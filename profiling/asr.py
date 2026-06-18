"""
asr.py — CrisperWhisper front-end for verbatim transcription.

Fixes applied vs. original:
  1. generate_kwargs includes num_beams=1 — THE confirmed fix for
     "size of tensor a (2) must match tensor b (0)". This is a documented
     transformers bug (huggingface/transformers #28007, #36093):
     WhisperGenerationMixin._extract_token_timestamps() mis-shapes
     `beam_indices` when return_timestamps="word" is combined with beam
     search. CrisperWhisper's generation_config.json defaults to multiple
     beams, which triggers it on every call. Forcing greedy decoding
     (num_beams=1) avoids the buggy code path entirely. The model card's
     own faster-whisper example also uses beam_size=1 for word timestamps.
  2. generate_kwargs only contains language='en' + task='transcribe' +
     num_beams=1 — NOT forced_decoder_ids and NOT chunk_length_s. Both
     of those were previously tried as "fixes" but actually caused the
     warnings seen in testing ("very experimental with seq2seq models",
     "custom logits processor ... will take precedence").
  3. _ensure_min_duration() pads clips under 1.2s with trailing silence,
     as a secondary safety net for extremely short clips (this alone did
     NOT fix the tensor mismatch — num_beams=1 is the real fix above).
  4. resample_to_16k() using numpy only (audioop removed in Python 3.13) →
     mic recordings at 44100 Hz are downsampled before ASR.
  5. RuntimeError (not silent fallback) when transformers is missing, so
     the UI shows a clear install message instead of spinning forever.

Note on CPU latency: CrisperWhisper is a ~3.2 GB seq2seq model. On CPU,
real-time-factor is typically 2-8x, meaning a 10-second clip can take
20-80 seconds, and longer clips scale accordingly. This is expected
model behaviour, not a bug — see app.py for the live progress logging
that surfaces this to the user instead of looking frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import math
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable
import wave

import numpy as np


# ── WAV resampler (numpy only, works on Python 3.13) ─────────────────────────

def resample_to_16k(wav_bytes: bytes) -> bytes:
    """Return WAV bytes resampled to 16 kHz mono int16.

    Uses only numpy — no audioop (removed in 3.13), no librosa, no soundfile.
    Called on mic recordings (typically 44100 Hz stereo) before ASR.
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf) as wf:
        channels  = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw       = wf.readframes(wf.getnframes())

    target_sr = 16_000
    if framerate == target_sr and channels == 1 and sampwidth == 2:
        return wav_bytes  # already the right format

    dtype = np.int16 if sampwidth == 2 else (
        np.int32 if sampwidth == 4 else np.uint8
    )
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # stereo → mono
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    elif channels > 2:
        samples = samples.reshape(-1, channels).mean(axis=1)

    # linear resample
    if framerate != target_sr:
        n_out   = max(1, int(len(samples) * target_sr / framerate))
        x_in    = np.linspace(0, len(samples) - 1, len(samples))
        x_out   = np.linspace(0, len(samples) - 1, n_out)
        samples = np.interp(x_out, x_in, samples)

    pcm = samples.clip(-32768, 32767).astype(np.int16).tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(pcm)
    return out.getvalue()


# ── VerbatimToken ─────────────────────────────────────────────────────────────

@dataclass
class VerbatimToken:
    word: str
    start: float | None = None
    end: float | None = None
    is_filler: bool = False
    is_stutter: bool = False
    source: str = "asr"
    profile_safe: bool = True

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "VerbatimToken":
        return cls(
            word=str(row.get("word", "")).strip(),
            start=_maybe_float(row.get("start")),
            end=_maybe_float(row.get("end")),
            is_filler=bool(row.get("is_filler", False)),
            is_stutter=bool(row.get("is_stutter", False)),
            source=str(row.get("source", "asr") or "asr"),
            profile_safe=bool(row.get("profile_safe", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("source") == "asr":
            data.pop("source", None)
        if data.get("profile_safe") is True:
            data.pop("profile_safe", None)
        return data


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _looks_like_vanilla_whisper(model_id: str) -> bool:
    low = model_id.lower()
    return low.startswith("openai/whisper") or low in {"whisper", "large-v3"}


def _normalise_word(word: str) -> str:
    return re.sub(r"^\W+|\W+$", "", word or "").lower()


# ── WAV timing fallback helpers ───────────────────────────────────────────────

def _pcm_rms(frame_bytes: bytes, sample_width: int) -> float:
    if not frame_bytes or sample_width <= 0:
        return 0.0
    values: list[int] = []
    for i in range(0, len(frame_bytes) - sample_width + 1, sample_width):
        raw = frame_bytes[i:i + sample_width]
        if sample_width == 1:
            values.append(raw[0] - 128)
        else:
            values.append(int.from_bytes(raw, "little", signed=True))
    if not values:
        return 0.0
    peak  = 128.0 if sample_width == 1 else float((1 << (8 * sample_width - 1)) - 1)
    power = sum((s / peak) ** 2 for s in values) / len(values)
    return min(1.0, math.sqrt(power))


def _merge_segments(segs: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in segs:
        if not merged or start - merged[-1][1] > gap:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], end)
    return merged


def _alpha_label(index: int) -> str:
    label = ""
    n = max(1, int(index))
    while n:
        n, rem = divmod(n - 1, 26)
        label = chr(97 + rem) + label
    return label


# ── Main ASR class ────────────────────────────────────────────────────────────

class CrisperWhisperASR:
    """CrisperWhisper wrapper with all hang / tensor-mismatch fixes applied."""

    FILLERS = {"uh", "um", "er", "erm", "ah", "hmm", "like"}

    def __init__(self, model_id: str | None = None, device: str | int | None = None):
        self.model_id = model_id or os.environ.get(
            "CRISPERWHISPER_MODEL", "nyrahealth/CrisperWhisper"
        )
        if _looks_like_vanilla_whisper(self.model_id):
            raise ValueError(
                "Use CrisperWhisper for verbatim stutter transcription; "
                "vanilla Whisper silently removes disfluencies."
            )
        self.device = device if device is not None else os.environ.get("ASR_DEVICE", "cpu")
        self._pipe = None
        self.last_timing: dict[str, float] = {}

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed.\n"
                "Run in your venv:\n"
                "    pip install transformers accelerate torch\n"
                "Then restart Streamlit."
            ) from exc

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "return_timestamps": "word",
            "model_kwargs": {"low_cpu_mem_usage": True},
            # ── language='en' + task='transcribe' ──────────────────────────────
            # This is the ONLY thing needed to skip Whisper's multilingual
            # language-detection pass. Do NOT also pass forced_decoder_ids
            # (None or otherwise) — the pipeline builds its own logits
            # processors from language/task, and passing forced_decoder_ids
            # on top of that creates duplicate SuppressTokensLogitsProcessor
            # instances that fight each other (the
            # "custom logits processor ... will take precedence" warnings).
            #
            # Do NOT pass chunk_length_s either — transformers explicitly
            # flags this as experimental/inaccurate for seq2seq models like
            # Whisper (see the "very experimental with seq2seq models"
            # warning). Whisper's pipeline already does its own internal
            # long-form chunking; chunk_length_s overrides that with a
            # less accurate sliding-window mechanism.
            #
            # ── num_beams=1 (THE ACTUAL FIX for the tensor mismatch) ───────────
            # "size of tensor a (2) must match tensor b (0)" is a confirmed,
            # documented transformers bug: WhisperGenerationMixin._extract_
            # token_timestamps() mis-shapes `beam_indices` when return_timestamps
            # ="word" is combined with beam search (num_beams > 1). CrisperWhisper's
            # own generation_config.json defaults to multiple beams, which is what
            # triggers it. See huggingface/transformers issues #28007 and #36093,
            # and the nyrahealth/CrisperWhisper model card, which explicitly
            # recommends beam_size=1 for word timestamps. Forcing greedy decoding
            # here sidesteps the buggy beam-indices reshape entirely — this is
            # what actually fixes the crash, not chunk_length_s or padding alone.
            #
            # ── max_new_tokens cap ──────────────────────────────────────────────
            # Bounds worst-case generation length so a single bad decode step
            # (e.g. repetition without a clean EOS) can't silently balloon
            # runtime on short clips. 256 is generous for clips under ~30s.
            "generate_kwargs": {
                "language": "en",
                "task": "transcribe",
                "num_beams": 1,
                "max_new_tokens": 256,
            },
        }
        if self.device not in (None, "cpu"):
            kwargs["device"] = self.device

        self._pipe = pipeline("automatic-speech-recognition", **kwargs)
        return self._pipe

    def transcribe(self, audio_path: str | Path) -> list[dict[str, Any]]:
        """Return verbatim token dicts.

        .json / .txt / .transcript  → fixture, no ASR
        .wav / .mp3 / .flac / .m4a → CrisperWhisper
        """
        path   = Path(audio_path)
        suffix = path.suffix.lower()

        # ── Fixtures (instant, no model) ──────────────────────────────────────
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("tokens", data) if isinstance(data, dict) else data
            return [VerbatimToken.from_mapping(row).to_dict() for row in rows]

        if suffix in {".txt", ".transcript"}:
            text = path.read_text(encoding="utf-8")
            return [t.to_dict() for t in self.tokens_from_text(text)]

        # ── Forced timing fallback (env override) ─────────────────────────────
        if os.environ.get("ASR_FORCE_AUDIO_FALLBACK", "").strip() == "1":
            return [t.to_dict() for t in self.tokens_from_audio_timing(path)]

        # ── Pad very short clips (secondary safety net) ────────────────────────
        # num_beams=1 in _load_pipeline() is the actual fix for the
        # "size of tensor a (2) must match tensor b (0)" error. This padding
        # is an additional safety net for extremely short clips (<1.2s) where
        # the encoder may not produce enough frames for reliable word-level
        # alignment even with greedy decoding.
        audio_for_asr = self._ensure_min_duration(path)

        # ── Real ASR, instrumented ──────────────────────────────────────────────
        # self.last_timing is read by app.py after this call to show a real
        # breakdown in the UI log instead of one opaque total — see future.md
        # Step 0. _load_pipeline() should be near-zero on a warm/cached call;
        # if it isn't, the model is being reconstructed every time. The pipe()
        # call itself is the actual generate() — if THIS is what's slow, the
        # bottleneck is inference, not loading, and future.md Step 2 applies.
        import time as _time
        _t0 = _time.perf_counter()
        pipe = self._load_pipeline()
        _t_load = _time.perf_counter() - _t0

        _t1 = _time.perf_counter()
        result = pipe(str(audio_for_asr))
        _t_infer = _time.perf_counter() - _t1

        self.last_timing = {
            "load_pipeline_seconds": round(_t_load, 3),
            "inference_seconds": round(_t_infer, 3),
        }

        chunks = result.get("chunks") or result.get("segments") or []
        if chunks:
            return [t.to_dict() for t in self._tokens_from_chunks(chunks)]
        # No word timestamps returned — fall back to text tokenisation
        return [t.to_dict() for t in self.tokens_from_text(result.get("text", ""))]

    def _ensure_min_duration(self, path: Path, min_seconds: float = 1.2) -> Path:
        """Pad a WAV file with trailing silence if shorter than min_seconds.

        Returns the original path unchanged for non-WAV files (mp3/flac/m4a
        are long enough in practice, and padding compressed formats requires
        decoding them anyway, which we avoid here to keep this dependency-free).
        """
        if path.suffix.lower() not in {".wav", ".wave"}:
            return path
        try:
            with wave.open(str(path), "rb") as wf:
                channels  = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                nframes   = wf.getnframes()
                raw       = wf.readframes(nframes)
            duration = nframes / framerate if framerate else 0
            if duration >= min_seconds:
                return path

            pad_frames = int((min_seconds - duration) * framerate)
            silence = b"\x00" * (pad_frames * channels * sampwidth)

            import tempfile
            padded_path = Path(tempfile.mktemp(suffix=".wav"))
            with wave.open(str(padded_path), "wb") as wf_out:
                wf_out.setnchannels(channels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                wf_out.writeframes(raw + silence)
            return padded_path
        except Exception:
            # If anything goes wrong inspecting/padding, just use the original —
            # worst case is the original tensor-mismatch error resurfaces for
            # that one clip, which is no worse than before this fix existed.
            return path

    def transcribe_bytes(self, wav_bytes: bytes) -> list[dict[str, Any]]:
        """Transcribe raw WAV bytes (e.g. from mic_recorder).

        Resamples to 16 kHz mono before passing to ASR.
        Writes a temp file, transcribes, cleans up.
        """
        import tempfile
        resampled = resample_to_16k(wav_bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tf.write(resampled)
            tmp_path = tf.name
        try:
            return self.transcribe(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _tokens_from_chunks(self, chunks: Iterable[dict[str, Any]]) -> list[VerbatimToken]:
        tokens: list[VerbatimToken] = []
        for chunk in chunks:
            word = str(chunk.get("text") or chunk.get("word") or "").strip()
            if not word:
                continue
            ts = chunk.get("timestamp") or (chunk.get("start"), chunk.get("end"))
            start = end = None
            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start, end = _maybe_float(ts[0]), _maybe_float(ts[1])
            else:
                start = _maybe_float(chunk.get("start"))
                end   = _maybe_float(chunk.get("end"))
            low = _normalise_word(word)
            tokens.append(VerbatimToken(
                word=word, start=start, end=end,
                is_filler=low in self.FILLERS or bool(chunk.get("is_filler", False)),
                is_stutter=bool(chunk.get("is_stutter", False)) or word.endswith("-"),
            ))
        return tokens

    def tokens_from_audio_timing(self, audio_path: str | Path) -> list[VerbatimToken]:
        """WAV timing fallback — produces speech_a, speech_b … tokens (no words)."""
        path = Path(audio_path)
        if path.suffix.lower() not in {".wav", ".wave"}:
            return []
        try:
            with wave.open(str(path), "rb") as wf:
                sr   = int(wf.getframerate())
                sw   = int(wf.getsampwidth())
                ch   = max(1, int(wf.getnchannels()))
                win  = max(1, int(sr * 0.03))
                fi   = 0
                wins: list[tuple[float, float, float]] = []
                while True:
                    data = wf.readframes(win)
                    if not data:
                        break
                    nr    = max(1, len(data) // max(sw * ch, 1))
                    wins.append((fi / sr, (fi + nr) / sr, _pcm_rms(data, sw)))
                    fi += nr
        except Exception:
            return []

        if not wins:
            return []
        rms = [r for _, _, r in wins]
        if max(rms) < 0.01:
            return []
        ordered = sorted(rms)
        nf = ordered[max(0, int(len(ordered) * 0.15))]
        pk = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        th = max(0.012, nf + 0.18 * (pk - nf))

        raw: list[tuple[float, float]] = []
        cs = ce = None
        for s, e, r in wins:
            if r >= th:
                cs = s if cs is None else cs
                ce = e
            elif cs is not None:
                raw.append((cs, ce))
                cs = ce = None
        if cs is not None:
            raw.append((cs, ce))

        segs = [(s, e) for s, e in _merge_segments(raw, 0.16) if e - s >= 0.08]
        return [
            VerbatimToken(
                word=f"speech_{_alpha_label(i)}",
                start=round(s, 3), end=round(e, 3),
                source="audio_timing_fallback", profile_safe=False,
            )
            for i, (s, e) in enumerate(segs, 1)
        ]

    def tokens_from_text(self, text: str) -> list[VerbatimToken]:
        words = re.findall(r"[A-Za-z]+-?|[.,!?;:]", text or "")
        return [
            VerbatimToken(
                word=w,
                is_filler=_normalise_word(w) in self.FILLERS,
                is_stutter=w.endswith("-"),
            )
            for w in words
        ]
