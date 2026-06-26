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

Backend note (added after measuring the above): the `transformers` pipeline
above pads every clip to Whisper's fixed 30s window before the encoder runs,
so the slow part is NOT the decode loop (max_new_tokens barely matters) —
it's one fixed-cost ~30s-window encoder forward pass in plain PyTorch fp32.
On CPU that was measured at ~650-680s regardless of clip length or token cap.

ASR_BACKEND controls which engine runs real audio (fixtures/.json/.txt never
need either):
  • "transformers" (default) — the only backend that currently produces
    correct word-level timestamps, which every feature in this app depends
    on (detection, calibration, profiling). Still has the ~30s-window
    encoder cost described above on CPU.
  • "auto" — currently identical to "transformers". Kept as an alias in
    case OpenVINO becomes usable again later (see below) and this should
    resume auto-selecting it.
  • "openvino" — DO NOT USE. Raises immediately with a clear message
    instead of attempting transcription. optimum-intel's
    OVModelForSpeechSeq2Seq does not return cross_attentions from
    generate() the way the plain PyTorch model does, and transformers'
    return_timestamps="word" post-processing requires them to compute
    word-level alignment — this is a confirmed, still-open upstream bug
    (github.com/huggingface/optimum-intel issue #561), not something
    fixable from this codebase. It was briefly the auto-selected default
    in an earlier round of this project, before a real test recording
    surfaced the crash — transcription would actually run (and run a fast
    encoder pass), then fail deep inside transformers' generation code
    with `TypeError: 'NoneType' object is not subscriptable` once it tried
    to extract word timestamps from the (absent) cross-attentions. See
    _transcribe_openvino's docstring for the full trace and what would need
    to change upstream before this becomes safe to re-enable.
  • "faster_whisper" — NOT auto-selected; tried and ruled out for
    CrisperWhisper specifically (its tokenizer wrapper hardcodes stock-
    Whisper special-token positions, and CrisperWhisper's fine-tune has a
    different layout — see _load_faster_whisper for the exact error). Left
    in for anyone who wants to retry against a future faster-whisper
    release that supports custom special-token layouts.
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
        self._ct2_model = None
        self._ov_pipe = None
        # NOTE: faster_whisper/CTranslate2 was tried and ruled out — its
        # tokenizer wrapper hardcodes stock-Whisper special-token positions
        # and CrisperWhisper's fine-tune has a different layout, so it can
        # never find "<|startoftranscript|>" no matter how the model is
        # converted. OpenVINO was ALSO tried as a fast path and is likewise
        # ruled out (for a different reason): its OVModelForSpeechSeq2Seq
        # doesn't return the cross_attentions that return_timestamps="word"
        # needs, so _transcribe_openvino() now raises immediately rather than
        # crashing minutes in (see its docstring + the module header). The
        # default backend is therefore "transformers" — the only one that
        # currently produces correct word-level timestamps.
        self.backend = os.environ.get("ASR_BACKEND", "transformers").strip().lower()
        self.ct2_model_dir = os.environ.get(
            "CRISPERWHISPER_CT2_DIR",
            str(Path(__file__).resolve().parents[1] / "models" / "crisperwhisper-ct2"),
        )
        self.ct2_compute_type = os.environ.get("CRISPERWHISPER_CT2_COMPUTE_TYPE", "int8")
        self.ov_model_dir = os.environ.get(
            "CRISPERWHISPER_OV_DIR",
            str(Path(__file__).resolve().parents[1] / "models" / "crisperwhisper-ov"),
        )
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
            "model_kwargs": {
                "low_cpu_mem_usage": True,
                # Silences: "WhisperModel is using WhisperSdpaAttention, but
                # scaled_dot_product_attention does not support output_attentions=True
                # ... Falling back to the manual attention implementation."
                # The pipeline requests output_attentions internally for word-timestamp
                # extraction. SDPA doesn't support that, so it falls back anyway —
                # setting "eager" explicitly just stops the warning without changing
                # which implementation actually runs (it was always falling back).
                "attn_implementation": "eager",
            },
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
            #
            # ── return_legacy_cache ─────────────────────────────────────────────
            # Silences: "From v4.47 onwards, when a model cache is to be returned,
            # `generate` will return a Cache instance instead by default."
            # We don't use the cache object at all — setting this avoids the
            # format-change warning without affecting output.
            "generate_kwargs": {
                "language": "en",
                "task": "transcribe",
                "num_beams": 1,
                "max_new_tokens": 256,
                "return_legacy_cache": True,
            },
        }
        if self.device not in (None, "cpu"):
            kwargs["device"] = self.device

        self._pipe = pipeline("automatic-speech-recognition", **kwargs)
        return self._pipe


    def _load_faster_whisper(self):
        """Load the CTranslate2-converted CrisperWhisper model (faster-whisper).

        Requires a one-time conversion of the HF weights — see README
        "Faster backend setup". We deliberately do NOT auto-convert here:
        conversion downloads + reprocesses the full ~3GB model and can itself
        take minutes, which would be a confusing surprise mid-transcription.
        """
        if self._ct2_model is not None:
            return self._ct2_model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster_whisper is not installed.\n"
                "Run in your venv:\n"
                "    pip install faster-whisper ctranslate2\n"
                "Then restart Streamlit. Or set ASR_BACKEND=transformers to\n"
                "use the slower (but dependency-light) original pipeline."
            ) from exc

        model_dir = Path(self.ct2_model_dir)
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise RuntimeError(
                f"CTranslate2 model not found at: {model_dir}\n"
                "Convert CrisperWhisper's weights once with:\n"
                f"    ct2-transformers-converter --model {self.model_id} "
                f"--output_dir \"{model_dir}\" --quantization {self.ct2_compute_type} "
                "--copy_files tokenizer.json preprocessor_config.json\n"
                "(install the converter first: pip install ctranslate2 transformers[torch])\n"
                "This is a one-time step — the converted files are reused on every "
                "future run. Or set ASR_BACKEND=transformers to use the original, "
                "slower pipeline meanwhile."
            )

        device = "cpu" if self.device in (None, "cpu") else str(self.device)
        self._ct2_model = WhisperModel(
            str(model_dir),
            device=device,
            compute_type=self.ct2_compute_type,
        )
        return self._ct2_model

    def _load_openvino_pipeline(self):
        """Load CrisperWhisper through OpenVINO (via optimum-intel).

        Unlike the faster_whisper path, this does NOT swap out the tokenizer
        or special-token handling — it accelerates the math (encoder/decoder
        matmuls) underneath the exact same `transformers` model/processor
        objects we already know produce correct output. That's what avoids
        the "<|startoftranscript|> not found" class of bug entirely: there's
        only ever one tokenizer implementation in play, the normal HF one.

        First call for a given model: exports + int8-quantizes into
        ov_model_dir (slower, one-time — comparable to the CTranslate2
        conversion we did before, but happens automatically in-process
        instead of a separate CLI step). Every call after that loads the
        cached export directly (fast).
        """
        if self._ov_pipe is not None:
            return self._ov_pipe

        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor, pipeline
        except ImportError as exc:
            raise RuntimeError(
                "optimum[openvino] is not installed.\n"
                "Run in your venv:\n"
                "    pip install \"optimum[openvino]\"\n"
                "Then restart Streamlit. Or set ASR_BACKEND=transformers to\n"
                "use the slower (but dependency-light) original pipeline."
            ) from exc

        model_dir = Path(self.ov_model_dir)
        processor = AutoProcessor.from_pretrained(self.model_id)

        if model_dir.exists() and any(model_dir.iterdir()):
            ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
                str(model_dir), compile=True,
            )
        else:
            # First-run export: downloads the HF weights (likely already
            # cached from earlier runs), converts to OpenVINO IR, and
            # int8-quantizes. This is the slow one-time step — expect it to
            # take a few minutes, not the ~700s-per-clip we were seeing.
            model_dir.mkdir(parents=True, exist_ok=True)
            ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, export=True, load_in_8bit=True, compile=True,
            )
            ov_model.save_pretrained(str(model_dir))
            processor.save_pretrained(str(model_dir))

        self._ov_pipe = pipeline(
            "automatic-speech-recognition",
            model=ov_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps="word",
        )
        return self._ov_pipe

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

        if self.backend == "faster_whisper":
            return self._transcribe_faster_whisper(audio_for_asr)
        if self.backend == "openvino":
            return self._transcribe_openvino(audio_for_asr)
        if self.backend == "transformers":
            return self._transcribe_transformers(audio_for_asr)

        # "auto" — currently identical to "transformers". OpenVINO is
        # intentionally NOT part of this chain; see the note above
        # _transcribe_openvino for why it can't be used as a silent
        # fallback target (or as a target at all, right now).
        return self._transcribe_transformers(audio_for_asr)

    @staticmethod
    def _clip_duration_seconds(audio_for_asr: Path) -> float | None:
        """Clip length in seconds read from the WAV header, or None for a
        non-WAV file (mp3/flac/m4a) whose duration we can't read header-only.
        Used both for the max_new_tokens budget and for last_timing's
        real-time-factor reporting (see benchmark_asr.py)."""
        try:
            with wave.open(str(audio_for_asr), "rb") as _wf:
                return _wf.getnframes() / max(_wf.getframerate(), 1)
        except Exception:
            return None

    @staticmethod
    def _audio_size_bytes(audio_for_asr: Path) -> int | None:
        """On-disk size of the clip being transcribed, or None if it can't be
        stat'd. Recorded in last_timing so a benchmark row can relate latency
        to input size, not just duration."""
        try:
            return int(Path(audio_for_asr).stat().st_size)
        except Exception:
            return None

    @classmethod
    def _max_new_tokens_for(cls, audio_for_asr: Path) -> int:
        """Cap decode steps proportional to clip duration (~6 tok/s, 20 floor,
        256 ceiling) so a clip with no clean EOS can't burn the full cap on
        repetition/garbage. Shared by the transformers and openvino paths
        (faster_whisper manages its own stopping via beam_size/VAD)."""
        dur = cls._clip_duration_seconds(audio_for_asr)
        if dur is None:
            return 256  # non-WAV (mp3/flac/m4a) — fall back to the flat cap
        return max(20, min(256, int(dur * 6) + 20))

    def _transcribe_faster_whisper(self, audio_for_asr: Path) -> list[dict[str, Any]]:
        """Fast path: CTranslate2-quantized CrisperWhisper.

        Fixes the actual bottleneck found in the transformers path — Whisper
        pads every clip to a fixed 30s window before the encoder runs, so the
        encoder forward pass (not decode steps) is the fixed ~650-680s cost on
        CPU/fp32. CTranslate2's int8 kernels target exactly that cost.
        """
        import time as _time
        _t0 = _time.perf_counter()
        model = self._load_faster_whisper()
        _t_load = _time.perf_counter() - _t0

        _t1 = _time.perf_counter()
        segments, _info = model.transcribe(
            str(audio_for_asr),
            language="en",
            task="transcribe",
            beam_size=1,          # same "THE fix" rationale as num_beams=1 above
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        chunks: list[dict[str, Any]] = []
        for seg in segments:
            words = getattr(seg, "words", None) or []
            for w in words:
                chunks.append({
                    "text": w.word,
                    "timestamp": (w.start, w.end),
                })
            if not words and (seg.text or "").strip():
                # Model returned segment-level text with no word alignment
                # (rare, e.g. very short/silent clips) — fall back to the
                # segment's own span so we still produce something.
                chunks.append({"text": seg.text.strip(), "timestamp": (seg.start, seg.end)})
        _t_infer = _time.perf_counter() - _t1

        clip_dur = self._clip_duration_seconds(audio_for_asr)
        self.last_timing = {
            "load_pipeline_seconds": round(_t_load, 3),
            "inference_seconds": round(_t_infer, 3),
            "clip_duration_seconds": round(clip_dur, 3) if clip_dur is not None else None,
            "max_new_tokens": None,  # faster_whisper stops via beam_size/VAD, not a token cap
            "audio_size_bytes": self._audio_size_bytes(audio_for_asr),
            "backend": "faster_whisper",
        }

        if chunks:
            return [t.to_dict() for t in self._tokens_from_chunks(chunks)]
        return []

    def _transcribe_transformers(self, audio_for_asr: Path) -> list[dict[str, Any]]:
        """Original (slow) path — kept as a fallback / comparison backend."""
        max_new_tokens = self._max_new_tokens_for(audio_for_asr)

        # ── Real ASR, instrumented ──────────────────────────────────────────────
        # self.last_timing is read by app.py after this call to show a real
        # breakdown in the UI log instead of one opaque total. _load_pipeline()
        # should be near-zero on a warm/cached call; if it isn't, the model is
        # being reconstructed every time.
        import time as _time
        _t0 = _time.perf_counter()
        pipe = self._load_pipeline()
        _t_load = _time.perf_counter() - _t0

        _t1 = _time.perf_counter()
        result = pipe(str(audio_for_asr), generate_kwargs={"max_new_tokens": max_new_tokens})
        _t_infer = _time.perf_counter() - _t1

        clip_dur = self._clip_duration_seconds(audio_for_asr)
        self.last_timing = {
            "load_pipeline_seconds": round(_t_load, 3),
            "inference_seconds": round(_t_infer, 3),
            "clip_duration_seconds": round(clip_dur, 3) if clip_dur is not None else None,
            "max_new_tokens": max_new_tokens,
            "audio_size_bytes": self._audio_size_bytes(audio_for_asr),
            "backend": "transformers",
        }

        chunks = result.get("chunks") or result.get("segments") or []
        if chunks:
            return [t.to_dict() for t in self._tokens_from_chunks(chunks)]
        # No word timestamps returned — fall back to text tokenisation
        return [t.to_dict() for t in self.tokens_from_text(result.get("text", ""))]

    def _transcribe_openvino(self, audio_for_asr: Path) -> list[dict[str, Any]]:
        """NOT a working fast path right now — kept for reference / future fix.

        Confirmed upstream bug: optimum-intel's OVModelForSpeechSeq2Seq does
        not return cross_attentions from generate() the way the plain
        PyTorch model does, and transformers' return_timestamps="word"
        post-processing requires them to compute word-level alignment. This
        crashes deep inside transformers/models/whisper/generation_whisper.py
        (_extract_token_timestamps) with `TypeError: 'NoneType' object is
        not subscriptable` — confirmed against
        github.com/huggingface/optimum-intel issue #561, still open upstream
        with no known workaround as of this writing.

        Since every feature in this app (detection, calibration, profiling)
        depends on word-level timestamps, OpenVINO cannot currently be used
        here regardless of its speed advantage — a fast transcript with no
        word timings is useless for this pipeline. This method now fails
        fast with a clear message instead of letting the person hit the
        confusing NoneType crash several minutes into a real transcription.

        Revisit if a future optimum-intel release fixes #561, or if anyone
        finds a workaround that restores cross_attentions through the OV
        decoder — at that point this can become the real default again.
        """
        raise RuntimeError(
            "ASR_BACKEND=openvino is not currently usable: optimum-intel's "
            "OpenVINO Whisper model can't produce word-level timestamps "
            "(github.com/huggingface/optimum-intel issue #561), and this "
            "app depends on word timestamps for every feature. Use "
            "ASR_BACKEND=transformers (the default) instead."
        )

    def _transcribe_openvino_DISABLED_reference_impl(self, audio_for_asr: Path) -> list[dict[str, Any]]:
        """Original implementation, kept only as a reference for anyone
        revisiting this once optimum-intel issue #561 is fixed upstream.
        Not called anywhere — _transcribe_openvino above raises before
        reaching any of this.
        """
        max_new_tokens = self._max_new_tokens_for(audio_for_asr)

        import time as _time
        _t0 = _time.perf_counter()
        pipe = self._load_openvino_pipeline()
        _t_load = _time.perf_counter() - _t0

        _t1 = _time.perf_counter()
        result = pipe(
            str(audio_for_asr),
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
            },
        )
        _t_infer = _time.perf_counter() - _t1

        self.last_timing = {
            "load_pipeline_seconds": round(_t_load, 3),
            "inference_seconds": round(_t_infer, 3),
        }

        chunks = result.get("chunks") or result.get("segments") or []
        if chunks:
            return [t.to_dict() for t in self._tokens_from_chunks(chunks)]
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
