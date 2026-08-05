"""repetition_classifier.py — applies the trained word_repetition/
sound_repetition corroboration classifier (VALIDATION.md §12.6.2's
decision, evidence-constrained per `CLAUDE.md` standing rule 8) as a hard
gate on candidate events — the same architectural role Praat-gating
already plays for `prolongation`
(`require_praat_stability_for_prolongation`).

Model artifact: `models/repetition_corroboration_classifier.npz`, trained
by `profiling/evaluation/train_repetition_classifier.py` on real
LibriStutter audio (250 clips, 388 events) — small (~17KB), committed to
the repo (unlike the huge pretrained CrisperWhisper weights under
`.cache/`, which are gitignored).

**Graceful, multi-layer no-op** when anything needed is unavailable:
`transformers`/`torch` not installed, the weights file missing, no audio,
or the encoder pass fails for any reason — matches every other optional
acoustic component in this codebase (`_AcousticContext` in `detect.py`,
Silero VAD / Praat in `acoustic.py`): never blocks or crashes, degrades to
"this gate is skipped, the token-path event fires exactly as it would
have before this feature existed" — the same "no extra evidence is not a
failure" principle documented throughout this project.

**Real, known cost, stated plainly**: unlike Praat/VAD (cheap, direct
signal-processing on the waveform), this gate requires a second
CrisperWhisper encoder pass distinct from the one already run for
transcription — measured at ~30-90s per clip
(`profiling/encoder_embedding.py`'s module docstring, `ARCHITECTURE.md`
§3/§7). Enabling this in the live app adds that latency on top of the
existing ASR cost. See `ARCHITECTURE.md`'s known-limitations section for
the follow-on engineering work that would remove this (capturing encoder
states during the same forward pass `asr.py` already makes for
transcription, instead of a second, separate pass) — not attempted here,
a real, separately-scoped next step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "repetition_corroboration_classifier.npz"

# Sentinel distinguishing "not tried yet" (None) from "tried and
# unavailable" ({}) so a missing/broken model file is only ever detected
# once per process, not re-attempted on every clip.
_weights_cache: dict[str, Any] | None = None


def _load_weights() -> dict[str, Any] | None:
    global _weights_cache
    if _weights_cache is not None:
        return _weights_cache or None
    try:
        d = np.load(_MODEL_PATH, allow_pickle=True)
        _weights_cache = {
            "weights": d["weights"].astype(np.float64),
            "bias": float(d["bias"]),
            "mean": d["mean"].astype(np.float64),
            "std": d["std"].astype(np.float64),
        }
    except Exception:
        _weights_cache = {}
    return _weights_cache or None


class RepetitionClassifierContext:
    """Per-clip context, mirroring `_AcousticContext`'s lazy/graceful
    pattern in `detect.py`. **Loading is deferred to the first actual
    query** (`available`/`confirms_repetition`), not done in `__init__` —
    a clip with no `word_repetition`/`sound_repetition` candidates at all
    must never pay the encoder-load cost. This matters concretely: this
    project's fast, real-model-free unit test suite constructs
    `detect_disfluencies()` with synthetic `audio_bytes` throughout, and an
    eager load here would silently turn every one of those into a
    multi-second-plus real-model-loading test the moment this class is
    constructed, whether or not a repetition candidate is ever checked."""

    def __init__(self, audio_bytes: bytes | None, enabled: bool):
        self.enabled = enabled
        self._audio_bytes = audio_bytes
        self._states = None
        self._weights: dict[str, Any] | None = None
        self._load_attempted = False

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.enabled or not self._audio_bytes:
            return

        self._weights = _load_weights()
        if self._weights is None:
            return  # model artifact missing -- graceful no-op

        try:
            from profiling.acoustic import load_wav_samples
            from profiling.encoder_embedding import extract_last_layer_states, load_encoder

            samples, sr = load_wav_samples(self._audio_bytes)
            if samples is None:
                return
            processor, encoder = load_encoder()
            self._states = extract_last_layer_states(processor, encoder, samples, sr)
        except Exception:
            # transformers/torch not installed, encoder load/inference
            # failed, or anything else went wrong -- graceful no-op, same
            # principle as every other optional acoustic component here.
            self._states = None

    @property
    def available(self) -> bool:
        self._ensure_loaded()
        return self._states is not None and self._weights is not None

    def confirms_repetition(self, start: float | None, end: float | None) -> bool:
        """True if the classifier confirms this span looks like a genuine
        repetition, OR the gate can't be evaluated at all (graceful
        no-op — an unavailable signal is never treated as evidence
        *against* firing, only ever an additional requirement when it IS
        available, matching `word_praat_stable`'s identical convention)."""
        if not self.available or start is None or end is None:
            return True

        from profiling.encoder_embedding import pool_span

        span_vec = pool_span(self._states, start, end)
        if span_vec is None:
            return True

        w = self._weights
        x = (span_vec.astype(np.float64) - w["mean"]) / w["std"]
        z = float(np.dot(x, w["weights"]) + w["bias"])
        proba = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        return bool(proba >= 0.5)
