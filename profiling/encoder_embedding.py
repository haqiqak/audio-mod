"""encoder_embedding.py — CrisperWhisper encoder-embedding primitives.

Core module (unlike `profiling/evaluation/`, this IS needed to run the app
itself when `require_repetition_classifier_confirmation` is enabled — see
`profiling/repetition_classifier.py`). Extracted from `profiling/
evaluation/encoder_features.py` (built for VALIDATION.md §11's Stage 1)
once the same extraction logic was needed by the live detector too
(§12.6.2's decision) — evaluation code must never be imported into core app
code (`profiling/evaluation/`'s own module docstring: "Not needed to RUN
the app itself"), so the shared primitives live here instead, and
`profiling/evaluation/encoder_features.py` now imports them from this
module rather than defining its own copies.

`profiling/asr.py` calls CrisperWhisper through `transformers.pipeline()`,
which never exposes hidden states — only the decoded text/timestamps. This
module bypasses that wrapper for a direct model call, but deliberately
loads *only the encoder* (`model.get_encoder()`), not the full seq2seq
model: only the last encoder layer's output (`last_hidden_state`, the
encoder's default primary output — no `output_hidden_states=True` flag or
decoding step needed at all) is used, so no generation/decoding happens
here.

**Real, known cost, not swept under the rug**: Whisper always pads to a
fixed 30s window before the encoder runs, so the encoder pass alone is the
dominant cost of a full transcription (~44s of the measured ~54s on
CPU/fp32 for a 4s clip — `ARCHITECTURE.md` §3) and is NOT proportionally
cheaper for shorter clips. Calling this from the live app means a SECOND
encoder pass distinct from the one `asr.py`'s `pipeline()` call already
does for transcription — real added latency, not the "zero added latency"
this project's own architecture review once assumed before this module
existed to measure it directly (see `PHASE_3_ARCHITECTURE_REVIEW.md` §5.1
for that original assumption, and `ARCHITECTURE.md`'s known-limitations
section for the now-corrected picture and the follow-on engineering work
that would remove this cost).
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

FRAME_SECONDS = 0.02  # Whisper encoder's fixed output resolution: a 30s
                       # input window always produces 1500 frames (30 / 1500).

_encoder_cache: dict[str, tuple] = {}  # model_id -> (processor, encoder), process-lifetime cache


@dataclass
class EncoderStates:
    hidden_states: np.ndarray  # [n_frames, hidden_dim], float32, last layer only
    frame_seconds: float = FRAME_SECONDS


def load_encoder(model_id: str | None = None):
    """Load CrisperWhisper's processor + encoder-only submodule directly.

    Returns (processor, encoder). Requires `transformers`/`torch` — not
    imported at module level so this file's pure-math functions (pool_span,
    cosine_distance) stay importable/testable without them, and so
    `profiling/detect.py` doesn't gain a hard `transformers` dependency
    merely by importing this module (matching `profiling/acoustic.py`'s own
    pattern for optional heavy dependencies like Praat/Silero-VAD).

    **Process-lifetime cached** (unlike `profiling/evaluation/`'s scripts,
    which call this once per run): the live app may call
    `detect_disfluencies()` many times per process, and reloading a ~3.2GB
    model on every call would be far worse than the encoder-pass cost
    itself. Same caching principle `profiling/acoustic.py` already uses for
    the Silero VAD model.
    """
    mid = model_id or os.environ.get("CRISPERWHISPER_MODEL", "nyrahealth/CrisperWhisper")
    if mid in _encoder_cache:
        return _encoder_cache[mid]

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(mid)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(mid, low_cpu_mem_usage=True)
    model.eval()
    result = (processor, model.get_encoder())
    _encoder_cache[mid] = result
    return result


def extract_last_layer_states(processor, encoder, samples: np.ndarray, sr: int) -> EncoderStates:
    """samples: float32 mono audio in [-1, 1] — the exact shape
    `profiling.acoustic.load_wav_samples()` already returns, reused here
    rather than a second WAV decoder."""
    import torch

    inputs = processor(samples, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = encoder(inputs.input_features)
    hidden = out.last_hidden_state[0].to(torch.float32).numpy()
    return EncoderStates(hidden_states=hidden)


def pool_span(states: EncoderStates, start: float | None, end: float | None) -> np.ndarray | None:
    """Mean-pool encoder frames overlapping [start, end] seconds
    (VALIDATION.md §11.1's "mean-pooled encoder hidden state across the
    encoder frames whose time range overlaps [start, end]")."""
    if start is None or end is None:
        return None
    n = states.hidden_states.shape[0]
    if n == 0:
        return None
    f0 = max(0, min(n - 1, int(round(start / states.frame_seconds))))
    f1 = max(0, min(n - 1, int(round(end / states.frame_seconds))))
    if f1 < f0:
        f0, f1 = f1, f0
    return states.hidden_states[f0:f1 + 1].mean(axis=0)


def cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """1 - cosine_similarity(a, b), or None if either vector is missing or
    zero-norm (VALIDATION.md §11.2 step 3)."""
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(1.0 - np.dot(a, b) / (na * nb))
