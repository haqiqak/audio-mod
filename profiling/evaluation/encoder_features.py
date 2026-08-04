"""encoder_features.py — Stage 1 (VALIDATION.md §11): extract
CrisperWhisper's own last-layer encoder hidden states and test them as a
zero-training corroboration signal for word_repetition/sound_repetition/
filler, per the protocol pre-registered before this module was written.

`profiling/asr.py` calls CrisperWhisper through `transformers.pipeline()`,
which never exposes hidden states — only the decoded text/timestamps. This
module bypasses that wrapper for a direct model call, but deliberately
loads *only the encoder* (`model.get_encoder()`), not the full seq2seq
model: Stage 1 only needs the last encoder layer's output
(`last_hidden_state`, the encoder's default primary output — no
`output_hidden_states=True` flag or decoding step needed at all), so no
generation/decoding happens here. This is cheaper than the pre-
registration assumed, but still real: Whisper always pads to a fixed 30s
window before the encoder runs, so the encoder pass alone is the dominant
cost of a full transcription (~44s of the measured ~54s on CPU/fp32 for a
4s clip — `ARCHITECTURE.md` §3) and is NOT proportionally cheaper for
shorter clips. See `run_encoder_signal_stage1.py`'s module docstring for
the resulting run-time/scoping decision this forced.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

FRAME_SECONDS = 0.02  # Whisper encoder's fixed output resolution: a 30s
                       # input window always produces 1500 frames (30 / 1500).


@dataclass
class EncoderStates:
    hidden_states: np.ndarray  # [n_frames, hidden_dim], float32, last layer only
    frame_seconds: float = FRAME_SECONDS


def load_encoder(model_id: str | None = None):
    """Load CrisperWhisper's processor + encoder-only submodule directly.

    Returns (processor, encoder). Requires `transformers`/`torch` — not
    imported at module level so this file's pure-math functions (pool_span,
    cosine_distance, fluent_centroid) stay importable/testable without them.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    mid = model_id or os.environ.get("CRISPERWHISPER_MODEL", "nyrahealth/CrisperWhisper")
    processor = AutoProcessor.from_pretrained(mid)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(mid, low_cpu_mem_usage=True)
    model.eval()
    return processor, model.get_encoder()


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


def fluent_centroid(
    states: EncoderStates, tokens: list[dict[str, Any]], ground_truth: dict[int, str],
) -> np.ndarray | None:
    """Mean pooled embedding over every token NOT in `ground_truth` — the
    clip's own fluent baseline (VALIDATION.md §11.2 step 2). None if the
    clip has no fluent tokens at all (degenerate, not expected in practice)."""
    vecs = []
    for i, tok in enumerate(tokens):
        if i in ground_truth:
            continue
        v = pool_span(states, tok.get("start"), tok.get("end"))
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """1 - cosine_similarity(a, b), or None if either vector is missing or
    zero-norm (VALIDATION.md §11.2 step 3)."""
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(1.0 - np.dot(a, b) / (na * nb))


def _event_span(event: dict[str, Any]) -> tuple[float | None, float | None]:
    """Same span preference as metrics._event_span (prefer the acoustic-
    native detector's precise region over the token's full nominal span) —
    duplicated here (not imported) to keep this module's only intra-package
    dependency the pure-math one, not a coupling to metrics.py's internals."""
    if event.get("acoustic_start") is not None:
        return event["acoustic_start"], event.get("acoustic_end")
    return event.get("start"), event.get("end")


def attach_encoder_distances(
    states: EncoderStates,
    tokens: list[dict[str, Any]],
    ground_truth: dict[int, str],
    events: list[dict[str, Any]],
    scorable_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return a copy of `events` with an `encoder_distance` key attached to
    every event whose type is in `scorable_types` (VALIDATION.md §11: the
    distance-to-fluent-centroid signal, ready for
    metrics.encoder_distance_stats() to aggregate). Events of other types
    are copied through unchanged (no key added) — this module's job is
    Stage 1's measurement only, never to change detector behavior.
    """
    centroid = fluent_centroid(states, tokens, ground_truth)
    out = []
    for e in events:
        e2 = dict(e)
        if e2.get("type") in scorable_types and centroid is not None:
            start, end = _event_span(e2)
            span_vec = pool_span(states, start, end)
            e2["encoder_distance"] = cosine_distance(span_vec, centroid)
        out.append(e2)
    return out


# ── §12 (VALIDATION.md): raw per-event data for the corroboration-mechanism comparison ─

REPEAT_PARTNER_TYPES = ("word_repetition", "sound_repetition")


def collect_raw_records(
    states: EncoderStates,
    clip_id: str,
    tokens: list[dict[str, Any]],
    ground_truth: dict[int, str],
    events: list[dict[str, Any]],
    scorable_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Per-event raw records for VALIDATION.md §12's comparison: the full
    embedding vectors needed to compute *any* signal/mechanism combination
    post-hoc, not just the one scalar Stage 1 (§11) tested. One dict per
    scorable event: {clip_id, index, type, label (1=TP/0=FP), embedding,
    centroid, partner_embedding (None if the event has no preceding token,
    or its type isn't in REPEAT_PARTNER_TYPES -- e.g. filler)}.

    `label` is computed here (not left to the caller) so the raw-data
    artifact is self-contained: TP/FP against `ground_truth`, matching
    score_word_level's own exact-type-match convention exactly.
    """
    centroid = fluent_centroid(states, tokens, ground_truth)
    if centroid is None:
        return []

    records = []
    for e in events:
        t = e.get("type")
        if t not in scorable_types:
            continue
        idx = e["index"]
        start, end = _event_span(e)
        embedding = pool_span(states, start, end)
        if embedding is None:
            continue

        partner_embedding = None
        if t in REPEAT_PARTNER_TYPES and idx - 1 >= 0:
            prev_tok = tokens[idx - 1]
            partner_embedding = pool_span(states, prev_tok.get("start"), prev_tok.get("end"))

        records.append({
            "clip_id": clip_id,
            "index": idx,
            "type": t,
            "label": 1 if ground_truth.get(idx) == t else 0,
            "embedding": embedding,
            "centroid": centroid,
            "partner_embedding": partner_embedding,
        })
    return records
