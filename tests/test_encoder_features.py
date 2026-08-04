"""Tests for profiling/evaluation/encoder_features.py's pure-math functions
(VALIDATION.md §11, Phase 3 Stage 1): span pooling, cosine distance, the
per-clip fluent centroid, and attaching encoder_distance onto events.

Deliberately does NOT test load_encoder()/extract_last_layer_states() —
those require downloading and running the real ~3.2GB CrisperWhisper model,
which this project's fast unit-test suite has never depended on (matching
tests/test_acoustic.py's own split between pure-function tests here and
real-model runs left to the evaluation scripts). Uses hand-constructed
EncoderStates with a small synthetic hidden_dim throughout.

    pytest tests/test_encoder_features.py
    python tests/test_encoder_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.evaluation.encoder_features import (
    EncoderStates,
    attach_encoder_distances,
    collect_raw_records,
    cosine_distance,
    fluent_centroid,
    pool_span,
)


def _states(n_frames: int, hidden_dim: int = 4, frame_seconds: float = 0.02) -> EncoderStates:
    # Frame i is filled with value i (broadcast across hidden_dim) so pooled
    # means are easy to hand-verify.
    hidden = np.tile(np.arange(n_frames, dtype=np.float32).reshape(-1, 1), (1, hidden_dim))
    return EncoderStates(hidden_states=hidden, frame_seconds=frame_seconds)


def test_pool_span_averages_the_right_frames() -> None:
    states = _states(n_frames=10)  # frames 0..9, frame_seconds=0.02
    # start=0.10 -> frame 5, end=0.14 -> frame 7 -> mean(5,6,7) = 6
    pooled = pool_span(states, 0.10, 0.14)
    assert pooled is not None
    assert np.allclose(pooled, 6.0), pooled


def test_pool_span_clamps_to_valid_frame_range() -> None:
    states = _states(n_frames=10)
    pooled = pool_span(states, -5.0, 100.0)  # wildly out of range both sides
    assert pooled is not None
    assert np.allclose(pooled, np.arange(10).mean()), pooled  # whole clip


def test_pool_span_none_when_start_or_end_missing() -> None:
    states = _states(n_frames=10)
    assert pool_span(states, None, 0.1) is None
    assert pool_span(states, 0.1, None) is None


def test_pool_span_none_for_empty_states() -> None:
    states = EncoderStates(hidden_states=np.zeros((0, 4), dtype=np.float32))
    assert pool_span(states, 0.0, 0.1) is None


def test_cosine_distance_identical_vectors_is_zero() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = cosine_distance(a, a.copy())
    assert d is not None and abs(d - 0.0) < 1e-6, d


def test_cosine_distance_orthogonal_vectors_is_one() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    d = cosine_distance(a, b)
    assert d is not None and abs(d - 1.0) < 1e-6, d


def test_cosine_distance_opposite_vectors_is_two() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    d = cosine_distance(a, b)
    assert d is not None and abs(d - 2.0) < 1e-6, d


def test_cosine_distance_none_on_missing_or_zero_vector() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    zero = np.array([0.0, 0.0], dtype=np.float32)
    assert cosine_distance(None, a) is None
    assert cosine_distance(a, None) is None
    assert cosine_distance(a, zero) is None


def test_fluent_centroid_excludes_ground_truth_tokens() -> None:
    # 4 tokens, each 0.1s wide, frame_seconds=0.02 -> 5 frames/token.
    # frame value == frame index, so token i's pooled value is its own
    # frame range's mean.
    states = _states(n_frames=20)
    tokens = [{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.1} for i, w in enumerate("abcd")]
    ground_truth = {1: "filler"}  # token 1 is disfluent, 0/2/3 are fluent
    centroid = fluent_centroid(states, tokens, ground_truth)
    assert centroid is not None
    # pool_span rounds both ends to the nearest frame and includes both
    # endpoints: token 0 (0.0-0.1s) -> frames 0..round(5.0)=5 (mean 2.5);
    # token 2 (0.2-0.3s) -> frames 10..15 (mean 12.5); token 3 (0.3-0.4s)
    # -> frames 15..round(20.0)=20, clamped to 19 (n_frames=20) -> 15..19
    # (mean 17.0). Centroid of [2.5, 12.5, 17.0].
    expected = np.mean([2.5, 12.5, 17.0])
    assert np.allclose(centroid, expected), (centroid, expected)


def test_fluent_centroid_none_when_every_token_is_disfluent() -> None:
    states = _states(n_frames=10)
    tokens = [{"word": "a", "start": 0.0, "end": 0.1}]
    assert fluent_centroid(states, tokens, {0: "filler"}) is None


def test_attach_encoder_distances_only_touches_scorable_typed_events() -> None:
    states = _states(n_frames=20)
    tokens = [{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.1} for i, w in enumerate("abcd")]
    ground_truth = {1: "filler"}
    events = [
        {"index": 1, "type": "filler", "start": 0.1, "end": 0.2},
        {"index": 2, "type": "prolongation", "start": 0.2, "end": 0.3},  # not in scorable_types
    ]
    out = attach_encoder_distances(states, tokens, ground_truth, events, ("filler",))
    assert "encoder_distance" in out[0], out[0]
    assert out[0]["encoder_distance"] is not None
    assert "encoder_distance" not in out[1], out[1]
    # original list must not be mutated
    assert "encoder_distance" not in events[0]


def test_attach_encoder_distances_prefers_acoustic_span() -> None:
    # _states' frames only vary in magnitude (every dim holds the same
    # value), so any two pooled vectors point in the same direction and
    # cosine distance between them is always 0 regardless of which frames
    # were pooled -- useless for telling spans apart. Use direction-varying
    # frames here instead: dim0 rises with frame index, dim1 falls, so
    # pooling a different frame range genuinely changes the direction.
    n_frames = 20
    dim0 = np.arange(n_frames, dtype=np.float32)
    dim1 = float(n_frames) - dim0
    states = EncoderStates(hidden_states=np.stack([dim0, dim1], axis=1))
    tokens = [{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.1} for i, w in enumerate("abcd")]
    ground_truth = {1: "filler"}
    absurd = {
        "index": 1, "type": "filler", "start": 0.1, "end": 0.2,
        "acoustic_start": 0.0, "acoustic_end": 0.02,  # deliberately different span, must be used
    }
    normal = {"index": 1, "type": "filler", "start": 0.1, "end": 0.2}
    out_absurd = attach_encoder_distances(states, tokens, ground_truth, [absurd], ("filler",))
    out_normal = attach_encoder_distances(states, tokens, ground_truth, [normal], ("filler",))
    assert out_absurd[0]["encoder_distance"] != out_normal[0]["encoder_distance"], (
        out_absurd[0], out_normal[0],
    )


def test_collect_raw_records_shape_and_labels() -> None:
    states = _states(n_frames=20)
    tokens = [{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.1} for i, w in enumerate("abcd")]
    ground_truth = {1: "filler", 2: "word_repetition"}
    events = [
        {"index": 1, "type": "filler", "start": 0.1, "end": 0.2},          # TP
        {"index": 2, "type": "word_repetition", "start": 0.2, "end": 0.3},  # TP, has a partner (idx 1)
        {"index": 3, "type": "word_repetition", "start": 0.3, "end": 0.4},  # FP (idx 3 not in gt)
    ]
    records = collect_raw_records(
        states, "clipA", tokens, ground_truth, events, ("filler", "word_repetition"),
    )
    assert len(records) == 3
    by_index = {r["index"]: r for r in records}
    assert by_index[1]["label"] == 1 and by_index[1]["type"] == "filler"
    assert by_index[1]["partner_embedding"] is None  # filler has no partner concept
    assert by_index[2]["label"] == 1
    assert by_index[2]["partner_embedding"] is not None  # word_repetition, idx-1 exists
    assert by_index[3]["label"] == 0  # FP: index 3 not in ground_truth
    for r in records:
        assert r["clip_id"] == "clipA"
        assert r["embedding"] is not None
        assert r["centroid"] is not None


def test_collect_raw_records_no_partner_at_index_zero() -> None:
    states = _states(n_frames=20)
    tokens = [{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.1} for i, w in enumerate("ab")]
    ground_truth = {0: "word_repetition"}
    events = [{"index": 0, "type": "word_repetition", "start": 0.0, "end": 0.1}]
    records = collect_raw_records(states, "clipB", tokens, ground_truth, events, ("word_repetition",))
    assert len(records) == 1
    assert records[0]["partner_embedding"] is None  # index-1 == -1, no partner


def _run_all() -> int:
    tests = [
        test_pool_span_averages_the_right_frames,
        test_pool_span_clamps_to_valid_frame_range,
        test_pool_span_none_when_start_or_end_missing,
        test_pool_span_none_for_empty_states,
        test_cosine_distance_identical_vectors_is_zero,
        test_cosine_distance_orthogonal_vectors_is_one,
        test_cosine_distance_opposite_vectors_is_two,
        test_cosine_distance_none_on_missing_or_zero_vector,
        test_fluent_centroid_excludes_ground_truth_tokens,
        test_fluent_centroid_none_when_every_token_is_disfluent,
        test_attach_encoder_distances_only_touches_scorable_typed_events,
        test_attach_encoder_distances_prefers_acoustic_span,
        test_collect_raw_records_shape_and_labels,
        test_collect_raw_records_no_partner_at_index_zero,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
