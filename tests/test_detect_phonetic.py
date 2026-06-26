"""Tests for phonetic near-repetition (short words compared by pronunciation).

Short words use ARPAbet phoneme distance (via phonetic.phonemes / CMU dict);
longer and out-of-vocabulary words keep the spelling metric. No audio.

    pytest tests/test_detect_phonetic.py
    python tests/test_detect_phonetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.detect import detect_disfluencies, _phonetic_similarity

_CFG = {
    "filler_words": ["uh", "um", "er", "erm", "like"],
    "block_gap_seconds": 0.55,
    "prolongation_min_seconds": 0.65,
    "prolongation_percentile": 90,
    "near_repetition_similarity": 0.75,
    "phonetic_short_max_chars": 4,
    "phrase_repetition_min_words": 2,
    "sentence_initial_boost": 0.08,
}


def _toks(words: list[str], step: float = 0.3) -> list[dict]:
    out, t = [], 0.0
    for w in words:
        out.append({"word": w, "start": round(t, 3), "end": round(t + step, 3)})
        t += step
    return out


def _reps(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "repetition"]


def test_phonetic_similarity_helper() -> None:
    assert _phonetic_similarity("no", "know") == 1.0      # both N OW
    assert _phonetic_similarity("be", "bee") == 1.0       # both B IY
    s = _phonetic_similarity("cat", "dog")
    assert s is not None and s < 0.5
    assert _phonetic_similarity("zzqx", "qzzx") is None    # OOV -> None


def test_phonetic_catches_short_soundalike_edit_misses() -> None:
    """'no' then 'know' sound identical but edit-distance similarity is only
    0.5; the phonetic path flags it. (Documents the short-word behaviour.)"""
    events = detect_disfluencies(_toks(["no", "know"]), config=_CFG)
    reps = _reps(events)
    assert len(reps) == 1, reps
    assert "phonetic similarity" in reps[0]["evidence"], reps[0]["evidence"]


def test_long_words_still_use_edit_metric() -> None:
    """Words longer than the short cap keep the spelling metric."""
    events = detect_disfluencies(_toks(["walking", "walkin"]), config=_CFG)
    reps = _reps(events)
    assert len(reps) == 1, reps
    assert "edit similarity" in reps[0]["evidence"], reps[0]["evidence"]


def test_oov_short_words_fall_back_to_edit() -> None:
    """Short but out-of-vocabulary words can't use phonemes -> spelling metric,
    and a dissimilar pair is not flagged (no crash)."""
    events = detect_disfluencies(_toks(["xqz", "qzy"]), config=_CFG)
    assert _reps(events) == []


_DEMO = [
    {"word": "I",         "start": 0.00, "end": 0.18},
    {"word": "I",         "start": 0.18, "end": 0.36, "is_stutter": True},
    {"word": "want",      "start": 0.36, "end": 0.62},
    {"word": "to",        "start": 1.28, "end": 1.45},
    {"word": "uh",        "start": 1.45, "end": 1.72, "is_filler": True},
    {"word": "buy",       "start": 1.72, "end": 2.10},
    {"word": "buy-",      "start": 2.10, "end": 2.31, "is_stutter": True},
    {"word": "something", "start": 2.31, "end": 3.60},
    {"word": "special",   "start": 3.65, "end": 4.45},
]


def test_demo_regression_unchanged() -> None:
    assert len(detect_disfluencies(_DEMO)) == 7


def _run_all() -> int:
    tests = [
        test_phonetic_similarity_helper,
        test_phonetic_catches_short_soundalike_edit_misses,
        test_long_words_still_use_edit_metric,
        test_oov_short_words_fall_back_to_edit,
        test_demo_regression_unchanged,
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
