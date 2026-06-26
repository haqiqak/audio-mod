"""Tests for generalized phrase-repetition detection (any length, not just 2-3).

No audio — phrase repetition is text-only. Tokens get contiguous timing so no
blocks/prolongations interfere.

    pytest tests/test_detect_phrase.py
    python tests/test_detect_phrase.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.detect import detect_disfluencies

_CFG = {
    "filler_words": ["uh", "um", "er", "erm", "like"],
    "block_gap_seconds": 0.55,
    "prolongation_min_seconds": 0.65,
    "prolongation_percentile": 90,
    "near_repetition_similarity": 0.75,
    "phrase_repetition_min_words": 2,
    "phrase_repetition_max_words": 8,
    "sentence_initial_boost": 0.08,
}


def _toks(words: list[str], step: float = 0.3) -> list[dict]:
    """Contiguous tokens (no gaps -> no blocks, short -> no prolongations)."""
    out = []
    t = 0.0
    for w in words:
        out.append({"word": w, "start": round(t, 3), "end": round(t + step, 3)})
        t += step
    return out


def _phrase_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "repetition" and "phrase" in e["evidence"]]


def test_two_word_phrase_repeat() -> None:
    events = detect_disfluencies(_toks(["i", "want", "i", "want"]), config=_CFG)
    ph = _phrase_events(events)
    assert len(ph) == 1, ph
    assert ph[0]["index"] == 2, ph[0]          # start of the 2nd occurrence
    assert "2-word phrase" in ph[0]["evidence"], ph[0]["evidence"]


def test_four_word_phrase_repeat_now_caught() -> None:
    """The old 2-3 word window missed this; the generalized scan catches it."""
    words = ["please", "pass", "the", "salt", "please", "pass", "the", "salt"]
    events = detect_disfluencies(_toks(words), config=_CFG)
    ph = _phrase_events(events)
    assert len(ph) == 1, ph
    assert ph[0]["index"] == 4, ph[0]
    assert "4-word phrase" in ph[0]["evidence"], ph[0]["evidence"]


def test_no_false_phrase_on_non_repeat() -> None:
    events = detect_disfluencies(_toks(["the", "cat", "sat", "on", "a", "mat"]), config=_CFG)
    assert _phrase_events(events) == []


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
        test_two_word_phrase_repeat,
        test_four_word_phrase_repeat_now_caught,
        test_no_false_phrase_on_non_repeat,
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
