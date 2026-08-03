"""Tests for the 2026-08 restructuring of profiling/detect.py:

  1. Taxonomy split — sound_repetition is now a distinct type from
     word_repetition (previously both were the generic "repetition").
  2. Acoustic corroboration for filler / stutter_marker — previously these
     trusted the ASR flag with zero audio grounding; now a voiced-energy
     check adjusts confidence when audio is available.
  3. Weighted-confidence fusion — an acoustic-native candidate only replaces
     a token-path event when it is MORE confident (after fusion_weights.
     acoustic scaling), not on any overlap. Verified directly by forcing the
     crossover via config rather than hunting for a naturally-occurring one.
  4. The config-driven `detectors` enable-list actually gates behaviour.

No ASR model — WAV bytes built from silence + a 150 Hz tone, matching the
style of the existing tests/test_detect_*.py suite.

    pytest tests/test_detect_taxonomy_and_fusion.py
    python tests/test_detect_taxonomy_and_fusion.py
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.detect import detect_disfluencies

SR = 16_000


def _tone(seconds: float, freq: float = 150.0, amp: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.int16)


def _wav_bytes(parts: list[np.ndarray]) -> bytes:
    pcm = np.concatenate(parts).astype(np.int16).tobytes() if parts else b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    return buf.getvalue()


_CFG = {
    "filler_words": ["uh", "um", "er", "erm", "like"],
    "block_gap_seconds": 0.55,
    "prolongation_min_seconds": 0.65,
    "prolongation_percentile": 90,
    "near_repetition_similarity": 0.75,
    "phrase_repetition_min_words": 2,
    "sentence_initial_boost": 0.08,
    "acoustic": {
        "silence_rms_threshold": 0.015,
        "voiced_rms_threshold": 0.030,
        "voiced_zcr_threshold": 0.15,
    },
}


def _toks(words: list[str], step: float = 0.3) -> list[dict]:
    out, t = [], 0.0
    for w in words:
        out.append({"word": w, "start": round(t, 3), "end": round(t + step, 3)})
        t += step
    return out


# ── 1. Taxonomy split ─────────────────────────────────────────────────────────

def test_sound_repetition_is_its_own_type() -> None:
    """A sub-word fragment ('str-') repeated before the full word it fragments
    is sound-level, not word-level — must NOT be classed as word_repetition."""
    events = detect_disfluencies(_toks(["str-", "street"]), config=_CFG)
    kinds = {e["type"] for e in events}
    assert "sound_repetition" in kinds, events
    assert "word_repetition" not in kinds, events


def test_exact_word_repeat_is_word_repetition_not_sound() -> None:
    events = detect_disfluencies(_toks(["the", "the", "cat"]), config=_CFG)
    kinds = [e["type"] for e in events if e["index"] == 1]
    assert kinds == ["word_repetition"], events


# ── 2. Acoustic corroboration for filler / stutter_marker ─────────────────────

def test_filler_confidence_boosted_when_acoustically_confirmed() -> None:
    tokens = [{"word": "uh", "start": 0.0, "end": 0.3, "is_filler": True}]
    audio = _wav_bytes([_tone(0.3)])  # real voiced energy under the filler
    events = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio)
    fillers = [e for e in events if e["type"] == "filler"]
    assert len(fillers) == 1
    assert fillers[0]["confidence"] > 0.90, fillers[0]
    assert "acoustic-confirmed" in fillers[0]["evidence"], fillers[0]


def test_filler_confidence_down_weighted_when_span_is_near_silent() -> None:
    """A word the ASR marked as a filler but which is actually near-silence
    (a known ASR mistag/hallucination failure mode) should score lower than
    the no-audio baseline, not the same as a genuinely voiced filler."""
    tokens = [{"word": "uh", "start": 0.0, "end": 0.3, "is_filler": True}]
    audio = _wav_bytes([_silence(0.3)])
    events = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio)
    fillers = [e for e in events if e["type"] == "filler"]
    assert len(fillers) == 1
    assert fillers[0]["confidence"] < 0.90, fillers[0]
    assert "possible ASR misfire" in fillers[0]["evidence"], fillers[0]


def test_stutter_marker_same_acoustic_corroboration() -> None:
    tokens = [{"word": "b-", "start": 0.0, "end": 0.2, "is_stutter": True}]
    audio_voiced = _wav_bytes([_tone(0.2)])
    audio_silent = _wav_bytes([_silence(0.2)])
    ev_voiced = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio_voiced)
    ev_silent = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio_silent)
    conf_voiced = next(e["confidence"] for e in ev_voiced if e["type"] == "stutter_marker")
    conf_silent = next(e["confidence"] for e in ev_silent if e["type"] == "stutter_marker")
    assert conf_voiced > 0.85 > conf_silent, (conf_voiced, conf_silent)


def test_no_audio_filler_confidence_unchanged() -> None:
    """Without audio, behaviour must be byte-for-byte the original 0.90/0.85 —
    no regression to the timestamp-only path. Tokens start at index 1 (not 0)
    so the sentence-initial confidence boost doesn't confound the comparison."""
    tokens = [
        {"word": "well", "start": 0.0, "end": 0.2},
        {"word": "uh", "start": 0.2, "end": 0.5, "is_filler": True},
        {"word": "b-", "start": 0.5, "end": 0.7, "is_stutter": True},
    ]
    events = detect_disfluencies(tokens, config=_CFG)
    filler = next(e for e in events if e["type"] == "filler")
    stutter = next(e for e in events if e["type"] == "stutter_marker")
    assert filler["confidence"] == 0.9
    assert stutter["confidence"] == 0.85


# ── 3. Weighted-confidence fusion ──────────────────────────────────────────────

_FUSION_TOKENS = [
    {"word": "i", "start": 0.0, "end": 0.4},
    {"word": "waaant", "start": 0.6, "end": 2.2},
]
_FUSION_AUDIO = _wav_bytes([_tone(0.4), _silence(0.2), _tone(1.6)])


def test_tie_keeps_token_path_event_by_default() -> None:
    """Baseline (fusion_weights.acoustic == 1.0, the default): the token-path
    event is kept on a tie, since it carries word-level grounding the
    acoustic-only candidate doesn't have on its own."""
    events = detect_disfluencies(_FUSION_TOKENS, config=_CFG, audio_bytes=_FUSION_AUDIO)
    prolong = [e for e in events if e["type"] == "prolongation"]
    assert len(prolong) == 1, prolong
    assert prolong[0].get("source") != "acoustic", prolong[0]
    assert "voiced_duration" in prolong[0], prolong[0]


def test_acoustic_candidate_replaces_weaker_token_event_when_weighted_higher() -> None:
    """With fusion_weights.acoustic raised, the acoustic-native candidate's
    confidence is scaled up until it strictly exceeds the token-path event's
    — at that point it must replace it (still exactly one event, no double
    counting), proving the fusion is genuinely weighted rather than a fixed
    'token always wins' priority."""
    cfg = dict(_CFG, fusion_weights={"rule": 1.0, "acoustic": 10.0})
    events = detect_disfluencies(_FUSION_TOKENS, config=cfg, audio_bytes=_FUSION_AUDIO)
    prolong = [e for e in events if e["type"] == "prolongation"]
    assert len(prolong) == 1, prolong
    assert prolong[0].get("source") == "acoustic", prolong[0]
    assert "voiced_duration" not in prolong[0], prolong[0]


# ── 4. Config-driven detector enable-list ──────────────────────────────────────

def test_disabling_a_detector_suppresses_its_events() -> None:
    tokens = [
        {"word": "I",         "start": 0.00, "end": 0.18},
        {"word": "I",         "start": 0.18, "end": 0.36, "is_stutter": True},
        {"word": "want",      "start": 0.36, "end": 0.62},
    ]
    cfg_all = dict(_CFG)
    cfg_no_word_rep = dict(_CFG, detectors=[
        "filler", "stutter_marker", "phrase_repetition",
        "sound_repetition", "block", "prolongation", "acoustic_fusion",
    ])  # word_repetition deliberately omitted
    with_it = detect_disfluencies(tokens, config=cfg_all)
    without_it = detect_disfluencies(tokens, config=cfg_no_word_rep)
    assert any(e["type"] == "word_repetition" for e in with_it)
    assert not any(e["type"] == "word_repetition" for e in without_it)
    # everything else (the stutter marker on index 1) is untouched
    assert any(e["type"] == "stutter_marker" for e in without_it)


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


def test_demo_fixture_regression_still_7_events_new_taxonomy() -> None:
    """Same 9-token demo fixture, still 7 events — only the type labels for
    the two repetitions changed (repetition -> word_repetition), not the
    count. See README.md's "Verify it works" walkthrough for the full
    expected badge breakdown under the new taxonomy."""
    events = detect_disfluencies(_DEMO)
    assert len(events) == 7, f"expected 7, got {len(events)}: {events}"
    reps = [e for e in events if e["type"] == "word_repetition"]
    assert len(reps) == 2, events
    assert not any(e["type"] == "repetition" for e in events), events


def _run_all() -> int:
    tests = [
        test_sound_repetition_is_its_own_type,
        test_exact_word_repeat_is_word_repetition_not_sound,
        test_filler_confidence_boosted_when_acoustically_confirmed,
        test_filler_confidence_down_weighted_when_span_is_near_silent,
        test_stutter_marker_same_acoustic_corroboration,
        test_no_audio_filler_confidence_unchanged,
        test_tie_keeps_token_path_event_by_default,
        test_acoustic_candidate_replaces_weaker_token_event_when_weighted_higher,
        test_disabling_a_detector_suppresses_its_events,
        test_demo_fixture_regression_still_7_events_new_taxonomy,
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
