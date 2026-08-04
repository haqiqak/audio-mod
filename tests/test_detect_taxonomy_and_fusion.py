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


def test_word_repetition_sld_tag_by_syllable_count() -> None:
    """PHASE_2_RESEARCH_PLAN.md: a repeated monosyllabic word is tagged
    likely_sld=True (stuttering-like per the clinical SLD/OD literature); a
    repeated polysyllabic word is tagged likely_sld=False (an ordinary
    linguistic-planning disfluency, not stuttering-like). Descriptive
    metadata only — must not change the event's type or confidence."""
    mono = detect_disfluencies(_toks(["her", "her", "name"]), config=_CFG)
    mono_wr = [e for e in mono if e["type"] == "word_repetition"]
    assert len(mono_wr) == 1, mono
    assert mono_wr[0]["syllable_count"] == 1, mono_wr
    assert mono_wr[0]["likely_sld"] is True, mono_wr

    poly = detect_disfluencies(_toks(["happy", "happy", "birthday"]), config=_CFG)
    poly_wr = [e for e in poly if e["type"] == "word_repetition"]
    assert len(poly_wr) == 1, poly
    assert poly_wr[0]["syllable_count"] == 2, poly_wr
    assert poly_wr[0]["likely_sld"] is False, poly_wr


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
    """Same 9-token demo fixture, still 7 events. Updated 2026-08-04 (ROADMAP.md
    item 3, the sound_repetition fragment-ordering fix): the fixture's
    "buy" "buy-" pair (complete word followed by its own fragment) was
    previously misclassified as word_repetition (both sides normalize to
    "buy", caught by the exact-match check before any fragment-specific
    logic ran) — now correctly sound_repetition. Only "I" "I" remains
    word_repetition. See README.md's "Verify it works" walkthrough for the
    full expected badge breakdown."""
    events = detect_disfluencies(_DEMO)
    assert len(events) == 7, f"expected 7, got {len(events)}: {events}"
    word_reps = [e for e in events if e["type"] == "word_repetition"]
    assert len(word_reps) == 1, events
    sound_reps = [e for e in events if e["type"] == "sound_repetition"]
    assert len(sound_reps) == 1, events
    assert not any(e["type"] == "repetition" for e in events), events


def test_sound_repetition_fragment_after_word() -> None:
    """PHASE_2_RESEARCH_PLAN.md / ROADMAP.md item 3: a fragment repeated
    AFTER its complete word ("Rachel" "rachel-", LibriStutter's actual
    reconstruction convention) must be sound_repetition, not swallowed by
    the exact-match word_repetition check — the deeper bug the fix
    addresses, not just the already-covered "before" ordering. The
    fragment's trailing "-" also independently triggers stutter_marker
    (expected, multi-label) — what must NOT appear is word_repetition."""
    events = detect_disfluencies(_toks(["Rachel", "rachel-", "said"]), config=_CFG)
    kinds = {e["type"] for e in events if e["index"] == 1}
    assert "sound_repetition" in kinds, events
    assert "word_repetition" not in kinds, events


def test_sound_repetition_fragment_before_word_still_works() -> None:
    """The original, already-covered ordering must still work after the fix
    (no regression): a genuine partial fragment before its completed word."""
    events = detect_disfluencies(_toks(["str-", "street"]), config=_CFG)
    kinds = [e["type"] for e in events if e["index"] == 1]
    assert kinds == ["sound_repetition"], events


def test_rate_normalized_prolongation_flags_what_percentile_mode_misses() -> None:
    """VALIDATION.md section 9.5: with a short clip (<5 tokens, so the
    default percentile mode falls back to a flat 1.5x floor = 0.975s at
    this config's 0.65s min), an 0.8s token is NOT flagged. Enabling
    use_rate_normalized_prolongation computes a speaking-rate-relative
    threshold instead (~0.42s here: 4 syllables / 1.4s span = ~2.86
    syll/s, 1.2/2.86 ~= 0.42s) and DOES flag it -- same input, different
    mechanism, different (and opposite) result, proving the toggle
    actually changes behavior."""
    toks = [
        {"word": "the",   "start": 0.0, "end": 0.2},
        {"word": "cat",   "start": 0.2, "end": 0.4},
        {"word": "sat",   "start": 0.4, "end": 0.6},
        {"word": "there", "start": 0.6, "end": 1.4},  # 0.8s duration
    ]
    baseline = detect_disfluencies(toks, config=_CFG)
    assert not any(e["type"] == "prolongation" for e in baseline), baseline

    rate_cfg = dict(_CFG, use_rate_normalized_prolongation=True,
                     prolongation_rate_alpha=1.2, prolongation_rate_floor=1.5)
    rated = detect_disfluencies(toks, config=rate_cfg)
    prolongs = [e for e in rated if e["type"] == "prolongation"]
    assert len(prolongs) == 1 and prolongs[0]["index"] == 3, rated


def test_praat_gate_is_graceful_noop_without_audio() -> None:
    """require_praat_stability_for_prolongation must never crash or block
    when there's no audio to analyze (graceful no-op, same principle as
    every other acoustic check) -- duration-only detection still works."""
    toks = [
        {"word": "a",    "start": 0.0, "end": 0.1},
        {"word": "b",    "start": 0.1, "end": 0.2},
        {"word": "c",    "start": 0.2, "end": 0.3},
        {"word": "d",    "start": 0.3, "end": 0.4},
        {"word": "long", "start": 0.4, "end": 2.0},  # well over any threshold
    ]
    cfg = dict(_CFG, require_praat_stability_for_prolongation=True)
    events = detect_disfluencies(toks, config=cfg)  # no audio_bytes
    assert any(e["type"] == "prolongation" and e["index"] == 4 for e in events), events


def test_word_sandwiched_repetition_not_implemented() -> None:
    """A "word-sandwiched repetition" extension (tolerating a single
    non-filler word between a repeat pair) was implemented and benchmarked
    2026-08-04, then REVERTED — measured net harm (Track A Any F1
    0.835->0.793) outweighed its measured benefit. This negative result is
    locked in as a regression test: a coincidental repeat across an
    unrelated intervening word (e.g. two sentences both starting "It")
    must NOT be flagged. See VALIDATION.md section 8.4.4."""
    events = detect_disfluencies(_toks(["Rachel", "Lynde,", "Rachel"]), config=_CFG)
    reps = [e for e in events if e["type"] == "word_repetition" and e["index"] == 2]
    assert len(reps) == 0, events


def _run_all() -> int:
    tests = [
        test_sound_repetition_is_its_own_type,
        test_exact_word_repeat_is_word_repetition_not_sound,
        test_word_repetition_sld_tag_by_syllable_count,
        test_filler_confidence_boosted_when_acoustically_confirmed,
        test_filler_confidence_down_weighted_when_span_is_near_silent,
        test_stutter_marker_same_acoustic_corroboration,
        test_no_audio_filler_confidence_unchanged,
        test_tie_keeps_token_path_event_by_default,
        test_acoustic_candidate_replaces_weaker_token_event_when_weighted_higher,
        test_disabling_a_detector_suppresses_its_events,
        test_demo_fixture_regression_still_7_events_new_taxonomy,
        test_sound_repetition_fragment_after_word,
        test_sound_repetition_fragment_before_word_still_works,
        test_word_sandwiched_repetition_not_implemented,
        test_rate_normalized_prolongation_flags_what_percentile_mode_misses,
        test_praat_gate_is_graceful_noop_without_audio,
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
