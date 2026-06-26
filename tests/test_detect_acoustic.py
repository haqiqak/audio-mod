"""Tests for §1 Option A — acoustic cross-validation of word timestamps.

Covers the leading-silence-as-prolongation bug and its coupled percentile
poisoning: when audio is available, a word's prolongation duration is its VOICED
extent (silent edges trimmed), so clip-initial silence the ASR bills to the first
word neither flags that word nor inflates the clip-wide prolongation threshold.

No ASR model is needed — tests build WAV bytes directly (silence vs. a low-freq
tone) and matching token dicts.

    pytest tests/test_detect_acoustic.py
    python tests/test_detect_acoustic.py
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

from profiling.detect import detect_disfluencies, _AcousticContext

SR = 16_000


def _tone(seconds: float, freq: float = 150.0, amp: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.int16)


def _wav_bytes(segments: list[np.ndarray]) -> bytes:
    pcm = np.concatenate(segments).astype(np.int16).tobytes() if segments else b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    return buf.getvalue()


# Detection config matching config.yaml defaults (explicit for determinism).
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


def test_voiced_span_trims_leading_silence() -> None:
    """voiced_duration recovers the real voiced extent, not the padded span."""
    audio = _wav_bytes([_silence(1.0), _tone(0.5)])  # 1.0s silence + 0.5s tone
    ac = _AcousticContext(audio, _CFG)
    assert ac.available

    # Word nominally spans 0.0..1.5 but only 1.0..1.5 is voiced.
    vdur = ac.voiced_duration(0.0, 1.5)
    assert vdur is not None
    assert abs(vdur - 0.5) < 0.06, vdur  # ~0.5s of tone, within a frame or two

    # A fully-silent span trims to ~zero.
    assert ac.voiced_duration(0.0, 1.0) < 0.06

    # No audio -> None (graceful).
    assert _AcousticContext(None, _CFG).voiced_duration(0.0, 1.5) is None


def _leading_silence_scenario() -> tuple[list[dict], bytes]:
    """First word carries 1.2s of clip-initial silence; a later word ('ssssun')
    is a genuine ~1.2s prolongation. Audio matches the voiced regions."""
    tokens = [
        {"word": "I",      "start": 0.00, "end": 1.38},  # voiced only 1.20-1.38
        {"word": "saw",    "start": 1.38, "end": 1.62},
        {"word": "the",    "start": 1.62, "end": 1.86},
        {"word": "big",    "start": 1.86, "end": 2.10},
        {"word": "ssssun", "start": 2.10, "end": 3.30},  # genuine prolongation
        {"word": "today",  "start": 3.30, "end": 3.60},
    ]
    audio = _wav_bytes([
        _silence(1.20),            # 0.00-1.20  leading silence (billed to "I")
        _tone(0.18),               # 1.20-1.38  "I"
        _tone(0.24),               # 1.38-1.62  "saw"
        _tone(0.24),               # 1.62-1.86  "the"
        _tone(0.24),               # 1.86-2.10  "big"
        _tone(1.20),               # 2.10-3.30  "ssssun" (sustained)
        _tone(0.30),               # 3.30-3.60  "today"
    ])
    return tokens, audio


def test_leading_silence_not_flagged_and_real_prolongation_survives() -> None:
    tokens, audio = _leading_silence_scenario()
    events = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio)
    prolongations = {e["index"] for e in events if e["type"] == "prolongation"}

    # "I" (index 0) is mostly leading silence -> must NOT be a prolongation.
    assert 0 not in prolongations, f"leading-silence word wrongly flagged: {events}"
    # "ssssun" (index 4) is a genuine sustained sound -> must be flagged.
    assert 4 in prolongations, f"real prolongation missed: {events}"


def test_audio_fixes_percentile_poisoning_that_timestamps_alone_cannot() -> None:
    """The crux of Option A: clip-initial silence on "I" inflates the percentile
    threshold and suppresses the genuine prolongation on "ssssun". With the audio
    waveform we trim that silence and recover the real prolongation; with only
    timestamps we can't, so it stays suppressed. This contrast IS the fix."""
    tokens, audio = _leading_silence_scenario()

    # With audio: silence trimmed -> threshold not poisoned -> "ssssun" flagged.
    with_audio = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio)
    assert any(e["type"] == "prolongation" and e["index"] == 4 for e in with_audio)
    ev = next(e for e in with_audio if e["index"] == 4 and e["type"] == "prolongation")
    assert "voiced_duration" in ev and abs(ev["voiced_duration"] - 1.20) < 0.1, ev

    # Without audio: "I"'s raw 1.38s poisons the percentile; the real one is missed.
    no_audio = detect_disfluencies(tokens, config=_CFG, audio_bytes=None)
    assert not any(e["type"] == "prolongation" and e["index"] == 4 for e in no_audio), (
        "timestamp-only mode unexpectedly avoided the percentile poisoning; "
        "the with/without-audio contrast no longer demonstrates the fix"
    )


def test_no_audio_is_graceful() -> None:
    """Timestamp-only mode must not crash and still detects non-duration events
    (e.g. the stutter marker / repetition on the demo fixture path)."""
    tokens, _ = _leading_silence_scenario()
    events = detect_disfluencies(tokens, config=_CFG, audio_bytes=None)
    assert isinstance(events, list)  # no exception, returns a result


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


def test_demo_fixture_regression_unchanged() -> None:
    """The fix must not touch the no-audio fixture path: still 7 events."""
    events = detect_disfluencies(_DEMO)
    assert len(events) == 7, f"expected 7, got {len(events)}: {events}"


def _run_all() -> int:
    tests = [
        test_voiced_span_trims_leading_silence,
        test_leading_silence_not_flagged_and_real_prolongation_survives,
        test_audio_fixes_percentile_poisoning_that_timestamps_alone_cannot,
        test_no_audio_is_graceful,
        test_demo_fixture_regression_unchanged,
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
