"""Tests for acoustic→token fusion inside detect_disfluencies.

Verifies the acoustic detector (profiling/acoustic.py) is merged into the live
detector when audio is present: it catches a sustained sound the token path
missed, it does NOT double-count what the token path already found, and it does
nothing without audio.

No ASR model — WAV bytes built from silence + a 150 Hz tone.

    pytest tests/test_detect_fusion.py
    python tests/test_detect_fusion.py
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


def test_acoustic_catches_sustain_in_a_gap_with_no_token() -> None:
    """A 1.4s sustained sound sits in the gap between two short words — no token
    covers it, so the token path can't flag it. Acoustic fusion should, and
    attribute it to the following word."""
    tokens = [
        {"word": "go",  "start": 0.0, "end": 0.4},
        {"word": "now", "start": 2.4, "end": 2.8},
    ]
    audio = _wav_bytes([
        _tone(0.4),       # 0.0-0.4  "go"
        _silence(0.2),    # 0.4-0.6
        _tone(1.4),       # 0.6-2.0  sustained sound in the gap (no token)
        _silence(0.4),    # 2.0-2.4
        _tone(0.4),       # 2.4-2.8  "now"
    ])
    events = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio)
    prolong = [e for e in events if e["type"] == "prolongation"]
    assert len(prolong) == 1, prolong
    assert prolong[0].get("source") == "acoustic", prolong[0]
    # attributed to the word after the region ("now", index 1)
    assert prolong[0]["index"] == 1, prolong[0]


def test_no_double_count_when_token_path_already_flags_it() -> None:
    """When the token path already flags the prolongation, the overlapping
    acoustic candidate is suppressed — exactly one event, not two."""
    tokens = [
        {"word": "i",    "start": 0.0, "end": 0.4},
        {"word": "waaant", "start": 0.6, "end": 2.2},  # genuinely long voiced word
    ]
    audio = _wav_bytes([
        _tone(0.4),       # 0.0-0.4  "i"
        _silence(0.2),    # 0.4-0.6
        _tone(1.6),       # 0.6-2.2  "waaant" sustained
    ])
    events = detect_disfluencies(tokens, config=_CFG, audio_bytes=audio)
    prolong = [e for e in events if e["type"] == "prolongation"]
    assert len(prolong) == 1, prolong
    # it's the token-path event (carries voiced_duration), not the acoustic one
    assert prolong[0].get("source") != "acoustic", prolong[0]
    assert "voiced_duration" in prolong[0], prolong[0]


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


def test_no_audio_no_acoustic_events() -> None:
    """Without a waveform, fusion is skipped entirely — demo stays 7 events,
    none from the acoustic source."""
    events = detect_disfluencies(_DEMO)
    assert len(events) == 7, len(events)
    assert all(e.get("source") != "acoustic" for e in events)


def _run_all() -> int:
    tests = [
        test_acoustic_catches_sustain_in_a_gap_with_no_token,
        test_no_double_count_when_token_path_already_flags_it,
        test_no_audio_no_acoustic_events,
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
