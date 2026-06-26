"""Tests for profiling/acoustic.py — ASR-independent waveform disfluency cues.

No ASR model: audio is built from silence + a low-freq tone. Validates
segmentation and prolongation/block candidate derivation.

    pytest tests/test_acoustic.py
    python tests/test_acoustic.py
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

from profiling.acoustic import (
    AcousticConfig,
    analyze,
    detect_blocks,
    detect_prolongations,
    segment_voiced,
)

SR = 16_000


def _tone(seconds: float, freq: float = 150.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32)


def _wav_bytes(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)
    return buf.getvalue()


CFG = AcousticConfig()


def test_segmentation_splits_voiced_and_silence() -> None:
    samples = _cat(_silence(0.5), _tone(0.8), _silence(0.5))
    segs = segment_voiced(samples, SR, CFG)
    voiced = [s for s in segs if s.voiced]
    assert len(voiced) == 1, [(s.start, s.end, s.voiced) for s in segs]
    v = voiced[0]
    assert abs(v.start - 0.5) < 0.05 and abs(v.end - 1.3) < 0.05, (v.start, v.end)


def test_prolongation_detected_for_long_sustained_tone() -> None:
    samples = _cat(_silence(0.2), _tone(1.2), _silence(0.2))
    segs = segment_voiced(samples, SR, CFG)
    cands = detect_prolongations(segs, CFG)
    assert len(cands) == 1, cands
    c = cands[0]
    assert c.type == "prolongation"
    assert c.duration >= CFG.prolongation_min_seconds
    assert 0.0 < c.confidence <= 0.95


def test_short_tone_is_not_a_prolongation() -> None:
    samples = _cat(_silence(0.2), _tone(0.3), _silence(0.2))  # 0.3s < 0.65s min
    segs = segment_voiced(samples, SR, CFG)
    assert detect_prolongations(segs, CFG) == []


def test_block_detected_only_between_voiced_regions() -> None:
    # voiced — 0.8s silence — voiced  => one block
    samples = _cat(_tone(0.4), _silence(0.8), _tone(0.4))
    segs = segment_voiced(samples, SR, CFG)
    blocks = detect_blocks(segs, CFG)
    assert len(blocks) == 1, blocks
    assert blocks[0].type == "block"
    assert blocks[0].duration >= CFG.block_min_seconds


def test_leading_and_trailing_silence_are_not_blocks() -> None:
    # long silence at the edges must NOT be flagged (not flanked by voicing)
    samples = _cat(_silence(1.0), _tone(0.5), _silence(1.0))
    segs = segment_voiced(samples, SR, CFG)
    assert detect_blocks(segs, CFG) == []


def test_all_silence_yields_no_candidates() -> None:
    res = analyze(_wav_bytes(_silence(2.0)), config=None)
    assert res.prolongations == [] and res.blocks == []


def test_analyze_from_wav_bytes_end_to_end() -> None:
    samples = _cat(_silence(0.2), _tone(1.0), _silence(0.8), _tone(1.0), _silence(0.2))
    res = analyze(_wav_bytes(samples), config=None)
    assert len(res.prolongations) == 2, res.prolongations
    assert len(res.blocks) == 1, res.blocks
    # candidates are time-ordered and serializable
    cds = res.candidates
    assert [c.type for c in cds] == ["prolongation", "block", "prolongation"]
    d = cds[0].to_dict()
    assert d["source"] == "acoustic" and d["type"] == "prolongation"


def test_config_from_detection_cfg() -> None:
    cfg = AcousticConfig.from_detection_cfg({
        "prolongation_min_seconds": 0.9,
        "block_gap_seconds": 0.7,
        "acoustic": {"silence_rms_threshold": 0.02},
    })
    assert cfg.prolongation_min_seconds == 0.9
    assert cfg.block_min_seconds == 0.7
    assert cfg.silence_rms == 0.02


def _run_all() -> int:
    tests = [
        test_segmentation_splits_voiced_and_silence,
        test_prolongation_detected_for_long_sustained_tone,
        test_short_tone_is_not_a_prolongation,
        test_block_detected_only_between_voiced_regions,
        test_leading_and_trailing_silence_are_not_blocks,
        test_all_silence_yields_no_candidates,
        test_analyze_from_wav_bytes_end_to_end,
        test_config_from_detection_cfg,
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
