"""Unit tests for CrisperWhisperASR.last_timing (Step 1a — honest latency).

These verify that last_timing is populated and self-describing WITHOUT loading
the real ~3.2 GB CrisperWhisper model: a stub pipeline is injected into
asr._pipe so _load_pipeline() returns it immediately (its `is not None` guard
short-circuits the real load). The only real I/O is a tiny generated WAV.

Runnable two ways:
    pytest tests/test_asr_timing.py
    python tests/test_asr_timing.py        # falls back to a built-in runner
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.asr import CrisperWhisperASR


def _write_wav(path: Path, seconds: float, sr: int = 16_000) -> None:
    """Write `seconds` of silence as 16 kHz mono int16 WAV."""
    n = int(seconds * sr)
    pcm = np.zeros(n, dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


class _StubPipe:
    """Stands in for the transformers ASR pipeline. Returns one word chunk;
    records that it was called so the test can assert inference actually ran."""

    def __init__(self):
        self.calls = 0

    def __call__(self, audio_path, generate_kwargs=None):
        self.calls += 1
        self.last_generate_kwargs = generate_kwargs
        return {"chunks": [{"text": "hello", "timestamp": (0.0, 0.5)}]}


def _make_asr_with_stub() -> tuple[CrisperWhisperASR, _StubPipe]:
    asr = CrisperWhisperASR(device="cpu")
    stub = _StubPipe()
    asr._pipe = stub  # _load_pipeline() short-circuits on this being non-None
    return asr, stub


def test_last_timing_is_self_describing(tmp_path: Path | None = None) -> None:
    """A 2.0s WAV → last_timing has all five required fields, correctly typed."""
    base = tmp_path or Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    wav = base / "clip_2s.wav"
    _write_wav(wav, seconds=2.0)

    asr, stub = _make_asr_with_stub()
    tokens = asr._transcribe_transformers(wav)

    assert stub.calls == 1, "stub pipeline should have been invoked exactly once"
    assert tokens and tokens[0]["word"] == "hello"

    t = asr.last_timing
    # All Step-1a fields present
    for key in (
        "load_pipeline_seconds",
        "inference_seconds",
        "clip_duration_seconds",
        "max_new_tokens",
        "audio_size_bytes",
    ):
        assert key in t, f"last_timing missing required key: {key}"

    # load is near-zero because the pipeline was pre-cached (stub)
    assert t["load_pipeline_seconds"] >= 0.0
    assert t["load_pipeline_seconds"] < 1.0, "warm/cached load should be near-zero"
    assert t["inference_seconds"] >= 0.0

    # clip duration read from the WAV header (~2.0s)
    assert abs(t["clip_duration_seconds"] - 2.0) < 0.05, t["clip_duration_seconds"]

    # max_new_tokens budget for a 2s clip: int(2*6)+20 = 32 (within 20..256)
    assert t["max_new_tokens"] == 32, t["max_new_tokens"]
    # and the same budget was actually passed down to the pipeline call
    assert stub.last_generate_kwargs == {"max_new_tokens": 32}

    # audio size matches the file actually on disk
    assert t["audio_size_bytes"] == wav.stat().st_size

    assert t["backend"] == "transformers"


def test_max_new_tokens_floor_and_ceiling(tmp_path: Path | None = None) -> None:
    """Very short clip floors at 20; long clip ceilings at 256."""
    base = tmp_path or Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)

    short = base / "clip_short.wav"
    _write_wav(short, seconds=0.2)   # int(0.2*6)+20 = 21 -> but floor is 20; 21>20 so 21
    assert CrisperWhisperASR._max_new_tokens_for(short) == 21

    tiny = base / "clip_tiny.wav"
    _write_wav(tiny, seconds=0.0)    # int(0*6)+20 = 20 (the floor)
    assert CrisperWhisperASR._max_new_tokens_for(tiny) == 20

    long = base / "clip_long.wav"
    _write_wav(long, seconds=60.0)   # int(60*6)+20 = 380 -> clamped to 256
    assert CrisperWhisperASR._max_new_tokens_for(long) == 256


def test_non_wav_duration_is_none_and_budget_falls_back(tmp_path: Path | None = None) -> None:
    """A non-WAV file: clip_duration_seconds is None, max_new_tokens falls back to 256."""
    base = tmp_path or Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    fake_mp3 = base / "clip.mp3"
    fake_mp3.write_bytes(b"not really audio, just bytes")

    assert CrisperWhisperASR._clip_duration_seconds(fake_mp3) is None
    assert CrisperWhisperASR._max_new_tokens_for(fake_mp3) == 256
    assert CrisperWhisperASR._audio_size_bytes(fake_mp3) == fake_mp3.stat().st_size


def _run_all() -> int:
    import tempfile

    tests = [
        test_last_timing_is_self_describing,
        test_max_new_tokens_floor_and_ceiling,
        test_non_wav_duration_is_none_and_budget_falls_back,
    ]
    failures = 0
    for fn in tests:
        try:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
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
