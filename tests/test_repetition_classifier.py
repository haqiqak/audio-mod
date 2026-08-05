"""Tests for profiling/repetition_classifier.py — the trained
word_repetition/sound_repetition corroboration gate (VALIDATION.md
section 12.6.2's decision).

Deliberately does NOT test the real model load path end to end (requires
the real ~3.2GB CrisperWhisper model) — matches tests/test_encoder_
features.py's own split between pure-logic tests here and real-model runs
left to the evaluation/training scripts. Tests the graceful-degradation
paths directly, and the classifier math by injecting hand-constructed
weights/states into an otherwise-untouched instance.

    pytest tests/test_repetition_classifier.py
    python tests/test_repetition_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.encoder_embedding import EncoderStates
from profiling.repetition_classifier import RepetitionClassifierContext


def test_disabled_is_graceful_noop() -> None:
    ctx = RepetitionClassifierContext(audio_bytes=b"fake", enabled=False)
    assert ctx.available is False
    assert ctx.confirms_repetition(0.0, 0.5) is True  # no-op: never blocks


def test_no_audio_is_graceful_noop() -> None:
    ctx = RepetitionClassifierContext(audio_bytes=None, enabled=True)
    assert ctx.available is False
    assert ctx.confirms_repetition(0.0, 0.5) is True


def test_missing_model_file_is_graceful_noop(monkeypatch) -> None:
    import profiling.repetition_classifier as rc
    monkeypatch.setattr(rc, "_MODEL_PATH", Path("/nonexistent/does-not-exist.npz"))
    monkeypatch.setattr(rc, "_weights_cache", None)
    ctx = RepetitionClassifierContext(audio_bytes=b"fake-but-nonempty", enabled=True)
    assert ctx.available is False
    assert ctx.confirms_repetition(0.0, 0.5) is True


def test_missing_start_or_end_is_graceful_noop() -> None:
    ctx = RepetitionClassifierContext(audio_bytes=None, enabled=False)
    # Even if this were somehow "available", missing timing must no-op.
    ctx._weights = {"weights": np.zeros(4), "bias": 0.0,
                     "mean": np.zeros(4), "std": np.ones(4)}
    ctx._states = EncoderStates(hidden_states=np.zeros((10, 4), dtype=np.float32))
    assert ctx.confirms_repetition(None, 0.5) is True
    assert ctx.confirms_repetition(0.0, None) is True


def test_classifier_math_fires_on_positive_logit() -> None:
    ctx = RepetitionClassifierContext(audio_bytes=None, enabled=False)
    hidden = np.tile(np.array([5.0, 5.0], dtype=np.float32), (10, 1))  # every frame = [5,5]
    ctx._states = EncoderStates(hidden_states=hidden, frame_seconds=0.02)
    # mean=[0,0], std=[1,1] -> standardized span vec is [5,5]; weights=[1,1], bias=-1
    # -> z = 5*1 + 5*1 - 1 = 9 -> sigmoid(9) ~= 0.9999 >= 0.5 -> fires
    ctx._weights = {"weights": np.array([1.0, 1.0]), "bias": -1.0,
                     "mean": np.zeros(2), "std": np.ones(2)}
    assert ctx.available is True
    assert ctx.confirms_repetition(0.0, 0.1) is True


def test_classifier_math_blocks_on_negative_logit() -> None:
    ctx = RepetitionClassifierContext(audio_bytes=None, enabled=False)
    hidden = np.tile(np.array([-5.0, -5.0], dtype=np.float32), (10, 1))
    ctx._states = EncoderStates(hidden_states=hidden, frame_seconds=0.02)
    ctx._weights = {"weights": np.array([1.0, 1.0]), "bias": -1.0,
                     "mean": np.zeros(2), "std": np.ones(2)}
    # z = -5 - 5 - 1 = -11 -> sigmoid ~= 0 -> does not confirm
    assert ctx.confirms_repetition(0.0, 0.1) is False


def test_standardization_is_applied() -> None:
    """A span vector equal to the training mean should sit at the decision
    boundary's bias-only term (standardized to 0) -- confirms mean/std are
    actually used, not silently skipped."""
    ctx = RepetitionClassifierContext(audio_bytes=None, enabled=False)
    hidden = np.tile(np.array([100.0, 100.0], dtype=np.float32), (10, 1))
    ctx._states = EncoderStates(hidden_states=hidden, frame_seconds=0.02)
    ctx._weights = {"weights": np.array([1.0, 1.0]), "bias": 0.5,
                     "mean": np.array([100.0, 100.0]), "std": np.array([1.0, 1.0])}
    # standardized = [0,0] -> z = 0.5 -> sigmoid(0.5) ~= 0.62 >= 0.5 -> fires
    assert ctx.confirms_repetition(0.0, 0.1) is True


def _run_all() -> int:
    tests = [
        test_disabled_is_graceful_noop,
        test_no_audio_is_graceful_noop,
        test_missing_start_or_end_is_graceful_noop,
        test_classifier_math_fires_on_positive_logit,
        test_classifier_math_blocks_on_negative_logit,
        test_standardization_is_applied,
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

    # test_missing_model_file_is_graceful_noop needs pytest's monkeypatch
    # fixture -- run it manually here with a tiny hand-rolled substitute
    # so the standalone `python tests/test_X.py` path still covers it.
    class _FakeMonkeypatch:
        def __init__(self):
            self._restore = []
        def setattr(self, obj, name, value):
            self._restore.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        def undo(self):
            for obj, name, value in reversed(self._restore):
                setattr(obj, name, value)

    mp = _FakeMonkeypatch()
    try:
        test_missing_model_file_is_graceful_noop(mp)
        print("PASS  test_missing_model_file_is_graceful_noop")
    except AssertionError as exc:
        failures += 1
        print(f"FAIL  test_missing_model_file_is_graceful_noop: {exc}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        print(f"ERROR test_missing_model_file_is_graceful_noop: {type(exc).__name__}: {exc}")
    finally:
        mp.undo()

    total = len(tests) + 1
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
