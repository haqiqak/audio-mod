"""benchmark_asr.py — honest ASR latency measurement (Step 1b).

Runs the CrisperWhisper pipeline over a folder of WAV files and prints one
clean table:

    File | Duration(s) | Load(s) | Infer(s) | RTF | Tokens

  • Duration — clip length from the WAV header.
  • Load     — CrisperWhisperASR._load_pipeline() time for THIS call. Near-zero
               after the first clip, because the model is cached on the asr
               instance (self._pipe). A non-zero load on the 2nd+ row means the
               model is being rebuilt — investigate.
  • Infer    — pipeline inference time for THIS call.
  • RTF      — real-time factor = Infer ÷ Duration. <1 means faster than real
               time; >1 means slower (expected on CPU for a 3.2 GB model).
  • Tokens   — number of word tokens returned.

All numbers come straight from CrisperWhisperASR.last_timing — this harness
adds no timing of its own beyond reading that dict, so the table reflects what
the app itself measures.

Usage
─────
    python -m profiling.benchmark_asr                 # ./benchmark_clips/*.wav
    python -m profiling.benchmark_asr --clips-dir X   # custom folder
    python -m profiling.benchmark_asr --self-test     # mock model, no 3.2 GB load

The --self-test path verifies the table layout and the RTF math against a stub
pipeline and generated silence WAVs, so the harness format can be trusted
BEFORE spending minutes on a real CPU run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import wave

# Allow `python profiling/benchmark_asr.py` (script form) as well as -m.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.asr import CrisperWhisperASR


@dataclass
class BenchRow:
    filename: str
    clip_duration_seconds: float | None
    load_seconds: float | None
    inference_seconds: float | None
    rtf: float | None
    token_count: int
    error: str | None = None


def _rtf(inference_seconds: float | None, clip_duration_seconds: float | None) -> float | None:
    """Real-time factor = inference ÷ clip duration. None when undefined."""
    if (
        inference_seconds is None
        or clip_duration_seconds is None
        or clip_duration_seconds <= 0
    ):
        return None
    return inference_seconds / clip_duration_seconds


def benchmark_clip(asr: CrisperWhisperASR, path: Path) -> BenchRow:
    """Transcribe one clip and turn asr.last_timing into a BenchRow.

    Errors are captured into the row rather than aborting a whole batch, so one
    bad file doesn't lose the measurements for the others.
    """
    try:
        tokens = asr.transcribe(path)
    except Exception as exc:  # noqa: BLE001
        return BenchRow(
            filename=path.name,
            clip_duration_seconds=CrisperWhisperASR._clip_duration_seconds(path),
            load_seconds=None,
            inference_seconds=None,
            rtf=None,
            token_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    t = asr.last_timing or {}
    clip_dur = t.get("clip_duration_seconds")
    if clip_dur is None:
        clip_dur = CrisperWhisperASR._clip_duration_seconds(path)
    infer = t.get("inference_seconds")
    return BenchRow(
        filename=path.name,
        clip_duration_seconds=clip_dur,
        load_seconds=t.get("load_pipeline_seconds"),
        inference_seconds=infer,
        rtf=_rtf(infer, clip_dur),
        token_count=len(tokens),
    )


def benchmark_folder(asr: CrisperWhisperASR, clips_dir: Path) -> list[BenchRow]:
    """Benchmark every .wav in clips_dir, shortest first (so the first — and
    thus model-load-bearing — row is the cheapest clip to wait on)."""
    wavs = sorted(
        clips_dir.glob("*.wav"),
        key=lambda p: CrisperWhisperASR._clip_duration_seconds(p) or float("inf"),
    )
    rows: list[BenchRow] = []
    for wav in wavs:
        rows.append(benchmark_clip(asr, wav))
    return rows


# ── Table rendering ───────────────────────────────────────────────────────────

_HEADERS = ["File", "Duration(s)", "Load(s)", "Infer(s)", "RTF", "Tokens"]


def _fmt_num(value: float | None, places: int) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _row_cells(row: BenchRow) -> list[str]:
    if row.error:
        return [row.filename, _fmt_num(row.clip_duration_seconds, 2),
                "ERROR", row.error, "—", "—"]
    return [
        row.filename,
        _fmt_num(row.clip_duration_seconds, 2),
        _fmt_num(row.load_seconds, 2),
        _fmt_num(row.inference_seconds, 2),
        _fmt_num(row.rtf, 2),
        str(row.token_count),
    ]


def format_table(rows: list[BenchRow]) -> str:
    """Render rows as a fixed-width text table (header + separator + rows)."""
    body = [_row_cells(r) for r in rows]
    widths = [len(h) for h in _HEADERS]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [line(_HEADERS), "  ".join("-" * w for w in widths)]
    out.extend(line(cells) for cells in body)
    return "\n".join(out)


def _summary(rows: list[BenchRow]) -> str:
    ok = [r for r in rows if r.error is None and r.rtf is not None]
    if not ok:
        return "No successfully-timed clips."
    rtfs = [r.rtf for r in ok]
    loads = [r.load_seconds for r in ok if r.load_seconds is not None]
    first_load = ok[0].load_seconds
    warm_loads = [r.load_seconds for r in ok[1:] if r.load_seconds is not None]
    lines = [
        f"Clips timed: {len(ok)}",
        f"RTF range  : {min(rtfs):.2f}-{max(rtfs):.2f} (inference / clip duration)",
    ]
    if first_load is not None:
        lines.append(f"First-clip load: {first_load:.2f}s")
    if warm_loads:
        lines.append(
            f"Warm-clip load: {max(warm_loads):.2f}s max over {len(warm_loads)} "
            f"later clip(s) - should be near-zero if the model is cached"
        )
    return "\n".join(lines)


# ── Self-test (mock model — no 3.2 GB load) ───────────────────────────────────

class _StubPipe:
    def __init__(self, n_tokens: int = 3):
        self.n_tokens = n_tokens

    def __call__(self, audio_path, generate_kwargs=None):
        chunks = [
            {"text": f"w{i}", "timestamp": (float(i) * 0.3, float(i) * 0.3 + 0.25)}
            for i in range(self.n_tokens)
        ]
        return {"chunks": chunks}


def _write_silence_wav(path: Path, seconds: float, sr: int = 16_000) -> None:
    import numpy as np
    pcm = np.zeros(int(seconds * sr), dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)


def run_self_test() -> int:
    """Verify table/math with a stub model + generated WAVs. No real model."""
    import tempfile

    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _rtf math
    check("rtf basic", _rtf(30.0, 3.0) == 10.0, str(_rtf(30.0, 3.0)))
    check("rtf zero-duration is None", _rtf(5.0, 0.0) is None)
    check("rtf none-inference is None", _rtf(None, 3.0) is None)

    # 2. benchmark_clip wires last_timing → row, RTF consistent with fields
    with tempfile.TemporaryDirectory() as d:
        clips = Path(d)
        _write_silence_wav(clips / "b_long.wav", 4.0)
        _write_silence_wav(clips / "a_short.wav", 1.5)

        asr = CrisperWhisperASR(device="cpu")
        asr._pipe = _StubPipe(n_tokens=3)  # short-circuits the real model load

        rows = benchmark_folder(asr, clips)
        check("two rows produced", len(rows) == 2, str(len(rows)))
        # shortest-first ordering
        check("sorted shortest-first", rows[0].filename == "a_short.wav",
              rows[0].filename)
        for r in rows:
            check(f"{r.filename}: no error", r.error is None, str(r.error))
            check(f"{r.filename}: token_count==3", r.token_count == 3,
                  str(r.token_count))
            # RTF must equal inference ÷ duration for the row's own numbers
            expected = r.inference_seconds / r.clip_duration_seconds
            check(f"{r.filename}: rtf == infer/duration",
                  r.rtf is not None and abs(r.rtf - expected) < 1e-9,
                  f"{r.rtf} vs {expected}")
        # duration came from the WAV header
        short = next(r for r in rows if r.filename == "a_short.wav")
        check("short clip duration ~1.5s",
              abs(short.clip_duration_seconds - 1.5) < 0.05,
              str(short.clip_duration_seconds))

        # 3. table renders with headers + a row per clip
        table = format_table(rows)
        for h in _HEADERS:
            check(f"table has header {h!r}", h in table)
        check("table has both files",
              "a_short.wav" in table and "b_long.wav" in table)
        print("\n--- sample table ---")
        print(table)
        print("\n--- sample summary ---")
        print(_summary(rows))

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark CrisperWhisper ASR latency.")
    parser.add_argument(
        "--clips-dir",
        default=str(_ROOT / "benchmark_clips"),
        help="Folder of .wav clips to benchmark (default: ./benchmark_clips/).",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="ASR device (default: cpu).",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Verify table/RTF math with a mock model — no real 3.2 GB load.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    clips_dir = Path(args.clips_dir)
    if not clips_dir.exists():
        print(f"Clips folder does not exist: {clips_dir}")
        print("Create it and drop a few real .wav recordings (e.g. ~3s, ~8s, ~15s) in,")
        print("then re-run. Or use --self-test to verify the harness without the model.")
        return 2
    wavs = list(clips_dir.glob("*.wav"))
    if not wavs:
        print(f"No .wav files found in: {clips_dir}")
        print("Drop a few real recordings of different lengths (~3s, ~8s, ~15s) and re-run.")
        return 2

    print(f"Benchmarking {len(wavs)} clip(s) in {clips_dir} on device={args.device} …")
    print("(First clip includes the one-time model load — this is slow on CPU.)\n")
    asr = CrisperWhisperASR(device=args.device)
    rows = benchmark_folder(asr, clips_dir)
    print(format_table(rows))
    print()
    print(_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
