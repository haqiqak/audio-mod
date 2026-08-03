"""fetch_libristutter_audio.py — download the real audio (FLAC) matching an
already-downloaded LibriStutter annotation sample (see
fetch_libristutter_sample.py), so track_a.py can run audio-enabled
evaluation (--audio-dir) — the only way this project's audio-native
detection layer (Silero VAD, Praat, acoustic fusion) has ever been
evaluated against labeled ground truth. See PAPER_DECISION_LOG.md's
"Audio-enabled evaluation" entry (2026-08-03) for why this was prioritized
over acquiring a new dataset (e.g. SEP-28k's audio).

Mirrors the annotation sample's relative paths exactly (SpeakerID/ChapterID/
file.csv -> SpeakerID/ChapterID/file.flac) so loaders.load_libristutter_dir_
with_audio can pair them by path alone.

Usage
─────
    python -m profiling.evaluation.fetch_libristutter_audio \\
        [--annotations-dir DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import requests

_RAW_BASE = "https://raw.githubusercontent.com/hhzhang16/LibriStutterData/main/"
_AUDIO_PREFIX = "LibriStutter%20Audio/"


def fetch(annotations_dir: Path, out_dir: Path) -> tuple[int, int]:
    csvs = sorted(annotations_dir.rglob("*.csv"))
    print(f"{len(csvs)} annotation files to find matching audio for")

    sess = requests.Session()
    ok, failed = 0, 0
    t0 = time.time()
    for i, csv_path in enumerate(csvs):
        rel = csv_path.relative_to(annotations_dir).with_suffix(".flac")
        dest = out_dir / rel
        if dest.exists():
            ok += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _RAW_BASE + _AUDIO_PREFIX + str(rel).replace("\\", "/")
        try:
            r = sess.get(url, timeout=20)
            if r.status_code == 200 and len(r.content) > 0:
                dest.write_bytes(r.content)
                ok += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(csvs)} ok={ok} failed={failed} elapsed={time.time()-t0:.0f}s")

    print(f"Done: {ok} ok, {failed} failed, {time.time()-t0:.0f}s total")
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", default=None,
        help="Directory of already-downloaded annotation CSVs (default: "
             "eval_datasets/libristutter_sample next to this repo).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for audio (default: eval_datasets/libristutter_sample_audio).",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    ann_dir = Path(args.annotations_dir) if args.annotations_dir else \
        root / "eval_datasets" / "libristutter_sample"
    out_dir = Path(args.out) if args.out else root / "eval_datasets" / "libristutter_sample_audio"

    if not ann_dir.exists():
        print(f"Annotations directory does not exist: {ann_dir}")
        print("Run fetch_libristutter_sample.py first.")
        return 2

    ok, failed = fetch(ann_dir, out_dir)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
