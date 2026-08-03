"""fetch_libristutter_sample.py — download an evenly-distributed sample of
real LibriStutter annotation CSVs (word/timestamp/label, no audio needed for
Track A's timestamp-only mode) from the GitHub mirror
(hhzhang16/LibriStutterData) into eval_datasets/ (gitignored — the data
itself is never committed, only this acquisition script is).

Uses one git-trees API call (recursive listing, counts as a single request
against GitHub's unauthenticated rate limit) rather than listing directories
one at a time, then downloads the sampled files via raw.githubusercontent.com
(a CDN, not subject to that same rate limit).

This produced the sample behind the first real Track A result recorded in
VALIDATION.md §8.2 — see PAPER_DECISION_LOG.md's "Real LibriStutter/SEP-28k
schemas confirmed" entry (2026-08-03) for the full story, including why the
GitHub mirror was used over the dataset's official Borealis Dataverse hosting
(the Dataverse RAR archives need an extraction tool not available in this
environment; the GitHub mirror ships the same annotation files uncompressed).

Usage
─────
    python -m profiling.evaluation.fetch_libristutter_sample [--n N] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import requests

_TREE_URL = "https://api.github.com/repos/hhzhang16/LibriStutterData/git/trees/main?recursive=1"
_RAW_BASE = "https://raw.githubusercontent.com/hhzhang16/LibriStutterData/main/"
_ANNOTATIONS_PREFIX = "LibriStutter Annotations/"


def fetch(n_target: int, out_dir: Path) -> tuple[int, int]:
    print("Listing repository tree (one API call)...")
    tree = requests.get(_TREE_URL, timeout=30).json()
    csvs = sorted(
        t["path"] for t in tree.get("tree", [])
        if t["path"].startswith(_ANNOTATIONS_PREFIX) and t["path"].endswith(".csv")
    )
    if not csvs:
        raise RuntimeError(
            "No annotation CSVs found in the repo tree — the mirror may have "
            "been taken down (its own README says it's a temporary class-"
            "project repo). Fall back to the official Borealis Dataverse "
            "hosting (doi:10.5683/SP3/NKVOGQ) — RAR archives, need an "
            "extraction tool this environment didn't have when this script "
            "was written."
        )

    step = max(1, len(csvs) // n_target)
    sample = csvs[::step][:n_target]
    print(f"{len(csvs)} total annotation files, sampling {len(sample)} (every {step}th)")

    sess = requests.Session()
    ok, failed = 0, 0
    t0 = time.time()
    for i, path in enumerate(sample):
        rel = path[len(_ANNOTATIONS_PREFIX):]
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _RAW_BASE + path.replace(" ", "%20")
        try:
            r = sess.get(url, timeout=15)
            if r.status_code == 200 and r.text.strip():
                dest.write_text(r.text, encoding="utf-8")
                ok += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} ok={ok} failed={failed} elapsed={time.time()-t0:.0f}s")

    print(f"Done: {ok} ok, {failed} failed, {time.time()-t0:.0f}s total")
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500, help="Target sample size (default 500).")
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: eval_datasets/libristutter_sample next to this repo).",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out) if args.out else root / "eval_datasets" / "libristutter_sample"
    ok, failed = fetch(args.n, out_dir)
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
