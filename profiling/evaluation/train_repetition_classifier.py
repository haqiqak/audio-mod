"""train_repetition_classifier.py — trains and saves the final
`word_repetition`/`sound_repetition` corroboration classifier
(VALIDATION.md §12.6.2's decision: (S1, M3) adopted as the shipped
mechanism).

This is NOT a re-run of the cross-validated comparison
(compare_corroboration_mechanisms.py) — that measures generalization by
holding out folds; this script trains the one model that actually ships,
using ALL available labeled data. Final L2 is chosen by one more nested
(clip-split) cross-validation pass over the full dataset, the same
selection procedure §12.6 used per outer fold, just with no outer fold
held out this time since there's no longer a held-out evaluation to
protect — that already happened in compare_corroboration_mechanisms.py.

Output: models/repetition_corroboration_classifier.npz — weights, bias,
standardization mean/std, and training metadata (date, git commit, data
provenance, selected L2, n_events/n_clips). Small (~10s of KB), meant to
be committed to the repo, unlike the huge pretrained CrisperWhisper
weights under .cache/ (gitignored) — see ARCHITECTURE.md for how this
fits the rest of the pipeline.

Usage
─────
    python -m profiling.evaluation.train_repetition_classifier \\
        --data eval_results/stage1_raw_embeddings_250clip.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.evaluation.compare_corroboration_mechanisms import (
    L2_GRID,
    _fit_logistic_regression,
    _load,
    _select_l2_by_nested_cv,
    _standardize,
)

TARGET_TYPES = ("word_repetition", "sound_repetition")
DEFAULT_OUT = _ROOT / "models" / "repetition_corroboration_classifier.npz"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def train(npz_path: Path, out_path: Path) -> Path:
    d = _load(npz_path)
    types, labels, clip_ids, embeddings = d["types"], d["labels"], d["clip_ids"], d["embeddings"]

    mask = np.isin(types, TARGET_TYPES)
    X, y, clips_m = embeddings[mask], labels[mask].astype(np.float64), clip_ids[mask]
    n_tp, n_fp = int(y.sum()), int(len(y) - y.sum())
    print(f"Training on {len(y)} events ({n_tp} TP / {n_fp} FP) from "
          f"{len(set(clips_m.tolist()))} clips.")

    l2 = _select_l2_by_nested_cv(X, y, clips_m)
    print(f"Selected L2 (nested CV over full dataset): {l2}")

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_std = (X - mean) / std
    w, b = _fit_logistic_regression(X_std, y, l2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        weights=w.astype(np.float32),
        bias=np.float32(b),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        l2=np.float32(l2),
        hidden_dim=np.int64(X.shape[1]),
        target_types=np.array(TARGET_TYPES, dtype=object),
        n_events=np.int64(len(y)),
        n_tp=np.int64(n_tp),
        n_fp=np.int64(n_fp),
        n_clips=np.int64(len(set(clips_m.tolist()))),
        trained_at_utc=np.array(datetime.now(timezone.utc).isoformat(), dtype=object),
        git_commit=np.array(_git_commit() or "unknown", dtype=object),
        source_data=np.array(str(npz_path), dtype=object),
        l2_grid=np.array(L2_GRID, dtype=np.float32),
    )
    print(f"Saved: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    path = Path(args.data)
    if not path.exists():
        print(f"Not found: {path}")
        return 2
    train(path, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
