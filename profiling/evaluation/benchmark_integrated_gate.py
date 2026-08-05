"""benchmark_integrated_gate.py — VALIDATION.md §13: the real, honest
detector-level impact of shipping the (S1, M3) repetition-classifier gate
(section 12.6.2's decision), expressed as TypeCounts (TP/FP/FN/TN),
matching this project's standard benchmark format throughout Phase 1/2.

**Reuses already-collected, already-cross-validated predictions rather
than re-running the encoder a fourth time.** Naively applying the final
shipped model (trained on all 250 clips) back to that same data would
give an optimistic, in-sample result -- not what a real user would see.
Instead, this reconstructs each event's *out-of-fold* prediction: the
same 5-fold, clip-split outer CV split `compare_corroboration_
mechanisms.py` already used, where each fold's model only ever predicts
on data it was never trained on. Summed across all 5 held-out folds, this
gives the honest, cross-validated confusion matrix for "what would the
full detector's word_repetition/sound_repetition counts look like with
this gate active" -- without spending another multi-hour encoder run.

Usage
─────
    python -m profiling.evaluation.benchmark_integrated_gate \\
        --data eval_results/stage1_raw_embeddings_250clip.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.evaluation.compare_corroboration_mechanisms import (
    N_FOLDS,
    _clip_folds,
    _fit_logistic_regression,
    _load,
    _select_l2_by_nested_cv,
    _standardize,
)

TARGET_TYPES = ("word_repetition", "sound_repetition")


def _out_of_fold_predictions(
    embeddings: np.ndarray, labels: np.ndarray, clip_ids: np.ndarray,
) -> np.ndarray:
    """For every event, its prediction from the outer fold that held it
    out -- each event is scored by a model that never saw it during
    training, the honest cross-validated estimate."""
    fold_map = _clip_folds(clip_ids, n_folds=N_FOLDS)
    fold_ids = np.array([fold_map[c] for c in clip_ids])
    preds = np.full(len(labels), -1, dtype=int)

    for f in range(N_FOLDS):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        l2 = _select_l2_by_nested_cv(embeddings[train_mask], labels[train_mask], clip_ids[train_mask])
        X_train, X_test = _standardize(embeddings[train_mask], embeddings[test_mask])
        y_train = labels[train_mask].astype(np.float64)
        w, b = _fit_logistic_regression(X_train, y_train, l2)
        proba = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
        preds[test_mask] = (proba >= 0.5).astype(int)

    assert (preds >= 0).all(), "every event should have been in exactly one outer test fold"
    return preds


def run(npz_path: Path) -> None:
    d = _load(npz_path)
    types, labels, clip_ids, embeddings = d["types"], d["labels"], d["clip_ids"], d["embeddings"]

    mask = np.isin(types, TARGET_TYPES)
    y = labels[mask]
    gate_pred = _out_of_fold_predictions(embeddings[mask], y, clip_ids[mask])

    print(f"Loaded {len(y)} word_repetition/sound_repetition events from {npz_path}\n")
    print(f"{'Type':<20}{'Metric':<12}{'Gate OFF':>10}{'Gate ON':>10}")
    print("-" * 52)

    for type_filter, label in [
        (("word_repetition",), "word_repetition"),
        (("sound_repetition",), "sound_repetition"),
        (TARGET_TYPES, "Any (both types)"),
    ]:
        tmask = np.isin(types[mask], type_filter)
        y_t, pred_t = y[tmask], gate_pred[tmask]

        # Gate OFF: every candidate the token-path check produced fires,
        # exactly as it does today with require_repetition_classifier_
        # confirmation=False -- this IS what "TP" (label==1) / "FP"
        # (label==0) already mean in this dataset (VALIDATION.md section
        # 12.2: labels come from scoring the gate-off detector).
        tp_off = int(y_t.sum())
        fp_off = int(len(y_t) - y_t.sum())

        # Gate ON: only events the (out-of-fold) classifier confirms
        # still fire. A confirmed label==1 stays TP; a confirmed
        # label==0 stays FP. A rejected label==1 becomes a new FN
        # (suppressed true positive); a rejected label==0 becomes a
        # true negative (suppressed false positive, no longer counted).
        tp_on = int(((y_t == 1) & (pred_t == 1)).sum())
        fp_on = int(((y_t == 0) & (pred_t == 1)).sum())
        fn_new = int(((y_t == 1) & (pred_t == 0)).sum())

        print(f"{label:<20}{'TP':<12}{tp_off:>10}{tp_on:>10}")
        print(f"{'':<20}{'FP':<12}{fp_off:>10}{fp_on:>10}")
        print(f"{'':<20}{'FN (new, suppressed TP)':<12}{'-':>10}{fn_new:>10}")
        p_off = tp_off / (tp_off + fp_off) if (tp_off + fp_off) else float("nan")
        r_off = 1.0  # by definition here: all label==1 events fire when the gate is off
        p_on = tp_on / (tp_on + fp_on) if (tp_on + fp_on) else float("nan")
        r_on = tp_on / (tp_on + fn_new) if (tp_on + fn_new) else float("nan")
        f1_off = 2 * p_off * r_off / (p_off + r_off) if (p_off + r_off) else float("nan")
        f1_on = 2 * p_on * r_on / (p_on + r_on) if (p_on + r_on) else float("nan")
        print(f"{'':<20}{'Precision':<12}{p_off:>10.3f}{p_on:>10.3f}")
        print(f"{'':<20}{'Recall':<12}{r_off:>10.3f}{r_on:>10.3f}")
        print(f"{'':<20}{'F1':<12}{f1_off:>10.3f}{f1_on:>10.3f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    args = parser.parse_args(argv)
    path = Path(args.data)
    if not path.exists():
        print(f"Not found: {path}")
        return 2
    run(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
