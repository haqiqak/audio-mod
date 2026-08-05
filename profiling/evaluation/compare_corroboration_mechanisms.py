"""compare_corroboration_mechanisms.py — VALIDATION.md §12/§12.6: 5-fold,
clip-split cross-validated comparison of (S1,M1), (S1,M3), (S2,M1) —
see those sections for the full pre-registered protocol and reasoning.

Consumes the .npz produced by collect_raw_encoder_data.py — no encoder
pass here, this is pure numpy analysis over already-collected embeddings,
so it runs in seconds/minutes, not hours.

Deliberately implements logistic regression (M3) with plain numpy rather
than adding scikit-learn as a new dependency (§12.3's stated reasoning).
L2-regularized: with a ~1280-dim embedding and roughly a hundred training
events per fold, an *unregularized* fit would be badly underdetermined
(more parameters than samples) — this is a real, stated methodological
choice, not a hyperparameter search.

**Nested cross-validation (VALIDATION.md §12.6, added after §12's first
result used a fixed L2=5.0)**: for each outer fold, the L2 strength is
selected by an inner, clip-split k-fold CV over that fold's own training
data only (never touching the outer test fold), then the model is refit
on the full outer-training set with the selected value before evaluating
on the outer test fold. This directly answers whether §12.5's result was
an artifact of the arbitrary fixed L2=5.0 choice.

Usage
─────
    python -m profiling.evaluation.compare_corroboration_mechanisms \\
        --data eval_results/stage1_raw_embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

N_FOLDS = 5
N_INNER_FOLDS = 3
L2_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
LR = 0.5
EPOCHS = 800


def _load(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _cosine_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise 1 - cosine_similarity(a[i], b[i]). NaN where either row is NaN."""
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    denom_safe = np.where(denom == 0, np.nan, denom)
    sim = np.sum(a * b, axis=1) / denom_safe
    return 1.0 - sim


def _clip_folds(clip_ids: np.ndarray, n_folds: int = N_FOLDS) -> dict[str, int]:
    """Deterministic round-robin fold assignment over sorted unique clip ids
    (VALIDATION.md §12.3 -- no random seed dependency for the split itself)."""
    unique = sorted(set(clip_ids.tolist()))
    return {cid: i % n_folds for i, cid in enumerate(unique)}


def _prf1(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _best_threshold_by_f1(signal: np.ndarray, labels: np.ndarray) -> float:
    """M1: the threshold (>= fires) maximizing F1 on the given (training) data."""
    uniq = np.unique(signal)
    candidates = np.concatenate([[uniq[0] - 1.0], uniq, [uniq[-1] + 1.0]])
    best_f1, best_t = -1.0, candidates[0]
    for t in candidates:
        pred = (signal >= t).astype(int)
        _, _, f1 = _prf1(pred, labels)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _fit_logistic_regression(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    """M3: L2-regularized logistic regression via batch gradient descent."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(EPOCHS):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + (l2 / n) * w
        grad_b = float(np.mean(p - y))
        w -= LR * grad_w
        b -= LR * grad_b
    return w, b


def _select_l2_by_nested_cv(
    embeddings: np.ndarray, labels: np.ndarray, clip_ids: np.ndarray,
) -> float:
    """VALIDATION.md §12.6: pick the L2 strength that maximizes mean F1 over
    an inner, clip-split k-fold CV -- computed entirely within the caller's
    outer-training data, never touching the outer test fold. Falls back to
    the grid's median if there are too few distinct clips to form
    N_INNER_FOLDS non-empty inner folds (can happen on a small outer-train
    set), rather than crashing or silently picking a degenerate split."""
    inner_fold_map = _clip_folds(clip_ids, n_folds=N_INNER_FOLDS)
    inner_fold_ids = np.array([inner_fold_map[c] for c in clip_ids])
    n_distinct_folds = len(set(inner_fold_ids.tolist()))
    if n_distinct_folds < 2:
        return float(np.median(L2_GRID))

    best_l2, best_f1 = L2_GRID[0], -1.0
    for l2 in L2_GRID:
        fold_f1s = []
        for f in range(N_INNER_FOLDS):
            test_mask = inner_fold_ids == f
            train_mask = ~test_mask
            if test_mask.sum() == 0 or train_mask.sum() == 0:
                continue
            X_train, X_test = _standardize(embeddings[train_mask], embeddings[test_mask])
            y_train = labels[train_mask].astype(np.float64)
            if len(set(y_train.tolist())) < 2:
                continue  # can't fit/evaluate meaningfully with one class in inner-train
            w, b = _fit_logistic_regression(X_train, y_train, l2)
            proba = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
            pred = (proba >= 0.5).astype(int)
            _, _, f1 = _prf1(pred, labels[test_mask])
            fold_f1s.append(f1)
        if fold_f1s:
            mean_f1 = float(np.mean(fold_f1s))
            if mean_f1 > best_f1:
                best_f1, best_l2 = mean_f1, l2
    return best_l2


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-8
    return (train - mean) / std, (test - mean) / std


def _cv_threshold(signal: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray) -> list[tuple[float, float, float]]:
    results = []
    for f in range(N_FOLDS):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        t = _best_threshold_by_f1(signal[train_mask], labels[train_mask])
        pred = (signal[test_mask] >= t).astype(int)
        results.append(_prf1(pred, labels[test_mask]))
    return results


def _cv_classifier(
    embeddings: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray, clip_ids: np.ndarray,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Returns (per-fold precision/recall/F1, per-fold selected L2) -- the
    L2 list is reported alongside results so the chosen values are part of
    the permanent record, not just an internal implementation detail."""
    results = []
    chosen_l2s = []
    for f in range(N_FOLDS):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        l2 = _select_l2_by_nested_cv(embeddings[train_mask], labels[train_mask], clip_ids[train_mask])
        chosen_l2s.append(l2)

        X_train, X_test = _standardize(embeddings[train_mask], embeddings[test_mask])
        y_train = labels[train_mask].astype(np.float64)
        w, b = _fit_logistic_regression(X_train, y_train, l2)
        proba = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
        pred = (proba >= 0.5).astype(int)
        results.append(_prf1(pred, labels[test_mask]))
    return results, chosen_l2s


def _summarize(name: str, results: list[tuple[float, float, float]]) -> str:
    if not results:
        return f"{name}: n/a (no valid folds)"
    arr = np.array(results)  # [n_folds, 3] -> precision, recall, f1
    mean = arr.mean(axis=0)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    return (f"{name}: F1={mean[2]:.3f} (range {lo[2]:.3f}-{hi[2]:.3f}), "
            f"P={mean[0]:.3f}, R={mean[1]:.3f}, n_folds={len(results)}")


def run(npz_path: Path) -> None:
    d = _load(npz_path)
    types = d["types"]
    labels = d["labels"]
    clip_ids = d["clip_ids"]
    embeddings = d["embeddings"]
    centroids = d["centroids"]
    partner = d["partner_embeddings"]
    has_partner = d["has_partner"]

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    s1 = _cosine_distance_rows(embeddings, centroids)  # distance-to-fluent-centroid

    print(f"Loaded {len(labels)} events from {npz_path}\n")

    for type_filter, label in [
        (("word_repetition",), "word_repetition"),
        (("word_repetition", "sound_repetition"), "Any (word_repetition + sound_repetition)"),
    ]:
        mask = np.isin(types, type_filter)
        n_pos = int(labels[mask].sum())
        n_neg = int((mask.sum()) - n_pos)
        print(f"=== {label} (n_TP={n_pos}, n_FP={n_neg}) ===")
        if n_pos == 0 or n_neg == 0:
            print("  Cannot cross-validate: one class is empty in this data.\n")
            continue

        m = mask
        print("  " + _summarize("(S1, M1) distance-to-centroid + threshold",
                                 _cv_threshold(s1[m], labels[m], fold_ids[m])))
        m3_results, m3_l2s = _cv_classifier(embeddings[m], labels[m], fold_ids[m], clip_ids[m])
        print("  " + _summarize("(S1, M3) raw embedding + logistic regression (nested-CV L2)",
                                 m3_results))
        print(f"    selected L2 per outer fold: {[round(x, 2) for x in m3_l2s]}")

        partner_mask = m & has_partner
        n_pos_p = int(labels[partner_mask].sum())
        n_neg_p = int(partner_mask.sum() - n_pos_p)
        if n_pos_p > 0 and n_neg_p > 0:
            s2 = _cosine_distance_rows(embeddings[partner_mask], partner[partner_mask])
            print("  " + _summarize(
                f"(S2, M1) repeat-pair self-similarity + threshold (n={partner_mask.sum()}, partner-only subset)",
                _cv_threshold(s2, labels[partner_mask], fold_ids[partner_mask]),
            ))
        else:
            print("  (S2, M1): n/a -- one class empty among events with a partner token")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the .npz from collect_raw_encoder_data.py")
    args = parser.parse_args(argv)
    path = Path(args.data)
    if not path.exists():
        print(f"Not found: {path}")
        return 2
    run(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
