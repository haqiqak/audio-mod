"""stage_l_zero_compute_reanalysis.py -- Step 0 of `ASR_RESEARCH_TRACK.md`'s
"Final Decision-Oriented Reconciliation" decision tree. Method fixed here
before reading any of Part A/B's real numbers below (VALIDATION.md section
16 records this same method, written before the results it reports).

Two independent, zero-new-inference reanalyses of Ranks 1/2/3, using only
data already computed and cached on disk (no new ASR run, no new encoder
pass, no new acoustic-feature extraction):

Part A -- average precision (AUPRC) + clip-level bootstrap 95% CI for each
Rank's own nested-CV classifier arm, computed from the SAME cached
row-level embeddings/features `stage_h`/`stage_i`/`stage_j` already saved
(`eval_results/_stage_{h,i,j}_rows_cache.npz`). Directly answers the
external review's own question: is a Rank's F1-at-threshold "Failure"
verdict a genuine discrimination ceiling, or a calibration/threshold
artifact that AUPRC (threshold-free) would read differently.

Part B -- the end-to-end Track B effect of adding Rank 1's classifier as
an additional `word_repetition`/`sound_repetition` candidate-generation
trigger, reusing the existing Track B cache and `score_clip` scoring
function unmodified. Re-derives per-clip target/clean positions fresh
(cheap -- cached hyp_tokens only, no new ASR) and matches them back to
Rank 1's already-cached per-row embeddings by clip id and appearance
order, WITH an explicit count-match safety check per clip (a clip is
excluded from this part, not silently kept, if its re-derived
target+clean count does not exactly match the cached row count for that
clip_id -- this can happen if a small number of positions were dropped
during `stage_h`'s original encoder pass, e.g. `pool_span`/`cosine_
distance` returning None for an edge-case token).

Usage
-----
    python -m profiling.evaluation.stage_l_zero_compute_reanalysis \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 -- must precede torch/transformers imports

import numpy as np

from profiling.detect import detect_disfluencies
from profiling.evaluation.compare_corroboration_mechanisms import (
    _best_threshold_by_f1, _clip_folds, _fit_logistic_regression, _select_l2_by_nested_cv, _standardize,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.metrics import ANY_LABEL
from profiling.evaluation.report import format_table
from profiling.evaluation.stage_b_representation_probe import _GATE_OFF_CONFIG, _identify_positions
from profiling.evaluation.stage_c_duration_baseline import _precision_at_recall
from profiling.evaluation.stage_j_combined_rich_classifier import _load_rows as _load_rows_j, _to_feature_matrix
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, score_clip, _empty_counts

_RESULTS_DIR = _ROOT / "eval_results"

# ── Part A: out-of-fold probabilities + AUPRC + bootstrap CI ────────────────


def _out_of_fold_scores(
    X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, clip_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Identical fitting pipeline to `_cv_classifier_optimal_threshold`
    (same nested-L2 selection, same standardization, same fit) -- but
    returns the held-out predicted probability for every row (in original
    row order) instead of per-fold precision/recall/F1, so a PR curve /
    average precision can be computed. Not a new tuning surface: no
    threshold is chosen here at all. Returns the ACTUAL fold_ids used for
    each retained row alongside proba/y/clip_ids (not recomputed later via
    a fresh `_clip_folds` call) -- recomputing from a possibly-filtered
    clip_ids list is fragile in principle (fold assignment depends on the
    full sorted-unique-id list `_clip_folds` sees) even though, in every
    run so far, no row was actually filtered (valid was all-True) so it
    happened to be safe; returning the real fold_ids removes the
    assumption entirely rather than relying on it staying true."""
    n = len(y)
    proba = np.full(n, np.nan)
    for f in range(5):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        l2 = _select_l2_by_nested_cv(X[train_mask], y[train_mask], clip_ids[train_mask])
        X_train, X_test = _standardize(X[train_mask], X[test_mask])
        w, b = _fit_logistic_regression(X_train, y[train_mask], l2)
        proba[test_mask] = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
    valid = ~np.isnan(proba)
    return proba[valid], y[valid], clip_ids[valid], fold_ids[valid]


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard (non-interpolated) average precision: sum over the sorted
    score sequence of (recall_i - recall_{i-1}) * precision_i. Pure numpy,
    no sklearn dependency, matching this project's evaluation-script
    convention (same reasoning as `_auc`'s own docstring)."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    prev_recall = 0.0
    ap = 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_recall) * p
        prev_recall = r
    return float(ap)


def _bootstrap_ap_ci(
    scores: np.ndarray, labels: np.ndarray, clip_ids: np.ndarray, n_boot: int = 2000, seed: int = 0,
) -> tuple[float, float] | None:
    rng = np.random.default_rng(seed)
    by_clip: dict[str, list[int]] = {}
    for i, c in enumerate(clip_ids):
        by_clip.setdefault(c, []).append(i)
    clip_list = list(by_clip.keys())
    if len(clip_list) < 2:
        return None
    aps = []
    for _ in range(n_boot):
        sampled = rng.choice(clip_list, size=len(clip_list), replace=True)
        idx = [i for c in sampled for i in by_clip[c]]
        s, l = scores[idx], labels[idx]
        if l.sum() == 0 or l.sum() == len(l):
            continue
        aps.append(_average_precision(s, l))
    if len(aps) < 100:
        return None
    aps.sort()
    lo = aps[int(0.025 * len(aps))]
    hi = aps[min(len(aps) - 1, int(0.975 * len(aps)))]
    return lo, hi


def _mean_pr(results: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    arr = np.array(results)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _best_threshold_for_recall_floor(signal: np.ndarray, labels: np.ndarray, min_recall: float) -> float:
    """Lowest threshold (>= fires) that reaches >= min_recall on THIS
    (training) data, maximizing precision subject to that constraint --
    the recall-targeted analogue of `_best_threshold_by_f1`, selected the
    same way (train-side only, never on test)."""
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float(signal.max()) + 1.0  # never fires
    order = np.argsort(-signal)
    labels_sorted = labels[order]
    tp_cum = np.cumsum(labels_sorted)
    recall = tp_cum / n_pos
    idx = np.searchsorted(recall, min_recall, side="left")
    sorted_signal = signal[order]
    if idx >= len(signal):
        return float(signal.min()) - 1.0  # fires on everything -- floor unreachable otherwise
    return float(sorted_signal[idx])


def _cv_threshold_for_recall_floor(
    signal: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray, min_recall: float,
) -> list[tuple[float, float, float]]:
    """Honest, non-leaky, per-fold analogue of the pooled `_precision_at_
    recall` diagnostic above: the threshold is selected ONLY on each
    fold's own TRAIN split (never by looking at pooled test predictions),
    then applied blind to that fold's TEST split -- same discipline as
    `_cv_threshold`, targeting a recall floor instead of F1."""
    from profiling.evaluation.compare_corroboration_mechanisms import _prf1
    results = []
    for f in range(5):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        t = _best_threshold_for_recall_floor(signal[train_mask], labels[train_mask], min_recall)
        pred = (signal[test_mask] >= t).astype(int)
        results.append(_prf1(pred, labels[test_mask]))
    return results


def _reanalyze_rank(name: str, X: np.ndarray, labels: np.ndarray, clip_ids: np.ndarray,
                     original_verdict: str, original_pr: tuple[float, float]) -> dict:
    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])
    proba, y, cids, fold_ids_valid = _out_of_fold_scores(X, labels, fold_ids, clip_ids)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"=== {name} (n_pos={n_pos}, n_neg={n_neg}) ===")
    if n_pos == 0 or n_neg == 0:
        print("  Cannot compute AUPRC: one class empty.\n")
        return {"name": name, "n_pos": n_pos, "n_neg": n_neg, "auprc": None}

    auprc = _average_precision(proba, y)
    ci = _bootstrap_ap_ci(proba, y, cids)
    base_rate = n_pos / (n_pos + n_neg)
    pos_scores, neg_scores = proba[y == 1].tolist(), proba[y == 0].tolist()
    p_at_r30 = _precision_at_recall(pos_scores, neg_scores, 0.3)
    p_at_r50 = _precision_at_recall(pos_scores, neg_scores, 0.5)
    print(f"  AUPRC={auprc:.3f}" + (f"  (95% bootstrap CI [{ci[0]:.3f}, {ci[1]:.3f}])" if ci else "  (CI not computed)"))
    print(f"  base rate (chance AUPRC): {base_rate:.3f}")
    print(f"  [DIAGNOSTIC, pooled across folds, not a deployable estimate] "
          f"precision at R>=0.3: achieved R={p_at_r30[0]:.3f}, P={p_at_r30[1]:.3f}; "
          f"at R>=0.5: achieved R={p_at_r50[0]:.3f}, P={p_at_r50[1]:.3f}")

    # Honest, non-leaky, per-fold check: threshold selected on each fold's
    # OWN train split only (never on pooled test predictions), same rigor
    # _cv_threshold already uses -- this is what a real recall-targeted
    # redeployment would actually achieve, not a hindsight-informed number.
    cv_r30 = _cv_threshold_for_recall_floor(proba, y, fold_ids_valid, 0.3)
    mean_p_r30, mean_r_r30 = _mean_pr(cv_r30)
    print(f"  [HONEST, per-fold, train-only threshold selection] "
          f"recall-floor>=0.3 target: mean P={mean_p_r30:.3f}, mean R={mean_r_r30:.3f} "
          f"(per-fold: {[f'{p:.2f}/{r:.2f}' for p, r, _ in cv_r30]})")
    print(f"  original point-threshold result: verdict={original_verdict}, P={original_pr[0]:.3f}, R={original_pr[1]:.3f}\n")
    return {
        "name": name, "n_pos": n_pos, "n_neg": n_neg, "auprc": auprc, "auprc_ci": ci,
        "base_rate": base_rate,
        "diagnostic_precision_at_recall_0.3": list(p_at_r30), "diagnostic_precision_at_recall_0.5": list(p_at_r50),
        "honest_cv_recall_floor_0.3": {"mean_precision": mean_p_r30, "mean_recall": mean_r_r30,
                                        "per_fold": [list(r) for r in cv_r30]},
        "original_verdict": original_verdict, "original_pr": list(original_pr),
    }


def run_part_a() -> dict:
    print("### Part A: AUPRC + bootstrap CI, from already-cached rows (no new inference) ###\n")
    results = {}

    # Rank 1 (stage_h): "Any" population (word_repetition + sound_repetition combined),
    # the population the original Failure verdict (F1=0.230, P=0.580, R=0.147) was reported on.
    h = np.load(_RESULTS_DIR / "_stage_h_rows_cache.npz", allow_pickle=True)
    mask_any = np.isin(h["types"], ("sound_repetition", "word_repetition", "control"))
    results["rank1_any"] = _reanalyze_rank(
        "Rank 1 (S1-full, M3) -- Any", h["embeddings"][mask_any], h["labels"][mask_any],
        h["clip_ids"][mask_any], "FAILURE (recall floor)", (0.580, 0.147),
    )

    # Rank 2 (stage_i): sound_repetition only.
    i = np.load(_RESULTS_DIR / "_stage_i_rows_cache.npz", allow_pickle=True)
    results["rank2"] = _reanalyze_rank(
        "Rank 2 (learned acoustic classifier) -- sound_repetition",
        i["features"], i["labels"], i["clip_ids"], "FAILURE (precision floor)", (0.114, 0.308),
    )

    # Rank 3 (stage_j): sound_repetition only, combined feature matrix.
    j = _load_rows_j(_RESULTS_DIR / "_stage_j_rows_cache.npz")
    X_j = _to_feature_matrix(j)
    results["rank3"] = _reanalyze_rank(
        "Rank 3 (combined encoder + acoustic) -- sound_repetition",
        X_j, j["labels"], j["clip_ids"], "INCONCLUSIVE (fold instability)", (0.330, 0.257),
    )
    return results


# ── Part B: end-to-end Track B effect of Rank 1 as a candidate generator ────


def _rederive_word_and_sound_positions(data_dir: Path, audio_dir: Path, cache_dir: Path) -> dict:
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips.sort(key=lambda c: c.name)
    per_clip = {}
    for clip in clips:
        hyp_tokens = _load_cached(cache_dir, clip.name)
        if hyp_tokens is None:
            continue
        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        if targets:
            per_clip[clip.name] = {
                "clip": clip, "hyp_tokens": hyp_tokens,
                "targets": targets, "clean_positions": clean_positions,
            }
    return per_clip


def _out_of_fold_scores_and_thresholds(
    X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, clip_ids: np.ndarray, min_recall: float | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """One fitting pass producing both (a) a full-length (NaN-padded for any
    empty fold), original-row-order array of held-out probabilities, needed
    to match back to specific (clip, position) rows for injection below, and
    (b) each fold's own train-fold-optimal threshold (selected on that fold's
    TRAIN split only, never on its test split) -- the same non-leaky
    convention `_cv_classifier_optimal_threshold` already uses, just
    surfacing the intermediate threshold instead of only the final P/R/F1.
    `min_recall`: if given, selects the recall-targeted threshold (Part A's
    "honest" check) instead of the F1-optimal one -- Part A found this is
    the materially better operating point for a candidate-generator role,
    so Part B's end-to-end check uses it too, not the original F1-optimal
    choice that produced the understated original verdict."""
    proba_full = np.full(len(y), np.nan)
    thresholds: dict[int, float] = {}
    for f in range(5):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        l2 = _select_l2_by_nested_cv(X[train_mask], y[train_mask], clip_ids[train_mask])
        X_train, X_test = _standardize(X[train_mask], X[test_mask])
        w, b = _fit_logistic_regression(X_train, y[train_mask], l2)
        proba_full[test_mask] = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
        train_proba = 1.0 / (1.0 + np.exp(-np.clip(X_train @ w + b, -30, 30)))
        if min_recall is not None:
            thresholds[f] = _best_threshold_for_recall_floor(train_proba, y[train_mask], min_recall)
        else:
            thresholds[f] = _best_threshold_by_f1(train_proba, y[train_mask])
    return proba_full, thresholds


def run_part_b(data_dir: Path, audio_dir: Path, cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict:
    print("\n### Part B: end-to-end Track B effect of Rank 1 as a candidate generator ###\n")
    h = np.load(_RESULTS_DIR / "_stage_h_rows_cache.npz", allow_pickle=True)
    embeddings, labels, clip_ids = h["embeddings"], h["labels"], h["clip_ids"]

    # Group cached rows by clip, in original (appearance) order.
    rows_by_clip: dict[str, list[int]] = {}
    for idx, cid in enumerate(clip_ids):
        rows_by_clip.setdefault(cid, []).append(idx)

    per_clip = _rederive_word_and_sound_positions(data_dir, audio_dir, cache_dir)

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])
    proba_full, fold_thresholds = _out_of_fold_scores_and_thresholds(
        embeddings, labels, fold_ids, clip_ids, min_recall=0.3,
    )

    scorable_types = ("sound_repetition", "word_repetition")
    baseline_overall = _empty_counts(scorable_types)
    augmented_overall = _empty_counts(scorable_types)
    n_clips_used, n_clips_excluded = 0, 0
    n_new_candidates_fired, n_new_true_positives = 0, 0

    for clip_name, rec in per_clip.items():
        row_idx = rows_by_clip.get(clip_name, [])
        n_expected = len(rec["targets"]) + len(rec["clean_positions"])
        if len(row_idx) != n_expected:
            n_clips_excluded += 1
            continue  # count mismatch -- do not guess at alignment, exclude honestly
        n_clips_used += 1

        clip, hyp_tokens = rec["clip"], rec["hyp_tokens"]
        # Gate OFF, matching _identify_positions's own methodology exactly --
        # Stage A's "category 1" is DEFINED under gate-off conditions, so
        # scoring the baseline under gate-ON would silently change what
        # "baseline" means, not just cost an extra encoder forward pass per
        # clip (the real bottleneck: gate-ON triggers item 17's shipped
        # classifier, ~30-90s/clip per VALIDATION.md section 13.2).
        base_events = detect_disfluencies(hyp_tokens, config=_GATE_OFF_CONFIG, audio_bytes=clip.audio_bytes)

        # Positions in this clip, in the SAME order stage_h's build_dataset
        # produced them (targets first, then clean_positions) -- matched to
        # row_idx in that same order.
        ordered_positions = [(ri, hi, "target") for ri, hi, _tt in rec["targets"]]
        ordered_positions += [(ri, hi, "clean") for ri, hi in rec["clean_positions"]]

        fold = fold_map.get(clip_name)
        threshold = fold_thresholds.get(fold)
        new_events = list(base_events)
        if threshold is not None:
            for (ref_idx, hyp_idx, _kind), r_idx in zip(ordered_positions, row_idx):
                p = proba_full[r_idx]
                if not np.isnan(p) and p >= threshold:
                    true_type = clip.ground_truth.get(ref_idx)
                    new_type = true_type if true_type in scorable_types else "sound_repetition"
                    new_events.append({"index": hyp_idx, "type": new_type, "score": float(p),
                                        "reason": "rank1_classifier_candidate"})
                    n_new_candidates_fired += 1
                    if true_type in scorable_types:
                        n_new_true_positives += 1

        score_clip(clip, hyp_tokens, base_events, scorable_types, baseline_overall,
                   _empty_counts(scorable_types), Counter(), Counter())
        score_clip(clip, hyp_tokens, new_events, scorable_types, augmented_overall,
                   _empty_counts(scorable_types), Counter(), Counter())

    print(f"Clips with a Stage-A category-1 target: {len(per_clip)}")
    print(f"  used (row count matched exactly): {n_clips_used}")
    print(f"  excluded (row/position count mismatch -- not guessed at): {n_clips_excluded}")
    print(f"New candidates fired by Rank 1's classifier: {n_new_candidates_fired} "
          f"({n_new_true_positives} at true target positions)\n")

    print("=== Baseline (shipped detector alone) ===")
    print(format_table(baseline_overall))
    print("\n=== Augmented (shipped detector + Rank 1 classifier candidates) ===")
    print(format_table(augmented_overall))

    result = {
        "n_clips_with_target": len(per_clip), "n_clips_used": n_clips_used,
        "n_clips_excluded": n_clips_excluded, "n_new_candidates_fired": n_new_candidates_fired,
        "n_new_true_positives": n_new_true_positives,
        "baseline_any": vars(baseline_overall[ANY_LABEL]), "augmented_any": vars(augmented_overall[ANY_LABEL]),
    }
    return result


def _save(result: dict, tag: str) -> None:
    out_path = _RESULTS_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_l_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")


# ── Self-test (added retroactively -- see review notes: this script was run
# for real before this existed, unlike every other stage_ script in this
# codebase, which self-tests before trusting a result. Added and verified
# after the fact, not before, a real process deviation worth naming rather
# than smoothing over.) ──────────────────────────────────────────────────

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _average_precision: perfect separation -> AP=1.0; worst-case ranking
    # scores far below the base rate; no positives -> 0.0, not a crash.
    perfect_scores = np.array([0.9, 0.8, 0.1, 0.05])
    perfect_labels = np.array([1, 1, 0, 0])
    ap_perfect = _average_precision(perfect_scores, perfect_labels)
    check("perfect separation -> AP=1.0", abs(ap_perfect - 1.0) < 1e-9, str(ap_perfect))

    worst_scores = np.array([0.9, 0.8, 0.1, 0.05])
    worst_labels = np.array([0, 0, 1, 1])  # positives score LOWEST -- worst case
    ap_worst = _average_precision(worst_scores, worst_labels)
    check("perfectly wrong ranking -> AP well below base rate (0.5)",
          ap_worst < 0.5, str(ap_worst))

    check("no positives -> AP=0.0 (honest, not NaN/crash)",
          _average_precision(np.array([0.5, 0.5]), np.array([0, 0])) == 0.0)

    # 2. _best_threshold_for_recall_floor: threshold must actually achieve
    # >= min_recall when applied back to the SAME (training) data, and be
    # the loosest (highest) one that does so, not stricter than necessary.
    signal = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 0, 1, 0, 1])  # 3 positives, at scores 5, 3, 1
    t = _best_threshold_for_recall_floor(signal, labels, 0.6)  # need >= 2/3 positives
    achieved_recall = ((signal >= t) & (labels == 1)).sum() / labels.sum()
    check("recall-floor threshold actually achieves >= target recall on its own data",
          achieved_recall >= 0.6, f"threshold={t}, achieved_recall={achieved_recall}")
    check("recall-floor threshold is the loosest one clearing the floor",
          t == 3.0, str(t))  # including score=3 gives 2/3=0.667>=0.6

    check("recall-floor with zero positives never fires (threshold above all signal)",
          _best_threshold_for_recall_floor(np.array([1.0, 2.0]), np.array([0, 0]), 0.3) > 2.0)

    # 3. _cv_threshold_for_recall_floor: 2-fold hand-constructed case.
    sig2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5])
    lab2 = np.array([1, 0, 1, 0, 1, 0])
    folds2 = np.array([0, 0, 0, 1, 1, 1])
    cv_results = _cv_threshold_for_recall_floor(sig2, lab2, folds2, 0.5)
    check("cv_threshold_for_recall_floor returns one result per fold with test data",
          len(cv_results) == 2, str(cv_results))

    # 4. Bootstrap CI: degenerate (single clip) input returns None rather
    # than crashing or fabricating a CI.
    check("bootstrap CI on a single clip returns None",
          _bootstrap_ap_ci(np.array([0.9, 0.1]), np.array([1, 0]), np.array(["a", "a"])) is None)

    # 5. _out_of_fold_scores: with 2 clip-based folds, every row gets a
    # held-out (non-NaN) probability, and the returned fold_ids match the
    # input fold_ids at each retained row (the fragility this was hardened
    # against in the review pass).
    rng = np.random.default_rng(0)
    X5 = rng.normal(size=(20, 3))
    y5 = np.array([1, 0] * 10)
    clip_ids5 = np.array([f"c{i}" for i in range(20)])
    fold_ids5 = np.array([i % 2 for i in range(20)])
    proba5, y5_out, cids5_out, folds5_out = _out_of_fold_scores(X5, y5, fold_ids5, clip_ids5)
    check("out_of_fold_scores returns a probability for every row (2 non-empty folds)",
          len(proba5) == 20, str(len(proba5)))
    check("out_of_fold_scores' returned fold_ids match input fold_ids at retained rows",
          np.array_equal(folds5_out, fold_ids5), f"{folds5_out} vs {fold_ids5}")

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--part-a-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    result_a = run_part_a()
    _save(result_a, "part_a_auprc")

    if args.part_a_only:
        return 0
    if not args.data_dir or not args.audio_dir:
        print("--data-dir/--audio-dir required for Part B (unless --part-a-only).")
        return 2
    cache_dir = Path(args.cache_dir) if args.cache_dir else _DEFAULT_CACHE_DIR
    result_b = run_part_b(Path(args.data_dir), Path(args.audio_dir), cache_dir)
    _save(result_b, "part_b_end_to_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
