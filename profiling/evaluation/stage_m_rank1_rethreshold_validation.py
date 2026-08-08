"""stage_m_rank1_rethreshold_validation.py -- pre-registered in
VALIDATION.md section 17 ("Rank 1 re-thresholding follow-up validation").
Read that section before changing any logic here.

Focused, decision-closing follow-up on the Rank 1 re-thresholding proposal
(section 16's finding: a recall-targeted threshold reaches mean P=0.783 at
mean R=0.289, vs. the original F1-optimal point's P=0.580/R=0.147). Answers
five questions, all from already-cached data -- no new ASR run, no new
encoder pass, no new dependency:

1. Fold stability -- are the recall-targeted threshold VALUES (not just
   outcome P/R, which section 16 already reported) similar across folds,
   or is the result driven by one or two atypical folds? Includes a
   leave-one-fold-out jackknife on mean precision/recall.
2. Does the operating point defensibly approach R>=0.3 without an
   excessive precision cost, and is that a trade-off or a simultaneous
   improvement over the original F1-optimal point?
3. Corrects the Any-label/type issue in the original Part B (section
   16.3): that run defaulted an unlabeled false-positive firing to
   `sound_repetition`, an arbitrary, unjustified choice. This script
   separates true positives (true type known from ground truth, validly
   attributable) from false positives (no true type exists, not
   fabricated -- tracked as a separate, type-unattributable count).
4. Re-runs the end-to-end Track B measurement with the corrected
   type-handling, same 55-clip matched population, same threshold rule.
5. Fits ONE threshold on the FULL dataset (no held-out split, explicitly
   labeled in-sample/not a valid performance estimate) to answer "what
   would the deployable threshold value actually be," separate from the
   5 per-fold values that exist only to estimate out-of-sample
   performance.

Usage
-----
    python -m profiling.evaluation.stage_m_rank1_rethreshold_validation \\
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
    _clip_folds, _fit_logistic_regression, _prf1, _select_l2_by_nested_cv, _standardize,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.metrics import ANY_LABEL
from profiling.evaluation.report import format_table
from profiling.evaluation.stage_b_representation_probe import _GATE_OFF_CONFIG, _identify_positions
from profiling.evaluation.stage_l_zero_compute_reanalysis import (
    _best_threshold_for_recall_floor, _cv_threshold_for_recall_floor_with_values,
    _out_of_fold_scores, _rederive_word_and_sound_positions,
)
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _empty_counts, score_clip

_RESULTS_DIR = _ROOT / "eval_results"
MIN_RECALL = 0.3


# ── Q1/Q2: per-fold threshold values + jackknife stability ──────────────────


# Published in eval_results/20260807T193221_stage_h_candidate_generation_
# classifier.json, "Any (word_repetition + sound_repetition)" population,
# "chosen_l2s" field -- one value per outer fold (0..4), in fold order.
# Reuse is EXACT, not an approximation: _select_l2_by_nested_cv is a fully
# deterministic function (no random seed anywhere in the pipeline -- fixed
# grid, zero-initialized weights, fixed-epoch batch gradient descent) of
# (embeddings, labels, clip_ids) restricted to a given outer fold's train
# split. Since this script uses the identical cached embeddings and the
# identical deterministic fold assignment (_clip_folds is a pure function
# of sorted unique clip ids), recomputing the nested L2 search here would
# reproduce these exact same values -- confirmed directly by timing one
# such recomputation (363s for one outer fold's 21-fit inner search),
# which is why this reuse exists: it skips ~30 minutes of redundant,
# already-known-deterministic computation, not a methodological shortcut.
_KNOWN_CHOSEN_L2S_ANY = [0.5, 1.0, 2.0, 0.5, 0.5]


def _cv_threshold_values_and_pr(
    X: np.ndarray, y: np.ndarray, fold_ids: np.ndarray, clip_ids: np.ndarray, min_recall: float,
    known_l2s: list[float] | None = None,
) -> list[dict]:
    """Reproduces section 16's EXACT two-stage procedure -- (1)
    `_out_of_fold_scores` to get genuinely held-out probabilities per row
    (each row's proba comes from a model that never saw it in training),
    then (2) `_cv_threshold_for_recall_floor_with_values` to select a
    recall-targeted threshold using OTHER folds' own out-of-fold scores as
    "train" signal and evaluating on this fold's out-of-fold scores as
    "test" -- and additionally surfaces the threshold VALUE selected per
    fold, which section 16 did not report (only the outcome), so
    fold-to-fold stability could not previously be inspected.

    CORRECTED 2026-08-08 (see this module's review note below): an
    earlier version of this function selected the threshold from each
    fold's own IN-SAMPLE train-set predictions instead of from the
    out-of-fold signal above -- a real methodological difference, not
    equivalent to section 16's procedure. In-sample predictions from a
    ~1280-dim model on a few hundred rows are close to perfectly
    separated, so a "reach 30% recall" threshold selected against them
    lands at an extreme, unrealistic value that does not generalize --
    caught because the result (mean R=0.054) was drastically different
    from section 16's own already-reported number (mean R=0.289) for what
    should have been the identical computation, which is exactly the kind
    of surprising-result audit this project's standing rules require
    before trusting a number.

    `known_l2s`, if given, reuses already-published, exact L2 values
    (see `_KNOWN_CHOSEN_L2S_ANY` above) for the fitting step in stage (1)
    only -- mathematically identical given the deterministic pipeline
    (verified directly: a fresh recompute for fold 0 on this exact data
    returned 0.5, matching the published value). Does not touch stage
    (2)'s threshold-selection logic, which is unchanged from section 16."""
    proba, y_valid, cids_valid, fold_ids_valid = _out_of_fold_scores(X, y, fold_ids, clip_ids, known_l2s=known_l2s)
    return _cv_threshold_for_recall_floor_with_values(proba, y_valid, fold_ids_valid, min_recall)


def _jackknife_mean_pr(per_fold: list[dict]) -> list[dict]:
    """Leave-one-fold-out: recompute mean precision/recall excluding each
    fold once. If excluding any single fold shifts the mean far from the
    full-5-fold mean, that fold is disproportionately driving the result;
    if all 5 jackknife means stay close to each other, the result is
    spread across folds, not one or two."""
    n = len(per_fold)
    out = []
    for leave_out in range(n):
        kept = [f for i, f in enumerate(per_fold) if i != leave_out]
        mean_p = float(np.mean([f["precision"] for f in kept]))
        mean_r = float(np.mean([f["recall"] for f in kept]))
        out.append({"leave_out_fold": per_fold[leave_out]["fold"], "mean_precision_excl": mean_p,
                     "mean_recall_excl": mean_r})
    return out


# ── Q5: single, full-dataset (in-sample) threshold ──────────────────────────


def _single_dataset_threshold(X: np.ndarray, y: np.ndarray, clip_ids: np.ndarray, min_recall: float) -> dict:
    """Fits ONE threshold on the ENTIRE dataset (no held-out split) using
    the identical recall-targeted rule -- this is what a real deployment
    would ship, as opposed to the 5 per-fold thresholds above, which exist
    only to ESTIMATE out-of-sample performance. Its own precision/recall
    on this same data is in-sample and explicitly not reported as a valid
    performance estimate -- only the threshold VALUE and how it compares
    to the per-fold values is decision-relevant here."""
    l2 = _select_l2_by_nested_cv(X, y, clip_ids)
    X_std, _ = _standardize(X, X[:0])  # standardize against the full set; empty "test" half unused
    w, b = _fit_logistic_regression(X_std, y, l2)
    proba = 1.0 / (1.0 + np.exp(-np.clip(X_std @ w + b, -30, 30)))
    t = _best_threshold_for_recall_floor(proba, y, min_recall)
    pred = (proba >= t).astype(int)
    p, r, f1 = _prf1(pred, y)
    return {"threshold": float(t), "l2": float(l2), "in_sample_precision": p, "in_sample_recall": r,
            "in_sample_f1": f1, "note": "IN-SAMPLE -- optimistic, not a valid out-of-sample estimate"}


def run_q1_q2_q5() -> dict:
    print("### Q1/Q2/Q5: fold threshold stability + single deployable threshold ###\n")
    h = np.load(_RESULTS_DIR / "_stage_h_rows_cache.npz", allow_pickle=True)
    mask_any = np.isin(h["types"], ("sound_repetition", "word_repetition", "control"))
    X, y, clip_ids = h["embeddings"][mask_any], h["labels"][mask_any], h["clip_ids"][mask_any]

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    per_fold = _cv_threshold_values_and_pr(X, y, fold_ids, clip_ids, MIN_RECALL, known_l2s=_KNOWN_CHOSEN_L2S_ANY)
    print("Per-fold recall-targeted threshold (value, not just outcome; "
          "selected from out-of-fold signal, matching section 16's exact procedure):")
    for f in per_fold:
        print(f"  fold {f['fold']}: threshold={f['threshold']:.4f}  "
              f"n_test=({f['n_test_pos']}pos/{f['n_test_neg']}neg)  "
              f"P={f['precision']:.3f}  R={f['recall']:.3f}  F1={f['f1']:.3f}")

    thresholds = [f["threshold"] for f in per_fold]
    recalls = [f["recall"] for f in per_fold]
    precisions = [f["precision"] for f in per_fold]
    print(f"\nThreshold dispersion: min={min(thresholds):.4f} max={max(thresholds):.4f} "
          f"range={max(thresholds) - min(thresholds):.4f} std={np.std(thresholds):.4f}")
    print(f"Recall dispersion: min={min(recalls):.3f} max={max(recalls):.3f} "
          f"range={max(recalls) - min(recalls):.3f} std={np.std(recalls):.3f}")
    print(f"Precision dispersion: min={min(precisions):.3f} max={max(precisions):.3f} "
          f"range={max(precisions) - min(precisions):.3f} std={np.std(precisions):.3f}")

    jackknife = _jackknife_mean_pr(per_fold)
    full_mean_r = float(np.mean(recalls))
    full_mean_p = float(np.mean(precisions))
    print(f"\nFull 5-fold mean: P={full_mean_p:.3f} R={full_mean_r:.3f}")
    print("Leave-one-fold-out jackknife (mean recomputed excluding each fold once):")
    max_shift_r, max_shift_p = 0.0, 0.0
    for j in jackknife:
        shift_r = abs(j["mean_recall_excl"] - full_mean_r)
        shift_p = abs(j["mean_precision_excl"] - full_mean_p)
        max_shift_r, max_shift_p = max(max_shift_r, shift_r), max(max_shift_p, shift_p)
        print(f"  excl. fold {j['leave_out_fold']}: mean P={j['mean_precision_excl']:.3f} "
              f"(shift={shift_p:+.3f})  mean R={j['mean_recall_excl']:.3f} (shift={shift_r:+.3f})")
    print(f"Max shift from excluding any single fold: P={max_shift_p:.3f}, R={max_shift_r:.3f}")

    single = _single_dataset_threshold(X, y, clip_ids, MIN_RECALL)
    print(f"\nSingle full-dataset (in-sample) threshold: {single['threshold']:.4f} "
          f"(L2={single['l2']}, in-sample P={single['in_sample_precision']:.3f}, "
          f"in-sample R={single['in_sample_recall']:.3f} -- {single['note']})")
    print(f"Per-fold threshold range was [{min(thresholds):.4f}, {max(thresholds):.4f}] -- "
          f"single-dataset value {'FALLS WITHIN' if min(thresholds) <= single['threshold'] <= max(thresholds) else 'FALLS OUTSIDE'} that range.")

    return {
        "per_fold": per_fold, "threshold_dispersion": {"min": min(thresholds), "max": max(thresholds),
                                                          "range": max(thresholds) - min(thresholds),
                                                          "std": float(np.std(thresholds))},
        "recall_dispersion": {"min": min(recalls), "max": max(recalls), "range": max(recalls) - min(recalls),
                               "std": float(np.std(recalls))},
        "precision_dispersion": {"min": min(precisions), "max": max(precisions),
                                  "range": max(precisions) - min(precisions), "std": float(np.std(precisions))},
        "jackknife": jackknife, "full_mean_precision": full_mean_p, "full_mean_recall": full_mean_r,
        "max_jackknife_shift_precision": max_shift_p, "max_jackknife_shift_recall": max_shift_r,
        "single_dataset_threshold": single,
    }


# ── Q3/Q4: type-corrected end-to-end re-scoring ──────────────────────────────


def run_q3_q4(data_dir: Path, audio_dir: Path, cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict:
    print("\n### Q3/Q4: type-corrected end-to-end Track B re-scoring ###\n")
    h = np.load(_RESULTS_DIR / "_stage_h_rows_cache.npz", allow_pickle=True)
    embeddings, labels, clip_ids = h["embeddings"], h["labels"], h["clip_ids"]

    rows_by_clip: dict[str, list[int]] = {}
    for idx, cid in enumerate(clip_ids):
        rows_by_clip.setdefault(cid, []).append(idx)

    per_clip = _rederive_word_and_sound_positions(data_dir, audio_dir, cache_dir)

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    # CORRECTED 2026-08-08 (same fix as _cv_threshold_values_and_pr above,
    # and the same real bug this review found in Step 0's ORIGINAL Part B
    # (VALIDATION.md section 16.3), which used the byte-identical pattern
    # in stage_l's `_out_of_fold_scores_and_thresholds`): the threshold
    # per fold must be selected from OTHER folds' out-of-fold signal, not
    # from this fold's own in-sample train-set predictions -- in-sample
    # predictions from a ~1280-dim model are close to perfectly separated,
    # producing an unrealistically extreme threshold (~0.98, confirmed
    # directly by this review's own first, buggy Q1/Q2 run) that fires far
    # less often than a properly-validated threshold (~0.66-0.80, this
    # review's own corrected Q1/Q2 result) would. This means Step 0's
    # original end-to-end finding likely understated Rank 1's true
    # end-to-end effect, not only for the denominator-dilution reason
    # section 16.3 already named.
    proba_full, y_valid, cids_valid, fold_ids_valid = _out_of_fold_scores(
        embeddings, labels, fold_ids, clip_ids, known_l2s=_KNOWN_CHOSEN_L2S_ANY)
    per_fold_thresholds = _cv_threshold_for_recall_floor_with_values(proba_full, y_valid, fold_ids_valid, MIN_RECALL)
    fold_thresholds: dict[int, float] = {r["fold"]: r["threshold"] for r in per_fold_thresholds}
    # The position-matching loop below indexes proba_full by the ORIGINAL
    # per-clip row indices (rows_by_clip), which assumes _out_of_fold_
    # scores dropped no rows (its `valid` mask was all-True) -- true for
    # this exact population (confirmed: n_pos=66/n_neg=1780 match the
    # unfiltered cache exactly), but checked explicitly rather than
    # silently assumed, so a future change to the cached data would fail
    # loudly instead of silently misaligning scores to positions.
    if len(proba_full) != len(labels):
        raise AssertionError(
            f"_out_of_fold_scores dropped {len(labels) - len(proba_full)} row(s) -- the "
            "position-matching loop below assumes full-length, original-row-order output; "
            "investigate before trusting results (do not silently proceed)")

    scorable_types = ("sound_repetition", "word_repetition")
    # Pass 1: ALL firings injected (TP -> true type; FP -> placeholder type,
    # used ONLY to register in the type-agnostic Any-label aggregate).
    baseline_any_pass = _empty_counts(scorable_types)
    augmented_any_pass = _empty_counts(scorable_types)
    # Pass 2: ONLY true-positive firings injected (true type, ground-truth-
    # justified) -- valid per-type TP/FN breakdown, no fabricated FP-by-type.
    baseline_typed_pass = _empty_counts(scorable_types)
    augmented_typed_pass = _empty_counts(scorable_types)

    n_clips_used, n_clips_excluded = 0, 0
    n_new_candidates_fired, n_new_true_positives, n_new_false_positives_unattributed = 0, 0, 0

    for clip_name, rec in per_clip.items():
        row_idx = rows_by_clip.get(clip_name, [])
        n_expected = len(rec["targets"]) + len(rec["clean_positions"])
        if len(row_idx) != n_expected:
            n_clips_excluded += 1
            continue
        n_clips_used += 1

        clip, hyp_tokens = rec["clip"], rec["hyp_tokens"]
        base_events = detect_disfluencies(hyp_tokens, config=_GATE_OFF_CONFIG, audio_bytes=clip.audio_bytes)

        ordered_positions = [(ri, hi, "target") for ri, hi, _tt in rec["targets"]]
        ordered_positions += [(ri, hi, "clean") for ri, hi in rec["clean_positions"]]

        fold = fold_map.get(clip_name)
        threshold = fold_thresholds.get(fold)

        new_events_any = list(base_events)     # Pass 1: all firings (Any-label valid)
        new_events_typed = list(base_events)   # Pass 2: TP-only firings (per-type valid)

        if threshold is not None:
            for (ref_idx, hyp_idx, _kind), r_idx in zip(ordered_positions, row_idx):
                p = proba_full[r_idx]
                if np.isnan(p) or p < threshold:
                    continue
                n_new_candidates_fired += 1
                true_type = clip.ground_truth.get(ref_idx)
                if true_type in scorable_types:
                    # TP: true type is ground-truth-justified, valid for BOTH passes.
                    n_new_true_positives += 1
                    ev = {"index": hyp_idx, "type": true_type, "score": float(p),
                          "reason": "rank1_classifier_candidate"}
                    new_events_any.append(ev)
                    new_events_typed.append(ev)
                else:
                    # FP: no true type exists. Placeholder only for Any-label
                    # registration (Any doesn't care which specific type);
                    # deliberately NOT added to new_events_typed, so the
                    # per-type table never fabricates a type for this firing.
                    n_new_false_positives_unattributed += 1
                    new_events_any.append({"index": hyp_idx, "type": "sound_repetition",
                                            "score": float(p), "reason": "rank1_classifier_candidate_FP_unattributed_type"})

        score_clip(clip, hyp_tokens, base_events, scorable_types, baseline_any_pass,
                   _empty_counts(scorable_types), Counter(), Counter())
        score_clip(clip, hyp_tokens, new_events_any, scorable_types, augmented_any_pass,
                   _empty_counts(scorable_types), Counter(), Counter())
        score_clip(clip, hyp_tokens, base_events, scorable_types, baseline_typed_pass,
                   _empty_counts(scorable_types), Counter(), Counter())
        score_clip(clip, hyp_tokens, new_events_typed, scorable_types, augmented_typed_pass,
                   _empty_counts(scorable_types), Counter(), Counter())

    print(f"Clips with a Stage-A category-1 target: {len(per_clip)}")
    print(f"  used (row count matched exactly): {n_clips_used}")
    print(f"  excluded (row/position count mismatch): {n_clips_excluded}")
    print(f"New candidates fired: {n_new_candidates_fired} "
          f"({n_new_true_positives} true positives, {n_new_false_positives_unattributed} "
          f"false positives -- type unattributable, not fabricated)\n")

    print("=== Pass 1 (Any-label valid; per-type breakdown here is NOT reliable, see note) ===")
    print("--- Baseline ---")
    print(format_table(baseline_any_pass))
    print("--- Augmented ---")
    print(format_table(augmented_any_pass))

    print("\n=== Pass 2 (per-type TP/FN valid, ground-truth-justified; FPs deliberately excluded here) ===")
    print("--- Baseline ---")
    print(format_table(baseline_typed_pass))
    print("--- Augmented ---")
    print(format_table(augmented_typed_pass))

    result = {
        "n_clips_with_target": len(per_clip), "n_clips_used": n_clips_used,
        "n_clips_excluded": n_clips_excluded, "n_new_candidates_fired": n_new_candidates_fired,
        "n_new_true_positives": n_new_true_positives,
        "n_new_false_positives_unattributed": n_new_false_positives_unattributed,
        "any_label_valid": {
            "baseline": vars(baseline_any_pass[ANY_LABEL]), "augmented": vars(augmented_any_pass[ANY_LABEL]),
        },
        "per_type_tp_fn_valid": {
            t: {"baseline": vars(baseline_typed_pass[t]), "augmented": vars(augmented_typed_pass[t])}
            for t in scorable_types
        },
    }
    return result


def _save(result: dict, tag: str) -> None:
    out_path = _RESULTS_DIR / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_m_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")


# ── Self-test (per instruction 9: written and run BEFORE trusting real results) ──

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _cv_threshold_values_and_pr: a threshold selected on train data,
    # applied to test data, must actually be usable (produces a real P/R).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    X[:20] += 2.0  # first 20 rows are the "positive" class, separable
    y = np.array([1] * 20 + [0] * 20)
    clip_ids = np.array([f"c{i}" for i in range(40)])
    fold_ids = np.array([i % 5 for i in range(40)])
    per_fold = _cv_threshold_values_and_pr(X, y, fold_ids, clip_ids, 0.5)
    check("per-fold results cover all 5 folds on well-separated synthetic data",
          len(per_fold) == 5, str(len(per_fold)))
    check("recall-targeted threshold achieves reasonable recall on held-out data "
          "for a well-separated synthetic case",
          np.mean([f["recall"] for f in per_fold]) > 0.3,
          str([f["recall"] for f in per_fold]))
    check("per-fold result exposes the threshold VALUE (the whole point of this function "
          "vs. section 16's original, which only reported outcomes)",
          all("threshold" in f for f in per_fold), str(per_fold))

    # 1b. known_l2s reuse path: must produce IDENTICAL results to the
    # freshly-recomputed path when given the L2 values that would have
    # been recomputed anyway (proves reuse doesn't change the answer),
    # and must not call the expensive nested-search function at all --
    # verified by monkeypatching it in the module it actually lives in
    # (stage_l, since _out_of_fold_scores is imported from there, not
    # re-implemented in this module), so any accidental call is caught
    # loudly rather than just being slow.
    import profiling.evaluation.stage_l_zero_compute_reanalysis as _stage_l_mod
    original_select_l2 = _stage_l_mod._select_l2_by_nested_cv

    def _boom(*a, **kw):
        raise AssertionError("nested L2 search was called despite known_l2s being provided")

    # First, recompute genuinely fresh (to know what the "real" per-fold
    # L2 values are for this synthetic case) with the nested search intact.
    _select_l2_used = []
    real_select_l2 = _stage_l_mod._select_l2_by_nested_cv

    def _spy(*a, **kw):
        v = real_select_l2(*a, **kw)
        _select_l2_used.append(v)
        return v

    _stage_l_mod._select_l2_by_nested_cv = _spy
    try:
        per_fold_fresh = _cv_threshold_values_and_pr(X, y, fold_ids, clip_ids, 0.5)
    finally:
        _stage_l_mod._select_l2_by_nested_cv = original_select_l2
    check("spy captured one L2 selection per fold", len(_select_l2_used) == 5, str(_select_l2_used))

    _stage_l_mod._select_l2_by_nested_cv = _boom
    try:
        per_fold_known = _cv_threshold_values_and_pr(X, y, fold_ids, clip_ids, 0.5, known_l2s=_select_l2_used)
        check("known_l2s path does not call the expensive nested search (would have raised)",
              len(per_fold_known) == 5, "did not raise -- good")
        check("known_l2s path reproduces IDENTICAL thresholds to the freshly-recomputed path "
              "when given the same L2 values -- proves reuse doesn't change the answer",
              all(abs(a["threshold"] - b["threshold"]) < 1e-9 for a, b in zip(per_fold_fresh, per_fold_known)),
              f"fresh={[f['threshold'] for f in per_fold_fresh]} known={[f['threshold'] for f in per_fold_known]}")
    finally:
        _stage_l_mod._select_l2_by_nested_cv = original_select_l2

    # 2. _jackknife_mean_pr: hand-constructed case where one fold is an
    # outlier -- excluding it should shift the mean noticeably; excluding
    # a typical fold should barely move it.
    hand_folds = [
        {"fold": 0, "precision": 0.8, "recall": 0.8},
        {"fold": 1, "precision": 0.8, "recall": 0.8},
        {"fold": 2, "precision": 0.8, "recall": 0.8},
        {"fold": 3, "precision": 0.8, "recall": 0.8},
        {"fold": 4, "precision": 0.0, "recall": 0.0},  # the outlier
    ]
    jk = _jackknife_mean_pr(hand_folds)
    full_mean_r = np.mean([f["recall"] for f in hand_folds])
    excl_outlier = next(j for j in jk if j["leave_out_fold"] == 4)
    excl_typical = next(j for j in jk if j["leave_out_fold"] == 0)
    check("excluding the outlier fold shifts the mean recall a lot",
          abs(excl_outlier["mean_recall_excl"] - full_mean_r) > 0.1,
          f"full={full_mean_r}, excl_outlier={excl_outlier['mean_recall_excl']}")
    check("excluding a typical (non-outlier) fold shifts the mean much less "
          "than excluding the outlier",
          abs(excl_typical["mean_recall_excl"] - full_mean_r) < abs(excl_outlier["mean_recall_excl"] - full_mean_r),
          f"excl_typical={excl_typical['mean_recall_excl']}, excl_outlier={excl_outlier['mean_recall_excl']}")

    # 3. _single_dataset_threshold: in-sample recall must actually reach
    # the target on a well-separated synthetic case, and the function must
    # not crash on the "empty second half" standardization call.
    single = _single_dataset_threshold(X, y, clip_ids, 0.5)
    check("single-dataset threshold reaches its own target recall in-sample",
          single["in_sample_recall"] >= 0.5, str(single))

    # 4. Type-attribution logic (hand-constructed, no real ASR/audio):
    # a TP firing must carry the ground-truth type; a control-position
    # firing (no true type) must be excluded from the per-type-valid pass
    # but still registered in the Any-label-valid pass.
    class _FakeClip:
        def __init__(self, ground_truth):
            self.ground_truth = ground_truth
    fake_clip = _FakeClip(ground_truth={2: "sound_repetition"})
    # Simulate the classification branch directly (same logic as run_q3_q4's loop body).
    scorable_types = ("sound_repetition", "word_repetition")
    true_type_tp = fake_clip.ground_truth.get(2)
    true_type_fp = fake_clip.ground_truth.get(5)  # a clean/control position, no entry
    check("a true-positive position resolves to its real ground-truth type",
          true_type_tp == "sound_repetition", str(true_type_tp))
    check("a control position (no ground truth) resolves to None, not a fabricated type",
          true_type_fp is None, str(true_type_fp))
    check("None is correctly excluded from the scorable-type-committed branch",
          true_type_fp not in scorable_types, str(true_type_fp))

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--q1-q2-q5-only", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    result_1 = run_q1_q2_q5()
    _save(result_1, "q1_q2_q5_fold_stability")

    if args.q1_q2_q5_only:
        return 0
    if not args.data_dir or not args.audio_dir:
        print("--data-dir/--audio-dir required for Q3/Q4 (unless --q1-q2-q5-only).")
        return 2
    cache_dir = Path(args.cache_dir) if args.cache_dir else _DEFAULT_CACHE_DIR
    result_2 = run_q3_q4(Path(args.data_dir), Path(args.audio_dir), cache_dir)
    _save(result_2, "q3_q4_type_corrected_end_to_end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
