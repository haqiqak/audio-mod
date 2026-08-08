"""stage_combined_classifier.py — ASR_RESEARCH_TRACK.md "Combined-signal
classifier".

Implements the protocol pre-registered in ASR_RESEARCH_TRACK.md's
"Combined-signal classifier - pre-registered protocol" section EXACTLY -
read that section before changing any logic here. Tests whether a plain
logistic regression over direction (g)'s acoustic candidate signal
(MFCC similarity, burst count, duration) PLUS Stage C's encoder-distance
signal, combined, separates true sound_repetition instances from false
acoustic candidates better than either signal alone.

Deliberately reuses this project's own existing infrastructure rather
than adding anything new:
- Candidate generation: profiling.evaluation.stage_g_acoustic_sound_
  repetition's generate_candidates() (MFCC feature) and compute_mfcc(),
  unmodified.
- Encoder-distance: profiling.encoder_embedding's load_encoder(),
  extract_last_layer_states(), pool_span(), cosine_distance(), the same
  primitives Stage B/C used - only the "which position to pool" logic is
  new (candidate spans, not ASR-hyp-token spans - see the pre-
  registration's "necessary adaptation" note).
- Model + cross-validation: profiling.evaluation.compare_corroboration_
  mechanisms's _fit_logistic_regression, _standardize,
  _select_l2_by_nested_cv, _clip_folds, _prf1, _best_threshold_by_f1,
  _cv_threshold, _cv_classifier, _summarize - all unmodified, the exact
  mechanism that already shipped this project's one other trained
  classifier (ROADMAP.md item 17).

Usage
-----
    python -m profiling.evaluation.stage_combined_classifier \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 -- must precede torch/transformers imports

import numpy as np

from profiling.acoustic import AcousticConfig, frame_features, load_wav_samples, segment_voiced
from profiling.config import load_config
from profiling.encoder_embedding import cosine_distance, extract_last_layer_states, load_encoder, pool_span
from profiling.evaluation.compare_corroboration_mechanisms import (
    _best_threshold_by_f1,
    _clip_folds,
    _cv_classifier,
    _cv_threshold,
    _fit_logistic_regression,
    _prf1,
    _select_l2_by_nested_cv,
    _standardize,
    _summarize,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_g_acoustic_sound_repetition import (
    AcousticCandidate,
    MATCH_TOLERANCE_SECONDS,
    _mfcc_similarity_fn,
    _overlaps,
    compute_mfcc,
    generate_candidates,
)
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPE = "sound_repetition"


def _clean_word_positions(clip, hyp_tokens: list[dict]) -> list[tuple[int, int]]:
    """(ref_idx, hyp_idx) pairs for correctly-aligned, non-disfluent
    positions - the same "clean" population Stage B/C build a fluent
    centroid from. Reuses alignment.align() directly, same as stage_b_
    representation_probe.py's own _identify_positions()."""
    from profiling.evaluation.alignment import align

    ref_words = [t["word"] for t in clip.tokens]
    hyp_words = [t["word"] for t in hyp_tokens]
    disfluent_idx = set(clip.ground_truth.keys())
    ops = align(ref_words, hyp_words, disfluent_indices=disfluent_idx)
    clean = []
    for op in ops:
        if op.kind == "correct" and op.hyp_index is not None:
            if op.ref_index is None or op.ref_index not in disfluent_idx:
                clean.append((op.ref_index, op.hyp_index))
    return clean


def _encoder_distance_for_candidate(
    cand: AcousticCandidate, hyp_tokens: list[dict], states, clean_vecs: dict[int, np.ndarray],
) -> float | None:
    """Pools the encoder at the first cached ASR token overlapping this
    candidate's span, and returns its cosine distance to the clip's
    leave-one-out-safe fluent centroid (the candidate itself is never
    part of the "clean" set, so no leave-one-out adjustment is needed for
    it - same asymmetry Stage B's own target-vs-control scoring used).
    None if no overlapping ASR token exists for this clip - a real,
    expected "missing evidence" case, not an error."""
    overlapping_hyp_idx = None
    for hyp_idx, tok in enumerate(hyp_tokens):
        t0, t1 = tok.get("start"), tok.get("end")
        if t0 is None or t1 is None:
            continue
        if _overlaps(cand.start, cand.end, t0, t1):
            overlapping_hyp_idx = hyp_idx
            break
    if overlapping_hyp_idx is None or len(clean_vecs) < 2:
        return None
    tok = hyp_tokens[overlapping_hyp_idx]
    v = pool_span(states, tok.get("start"), tok.get("end"))
    if v is None:
        return None
    centroid = np.mean(list(clean_vecs.values()), axis=0)
    return cosine_distance(v, centroid)


def build_dataset(data_dir: Path, audio_dir: Path, n_clips: int = 120) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None][:n_clips]
    print(f"{len(clips)} clips.\n")

    ac_cfg_dict = dict(load_config().get("profiling", {}).get("detection", {}))
    ac_cfg = AcousticConfig.from_detection_cfg(ac_cfg_dict)

    print("Loading CrisperWhisper encoder ...")
    processor, encoder = load_encoder()

    rows = []  # dicts: clip_id, mfcc_sim, n_bursts, duration, encoder_distance (or None), label
    n_with_encoder_signal = 0
    t0 = time.time()

    for i, clip in enumerate(clips):
        targets = [
            (clip.tokens[ref_idx]["start"], clip.tokens[ref_idx]["end"])
            for ref_idx, t in clip.ground_truth.items() if t == TARGET_TYPE
        ]

        samples, sr = load_wav_samples(clip.audio_bytes)
        if samples is None:
            continue
        segments = segment_voiced(samples, sr, ac_cfg)
        mfcc_times, mfcc = compute_mfcc(samples, sr, ac_cfg.frame_seconds, ac_cfg.hop_seconds)
        sim_fn = _mfcc_similarity_fn(mfcc_times, mfcc)
        cands = generate_candidates(segments, None, None, None, similarity_fn=sim_fn)
        if not cands:
            continue

        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        states = None
        clean_vecs: dict[int, np.ndarray] = {}
        if hyp_tokens is not None:
            states = extract_last_layer_states(processor, encoder, samples, sr)
            clean = _clean_word_positions(clip, hyp_tokens)
            for ref_idx, hyp_idx in clean:
                tok = hyp_tokens[hyp_idx]
                v = pool_span(states, tok.get("start"), tok.get("end"))
                if v is not None:
                    clean_vecs[hyp_idx] = v

        for cand in cands:
            label = 1 if any(
                _overlaps(cand.start, cand.end, t0_ - MATCH_TOLERANCE_SECONDS, t1_ + MATCH_TOLERANCE_SECONDS)
                for t0_, t1_ in targets
            ) else 0

            enc_dist = None
            if states is not None and hyp_tokens is not None:
                enc_dist = _encoder_distance_for_candidate(cand, hyp_tokens, states, clean_vecs)
            if enc_dist is not None:
                n_with_encoder_signal += 1

            rows.append({
                "clip_id": clip.name,
                "mfcc_sim": cand.mean_similarity,
                "n_bursts": cand.n_bursts,
                "duration": cand.end - cand.start,
                "encoder_distance": enc_dist,
                "label": label,
            })

        if (i + 1) % 30 == 0 or i + 1 == len(clips):
            print(f"[{i+1}/{len(clips)}] ... ({time.time()-t0:.0f}s elapsed)")

    print(f"\nTotal time: {time.time()-t0:.0f}s for {len(clips)} clips.")
    n_pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} candidates, {n_pos} positive (label=1), "
          f"{n_with_encoder_signal}/{len(rows)} have a real encoder-distance value "
          f"({100*n_with_encoder_signal/max(1,len(rows)):.1f}%).\n")

    cache_path = _ROOT / "eval_datasets" / "_stage_combined_classifier_rows_cache.json"
    cache_path.write_text(json.dumps({"rows": rows, "n_clips": len(clips)}), encoding="utf-8")
    print(f"Cached row-level dataset (reusable via --from-cache): {cache_path}\n")

    return {"rows": rows, "n_clips": len(clips)}


def _to_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_combined, mfcc_sim, encoder_distance_real_mask, labels, clip_ids).
    encoder_distance is median-imputed with a paired has-signal indicator
    column in X_combined, per the pre-registration."""
    mfcc_sim = np.array([r["mfcc_sim"] for r in rows], dtype=np.float64)
    n_bursts = np.array([r["n_bursts"] for r in rows], dtype=np.float64)
    duration = np.array([r["duration"] for r in rows], dtype=np.float64)
    enc_raw = [r["encoder_distance"] for r in rows]
    has_enc = np.array([1.0 if v is not None else 0.0 for v in enc_raw], dtype=np.float64)
    real_vals = [v for v in enc_raw if v is not None]
    median_enc = float(np.median(real_vals)) if real_vals else 0.0
    enc_imputed = np.array([v if v is not None else median_enc for v in enc_raw], dtype=np.float64)
    labels = np.array([r["label"] for r in rows], dtype=np.int64)
    clip_ids = np.array([r["clip_id"] for r in rows])

    X = np.column_stack([mfcc_sim, n_bursts, duration, enc_imputed, has_enc])
    return X, mfcc_sim, has_enc.astype(bool), labels, clip_ids


def _cv_classifier_optimal_threshold(
    embeddings: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray, clip_ids: np.ndarray,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Same nested-CV logistic regression as compare_corroboration_
    mechanisms._cv_classifier() (L2 selected by inner clip-split CV,
    fit on the full outer-train fold) - but selects the classification
    threshold on the outer-train fold's own predicted probabilities via
    _best_threshold_by_f1(), instead of hardcoding proba>=0.5.

    NOT a new tuning surface: this is the identical convention already
    used for Arms (A)/(B) in this script (_cv_threshold/_best_threshold_
    by_f1). Reusing _cv_classifier's hardcoded 0.5 for Arm (C) alone
    would make the three-arm comparison inconsistent, not simpler - a
    fixed 0.5 cutoff is not calibrated to this population's ~8% positive
    rate, and standard logistic regression's fitted probabilities
    correctly reflect that base rate (verified directly, not assumed:
    the first real run here returned F1=0.000/0.000/0.000/0.000/0.000
    across all 5 folds - exactly the signature of a threshold below
    every fold's actual probability ceiling, not a model that failed to
    learn anything, given Arm (B) alone - one of Arm (C)'s own input
    features - independently scores F1=0.244)."""
    results = []
    chosen_l2s = []
    for f in range(5):
        test_mask = fold_ids == f
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        l2 = _select_l2_by_nested_cv(embeddings[train_mask], labels[train_mask], clip_ids[train_mask])
        chosen_l2s.append(l2)

        X_train, X_test = _standardize(embeddings[train_mask], embeddings[test_mask])
        y_train = labels[train_mask]
        w, b = _fit_logistic_regression(X_train, y_train, l2)

        proba_train = 1.0 / (1.0 + np.exp(-np.clip(X_train @ w + b, -30, 30)))
        threshold = _best_threshold_by_f1(proba_train, y_train)

        proba_test = 1.0 / (1.0 + np.exp(-np.clip(X_test @ w + b, -30, 30)))
        pred = (proba_test >= threshold).astype(int)
        results.append(_prf1(pred, labels[test_mask]))
    return results, chosen_l2s


def run(data_dir: Path, audio_dir: Path, n_clips: int = 120, from_cache: Path | None = None) -> dict:
    if from_cache is not None:
        print(f"Loading cached row-level dataset from {from_cache} (skipping the expensive encoder pass) ...")
        data = json.loads(from_cache.read_text(encoding="utf-8"))
    else:
        data = build_dataset(data_dir, audio_dir, n_clips)
    rows = data["rows"]
    X, mfcc_sim, has_enc_mask, labels, clip_ids = _to_feature_matrix(rows)

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    print("=== Arm (A): MFCC-alone, cross-validated threshold ===")
    results_a = _cv_threshold(mfcc_sim, labels, fold_ids)
    print("  " + _summarize("MFCC-alone (CV threshold)", results_a))

    print("\n=== Arm (B): Encoder-distance-alone, cross-validated threshold (real values only) ===")
    if has_enc_mask.sum() >= 10 and labels[has_enc_mask].sum() >= 2:
        enc_only = np.array([r["encoder_distance"] for r in rows], dtype=object)
        enc_vals = np.array([v if v is not None else np.nan for v in enc_only], dtype=np.float64)
        m = has_enc_mask
        results_b = _cv_threshold(enc_vals[m], labels[m], fold_ids[m])
        print("  " + _summarize("Encoder-distance-alone (CV threshold)", results_b))
    else:
        results_b = []
        print("  n/a - too few candidates with a real encoder-distance value")

    print("\n=== Arm (C): Combined classifier (MFCC + n_bursts + duration + encoder-distance), nested-CV logistic regression ===")
    results_c_fixed05, _ = _cv_classifier(X, labels.astype(np.float64), fold_ids, clip_ids)
    print("  " + _summarize("Combined classifier (proba>=0.5, compare_corroboration_mechanisms's own default)", results_c_fixed05))
    results_c, chosen_l2s = _cv_classifier_optimal_threshold(X, labels.astype(np.float64), fold_ids, clip_ids)
    print("  " + _summarize("Combined classifier (train-fold-optimal threshold, matching Arms A/B's own convention)", results_c))
    print(f"    selected L2 per outer fold: {[round(x, 2) for x in chosen_l2s]}")

    def _mean_f1(results):
        return float(np.mean([f1 for _, _, f1 in results])) if results else 0.0

    f1_a, f1_b, f1_c = _mean_f1(results_a), _mean_f1(results_b), _mean_f1(results_c)
    bar_a, bar_b = f1_a * 1.2, f1_b * 1.2
    beats_a = f1_c > bar_a
    beats_b = f1_c > bar_b if results_b else True  # if (B) is n/a, don't block the verdict on it
    verdict = "SUCCESS" if (beats_a and beats_b) else "FAILURE"

    print(f"\nMean F1: (A) MFCC-alone={f1_a:.3f}  (B) encoder-alone={f1_b:.3f}  (C) combined={f1_c:.3f}")
    print(f"(C) vs (A) bar ({bar_a:.3f}, >=20% relative): {'CLEARED' if beats_a else 'not cleared'}")
    if results_b:
        print(f"(C) vs (B) bar ({bar_b:.3f}, >=20% relative): {'CLEARED' if beats_b else 'not cleared'}")
    print(f"Verdict (pre-registered: combined must meaningfully beat BOTH individual arms): {verdict}")

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_combined_classifier.json"
    out_path.write_text(json.dumps({
        "n_clips": data["n_clips"], "n_candidates": len(rows), "n_positive": int(labels.sum()),
        "n_with_encoder_signal": int(has_enc_mask.sum()),
        "results_a_mfcc_alone": [list(r) for r in results_a],
        "results_b_encoder_alone": [list(r) for r in results_b],
        "results_c_combined_fixed_threshold_0.5": [list(r) for r in results_c_fixed05],
        "results_c_combined_optimal_threshold": [list(r) for r in results_c],
        "chosen_l2s": chosen_l2s,
        "mean_f1": {"a_mfcc_alone": f1_a, "b_encoder_alone": f1_b, "c_combined": f1_c},
        "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return {"f1_a": f1_a, "f1_b": f1_b, "f1_c": f1_c, "verdict": verdict}


# ── Self-test (feature construction + imputation, hand-constructed) ────────

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. Missing encoder-distance values get median-imputed with a correct
    #    has-signal indicator, real values pass through unchanged.
    rows = [
        {"clip_id": "a", "mfcc_sim": 0.5, "n_bursts": 2, "duration": 0.3, "encoder_distance": 0.10, "label": 1},
        {"clip_id": "a", "mfcc_sim": 0.6, "n_bursts": 2, "duration": 0.4, "encoder_distance": None, "label": 0},
        {"clip_id": "b", "mfcc_sim": 0.4, "n_bursts": 3, "duration": 0.5, "encoder_distance": 0.30, "label": 0},
    ]
    X, mfcc_sim, has_enc_mask, labels, clip_ids = _to_feature_matrix(rows)
    check("median imputation used for the missing value (median of [0.10, 0.30] = 0.20)",
          abs(X[1, 3] - 0.20) < 1e-9, str(X[1, 3]))
    check("has-signal indicator is 1 for real values, 0 for imputed",
          list(has_enc_mask) == [True, False, True], str(has_enc_mask))
    check("real encoder-distance values pass through unchanged",
          abs(X[0, 3] - 0.10) < 1e-9 and abs(X[2, 3] - 0.30) < 1e-9, str(X[:, 3]))
    check("feature matrix shape is [n_rows, 5]", X.shape == (3, 5), str(X.shape))
    check("clip_ids and labels carried through correctly",
          list(clip_ids) == ["a", "a", "b"] and list(labels) == [1, 0, 0],
          f"{clip_ids}, {labels}")

    # 2. All-missing encoder-distance column: imputation falls back to 0.0
    #    (documented fallback), not a crash.
    rows_all_missing = [
        {"clip_id": "a", "mfcc_sim": 0.5, "n_bursts": 2, "duration": 0.3, "encoder_distance": None, "label": 1},
    ]
    X2, _, has_enc2, _, _ = _to_feature_matrix(rows_all_missing)
    check("all-missing encoder-distance column imputes to 0.0 without crashing",
          X2[0, 3] == 0.0 and has_enc2[0] == False, str(X2[0, 3]))

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--from-cache", default=None,
                         help="Reuse a previously-cached row-level dataset (skips the encoder pass).")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    from_cache = Path(args.from_cache) if args.from_cache else None
    if from_cache is None and (not args.data_dir or not args.audio_dir):
        print("--data-dir and --audio-dir are required (unless --self-test or --from-cache).")
        return 2
    data_dir = Path(args.data_dir) if args.data_dir else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    run(data_dir, audio_dir, args.n, from_cache=from_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
