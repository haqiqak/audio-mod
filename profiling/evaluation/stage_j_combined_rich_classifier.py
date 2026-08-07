"""stage_j_combined_rich_classifier.py — Rank 3, pre-registered in
`ASR_RESEARCH_TRACK.md`'s "Rank 3: combined rich-feature classifier"
section. Read that section before changing any logic here.

Per the project owner's explicit instruction, this is the final bounded
low-cost experiment in this track's Rank 1/2/3 sequence: tests whether a
nested-CV logistic regression given BOTH the full CrisperWhisper encoder
embedding (Rank 1's own feature) AND the full 26-dim raw acoustic MFCC
statistics (Rank 2's own feature) together, on the exact same population
Phase A's original combined-signal classifier used, separates true
`sound_repetition` candidates better than either rich signal alone or
than Phase A's own scalar-only combination (F1=0.242).

Deliberately reuses this project's own existing infrastructure rather
than adding anything new:
- Acoustic candidate generation + 26-dim feature + MFCC-similarity
  baseline: `stage_i_learned_acoustic_classifier`'s `_generate_
  candidates_with_bursts`, `_burst_feature_vector`, `_mean_mfcc_
  similarity` — unmodified.
- Encoder pooling + clean-centroid logic: `stage_combined_classifier`'s
  `_clean_word_positions` — unmodified. This script adds ONE new
  function, `_encoder_embedding_for_candidate`, mirroring `stage_
  combined_classifier._encoder_distance_for_candidate`'s exact overlap-
  finding logic but returning the raw pooled vector (for Rank 3's
  classifier input) alongside the distance scalar (for Arm B's
  baseline) instead of the distance alone -- the smallest change that
  reuses the validated overlap logic without duplicating it wholesale.
- CV/classifier machinery: `compare_corroboration_mechanisms`'s
  `_clip_folds`/`_cv_threshold`/`_summarize` and `stage_combined_
  classifier`'s `_cv_classifier_optimal_threshold` — all unmodified.

Usage
-----
    python -m profiling.evaluation.stage_j_combined_rich_classifier \\
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

from profiling.acoustic import AcousticConfig, load_wav_samples, segment_voiced
from profiling.config import load_config
from profiling.encoder_embedding import cosine_distance, extract_last_layer_states, load_encoder, pool_span
from profiling.evaluation.compare_corroboration_mechanisms import (
    _clip_folds,
    _cv_threshold,
    _summarize,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_combined_classifier import _clean_word_positions, _cv_classifier_optimal_threshold
from profiling.evaluation.stage_g_acoustic_sound_repetition import MATCH_TOLERANCE_SECONDS, _overlaps, compute_mfcc
from profiling.evaluation.stage_i_learned_acoustic_classifier import (
    CandidateWithBursts,
    _burst_feature_vector,
    _generate_candidates_with_bursts,
    _mean_mfcc_similarity,
)
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached

TARGET_TYPE = "sound_repetition"
N_CLIPS = 120  # matches Phase A's original combined-signal classifier exactly


def _encoder_embedding_for_candidate(
    cand: CandidateWithBursts, hyp_tokens: list[dict], states, clean_vecs: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, float | None]:
    """Mirrors `stage_combined_classifier._encoder_distance_for_candidate`'s
    exact overlap-finding logic, but returns (raw_vector, distance) instead
    of distance alone -- the vector is Rank 3's classifier input, the
    distance is Arm B's baseline value, both from one pooling call."""
    overlapping_hyp_idx = None
    for hyp_idx, tok in enumerate(hyp_tokens):
        t0, t1 = tok.get("start"), tok.get("end")
        if t0 is None or t1 is None:
            continue
        if _overlaps(cand.start, cand.end, t0, t1):
            overlapping_hyp_idx = hyp_idx
            break
    if overlapping_hyp_idx is None or len(clean_vecs) < 2:
        return None, None
    tok = hyp_tokens[overlapping_hyp_idx]
    v = pool_span(states, tok.get("start"), tok.get("end"))
    if v is None:
        return None, None
    centroid = np.mean(list(clean_vecs.values()), axis=0)
    return v, cosine_distance(v, centroid)


def build_dataset(data_dir: Path, audio_dir: Path, n_clips: int = N_CLIPS) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None][:n_clips]
    print(f"{len(clips)} clips.\n")

    ac_cfg_dict = dict(load_config().get("profiling", {}).get("detection", {}))
    cfg = AcousticConfig.from_detection_cfg(ac_cfg_dict)

    print("Loading CrisperWhisper encoder ...")
    processor, encoder = load_encoder()

    rows: list[dict] = []
    n_with_encoder = 0
    t0 = time.time()

    for i, clip in enumerate(clips):
        targets = [
            (clip.tokens[ref_idx]["start"], clip.tokens[ref_idx]["end"])
            for ref_idx, t in clip.ground_truth.items() if t == TARGET_TYPE
        ]

        samples, sr = load_wav_samples(clip.audio_bytes)
        if samples is None:
            continue
        segments = segment_voiced(samples, sr, cfg)
        times, mfcc = compute_mfcc(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
        cands = _generate_candidates_with_bursts(segments)
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
            acoustic_feat = _burst_feature_vector(cand, times, mfcc)
            if acoustic_feat is None:
                continue
            sim = _mean_mfcc_similarity(cand, times, mfcc)

            enc_vec, enc_dist = None, None
            if states is not None and hyp_tokens is not None:
                enc_vec, enc_dist = _encoder_embedding_for_candidate(cand, hyp_tokens, states, clean_vecs)
            if enc_vec is not None:
                n_with_encoder += 1

            label = 1 if any(
                _overlaps(cand.start, cand.end, t0_ - MATCH_TOLERANCE_SECONDS, t1_ + MATCH_TOLERANCE_SECONDS)
                for t0_, t1_ in targets
            ) else 0

            rows.append({
                "clip_id": clip.name, "label": label,
                "acoustic_feat": acoustic_feat, "mfcc_similarity": sim,
                "encoder_embedding": enc_vec, "encoder_distance": enc_dist,
            })

        if (i + 1) % 30 == 0 or i + 1 == len(clips):
            print(f"[{i+1}/{len(clips)}] ... ({time.time()-t0:.0f}s elapsed)")

    print(f"\nTotal time: {time.time()-t0:.0f}s for {len(clips)} clips.")
    n_pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} candidates, {n_pos} positive (label=1), "
          f"{n_with_encoder}/{len(rows)} have a real encoder embedding "
          f"({100*n_with_encoder/max(1,len(rows)):.1f}%).\n")

    out_path = _ROOT / "eval_results" / "_stage_j_rows_cache.npz"
    _save_rows(rows, out_path)
    print(f"Cached row-level dataset (reusable via --from-cache): {out_path}\n")
    return {"rows": rows}


def _save_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(
            out_path, acoustic_feats=np.zeros((0, 0), dtype=np.float64),
            mfcc_similarities=np.zeros(0, dtype=np.float64),
            encoder_embeddings=np.zeros((0, 0), dtype=np.float32),
            encoder_distances=np.zeros(0, dtype=np.float64),
            has_encoder=np.zeros(0, dtype=bool),
            labels=np.zeros(0, dtype=np.int64), clip_ids=np.empty(0, dtype=object),
        )
        return
    acoustic_dim = rows[0]["acoustic_feat"].shape[0]
    hidden_dim = next((r["encoder_embedding"].shape[0] for r in rows if r["encoder_embedding"] is not None), 1280)
    n = len(rows)
    acoustic_feats = np.zeros((n, acoustic_dim), dtype=np.float64)
    mfcc_similarities = np.zeros(n, dtype=np.float64)
    encoder_embeddings = np.full((n, hidden_dim), np.nan, dtype=np.float32)
    encoder_distances = np.full(n, np.nan, dtype=np.float64)
    has_encoder = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int64)
    clip_ids = np.empty(n, dtype=object)
    for i, r in enumerate(rows):
        acoustic_feats[i] = r["acoustic_feat"]
        mfcc_similarities[i] = r["mfcc_similarity"]
        labels[i] = r["label"]
        clip_ids[i] = r["clip_id"]
        if r["encoder_embedding"] is not None:
            encoder_embeddings[i] = r["encoder_embedding"]
            encoder_distances[i] = r["encoder_distance"]
            has_encoder[i] = True
    np.savez_compressed(
        out_path, acoustic_feats=acoustic_feats, mfcc_similarities=mfcc_similarities,
        encoder_embeddings=encoder_embeddings, encoder_distances=encoder_distances,
        has_encoder=has_encoder, labels=labels, clip_ids=clip_ids,
    )


def _load_rows(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in (
        "acoustic_feats", "mfcc_similarities", "encoder_embeddings",
        "encoder_distances", "has_encoder", "labels", "clip_ids",
    )}


def _to_feature_matrix(d: dict) -> np.ndarray:
    """Concatenates [acoustic (26)] + [median-imputed encoder embedding
    (hidden_dim)] + [has-encoder-signal indicator (1)] -- same median-
    imputation + paired-indicator convention `stage_combined_classifier.
    _to_feature_matrix()` already established and validated, extended
    from scalar to vector imputation."""
    acoustic = d["acoustic_feats"]
    emb = d["encoder_embeddings"]
    has_enc = d["has_encoder"]
    if has_enc.any():
        median_vec = np.nanmedian(emb[has_enc], axis=0)
    else:
        median_vec = np.zeros(emb.shape[1])
    emb_imputed = np.where(has_enc[:, None], emb, median_vec[None, :])
    return np.column_stack([acoustic, emb_imputed, has_enc.astype(np.float64)])


def _mean_f1(results: list[tuple[float, float, float]]) -> float:
    return float(np.mean([f1 for _, _, f1 in results])) if results else 0.0


def _mean_pr(results: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    arr = np.array(results)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _decide(results_a: list, results_b: list, results_c: list) -> dict:
    """Pre-registered gate (ASR_RESEARCH_TRACK.md "Rank 3" success/failure
    criteria): stability check vs. Arm B first, then classifier must beat
    BOTH Arm A and Arm B by >=20% relative AND clear the absolute floor."""
    b_f1s = [f1 for _, _, f1 in results_b]
    c_f1s = [f1 for _, _, f1 in results_c]
    n = min(len(b_f1s), len(c_f1s))
    wins = sum(1 for i in range(n) if c_f1s[i] > b_f1s[i])
    losses = sum(1 for i in range(n) if b_f1s[i] > c_f1s[i])

    if not (wins >= 4 or losses >= 4):
        return {"verdict": "INCONCLUSIVE", "wins": wins, "losses": losses, "n_folds": n}

    f1_a, f1_b, f1_c = _mean_f1(results_a), _mean_f1(results_b), _mean_f1(results_c)
    bar_a, bar_b = f1_a * 1.2, f1_b * 1.2
    beats_a, beats_b = f1_c > bar_a, f1_c > bar_b
    prec, rec = _mean_pr(results_c)
    clears_floor = prec >= 0.15 and rec >= 0.3
    verdict = "SUCCESS" if (beats_a and beats_b and clears_floor) else "FAILURE"
    return {
        "verdict": verdict, "wins": wins, "losses": losses, "n_folds": n,
        "f1_a": f1_a, "f1_b": f1_b, "f1_c": f1_c, "bar_a": bar_a, "bar_b": bar_b,
        "beats_a": beats_a, "beats_b": beats_b,
        "mean_precision": prec, "mean_recall": rec, "clears_floor": clears_floor,
    }


def run(data_dir: Path | None, audio_dir: Path | None, from_cache: Path | None = None) -> dict:
    if from_cache is not None:
        print(f"Loading cached row-level dataset from {from_cache} (skipping the encoder pass) ...")
        d = _load_rows(from_cache)
    else:
        build_dataset(data_dir, audio_dir)
        d = _load_rows(_ROOT / "eval_results" / "_stage_j_rows_cache.npz")

    labels, clip_ids = d["labels"], d["clip_ids"]
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"=== {TARGET_TYPE} (n_pos={n_pos}, n_neg={n_neg}) ===")
    if n_pos == 0 or n_neg == 0:
        print("Cannot cross-validate: one class is empty.")
        return {}

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    print("=== Arm A: MFCC-similarity-alone, CV threshold ===")
    results_a = _cv_threshold(d["mfcc_similarities"], labels, fold_ids)
    print("  " + _summarize("Arm A (MFCC-alone)", results_a))

    print("\n=== Arm B: encoder-distance-alone, CV threshold (real values only) ===")
    has_enc = d["has_encoder"]
    if has_enc.sum() >= 10 and labels[has_enc].sum() >= 2:
        results_b = _cv_threshold(d["encoder_distances"][has_enc], labels[has_enc], fold_ids[has_enc])
        print("  " + _summarize("Arm B (encoder-distance-alone)", results_b))
    else:
        results_b = []
        print("  n/a -- too few candidates with a real encoder embedding")

    print("\n=== Rank 3: combined (full acoustic + full embedding), nested-CV logistic regression ===")
    X = _to_feature_matrix(d)
    results_c, l2s = _cv_classifier_optimal_threshold(X, labels.astype(np.float64), fold_ids, clip_ids)
    print("  " + _summarize("Rank 3 combined", results_c))
    print(f"    selected L2 per outer fold: {[round(x, 2) for x in l2s]}")

    decision = _decide(results_a, results_b, results_c) if results_b else {"verdict": "N/A -- Arm B unavailable"}
    print(f"\nPer-fold win/loss (Rank 3 vs Arm B F1): {decision.get('wins','-')}W/{decision.get('losses','-')}L "
          f"of {decision.get('n_folds','-')} folds -> {decision['verdict']}")
    if decision["verdict"] not in ("INCONCLUSIVE", "N/A -- Arm B unavailable"):
        print(f"  mean F1: A={decision['f1_a']:.3f}  B={decision['f1_b']:.3f}  C(combined)={decision['f1_c']:.3f}")
        print(f"  bar_A(+20% rel)={decision['bar_a']:.3f} beats_a={decision['beats_a']}  "
              f"bar_B(+20% rel)={decision['bar_b']:.3f} beats_b={decision['beats_b']}")
        print(f"  combined mean precision={decision['mean_precision']:.3f}  "
              f"mean recall={decision['mean_recall']:.3f}  clears_floor(P>=0.15,R>=0.3)={decision['clears_floor']}")

    result = {
        "n_pos": n_pos, "n_neg": n_neg, "n_with_encoder": int(has_enc.sum()),
        "results_a": [list(r) for r in results_a],
        "results_b": [list(r) for r in results_b],
        "results_c": [list(r) for r in results_c],
        "chosen_l2s": l2s,
        "decision": decision,
    }
    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_j_combined_rich_classifier.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return result


# ── Self-test (feature matrix construction, decision gate, row round-trip) ─

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _to_feature_matrix: median imputation for missing embeddings, has-
    #    signal indicator correct, real values pass through unchanged.
    d = {
        "acoustic_feats": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "encoder_embeddings": np.array([[10.0, 20.0], [np.nan, np.nan], [30.0, 40.0]], dtype=np.float32),
        "has_encoder": np.array([True, False, True]),
    }
    X = _to_feature_matrix(d)
    check("feature matrix shape [3, 2+2+1]", X.shape == (3, 5), str(X.shape))
    check("missing embedding row imputed to median of [10,30]/[20,40] = [20,30]",
          np.allclose(X[1, 2:4], [20.0, 30.0]), str(X[1, 2:4]))
    check("real embedding rows pass through unchanged",
          np.allclose(X[0, 2:4], [10.0, 20.0]) and np.allclose(X[2, 2:4], [30.0, 40.0]), str(X[:, 2:4]))
    check("has-signal indicator correct", list(X[:, 4]) == [1.0, 0.0, 1.0], str(X[:, 4]))

    # 2. _to_feature_matrix: all-missing embeddings falls back to zero
    #    imputation, no crash (mirrors stage_combined_classifier's own
    #    documented all-missing fallback).
    d_all_missing = {
        "acoustic_feats": np.array([[1.0, 2.0]]),
        "encoder_embeddings": np.array([[np.nan, np.nan]], dtype=np.float32),
        "has_encoder": np.array([False]),
    }
    X2 = _to_feature_matrix(d_all_missing)
    check("all-missing embeddings impute to 0.0 without crashing",
          np.allclose(X2[0, 2:4], [0.0, 0.0]), str(X2))

    # 3. Decision gate: unanimous win vs B, clears both relative bars + floor -> SUCCESS.
    results_a = [(0.1, 0.1, 0.10)] * 5
    results_b = [(0.2, 0.2, 0.20)] * 5
    results_c_good = [(0.3, 0.5, 0.375)] * 5  # > 1.2*0.10=0.12 and > 1.2*0.20=0.24; P=0.3>=0.15, R=0.5>=0.3
    dcs = _decide(results_a, results_b, results_c_good)
    check("beats both bars + clears floor -> SUCCESS", dcs["verdict"] == "SUCCESS", str(dcs))

    # 4. Decision gate: beats Arm A's bar (0.12) but not Arm B's (0.252) ->
    #    FAILURE. Arm B's own F1 raised slightly above C's so the per-fold
    #    stability check registers a consistent loss (not a tie -> Inconclusive).
    results_b_higher = [(0.21, 0.21, 0.21)] * 5
    results_c_partial = [(0.2, 0.2, 0.20)] * 5  # 0.20 > 0.12 (beats A) but 0.20 < 0.252 (fails B)
    dcp = _decide(results_a, results_b_higher, results_c_partial)
    check("beats Arm A's bar only, not Arm B's -> FAILURE", dcp["verdict"] == "FAILURE" and dcp["beats_a"] and not dcp["beats_b"], str(dcp))

    # 5. Decision gate: mixed per-fold direction vs B -> INCONCLUSIVE.
    results_b_mixed = [(0.1, 0.1, 0.10), (0.1, 0.1, 0.30)] * 2 + [(0.1, 0.1, 0.10)]
    results_c_mixed = [(0.2, 0.2, 0.20), (0.1, 0.1, 0.15)] * 2 + [(0.2, 0.2, 0.20)]
    dmix = _decide(results_a, results_b_mixed, results_c_mixed)
    check("mixed per-fold direction vs Arm B -> INCONCLUSIVE", dmix["verdict"] == "INCONCLUSIVE", str(dmix))

    # 6. Row packing round-trips correctly, including None embeddings.
    rows = [
        {"clip_id": "a", "label": 1, "acoustic_feat": np.arange(26.0), "mfcc_similarity": 0.5,
         "encoder_embedding": np.full(1280, 2.0, dtype=np.float32), "encoder_distance": 0.3},
        {"clip_id": "a", "label": 0, "acoustic_feat": np.arange(26.0) * 2, "mfcc_similarity": 0.2,
         "encoder_embedding": None, "encoder_distance": None},
    ]
    tmp = _ROOT / "eval_results" / "_stage_j_selftest_tmp.npz"
    _save_rows(rows, tmp)
    loaded = _load_rows(tmp)
    check("row round-trip: acoustic_feats shape", loaded["acoustic_feats"].shape == (2, 26), str(loaded["acoustic_feats"].shape))
    check("row round-trip: has_encoder", list(loaded["has_encoder"]) == [True, False], str(loaded["has_encoder"]))
    check("row round-trip: real embedding preserved", np.allclose(loaded["encoder_embeddings"][0], 2.0), "")
    tmp.unlink(missing_ok=True)

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
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
    run(data_dir, audio_dir, from_cache=from_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
