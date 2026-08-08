"""stage_i_learned_acoustic_classifier.py — Rank 2, pre-registered in
`ASR_RESEARCH_TRACK.md`'s "Rank 2: learned acoustic candidate-generation
classifier" section. Read that section before changing any logic here.

Tests whether a nested-CV logistic-regression classifier over each
acoustic candidate's *raw* per-coefficient MFCC statistics (mean + std
across the candidate's bursts) separates true `sound_repetition`
instances from incidental short-burst runs better than direction (g)'s
own hand-designed single-scalar similarity metric (mean pairwise cosine
similarity between consecutive bursts' MFCC vectors) does — under
identical 5-fold clip-split CV, over the SAME candidate population
(candidate spans depend only on voicing segmentation, not on which
similarity feature is computed from them).

No ASR, no encoder anywhere in this script — acoustic-only, matching
direction (g)'s own scope. Deliberately reuses this project's own
existing infrastructure:
- Voicing segmentation + MFCC extraction: `profiling.acoustic.
  segment_voiced`, `stage_g_acoustic_sound_repetition.compute_mfcc` —
  unmodified.
- Candidate-span detection: `stage_g_acoustic_sound_repetition`'s exact
  run-detection loop (`_voiced_runs_with_short_gaps` + the `generate_
  candidates` grouping logic) — re-implemented here as `_generate_
  candidates_with_bursts()` ONLY because `AcousticCandidate` itself
  doesn't expose the individual burst segments this script needs; the
  span-detection logic itself is not changed.
- Baseline + classifier machinery: `compare_corroboration_mechanisms`'s
  `_clip_folds`/`_cv_threshold`/`_summarize` and `stage_combined_
  classifier`'s `_cv_classifier_optimal_threshold` — all unmodified.

Usage
-----
    python -m profiling.evaluation.stage_i_learned_acoustic_classifier \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 -- kept for convention; no torch used here

import numpy as np

from profiling.acoustic import AcousticConfig, Segment, load_wav_samples, segment_voiced
from profiling.config import load_config
from profiling.evaluation.compare_corroboration_mechanisms import (
    _clip_folds,
    _cv_threshold,
    _summarize,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_combined_classifier import _cv_classifier_optimal_threshold
from profiling.evaluation.stage_g_acoustic_sound_repetition import (
    MATCH_TOLERANCE_SECONDS,
    MAX_GAP_SECONDS,
    SHORT_BURST_MAX_SECONDS,
    _overlaps,
    _resampled_mfcc_vector,
    _voiced_runs_with_short_gaps,
    compute_mfcc,
)

TARGET_TYPE = "sound_repetition"
N_MFCC_KEPT = 12  # coefficients 1..12, coefficient 0 (energy) excluded -- established fix


@dataclass
class CandidateWithBursts:
    start: float
    end: float
    n_bursts: int
    burst_spans: list[tuple[float, float]]


def _generate_candidates_with_bursts(
    segments: list[Segment], short_max_s: float = SHORT_BURST_MAX_SECONDS, max_gap_s: float = MAX_GAP_SECONDS,
) -> list[CandidateWithBursts]:
    """Mirrors `stage_g_acoustic_sound_repetition.generate_candidates()`'s
    exact run-detection loop (same thresholds, same grouping condition),
    but returns each candidate's individual burst spans instead of
    pairwise-similarity scalars -- see module docstring for why this is a
    deliberate small duplication, not a modification of the original."""
    runs = _voiced_runs_with_short_gaps(segments, max_gap_s)
    candidates: list[CandidateWithBursts] = []
    for run in runs:
        i = 0
        while i < len(run):
            if run[i].duration >= short_max_s:
                i += 1
                continue
            j = i
            while j + 1 < len(run) and run[j + 1].duration < short_max_s:
                j += 1
            n_bursts = j - i + 1
            if n_bursts >= 2:
                spans = [(run[k].start, run[k].end) for k in range(i, j + 1)]
                candidates.append(CandidateWithBursts(
                    start=run[i].start, end=run[j].end, n_bursts=n_bursts, burst_spans=spans,
                ))
                i = j + 1
            else:
                i += 1
    return candidates


def _burst_feature_vector(cand: CandidateWithBursts, times: np.ndarray, mfcc: np.ndarray) -> np.ndarray | None:
    """26-dim: [mean-across-bursts MFCC (12, c0 excluded), std-across-bursts
    MFCC (12), n_bursts, duration]. None if fewer than 2 bursts have a
    computable MFCC vector (degenerate span, e.g. off the end of the clip)."""
    vecs = []
    for b_start, b_end in cand.burst_spans:
        v = _resampled_mfcc_vector(times, mfcc, b_start, b_end)
        if v is not None:
            vecs.append(v[1:N_MFCC_KEPT + 1])  # drop coefficient 0
    if len(vecs) < 2:
        return None
    arr = np.stack(vecs, axis=0)  # [n_bursts, 12]
    mean_vec = arr.mean(axis=0)
    std_vec = arr.std(axis=0)
    duration = cand.end - cand.start
    return np.concatenate([mean_vec, std_vec, [float(cand.n_bursts), duration]])


def _mean_mfcc_similarity(cand: CandidateWithBursts, times: np.ndarray, mfcc: np.ndarray) -> float | None:
    """Direction (g)'s own baseline feature, recomputed here (not
    imported) because it operates on burst spans this script derives
    independently -- same formula as `_burst_similarity_mfcc`
    (cosine similarity, c0 excluded), averaged over consecutive pairs."""
    vecs = []
    for b_start, b_end in cand.burst_spans:
        v = _resampled_mfcc_vector(times, mfcc, b_start, b_end)
        vecs.append(v[1:] if v is not None else None)
    sims = []
    for a, b in zip(vecs, vecs[1:]):
        if a is None or b is None:
            sims.append(0.0)
            continue
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        sims.append(float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0)
    return sum(sims) / len(sims) if sims else 0.0


def build_dataset(data_dir: Path, audio_dir: Path) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    print(f"{len(clips)} clips have usable audio.\n")

    ac_cfg_dict = dict(load_config().get("profiling", {}).get("detection", {}))
    cfg = AcousticConfig.from_detection_cfg(ac_cfg_dict)

    rows: list[dict] = []
    n_targets_total = 0
    n_clips_with_target = 0
    t0 = time.time()

    for i, clip in enumerate(clips):
        targets = [
            (clip.tokens[ref_idx]["start"], clip.tokens[ref_idx]["end"])
            for ref_idx, t in clip.ground_truth.items() if t == TARGET_TYPE
        ]
        if targets:
            n_clips_with_target += 1
        n_targets_total += len(targets)

        samples, sr = load_wav_samples(clip.audio_bytes)
        if samples is None:
            continue
        segments = segment_voiced(samples, sr, cfg)
        times, mfcc = compute_mfcc(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
        cands = _generate_candidates_with_bursts(segments)

        for cand in cands:
            feat = _burst_feature_vector(cand, times, mfcc)
            sim = _mean_mfcc_similarity(cand, times, mfcc)
            if feat is None:
                continue
            label = 1 if any(
                _overlaps(cand.start, cand.end, t0_ - MATCH_TOLERANCE_SECONDS, t1_ + MATCH_TOLERANCE_SECONDS)
                for t0_, t1_ in targets
            ) else 0
            rows.append({
                "clip_id": clip.name, "label": label,
                "feature": feat, "mfcc_similarity": sim,
            })

        if (i + 1) % 50 == 0 or i + 1 == len(clips):
            print(f"[{i+1}/{len(clips)}] ... ({time.time()-t0:.0f}s elapsed)")

    print(f"\nTotal time: {time.time()-t0:.0f}s for {len(clips)} clips.")
    n_pos = sum(r["label"] for r in rows)
    print(f"{n_targets_total} ground-truth {TARGET_TYPE} instances across {n_clips_with_target} clips.")
    print(f"{len(rows)} acoustic candidates generated, {n_pos} positive (label=1).\n")

    out_path = _ROOT / "eval_results" / "_stage_i_rows_cache.npz"
    _save_rows(rows, out_path)
    print(f"Cached row-level dataset (reusable via --from-cache): {out_path}\n")
    return {"rows": rows, "n_targets_total": n_targets_total, "n_clips_with_target": n_clips_with_target}


def _save_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(
            out_path, features=np.zeros((0, 0), dtype=np.float64),
            similarities=np.zeros(0, dtype=np.float64), labels=np.zeros(0, dtype=np.int64),
            clip_ids=np.empty(0, dtype=object),
        )
        return
    dim = rows[0]["feature"].shape[0]
    n = len(rows)
    features = np.zeros((n, dim), dtype=np.float64)
    similarities = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    clip_ids = np.empty(n, dtype=object)
    for i, r in enumerate(rows):
        features[i] = r["feature"]
        similarities[i] = r["mfcc_similarity"]
        labels[i] = r["label"]
        clip_ids[i] = r["clip_id"]
    np.savez_compressed(
        out_path, features=features, similarities=similarities, labels=labels, clip_ids=clip_ids,
    )


def _load_rows(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "features": d["features"], "similarities": d["similarities"],
        "labels": d["labels"], "clip_ids": d["clip_ids"],
    }


def _mean_f1(results: list[tuple[float, float, float]]) -> float:
    return float(np.mean([f1 for _, _, f1 in results])) if results else 0.0


def _mean_pr(results: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    arr = np.array(results)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _decide(baseline: list[tuple[float, float, float]], classifier: list[tuple[float, float, float]]) -> dict:
    """Same pre-registered decision-gate structure as Rank 1's own
    `_decide()` (ASR_RESEARCH_TRACK.md "Rank 2" success/failure criteria):
    per-fold stability check first, then relative-F1 + absolute
    precision/recall floor. Reimplemented here (not imported) since Rank
    1's version lives in a different, independently-focused module, but
    the logic is intentionally identical -- see the shared floor values."""
    base_f1s = [f1 for _, _, f1 in baseline]
    clf_f1s = [f1 for _, _, f1 in classifier]
    n = min(len(base_f1s), len(clf_f1s))
    wins = sum(1 for i in range(n) if clf_f1s[i] > base_f1s[i])
    losses = sum(1 for i in range(n) if base_f1s[i] > clf_f1s[i])

    if not (wins >= 4 or losses >= 4):
        return {"verdict": "INCONCLUSIVE", "wins": wins, "losses": losses, "n_folds": n}

    f1_base, f1_clf = _mean_f1(baseline), _mean_f1(classifier)
    bar = f1_base * 1.2
    beats_bar = f1_clf > bar
    prec, rec = _mean_pr(classifier)
    clears_floor = prec >= 0.15 and rec >= 0.3
    verdict = "SUCCESS" if (beats_bar and clears_floor) else "FAILURE"
    return {
        "verdict": verdict, "wins": wins, "losses": losses, "n_folds": n,
        "f1_baseline": f1_base, "f1_classifier": f1_clf, "bar": bar,
        "beats_bar": beats_bar, "mean_precision": prec, "mean_recall": rec,
        "clears_floor": clears_floor,
    }


def run(data_dir: Path | None, audio_dir: Path | None, from_cache: Path | None = None) -> dict:
    if from_cache is not None:
        print(f"Loading cached row-level dataset from {from_cache} (skipping candidate generation) ...")
        d = _load_rows(from_cache)
    else:
        build_dataset(data_dir, audio_dir)
        d = _load_rows(_ROOT / "eval_results" / "_stage_i_rows_cache.npz")

    features, similarities = d["features"], d["similarities"]
    labels, clip_ids = d["labels"], d["clip_ids"]

    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"=== {TARGET_TYPE} (n_pos={n_pos}, n_neg={n_neg}) ===")
    if n_pos == 0 or n_neg == 0:
        print("Cannot cross-validate: one class is empty.")
        return {}

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    baseline = _cv_threshold(similarities, labels, fold_ids)
    print("  " + _summarize("Baseline (mean MFCC similarity + CV threshold)", baseline))

    classifier, l2s = _cv_classifier_optimal_threshold(features, labels.astype(np.float64), fold_ids, clip_ids)
    print("  " + _summarize("Rank 2 (raw per-coefficient MFCC stats + nested-CV logistic regression)", classifier))
    print(f"    selected L2 per outer fold: {[round(x, 2) for x in l2s]}")

    decision = _decide(baseline, classifier)
    print(f"\nPer-fold win/loss (classifier vs baseline F1): {decision['wins']}W/{decision['losses']}L "
          f"of {decision['n_folds']} folds -> {decision['verdict']}")
    if decision["verdict"] != "INCONCLUSIVE":
        print(f"  mean F1: baseline={decision['f1_baseline']:.3f}  classifier={decision['f1_classifier']:.3f}  "
              f"bar(+20% rel)={decision['bar']:.3f}  beats_bar={decision['beats_bar']}")
        print(f"  classifier mean precision={decision['mean_precision']:.3f}  "
              f"mean recall={decision['mean_recall']:.3f}  clears_floor(P>=0.15,R>=0.3)={decision['clears_floor']}")

    result = {
        "n_pos": n_pos, "n_neg": n_neg,
        "baseline": [list(r) for r in baseline],
        "classifier": [list(r) for r in classifier],
        "chosen_l2s": l2s,
        "decision": decision,
    }
    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_i_learned_acoustic_classifier.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return result


# ── Self-test (candidate/burst extraction, feature construction, decision gate) ─

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _generate_candidates_with_bursts: a 2-short-burst run followed by a
    #    longer segment yields exactly one candidate with 2 burst spans.
    segs = [
        Segment(start=0.0, end=0.15, voiced=True, rms=0.1, zcr=0.1),
        Segment(start=0.15, end=0.20, voiced=False, rms=0.0, zcr=0.0),
        Segment(start=0.20, end=0.35, voiced=True, rms=0.1, zcr=0.1),
        Segment(start=0.35, end=0.40, voiced=False, rms=0.0, zcr=0.0),
        Segment(start=0.40, end=1.0, voiced=True, rms=0.1, zcr=0.1),
    ]
    cands = _generate_candidates_with_bursts(segs)
    check("one candidate, 2 burst spans, matching (start,end)",
          len(cands) == 1 and cands[0].n_bursts == 2 and cands[0].burst_spans == [(0.0, 0.15), (0.20, 0.35)],
          str(cands))

    # 2. A single short burst with no repeat does not yield a candidate.
    single = [
        Segment(start=0.0, end=0.15, voiced=True, rms=0.1, zcr=0.1),
        Segment(start=0.15, end=1.0, voiced=False, rms=0.0, zcr=0.0),
    ]
    check("a single short burst with no repeat yields no candidate",
          len(_generate_candidates_with_bursts(single)) == 0, "")

    # 3. _burst_feature_vector: identical bursts (same MFCC vector every
    #    frame) produce zero inter-burst std and a well-formed 26-dim vector.
    sr = 16000
    times = np.arange(0.0, 0.5, 0.01)
    mfcc = np.tile(np.arange(1.0, 14.0), (len(times), 1))  # every frame identical, coeffs 1..13
    cand = CandidateWithBursts(start=0.0, end=0.3, n_bursts=2, burst_spans=[(0.0, 0.1), (0.15, 0.25)])
    feat = _burst_feature_vector(cand, times, mfcc)
    check("feature vector has 26 dims", feat is not None and feat.shape == (26,), str(feat))
    check("identical bursts -> zero inter-burst std (dims 12:24)",
          feat is not None and np.allclose(feat[12:24], 0.0), str(feat[12:24]) if feat is not None else "None")
    check("mean vector (dims 0:12) matches the constant coefficients (2..13, c0 dropped)",
          feat is not None and np.allclose(feat[0:12], np.arange(2.0, 14.0)), str(feat[0:12]) if feat is not None else "None")
    check("n_bursts/duration carried through (dims 24,25)",
          feat is not None and feat[24] == 2.0 and abs(feat[25] - 0.3) < 1e-9, str(feat[24:26]) if feat is not None else "None")

    # 4. _burst_feature_vector: fewer than 2 resolvable bursts -> None, not a crash.
    cand_degenerate = CandidateWithBursts(start=10.0, end=10.3, n_bursts=2, burst_spans=[(10.0, 10.1), (10.15, 10.25)])
    feat_none = _burst_feature_vector(cand_degenerate, times, mfcc)  # spans entirely outside `times`
    check("out-of-range burst spans -> None (no crash)", feat_none is None, str(feat_none))

    # 5. _mean_mfcc_similarity: identical bursts score similarity ~1.0.
    sim = _mean_mfcc_similarity(cand, times, mfcc)
    check("identical bursts -> similarity close to 1.0", sim is not None and sim > 0.999, str(sim))

    # 6. Decision gate: unanimous win + clears floor -> SUCCESS (same logic as Rank 1's).
    baseline = [(0.1, 0.1, 0.1)] * 5
    classifier_good = [(0.3, 0.5, 0.375)] * 5
    d = _decide(baseline, classifier_good)
    check("decision gate: unanimous win + clears floor -> SUCCESS", d["verdict"] == "SUCCESS", str(d))

    # 7. Decision gate: mixed direction -> INCONCLUSIVE.
    baseline_mixed = [(0.1, 0.1, 0.10), (0.1, 0.1, 0.30)] * 2 + [(0.1, 0.1, 0.10)]
    classifier_mixed = [(0.2, 0.2, 0.20), (0.1, 0.1, 0.15)] * 2 + [(0.2, 0.2, 0.20)]
    d2 = _decide(baseline_mixed, classifier_mixed)
    check("decision gate: mixed per-fold direction -> INCONCLUSIVE", d2["verdict"] == "INCONCLUSIVE", str(d2))

    # 8. Row packing round-trips correctly.
    rows = [
        {"clip_id": "a", "label": 1, "feature": np.arange(26.0), "mfcc_similarity": 0.5},
        {"clip_id": "a", "label": 0, "feature": np.arange(26.0) * 2, "mfcc_similarity": 0.2},
    ]
    tmp = _ROOT / "eval_results" / "_stage_i_selftest_tmp.npz"
    _save_rows(rows, tmp)
    loaded = _load_rows(tmp)
    check("row round-trip: features shape", loaded["features"].shape == (2, 26), str(loaded["features"].shape))
    check("row round-trip: labels", list(loaded["labels"]) == [1, 0], str(loaded["labels"]))
    tmp.unlink(missing_ok=True)

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--from-cache", default=None,
                         help="Reuse a previously-cached row-level dataset (skips candidate generation).")
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
