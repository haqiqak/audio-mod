"""stage_h_candidate_generation_classifier.py — Rank 1, pre-registered in
`ASR_RESEARCH_TRACK.md`'s "Rank 1: full-embedding candidate-generation
classifier" section. Read that section before changing any logic here.

Tests whether a full-raw-embedding, nested-CV logistic-regression
classifier (item 17's own shipped mechanism, (S1-full, M3)) separates
Stage B/C's real candidate-generation-gap population (`sound_repetition`/
`word_repetition` positions where CrisperWhisper's decoded text gives NO
surface evidence at all, vs. clean positions) better than Stage C's own
single-scalar (S1, M1) baseline (cosine distance to the fluent centroid +
cross-validated threshold), under identical 5-fold clip-split CV.

Deliberately reuses this project's own existing infrastructure rather than
adding anything new:
- Population identification: `stage_b_representation_probe._identify_
  positions()`, unmodified — the exact "category 1" target/control
  definition Stage B/C already used.
- Encoder primitives: `profiling.encoder_embedding` (`load_encoder`,
  `extract_last_layer_states`, `pool_span`, `cosine_distance`) — same
  ones Stage B/C used.
- Baseline + classifier machinery: `compare_corroboration_mechanisms`'s
  `_cv_threshold`/`_clip_folds`/`_summarize` and `stage_combined_
  classifier`'s `_cv_classifier_optimal_threshold` (the train-fold-
  optimal-threshold fix, needed from the start here given this
  population is even more imbalanced than the one that fix was built
  for) — all unmodified.

Usage
-----
    python -m profiling.evaluation.stage_h_candidate_generation_classifier \\
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

from profiling.acoustic import load_wav_samples
from profiling.encoder_embedding import cosine_distance, extract_last_layer_states, load_encoder, pool_span
from profiling.evaluation.compare_corroboration_mechanisms import (
    _clip_folds,
    _cv_threshold,
    _summarize,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_b_representation_probe import _identify_positions
from profiling.evaluation.stage_combined_classifier import _cv_classifier_optimal_threshold
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached

TARGET_TYPES = ("sound_repetition", "word_repetition")

# Stage B/C's own already-published counts, for the provenance check (§4's
# pre-registered "Provenance note") -- NOT asserted equal (this cache may
# legitimately differ from Stage B/C's own session state), only compared
# and reported.
_STAGE_B_REFERENCE_COUNTS = {"sound_repetition": 19, "word_repetition": 17, "control": 966}


def build_dataset(data_dir: Path, audio_dir: Path, cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips.sort(key=lambda c: c.name)  # deterministic iteration order
    print(f"{len(clips)} clips have usable audio.\n")

    # Pass 1: identify target/clean positions per clip using only cached
    # hyp_tokens + text-based candidate matching -- no encoder cost yet.
    per_clip = {}
    n_scanned = 0
    for clip in clips:
        hyp_tokens = _load_cached(cache_dir, clip.name)
        if hyp_tokens is None:
            continue
        n_scanned += 1
        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        if targets:
            per_clip[clip.name] = {
                "clip": clip, "hyp_tokens": hyp_tokens,
                "targets": targets, "clean_positions": clean_positions,
            }

    print(f"{n_scanned} clips had cached Track B ASR output and were scanned.")
    print(f"{len(per_clip)} clips contain at least one Stage-A category-1 target position.")
    n_by_type: dict[str, int] = {}
    for v in per_clip.values():
        for _, _, t in v["targets"]:
            n_by_type[t] = n_by_type.get(t, 0) + 1
    print(f"Target counts by type: {n_by_type}")
    for t, ref_n in _STAGE_B_REFERENCE_COUNTS.items():
        if t == "control":
            continue
        got = n_by_type.get(t, 0)
        if got != ref_n:
            print(f"  NOTE: {t} count ({got}) differs from Stage B/C's own saved "
                  f"figure ({ref_n}) -- expected if the Track B cache has changed "
                  f"since that session (pre-registration's Provenance note), not "
                  f"treated as an error.")
    print()

    # Pass 2: real encoder pass, one per clip that has a target.
    processor, encoder = load_encoder()
    rows: list[dict] = []
    t0 = time.time()
    for i, (name, rec) in enumerate(per_clip.items()):
        clip, hyp_tokens = rec["clip"], rec["hyp_tokens"]
        c0 = time.time()
        samples, sr = load_wav_samples(clip.audio_bytes)
        if samples is None:
            continue
        states = extract_last_layer_states(processor, encoder, samples, sr)
        print(f"[{i+1}/{len(per_clip)}] {name} ... ({time.time()-c0:.0f}s, {time.time()-t0:.0f}s elapsed)")

        clean = rec["clean_positions"]
        clean_vecs: dict[int, np.ndarray] = {}
        for ref_idx, hyp_idx in clean:
            tok = hyp_tokens[hyp_idx]
            v = pool_span(states, tok.get("start"), tok.get("end"))
            if v is not None:
                clean_vecs[hyp_idx] = v
        if len(clean_vecs) < 2:
            continue  # can't build a meaningful centroid/leave-one-out baseline

        all_vecs = list(clean_vecs.values())
        sum_vec = np.sum(all_vecs, axis=0)
        n_clean = len(all_vecs)
        full_centroid = sum_vec / n_clean

        # Targets: full embedding + distance to the FULL clean centroid
        # (targets are never part of the clean set -- no leave-one-out
        # needed, same asymmetry Stage B/C used).
        for ref_idx, hyp_idx, true_type in rec["targets"]:
            tok = hyp_tokens[hyp_idx]
            v = pool_span(states, tok.get("start"), tok.get("end"))
            if v is None:
                continue
            d = cosine_distance(v, full_centroid)
            if d is None:
                continue
            rows.append({
                "clip_id": name, "type": true_type, "label": 1,
                "embedding": v, "distance": d,
            })

        # Controls: full embedding + leave-one-out distance.
        for hyp_idx, v in clean_vecs.items():
            loo_centroid = (sum_vec - v) / (n_clean - 1)
            d = cosine_distance(v, loo_centroid)
            if d is None:
                continue
            rows.append({
                "clip_id": name, "type": "control", "label": 0,
                "embedding": v, "distance": d,
            })

    print(f"\nTotal encoder time: {time.time()-t0:.0f}s for {len(per_clip)} clips.\n")

    out_path = _ROOT / "eval_results" / "_stage_h_rows_cache.npz"
    _save_rows(rows, out_path)
    print(f"Cached row-level dataset (reusable via --from-cache): {out_path}\n")
    return {"rows": rows}


def _save_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        np.savez_compressed(
            out_path, embeddings=np.zeros((0, 0), dtype=np.float32),
            distances=np.zeros(0, dtype=np.float64), labels=np.zeros(0, dtype=np.int64),
            types=np.empty(0, dtype=object), clip_ids=np.empty(0, dtype=object),
        )
        return
    hidden_dim = rows[0]["embedding"].shape[0]
    n = len(rows)
    embeddings = np.zeros((n, hidden_dim), dtype=np.float32)
    distances = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    types = np.empty(n, dtype=object)
    clip_ids = np.empty(n, dtype=object)
    for i, r in enumerate(rows):
        embeddings[i] = r["embedding"]
        distances[i] = r["distance"]
        labels[i] = r["label"]
        types[i] = r["type"]
        clip_ids[i] = r["clip_id"]
    np.savez_compressed(
        out_path, embeddings=embeddings, distances=distances,
        labels=labels, types=types, clip_ids=clip_ids,
    )


def _load_rows(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "embeddings": d["embeddings"], "distances": d["distances"],
        "labels": d["labels"], "types": d["types"], "clip_ids": d["clip_ids"],
    }


def _mean_f1(results: list[tuple[float, float, float]]) -> float:
    return float(np.mean([f1 for _, _, f1 in results])) if results else 0.0


def _mean_pr(results: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    arr = np.array(results)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _per_fold_f1(results: list[tuple[float, float, float]]) -> list[float]:
    return [f1 for _, _, f1 in results]


def _decide(baseline: list[tuple[float, float, float]], classifier: list[tuple[float, float, float]]) -> dict:
    """Applies the pre-registered decision gate (ASR_RESEARCH_TRACK.md
    "Rank 1" §6) exactly: per-fold win/loss stability check first, then
    the relative-F1 + absolute-precision/recall floor."""
    base_f1s = _per_fold_f1(baseline)
    clf_f1s = _per_fold_f1(classifier)
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


def run(data_dir: Path | None, audio_dir: Path | None, from_cache: Path | None = None,
        cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict:
    if from_cache is not None:
        print(f"Loading cached row-level dataset from {from_cache} (skipping the encoder pass) ...")
        d = _load_rows(from_cache)
    else:
        data = build_dataset(data_dir, audio_dir, cache_dir)
        d = _load_rows(_ROOT / "eval_results" / "_stage_h_rows_cache.npz")

    embeddings, distances = d["embeddings"], d["distances"]
    labels, types, clip_ids = d["labels"], d["types"], d["clip_ids"]

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])

    n_pos_total = {t: int((types == t).sum()) for t in TARGET_TYPES}
    n_control = int((types == "control").sum())
    print(f"Row-level population: {n_pos_total}, control={n_control}\n")

    populations = [
        ("Any (word_repetition + sound_repetition)", TARGET_TYPES),
        ("sound_repetition", ("sound_repetition",)),
        ("word_repetition", ("word_repetition",)),
    ]

    all_results = {}
    for label_name, type_filter in populations:
        mask = np.isin(types, type_filter + ("control",))
        n_pos = int(labels[mask].sum())
        n_neg = int(mask.sum() - n_pos)
        print(f"=== {label_name} (n_pos={n_pos}, n_neg={n_neg}) ===")
        if n_pos == 0 or n_neg == 0:
            print("  Cannot cross-validate: one class is empty.\n")
            continue

        baseline = _cv_threshold(distances[mask], labels[mask], fold_ids[mask])
        print("  " + _summarize("Baseline (S1, M1) distance-to-centroid + CV threshold", baseline))

        classifier, l2s = _cv_classifier_optimal_threshold(
            embeddings[mask], labels[mask].astype(np.float64), fold_ids[mask], clip_ids[mask],
        )
        print("  " + _summarize("Rank 1 (S1-full, M3) full embedding + nested-CV logistic regression", classifier))
        print(f"    selected L2 per outer fold: {[round(x, 2) for x in l2s]}")

        decision = _decide(baseline, classifier)
        print(f"  Per-fold win/loss (classifier vs baseline F1): {decision['wins']}W/{decision['losses']}L "
              f"of {decision['n_folds']} folds -> {decision['verdict']}")
        if decision["verdict"] != "INCONCLUSIVE":
            print(f"    mean F1: baseline={decision['f1_baseline']:.3f}  classifier={decision['f1_classifier']:.3f}  "
                  f"bar(+20% rel)={decision['bar']:.3f}  beats_bar={decision['beats_bar']}")
            print(f"    classifier mean precision={decision['mean_precision']:.3f}  "
                  f"mean recall={decision['mean_recall']:.3f}  clears_floor(P>=0.15,R>=0.3)={decision['clears_floor']}")
        print()

        all_results[label_name] = {
            "n_pos": n_pos, "n_neg": n_neg,
            "baseline": [list(r) for r in baseline],
            "classifier": [list(r) for r in classifier],
            "chosen_l2s": l2s,
            "decision": decision,
        }

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_h_candidate_generation_classifier.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out_path}")
    return all_results


# ── Self-test (decision-gate logic + row packing, hand-constructed) ────────

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. Decision gate: unanimous win (5/5) + clears both bars -> SUCCESS.
    baseline = [(0.1, 0.1, 0.1)] * 5
    classifier = [(0.3, 0.5, 0.375)] * 5  # F1=0.375 > 0.1*1.2=0.12; P=0.3>=0.15, R=0.5>=0.3
    d = _decide(baseline, classifier)
    check("unanimous win + clears floor -> SUCCESS", d["verdict"] == "SUCCESS", str(d))

    # 2. Decision gate: unanimous win but fails absolute floor -> FAILURE.
    classifier_low_prec = [(0.05, 0.5, 0.091)] * 5  # beats relative bar but P<0.15
    d2 = _decide(baseline, classifier_low_prec)
    check("beats relative bar but fails absolute floor -> FAILURE", d2["verdict"] == "FAILURE", str(d2))

    # 3. Decision gate: mixed per-fold direction (3 wins, 2 losses out of 5) -> INCONCLUSIVE.
    baseline_mixed = [(0.1, 0.1, 0.10), (0.1, 0.1, 0.30), (0.1, 0.1, 0.10), (0.1, 0.1, 0.30), (0.1, 0.1, 0.10)]
    classifier_mixed = [(0.2, 0.2, 0.20), (0.1, 0.1, 0.15), (0.2, 0.2, 0.20), (0.1, 0.1, 0.15), (0.2, 0.2, 0.20)]
    d3 = _decide(baseline_mixed, classifier_mixed)
    check("mixed per-fold direction (3W/2L) -> INCONCLUSIVE", d3["verdict"] == "INCONCLUSIVE", str(d3))

    # 4. Decision gate: unanimous loss (5/5) -> FAILURE (never SUCCESS on a loss).
    classifier_worse = [(0.05, 0.05, 0.05)] * 5
    d4 = _decide(baseline, classifier_worse)
    check("unanimous loss -> FAILURE, not SUCCESS", d4["verdict"] == "FAILURE", str(d4))

    # 5. Row packing/unpacking round-trips embeddings, labels, types, clip_ids correctly.
    rows = [
        {"clip_id": "a", "type": "sound_repetition", "label": 1, "embedding": np.array([1.0, 2.0, 3.0]), "distance": 0.5},
        {"clip_id": "a", "type": "control", "label": 0, "embedding": np.array([4.0, 5.0, 6.0]), "distance": 0.2},
    ]
    tmp = _ROOT / "eval_results" / "_stage_h_selftest_tmp.npz"
    _save_rows(rows, tmp)
    loaded = _load_rows(tmp)
    check("row round-trip: embeddings shape", loaded["embeddings"].shape == (2, 3), str(loaded["embeddings"].shape))
    check("row round-trip: labels", list(loaded["labels"]) == [1, 0], str(loaded["labels"]))
    check("row round-trip: types", list(loaded["types"]) == ["sound_repetition", "control"], str(loaded["types"]))
    tmp.unlink(missing_ok=True)

    # 6. Empty rows list saves/loads a well-shaped, zero-length file (no crash).
    tmp2 = _ROOT / "eval_results" / "_stage_h_selftest_empty_tmp.npz"
    _save_rows([], tmp2)
    loaded_empty = _load_rows(tmp2)
    check("empty rows: zero-length arrays, no crash", len(loaded_empty["labels"]) == 0, str(loaded_empty["labels"]))
    tmp2.unlink(missing_ok=True)

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
