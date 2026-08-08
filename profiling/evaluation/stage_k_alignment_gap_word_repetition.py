"""stage_k_alignment_gap_word_repetition.py -- pre-registered in
VALIDATION.md section 15 ("word_repetition alignment-gap / duration-residual
candidate generator"). Read that section before changing any logic here.

Tests whether a single, zero-training, zero-new-dependency signal --
CrisperWhisper's own already-computed word-level timestamps, turned into a
clip-relative duration+silence residual z-score -- separates Stage A's
`word_repetition` category-1 positions (ASR transcribed the labeled word
correctly, but generated no candidate at all because the *other* half of
the repeated pair was deleted from its own output) from clean positions in
the same clips.

Deliberately reuses this project's own existing infrastructure rather than
adding anything new:
- Population identification: `stage_b_representation_probe._identify_
  positions()`, unmodified, filtered to `word_repetition` only.
- Threshold/CV machinery: `compare_corroboration_mechanisms`'s
  `_clip_folds`/`_cv_threshold`/`_summarize`, unmodified.
- AUC: `stage_c_duration_baseline._auc`, unmodified.
- Cohen's d: `stage_b_representation_probe._cohens_d`, unmodified.
- ASR output: the existing Track B cache (`eval_datasets/_track_b_cache`)
  -- no new ASR inference, no new model, no new library.

Usage
-----
    python -m profiling.evaluation.stage_k_alignment_gap_word_repetition \\
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

from profiling.evaluation.compare_corroboration_mechanisms import _clip_folds, _cv_threshold, _summarize
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_b_representation_probe import _cohens_d, _identify_positions
from profiling.evaluation.stage_c_duration_baseline import _auc
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached

TARGET_TYPE = "word_repetition"

# This project's own already-established candidate-generator floor (used
# identically for Rank 1 and Rank 2, VALIDATION.md section 12.4/`ASR_
# RESEARCH_TRACK.md` "Rank 1"/"Rank 2" pre-registrations) -- reused here
# for consistency, not tuned to this experiment's own result (standing
# rule 4).
PRECISION_FLOOR = 0.15
RECALL_FLOOR = 0.3


def _residual(hyp_tokens: list[dict], h: int) -> float | None:
    """duration(h) + gap_before(h), VALIDATION.md section 15.4. None if
    either timestamp needed is missing (rare, e.g. a beam-search timestamp
    gap already documented in `asr.py`) -- excluded, not imputed."""
    tok = hyp_tokens[h]
    start, end = tok.get("start"), tok.get("end")
    if start is None or end is None:
        return None
    duration = end - start
    gap_before = 0.0
    if h > 0:
        prev_end = hyp_tokens[h - 1].get("end")
        if prev_end is not None and start is not None:
            gap_before = max(0.0, start - prev_end)
    return duration + gap_before


def _clip_residual_stats(hyp_tokens: list[dict]) -> tuple[float, float] | None:
    """Clip-relative mean/SD of `_residual` across every hyp token in the
    clip (not just target/clean positions) -- VALIDATION.md section 15.4's
    clip-relative z-scoring, controlling for narrator speaking-rate
    differences between clips the same way Stage C already did for
    duration. None if fewer than 2 usable values or zero variance."""
    vals = [r for h in range(len(hyp_tokens)) if (r := _residual(hyp_tokens, h)) is not None]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    sd = var ** 0.5
    if sd == 0:
        return None
    return mean, sd


def build_dataset(data_dir: Path, audio_dir: Path, cache_dir: Path = _DEFAULT_CACHE_DIR) -> list[dict]:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips.sort(key=lambda c: c.name)  # deterministic iteration order
    print(f"{len(clips)} clips have usable audio.\n")

    rows: list[dict] = []
    n_scanned = 0
    n_with_target = 0
    for clip in clips:
        hyp_tokens = _load_cached(cache_dir, clip.name)
        if hyp_tokens is None:
            continue  # no new ASR run -- pre-registration section 15.2
        n_scanned += 1

        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        targets = [(ri, hi, tt) for (ri, hi, tt) in targets if tt == TARGET_TYPE]
        if not targets:
            continue
        n_with_target += 1

        stats = _clip_residual_stats(hyp_tokens)
        if stats is None:
            continue
        mean_r, sd_r = stats

        def z(h: int) -> float | None:
            r = _residual(hyp_tokens, h)
            return None if r is None else (r - mean_r) / sd_r

        for _ref_idx, hyp_idx, _true_type in targets:
            zi = z(hyp_idx)
            if zi is not None:
                rows.append({"clip_id": clip.name, "label": 1, "residual_z": zi})

        for _ref_idx, hyp_idx in clean_positions:
            zi = z(hyp_idx)
            if zi is not None:
                rows.append({"clip_id": clip.name, "label": 0, "residual_z": zi})

    print(f"{n_scanned} clips had cached Track B ASR output and were scanned.")
    print(f"{n_with_target} clips contain at least one word_repetition category-1 target position.\n")
    return rows


def _bootstrap_auc_ci(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """Clip-level bootstrap 95% CI on AUC -- resamples CLIPS, not rows, so
    a clip's own correlated positions move together. Added in direct
    response to the external review's own critique (`ASR_RESEARCH_
    TRACK.md`'s round-3 reconciliation, section 5 of `EXTERNAL_
    REVIEW_2026-08-07.md`) that no confidence interval had ever been
    computed anywhere in this track -- this experiment does not repeat
    that gap."""
    rng = np.random.default_rng(seed)
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    clip_list = list(by_clip.keys())
    if len(clip_list) < 2:
        return None
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(clip_list, size=len(clip_list), replace=True)
        pos: list[float] = []
        neg: list[float] = []
        for c in sampled:
            for r in by_clip[c]:
                (pos if r["label"] == 1 else neg).append(r["residual_z"])
        if pos and neg:
            aucs.append(_auc(pos, neg))
    if len(aucs) < 100:
        return None
    aucs.sort()
    lo = aucs[int(0.025 * len(aucs))]
    hi = aucs[min(len(aucs) - 1, int(0.975 * len(aucs)))]
    return lo, hi


def _mean_pr(results: list[tuple[float, float, float]]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    arr = np.array(results)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def _decide(auc: float, auc_ci: tuple[float, float] | None, cv_results: list[tuple[float, float, float]]) -> dict:
    """VALIDATION.md section 15.6, applied exactly as pre-registered
    before this script was run against real data."""
    if auc_ci is None:
        signal_real = None  # not enough clips to bootstrap -- honestly unresolved, not guessed at
    else:
        signal_real = auc_ci[0] > 0.5  # CI excludes chance
    mean_p, mean_r = _mean_pr(cv_results)
    clears_floor = mean_p >= PRECISION_FLOOR and mean_r >= RECALL_FLOOR

    if signal_real is None:
        verdict = "INCONCLUSIVE"
    elif not signal_real:
        verdict = "FAILURE"
    elif signal_real and clears_floor:
        verdict = "SUCCESS"
    else:
        verdict = "FAILURE"  # real signal, but not a non-trivial recovery -- same pattern as Stage B/C

    return {
        "verdict": verdict, "auc": auc, "auc_ci": auc_ci, "signal_real": signal_real,
        "mean_precision": mean_p, "mean_recall": mean_r, "clears_floor": clears_floor,
        "precision_floor": PRECISION_FLOOR, "recall_floor": RECALL_FLOOR,
    }


def run(data_dir: Path, audio_dir: Path, cache_dir: Path = _DEFAULT_CACHE_DIR) -> dict:
    rows = build_dataset(data_dir, audio_dir, cache_dir)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"=== word_repetition alignment-gap residual (n_pos={n_pos}, n_neg={n_neg}) ===\n")

    if n_pos == 0 or n_neg == 0:
        print("Cannot evaluate: one class is empty (too small a sample in this cache).")
        result = {"verdict": "INCONCLUSIVE", "reason": "empty class", "n_pos": n_pos, "n_neg": n_neg}
        _save(result)
        return result

    labels = np.array([r["label"] for r in rows])
    clip_ids = np.array([r["clip_id"] for r in rows])
    residual_z = np.array([r["residual_z"] for r in rows])

    pos_scores = residual_z[labels == 1].tolist()
    neg_scores = residual_z[labels == 0].tolist()
    d = _cohens_d(pos_scores, neg_scores)
    auc = _auc(pos_scores, neg_scores)
    auc_ci = _bootstrap_auc_ci(rows)
    print(f"Cohen's d: {d}")
    print(f"AUC: {auc:.3f}" + (f"  (95% bootstrap CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}])" if auc_ci else "  (CI not computed -- too few clips)"))

    fold_map = _clip_folds(clip_ids)
    fold_ids = np.array([fold_map[c] for c in clip_ids])
    cv_results = _cv_threshold(residual_z, labels, fold_ids)
    print("  " + _summarize("Alignment-gap residual_z + CV threshold", cv_results))

    decision = _decide(auc, auc_ci, cv_results)
    print(f"\nVerdict: {decision['verdict']}")
    print(f"  mean precision={decision['mean_precision']:.3f}  mean recall={decision['mean_recall']:.3f}  "
          f"floor(P>={PRECISION_FLOOR}, R>={RECALL_FLOOR})={decision['clears_floor']}")

    result = {
        "n_pos": n_pos, "n_neg": n_neg, "cohens_d": d, "auc": auc, "auc_ci": auc_ci,
        "cv_results": [list(r) for r in cv_results], "decision": decision,
    }
    _save(result)
    return result


def _save(result: dict) -> None:
    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_k_alignment_gap_word_repetition.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {out_path}")


# ── Self-test (residual math + decision-gate logic, hand-constructed, no real ASR) ──

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _residual: duration + gap-before, first token has no gap.
    toks = [
        {"word": "i", "start": 0.0, "end": 0.2},
        {"word": "want", "start": 0.5, "end": 0.7},  # 0.3s gap before it
        {"word": "coffee", "start": 0.7, "end": 0.9},
    ]
    check("first token has zero gap_before", _residual(toks, 0) == 0.2, str(_residual(toks, 0)))
    check("second token's residual includes the 0.3s gap",
          abs(_residual(toks, 1) - (0.2 + 0.3)) < 1e-9, str(_residual(toks, 1)))
    check("missing timestamp returns None",
          _residual([{"word": "x", "start": None, "end": None}], 0) is None)

    # 2. _clip_residual_stats: mean/SD over all tokens, ignoring missing ones.
    stats = _clip_residual_stats(toks)
    check("clip stats computed when enough usable values", stats is not None, str(stats))

    # 3. Decision gate: real signal (CI excludes 0.5) + clears floor -> SUCCESS.
    d = _decide(auc=0.8, auc_ci=(0.6, 0.95), cv_results=[(0.3, 0.5, 0.375)] * 5)
    check("CI excludes chance + clears floor -> SUCCESS", d["verdict"] == "SUCCESS", str(d))

    # 4. Decision gate: CI excludes chance but recall/precision too low -> FAILURE.
    d2 = _decide(auc=0.65, auc_ci=(0.55, 0.8), cv_results=[(0.05, 0.1, 0.067)] * 5)
    check("real signal but below floor -> FAILURE", d2["verdict"] == "FAILURE", str(d2))

    # 5. Decision gate: CI includes 0.5 -> FAILURE regardless of floor.
    d3 = _decide(auc=0.55, auc_ci=(0.45, 0.7), cv_results=[(0.3, 0.5, 0.375)] * 5)
    check("CI includes chance -> FAILURE even if floor cleared", d3["verdict"] == "FAILURE", str(d3))

    # 6. Decision gate: too few clips to bootstrap -> INCONCLUSIVE, not guessed.
    d4 = _decide(auc=0.8, auc_ci=None, cv_results=[(0.3, 0.5, 0.375)] * 5)
    check("no CI available -> INCONCLUSIVE, not forced to SUCCESS/FAILURE",
          d4["verdict"] == "INCONCLUSIVE", str(d4))

    # 7. Bootstrap CI: degenerate single-clip input returns None, not a crash.
    one_clip_rows = [{"clip_id": "a", "label": 1, "residual_z": 1.0},
                      {"clip_id": "a", "label": 0, "residual_z": 0.0}]
    check("bootstrap CI on a single clip returns None (can't resample variation)",
          _bootstrap_auc_ci(one_clip_rows) is None)

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.data_dir or not args.audio_dir:
        print("--data-dir and --audio-dir are required (unless --self-test).")
        return 2
    data_dir, audio_dir = Path(args.data_dir), Path(args.audio_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else _DEFAULT_CACHE_DIR
    run(data_dir, audio_dir, cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
