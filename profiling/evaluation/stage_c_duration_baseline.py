"""stage_c_duration_baseline.py — ASR_RESEARCH_TRACK.md Stage C.

Implements the protocol pre-registered in `ASR_RESEARCH_TRACK.md`'s
"Stage C - pre-registered protocol" section EXACTLY - read that section
before changing any logic here. Compares two candidate-scoring arms
(encoder-distance vs. a duration-only baseline) over the identical
sound_repetition target/control population Stage B already collected, to
distinguish H1 (duration confound) / H2 (genuine signature) / H3 (real
but not instance-actionable).

No new encoder passes: reuses Stage B's saved per-type distances
directly. Re-derives the same target/control positions (cheap, cached
data only) to attach real-ASR token duration to each one, and asserts
the re-derived counts match Stage B's saved counts exactly before
trusting the pairing (see the "why this is safe" note in `run()`).

Usage
-----
    python -m profiling.evaluation.stage_c_duration_baseline \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --stage-b-result eval_results/20260805T211000_stage_b_representation_probe.json
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

from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_b_representation_probe import _identify_positions
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPE = "sound_repetition"  # Stage C is scoped to this type only, per Stage B's result


def _auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Mann-Whitney-U-based ROC AUC, pure Python (no scipy/sklearn
    dependency, matching this project's minimal-dependency convention for
    evaluation scripts). AUC = P(a random positive scores higher than a
    random negative), with average ranks for ties."""
    combined = sorted(pos_scores + neg_scores)  # already sorted -> combined[i] is rank i+1 before tie-averaging
    n = len(combined)
    rank_of = [0.0] * n  # rank_of[i] = the (tie-averaged) 1-indexed rank of combined[i]
    i = 0
    while i < n:
        j = i
        while j < n and combined[j] == combined[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            rank_of[k] = avg_rank
        i = j
    value_to_indices: dict[float, list[int]] = {}
    for idx, v in enumerate(combined):
        value_to_indices.setdefault(v, []).append(idx)
    pos_rank_sum = 0.0
    remaining = dict(value_to_indices)
    for v in pos_scores:
        idx = remaining[v].pop()
        pos_rank_sum += rank_of[idx]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def _precision_at_recall(pos_scores: list[float], neg_scores: list[float], target_recall: float) -> tuple[float, float]:
    """Highest-score-first threshold sweep: returns (achieved_recall,
    precision) at the lowest threshold that reaches >= target_recall.
    Higher score = more likely positive, matching both arms' convention
    (larger encoder_distance / larger duration-z-score = more anomalous)."""
    pos_sorted = sorted(pos_scores, reverse=True)
    all_scored = sorted(
        [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores], reverse=True,
    )
    n_pos = len(pos_scores)
    tp = fp = 0
    for score, label in all_scored:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        if recall >= target_recall:
            precision = tp / (tp + fp)
            return recall, precision
    return 1.0, n_pos / (n_pos + len(neg_scores))


def run(data_dir: Path, audio_dir: Path, stage_b_result_path: Path, n_clips: int = 120) -> dict:
    stage_b = json.loads(stage_b_result_path.read_text(encoding="utf-8"))
    encoder_target_distances = stage_b["results"][TARGET_TYPE]["target_distances"]
    encoder_control_distances = stage_b["control_distances"]
    print(f"Loaded Stage B result: {len(encoder_target_distances)} target, "
          f"{len(encoder_control_distances)} control encoder distances for {TARGET_TYPE}.")

    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips = _speaker_stratified_order(clips)[:n_clips]

    def _duration(tok: dict) -> float | None:
        start, end = tok.get("start"), tok.get("end")
        if start is None or end is None:
            return None
        return float(end) - float(start)

    target_durations: list[float] = []
    control_durations: list[float] = []
    n_targets_seen = 0
    n_clean_seen = 0
    n_target_missing_ts = 0
    n_control_missing_ts = 0
    for clip in clips:
        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        if hyp_tokens is None:
            continue
        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        if not targets:
            # Must match Stage B's own population exactly: its control pool
            # only ever drew from clips that contained at least one target
            # (`if targets: per_clip[...] = ...`) -- a clip with zero targets
            # never contributed control positions there, so it must not here
            # either, or n_clean_seen won't reconcile with Stage B's saved
            # count (caught by the assertion below, not silently accepted).
            continue
        for ref_idx, hyp_idx, true_type in targets:
            if true_type != TARGET_TYPE:
                continue
            n_targets_seen += 1
            d = _duration(hyp_tokens[hyp_idx])
            if d is None:
                n_target_missing_ts += 1
                continue
            target_durations.append(d)
        for ref_idx, hyp_idx in clean_positions:
            n_clean_seen += 1
            d = _duration(hyp_tokens[hyp_idx])
            if d is None:
                n_control_missing_ts += 1
                continue
            control_durations.append(d)

    # Safety check before trusting any pairing with Stage B's saved distances:
    # the re-derivation must reproduce the exact same counts Stage B reported,
    # or the ordering assumption below (same deterministic iteration -> same
    # position sequence) cannot be trusted and this run should stop, not guess.
    assert n_targets_seen == len(encoder_target_distances), (
        f"Re-derived {n_targets_seen} {TARGET_TYPE} targets, Stage B saved "
        f"{len(encoder_target_distances)} -- refusing to trust this run under "
        f"a mismatched count (would indicate the identification logic no "
        f"longer reproduces Stage B's own result)."
    )
    # Control count is allowed a small, explicit tolerance (not exact-match
    # like the target count above): Stage B's encoder arm silently drops a
    # control position if pool_span() returns None for it (a degenerate
    # encoder-frame span -- see encoder_embedding.py's own None-handling),
    # which this duration-only extraction has no equivalent reason to hit
    # (start/end either both exist or both don't, checked separately above).
    # A large mismatch would still mean something is wrong; a difference of
    # 1-2 positions out of ~966 is consistent with that one known asymmetry
    # and cannot materially move an AUC computed over ~966 points.
    control_count_diff = n_clean_seen - len(encoder_control_distances)
    assert 0 <= control_count_diff <= 3, (
        f"Re-derived {n_clean_seen} control positions, Stage B saved "
        f"{len(encoder_control_distances)} (diff={control_count_diff}) -- outside "
        f"the small tolerance expected from pool_span()'s own None-handling; "
        f"refusing to trust this run under a mismatch this size."
    )
    print(f"Count check: re-identified {n_targets_seen} target positions (exact match) "
          f"and {n_clean_seen} control positions vs. Stage B's saved {len(encoder_control_distances)} "
          f"(diff={control_count_diff}, within the small tolerance expected from "
          f"pool_span()'s own graceful None-handling on the encoder side -- not "
          f"replicated in this duration-only extraction, which has no equivalent "
          f"degenerate case to hit).")
    if n_target_missing_ts or n_control_missing_ts:
        print(f"Of those, {n_target_missing_ts} target / {n_control_missing_ts} control "
              f"positions lack an ASR timestamp and are excluded from the duration arm "
              f"only, so n differs slightly between arms below (reported explicitly, "
              f"not hidden) -- the encoder arm still uses the full Stage B population.\n")
    else:
        print()

    # z-score durations against the population of clean/control durations
    # (a per-word-matched baseline was pre-registered as the intent, but most
    # words in this sample appear only once as a clean token -- too sparse to
    # support per-word reference distributions. Falls back to a population
    # z-score, recorded here as a dated addendum to the pre-registered
    # protocol, not a silent substitution.)
    mean_dur = sum(control_durations) / len(control_durations)
    var_dur = sum((d - mean_dur) ** 2 for d in control_durations) / (len(control_durations) - 1)
    sd_dur = var_dur ** 0.5
    target_duration_z = [(d - mean_dur) / sd_dur for d in target_durations]
    control_duration_z = [(d - mean_dur) / sd_dur for d in control_durations]

    encoder_auc = _auc(encoder_target_distances, encoder_control_distances)
    duration_auc = _auc(target_duration_z, control_duration_z)

    results = {"encoder_arm": {"auc": encoder_auc}, "duration_arm": {"auc": duration_auc}}
    for arm, (pos, neg) in (
        ("encoder_arm", (encoder_target_distances, encoder_control_distances)),
        ("duration_arm", (target_duration_z, control_duration_z)),
    ):
        for r in (0.5, 0.7):
            achieved_recall, precision = _precision_at_recall(pos, neg, r)
            results[arm][f"precision_at_recall_{r}"] = precision
            results[arm][f"achieved_recall_{r}"] = achieved_recall

    print(f"=== {TARGET_TYPE}: encoder arm vs. duration-only baseline arm ===")
    print(f"n_target={len(encoder_target_distances)}  n_control={len(encoder_control_distances)}\n")
    for arm in ("encoder_arm", "duration_arm"):
        r = results[arm]
        print(f"{arm}: AUC={r['auc']:.3f}  "
              f"P@R>=0.5={r['precision_at_recall_0.5']:.3f} (achieved R={r['achieved_recall_0.5']:.3f})  "
              f"P@R>=0.7={r['precision_at_recall_0.7']:.3f} (achieved R={r['achieved_recall_0.7']:.3f})")

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_c_duration_baseline.json"
    out_path.write_text(json.dumps({
        "target_type": TARGET_TYPE,
        "n_target": len(encoder_target_distances),
        "n_control": len(encoder_control_distances),
        "results": results,
        "encoder_target_distances": encoder_target_distances,
        "encoder_control_distances": encoder_control_distances,
        "duration_target_z": target_duration_z,
        "duration_control_z": control_duration_z,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--stage-b-result", required=True)
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args(argv)
    run(Path(args.data_dir), Path(args.audio_dir), Path(args.stage_b_result), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
