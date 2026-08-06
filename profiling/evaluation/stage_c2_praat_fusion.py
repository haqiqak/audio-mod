"""stage_c2_praat_fusion.py — ASR_RESEARCH_TRACK.md Stage C2.

Implements the protocol pre-registered in `ASR_RESEARCH_TRACK.md`'s
"Stage C2 - Fusion with acoustic voice-quality evidence" section EXACTLY
- read that section before changing any logic here. Tests whether
Praat-derived voice-quality features (pitch stability, jitter, shimmer,
HNR) carry information Stage C's encoder-distance signal doesn't, over
the identical n=19/966 sound_repetition population Stage B/C used.

No new audio, no encoder pass - reuses Stage C's saved encoder distances
and re-derives the same target/control positions (with the same
count-check discipline every prior stage in this track has used) to
attach Praat features, computed fresh via profiling/acoustic.py's
existing _praat_features (CPU-only, no model download).

Usage
-----
    python -m profiling.evaluation.stage_c2_praat_fusion \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --stage-c-result eval_results/20260805T215037_stage_c_duration_baseline.json
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

from profiling.acoustic import _praat_features, load_wav_samples
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_b_representation_probe import _identify_positions
from profiling.evaluation.stage_c_duration_baseline import _auc, _precision_at_recall
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPE = "sound_repetition"
PRAAT_FEATURES = ("pitch_hz", "pitch_std_hz", "jitter", "shimmer", "hnr")
# Direction each feature is scored so "higher score = more anomalous",
# matching the encoder-distance/duration-z convention every prior stage used.
HIGHER_IS_ANOMALOUS = {"pitch_hz": False, "pitch_std_hz": True, "jitter": True, "shimmer": True, "hnr": False}
SCREEN_AUC_BAR = 0.55


def _anomaly_score(feat: dict, name: str) -> float | None:
    v = feat.get(name)
    if v is None:
        return None
    return v if HIGHER_IS_ANOMALOUS[name] else -v


def run(data_dir: Path, audio_dir: Path, stage_c_result_path: Path, n_clips: int = 120) -> dict:
    stage_c = json.loads(stage_c_result_path.read_text(encoding="utf-8"))
    encoder_target_distances = stage_c["encoder_target_distances"]
    encoder_control_distances = stage_c["encoder_control_distances"]
    print(f"Loaded Stage C result: {len(encoder_target_distances)} target, "
          f"{len(encoder_control_distances)} control encoder distances for {TARGET_TYPE}.")

    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips = _speaker_stratified_order(clips)[:n_clips]

    # Praat feature dicts, parallel to the encoder-distance lists above
    # (same iteration order/population as stage_c_duration_baseline.py's
    # own re-derivation -- verified below via the same count-check
    # discipline every prior stage in this track has used).
    target_feats: list[dict] = []
    control_feats: list[dict] = []
    n_targets_seen = 0
    t0 = time.time()
    for clip in clips:
        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        if hyp_tokens is None:
            continue
        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        if not targets:
            continue  # must match Stage B/C's own population exactly
        samples, sr = load_wav_samples(clip.audio_bytes)
        if samples is None:
            continue
        for ref_idx, hyp_idx, true_type in targets:
            if true_type != TARGET_TYPE:
                continue
            n_targets_seen += 1
            tok = hyp_tokens[hyp_idx]
            start, end = tok.get("start"), tok.get("end")
            feat = _praat_features(samples, sr, start, end) if start is not None and end is not None else {
                k: None for k in PRAAT_FEATURES
            }
            target_feats.append(feat)
        for ref_idx, hyp_idx in clean_positions:
            tok = hyp_tokens[hyp_idx]
            start, end = tok.get("start"), tok.get("end")
            feat = _praat_features(samples, sr, start, end) if start is not None and end is not None else {
                k: None for k in PRAAT_FEATURES
            }
            control_feats.append(feat)
    print(f"Praat extraction: {time.time()-t0:.0f}s for {len(clips)} clips scanned.\n")

    assert n_targets_seen == len(encoder_target_distances), (
        f"Re-derived {n_targets_seen} {TARGET_TYPE} targets, Stage C saved "
        f"{len(encoder_target_distances)} -- refusing to pair Praat features "
        f"with encoder distances under a mismatched count."
    )
    control_count_diff = len(control_feats) - len(encoder_control_distances)
    assert 0 <= control_count_diff <= 3, (
        f"Re-derived {len(control_feats)} control positions, Stage C saved "
        f"{len(encoder_control_distances)} (diff={control_count_diff}) -- "
        f"outside the small tolerance seen in prior stages; refusing to trust "
        f"this run under a mismatch this size."
    )
    print(f"Count check: {n_targets_seen} target (exact match), {len(control_feats)} "
          f"control (diff={control_count_diff} vs Stage C, within tolerance).\n")
    # Encoder distances and control_feats can differ by the small tolerance
    # above (same pool_span-None asymmetry documented in Stage C) -- trim the
    # longer list's tail so both are the same length before pairing by index.
    n_pair = min(len(control_feats), len(encoder_control_distances))
    control_feats = control_feats[:n_pair]
    encoder_control_distances_paired = encoder_control_distances[:n_pair]

    # Step 1: screen each Praat feature individually.
    print("=== Screening individual Praat features (AUC vs. chance=0.5) ===")
    screened = {}
    for name in PRAAT_FEATURES:
        tgt_scores = [_anomaly_score(f, name) for f in target_feats]
        ctl_scores = [_anomaly_score(f, name) for f in control_feats]
        n_tgt_missing = sum(1 for s in tgt_scores if s is None)
        n_ctl_missing = sum(1 for s in ctl_scores if s is None)
        tgt_valid = [s for s in tgt_scores if s is not None]
        ctl_valid = [s for s in ctl_scores if s is not None]
        if len(tgt_valid) < 2 or len(ctl_valid) < 2:
            print(f"  {name:14s}  SKIPPED (too few valid values: "
                  f"n_target_valid={len(tgt_valid)}, n_control_valid={len(ctl_valid)})")
            continue
        auc = _auc(tgt_valid, ctl_valid)
        passed = auc >= SCREEN_AUC_BAR
        print(f"  {name:14s}  AUC={auc:.3f}  target_missing={n_tgt_missing}/{len(tgt_scores)} "
              f"control_missing={n_ctl_missing}/{len(ctl_scores)}  {'PASS' if passed else 'below bar'}")
        screened[name] = {
            "auc": auc, "n_target_missing": n_tgt_missing, "n_control_missing": n_ctl_missing,
            "passed_screen": passed,
        }

    passing = [n for n, r in screened.items() if r["passed_screen"]]
    print(f"\nFeatures clearing AUC >= {SCREEN_AUC_BAR}: {passing or 'NONE'}\n")

    result: dict = {"screened_features": screened, "passing_features": passing}

    if not passing:
        print("No Praat feature cleared the screening bar -- fusion not attempted "
              "(per the pre-registered protocol: this is its own distinct finding, "
              "not the same as 'fusion didn't help').")
        result["fusion_attempted"] = False
    else:
        # Step 2: combine each passing feature (max of z-scores) with encoder
        # distance; report the best-performing combination, but show all.
        def _zscore(values, ref_values):
            m = sum(ref_values) / len(ref_values)
            var = sum((x - m) ** 2 for x in ref_values) / (len(ref_values) - 1)
            sd = var ** 0.5
            return [(v - m) / sd if sd > 0 else 0.0 for v in values]

        enc_ctl_z = _zscore(encoder_control_distances_paired, encoder_control_distances_paired)
        enc_tgt_z = _zscore(encoder_target_distances, encoder_control_distances_paired)

        result["fusion_attempted"] = True
        result["combinations"] = {}
        for name in passing:
            tgt_scores = [_anomaly_score(f, name) for f in target_feats]
            ctl_scores = [_anomaly_score(f, name) for f in control_feats]
            # Missing Praat value -> treat as "no extra evidence" (0 after
            # z-scoring against the control mean), never as anomalous by
            # default -- matches this codebase's own graceful-degradation
            # convention for optional acoustic signals.
            ctl_valid_for_z = [s for s in ctl_scores if s is not None]
            feat_ctl_z_full = _zscore(
                [s if s is not None else (sum(ctl_valid_for_z) / len(ctl_valid_for_z)) for s in ctl_scores],
                ctl_valid_for_z,
            )
            feat_tgt_z_full = _zscore(
                [s if s is not None else (sum(ctl_valid_for_z) / len(ctl_valid_for_z)) for s in tgt_scores],
                ctl_valid_for_z,
            )
            combo_tgt = [max(e, f) for e, f in zip(enc_tgt_z, feat_tgt_z_full)]
            combo_ctl = [max(e, f) for e, f in zip(enc_ctl_z, feat_ctl_z_full)]
            auc = _auc(combo_tgt, combo_ctl)
            r5 = _precision_at_recall(combo_tgt, combo_ctl, 0.5)
            r7 = _precision_at_recall(combo_tgt, combo_ctl, 0.7)
            result["combinations"][name] = {
                "auc": auc, "precision_at_recall_0.5": r5[1], "achieved_recall_0.5": r5[0],
                "precision_at_recall_0.7": r7[1], "achieved_recall_0.7": r7[0],
            }
            print(f"encoder + {name} (max of z-scores): AUC={auc:.3f}  "
                  f"P@R>=0.5={r5[1]:.3f} (R={r5[0]:.3f})  P@R>=0.7={r7[1]:.3f} (R={r7[0]:.3f})")

    result["encoder_only_auc"] = _auc(encoder_target_distances, encoder_control_distances_paired)
    result["n_target"] = len(encoder_target_distances)
    result["n_control"] = n_pair
    print(f"\n(For reference, Stage C's own encoder-only AUC on this exact "
          f"paired population: {result['encoder_only_auc']:.3f})")

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_c2_praat_fusion.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--stage-c-result", required=True)
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args(argv)
    run(Path(args.data_dir), Path(args.audio_dir), Path(args.stage_c_result), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
