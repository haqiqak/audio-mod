"""stage_b_representation_probe.py — ASR_RESEARCH_TRACK.md Stage B.

Implements the protocol pre-registered in `ASR_RESEARCH_TRACK.md`'s
"Stage B - pre-registered protocol" section EXACTLY - read that section
before changing any logic here. Answers RQ2: does CrisperWhisper's
encoder retain discriminative information at real-ASR positions where
transcript-level evidence for `sound_repetition`/`word_repetition` was
normalized away (Stage A's category 1), or doesn't it. A positive,
negative, or inconclusive result are all acceptable, correctly-reported
outcomes.

Reuses Stage 1's exact encoder-extraction primitives
(`profiling.encoder_embedding`) unmodified - only the span-selection and
centroid-construction layer is new, because Stage B operates on real ASR
hyp-token boundaries (Track B), not ground-truth token boundaries
(Track A, what Stage 1 used).

Usage
-----
    python -m profiling.evaluation.stage_b_representation_probe \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --n 120

Deliberately reuses the already-cached Track B ASR output
(`eval_datasets/_track_b_cache`) - no new ASR inference. The real cost
here is the encoder pass alone, scoped to the ~38 clips that actually
contain a Stage-A category-1 target position before this script runs
(see `ASR_RESEARCH_TRACK.md`'s "Cost, scoped before running").
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from profiling.config import load_config
from profiling.detect import detect_disfluencies
from profiling.encoder_embedding import cosine_distance, extract_last_layer_states, load_encoder, pool_span
from profiling.evaluation.alignment import align
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.track_b import _load_cached, _speaker_stratified_order

_DEFAULT_CACHE_DIR = _ROOT / "eval_datasets" / "_track_b_cache"
TARGET_TYPES = ("sound_repetition", "word_repetition")

# Gate forced off via an explicit config override (never touches config.yaml
# on disk) so identification matches Stage A's own methodology exactly: real
# audio (so acoustic-native detectors, e.g. block/prolongation, are active --
# passing audio_bytes=None would silently disable them and could misclassify
# a "mis-routed to block" case as "no candidate at all"), classifier gate off
# (so a classifier-suppressed candidate is never confused with "no candidate
# was ever generated" -- the two are different things and this script must
# not conflate them).
_GATE_OFF_CONFIG = dict(load_config().get("profiling", {}).get("detection", {}))
_GATE_OFF_CONFIG["require_repetition_classifier_confirmation"] = False


def _identify_positions(clip, hyp_tokens):
    """Re-derives, for one clip, every disfluent ref position's alignment
    kind/hyp_idx AND (independently) the raw candidate set at every hyp
    index, under real audio with the classifier gate explicitly forced off
    (see _GATE_OFF_CONFIG) -- matching Stage A's own "gate off" methodology
    exactly, not an approximation of it."""
    ref_words = [t["word"] for t in clip.tokens]
    hyp_words = [t["word"] for t in hyp_tokens]
    disfluent_idx = set(clip.ground_truth.keys())
    ops = align(ref_words, hyp_words, disfluent_indices=disfluent_idx)
    hyp_kind_by_ref = {op.ref_index: (op.kind, op.hyp_index) for op in ops if op.ref_index is not None}

    events = detect_disfluencies(hyp_tokens, config=_GATE_OFF_CONFIG, audio_bytes=clip.audio_bytes)
    predicted_by_hyp_idx = defaultdict(set)
    for e in events:
        predicted_by_hyp_idx[e["index"]].add(e["type"])

    targets = []  # (ref_idx, hyp_idx, true_type)
    clean_positions = []  # (ref_idx, hyp_idx) -- not ground-truth-disfluent, align=correct
    for ref_idx in range(len(ref_words)):
        kind, hyp_idx = hyp_kind_by_ref.get(ref_idx, ("deletion", None))
        if kind != "correct" or hyp_idx is None:
            continue
        true_type = clip.ground_truth.get(ref_idx)
        if true_type is None:
            clean_positions.append((ref_idx, hyp_idx))
            continue
        if true_type in TARGET_TYPES:
            predicted_here = predicted_by_hyp_idx.get(hyp_idx, set())
            if not predicted_here:  # category 1 only: nothing predicted at all
                targets.append((ref_idx, hyp_idx, true_type))
    return targets, clean_positions


def _cohens_d(a: list[float], b: list[float]) -> float | None:
    """Pooled-SD Cohen's d. None when either group has fewer than 2
    samples -- an honest 'not enough data', matching metrics.py's
    encoder_distance_stats() convention exactly (same formula, duplicated
    here rather than imported, since that function's helper is private
    and shaped around TP/FP labels, not this stage's target/control
    groups)."""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = (((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)) ** 0.5
    if pooled == 0:
        return None
    return (ma - mb) / pooled


def run(data_dir: Path, audio_dir: Path, n_clips: int, cache_dir: Path = _DEFAULT_CACHE_DIR,
        speaker_stratified: bool = True) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    if speaker_stratified:
        clips = _speaker_stratified_order(clips)
    clips = clips[:n_clips]

    # Pass 1: identify target/clean positions per clip using only cached
    # hyp_tokens + text-based candidate matching -- no encoder cost yet.
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

    print(f"{len(per_clip)} clips contain at least one Stage-A category-1 target position.")
    n_targets = sum(len(v["targets"]) for v in per_clip.values())
    print(f"{n_targets} total target positions across those clips.\n")

    # Pass 2: real encoder pass, one per clip that has a target.
    processor, encoder = load_encoder()
    target_distances = defaultdict(list)   # type -> [distance, ...]
    control_distances = defaultdict(list)  # type -> [distance, ...] (leave-one-out)
    t0 = time.time()
    for i, (name, rec) in enumerate(per_clip.items()):
        clip, hyp_tokens = rec["clip"], rec["hyp_tokens"]
        c0 = time.time()
        samples, sr = load_wav_samples(clip.audio_bytes)
        states = extract_last_layer_states(processor, encoder, samples, sr)
        print(f"[{i+1}/{len(per_clip)}] {name} ... ({time.time()-c0:.0f}s, {time.time()-t0:.0f}s elapsed)")

        clean = rec["clean_positions"]
        clean_vecs = {}
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

        # Target positions: distance to the FULL clean centroid (targets are
        # never part of the clean set, so no leave-one-out needed for them).
        for ref_idx, hyp_idx, true_type in rec["targets"]:
            tok = hyp_tokens[hyp_idx]
            v = pool_span(states, tok.get("start"), tok.get("end"))
            d = cosine_distance(v, full_centroid)
            if d is not None:
                target_distances[true_type].append(d)

        # Control: every clean position, leave-one-out centroid.
        for hyp_idx, v in clean_vecs.items():
            loo_centroid = (sum_vec - v) / (n_clean - 1)
            d = cosine_distance(v, loo_centroid)
            if d is not None:
                # Controls aren't typed by disfluency type; pooled under "control".
                control_distances["control"].append(d)

    print(f"\nTotal encoder time: {time.time()-t0:.0f}s for {len(per_clip)} clips.\n")

    results = {}
    for t in TARGET_TYPES:
        tgt = target_distances[t]
        ctl = control_distances["control"]
        d = _cohens_d(tgt, ctl)
        results[t] = {
            "n_target": len(tgt),
            "target_mean": sum(tgt) / len(tgt) if tgt else None,
            "n_control": len(ctl),
            "control_mean": sum(ctl) / len(ctl) if ctl else None,
            "cohens_d": d,
            "target_distances": tgt,
        }
        print(f"=== {t} ===")
        print(f"  n_target={len(tgt)}  target_mean_distance={results[t]['target_mean']}")
        print(f"  n_control={len(ctl)}  control_mean_distance={results[t]['control_mean']}")
        print(f"  cohens_d={d}")

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_b_representation_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_clips_with_targets": len(per_clip),
        "results": results,
        "control_distances": control_distances["control"],
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir else _DEFAULT_CACHE_DIR
    run(Path(args.data_dir), Path(args.audio_dir), args.n, cache_dir=cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
