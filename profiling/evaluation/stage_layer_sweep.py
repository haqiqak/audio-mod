"""stage_layer_sweep.py — ASR_RESEARCH_TRACK.md "Encoder layer-depth sweep".

Implements the protocol pre-registered in ASR_RESEARCH_TRACK.md's
"Encoder layer-depth sweep - pre-registered protocol" section EXACTLY -
read that section before changing any logic here. Tests whether a layer
other than CrisperWhisper's default last encoder layer carries a
stronger sound_repetition signal at Stage A's category-1 positions,
matching arXiv:2311.05203's finding that deeper Whisper-encoder layers
carry more disfluency-relevant signal for a comparable task.

One forward pass per clip (output_hidden_states=True returns every
layer's activations from a single call, not one call per layer) over
the identical 31 clips / 19 target / 966 control population Stage B/C
already established - reuses profiling.encoder_embedding's pool_span/
cosine_distance primitives unmodified, only the extraction call itself
is new (kept local to this script rather than added to the core
encoder_embedding.py module, since no live-app code needs it).

Usage
-----
    python -m profiling.evaluation.stage_layer_sweep \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio
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
from profiling.encoder_embedding import EncoderStates, cosine_distance, load_encoder, pool_span
from profiling.evaluation.stage_b_representation_probe import _identify_positions
from profiling.evaluation.stage_c_duration_baseline import _auc
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPE = "sound_repetition"


def _extract_all_layers(processor, encoder, samples: np.ndarray, sr: int) -> list[EncoderStates]:
    """One forward pass, every layer's hidden states returned -- not one
    pass per layer. Layer 0 is the embedding-layer output (before any
    transformer block); layer -1 (last) is what Stage B/C already used
    (`extract_last_layer_states`'s exact output, verified equal before
    this script was trusted -- see the pre-run API check in
    PAPER_DECISION_LOG.md)."""
    import torch

    inputs = processor(samples, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = encoder(inputs.input_features, output_hidden_states=True)
    return [EncoderStates(hidden_states=h[0].to(torch.float32).numpy()) for h in out.hidden_states]


def run(data_dir: Path, audio_dir: Path, n_clips: int = 120, model_id: str | None = None) -> dict:
    from profiling.evaluation.loaders import load_libristutter_dir_with_audio

    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips = _speaker_stratified_order(clips)[:n_clips]

    per_clip = {}
    for clip in clips:
        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        if hyp_tokens is None:
            continue
        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        targets = [t for t in targets if t[2] == TARGET_TYPE]
        if targets:
            per_clip[clip.name] = {"clip": clip, "hyp_tokens": hyp_tokens,
                                    "targets": targets, "clean_positions": clean_positions}

    n_targets = sum(len(v["targets"]) for v in per_clip.values())
    print(f"{len(per_clip)} clips, {n_targets} target positions ({TARGET_TYPE}).\n")

    print(f"Encoder: {model_id or 'nyrahealth/CrisperWhisper (default)'}\n")
    processor, encoder = load_encoder(model_id)
    n_layers = None
    target_distances_by_layer: dict[int, list[float]] = defaultdict(list)
    control_distances_by_layer: dict[int, list[float]] = defaultdict(list)
    t0 = time.time()

    for i, (name, rec) in enumerate(per_clip.items()):
        clip, hyp_tokens = rec["clip"], rec["hyp_tokens"]
        c0 = time.time()
        samples, sr = load_wav_samples(clip.audio_bytes)
        layer_states = _extract_all_layers(processor, encoder, samples, sr)
        if n_layers is None:
            n_layers = len(layer_states)
        print(f"[{i+1}/{len(per_clip)}] {name} ... ({time.time()-c0:.0f}s, {time.time()-t0:.0f}s elapsed)")

        clean = rec["clean_positions"]
        for layer_idx, states in enumerate(layer_states):
            clean_vecs = {}
            for ref_idx, hyp_idx in clean:
                tok = hyp_tokens[hyp_idx]
                v = pool_span(states, tok.get("start"), tok.get("end"))
                if v is not None:
                    clean_vecs[hyp_idx] = v
            if len(clean_vecs) < 2:
                continue
            all_vecs = list(clean_vecs.values())
            sum_vec = np.sum(all_vecs, axis=0)
            n_clean = len(all_vecs)
            full_centroid = sum_vec / n_clean

            for ref_idx, hyp_idx, true_type in rec["targets"]:
                tok = hyp_tokens[hyp_idx]
                v = pool_span(states, tok.get("start"), tok.get("end"))
                d = cosine_distance(v, full_centroid)
                if d is not None:
                    target_distances_by_layer[layer_idx].append(d)

            for hyp_idx, v in clean_vecs.items():
                loo_centroid = (sum_vec - v) / (n_clean - 1)
                d = cosine_distance(v, loo_centroid)
                if d is not None:
                    control_distances_by_layer[layer_idx].append(d)

    print(f"\nTotal encoder time: {time.time()-t0:.0f}s for {len(per_clip)} clips, {n_layers} layers each.\n")

    results = {}
    for layer_idx in range(n_layers):
        tgt = target_distances_by_layer[layer_idx]
        ctl = control_distances_by_layer[layer_idx]
        if len(tgt) < 2 or len(ctl) < 2:
            continue
        auc = _auc(tgt, ctl)
        results[layer_idx] = {"auc": auc, "n_target": len(tgt), "n_control": len(ctl)}

    print("=== AUC by layer (0 = embedding output, "
          f"{n_layers-1} = last transformer layer, matches Stage B/C) ===")
    for layer_idx in sorted(results):
        r = results[layer_idx]
        marker = "  <- Stage B/C's layer" if layer_idx == n_layers - 1 else ""
        print(f"  layer {layer_idx:2d}: AUC={r['auc']:.3f}  (n_target={r['n_target']}, n_control={r['n_control']}){marker}")

    best_layer = max(results, key=lambda k: results[k]["auc"])
    last_layer_auc = results[n_layers - 1]["auc"]
    print(f"\nBest layer: {best_layer} (AUC={results[best_layer]['auc']:.3f}) "
          f"vs. last layer {n_layers-1} (AUC={last_layer_auc:.3f})")

    tag = "_arm2_" + model_id.replace("/", "_") if model_id else ""
    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_layer_sweep{tag}.json"
    out_path.write_text(json.dumps({
        "model_id": model_id or "nyrahealth/CrisperWhisper",
        "target_type": TARGET_TYPE, "n_layers": n_layers,
        "results_by_layer": results, "best_layer": best_layer,
        "last_layer_index": n_layers - 1,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--model-id", default=None,
                         help="Encoder checkpoint to sweep (default: CrisperWhisper). "
                              "Phase 2 Arm 2 uses 'openai/whisper-large-v3'.")
    args = parser.parse_args(argv)
    run(Path(args.data_dir), Path(args.audio_dir), args.n, model_id=args.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
