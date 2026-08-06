"""stage_arm3_wavlm.py — ASR_RESEARCH_TRACK.md Phase 2, Arm 3.

Implements the protocol pre-registered in ASR_RESEARCH_TRACK.md's "Phase 2
of this research track" -> "Pre-registered protocol" -> "Arm 3" section
EXACTLY - read that section before changing any logic here. Answers RQ-B(ii):
does WavLM-Large's encoder representation - a genuinely different
architecture, never fine-tuned for ASR, explicitly designed for
paralinguistic sensitivity (arXiv:2110.13900) - carry a stronger
sound_repetition/word_repetition signal than anything in the Whisper family
(CrisperWhisper's own Stage B/C result, d=0.894/AUC=0.723 for
sound_repetition; Arm 2's stock whisper-large-v3 result, AUC=0.680)?

Sequenced after Arm 1 and Arm 2 (both came back Failure per their own
pre-registered criteria - see PAPER_DECISION_LOG.md), which is exactly the
outcome-to-conclusion mapping's trigger for running this arm.

Confounders named in the pre-registration, not resolved here:
- Model size: WavLM-Large (~315M params) is smaller than CrisperWhisper's
  encoder. Any difference found cannot be cleanly attributed to
  "pretraining objective" alone.
- Frame-rate/pooling: VERIFIED DIRECTLY before this script was written
  (not assumed) - WavLM-Large's conv feature encoder has total stride 320
  at 16kHz => exactly 20ms/frame, identical to profiling/encoder_embedding.
  py's FRAME_SECONDS convention. LibriStutter's own audio files are
  natively 16kHz (verified directly), matching WavLM's expected input rate
  with no resampling needed. This means pool_span/cosine_distance are
  reusable UNMODIFIED - the "non-trivial engineering" characterization in
  the original pre-registration was more cautious than the actual
  situation turned out to be once checked; recorded as a correction, not a
  silent scope change.
- WavLM has no fixed 30s-window padding (unlike Whisper) - its output
  frame count is proportional to actual clip length. pool_span already
  only depends on states.hidden_states.shape[0], so this needs no code
  change either.

Usage
-----
    python -m profiling.evaluation.stage_arm3_wavlm \\
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
from profiling.encoder_embedding import EncoderStates, cosine_distance, pool_span
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_b_representation_probe import _cohens_d, _identify_positions
from profiling.evaluation.stage_c_duration_baseline import _auc
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPES = ("sound_repetition", "word_repetition")
WAVLM_MODEL_ID = "microsoft/wavlm-large"

# CrisperWhisper's own reference numbers, printed for comparison only -
# never recomputed here. sound_repetition: Stage C's d/AUC (full 31-clip
# population, leave-one-out). word_repetition: Stage B never found a
# usable signal for it (fewer TPs) - no comparable number exists to cite.
CRISPERWHISPER_REFERENCE = {
    "sound_repetition": {"cohens_d": 0.894, "auc": 0.723},
}
ARM2_STOCK_WHISPER_REFERENCE = {"sound_repetition": {"auc": 0.680}}


def _load_wavlm_encoder(model_id: str = WAVLM_MODEL_ID):
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = WavLMModel.from_pretrained(model_id)
    model.eval()
    return feature_extractor, model


def _extract_last_layer(feature_extractor, model, samples: np.ndarray, sr: int) -> EncoderStates:
    import torch

    inputs = feature_extractor(samples, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values)
    hidden = out.last_hidden_state[0].to(torch.float32).numpy()
    return EncoderStates(hidden_states=hidden, frame_seconds=0.02)


def _extract_all_layers(feature_extractor, model, samples: np.ndarray, sr: int) -> list[EncoderStates]:
    import torch

    inputs = feature_extractor(samples, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs.input_values, output_hidden_states=True)
    return [EncoderStates(hidden_states=h[0].to(torch.float32).numpy(), frame_seconds=0.02) for h in out.hidden_states]


def run(data_dir: Path, audio_dir: Path, n_clips: int = 120, do_layer_sweep: bool = True) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips = _speaker_stratified_order(clips)[:n_clips]

    # Same target/control position source as Arm 2: CrisperWhisper's own
    # cached hyp_tokens define WHERE to pool from - only the encoder that
    # processes the audio changes. This holds position-selection fixed
    # across every arm, per the pre-registration.
    per_clip = {}
    for clip in clips:
        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        if hyp_tokens is None:
            continue
        targets, clean_positions = _identify_positions(clip, hyp_tokens)
        targets = [t for t in targets if t[2] in TARGET_TYPES]
        if targets:
            per_clip[clip.name] = {"clip": clip, "hyp_tokens": hyp_tokens,
                                    "targets": targets, "clean_positions": clean_positions}

    n_targets = sum(len(v["targets"]) for v in per_clip.values())
    print(f"{len(per_clip)} clips, {n_targets} target positions (sound_repetition + word_repetition).\n")

    print(f"Loading {WAVLM_MODEL_ID} (downloads on first use) ...")
    feature_extractor, model = _load_wavlm_encoder()

    target_distances: dict[str, list[float]] = defaultdict(list)
    control_distances: list[float] = []
    target_distances_by_layer: dict[int, list[float]] = defaultdict(list)
    control_distances_by_layer: dict[int, list[float]] = defaultdict(list)
    n_layers = None
    t0 = time.time()

    for i, (name, rec) in enumerate(per_clip.items()):
        clip, hyp_tokens = rec["clip"], rec["hyp_tokens"]
        c0 = time.time()
        samples, sr = load_wav_samples(clip.audio_bytes)

        if do_layer_sweep:
            layer_states = _extract_all_layers(feature_extractor, model, samples, sr)
            if n_layers is None:
                n_layers = len(layer_states)
            last_states = layer_states[-1]
        else:
            last_states = _extract_last_layer(feature_extractor, model, samples, sr)
            layer_states = None

        print(f"[{i+1}/{len(per_clip)}] {name} ... ({time.time()-c0:.0f}s, {time.time()-t0:.0f}s elapsed)")

        clean = rec["clean_positions"]

        def _clean_vecs(states):
            out = {}
            for ref_idx, hyp_idx in clean:
                tok = hyp_tokens[hyp_idx]
                v = pool_span(states, tok.get("start"), tok.get("end"))
                if v is not None:
                    out[hyp_idx] = v
            return out

        clean_vecs = _clean_vecs(last_states)
        if len(clean_vecs) >= 2:
            all_vecs = list(clean_vecs.values())
            sum_vec = np.sum(all_vecs, axis=0)
            n_clean = len(all_vecs)
            full_centroid = sum_vec / n_clean

            for ref_idx, hyp_idx, true_type in rec["targets"]:
                tok = hyp_tokens[hyp_idx]
                v = pool_span(last_states, tok.get("start"), tok.get("end"))
                d = cosine_distance(v, full_centroid)
                if d is not None:
                    target_distances[true_type].append(d)

            for hyp_idx, v in clean_vecs.items():
                loo_centroid = (sum_vec - v) / (n_clean - 1)
                d = cosine_distance(v, loo_centroid)
                if d is not None:
                    control_distances.append(d)

        if do_layer_sweep and layer_states is not None:
            for layer_idx, states in enumerate(layer_states):
                lcv = _clean_vecs(states)
                if len(lcv) < 2:
                    continue
                lall = list(lcv.values())
                lsum = np.sum(lall, axis=0)
                ln = len(lall)
                lcentroid = lsum / ln
                for ref_idx, hyp_idx, true_type in rec["targets"]:
                    if true_type != "sound_repetition":
                        continue  # layer sweep scoped to sound_repetition, matching Arm 2
                    tok = hyp_tokens[hyp_idx]
                    v = pool_span(states, tok.get("start"), tok.get("end"))
                    d = cosine_distance(v, lcentroid)
                    if d is not None:
                        target_distances_by_layer[layer_idx].append(d)
                for hyp_idx, v in lcv.items():
                    loo = (lsum - v) / (ln - 1)
                    d = cosine_distance(v, loo)
                    if d is not None:
                        control_distances_by_layer[layer_idx].append(d)

    print(f"\nTotal encoder time: {time.time()-t0:.0f}s for {len(per_clip)} clips.\n")

    results = {}
    print("=== Arm 3: WavLM-Large last-layer encoder-distance (Cohen's d / AUC) ===")
    for t in TARGET_TYPES:
        tgt = target_distances[t]
        d = _cohens_d(tgt, control_distances)
        auc = _auc(tgt, control_distances) if len(tgt) >= 2 and len(control_distances) >= 2 else None
        results[t] = {
            "n_target": len(tgt), "n_control": len(control_distances),
            "target_mean": sum(tgt) / len(tgt) if tgt else None,
            "control_mean": sum(control_distances) / len(control_distances) if control_distances else None,
            "cohens_d": d, "auc": auc,
        }
        print(f"\n{t}: n_target={len(tgt)} n_control={len(control_distances)}")
        print(f"  cohens_d={d}  auc={auc}")
        ref = CRISPERWHISPER_REFERENCE.get(t)
        if ref:
            print(f"  (CrisperWhisper Stage B/C reference: d={ref['cohens_d']}, AUC={ref['auc']})")
        ref2 = ARM2_STOCK_WHISPER_REFERENCE.get(t)
        if ref2:
            print(f"  (Arm 2 stock whisper-large-v3 reference: AUC={ref2['auc']})")

    layer_results = {}
    if do_layer_sweep and n_layers:
        print("\n=== Arm 3: layer sweep (sound_repetition only, matching Arm 2's scope) ===")
        for layer_idx in range(n_layers):
            tgt = target_distances_by_layer[layer_idx]
            ctl = control_distances_by_layer[layer_idx]
            if len(tgt) < 2 or len(ctl) < 2:
                continue
            auc = _auc(tgt, ctl)
            layer_results[layer_idx] = {"auc": auc, "n_target": len(tgt), "n_control": len(ctl)}
            marker = "  <- last layer" if layer_idx == n_layers - 1 else ""
            print(f"  layer {layer_idx:2d}: AUC={auc:.3f}  (n_target={len(tgt)}, n_control={len(ctl)}){marker}")

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_arm3_wavlm.json"
    out_path.write_text(json.dumps({
        "model_id": WAVLM_MODEL_ID,
        "n_clips": len(per_clip), "n_targets": n_targets,
        "last_layer_results": results,
        "layer_sweep_results": layer_results,
        "n_layers": n_layers,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return {"last_layer_results": results, "layer_sweep_results": layer_results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--no-layer-sweep", action="store_true",
                         help="Skip the per-layer sweep, last-layer stats only (faster).")
    args = parser.parse_args(argv)
    run(Path(args.data_dir), Path(args.audio_dir), args.n, do_layer_sweep=not args.no_layer_sweep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
