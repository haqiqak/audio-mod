"""collect_raw_encoder_data.py — VALIDATION.md §12 data collection: persist
raw per-event encoder embeddings (not just the aggregate distance Stage 1
saved) so the corroboration-mechanism comparison (§12) can compute any
signal/mechanism combination post-hoc, without a second encoder pass.

Same cost profile as run_encoder_signal_stage1.py (the encoder pass
dominates, ~40-50s/clip regardless of clip length — see that script's
module docstring) — this is a separate run because Stage 1's original
runner only ever computed and saved the aggregate distance-to-centroid
number, never the raw embedding vectors §12's comparison needs.

Usage
─────
    python -m profiling.evaluation.collect_raw_encoder_data \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --n-clips 90 \\
        --out eval_results/stage1_raw_embeddings.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.acoustic import load_wav_samples
from profiling.detect import detect_disfluencies
from profiling.evaluation.encoder_features import (
    collect_raw_records,
    extract_last_layer_states,
    load_encoder,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio

# Same scope as run_encoder_signal_stage1.py / ROADMAP.md item 17.
TARGET_TYPES = ("word_repetition", "sound_repetition", "filler")


def run(data_dir: Path, audio_dir: Path, n_clips: int | None, out_path: Path) -> Path:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    if n_clips is not None:
        clips = clips[:n_clips]
    print(f"{len(clips)} clips loaded.\n")

    print("Loading CrisperWhisper's processor + encoder (not the full "
          "model -- no decoding) ...")
    t0 = time.perf_counter()
    processor, encoder = load_encoder()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    all_records: list[dict] = []
    t_encode_total = 0.0
    for i, clip in enumerate(clips, 1):
        samples, sr = load_wav_samples(clip.audio_bytes) if clip.audio_bytes else (None, None)
        if samples is None:
            print(f"[{i}/{len(clips)}] {clip.name}: no audio, skipped")
            continue
        events = detect_disfluencies(clip.tokens, audio_bytes=clip.audio_bytes)

        t1 = time.perf_counter()
        states = extract_last_layer_states(processor, encoder, samples, sr)
        dt = time.perf_counter() - t1
        t_encode_total += dt

        records = collect_raw_records(
            states, clip.name, clip.tokens, clip.ground_truth, events, TARGET_TYPES,
        )
        all_records.extend(records)
        print(f"[{i}/{len(clips)}] {clip.name}: encoder pass {dt:.1f}s, "
              f"{len(records)} scorable events")

    print(f"\nTotal encoder time: {t_encode_total:.1f}s "
          f"({t_encode_total / max(1, len(clips)):.1f}s/clip average)")
    print(f"Total scorable events collected: {len(all_records)}\n")

    _save_npz(all_records, out_path)
    print(f"Saved: {out_path}")
    return out_path


def _save_npz(records: list[dict], out_path: Path) -> None:
    """Pack the variable-shape record list into fixed-width parallel arrays
    for .npz storage. partner_embedding rows that are None are stored as
    NaN-filled vectors with a parallel has_partner boolean array, rather
    than omitted, so array lengths stay aligned across all fields."""
    if not records:
        raise ValueError("No records collected -- nothing to save.")

    hidden_dim = records[0]["embedding"].shape[0]
    n = len(records)
    embeddings = np.zeros((n, hidden_dim), dtype=np.float32)
    centroids = np.zeros((n, hidden_dim), dtype=np.float32)
    partner_embeddings = np.full((n, hidden_dim), np.nan, dtype=np.float32)
    has_partner = np.zeros(n, dtype=bool)
    labels = np.zeros(n, dtype=np.int64)
    clip_ids = np.empty(n, dtype=object)
    indices = np.zeros(n, dtype=np.int64)
    types = np.empty(n, dtype=object)

    for i, r in enumerate(records):
        embeddings[i] = r["embedding"]
        centroids[i] = r["centroid"]
        labels[i] = r["label"]
        clip_ids[i] = r["clip_id"]
        indices[i] = r["index"]
        types[i] = r["type"]
        if r["partner_embedding"] is not None:
            partner_embeddings[i] = r["partner_embedding"]
            has_partner[i] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        centroids=centroids,
        partner_embeddings=partner_embeddings,
        has_partner=has_partner,
        labels=labels,
        clip_ids=clip_ids,
        indices=indices,
        types=types,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--n-clips", type=int, default=30)
    parser.add_argument(
        "--out", default=str(_ROOT / "eval_results" / "stage1_raw_embeddings.npz"),
    )
    args = parser.parse_args(argv)

    data_dir, audio_dir = Path(args.data_dir), Path(args.audio_dir)
    if not data_dir.exists() or not audio_dir.exists():
        print("--data-dir and --audio-dir must both exist.")
        return 2

    n_clips = None if args.n_clips in (0, None) else args.n_clips
    run(data_dir, audio_dir, n_clips, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
