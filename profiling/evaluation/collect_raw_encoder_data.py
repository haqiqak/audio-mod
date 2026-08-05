"""collect_raw_encoder_data.py — VALIDATION.md §12 data collection: persist
raw per-event encoder embeddings (not just the aggregate distance Stage 1
saved) so the corroboration-mechanism comparison (§12) can compute any
signal/mechanism combination post-hoc, without a second encoder pass.

Same cost profile as run_encoder_signal_stage1.py (the encoder pass
dominates, ~40-50s/clip regardless of clip length — see that script's
module docstring) — this is a separate run because Stage 1's original
runner only ever computed and saved the aggregate distance-to-centroid
number, never the raw embedding vectors §12's comparison needs.

**Checkpointing (added 2026-08-04, after a real run was killed mid-way by
an unrelated session interruption and lost all progress — nothing had
been saved, since the original version only wrote output at the very
end)**: the output `.npz` is now saved after *every* clip, not just at
completion — cheap relative to the ~30-90s/clip encoder cost this
project has already measured. If `--out` already exists when this script
starts, it's loaded as a checkpoint: clips already recorded as processed
are skipped, and newly-collected records are appended to (not replacing)
what's already there. Re-running the exact same command after an
interruption resumes rather than restarting from clip 1.

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

    all_records: list[dict] = []
    processed_clips: set[str] = set()
    if out_path.exists():
        all_records, processed_clips = _load_checkpoint(out_path)
        print(f"Resuming from checkpoint: {out_path} "
              f"({len(processed_clips)} clips already processed, "
              f"{len(all_records)} records so far).\n")

    remaining = [c for c in clips if c.name not in processed_clips]
    if not remaining:
        print("Nothing left to do -- every requested clip is already in the checkpoint.")
        return out_path
    print(f"{len(remaining)} clips remaining to process "
          f"({len(clips) - len(remaining)} already done).\n")

    print("Loading CrisperWhisper's processor + encoder (not the full "
          "model -- no decoding) ...")
    t0 = time.perf_counter()
    processor, encoder = load_encoder()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    t_encode_total = 0.0
    for i, clip in enumerate(remaining, 1):
        samples, sr = load_wav_samples(clip.audio_bytes) if clip.audio_bytes else (None, None)
        if samples is None:
            print(f"[{i}/{len(remaining)}] {clip.name}: no audio, skipped")
            processed_clips.add(clip.name)
            _save_npz(all_records, processed_clips, out_path)
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
        processed_clips.add(clip.name)

        # Checkpoint after every clip: cheap (a small compressed array
        # write) relative to the 30-90s/clip encoder cost, and means an
        # interruption loses at most one clip's work, not the whole run.
        _save_npz(all_records, processed_clips, out_path)

        print(f"[{i}/{len(remaining)}] {clip.name}: encoder pass {dt:.1f}s, "
              f"{len(records)} scorable events (checkpointed)")

    print(f"\nTotal encoder time this run: {t_encode_total:.1f}s "
          f"({t_encode_total / max(1, len(remaining)):.1f}s/clip average)")
    print(f"Total scorable events (all checkpoints): {len(all_records)}")
    print(f"Total clips processed (all checkpoints): {len(processed_clips)}\n")
    print(f"Saved: {out_path}")
    return out_path


def _load_checkpoint(out_path: Path) -> tuple[list[dict], set[str]]:
    """Reconstruct the record-dict list + processed-clips set from a
    previously-saved .npz, so a resumed run can append to it correctly."""
    data = np.load(out_path, allow_pickle=True)
    n = len(data["labels"])
    records = []
    for i in range(n):
        partner = None
        if bool(data["has_partner"][i]):
            partner = data["partner_embeddings"][i]
        records.append({
            "clip_id": str(data["clip_ids"][i]),
            "index": int(data["indices"][i]),
            "type": str(data["types"][i]),
            "label": int(data["labels"][i]),
            "embedding": data["embeddings"][i],
            "centroid": data["centroids"][i],
            "partner_embedding": partner,
        })
    processed_clips = set(str(c) for c in data["processed_clips"].tolist())
    return records, processed_clips


def _save_npz(records: list[dict], processed_clips: set[str], out_path: Path) -> None:
    """Pack the variable-shape record list into fixed-width parallel arrays
    for .npz storage. partner_embedding rows that are None are stored as
    NaN-filled vectors with a parallel has_partner boolean array, rather
    than omitted, so array lengths stay aligned across all fields.
    `processed_clips` is saved separately from the records themselves so a
    clip that legitimately produced zero scorable events is still
    remembered as done (not re-processed on resume)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        # A checkpoint can legitimately have zero records so far (e.g. the
        # first several clips had no scorable events) -- save an empty-but-
        # well-shaped file rather than raising, so resume logic still works.
        np.savez_compressed(
            out_path,
            embeddings=np.zeros((0, 0), dtype=np.float32),
            centroids=np.zeros((0, 0), dtype=np.float32),
            partner_embeddings=np.zeros((0, 0), dtype=np.float32),
            has_partner=np.zeros(0, dtype=bool),
            labels=np.zeros(0, dtype=np.int64),
            clip_ids=np.empty(0, dtype=object),
            indices=np.zeros(0, dtype=np.int64),
            types=np.empty(0, dtype=object),
            processed_clips=np.array(sorted(processed_clips), dtype=object),
        )
        return

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
        processed_clips=np.array(sorted(processed_clips), dtype=object),
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
