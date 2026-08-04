"""run_encoder_signal_stage1.py — Phase 3, Stage 1 (VALIDATION.md §11):
measure whether CrisperWhisper's own last-layer encoder hidden states
carry a signal that separates the current detector's true positives from
its false positives, for word_repetition/sound_repetition/filler.

Loads real audio + labels, runs today's shipped detector (default config,
unchanged) to get the current TP/FP events, runs CrisperWhisper's encoder
(NOT the full model — no decoding, see encoder_features.py's module
docstring) once per clip to get its last-layer hidden states, computes
each scorable event's cosine distance to that clip's own fluent-token
centroid, then aggregates with metrics.encoder_distance_stats().

Cost, discovered while implementing this (recorded as a dated addendum to
VALIDATION.md §11, not silently absorbed): Whisper always pads to a fixed
30s window before the encoder runs, so the encoder pass is NOT
proportionally cheaper for short clips — it is the dominant cost of a
full transcription (~44s of ~54s measured for a 4s clip on this project's
CPU environment, ARCHITECTURE.md §3) and skipping decoding only saves the
remaining ~10s. A full 499-clip run is therefore ~6 hours of CPU time,
not a few minutes. Defaults to a smaller pilot (--n-clips, default 30,
matching this project's own Track B pilot-then-scale precedent) rather
than silently running the full sample; pass a larger --n-clips explicitly
to scale up once the pilot's result is reviewed.

Usage
─────
    python -m profiling.evaluation.run_encoder_signal_stage1 \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --n-clips 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.acoustic import load_wav_samples
from profiling.detect import detect_disfluencies
from profiling.evaluation.encoder_features import (
    attach_encoder_distances,
    extract_last_layer_states,
    load_encoder,
)
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.metrics import encoder_distance_stats
from profiling.evaluation.report import format_encoder_distance_stats, save_run

_DEFAULT_RESULT_DIR = _ROOT / "eval_results"

# word_repetition/sound_repetition/filler only — ROADMAP.md item 17's
# scope; prolongation/block are already audio-native, phrase_repetition
# is excluded for the same reconstruction-limitation reason it's excluded
# from other detector-improvement work (VALIDATION.md §8.2).
STAGE1_TARGET_TYPES = ("word_repetition", "sound_repetition", "filler")


def run(
    data_dir: Path, audio_dir: Path, n_clips: int | None,
    result_dir: Path = _DEFAULT_RESULT_DIR,
) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    if n_clips is not None:
        clips = clips[:n_clips]
    print(f"{len(clips)} clips loaded (of the full sample, per --n-clips).\n")

    print("Loading CrisperWhisper's processor + encoder (not the full "
          "model -- no decoding) ...")
    t0 = time.perf_counter()
    processor, encoder = load_encoder()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s.\n")

    all_events: list[list[dict]] = []
    t_encode_total = 0.0
    for i, clip in enumerate(clips, 1):
        samples, sr = load_wav_samples(clip.audio_bytes) if clip.audio_bytes else (None, None)
        events = detect_disfluencies(clip.tokens, audio_bytes=clip.audio_bytes)
        if samples is None:
            all_events.append(events)
            print(f"[{i}/{len(clips)}] {clip.name}: no audio, skipped encoder pass")
            continue

        t1 = time.perf_counter()
        states = extract_last_layer_states(processor, encoder, samples, sr)
        dt = time.perf_counter() - t1
        t_encode_total += dt

        events = attach_encoder_distances(
            states, clip.tokens, clip.ground_truth, events, STAGE1_TARGET_TYPES,
        )
        all_events.append(events)
        print(f"[{i}/{len(clips)}] {clip.name}: encoder pass {dt:.1f}s")

    print(f"\nTotal encoder time: {t_encode_total:.1f}s "
          f"({t_encode_total / max(1, len(clips)):.1f}s/clip average)\n")

    stats = encoder_distance_stats(clips, all_events, STAGE1_TARGET_TYPES)
    print(format_encoder_distance_stats(stats))

    path = save_run(
        result_dir, dataset="libristutter", track="stage1-encoder-signal",
        n_clips=len(clips), counts={},  # not a TypeCounts table -- distance stats saved via extra_metadata
        extra_metadata={
            "encoder_distance_stats": stats,
            "target_types": list(STAGE1_TARGET_TYPES),
            "total_encoder_seconds": round(t_encode_total, 1),
        },
    )
    print(f"\nSaved: {path}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Annotations directory.")
    parser.add_argument("--audio-dir", required=True, help="Matching real-audio directory.")
    parser.add_argument(
        "--n-clips", type=int, default=30,
        help="Limit to the first N clips (default 30, a pilot -- see module "
             "docstring for why the full 499-clip sample is NOT the default). "
             "Pass 0 or a number >= the sample size to run the full sample.",
    )
    args = parser.parse_args(argv)

    data_dir, audio_dir = Path(args.data_dir), Path(args.audio_dir)
    if not data_dir.exists() or not audio_dir.exists():
        print("--data-dir and --audio-dir must both exist.")
        return 2

    n_clips = None if args.n_clips in (0, None) else args.n_clips
    run(data_dir, audio_dir, n_clips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
