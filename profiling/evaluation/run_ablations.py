"""run_ablations.py — ablation studies against the 2026-08-03 research
baseline (VALIDATION.md §8.3): which components of the audio-native
detection layer actually drive its measured F1 gain over text-only
detection?

Loads the labeled clips + real audio ONCE (FLAC decode and the Silero VAD
model load are the expensive parts — the VAD model is cached at module
level in profiling/acoustic.py, so running every variant in one process is
materially faster than N separate CLI invocations) and re-runs
detect_disfluencies() once per config variant, holding everything else
constant. Each variant is scored the same way as track_a.py and saved via
report.save_run with a descriptive track tag (e.g. "ablation-novad").

Usage
─────
    python -m profiling.evaluation.run_ablations \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio

See VALIDATION.md §9 for the full results table and interpretation.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.config import load_config
from profiling.evaluation.loaders import LIBRISTUTTER_SCORABLE_TYPES, load_libristutter_dir_with_audio
from profiling.evaluation.report import format_table, save_run
from profiling.evaluation.track_a import evaluate

_DEFAULT_RESULT_DIR = _ROOT / "eval_results"


def _base_config() -> dict:
    return copy.deepcopy(load_config()["profiling"]["detection"])


def _variants() -> list[tuple[str, dict]]:
    """(name, detection-config dict) pairs. The first is always the
    unmodified baseline (fusion_weights.acoustic=1.0, use_vad=True,
    use_praat=True, prolongation_min_seconds=1.0, matching VALIDATION.md
    §8.3 exactly) — every other variant is diffed against it."""
    base = _base_config()
    variants: list[tuple[str, dict]] = [("baseline", copy.deepcopy(base))]

    novad = copy.deepcopy(base)
    novad["acoustic"]["use_vad"] = False
    variants.append(("vad_off", novad))

    nopraat = copy.deepcopy(base)
    nopraat["acoustic"]["use_praat"] = False
    variants.append(("praat_off", nopraat))

    for w in (0.5, 2.0, 5.0):
        cfg = copy.deepcopy(base)
        cfg["fusion_weights"] = {"rule": 1.0, "acoustic": w}
        variants.append((f"fusion_weight_{w}", cfg))

    for thr in (0.65, 0.85, 1.2, 1.4):
        cfg = copy.deepcopy(base)
        cfg["prolongation_min_seconds"] = thr
        variants.append((f"prolong_threshold_{thr}", cfg))

    return variants


def run(data_dir: Path, audio_dir: Path, result_dir: Path = _DEFAULT_RESULT_DIR) -> list[dict]:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    print(f"{len(clips)} clips loaded.\n")

    rows = []
    for name, cfg in _variants():
        print(f"--- {name} ---")
        counts, localization = evaluate(clips, LIBRISTUTTER_SCORABLE_TYPES, config=cfg)
        print(format_table(counts, localization))
        path = save_run(
            result_dir, dataset="libristutter", track=f"ablation-{name}",
            n_clips=len(clips), counts=counts, localization=localization,
            config=cfg, extra_metadata={"ablation_variant": name},
        )
        print(f"Saved: {path}\n")
        any_c = counts["Any"]
        prolong_c = counts["prolongation"]
        rows.append({
            "variant": name,
            "any_tp": any_c.tp, "any_fp": any_c.fp, "any_fn": any_c.fn,
            "any_precision": any_c.precision, "any_recall": any_c.recall, "any_f1": any_c.f1,
            "prolong_tp": prolong_c.tp, "prolong_fp": prolong_c.fp, "prolong_fn": prolong_c.fn,
            "prolong_precision": prolong_c.precision, "prolong_recall": prolong_c.recall,
            "prolong_f1": prolong_c.f1,
        })

    print("\n=== Summary (Any label) ===")
    print(f"{'variant':<22}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    for r in rows:
        print(
            f"{r['variant']:<22}{r['any_tp']:>5}{r['any_fp']:>5}{r['any_fn']:>5}"
            f"{(r['any_precision'] or 0):>8.3f}{(r['any_recall'] or 0):>8.3f}{(r['any_f1'] or 0):>8.3f}"
        )
    print("\n=== Summary (prolongation) ===")
    print(f"{'variant':<22}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    for r in rows:
        print(
            f"{r['variant']:<22}{r['prolong_tp']:>5}{r['prolong_fp']:>5}{r['prolong_fn']:>5}"
            f"{(r['prolong_precision'] or 0):>8.3f}{(r['prolong_recall'] or 0):>8.3f}{(r['prolong_f1'] or 0):>8.3f}"
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Annotations directory.")
    parser.add_argument("--audio-dir", required=True, help="Matching real-audio directory.")
    args = parser.parse_args(argv)

    data_dir, audio_dir = Path(args.data_dir), Path(args.audio_dir)
    if not data_dir.exists() or not audio_dir.exists():
        print("--data-dir and --audio-dir must both exist.")
        return 2

    run(data_dir, audio_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
