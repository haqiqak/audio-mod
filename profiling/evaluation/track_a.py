"""track_a.py — detector-only evaluation (ASR bypassed).

Feeds a dataset's own ground-truth words/timestamps directly into
detect_disfluencies(), skipping CrisperWhisper entirely. Answers: given a
perfect transcript, how good is the detection logic itself? See
VALIDATION.md §3 for why this is tracked separately from the full pipeline
(Track B, not yet built — see VALIDATION.md §6 sequencing).

"ASR bypassed" is about the TRANSCRIPT source, not about audio. Audio is a
second, independent axis: if a clip's ground truth (loaders.LabeledClip)
carries `audio_bytes`, this module passes it straight through to
detect_disfluencies(), which activates the full audio-native layer (Silero
VAD corroboration, Praat pitch/jitter/shimmer, weighted acoustic-vs-token
fusion) — same as a live recording would. Without it, only the text/timing-
based checks run. Both modes matter: text-only isolates the detector logic
from any audio quality/decoding variance; audio-enabled is the only way the
audio-native layer (this project's main 2026-08 architectural change) gets
evaluated against labeled ground truth at all, and is closer to what an
actual user experiences. See PAPER_DECISION_LOG.md's "Audio-enabled
evaluation" entry for the reasoning behind prioritizing this over acquiring
a new dataset.

Usage
─────
    python -m profiling.evaluation.track_a                          # bundled synthetic sample
    python -m profiling.evaluation.track_a --dataset libristutter --data-dir DIR
    python -m profiling.evaluation.track_a --dataset libristutter --data-dir DIR --audio-dir DIR2
    python -m profiling.evaluation.track_a --self-test               # verify the harness itself

Results are printed and, unless --no-save is passed, written to
eval_results/ as a timestamped JSON file (see report.py) — gitignored, not
committed, since results depend on whatever real dataset is on the local
machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.detect import detect_disfluencies
from profiling.evaluation.loaders import (
    LIBRISTUTTER_SCORABLE_TYPES,
    LabeledClip,
    load_libristutter_dir,
    load_libristutter_dir_with_audio,
    synthetic_libristutter_sample,
)
from profiling.evaluation.metrics import (
    ANY_LABEL,
    format_confusion_matrix,
    localization_rate,
    score_word_level,
)
from profiling.evaluation.report import format_table, save_run

_DEFAULT_RESULT_DIR = _ROOT / "eval_results"

# dataset name -> (scorable types, loader-from-dir, synthetic-sample-for-self-test)
_DATASETS = {
    "libristutter": (
        LIBRISTUTTER_SCORABLE_TYPES, load_libristutter_dir, synthetic_libristutter_sample,
    ),
}
# datasets that also support an --audio-dir (paired loader: annotations dir, audio dir)
_AUDIO_LOADERS = {
    "libristutter": load_libristutter_dir_with_audio,
}


def evaluate(
    clips: list[LabeledClip], scorable_types: tuple[str, ...], config: dict | None = None,
) -> tuple[dict, dict]:
    """Run detect_disfluencies() over every clip's ground-truth tokens and
    score the result. Uses each clip's own `audio_bytes` (None for text-only
    loaders, real WAV bytes for audio-enabled ones — see the module
    docstring) so this one function serves both modes; the caller decides
    which by which loader it used. Returns (counts, localization)."""
    predictions = [
        detect_disfluencies(c.tokens, config=config, audio_bytes=c.audio_bytes)
        for c in clips
    ]
    counts = score_word_level(clips, predictions, scorable_types)
    localization = localization_rate(clips, predictions, scorable_types)
    return counts, localization


def run(
    dataset: str, data_dir: Path | None, audio_dir: Path | None = None,
    save_results: bool = True, result_dir: Path = _DEFAULT_RESULT_DIR,
) -> int:
    if dataset not in _DATASETS:
        print(f"Unknown dataset {dataset!r}. Available: {', '.join(_DATASETS)}")
        return 2
    scorable_types, load_dir, synthetic = _DATASETS[dataset]

    if audio_dir and not data_dir:
        print("--audio-dir requires --data-dir (the annotations directory).")
        return 2

    if audio_dir:
        if dataset not in _AUDIO_LOADERS:
            print(f"{dataset!r} has no audio-enabled loader yet.")
            return 2
        if not audio_dir.exists():
            print(f"--audio-dir does not exist: {audio_dir}")
            return 2
        print(f"Evaluating with real audio from {audio_dir} (this activates the full "
              f"audio-native detection layer, not just text/timing checks)...")
        clips = _AUDIO_LOADERS[dataset](data_dir, audio_dir)
    elif data_dir:
        if not data_dir.exists():
            print(f"--data-dir does not exist: {data_dir}")
            return 2
        clips = load_dir(data_dir)
    else:
        clips = synthetic()
        print(
            f"No --data-dir given - evaluating {len(clips)} bundled synthetic clip(s) "
            f"for {dataset}.\nThis is a schema smoke-test, NOT a real accuracy number. "
            f"Point --data-dir at a real download for an actual benchmark result - "
            f"see VALIDATION.md section 2.\n"
        )

    if not clips:
        print(f"No labeled clips found under: {data_dir}")
        return 2
    print(f"Evaluating {len(clips)} labeled clip(s) from {data_dir or 'synthetic sample'} ...")

    counts, localization = evaluate(clips, scorable_types)
    print(format_table(counts, localization))

    if save_results:
        n_with_audio = sum(1 for c in clips if c.audio_bytes is not None)
        path = save_run(
            result_dir, dataset=dataset, track=("A+audio" if audio_dir else "A"),
            n_clips=len(clips), counts=counts, localization=localization,
            extra_metadata={
                "data_dir": str(data_dir) if data_dir else "synthetic_sample",
                "audio_dir": str(audio_dir) if audio_dir else None,
                "n_clips_with_audio": n_with_audio,
            },
        )
        print(f"\nSaved: {path}")
    return 0


# ── Self-test (verifies scoring math directly, independent of the detector) ──

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. score_word_level against hand-computed TP/FP/FN/TN — independent of the detector.
    clip = LabeledClip(
        name="hand", tokens=[{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.2}
                              for i, w in enumerate(["a", "b", "c", "d"])],
        ground_truth={1: "filler", 3: "prolongation"},
    )
    fake_predictions = [[
        {"index": 1, "type": "filler", "start": 0.3, "end": 0.5},
        {"index": 2, "type": "word_repetition", "start": 0.6, "end": 0.8},
    ]]
    counts = score_word_level([clip], fake_predictions, LIBRISTUTTER_SCORABLE_TYPES)
    check("filler TP", counts["filler"].tp == 1, str(counts["filler"]))
    check("word_repetition FP", counts["word_repetition"].fp == 1, str(counts["word_repetition"]))
    check("prolongation FN", counts["prolongation"].fn == 1, str(counts["prolongation"]))
    check("filler precision 1.0", counts["filler"].precision == 1.0, str(counts["filler"].precision))
    check("prolongation recall 0.0", counts["prolongation"].recall == 0.0, str(counts["prolongation"].recall))
    check("sound_repetition all TN -> precision n/a",
          counts["sound_repetition"].precision is None, str(counts["sound_repetition"].precision))
    check("sound_repetition TN counted", counts["sound_repetition"].tn == 4, str(counts["sound_repetition"]))
    check(f"{ANY_LABEL} label present", ANY_LABEL in counts, str(list(counts)))
    check(f"{ANY_LABEL} TP counts the filler hit", counts[ANY_LABEL].tp == 1, str(counts[ANY_LABEL]))
    print("\n--- sample confusion matrix ---")
    print(format_confusion_matrix("filler", counts["filler"]))

    # 2. localization_rate — IoU against the ground-truth token's own span.
    loc = localization_rate([clip], fake_predictions, LIBRISTUTTER_SCORABLE_TYPES)
    check("filler localization is a perfect overlap (IoU=1.0 -> counted)",
          loc["filler"] == 1.0, str(loc["filler"]))
    check("prolongation localization is None (zero true positives)",
          loc["prolongation"] is None, str(loc["prolongation"]))
    # A predicted span that barely overlaps should fail the IoU>=0.5 threshold.
    poor_overlap_predictions = [[{"index": 1, "type": "filler", "start": 0.49, "end": 0.51}]]
    loc_poor = localization_rate([clip], poor_overlap_predictions, LIBRISTUTTER_SCORABLE_TYPES)
    check("poor-overlap prediction fails IoU>=0.5", loc_poor["filler"] == 0.0, str(loc_poor["filler"]))
    # acoustic_start/acoustic_end must be preferred over the nominal span.
    acoustic_predictions = [[{
        "index": 1, "type": "filler", "start": 0.3, "end": 0.5,
        "acoustic_start": 100.0, "acoustic_end": 200.0,  # deliberately absurd, must be used anyway
    }]]
    loc_acoustic = localization_rate([clip], acoustic_predictions, LIBRISTUTTER_SCORABLE_TYPES)
    check("localization prefers acoustic_start/end over nominal span",
          loc_acoustic["filler"] == 0.0, str(loc_acoustic["filler"]))

    # 3. CSV round-trip (loaders.py)
    import csv
    import tempfile
    from profiling.evaluation.loaders import load_libristutter_csv, load_sep28k_labels

    # 3a. Defensive fallback path: a label directly on a real word's own row
    # (not the confirmed real convention, but must not silently drop data
    # if it's ever encountered).
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hello", "0.0", "0.2", "0"])
            w.writerow(["uh", "0.2", "0.5", "1"])
        loaded = load_libristutter_csv(p)
        check("fallback: csv loaded 2 tokens", len(loaded.tokens) == 2, str(len(loaded.tokens)))
        check("fallback: label on a real word's own row is honored",
              loaded.ground_truth == {1: "filler"}, str(loaded.ground_truth))

    # 3b. The CONFIRMED real convention: a separate "STUTTER" placeholder
    # row between two real words (verified 2026-08-03 against real
    # downloaded LibriStutter files — see PAPER_DECISION_LOG.md).
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "y.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mrs.", "1.1", "1.8", "0"])
            w.writerow(["Rachel", "1.8", "2.1", "0"])
            w.writerow(["STUTTER", "2.1", "3.08", "3"])  # word_repetition
            w.writerow(["Lynde", "3.08", "3.48", "0"])
        loaded = load_libristutter_csv(p)
        check("STUTTER-row: 4 tokens (marker reconstructed, not dropped)",
              len(loaded.tokens) == 4, str(len(loaded.tokens)))
        check("STUTTER-row: reconstructed word repeats the preceding real word",
              loaded.tokens[2]["word"] == "Rachel", str(loaded.tokens))
        check("STUTTER-row: ground truth attaches to the reconstructed token, not a real word",
              loaded.ground_truth == {2: "word_repetition"}, str(loaded.ground_truth))
        check("STUTTER-row: literal word 'STUTTER' never reaches the token list",
              all(t["word"] != "STUTTER" for t in loaded.tokens), str(loaded.tokens))
        # This reconstruction must actually produce a detectable event.
        events = detect_disfluencies(loaded.tokens, audio_bytes=None)
        check("STUTTER-row reconstruction is actually detectable as word_repetition",
              any(e["index"] == 2 and e["type"] == "word_repetition" for e in events),
              str(events))

    # 3b-audio. Audio-enabled loading (2026-08-03 addition): FLAC decode ->
    # 16kHz WAV -> attached to LabeledClip.audio_bytes -> actually activates
    # the audio-native detection layer. Synthetic FLAC (no real download
    # needed for a self-test), same tone-based convention as every other
    # audio test in this project.
    import numpy as np
    import soundfile as sf
    from profiling.evaluation.loaders import (
        _flac_bytes_to_wav16k, load_libristutter_csv_with_audio, load_libristutter_dir_with_audio,
    )
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "ann").mkdir()
        (root / "audio").mkdir()
        sr = 22050
        t = np.arange(int(1.5 * sr)) / sr
        tone = (np.sin(2 * np.pi * 150 * t) * 0.2).astype(np.float32)
        flac_bytes_buf = Path(d) / "_tmp.flac"
        sf.write(flac_bytes_buf, tone, sr, format="FLAC")
        flac_bytes = flac_bytes_buf.read_bytes()

        wav16k = _flac_bytes_to_wav16k(flac_bytes)
        check("flac decode produces WAV bytes", wav16k is not None and len(wav16k) > 0,
              str(len(wav16k) if wav16k else None))
        import wave as _wave
        import io as _io
        with _wave.open(_io.BytesIO(wav16k)) as _wf:
            check("flac decode resamples to 16kHz", _wf.getframerate() == 16000,
                  str(_wf.getframerate()))
            pcm = np.frombuffer(_wf.readframes(_wf.getnframes()), dtype=np.int16)
        # Directly checks for non-silence — this is the exact assertion that
        # would have caught a real bug found 2026-08-03: sf.read(...,
        # dtype="int16") silently returned all-zero samples for real
        # LibriStutter FLAC files (a soundfile/libsndfile quirk with those
        # files' encoding), which this self-test's original "bytes were
        # produced, frame rate is right" checks did not catch, because the
        # synthetic FLAC below happened not to trigger the same quirk. An
        # entire real evaluation run executed successfully against
        # completely silent "audio" before this was caught by manually
        # inspecting RMS on real downloaded data — see
        # PAPER_DECISION_LOG.md. Never trust "it produced output" as
        # equivalent to "the output is real audio" again.
        check("flac decode produces genuinely non-silent audio (RMS > 0)",
              float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) > 100,
              f"RMS={np.sqrt(np.mean(pcm.astype(np.float64) ** 2)):.2f}")
        check("flac decode on garbage bytes degrades to None, doesn't raise",
              _flac_bytes_to_wav16k(b"not a flac file") is None)

        csv_path = root / "ann" / "spk" / "chap" / "spk-chap-0000.csv"
        csv_path.parent.mkdir(parents=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["so", "0.0", "0.3", "0"])
            w.writerow(["saaaw", "0.3", "1.5", "5"])  # prolongation, matches the tone's real span
        flac_path = root / "audio" / "spk" / "chap" / "spk-chap-0000.flac"
        flac_path.parent.mkdir(parents=True)
        flac_path.write_bytes(flac_bytes)

        clip = load_libristutter_csv_with_audio(csv_path, flac_path)
        check("with-audio loader attaches audio_bytes", clip.audio_bytes is not None,
              str(clip.audio_bytes is None))
        events_with_audio = detect_disfluencies(clip.tokens, audio_bytes=clip.audio_bytes)
        events_without = detect_disfluencies(clip.tokens, audio_bytes=None)
        check("audio-enabled evaluation actually differs from text-only for a real prolongation "
              "(more/annotated evidence, not identical output)",
              events_with_audio != events_without or any("RMS" in e.get("evidence", "") for e in events_with_audio),
              f"with={events_with_audio}\nwithout={events_without}")

        clips_dir = load_libristutter_dir_with_audio(root / "ann", root / "audio")
        check("dir-level with-audio loader finds the one clip", len(clips_dir) == 1, str(len(clips_dir)))
        check("dir-level with-audio loader pairs it with audio",
              clips_dir[0].audio_bytes is not None, str(clips_dir[0].audio_bytes is None))

        # Missing audio must degrade gracefully (text-only for that clip), not crash the batch.
        missing_dir = root / "audio_missing"
        missing_dir.mkdir()
        clips_missing = load_libristutter_dir_with_audio(root / "ann", missing_dir)
        check("missing audio degrades to text-only, not a crash",
              len(clips_missing) == 1 and clips_missing[0].audio_bytes is None,
              str(clips_missing[0].audio_bytes if clips_missing else "no clips"))

    # 3c. SEP-28k's real, confirmed clip-level schema (2026-08-03) —
    # comma-space-separated, counts out of 3 annotators, no transcript.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sep28k.csv"
        p.write_text(
            "Show,EpId,ClipId,Start,Stop,Unsure,PoorAudioQuality,Prolongation,"
            "Block,SoundRep,WordRep,DifficultToUnderstand,Interjection,"
            "NoStutteredWords,NaturalPause,Music,NoSpeech\n"
            "HeStutters, 0, 0, 0, 48000, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0\n"   # majority Prolongation
            "HeStutters, 0, 1, 48000, 96000, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0\n",  # only 1/3 -> not counted
            encoding="utf-8",
        )
        clips = load_sep28k_labels(p)
        check("sep28k: 2 clips parsed", len(clips) == 2, str(len(clips)))
        check("sep28k: clip name built from Show_EpId_ClipId",
              clips[0].name == "HeStutters_0_0", clips[0].name)
        check("sep28k: majority-agreement prolongation counted as present",
              clips[0].present_types == {"prolongation"}, str(clips[0].present_types))
        check("sep28k: 1-of-3 agreement (below default threshold) not counted",
              clips[1].present_types == set(), str(clips[1].present_types))
        check("sep28k: audio_path is None (labels only, no download attempted)",
              clips[0].audio_path is None, str(clips[0].audio_path))

    # 4. End-to-end against the bundled synthetic sample + the real detector
    sample = synthetic_libristutter_sample()
    result, loc_e2e = evaluate(sample, LIBRISTUTTER_SCORABLE_TYPES)
    check("evaluate() returns all scorable types + Any",
          set(result) == set(LIBRISTUTTER_SCORABLE_TYPES) | {ANY_LABEL}, str(set(result)))
    table = format_table(result, loc_e2e)
    for h in ("Type", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1"):
        check(f"table has header {h!r}", h in table)
    print("\n--- sample table (synthetic data, not a real accuracy claim) ---")
    print(table)

    # 5. report.py — save_run writes a well-formed, timestamped, non-clobbering file
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result_dir = Path(d)
        path1 = save_run(result_dir, dataset="libristutter", track="A",
                          n_clips=len(sample), counts=result, localization=loc_e2e)
        path2 = save_run(result_dir, dataset="libristutter", track="A",
                          n_clips=len(sample), counts=result, localization=loc_e2e)
        check("save_run produces a file", path1.exists(), str(path1))
        check("two calls do not clobber each other", path1 != path2, f"{path1} vs {path2}")
        with open(path1, "r", encoding="utf-8") as f:
            payload = json.load(f)
        check("saved payload has dataset field", payload.get("dataset") == "libristutter", str(payload))
        check("saved payload has counts for filler", "filler" in payload.get("counts", {}), str(payload))

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Track A: score detect_disfluencies() against labeled clips, ASR bypassed."
    )
    parser.add_argument("--dataset", default="libristutter", choices=list(_DATASETS))
    parser.add_argument(
        "--data-dir", default=None,
        help="Folder of labeled annotations for --dataset (searched recursively). "
             "Defaults to a small bundled synthetic sample if omitted.",
    )
    parser.add_argument(
        "--audio-dir", default=None,
        help="Folder of matching real audio (requires --data-dir). Activates the full "
             "audio-native detection layer (Silero VAD, Praat, acoustic fusion) instead "
             "of text/timing-only evaluation. See the module docstring.",
    )
    parser.add_argument("--no-save", action="store_true", help="Don't write a result file.")
    parser.add_argument(
        "--self-test", action="store_true",
        help="Verify the scoring math itself against hand-computed expectations.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    data_dir = Path(args.data_dir) if args.data_dir else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    return run(args.dataset, data_dir, audio_dir, save_results=not args.no_save)


if __name__ == "__main__":
    raise SystemExit(main())
