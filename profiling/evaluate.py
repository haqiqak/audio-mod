"""evaluate.py — the first objective accuracy number for detect_disfluencies.

Before this file existed, the detector's only accuracy evidence was the demo
fixture's fixed 9-token/7-event regression count and informal real-audio spot
checks logged in PAPER_DECISION_LOG.md — no run against a public, labeled
benchmark. This harness closes that gap using LibriStutter (Zhang et al.,
Queen's University), the labeled dataset whose schema is the closest
structural match to this app's own token format: word, start, end, and a
per-word disfluency label, generated from LibriSpeech with time-aligned
transcriptions.

LibriStutter's label scheme (confirmed against the dataset's own README):
    0 clean | 1 interjection | 2 sound repetition | 3 word repetition
    4 phrase repetition | 5 prolongation
maps directly onto this app's taxonomy (profiling/detect.py):
    interjection -> filler, sound repetition -> sound_repetition,
    word repetition -> word_repetition, phrase repetition -> phrase_repetition,
    prolongation -> prolongation.
LibriStutter has no "block" label (a block is fundamentally an absence of
speech, harder to synthesize the way LibriStutter's other categories are) —
block and stutter_marker are therefore outside what THIS harness can score;
a future pass against SEP-28k or KSoF (which do label blocks) is the natural
next step, not done here.

Scope boundary, stated honestly: this harness runs detect_disfluencies() in
TIMESTAMP-ONLY mode (audio_bytes=None). LibriStutter ships FLAC audio, and
decoding FLAC isn't a dependency this project currently has (adding one is a
separate, deliberate decision, not a silent side-effect of building this
harness). This still exercises the majority of the taxonomy — every
text/timing-based detector (filler, all three repetition types, and
timestamp-duration prolongation) — just not the acoustic-fusion path. That
path is separately covered by tests/test_detect_fusion.py's synthetic-audio
tests. Extending this harness to real audio is documented future work, not
a gap papered over here.

Usage
─────
    python -m profiling.evaluate                    # bundled synthetic sample
    python -m profiling.evaluate --data-dir DIR      # real LibriStutter-format tree
                                                      #   DIR/**/*.csv (+ matching audio, unused)
    python -m profiling.evaluate --self-test         # verify the scoring math itself

The bundled sample (generated on the fly, not committed as binary fixtures —
same convention as benchmark_asr.py's --self-test WAVs) is a small,
schema-compatible synthetic set for exercising the harness end-to-end. It is
NOT a substitute for running against the real corpus and is not sized or
sourced to produce a publishable accuracy claim — see README.md for how to
point --data-dir at a real download.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from profiling.detect import detect_disfluencies

# LibriStutter integer label -> this app's event type. 0 (clean) has no entry.
LIBRISTUTTER_LABEL_MAP: dict[int, str] = {
    1: "filler",
    2: "sound_repetition",
    3: "word_repetition",
    4: "phrase_repetition",
    5: "prolongation",
}
# Only these types are meaningfully scorable against LibriStutter (see module
# docstring — no "block" label exists in this dataset).
SCORABLE_TYPES = tuple(LIBRISTUTTER_LABEL_MAP.values())


@dataclass
class LabeledClip:
    name: str
    tokens: list[dict]                 # [{"word", "start", "end"}, ...]
    ground_truth: dict[int, str]       # token index -> our event type (only labeled rows)


@dataclass
class TypeCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return None if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return None if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)


# ── LibriStutter CSV loading ─────────────────────────────────────────────────

def load_libristutter_csv(csv_path: Path) -> LabeledClip:
    """Parse one LibriStutter-format annotation file: rows of
    (word, start_seconds, end_seconds, label 0-5), no header row.

    Assumption (documented, not yet verified against a real download in this
    environment — flag for whoever runs this against the real corpus first):
    each row is treated as one token, and a non-zero label marks THAT row's
    own word as the disfluency event. If real data turns out to use a
    separate marker-row convention (e.g. a distinct 'STUTTER' row preceding
    the affected word) this function's row-to-token mapping is the one place
    to adjust — everything downstream (scoring) is unaffected by that detail.
    """
    tokens: list[dict] = []
    ground_truth: dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row or len(row) < 4:
                continue
            word, start, end, label = row[0], row[1], row[2], row[3]
            tokens.append({"word": word, "start": float(start), "end": float(end)})
            label_int = int(float(label))
            if label_int in LIBRISTUTTER_LABEL_MAP:
                ground_truth[i] = LIBRISTUTTER_LABEL_MAP[label_int]
    return LabeledClip(name=csv_path.stem, tokens=tokens, ground_truth=ground_truth)


def load_data_dir(data_dir: Path) -> list[LabeledClip]:
    """Every *.csv under data_dir (recursive — matches LibriStutter's
    SpeakerID/ChapterID/*.csv layout), each parsed as one labeled clip."""
    return [load_libristutter_csv(p) for p in sorted(data_dir.rglob("*.csv"))]


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_clips(
    clips: list[LabeledClip], predictions: list[list[dict]],
) -> dict[str, TypeCounts]:
    """Per-type TP/FP/FN over all (clip, token index) pairs, for the types in
    SCORABLE_TYPES. Pure function of ground truth + already-computed
    predictions — kept separate from running the detector so the scoring
    math itself can be unit-tested (see run_self_test) independent of
    detect_disfluencies.
    """
    counts = {t: TypeCounts() for t in SCORABLE_TYPES}
    for clip, events in zip(clips, predictions):
        predicted_types_by_index: dict[int, set[str]] = {}
        for e in events:
            predicted_types_by_index.setdefault(e["index"], set()).add(e["type"])

        all_indices = set(range(len(clip.tokens)))
        for idx in all_indices:
            predicted_here = predicted_types_by_index.get(idx, set())
            true_type = clip.ground_truth.get(idx)
            for t in SCORABLE_TYPES:
                predicted_t = t in predicted_here
                true_t = true_type == t
                if predicted_t and true_t:
                    counts[t].tp += 1
                elif predicted_t and not true_t:
                    counts[t].fp += 1
                elif true_t and not predicted_t:
                    counts[t].fn += 1
    return counts


def evaluate(clips: list[LabeledClip], config: dict | None = None) -> dict[str, TypeCounts]:
    predictions = [detect_disfluencies(c.tokens, config=config, audio_bytes=None) for c in clips]
    return score_clips(clips, predictions)


# ── Table rendering (mirrors benchmark_asr.py's format_table convention) ──────

_HEADERS = ["Type", "TP", "FP", "FN", "Precision", "Recall", "F1"]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_table(counts: dict[str, TypeCounts]) -> str:
    body = [
        [t, str(c.tp), str(c.fp), str(c.fn), _fmt(c.precision), _fmt(c.recall), _fmt(c.f1)]
        for t, c in counts.items()
    ]
    widths = [len(h) for h in _HEADERS]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [line(_HEADERS), "  ".join("-" * w for w in widths)]
    out.extend(line(cells) for cells in body)
    return "\n".join(out)


# ── Synthetic, schema-compatible sample (self-test + no-data-dir default) ─────

def _synthetic_sample() -> list[LabeledClip]:
    """A small, hand-built set in LibriStutter's exact (word, start, end,
    label) schema. Generated in-process (not committed binary fixtures),
    same convention as benchmark_asr.py's _write_silence_wav self-test WAVs.
    Deliberately NOT a substitute for the real corpus — see module docstring.
    """
    def clip(name: str, rows: list[tuple[str, float, float, int]]) -> LabeledClip:
        tokens = [{"word": w, "start": s, "end": e} for w, s, e, _ in rows]
        gt = {i: LIBRISTUTTER_LABEL_MAP[lbl] for i, (_, _, _, lbl) in enumerate(rows) if lbl}
        return LabeledClip(name=name, tokens=tokens, ground_truth=gt)

    return [
        clip("sample_filler", [
            ("well", 0.0, 0.2, 0), ("uh", 0.2, 0.5, 1), ("i", 0.5, 0.7, 0),
            ("think", 0.7, 1.0, 0), ("so", 1.0, 1.2, 0),
        ]),
        clip("sample_word_repetition", [
            ("the", 0.0, 0.2, 0), ("the", 0.2, 0.4, 3), ("cat", 0.4, 0.7, 0),
            ("sat", 0.7, 1.0, 0), ("down", 1.0, 1.3, 0),
        ]),
        clip("sample_phrase_repetition", [
            ("i", 0.0, 0.2, 0), ("want", 0.2, 0.5, 0), ("to", 0.5, 0.7, 0),
            ("i", 0.7, 0.9, 4), ("want", 0.9, 1.2, 0), ("to", 1.2, 1.4, 0),
            ("go", 1.4, 1.7, 0),
        ]),
        clip("sample_prolongation", [
            ("so", 0.0, 0.2, 0), ("i", 0.2, 0.4, 0), ("saaaaaw", 0.4, 1.7, 5),
            ("it", 1.7, 1.9, 0), ("there", 1.9, 2.2, 0),
        ]),
    ]


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

    # 1. score_clips against hand-computed TP/FP/FN — independent of the detector.
    clip = LabeledClip(
        name="hand", tokens=[{"word": w, "start": i * 0.3, "end": i * 0.3 + 0.2}
                              for i, w in enumerate(["a", "b", "c", "d"])],
        ground_truth={1: "filler", 3: "prolongation"},
    )
    # Predictions: index1 correctly predicted filler (TP), index2 a spurious
    # word_repetition (FP, ground truth is clean), index3 prolongation missed (FN).
    fake_predictions = [[
        {"index": 1, "type": "filler"},
        {"index": 2, "type": "word_repetition"},
    ]]
    counts = score_clips([clip], fake_predictions)
    check("filler TP", counts["filler"].tp == 1, str(counts["filler"]))
    check("word_repetition FP", counts["word_repetition"].fp == 1, str(counts["word_repetition"]))
    check("prolongation FN", counts["prolongation"].fn == 1, str(counts["prolongation"]))
    check("filler precision 1.0", counts["filler"].precision == 1.0, str(counts["filler"].precision))
    check("prolongation recall 0.0", counts["prolongation"].recall == 0.0, str(counts["prolongation"].recall))
    check("sound_repetition all-zero -> precision n/a",
          counts["sound_repetition"].precision is None, str(counts["sound_repetition"].precision))

    # 2. CSV round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["hello", "0.0", "0.2", "0"])
            w.writerow(["uh", "0.2", "0.5", "1"])
        loaded = load_libristutter_csv(p)
        check("csv loaded 2 tokens", len(loaded.tokens) == 2, str(len(loaded.tokens)))
        check("csv ground truth maps label 1 -> filler",
              loaded.ground_truth == {1: "filler"}, str(loaded.ground_truth))

    # 3. End-to-end against the bundled synthetic sample + the real detector
    sample = _synthetic_sample()
    result = evaluate(sample)
    check("evaluate() returns all scorable types", set(result) == set(SCORABLE_TYPES), str(set(result)))
    table = format_table(result)
    for h in _HEADERS:
        check(f"table has header {h!r}", h in table)
    print("\n--- sample table (synthetic data, not a real accuracy claim) ---")
    print(table)

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score detect_disfluencies() against LibriStutter-format labeled clips."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Folder of LibriStutter-format *.csv annotations (searched recursively). "
             "Defaults to a small bundled synthetic sample if omitted.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Verify the scoring math itself against hand-computed expectations.",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"--data-dir does not exist: {data_dir}")
            return 2
        clips = load_data_dir(data_dir)
        if not clips:
            print(f"No .csv annotation files found under: {data_dir}")
            return 2
        print(f"Evaluating {len(clips)} labeled clip(s) from {data_dir} …")
    else:
        clips = _synthetic_sample()
        print(
            f"No --data-dir given — evaluating {len(clips)} bundled synthetic clip(s).\n"
            "This is a schema smoke-test, NOT a real accuracy number. Point --data-dir at a "
            "real LibriStutter download for an actual benchmark result.\n"
        )

    result = evaluate(clips)
    print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
