"""loaders.py — per-dataset loading into a shared clip representation.

See this package's __init__.py for why LibriStutter and SEP-28k-style
datasets need different LabeledClip granularities rather than one shape
forced onto both.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class LabeledClip:
    """Word-level ground truth: one entry per labeled token index.

    Matches this app's own VerbatimToken shape (word/start/end) directly, so
    `tokens` can be fed straight into detect_disfluencies() for Track A
    (see track_a.py) with no ASR involved.

    `audio_bytes` is optional (None by default — the original,
    annotation-only Track A behavior). When present (16kHz mono PCM WAV),
    `track_a.evaluate()` passes it straight through to
    `detect_disfluencies()`, which activates the full audio-native detection
    layer (Silero VAD corroboration, Praat pitch/jitter/shimmer, weighted
    acoustic-vs-token fusion) — this is the ONLY way that layer has ever
    been evaluated against labeled ground truth; the original text-only
    LabeledClip usage never exercised it. See PAPER_DECISION_LOG.md's
    "Audio-enabled evaluation" entry for why this was prioritized over
    acquiring a new dataset.
    """
    name: str
    tokens: list[dict[str, Any]]        # [{"word", "start", "end"}, ...]
    ground_truth: dict[int, str]        # token index -> our event type (only labeled rows)
    audio_bytes: bytes | None = None    # 16kHz mono PCM WAV, or None for text-only evaluation


@dataclass
class ClipLevelLabels:
    """Clip-level ground truth: does this clip contain each type ANYWHERE,
    with no reference transcript at all (e.g. SEP-28k's schema). Cannot
    drive detect_disfluencies() directly — there's no ground-truth word
    sequence to feed it — so this shape is for Track B (full pipeline,
    including our own ASR) once that exists, or for scoring the
    acoustic-native detectors (block/prolongation, which need no transcript)
    directly against the audio. Not yet wired to a runner — see
    VALIDATION.md §6 sequencing.
    """
    name: str
    audio_path: str | None              # None if only labels are available (audio not yet fetched)
    present_types: set[str]             # our event types this clip's ground truth says are present


# ── LibriStutter (word-level) ────────────────────────────────────────────────

# LibriStutter integer label -> this app's event type. 0 (clean) has no entry.
# Confirmed against the dataset's own documentation (hhzhang16/LibriStutterData).
LIBRISTUTTER_LABEL_MAP: dict[int, str] = {
    1: "filler",
    2: "sound_repetition",
    3: "word_repetition",
    4: "phrase_repetition",
    5: "prolongation",
}
LIBRISTUTTER_SCORABLE_TYPES = tuple(LIBRISTUTTER_LABEL_MAP.values())


_STUTTER_MARKER = "STUTTER"


def load_libristutter_csv(csv_path: Path) -> LabeledClip:
    """Parse one LibriStutter-format annotation file: rows of
    (word, start_seconds, end_seconds, label 0-5), no header row.

    CONFIRMED (2026-08-03, against real downloaded files — this is not the
    guessed schema the original version of this function assumed; that
    assumption was wrong and is corrected here): disfluencies are NOT a
    label on a real word's own row. Every non-zero-label row has the literal
    placeholder word "STUTTER" instead of a real transcribed word — the
    label and timestamps describe a synthetically-inserted disfluency
    segment sitting between two real words, e.g.:

        Rachel,1.8,2.1,0
        STUTTER,2.1,3.08,3          <- label 3 = word_repetition
        Lynde,3.08,3.48,0

    Feeding "STUTTER" to detect_disfluencies() as a literal word would (a)
    corrupt text-based word/phrase-repetition matching with a word nobody
    said, and (b) leave the ground-truth span disconnected from any real
    token, breaking IoU localization scoring. Instead, each STUTTER row is
    reconstructed into a plausible real token, using the immediately
    preceding real word (falling back to the following word if the STUTTER
    row is first) — this is a documented approximation, not a verified
    transcription of what LibriStutter's synthesis actually spliced in:
      - word_repetition / sound_repetition / phrase_repetition: the
        STUTTER row becomes an exact copy of the adjacent word's text (a
        repeat), keeping the STUTTER row's own timestamps — this matches
        detect_disfluencies()'s exact-repeat and duration-based checks
        directly. Sound_repetition additionally gets a trailing "-" (e.g.
        "rachel-") to look like a sub-word fragment, matching the fragment
        check's expected shape. Phrase_repetition is NOT reconstructed as a
        true multi-word repeat (this loader doesn't know the repeated
        phrase's length) — it's approximated as a single-word repeat, which
        means our phrase-repetition detector (which needs a genuine
        multi-word match) will honestly score low recall on this type here;
        that's a known limitation of this reconstruction, not a detector
        bug — see VALIDATION.md.
      - prolongation: the STUTTER row becomes a copy of the adjacent word's
        text with the STUTTER row's own (long) duration — a direct fit for
        detect_disfluencies()'s duration-based prolongation check.
      - interjection (-> filler): the STUTTER row becomes the word "uh"
        with `is_filler=True` set explicitly, since the actual inserted
        filler word's identity isn't recoverable from this row alone, and
        the `is_filler` flag makes the specific word choice not
        scoring-relevant (detect_disfluencies() checks the flag first).
    """
    raw_rows: list[tuple[str, float, float, int]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 4:
                continue
            word, start, end, label = row[0], row[1], row[2], row[3]
            raw_rows.append((word, float(start), float(end), int(float(label))))

    tokens: list[dict[str, Any]] = []
    ground_truth: dict[int, str] = {}

    for i, (word, start, end, label_int) in enumerate(raw_rows):
        if word != _STUTTER_MARKER:
            tokens.append({"word": word, "start": start, "end": end})
            # Defensive fallback, not the confirmed primary case (every real
            # file inspected uses the STUTTER-marker convention below): if a
            # real word's own row ever carries a non-zero label directly,
            # honor it rather than silently dropping it.
            if label_int in LIBRISTUTTER_LABEL_MAP:
                ground_truth[len(tokens) - 1] = LIBRISTUTTER_LABEL_MAP[label_int]
            continue
        if label_int not in LIBRISTUTTER_LABEL_MAP:
            continue  # a STUTTER row should always carry a real label, but degrade gracefully if not

        # Find the nearest real (non-STUTTER) word: prefer preceding, fall
        # back to following (only relevant if STUTTER is the very first row).
        adjacent = next(
            (raw_rows[j][0] for j in range(i - 1, -1, -1) if raw_rows[j][0] != _STUTTER_MARKER),
            None,
        ) or next(
            (raw_rows[j][0] for j in range(i + 1, len(raw_rows)) if raw_rows[j][0] != _STUTTER_MARKER),
            None,
        )
        if adjacent is None:
            continue  # pathological: a clip that's entirely STUTTER rows — nothing to attach to

        our_type = LIBRISTUTTER_LABEL_MAP[label_int]
        if our_type == "filler":
            reconstructed = "uh"
        elif our_type == "sound_repetition":
            reconstructed = adjacent.rstrip(".,!?;:") + "-"
        else:  # word_repetition, phrase_repetition (approximated), prolongation
            reconstructed = adjacent

        token: dict[str, Any] = {"word": reconstructed, "start": start, "end": end}
        if our_type == "filler":
            token["is_filler"] = True
        tokens.append(token)
        ground_truth[len(tokens) - 1] = our_type

    return LabeledClip(name=csv_path.stem, tokens=tokens, ground_truth=ground_truth)


def load_libristutter_dir(data_dir: Path) -> list[LabeledClip]:
    """Every *.csv under data_dir (recursive — matches LibriStutter's
    SpeakerID/ChapterID/*.csv layout), each parsed as one labeled clip."""
    return [load_libristutter_csv(p) for p in sorted(data_dir.rglob("*.csv"))]


def _flac_bytes_to_wav16k(flac_bytes: bytes) -> bytes | None:
    """Decode FLAC bytes (LibriStutter's real audio format, e.g. 22050 Hz —
    not one of Silero VAD's supported rates of 8000/16000 Hz) into 16kHz
    mono PCM WAV bytes, reusing profiling/asr.py's existing, already-tested
    resample_to_16k rather than duplicating resampling logic. Returns None
    on any decode failure — callers must degrade to text-only for that clip,
    not crash a whole batch over one bad file.

    IMPORTANT: read with soundfile's default dtype (float64), NOT
    dtype="int16". Confirmed by direct testing (2026-08-03) that
    sf.read(..., dtype="int16") silently returns an all-zero array for real
    LibriStutter FLAC files (soundfile/libsndfile quirk with these files'
    encoding, not a corrupt-file problem — the default float64 read decodes
    the same bytes correctly). This produced a real, initially-confusing
    result: an evaluation run against completely-silent "audio" that still
    executed without error, silently discarding the entire audio-native
    detection layer instead of exercising it. Do not "simplify" this back to
    dtype="int16" without re-verifying against a real file first.
    """
    try:
        import io
        import wave

        import numpy as np
        import soundfile as sf

        from profiling.asr import resample_to_16k

        data, sr = sf.read(io.BytesIO(flac_bytes), dtype="float64")
        if data.ndim > 1:
            data = data.mean(axis=1)
        pcm = np.clip(data * 32767.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return resample_to_16k(buf.getvalue())
    except Exception:
        return None


def load_libristutter_csv_with_audio(csv_path: Path, flac_path: Path) -> LabeledClip:
    """Like load_libristutter_csv, but also decodes and attaches the
    matching real audio (see LabeledClip.audio_bytes) so Track A can
    exercise the full audio-native detection layer, not just the text/
    timing-based checks. Falls back to text-only (audio_bytes=None) if the
    FLAC file is missing or fails to decode — never raises for a bad clip."""
    clip = load_libristutter_csv(csv_path)
    if flac_path.exists():
        clip.audio_bytes = _flac_bytes_to_wav16k(flac_path.read_bytes())
    return clip


def load_libristutter_dir_with_audio(annotations_dir: Path, audio_dir: Path) -> list[LabeledClip]:
    """Every *.csv under annotations_dir, each paired with the matching
    *.flac under audio_dir (same relative path, .csv -> .flac) if present.
    Clips whose audio is missing/undecodable still get scored (text-only) —
    a partial audio sample isn't discarded, it's the honest, documented
    fraction actually evaluated with audio (see VALIDATION.md)."""
    clips = []
    n_with_audio = 0
    for csv_path in sorted(annotations_dir.rglob("*.csv")):
        rel = csv_path.relative_to(annotations_dir)
        flac_path = audio_dir / rel.with_suffix(".flac")
        clip = load_libristutter_csv_with_audio(csv_path, flac_path)
        if clip.audio_bytes is not None:
            n_with_audio += 1
        clips.append(clip)
    print(f"  ({n_with_audio}/{len(clips)} clips have usable audio)")
    return clips


def synthetic_libristutter_sample() -> list[LabeledClip]:
    """A small, hand-built set in LibriStutter's exact (word, start, end,
    label) schema. Generated in-process (not committed binary fixtures),
    same convention as benchmark_asr.py's self-test WAVs. Deliberately NOT a
    substitute for the real corpus — see VALIDATION.md §1/§7.
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


# ── SEP-28k (clip-level) ────────────────────────────────────────────────────
#
# Schema confirmed 2026-08-03 directly against the real downloaded file
# (github.com/apple/ml-stuttering-events-dataset/SEP-28k_labels.csv) — not
# guessed. Header row (fields are comma-*space*-separated, hence
# skipinitialspace below):
#
#   Show,EpId,ClipId,Start,Stop,Unsure,PoorAudioQuality,Prolongation,Block,
#   SoundRep,WordRep,DifficultToUnderstand,Interjection,NoStutteredWords,
#   NaturalPause,Music,NoSpeech
#
# Start/Stop are SAMPLE indices (Stop-Start == 48000 consistently == 3.0s at
# 16kHz — confirms every clip is exactly 3 seconds), not seconds and not a
# word location — there is no reference transcript anywhere in this dataset,
# confirming ClipLevelLabels (not LabeledClip) is the only correct shape.
# Each disfluency-type column is a COUNT out of 3 annotators who selected it
# for that clip, not a boolean — load_sep28k_labels() below takes an
# agreement threshold (default: majority, >=2) rather than hardcoding one,
# since different published work uses different thresholds and this should
# be a stated, changeable choice, not a silent one.

SEP28K_TYPE_COLUMNS: dict[str, str] = {
    "Prolongation": "prolongation",
    "Block": "block",
    "SoundRep": "sound_repetition",
    "WordRep": "word_repetition",
    "Interjection": "filler",
}
SEP28K_SCORABLE_TYPES = tuple(SEP28K_TYPE_COLUMNS.values())


def load_sep28k_labels(
    csv_path: Path, min_annotator_agreement: int = 2,
) -> list[ClipLevelLabels]:
    """Parse SEP-28k_labels.csv (or SEP-28k-E's equivalent) into
    ClipLevelLabels — one per 3-second clip, no reference transcript.

    `audio_path` is always None here: matching a clip to actual audio bytes
    requires SEP-28k_episodes.csv (podcast URLs) plus the upstream
    download_audio.py/extract_clips.py pipeline — a real download, not
    attempted by this function. Populating audio_path (once audio has been
    fetched) is a separate, later step — see VALIDATION.md §6.
    """
    clips: list[ClipLevelLabels] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            name = f"{row.get('Show','?')}_{row.get('EpId','?')}_{row.get('ClipId','?')}"
            present: set[str] = set()
            for column, our_type in SEP28K_TYPE_COLUMNS.items():
                try:
                    count = int(row[column])
                except (KeyError, ValueError, TypeError):
                    continue
                if count >= min_annotator_agreement:
                    present.add(our_type)
            clips.append(ClipLevelLabels(name=name, audio_path=None, present_types=present))
    return clips
