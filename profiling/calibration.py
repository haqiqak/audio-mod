"""
calibration.py — one-time speaker-rate baseline, used to personalize
detection thresholds instead of applying the same fixed block/prolongation
cutoffs to every speaker.

Why this exists
────────────────
detect.py's block (silence-gap) and prolongation (duration) thresholds are
global constants (0.55s, 0.65s). A speaker who naturally talks slowly will
trip those thresholds on totally normal speech; a fast talker's real blocks
and prolongations may never clear them. The fix isn't a smarter constant —
it's making the constant relative to *that speaker's* normal rate.

How it works
────────────
1. The speaker reads one fixed, phonetically-neutral calibration sentence —
   no stutter-prone clusters, no rare words, so it measures *rate*, not
   difficulty. This happens once per account, not once per session.
2. From that one clean read we compute the speaker's median word duration
   and median inter-word gap, plus the spread (IQR) of each — a *range*,
   not a single number, because the same person's tempo varies run to run.
3. Detection thresholds become `max(global_floor, baseline_median * k)`
   instead of pure global constants. The global floor is always kept, so a
   speaker who skips calibration (or whose calibration looks broken) gets
   exactly today's fixed-threshold behaviour — calibration only ever makes
   thresholds speaker-aware, never removes the safety floor.

A speaker can re-run calibration later (e.g. after months) if their natural
tempo has drifted; nothing forces a fresh calibration per session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable

# Fixed, phonetically-balanced calibration sentence. Deliberately avoids
# plosive-heavy clusters and rare words — this measures the speaker's
# natural tempo, not their difficulty profile (that's what real sessions
# are for). ~20 words, comfortable to read in one breath at a normal pace.
CALIBRATION_SENTENCE = (
    "The lake was calm this morning, and we walked along the shore "
    "while the sun slowly rose over the hills."
)

# Number of consecutive calibration reads we keep a rolling baseline over.
# A *range* across recent reads, not a single locked-in number.
MAX_BASELINE_SAMPLES = 5


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    s = sorted(values)
    n = len(s)
    q1 = s[max(0, n // 4)]
    q3 = s[min(n - 1, (3 * n) // 4)]
    return max(0.0, q3 - q1)


@dataclass
class SpeakerBaseline:
    """Median + spread of word duration and inter-word gap, pooled across
    up to MAX_BASELINE_SAMPLES calibration reads (a range, not a point)."""

    word_dur_median: float = 0.0
    word_dur_iqr: float = 0.0
    gap_median: float = 0.0
    gap_iqr: float = 0.0
    sample_count: int = 0
    last_calibrated_at: str | None = None

    @property
    def is_usable(self) -> bool:
        # One real calibration read is already enough to start personalizing
        # thresholds — we don't make someone calibrate repeatedly before it
        # does anything. Additional reads (up to MAX_BASELINE_SAMPLES) refine
        # the range further; they aren't a gate on using it at all.
        return self.sample_count >= 1 and self.word_dur_median > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SpeakerBaseline":
        data = data or {}
        return cls(
            word_dur_median=float(data.get("word_dur_median", 0.0)),
            word_dur_iqr=float(data.get("word_dur_iqr", 0.0)),
            gap_median=float(data.get("gap_median", 0.0)),
            gap_iqr=float(data.get("gap_iqr", 0.0)),
            sample_count=int(data.get("sample_count", 0)),
            last_calibrated_at=data.get("last_calibrated_at"),
        )


def measure_calibration_read(
    tokens: Iterable[dict[str, Any]],
) -> dict[str, list[float]] | None:
    """Extract raw word-duration and inter-word-gap samples from one
    calibration read's timestamped tokens.

    Returns None if there isn't enough timing data to be useful (e.g. a
    fixture with no timestamps, or a read that's mostly silence).
    """
    rows = [t for t in tokens if isinstance(t, dict)]
    durations: list[float] = []
    gaps: list[float] = []
    prev_end: float | None = None

    for row in rows:
        start, end = row.get("start"), row.get("end")
        if start is None or end is None:
            prev_end = None
            continue
        try:
            start_f, end_f = float(start), float(end)
        except (TypeError, ValueError):
            prev_end = None
            continue
        if end_f > start_f:
            durations.append(end_f - start_f)
        if prev_end is not None and start_f >= prev_end:
            gap = start_f - prev_end
            # Exclude obvious sentence-boundary-sized pauses (>1.2s) from
            # the *gap* baseline — those are reading pauses, not the
            # speaker's natural word-to-word rhythm, and would otherwise
            # inflate the personalized block threshold sky-high.
            if gap <= 1.2:
                gaps.append(gap)
        prev_end = end_f

    if len(durations) < 3:
        return None
    return {"durations": durations, "gaps": gaps}


def update_baseline(
    existing: SpeakerBaseline, new_samples: dict[str, list[float]]
) -> SpeakerBaseline:
    """Fold one new calibration read into the rolling baseline.

    Pools the new read's raw samples with a bounded amount of "weight"
    from the previous baseline (approximated by re-injecting its median a
    few times) so a handful of recent reads shape the range together,
    rather than the newest read silently overwriting everything before it.
    """
    durations = list(new_samples.get("durations", []))
    gaps = list(new_samples.get("gaps", []))

    if existing.is_usable and existing.sample_count > 0:
        carry_weight = min(existing.sample_count, MAX_BASELINE_SAMPLES - 1)
        durations.extend([existing.word_dur_median] * carry_weight)
        if existing.gap_median > 0:
            gaps.extend([existing.gap_median] * carry_weight)

    return SpeakerBaseline(
        word_dur_median=round(median(durations), 4) if durations else 0.0,
        word_dur_iqr=round(_iqr(durations), 4) if durations else 0.0,
        gap_median=round(median(gaps), 4) if gaps else 0.0,
        gap_iqr=round(_iqr(gaps), 4) if gaps else 0.0,
        sample_count=min(existing.sample_count + 1, MAX_BASELINE_SAMPLES),
        last_calibrated_at=datetime.now(timezone.utc).isoformat(),
    )


def adjusted_thresholds(
    baseline: SpeakerBaseline,
    global_block_gap: float,
    global_prolong_min: float,
    gap_k: float = 2.2,
    duration_k: float = 1.8,
) -> dict[str, float]:
    """Personalize block/prolongation thresholds from a speaker baseline.

    Always returns at least the global floor — calibration can only raise
    a speaker's own bar above the default, never lower detection sensitivity
    below what an uncalibrated speaker gets. A naturally slow speaker's
    longer baseline gap means we require an even longer pause before
    calling it a "block"; a naturally fast speaker keeps the global floor.

    gap_k / duration_k are how many baseline-spread-widths above the
    speaker's own median we require before flagging — i.e. "longer than
    your own normal range", not an absolute number borrowed from someone
    else's speech.
    """
    if not baseline.is_usable:
        return {"block_gap_seconds": global_block_gap, "prolongation_min_seconds": global_prolong_min}

    personal_gap = baseline.gap_median + gap_k * baseline.gap_iqr
    personal_dur = baseline.word_dur_median + duration_k * baseline.word_dur_iqr

    return {
        "block_gap_seconds": round(max(global_block_gap, personal_gap), 4),
        "prolongation_min_seconds": round(max(global_prolong_min, personal_dur), 4),
    }
