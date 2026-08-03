"""metrics.py — dataset-agnostic scoring: precision/recall/F1, per-type
confusion matrices, the combined "Any" label, and IoU-based localization.

See VALIDATION.md §4 for why each of these was chosen (matching SEP-28k's own
reporting convention for comparability; per-type binary confusion matrices
rather than a single multi-class one, per the field's multi-label literature;
IoU>=0.5 as the localization threshold used in the dysfluency-localization
literature reviewed for this project).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .loaders import LabeledClip

ANY_LABEL = "Any"
DEFAULT_IOU_THRESHOLD = 0.5


@dataclass
class TypeCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

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

    @property
    def accuracy(self) -> float | None:
        denom = self.tp + self.fp + self.fn + self.tn
        return None if denom == 0 else (self.tp + self.tn) / denom


def _predicted_types_by_index(events: list[dict]) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for e in events:
        out.setdefault(e["index"], set()).add(e["type"])
    return out


def score_word_level(
    clips: list[LabeledClip],
    predictions: list[list[dict]],
    scorable_types: Iterable[str],
    include_any: bool = True,
) -> dict[str, TypeCounts]:
    """Per-type TP/FP/FN/TN over all (clip, token index) pairs, for word-level
    ground truth (e.g. LibriStutter). Pure function of ground truth +
    already-computed predictions — kept separate from running the detector
    so the scoring math itself can be unit-tested independent of
    detect_disfluencies (see the package's tests).

    include_any additionally reports a combined "Any" label (disfluent vs.
    clean, ignoring type) — matches SEP-28k's own paper's reporting
    convention, for comparability with published baselines.
    """
    scorable_types = tuple(scorable_types)
    counts = {t: TypeCounts() for t in scorable_types}
    if include_any:
        counts[ANY_LABEL] = TypeCounts()

    for clip, events in zip(clips, predictions):
        predicted_by_index = _predicted_types_by_index(events)

        for idx in range(len(clip.tokens)):
            predicted_here = predicted_by_index.get(idx, set())
            true_type = clip.ground_truth.get(idx)

            for t in scorable_types:
                predicted_t = t in predicted_here
                true_t = true_type == t
                c = counts[t]
                if predicted_t and true_t:
                    c.tp += 1
                elif predicted_t and not true_t:
                    c.fp += 1
                elif true_t and not predicted_t:
                    c.fn += 1
                else:
                    c.tn += 1

            if include_any:
                predicted_any = bool(predicted_here & set(scorable_types))
                true_any = true_type in scorable_types
                c = counts[ANY_LABEL]
                if predicted_any and true_any:
                    c.tp += 1
                elif predicted_any and not true_any:
                    c.fp += 1
                elif true_any and not predicted_any:
                    c.fn += 1
                else:
                    c.tn += 1

    return counts


def score_clip_level(
    clip_names: list[str],
    ground_truth_types: list[set[str]],
    predicted_types: list[set[str]],
    scorable_types: Iterable[str],
    include_any: bool = True,
) -> dict[str, TypeCounts]:
    """Per-type TP/FP/FN/TN at clip granularity: did this clip contain type T
    ANYWHERE (ground truth) vs. did our pipeline predict type T ANYWHERE in
    it (prediction) — no word-level location involved. This is the shape
    SEP-28k-style datasets need (see loaders.ClipLevelLabels): they provide
    no reference transcript, so word-index matching isn't possible, only
    presence/absence per clip. `predicted_types` must come from something
    that can run on that clip's audio without a reference transcript
    (acoustic-only detection, or a full Track B pipeline) — not yet wired to
    a runner, see VALIDATION.md §6.
    """
    scorable_types = tuple(scorable_types)
    counts = {t: TypeCounts() for t in scorable_types}
    if include_any:
        counts[ANY_LABEL] = TypeCounts()

    for true_set, pred_set in zip(ground_truth_types, predicted_types):
        for t in scorable_types:
            predicted_t, true_t = t in pred_set, t in true_set
            c = counts[t]
            if predicted_t and true_t:
                c.tp += 1
            elif predicted_t and not true_t:
                c.fp += 1
            elif true_t and not predicted_t:
                c.fn += 1
            else:
                c.tn += 1
        if include_any:
            predicted_any = bool(pred_set & set(scorable_types))
            true_any = bool(true_set & set(scorable_types))
            c = counts[ANY_LABEL]
            if predicted_any and true_any:
                c.tp += 1
            elif predicted_any and not true_any:
                c.fp += 1
            elif true_any and not predicted_any:
                c.fn += 1
            else:
                c.tn += 1

    return counts


# ── Localization (IoU) ─────────────────────────────────────────────────────

def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return 0.0
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def _event_span(event: dict) -> tuple[float, float] | None:
    """Prefer the acoustic-native detector's precise region
    (acoustic_start/acoustic_end) over the attributed token's full nominal
    span, the same preference app.py's Event table uses (see
    PAPER_DECISION_LOG.md's 2026-08-03 display-fix entry) — for the same
    reason: it's the more accurate claim of where the event actually is."""
    if event.get("acoustic_start") is not None:
        return event["acoustic_start"], event.get("acoustic_end")
    start, end = event.get("start"), event.get("end")
    if start is None or end is None:
        return None
    return start, end


def localization_rate(
    clips: list[LabeledClip],
    predictions: list[list[dict]],
    scorable_types: Iterable[str],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float | None]:
    """For each type, the fraction of word-level true positives (correct
    type at the correct token index) whose predicted span also clears
    `iou_threshold` against the ground-truth token's own span. None when a
    type has zero true positives to measure localization against — this is
    a "how precise are our correct detections", not a detection-rate metric
    (that's precision/recall from score_word_level).
    """
    scorable_types = tuple(scorable_types)
    hits = {t: 0 for t in scorable_types}
    totals = {t: 0 for t in scorable_types}

    for clip, events in zip(clips, predictions):
        predicted_by_index: dict[int, list[dict]] = {}
        for e in events:
            predicted_by_index.setdefault(e["index"], []).append(e)

        for idx, true_type in clip.ground_truth.items():
            if true_type not in scorable_types:
                continue
            matches = [e for e in predicted_by_index.get(idx, []) if e["type"] == true_type]
            if not matches:
                continue  # false negative — not a localization question
            totals[true_type] += 1
            gt_span = (clip.tokens[idx]["start"], clip.tokens[idx]["end"])
            if gt_span[0] is None or gt_span[1] is None:
                continue
            best_iou = max(
                (_iou(*_event_span(e), *gt_span) for e in matches if _event_span(e) is not None),
                default=0.0,
            )
            if best_iou >= iou_threshold:
                hits[true_type] += 1

    return {
        t: (hits[t] / totals[t] if totals[t] else None)
        for t in scorable_types
    }


# ── Confusion matrix rendering ─────────────────────────────────────────────

def format_confusion_matrix(type_name: str, c: TypeCounts) -> str:
    """A readable 2x2 binary confusion matrix for one type. Deliberately
    per-type/binary, not a single N x N multi-class matrix — disfluencies
    co-occur (Bayerl et al., "A Stutter Seldom Comes Alone," Interspeech
    2023), and forcing multi-label data into one multi-class matrix is a
    documented methodological error for this kind of data — see
    VALIDATION.md §4.
    """
    return (
        f"{type_name}\n"
        f"                 predicted +   predicted -\n"
        f"  actual +       TP={c.tp:<8}  FN={c.fn:<8}\n"
        f"  actual -       FP={c.fp:<8}  TN={c.tn:<8}"
    )
