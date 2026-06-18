"""Rule-based disfluency detection over verbatim ASR tokens."""

from __future__ import annotations

from statistics import quantiles
import re
from typing import Any, Iterable

from .config import load_config


def _as_dict(token: Any) -> dict[str, Any]:
    if isinstance(token, dict):
        return token
    if hasattr(token, "to_dict"):
        return token.to_dict()
    return {
        "word": getattr(token, "word", ""),
        "start": getattr(token, "start", None),
        "end": getattr(token, "end", None),
        "is_filler": getattr(token, "is_filler", False),
        "is_stutter": getattr(token, "is_stutter", False),
        "source": getattr(token, "source", None),
        "profile_safe": getattr(token, "profile_safe", True),
    }


def _norm(word: str) -> str:
    return re.sub(r"[^a-z]", "", (word or "").lower())


def _duration(token: dict[str, Any]) -> float | None:
    try:
        start = token.get("start")
        end = token.get("end")
        if start is None or end is None:
            return None
        return max(0.0, float(end) - float(start))
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) < 3:
        return max(values)
    try:
        cuts = quantiles(values, n=100, method="inclusive")
        idx = min(98, max(0, int(pct) - 1))
        return cuts[idx]
    except Exception:
        return max(values)


def detect_disfluencies(
    tokens: Iterable[Any],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Flag repetitions, prolongations, blocks, fillers, and ASR stutter marks.

    The detector is deliberately transparent and conservative. It returns one
    event per flagged token with ``word``, ``index``, ``type``, ``confidence``,
    and ``evidence`` fields.
    """

    rows = [_as_dict(t) for t in tokens]
    cfg = config or load_config().get("profiling", {}).get("detection", {})
    filler_words = set(cfg.get("filler_words", ["uh", "um", "er", "erm", "like"]))
    block_gap = float(cfg.get("block_gap_seconds", 0.55))
    prolong_min = float(cfg.get("prolongation_min_seconds", 0.65))
    prolong_pct = float(cfg.get("prolongation_percentile", 90))

    durations = [d for d in (_duration(t) for t in rows) if d is not None]
    prolong_threshold = max(prolong_min, _percentile(durations, prolong_pct))

    events: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def add(index: int, kind: str, confidence: float, evidence: str) -> None:
        key = (index, kind)
        if key in seen:
            return
        seen.add(key)
        event = {
            "word": rows[index].get("word", ""),
            "index": index,
            "start": rows[index].get("start"),
            "end": rows[index].get("end"),
            "type": kind,
            "confidence": round(confidence, 3),
            "evidence": evidence,
        }
        if rows[index].get("source"):
            event["source"] = rows[index].get("source")
        if rows[index].get("profile_safe") is False:
            event["profile_safe"] = False
        events.append(event)

    for i, token in enumerate(rows):
        word = str(token.get("word", ""))
        low = _norm(word)
        if not low:
            continue

        if token.get("is_filler") or low in filler_words:
            add(i, "filler", 0.9, "ASR filler marker or known filler word")

        if token.get("is_stutter") or word.endswith("-"):
            add(i, "stutter_marker", 0.85, "ASR stutter marker or trailing fragment")

        if i > 0:
            prev_low = _norm(str(rows[i - 1].get("word", "")))
            prev_word = str(rows[i - 1].get("word", ""))
            if low and prev_low and low == prev_low:
                add(i, "repetition", 0.92, "same token repeated back-to-back")
            if prev_word.endswith("-") and prev_low and low.startswith(prev_low):
                add(i, "repetition", 0.86, "sub-word fragment before this word")

            prev_end = rows[i - 1].get("end")
            start = token.get("start")
            if prev_end is not None and start is not None:
                try:
                    gap = float(start) - float(prev_end)
                    if gap >= block_gap:
                        add(i, "block", min(0.95, gap / max(block_gap, 0.01)), f"silent gap {gap:.2f}s")
                except Exception:
                    pass

        dur = _duration(token)
        if dur is not None and dur >= prolong_threshold and low not in filler_words:
            add(i, "prolongation", min(0.95, dur / max(prolong_threshold, 0.01)), f"duration {dur:.2f}s")

    return sorted(events, key=lambda e: (e["index"], e["type"]))
