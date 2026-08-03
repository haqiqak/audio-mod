"""report.py — table rendering + timestamped, reproducible result files.

VALIDATION.md §5 requires every run to record enough to be trusted months
later: which git commit, which config, which dataset, how many clips. Without
that, "F1 improved from X to Y" can't be distinguished from "we changed the
config and forgot."
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from .metrics import TypeCounts

_HEADERS = ["Type", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "Localization(IoU>=0.5)"]


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_table(
    counts: dict[str, TypeCounts],
    localization: dict[str, float | None] | None = None,
) -> str:
    localization = localization or {}
    body = []
    for t, c in counts.items():
        body.append([
            t, str(c.tp), str(c.fp), str(c.fn), str(c.tn),
            _fmt(c.precision), _fmt(c.recall), _fmt(c.f1),
            _fmt(localization.get(t)),
        ])
    widths = [len(h) for h in _HEADERS]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [line(_HEADERS), "  ".join("-" * w for w in widths)]
    out.extend(line(cells) for cells in body)
    return "\n".join(out)


def git_commit() -> str | None:
    """Current commit hash, or None if not in a git repo / git unavailable —
    never raises, since a missing commit hash shouldn't block saving a run."""
    try:
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def counts_to_dict(counts: dict[str, TypeCounts]) -> dict:
    return {
        t: {
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn,
            "precision": c.precision, "recall": c.recall, "f1": c.f1,
        }
        for t, c in counts.items()
    }


def save_run(
    result_dir: Path,
    *,
    dataset: str,
    track: str,
    n_clips: int,
    counts: dict[str, TypeCounts],
    localization: dict[str, float | None] | None = None,
    config: dict | None = None,
    extra_metadata: dict | None = None,
) -> Path:
    """Write one timestamped JSON result file. Never overwrites a prior run —
    each call gets its own filename, so a history of runs accumulates instead
    of the latest silently replacing the last (matching VALIDATION.md §8's
    run log, which is meant to grow, not be overwritten)."""
    result_dir.mkdir(parents=True, exist_ok=True)
    # Microsecond precision, not just seconds: two runs in the same second
    # (e.g. back-to-back self-test calls) must not silently overwrite each
    # other — confirmed by a direct test that second-resolution timestamps
    # collide in exactly this way.
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    path = result_dir / f"{ts}_{dataset}_{track}.json"
    payload = {
        "timestamp_utc": ts,
        "dataset": dataset,
        "track": track,
        "git_commit": git_commit(),
        "n_clips": n_clips,
        "config": config,
        "counts": counts_to_dict(counts),
        "localization": localization,
        "extra_metadata": extra_metadata or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
