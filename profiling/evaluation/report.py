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


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f},{ci[1]:.3f}]"


def format_table_with_ci(
    counts: dict[str, TypeCounts],
    localization: dict[str, float | None] | None = None,
) -> str:
    """Same as format_table, plus Wilson 95% CI columns for precision and
    recall (ROADMAP.md item 8) — a separate, opt-in function rather than
    changing format_table's default columns, so existing callers/output
    parsing are unaffected. Use where a run's small sample size makes the
    point estimate alone potentially misleading (see VALIDATION.md
    §8.4.3/§8.4.4's small-n cases, exactly what this was built for)."""
    localization = localization or {}
    headers = ["Type", "TP", "FP", "FN", "TN", "Precision", "Precision 95% CI",
               "Recall", "Recall 95% CI", "F1", "Localization(IoU>=0.5)"]
    body = []
    for t, c in counts.items():
        body.append([
            t, str(c.tp), str(c.fp), str(c.fn), str(c.tn),
            _fmt(c.precision), _fmt_ci(c.precision_ci()),
            _fmt(c.recall), _fmt_ci(c.recall_ci()),
            _fmt(c.f1), _fmt(localization.get(t)),
        ])
    widths = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(cells) for cells in body)
    return "\n".join(out)


def format_confidence_stats(stats: dict[str, dict[str, float | int | None]]) -> str:
    """Readable table for metrics.confidence_stats' output (ROADMAP.md
    item 7) — mean confidence of TP vs. FP events, per type."""
    headers = ["Type", "TP mean conf", "FP mean conf", "Gap (TP-FP)", "n_TP", "n_FP"]
    body = []
    for t, s in stats.items():
        tp_m, fp_m = s.get("tp_mean_confidence"), s.get("fp_mean_confidence")
        gap = (tp_m - fp_m) if (tp_m is not None and fp_m is not None) else None
        body.append([
            t,
            "n/a" if tp_m is None else f"{tp_m:.3f}",
            "n/a" if fp_m is None else f"{fp_m:.3f}",
            "n/a" if gap is None else f"{gap:+.3f}",
            str(s.get("n_tp", 0)), str(s.get("n_fp", 0)),
        ])
    widths = [len(h) for h in headers]
    for cells in body:
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    out = [line(headers), "  ".join("-" * w for w in widths)]
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
            # Wilson 95% CIs (ROADMAP.md item 8) -- saved so a small-n run's
            # uncertainty is part of the permanent record, not just visible
            # if someone happens to print format_table_with_ci at run time.
            "precision_ci": c.precision_ci(), "recall_ci": c.recall_ci(),
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
