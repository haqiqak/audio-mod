"""Configuration loading for the fluency roadmap modules.

The project keeps the public knobs in the repository root ``config.yaml``.
PyYAML is optional: if it is not installed, the modules continue with the
checked-in defaults so tests and offline use still work.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: dict[str, Any] = {
    "profiling": {
        "ewma_alpha": 0.35,
        "confidence_events": 30,
        "weights": {
            "onset": 0.45,
            "length": 0.25,
            "frequency": 0.20,
            "grammatical_class": 0.10,
        },
        "detection": {
            "repetition_window": 2,
            "prolongation_percentile": 90,
            "prolongation_min_seconds": 0.65,
            "block_gap_seconds": 0.55,
            "filler_words": ["uh", "um", "er", "erm", "like"],
        },
    },
    "rewrite": {
        "lambda": 0.55,
        "mu": 0.12,
        "tau": 0.80,
        "top_k": 12,
        "risk_threshold": 0.55,
        "min_improvement": 0.01,
        "lambda_sweep": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    "eval": {"output_dir": "outputs"},
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return merged roadmap config.

    If ``config.yaml`` cannot be read or PyYAML is unavailable, defaults are
    returned. Callers may still pass per-call overrides for experiments.
    """

    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        return deepcopy(DEFAULT_CONFIG)
    try:
        import yaml  # type: ignore

        with open(cfg_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            return deepcopy(DEFAULT_CONFIG)
        return _deep_merge(DEFAULT_CONFIG, loaded)
    except Exception:
        return deepcopy(DEFAULT_CONFIG)


def section(name: str, path: str | Path | None = None) -> dict[str, Any]:
    return dict(load_config(path).get(name, {}))
