"""Cold-start priors for the multi-factor fluency profile."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import phonetic

PRIORS_PATH = Path(__file__).resolve().parent / "default_onset_priors.json"


def _key(onset: Iterable[str] | str) -> str:
    if isinstance(onset, str):
        return onset.strip().upper()
    return " ".join(str(p).upper() for p in onset if p)


def load_population_priors(path: str | Path | None = None) -> dict[str, float]:
    priors_path = Path(path) if path else PRIORS_PATH
    try:
        with open(priors_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k).upper(): float(v) for k, v in data.items()}
    except Exception:
        return {}


def self_report_prior(self_reported_sounds: Iterable[str]) -> dict[str, float]:
    """Convert user-typed sounds into onset-risk seeds."""

    seeded: dict[str, float] = {}
    for raw in self_reported_sounds or []:
        onset = phonetic.normalize_pattern(str(raw))
        if onset:
            seeded[_key(onset)] = 0.82
    return seeded


def fused_cold_start(
    self_reported_sounds: Iterable[str] | None = None,
    observed_event_count: int = 0,
    population_priors: dict[str, float] | None = None,
    confidence_events: int = 30,
) -> dict[str, float]:
    """Blend population priors and self-report.

    The weight of the prior decays as observed data accumulates. The profile
    updater applies personal observations after this seed, so personal data
    naturally takes over.
    """

    population = dict(population_priors or load_population_priors())
    reported = self_report_prior(self_reported_sounds or [])
    prior_weight = max(0.0, 1.0 - min(observed_event_count, confidence_events) / max(confidence_events, 1))

    out: dict[str, float] = {}
    for key in set(population) | set(reported):
        pop = population.get(key, 0.18)
        rep = reported.get(key, pop)
        out[key] = round(prior_weight * max(pop, rep) + (1.0 - prior_weight) * pop, 4)
    return out
