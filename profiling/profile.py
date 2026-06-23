"""Personalized multi-factor word-difficulty profile.

This is the corrected roadmap core:

    difficulty(word) =
        w_onset * onset_risk(word)
      + w_length * normalized_syllable_count(word)
      + w_freq * (1 - zipf_norm(word))
      + w_class * is_content_word(word)

Onset risk is seeded from self-report and Brown-style population priors, then
updated from observed disfluency events with EWMA decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

import nltk
from nltk import pos_tag, word_tokenize

import phonetic
import semantic
from freq import zipf_frequency

from .coldstart import fused_cold_start, load_population_priors
from .config import load_config

ROOT = Path(__file__).resolve().parents[1]
USERS_DIR = ROOT / "users"
USERS_DIR.mkdir(exist_ok=True)


def _safe_username(username: str) -> str:
    safe = re.sub(r"[^a-z0-9_\-]", "", (username or "default").lower())
    return safe or "default"


def profile_path(username: str) -> Path:
    return USERS_DIR / f"{_safe_username(username)}.fluency_profile.json"


def _onset_key(word_or_onset: str | Iterable[str]) -> str:
    if isinstance(word_or_onset, str):
        s = word_or_onset.strip()
        # Treat as a pre-computed ARPAbet onset code (not an English word) when:
        #   • it contains a space  → must be multi-phoneme e.g. "S T R", "K L"
        #   • it's multi-character AND all-uppercase  → e.g. "SH", "TH", "CH"
        # A single uppercase letter like "I" or "A" is ambiguous — it could be
        # the English pronoun/article. Do NOT shortcut those; always run them
        # through phonetic.onset() so vowel-initial words correctly get an
        # empty onset rather than a phantom phoneme label.
        is_arpabet_code = (
            " " in s
            or (len(s) > 1 and s.isupper() and not s.isalpha() is False)
        )
        # Simpler guard: space-separated or 2+ chars all-caps
        is_arpabet_code = " " in s or (len(s) >= 2 and s.isupper())
        if is_arpabet_code:
            return s.upper()
        onset = phonetic.onset(s)
    else:
        onset = tuple(word_or_onset)
    return " ".join(str(p).upper() for p in onset if p)


def _syllable_count(word: str) -> int:
    try:
        return int(getattr(phonetic, "_syllable_count")(word))
    except Exception:
        groups = re.findall(r"[aeiouy]+", (word or "").lower())
        return max(1, len(groups))


def _content_from_tag(tag: str | None) -> bool:
    return bool(tag and tag[:1] in {"N", "V", "J", "R"})


def _guess_tag(word: str) -> str | None:
    try:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        return pos_tag([word])[0][1]
    except Exception:
        return None


@dataclass
class SpeakerDifficultyProfile:
    username: str = "default"
    onset_risk: dict[str, float] = field(default_factory=dict)
    onset_observations: dict[str, dict[str, float]] = field(default_factory=dict)
    self_reported_sounds: list[str] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)
    event_count: int = 0
    config: dict[str, Any] = field(default_factory=load_config)
    population_priors: dict[str, float] = field(default_factory=load_population_priors)

    @classmethod
    def load(cls, username: str = "default") -> "SpeakerDifficultyProfile":
        path = profile_path(username)
        cfg = load_config()
        if not path.exists():
            profile = cls(username=_safe_username(username), config=cfg)
            profile.onset_risk = fused_cold_start(
                [],
                observed_event_count=0,
                confidence_events=int(cfg.get("profiling", {}).get("confidence_events", 30)),
            )
            return profile
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, config=cfg)
        except Exception:
            return cls(username=_safe_username(username), config=cfg)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], config: dict[str, Any] | None = None
    ) -> "SpeakerDifficultyProfile":
        profile = cls(
            username=_safe_username(data.get("username", "default")),
            onset_risk={str(k).upper(): float(v) for k, v in data.get("onset_risk", {}).items()},
            onset_observations=dict(data.get("onset_observations", {})),
            self_reported_sounds=list(data.get("self_reported_sounds", [])),
            sessions=list(data.get("sessions", [])),
            event_count=int(data.get("event_count", 0)),
            config=config or load_config(),
        )
        if not profile.onset_risk:
            profile.onset_risk = fused_cold_start(profile.self_reported_sounds)
        return profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "username": self.username,
            "onset_risk": self.onset_risk,
            "onset_observations": self.onset_observations,
            "self_reported_sounds": self.self_reported_sounds,
            "sessions": self.sessions[-100:],
            "event_count": self.event_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self) -> None:
        path = profile_path(self.username)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.config.get("profiling", {}).get("weights", {}))

    @property
    def alpha(self) -> float:
        return float(self.config.get("profiling", {}).get("ewma_alpha", 0.35))

    def onboarding(self, self_reported_sounds: Iterable[str]) -> None:
        """Seed or refresh the cold-start prior from self-report.

        Only seeds onsets that have *no* observed data yet (onset_observations
        is empty for that key).  For onsets that already have real session
        data, the cold-start value is ignored so we never inflate a
        trained-down risk score back toward the self-report prior.
        """

        sounds = []
        seen = set()
        for raw in self_reported_sounds or []:
            val = str(raw).strip().lower()
            if val and val not in seen:
                seen.add(val)
                sounds.append(val)
        self.self_reported_sounds = sounds
        seed = fused_cold_start(
            sounds,
            observed_event_count=self.event_count,
            population_priors=self.population_priors,
            confidence_events=int(self.config.get("profiling", {}).get("confidence_events", 30)),
        )
        for onset, value in seed.items():
            # Only apply the seed if we have no real observations for this onset.
            # If onset_observations has data, the real EWMA value takes precedence.
            if onset not in self.onset_observations:
                self.onset_risk[onset] = max(float(self.onset_risk.get(onset, 0.0)), float(value))
            # else: real session data already trained this onset — leave it alone.

    def update(
        self,
        session_events: Iterable[dict[str, Any]],
        alpha: float | None = None,
        session_id: str | None = None,
    ) -> None:
        """Fold observed events into onset risk with EWMA.

        Each event can be ``{"word": "buy", "disfluent": true}``, a detector
        output row (presence implies disfluent), or a fluent row with
        ``disfluent: false``.
        """

        a = self.alpha if alpha is None else float(alpha)
        buckets: dict[str, list[float]] = {}
        saved_events: list[dict[str, Any]] = []

        for event in session_events or []:
            # Skip timing-only fallback events that have no reliable word/onset
            # data. These are marked profile_safe=False by the ASR layer.
            if event.get("profile_safe") is False:
                continue
            word_raw = str(event.get("word", "")).strip()
            if not word_raw:
                continue
            # Strip leading/trailing punctuation before onset lookup.
            # CrisperWhisper attaches punctuation to the last word of each
            # sentence ("recording." "place.") — without stripping, _onset_key
            # would receive "recording." which phonetic.onset() can't look up
            # in CMUdict, falling back to grapheme rules that may return wrong
            # or empty results. "recording" → R correctly every time.
            word = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word_raw) or word_raw
            onset = _onset_key(word)
            if not onset:
                continue
            # An event is disfluent if:
            #   • it has an explicit "disfluent" key (bool), OR
            #   • it comes from detect.py and has a "type" field (all detect events
            #     represent disfluencies), OR
            #   • neither key is present → default True (detector-only path)
            if "disfluent" in event:
                disfluent = bool(event["disfluent"])
            else:
                disfluent = event.get("type") is not None
            value = 1.0 if disfluent else 0.0
            buckets.setdefault(onset, []).append(value)
            saved_events.append(
                {
                    "word": word,   # punctuation-stripped
                    "onset": onset,
                    "disfluent": disfluent,
                    "type": event.get("type", "observed" if disfluent else "fluent"),
                }
            )

        for onset, values in buckets.items():
            observed_rate = sum(values) / max(len(values), 1)
            previous = float(self.onset_risk.get(onset, self.population_priors.get(onset, 0.18)))
            updated = a * observed_rate + (1.0 - a) * previous
            self.onset_risk[onset] = round(max(0.0, min(1.0, updated)), 4)
            obs = dict(self.onset_observations.get(onset, {"events": 0, "disfluent": 0}))
            obs["events"] = int(obs.get("events", 0)) + len(values)
            obs["disfluent"] = int(obs.get("disfluent", 0)) + int(sum(values))
            self.onset_observations[onset] = obs

        if saved_events:
            self.event_count += len(saved_events)
            self.sessions.append(
                {
                    "id": session_id or datetime.now(timezone.utc).isoformat(),
                    "events": saved_events,
                    "count": len(saved_events),
                }
            )

    def factors_for_word(self, word: str, tag: str | None = None) -> dict[str, float | str | bool]:
        cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", word or "")
        if not cleaned:
            return {
                "word": word,
                "onset": "",
                "onset_risk": 0.0,
                "length": 0.0,
                "frequency": 0.0,
                "grammatical_class": 0.0,
                "is_content": False,
                "difficulty": 0.0,
            }

        onset = _onset_key(cleaned)
        onset_score = self._risk_for_onset(onset)
        length_score = min(_syllable_count(cleaned) / 4.0, 1.0)
        zipf = zipf_frequency(cleaned.lower(), "en")
        freq_score = 1.0 - min(max(zipf, 0.0) / 7.0, 1.0)
        tag = tag or _guess_tag(cleaned)
        is_content = _content_from_tag(tag) and cleaned.lower() not in semantic._PROTECTED_SINGLE
        class_score = 1.0 if is_content else 0.0

        w = self.weights
        difficulty = (
            float(w.get("onset", 0.45)) * onset_score
            + float(w.get("length", 0.25)) * length_score
            + float(w.get("frequency", 0.20)) * freq_score
            + float(w.get("grammatical_class", 0.10)) * class_score
        )
        return {
            "word": cleaned,
            "onset": onset,
            "onset_risk": round(onset_score, 4),
            "length": round(length_score, 4),
            "frequency": round(freq_score, 4),
            "grammatical_class": round(class_score, 4),
            "is_content": bool(is_content),
            "difficulty": round(max(0.0, min(1.0, difficulty)), 4),
        }

    def difficulty(self, word: str, tag: str | None = None) -> float:
        return float(self.factors_for_word(word, tag).get("difficulty", 0.0))

    def _risk_for_onset(self, onset: str) -> float:
        if not onset:
            return 0.0
        parts = onset.split()
        # Try the full onset first, then progressively shorter prefixes.
        # e.g. "S T R" -> tries "S T R", "S T", "S" in order.
        candidates = [" ".join(parts[:i]) for i in range(len(parts), 0, -1)]
        for key in candidates:
            if key in self.onset_risk:
                return float(self.onset_risk[key])
        # No match found — fall back to population prior for the first phoneme.
        return float(self.population_priors.get(parts[0], 0.18 if parts else 0.0))

    def sentence_difficulty(self, text_or_words: str | Iterable[str]) -> float:
        if isinstance(text_or_words, str):
            try:
                words = word_tokenize(text_or_words)
            except Exception:
                words = re.findall(r"[A-Za-z][A-Za-z'-]*", text_or_words)
        else:
            words = list(text_or_words)
        scored = [
            self.difficulty(str(word))
            for word in words
            if re.search(r"[A-Za-z]", str(word))
        ]
        return round(sum(scored) / len(scored), 4) if scored else 0.0

    def risk_count(self, text_or_words: str | Iterable[str], threshold: float = 0.55) -> int:
        if isinstance(text_or_words, str):
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", text_or_words)
        else:
            words = list(text_or_words)
        return sum(1 for word in words if self.difficulty(str(word)) >= threshold)

    def top_onsets(self, n: int = 8) -> list[tuple[str, float]]:
        return sorted(self.onset_risk.items(), key=lambda item: (-item[1], item[0]))[:n]

    def decay_sensitivity(
        self,
        events: Iterable[dict[str, Any]],
        alphas: Iterable[float] = (0.15, 0.25, 0.35, 0.5, 0.75),
    ) -> list[dict[str, Any]]:
        rows = []
        base = self.to_dict()
        for a in alphas:
            clone = SpeakerDifficultyProfile.from_dict(base, config=self.config)
            clone.update(events, alpha=float(a), session_id=f"sensitivity-{a}")
            rows.append(
                {
                    "alpha": float(a),
                    "event_count": clone.event_count,
                    "top_onsets": clone.top_onsets(5),
                }
            )
        return rows
