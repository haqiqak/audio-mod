"""Rule-based disfluency detection over verbatim ASR tokens.

Improvements over the previous version
───────────────────────────────────────
1. Punctuation-stripped prolongation threshold
   Words like "recording." no longer skew the per-clip duration percentile —
   punctuation is stripped before all duration comparisons and word matching.

2. Sentence-position awareness
   Stuttering is overwhelmingly sentence-initial (first word, or first word
   after a sentence-boundary pause). Events at those positions get a small
   confidence boost (+0.08) and their evidence strings note the position.

3. Short-clip prolongation guard
   With fewer than 5 tokens the 90th-percentile is meaningless (every word
   looks prolonged). For short clips we fall back to a flat 1.5× the absolute
   minimum instead of the percentile, preventing every single word from being
   flagged as a prolongation.

4. Interjection-sandwiched repetition ("I uh I")
   A filler between two identical words is a classic stuttering pattern not
   caught by back-to-back exact-match alone. Now detected as a repetition on
   the second word with a note that a filler intervened.

5. Acoustic validation (carried over from previous version)
   Blocks confirmed by near-zero gap RMS; prolongations confirmed by sustained
   voiced energy + low zero-crossing rate.

6. Near-repetition + phrase repetition (carried over)

All acoustic thresholds are configurable in config.yaml under
profiling.detection.acoustic.*. Detector degrades gracefully to
timestamp-only mode when audio_bytes is None.
"""

from __future__ import annotations

import wave
from io import BytesIO
from statistics import quantiles
import re
from typing import Any, Iterable

import numpy as np

from .config import load_config


# ── Token normalisation ───────────────────────────────────────────────────────

def _as_dict(token: Any) -> dict[str, Any]:
    if isinstance(token, dict):
        return token
    if hasattr(token, "to_dict"):
        return token.to_dict()
    return {
        "word":         getattr(token, "word",         ""),
        "start":        getattr(token, "start",        None),
        "end":          getattr(token, "end",           None),
        "is_filler":    getattr(token, "is_filler",    False),
        "is_stutter":   getattr(token, "is_stutter",   False),
        "source":       getattr(token, "source",        None),
        "profile_safe": getattr(token, "profile_safe", True),
    }


def _norm(word: str) -> str:
    """Lowercase alphabetic only — strips punctuation, numbers, spaces."""
    return re.sub(r"[^a-z]", "", (word or "").lower())


def _strip_punct(word: str) -> str:
    """Strip leading/trailing punctuation, keep internal apostrophes/hyphens."""
    return re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", (word or ""))


def _duration(token: dict[str, Any]) -> float | None:
    try:
        start = token.get("start")
        end   = token.get("end")
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
        idx  = min(98, max(0, int(pct) - 1))
        return cuts[idx]
    except Exception:
        return max(values)


# ── Edit distance (near-repetition) ──────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 1.0 - _edit_distance(a, b) / max(len(a), len(b))


# ── Acoustic feature extraction ───────────────────────────────────────────────

def _load_wav_samples(audio_bytes: bytes) -> tuple[np.ndarray, int] | tuple[None, None]:
    try:
        with wave.open(BytesIO(audio_bytes), "rb") as wf:
            sr       = wf.getframerate()
            n_frames = wf.getnframes()
            n_ch     = wf.getnchannels()
            raw      = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        return samples, sr
    except Exception:
        return None, None


def _slice(samples: np.ndarray, sr: int,
           start: float | None, end: float | None,
           pad_s: float = 0.0) -> np.ndarray | None:
    if samples is None or start is None or end is None:
        return None
    i0 = max(0, int((start - pad_s) * sr))
    i1 = min(len(samples), int((end + pad_s) * sr))
    if i1 <= i0:
        return None
    return samples[i0:i1]


def _rms(chunk: np.ndarray | None) -> float:
    if chunk is None or len(chunk) == 0:
        return 0.0
    return float(np.sqrt(np.mean(chunk ** 2)))


def _zcr(chunk: np.ndarray | None) -> float:
    if chunk is None or len(chunk) < 2:
        return 0.5
    signs = np.sign(chunk)
    signs[signs == 0] = 1
    return float(np.sum(signs[:-1] != signs[1:])) / (len(chunk) - 1)


class _AcousticContext:
    """Per-clip acoustic feature cache. Built once, queried per-token."""

    def __init__(self, audio_bytes: bytes | None, cfg: dict[str, Any]):
        acoustic_cfg     = cfg.get("acoustic", {})
        self.silence_rms = float(acoustic_cfg.get("silence_rms_threshold", 0.015))
        self.voiced_rms  = float(acoustic_cfg.get("voiced_rms_threshold",  0.030))
        self.voiced_zcr  = float(acoustic_cfg.get("voiced_zcr_threshold",  0.15))
        self.samples: np.ndarray | None = None
        self.sr: int | None = None
        if audio_bytes:
            self.samples, self.sr = _load_wav_samples(audio_bytes)

    @property
    def available(self) -> bool:
        return self.samples is not None and self.sr is not None

    def gap_is_silent(self, gap_start: float, gap_end: float) -> bool:
        if not self.available:
            return True
        chunk = _slice(self.samples, self.sr, gap_start, gap_end)
        return _rms(chunk) < self.silence_rms

    def word_is_prolonged(self, start: float | None, end: float | None) -> bool:
        if not self.available:
            return True
        chunk = _slice(self.samples, self.sr, start, end)
        return _rms(chunk) >= self.voiced_rms and _zcr(chunk) <= self.voiced_zcr

    def word_rms(self, start: float | None, end: float | None) -> float:
        if not self.available or start is None or end is None:
            return 0.0
        return _rms(_slice(self.samples, self.sr, start, end))

    def word_zcr(self, start: float | None, end: float | None) -> float:
        if not self.available or start is None or end is None:
            return 0.5
        return _zcr(_slice(self.samples, self.sr, start, end))


# ── Sentence-boundary detection ───────────────────────────────────────────────

# Gaps this large between words are treated as sentence boundaries.
# Stuttering at sentence-initial position is clinically more significant.
_SENTENCE_BOUNDARY_GAP = 1.5  # seconds

def _sentence_initial_indices(rows: list[dict[str, Any]]) -> set[int]:
    """Return the set of token indices that start a new sentence.

    A sentence boundary is defined as:
      • The very first token (index 0).
      • Any token whose gap from the previous token is ≥ _SENTENCE_BOUNDARY_GAP.
    """
    result = {0}
    for i in range(1, len(rows)):
        prev_end = rows[i - 1].get("end")
        curr_start = rows[i].get("start")
        if prev_end is not None and curr_start is not None:
            try:
                if float(curr_start) - float(prev_end) >= _SENTENCE_BOUNDARY_GAP:
                    result.add(i)
            except Exception:
                pass
    return result


# ── Main detector ─────────────────────────────────────────────────────────────

def detect_disfluencies(
    tokens: Iterable[Any],
    config: dict[str, Any] | None = None,
    audio_bytes: bytes | None = None,
    speaker_baseline: "Any | None" = None,
) -> list[dict[str, Any]]:
    """Flag repetitions, prolongations, blocks, fillers, and ASR stutter marks.

    Parameters
    ----------
    tokens           : iterable of VerbatimToken or dict with word/start/end fields
    config           : optional profiling config dict (loaded from config.yaml if None)
    audio_bytes      : optional 16 kHz mono WAV bytes for acoustic validation.
    speaker_baseline : optional calibration.SpeakerBaseline. When provided and
                        usable, block_gap_seconds and prolongation_min_seconds
                        are personalized to the speaker's own calibrated tempo
                        (never below the config/global floor — see
                        calibration.adjusted_thresholds). Omit for the
                        original fixed-threshold behaviour.

    Returns
    -------
    Sorted list of event dicts: word, index, start, end, type, confidence, evidence.
    Optional extra fields: source, profile_safe, acoustic_rms, acoustic_zcr,
    sentence_initial.
    """
    rows = [_as_dict(t) for t in tokens]
    if not rows:
        return []

    cfg           = config or load_config().get("profiling", {}).get("detection", {})
    ac            = _AcousticContext(audio_bytes, cfg)

    filler_words   = set(cfg.get("filler_words", ["uh", "um", "er", "erm", "like"]))
    block_gap      = float(cfg.get("block_gap_seconds",           0.55))
    prolong_min    = float(cfg.get("prolongation_min_seconds",    0.65))
    prolong_pct    = float(cfg.get("prolongation_percentile",     90))

    # ── Personalize thresholds from a speaker's calibration baseline ──────────
    # Only ever raises a speaker's own bar above the global floor — never
    # lowers detection sensitivity below what an uncalibrated speaker gets.
    if speaker_baseline is not None and getattr(speaker_baseline, "is_usable", False):
        from .calibration import adjusted_thresholds
        adjusted = adjusted_thresholds(speaker_baseline, block_gap, prolong_min)
        block_gap = adjusted["block_gap_seconds"]
        prolong_min = adjusted["prolongation_min_seconds"]
    near_rep_sim   = float(cfg.get("near_repetition_similarity",  0.75))
    phrase_rep_len = int(  cfg.get("phrase_repetition_min_words", 2))
    # Confidence boost for sentence-initial disfluencies (clinically more
    # significant — stuttering almost always happens at word/sentence onset)
    sent_init_boost = float(cfg.get("sentence_initial_boost", 0.08))

    # ── Prolongation threshold ─────────────────────────────────────────────────
    # Guard: with < 5 tokens the 90th-percentile is meaningless (every word
    # looks prolonged relative to itself). Use 1.5× the absolute minimum
    # for short clips so we don't flag every single word.
    durations = [d for d in (_duration(t) for t in rows) if d is not None]
    if len(durations) >= 5:
        prolong_threshold = max(prolong_min, _percentile(durations, prolong_pct))
    else:
        prolong_threshold = prolong_min * 1.5

    # ── Pre-compute derived sequences once ────────────────────────────────────
    norms      = [_norm(str(r.get("word", ""))) for r in rows]
    sent_init  = _sentence_initial_indices(rows)

    # ── Phrase-repetition pre-pass ────────────────────────────────────────────
    phrase_rep_indices: set[int] = set()
    for wlen in range(phrase_rep_len, min(phrase_rep_len + 2, len(rows) + 1)):
        for i in range(wlen * 2, len(rows) + 1):
            seq_a = tuple(norms[i - wlen * 2 : i - wlen])
            seq_b = tuple(norms[i - wlen : i])
            if (
                len(seq_a) == wlen
                and seq_a == seq_b
                and all(s for s in seq_a)
            ):
                phrase_rep_indices.add(i - wlen)

    # ── Event accumulator ─────────────────────────────────────────────────────
    events: list[dict[str, Any]] = []
    seen:   set[tuple[int, str]] = set()

    def add(
        index: int,
        kind: str,
        confidence: float,
        evidence: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        key = (index, kind)
        if key in seen:
            return
        seen.add(key)
        is_sent_init = index in sent_init
        event: dict[str, Any] = {
            "word":             rows[index].get("word", ""),
            "index":            index,
            "start":            rows[index].get("start"),
            "end":              rows[index].get("end"),
            "type":             kind,
            "confidence":       round(min(0.99, confidence + (sent_init_boost if is_sent_init else 0.0)), 3),
            "evidence":         evidence + (" [sentence-initial]" if is_sent_init else ""),
            "sentence_initial": is_sent_init,
        }
        if rows[index].get("source"):
            event["source"] = rows[index].get("source")
        if rows[index].get("profile_safe") is False:
            event["profile_safe"] = False
        if extra:
            event.update(extra)
        events.append(event)

    # ── Per-token loop ────────────────────────────────────────────────────────
    for i, token in enumerate(rows):
        word  = str(token.get("word", ""))
        low   = norms[i]
        clean = _strip_punct(word)   # "recording." → "recording"
        if not low:
            continue

        # ── Filler ───────────────────────────────────────────────────────────
        if token.get("is_filler") or low in filler_words:
            add(i, "filler", 0.90, "ASR filler marker or known filler word")

        # ── Stutter marker ────────────────────────────────────────────────────
        if token.get("is_stutter") or word.endswith("-"):
            add(i, "stutter_marker", 0.85, "ASR stutter marker or trailing fragment")

        # ── Phrase repetition ─────────────────────────────────────────────────
        if i in phrase_rep_indices:
            add(i, "repetition", 0.88, f"phrase repeated starting at token {i}")

        if i > 0:
            prev_low  = norms[i - 1]
            prev_word = str(rows[i - 1].get("word", ""))

            # ── Exact back-to-back repetition ─────────────────────────────────
            if low and prev_low and low == prev_low:
                add(i, "repetition", 0.92, "same token repeated back-to-back")

            # ── Near-repetition ───────────────────────────────────────────────
            elif low and prev_low and len(low) >= 2 and len(prev_low) >= 2:
                sim = _similarity(low, prev_low)
                if sim >= near_rep_sim:
                    add(i, "repetition", round(0.75 * sim, 3),
                        f"near-repetition (similarity {sim:.2f}): "
                        f"'{prev_word}' → '{word}'")
                elif prev_word.endswith("-") and low.startswith(prev_low):
                    add(i, "repetition", 0.86, "sub-word fragment before this word")

            # ── Interjection-sandwiched repetition ("I uh I") ─────────────────
            # Pattern: token[i-2] == token[i] and token[i-1] is a filler.
            # The speaker said the word, stuttered into a filler, then
            # repeated the word — all three together form one event.
            if i >= 2:
                two_back_low = norms[i - 2]
                mid_low      = norms[i - 1]
                if (
                    low
                    and two_back_low
                    and low == two_back_low
                    and mid_low in filler_words
                ):
                    add(i, "repetition", 0.89,
                        f"filler-sandwiched repetition: "
                        f"'{rows[i-2].get('word','')}' + "
                        f"'{rows[i-1].get('word','')}' + '{word}'")

            # ── Block (with acoustic confirmation) ────────────────────────────
            prev_end  = rows[i - 1].get("end")
            curr_start = token.get("start")
            if prev_end is not None and curr_start is not None:
                try:
                    gap = float(curr_start) - float(prev_end)
                    if gap >= block_gap:
                        if ac.gap_is_silent(float(prev_end), float(curr_start)):
                            extra_fields: dict[str, Any] = {}
                            if ac.available:
                                rms_val = ac.word_rms(float(prev_end), float(curr_start))
                                extra_fields["acoustic_rms"] = round(float(rms_val), 5)
                                evidence = (
                                    f"silent gap {gap:.2f}s "
                                    f"(confirmed: RMS={rms_val:.4f})"
                                )
                            else:
                                evidence = f"silent gap {gap:.2f}s"
                            add(i, "block",
                                min(0.95, gap / max(block_gap, 0.01)),
                                evidence, extra_fields or None)
                except Exception:
                    pass

        # ── Prolongation (with acoustic confirmation + punctuation-aware) ─────
        # Use _norm-stripped low for filler check, but duration comes from
        # the raw timestamps — unaffected by punctuation.
        # clean_low strips punctuation for filler-word matching so "uh." isn't
        # missed as a filler and then accidentally flagged as prolongation too.
        clean_low = _norm(clean)
        dur = _duration(token)
        if (
            dur is not None
            and dur >= prolong_threshold
            and clean_low not in filler_words
            and low not in filler_words
        ):
            start_t = token.get("start")
            end_t   = token.get("end")
            if ac.word_is_prolonged(start_t, end_t):
                extra_fields = {}
                if ac.available:
                    rms_val = ac.word_rms(start_t, end_t)
                    zcr_val = ac.word_zcr(start_t, end_t)
                    extra_fields["acoustic_rms"] = round(float(rms_val), 5)
                    extra_fields["acoustic_zcr"] = round(float(zcr_val), 4)
                    evidence = (
                        f"duration {dur:.2f}s on '{clean}' "
                        f"(confirmed: RMS={rms_val:.4f}, ZCR={zcr_val:.3f})"
                    )
                else:
                    evidence = f"duration {dur:.2f}s on '{clean}'"
                add(i, "prolongation",
                    min(0.95, dur / max(prolong_threshold, 0.01)),
                    evidence, extra_fields or None)

    return sorted(events, key=lambda e: (e["index"], e["type"]))
