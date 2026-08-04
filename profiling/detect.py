"""Rule-based, audio-native-primary disfluency detection over verbatim ASR tokens.

Restructured (2026-08) so the audio signal is a first-class detector, not just
a confirmation filter bolted onto ASR-derived boundaries. Concretely:

1. Standard 5-class taxonomy
   The old generic "repetition" type is split into `sound_repetition` (a
   sub-word/fragment repeated, e.g. "b- buy"), `word_repetition` (a whole
   word repeated, exactly or near-exactly, including filler-sandwiched and
   phrase-adjacent single-word repeats), and `phrase_repetition` (an
   immediately-repeated multi-word phrase). Combined with the existing
   `block`, `prolongation`, and `filler` types, this now matches the field's
   standard taxonomy (SEP-28k / FluencyBank / KSoF: block, prolongation,
   sound repetition, word repetition, interjection) instead of an ad hoc set
   — the point being that output is now directly benchmarkable against public
   datasets, not just internally self-consistent.

2. Acoustic corroboration for filler / stutter-marker events
   These previously trusted the ASR's `is_filler`/`is_stutter` flags (or a
   trailing "-") with zero audio grounding — the weakest link in the
   pipeline, since even fine-tuned verbatim ASR models are known to mis-tag
   these tokens. When audio is available, a quick voiced-energy check now
   corroborates or down-weights these events, the same way block/prolongation
   have always been acoustically confirmed.

3. Weighted-confidence fusion (not fixed priority)
   The acoustic-native detector (profiling/acoustic.py) used to be dropped
   outright whenever it overlapped a token-path event of the same type —
   "token always wins," even if the token-derived confidence was weaker. It
   now only yields to the token-path event when that event's confidence is
   at least as high; a materially more confident acoustic-native candidate
   replaces the weaker token-path guess instead of being silently discarded.
   On a tie, the token-path event is kept deliberately: it carries word-level
   grounding (which word, which onset) that a signal-only candidate doesn't
   have on its own.

4. Config-driven detector enable-list
   `profiling.detection.detectors` in config.yaml lists which named checks
   run. Adding a future detector is a registration in that list plus one
   function, not a rewrite of this file's control flow.

All acoustic thresholds are configurable in config.yaml under
profiling.detection.acoustic.*. Every check here still degrades gracefully to
its original timestamp/text-only behaviour when audio_bytes is None — see
ARCHITECTURE.md for the regression-tested demo-fixture contract.
"""

from __future__ import annotations

import wave
from io import BytesIO
from statistics import quantiles
import re
from typing import Any, Iterable

import numpy as np

import phonetic

from .acoustic import _praat_features
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


def _phonetic_similarity(a: str, b: str) -> float | None:
    """Similarity of two words by ARPAbet phoneme edit distance, or None when
    either word is out-of-vocabulary (no CMU pronunciation). `_edit_distance`
    works on phoneme tuples as well as strings (it only compares elements)."""
    pa = phonetic.phonemes(a)
    pb = phonetic.phonemes(b)
    if not pa or not pb:
        return None
    return 1.0 - _edit_distance(pa, pb) / max(len(pa), len(pb))


def _word_repetition_extra(word: str) -> dict[str, Any]:
    """Descriptive syllable-count metadata for a `word_repetition` event,
    added 2026-08 per PHASE_2_RESEARCH_PLAN.md's literature review.

    The clinical stuttering-like-disfluency (SLD) vs. other-disfluency (OD)
    literature (Ambrose & Yairi's framework) treats a repeated monosyllabic
    word ("her-her-her") as motoric/stuttering-like, and a repeated
    polysyllabic word ("I see... I see her") as an ordinary linguistic-
    planning disfluency, not stuttering. This tag surfaces that distinction
    — it is a heuristic descriptive signal computed from syllable count
    alone, NOT a clinical diagnosis: the field's own severity instrument
    (SSI-3/4) only counts a monosyllabic repeat as stuttering when it also
    sounds perceptibly tense, which this project has no acoustic check for
    yet. No dataset this project benchmarks against labels this split, so
    it carries no accuracy claim — see PHASE_2_RESEARCH_PLAN.md section 5.
    """
    n = phonetic._syllable_count(word)
    return {"syllable_count": n, "likely_sld": n <= 1}


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
        # Same thresholds the acoustic-native fusion path (profiling/acoustic.py)
        # already uses for confidence adjustment — reused here, 2026-08-04,
        # VALIDATION.md section 9.5, as an optional HARD GATE for the
        # token-path prolongation check specifically (require_praat_
        # stability_for_prolongation). Not used unless that flag is on.
        self.pitch_std_max_hz = float(acoustic_cfg.get("pitch_std_max_hz", 25.0))
        self.jitter_max       = float(acoustic_cfg.get("jitter_max", 0.02))
        self.shimmer_max      = float(acoustic_cfg.get("shimmer_max", 0.08))
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

    def word_praat_stable(self, start: float | None, end: float | None) -> bool:
        """True if Praat pitch/jitter/shimmer are available AND all within
        the stable-voicing thresholds; True (graceful no-op, never blocks)
        when Praat is unavailable or the segment is too short/unvoiced for
        reliable tracking — same "None means no extra evidence, not a
        failure" principle as everywhere else Praat features are used in
        this codebase (see acoustic.py's _praat_features docstring). Added
        2026-08-04 (VALIDATION.md section 9.5) specifically so the
        token-path prolongation check can use this as a hard gate, not
        just a confidence adjustment (which is all the acoustic-native
        fusion path could ever do, per section 9.3's metric-blindness
        finding)."""
        if not self.available or start is None or end is None:
            return True
        feats = _praat_features(self.samples, self.sr, float(start), float(end))
        pitch_std, jitter, shimmer = feats["pitch_std_hz"], feats["jitter"], feats["shimmer"]
        if pitch_std is None:
            return True  # no reliable Praat read on this span -- don't block
        stable_pitch   = pitch_std <= self.pitch_std_max_hz
        stable_jitter  = jitter is None or jitter <= self.jitter_max
        stable_shimmer = shimmer is None or shimmer <= self.shimmer_max
        return stable_pitch and stable_jitter and stable_shimmer

    def has_voiced_energy(self, start: float | None, end: float | None) -> bool:
        """Cheap plausibility check for filler/stutter-marker corroboration:
        is there any real acoustic energy here at all, or is the ASR flag
        sitting on what's actually near-silence (a known ASR mistag/
        hallucination failure mode)? Deliberately simple — see module
        docstring point 2; a more elaborate offset-shape check for stutter
        fragments specifically is a documented future refinement, not done
        here without real recordings to validate it against."""
        if not self.available:
            return True  # no audio to check against -> don't penalize
        return self.word_rms(start, end) >= self.silence_rms

    def voiced_span(
        self, start: float | None, end: float | None, frame_s: float = 0.02,
    ) -> tuple[float, float] | None:
        """Absolute-time (start, end) of the voiced portion within [start, end],
        trimming leading/trailing silence by frame-wise RMS.

        This is the core of the word-timestamp / audio cross-check: ASR anchors a
        word's `start` to the chunk boundary, so clip-initial silence gets billed
        to the first word. Trimming silent edges recovers the word's real voiced
        extent. Only edges are trimmed (first..last voiced frame), so a brief
        mid-word dip doesn't shorten a genuinely sustained sound.

        Returns None when no audio is available or the span is empty; returns a
        zero-length span at `start` when the whole span is below silence.
        """
        if not self.available or start is None or end is None:
            return None
        s, e = float(start), float(end)
        i0 = max(0, int(s * self.sr))
        i1 = min(len(self.samples), int(e * self.sr))
        if i1 <= i0:
            return None
        chunk = self.samples[i0:i1]
        fr = max(1, int(self.sr * frame_s))
        n = len(chunk) // fr
        if n == 0:
            # Too short to frame — voiced iff the whole slice has energy.
            return (s, e) if _rms(chunk) >= self.silence_rms else (s, s)
        frames = chunk[: n * fr].reshape(n, fr)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        voiced = np.nonzero(frame_rms >= self.silence_rms)[0]
        if len(voiced) == 0:
            return (s, s)
        v0, v1 = int(voiced[0]), int(voiced[-1])
        vstart = s + (v0 * fr) / self.sr
        vend = s + ((v1 + 1) * fr) / self.sr
        return (min(e, vstart), min(e, vend))

    def voiced_duration(
        self, start: float | None, end: float | None,
    ) -> float | None:
        """Duration of the voiced portion within [start, end] (silent edges
        trimmed), or None if no audio is available."""
        span = self.voiced_span(start, end)
        if span is None:
            return None
        return max(0.0, span[1] - span[0])


# ── Sentence-boundary detection ───────────────────────────────────────────────

# Gaps this large between words are treated as sentence boundaries.
# Stuttering at sentence-initial position is clinically more significant.
_SENTENCE_BOUNDARY_GAP = 1.5  # seconds

def _spans_overlap(a0: Any, a1: Any, b0: float, b1: float) -> bool:
    """True if time span [a0,a1] overlaps [b0,b1] (a0/a1 may be None)."""
    if a0 is None or a1 is None:
        return False
    try:
        return min(float(a1), b1) > max(float(a0), b0)
    except (TypeError, ValueError):
        return False


def _token_index_for_span(rows: list[dict[str, Any]], start: float, end: float) -> int | None:
    """Token index best matching an acoustic time region: the token it overlaps
    most; failing any overlap (e.g. a silent block between words) the first token
    starting at/after the region; failing that the nearest by midpoint."""
    best_idx, best_ov = None, 0.0
    for i, r in enumerate(rows):
        s, e = r.get("start"), r.get("end")
        if s is None or e is None:
            continue
        try:
            ov = min(end, float(e)) - max(start, float(s))
        except (TypeError, ValueError):
            continue
        if ov > best_ov:
            best_ov, best_idx = ov, i
    if best_idx is not None and best_ov > 0:
        return best_idx
    after = [i for i, r in enumerate(rows)
             if r.get("start") is not None and _maybe_ge(r["start"], start)]
    if after:
        return after[0]
    mid = (start + end) / 2.0
    near = [(abs((float(r["start"]) + float(r["end"])) / 2.0 - mid), i)
            for i, r in enumerate(rows)
            if r.get("start") is not None and r.get("end") is not None]
    return min(near)[1] if near else None


def _maybe_ge(value: Any, threshold: float) -> bool:
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


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


# ── Phrase-repetition pre-pass ────────────────────────────────────────────────

def _find_phrase_repetitions(
    norms: list[str], phrase_rep_len: int, phrase_rep_max: int,
) -> dict[int, int]:
    """Start index of the 2nd occurrence of an immediately-repeated phrase ->
    phrase length (in words). Scans window lengths from phrase_rep_len up to
    min(phrase_rep_max, len(norms)//2) — a phrase can't repeat within fewer
    than twice its own length. Longest match wins per start index."""
    spans: dict[int, int] = {}
    upper = min(max(phrase_rep_len, phrase_rep_max), len(norms) // 2)
    for wlen in range(phrase_rep_len, upper + 1):
        for i in range(wlen * 2, len(norms) + 1):
            seq_a = tuple(norms[i - wlen * 2: i - wlen])
            seq_b = tuple(norms[i - wlen: i])
            if len(seq_a) == wlen and seq_a == seq_b and all(s for s in seq_a):
                start = i - wlen
                spans[start] = max(spans.get(start, 0), wlen)
    return spans


# ── Main detector ─────────────────────────────────────────────────────────────

_ALL_DETECTORS = (
    "filler", "stutter_marker", "phrase_repetition", "word_repetition",
    "sound_repetition", "block", "prolongation", "acoustic_fusion",
)


def detect_disfluencies(
    tokens: Iterable[Any],
    config: dict[str, Any] | None = None,
    audio_bytes: bytes | None = None,
    speaker_baseline: "Any | None" = None,
) -> list[dict[str, Any]]:
    """Flag disfluencies against the standard taxonomy: filler, stutter_marker,
    sound_repetition, word_repetition, phrase_repetition, block, prolongation.

    Parameters
    ----------
    tokens           : iterable of VerbatimToken or dict with word/start/end fields
    config           : optional profiling config dict (loaded from config.yaml if None)
    audio_bytes      : optional 16 kHz mono WAV bytes for acoustic validation/fusion.
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
    voiced_duration, sentence_initial, acoustic_corroborated.
    """
    rows = [_as_dict(t) for t in tokens]
    if not rows:
        return []

    cfg           = config or load_config().get("profiling", {}).get("detection", {})
    ac            = _AcousticContext(audio_bytes, cfg)

    enabled       = set(cfg.get("detectors", list(_ALL_DETECTORS)))
    fusion_weights = cfg.get("fusion_weights", {"rule": 1.0, "acoustic": 1.0})
    acoustic_weight = float(fusion_weights.get("acoustic", 1.0))

    filler_words   = set(cfg.get("filler_words", ["uh", "um", "er", "erm", "like"]))
    block_gap      = float(cfg.get("block_gap_seconds",           0.55))
    prolong_min    = float(cfg.get("prolongation_min_seconds",    1.0))
    prolong_pct    = float(cfg.get("prolongation_percentile",     90))
    use_rate_norm  = bool(cfg.get("use_rate_normalized_prolongation", False))
    rate_alpha     = float(cfg.get("prolongation_rate_alpha",     1.2))
    rate_floor     = float(cfg.get("prolongation_rate_floor",     1.5))
    require_praat_stable = bool(cfg.get("require_praat_stability_for_prolongation", False))

    # ── Personalize thresholds from a speaker's calibration baseline ──────────
    # Only ever raises a speaker's own bar above the global floor — never
    # lowers detection sensitivity below what an uncalibrated speaker gets.
    if speaker_baseline is not None and getattr(speaker_baseline, "is_usable", False):
        from .calibration import adjusted_thresholds
        adjusted = adjusted_thresholds(speaker_baseline, block_gap, prolong_min)
        block_gap = adjusted["block_gap_seconds"]
        prolong_min = adjusted["prolongation_min_seconds"]
    near_rep_sim   = float(cfg.get("near_repetition_similarity",  0.75))
    # For words this short or shorter, orthographic edit distance is noisy (one
    # changed letter is a huge % of a 2-3 letter word), so we compare ARPAbet
    # pronunciations instead — closer to what a stutter near-repeat actually is.
    # Longer (and out-of-vocabulary) words keep the spelling metric.
    phon_short_max = int(  cfg.get("phonetic_short_max_chars",    4))
    phrase_rep_len = int(  cfg.get("phrase_repetition_min_words", 2))
    # Confidence boost for sentence-initial disfluencies (clinically more
    # significant — stuttering almost always happens at word/sentence onset)
    sent_init_boost = float(cfg.get("sentence_initial_boost", 0.08))

    # ── Effective (voiced) duration ─────────────────────────────────────────────
    # When audio is available, a word's duration for prolongation purposes is its
    # VOICED extent (silent edges trimmed), not the raw ASR timestamp span. This
    # fixes two coupled failures from clip-initial silence the ASR bills to the
    # first word: (1) the word itself looking falsely prolonged, and (2) — just as
    # important — that inflated value entering the percentile below and raising the
    # bar so genuine prolongations elsewhere in the clip get suppressed. Without
    # audio this is identical to the raw timestamp duration (prior behaviour), so
    # fixtures and timestamp-only clips are unaffected.
    def _effective_duration(tok: dict[str, Any]) -> float | None:
        nominal = _duration(tok)
        if nominal is None:
            return None
        if ac.available:
            voiced = ac.voiced_duration(tok.get("start"), tok.get("end"))
            if voiced is not None:
                return voiced
        return nominal

    # ── Prolongation threshold ─────────────────────────────────────────────────
    if use_rate_norm:
        # Rate-normalized mechanism (2026-08-04, VALIDATION.md section 9.5):
        # T = rate_alpha / speaking_rate, the literature's standard technique
        # (Esmaili et al. 2017) — REPLACES the percentile mechanism below
        # when enabled, rather than combining with it, so the two are
        # cleanly comparable in the pre-registered ablation. Speaking rate
        # estimated as total syllables (phonetic._syllable_count, the same
        # function already used for the word_repetition SLD/OD tag) over
        # the clip's total time span; rate_floor guards against instability
        # on very short/sparse clips (a handful of words spanning a long
        # silence would otherwise imply an implausibly slow rate and an
        # implausibly high threshold).
        total_syllables = sum(
            phonetic._syllable_count(_norm(str(r.get("word", ""))))
            for r in rows if r.get("word")
        )
        starts = [r.get("start") for r in rows if r.get("start") is not None]
        ends   = [r.get("end")   for r in rows if r.get("end")   is not None]
        clip_span = (max(ends) - min(starts)) if (starts and ends) else None
        if clip_span and clip_span > 0 and total_syllables > 0:
            speaking_rate = total_syllables / clip_span
        else:
            speaking_rate = rate_floor  # no timing info -- fall back to the floor rate
        prolong_threshold = rate_alpha / max(speaking_rate, rate_floor)
    else:
        # Original mechanism: Guard: with < 5 tokens the 90th-percentile is
        # meaningless (every word looks prolonged relative to itself). Use
        # 1.5x the absolute minimum for short clips so we don't flag every
        # single word.
        durations = [d for d in (_effective_duration(t) for t in rows) if d is not None]
        if len(durations) >= 5:
            prolong_threshold = max(prolong_min, _percentile(durations, prolong_pct))
        else:
            prolong_threshold = prolong_min * 1.5

    # ── Pre-compute derived sequences once ────────────────────────────────────
    norms      = [_norm(str(r.get("word", ""))) for r in rows]
    sent_init  = _sentence_initial_indices(rows)

    phrase_rep_max = int(cfg.get("phrase_repetition_max_words", 8))
    phrase_rep_spans = (
        _find_phrase_repetitions(norms, phrase_rep_len, phrase_rep_max)
        if "phrase_repetition" in enabled else {}
    )

    # ── Event accumulator ─────────────────────────────────────────────────────
    events: list[dict[str, Any]] = []
    seen:   set[tuple[int, str]] = set()

    def add(
        index: int,
        kind: str,
        confidence: float,
        evidence: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        key = (index, kind)
        if key in seen:
            return None
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
        return event

    def discard(event: dict[str, Any]) -> None:
        """Remove a previously-added event so a stronger candidate for the
        same (index, type) can take its place — see the fusion note below."""
        try:
            events.remove(event)
        except ValueError:
            pass
        seen.discard((event["index"], event["type"]))

    # ── Per-token loop ────────────────────────────────────────────────────────
    for i, token in enumerate(rows):
        word  = str(token.get("word", ""))
        low   = norms[i]
        clean = _strip_punct(word)   # "recording." → "recording"
        if not low:
            continue

        # ── Filler (acoustically corroborated when audio is available) ───────
        if "filler" in enabled and (token.get("is_filler") or low in filler_words):
            conf, note = 0.90, "ASR filler marker or known filler word"
            if ac.available:
                rms_val = ac.word_rms(token.get("start"), token.get("end"))
                if rms_val >= ac.silence_rms:
                    conf = min(0.95, conf * 1.05)
                    note += f" (acoustic-confirmed: RMS={rms_val:.4f})"
                else:
                    conf *= 0.6
                    note += f" (acoustic check: near-silent, RMS={rms_val:.4f} — possible ASR misfire)"
            add(i, "filler", conf, note)

        # ── Stutter marker (same acoustic-plausibility corroboration) ────────
        if "stutter_marker" in enabled and (token.get("is_stutter") or word.endswith("-")):
            conf, note = 0.85, "ASR stutter marker or trailing fragment"
            if ac.available:
                rms_val = ac.word_rms(token.get("start"), token.get("end"))
                if rms_val >= ac.silence_rms:
                    conf = min(0.95, conf * 1.05)
                    note += f" (acoustic-confirmed: RMS={rms_val:.4f})"
                else:
                    conf *= 0.6
                    note += f" (acoustic check: near-silent, RMS={rms_val:.4f} — possible ASR misfire)"
            add(i, "stutter_marker", conf, note)

        # ── Phrase repetition ─────────────────────────────────────────────────
        if "phrase_repetition" in enabled and i in phrase_rep_spans:
            wlen = phrase_rep_spans[i]
            add(i, "phrase_repetition", 0.88,
                f"{wlen}-word phrase repeated starting at token {i}")

        if i > 0:
            prev_low  = norms[i - 1]
            prev_word = str(rows[i - 1].get("word", ""))

            # ── Sound-level fragment repeat (checked BEFORE word-level exact
            # match — see VALIDATION.md section 8.2's 2026-08-04 addendum).
            # A fragment reconstructed/transcribed as "word-" normalizes
            # (trailing "-" stripped by _norm) to the SAME string as its
            # complete-word counterpart, in EITHER order ("rachel- Rachel"
            # or "Rachel rachel-") — so this must run before the exact-match
            # check below, or that check intercepts both orderings and
            # sound_repetition can never fire for this reconstruction
            # pattern at all, regardless of which side the fragment is on.
            is_fragment_repeat = False
            fragment_first = False
            if ("sound_repetition" in enabled and low and prev_low
                    and len(low) >= 2 and len(prev_low) >= 2):
                if prev_word.endswith("-") and (low == prev_low or low.startswith(prev_low)):
                    is_fragment_repeat, fragment_first = True, True
                elif word.endswith("-") and (prev_low == low or prev_low.startswith(low)):
                    is_fragment_repeat, fragment_first = True, False

            if is_fragment_repeat:
                where = "before" if fragment_first else "after"
                add(i, "sound_repetition", 0.86,
                    f"sub-word fragment repeated {where} this word")

            # ── Exact back-to-back repetition (word-level) ────────────────────
            elif "word_repetition" in enabled and low and prev_low and low == prev_low:
                add(i, "word_repetition", 0.92, "same word repeated back-to-back",
                    _word_repetition_extra(clean))

            # ── Near-repetition (word-level) ──────────────────────────────────
            elif low and prev_low and len(low) >= 2 and len(prev_low) >= 2:
                # Short words: compare pronunciations (phonetic); longer/OOV
                # words: keep the spelling metric. See _phonetic_similarity.
                metric = "edit"
                sim: float | None = None
                if len(low) <= phon_short_max and len(prev_low) <= phon_short_max:
                    sim = _phonetic_similarity(low, prev_low)
                    if sim is not None:
                        metric = "phonetic"
                if sim is None:
                    sim = _similarity(low, prev_low)
                if "word_repetition" in enabled and sim >= near_rep_sim:
                    add(i, "word_repetition", round(0.75 * sim, 3),
                        f"near-repetition ({metric} similarity {sim:.2f}): "
                        f"'{prev_word}' → '{word}'",
                        _word_repetition_extra(clean))

            # ── Interjection-sandwiched repetition ("I uh I") ─────────────────
            # Pattern: token[i-2] == token[i] and token[i-1] is a filler.
            # The speaker said the word, stuttered into a filler, then
            # repeated the word — all three together form one word-level event.
            if "word_repetition" in enabled and i >= 2:
                two_back_low = norms[i - 2]
                mid_low      = norms[i - 1]
                if (
                    low
                    and two_back_low
                    and low == two_back_low
                    and mid_low in filler_words
                ):
                    add(i, "word_repetition", 0.89,
                        f"filler-sandwiched repetition: "
                        f"'{rows[i-2].get('word','')}' + "
                        f"'{rows[i-1].get('word','')}' + '{word}'",
                        _word_repetition_extra(clean))

                # NOTE: a "word-sandwiched repetition" extension (tolerating
                # a single non-filler word between a repeat pair) was
                # implemented and benchmarked 2026-08-04, then REVERTED —
                # measured net harm (Track A Any F1 0.835->0.793, +102 FP,
                # 0 new TP) far outweighed its measured benefit (Track B:
                # +1 TP at a cost of +24-29 FP). Do not re-add without new
                # evidence — full negative-result writeup in
                # VALIDATION.md section 8.4.4 and PAPER_DECISION_LOG.md.

            # ── Block (with acoustic confirmation) ────────────────────────────
            if "block" in enabled:
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
        if "prolongation" in enabled:
            clean_low = _norm(clean)
            dur = _effective_duration(token)
            if (
                dur is not None
                and dur >= prolong_threshold
                and clean_low not in filler_words
                and low not in filler_words
            ):
                start_t = token.get("start")
                end_t   = token.get("end")
                # Praat stability GATE (2026-08-04, VALIDATION.md section
                # 9.5) — optional, off by default. Unlike the acoustic-
                # native fusion path's confidence-only use of these same
                # features (section 9.3), this can reject a candidate the
                # RMS/ZCR/duration gate already passed. Graceful no-op
                # (True) when Praat is unavailable or require_praat_stable
                # is off, same as every other acoustic check here.
                praat_ok = (not require_praat_stable) or ac.word_praat_stable(start_t, end_t)
                if praat_ok and ac.word_is_prolonged(start_t, end_t):
                    extra_fields = {}
                    if ac.available:
                        rms_val = ac.word_rms(start_t, end_t)
                        zcr_val = ac.word_zcr(start_t, end_t)
                        extra_fields["acoustic_rms"] = round(float(rms_val), 5)
                        extra_fields["acoustic_zcr"] = round(float(zcr_val), 4)
                        extra_fields["voiced_duration"] = round(float(dur), 4)
                        evidence = (
                            f"voiced duration {dur:.2f}s on '{clean}' "
                            f"(confirmed: RMS={rms_val:.4f}, ZCR={zcr_val:.3f})"
                        )
                    else:
                        evidence = f"duration {dur:.2f}s on '{clean}'"
                    add(i, "prolongation",
                        min(0.95, dur / max(prolong_threshold, 0.01)),
                        evidence, extra_fields or None)

    # ── Acoustic fusion (only when we actually have the waveform) ────────────────
    # Cross-check with ASR-independent cues from profiling/acoustic.py: catch
    # prolongations/blocks the token path missed — e.g. a sustained sound that
    # falls in a gap with no token of its own, or one the ASR's word timestamps
    # under-shot. Each kept candidate is attributed to the best-matching token so
    # it carries a word/onset for the profile.
    #
    # Weighted-confidence fusion (not fixed priority): an acoustic candidate that
    # overlaps an existing event of the same type only replaces it when the
    # acoustic candidate is MORE confident (after fusion_weights.acoustic scaling)
    # than that event's own confidence. On a tie, the existing (token-path) event
    # is kept deliberately — it carries word-level grounding an audio-only
    # candidate doesn't have on its own. When it wins, the weaker event is
    # discarded and the acoustic one takes its place, so exactly one event
    # survives per (index, type) either way — no double counting.
    if "acoustic_fusion" in enabled and ac.available:
        from .acoustic import (
            AcousticConfig, detect_blocks, detect_prolongations, segment_voiced,
        )
        acfg = AcousticConfig.from_detection_cfg(cfg)
        acfg.prolongation_min_seconds = prolong_min   # honour calibrated floors
        acfg.block_min_seconds = block_gap
        segs = segment_voiced(ac.samples, ac.sr, acfg)
        for cand in detect_prolongations(segs, acfg) + detect_blocks(segs, acfg):
            weighted_conf = min(0.99, cand.confidence * acoustic_weight)
            overlapping = [
                ev for ev in events
                if ev["type"] == cand.type
                and _spans_overlap(ev.get("start"), ev.get("end"), cand.start, cand.end)
            ]
            if overlapping:
                best = max(overlapping, key=lambda e: e["confidence"])
                if weighted_conf <= best["confidence"]:
                    # Existing (token-path) event already at least as confident —
                    # annotate it with the acoustic corroboration, don't duplicate.
                    if not best.get("acoustic_corroborated"):
                        best["acoustic_corroborated"] = True
                        best["evidence"] += f" [acoustic corroboration: {cand.evidence}]"
                    continue
                for ev in overlapping:
                    discard(ev)

            idx = _token_index_for_span(rows, cand.start, cand.end)
            if idx is None:
                continue
            add(idx, cand.type, weighted_conf,
                f"[acoustic] {cand.evidence}",
                {"source": "acoustic",
                 "acoustic_start": round(cand.start, 3),
                 "acoustic_end": round(cand.end, 3)})

    return sorted(events, key=lambda e: (e["index"], e["type"]))
