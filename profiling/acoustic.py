"""acoustic.py — ASR-independent disfluency cues straight from the waveform.

The token-based detector in detect.py can only flag what CrisperWhisper puts in
the transcript, and it can only run after the (slow) ASR finishes. This module is
the foundation for the two things that need more than that:

  • **Research/accuracy:** catching disfluencies the ASR smooths away — a
    sub-word prolongation or a silent block that never became its own token.
  • **Realtime:** the benchmark showed transcription is inference-bound and
    ~5-13× slower than real time, so a realtime path can't wait on ASR. Acoustic
    cues (energy envelope, voicing) can be computed on an audio *stream* with no
    model at all, and only reconciled with a transcript later.

It segments a waveform into voiced/silent regions by frame energy, then derives:
  • **prolongation candidates** — long, sustained, low-ZCR voiced regions, and
  • **block candidates** — long silences *between* voiced regions (intra-speech).

Pure NumPy. No ASR, no model, no torch. Designed to run both on a whole clip and
(later) on a sliding window for streaming — the segmentation is windowable.

NOTE: this is intentionally NOT yet wired into detect_disfluencies. It's an
additive primitive; merging its candidates with the token-based detector (and
validating against real stutter recordings) is a deliberate next step recorded
in PAPER_DECISION_LOG.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import wave

import numpy as np

# ── Optional, pretrained corroborating signals ──────────────────────────────
# Both are additive: if either import fails (not installed) or errors out on a
# given clip, every function below degrades to the original RMS/ZCR-only
# behaviour. Neither is a hard dependency of this module.
try:
    import parselmouth  # praat-parselmouth: pitch/jitter/shimmer/HNR
    _HAS_PARSELMOUTH = True
except Exception:
    _HAS_PARSELMOUTH = False

try:
    import torch as _torch
    from silero_vad import get_speech_timestamps as _vad_get_speech_timestamps
    from silero_vad import load_silero_vad as _vad_load_model
    _HAS_SILERO = True
except Exception:
    _HAS_SILERO = False

_VAD_MODEL = None
_VAD_LOAD_FAILED = False


def _get_vad_model():
    """Lazy-load the Silero VAD model once per process. Returns None if the
    package isn't installed or the (one-time) load failed — callers must
    treat None as "VAD unavailable, fall back", never raise."""
    global _VAD_MODEL, _VAD_LOAD_FAILED
    if _VAD_MODEL is not None or _VAD_LOAD_FAILED or not _HAS_SILERO:
        return _VAD_MODEL
    try:
        _VAD_MODEL = _vad_load_model(onnx=False)
    except Exception:
        _VAD_LOAD_FAILED = True
    return _VAD_MODEL


# ── Tunables (mirror config.yaml's profiling.detection.* defaults) ─────────────

@dataclass
class AcousticConfig:
    silence_rms: float = 0.015          # below this a frame is "silence"
    voiced_rms: float = 0.030           # at/above this a region counts as voiced energy
    voiced_zcr: float = 0.15            # at/below this voicing is sustained (vowel-like)
    prolongation_min_seconds: float = 0.65
    block_min_seconds: float = 0.55
    frame_seconds: float = 0.025
    hop_seconds: float = 0.010
    # ── Corroborating-evidence tunables (additive, never make detection
    # stricter than the RMS/ZCR baseline when unavailable — see docstring) ──
    use_vad: bool = True                # gate voiced segments against Silero VAD when it fires on this clip
    vad_min_coverage: float = 0.30      # min fraction of a voiced segment VAD must confirm as speech
    jitter_max: float = 0.02            # local jitter (F0 period-to-period) below this = stable voicing
    shimmer_max: float = 0.08           # local shimmer (amplitude) below this = stable voicing
    pitch_std_max_hz: float = 25.0      # F0 std-dev within a segment below this = stable pitch (sustained vowel)
    min_praat_segment_seconds: float = 0.15  # too short for reliable pitch/jitter/shimmer tracking

    @classmethod
    def from_detection_cfg(cls, cfg: dict | None) -> "AcousticConfig":
        """Build from a profiling.detection config dict (so this module and the
        token detector stay in sync on thresholds)."""
        cfg = cfg or {}
        ac = cfg.get("acoustic", {})
        return cls(
            silence_rms=float(ac.get("silence_rms_threshold", 0.015)),
            voiced_rms=float(ac.get("voiced_rms_threshold", 0.030)),
            voiced_zcr=float(ac.get("voiced_zcr_threshold", 0.15)),
            prolongation_min_seconds=float(cfg.get("prolongation_min_seconds", 0.65)),
            block_min_seconds=float(cfg.get("block_gap_seconds", 0.55)),
            use_vad=bool(ac.get("use_vad", True)),
            vad_min_coverage=float(ac.get("vad_min_coverage", 0.30)),
            jitter_max=float(ac.get("jitter_max", 0.02)),
            shimmer_max=float(ac.get("shimmer_max", 0.08)),
            pitch_std_max_hz=float(ac.get("pitch_std_max_hz", 25.0)),
            min_praat_segment_seconds=float(ac.get("min_praat_segment_seconds", 0.15)),
        )


@dataclass
class Segment:
    start: float
    end: float
    voiced: bool
    rms: float
    zcr: float
    # ── Corroborating features, populated only for voiced segments long
    # enough to matter (see AcousticConfig.min_praat_segment_seconds) and
    # only when the optional pretrained signal is available. None means
    # "no evidence either way" — callers must never treat None as failing
    # a check, only as "nothing extra to corroborate with" (see
    # detect_prolongations). ──
    vad_coverage: float | None = None   # fraction of this segment VAD confirms as real speech
    pitch_hz: float | None = None       # mean voiced F0 (praat)
    pitch_std_hz: float | None = None   # F0 std-dev within the segment (stability)
    jitter: float | None = None         # local jitter (praat)
    shimmer: float | None = None        # local shimmer (praat)
    hnr: float | None = None            # mean harmonics-to-noise ratio, dB (praat)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Candidate:
    start: float
    end: float
    type: str          # "prolongation" | "block"
    confidence: float
    evidence: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "source": "acoustic",
        }


@dataclass
class AcousticAnalysis:
    segments: list[Segment] = field(default_factory=list)
    prolongations: list[Candidate] = field(default_factory=list)
    blocks: list[Candidate] = field(default_factory=list)

    @property
    def candidates(self) -> list[Candidate]:
        return sorted(self.prolongations + self.blocks, key=lambda c: c.start)


# ── WAV loading (standalone — no dependency on detect.py) ──────────────────────

def load_wav_samples(audio_bytes: bytes) -> tuple[np.ndarray | None, int | None]:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_ch = wf.getnchannels()
            raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        return samples, sr
    except Exception:
        return None, None


# ── Frame-level features ──────────────────────────────────────────────────────

def frame_features(
    samples: np.ndarray, sr: int, frame_s: float, hop_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (time, rms, zcr) arrays. Time is the frame's start in seconds."""
    frame = max(1, int(sr * frame_s))
    hop = max(1, int(sr * hop_s))
    n = len(samples)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty

    starts = np.arange(0, max(1, n - frame + 1), hop)
    times = starts / sr
    rms = np.empty(len(starts), dtype=np.float32)
    zcr = np.empty(len(starts), dtype=np.float32)
    for k, s0 in enumerate(starts):
        chunk = samples[s0:s0 + frame]
        rms[k] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0.0
        if len(chunk) >= 2:
            signs = np.sign(chunk)
            signs[signs == 0] = 1
            zcr[k] = np.sum(signs[:-1] != signs[1:]) / (len(chunk) - 1)
        else:
            zcr[k] = 0.0
    return times, rms, zcr


# ── Corroborating signals: Silero VAD (voicing) + Praat (voice quality) ────────
#
# Both are deliberately *additive* evidence layered on top of the original
# RMS/ZCR segmentation, not a replacement for it. Silero VAD is trained on
# real speech and correctly reports "no speech" on non-speech signals (e.g.
# a synthetic sine tone, or the tone+silence fixtures this module's own test
# suite is built from) — so gating segmentation on VAD alone would silently
# break every synthetic-audio test and the project's model-free testing
# philosophy. Instead: VAD only *gates/down-weights* a candidate when it (a)
# is available, AND (b) found real speech somewhere in this clip at all (i.e.
# it's actually applicable to this signal) — on a clip where VAD finds
# nothing anywhere (pure tones, synthetic fixtures), it silently opts out and
# behaviour is identical to the original RMS/ZCR-only logic. Praat pitch/
# jitter/shimmer degrade the same way: None on failure/unavailability, never
# used to fail a check that would otherwise pass.

def vad_speech_ranges(samples: np.ndarray, sr: int) -> list[tuple[float, float]] | None:
    """Real-speech time ranges per Silero VAD, or None if VAD is unavailable
    or errored on this clip (caller must fall back, not fail)."""
    if not _HAS_SILERO:
        return None
    model = _get_vad_model()
    if model is None:
        return None
    try:
        audio = _torch.from_numpy(np.asarray(samples, dtype=np.float32))
        spans = _vad_get_speech_timestamps(
            audio, model, sampling_rate=sr, return_seconds=True,
        )
        return [(float(s["start"]), float(s["end"])) for s in spans]
    except Exception:
        return None


def _range_coverage(start: float, end: float, ranges: list[tuple[float, float]]) -> float:
    """Fraction of [start,end] covered by the union of `ranges`."""
    dur = max(0.0, end - start)
    if dur <= 0 or not ranges:
        return 0.0
    covered = 0.0
    for r0, r1 in ranges:
        ov = min(end, r1) - max(start, r0)
        if ov > 0:
            covered += ov
    return min(1.0, covered / dur)


def _praat_features(
    samples: np.ndarray, sr: int, start: float, end: float,
) -> dict[str, float | None]:
    """Mean/std F0, local jitter, local shimmer, and mean HNR for one segment
    via Praat (through parselmouth). Returns all-None on failure or when the
    segment is too short/unvoiced for reliable pronunciation-quality tracking
    — callers must treat None as "no extra evidence", never as a failed check."""
    empty = {"pitch_hz": None, "pitch_std_hz": None, "jitter": None, "shimmer": None, "hnr": None}
    if not _HAS_PARSELMOUTH:
        return empty
    try:
        i0 = max(0, int(start * sr))
        i1 = min(len(samples), int(end * sr))
        if i1 <= i0:
            return empty
        chunk = np.asarray(samples[i0:i1], dtype=np.float64)
        snd = parselmouth.Sound(chunk, sampling_frequency=sr)

        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        pitch_hz = float(np.mean(f0)) if len(f0) else None
        pitch_std_hz = float(np.std(f0)) if len(f0) else None

        harm = snd.to_harmonicity()
        hv = harm.values[np.isfinite(harm.values) & (harm.values > -200)]
        hnr = float(np.mean(hv)) if hv.size else None

        jitter = shimmer = None
        try:
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            j = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            s = parselmouth.praat.call(
                [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
            jitter = float(j) if j == j else None      # NaN check (unvoiced/aperiodic)
            shimmer = float(s) if s == s else None
        except Exception:
            pass

        return {
            "pitch_hz": pitch_hz, "pitch_std_hz": pitch_std_hz,
            "jitter": jitter, "shimmer": shimmer, "hnr": hnr,
        }
    except Exception:
        return empty


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_voiced(samples: np.ndarray, sr: int, cfg: AcousticConfig) -> list[Segment]:
    """Merge consecutive frames of the same voiced/silent class into segments.

    Boundary detection itself stays pure RMS/ZCR (unchanged, and exactly what
    the existing synthetic-tone test suite validates). Voiced segments long
    enough to matter are then *annotated* with VAD coverage and praat voice-
    quality features as additional, optional evidence — see the module-level
    note above for why this is additive rather than a hard replacement.
    """
    times, rms, zcr = frame_features(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
    if len(times) == 0:
        return []
    hop = max(1, int(sr * cfg.hop_seconds)) / sr
    voiced_flags = rms >= cfg.silence_rms

    vad_ranges = vad_speech_ranges(samples, sr) if cfg.use_vad else None
    vad_applicable = bool(vad_ranges)  # False if unavailable OR found nothing anywhere in this clip

    segments: list[Segment] = []
    run_start = 0
    for i in range(1, len(times) + 1):
        at_end = i == len(times)
        if at_end or voiced_flags[i] != voiced_flags[run_start]:
            seg_rms = float(np.mean(rms[run_start:i]))
            seg_zcr = float(np.mean(zcr[run_start:i]))
            start_t = float(times[run_start])
            end_t = float(times[i - 1]) + hop
            seg_voiced = bool(voiced_flags[run_start])

            vad_coverage = None
            pitch_hz = pitch_std_hz = jitter = shimmer = hnr = None
            if seg_voiced and (end_t - start_t) >= cfg.min_praat_segment_seconds:
                if vad_applicable:
                    vad_coverage = _range_coverage(start_t, end_t, vad_ranges)
                feats = _praat_features(samples, sr, start_t, end_t)
                pitch_hz, pitch_std_hz = feats["pitch_hz"], feats["pitch_std_hz"]
                jitter, shimmer, hnr = feats["jitter"], feats["shimmer"], feats["hnr"]

            segments.append(Segment(
                start=start_t, end=end_t, voiced=seg_voiced,
                rms=seg_rms, zcr=seg_zcr,
                vad_coverage=vad_coverage,
                pitch_hz=pitch_hz, pitch_std_hz=pitch_std_hz,
                jitter=jitter, shimmer=shimmer, hnr=hnr,
            ))
            run_start = i
    return segments


# ── Candidate derivation ──────────────────────────────────────────────────────

def detect_prolongations(segments: list[Segment], cfg: AcousticConfig) -> list[Candidate]:
    """Voiced segments that are long, energetic, and sustained (low ZCR).

    Corroborating evidence (VAD coverage, pitch stability, jitter, shimmer —
    see the module-level note above `segment_voiced`) adjusts confidence
    up or down when available, but the core RMS/ZCR/duration gate below is
    unchanged: with no corroborating evidence (VAD/praat unavailable, or a
    non-speech test signal VAD legitimately found nothing in) this behaves
    exactly as before.
    """
    out: list[Candidate] = []
    for seg in segments:
        if not (
            seg.voiced
            and seg.duration >= cfg.prolongation_min_seconds
            and seg.rms >= cfg.voiced_rms
            and seg.zcr <= cfg.voiced_zcr
        ):
            continue

        conf = min(0.95, seg.duration / max(cfg.prolongation_min_seconds, 0.01))
        notes = [f"sustained voiced region {seg.duration:.2f}s (RMS={seg.rms:.4f}, ZCR={seg.zcr:.3f})"]

        if seg.vad_coverage is not None:
            if seg.vad_coverage < cfg.vad_min_coverage:
                conf *= 0.6   # down-weight, don't hard-reject — RMS/ZCR already passed
                notes.append(f"low VAD speech coverage ({seg.vad_coverage:.0%})")
            else:
                notes.append(f"VAD-confirmed speech ({seg.vad_coverage:.0%})")

        stable_pitch = seg.pitch_std_hz is not None and seg.pitch_std_hz <= cfg.pitch_std_max_hz
        stable_jitter = seg.jitter is None or seg.jitter <= cfg.jitter_max
        stable_shimmer = seg.shimmer is None or seg.shimmer <= cfg.shimmer_max
        if seg.pitch_std_hz is not None:
            if stable_pitch and stable_jitter and stable_shimmer:
                conf = min(0.95, conf * 1.08)
                notes.append(
                    f"stable pitch (F0≈{seg.pitch_hz:.0f}Hz, "
                    f"σ={seg.pitch_std_hz:.1f}Hz, jitter/shimmer within range)"
                )
            elif not stable_pitch:
                conf *= 0.85
                notes.append(f"unstable pitch (σ={seg.pitch_std_hz:.1f}Hz)")

        conf = round(min(0.95, max(0.05, conf)), 4)
        out.append(Candidate(
            start=seg.start, end=seg.end, type="prolongation",
            confidence=conf, evidence="; ".join(notes),
        ))
    return out


def detect_blocks(segments: list[Segment], cfg: AcousticConfig) -> list[Candidate]:
    """Silences long enough to be a block, *flanked by voiced segments on both
    sides* — i.e. intra-speech silence, not leading/trailing dead air."""
    out: list[Candidate] = []
    for idx in range(1, len(segments) - 1):
        seg = segments[idx]
        if (
            not seg.voiced
            and seg.duration >= cfg.block_min_seconds
            and segments[idx - 1].voiced
            and segments[idx + 1].voiced
        ):
            conf = min(0.95, seg.duration / max(cfg.block_min_seconds, 0.01))
            out.append(Candidate(
                start=seg.start, end=seg.end, type="block",
                confidence=conf,
                evidence=f"silent gap {seg.duration:.2f}s between voiced regions",
            ))
    return out


def analyze(
    audio: bytes | np.ndarray,
    sr: int | None = None,
    config: dict | None = None,
) -> AcousticAnalysis:
    """Full pipeline: load (if bytes) → segment → derive prolongation/block cues.

    `audio` may be WAV bytes (sr ignored, read from header) or a mono float32
    NumPy array (sr required). `config` is a profiling.detection config dict.
    """
    cfg = AcousticConfig.from_detection_cfg(config)
    if isinstance(audio, (bytes, bytearray)):
        samples, sr = load_wav_samples(bytes(audio))
    else:
        samples = np.asarray(audio, dtype=np.float32)
    if samples is None or sr is None or len(samples) == 0:
        return AcousticAnalysis()

    segments = segment_voiced(samples, sr, cfg)
    return AcousticAnalysis(
        segments=segments,
        prolongations=detect_prolongations(segments, cfg),
        blocks=detect_blocks(segments, cfg),
    )
