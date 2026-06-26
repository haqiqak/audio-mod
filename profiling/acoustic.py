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
        )


@dataclass
class Segment:
    start: float
    end: float
    voiced: bool
    rms: float
    zcr: float

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


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_voiced(samples: np.ndarray, sr: int, cfg: AcousticConfig) -> list[Segment]:
    """Merge consecutive frames of the same voiced/silent class into segments."""
    times, rms, zcr = frame_features(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
    if len(times) == 0:
        return []
    hop = max(1, int(sr * cfg.hop_seconds)) / sr
    voiced_flags = rms >= cfg.silence_rms

    segments: list[Segment] = []
    run_start = 0
    for i in range(1, len(times) + 1):
        at_end = i == len(times)
        if at_end or voiced_flags[i] != voiced_flags[run_start]:
            seg_rms = float(np.mean(rms[run_start:i]))
            seg_zcr = float(np.mean(zcr[run_start:i]))
            start_t = float(times[run_start])
            end_t = float(times[i - 1]) + hop
            segments.append(Segment(
                start=start_t, end=end_t,
                voiced=bool(voiced_flags[run_start]),
                rms=seg_rms, zcr=seg_zcr,
            ))
            run_start = i
    return segments


# ── Candidate derivation ──────────────────────────────────────────────────────

def detect_prolongations(segments: list[Segment], cfg: AcousticConfig) -> list[Candidate]:
    """Voiced segments that are long, energetic, and sustained (low ZCR)."""
    out: list[Candidate] = []
    for seg in segments:
        if (
            seg.voiced
            and seg.duration >= cfg.prolongation_min_seconds
            and seg.rms >= cfg.voiced_rms
            and seg.zcr <= cfg.voiced_zcr
        ):
            conf = min(0.95, seg.duration / max(cfg.prolongation_min_seconds, 0.01))
            out.append(Candidate(
                start=seg.start, end=seg.end, type="prolongation",
                confidence=conf,
                evidence=(
                    f"sustained voiced region {seg.duration:.2f}s "
                    f"(RMS={seg.rms:.4f}, ZCR={seg.zcr:.3f})"
                ),
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
