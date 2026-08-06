"""stage_g_acoustic_sound_repetition.py — ASR_RESEARCH_TRACK.md direction (g).

Implements the protocol pre-registered in ASR_RESEARCH_TRACK.md's
"Direction (g): an acoustic-native sound_repetition candidate generator"
section (and its "MFCC escalation" addendum) EXACTLY - read those
sections before changing any logic here. Tests whether sound_repetition
has a recoverable acoustic signature - 2+ short, spectrally self-similar
voiced bursts in immediate succession - detectable directly from the
waveform, with NO ASR involved anywhere in this script (Track-A-style:
ground truth timestamps come straight from LibriStutter's own labels,
not from any hypothesis alignment).

Two per-burst similarity features are implemented, selected via
--feature: "rmszcr" (default, the cheap first pass - RMS/ZCR envelope
shape) and "mfcc" (the pre-registered escalation - spectral shape via a
hand-rolled MFCC extractor, coefficient 0 excluded, see
_burst_similarity_mfcc()'s docstring for why). Candidate run-detection
(generate_candidates()) is identical between both - only the similarity
feature passed to it differs - so any result difference between the two
is attributable to the feature alone. Both were run against the same
120-clip sample; both came back Failure - see ASR_RESEARCH_TRACK.md's
"Direction (g) results" and "MFCC escalation results" for the numbers.

Reuses profiling.acoustic's existing segment_voiced() (the same RMS/ZCR
segmentation block/prolongation already build on) and frame_features()
unmodified - only the candidate-generation and scoring logic here is new,
and neither touches profiling/acoustic.py or profiling/detect.py, matching
the pre-registration's "new function, not touching detect_prolongations/
detect_blocks."

Usage
-----
    python -m profiling.evaluation.stage_g_acoustic_sound_repetition \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --feature rmszcr  # or --feature mfcc

    python -m profiling.evaluation.stage_g_acoustic_sound_repetition --self-test
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 -- must precede torch/transformers imports (kept for convention; no torch used here)

import numpy as np

from profiling.acoustic import AcousticConfig, Segment, frame_features, load_wav_samples, segment_voiced
from profiling.config import load_config
from profiling.evaluation.loaders import load_libristutter_dir_with_audio

TARGET_TYPE = "sound_repetition"

# Pre-registered starting thresholds (ASR_RESEARCH_TRACK.md "Direction (g)")
# - starting points, not tuned against results (rule 4).
SHORT_BURST_MAX_SECONDS = 0.35   # a "short burst" candidate segment must be under this
MAX_GAP_SECONDS = 0.20           # tolerate this much silence between bursts in one run
MATCH_TOLERANCE_SECONDS = 0.20   # +/- padding around ground-truth span for a match
SIMILARITY_N_BINS = 8            # fixed-length resampled feature vector size per burst
SIMILARITY_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


@dataclass
class AcousticCandidate:
    start: float
    end: float
    n_bursts: int
    similarities: list[float]   # pairwise similarity, consecutive bursts in the run
    followed_by_longer: bool

    @property
    def mean_similarity(self) -> float:
        return sum(self.similarities) / len(self.similarities) if self.similarities else 0.0


# ── Candidate generation (new; does not touch profiling/acoustic.py) ───────

def _voiced_runs_with_short_gaps(segments: list[Segment], max_gap_s: float) -> list[list[Segment]]:
    """Groups voiced segments into runs, tolerating short silent gaps
    between them (segment_voiced() strictly alternates voiced/silent, so
    the gap between two voiced segments is exactly the intervening silent
    segment's duration - no need to look at the silent segments directly)."""
    voiced = [s for s in segments if s.voiced]
    runs: list[list[Segment]] = []
    current: list[Segment] = []
    for seg in voiced:
        if current and (seg.start - current[-1].end) > max_gap_s:
            runs.append(current)
            current = []
        current.append(seg)
    if current:
        runs.append(current)
    return runs


def _resampled_vector(times: np.ndarray, values: np.ndarray, start: float, end: float, n_bins: int) -> np.ndarray | None:
    mask = (times >= start) & (times < end)
    v = values[mask]
    if len(v) == 0:
        return None
    if len(v) == 1:
        return np.full(n_bins, v[0], dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(v))
    x_new = np.linspace(0.0, 1.0, n_bins)
    return np.interp(x_new, x_old, v)


def _burst_similarity(
    seg_a: Segment, seg_b: Segment, frame_times: np.ndarray, frame_rms: np.ndarray, frame_zcr: np.ndarray,
    n_bins: int = SIMILARITY_N_BINS,
) -> float | None:
    """Cosine similarity between two bursts' RMS+ZCR envelope shape - the
    cheapest signal already available in this codebase's dependency set
    (frame_features(), the same primitive segment_voiced() itself uses),
    per the pre-registration's 'start cheap' instruction."""
    ra = _resampled_vector(frame_times, frame_rms, seg_a.start, seg_a.end, n_bins)
    rb = _resampled_vector(frame_times, frame_rms, seg_b.start, seg_b.end, n_bins)
    za = _resampled_vector(frame_times, frame_zcr, seg_a.start, seg_a.end, n_bins)
    zb = _resampled_vector(frame_times, frame_zcr, seg_b.start, seg_b.end, n_bins)
    if ra is None or rb is None:
        return None

    def _norm(x: np.ndarray | None) -> np.ndarray:
        if x is None:
            return np.zeros(n_bins)
        n = np.linalg.norm(x)
        return x / n if n > 0 else x

    va = np.concatenate([_norm(ra), _norm(za)])
    vb = np.concatenate([_norm(rb), _norm(zb)])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


# ── MFCC escalation (ASR_RESEARCH_TRACK.md "Direction (g), MFCC
# escalation" addendum) - hand-rolled, no new dependency (librosa is not
# installed in this project's environment; scipy is already a transitive
# dependency and provides the DCT primitive). Standard pipeline: windowed
# FFT -> power spectrum -> mel filterbank -> log -> DCT-II. ──

MFCC_N_MELS = 26
MFCC_N_COEFF = 13
MFCC_FMIN_HZ = 20.0


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sr: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Standard triangular mel filterbank, shape [n_mels, n_fft//2 + 1]."""
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / (right - center)
    return fb


def compute_mfcc(
    samples: np.ndarray, sr: int, frame_s: float, hop_s: float,
    n_mels: int = MFCC_N_MELS, n_mfcc: int = MFCC_N_COEFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (frame_start_times, mfcc[n_frames, n_mfcc]). Frame/hop match
    AcousticConfig's own frame_seconds/hop_seconds so this lines up with
    the same frame grid frame_features() already uses."""
    from scipy.fftpack import dct

    frame_len = max(1, int(sr * frame_s))
    hop_len = max(1, int(sr * hop_s))
    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    n = len(samples)
    if n < frame_len:
        return np.array([]), np.zeros((0, n_mfcc))

    starts = np.arange(0, max(1, n - frame_len + 1), hop_len)
    window = np.hanning(frame_len)
    fb = _mel_filterbank(n_fft, sr, n_mels, MFCC_FMIN_HZ, sr / 2.0)

    mfcc = np.empty((len(starts), n_mfcc), dtype=np.float64)
    for k, s0 in enumerate(starts):
        chunk = samples[s0:s0 + frame_len] * window
        spectrum = np.fft.rfft(chunk, n=n_fft)
        power = (np.abs(spectrum) ** 2) / n_fft
        mel_energies = fb @ power
        log_mel = np.log(mel_energies + 1e-10)
        coeffs = dct(log_mel, type=2, norm="ortho")
        mfcc[k] = coeffs[:n_mfcc]

    times = starts / sr
    return times, mfcc


def _resampled_mfcc_vector(times: np.ndarray, mfcc: np.ndarray, start: float, end: float) -> np.ndarray | None:
    """Mean MFCC vector across the frames overlapping [start, end) - a
    single spectral-shape summary per burst, replacing the RMS/ZCR
    resampled-envelope vector."""
    mask = (times >= start) & (times < end)
    if not np.any(mask):
        return None
    return mfcc[mask].mean(axis=0)


def _burst_similarity_mfcc(seg_a: Segment, seg_b: Segment, times: np.ndarray, mfcc: np.ndarray) -> float | None:
    """Cosine similarity between two bursts' mean MFCC vectors, EXCLUDING
    coefficient 0 (overall log-energy) - standard MFCC practice, and a real
    bug caught here before trusting the first real run: coefficient 0
    dominates the vector's norm, so any two voiced (energetic) segments
    score >=0.9 similarity regardless of spectral shape (measured directly:
    90% of all burst pairs in the sample scored >=0.9 including c0, vs.
    22% excluding it, with a real spread - mean 0.961 -> 0.433). Including
    it made this feature measure 'is this voiced speech' rather than 'does
    this burst sound like that burst', silently defeating the point of
    escalating past the RMS/ZCR envelope feature."""
    va = _resampled_mfcc_vector(times, mfcc, seg_a.start, seg_a.end)
    vb = _resampled_mfcc_vector(times, mfcc, seg_b.start, seg_b.end)
    if va is None or vb is None:
        return None
    va, vb = va[1:], vb[1:]
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _mfcc_similarity_fn(times: np.ndarray, mfcc: np.ndarray):
    return lambda a, b: _burst_similarity_mfcc(a, b, times, mfcc)


def _rmszcr_similarity_fn(frame_times: np.ndarray, frame_rms: np.ndarray, frame_zcr: np.ndarray):
    return lambda a, b: _burst_similarity(a, b, frame_times, frame_rms, frame_zcr)


def generate_candidates(
    segments: list[Segment], frame_times: np.ndarray, frame_rms: np.ndarray, frame_zcr: np.ndarray,
    short_max_s: float = SHORT_BURST_MAX_SECONDS, max_gap_s: float = MAX_GAP_SECONDS,
    similarity_fn=None,
) -> list[AcousticCandidate]:
    """Every maximal run of >=2 consecutive short voiced bursts (short-gap
    tolerant). Similarity is NOT gated here - every qualifying run becomes
    a candidate, carrying its own similarity scores, so scoring can compare
    the similarity-gated mechanism against the duration-only baseline
    (every candidate here, ungated) without regenerating candidates twice.

    `similarity_fn(seg_a, seg_b) -> float | None` is pluggable so the MFCC
    escalation (ASR_RESEARCH_TRACK.md "Direction (g), MFCC escalation")
    can reuse this exact run-detection logic unmodified, swapping only the
    per-burst feature - any difference in result is then attributable to
    the feature, not a confounded re-design. Defaults to the original
    RMS/ZCR envelope-shape similarity."""
    if similarity_fn is None:
        similarity_fn = _rmszcr_similarity_fn(frame_times, frame_rms, frame_zcr)
    runs = _voiced_runs_with_short_gaps(segments, max_gap_s)
    candidates: list[AcousticCandidate] = []
    for run in runs:
        i = 0
        while i < len(run):
            if run[i].duration >= short_max_s:
                i += 1
                continue
            j = i
            sims: list[float] = []
            while j + 1 < len(run) and run[j + 1].duration < short_max_s:
                sim = similarity_fn(run[j], run[j + 1])
                sims.append(sim if sim is not None else 0.0)
                j += 1
            n_bursts = j - i + 1
            if n_bursts >= 2:
                followed_by_longer = j + 1 < len(run) and run[j + 1].duration >= short_max_s
                candidates.append(AcousticCandidate(
                    start=run[i].start, end=run[j].end, n_bursts=n_bursts,
                    similarities=sims, followed_by_longer=followed_by_longer,
                ))
                i = j + 1
            else:
                i += 1
    return candidates


# ── Scoring (Track-A-style: ground truth timestamps, no ASR) ───────────────

def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def score(
    per_clip_targets_and_candidates: list[tuple[list[tuple[float, float]], list[AcousticCandidate]]],
    tol: float = MATCH_TOLERANCE_SECONDS,
) -> dict:
    """Scores strictly WITHIN each clip - never matches a candidate in one
    clip against a target's raw timestamp in a different clip. This is not
    optional bookkeeping: every clip's timeline independently starts at 0,
    clips are ~10-15s each, and pooling raw timestamps globally (an earlier,
    now-fixed version of this function did exactly that) produces massive
    spurious cross-clip matches purely from second-offset coincidence -
    caught by rule 3's "audit surprising results" discipline when the first
    real run returned an implausible recall=1.000/precision=0.966."""
    matched_targets = 0
    tp_candidates = 0
    n_targets = 0
    n_candidates = 0
    for targets, candidates in per_clip_targets_and_candidates:
        n_targets += len(targets)
        n_candidates += len(candidates)
        for t_start, t_end in targets:
            if any(_overlaps(c.start, c.end, t_start - tol, t_end + tol) for c in candidates):
                matched_targets += 1
        for c in candidates:
            if any(_overlaps(c.start, c.end, t_start - tol, t_end + tol) for t_start, t_end in targets):
                tp_candidates += 1
    recall = matched_targets / n_targets if n_targets else 0.0
    precision = tp_candidates / n_candidates if n_candidates else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_targets": n_targets, "n_candidates": n_candidates,
        "matched_targets": matched_targets, "tp_candidates": tp_candidates,
        "recall": recall, "precision": precision, "f1": f1,
    }


def run(data_dir: Path, audio_dir: Path, n_clips: int = 120, feature: str = "rmszcr") -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    print(f"Feature: {feature}\n")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None][:n_clips]
    print(f"{len(clips)} clips.\n")

    ac_cfg_dict = dict(load_config().get("profiling", {}).get("detection", {}))
    cfg = AcousticConfig.from_detection_cfg(ac_cfg_dict)

    per_clip: list[tuple[list[tuple[float, float]], list[AcousticCandidate]]] = []
    n_clips_with_target = 0
    n_targets_total = 0
    n_candidates_total = 0
    t0 = time.time()

    for i, clip in enumerate(clips):
        targets = [
            (clip.tokens[ref_idx]["start"], clip.tokens[ref_idx]["end"])
            for ref_idx, t in clip.ground_truth.items() if t == TARGET_TYPE
        ]
        if targets:
            n_clips_with_target += 1
        n_targets_total += len(targets)

        samples, sr = load_wav_samples(clip.audio_bytes)
        if samples is None:
            per_clip.append((targets, []))
            continue
        segments = segment_voiced(samples, sr, cfg)
        if feature == "mfcc":
            mfcc_times, mfcc = compute_mfcc(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
            sim_fn = _mfcc_similarity_fn(mfcc_times, mfcc)
            frame_times, frame_rms, frame_zcr = frame_features(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
            cands = generate_candidates(segments, frame_times, frame_rms, frame_zcr, similarity_fn=sim_fn)
        else:
            frame_times, frame_rms, frame_zcr = frame_features(samples, sr, cfg.frame_seconds, cfg.hop_seconds)
            cands = generate_candidates(segments, frame_times, frame_rms, frame_zcr)
        n_candidates_total += len(cands)
        per_clip.append((targets, cands))

        if (i + 1) % 50 == 0 or i + 1 == len(clips):
            print(f"[{i+1}/{len(clips)}] ... ({time.time()-t0:.0f}s elapsed)")

    print(f"\nTotal time: {time.time()-t0:.0f}s for {len(clips)} clips.")
    print(f"{n_targets_total} ground-truth {TARGET_TYPE} instances across {n_clips_with_target} clips.")
    print(f"{n_candidates_total} total acoustic candidates generated (ungated, duration-only baseline).\n")

    print("=== Baseline: duration-only (every >=2-short-burst run counts, no similarity gate) ===")
    baseline = score(per_clip)
    print(f"  recall={baseline['recall']:.3f}  precision={baseline['precision']:.3f}  f1={baseline['f1']:.3f}  "
          f"(n_targets={baseline['n_targets']}, n_candidates={baseline['n_candidates']})")

    print("\n=== Similarity-gated: precision/recall by threshold ===")
    gated_by_threshold = {}
    for thr in SIMILARITY_THRESHOLDS:
        gated_per_clip = [(targets, [c for c in cands if c.mean_similarity >= thr]) for targets, cands in per_clip]
        result = score(gated_per_clip)
        gated_by_threshold[thr] = result
        print(f"  threshold={thr:.2f}: recall={result['recall']:.3f}  precision={result['precision']:.3f}  "
              f"f1={result['f1']:.3f}  (n_candidates={result['n_candidates']})")

    best_thr = max(gated_by_threshold, key=lambda t: gated_by_threshold[t]["f1"])
    best = gated_by_threshold[best_thr]
    print(f"\nBest-F1 threshold: {best_thr:.2f} (F1={best['f1']:.3f}) vs. baseline F1={baseline['f1']:.3f}")
    verdict = "MEANINGFULLY ABOVE baseline" if best["f1"] > baseline["f1"] * 1.2 and best["recall"] > 0 else (
        "NOT meaningfully above baseline" if best["f1"] <= baseline["f1"] * 1.2 else "inconclusive"
    )
    print(f"Verdict (pre-registered success criterion: similarity check meaningfully beats duration-only baseline): {verdict}")

    tag = "_mfcc" if feature == "mfcc" else ""
    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_g_acoustic_sound_repetition{tag}.json"
    out_path.write_text(json.dumps({
        "feature": feature,
        "n_clips": len(clips), "n_targets": n_targets_total, "n_clips_with_target": n_clips_with_target,
        "baseline": baseline,
        "gated_by_threshold": {str(k): v for k, v in gated_by_threshold.items()},
        "best_threshold": best_thr,
        "verdict": verdict,
        "config": {
            "short_burst_max_seconds": SHORT_BURST_MAX_SECONDS, "max_gap_seconds": MAX_GAP_SECONDS,
            "match_tolerance_seconds": MATCH_TOLERANCE_SECONDS, "similarity_n_bins": SIMILARITY_N_BINS,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return {"baseline": baseline, "gated_by_threshold": gated_by_threshold, "best_threshold": best_thr, "verdict": verdict}


# ── Self-test (candidate-generation + scoring math, hand-constructed) ──────

def run_self_test() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. _voiced_runs_with_short_gaps: groups voiced segments across a short
    #    silent gap, breaks across a long one.
    segs = [
        Segment(start=0.0, end=0.1, voiced=True, rms=0.1, zcr=0.1),
        Segment(start=0.1, end=0.15, voiced=False, rms=0.0, zcr=0.0),  # 0.05s gap - short
        Segment(start=0.15, end=0.25, voiced=True, rms=0.1, zcr=0.1),
        Segment(start=0.25, end=0.9, voiced=False, rms=0.0, zcr=0.0),  # 0.65s gap - long
        Segment(start=0.9, end=1.0, voiced=True, rms=0.1, zcr=0.1),
    ]
    runs = _voiced_runs_with_short_gaps(segs, max_gap_s=0.20)
    check("short gap keeps two voiced segments in one run, long gap starts a new run",
          len(runs) == 2 and len(runs[0]) == 2 and len(runs[1]) == 1, str(runs))

    # 2. generate_candidates: a run of 2 short similar bursts followed by a
    #    longer segment should produce exactly one candidate with n_bursts=2.
    burst_segs = [
        Segment(start=0.0, end=0.15, voiced=True, rms=0.1, zcr=0.1),   # short burst 1
        Segment(start=0.15, end=0.20, voiced=False, rms=0.0, zcr=0.0),  # short gap
        Segment(start=0.20, end=0.35, voiced=True, rms=0.1, zcr=0.1),   # short burst 2
        Segment(start=0.35, end=0.40, voiced=False, rms=0.0, zcr=0.0),  # short gap
        Segment(start=0.40, end=1.0, voiced=True, rms=0.1, zcr=0.1),    # completed word (long)
    ]
    times = np.arange(0.0, 1.0, 0.01)
    # Two nearly-identical short bursts: same synthetic RMS/ZCR shape both times.
    rms = np.where((times >= 0.0) & (times < 0.35), 0.15 + 0.02 * np.sin(50 * times), 0.12)
    zcr = np.full_like(times, 0.1)
    cands = generate_candidates(burst_segs, times, rms, zcr)
    check("a 2-short-burst run followed by a long segment yields exactly one candidate",
          len(cands) == 1 and cands[0].n_bursts == 2 and cands[0].followed_by_longer,
          str(cands))
    check("near-identical bursts score high similarity",
          len(cands) == 1 and cands[0].mean_similarity > 0.8, str(cands[0].similarities) if cands else "no candidate")

    # 3. A single short burst alone (no second burst) must NOT become a candidate.
    single_burst = [
        Segment(start=0.0, end=0.15, voiced=True, rms=0.1, zcr=0.1),
        Segment(start=0.15, end=1.0, voiced=False, rms=0.0, zcr=0.0),
    ]
    cands_single = generate_candidates(single_burst, times, rms, zcr)
    check("a single short burst with no repeat does not yield a candidate",
          len(cands_single) == 0, str(cands_single))

    # 4. score(): a candidate overlapping a target (within tolerance) counts
    #    as both a recall hit and a precision TP; one that doesn't overlap
    #    anything is a pure FP that hurts precision, not recall.
    targets = [(0.0, 0.35)]
    good_cand = AcousticCandidate(start=0.02, end=0.33, n_bursts=2, similarities=[0.9], followed_by_longer=True)
    fp_cand = AcousticCandidate(start=5.0, end=5.3, n_bursts=2, similarities=[0.9], followed_by_longer=True)
    result_hit = score([(targets, [good_cand])])
    check("matching candidate: recall=1.0, precision=1.0",
          result_hit["recall"] == 1.0 and result_hit["precision"] == 1.0, str(result_hit))
    result_fp = score([(targets, [fp_cand])])
    check("non-matching candidate: recall=0.0 (target missed), precision=0.0 (candidate is a pure FP)",
          result_fp["recall"] == 0.0 and result_fp["precision"] == 0.0, str(result_fp))
    result_both = score([(targets, [good_cand, fp_cand])])
    check("one hit + one FP: recall=1.0 (target still covered), precision=0.5 (1 of 2 candidates right)",
          result_both["recall"] == 1.0 and result_both["precision"] == 0.5, str(result_both))

    # 5. Cross-clip isolation: a candidate in clip B must NEVER be allowed to
    #    match a target in clip A, even if their raw (clip-local) timestamps
    #    coincidentally overlap - this is the exact bug the first real run
    #    surfaced (implausible recall=1.000/precision=0.966 from pooling
    #    raw timestamps across all 120 clips' independent timelines).
    clip_a_targets = [(1.0, 1.3)]
    clip_b_candidates = [AcousticCandidate(start=1.0, end=1.3, n_bursts=2, similarities=[0.9], followed_by_longer=True)]
    cross_clip_result = score([(clip_a_targets, []), ([], clip_b_candidates)])
    check("a candidate in one clip never matches a target in a different clip, even with identical raw timestamps",
          cross_clip_result["recall"] == 0.0 and cross_clip_result["precision"] == 0.0,
          str(cross_clip_result))

    # 6. MFCC extractor validation on synthetic tones (pre-registered
    #    requirement, ASR_RESEARCH_TRACK.md "Direction (g), MFCC escalation":
    #    two instances of an identical tone must score high similarity; two
    #    clearly different frequencies must score low - checked before this
    #    feature is trusted on real audio).
    sr = 16000
    dur = 0.3
    t = np.arange(0, dur, 1.0 / sr)
    tone_a1 = 0.5 * np.sin(2 * np.pi * 300.0 * t)
    tone_a2 = 0.5 * np.sin(2 * np.pi * 300.0 * (t + 0.001))  # same freq, phase-shifted
    tone_b = 0.5 * np.sin(2 * np.pi * 2000.0 * t)  # a clearly different frequency

    def _mfcc_mean(sig: np.ndarray) -> np.ndarray:
        times, mfcc = compute_mfcc(sig, sr, 0.025, 0.010)
        return mfcc.mean(axis=0)

    v_a1, v_a2, v_b = _mfcc_mean(tone_a1), _mfcc_mean(tone_a2), _mfcc_mean(tone_b)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_same = _cos(v_a1, v_a2)
    sim_diff = _cos(v_a1, v_b)
    check("MFCC: two instances of the same tone score high similarity",
          sim_same > 0.95, f"sim_same={sim_same:.3f}")
    check("MFCC: two clearly different frequencies score meaningfully lower",
          sim_diff < sim_same - 0.1, f"sim_same={sim_same:.3f} sim_diff={sim_diff:.3f}")

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--feature", choices=["rmszcr", "mfcc"], default="rmszcr",
                         help="Per-burst similarity feature (default rmszcr; mfcc is the pre-registered escalation).")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.data_dir or not args.audio_dir:
        print("--data-dir and --audio-dir are required (unless --self-test).")
        return 2
    run(Path(args.data_dir), Path(args.audio_dir), args.n, feature=args.feature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
