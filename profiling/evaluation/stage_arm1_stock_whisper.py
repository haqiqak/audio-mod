"""stage_arm1_stock_whisper.py — ASR_RESEARCH_TRACK.md Phase 2, Arm 1.

Implements the protocol pre-registered in ASR_RESEARCH_TRACK.md's "Phase 2
of this research track" -> "Pre-registered protocol" -> "Arm 1" section
EXACTLY - read that section before changing any logic here. Answers RQ-A:
does stock whisper-large-v3 (same architecture as CrisperWhisper, no
verbatim fine-tuning) preserve sound_repetition/word_repetition evidence at
the 36 positions this track already knows CrisperWhisper normalizes away
(Stage A category 1), better than CrisperWhisper does?

Population: re-derived from the existing Track B cache (CrisperWhisper's
own hyp_tokens), NOT re-scanned fresh - these 31 clips / 36 ref positions
are by construction exactly Stage A's category-1 ("normalized away") cases
for sound_repetition/word_repetition, i.e. under CrisperWhisper every one of
them starts at category 1. This script re-transcribes the SAME clips with
stock whisper-large-v3 and re-categorizes those SAME ref positions under
the new transcript, using the same 4-category scheme Stage A defined
(ASR_RESEARCH_TRACK.md section 8, "Stage A: done").

Deviation from the letter of the pre-registration, recorded here rather
than silently: the "45.2%/40.5%" comparison baseline cited as the success
criterion was computed over Stage A's full 42-position audit population,
not the 36-position category-1 subset alone. Re-running that full audit
for stock Whisper (hand-tracing every one of the 120-clip sample's FN
cases again) is out of proportion to the 30-55 min cost this arm was
priced at. This script instead reports what fraction of the 36 KNOWN
losses remain in category 1 under stock Whisper - a more targeted, and
arguably more direct, version of the same question ("does a different ASR
recover the specific evidence CrisperWhisper is known to lose") - and
states this substitution explicitly rather than quietly redefining the
metric.

Usage
-----
    python -m profiling.evaluation.stage_arm1_stock_whisper \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 -- must precede torch/transformers imports

from profiling.detect import detect_disfluencies
from profiling.evaluation.alignment import align, word_error_rate
from profiling.evaluation.loaders import load_libristutter_dir_with_audio
from profiling.evaluation.stage_b_representation_probe import _GATE_OFF_CONFIG, _identify_positions
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPES = ("sound_repetition", "word_repetition")
STOCK_MODEL_ID = "openai/whisper-large-v3"
_ARM1_CACHE_DIR = _ROOT / "eval_datasets" / "_arm1_stock_whisper_cache"

# Original Stage A rates (full 42-position audit, ASR_RESEARCH_TRACK.md
# section 8's table) - kept here only as the printed reference point, not
# recomputed by this script.
CRISPERWHISPER_NORMALIZED_AWAY_RATE = {"sound_repetition": 0.452, "word_repetition": 0.405}


def _select_population(data_dir: Path, audio_dir: Path, n_scan: int = 120):
    """Re-derives the exact 31-clip/36-position population Stage B/C/layer-
    sweep already used: clips (scanned in the same speaker-stratified order,
    same n_scan) that contain at least one Stage-A category-1 target
    position for sound_repetition/word_repetition, under CrisperWhisper's
    own cached Track B output. Returns [(clip, [(ref_idx, true_type), ...])]."""
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips = _speaker_stratified_order(clips)[:n_scan]

    selected = []
    for clip in clips:
        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        if hyp_tokens is None:
            continue
        targets, _clean = _identify_positions(clip, hyp_tokens)
        targets = [(ref_idx, true_type) for ref_idx, _hyp_idx, true_type in targets if true_type in TARGET_TYPES]
        if targets:
            selected.append((clip, targets))
    return selected


# ── Stock-Whisper transcription (bypasses CrisperWhisperASR's deliberate
# vanilla-Whisper guard in profiling/asr.py - that guard exists to protect
# the live app from silently losing disfluencies; this script's entire
# purpose is to measure exactly that behavior on stock Whisper, so calling
# transformers.pipeline() directly here is correct, not a workaround). ──

def _load_stock_pipeline(model_id: str):
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        return_timestamps="word",
        model_kwargs={"low_cpu_mem_usage": True, "attn_implementation": "eager"},
        # num_beams=1: same live-app decoding configuration CrisperWhisper
        # uses (ASR_RESEARCH_TRACK.md Phase 2's "Preprocessing confound
        # (Arm 1)" note) - a fair comparison, not the model's unconstrained
        # default (num_beams=5).
        generate_kwargs={
            "language": "en", "task": "transcribe", "num_beams": 1,
            "max_new_tokens": 256, "return_legacy_cache": True,
        },
    )


def _transcribe(pipe, audio_bytes: bytes) -> list[dict]:
    import io
    import tempfile
    import wave

    from profiling.asr import resample_to_16k

    resampled = resample_to_16k(audio_bytes)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(resampled)
        tmp_path = tf.name
    try:
        result = pipe(tmp_path, generate_kwargs={"max_new_tokens": 256})
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    chunks = result.get("chunks") or result.get("segments") or []
    tokens = []
    for chunk in chunks:
        word = str(chunk.get("text") or chunk.get("word") or "").strip()
        if not word:
            continue
        ts = chunk.get("timestamp") or (chunk.get("start"), chunk.get("end"))
        start = end = None
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start = float(ts[0]) if ts[0] is not None else None
            end = float(ts[1]) if ts[1] is not None else None
        tokens.append({"word": word, "start": start, "end": end})
    return tokens


def _cache_path(clip_name: str) -> Path:
    safe = clip_name.replace("/", "_").replace("\\", "_")
    return _ARM1_CACHE_DIR / f"{safe}.json"


def _load_cache(clip_name: str) -> list[dict] | None:
    p = _cache_path(clip_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))["hyp_tokens"]
    except Exception:
        return None


def _save_cache(clip_name: str, hyp_tokens: list[dict]) -> None:
    _ARM1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(clip_name).write_text(
        json.dumps({"hyp_tokens": hyp_tokens}, ensure_ascii=False), encoding="utf-8",
    )


# ── Stage-A-style 4-category classification, generalized from
# stage_b_representation_probe._identify_positions (which only computes
# category 1) to all 4 categories, plus a 5th outcome (recovered_tp) for
# positions where the new ASR's transcript lets our own detector actually
# get it right - not one of Stage A's original loss categories, but the
# most important possible outcome for this specific arm's question. ──

def _categorize(clip, hyp_tokens: list[dict], target_ref_indices: list[tuple[int, str]]) -> dict[int, str]:
    ref_words = [t["word"] for t in clip.tokens]
    hyp_words = [t["word"] for t in hyp_tokens]
    disfluent_idx = set(clip.ground_truth.keys())
    ops = align(ref_words, hyp_words, disfluent_indices=disfluent_idx)
    hyp_kind_by_ref = {op.ref_index: (op.kind, op.hyp_index) for op in ops if op.ref_index is not None}

    events = detect_disfluencies(hyp_tokens, config=_GATE_OFF_CONFIG, audio_bytes=clip.audio_bytes)
    predicted_by_hyp_idx: dict[int, set[str]] = defaultdict(set)
    for e in events:
        predicted_by_hyp_idx[e["index"]].add(e["type"])

    out = {}
    for ref_idx, true_type in target_ref_indices:
        kind, hyp_idx = hyp_kind_by_ref.get(ref_idx, ("deletion", None))
        predicted_here = predicted_by_hyp_idx.get(hyp_idx, set()) if hyp_idx is not None else set()
        if kind == "correct":
            if true_type in predicted_here:
                out[ref_idx] = "recovered_tp"
            elif not predicted_here:
                out[ref_idx] = "1_normalized_away"
            else:
                out[ref_idx] = "2_mis_routed"
        else:
            out[ref_idx] = "4_asr_error_coincidental" if predicted_here else "3_asr_error"
    return out


def run(data_dir: Path, audio_dir: Path, n_scan: int = 120, use_cache: bool = True) -> dict:
    print(f"Re-deriving the 31-clip/36-position population from {data_dir} / {audio_dir} "
          f"(CrisperWhisper's cached Track B output) ...")
    population = _select_population(data_dir, audio_dir, n_scan=n_scan)
    n_positions = sum(len(t) for _c, t in population)
    print(f"{len(population)} clips, {n_positions} target positions "
          f"(all category-1/'normalized away' under CrisperWhisper, by construction).\n")

    pipe = None
    category_counts: Counter = Counter()  # per type
    category_counts_by_type: dict[str, Counter] = {t: Counter() for t in TARGET_TYPES}
    wers: list[float] = []
    n_cache_hits = 0
    t0 = time.time()

    per_position_detail = []

    for i, (clip, targets) in enumerate(population):
        print(f"[{i+1}/{len(population)}] {clip.name} ...", end=" ", flush=True)
        c0 = time.time()
        hyp_tokens = _load_cache(clip.name) if use_cache else None
        if hyp_tokens is not None:
            n_cache_hits += 1
            tag = "cached"
        else:
            if pipe is None:
                print(f"\n  (loading {STOCK_MODEL_ID} - downloads on first use, ~3GB)")
                pipe = _load_stock_pipeline(STOCK_MODEL_ID)
            hyp_tokens = _transcribe(pipe, clip.audio_bytes)
            if use_cache:
                _save_cache(clip.name, hyp_tokens)
            tag = "ASR"

        cats = _categorize(clip, hyp_tokens, targets)
        for ref_idx, true_type in targets:
            cat = cats[ref_idx]
            category_counts[cat] += 1
            category_counts_by_type[true_type][cat] += 1
            per_position_detail.append({
                "clip": clip.name, "ref_idx": ref_idx, "true_type": true_type, "category": cat,
            })

        ref_words = [t["word"] for t in clip.tokens]
        hyp_words = [t["word"] for t in hyp_tokens]
        wer = word_error_rate(ref_words, hyp_words)
        wers.append(wer)
        print(f"WER={wer:.2f} [{tag}] ({time.time()-c0:.0f}s, {time.time()-t0:.0f}s elapsed)")

    print(f"\nTotal time: {time.time()-t0:.0f}s for {len(population)} clips "
          f"({n_cache_hits} from cache, {len(population)-n_cache_hits} real ASR runs).\n")

    print("=== Arm 1: category breakdown at the 36 known-loss positions, under stock whisper-large-v3 ===")
    for t in TARGET_TYPES:
        c = category_counts_by_type[t]
        n = sum(c.values())
        if n == 0:
            continue
        print(f"\n{t} (n={n}):")
        for cat in ("recovered_tp", "1_normalized_away", "2_mis_routed", "3_asr_error", "4_asr_error_coincidental"):
            cnt = c.get(cat, 0)
            print(f"  {cat}: {cnt} ({100*cnt/n:.1f}%)")
        stock_rate = c.get("1_normalized_away", 0) / n
        baseline_rate = CRISPERWHISPER_NORMALIZED_AWAY_RATE.get(t)
        print(f"  --> stock whisper-large-v3 'still normalized away' rate: {100*stock_rate:.1f}% "
              f"(CrisperWhisper's full-audit baseline for reference: {100*baseline_rate:.1f}%; "
              f"NOTE: baselines are not computed over the identical denominator, see module docstring)")

    mean_wer = sum(wers) / max(1, len(wers))
    print(f"\nMean WER (stock whisper-large-v3, 31 clips): {mean_wer:.3f}")

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_arm1_stock_whisper.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model_id": STOCK_MODEL_ID,
        "n_clips": len(population),
        "n_positions": n_positions,
        "category_counts_by_type": {t: dict(c) for t, c in category_counts_by_type.items()},
        "mean_wer": mean_wer,
        "per_position_detail": per_position_detail,
        "n_cache_hits": n_cache_hits,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return {"category_counts_by_type": category_counts_by_type, "mean_wer": mean_wer}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--n-scan", type=int, default=120,
                         help="How many speaker-stratified clips to scan when re-deriving the population (default 120, matching Stage B/C).")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    run(Path(args.data_dir), Path(args.audio_dir), n_scan=args.n_scan, use_cache=not args.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
