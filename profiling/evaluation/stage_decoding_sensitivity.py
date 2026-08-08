"""stage_decoding_sensitivity.py — ASR_RESEARCH_TRACK.md "Decoding-parameter
sensitivity (num_beams)".

Implements the protocol pre-registered in ASR_RESEARCH_TRACK.md's
"Decoding-parameter sensitivity (num_beams) - pre-registered protocol"
section EXACTLY - read that section before changing any logic here.
Tests whether num_beams=5 (CrisperWhisper's own trained default) recovers
any of the 36 sound_repetition/word_repetition positions Stage A found
lost under the live app's forced num_beams=1 (a confirmed transformers
bug workaround for word-timestamp extraction, unrelated to this
question).

Calls model.generate() directly (bypassing pipeline() and
return_timestamps="word" entirely) - only decoded TEXT is needed to test
this hypothesis, not timestamps, so the known beam-search timestamp
crash (huggingface/transformers #28007/#36093) is never triggered.

Real, new ASR cost: unlike every stage since Stage B, this cannot reuse
cached transcription - decoding parameters only take effect during
generation. Both num_beams=1 and num_beams=5 are generated fresh via the
identical code path (direct generate(), not pipeline()), so any
difference is attributable to num_beams alone, not to a pipeline-vs-
direct-call confound.

Usage
-----
    python -m profiling.evaluation.stage_decoding_sensitivity \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 -- must precede torch/transformers imports

from profiling.acoustic import load_wav_samples
from profiling.evaluation.alignment import align, word_error_rate
from profiling.evaluation.stage_b_representation_probe import _identify_positions
from profiling.evaluation.track_b import _DEFAULT_CACHE_DIR, _load_cached, _speaker_stratified_order

TARGET_TYPES = ("sound_repetition", "word_repetition")


def _max_new_tokens_for(duration_s: float) -> int:
    """Identical formula to CrisperWhisperASR._max_new_tokens_for (profiling/asr.py),
    reimplemented here since that method takes a Path and reads a WAV header;
    this script already has samples/sr in memory."""
    return max(20, min(256, int(duration_s * 6) + 20))


def _generate_words(model, processor, samples, sr: int, num_beams: int) -> list[str]:
    import torch

    duration_s = len(samples) / sr
    inputs = processor(samples, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            inputs.input_features,
            language="en", task="transcribe", num_beams=num_beams,
            max_new_tokens=_max_new_tokens_for(duration_s),
        )
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text.split()


def run(data_dir: Path, audio_dir: Path, n_clips: int = 120) -> dict:
    from profiling.evaluation.loaders import load_libristutter_dir_with_audio
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    clips = _speaker_stratified_order(clips)[:n_clips]

    per_clip = {}
    for clip in clips:
        hyp_tokens = _load_cached(_DEFAULT_CACHE_DIR, clip.name)
        if hyp_tokens is None:
            continue
        targets, _ = _identify_positions(clip, hyp_tokens)
        targets = [t for t in targets if t[2] in TARGET_TYPES]
        if targets:
            per_clip[clip.name] = {"clip": clip, "targets": targets}

    n_targets = sum(len(v["targets"]) for v in per_clip.values())
    print(f"{len(per_clip)} clips, {n_targets} target positions (sound/word_repetition).\n")

    print("Loading CrisperWhisper model directly (bypassing pipeline())...")
    processor = AutoProcessor.from_pretrained("nyrahealth/CrisperWhisper")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("nyrahealth/CrisperWhisper", low_cpu_mem_usage=True)
    model.eval()

    recovered = {"sound_repetition": [], "word_repetition": []}
    still_lost = {"sound_repetition": [], "word_repetition": []}
    wers_b1, wers_b5 = [], []
    t0 = time.time()

    for i, (name, rec) in enumerate(per_clip.items()):
        clip = rec["clip"]
        c0 = time.time()
        samples, sr = load_wav_samples(clip.audio_bytes)
        ref_words = [t["word"] for t in clip.tokens]

        hyp1 = _generate_words(model, processor, samples, sr, num_beams=1)
        hyp5 = _generate_words(model, processor, samples, sr, num_beams=5)
        t_clip = time.time() - c0
        print(f"[{i+1}/{len(per_clip)}] {name} ... ({t_clip:.0f}s, {time.time()-t0:.0f}s elapsed)")

        disfluent_idx = set(clip.ground_truth.keys())
        ops1 = align(ref_words, hyp1, disfluent_indices=disfluent_idx)
        ops5 = align(ref_words, hyp5, disfluent_indices=disfluent_idx)
        kind1 = {op.ref_index: op.kind for op in ops1 if op.ref_index is not None}
        kind5 = {op.ref_index: (op.kind, op.hyp_index) for op in ops5 if op.ref_index is not None}
        wers_b1.append(word_error_rate(ref_words, hyp1))
        wers_b5.append(word_error_rate(ref_words, hyp5))

        for ref_idx, hyp_idx_b1, true_type in rec["targets"]:
            k5, hyp_idx5 = kind5.get(ref_idx, ("deletion", None))
            if true_type == "sound_repetition":
                # "Recovered" = a literal fragment-shaped token now precedes
                # the word at this position in the beam=5 hypothesis (the
                # exact structural signature Stage A's category-1 lacked).
                # min length 2 on the fragment guards against a real false-
                # positive class: single-letter words ("a", "I") trivially
                # "prefix-match" almost any following word by coincidence,
                # not because a genuine fragment appeared (verified with a
                # hand-constructed case, e.g. "a apple", before this guard
                # was added -- see PAPER_DECISION_LOG.md).
                recovered_here = False
                if hyp_idx5 is not None and hyp_idx5 > 0:
                    prev_w = hyp5[hyp_idx5 - 1].lower().strip(".,!?;:-")
                    cur_w = hyp5[hyp_idx5].lower().strip(".,!?;:-")
                    if len(prev_w) >= 2 and cur_w.startswith(prev_w) and prev_w != cur_w:
                        recovered_here = True
            else:  # word_repetition: "recovered" = the pair is now intact and adjacent
                recovered_here = False
                if hyp_idx5 is not None and hyp_idx5 > 0:
                    prev_w = hyp5[hyp_idx5 - 1].lower().strip(".,!?;:-")
                    cur_w = hyp5[hyp_idx5].lower().strip(".,!?;:-")
                    if prev_w == cur_w and prev_w:
                        recovered_here = True
            bucket = recovered if recovered_here else still_lost
            bucket[true_type].append({"clip": name, "ref_idx": ref_idx, "ref_word": ref_words[ref_idx],
                                       "hyp5_context": hyp5[max(0, (hyp_idx5 or 0) - 1):(hyp_idx5 or 0) + 2]})

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s for {len(per_clip)} clips x 2 conditions.\n")
    print(f"Mean WER: num_beams=1: {sum(wers_b1)/len(wers_b1):.3f}  "
          f"num_beams=5: {sum(wers_b5)/len(wers_b5):.3f}")

    result = {"n_clips": len(per_clip), "n_targets": n_targets,
              "mean_wer_beams1": sum(wers_b1) / len(wers_b1), "mean_wer_beams5": sum(wers_b5) / len(wers_b5),
              "total_time_s": total_time}
    for t in TARGET_TYPES:
        n_r, n_l = len(recovered[t]), len(still_lost[t])
        print(f"\n=== {t}: {n_r}/{n_r+n_l} recovered under num_beams=5 ===")
        for ex in recovered[t][:5]:
            print(f"  RECOVERED  clip={ex['clip']} ref[{ex['ref_idx']}]='{ex['ref_word']}' "
                  f"beam5_context={ex['hyp5_context']}")
        result[t] = {"n_recovered": n_r, "n_total": n_r + n_l,
                      "recovered_examples": recovered[t], "still_lost_examples": still_lost[t]}

    out_path = _ROOT / "eval_results" / f"{time.strftime('%Y%m%dT%H%M%S')}_stage_decoding_sensitivity.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args(argv)
    run(Path(args.data_dir), Path(args.audio_dir), args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
