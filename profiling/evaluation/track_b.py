"""track_b.py — full-pipeline evaluation: real ASR (CrisperWhisper) on real
audio, our own detector on our own transcript, aligned back to ground truth.

Implements the protocol pre-registered in VALIDATION.md §5.1 EXACTLY —
read that section before changing any scoring logic here. This module does
not decide what counts as an ASR-attributable vs. detector-attributable
error; it computes the decomposition §5.1 already defined.

Usage
─────
    python -m profiling.evaluation.track_b \\
        --data-dir eval_datasets/libristutter_sample \\
        --audio-dir eval_datasets/libristutter_sample_audio \\
        --n 30

Deliberately clip-limited by default (--n) — CrisperWhisper inference is
54-102s/clip on CPU (measured, ARCHITECTURE.md §3), so this is a pilot tool,
not meant to be pointed at the full 499-clip sample without a deliberate,
separate decision to do so (see VALIDATION.md §5.1 "Scope of the pilot").
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paths  # noqa: F401 — must precede torch/transformers imports

from profiling.detect import detect_disfluencies
from profiling.evaluation.alignment import align, word_error_rate
from profiling.evaluation.loaders import LIBRISTUTTER_SCORABLE_TYPES, LabeledClip, load_libristutter_dir_with_audio
from profiling.evaluation.metrics import ANY_LABEL, TypeCounts
from profiling.evaluation.report import counts_to_dict, format_table, save_run

_DEFAULT_RESULT_DIR = _ROOT / "eval_results"
_DEFAULT_CACHE_DIR = _ROOT / "eval_datasets" / "_track_b_cache"


# ── Per-clip ASR+detector result cache ──────────────────────────────────────
# CrisperWhisper inference is the expensive part of every Track B run
# (54-102s/clip). Caching raw hyp_tokens/events per clip means a future
# metric refinement (e.g. this round's context-strict preserved subset) can
# be evaluated by RE-SCORING cached output, not re-running the model —
# real inference is only needed once per clip. Cached under eval_datasets/
# (gitignored, like the datasets themselves — this is derived, reproducible
# data, not something to commit).

def _cache_path(cache_dir: Path, clip_name: str) -> Path:
    safe = clip_name.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe}.json"


def _load_cached(cache_dir: Path, clip_name: str) -> tuple[list[dict], list[dict]] | None:
    p = _cache_path(cache_dir, clip_name)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["hyp_tokens"], data["events"]
    except Exception:
        return None


def _save_cache(cache_dir: Path, clip_name: str, hyp_tokens: list[dict], events: list[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, clip_name).write_text(
        json.dumps({"hyp_tokens": hyp_tokens, "events": events}, ensure_ascii=False),
        encoding="utf-8",
    )


def _empty_counts(scorable_types) -> dict[str, TypeCounts]:
    counts = {t: TypeCounts() for t in scorable_types}
    counts[ANY_LABEL] = TypeCounts()
    return counts


def score_clip(
    clip: LabeledClip,
    hyp_tokens: list[dict],
    events: list[dict],
    scorable_types: tuple[str, ...],
    overall: dict[str, TypeCounts],
    preserved: dict[str, TypeCounts],
    fp_attribution: Counter,
    op_counts: Counter,
    preserved_ctx1: dict[str, TypeCounts] | None = None,
) -> None:
    """Score one clip in place into the running `overall`/`preserved`
    TypeCounts and the `fp_attribution`/`op_counts` diagnostics — see
    VALIDATION.md §5.1 for exactly what each of these means.

    `preserved_ctx1` (optional): the context-strict preserved subset
    pre-registered as a dated addendum to §5.1 after the pilot's hand-
    verification found adjacent-word ASR errors can break context-dependent
    detector checks even when the disfluent word itself aligns `correct`.
    A reference index counts here only if it AND its immediately preceding
    index both align `correct` — see the addendum for exactly what this
    does and doesn't fix.
    """
    ref_words = [t["word"] for t in clip.tokens]
    hyp_words = [t["word"] for t in hyp_tokens]
    disfluent_idx = set(clip.ground_truth.keys())
    ops = align(ref_words, hyp_words, disfluent_indices=disfluent_idx)

    correct_map = {op.ref_index: op.hyp_index for op in ops if op.kind == "correct"}
    # every hyp index's alignment outcome, for FP attribution
    hyp_kind = {}
    for op in ops:
        if op.hyp_index is not None:
            hyp_kind[op.hyp_index] = (op.kind, op.ref_index)

    for op in ops:
        bucket = "disfluent" if (op.ref_index is not None and op.ref_index in disfluent_idx) else (
            "insertion" if op.ref_index is None else "clean"
        )
        op_counts[(bucket, op.kind)] += 1

    predicted_by_hyp_idx: dict[int, set[str]] = {}
    for e in events:
        predicted_by_hyp_idx.setdefault(e["index"], set()).add(e["type"])

    def _score_into(counts: dict[str, TypeCounts], true_type: str | None, predicted_here: set[str]) -> None:
        for t in scorable_types:
            pt, tt = t in predicted_here, true_type == t
            c = counts[t]
            if pt and tt:
                c.tp += 1
            elif pt and not tt:
                c.fp += 1
            elif tt and not pt:
                c.fn += 1
            else:
                c.tn += 1
        pa, ta = bool(predicted_here & set(scorable_types)), true_type in scorable_types
        c = counts[ANY_LABEL]
        if pa and ta:
            c.tp += 1
        elif pa and not ta:
            c.fp += 1
        elif ta and not pa:
            c.fn += 1
        else:
            c.tn += 1

    # ── Overall: every reference position, real end-user-facing outcome ──
    for ref_idx in range(len(ref_words)):
        true_type = clip.ground_truth.get(ref_idx)
        hyp_idx = correct_map.get(ref_idx)  # only "correct" gives a trustworthy predicted set
        predicted_here = predicted_by_hyp_idx.get(hyp_idx, set()) if hyp_idx is not None else set()
        _score_into(overall, true_type, predicted_here)

    # ── Preserved subset: only reference positions ASR transcribed correctly ──
    for ref_idx, hyp_idx in correct_map.items():
        true_type = clip.ground_truth.get(ref_idx)
        predicted_here = predicted_by_hyp_idx.get(hyp_idx, set())
        _score_into(preserved, true_type, predicted_here)

    # ── Context-strict preserved subset (§5.1 addendum): also require the
    # immediately preceding reference index to align "correct" — targets
    # the exact 1-word-back dependency word_repetition/sound_repetition
    # checks have, per the hand-verified example (PAPER_DECISION_LOG.md). ──
    if preserved_ctx1 is not None:
        for ref_idx, hyp_idx in correct_map.items():
            if ref_idx > 0 and ref_idx - 1 not in correct_map:
                continue  # preceding word wasn't also correctly transcribed
            true_type = clip.ground_truth.get(ref_idx)
            predicted_here = predicted_by_hyp_idx.get(hyp_idx, set())
            _score_into(preserved_ctx1, true_type, predicted_here)

    # ── FP attribution: for every predicted event, was its hyp position a
    # correct/substitution/insertion alignment? Substitution/insertion FPs
    # are ASR-attributable by construction (VALIDATION.md §5.1 point 3);
    # correct-position FPs on a clean word are detector-attributable, same
    # as Track A. This is diagnostic bookkeeping ALONGSIDE overall[t].fp
    # (which already counts every real FP, matching what a user experiences)
    # — not a replacement for it. ──
    for hyp_idx, types in predicted_by_hyp_idx.items():
        kind, ref_idx = hyp_kind.get(hyp_idx, ("insertion", None))
        if kind == "correct" and ref_idx is not None and clip.ground_truth.get(ref_idx) not in types:
            fp_attribution["detector_attributable"] += len(types)
        elif kind in ("substitution", "insertion"):
            fp_attribution["asr_attributable"] += len(types)


def _print_disfluent_word_diagnostics(clip: LabeledClip, hyp_tokens: list[dict], events: list[dict]) -> None:
    """Per-clip detail for the pre-registered methodological hand-check
    (VALIDATION.md §5.1, point 4: "a random sample of at least 10 clips'
    alignments hand-checked directly"). Prints only the disfluent-word rows
    (not the whole transcript) — what the reference says, what alignment
    classified it as, what ASR actually produced there, and what the
    detector predicted at that hypothesis position — enough to judge by eye
    whether the alignment's classification is correct."""
    ref_words = [t["word"] for t in clip.tokens]
    hyp_words = [t["word"] for t in hyp_tokens]
    disfluent_idx = set(clip.ground_truth.keys())
    ops = align(ref_words, hyp_words, disfluent_indices=disfluent_idx)
    hyp_kind_by_ref = {op.ref_index: (op.kind, op.hyp_index) for op in ops if op.ref_index is not None}
    predicted_by_hyp_idx: dict[int, set[str]] = {}
    for e in events:
        predicted_by_hyp_idx.setdefault(e["index"], set()).add(e["type"])

    print(f"    ref (ground truth): {ref_words}")
    print(f"    hyp (ASR output):   {hyp_words}")
    for ref_idx in sorted(disfluent_idx):
        true_type = clip.ground_truth[ref_idx]
        kind, hyp_idx = hyp_kind_by_ref.get(ref_idx, ("deletion", None))
        hyp_word = hyp_words[hyp_idx] if hyp_idx is not None else "(none)"
        predicted = predicted_by_hyp_idx.get(hyp_idx, set()) if hyp_idx is not None else set()
        context = ref_words[max(0, ref_idx - 1):ref_idx + 2]
        print(
            f"    ref[{ref_idx}]={ref_words[ref_idx]!r} (true={true_type}, context={context}) "
            f"-> align={kind}, hyp_word={hyp_word!r}, detector_predicted={predicted or '{}'}"
        )


def run(
    data_dir: Path, audio_dir: Path, n_clips: int | None,
    result_dir: Path = _DEFAULT_RESULT_DIR, device: str = "cpu", verbose: bool = False,
    cache_dir: Path | None = _DEFAULT_CACHE_DIR, use_cache: bool = True,
) -> dict:
    print(f"Loading clips + real audio from {data_dir} / {audio_dir} ...")
    clips = load_libristutter_dir_with_audio(data_dir, audio_dir)
    clips = [c for c in clips if c.audio_bytes is not None]
    if n_clips:
        clips = clips[:n_clips]
    print(f"{len(clips)} clips selected for this Track B run "
          f"(pilot scope - see VALIDATION.md section 5.1).\n")

    asr = None  # lazy: only load the (slow) model if any clip is actually a cache miss

    scorable_types = LIBRISTUTTER_SCORABLE_TYPES
    overall = _empty_counts(scorable_types)
    preserved = _empty_counts(scorable_types)
    preserved_ctx1 = _empty_counts(scorable_types)
    fp_attribution: Counter = Counter()
    op_counts: Counter = Counter()
    wers: list[float] = []
    n_cache_hits = 0
    t0 = time.time()

    for i, clip in enumerate(clips):
        print(f"[{i+1}/{len(clips)}] {clip.name} ...", end=" ", flush=True)
        c0 = time.time()
        cached = _load_cached(cache_dir, clip.name) if (use_cache and cache_dir) else None
        if cached:
            hyp_tokens, events = cached
            n_cache_hits += 1
            tag = "cached"
        else:
            if asr is None:
                print("\n  (loading CrisperWhisper - cached locally if already downloaded)")
                from profiling.asr import CrisperWhisperASR
                asr = CrisperWhisperASR(device=device)
            hyp_tokens = asr.transcribe_bytes(clip.audio_bytes)
            events = detect_disfluencies(hyp_tokens, audio_bytes=clip.audio_bytes)
            if use_cache and cache_dir:
                _save_cache(cache_dir, clip.name, hyp_tokens, events)
            tag = "ASR"
        ref_words = [t["word"] for t in clip.tokens]
        hyp_words = [t["word"] for t in hyp_tokens]
        wer = word_error_rate(ref_words, hyp_words)
        wers.append(wer)
        score_clip(clip, hyp_tokens, events, scorable_types, overall, preserved,
                   fp_attribution, op_counts, preserved_ctx1=preserved_ctx1)
        print(f"WER={wer:.2f} [{tag}] ({time.time()-c0:.0f}s, {time.time()-t0:.0f}s elapsed)")
        if verbose:
            _print_disfluent_word_diagnostics(clip, hyp_tokens, events)

    print(f"\nTotal time: {time.time()-t0:.0f}s for {len(clips)} clips "
          f"({n_cache_hits} from cache, {len(clips)-n_cache_hits} real ASR runs).\n")

    print("=== Track B: overall (real end-user experience) ===")
    print(format_table(overall))
    print("\n=== Track B: ASR-preserved subset, original definition (word itself correct) ===")
    print(format_table(preserved))
    print("\n=== Track B: ASR-preserved subset, context-strict (word AND preceding word correct) ===")
    print("(See VALIDATION.md section 5.1 addendum for what this does and doesn't fix.)")
    print(format_table(preserved_ctx1))

    print("\n=== Alignment diagnostics ===")
    print(f"Mean WER: {sum(wers)/max(1,len(wers)):.3f}")
    for bucket in ("disfluent", "clean", "insertion"):
        row = {k: v for (b, k), v in op_counts.items() if b == bucket}
        print(f"  {bucket}: {dict(row)}")
    print(f"FP attribution: {dict(fp_attribution)}")

    print("\n=== ASR-attributable vs detector-attributable recall gap (Any label) ===")
    r_b_overall = overall[ANY_LABEL].recall
    r_b_preserved = preserved[ANY_LABEL].recall
    r_b_preserved_ctx1 = preserved_ctx1[ANY_LABEL].recall
    print(f"Track B overall recall (R_B|overall):              {r_b_overall}")
    print(f"Track B preserved recall (R_B|preserved):          {r_b_preserved}")
    print(f"Track B preserved recall (R_B|preserved_ctx1):     {r_b_preserved_ctx1}")
    print("(Compare against Track A recall from VALIDATION.md section 8.2/8.3 to")
    print(" compute Detector-attributable gap = R_A - R_B|preserved and")
    print(" ASR-attributable gap = R_B|preserved - R_B|overall - see section 5.1")
    print(" and its addendum for the context-strict variant.)")

    path = save_run(
        result_dir, dataset="libristutter", track="B",
        n_clips=len(clips), counts=overall, localization=None,
        extra_metadata={
            "preserved_counts": counts_to_dict(preserved),
            "preserved_ctx1_counts": counts_to_dict(preserved_ctx1),
            "fp_attribution": dict(fp_attribution),
            "alignment_ops": {f"{b}_{k}": v for (b, k), v in op_counts.items()},
            "mean_wer": sum(wers) / max(1, len(wers)),
            "clip_names": [c.name for c in clips],
            "n_cache_hits": n_cache_hits,
        },
    )
    print(f"\nSaved: {path}")
    return {
        "overall": overall, "preserved": preserved, "preserved_ctx1": preserved_ctx1,
        "fp_attribution": fp_attribution, "op_counts": op_counts,
        "wers": wers, "clips": clips,
    }


# ── Self-test (score_clip's decomposition math, independent of real ASR) ────

def run_self_test() -> int:
    """Verifies score_clip's ASR-preserved/overall decomposition against a
    hand-constructed case, with no real ASR involved — same convention as
    track_a.py's self-test. This is what would have caught a bug in the
    decomposition math before it ever touched a real (slow) ASR run."""
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}: {detail}")

    # 1. Alignment: exact match, deletion, and the disfluent-word cost bias.
    ops = align(["the", "cat", "sat"], ["the", "cat", "sat"])
    check("exact match all correct", all(o.kind == "correct" for o in ops), str(ops))

    ops = align(["i", "want", "want", "to", "go"], ["i", "want", "to", "go"], disfluent_indices={2})
    kinds = [(o.kind, o.ref_index) for o in ops]
    check("deletion case has exactly one deletion", sum(1 for k, _ in kinds if k == "deletion") == 1, str(kinds))

    # Genuine substitution-vs-deletion bias test (see PAPER_DECISION_LOG.md).
    biased = align(["i", "need", "need", "coffee"], ["i", "nerd", "coffee"],
                    disfluent_indices={2}, disfluent_cost_multiplier=1.5)
    unbiased = align(["i", "need", "need", "coffee"], ["i", "nerd", "coffee"],
                      disfluent_indices={2}, disfluent_cost_multiplier=1.0)
    biased_disfluent_op = next(o.kind for o in biased if o.ref_index == 2)
    unbiased_disfluent_op = next(o.kind for o in unbiased if o.ref_index == 2)
    check("bias makes the disfluent word a deletion, not a forced substitution",
          biased_disfluent_op == "deletion" and unbiased_disfluent_op == "substitution",
          f"biased={biased_disfluent_op} unbiased={unbiased_disfluent_op}")

    check("WER counts one substitution correctly",
          word_error_rate(["a", "b", "c", "d"], ["a", "x", "c", "d"]) == 0.25,
          str(word_error_rate(["a", "b", "c", "d"], ["a", "x", "c", "d"])))

    # 2. score_clip: hand-constructed case with a known decomposition.
    #    Reference: "i" "want" "want" "coffee" — index 2 ("want" repeat) is
    #    the only disfluency (word_repetition). ASR hypothesis SUBSTITUTES
    #    it away ("i" "want" "wont" "coffee") — so this disfluent word is
    #    NOT in the ASR-preserved subset, and must show up as an automatic
    #    FN in "overall" but simply be ABSENT from "preserved" (not counted
    #    as an attempt at all there).
    clip = LabeledClip(
        name="hand", tokens=[
            {"word": "i", "start": 0.0, "end": 0.2}, {"word": "want", "start": 0.2, "end": 0.4},
            {"word": "want", "start": 0.4, "end": 0.6}, {"word": "coffee", "start": 0.6, "end": 0.9},
        ],
        ground_truth={2: "word_repetition"},
    )
    hyp_tokens = [
        {"word": "i", "start": 0.0, "end": 0.2}, {"word": "want", "start": 0.2, "end": 0.4},
        {"word": "wont", "start": 0.4, "end": 0.6}, {"word": "coffee", "start": 0.6, "end": 0.9},
    ]
    fake_events: list[dict] = []  # detector found nothing (irrelevant to a lost word either way)
    overall = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    preserved = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    fp_attr: Counter = Counter()
    op_counts: Counter = Counter()
    score_clip(clip, hyp_tokens, fake_events, LIBRISTUTTER_SCORABLE_TYPES, overall, preserved, fp_attr, op_counts)
    check("substituted disfluent word counts as FN in overall",
          overall["word_repetition"].fn == 1, str(overall["word_repetition"]))
    check("substituted disfluent word is NOT counted at all in preserved (denominator excludes it)",
          preserved["word_repetition"].tp + preserved["word_repetition"].fn
          + preserved["word_repetition"].fp + preserved["word_repetition"].tn == 3,  # only the 3 clean words
          str(preserved["word_repetition"]))
    check("disfluent bucket recorded as a substitution in op_counts",
          op_counts[("disfluent", "substitution")] == 1, str(dict(op_counts)))

    # 3. Same case, but ASR gets the disfluent word right and the detector
    #    correctly flags it — must show up as TP in BOTH overall and preserved.
    hyp_tokens_correct = [
        {"word": "i", "start": 0.0, "end": 0.2}, {"word": "want", "start": 0.2, "end": 0.4},
        {"word": "want", "start": 0.4, "end": 0.6}, {"word": "coffee", "start": 0.6, "end": 0.9},
    ]
    events_correct = [{"index": 2, "type": "word_repetition"}]
    overall2 = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    preserved2 = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    score_clip(clip, hyp_tokens_correct, events_correct, LIBRISTUTTER_SCORABLE_TYPES,
               overall2, preserved2, Counter(), Counter())
    check("ASR-preserved + detector-correct is TP in overall",
          overall2["word_repetition"].tp == 1, str(overall2["word_repetition"]))
    check("ASR-preserved + detector-correct is TP in preserved",
          preserved2["word_repetition"].tp == 1, str(preserved2["word_repetition"]))

    # 4. FP attribution: a prediction on a pure ASR insertion must be
    #    classified asr_attributable, never detector_attributable.
    hyp_with_insertion = hyp_tokens_correct + [{"word": "extra", "start": 0.9, "end": 1.0}]
    events_with_fp_on_insertion = events_correct + [{"index": 4, "type": "filler"}]
    fp_attr2: Counter = Counter()
    score_clip(clip, hyp_with_insertion, events_with_fp_on_insertion, LIBRISTUTTER_SCORABLE_TYPES,
               _empty_counts(LIBRISTUTTER_SCORABLE_TYPES), _empty_counts(LIBRISTUTTER_SCORABLE_TYPES),
               fp_attr2, Counter())
    check("FP on a pure ASR insertion is asr_attributable, not detector_attributable",
          fp_attr2.get("asr_attributable", 0) >= 1 and fp_attr2.get("detector_attributable", 0) == 0,
          str(dict(fp_attr2)))

    # 5. Context-strict preserved subset (VALIDATION.md section 5.1 addendum):
    #    the exact scenario hand-verification found — the disfluent word
    #    itself is transcribed correctly, but the PRECEDING word is not, so
    #    it must count in the original `preserved` but be EXCLUDED from
    #    `preserved_ctx1`.
    clip_ctx = LabeledClip(
        name="hand_ctx", tokens=[
            {"word": "hi", "start": 0.0, "end": 0.2}, {"word": "Rachel", "start": 0.2, "end": 0.4},
            {"word": "Rachel", "start": 0.4, "end": 0.6}, {"word": "Lynde", "start": 0.6, "end": 0.9},
        ],
        ground_truth={2: "word_repetition"},  # the second "Rachel"
    )
    # ASR inserts "Lynde," between the two Rachels (the real hand-verified
    # case) — ref[2]="Rachel" still aligns correct, but ref[1]="Rachel" (the
    # preceding word) does not, since the hypothesis word at that position
    # is now the inserted "Lynde,".
    hyp_ctx = [
        {"word": "hi", "start": 0.0, "end": 0.2}, {"word": "Lynde,", "start": 0.2, "end": 0.3},
        {"word": "Rachel", "start": 0.3, "end": 0.5}, {"word": "Lynde", "start": 0.5, "end": 0.8},
    ]
    overall3 = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    preserved3 = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    preserved3_ctx1 = _empty_counts(LIBRISTUTTER_SCORABLE_TYPES)
    score_clip(clip_ctx, hyp_ctx, [], LIBRISTUTTER_SCORABLE_TYPES, overall3, preserved3,
               Counter(), Counter(), preserved_ctx1=preserved3_ctx1)
    # The disfluent position (ref_idx=2) is the ONLY position with
    # true_type="word_repetition" in this clip. If it's scored (included),
    # since nothing was predicted there (events=[]) it must show up as an
    # FN. Other clean positions (0, 3) legitimately also get scored into
    # preserved_ctx1 (their own context is fine) contributing TNs — that's
    # correct and expected, not what this check is about.
    check("context-broken word counts (as FN) in original preserved (word itself is correct)",
          preserved3["word_repetition"].fn == 1, str(preserved3["word_repetition"]))
    check("context-broken word is EXCLUDED from context-strict preserved (preceding word is not correct) "
          "- no FN recorded for it, unlike the original preserved subset",
          preserved3_ctx1["word_repetition"].fn == 0, str(preserved3_ctx1["word_repetition"]))

    # 6. Cache round-trip.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cd = Path(d)
        check("cache miss returns None", _load_cached(cd, "nope") is None)
        _save_cache(cd, "clip-a", [{"word": "hi", "start": 0.0, "end": 0.1}], [{"index": 0, "type": "filler"}])
        loaded = _load_cached(cd, "clip-a")
        check("cache round-trip preserves hyp_tokens/events",
              loaded is not None and loaded[0][0]["word"] == "hi" and loaded[1][0]["type"] == "filler",
              str(loaded))

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument("--n", type=int, default=30, help="Pilot clip count (default 30).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--self-test", action="store_true",
        help="Verify the alignment + decomposition math - no real ASR involved.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-clip disfluent-word alignment detail (for hand-verification, "
             "VALIDATION.md section 5.1 point 4).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force real ASR for every clip, ignoring any cached results from a prior run.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help=f"Where to cache per-clip ASR+detector output (default: {_DEFAULT_CACHE_DIR}).",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.data_dir or not args.audio_dir:
        print("--data-dir and --audio-dir are required (unless --self-test).")
        return 2
    data_dir, audio_dir = Path(args.data_dir), Path(args.audio_dir)
    if not data_dir.exists() or not audio_dir.exists():
        print("--data-dir and --audio-dir must both exist.")
        return 2

    cache_dir = Path(args.cache_dir) if args.cache_dir else _DEFAULT_CACHE_DIR
    run(data_dir, audio_dir, args.n, device=args.device, verbose=args.verbose,
        cache_dir=cache_dir, use_cache=not args.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
