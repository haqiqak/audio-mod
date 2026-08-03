# ROADMAP.md — what's next, in priority order

A single forward-looking list across the whole project, consolidated from
items scattered across `ARCHITECTURE.md`, `VALIDATION.md`, and
`PAPER_DECISION_LOG.md`. This file points *at* the doc/entry with the full
reasoning rather than duplicating it — see `DOCS.md` for how these files
relate. Update this whenever priorities shift; when an item completes or is
dropped, move it to "Completed" or "Explicitly rejected" with a link to the
`CHANGELOG.md`/`PAPER_DECISION_LOG.md` entry that closed it out, rather than
deleting the line.

---

## Immediate (next up)

1. **Build the `profiling/evaluation/` package** per `VALIDATION.md` §6 —
   blocked on the current baseline being committed first (owner's explicit
   sequencing, `PAPER_DECISION_LOG.md` 2026-08-03).
2. **Run Track A against real LibriStutter and SEP-28k data** — the first
   objective accuracy numbers this project will have ever had against public
   benchmarks. See `VALIDATION.md` §8 for the results template waiting to be
   filled in.
3. **Build Track B (full pipeline + ASR↔reference alignment)** and directly
   re-examine the missed-word-repetition finding from the 2026-08-03 real-mic
   test through it — confirm whether that gap is ASR-attributable (expected)
   or also a detector issue (would be new information).

## Near-term

4. **Validate `block` and `stutter_marker` against SEP-28k/KSoF specifically**
   — LibriStutter's taxonomy can't score either type at all (`VALIDATION.md`
   §2); this needs the Tier-2 dataset.
5. **The deferred learned tier** — a frozen WavLM-base or wav2vec2-base
   classifier for repetition-subtype discrimination and cross-speaker
   generalization (Shih et al. 2024; the multi-task/adversarial-learning
   literature), trained/evaluated on SEP-28k + FluencyBank Timestamped. The
   clear evidence-backed next architectural step, deliberately deferred until
   §8's baseline numbers exist to prove it actually helps — see
   `PAPER_DECISION_LOG.md`'s 2026-08 restructuring entry for the full
   reasoning on why this wasn't built immediately.
6. **Ablation studies** — `VALIDATION.md` §9 has the candidate list (Silero
   VAD on/off, Praat features on/off, fusion-weight sweep, calibration on/off,
   threshold sweep). Run once a baseline result exists to ablate against.
7. **Extend `evaluate.py`'s LibriStutter path to real audio** (FLAC decoding
   or pre-conversion) so it can also exercise the acoustic-fusion path, today
   only covered by synthetic-tone unit tests.
8. **A more sophisticated stutter_marker acoustic check** — an offset-shape /
   abrupt-energy-drop heuristic for cut-off fragments, replacing the simpler
   voiced-energy-presence check shipped 2026-08. Explicitly not built without
   real recordings to validate it against.

## Longer-term

9. **True streaming / real-time transcription** — `ARCHITECTURE.md` §7's
   reasoning still holds: CPU ASR latency (54–102s/clip) makes this moot
   until addressed separately (a faster/smaller model, or GPU). The
   `profiling/acoustic.py` foundation is windowable and was built with this
   in mind, but no streaming work has started.
10. **Multi-language support** — English only, hardcoded, throughout
    (`filler_words`, CMU-dict-based phonetics, `nltk` POS tagging). KSoF
    (German) is the one dataset in `VALIDATION.md` §2 that would force this
    question if pursued.
11. **Product/ethics review of the downstream "replace flagged words" use
    case** — the end-user-alignment literature reviewed during the 2026-08
    research round (Aligning Stuttered-Speech Research with End-User Needs,
    2026; Disability-First AI Dataset Annotation) raises a real question
    about automatically flagging a word as a stutter site and feeding it to a
    rewrite pipeline, which risks conflating "hard to say" with "should be
    avoided." Not an engineering task — flagged here so it doesn't get lost.
12. **CTC forced alignment as a Stage-1 accuracy check** — considered during
    the 2026-08 research round and not adopted (CrisperWhisper's own
    benchmarked word-timestamp accuracy was the reason it was chosen in the
    first place, and a second heavy model wasn't justified without evidence
    the existing timestamps are the bottleneck). Revisit only if `VALIDATION.md`
    results show word-boundary precision is actually a limiting factor.

## Explicitly rejected

These were considered and turned down with reasons recorded, not silently
dropped — do not re-litigate without new evidence.

- **End-to-end audio→dysfluency-region models** (YOLO-Stutter/
  Stutter-Solver/SSDM-class) as a replacement for the two-stage
  ASR-then-detector pipeline. Rejected: still require a speech-text alignment
  as input (don't remove the ASR stage), and a 2025 comparative study found
  the most complex of these (SSDM) irreproducible by an independent team —
  see `PAPER_DECISION_LOG.md`'s 2026-08 restructuring entry.
- **OpenVINO as the ASR backend** — confirmed upstream bug (optimum-intel
  #561), can't produce word timestamps this app depends on for everything.
  See `ARCHITECTURE.md` §3.
- **faster-whisper as the ASR backend** — CrisperWhisper's fine-tune has a
  pruned/shifted tokenizer faster-whisper's special-token handling can't
  cope with. See `ARCHITECTURE.md` §3.
- **A hard replacement of RMS/ZCR segmentation with Silero VAD** — VAD is
  trained on real speech and returns nothing on this project's synthetic
  test tones, which would have silently broken the entire synthetic-audio
  test suite. Implemented instead as a self-disabling corroboration signal.
  See `PAPER_DECISION_LOG.md`'s 2026-08 restructuring entry.

## Completed

- **Audio-native-primary detector restructuring** (taxonomy split, acoustic
  corroboration for filler/stutter_marker, weighted-confidence fusion, Silero
  VAD + Praat integration, `profiling/evaluate.py` v1) — 2026-08-03. See
  `CHANGELOG.md` and `PAPER_DECISION_LOG.md`.
- **Event-table display fix** (acoustic-sourced events now show their real
  detected region, not the full attributed word's span) — 2026-08-03. See
  `CHANGELOG.md`.
- **Evaluation methodology research and plan** — 2026-08-03, written up as
  `VALIDATION.md`. Implementation not yet started (this file's "Immediate"
  section above).
