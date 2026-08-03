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

## Phase 1 is closed (2026-08-03)

Validation, Benchmarking, Analysis, and Scientific Understanding — see
`PHASE_1_SUMMARY.md` for the full closing summary (confirmed findings,
scope/generalization limits, and the readiness argument for Phase 2). All
Phase 1 work (dataset evaluation, ablations, Track A/B, the critical
methodology review) is in "Completed" below, not repeated here. Everything
in this section is **Phase 2: evidence-driven improvement** priority, in
order, each directly justified by a specific Phase 1 finding (linked).

## Immediate (next up)

1. **[Top priority — Phase 1's own critical review, `VALIDATION.md` §7.2
   item 3] Validate the confirmed "ASR is the bottleneck" conclusion
   against a second ASR backend and/or real (non-synthetic) disfluent
   speech.** Every Track B number to date comes from exactly one ASR model
   (CrisperWhisper) on exactly one dataset family (LibriStutter's
   synthetically-spliced disfluencies). The absolute Track A→B recall drop
   is not in question, but whether ASR-fidelity is *generally* the
   bottleneck (vs. specific to this backend/dataset combination) is not
   yet established at the same rigor as the rest of the finding. Two
   independent ways to check, either is valuable on its own: (a) run a
   second ASR backend (e.g. stock whisper-large-v3) through the same
   Track B pipeline on the same audio already downloaded — cheaper than a
   new dataset, no new licensing/acquisition; (b) integrate FluencyBank
   Timestamped (real people who stutter, word-level timestamps — see item
   4 below) and re-run Track B against it. Judged the single
   highest-value-per-effort Phase 2 validation item.
2. **[Top priority — `VALIDATION.md` §7.2 item 2] Re-sample Track B across
   speakers, not just by clip count.** The 90-clip scaled run covers only
   7 of the 499-clip Track A sample's 40 distinct speakers (a deterministic
   "first-N" prefix of a speaker-ordered file list, not a representative
   cross-section) — a real, previously-undocumented limitation found during
   the Phase 1 closing review. Before trusting the confirmed §8.4.2
   conclusion as speaker-general, re-run Track B on a speaker-stratified
   sample (e.g. every-Kth clip across the full 499, or N clips per speaker)
   rather than a prefix. Needs fresh CrisperWhisper inference on the newly
   included clips (the per-clip cache means previously-run clips are still
   free); a real but bounded time cost, not a re-architecture.
3. **[High priority, evidence-based, confirmed pattern] Fix the
   word_repetition/sound_repetition type-classification gap
   `preserved_ctx1` surfaced.** Not a recall problem — a type-label problem.
   At n=90, only 2/7 (29%) of context-strict-preserved disfluent instances
   got the *exact* correct type label; the other 5 (`word_repetition` ×4,
   `sound_repetition` ×1) were still flagged disfluent (`Any` correct) but
   mislabeled, mostly as `phrase_repetition`/`block`. Traced mechanism
   (`VALIDATION.md` §8.4.1/§8.4.2, clip `103-1240-0000`): the detector's
   word-adjacency check operates on the raw ASR hypothesis token stream,
   which can have an inserted word between two otherwise-correctly-
   transcribed repeated words, breaking literal back-to-back adjacency even
   though a reference-level alignment says both words are "correct." Two
   candidate fixes, not yet scoped: (a) make the detector's own adjacency
   check tolerant of a single inserted/filler token between repeat
   candidates, or (b) build the hypothesis-side-contiguity-aware metric
   already flagged in §8.4.1 to confirm the fix actually helps before
   shipping it. This is the one place in the whole Track B analysis with a
   real, scoped, *detector*-side action item — everything else points at
   ASR (item 1).
4. **[De-prioritized by evidence, was previously higher] Tuning
   `profiling/detect.py`'s detection thresholds/logic purely to raise
   recall.** Phase 1's confirmed conclusion is that recall is not being
   lost in the detector when given intact input — `prolongation_min_seconds`
   (§9) and similar thresholds still matter for *precision* (§9.1) and
   remain worth validating on real ASR timestamps eventually, but are no
   longer believed to be where real-world *recall* is being lost, so
   further threshold-only tuning work for that purpose specifically is
   explicitly deprioritized until items 1–3 are further along.
5. **[Cheap, parallel, no new data needed] Build a confidence-sensitive
   metric** (mean confidence of TP vs. FP, or the EER metric already
   flagged in §4) so Silero VAD and Praat corroboration can actually be
   evaluated — §9.3's finding is that the current presence/absence metric
   is structurally blind to their designed effect, not that they don't
   help.
6. **[Cheap, parallel, no new data needed — new from the Phase 1 closing
   review, `VALIDATION.md` §7.2 item 5] Add confidence intervals to every
   reported recall/precision number.** Every Phase 1 number is a point
   estimate with only qualitative small-sample caveats (e.g. "too few
   instances to trust individually"); `metrics.py` already has every raw
   count needed for a Wilson/Clopper-Pearson interval. Especially valuable
   at the extreme small-n cases (`R_B|preserved_ctx1`'s n=2/n=7) where a
   formal interval would make the existing qualitative caveat concrete.
7. **[Medium priority — new from the Phase 1 closing review, `VALIDATION.md`
   §7.2 item 4] Build a Track B localization (IoU/temporal) metric.**
   `track_b.py`'s `score_clip` currently hardcodes `localization=None` —
   Track B has never validated *how precisely timed* a caught disfluency
   is under real ASR conditions, only Track A has (§4 point 3). Feasible
   in principle (both a predicted acoustic span and a ground-truth
   reference span exist in real audio time) but real, non-trivial work —
   not a quick fix.
8. **Investigate `sound_repetition`'s 0% recall on Track A (confirmed
   structural finding, not yet resolved, distinct from item 3's
   context-corruption issue)** — the fragment-repeat check in
   `profiling/detect.py` only catches "fragment-then-complete-word"
   ordering; real LibriStutter data (and possibly real speech generally)
   also has "complete-word-then-fragment" patterns. See `VALIDATION.md` §8.2.
9. **Standardize non-ASCII-in-console-output prevention** — the same
   Windows `cp1252` encoding bug has now been fixed three separate times
   (`track_a.py`, `report.py`, `track_b.py`) as new print statements were
   added. Worth a lint rule or pre-commit check rather than fixing it
   reactively a fourth time.
10. **Expand the LibriStutter sample** (499/4,736 files, ~10.5% of the
    corpus) for more statistical power, and specifically target a sample
    with `filler`/interjection instances (the 499-file sample happened to
    have none) — `profiling/evaluation/fetch_libristutter_sample.py --n`.
    Lower priority than items 1–7 per the evidence above.
11. **SEP-28k: acquire real audio, then run Track A (acoustic-only) or
    Track B** — `load_sep28k_labels` is built and verified against the real
    labels file, but SEP-28k has no reference transcript at all, so nothing
    can be *scored* yet without audio (bandwidth/storage/time cost,
    materially larger than LibriStutter's annotation-only footprint — check
    with the project owner before attempting). Track B's alignment
    machinery now exists and would need adapting to clip-level rather than
    word-level ground truth to be usable here. Only partially addresses
    item 1's ASR-generalization question (real speech, but no word-level
    transcript) — item 1's FluencyBank option is the more direct check.
12. **FluencyBank Timestamped** — real people who stutter (not synthetic
    injection like LibriStutter), word-level timestamps + disfluency labels.
    Scientifically the strongest dataset option researched so far, but not
    chosen for the 2026-08-03 phase due to unconfirmed integration risk:
    hosted on TalkBank in CHAT format (needs a dedicated parser) and
    possibly access-gated (unconfirmed). Investigate access/format directly
    before committing engineering time. This is now item 1's most direct
    dataset-side option, not just a "nice to have."

## Near-term

- **Validate `block` and `stutter_marker` against SEP-28k/KSoF specifically**
  — LibriStutter's taxonomy can't score either type at all (`VALIDATION.md`
  §2); this needs the Tier-2 dataset.
- **The deferred learned tier** — a frozen WavLM-base or wav2vec2-base
  classifier for repetition-subtype discrimination and cross-speaker
  generalization (Shih et al. 2024; the multi-task/adversarial-learning
  literature), trained/evaluated on SEP-28k + FluencyBank Timestamped. The
  clear evidence-backed next architectural step, deliberately deferred until
  a baseline exists to prove it actually helps — that baseline now exists
  (§8.3) — see `PAPER_DECISION_LOG.md`'s 2026-08 restructuring entry for the
  full reasoning on why this wasn't built immediately.
- ~~Ablation studies~~ — **done**, see "Immediate" above and `VALIDATION.md`
  §9. Follow-on fine-grained work (e.g. isolating VAD's effect once a
  confidence-sensitive metric exists) stays here.
- ~~Extend the LibriStutter path to real audio~~ — **done**, this project's
  first research baseline (see "Immediate" above). Real audio via
  `fetch_libristutter_audio.py`; a real FLAC-decode bug was found and fixed
  in the process (`PAPER_DECISION_LOG.md`).
- **A more sophisticated stutter_marker acoustic check** — an offset-shape /
  abrupt-energy-drop heuristic for cut-off fragments, replacing the simpler
  voiced-energy-presence check shipped 2026-08. Explicitly not built without
  real recordings to validate it against.

## Longer-term

- **True streaming / real-time transcription** — `ARCHITECTURE.md` §7's
  reasoning still holds: CPU ASR latency (54–102s/clip) makes this moot
  until addressed separately (a faster/smaller model, or GPU). The
  `profiling/acoustic.py` foundation is windowable and was built with this
  in mind, but no streaming work has started.
- **Multi-language support** — English only, hardcoded, throughout
  (`filler_words`, CMU-dict-based phonetics, `nltk` POS tagging). KSoF
  (German) is the one dataset in `VALIDATION.md` §2 that would force this
  question if pursued.
- **Product/ethics review of the downstream "replace flagged words" use
  case** — the end-user-alignment literature reviewed during the 2026-08
  research round (Aligning Stuttered-Speech Research with End-User Needs,
  2026; Disability-First AI Dataset Annotation) raises a real question
  about automatically flagging a word as a stutter site and feeding it to a
  rewrite pipeline, which risks conflating "hard to say" with "should be
  avoided." Not an engineering task — flagged here so it doesn't get lost.
- **CTC forced alignment as a Stage-1 accuracy check** — considered during
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

- **Phase 1 (Validation, Benchmarking, Analysis, Scientific Understanding)
  formally closed** — 2026-08-03. Critical review of the full methodology
  (`VALIDATION.md` §7.2–§7.4): fixed the Track B decomposition's $R_A$
  approximation (now exact, not approximated); identified and scoped two
  new generalization limits (speaker-clustering in the Track B subset,
  items 1–2 above; single-ASR-backend/single-dataset dependence, item 1
  above); reconciled documentation drift across `README.md`/
  `ARCHITECTURE.md`/`VALIDATION.md`. Closing summary: `PHASE_1_SUMMARY.md`.
  See `PAPER_DECISION_LOG.md`.
- **Track B scaled 30 → 90 clips — context-strict finding confirmed as a
  major research conclusion** — 2026-08-03. `R_B|preserved_ctx1` recall =
  1.0 again at n=7 (up from n=2), stable ~5.5% sample-attrition rate. The
  detector's binary disfluent/clean judgment is effectively perfect given
  intact ASR input; the real-world recall shortfall is overwhelmingly
  ASR-attributable. A confirmed, scoped detector-side issue also surfaced:
  `word_repetition`/`sound_repetition` type-label accuracy is weak (2/7
  exact matches) even when binary detection succeeds. Synthesis and
  resulting priority changes: see "Immediate" items 8–10 above.
  `VALIDATION.md` §8.4.2 and `PAPER_DECISION_LOG.md`.
- **Context-strict preserved-subset scoring (`R_B|preserved_ctx1`)
  implemented and run** — 2026-08-03. Pre-registered in `VALIDATION.md`
  §5.1's addendum before implementation; added per-clip caching to
  `track_b.py` at the same time. Reverses the previous ~95%/5%
  detector/ASR attribution split to ~0%/~100% (n=2, extremely small —
  direction credible, exact split not). Surfaced a further,
  not-yet-fixed hypothesis-side-contiguity gap. See `VALIDATION.md` §8.4.1
  and `PAPER_DECISION_LOG.md`.
- **Track B implemented and piloted, protocol pre-registered first** —
  2026-08-03. `VALIDATION.md` §5.1 written before `alignment.py`/`track_b.py`
  existed. 30-clip pilot, hand-verified. Headline: Track A's ~99% recall
  drops to ~4–9% under real ASR conditions — this project's most important
  evaluation finding to date. A real methodological limitation
  (adjacent-context contamination of the "ASR-preserved" subset) was found
  during hand-verification and recorded as a dated addendum, not smoothed
  over. See `VALIDATION.md` §8.4 and `PAPER_DECISION_LOG.md`.
- **Full ablation study against the baseline** (10 variants: VAD on/off,
  Praat on/off, fusion-weight sweep, prolongation-threshold sweep) —
  2026-08-03. `prolongation_min_seconds` dominant by an order of magnitude;
  VAD/Praat measured zero effect (a metric-blindness finding, not a
  negative result); fusion weight has a small real effect. See
  `VALIDATION.md` §9 and `PAPER_DECISION_LOG.md`.
- **Audio-native-primary detector restructuring** (taxonomy split, acoustic
  corroboration for filler/stutter_marker, weighted-confidence fusion, Silero
  VAD + Praat integration, `profiling/evaluate.py` v1) — 2026-08-03. See
  `CHANGELOG.md` and `PAPER_DECISION_LOG.md`.
- **Event-table display fix** (acoustic-sourced events now show their real
  detected region, not the full attributed word's span) — 2026-08-03. See
  `CHANGELOG.md`.
- **Evaluation methodology research and plan** — 2026-08-03, written up as
  `VALIDATION.md`.
- **`profiling/evaluation/` package, steps 1–2** — 2026-08-03 (package
  skeleton + migration + IoU localization + confusion matrices + "Any"
  label).
- **`load_sep28k_labels` built and verified against the real, complete
  labels file** — 2026-08-03 (scoring against it still blocked on real
  audio — see "Immediate" above).
- **First real Track A result** (499 real LibriStutter clips, text-only) —
  2026-08-03. See `VALIDATION.md` §8.2 and `PAPER_DECISION_LOG.md`.
- **First research baseline: audio-enabled Track A result** (same 499
  clips, real audio, the audio-native layer evaluated against ground truth
  for the first time) — 2026-08-03. `Any` F1 0.773 → 0.835. Includes a
  real bug found and fixed (`soundfile` silently decoding real FLAC files
  as silence) before the result was trusted. See `VALIDATION.md` §8.3 and
  `PAPER_DECISION_LOG.md`.
