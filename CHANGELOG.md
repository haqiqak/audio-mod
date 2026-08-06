# CHANGELOG.md — fast-scan history

Reverse-chronological, one line per change. This is the "what changed and
when" index; for "why," follow the pointer into `PAPER_DECISION_LOG.md` (its
entries are titled to match). See `DOCS.md` for how these files relate.

---

## 2026-08-06 (asr-research branch)

- **Direction (g) pre-registered: an acoustic-native `sound_repetition`
  candidate generator.** Chosen over costing out Stage D as the next
  step (cheaper, no new infrastructure). A new waveform-only detection
  mechanism (short, spectrally self-similar voiced bursts in
  succession), evaluated against LibriStutter's own ground-truth
  timestamps rather than ASR output — deliberately escapes the earlier
  superseded fusion idea's starved-population problem by not depending
  on ASR alignment at all. Full protocol pre-registered before any
  code, per rule 1; implementation needs a separate go-ahead. →
  *PAPER_DECISION_LOG.md, "Direction (g) pre-registered..."*

- **Phase 2 integrative conclusion: Failure/Failure/Failure — Stage D
  costing is the evidence-motivated next step.** All three pre-registered
  arms ran and each independently failed its own success criterion —
  directions (a) (different pretrained ASR) and (b) (different pretrained
  representation, in-family or architecture-diverse) are now
  evidence-closed for `sound_repetition`/`word_repetition` normalization
  loss, not merely untested. Two live next steps named: formally cost
  out Stage D (fine-tuning/data acquisition, still gated on
  infrastructure this project doesn't have), and give direction (g)
  (extending the acoustic-native precedent) a real look first, since it's
  cheaper and was sequenced after (a)/(b), not instead of them. No
  change to `main`. → *PAPER_DECISION_LOG.md, "Phase 2 integrative
  conclusion..."*; full write-up `ASR_RESEARCH_TRACK.md` "Phase 2
  results."

- **Arm 3 done: WavLM-Large representation — negative on the primary
  metric, one genuine nuance.** `sound_repetition` at chance
  (d=-0.061, AUC=0.474) — weaker than both CrisperWhisper (d=0.894,
  AUC=0.723) and Arm 2's stock Whisper (AUC=0.680). Layer-depth profile
  genuinely differs (mid-network peak, not last-layer) but never exceeds
  either Whisper variant's peak. A small, small-sample `word_repetition`
  signal (d=0.259) appeared that CrisperWhisper's own Stage B never
  found — flagged as too small to trust standalone. Frame-rate/pooling
  confound resolved by direct verification (WavLM-Large: exactly
  20ms/frame at 16kHz, matching `FRAME_SECONDS`; LibriStutter audio
  natively 16kHz) rather than left as assumed engineering friction. →
  *PAPER_DECISION_LOG.md, "Arm 3 done..."*

- **Arm 2 done: stock whisper-large-v3 encoder, layer sweep — clean
  negative.** Same last-layer-only pattern as CrisperWhisper's own sweep
  (layer 32 AUC=0.680, all others 0.336-0.378, near/below chance).
  Same-population comparison: CrisperWhisper 0.721 vs. stock Whisper
  0.680 — slightly lower, not higher or more distributed. The
  concentration pattern is a Whisper-architecture property, not
  something CrisperWhisper's fine-tuning introduced. → *PAPER_DECISION_
  LOG.md, "Arm 2 done..."*

- **Arm 1 done: stock whisper-large-v3, full pipeline — clean negative.**
  0/36 known-loss positions recovered; normalized-away rate *higher* than
  CrisperWhisper's own baseline (89.5%/88.2% vs. 45.2%/40.5%). A
  materially bigger, more capable model from the same architecture
  family still normalizes these disfluencies away — real evidence
  against "just swap the ASR" as a fix. → *PAPER_DECISION_LOG.md, "Arm 1
  done..."*

- **Phase 2 of the ASR research track formally opened: full design-space
  plan, pre-registered.** Re-opened the complete design space from first
  principles (7 directions: different ASR, different representation,
  hybrid, decoding, fine-tuning, purpose-built, further acoustic-native
  extension) rather than assuming item 10 was automatically correct.
  New, verified literature (catastrophic forgetting/representational
  drift under fine-tuning; WavLM's own paper on its paralinguistic-
  sensitivity design objective; a real hybrid-beats-baselines result,
  arXiv:2605.12387) grounds a pre-registered 3-arm design: stock
  `whisper-large-v3` through Track B and through the layer-sweep
  methodology, plus WavLM-Large's representation. Exact success/failure
  criteria, confounders, costs, and outcome-to-conclusion mapping for
  every arm combination, plus an adversarial self-critique of the plan
  itself. Nothing implemented yet. → *PAPER_DECISION_LOG.md, "Phase 2
  of the ASR research track formally opened..."*; full plan
  `ASR_RESEARCH_TRACK.md`.

- **Integrative reassessment: current architecture's cheap investigation
  is exhausted; item 10 elevated as the next step.** Full evidence
  inventory across the entire track (Track B through decoding), evidence/
  inference/judgment separated. Conclusion: yes, the cheap, representation/
  decoding-only investigation of CrisperWhisper specifically is
  essentially exhausted (every lever tried, two positive-but-limited,
  five negative/inconclusive) — but **not** a recommendation to abandon
  CrisperWhisper, since `ROADMAP.md` item 10 (a second ASR backend) has
  never been run and is the one remaining piece of evidence that would
  actually inform that call. Item 10 elevated to this track's explicit
  next step. A real, unreconciled discrepancy with the literature (deeper
  layers help more, per arXiv:2311.05203 — the opposite of this track's
  own layer-sweep result) named explicitly, not smoothed over. →
  *PAPER_DECISION_LOG.md, "Integrative reassessment: current
  architecture's cheap investigation is exhausted..."*; full reasoning
  `ASR_RESEARCH_TRACK.md`.

- **Decoding-parameter sensitivity (num_beams) done: clean negative.**
  Second of the reassessment's two recommended experiments.
  `num_beams=5` (model's own trained default) recovered 0 of 14 tested
  `sound_repetition`/`word_repetition` positions lost under the live
  app's forced `num_beams=1`; mean WER identical between conditions
  (0.187 vs 0.187). A real cost overrun found via dry run (~5x compute
  for beam=5, not additive) handled by scoping down to 40 raw clips,
  committed before seeing further results. A false-positive bug in the
  recovery heuristic (single-letter words trivially prefix-matching) was
  caught and fixed by a unit test before any real audio ran. Both of the
  reassessment's recommended in-architecture experiments are now done,
  both negative — triggering the full integrative reassessment. →
  *PAPER_DECISION_LOG.md, "Decoding-parameter sensitivity (num_beams)
  done: clean negative"*; full results `ASR_RESEARCH_TRACK.md`.

- **Encoder layer-depth sweep done: the last layer is uniquely
  informative.** First of the reassessment's two recommended
  experiments. One forward pass per clip with `output_hidden_states=True`
  (not one pass per layer) over all 33 CrisperWhisper encoder layers.
  Last layer (32): AUC=0.721, decisively the best — runner-up 0.383,
  most other layers below chance. Verified against a real population
  deviation (18 clips/551 controls vs. Stage B/C's 31/966) via a
  near-identical last-layer AUC on both (0.721 vs 0.723) before
  trusting the result. Closes the layer-depth question: no untapped
  signal in an earlier layer. Decoding-parameter sensitivity is now the
  sole remaining untested in-architecture lever before RQ3/Stage D. →
  *PAPER_DECISION_LOG.md, "Encoder layer-depth sweep done: the last
  layer is uniquely informative"*; full results `ASR_RESEARCH_TRACK.md`.

- **First-principles reassessment: still the right trajectory, not yet
  time to leave CrisperWhisper.** Evidence/inference/judgment kept
  separate. Conclusion: moving to a different/purpose-built ASR now
  would be evidence-free (RQ3 never tested); only one encoder layer,
  threshold-only combination, and a modest sample have actually been
  tried within the current architecture. Two untested, cheap, in-
  architecture experiments identified as the immediate next steps ahead
  of RQ3/Stage D: encoder layer depth and decoding-parameter
  sensitivity. → *PAPER_DECISION_LOG.md, "First-principles reassessment:
  still the right trajectory, not yet time to leave CrisperWhisper"*;
  full reasoning `ASR_RESEARCH_TRACK.md` (new section, top of file).

- **Stage C2 done: Praat voice-quality fusion — clean negative result.**
  Continued from the prior session's handoff. Re-scoped the originally-
  proposed mis-routing fusion candidate (found degenerate/too small,
  n=4) to Praat-derived voice-quality features (pitch, jitter, shimmer,
  HNR), pre-registered, then tested on the same n=19/966 population
  Stage C used. None of five features cleared the AUC>=0.55 screening
  bar (all near chance) — fusion combination correctly not attempted.
  Rules out Praat features specifically for `sound_repetition`; Stage
  C's own encoder-distance conclusion unaffected. Three remaining
  options (scale up the sample / try the mis-routing rule directly /
  cost out Stage D) recorded in the handoff, none chosen yet. →
  *PAPER_DECISION_LOG.md, "Stage C2 done: Praat voice-quality fusion —
  clean negative result"*; full results `ASR_RESEARCH_TRACK.md` §8.

## 2026-08-05 (asr-research branch)

- **End-of-session documentation, consistency, and handoff pass.**
  Stated the project's objective hierarchy explicitly and consistently
  (`CLAUDE.md` new section; reinforced in `ARCHITECTURE.md`/`README.md`'s
  openings): audio is fundamental, ASR is one subsystem, the transcript
  is one evidence source not ground truth, representations are
  complementary and evidence-gated. Full staleness audit across
  `CLAUDE.md`/`HANDOFF.md`/`DOCS.md`/`VALIDATION.md` — corrected several
  "not yet implemented" statements left over from before Stage C
  completed. Wrote a formal end-of-session handoff into `ASR_RESEARCH_
  TRACK.md` (full-day summary, strongest conclusions, open questions, the
  exact proposed next stage and its hypotheses, an ordered next-session
  plan with success/stopping criteria, and explicit risks/recommendations)
  so a future session can continue without re-planning. No new experiment
  started, per explicit instruction. → *PAPER_DECISION_LOG.md, "End-of-
  session documentation, consistency, and handoff pass"*.
- **Stage C done: H1 (duration confound) refuted, H2 (genuine signature)
  supported, H3 (real but not instance-actionable) also supported.**
  Encoder-distance arm AUC=0.723 (clears chance decisively); duration-
  only baseline AUC=0.483 (essentially chance — and a correction to the
  pre-registered assumption's direction: target positions are very
  slightly *shorter*, not longer, than typical). Absolute precision at a
  useful recall is still low (4.7% at 52.6% recall) given realistic class
  imbalance — not ready to ship standalone; a fusion-style revision is
  the evidence-justified next step, not Stage D. Three real bugs in the
  new analysis script caught and fixed via its own safety-check
  assertions before trusting any number. → *PAPER_DECISION_LOG.md,
  "Stage C done: H1 refuted, H2 supported, H3 also supported"*; full
  results `ASR_RESEARCH_TRACK.md` §8.
- **`main`/`asr-research` pushed to GitHub and verified in sync**
  (hashes compared directly, not just trusted push output). Added an
  "Interpretation" section to `ASR_RESEARCH_TRACK.md` §8 before Stage C:
  what Stages A+B have and haven't established (a real aggregate effect
  for `sound_repetition`, not yet instance-level or confound-resolved),
  three competing hypotheses Stage C should distinguish (duration
  confound / genuine signature / real-but-not-actionable), and a
  concrete design consequence — Stage C needs a duration-only baseline
  comparison arm. → *PAPER_DECISION_LOG.md, "Interpretation added before
  Stage C: uncertainty, rationale, competing hypotheses"*.
- **Stage B (representation-level probe) done: mixed result, reported
  honestly.** Pre-registered hypothesis test (positive/negative/
  inconclusive all acceptable): does CrisperWhisper's encoder retain
  discriminative signal at real-ASR positions where transcript-level
  evidence was normalized away (Stage A). `sound_repetition`: **positive**
  (Cohen's d=0.894, n=19, clears the pre-registered d>=0.5 bar).
  `word_repetition`: **inconclusive** (d=0.428, n=17 — the more indirect
  of the two tests, exactly as flagged as plausible in advance). One
  identification bug caught and fixed before trusting the numbers
  (`audio_bytes=None` was silently disabling acoustic-native detectors
  too, not just the classifier gate). Stage C now proceeds scoped to
  `sound_repetition` only, not extended to `word_repetition` on this
  evidence. → *PAPER_DECISION_LOG.md, "Stage B (representation-level
  probe) done: mixed result, reported as such"*; full results
  `ASR_RESEARCH_TRACK.md` §8.
- **`asr-research` branch created; Stage A (systematic information-loss
  audit) done.** Categorized all 186 disfluent ground-truth positions in
  the 120-clip Track B sample into four causes (normalized away,
  mis-routed, genuine ASR error, ASR error + coincidental type). For
  `sound_repetition`/`word_repetition`, ~53% of losses happen even at
  correctly-transcribed positions, confirming item 19's finding
  generalizes. Found `word_repetition`'s specific mechanism: 22/23
  "correct" positions have the other half of the repeated pair deleted
  by ASR — different from `sound_repetition`'s fragment-token loss. A
  bug in the analysis script itself was caught and fixed before trusting
  the numbers (reconciled against the official scored table). One
  unrelated detector bug found incidentally (a genuine triple repeat
  missed) → new `ROADMAP.md` item 21, for `main`, not this track. →
  *PAPER_DECISION_LOG.md, "`asr-research` branch created; Stage A
  (systematic information-loss audit) done"*; full results
  `ASR_RESEARCH_TRACK.md` §8.

## 2026-08-05

- **A separate research track opened: `ASR_RESEARCH_TRACK.md`,
  `asr-research` branch (not yet created).** Item 19's finding — real
  ASR structurally discards information certain disfluency types need —
  judged a bigger-than-one-item checkpoint. New charter document: the
  reframed core question ("how do we preserve the speech-production
  information conventional ASR intentionally removes"), a real 13-source
  literature review (field-level ASR bias, CrisperWhisper's own design,
  continual-learning adaptation, multitask joint training, bypassing
  decoded text, SSL/encoder probing — including one paper independently
  corroborating this project's own Stage 1 result), six architectural
  directions explored without commitment, and a phased evidence-gated
  research plan (Stages A-E) with explicit criteria for when a
  purpose-built ASR/representation would be justified. Does not reopen
  `PHASE_3_ARCHITECTURE_REVIEW.md`'s two-stage-architecture conclusion.
  `main` stays stable; work happens on the new branch once created. →
  *PAPER_DECISION_LOG.md, "A separate research track opened:
  `ASR_RESEARCH_TRACK.md`, `asr-research` branch"*.
- **Track B validation of the shipped repetition-classifier gate (item
  19 executed)**: re-ran the existing 120-clip speaker-stratified sample,
  gate on/off, reusing cached ASR output (zero new inference). Gate's
  mechanism confirmed safe on real ASR text (37% FP reduction on
  `word_repetition` candidates, zero recall cost) but real-world impact
  negligible (`Any` F1 0.082 -> 0.085) because real ASR starves both
  gated types of candidates (`word_repetition` ~7x fewer/clip than
  Track A; `sound_repetition` zero candidates across all 120 clips,
  either condition). Hand-check found the specific mechanism: the
  fragment-token candidate check's input assumption essentially never
  holds for real (fluency-normalizing) ASR output. Gate stays enabled
  (shipped default unchanged). New `ROADMAP.md` item 20 (redesign
  candidate generation for real ASR) now the top-priority open item. →
  *PAPER_DECISION_LOG.md, "Track B validation of the shipped repetition-
  classifier gate (item 19 executed)"*; full results `VALIDATION.md`
  §14/§14.1.
- **First-principles reassessment written into `ROADMAP.md`**: re-checked
  the whole project against its stated objective from scratch. Confirmed
  by direct code inspection that Phase 3's entire shipped result (the
  repetition classifier, item 17) was validated only on Track A
  (ground-truth transcript tokens — `track_a.py`'s `evaluate()` never
  touches real ASR output), and that no Phase 2/3 detector change has
  been re-checked against Track B since it was last run (2026-08-04,
  before any of them shipped). New `ROADMAP.md` item 19 (re-run Track B
  before further Track-A-only work); item 10 and the deferred-learned-
  tier bullet re-flagged in place. Also trimmed ~5 duplicated entries in
  `ROADMAP.md`'s "Completed" section down to pointers. → *PAPER_DECISION_
  LOG.md, "First-principles reassessment of the whole project, written
  into ROADMAP.md"*.
- **Decision executed in full: (S1, M3) implemented, benchmarked, and
  shipped as the new default.** Trained the final classifier (`models/
  repetition_corroboration_classifier.npz`, this project's first
  internally-trained shipped artifact), refactored shared encoder-
  extraction code into a new core module (`profiling/encoder_
  embedding.py`) to preserve the evaluation/core-app boundary, wired the
  gate into `detect.py` (`require_repetition_classifier_confirmation`,
  default `true`), and benchmarked honestly via out-of-fold cross-
  validated predictions (not the final model scored on its own training
  data): `Any` (word+sound repetition) F1 0.631 -> 0.890 (89% FP
  reduction, 10% recall cost). **Two real bugs caught by the existing
  fast test suite before any result was trusted**: eager encoder loading
  that would have broken the real-model-free unit test suite, and a
  second, sharper version of the same bug (gate evaluated for every
  token pair, not just actual candidates) — both fixed, verified on real
  audio (a genuine TP still fires, a genuine FP is suppressed), full
  suite 66/66. A real live-app latency cost (a second ~30-90s encoder
  pass) is documented, not hidden. → *PAPER_DECISION_LOG.md, "Decision
  executed in full: (S1, M3) implemented, benchmarked, and shipped..."*;
  full results in `VALIDATION.md` §13.
- **Executing §12.6 (continued from 2026-08-04): 250-clip result —
  the classifier's advantage held and grew, decision rule satisfied.**
  See the 2026-08-04 section below for the pre-run scoping/checkpointing
  work; the run itself completed and was analyzed today. `word_repetition`
  gap +0.161 -> +0.223, `Any` gap +0.133 -> +0.178, 5/5 folds, fully
  non-overlapping ranges. → *VALIDATION.md* §12.6.2.

## 2026-08-04

- **Real cost of no checkpointing: a run was killed mid-way with zero
  recoverable progress; checkpointing added and verified before
  restarting.** The 250-clip collection was interrupted; confirmed zero
  progress was recoverable (output was only ever written at the end).
  Rewrote the collector to checkpoint after every clip and genuinely
  resume (skip already-processed clips, correctly handle a clip that
  legitimately produces zero records) rather than restart from scratch.
  Verified with a synthetic round-trip test and a real 2-then-4-clip
  collection/resume run before trusting it with the real dataset. →
  *PAPER_DECISION_LOG.md, "Real cost of no checkpointing..."*
- **Executing §12.6: nested-CV regularization tuning implemented; a real
  cost-escalation finding changed the run's scope before launching it.**
  Replaced the fixed L2=5.0 with per-outer-fold nested CV selection
  (smoke-tested on synthetic data first). Before scaling up, re-examined
  the 90-clip run's own per-clip timings and found a real ~2.7x
  slowdown within that single run (31s/clip -> 85s/clip, likely thermal
  throttling on the laptop CPU) — undersold by the flat-average
  extrapolation §12.6 was written against. Scoped the follow-up run to
  250 clips (not the unbounded full 499) as a result, with reasoning
  recorded rather than silently picked. → *PAPER_DECISION_LOG.md,
  "Executing §12.6: nested-CV regularization tuning implemented..."*;
  addendum in `VALIDATION.md` §12.6.1.

- **Standing principle established: architectural decisions are
  evidence-constrained, not preservation-constrained (`CLAUDE.md` rule
  8) — applied immediately to today's own open question.** Current
  architecture and every prior decision are hypotheses, not defaults to
  protect; simplicity/interpretability/ML/rule-based/pretrained/trained
  are all engineering choices, none privileged by default. Applied to
  the classifier-vs-threshold question: decided the evidence is real and
  positive enough to shift this project's *working expectation* toward
  the learned signal, but not yet decisive enough to ship (three named,
  evidence-based reasons — small-sample classifier variance, untuned
  regularization, reconstructed-timing overfitting risk — not a
  simplicity preference). Pre-registered the specific follow-up
  (`VALIDATION.md` §12.6: larger scale, nested-CV-tuned regularization,
  a fixed decision rule) rather than guessing or stalling. → *PAPER_
  DECISION_LOG.md, "Standing principle established: architectural
  decisions are evidence-constrained..."*
- **Corroboration-mechanism comparison result: the classifier wins
  clearly — the opposite of this project's own pre-registered
  prediction.** 5-fold CV: (S1, M1) threshold F1=0.588/0.755
  (word_repetition/Any) vs. **(S1, M3) classifier F1=0.749/0.888** —
  beating the threshold in 5/5 folds in both slices, verified directly
  before trusting the claim. The new repeat-pair-similarity signal (S2)
  did not outperform the threshold either. `VALIDATION.md` §12.4 had
  predicted the opposite (large Cohen's d -> limited classifier
  headroom) — reported as a contradicted prediction, not reframed as
  expected. Real limitations named alongside the strong result (modest
  sample, untuned regularization, reconstructed-timing data). Does not
  itself authorize implementation — the real costs of shipping this
  project's first trained-model artifact remain a separate decision. →
  *PAPER_DECISION_LOG.md, "Corroboration-mechanism comparison result:
  the classifier wins clearly..."*; full results in `VALIDATION.md`
  §12.5.
- **Corroboration-mechanism review: neither threshold nor classifier
  assumed as the answer — a broader candidate space reviewed, a
  comparison pre-registered.** Separated "which signal" from "which
  decision mechanism" (previously conflated), evaluated against all nine
  dimensions the owner named, found a new untested signal candidate
  (repeat-pair self-similarity) and corrected a framing error (the
  architecture never actually shared one strategy across types — it
  already uses type-specific signals under one shared fusion mechanism).
  Pre-registered a 5-fold, clip-split cross-validated comparison of
  three candidates (`VALIDATION.md` §12). Built the measurement
  infrastructure (raw-embedding collection, numpy logistic regression —
  not scikit-learn, to avoid a new dependency) and smoke-tested it on
  synthetic data before spending real compute. Full suite 59/59. Real
  comparison running. → *PAPER_DECISION_LOG.md, "Corroboration-mechanism
  review: neither 'threshold' nor 'classifier' assumed..."*; full review
  in `PHASE_3_ARCHITECTURE_REVIEW.md` §9.
- **Stage 1 result: a clear, stable, large-effect-size signal.**
  90-clip run: `word_repetition` Cohen's d = +1.047, `Any` d = +1.116 —
  both well past the revised `|d| >= 0.5` bar, and stable in sign and
  magnitude from the 30-clip pilot. CrisperWhisper's own encoder
  representation carries information the ASR transcript alone does not
  (a genuine vs. coincidental repeated word produce identical tokens but
  measurably different embeddings). A real, unconfirmed duration/word-
  identity confound named explicitly, not hidden by the headline result.
  Per pre-registration, this does not itself authorize implementing
  Stage 2 — that remains a separate, explicit decision. → *PAPER_DECISION_
  LOG.md, "Stage 1 result: a clear, stable, large-effect-size signal..."*;
  full results in `VALIDATION.md` §11.6.
- **30-clip Stage 1 pilot: a real, non-trivial gap found (`Any` +0.0895,
  15-20x the VAD/Praat null result's magnitude) — but a real gap in the
  pre-registration's own success criteria found first, and fixed before
  drawing any conclusion.** The original criteria specified a direction,
  not a rigorous way to judge magnitude against sample noise. Added
  Cohen's d + stdev to `encoder_distance_stats()` (mirroring why Wilson
  intervals were added in Phase 2), revised the success bar to require
  `|Cohen's d| >= 0.5`, and — since the 30-clip pilot didn't save raw
  per-event distances needed to compute it retroactively — launched a
  90-clip run directly rather than re-running 30 twice. → *PAPER_DECISION_
  LOG.md, "30-clip pilot: a real, non-trivial gap found; a real gap in
  the pre-registration's own success criteria found and fixed..."*
- **Stage 1 implemented exactly as pre-registered.**
  `profiling/evaluation/encoder_features.py` (extraction/pooling/
  distance), `metrics.encoder_distance_stats()`, `report.
  format_encoder_distance_stats()`, `run_encoder_signal_stage1.py`. 12 new
  unit tests, real-model smoke test passed. **Real cost discovered by
  measurement, not estimated**: the encoder pass alone is ~38s/clip
  (Whisper's fixed 30s-window encoder pass dominates transcription
  latency regardless of clip length) — a full 499-clip run is ~6 hours,
  not stated in the original pre-registration. Runner now defaults to a
  30-clip pilot, mirroring Track B's own pilot-then-scale precedent, not
  a methodology change. A real bug (a literal em-dash in a new `print()`
  call) was caught immediately by the existing ASCII lint rule. Full
  suite 57/57. → *PAPER_DECISION_LOG.md, "Stage 1 implemented exactly as
  pre-registered; a real cost discovered and priced..."*
- **Stage 1 (encoder-representation corroboration signal) pre-registered
  before any implementation.** Exact methodology fixed in `VALIDATION.md`
  §11: extract CrisperWhisper's last-layer encoder hidden states per
  event span, test a zero-training cosine-distance-to-fluent-centroid
  signal (reusing Phase 2's `confidence_stats()` TP-vs-FP-gap shape) on
  the 499-clip LibriStutter sample, restricted to `word_repetition`/
  `sound_repetition`/`filler`. Success criteria fixed in advance,
  including that a null result triggers the already-agreed Stage 1b
  escalation (frozen WavLM-Large) rather than being treated as a dead
  end. A real methodological subtlety caught during this pass: testing
  this signal requires running CrisperWhisper's encoder over real audio
  even under Track A, which normally bypasses ASR entirely. No code
  written yet. → *PAPER_DECISION_LOG.md, "Stage 1 (encoder-representation
  corroboration signal) pre-registered before implementation"*.
- **Encoder-reuse candidate refined to a specific, staged mechanism, then
  adversarially challenged from a clean-slate design stance.** Refined:
  compared cross-attention/encoder-hidden-states/decoder-confidence on
  accessibility (all free — already computed by the existing
  `pipeline()` call, just discarded) and evidence; encoder hidden states
  won, staged as zero-training corroboration first, trained classifier
  only later with explicit go-ahead. Then challenged: is an ASR-trained
  encoder even the right representation family vs. a purpose-built SSL
  encoder (wav2vec2/HuBERT/WavLM/SeamlessM4T)? Found the objection real
  and unresolved (no direct disfluency-specific head-to-head exists), so
  the plan now has an explicit, pre-stated escalation trigger to a frozen
  WavLM-Large pass if the cheap first test fails, rather than picking a
  winner by argument. → *PAPER_DECISION_LOG.md, "Encoder-reuse refined to
  a specific, staged mechanism; then adversarially challenged..."*; full
  reasoning in `PHASE_3_ARCHITECTURE_REVIEW.md` §5.1/§8.
- **Pre-Phase-3 architecture review: is ASR-first the right foundation? Kept, with a scoped extension identified.**
  Fresh 2024-2026 literature pass (newer Whisper models, disfluency-trained
  ASR, SSL representations, acoustic embeddings, hybrid/end-to-end
  architectures, joint ASR+detection training) cross-referenced against
  this project's own Track A/B findings. **Conclusion: two-stage
  architecture kept** (no alternative found is decisively more accurate at
  this project's task granularity without infrastructure this project
  doesn't have) **but a real next step identified: extend
  audio-native-primary detection (already proven for `block`/
  `prolongation`) to `word_repetition`/`sound_repetition`/`filler` via
  CrisperWhisper's own encoder representations** — the lowest-cost
  "richer representation" option since the forward pass already runs.
  Not implemented — a Phase 3 candidate requiring its own pre-registration
  first. → *PAPER_DECISION_LOG.md, "Pre-Phase-3 architecture review: is
  ASR-first the right foundation?"*; full review in
  `PHASE_3_ARCHITECTURE_REVIEW.md`.
- **Prolongation redesign decided: Praat-gating adopted as the new
  default, rate-normalization rejected.** Pre-registered 4-variant
  ablation (13-variant full re-run) against the 499-clip real-audio
  sample. `prolong_praat_gated` was the only variant of 13 to improve
  both `Any` F1 (0.835->0.888) and prolongation-specific F1 (0.064->
  0.084) simultaneously — `config.yaml`'s
  `require_praat_stability_for_prolongation` flipped to `true`.
  `prolong_rate_normalized` regressed badly on both metrics (`Any` F1
  0.835->0.347) — `use_rate_normalized_prolongation` stays `false`. Full
  suite 45/45 after the config change. → *PAPER_DECISION_LOG.md,
  "Prolongation redesign ablation run; Praat-gating adopted as the new
  default, rate-normalization rejected"*; full results table in
  `VALIDATION.md` §9.5.1.
- **Explicit scope decisions on remaining lower-priority Phase 2 items (9,
  14-16): deferred to Phase 3 with reasoning, not silently dropped.**
  Mirrors how Phase 1 closed. → *PAPER_DECISION_LOG.md, "Explicit scope
  decisions on remaining lower-priority Phase 2 items (9, 14-16)..."*
- **ASCII-console-output lint rule built (`ROADMAP.md` item 13) — caught
  two real, previously-unnoticed non-ASCII characters in
  `benchmark_asr.py`'s `print()` calls on its first run.** AST-based
  check under `tests/test_ascii_console_output.py`, scoped to `print()`
  literals under `profiling/` (not a whole-file scan, which would
  false-positive on 31 files' legitimate docstring em-dashes). Full suite
  now 45/45. → *PAPER_DECISION_LOG.md, "ASCII-console-output lint rule
  built; caught two real, previously-unnoticed violations immediately"*
- **UCLASS annotation-schema check: inconclusive from public sources, the
  "severity annotation" claim behind item 11 is unsubstantiated by its own
  cited primary source.** Checked the primary UCLASS paper (Howell et al.
  2009), its externally-referenced annotation page (link rot — TLS cert
  no longer matches), and UCLASS's current raw file listing (files
  present, no methodology doc). → *PAPER_DECISION_LOG.md, "UCLASS
  annotation-schema verification: inconclusive from public sources..."*;
  full trace in `PHASE_2_RESEARCH_PLAN.md` §5 point 3's addendum.
- **Confidence-sensitive metric run against real data: VAD/Praat's
  designed confidence effect is not showing up (negative-to-null
  result).** `confidence_stats()` run on the full 499-clip real-audio
  LibriStutter sample under production config — TP-vs-FP confidence gap
  ~zero everywhere measurable, slightly negative for the combined `Any`
  label. Closes `VALIDATION.md` §9.3's open question (whether VAD/Praat
  corroboration's confidence effect was real but metric-invisible) with a
  real measurement rather than a caveat. → *PAPER_DECISION_LOG.md,
  "Confidence-sensitive metric run against real data..."*; full table in
  `VALIDATION.md` §9.3.1.
- **Wilson 95% confidence intervals applied to `R_B|preserved_ctx1`'s
  n=2/n=7/n=15 recall figures.** n=7 and n=15 intervals overlap
  substantially — makes concrete that the earlier "1.0 recall" point
  estimate was never precise enough to rule out the later 0.667 estimate.
  → *PAPER_DECISION_LOG.md, "Wilson confidence intervals applied to the
  project's own extreme-small-n numbers"*; table in `VALIDATION.md`
  §8.4.3.
- **Negative result, documented honestly: a "word-sandwiched repetition"
  detector extension was built, measured, and reverted.** Built a
  hypothesis-side-contiguity metric first (`ROADMAP.md` item 4's safer
  candidate fix), reusing the existing Track B cache at zero ASR cost —
  found only 1 truly close-range (`gap=1`) recall-miss case and 2
  type-accuracy cases out of 120 clips (n=3 total), too thin to decide
  from alone. Implemented a narrow, conservative extension anyway (exact-
  match only, lower confidence) and let a full 499-clip Track A benchmark
  decide empirically, as pre-registered. **Result: `Any` F1 regressed
  0.835→0.793 with zero new true positives** (Track A can only show this
  fix's cost, not its benefit, since it has no ASR to corrupt); Track B
  showed only +1 TP at a cost of +24-29 new FP. **Reverted** — a
  regression test now locks in the correct (non-firing) behavior. → 
  *PAPER_DECISION_LOG.md, "Hypothesis-side-contiguity metric built; a
  narrow detector extension implemented, measured, and reverted (negative
  result)"; full writeup in `VALIDATION.md` §8.4.4*
- **`sound_repetition` fragment-ordering bug fixed — first real detector
  code change of Phase 2 — recall 0.000 → 0.920.** Root cause was deeper
  than previously documented: a reconstructed fragment ("word" + trailing
  `-`) normalizes identically to its complete-word counterpart, so the
  detector's exact-match `word_repetition` check intercepted it in *both*
  orderings, not just the one previously flagged — a simple reverse-order
  check (the originally proposed fix) would not have worked. Fixed by
  reordering the fragment-pair check ahead of the exact-match check.
  Pre-registered before implementing; measured on the full 499-clip
  benchmark: `sound_repetition` recall 0%→92.0%, `word_repetition` FP
  −188 (TP/FN unchanged — the bug inflated FPs, not TPs), `Any` (combined)
  label **exactly unchanged** (byte-for-byte), confirming a pure
  type-reclassification fix. Also found and fixed, alongside this: Track
  B's per-clip cache was storing stale detector output, meaning every
  future Track B run would have silently kept using the pre-fix
  classification — cache now stores only ASR output, detector output is
  always recomputed fresh. Full suite 41/41 (was 39); Track B self-test
  19/19 (was 18). → *PAPER_DECISION_LOG.md, "sound_repetition
  fragment-ordering fix: root cause deeper than documented, fixed and
  measured; a related cache-staleness bug found and fixed alongside it";
  full results in `VALIDATION.md` §8.2.1*
- **Roadmap reprioritized by evidence, not implementation order: the
  second-ASR-backend gate lifted for three detector-side items.**
  Explicitly re-examined whether `ROADMAP.md`'s "validate against a second
  ASR backend" item was still a necessary prerequisite for the
  prolongation redesign and the newly-elevated `sound_repetition`/
  `phrase_repetition` fixes, now that speaker-stratified Track B (above)
  has completed. **Decision: no** — traced each item's evidence back to
  its source and found all three rest on Track A and/or peer-reviewed
  literature that never involved ASR at all, so no second-backend result
  could invalidate them. The second-ASR-backend/FluencyBank item remains
  valuable (it still tests whether the *general* "ASR is the bottleneck"
  claim holds beyond CrisperWhisper) but is re-scoped from blocking
  prerequisite to parallel generalization check, moved later in the
  priority list. → *PAPER_DECISION_LOG.md, "Is a second ASR backend still
  necessary before detector-side work, or does current evidence already
  justify proceeding?"; reordered priorities in `ROADMAP.md`*
- **"Whisper did not predict an ending timestamp" console warning
  investigated and confirmed external.** Verified, not assumed: grepped
  our own code (no match), found the trigger in `transformers`' installed
  source directly; ruled out our `max_new_tokens` budget as the cause by
  measuring actual generated-token counts against budget on cached real
  transcriptions (consistently ~30-50% utilization, nowhere near the cap).
  No fix needed — affected words aren't lost, and this project's own code
  already handles the resulting missing timestamp gracefully. → 
  *PAPER_DECISION_LOG.md, "Whisper 'did not predict an ending timestamp'
  warning: investigated, confirmed external"; full account in
  `ARCHITECTURE.md` §3*
- **Speaker-stratified Track B: the "~0% detector-attributable" finding
  revised, not confirmed.** Directly resolved the Phase 1 closing review's
  speaker-clustering caveat — the earlier confirmed conclusion had only
  been measured on 7 of 40 speakers. Pre-registered a speaker-stratified
  sampling method (round-robin across all 40 speakers, 120 clips) before
  running, explicitly allowing "the finding weakens" as a valid outcome.
  **It weakened**: `R_B|preserved_ctx1` recall dropped from 1.0 (7
  speakers) to 0.667 (40 speakers); decomposition revised from ~0%/100% to
  **35.1% detector-attributable / 64.9% ASR-attributable**. ASR-fidelity
  remains the majority driver — not reversed. Hand-traced all 5 misses:
  every one is `sound_repetition` or `phrase_repetition`, both types with
  *already-known* structural gaps unrelated to ASR context (fragment-
  ordering mismatch; LibriStutter single-word-reconstruction limitation).
  **`word_repetition` itself remains at 100% recall given intact context,
  across 10 instances and two independent sampling methods** — the
  earlier small samples were `word_repetition`-heavy by chance, not by
  selection. Historical §8.4.1/§8.4.2 numbers kept as an accurate record,
  not rewritten — dated pointers added instead. → *PAPER_DECISION_LOG.md,
  "Speaker-stratified Track B: the '~0% detector-attributable' finding
  revised, not confirmed"; full results in `VALIDATION.md` §8.4.3*

## 2026-08-03

- **Per-type definition audit: literature vs. dataset vs. implementation.**
  For each of the 7 disfluency types, checked whether our code's exact
  operational trigger detects the phenomenon as the clinical literature
  defines it, or only approximates the dataset's own labeling shortcut.
  Two real, now-quantified gaps found — both already top Phase 2
  priorities, now sharper: **`prolongation`'s effective ~1.0s threshold is
  ~2–4× higher than the literature's rate-normalized standard**
  (`T_min = 1.2/speaking_rate`, Esmaili et al. 2017), a deliberate
  real-mic-tuned trade-off, not derived from the rate-normalized
  definition; **`block`'s silence-only rule tests a necessary-but-not-
  sufficient proxy for the clinical (effort/struggle-based) definition** —
  SEP-28k's own schema has a separate `NaturalPause` column, meaning its
  human annotators already make a distinction this detector structurally
  cannot. Every other type's simplification (filler word-list vs.
  discourse judgment, no iteration counting for sound_repetition,
  stutter_marker having no external definition at all) matches a
  simplification the benchmark datasets themselves also make — documented,
  not actioned, since none are validatable against current data.
  `phrase_repetition` found to be an unusual case where this project's
  implementation is arguably *more* faithful to the literature than any
  available dataset's own label. No code changed — research/documentation
  only. → *PAPER_DECISION_LOG.md, "Per-type definition audit: literature
  vs. dataset vs. implementation"; full audit in `PHASE_2_RESEARCH_PLAN.md`
  §10*
- **Adversarial self-review of the Phase 2 plan, then its first
  implementation milestone.** Actively tried to disprove
  `PHASE_2_RESEARCH_PLAN.md`'s own conclusions before implementing anything.
  Found the prolongation-first case was leaning on one non-peer-reviewed
  preprint — replaced with two independent, peer-reviewed sources (Esmaili
  et al. 2017; a 14-study PMC systematic review) that confirm the same
  conclusion more solidly. Directly addressed the rule-based-vs-deep-
  learning tension the field's own trend raises: explainability is a
  co-equal stated project objective, so staying interpretable-first is a
  deliberate, justified choice, not an oversight. **No direction found
  strong enough to change the plan's ordering.** Then implemented Step 1:
  `word_repetition` events now carry a `syllable_count`/`likely_sld` tag
  (monosyllabic = stuttering-like, polysyllabic = ordinary disfluency, per
  the clinical SLD/OD literature) — purely additive, surfaced in the app's
  Event table with an explicit "not a validated clinical measure" caption;
  `phrase_repetition`/`stutter_marker` now explicitly labeled
  "not validated against any public dataset" in `README.md` and the UI; the
  silent-only `block` limitation documented in `ARCHITECTURE.md`.
  **Benchmarked against the frozen Phase 1 Track A baseline — confirmed
  byte-for-byte identical** (`Any` F1 0.835, unchanged across all 499
  clips), proving the change is purely additive. New test added, full
  suite 39/39. → *PAPER_DECISION_LOG.md, "Adversarial self-review of the
  Phase 2 plan, and its first implementation milestone (Step 1)"; full
  review in `PHASE_2_RESEARCH_PLAN.md` §9*
- **Phase 2 opening literature review: taxonomy checked against the
  field, not assumed optimal.** Before any Phase 2 implementation,
  reviewed clinical speech-pathology taxonomy (stuttering-like vs. other
  disfluencies), computational detection literature, and dataset
  annotation conventions (SEP-28k/FluencyBank/KSoF/UCLASS) against a wider
  sweep than Phase 1's original comparison. **Core 5-type taxonomy
  confirmed scientifically sound — no redesign needed.** Two real, newly
  identified gaps: `word_repetition` ignores a clinically meaningful
  monosyllabic/polysyllabic split (cheap, additive fix available); `block`
  detection is confirmed silent-only, missing the literature's "audible/
  struggle" sub-type entirely (not currently validatable against any
  available dataset, so deliberately not built yet). **Prolongation
  identified as the highest-confidence Phase 2 detector-side target** —
  three independent sources converge on it (Phase 1's own ablation,
  the literature's rule-based-detection results, and full dataset
  support). Two external preprints independently corroborate findings
  Phase 1 already reached on its own (type-confusion between adjacent
  disfluencies; synthetic-data generalization risk). Structured Phase 2
  plan: taxonomy/documentation refinements first (cheap, immediate),
  ASR-backend/speaker-diversity validation in parallel (already Phase 1's
  top priority, reinforced not superseded), then the prolongation redesign
  gated on that validation's outcome. No code changed — research and
  planning only. → *PAPER_DECISION_LOG.md, "Phase 2 opening literature
  review: is our taxonomy scientifically sound?"; full review in
  `PHASE_2_RESEARCH_PLAN.md`*
- **Phase 1 (Validation, Benchmarking, Analysis) formally closed.** Critical
  review of the full evaluation methodology at the project owner's request:
  fixed a real gap (Track B's decomposition used an *approximated* Track A
  recall from the full 499-clip sample instead of the exact matched-clip
  subset — now computed exactly, turning the context-strict
  detector-attributable share from "≈0%" into **exactly 0%**); identified
  two new, real generalization limits on the confirmed Track B conclusion
  (the clip subset is speaker-clustered — 7/40 speakers at n=90 — and the
  whole finding rests on one ASR backend + one synthetic dataset), neither
  fixed this session but both promoted to top Phase 2 priorities; fixed
  several cases of stale documentation (`README.md`'s config table showed
  the pre-Part-D `prolongation_min_seconds` default; `ARCHITECTURE.md`
  pointed at the retired `august.md`; `VALIDATION.md` described Track B as
  "not yet built" in several places after it was built). Added `CLAUDE.md`
  as a stable session-start orientation file. Closing summary:
  `PHASE_1_SUMMARY.md` (new). → *PAPER_DECISION_LOG.md, "Phase 1 closing
  review: exact-subset Track A recall, critical methodology review,
  documentation consolidation"; full critical review in `VALIDATION.md`
  §7.2–§7.4*
- **Track B scaled 30 → 90 clips — context-strict finding CONFIRMED, now a
  major research conclusion.** `R_B|preserved_ctx1` recall = 1.0 again, this
  time on n=7 positive instances (up from n=2), with a stable ~5.5%
  sample-attrition rate (127 disfluent instances → 7 surviving both the
  word-correct and context-correct filter). **The detector's binary
  disfluent/clean judgment is effectively perfect given intact ASR input;
  the real-world recall shortfall is overwhelmingly an ASR-fidelity
  problem, not a detector problem.** A second, sharpened finding: even when
  binary detection succeeds, exact type labeling for `word_repetition`/
  `sound_repetition` is frequently wrong (2/7 exact matches) — a real,
  scoped detector-side issue distinct from the recall question. Synthesis:
  future development priority shifts toward ASR robustness on disfluent
  speech + this scoped type-classification fix; detector-recall tuning is
  explicitly de-prioritized. `ROADMAP.md` restructured accordingly. →
  *PAPER_DECISION_LOG.md, "Track B scaled 30 → 90 clips: context-strict
  finding confirmed, treated as a major conclusion"; full results in
  `VALIDATION.md` §8.4.2*
- **Context-strict preserved-subset scoring implemented and run — reverses
  the previous detector/ASR attribution split.** Pre-registered in
  `VALIDATION.md` §5.1's addendum before implementing (same discipline as
  Track B itself). `R_B|preserved_ctx1` requires both a disfluent word *and*
  its immediately preceding word to survive ASR intact. Result: **~95%
  detector-attributable / ~5% ASR-attributable → ~0% detector-attributable
  / ~100% ASR-attributable** (`R_B|preserved_ctx1` = 1.0, n=2 — only 2 of 48
  disfluent instances survive the stricter filter, itself a finding about
  how thoroughly ASR corrupts local context). Added per-clip caching to
  `track_b.py` so this and future metric refinements skip re-running
  CrisperWhisper. A further nuance: even the 2 surviving instances got the
  wrong *exact* type label (`word_repetition` predicted as
  `phrase_repetition`/`block`) — traced to a hypothesis-side-contiguity gap
  the new metric still doesn't check, flagged as the next candidate
  refinement, not implemented yet. → *PAPER_DECISION_LOG.md, "Context-strict
  preserved-subset scoring implemented and run"; full results in
  `VALIDATION.md` §8.4.1*
- **Track B implemented and piloted — evaluation protocol pre-registered
  first, exactly as requested.** Wrote the full protocol into
  `VALIDATION.md` §5.1 (Levenshtein alignment with a disfluent-word cost
  bias; exact metrics at three levels — Track A, Track B ASR-preserved
  subset, Track B overall; the ASR-vs-detector decomposition formula; two
  explicit success criteria) *before* writing any Track B code. Built
  `alignment.py` + `track_b.py`, self-tested (10 checks against
  hand-computed expectations), then ran a 30-clip pilot with real
  CrisperWhisper ASR. **Headline finding: Track A's ~99% recall drops to
  ~4–9% under real ASR conditions** — the most important result this
  evaluation effort has produced, and direct proof Track A alone
  overstated real-world performance. Hand-verified alignment quality on 10
  clips (the pre-registered methodological gate — passed) and found a real
  limitation in the process: ASR can correctly transcribe a disfluent word
  while still breaking the *adjacent* context a detector check depends on,
  meaning the mechanical 95%/5% detector/ASR split likely overstates
  genuine detector-only failure — recorded as a dated addendum to §5.1, not
  smoothed over. → *PAPER_DECISION_LOG.md, "Track B evaluation protocol
  pre-registered before implementation" and "Track B implemented and
  piloted exactly per the pre-registered protocol"; full results in
  `VALIDATION.md` §8.4*
- **Complete ablation study against the baseline** (10 variants, same 499
  real clips): `prolongation_min_seconds` dominates by an order of
  magnitude (`Any` F1 range 0.639–0.933 across the sweep); `fusion_weights.
  acoustic` has a small real effect (+0.003 F1, saturating at 2×); Silero
  VAD and Praat corroboration both measured **zero** effect on the
  presence/absence metric used — traced to a real methodological gap (the
  metric can't see confidence adjustments, their designed effect), not
  evidence they're unhelpful; speaker calibration not applicable to this
  dataset. Added a `use_praat` config toggle to `profiling/acoustic.py`
  (defaults `True`, verified no behavior change) as ablation
  infrastructure. → *PAPER_DECISION_LOG.md, "Full ablation study against the
  baseline"; full tables in `VALIDATION.md` §9*
- **First research baseline established: audio-native-layer evaluated
  against labeled ground truth for the first time.** Re-examined the
  validation strategy against the project's core objective and prioritized
  closing a self-identified gap (the audio-native detection layer had never
  been evaluated at all — Track A always ran with `audio_bytes=None`) ahead
  of acquiring SEP-28k's audio, the previously-assumed next step. Built
  audio-enabled Track A support (`LabeledClip.audio_bytes`,
  `load_libristutter_*_with_audio`, FLAC→16kHz-WAV conversion via new
  `soundfile` dependency), downloaded matching real audio for the same
  499-clip sample (499/499, zero failures). **Found and fixed a real bug**
  before trusting the result: `soundfile.read(..., dtype="int16")` silently
  zeroed real LibriStutter FLAC files; caught by direct RMS audit, not by
  the metrics table looking wrong. Corrected result: `Any`-label F1
  0.773 → 0.835 with audio (157 fewer false positives, ~0 recall cost) —
  real evidence the architecture's design goal is being met.
  `prolongation` recall dropped (37→21 TP), mechanistically traced to
  voiced-duration trimming on reconstructed-token timing, not a detector
  bug. This checkpoint is now the project's first established research
  baseline — future algorithm changes are expected to be measured against
  it. → *PAPER_DECISION_LOG.md, "Audio-enabled evaluation", "Bug found and
  fixed: `_flac_bytes_to_wav16k` silently produced silent audio", "First
  audio-native-layer result (LibriStutter, 499 clips, corrected) — baseline
  established"; full results in `VALIDATION.md` §8.3*
- **First real Track A result**: 499 real LibriStutter clips (17,970 tokens),
  scored via `profiling/evaluation/track_a.py`. Headline: `Any` combined
  label 99.1% recall / 63.3% precision. Per-type numbers required real
  interpretation — a direct FP audit confirmed `word_repetition`'s alarming
  22.2% raw precision is mostly a reconstruction artifact (87.1% against
  clean speech only), while `prolongation`'s poor precision (4.8%) is
  substantially real even after the same correction. `sound_repetition`
  found at 0% recall, traced to a structural fragment-ordering mismatch
  between the reconstruction and the detector's fragment check — flagged
  for investigation, not fixed this round. → *PAPER_DECISION_LOG.md, "First
  real Track A result (LibriStutter, 499 clips)"; full results in
  `VALIDATION.md` §8.2*
- **Real LibriStutter/SEP-28k schemas confirmed against downloaded files —
  one loader bug found and fixed, one loader newly built.** LibriStutter's
  real format uses a `"STUTTER"` placeholder row (not a label on a real
  word, as originally assumed) — `load_libristutter_csv` fixed to
  reconstruct it into a plausible token instead of feeding the literal word
  "STUTTER" to the detector. `load_sep28k_labels` written and verified
  against the real, complete 28,177-row file (schema confirmed: clip-level,
  counts-out-of-3-annotators, no transcript). → *PAPER_DECISION_LOG.md, "Real
  LibriStutter/SEP-28k schemas confirmed; SEP-28k labels loader built"*
- **`profiling/evaluation/` package built** (sequencing steps 1–2 of
  `VALIDATION.md` §6): word-level scoring (`score_word_level`) migrated from
  `evaluate.py` v1 with no behavior change, plus new IoU-based localization,
  per-type confusion matrices (TP/FP/FN/TN), and the combined "Any" label.
  `profiling/evaluate.py` is now a thin backward-compatible shim.
  **Finding**: SEP-28k is clip-level labeled with no reference transcript at
  all (confirmed against its README) — added `ClipLevelLabels`/
  `score_clip_level` to handle that shape correctly; its CSV parser itself
  deferred pending the real file (exact column names unconfirmed). Two real
  bugs caught and fixed while building this: a same-second result-file
  timestamp collision, and non-ASCII characters breaking under this
  machine's Windows console codepage. → *PAPER_DECISION_LOG.md, "Building the
  profiling/evaluation package"*
- **Documentation architecture established.** Added `DOCS.md` (documentation
  map), `VALIDATION.md` (evaluation methodology + results/ablation
  placeholders), `ROADMAP.md` (consolidated forward-looking priorities), and
  this file. Retired `august.md` into a stub — its content is now split
  between `PAPER_DECISION_LOG.md` (reasoning) and `VALIDATION.md`
  (methodology). → *PAPER_DECISION_LOG.md, "Documentation architecture
  established"*
- **Evaluation methodology researched and planned; no implementation yet.**
  Datasets compared (LibriStutter, SEP-28k, KSoF, UCLASS), a two-track
  (ASR-bypass / full-pipeline) methodology designed, metrics selected
  (precision/recall/F1, per-type confusion matrices, IoU localization, EER).
  Explicitly deferred pending the baseline commit. → *VALIDATION.md;
  PAPER_DECISION_LOG.md, "Evaluation methodology research (VALIDATION.md)"*
- **Real-audio validation pass; one bug found and fixed.** Two real
  microphone recordings tested end-to-end. Found and fixed a pre-existing
  display bug: the Event table showed an acoustic-sourced event's full
  attributed-word span instead of the actual detected region
  (`acoustic_start`/`acoustic_end`). Also surfaced (not fixed — no ground
  truth to act on): a missed word-repetition traceable to ASR transcription
  fidelity, and possible block-detection over-sensitivity. →
  *PAPER_DECISION_LOG.md, "Real-audio validation pass + event-table display
  fix"*
- **Audio-native-primary detector restructuring.** `profiling/detect.py`
  taxonomy split (`repetition` → `sound_repetition`/`word_repetition`/
  `phrase_repetition`, matching the SEP-28k/FluencyBank/KSoF standard);
  acoustic corroboration added for `filler`/`stutter_marker` (previously zero
  audio grounding); token-vs-acoustic fusion changed from fixed priority to
  weighted-confidence; config-driven `detectors` enable-list.
  `profiling/acoustic.py` enriched with Silero VAD (self-disabling
  corroboration signal, not a hard replacement — see the entry for why) and
  Praat pitch/jitter/shimmer/HNR. New `profiling/evaluate.py` (v1,
  timestamp-only, LibriStutter-schema, self-tested). 38/38 tests passing
  (28 pre-existing + 10 new); only 2 pre-existing tests needed a deliberate
  taxonomy-label update. → *PAPER_DECISION_LOG.md, "Audio-native-primary
  detector restructuring"*
- **Vision-alignment review + background research: architecture decision.**
  Full codebase re-read against the restated mission (audio-based disfluency
  localization, transcription as scaffolding). Literature/dataset/model
  review across ~25 papers and 6 datasets. Decision: keep the two-stage
  (ASR + detector) design, reject end-to-end region-detection architectures,
  restructure the detector to be audio-native-primary. → *PAPER_DECISION_LOG.md,
  "Vision alignment review + architecture decision"*

## 2026-06-27

- **Real-audio validation + prolongation threshold tune (Part D).**
  `prolongation_min_seconds` raised 0.65 → 1.0 after real fluent-speech false
  positives (naturally emphasized vowels). Long-recording ASR truncation
  found, not yet fixed (open issue). → *PAPER_DECISION_LOG.md*
- **Quality: phonetic near-repetition for short words.** ARPAbet
  phoneme-distance comparison for words ≤4 chars, spelling-based edit
  distance otherwise. → *PAPER_DECISION_LOG.md*
- **Quality: generalized phrase-repetition (any length).** Scan window
  widened from a hardcoded 2–3 words to a configurable
  `phrase_repetition_max_words` (default 8). → *PAPER_DECISION_LOG.md*
- **Fuse acoustic cues into the live detector (Option B, step 2).**
  `profiling/acoustic.py`'s candidates merged into `detect_disfluencies()`
  for the first time, with overlap-based dedup (token path always won —
  changed 2026-08-03, see above). → *PAPER_DECISION_LOG.md*
- **Realtime foundation: ASR-independent acoustic detection (Option B,
  step 1).** `profiling/acoustic.py` added — pure-NumPy waveform
  segmentation and prolongation/block candidate derivation, no ASR. Not
  wired into the live detector yet at this point. → *PAPER_DECISION_LOG.md*
- **§1 Option A: acoustic cross-validation of word timestamps.** Fixed
  clip-initial silence being billed to the first word (falsely inflating
  both that word's duration and the clip-wide prolongation percentile
  threshold). → *PAPER_DECISION_LOG.md*
- **Step 1c (real run) + Step 3: measured ASR latency & profiling.** First
  real benchmark numbers on a 16GB machine: ~54s (4s clip) to ~102s (20s
  clip), one-time ~29s model load. Corrected prior stale "~650-680s
  regardless of clip length" assumption. → *PAPER_DECISION_LOG.md*

## 2026-06-26

- **Docs: fixed a stale OpenVINO comment** contradicting the actual default
  backend (`transformers`). → *PAPER_DECISION_LOG.md*
- **Step 1c: real benchmark blocked** — model load OOM/segfault on a 2.2GB-
  free-RAM dev machine; deferred to a 16GB machine (resolved next day, see
  above). → *PAPER_DECISION_LOG.md*
- **Step 1b: benchmark harness** (`profiling/benchmark_asr.py`) added, with
  a `--self-test` mode verified against a stub model before any real run. →
  *PAPER_DECISION_LOG.md*
- **Step 1a: `last_timing` made self-describing** — added
  `clip_duration_seconds`, `max_new_tokens`, `audio_size_bytes`, `backend` to
  `CrisperWhisperASR.last_timing`. → *PAPER_DECISION_LOG.md*

## Before 2026-06-26

Earlier project history (the original planning docs `improve.md`,
`future.md`, `for-claude.md` and their inline results logs) predates
`PAPER_DECISION_LOG.md` and was consolidated into `ARCHITECTURE.md` at commit
`7c7e808`. Not individually reconstructed here — see `ARCHITECTURE.md` for
what survived as still-accurate technical detail.
