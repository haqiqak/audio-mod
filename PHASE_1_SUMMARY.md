# Phase 1 closing summary — Validation, Benchmarking, Analysis, and Scientific Understanding

**Closed 2026-08-03.** This is a one-time snapshot written when Phase 1 was
formally closed, not a living document — see `DOCS.md` for why it isn't
updated after this point. For full detail behind every claim here, follow
the section pointers into `VALIDATION.md` and `PAPER_DECISION_LOG.md`.

---

## 1. What Phase 1 set out to do

Establish, with real evidence rather than anecdotal testing, whether
`audio-mod`'s disfluency detector actually works — and specifically whether
its central architectural bet (audio-native-primary detection: acoustic
evidence co-equal with transcript/timing checks, not subordinate to them)
delivers on its design goal. The starting point was two informal
self-recorded microphone tests, which surfaced real findings but could not
distinguish detector bugs from ASR errors from natural speech variation —
not a basis for tuning anything. Phase 1's job was to replace that with
dataset-based, reproducible, pre-registered evaluation.

## 2. What was built

- `profiling/evaluation/`: a full evaluation package — dataset loaders
  (LibriStutter word-level, SEP-28k clip-level), metrics (precision/recall/
  F1, per-type confusion matrices, IoU localization, the combined "Any"
  label), **Track A** (detector-only, ASR bypassed, ground-truth transcript)
  and **Track B** (full pipeline — real CrisperWhisper ASR, word-level
  Levenshtein alignment back to ground truth, per-clip result caching), an
  ablation-sweep runner, and reproducible timestamped result files for
  every run.
- Real data acquisition: 499 real LibriStutter clips (annotations + audio),
  a real, complete SEP-28k labels file (audio not yet acquired — §5 below).
- `VALIDATION.md`: the methodology + results reference, now ~1,300 lines,
  including a pre-registered Track B protocol (written *before* the code
  that implements it) and a critical review of the methodology itself (§7).

## 3. Confirmed findings — established with confidence

These are treated as solid, evidence-backed conclusions, not hypotheses,
though each has a stated scope (§6 below draws the line explicitly):

1. **The audio-native architecture change delivers on its design goal.**
   Adding real audio (Silero VAD, Praat, weighted acoustic/token fusion) to
   the detector, evaluated against ground-truth transcripts on 499 real
   clips, improved `Any`-label F1 0.773 → 0.835 — almost entirely a
   precision gain (157 fewer false positives) at ~0 recall cost.
   (`VALIDATION.md` §8.3)
2. **Track A (ground-truth transcript) drastically overstates real-world
   performance.** Under real ASR conditions (Track B), recall collapses
   from Track A's ~99% to ~4–13% depending on exact subset/definition. This
   is the single most consequential finding of Phase 1: a detector that
   looks excellent against a perfect transcript can still fail badly in
   real deployment, for reasons that have nothing to do with the detector
   itself. (`VALIDATION.md` §8.4)
3. **Once corrected for a real methodological gap, the recall shortfall is
   overwhelmingly attributable to ASR, not the detector.** The original
   decomposition (word-only "ASR preserved this word") attributed ~93–95%
   of the gap to the detector. Hand-verification found this was measuring
   the wrong thing: ASR can transcribe a disfluent word correctly while
   still corrupting the word *next to it*, which breaks context-dependent
   detector checks through no fault of the detector's core logic. A
   context-strict metric (`R_B|preserved_ctx1` — both the word and its
   immediate predecessor must survive ASR intact) was pre-registered,
   implemented, and confirmed at two independent sample sizes (n=2 at 30
   clips, n=7 at 90 clips): recall on this subset is **exactly 1.0** both
   times, using an exact (not approximated) Track A comparison. The
   decomposition flips to **0% detector-attributable / 100%
   ASR-attributable, exactly**, on the measured subsets. (`VALIDATION.md`
   §8.4.1/§8.4.2)
4. **`prolongation_min_seconds` dominates every other tunable component by
   an order of magnitude** for aggregate precision/recall trade-off, and
   the current 1.0s setting is not the optimum for either aggregate `Any`
   F1 or prolongation-specific F1 on the (synthetic, reconstructed-timing)
   data measured. Not acted on — recorded as evidence for a future,
   separately-approved tuning pass. (`VALIDATION.md` §9.1/§9.2)
5. **`sound_repetition` has a genuine, structural 0% recall gap**: the
   detector's fragment-repeat check only handles "fragment-before-word"
   ordering; the data shows "fragment-after-word" patterns too. Confirmed,
   not yet fixed. (`VALIDATION.md` §8.2)

## 4. A real, scoped detector-side issue (distinct from findings 2–3 above)

Even within the context-strict-preserved subset — where the detector caught
100% of instances at the binary "disfluent or not" level — only 2 of 7
(29%) got the *exact* correct type label. `word_repetition`/
`sound_repetition` are frequently mislabeled as `phrase_repetition`/`block`
when an ASR-inserted word breaks literal token-stream adjacency, even
though a reference-level alignment says the surrounding words are
"correct." This is the one place in the whole Track B analysis with a real,
actionable *detector*-side fix candidate — everything else in findings 2–3
points at ASR fidelity, not detection logic. (`VALIDATION.md` §8.4.2)

## 5. What remains a hypothesis, explicitly not yet confirmed

Identified by the Phase 1 closing critical review (`VALIDATION.md` §7.2),
not swept under a confident-sounding headline:

- **Whether "ASR fidelity is the bottleneck" generalizes beyond this
  specific measurement.** Every Track B number comes from exactly one ASR
  backend (CrisperWhisper) on exactly one dataset family (LibriStutter's
  synthetically-spliced disfluencies). A different ASR model could have a
  materially different error profile around disfluent speech; synthetic
  splice artifacts may not behave like real stuttered speech acoustically.
  This is the single largest open question Phase 1 leaves for Phase 2.
- **Whether the confirmed conclusion (finding 3) is speaker-general.** The
  Track B subset (a deterministic prefix of the 499-clip sample) covers
  only 7 of 40 distinct speakers in the sample — found during this closing
  review, not previously documented.
- **Whether the type-classification fix (§4) actually improves outcomes**
  once implemented — currently a well-evidenced hypothesis with a traced
  mechanism, not yet built or tested.
- **Whether the ablation study's threshold findings (finding 4) transfer to
  real, non-reconstructed audio** — not yet tested on Track B data.
- Two measurement gaps, not yet built: Track B has no localization (IoU)
  metric at all; no confidence intervals exist anywhere in Phase 1's
  results, only qualitative small-sample caveats.

## 6. Why this is a sufficient, sound basis to close Phase 1

Full reasoning: `VALIDATION.md` §7.4. In short: every result currently
reported has been audited, hand-verified where alignment-dependent, and
(where a real gap was found — the $R_A$ approximation, the speaker
clustering, the single-ASR-backend dependence) either fixed or explicitly
scoped as a named, evidence-backed next step rather than silently left
implicit. No dataset or validation strategy considered during the closing
review (FluencyBank Timestamped, SEP-28k audio, a second ASR backend) was
skipped by oversight — each was evaluated and deliberately deferred to
Phase 2 because integrating it is real engineering/data-acquisition work,
not a same-session fix, and Phase 2 is explicitly where that work belongs.

## 7. What this means for Phase 2

Phase 2 ("evidence-driven model improvement") is now justified to prioritize,
**in this order** (full reasoning and links: `ROADMAP.md`):

1. Validate the ASR-is-the-bottleneck conclusion against a second ASR
   backend and/or real disfluent speech (FluencyBank Timestamped) — the
   top item, because everything else's priority depends on whether this
   holds up.
2. Re-sample Track B across speakers, not just by clip count, to confirm
   finding 3 is speaker-general.
3. Fix the word/sound-repetition type-classification gap (§4) — the one
   confirmed, scoped detector-side issue.
4. Everything downstream of that (further detector threshold tuning, new
   dataset integration, etc.) is explicitly de-prioritized relative to 1–3,
   because Phase 1's evidence says that's not where current performance is
   being lost.

This ordering is itself a direct product of Phase 1's evidence, not a
carried-over assumption — which is the point of having done Phase 1 first.
