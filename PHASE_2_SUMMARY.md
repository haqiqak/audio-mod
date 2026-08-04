# Phase 2 closing summary — Evidence-driven improvement

**Closed 2026-08-04.** This is a one-time snapshot written when Phase 2 was
formally closed, not a living document — see `DOCS.md` for why it isn't
updated after this point (mirrors `PHASE_1_SUMMARY.md`'s own convention).
For full detail behind every claim here, follow the section pointers into
`VALIDATION.md`, `PAPER_DECISION_LOG.md`, `PHASE_2_RESEARCH_PLAN.md`, and
`ROADMAP.md`.

---

## 1. What Phase 2 set out to do

Phase 1 closed with a validated baseline and a specific, evidence-ranked
priority list (`PHASE_1_SUMMARY.md` §7). Before touching any of it, Phase 2
opened with a literature-grounded review of whether the disfluency taxonomy
itself was scientifically sound (`PHASE_2_RESEARCH_PLAN.md`) — since
building on an unvalidated taxonomy would be building on sand. That review
confirmed the core 5-type taxonomy is correct, found two real gaps, and set
the actual order of Phase 2 implementation work. Phase 2's job was then to
work through that evidence-ranked list — pre-registering methodology,
implementing carefully, validating objectively, and documenting every
result honestly, including the negative ones — until nothing scientifically
justified and evidence-backed remained undone.

## 2. What was built

- **Taxonomy refinements** (additive, backward-compatible): a computed
  monosyllabic/polysyllabic (SLD/OD-likely) sub-tag on `word_repetition`
  events; explicit "not validated against any public dataset" labeling for
  `phrase_repetition`/`stutter_marker`; the silent-only `block` limitation
  documented in `ARCHITECTURE.md`.
- **Two real detector-code fixes**: `sound_repetition`'s fragment-ordering
  bug (recall 0.000→0.920 — the root cause was deeper than previously
  documented, affecting both fragment orderings, not one); a related
  Track B cache-staleness bug (the cache was storing detector *output*,
  which would have silently kept serving pre-fix classifications forever —
  now stores only ASR output, detector output always recomputed fresh).
- **The prolongation redesign**: two separately-toggleable mechanisms
  (rate-normalized threshold per Esmaili et al. 2017; Praat pitch/jitter/
  shimmer stability as a hard detection gate, not just a confidence
  adjustment), each pre-registered in `VALIDATION.md` §9.5 before
  implementation, evaluated via a 4-variant ablation, and decided by the
  pre-registered criteria — see §3 below for the result.
- **Measurement infrastructure**: a confidence-sensitive metric
  (`metrics.confidence_stats()`, mean TP vs. FP confidence) so VAD/Praat
  corroboration's designed effect — invisible to presence/absence
  scoring — could actually be evaluated; Wilson 95% confidence intervals
  (`metrics.wilson_interval()`, `TypeCounts.precision_ci()`/`.recall_ci()`,
  `report.format_table_with_ci()`) for every future run, applied
  retroactively to this project's own extreme-small-n recall claims.
- **A repository-hygiene fix that paid for itself immediately**: an
  AST-based lint check (`tests/test_ascii_console_output.py`) for
  non-ASCII characters in console `print()` output — the Windows `cp1252`
  bug that had been fixed reactively three times before. Running it once
  caught two real, previously-unnoticed violations in `benchmark_asr.py`.
- **One narrow detector extension, implemented, measured, and reverted** —
  a documented negative result, not a silent dead end (§3 below).

## 3. Confirmed findings — established with confidence

1. **`sound_repetition`'s 0% recall gap is fixed and measured.** Root
   cause: a reconstructed fragment ("word" + trailing `-`) normalizes
   identically to its complete-word counterpart, so the exact-match
   `word_repetition` check intercepted it in *both* fragment orderings,
   not just the one previously documented. Recall 0.000→0.920 on the full
   499-clip benchmark, `Any` label exactly unchanged (a pure
   type-reclassification fix). (`VALIDATION.md` §8.2.1)
2. **The speaker-clustering caveat Phase 1 flagged was real: the
   "~0% detector-attributable" finding was revised, not just confirmed,
   once speaker diversity was accounted for.** At 7-speaker samples (n=2,
   n=7), `R_B|preserved_ctx1` recall was a clean 1.0. At full 40-speaker
   diversity (n=15, 120 clips), it drops to 0.667 — a real, hand-traced
   change (all 5 new misses are `sound_repetition`/`phrase_repetition`
   instances with already-known structural gaps, not a new detector
   weakness), revising the decomposition from ~0%/100% to
   **35.1% detector-attributable / 64.9% ASR-attributable**. ASR-fidelity
   remains the majority driver — the headline does not reverse — but the
   earlier "detector is essentially perfect given fair input" framing was
   too strong. Wilson 95% CIs computed for all three sample sizes confirm
   the n=7 and n=15 intervals overlap substantially — the earlier point
   estimate was never precise enough to rule this out. (`VALIDATION.md`
   §8.4.3)
3. **The prolongation redesign has a real, measured winner and a real,
   measured loser — decided by pre-registered criteria, not judgment
   after the fact.** Praat-feature gating (a candidate must pass
   pitch-stability/jitter/shimmer checks to fire at all, not just get a
   confidence boost) is the only variant in a 13-variant ablation to
   improve both `Any` F1 (0.835→0.888) and prolongation-specific F1
   (0.064→0.084) simultaneously — now the shipped default
   (`require_praat_stability_for_prolongation: true`). Rate-normalization
   (Esmaili et al. 2017's peer-reviewed formula) regressed both metrics
   severely on its own (`Any` F1 0.835→0.347) and combined with
   Praat-gating (0.835→0.743) — stays off by default, with a stated
   hypothesis for why (LibriStutter's short reconstructed clips likely
   destabilize the per-clip speaking-rate estimate the formula divides
   by), not confirmed further. (`VALIDATION.md` §9.5.1)
4. **VAD/Praat corroboration's confidence-adjustment effect is ~zero (and
   slightly negative for the combined label) on real data, closing a
   question Phase 1's ablation left open.** Phase 1's own ablation found
   zero measured effect from VAD/Praat on presence/absence counts, but
   flagged that as a metric-blindness finding, not evidence they don't
   help. The confidence-sensitive metric built to test that directly
   found the TP-vs-FP confidence gap is ~zero everywhere measurable
   (largest per-type gap +0.003) and -0.007 for the combined `Any` label —
   a real, audited (not just read-off) negative-to-null result. Not acted
   on (no code/config change); flagged as a Phase 3 candidate decision.
   (`VALIDATION.md` §9.3.1)
5. **UCLASS's claimed "severity" annotation is unsubstantiated by its own
   cited primary source.** The rule-based-detection preprint motivating
   this check cites Howell et al. 2009 for its claim that UCLASS has
   event-level severity annotations; the primary paper itself describes
   only recording-level perceptual quality ratings, not event-level
   severity. Combined with link rot on the referenced annotation
   documentation and no methodology doc alongside UCLASS's raw file
   listing, this is inconclusive from every public source checked — not a
   confirmation UCLASS lacks the distinction, but not a basis to build
   audible/tense block detection on either. (`PHASE_2_RESEARCH_PLAN.md`
   §5 point 3)

## 4. A documented negative result, in full (not a silent dead end)

A candidate "word-sandwiched repetition" detector extension (tolerating
one non-filler intervening word between a repeated pair) was built, per
the pre-registered discipline of measuring before committing to a change.
The diagnostic metric built first (hypothesis-side-contiguity, computed
from real ASR alignment) found only n=3 addressable cases out of 120
already-scored clips — thin evidence. Implemented anyway, narrowly scoped,
and let a full 499-clip benchmark decide: **`Any` F1 regressed
0.835→0.793 with zero new true positives** (Track A structurally can only
show this fix's cost, never its benefit, since the benefit requires real
ASR errors that don't exist in ground-truth-transcript scoring); Track B
showed only +1 TP at a cost of +24–29 new FP. **Reverted**, with a
regression test locking in the correct (non-firing) behavior. See
`VALIDATION.md` §8.4.4 for the full writeup.

## 5. What remains a hypothesis, explicitly not yet confirmed

- **Whether "ASR fidelity is the dominant bottleneck" generalizes beyond
  CrisperWhisper on LibriStutter.** Re-scoped mid-Phase-2 from "blocking
  prerequisite" to "external-validity strengthening" (`PAPER_DECISION_LOG.md`,
  2026-08-04) once each detector-side fix's own evidence was traced back
  to Track A/literature sources that never involved ASR — but the general
  claim itself is still only measured against one backend, one dataset
  family. (`ROADMAP.md` item 10)
- **Whether the prolongation redesign's Praat-gating win transfers to
  real (non-reconstructed-timing) speech.** Measured entirely on
  LibriStutter's reconstructed tokens, the same limitation carried over
  from Phase 1's original threshold sweep. (`VALIDATION.md` §9.4/§9.5)
- **Why rate-normalization regressed so severely** — a stated, plausible
  hypothesis (unstable speaking-rate estimates on short reconstructed
  clips), not confirmed by further investigation, which was out of this
  ablation's scope.
- **Whether the confidence-sensitive metric's near-zero VAD/Praat result
  holds beyond this one dataset and run.** One dataset, one run is
  evidence, not proof the corroboration mechanism is worthless in
  general.
- **Whether UCLASS actually does or doesn't distinguish silent from
  audible/tense blocks** — genuinely unresolved; would require direct
  inspection of UCLASS's raw annotation files, not attempted this phase.
- **Track B has no localization (IoU) metric** — Track A has had one
  since Phase 1; Track B's timing precision under real ASR conditions has
  never been measured. (`ROADMAP.md` item 9)

## 6. Why this is a sufficient, sound basis to close Phase 2

Every item on the evidence-ranked Phase 2 priority list (`ROADMAP.md`,
as reordered by `PHASE_2_RESEARCH_PLAN.md`'s opening review) reached one
of three honest end states: **implemented and validated** (taxonomy
refinements, the two detector fixes, the prolongation redesign, the
measurement infrastructure, the ASCII lint rule); **investigated and
explicitly documented as negative or inconclusive** (the word-sandwiched
extension, UCLASS's schema, rate-normalized prolongation); or
**evaluated and deliberately deferred to Phase 3 with stated reasoning**
(the Track B localization metric, LibriStutter sample expansion, SEP-28k
audio acquisition, FluencyBank integration, the second-ASR-backend
generalization check) — mirroring exactly how Phase 1 closed
(`PHASE_1_SUMMARY.md` §6). Nothing was left as a silent gap: every open
question in §5 above is named, scoped, and pointed at the section that
explains it, not glossed over by a confident-sounding headline. The full
regression suite (45/45) passes with every change in place, and the
config-default change made this phase (Praat-gating) followed mechanically
from criteria fixed *before* the ablation that decided it ran.

## 7. What this means for Phase 3

No single "next big thing" the way Phase 2 had prolongation — the
evidence-ranked remaining work is a set of independent, genuinely deferred
items, each with its own stated reasoning (`ROADMAP.md` for the full
list). In rough order of leverage-per-effort:

1. **Track B localization metric** (`ROADMAP.md` item 9) — the cheapest
   way to extend this project's "is it precisely timed, not just present"
   validation to real ASR conditions, where it has never been measured.
2. **Expand the LibriStutter sample** (`ROADMAP.md` item 14) — cheapest
   of the dataset items (existing fetch script), and specifically closes
   the `filler`/`phrase_repetition` n=0 sampling gap this phase's
   confidence-stats run re-surfaced.
3. **A second ASR backend, and/or FluencyBank Timestamped**
   (`ROADMAP.md` items 10/16) — the remaining piece of external validity
   this project's central "ASR fidelity is the bottleneck" claim still
   needs before it can be stated as a general finding rather than a
   CrisperWhisper-on-LibriStutter-specific one.
4. **Revisit whether VAD/Praat's confidence-weighting logic is worth its
   complexity**, given this phase's near-zero measured confidence-gap
   result — a simplification candidate, not an urgent fix.
5. **SEP-28k audio acquisition** (`ROADMAP.md` item 15) — the largest
   engineering/bandwidth lift of the deferred items; the only path to
   validating `block` detection against a second real-speech dataset.

This ordering, like Phase 1→2's, is itself a product of Phase 2's
evidence — not a carried-over assumption.
