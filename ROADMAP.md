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

## Phase 2 opening literature review is done (2026-08-03)

Before any Phase 2 implementation, the current 7-type taxonomy was checked
against clinical speech-pathology and computational-detection literature —
see `PHASE_2_RESEARCH_PLAN.md` for the full review, gap analysis, and
structured plan. **Headline: the core 5-type taxonomy is scientifically
sound, no redesign needed.** This found two real gaps (a clinically
meaningful monosyllabic/polysyllabic split `word_repetition` ignores; a
confirmed silent-only limitation in `block` detection), independently
corroborated two of Phase 1's own findings via unrelated external sources,
and identified prolongation as the single highest-confidence Phase 2
detector-side target (three independent lines of evidence converge on it —
Phase 1's own ablation, the literature's rule-based-detection results, and
full dataset support). The items below reflect this review's structured
plan (`PHASE_2_RESEARCH_PLAN.md` §7) as it has evolved with evidence since
— see the 2026-08-04 note before item 3 for the most recent reordering
(the second-ASR-backend gate lifted for the detector-side items, moved to
item 10).

## Phase 2 is closed (2026-08-04)

See `PHASE_2_SUMMARY.md` for the full closing summary (confirmed findings,
the one fully-documented negative result, what remains a hypothesis, and
the evidence-ranked case for Phase 3). Every item below reached one of
three honest end states: **done** (struck through, with the measured
result), **investigated and documented as negative/inconclusive** (11,
and by extension 12), or **explicitly deferred to Phase 3 with reasoning**
(9, 10, 14, 15, 16 — look for each item's own "Scope decision" or
"re-scoped" note). Kept as one numbered list rather than split into
"done"/"Phase 3" sections, so each item's full reasoning trail stays in
one place and nothing needs renumbering (other docs cite these items by
number — `VALIDATION.md`, `PAPER_DECISION_LOG.md`, `PHASE_2_RESEARCH_
PLAN.md`, `HANDOFF.md`). **For Phase 3 planning, start from `PHASE_2_
SUMMARY.md` §7's leverage-ranked shortlist**, not by re-reading every item
below from scratch.

1. ~~[Cheap, immediate, low-risk] Taxonomy/documentation refinements~~ —
   **implemented, 2026-08-03**. All three additive, backward-compatible
   changes from `PHASE_2_RESEARCH_PLAN.md` §7 Step 1 are in place: (a)
   `word_repetition` events now carry `syllable_count`/`likely_sld` fields
   (`_word_repetition_extra()` in `detect.py`) — monosyllabic repeats
   tagged stuttering-like (SLD), polysyllabic tagged an ordinary
   linguistic-planning disfluency (OD), per the clinical literature;
   surfaced in the app's Event table as a "Class" column, with an explicit
   caption that it's a descriptive heuristic, not a validated clinical
   measure. (b) `phrase_repetition`/`stutter_marker` explicitly labeled
   "not validated against any current public dataset" in `README.md`'s
   taxonomy table and the app's event-table caption. (c) The silent-only
   `block` coverage gap documented in `ARCHITECTURE.md`'s known-limitations
   section and §4. New unit test added
   (`test_word_repetition_sld_tag_by_syllable_count`,
   `tests/test_detect_taxonomy_and_fusion.py`, now 11/11 — all pass; full
   suite 39/39). **Benchmarked against the frozen Phase 1 Track A baseline
   (§8.3, 499 real clips) — confirmed byte-for-byte identical** (`Any`
   TP=801/FP=308/FN=8/F1=0.835, matching §8.3 exactly across every type),
   proving the change is purely additive with zero effect on detection or
   scoring. See `PAPER_DECISION_LOG.md`.
2. ~~[Top priority] Re-sample Track B across speakers, not just by clip
   count~~ — **done, 2026-08-04**. Speaker-stratified run (120 clips, all
   40 speakers, `--speaker-stratified`) — see `VALIDATION.md` §8.4.3. The
   suspicion behind this item was justified: `R_B|preserved_ctx1` recall
   dropped from 1.0 (7 speakers) to 0.667 (40 speakers), decomposition
   revised to 35.1% detector-attributable / 64.9% ASR-attributable. Hand-
   traced: every miss is `sound_repetition`/`phrase_repetition` (item 3
   below), not `word_repetition`, and not a new failure mode — both trace
   to already-known structural gaps. See `PAPER_DECISION_LOG.md`.

**Items 3–5 below were gated on completing item 10 (the second-ASR-backend
check) until 2026-08-04, when that gate was explicitly re-examined and
lifted for these three items specifically — not blanket-lifted for
everything.** Reasoning (`PAPER_DECISION_LOG.md`, "Is a second ASR backend
still necessary before detector-side work..."): each of the three rests on
Track A and/or peer-reviewed-literature evidence that never involved ASR
at all, so no plausible second-backend result could invalidate them. What
a second backend would still test — whether the *general* claim
"ASR-fidelity is the dominant real-world bottleneck" holds beyond
CrisperWhisper — remains open and valuable, just no longer a prerequisite
for these three (see item 10's revised framing below).

3. ~~[Top priority] Fix `sound_repetition`'s fragment-ordering gap~~ —
   **(a) done, 2026-08-04; (b) not a detector bug, no action needed.**
   (a) `sound_repetition`: fixed and measured — recall 0.000→0.920 on the
   full 499-clip Track A benchmark (`Any` label exactly unchanged,
   confirming a pure type-reclassification fix). Root cause was deeper
   than the previously-documented "ordering" framing: a reconstructed
   fragment normalizes identically to its complete-word counterpart, so
   the exact-match `word_repetition` check intercepted it in *both*
   orderings, not just one — a reverse-order-only fix would not have
   worked; the actual fix reorders the fragment check ahead of the
   exact-match check, handling both orderings in one branch. 16 residual
   FN (8%) remain, not investigated further — secondary to the closed
   order-of-magnitude gap. See `VALIDATION.md` §8.2.1 and
   `PAPER_DECISION_LOG.md`. A related infrastructure bug was found and
   fixed alongside this: Track B's per-clip cache stored the detector's
   *output*, not just ASR output, so it would have silently kept serving
   pre-fix classifications on every future Track B run — cache now stores
   only ASR output, detector output is always recomputed fresh.
   (b) `phrase_repetition`'s LibriStutter ground truth is often a
   single-word approximation of a true multi-word repeat (§8.2), which
   can't be detected by any phrase-repetition check operating on the real
   (non-reconstructed) word sequence almost by definition — this is a
   dataset-representation limit, not a detector bug, and the fix (if any)
   is a better dataset/reconstruction, not different detection logic; no
   code action planned.
4. ~~Fix the word_repetition/sound_repetition type-classification gap~~ —
   **investigated fully, 2026-08-04; net conclusion: not fixable at
   acceptable cost with current evidence, negative result documented.**
   Built candidate fix (b) first (the hypothesis-side-contiguity metric,
   `VALIDATION.md` §8.4.4) using the existing Track B cache at zero ASR
   cost — found only n=3 addressable cases (1 genuine close-range miss, 2
   type-accuracy-only cases) out of 120 clips; the other 4 "gap" misses
   traced to the *already-known* `phrase_repetition` reconstruction
   limitation (§8.2), not ASR-insertion corruption. Implemented candidate
   fix (a) anyway, narrowly scoped, and let a full Track A benchmark
   decide empirically: **regressed `Any` F1 0.835→0.793 with zero new true
   positives** on Track A, and only +1 TP at a cost of +24–29 FP on Track
   B. **Reverted** — a regression test locks in the correct behavior. See
   `VALIDATION.md` §8.4.4 and `PAPER_DECISION_LOG.md`. Not revisit-worthy
   without materially more Track B data (the addressable-case count would
   need to grow well past n=3 to justify the demonstrated false-positive
   exposure).
5. ~~[High-confidence, gate lifted 2026-08-04 — from the literature review,
   `PHASE_2_RESEARCH_PLAN.md` §6/§7 Step 3] Redesign `prolongation`'s core
   detection to use rate-normalization and already-computed Praat features
   as detection criteria, not just post-hoc confidence adjustment~~ —
   **done, 2026-08-04. Praat-gating adopted as the new default;
   rate-normalization implemented, tested, and rejected by evidence.**
   Both changes built as separately-toggleable config options (`use_rate_
   normalized_prolongation`, `require_praat_stability_for_prolongation`),
   pre-registered in `VALIDATION.md` §9.5 before implementation, then
   evaluated via a 4-variant ablation (folded into the standing
   13-variant harness) against the full 499-clip real-audio sample.
   **Result**: Praat-gating alone is the only variant of 13 to clear the
   pre-registered bar (improve *both* `Any` F1 [0.835->0.888] *and*
   prolongation-specific F1 [0.064->0.084] simultaneously) — now the
   shipped `config.yaml` default. Rate-normalization alone regressed both
   metrics severely (`Any` F1 0.835->0.347), and combined with
   Praat-gating still failed the bar (`Any` F1 0.835->0.743) — stays off
   by default; code kept in place (fully toggleable) as a documented
   negative result and possible future starting point if a real-speech
   (non-reconstructed-timing) dataset becomes available. Full regression
   suite: 45/45 after the config change. See `VALIDATION.md` §9.5.1 and
   `PAPER_DECISION_LOG.md`.
6. **[De-prioritized by evidence — naive threshold-only tuning specifically,
   not the literature-informed redesign in item 5] Tuning
   `profiling/detect.py`'s detection thresholds purely to raise recall
   without changing the underlying detection logic.** Phase 1's confirmed
   conclusion is that recall is not being lost in the detector when given
   intact input, so a naive threshold sweep for that purpose specifically
   is still de-prioritized — item 5 is a different, literature-motivated
   kind of change (new detection criteria, not just moving a threshold) and
   is not affected by this de-prioritization.
7. ~~[Cheap, parallel, no new data needed] Build a confidence-sensitive
   metric~~ — **done, 2026-08-04.** `metrics.confidence_stats()` built,
   unit-tested, then run against the full 499-clip real-audio LibriStutter
   sample under current production config. **Result: the TP-vs-FP
   confidence gap is ~zero everywhere measurable (largest per-type gap
   +0.003) and slightly negative for the combined `Any` label (-0.007)** —
   see `VALIDATION.md` §9.3.1. This closes §9.3's open question with a
   negative-to-null result: VAD/Praat corroboration's designed
   confidence-adjustment effect is not producing a meaningful TP/FP
   separation on this dataset, as measured. Not acted on (no config/logic
   change) per standing rule 4 — flagged as a Phase 3 candidate decision
   (simplify or remove the corroboration weighting) rather than applied
   automatically.
8. ~~[Cheap, parallel, no new data needed — new from the Phase 1 closing
   review, `VALIDATION.md` §7.2 item 5] Add confidence intervals to every
   reported recall/precision number~~ — **infrastructure done, 2026-08-04.**
   `metrics.wilson_interval()`, `TypeCounts.precision_ci()`/`.recall_ci()`,
   `report.format_table_with_ci()` built and unit-tested; `save_run()` now
   persists CIs for every future run automatically. Applied directly to
   the exact small-n cases this item named: `R_B|preserved_ctx1`'s
   n=2/n=7/n=15 recall figures now have Wilson 95% CIs recorded in
   `VALIDATION.md` §8.4.3 ([0.342,1.000], [0.646,1.000], [0.417,0.848]
   respectively) — the n=7 and n=15 intervals overlap substantially,
   making concrete exactly how much the earlier "1.0 recall" point
   estimate was not a precise claim. Retrofitting CIs onto every other
   historical number in `VALIDATION.md` §8 is not done (would mean
   re-deriving many past run's raw k/n from JSON result files for no new
   decision it would change) — scoped down to "infrastructure exists and
   is used at the point it was requested for," not a full historical
   backfill.
9. **[Medium priority — new from the Phase 1 closing review, `VALIDATION.md`
   §7.2 item 4] Build a Track B localization (IoU/temporal) metric.**
   `track_b.py`'s `score_clip` currently hardcodes `localization=None` —
   Track B has never validated *how precisely timed* a caught disfluency
   is under real ASR conditions, only Track A has (§4 point 3). Feasible
   in principle (both a predicted acoustic span and a ground-truth
   reference span exist in real audio time) but real, non-trivial work —
   not a quick fix. **Scope decision (2026-08-04): explicitly deferred to
   Phase 3, not started this phase.** Reasoning: every Phase 2 detector-
   side change actually made (prolongation redesign, `sound_repetition`
   fix) was validated on Track A, where this localization metric already
   exists (§4 point 3) — this item would strengthen the Track B picture
   specifically, which is valuable but not blocking for anything decided
   in Phase 2, and the "real, non-trivial work" scoping note above still
   holds (it needs its own alignment-to-timing design, not a quick
   addition). Consistent with how Phase 1 closed: real, valuable,
   evaluated-and-deliberately-deferred, not silently dropped.
10. **[Re-scoped 2026-08-04 — no longer a blocking prerequisite, still a
    valuable generalization check — `VALIDATION.md` §7.2 item 3] Validate
    the confirmed "ASR is the bottleneck" conclusion against a second ASR
    backend and/or real (non-synthetic) disfluent speech.** Every Track B
    number to date comes from exactly one ASR model (CrisperWhisper) on
    exactly one dataset family (LibriStutter's synthetically-spliced
    disfluencies). The absolute Track A→B recall drop is not in question.
    What remains open is whether the *general* claim "ASR-fidelity is the
    dominant bottleneck" (as opposed to specifically for CrisperWhisper on
    LibriStutter) holds — this still matters for how broadly this project
    can eventually state its central finding in a paper, but (per
    `PAPER_DECISION_LOG.md`, 2026-08-04) does not need to be resolved
    before items 3–5, whose evidence is independently ASR-backend-agnostic.
    Two independent ways to check, either is valuable on its own: (a) run
    a second ASR backend (e.g. stock whisper-large-v3) through the same
    Track B pipeline on the same audio already downloaded — cheaper than a
    new dataset, no new licensing/acquisition; (b) integrate FluencyBank
    Timestamped (real people who stutter, word-level timestamps — see item
    16 below) and re-run Track B against it.
11. ~~[New, small, scoped — from the literature review, `PHASE_2_RESEARCH_
    PLAN.md` §5 point 3] Verify UCLASS's exact annotation schema directly~~
    — **investigated, 2026-08-04; conclusion: inconclusive from every
    public secondary source, and the specific "severity" claim is not
    substantiated by the primary source it cites.** Checked the primary
    UCLASS archive paper (Howell et al. 2009), the external annotation
    page it points to (link rot — certificate no longer matches the old
    domain), and UCLASS's current raw file listing (files present, no
    accompanying methodology doc). See `PHASE_2_RESEARCH_PLAN.md` §5
    point 3's addendum for the full trace. Resolving this further would
    require downloading and directly inspecting UCLASS's raw annotation
    files under its own access process — out of scope for this check.
12. **Audible/tense block detection — still not started; item 11's
    investigation did not find evidence this is validatable yet.**
    A real, literature-identified, previously-undocumented gap (this
    project's `block` detector is confirmed silent-only) and the type even
    the literature's own rule-based systems handle worst — but not
    currently validatable against any dataset this project has access to
    (SEP-28k/KSoF/FluencyBank's `block` label doesn't sub-type this).
    Building it without a way to measure whether it works would repeat the
    anecdotal-validation mistake Phase 1 was built to avoid. **Sharpened by
    the 2026-08-03 per-type definition audit** (`PHASE_2_RESEARCH_PLAN.md`
    §10.5): the clinical definition is effort/struggle-based (SSI-4 scores
    "physical concomitants" — signs not present in audio at all), and
    SEP-28k's own schema has a *separate* `NaturalPause` column distinct
    from `Block` — meaning trained human annotators already make a
    pause-vs-block distinction our silence-duration-only rule structurally
    cannot. This is a mechanistic reason to expect a pure-silence rule to
    underperform, not just an empirical one. Item 11's investigation
    (2026-08-04) did not find evidence UCLASS supports this — revisit only
    if direct inspection of UCLASS's raw annotation files (not attempted)
    or a different dataset ever does.
13. ~~Standardize non-ASCII-in-console-output prevention~~ — **done,
    2026-08-04.** `tests/test_ascii_console_output.py`: an AST-based check
    (not a whole-file byte scan — this codebase's docstrings/comments
    legitimately use em-dashes throughout, never sent to a console) that
    fails if any `print()` call under `profiling/` contains a non-ASCII
    string literal. Running it immediately caught two real, previously-
    unnoticed violations in `benchmark_asr.py` (an ellipsis and an
    em-dash) — fixed alongside adding the check. Now part of the standard
    test suite (45/45), so this recurs automatically on every future run
    instead of needing a fourth reactive fix.
14. **Expand the LibriStutter sample** (499/4,736 files, ~10.5% of the
    corpus) for more statistical power, and specifically target a sample
    with `filler`/interjection instances (the 499-file sample happened to
    have none) — `profiling/evaluation/fetch_libristutter_sample.py --n`.
    Lower priority than items 3–9 per the evidence above.
15. **SEP-28k: acquire real audio, then run Track A (acoustic-only) or
    Track B** — `load_sep28k_labels` is built and verified against the real
    labels file, but SEP-28k has no reference transcript at all, so nothing
    can be *scored* yet without audio (bandwidth/storage/time cost,
    materially larger than LibriStutter's annotation-only footprint — check
    with the project owner before attempting). Track B's alignment
    machinery now exists and would need adapting to clip-level rather than
    word-level ground truth to be usable here. Only partially addresses
    item 10's ASR-generalization question (real speech, but no word-level
    transcript) — item 10's FluencyBank option is the more direct check.
16. **FluencyBank Timestamped** — real people who stutter (not synthetic
    injection like LibriStutter), word-level timestamps + disfluency labels.
    Scientifically the strongest dataset option researched so far, but not
    chosen for the 2026-08-03 phase due to unconfirmed integration risk:
    hosted on TalkBank in CHAT format (needs a dedicated parser) and
    possibly access-gated (unconfirmed). Investigate access/format directly
    before committing engineering time. This is now item 10's most direct
    dataset-side option, not just a "nice to have."

**Scope decision on items 14-16 (2026-08-04): all three explicitly deferred
to Phase 3, none started this phase.** Reasoning, common to all three:
each is real data-acquisition/engineering work (a larger LibriStutter
sample, SEP-28k audio bandwidth/storage, a new CHAT-format parser for
FluencyBank), not a same-session fix, and none is a prerequisite for any
decision Phase 2 actually needed to make — every Phase 2 detector-side
change (prolongation redesign, `sound_repetition` fix, the confidence-
stats/CI infrastructure) was fully evaluable on the existing 499-clip
LibriStutter sample already in hand. This mirrors exactly how Phase 1
closed (`PHASE_1_SUMMARY.md` §6: "no dataset or validation strategy... was
skipped by oversight — each was evaluated and deliberately deferred...
because integrating it is real engineering/data-acquisition work"), now
applied a second time at Phase 2's close. Individually: item 14
(LibriStutter expansion) is the cheapest of the three (existing fetch
script, `--n` flag) and the best Phase 3 starting point specifically to
close the `filler`/`phrase_repetition` n=0 gap this session's
confidence-stats run re-surfaced (§9.3.1); items 15-16 both require an
explicit acquisition go-ahead per their own text above and are the
larger lift.

## Phase 3 architecture review is done (2026-08-04)

Before starting Phase 3 implementation, the project owner asked for a
first-principles challenge to the ASR-first two-stage architecture
itself — not just its components — grounded in Phase 1/2's own evidence
plus a fresh 2024-2026 literature pass. Full review: `PHASE_3_
ARCHITECTURE_REVIEW.md`. **Headline: the two-stage architecture is kept**
(no alternative found — SSL classifiers, end-to-end audio-region models,
joint ASR+detection training — is decisively more accurate at this
project's task granularity without infrastructure this project doesn't
have), **but the review surfaced one new, well-evidenced, actionable
item, added below as item 17.** See `PHASE_2_SUMMARY.md` §7 for the rest
of Phase 3's evidence-ranked shortlist (items 9, 14, 10/16, 15, and a VAD/
Praat-complexity revisit) — item 17 below now sits alongside that list,
not separate from it.

17. **[New, 2026-08-04, from `PHASE_3_ARCHITECTURE_REVIEW.md` §5.1/§8,
    refined same day via an adversarial self-review] Extend
    audio-native-primary detection to `word_repetition`/
    `sound_repetition`/`filler` — staged, evidence-gated, starting from
    CrisperWhisper's own encoder representations.** These three types
    remain almost entirely token-text-dependent today (`ARCHITECTURE.md`:
    the acoustic-native detector "has no repetition or filler logic at
    all") — the same architectural gap `block` (Phase 1) and
    `prolongation` (Phase 2) each closed by becoming less token-dependent
    and more acoustic-native, both times with a measured accuracy
    improvement. Independently corroborated by outside literature
    (`PHASE_3_ARCHITECTURE_REVIEW.md` §3.6): ASR damages exactly these
    three types most severely (35-47% WER impact) while damaging `block`
    least (~20%).

    **Staged plan** (full reasoning: `PHASE_3_ARCHITECTURE_REVIEW.md`
    §5.1, §8):
    1. **Stage 1**: CrisperWhisper's own last-layer encoder hidden
       states (already computed on every clip via the pipeline's
       existing forward pass — bypassing `pipeline()`'s wrapper is the
       only new engineering cost; no new model, no new forward pass,
       zero added latency) as a zero-training, non-parametric
       corroboration signal, the same fusion role VAD/Praat already
       play. Best available specific evidence: a Whisper encoder's last
       layer alone reached F1 0.88/0.85/0.87 on SEP-28k+FluencyBank
       clip-level classification (arXiv:2406.05784) — not word-level,
       not CrisperWhisper-specific, a real gap this project's own Stage
       1 evaluation would close.
    2. **Explicit escalation trigger, fixed before Stage 1 runs**: a
       weak/null Stage 1 result is itself the evidence that justifies
       Stage 1b below, not a reason to quietly drop the whole idea.
    3. **Stage 1b (evidence-gated)**: a frozen **WavLM-Large** pass — a
       genuinely new model and a real added-latency cost this project
       doesn't pay today, honestly priced as such, but the strongest
       theoretical candidate if Stage 1 fails: no competing ASR training
       objective diluting its acoustic sensitivity, the best published
       word-level stuttering F1 found in this review (0.554,
       arXiv:2409.10704), and a comparable-to-Whisper showing in the one
       direct paralinguistic head-to-head found (arXiv:2502.19387, tone
       classification, not disfluency-specific — a stated caveat).
    4. **Stage 2 (either path)**: a small trained classification head,
       only with explicit go-ahead — the first real departure from this
       project's zero-training-component philosophy, decided
       deliberately rather than backed into.

    **Stage 1: done, 2026-08-04 — clear, stable, large-effect-size
    result.** Pre-registered (`VALIDATION.md` §11), implemented, run at
    30 then 90 clips: `word_repetition` Cohen's d = +1.047, `Any` d =
    +1.116 (both well past the revised `|d| >= 0.5` bar, stable across
    both sample sizes). CrisperWhisper's encoder carries information the
    transcript alone does not. A real, unconfirmed duration/word-identity
    confound named explicitly (`VALIDATION.md` §11.6), not hidden by the
    positive headline. **Stage 1b (WavLM-Large) is therefore not
    triggered** — the escalation condition (a weak/null Stage 1 result)
    did not occur. **Stage 2 (a trained classification head, or
    possibly a simpler zero-training threshold on the same distance
    measure — an option this result itself surfaced, not originally
    planned) awaits an explicit go-ahead, per `VALIDATION.md` §11.5 and
    standing rule 4** — a clear result does not self-authorize the next
    step. See `PAPER_DECISION_LOG.md`'s 2026-08-04 "Stage 1 result" entry
    for the full audit and the stated limitation.

    **Corroboration-mechanism review, same day**: re-opened the "what
    happens with this signal" question from first principles rather than
    defaulting to Stage 2 as originally scoped — see
    `PHASE_3_ARCHITECTURE_REVIEW.md` §9. Found the mechanism question
    (threshold vs. relative threshold vs. classifier) and the signal
    question (distance-to-centroid vs. a new, untested repeat-pair
    self-similarity candidate) are separable, and that at least three
    combinations remain genuinely plausible — resolved by a pre-registered
    5-fold cross-validated comparison (`VALIDATION.md` §12), not by
    argument.

    **Comparison result, same day: the classifier wins clearly — the
    opposite of this project's own pre-registered prediction.**
    `word_repetition` F1: threshold 0.588 vs. **classifier 0.749**;
    `Any` F1: threshold 0.755 vs. **classifier 0.888** — beating the
    threshold in 5/5 folds in both slices (verified directly, not just
    from the mean). `VALIDATION.md` §12.4 had predicted a large Cohen's
    d would leave little room for a classifier to improve on a
    threshold; this is the opposite result, reported as a contradicted
    prediction, not reframed as expected. The new repeat-pair-similarity
    signal did not outperform the threshold either — a real, reported
    negative result for a candidate this review specifically proposed.
    **Real limitations named alongside the strong result**: modest
    sample (93-130 events across 5 folds), untuned L2 regularization
    strength, still LibriStutter's reconstructed-timing data.

    **Reasoned decision, same day, under the new evidence-constrained-
    architecture standing rule (`CLAUDE.md` rule 8)**: the §12.5 result
    is real and positive enough to change this project's *working
    expectation* — a learned corroboration signal is, on current
    evidence, more likely the right direction for these two types than a
    hand-calibrated threshold. **Not yet strong enough to ship as the
    default**, for three named, evidence-based reasons (not a preference
    for simplicity): a high-dimensional classifier's cross-validated
    estimate is inherently less trustworthy than a threshold's at this
    same small sample size; the regularization strength was fixed, not
    tuned; and LibriStutter's reconstructed timing has fooled a
    higher-capacity mechanism in this exact project before
    (`VALIDATION.md` §8.3's prolongation-threshold history). **Next step
    pre-registered, not run this session**: a larger-scale re-run with
    nested-CV-tuned regularization and a fixed decision rule
    (`VALIDATION.md` §12.6) — if the classifier's advantage holds,
    implementation becomes the justified next step; if it doesn't, the
    threshold (or no new mechanism) remains the default, recorded with
    equal rigor either way. See `PAPER_DECISION_LOG.md`'s 2026-08-04
    "Standing principle established" entry for the full reasoning.

## Near-term

- **Validate `block` against SEP-28k/KSoF specifically** — LibriStutter's
  taxonomy can't score it at all (`VALIDATION.md` §2); this needs the
  Tier-2 dataset. Correction (2026-08-03, Phase 2 literature review,
  `PHASE_2_RESEARCH_PLAN.md`): SEP-28k/KSoF *do* label `block` (unlike
  `stutter_marker`, which no reviewed dataset labels at all — see
  "Phase 2 is closed" items 5/11/12 above) but as a single undifferentiated
  category, not split into the literature's silent/audible sub-types —
  even the Tier-2 datasets won't validate that specific split without
  UCLASS's possibly-finer annotations (item 11 above).
- **The deferred learned tier** — a frozen WavLM-base or wav2vec2-base
  classifier for repetition-subtype discrimination and cross-speaker
  generalization (Shih et al. 2024; the multi-task/adversarial-learning
  literature), trained/evaluated on SEP-28k + FluencyBank Timestamped. The
  clear evidence-backed next architectural step, deliberately deferred until
  a baseline exists to prove it actually helps — that baseline now exists
  (§8.3) — see `PAPER_DECISION_LOG.md`'s 2026-08 restructuring entry for the
  full reasoning on why this wasn't built immediately.
- ~~Ablation studies~~ — **done**, see "Phase 2 is closed" above and `VALIDATION.md`
  §9. Follow-on fine-grained work (e.g. isolating VAD's effect once a
  confidence-sensitive metric exists) stays here.
- ~~Extend the LibriStutter path to real audio~~ — **done**, this project's
  first research baseline (see "Phase 2 is closed" above). Real audio via
  `fetch_libristutter_audio.py`; a real FLAC-decode bug was found and fixed
  in the process (`PAPER_DECISION_LOG.md`).
- **A more sophisticated stutter_marker acoustic check** — an offset-shape /
  abrupt-energy-drop heuristic for cut-off fragments, replacing the simpler
  voiced-energy-presence check shipped 2026-08. Explicitly not built without
  real recordings to validate it against. Further de-prioritized (2026-08-03,
  `PHASE_2_RESEARCH_PLAN.md` §6 item 5): no reviewed dataset labels
  `stutter_marker` at all, so even with real recordings, an improvement
  here can't be benchmarked the way every other type in this project can —
  revisit only if a matching dataset appears or real usage flags it as a
  problem.

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
- **[New, 2026-08-04, from `PHASE_3_ARCHITECTURE_REVIEW.md` §3.8] Joint
  ASR + disfluency-detection multi-task training** — the most direct
  literature-found challenge to this project's two-stage assumption
  (arXiv:2505.22005 reports a 37.71% relative CER reduction *and* a
  46.58% relative detection-F1 improvement from training the two tasks
  jointly rather than as separate stages, on Mandarin data). Not adopted
  now: requires fine-tuning an ASR model jointly with detection labels,
  which needs both a training pipeline this project doesn't have and a
  paired transcript+word-level-disfluency English dataset this project
  doesn't have (SEP-28k/LibriStutter don't pair the two the way this
  training regime needs). This project has also already found that even
  *swapping* ASR backends without retraining breaks compatibility
  (faster-whisper's tokenizer incompatibility, `ARCHITECTURE.md` §3) —
  fine-tuning the model itself is a substantially larger lift. Revisit
  only if a training pipeline and a suitable paired English dataset both
  become available.

## Explicitly rejected

These were considered and turned down with reasons recorded, not silently
dropped — do not re-litigate without new evidence.

- **End-to-end audio→dysfluency-region models** (YOLO-Stutter/
  Stutter-Solver/SSDM-class) as a replacement for the two-stage
  ASR-then-detector pipeline. Rejected: still require a speech-text alignment
  as input (don't remove the ASR stage), and a 2025 comparative study found
  the most complex of these (SSDM) irreproducible by an independent team —
  see `PAPER_DECISION_LOG.md`'s 2026-08 restructuring entry. **Reaffirmed,
  2026-08-04**, in the pre-Phase-3 architecture review: SSDM 2.0 (SSDM's
  direct successor) is heavier still (adds a neural articulatory flow and
  an LLM-integration pipeline, needs specialized corpora this project has
  no access to) — the field's most complex end is getting more
  specialized, not more accessible. See `PHASE_3_ARCHITECTURE_REVIEW.md`
  §3.7.
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

- **`sound_repetition` fragment-ordering bug fixed and measured — the
  first real detector code change of Phase 2** — 2026-08-04. Recall
  0.000→0.920 on the full 499-clip benchmark; `Any` label exactly
  unchanged (pure type-reclassification). Root cause was deeper than
  previously documented (both orderings were broken, not just one) — see
  `VALIDATION.md` §8.2.1. A related Track B cache-staleness bug found and
  fixed alongside it (cache now stores ASR output only, detector output
  always recomputed fresh). See `PAPER_DECISION_LOG.md`.
- **Roadmap reprioritized by evidence: second-ASR-backend gate lifted for
  three detector-side items** — 2026-08-04. Explicitly re-examined
  whether item 10 (second ASR backend) remained a necessary prerequisite
  for the prolongation redesign and the `sound_repetition`/
  `phrase_repetition` fixes, now that speaker-stratified Track B had
  completed. Decision: no — all three rest on Track A/literature evidence
  that never involved ASR. Item 10 re-scoped from blocking prerequisite to
  parallel generalization check. See `PAPER_DECISION_LOG.md`.
- **"Whisper did not predict an ending timestamp" warning investigated and
  confirmed external** — 2026-08-04. Verified via direct evidence (not
  assumed): our own token budget has ample headroom on every clip checked;
  the warning originates from `transformers`' own Whisper decoding logic.
  No fix needed. See `ARCHITECTURE.md` §3 and `PAPER_DECISION_LOG.md`.
- **Speaker-stratified Track B: the "~0% detector-attributable" finding
  revised, not confirmed** — 2026-08-04. Resolved the Phase 1 closing
  review's speaker-clustering caveat directly: re-ran Track B across all
  40 speakers (120 clips, pre-registered before running). `R_B|preserved_
  ctx1` recall dropped from 1.0 (7 speakers) to 0.667 (40 speakers);
  decomposition revised to 35.1% detector-attributable / 64.9%
  ASR-attributable — ASR-fidelity remains the majority driver. Hand-traced
  every miss to `sound_repetition`/`phrase_repetition`'s already-known
  structural gaps (item 10), not `word_repetition` or a new failure mode.
  See `VALIDATION.md` §8.4.3 and `PAPER_DECISION_LOG.md`.
- **Per-type definition audit: literature vs. dataset vs. implementation**
  — 2026-08-03. Full audit: `PHASE_2_RESEARCH_PLAN.md` §10. For each of
  the 7 types, checked whether our detector's exact operational trigger
  matches the clinical/scientific definition or only the dataset's own
  labeling shortcut. Two real, now-quantified gaps found — both already
  top priorities, sharpened not newly discovered: `prolongation`'s
  effective threshold is ~2–4× higher than the literature's rate-
  normalized standard (Esmaili et al. 2017); `block`'s silence-only rule
  tests a necessary-but-not-sufficient proxy for the clinical
  (effort/struggle-based) definition, and SEP-28k's own schema shows human
  annotators already make a pause-vs-block distinction this detector
  cannot. Every other type's simplification matches one the benchmark
  datasets themselves make (documented, not actioned);
  `phrase_repetition` found to be more faithful to the literature than any
  available dataset's own label. See `PAPER_DECISION_LOG.md`.
- **Adversarial self-review of the Phase 2 plan + Step 1 implemented and
  benchmarked** — 2026-08-03. Actively tried to disprove
  `PHASE_2_RESEARCH_PLAN.md`'s own conclusions (`PHASE_2_RESEARCH_PLAN.md`
  §9); found and fixed weak sourcing on the prolongation-first claim
  (upgraded from one preprint to two peer-reviewed sources), directly
  addressed the rule-based-vs-deep-learning tension, found no direction
  strong enough to change the plan's ordering. Implemented Step 1
  (taxonomy/documentation refinements: `word_repetition` SLD/OD sub-tag,
  explicit dataset-validation-status labeling, silent-only-block
  documentation) and **benchmarked it against the frozen Phase 1 Track A
  baseline — confirmed byte-for-byte identical**, proving the change is
  purely additive. See `PAPER_DECISION_LOG.md`.
- **Phase 2 opening literature review: taxonomy checked against the field
  before any implementation** — 2026-08-03. Full review:
  `PHASE_2_RESEARCH_PLAN.md`. Core 5-type taxonomy (filler, sound_repetition,
  word_repetition, block, prolongation) confirmed scientifically sound
  against clinical speech-pathology and computational-detection literature —
  no redesign needed. Found a real, cheap, additive taxonomy refinement
  (monosyllabic/polysyllabic split for `word_repetition`); confirmed a real,
  previously-undocumented architectural gap (`block` is silent-only, no
  audible/struggle sub-type); identified prolongation as the
  highest-confidence Phase 2 detector-side target via three converging
  sources of evidence; independently corroborated two of Phase 1's own
  findings via unrelated external studies. See `PAPER_DECISION_LOG.md`.
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
  resulting priority changes: see "Phase 2 is closed" items 8-10 above.
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
  audio — see "Phase 2 is closed" above).
- **First real Track A result** (499 real LibriStutter clips, text-only) —
  2026-08-03. See `VALIDATION.md` §8.2 and `PAPER_DECISION_LOG.md`.
- **First research baseline: audio-enabled Track A result** (same 499
  clips, real audio, the audio-native layer evaluated against ground truth
  for the first time) — 2026-08-03. `Any` F1 0.773 → 0.835. Includes a
  real bug found and fixed (`soundfile` silently decoding real FLAC files
  as silence) before the result was trusted. See `VALIDATION.md` §8.3 and
  `PAPER_DECISION_LOG.md`.
