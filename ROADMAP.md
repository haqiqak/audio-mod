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

## First-principles reassessment (2026-08-05) — a strategic gut-check, not an evidence log

**What this section is, and isn't.** Every other section of this file is a
chronological record: what was proposed, what was found, what shipped. This
section is different in kind — it's a deliberate exercise in setting that
whole record aside and asking, from the project's stated objective alone
("given arbitrary speech recorded through a microphone, detect, classify,
and localize disfluencies as accurately as possible so they can later be
corrected"), whether the path the record describes is still the path worth
being on. It was written in response to the project owner asking exactly
that — not "what did we do" but "if you'd never seen this roadmap, what
would you do next?" Where it disagrees with the rest of this file, that
disagreement is the point; it is not a correction of prior entries (which
stay, per this file's own append-discipline) but a fresh answer to a
different question, checked against the actual code rather than against
this file's own prior narrative.

**The finding that anchors it.** `profiling/evaluation/track_a.py`'s
`evaluate()` calls `detect_disfluencies(clip.tokens, audio_bytes=clip.
audio_bytes)` — and `clip.tokens` are LibriStutter's own ground-truth/
reconstructed annotation tokens (via `load_libristutter_dir_with_audio`),
never CrisperWhisper's real transcription output. That is Track A by
design (§3 of this project's evaluation methodology) — a legitimate,
pre-registered, honestly-labeled thing to measure. But it means **every
piece of Phase 3 work to date** — Stage 1's encoder-signal validation
(`run_encoder_signal_stage1.py`), the raw-embedding collection
(`collect_raw_encoder_data.py`), the corroboration-mechanism comparison
(`compare_corroboration_mechanisms.py`), and the integrated-gate benchmark
that produced item 17's headline `Any` F1 0.631→0.890 result
(`benchmark_integrated_gate.py`) — reuses that same ground-truth-token
loading path. None of it has ever seen real ASR output. And separately:
Track B (the one harness that *does* run real ASR) was last executed as
120 speaker-stratified clips on 2026-08-04, **before** the `sound_
repetition` fix, the prolongation redesign, or the repetition classifier
existed. Nothing shipped in Phase 2 or Phase 3 has been re-checked against
Track B since. Both facts are independently verifiable by reading the
scripts named above; neither is a new measurement, just a noticing.

### 1. Would we still choose this architecture, starting today?

The two-stage ASR-then-detect split itself: yes — `PHASE_3_ARCHITECTURE_
REVIEW.md` already re-litigated this from scratch against 2024-2026
literature and found no alternative (SSL classifiers, end-to-end
audio-region models, joint ASR+detection training) decisively better
without infrastructure this project doesn't have. No new evidence here
changes that.

What I would *not* rebuild the same way is this project's **evaluation
architecture**. Track A is fast (no ~30-90s/clip ASR cost) and free to
iterate against, so nearly every fast feedback loop built this session —
encoder-distance stats, nested cross-validation, the classifier itself —
plugged into it by default, because it's the metric that's cheap to check
against, not necessarily the one that answers the project's actual
question. That's a defensible engineering shortcut for any single
experiment. Left unexamined across two full phases, it quietly became the
project's real optimization target.

### 2. Are we optimizing the right problems?

Split the last two phases' detector-side changes into two categories, and
the answer differs by category:

- **Transcript-fidelity-independent fixes** — the `sound_repetition`
  ordering fix (item 3) and the `prolongation` Praat-gating redesign
  (item 5). These are legitimately validated by Track A alone, because
  their correctness doesn't depend on what ASR wrote down: a Praat-measured
  pause is a Praat-measured pause regardless of the transcript, and an
  ordering bug in candidate matching only ever helps once a candidate's
  text has already survived to the detector. Track A is the right tool for
  these, and shipping them on Track A evidence was sound reasoning, not a
  shortcut.
- **Transcript-fidelity-dependent work** — the repetition classifier (item
  17). Its entire premise is "given a text-identical repeated-word
  candidate, is it a real disfluency or a coincidental repeat" — a
  question about candidates that *survived* to the detector. It says
  nothing about candidates ASR corrupted or dropped before they ever
  became candidates, and Phase 1's own numbers say that's where most of
  the real-world recall loss already lives (~99%→~6-15% Track A→B recall;
  35.1% detector-attributable vs. 64.9% ASR-attributable at full speaker
  diversity, item 2). The classifier improves precision among survivors —
  real, but narrower than "improves disfluency detection on arbitrary
  microphone speech," and untested against the actual bottleneck.

Verdict: not misdirected, but increasingly narrow. Two phases have made
the "given an intact transcript" case measurably better without a single
re-check of whether that translates to the "given real ASR output" case —
which is the actual stated objective, and which this project's own Phase 1
proved is not the same case.

### 3. Assumptions that no longer deserve to survive unexamined

1. **"Track A improvement → ship it, Track B validation can wait."**
   Correct for the two transcript-agnostic fixes above. It was never
   re-derived per decision, though, and the repetition classifier shipped
   under the same blanket assumption without the property (fidelity
   independence) that made it safe for the earlier two. This needs to be a
   per-decision check, not a standing default.
2. **"No training pipeline exists, so the learned tier stays deferred."**
   Literally false now. `train_repetition_classifier.py`, checkpointed
   data collection, and nested-CV regularization selection are real,
   working infrastructure, built this session. The "Near-term" section's
   "deferred learned tier" bullet still argues from the old premise —
   worth re-opening on its own merits (§6 below), not left parked on
   reasoning that predates the thing it's reasoning about.
3. **Item 10's framing (second ASR backend / real speech as a "parallel
   generalization check," not a prerequisite)** was correct reasoning for
   Phase 2's transcript-agnostic fixes. It does not straightforwardly carry
   over to the classifier, whose validity is specifically a question about
   real ASR behavior.
4. **The implicit assumption that a Track A F1 gain is informative about
   deployment value on its own, absent any Track B check.** This is the
   one most worth retiring as a default: Phase 1's own headline finding
   already contradicts it in general (most of this app's real-world
   behavior is governed by what ASR did to the audio, not by what the
   detector does with clean text), and nothing since has re-tested whether
   that finding still applies to what's shipped since.

### 4. Is the current roadmap still what I'd choose?

Not in its current ordering. Items 14-16 (dataset expansion) and item 10
(second ASR backend / real disfluent speech) currently read as
lower-priority / parallel — accurate for the era when the only shipped
work was transcript-agnostic. As written today, the roadmap's most natural
next step is item 18 (removing the classifier's live-app latency cost) —
polishing the delivery of a component whose real-world benefit hasn't yet
been checked at all. That's effort spent making an unvalidated result
faster, before spending a much cheaper amount of effort finding out
whether it's valid.

### 5. Locally optimal, or globally optimal?

Locally optimal, in a specific and nameable way: the project climbed the
gradient of the metric that's cheapest to iterate against (Track A — no
ASR cost, existing infrastructure) and built increasingly sophisticated
machinery on top of it (encoder embeddings, nested cross-validation, a
trained classifier), while the metric that's actually proven to reflect
the stated objective (Track B) has been run exactly three times in this
project's history — a 30-clip pilot, a 90-clip run, and the 120-clip
speaker-stratified run — all of them **before** any Phase 2 or Phase 3
detector change shipped. Zero Track B runs since. That is the textbook
shape of local optimization: real, honest, well-validated progress on the
easy-to-measure proxy, with no recent check on whether it moved the hard-
to-measure target.

### 6. Higher-impact directions that should replace some current Phase 3 work

**Elevate:**
- Re-run Track B (the existing 120-clip speaker-stratified sample — no new
  acquisition needed) with today's code, classifier gate on and off. Cheap
  relative to the work it would be validating, and directly answers
  whether anything shipped in Phase 2 or Phase 3 has moved the real-world
  number at all. This is the single highest-leverage thing not yet done.
- Item 10 (second ASR backend / FluencyBank): promote from "parallel
  generalization check" to a real prerequisite for trusting the next
  Track-A-only detector change, given the classifier precedent above.
- The deferred learned tier (WavLM/wav2vec2 classifier, "Near-term"
  section): re-open as a live candidate now that a training pipeline
  exists — evaluated Track-B-first this time, not Track-A-alone.

**De-prioritize:**
- Item 18 (removing the classifier's latency cost). Optimizing the
  delivery cost of a component whose real-world benefit is unverified is
  premature — verify the benefit first; the latency work is still worth
  doing, just not next.

### 7. The single biggest bottleneck

Not a specific undetected disfluency type, not the absence of a training
pipeline (that objection no longer holds — §3.2), not latency. It is that
**this project has no closed feedback loop between "a change looks good on
Track A" and a confirmed check that it helped on Track B.** Every answer
above traces back to this one gap. Closing it is not expensive — the
alignment and caching infrastructure already exists, and the sample
already exists — it has simply not been re-run since the things it would
be validating were built.

### What this implies for next steps

This section is analysis, not an authorization to act — per standing rule
4, no threshold/config/architecture change follows from it automatically.
The concrete, evidence-seeking next step it argued for is tracked as item
19 below, and items 10 and the deferred-learned-tier bullet were
re-flagged (not re-ordered or renumbered) accordingly. If a fresh Track B
run confirms the shipped Track-A wins transfer, this section's concern is
answered and the existing Phase 3 ordering can resume as-is. If it
doesn't, that gap — not any item currently below — becomes the project's
real next finding to chase, and the reason will be traceable (candidate-
generation loss vs. genuine transfer failure) using infrastructure that
already exists.

**Update, 2026-08-05 — item 19 executed, verdict: partial confirmation,
with a sharper finding than either branch anticipated.** The gate's own
mechanism transfers safely to real ASR text (a real, non-harmful 37% FP
reduction on the `word_repetition` candidates that exist, zero recall
cost) — but its real-world impact on this sample is negligible (`Any` F1
0.082 -> 0.085) because real ASR output starves both gated types of
candidates before the gate ever gets a chance to matter:
`word_repetition` candidate volume is ~7x lower per clip than Track A's
ground-truth-token rate, and `sound_repetition` produced **zero**
candidates across all 120 clips, either condition. Direct hand-check
(`VALIDATION.md` §14.1, point 3) found a specific, previously-
undocumented mechanism for the `sound_repetition` collapse: its candidate
check requires a literal sub-word fragment token in the ASR transcript,
and real verbatim ASR essentially never produces one — even on positions
it otherwise transcribes correctly — because it normalizes disfluent
fragments into the full word. This is narrower and more actionable than
the general "ASR fidelity is the bottleneck" framing: it names the exact
structural mismatch between the current detection strategy and what real
ASR output actually looks like. New item 20 below tracks the resulting,
now-highest-priority follow-up. Full results: `VALIDATION.md` §14.1.

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
    16 below) and re-run Track B against it. **Re-flagged, 2026-08-05
    (see "First-principles reassessment" above): this item's own framing
    below ("no longer a blocking prerequisite") was reasoned about
    transcript-agnostic Phase 2 fixes specifically — it does not
    automatically extend to item 17's classifier, whose validity is a
    question about real ASR behavior. Item 19 (a Track B re-run) is the
    cheaper, more immediate check; this item remains the deeper
    cross-backend generalization question behind it.**
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
item, added below as item 17 — now implemented and shipped (2026-08-05),
see item 17's own entry for the full result, and item 18 for the one
follow-on cost it left open.** See `PHASE_2_SUMMARY.md` §7 for the rest
of Phase 3's evidence-ranked shortlist (items 9, 14, 10/16, 15, and a VAD/
Praat-complexity revisit) — items 17/18 below now sit alongside that
list, not separate from it.

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

    **STATUS: DONE, 2026-08-05 — implemented, benchmarked, and shipped
    as the default.** §12.6's pre-registered follow-up (250 clips,
    nested-CV-tuned regularization) confirmed the classifier's advantage
    held and grew (`word_repetition` F1 gap +0.161→+0.223, `Any`
    +0.133→+0.178, 5/5 folds, non-overlapping ranges) — the decision
    rule was satisfied, so `CLAUDE.md` rule 8 was applied to its own
    conclusion: (S1, M3) adopted. Implemented as `profiling/
    repetition_classifier.py` (`require_repetition_classifier_
    confirmation`, default `true`), trained artifact committed at
    `models/repetition_corroboration_classifier.npz` — this project's
    first internally-trained, shipped model. Integrated-detector
    benchmark (honest, out-of-fold, not in-sample): `Any` F1 0.631→0.890
    (89% FP reduction, 10% recall cost). Two real bugs (eager model
    loading that would have broken the fast test suite) caught and fixed
    before the result was trusted. **One real, accepted, documented
    limitation carried forward, not resolved**: enabling this in the
    live app adds a second ~30-90s CrisperWhisper encoder pass on
    affected clips — `ARCHITECTURE.md` §4b, `VALIDATION.md` §13.2. See
    `PAPER_DECISION_LOG.md`'s 2026-08-05 "Decision executed in full"
    entry for the complete record.

18. **[New, 2026-08-05, follow-on from item 17] Remove the repetition-
    classifier gate's live-app latency cost** by restructuring
    `profiling/asr.py`'s core transcription call to capture encoder
    hidden states during the same forward pass it already makes, instead
    of `profiling/repetition_classifier.py` making a second, separate
    encoder-only call. Would make item 17's gate genuinely free at
    inference time (matching the original "zero added latency"
    assumption in `PHASE_3_ARCHITECTURE_REVIEW.md` §5.1) rather than
    costing a real ~30-90s on affected clips (`VALIDATION.md` §13.2).
    **Not attempted alongside item 17 deliberately**: `asr.py`'s
    transcription call is delicate and multiply-patched (documented
    workarounds for a beam-search timestamp bug, a tokenizer-mismatch
    issue, and more — see `asr.py`'s own module docstring) — touching it
    in the same session as validating a brand-new classifier would mix
    two real, separately-risky changes together. Scoped as its own
    future step, with its own testing (a full mic-record round-trip,
    per `requirements.txt`'s own standing warning about touching this
    code path) before being attempted.

19. ~~[New, 2026-08-05, from the "First-principles reassessment" section
    above] Re-run Track B on the existing 120-clip speaker-stratified
    sample with today's code, repetition-classifier gate on and off, before
    any further Track-A-only detector work~~ — **done, 2026-08-05.**
    Both conditions run against the identical, already-cached 120-clip
    sample (zero new ASR inference: `120/120 from cache` both runs), gate
    toggled via config only, per the pre-registered protocol
    (`VALIDATION.md` §14). **Result: technically "transfers"** (`Any` F1
    0.082 -> 0.085, `word_repetition` FP reduced 19 -> 12 at zero recall
    cost — the gate's mechanism is safe and correctly-behaved on real ASR
    text) **but the practical impact is negligible**, because real ASR
    starves both gated types of candidates before the gate ever gets a
    chance to matter: `word_repetition` candidate volume is ~7x lower per
    clip than Track A's ground-truth-token rate; `sound_repetition`
    produced **zero** candidates across all 120 clips, either condition.
    A direct hand-check (standing rule 3, since a flat zero across an
    entire sample is exactly the kind of dramatic result that needs
    auditing before being trusted) found the specific mechanism: LibriStutter's
    `sound_repetition` ground truth is a reconstructed fragment token, and
    real verbatim ASR normalizes disfluent fragments into the clean full
    word even when it transcribes that position correctly — so the
    fragment-repeat candidate check's input assumption (a literal
    sub-word fragment token in the transcript) essentially never holds for
    real ASR output, independent of ASR accuracy. This is a narrower,
    more actionable finding than the general "ASR is the bottleneck"
    framing — it names the exact structural mismatch. Full results and
    the hand-checked examples: `VALIDATION.md` §14.1. See
    `PAPER_DECISION_LOG.md`'s 2026-08-05 "Track B validation of the
    shipped repetition-classifier gate" entry for the complete record.
    **Classifier gate stays enabled (shipped default unchanged)** — it is
    confirmed safe and mildly beneficial, just not sufficient on its own.
    Item 20 below is the resulting, now-highest-priority follow-up.

20. ~~[New, 2026-08-05, highest priority — directly from item 19's
    result; now the opening move of a dedicated research track, see
    below] Investigate and, if a fix is evidence-supported, redesign how
    `sound_repetition` (and secondarily `word_repetition`) candidates are
    generated from real ASR output — not just how they're gated once
    found~~ — **Stage A done, 2026-08-05 (on the `asr-research` branch;
    not yet merged to `main`).** Systematically categorized all 186
    disfluent ground-truth positions in the 120-clip sample (not a
    hand-picked few) into four causes. Headline: for `sound_repetition`
    and `word_repetition`, ~50% of losses happen even when ASR
    transcribed the position correctly — confirming item 19's finding
    generalizes, not an edge case. The two types lose signal by different
    mechanisms: `sound_repetition` loses the literal fragment token;
    `word_repetition` loses the *pair* — a targeted follow-up found 22/23
    "correctly transcribed" `word_repetition` positions have the *other*
    half of the repeat deleted or displaced by ASR. "Mis-routed to a
    different type" (the original `block`-instead-of-`sound_repetition`
    lead) is real but modest (~10%), not the dominant recovery
    opportunity. The remaining ~45% of losses for both types are ordinary
    ASR transcription error, unrelated to normalization — a different,
    already-known problem this track doesn't expect to fix. Full
    breakdown, per-category examples, and the small-sample caveats:
    `ASR_RESEARCH_TRACK.md` §8 (Stage A results). Stage B (representation
    probe) is next.

21. **[New, 2026-08-05, small, detector-side, independent of the ASR
    research track — found while doing item 20's Stage A hand-trace]
    `word_repetition`'s exact-match candidate check appears to miss runs
    of 3+ identical adjacent words.** One case (clip `2092-145706-0025`)
    has a genuine triple repeat fully intact and adjacent in the ASR
    output (`['wolf', 'wolf', 'wolf,']`) yet was not flagged. This is a
    detector-logic question, not an ASR-representation one — out of
    `ASR_RESEARCH_TRACK.md`'s scope, flagged here instead. Not
    investigated further this session (n=1, found incidentally); worth a
    small, cheap look on `main` independent of the research track,
    starting with a targeted synthetic test for a 3-repeat run before
    assuming a fix.

**A separate research track opened from this checkpoint, 2026-08-05: see
`ASR_RESEARCH_TRACK.md`.** Item 19's finding — that real ASR structurally
discards information certain disfluency types need, independent of any
detector-side fix — was judged a bigger-than-one-item checkpoint: not
"how do we improve the detector" but "how do we preserve the
speech-production information conventional ASR intentionally removes."
That question gets its own charter document (problem statement,
literature review, architectural options explored without commitment,
phased evidence-gated research plan) and its own branch (`asr-research`),
kept separate so `main` stays stable and shippable throughout. Item 20
above was that track's Stage A (the systematic, no-new-data information-
loss audit, now done) — Stage B and beyond are defined in
`ASR_RESEARCH_TRACK.md` §8, not duplicated here. **Stage B (the encoder
representation-level probe) is also done, 2026-08-05: a mixed result —
positive for `sound_repetition` (Cohen's d = 0.894, clears the
pre-registered bar), inconclusive for `word_repetition` (d = 0.428, the
more indirect of the two tests). Stage C now proceeds scoped specifically
to `sound_repetition`, not extended to `word_repetition` on this
evidence.** **Stage C (encoder vs. duration-only baseline) is also
done, 2026-08-05: H1 (duration confound) refuted — the duration arm sits
at chance (AUC=0.483); H2 (genuine signature) supported — the encoder
arm clears chance decisively (AUC=0.723); H3 (real but not yet
actionable) also supported — absolute precision at a useful recall is
still low (4.7% at 52.6% recall) given realistic class imbalance. Not
ready to ship as a standalone candidate generator; the evidence-justified
next step is a fusion-style revision combining this signal with others,
not Stage D (fine-tuning) — the confound Stage D exists to react to has
been refuted, not confirmed.** **Stage C2 (Praat voice-quality fusion) is
also done, 2026-08-06: a clean negative — none of five Praat features
(pitch, pitch stability, jitter, shimmer, HNR) cleared even a low
AUC>=0.55 screening bar (all near chance), so the fusion combination step
was correctly not attempted. Rules out Praat features specifically as
this track's next signal for `sound_repetition`; does not touch Stage
C's own encoder-distance conclusion. With the mis-routing lead (n=4, too
small to test) and Praat both explored, low-cost fusion candidates are
largely exhausted at this sample size — three options recorded, not yet
chosen, in `ASR_RESEARCH_TRACK.md`'s end-of-session handoff update.**
**First-principles reassessment, 2026-08-06 — is this track (and
CrisperWhisper specifically) still the right trajectory, or attachment
to it?** Reassessed explicitly, evidence/inference/judgment kept
separate: **not** yet time to move to a different or purpose-built ASR —
that would be evidence-free in the opposite direction (RQ3, whether any
of this is CrisperWhisper-specific, has never been tested). What *is*
true: only the narrowest slice of the richer-representation direction
has been tried (one encoder layer, threshold-only, no decoding variation,
a modest sample) and that narrow slice is now returning diminishing/
negative results. Two concrete, cheap, never-tried experiments remain
squarely within the current architecture and are more evidence-motivated
than either extreme (preserving CrisperWhisper by default, or abandoning
it by default): **encoder layer depth** (only the last layer has ever
been used; literature found deeper layers carry more signal for a
comparable task) and **decoding-parameter sensitivity** (never varied;
the most direct available test of whether the loss is decoder-side, as
inferred, or something deeper). Both come before RQ3 and Stage D in the
plan now. **Update, 2026-08-06: the layer-depth sweep is done — a
decisive, clean negative.** The last encoder layer (Stage B/C's own
choice) is uniquely informative; no other of the 32 remaining layers
comes close (best runner-up AUC 0.383 vs. the last layer's 0.721-0.723,
verified consistent across two different control populations). Closes
that specific question — layer depth was never the missing lever. The
decoding-parameter experiment is now the sole remaining untested
in-architecture lever before this track's own stated logic would shift
weight toward RQ3/Stage D. Full reasoning: `ASR_RESEARCH_TRACK.md`'s new
"First-principles reassessment" section, at the top of the file.
Full results, controls, and limitations: `ASR_RESEARCH_TRACK.md` §8.

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
  full reasoning on why this wasn't built immediately. **Re-flagged,
  2026-08-05: the other half of the original deferral reasoning — "no
  training pipeline exists" — is now false (`train_repetition_classifier.py`
  and its nested-CV/checkpointing infrastructure exist and are proven on
  item 17). See "First-principles reassessment" above, §3 point 2 and §6:
  this item should be re-opened as a live candidate, evaluated against
  Track B from the start rather than Track A alone. **Sequencing update,
  2026-08-05**: item 20 now sits ahead of this one too — item 19's result
  shows real ASR barely produces any `word_repetition`/`sound_repetition`
  candidates in the first place, and a learned classifier trained on the
  same starved candidate population would inherit the identical ceiling.
  Understand and address candidate generation first (item 20); revisit
  this once there's a healthier candidate population to train against.**
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

Full detail for items already covered above (struck-through in the
numbered Phase 2/3 list) is not repeated here — see item 1-16's own text
for `sound_repetition`'s fix, the second-ASR-backend gate lift, the
speaker-stratified Track B result, and the Phase 1/2 opening reviews. This
section covers everything else: earlier work with no corresponding
numbered item above, plus a couple of one-line pointers back for
chronological completeness.

- **`sound_repetition` fragment-ordering fix** — see item 3 above.
- **Second-ASR-backend gate lifted for three detector-side items** — see
  the note directly above item 3 above.
- **"Whisper did not predict an ending timestamp" warning investigated and
  confirmed external** — 2026-08-04. Verified via direct evidence (not
  assumed): our own token budget has ample headroom on every clip checked;
  the warning originates from `transformers`' own Whisper decoding logic.
  No fix needed. See `ARCHITECTURE.md` §3 and `PAPER_DECISION_LOG.md`.
- **Speaker-stratified Track B (35.1%/64.9% detector/ASR attribution
  split)** — see item 2 above.
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
- **Adversarial self-review of the Phase 2 plan** — 2026-08-03. Actively
  tried to disprove `PHASE_2_RESEARCH_PLAN.md`'s own conclusions
  (`PHASE_2_RESEARCH_PLAN.md` §9): found and fixed weak sourcing on the
  prolongation-first claim (upgraded from one preprint to two peer-reviewed
  sources), directly addressed the rule-based-vs-deep-learning tension,
  found no direction strong enough to change the plan's ordering. See
  `PAPER_DECISION_LOG.md`. (Step 1's implementation itself is item 1
  above.)
- **Phase 2 opening literature review** — see "Phase 2 opening literature
  review is done" section above.
- **Phase 1 formally closed** — see "Phase 1 is closed" section above;
  full closing summary in `PHASE_1_SUMMARY.md`.
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
