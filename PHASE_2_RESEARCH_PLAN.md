# PHASE_2_RESEARCH_PLAN.md — literature-grounded taxonomy review and Phase 2 direction

**Status: §1–8 is the original literature review and plan (research and
planning only, no code changed as a result of it). §9 is a dated
adversarial self-review conducted before implementation began, per the
project owner's explicit instruction to actively try to disprove this
plan's own conclusions rather than assume them — it reinforced the
original ordering (with citations strengthened and one new nuance
recorded, see §9.5) rather than overturning it. §10 is a dated per-type
definition audit (literature vs. dataset vs. implementation) conducted
before the next implementation milestone — it found `prolongation` and
`block` have real, now-quantified definitional gaps (both already this
project's top two detector-side priorities, now sharpened rather than
newly discovered) and confirmed every other type's simplifications are
ones this project's own benchmark datasets also make. Step 1's
implementation follows this document**, tracked in `PAPER_DECISION_LOG.md`
and `CHANGELOG.md`. See `PHASE_1_SUMMARY.md` for where Phase 1 left off
and `ROADMAP.md` for how this document's conclusions are reflected in
current priorities.

**2026-08-04 note**: the speaker-stratified Track B run this plan's §7
Step 2 called for (`ROADMAP.md` item 3) has completed and revised — not
overturned — the underlying Phase 1 finding this document builds on: the
"~0% detector-attributable" figure held only at 7 of 40 speakers and is
now 35.1% at full diversity, traced to `sound_repetition`/
`phrase_repetition`'s already-known structural gaps (`word_repetition`
remains unaffected, at 100% recall given intact context). ASR-fidelity
remains the majority driver, so this document's priority ordering (§7) is
unaffected — full detail: `VALIDATION.md` §8.4.3,
`PAPER_DECISION_LOG.md`.

---

## 1. Purpose and scope

Phase 1 established *whether* the system works and *where* its real-world
bottleneck is (`PHASE_1_SUMMARY.md`). It did not question whether the
7-type taxonomy itself — `filler`, `sound_repetition`, `word_repetition`,
`phrase_repetition`, `block`, `prolongation`, `stutter_marker` — is
scientifically the right representation of disfluency in the first place.
That taxonomy was chosen in the 2026-08 restructuring to match SEP-28k/
FluencyBank/KSoF's own labels (`PAPER_DECISION_LOG.md`, "Vision alignment
review + architecture decision"), which was a reasonable practical choice
at the time, but was not itself checked against the speech-pathology and
computational-detection literature in depth. This document does that check:
what does the literature say each disfluency type actually *is*, acoustically
and linguistically; how do detection systems actually detect each one; how
do our specific candidate datasets annotate them; and, given all of that,
where is our current taxonomy/implementation aligned with the field, where
does it diverge, and does the divergence matter enough to fix — and if so,
in what order.

**Method**: targeted literature search (not an exhaustive systematic
review) across clinical speech-pathology taxonomy, acoustic/computational
detection literature, and the specific datasets already in scope from Phase
1 (§2 of `VALIDATION.md`). Every claim below is sourced; a claim from a
single non-peer-reviewed preprint is flagged as such and weighted
accordingly, consistent with this project's "measurement-first, cite what's
actually verified" discipline (`DOCS.md`).

---

## 2. What the literature says: clinical taxonomy

### 2.1 The stuttering-like vs. other-disfluencies (SLD/OD) distinction

The dominant clinical framework (Ambrose & Yairi, and the broader stuttering
research literature building on it) splits disfluencies into two categories
that are treated very differently:

- **Stuttering-Like Disfluencies (SLDs)**: part-word/sound repetitions,
  **monosyllabic** whole-word repetitions, blocks, prolongations, and broken
  words. These indicate problems at the *motoric* level and are the
  disfluencies that actually differ significantly in frequency between
  people who stutter (PWS) and people who don't (PWNS).
- **Other/Typical Disfluencies (ODs/TDs)**: interjections (fillers),
  **polysyllabic/multi-syllable** word repetitions, phrase repetitions,
  revisions, and interruptions. These are thought to reflect *linguistic
  planning* (stalls, prospective; revisions, retrospective), occur in
  virtually everyone's speech, and — critically — **did not differ
  significantly between stuttering and fluent groups** in the studies
  reviewed.

Two nuances worth stating precisely, because they bear directly on our
taxonomy:

1. **Word repetition's clinical classification depends on syllable count,
   not just "was a word repeated."** `"her-her-her"` (monosyllabic) is an
   SLD; `"I see... I see her"` (polysyllabic word/phrase repeated) is an OD.
   Our current `word_repetition` type does not make this distinction at
   all — every repeated word is scored identically regardless of syllable
   count.
2. **Even the clinical field has internal disagreement at the boundary**:
   the standard severity instrument (SSI-3/4) does *not* count monosyllabic
   word repetitions toward stuttering severity by default, unless they sound
   perceptibly tense/stuttered — i.e., even "SLD vs. OD" is not a clean
   acoustic bright line in practice, it's a graded judgment call clinicians
   make. This matters for how confidently any automated re-classification
   should be presented (as a probabilistic tag, not a hard reclassification).

*(Sources: PMC10465157 "Stalling for Time"; PMC3482136 "Disfluency Patterns
and Phonological Skills Near Stuttering Onset"; the Ambrose & Yairi
normative-disfluency-data literature; PMC4724559 "The Function of Repeating:
The Relation Between Word Class and Repetition Type in Developmental
Stuttering.")*

### 2.2 Block vs. prolongation: two acoustically distinct phenomena, and a
third sub-case we don't implement

Prolongations extend a consonant or vowel sound beyond its typical
duration — continuous sound production, extended in time. Blocks are
fundamentally different: a stoppage of airflow and voicing due to laryngeal
or articulatory tension. Critically, the literature identifies **two
acoustically distinct block sub-types**:

- **Silent blocks**: genuine silence, detectable via extended
  no-audio-energy duration.
- **Audible (struggle) blocks**: sustained low-amplitude, tension-indicating
  energy — the speaker is visibly/audibly struggling but not fully silent.
  This requires different acoustic features than silence detection (energy
  *pattern*, not energy *absence*).

**Our current implementation only detects the silent variant.** Verified
directly against the code (`profiling/detect.py`'s block branch, `§4` below
has the specifics): the block detector requires a literal inter-token time
gap (`gap >= block_gap_seconds`) *and* `_AcousticContext.gap_is_silent()`,
which is a pure RMS-below-threshold check. There is no code path anywhere
in `detect.py`/`acoustic.py` that can fire on an audible/tense block — a
speaker straining audibly through a block, without a clean silent gap
between ASR-recognized tokens, is invisible to this detector today. This is
a genuine, literature-identified, previously-undocumented coverage gap —
distinct from the already-known `sound_repetition` fragment-ordering gap
(`VALIDATION.md` §8.2).

*(Sources: multiple clinical overviews of the three stuttering types
[Expressable, GreatSpeech, StatPearls]; the rule-based detection paper in
§3 below, which explicitly implements both sub-types and reports blocks as
its hardest-to-detect category.)*

---

## 3. What the literature says: computational/acoustic detection strategies per type

### 3.1 Dataset-standard taxonomy: strong, direct validation of our core 5

Every major computational stuttering dataset reviewed uses essentially the
**same 5-type taxonomy**: block, prolongation, sound repetition, word
repetition, and interjection (filler).

- **SEP-28k** (Lea et al. 2021, ICASSP): block, prolongation, sound
  repetition, word repetition, interjections — 5 types, clip-level,
  3-annotator agreement, no reference transcript.
- **FluencyBank** (Bernstein Ratner & MacWhinney): same taxonomy, CHAT-format
  transcriptions, standardized across languages; ~4,000 3-second clips
  bundled alongside SEP-28k specifically.
- **KSoF** (Bayerl et al. 2022, LREC): the same 5 types **plus** "modified
  speech" (fluency-shaping, therapy-specific — not applicable outside a
  therapy context).
- **UCLASS** (Howell et al. 2009): 118 recordings, event-level annotations
  by trained SLPs including "dysfluency type, duration, and severity" per
  the rule-based paper in §3.2 (worth directly re-verifying UCLASS's exact
  schema before relying on the "severity" claim — flagged as an open
  verification item in §7, not yet independently confirmed by this project).

**This directly confirms Phase 1's original taxonomy alignment decision was
right**: our `filler`, `sound_repetition`, `word_repetition`, `block`,
`prolongation` core-5 matches the field's actual annotation practice across
every reviewed dataset, not just SEP-28k in isolation. This review does
**not** find grounds to change the core 5.

**What is *not* validated anywhere**: `phrase_repetition` and
`stutter_marker` (our two remaining types) appear in **none** of the
reviewed datasets' label sets as distinct categories. This was already
known from Phase 1's own dataset comparison (`VALIDATION.md` §2) — this
review's contribution is confirming it holds across a wider literature
sweep, not just the datasets Phase 1 happened to pick. `phrase_repetition`
also maps, per §2.1's SLD/OD framework, to the *Other Disfluencies* category
— i.e., even where it's clinically recognized at all, it's explicitly the
*less diagnostically significant* category, not an oversight in how
datasets label things.

### 3.2 Per-type detection strategy, and which types rule-based methods
handle well

A directly relevant preprint, "Revisiting Rule-Based Stuttering Detection:
A Comprehensive Analysis of Interpretable Models for Clinical Applications"
(Zhang, SSHealth Team, arXiv:2508.16681, **preprint, not peer-reviewed —
weighted as suggestive evidence, not settled fact**), evaluates a rule-based
system directly comparable in spirit to this project's own architecture
(interpretable, acoustic-feature-based, no training data required) across
UCLASS/FluencyBank/SEP-28k. Findings directly relevant to this project:

- **Prolongation is where rule-based/interpretable methods are strongest**:
  0.97–0.99% accuracy is reported as achievable with speaking-rate-normalized
  duration thresholds (`T_min = α / speaking_rate`, `α≈1.2`) combined with
  MFCC frame-to-frame spectral correlation (>0.92), F0 stability (Δf0 <
  15Hz), and harmonic-to-noise ratio (>10dB) as *joint, core* detection
  criteria — not applied as post-hoc confidence adjustments the way this
  project currently uses its own Praat features (jitter/shimmer/pitch/HNR;
  `VALIDATION.md` §9.3 already found these are confidence-only and
  therefore invisible to a presence/absence metric). This paper's own
  per-type table (UCLASS) reports prolongation F1 0.98 for their rule-based
  system, essentially matching neural SSDM's 0.96 — the smallest rule-vs-
  neural gap of any type.
- **Block is where rule-based methods are weakest**: F1 0.69 vs. neural
  SSDM's 0.79 — the largest gap of any type, attributed to blocks requiring
  "complex acoustic-prosodic interactions" beyond simple threshold rules,
  consistent with §2.2's silent/audible sub-type distinction adding real
  complexity a threshold-only rule can't capture.
- **Speaking-rate normalization is treated as essential, not optional.**
  "Fixed threshold" duration-based detection collapses from F1 0.81 (normal
  rate) to 0.39 (2× rate) in their own ablation; their rate-normalized
  version stays at 0.77 under the same stress test. **Our own prolongation
  detector does not do true syllable-rate normalization** — it uses a
  90th-percentile-of-the-clip's-own-token-durations threshold plus a fixed
  floor (`prolongation_min_seconds`), which adapts *coarsely* to a clip's
  overall pace but is not the same mechanism, and (unlike syllable-rate
  normalization) is sensitive to how many tokens are in the clip (already
  the reason Phase 1's own detector falls back to a flat multiplier below 5
  tokens, `ARCHITECTURE.md` §4).
- **A conflict-resolution precedence** (blocks > sound repetitions >
  prolongations > word repetitions, when multiple *type* rules fire on
  overlapping segments) is used for a different problem than our own fusion
  logic (which resolves *source* — acoustic vs. token-path — not *type*
  conflicts) — not directly transferable, but a useful reference point if
  cross-type conflicts are ever found to matter here.
- **Error analysis independently corroborates two of this project's own
  findings**, from a completely different codebase and dataset: the
  single largest reported error category besides boundary imprecision is
  "missed coarticulation" — specifically **sound-repetition immediately
  preceding a prolongation being conflated** — and the third-largest is
  "natural pause labeled as block" false positives. Both are the *same
  class* of confusion this project independently found: the Track B
  type-classification gap (`word_repetition`/`sound_repetition` mislabeled
  as `phrase_repetition`/`block`, `VALIDATION.md` §8.4.2) and the original
  real-mic validation pass's "possible block over-flagging" observation
  (`PAPER_DECISION_LOG.md`, 2026-08-03 real-audio validation entry). This is
  independent, cross-project evidence that **type-confusion between
  adjacent/co-occurring disfluencies is a known, general difficulty in this
  field, not an implementation-specific defect** — reassuring in one sense
  (our detector isn't uniquely bad at this), and still a real, worthwhile
  engineering target in another (the field hasn't solved it either, so
  solving it well would be a genuine contribution, not a trivial fix).

### 3.3 Multi-label, not multi-class: an existing design choice this review
validates rather than challenges

Bayerl et al.'s "A Stutter Seldom Comes Alone" (Interspeech 2023,
arXiv:2305.19255) makes the case, across English/German/Mandarin, four
corpora, that stuttering detection should be treated as a **multi-label**
problem (a moment of speech can carry more than one true label
simultaneously) rather than forcing a single class per segment — and finds
that "performance on samples with multiple labels stays below overall
detection results," i.e., co-occurrence is a genuinely harder case, not a
labeling technicality. **This project's architecture already does this
correctly**: `detect_disfluencies()` allows multiple event types per token
(e.g., a fragment can carry both `stutter_marker` and, if followed by the
completed word, `sound_repetition`), and Phase 1's own metrics score
per-type independently rather than forcing single-label classification
(`VALIDATION.md` §4 point 2, citing this same paper). **No change indicated
here** — this is a point where the existing architecture is already aligned
with the literature's recommended approach, worth stating explicitly rather
than only cataloguing gaps.

### 3.4 Synthetic disfluency data: independent confirmation of Phase 1's
own generalization concern

Zhang et al., "Analysis and Evaluation of Synthetic Data Generation in
Speech Dysfluency Detection" (arXiv:2505.22029, **preprint**), evaluating
splicing- and TTS-based synthetic disfluency generation including
LibriStutter-style corpora, finds that **models trained/evaluated on
synthetic disfluent data show measurable performance degradation when
tested on authentic disfluent speech**, and that synthetic disfluencies
have identifiable acoustic differences from genuine ones — augmenting with
real disfluent data outperforms purely synthetic approaches. **This is
independent, external confirmation of exactly the concern this project's
own Phase 1 closing review raised** (`VALIDATION.md` §7.2 item 3: "LibriStutter's
disfluencies are synthetically spliced... may have acoustic/prosodic
discontinuities... that a real stutterer's natural disfluent speech doesn't
have"). It does not change Phase 1's conclusion, but it raises confidence
that the concern is real and general, not a one-off worry specific to this
project's setup — and reinforces that the top Phase 2 validation
priority already identified (checking the ASR-bottleneck finding against
real, non-synthetic disfluent speech — `ROADMAP.md` item 1) is
well-justified by the broader field's own findings, not just this
project's.

### 3.5 Evaluation-consistency issues across the field

Bayerl et al.'s "Classification of Stuttering – The ComParE Challenge and
Beyond" (Computer Speech & Language 2023) is cited (via the rule-based
paper, §3.2) as documenting cross-study evaluation inconsistency: event-based
vs. interval-based segmentation produce incomparable metrics, **label
definitions vary across corpora** (their specific example: whether filled
pauses even count as "interjections" differs by study), and speaker-
dependent vs. independent splits materially change reported numbers. This
is exactly the discipline `VALIDATION.md` already tries to enforce (never
blend Track A/B numbers, always report per-dataset, the speaker-exclusive-
splits principle in §4 point 5) — this review finds the literature agrees
this discipline is necessary, not overcautious, and specifically flags that
this project's own filler/interjection definition should be double-checked
against each dataset's specific convention before any cross-dataset filler
comparison is drawn (not yet done — `VALIDATION.md` §8.2 already found zero
filler ground truth in the current LibriStutter sample, so this hasn't been
exercised yet in practice).

---

## 4. Gap analysis: our current taxonomy/implementation vs. the literature

| Type | Literature/dataset status | Our implementation | Gap found | Action |
|---|---|---|---|---|
| `filler`/interjection | Core 5, all datasets. Clinically an OD (linguistic, not motoric). | ASR flag or known-word list, acoustically corroborated. | None structural. Definition-consistency with each dataset's own filler convention not yet double-checked (§3.5). | Verify definitions per-dataset before any cross-dataset filler claim; otherwise no change. |
| `sound_repetition` | Core 5, all datasets. SLD (motoric). | Fragment-then-repeat pattern, one specific ordering only (`VALIDATION.md` §8.2's known 0% recall gap). | Confirmed structural gap, already known — literature's DTW/periodicity-based approach (§3.2) is a more general alternative worth considering when this is eventually fixed. | Already tracked in `ROADMAP.md`; this review adds a candidate technique (periodicity/DTW) to consider when scoping the fix. |
| `word_repetition` | Core 5, all datasets, but **clinically split** by syllable count (SLD if monosyllabic, OD if polysyllabic — §2.1). | No syllable-count distinction; every repeat scored identically. | Real, literature-backed, currently unaddressed distinction with clinical significance. | **New, scoped, low-risk candidate** — see §5. |
| `phrase_repetition` | **Not in any reviewed dataset's label set.** Clinically an OD when recognized at all. | Detected, reported as co-equal with SLD types. | Confirmed (not new) — already known from Phase 1's own dataset review, now confirmed across a wider sweep. | Keep implemented (real signal, just not benchmarkable) but label its diagnostic/validation status explicitly wherever reported — see §5. |
| `block` | Core 5, all datasets. **Two acoustically distinct sub-types** (silent, audible/tense) in the literature — §2.2. | **Silent-only.** No code path for audible/tense blocks (verified directly, §2.2). | **New, real, previously-undocumented coverage gap.** Also the type rule-based methods handle worst in the literature (§3.2) — a harder problem generally, not just for us. | Documented as a real gap; **not** a Phase-2-opening priority — see §5/§6 for why (dataset-validatability). |
| `prolongation` | Core 5, all datasets. Rule-based methods' *strongest* type in the literature, given rate-normalization + multi-feature core rules (§3.2). | Percentile+floor duration threshold; Praat features (jitter/shimmer/F0/HNR) computed but used only as post-hoc confidence adjustment, never core detection criteria (confirmed inert-to-current-metric by Phase 1's own ablation, `VALIDATION.md` §9.3). | **Highest-confidence improvement candidate found by this review** — converges with Phase 1's own ablation finding (this threshold already dominates measured performance) and with dataset compatibility (fully validated everywhere). | **Top candidate for Phase 2's first architecture-level change** — see §5/§6. |
| `stutter_marker` | **Not in any reviewed dataset or clinical framework** — an ASR-artifact-derived category specific to this project's pipeline (a trailing `-` or ASR's own `is_stutter` flag). | Detected, reported as co-equal with SLD types. | Confirmed (not new). | Same treatment as `phrase_repetition` — see §5. |
| Multi-label event structure | Recommended by the literature (§3.3). | Already implemented this way. | **None — a validated existing strength**, not a gap. | No change. |

---

## 5. Dataset-compatibility analysis for each candidate change

Per the explicit instruction not to redesign the taxonomy in a way that
breaks benchmarking against the datasets Phase 1 already validated against:

1. **Monosyllabic/polysyllabic sub-tagging of `word_repetition`**: SEP-28k/
   KSoF/FluencyBank do **not** annotate this split as a separate label —
   they just say "word repetition." But syllable count is **computable
   directly from the repeated word itself**, with no new annotation needed
   — and this project already computes syllable count for
   `profile.difficulty()` (`profiling/profile.py`, `factors_for_word()`).
   Adding an internal, computed `sld_likely: bool` (or similar) tag to
   `word_repetition` events would be **fully additive and backward
   compatible**: Track A/B's existing scoring against SEP-28k/LibriStutter's
   binary `word_repetition` label is completely unaffected (the sub-tag is
   extra metadata, not a new required field), so this does not break any
   existing benchmark. **Feasible, low-risk, literature-justified.**
2. **Explicit dataset-validation-status labeling for `phrase_repetition`/
   `stutter_marker`**: pure documentation/reporting change (already
   partially done in `VALIDATION.md` §2's "no dataset validates
   `stutter_marker`" note) — zero compatibility risk, since nothing about
   detection or scoring changes, only how results are presented.
3. **Audible/tense block detection**: **not validatable against any
   currently-available dataset** — SEP-28k/KSoF/FluencyBank's `block` label
   is a single undifferentiated category (no silent/audible sub-type in
   the label itself, confirmed via the same dataset review Phase 1 already
   did, `VALIDATION.md` §2). Building this without a way to *measure*
   whether it's working would repeat exactly the mistake Phase 1's whole
   premise was designed to avoid (anecdotal, unmeasurable self-testing).
   **UCLASS is flagged as worth directly re-verifying** — the rule-based
   paper's claim that UCLASS has "severity" annotations *might* imply finer
   sub-typing, but this project has not independently confirmed that yet
   (Phase 1's own dataset review rated UCLASS "Tier 3... older/less
   standardized annotation format," `VALIDATION.md` §2 — these two
   characterizations aren't necessarily in conflict, but the specific
   question "does UCLASS distinguish silent vs. audible blocks" has not
   been directly checked). **Recommendation: do not build this in Phase 2's
   opening work; flag as a specifically-scoped follow-up investigation
   (verify UCLASS's exact schema) before deciding whether it's buildable
   with real validation at all.**

   **Addendum (2026-08-04, `ROADMAP.md` item 11): the follow-up
   investigation this point recommended has now been done.** Checked, in
   order: (a) the primary UCLASS archive paper itself (Howell et al. 2009,
   PMC open-access copy) — describes only *recording-level* perceptual
   quality ratings (background noise, clarity), explicitly **not**
   event-level severity, and points to an external "How We Transcribe"
   page (`speech.psychol.ucl.ac.uk`, "Shared Resources") for the actual
   dysfluency-type/annotation conventions rather than specifying them in
   the paper; (b) that external page's certificate no longer matches its
   old domain — link rot, consistent with `VALIDATION.md` §7's standing
   caveat that some of this project's dataset-acquisition links can rot
   over time; (c) UCLASS's own current file-directory page
   (`uclass.psychol.ucl.ac.uk/Transcript/TAligned/Annotation/`) is reachable
   but is a bare listing of downloadable SFS/CHAT/PRAAT-TextGrid annotation
   files with no methodology documentation alongside it; (d) the specific
   rule-based-detection preprint that originally motivated this check does
   claim UCLASS has "detailed event-level annotations including dysfluency
   type, duration, and severity," but cites only Howell et al. 2009 for
   that claim — the same primary paper that (a) shows does not itself
   describe event-level severity.

   **Conclusion: inconclusive from every public secondary source checked,
   and the one claim asserting UCLASS has event-level severity is not
   substantiated by the primary source it cites.** This does not prove
   UCLASS *lacks* a silent/audible block distinction — the actual answer
   may only be recoverable by downloading and directly inspecting the raw
   SFS/CHAT/TextGrid annotation files themselves (not attempted here; a
   materially larger effort than this scoped literature check, and gated
   behind UCLASS's own access process per `VALIDATION.md` §2's "Tier 3...
   request-access" characterization). **Decision: `ROADMAP.md` item 12
   (audible/tense block detection) stays exactly where it was — not
   started, pending real validation data — with this addendum replacing
   "not yet independently confirmed" with a concrete, dated negative
   result** rather than leaving the question open indefinitely. Revisit
   only if this project ever gets direct access to UCLASS's raw annotation
   files, or a primary source with more explicit annotation documentation
   surfaces.

4. **Prolongation core-detection redesign (rate-normalization + Praat
   features as detection criteria, not just confidence)**: prolongation is
   validated in every candidate dataset (LibriStutter, SEP-28k, KSoF,
   FluencyBank, UCLASS) — **the safest possible type to redesign from a
   benchmarking-compatibility standpoint**, since Track A/B's existing
   scoring machinery needs no changes to evaluate it, only a re-run.

---

## 6. Should Phase 2 focus on a subset of types rather than all 7 at once?

**Yes — recommended explicitly, with a concrete priority order, justified
by three independent, converging sources of evidence** (not just one):

1. **Phase 1's own ablation study** (`VALIDATION.md` §9): `prolongation_min_seconds`
   already dominates measured performance by an order of magnitude over
   every other tunable component — the biggest lever this project has
   already found empirically, before this literature review even started.
2. **This review's literature findings** (§3.2): prolongation is the type
   where rule-based/interpretable methods are most reliable and best
   validated, given the right (rate-normalized, multi-feature) detection
   rule — and this project already computes most of the needed features
   (Praat pitch/jitter/shimmer/HNR) but doesn't use them as core detection
   criteria yet.
3. **Dataset compatibility** (§5): prolongation is validated everywhere,
   with zero benchmarking risk to redesign.

No other type has this three-way convergence. Recommended priority order
for Phase 2's *detector-side* work specifically (distinct from, and not a
replacement for, the ASR-robustness/speaker-diversity validation work
`ROADMAP.md` already prioritizes from Phase 1):

1. **Prolongation** — redesign core detection to use rate-normalization and
   already-computed Praat features as detection criteria, not just
   confidence adjustment. Highest-confidence candidate found.
2. **`word_repetition`/`sound_repetition` type-classification fix** —
   already scoped and confirmed by Phase 1 (`ROADMAP.md`, the
   hypothesis-side-contiguity gap); this review adds the SLD/OD
   monosyllabic sub-tag as a cheap, compatible addition to the same area of
   code.
3. **`sound_repetition`'s structural fragment-ordering gap** — already
   known (`VALIDATION.md` §8.2); this review adds a candidate alternative
   technique (periodicity/DTW matching, §3.2) worth considering when it's
   eventually scoped.
4. **Block** — keep the existing silent-block detector as-is (reasonably
   evidenced, not touched by this review's findings); explicitly do **not**
   build audible/tense-block detection yet, pending the UCLASS
   schema-verification flagged in §5.
5. **`phrase_repetition`/`stutter_marker`** — de-prioritized, not abandoned.
   No dataset can currently validate improvements to either, so further
   engineering investment here cannot be evidence-based right now under
   this project's own standing discipline. Keep them running (they're not
   broken, just unvalidated) and revisit only if a dataset with matching
   annotations becomes available or real product usage specifically
   surfaces them as a problem.
6. **Filler** — lowest priority. Phase 1 found zero ground-truth filler
   instances in the current LibriStutter sample (a sampling gap, not a
   detector problem, `VALIDATION.md` §8.2), and the clinical literature
   independently classifies interjections as the lowest-diagnostic-value
   category (§2.1). Consistent low priority from both the data and the
   clinical framing.

---

## 7. Structured Phase 2 plan: what the first step actually is

The project owner asked directly: should Phase 2 begin with redefining the
taxonomy, redesigning architecture, improving ASR robustness, improving
individual classifiers, or something else? **The honest answer is that
this isn't a single choice — the evidence points to a specific ordering
across more than one of these, not one winner-take-all direction:**

**Step 0 (this document): literature-grounded taxonomy review — done.**
Conclusion: the core 5-type taxonomy is scientifically sound and matches
the field; no wholesale taxonomy redesign is justified. Two scoped
refinements and one high-confidence architecture-level candidate were
identified instead.

**Step 1 (immediate, cheap, low-risk — taxonomy/documentation refinement,
not a detection-logic rewrite):**
- Add a computed monosyllabic/polysyllabic (SLD-likely/OD-likely) sub-tag
  to `word_repetition` events (§5 point 1) — additive, backward-compatible,
  literature-justified.
- Explicitly label `phrase_repetition`/`stutter_marker` as
  "not validated against any current public dataset" everywhere they're
  reported (extending the existing `VALIDATION.md` §2 note to `README.md`
  and the app's own UI/event table), and explicitly document the
  now-confirmed silent-only block coverage gap (§2.2) in `ARCHITECTURE.md`'s
  known-limitations section.
- This step produces no new detection behavior and needs no new Track A/B
  runs to validate — it's a documentation and metadata change, appropriate
  to do before anything else because it's how Step 3 below gets reported
  honestly once results start coming in.

**Step 2 (in parallel with Step 1, already Phase 1's own top `ROADMAP.md`
priority — this review does not change its priority, only reinforces it):**
Validate the confirmed "ASR is the bottleneck" conclusion against a second
ASR backend and/or real (non-synthetic) disfluent speech, and re-sample
Track B across speakers. §3.4 of this review independently reinforces why
this matters — the synthetic-data generalization concern is a known,
general finding in the field, not a one-off worry.

**Step 3 (pre-registered before implementation per this project's standing
discipline): the prolongation core-detection redesign** (§6 item 1) — the
single highest-confidence architecture-level change this review
identified. **Originally gated on Step 2's outcome in full; that gate was
explicitly re-examined and lifted for this item on 2026-08-04**
(`PAPER_DECISION_LOG.md`, "Is a second ASR backend still necessary before
detector-side work..."; `ROADMAP.md` item 5): Step 3's own evidence
(Phase 1's Track A ablation, peer-reviewed rate-normalization literature)
never depended on which ASR backend this project uses, so it does not need
Step 2's remaining half (a second ASR backend, `ROADMAP.md` item 10) to
proceed. The speaker-diversity half of Step 2 (`ROADMAP.md` item 2) *did*
complete first and materially informed this decision — see §10 of this
document's 2026-08-04 update and `VALIDATION.md` §8.4.3.

**Step 3b, elevated to equal priority alongside Step 3 on 2026-08-04**:
fixing `sound_repetition`'s fragment-ordering gap and `phrase_repetition`'s
reconstruction-caused unvalidatability (`ROADMAP.md` item 3) — these are
now the *entire, confirmed* explanation for the detector-attributable
share of the Track B gap (`VALIDATION.md` §8.4.3), proven via Track A
alone, and equally gate-independent.

**Step 4: the `word_repetition`/`sound_repetition` type-classification fix**
already scoped in `ROADMAP.md`, informed by this review's SLD/OD framing
where relevant (§6 item 2).

**Not Phase 2's opening work, explicitly deferred with reasoning (not
silently dropped)**: audible/tense block detection (§5 point 3 — not
currently validatable); `phrase_repetition`/`stutter_marker` algorithm
improvements (§6 item 5 — not currently validatable); a wholesale taxonomy
redesign (§3.1/§6 — not supported by the evidence; the core 5 is already
correct).

---

## 8. What this review deliberately did not do

- **Did not implement any of the above.** Per the project owner's explicit
  instruction, this document is research and planning; Step 1's actual
  code changes, and any pre-registration for Step 3, are separate,
  future, explicitly-approved work.
- **Did not treat the two preprints (arXiv:2508.16681, arXiv:2505.22029)
  as peer-reviewed fact.** Both are cited as suggestive, converging
  evidence — particularly valuable where they *independently* corroborate
  a finding this project already reached on its own (§3.2's error-pattern
  overlap, §3.4's synthetic-data concern), less weight is placed on any
  claim from them that stands alone.
- **Did not exhaustively re-verify every dataset's exact schema from
  scratch** — built on Phase 1's own dataset review (`VALIDATION.md` §2)
  where it already existed, and flagged the one specific claim (UCLASS's
  possible severity/sub-type annotations) that would need direct
  verification before being relied on, rather than assuming the literature
  search's summary of it is precise enough to act on.

---

## 9. Adversarial self-review (2026-08-03, same day, before any
implementation) — does this plan survive scrutiny?

Written as a dated addendum, not a silent rewrite of §1–8, per this file's
own stated discipline (`DOCS.md`). The project owner asked for this plan to
be actively challenged, not assumed correct — including checking for
stronger evidence, better sources, overlooked directions, and asking
whether the proposed *ordering* is actually optimal. Method: re-searched
specifically for counter-evidence and stronger sources, not confirmation.

### 9.1 Was the prolongation-first recommendation resting on weak evidence?

**Partially, yes — and this review fixes it, without changing the
conclusion.** §3.2's case for prolongation leaned heavily on a single
non-peer-reviewed preprint (arXiv:2508.16681). Actively hunting for
independent confirmation or disconfirmation found two much stronger
sources that were missing from the original pass:

- **Esmaili et al. 2017** (*Journal of Medical Signals and Sensors*,
  peer-reviewed, PubMed-indexed) — independently reports **99%/97.1%
  prolongation-detection accuracy on UCLASS/Persian corpora respectively**,
  using speaking-rate-normalized spectral-correlation thresholds, with
  demonstrated robustness from 70–130% of normal speaking rate. This is
  the *same* rate-normalization mechanism §3.2/§6 recommend, reported
  independently of the shaky preprint, on a real peer-reviewed study,
  years before it.
- **A genuine peer-reviewed systematic review** ("Computational
  Intelligence-Based Stuttering Detection: A Systematic Review," PMC10706171,
  synthesizing 14 studies) **independently confirms both halves of the
  claim**: prolongation and interjection are consistently reported as the
  *easiest* types to detect reliably; blocks are consistently reported as
  the *hardest* ("particularly difficult to detect," attributed to
  "variable manifestation and brief temporal duration").

**Net effect: the prolongation-first conclusion is now resting on stronger
ground than when §3.2/§6 were first written**, not weaker — two
peer-reviewed, independent sources replace reliance on one preprint.
**Correction applied**: the preprint is demoted to tertiary/consistent-but-
unverified evidence; Esmaili 2017 and the PMC systematic review are now the
primary citations for "prolongation is the most tractable type for
interpretable detection, block is the hardest." This is exactly what an
adversarial check is supposed to do — not necessarily reverse a conclusion,
but find out whether it was standing on solid ground, and fix it if it
wasn't.

**A real discrepancy was also caught in the process, worth flagging rather
than quietly resolving**: the original preprint claims UCLASS has 118
recordings; the systematic review states 457. This project has not
independently verified which (if either) is exactly right — both may be
describing different UCLASS subsets/versions. This reinforces, rather than
introduces, the already-flagged action item (`ROADMAP.md` item 11: verify
UCLASS's schema directly) — now also verify its actual size before citing
either number again.

### 9.2 Is a discrete-event taxonomy the wrong representation entirely?

**Actively checked; not overturned, but a real, honest tension is now
named instead of left implicit.** The most credible technical challenge to
this project's whole taxonomy-based approach comes from a specific,
well-resourced research thread (Lian et al., Anumanchipalli lab at
Berkeley: SSDM/SSDM 2.0, Dysfluent-WFST, YOLO-Stutter, Stutter-Solver),
which models dysfluency as a **continuous, phonetic/articulatory
representation** (edit operations against a fluent reference, via
articulatory-gesture-based forced alignment) rather than a fixed set of
discrete event *types* at all. This is a genuinely different, and in some
ways more expressive, scientific paradigm — it doesn't have to force every
real disfluency into one of 5–7 boxes.

**Why this project does not adopt it now, stated as a reasoned decision,
not a gap**: Phase 1 already directly considered and rejected end-to-end
neural dysfluency-region models for this exact reason class
(`ROADMAP.md`, "Explicitly rejected" — SSDM was independently found
irreproducible by an outside team; these approaches also still require a
speech-text alignment step, they don't eliminate the ASR-dependency problem
Phase 1's whole Track B investigation is about). Nothing found in this
adversarial pass overturns that: SSDM 2.0 is presented as improving on
SSDM's own limitations (representation-learning complexity, alignment
coverage), which is normal research progress, not evidence the
irreproducibility finding was wrong. More importantly, **the continuous/
phonetic representation and the discrete-event taxonomy are not actually
in conflict for this project's purposes** — every dataset this project
benchmarks against (SEP-28k, FluencyBank, KSoF) and Track A/B's entire
scoring machinery are built on discrete event labels; adopting a continuous
internal representation would still need to be *reported* against those
same discrete labels to remain benchmarkable (§5's own compatibility
discipline), so it would not remove the taxonomy question, only move where
it's answered. **Conclusion: not adopted, for the same
interpretability-and-benchmark-compatibility reasons as before — but this
is now stated as an actively-considered and rejected alternative, not an
unexamined default.**

### 9.3 Is staying rule-based/interpretable itself defensible, given the
field has moved toward deep learning?

**This is the most important tension this adversarial review surfaced, and
it deserves a direct answer rather than a reflexive defense.** The PMC
systematic review (§9.1) is unambiguous: "most studies have leaned toward
applying deep learning models... deep learning models consistently
outperform traditional machine learning" (only 3 of 14 reviewed studies
used classical ML exclusively). Taken alone, this could be read as evidence
this project's entire rule-based architecture is swimming against the
field's own findings.

**It is not, once the actual objective is stated precisely** — and the
project owner's own framing of this task ("accurate, robust, **explainable**,
and scientifically grounded") makes the tradeoff explicit rather than
leaving it to be inferred: the systematic review measures raw detection
performance, not interpretability, and does not weigh in on the
clinical-deployment tradeoffs at all. This project's own literature review
(§3.2) already found the same pattern from a different angle — rule-based
methods trail neural ones by a modest but real margin (~6% F1 in the one
detailed comparison found) in exchange for complete decision traceability,
10–15× lower compute cost, and no training-data dependency. **Given
explainability is a co-equal, explicitly stated project objective, not an
afterthought, staying interpretable-first is the correct choice for this
project specifically — not a universal claim that rule-based is better in
general.** This should be stated this plainly in the project's own
documentation rather than assumed silently, which is what this section
now does.

**This does not mean pretrained/learned components are off the table
entirely** — Phase 1's own `ROADMAP.md` already carries a deliberately
deferred "learned tier" item (a frozen WavLM/wav2vec2 classifier for
repetition-subtype discrimination) for exactly this reason: use a learned
signal as an *additional*, clearly-labeled feature or confidence input,
not a replacement for the interpretable decision layer. **This adversarial
review adds one specific, evidence-backed refinement to that existing
idea**: since blocks are now confirmed (by two independent peer-reviewed
sources, §9.1) to be the type rule-based methods handle worst, a
pretrained embedding used *specifically* as an auxiliary confidence signal
for block detection — not prolongation, not the other already-well-served
types — is the most targeted, evidence-justified place such a component
could eventually go. **Not promoted to a Phase 2 priority** (it still
requires a dataset that sub-types blocks to validate against, which §5
already found doesn't exist yet — the same blocker as the audible-block
detector itself), but recorded as a more specific version of the existing
deferred item rather than a vague "someday, learned models."

### 9.4 Other findings from this pass, smaller but worth recording

- **FluencyBank underrepresents prolongations** (per the PMC systematic
  review) — a specific, previously-unrecorded caveat. Relevant because
  prolongation is now this project's #1 detector-side priority (§6): if
  FluencyBank Timestamped is integrated per `ROADMAP.md` item 16, its
  prolongation numbers specifically should be read with this in mind, not
  treated as a clean cross-check.
- **A concrete technique for the synthetic-to-real generalization gap**
  (§3.4/`VALIDATION.md` §7.2 item 3): the systematic review cites
  Kourkounakis et al. — LibriStutter's *own original authors* — proposing
  domain-adaptation via adversarial networks specifically to bridge
  synthetic (LibriStutter) and real stuttered speech. This is a concrete,
  literature-sourced candidate technique for `ROADMAP.md` item 2 (the
  ASR-backend/real-speech validation work), not a new priority — added to
  that item's toolbox, not competing with it.
- **Independent reinforcement of an already-planned item**: the systematic
  review's own recommended research directions — balanced/targeted
  sampling to address class imbalance, and multiclass/multi-label learning
  for simultaneous stuttering-type detection — match, respectively,
  `ROADMAP.md`'s already-planned "expand the LibriStutter sample... target
  filler instances" item and this project's already-implemented multi-label
  event architecture (§3.3). Both are confirmations, not new information,
  but confirmations from a stronger source than was available when those
  items/decisions were first made.

### 9.5 Verdict: does the plan's ordering survive this review?

**Yes, reinforced rather than overturned — with one new nuance added
explicitly (§9.3) and citations strengthened (§9.1), not a reordering.**
Restating the ordering from §7 with this review's changes folded in:

1. Step 1 (taxonomy/documentation refinements) — unchanged, unaffected by
   anything found in this pass.
2. Step 2 (ASR-backend/speaker-diversity/real-speech validation) —
   unchanged priority; gained a concrete candidate technique (domain
   adaptation, §9.4) for the generalization half of this work.
3. Step 3 (prolongation core-detection redesign) — **strengthened**, not
   weakened: now resting on two peer-reviewed independent sources instead
   of one preprint, still explicitly gated on Step 2's outcome per §7's
   original reasoning.
4. Step 4 (word/sound-repetition type-classification fix) — unchanged.
5. **New, explicit, not previously stated**: staying rule-based/
   interpretable-first is a deliberate, evidence-and-mission-aligned
   choice given explainability is a co-equal stated objective, not an
   unexamined default — with a specific, targeted (block-only) future use
   for pretrained embeddings identified if that gap is ever revisited.

No direction found in this adversarial pass — a continuous/phonetic
taxonomy, a wholesale shift to deep learning, a different dataset priority
order, a different first implementation step — produced evidence strong
enough to change §7's ordering. The plan proceeds as originally
structured, with the above corrections/additions folded into how it's
justified and cited.

---

## 10. Per-type definition audit: literature vs. dataset vs. implementation
(2026-08-03, dated addendum, before the next implementation milestone)

§2–§4 already established *which types* belong in the taxonomy and *how*
detection strategies typically work per type. This section asks a sharper,
different question the project owner raised directly: for each type we
already implement, is our code's **operational trigger condition** — the
literal threshold/rule that decides "yes, flag this" — actually detecting
the phenomenon **as the clinical/scientific literature defines it**, or
only approximating **the dataset's own operational shortcut** for
labeling it? These are not always the same thing, and the gap between them
matters for what "improving accuracy" even means going forward.

Method: for each type, state (a) the clinical/scientific definition with
citation, (b) the exact dataset annotation/generation protocol, (c) our
code's exact operational trigger (verified by reading `detect.py`/
`acoustic.py` directly, not recalled from memory), then judge alignment.

### 10.1 `filler` / interjection

- **Literature**: an interjection/filler is a discourse-level "stall" — a
  word or sound (uh, um, like, you know) inserted while the speaker plans
  upcoming content, linguistically motivated rather than motoric (§2.1).
  Clinically classified as an Other Disfluency (OD), not stuttering-like.
- **Dataset**: SEP-28k's `Interjection` column is a **count out of 3
  annotators** for the whole 3-second clip (no word-level location) —
  annotators are trained to identify the *category*, not asked to apply a
  fixed word list (confirmed directly from the CSV schema comment in
  `profiling/evaluation/loaders.py`: `Show,EpId,ClipId,Start,Stop,Unsure,
  PoorAudioQuality,Prolongation,Block,SoundRep,WordRep,
  DifficultToUnderstand,Interjection,...`). LibriStutter injects
  interjections synthetically as one of its 5 simulated types (Kourkounakis
  et al., splicing onto GCSTT-timestamped LibriSpeech audio).
- **Our implementation**: `token.get("is_filler") or low in filler_words`
  — a **fixed 5-word list** (`uh`, `um`, `er`, `erm`, `like`) plus
  CrisperWhisper's own flag, matched **unconditionally** — "like" is
  always flagged whether it's a discourse filler ("it's, like, really far")
  or its literal meaning ("I like that"). SEP-28k's human annotators are
  presumably judging discourse function, not string-matching — this is a
  real, previously-unstated gap between what a *word list* can do and what
  the clinical category actually requires (distinguishing filler-"like"
  from content-"like" needs syntactic/semantic context, not just the
  token).
- **Verdict: our implementation approximates the dataset's operational
  category using a cruder mechanism than either the dataset's own
  annotators or the clinical definition require.** Not previously stated
  this precisely. **Not a Phase 2 priority to fix** — filler is already
  low-priority per §6 (Phase 1 found zero ground-truth filler instances in
  the current LibriStutter sample, so there's no way to even measure a fix
  yet), but the mechanism-level gap is real and now recorded rather than
  assumed away.

### 10.2 `sound_repetition`

- **Literature**: repetition of a sub-word fragment/phoneme (e.g. "b-b-
  ball"), classified as an SLD (motoric). The clinical severity instrument
  (SSI-4) scores stuttering partly by **counting iterations** — "b-b-b-
  ball" (3 iterations) is scored differently from "b-ball" (1 iteration) —
  iteration count is a first-class clinical measurement, not incidental.
- **Dataset**: SEP-28k's `SoundRep` is again a clip-level annotator count,
  no iteration count captured in the label itself. LibriStutter's real
  annotation format marks the disfluency with a single `"STUTTER"`
  placeholder row per event (confirmed directly, `VALIDATION.md` §8.2) —
  it cannot represent "how many times" a fragment repeated even if the
  underlying synthetic audio has multiple iterations spliced in; our
  `load_libristutter_csv` reconstruction inherits this same one-shot
  limitation.
- **Our implementation**: `prev_word.endswith("-") and low.startswith
  (prev_low)` — fires once, on the fragment-then-word transition. Per
  `ARCHITECTURE.md`'s own documented edge-case testing, a multi-iteration
  case like "str- str- street" produces multiple `stutter_marker` events
  (one per fragment) plus one `sound_repetition` event on the final
  transition — but **nowhere is an iteration count computed or stored**,
  on any event.
- **Verdict: matches the dataset's own (also iteration-blind) operational
  definition, but not the clinical literature's, which treats iteration
  count as clinically meaningful.** This is a genuine, previously-unstated
  simplification — one this project's benchmark datasets *also* make, so
  it is not a benchmarking-compatibility problem, but it does mean this
  project cannot currently produce a clinically-complete severity-style
  count even if a downstream use case wanted one. Recorded, not
  actioned — no dataset to validate an iteration-counting feature against,
  same reasoning pattern as §5's dataset-compatibility discipline
  elsewhere in this document.

### 10.3 `word_repetition`

- **Literature**: whole-word repeated, exactly or near-exactly; §2.1's
  monosyllabic/polysyllabic SLD/OD split already reviewed and (as of §7
  Step 1) implemented as a descriptive sub-tag.
- **Dataset**: SEP-28k's `WordRep` clip-level count; LibriStutter's
  `STUTTER`-row reconstruction (word/sound/prolongation types get a copy
  of the adjacent real word, §8.2) — neither dataset distinguishes
  monosyllabic from polysyllabic at the label level (already established,
  §5 point 1).
- **Our implementation**: exact back-to-back match, phonetic/edit-distance
  near-match, and filler-sandwiched match (3 call sites, `detect.py`) —
  now also tagged `syllable_count`/`likely_sld` (§7 Step 1, implemented).
- **Verdict: already the most closely-aligned type in this audit.** The
  literature-motivated refinement is implemented, additive, and doesn't
  break dataset compatibility (confirmed by the identical-baseline
  benchmark in the previous milestone). No further gap found this pass.

### 10.4 `phrase_repetition`

- **Literature**: an immediately-repeated multi-word phrase; clinically an
  OD (§2.1), not stuttering-like.
- **Dataset**: **no reviewed dataset labels this as a distinct category at
  all** (already established, §3.1/§4) — LibriStutter approximates it as a
  single-word marker (the true repeated-phrase length is not recoverable
  from its `STUTTER` row, `VALIDATION.md` §8.2), and SEP-28k/KSoF/
  FluencyBank have no equivalent column whatsoever.
- **Our implementation**: `_find_phrase_repetitions()` scans genuine
  multi-word windows (`phrase_repetition_min_words` to `_max_words`) — this
  is actually **closer to the literature's real definition than any
  available dataset can even represent**, since it operates on genuine
  multi-word spans, not a single-word proxy.
- **Verdict: an unusual case — our implementation is arguably more
  faithful to the scientific definition than any available dataset's own
  operational shortcut, which is precisely why this type cannot be
  meaningfully benchmarked (§5, §6 item 5) — there is no dataset whose
  *label* matches what our *detector* actually does closely enough to
  score it fairly.** This reframes the existing "not validated" status:
  it is not that our implementation is a crude approximation of a good
  dataset label — the datasets' own labels are the cruder approximation
  here.

### 10.5 `block`

- **Literature**: a stoppage of airflow/voicing from laryngeal or
  articulatory tension — critically, an **effort/struggle-based**
  definition, not purely acoustic. SSI-4 scores blocks partly via
  "physical concomitants" (facial grimaces, head/limb movements) — signs
  observable to a clinician but **not present in the audio signal at
  all**. Two acoustic sub-types exist in the literature (silent, audible/
  struggle — §2.2, already documented as a gap).
- **Dataset**: SEP-28k's annotators judge `Block` perceptually from audio
  alone (no video), presumably trained to recognize the *sound* of
  struggle/tension, not applying a fixed silence-duration rule — and
  notably, SEP-28k's schema has a **separate `NaturalPause` column**
  (confirmed in the CSV header, `loaders.py`), meaning SEP-28k's own
  annotators are explicitly trained to distinguish "this is a natural
  pause" from "this is a stuttering block" — a distinction our detector
  cannot make at all (see below).
- **Our implementation**: `gap >= block_gap_seconds` (a fixed/calibrated
  time threshold) **and** `gap_is_silent()` (RMS below a fixed threshold)
  — a pure silence-duration rule with no effort/tension signal and no way
  to distinguish a natural pause from a struggle. This was already found
  to be silent-only (§2.2); this pass adds a sharper point: **even the
  silent sub-type it does detect is not actually testing for "struggle,"
  only for "gap exceeded a duration threshold," which is a different,
  weaker claim than the clinical definition makes.** A speaker who simply
  pauses to think for longer than `block_gap_seconds` (default 0.55s,
  personalizable) is, by this detector's logic, indistinguishable from one
  genuinely struggling to initiate speech — exactly the ambiguity SEP-28k's
  own separate `NaturalPause` column exists to resolve, which this
  detector has no equivalent mechanism for.
- **Verdict: the largest literature-vs-implementation gap found in this
  audit.** Our `block` detector operationalizes a *necessary but not
  sufficient* acoustic correlate of the clinical definition (silence can
  indicate a block, but not every qualifying silence is one), and datasets
  built by trained annotators are explicitly designed to make a
  distinction (pause vs. block) that this project's implementation
  structurally cannot make. This is consistent with, and sharpens, the
  already-known fact that block is the hardest type in the literature
  (§9.1) — now with a specific mechanistic reason *why* a pure-silence
  rule is expected to underperform, not just an empirical observation that
  it does.

### 10.6 `prolongation`

- **Literature**: no single universal millisecond threshold exists across
  the field (confirmed directly — searched specifically for one and found
  none), but the dominant computational approach is **speaking-rate-
  relative**, not absolute: Esmaili et al. 2017's validated formula is
  `T_min = α / speaking_rate` with `α ≈ 1.2`. At a typical conversational
  rate of ~4–5 syllables/second, this gives `T_min ≈ 0.24–0.30s`; even at a
  slow, deliberate ~2.5 syllables/second, `T_min ≈ 0.48s`. SSI-4 separately
  measures duration as "the average length of the three longest stuttering
  events" in a sample — a relative, within-sample statistic, not a fixed
  external number either.
- **Dataset**: SEP-28k's `Prolongation` is a clip-level annotator count
  (perceptual judgment, no stated duration rule); LibriStutter's injected
  prolongations inherit its `STUTTER`-row's often-long duration by
  construction (already flagged as a reconstruction artifact,
  `VALIDATION.md` §8.3).
- **Our implementation, computed precisely from the code**:
  `prolong_threshold = max(prolongation_min_seconds, 90th-percentile-of-
  this-clip's-own-token-durations)`, with `prolongation_min_seconds`
  defaulting to **1.0 second** (raised from 0.65s specifically after
  real-mic false positives on naturally emphasized vowels, Part D tune,
  `PAPER_DECISION_LOG.md`). Typical adult conversational word durations
  (200–400ms) mean a clip's own 90th percentile rarely exceeds 1.0s in
  practice — **the flat 1.0s floor is the effective, binding threshold in
  most real clips**, not the percentile term.
- **Verdict: a real, now-quantified gap.** Our effective ~1.0s threshold is
  roughly **2–4× higher** than what rate-normalized literature methods
  (Esmaili 2017, independently verified peer-reviewed) use at normal-to-
  slow speaking rates. This is not a mistake — it was a deliberate,
  documented, evidence-based response to a real false-positive problem on
  this project's own real-mic testing (`PAPER_DECISION_LOG.md`, Part D) —
  but it means the current threshold is **calibrated for this project's
  specific precision/recall trade-off on limited real-mic data, not
  derived from (or validated against) the rate-normalized definition the
  literature treats as standard.** This directly sharpens §6/§7 Step 3's
  already-planned prolongation redesign: the redesign is not just "use
  Praat features as core criteria instead of confidence adjustments" (as
  §6 already states) — it should specifically test whether a rate-
  normalized threshold, evaluated properly via Track A/B rather than
  eyeballed against a handful of real-mic clips, changes the
  precision/recall balance found in `VALIDATION.md` §9's ablation. This is
  a refinement of Step 3's scope, not a new step — folded into §7 Step 3's
  existing gate on Step 2's outcome.

### 10.7 `stutter_marker`

- **Literature**: no clinical or computational-detection equivalent found
  anywhere in this project's research to date (already established,
  §3.1/§4) — it is an artifact of this pipeline's own ASR output (a
  trailing `-` or CrisperWhisper's `is_stutter` flag on a sub-word
  fragment), not a category any external definition describes.
- **Dataset**: none label it.
- **Our implementation**: `token.get("is_stutter") or word.endswith("-")`.
- **Verdict: not a "different definition" situation — there is no external
  definition to be different from.** This type is best understood as a
  transcript-level *signal* (the ASR flagged something as a fragment) that
  partially overlaps `sound_repetition`'s territory when a fragment is
  followed by its completed word, and stands alone (uncorroborated by any
  clinical category) otherwise. Consistent with its existing
  de-prioritization (§6 item 5) — this pass found no new reason to revisit
  that.

### 10.8 Overall verdict: should implementation move closer to the
scientific definition, and how, while staying dataset-compatible?

**Not a uniform answer — it genuinely differs by type, and stating it as
one summary would misrepresent what was found:**

- `word_repetition`, `phrase_repetition`: **already aligned or better
  than the datasets' own operational shortcuts.** No action.
- `filler`, `sound_repetition`, `stutter_marker`: **simplified relative to
  the full clinical definition, but in ways this project's own benchmark
  datasets *also* simplify** (word-list matching vs. discourse judgment;
  no iteration counting; no external definition at all, respectively) —
  moving closer to the clinical definition here would not be *validatable*
  against any current dataset, so it is correctly left as documented,
  known simplification rather than actioned now, consistent with §5's
  established compatibility discipline.
- `prolongation`: **a real, now-quantified gap between our empirically-
  tuned absolute threshold and the literature's rate-normalized standard**
  — already the top Phase 2 detector-side priority (§6/§7 Step 3); this
  audit sharpens exactly what "redesign" should test (rate-normalization
  specifically, evaluated via Track A/B, not eyeballed) rather than adding
  a new priority. **Addendum (2026-08-04): tested exactly as this audit
  recommended.** Rate-normalization, evaluated on its own merits via a
  pre-registered ablation, did not transfer well to this project's
  benchmark data (regressed both `Any` and prolongation-specific F1) —
  the literature's formula itself is not in question, but it does not
  transfer as-is to LibriStutter's short, reconstructed-timing clips.
  Praat-feature gating (this audit's other identified lever) did clear
  the pre-registered bar and is now the shipped default. See
  `VALIDATION.md` §9.5.1 for the full result.
- `block`: **the clearest case for eventually moving closer to the
  clinical definition** — a pure silence-duration rule is a structurally
  weaker claim than "struggle/tension," and this project's own benchmark
  datasets (SEP-28k's separate `NaturalPause` column) show human
  annotators already make a distinction this detector cannot. **Not
  actionable yet**, for the same dataset-compatibility reason established
  in §5: no available dataset sub-types blocks by silent vs. struggle, so
  a more clinically-faithful detector could not be benchmarked against
  anything today. This strengthens, rather than changes, the existing
  plan: item 11 in `ROADMAP.md` (verify UCLASS's schema) is now the
  single most important open question standing between "this is a
  documented limitation" and "this is buildable with real validation."

**General principle this audit establishes, worth stating explicitly for
future definitional questions**: this project's discipline is not "always
move closer to the scientific definition" — it is **"move closer to the
scientific definition exactly where doing so remains benchmarkable against
real data, and otherwise document the simplification honestly rather than
either ignoring it or acting on unvalidatable intuition."** Every
divergence found in this audit was already, independently, a divergence
the datasets themselves make (§10.1, §10.2, §10.7) — with the two
exceptions that matter (`prolongation`'s threshold, `block`'s effort-vs-
silence gap) already being this project's top two identified detector-side
priorities before this audit began, now with sharper, literature-grounded
specifics rather than general direction.
