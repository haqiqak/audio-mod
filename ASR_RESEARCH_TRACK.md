# ASR_RESEARCH_TRACK.md — Preserving Speech-Production Information: A Research Track

**Status: OPEN, 2026-08-05.** This is the charter document for a new,
deliberately-separate research track, developed on its own branch
(`asr-research`) so `main` stays stable and shippable throughout. It is
**not** implementation, and it does not commit this project to building a
custom ASR. It defines the problem, reviews what's already known, lays
out architectural directions without picking one, and sets the phased
research plan and decision criteria that will govern work on that branch.
Per this project's own standing discipline (`CLAUDE.md` rule 1; this
document *is* that pre-registration, at the level of a whole research
track rather than a single experiment) — no implementation code should be
written against this track until the plan below says a specific
experiment is next.

---

## 0. The checkpoint that opened this track

`VALIDATION.md` §14/§14.1 (2026-08-05) re-ran the shipped repetition-
classifier gate against real ASR output for the first time (`ROADMAP.md`
item 19). The classifier's own mechanism transferred safely — but its
real-world impact was negligible, because real CrisperWhisper output
produced almost no `word_repetition`/`sound_repetition` candidates to
gate in the first place (`sound_repetition`: **zero** candidates across
all 120 real-audio clips, either condition). A direct hand-check (not
just trusting the aggregate number) found why: the candidate check needs
a literal sub-word fragment token in the transcript, and real verbatim
ASR normalizes disfluent fragments into the clean full word — even at
positions it otherwise transcribes correctly. This was not a detector bug
and not a classifier-precision problem. It was the first direct, measured
evidence that the ASR stage itself may be discarding information this
project's downstream objective structurally depends on, before the
detector ever gets a chance to see it.

That is a different *kind* of finding than anything this project has
acted on before. Every prior fix (`sound_repetition`'s ordering bug, the
`prolongation` redesign, the repetition classifier itself) improved what
the detector does with the transcript it's given. This finding is about
whether the transcript itself is the right representation to hand the
detector at all, for certain disfluency types. That is why this is being
treated as the start of a research track, not another `ROADMAP.md` item.

---

## 1. The reframed core question

The question this track investigates is not "how do we improve the
detector." It is:

> **How do we preserve the speech-production information that
> conventional ASR intentionally removes?**

General-purpose ASR is built to recover a speaker's *intended* linguistic
content — that is the product it is optimized and evaluated for. This
project's objective is different in kind: to preserve and understand
*how* speech was produced, because the production characteristics
themselves — the repetitions, the blocks, the prolongations, the
fragments — are exactly what the downstream modules need. A transcript
can be an excellent ASR output by every standard metric while
simultaneously being a poor representation for disfluency analysis,
because the two objectives are not the same objective wearing different
clothes — they are different objectives, and general-purpose ASR was
never asked to serve this project's.

The project owner's own framing of this, stated in the conversation that
opened this track, is worth preserving verbatim as the plainest statement
of why this matters:

> *"That dust is now gold to us."* — Haqiq, 2026-08-05

What conventional ASR treats as noise to be cleaned away — the false
starts, the fragments, the repeats — is this project's primary signal.

---

## 2. Formal problem statement

**What this track is investigating**: whether the *representation*
produced by the ASR stage (not the stage's existence) is rich enough to
support this project's stated objective, and if not, what would make it
so — richer intermediate representations, decoding changes, fine-tuning,
multitask training, or a hybrid/acoustic-native path that reduces
reliance on ASR's decoded text for certain disfluency types.

**What this track is *not* reopening**: `PHASE_3_ARCHITECTURE_REVIEW.md`
already asked, from first principles, whether the two-stage
(ASR-then-detect) architecture should be replaced by an end-to-end
audio-to-disfluency model, and — evaluated against 2024-2026 literature —
concluded no viable alternative was decisively better without
infrastructure this project doesn't have. Nothing found this session
contradicts that conclusion, and this track does not re-litigate it
without new evidence, per standing rule 8. The two questions are
genuinely different: that review asked *"should there be a distinct ASR
stage at all;"* this track asks *"given that there is one, is what it
hands the detector rich enough, and if not, what has to change about
it."* The second question can have real, evidence-supported architectural
consequences (this project's first fine-tuned or heavily-adapted model
component, most obviously) without touching the first question's
conclusion.

**Why this checkpoint changes the project's direction, precisely stated**:
prior to item 19, the working assumption — reasonable, evidence-based at
the time — was that the two-stage architecture's *detector* side was
where remaining accuracy was to be found (Phase 2's fixes, Phase 3's
classifier). Item 19 is the first direct evidence that for at least one
disfluency type, the ceiling is not in the detector at all — it is set
upstream, by what the ASR stage's output representation is capable of
expressing. No amount of detector-side work can recover information the
representation it's given never contained.

---

## 3. What information this project actually needs — restated here so it
stays in view before any "how"

Per the project owner's explicit instruction: this document keeps the
target in hindsight throughout, not just at the start. This project's
taxonomy (`README.md`, `PHASE_2_RESEARCH_PLAN.md`'s literature-grounded
review) defines seven disfluency types this project must detect,
classify, and localize:

| Type | What must survive in the representation to detect it |
|---|---|
| `filler` | A recognizable interjection ("um," "uh") as a distinct unit, not smoothed into surrounding fluent speech. |
| `sound_repetition` | A **sub-word fragment** (e.g. a repeated initial sound) as evidence distinct from the word it precedes — exactly what item 19 found real ASR does not preserve as a token. |
| `word_repetition` | A whole word repeated back-to-back — survives more often than fragments, but item 19 measured real ASR still produces ~7x fewer recoverable candidates per clip than a ground-truth transcript would. |
| `phrase_repetition` | A multi-word span repeated — same class of risk as `word_repetition`, unconfirmed at this granularity yet. |
| `block` | A silence with specific acoustic context (flanked by voiced speech, not just any pause) — already handled acoustic-natively (`profiling/acoustic.py`), not purely token-dependent. |
| `prolongation` | An abnormally sustained sound — already handled acoustic-natively with Praat-gating (`VALIDATION.md` §9.5.1), the project's own precedent for "stop depending on the transcript, use the waveform directly." |
| `stutter_marker` | An ASR-level cut-off fragment marker — inherently dependent on what the ASR stage is willing to emit at all; no dataset validates this type today (`ROADMAP.md` "Phase 2 is closed," item 11/12). |

Per `PHASE_3_ARCHITECTURE_REVIEW.md` §3.6 (external literature, cited
there before this track existed): ASR damages `word_repetition`,
`sound_repetition`, and `filler` most severely (35-47% WER impact) and
`block` least (~20%) — consistent with the pattern above, where `block`
and `prolongation` are exactly the two types this project already made
acoustic-native and least token-dependent. **This is a meaningful,
already-available clue**: the two types least damaged by ASR normalization
are also the two types this project already stopped depending on ASR
text for. That is not a coincidence to ignore; see §6(f) and §8.

---

## 4. How modern general-purpose ASR is designed, and why normalization
is generally considered desirable

This is not a design flaw to be indignant about — it is a deliberate,
well-motivated engineering choice for the objective mainstream ASR
serves, and understanding *why* is necessary before proposing to deviate
from it.

- **Training references are themselves normalized.** Standard ASR
  training/benchmark corpora (LibriSpeech, CommonVoice, and similar) are
  transcribed by humans who routinely clean up disfluencies as a matter
  of transcription convention — the reference text a model is trained
  toward already has much of the "dust" removed before training even
  starts. A model trained to match those references learns to produce
  clean output, not because it was told to discard disfluencies, but
  because disfluencies were rarely represented as worth reproducing in
  what it was shown.
- **Word Error Rate, the field's dominant metric, does not penalize
  normalization when references are already clean** — a transcript that
  smooths a stutter into the fluent word it was aiming for can score
  *better* on WER than one that faithfully reproduces the disfluency, if
  the reference itself was cleaned. The metric and the training data
  reinforce the same bias together.
- **Decoding itself smooths toward fluency.** Beam search combined with
  an implicit or explicit language-model prior favors higher-probability,
  well-formed sequences — a disfluent, low-probability fragment sequence
  is exactly what such decoding is built to suppress in favor of the
  fluent alternative it's "supposed to" represent.
- **The dominant use cases are dictation, captioning, and voice
  assistants** — products where the intended message, not the manner of
  its production, is the deliverable. Normalization is a feature for
  those products, not an oversight.

Recent field-level evidence confirms this bias is real and measurable,
not just a plausible-sounding story: **"Lost in Transcription"**
(arXiv:2405.06150) evaluated six leading ASR systems and found "a
consistent and statistically significant accuracy bias across all ASRs
against disfluent speech," using a LibriSpeech-based synthetic-stuttering
methodology structurally similar to this project's own LibriStutter
usage. **FluencyBank Timestamped**'s own framing (Bishop et al.,
*Journal of Speech, Language, and Hearing Research*, 2024) draws the
same distinction this document does at the field level: "clinicians
require verbatim transcripts where disfluencies are transcribed... [while]
dictation services benefit from transcribing only the intended... speech"
— two legitimately different objectives, and general-purpose ASR is built
for the second one.

---

## 5. Existing research toward disfluency- and production-preserving ASR

A genuine literature pass (not assumed, checked directly this session),
organized by the architectural direction each line of work speaks to.
None of these papers solve this project's exact problem (a CPU-only,
no-GPU-training-infra-until-recently, seven-type taxonomy, LibriStutter-
plus-real-audio validated system) — but real, active work exists in
every direction this section explores, and one result independently
corroborates a finding this project already made on its own.

**Verbatim-transcription-focused ASR (already adopted, worth
re-examining precisely).** CrisperWhisper itself (Wagner, Thallinger,
Zusag; arXiv:2408.16589; INTERSPEECH 2024) achieves verbatim
transcription primarily through a **retokenized input layer and a custom
attention-loss timestamp-alignment mechanism** (Dynamic Time Warping over
cross-attention scores), trained to reduce hallucination and improve
timestamp precision around disfluencies and pauses. This is real,
validated, and is why this project chose it (`ARCHITECTURE.md` §3). But
its innovation is concentrated in *tokenization and alignment*, not
necessarily in preserving genuine sub-word acoustic fragments as distinct
emitted tokens — which is exactly consistent with, and now a candidate
explanation for, item 19's finding that `sound_repetition` fragments
still don't survive to the transcript. **Open, sharpened question this
track should answer early**: does CrisperWhisper's verbatim claim cover
word/phrase-level disfluencies well (repeats, fillers) while still
normalizing sub-word fragments specifically — i.e. is the gap item 19
found a known, bounded limitation of even the best current
verbatim-focused ASR, not a sign CrisperWhisper was the wrong choice?

**Fine-tuning / continual adaptation of an existing ASR.** "Learning to
Hear Hesitation: Continual Learning for Disfluency-Aware ASR"
(arXiv:2606.14391) investigates continual-learning methods to integrate
disfluency tokens into an ASR's output over successive training rounds,
reporting improved disfluency-marker retention without discarding general
ASR performance — a directly relevant existence proof that adapting an
existing model (rather than training one from scratch) is a viable
direction, at a cost this track has not yet priced for this project's
own infrastructure.

**Multitask / joint ASR-and-disfluency-detection training.** Several
independent lines of work train transcription and disfluency detection
as coupled objectives rather than a strict two-stage pipeline: "Streaming
Joint Speech Recognition and Disfluency Detection" (arXiv:2211.08726,
two output layers with a token-dependency bridge), "Augmenting Automatic
Speech Recognition Models with Disfluency Detection" (arXiv:2409.10177),
and the earlier "Multi-Task Self-Supervised Learning for Disfluency
Detection" (arXiv:1908.05378). `ROADMAP.md`'s "Longer-term" section
already cites a further joint-training result reporting large relative
CER and detection-F1 gains from this style of training on Mandarin data —
not re-verified independently this session, so treated as an existing,
not newly-confirmed, citation.

**Bypassing decoded text for detection entirely.** "Automatic Disfluency
Detection from Untranscribed Speech" (arXiv:2311.00867) investigates
detecting disfluencies directly from audio without relying on a
transcript as the intermediate representation at all — the same
direction this project already has real, shipped precedent for
(`profiling/acoustic.py`'s `block`/`prolongation` detection, built
specifically because token-based detection couldn't reach them).

**Probing whether pretrained speech representations already carry the
signal, independent of decoded text.** This is the most directly
corroborating find of this literature pass. "Whisper in Focus: Enhancing
Stuttered Speech Classification with Encoder Layer Optimization"
(arXiv:2311.05203) classifies stuttering **using Whisper's encoder
layers directly**, finding that **deeper encoder layers carry more
disfluency-relevant information than shallow ones**, reaching an average
F1 of 0.81 — independent, external evidence that a Whisper-family
encoder's internal representation carries recoverable disfluency signal
beyond what the decoded transcript expresses. **This is the same
conclusion this project's own Stage 1 experiment reached independently**
(`VALIDATION.md` §11: CrisperWhisper's last-layer encoder states carry a
large, stable, TP-vs-FP-discriminating signal, Cohen's d > 1.0) —
meaning this track is not starting from zero; it is extending a direction
this project already has one successful, literature-corroborated result
in. Separately, "Self-supervised Speech Models for Word-Level Stuttered
Speech Detection" (arXiv:2409.10704, already cited in `PHASE_3_
ARCHITECTURE_REVIEW.md` as the Stage 1b/WavLM candidate) reports the best
published word-level stuttering F1 (0.554) found in that review using a
frozen WavLM-Large representation, not a fine-tuned ASR.

**Hybrid / fused representations.** "StutterFuse: Mitigating Modality
Collapse in Stuttering Detection with Jaccard-Weighted Metric Learning
and Gated Fusion" (arXiv:2512.13632) fuses multiple representations
(implicitly, text and acoustic) with a gating mechanism designed
specifically to prevent one modality from dominating and washing out the
other's signal — directly relevant to how this project's own weighted
token/acoustic fusion (`ARCHITECTURE.md` §4) already works, and a
possible source of design ideas for making that fusion more principled.

**One title-relevant paper not independently deep-verified this
session, flagged honestly rather than cited with unearned confidence**:
"On the Difficulty of Token-Level Modeling of Dysfluency and Fluency
Shaping Artifacts" (arXiv:2512.02027) — its title matches this track's
problem almost exactly. Worth a dedicated, careful read as an early Stage
A task (§8), not cited further here until that read happens.

**What this literature pass supports, stated as a conclusion, not an
assumption**: no existing work directly answers "how should *this*
project's ASR stage be adapted for *this* seven-type taxonomy under *these*
compute constraints" — that gap is real and is what this track exists to
close. But every architectural direction this track might pursue has real
prior art to learn from, not a blank page, and one of those directions
(richer intermediate representations) already has both external
literature support and this project's own successful pilot result behind
it before this track has run a single new experiment.

---

## 6. Architectural directions — explored, none committed to

Per the project owner's explicit instruction, these are laid out as
options to investigate, not a predetermined plan. Each is mapped to what
this project already has, to keep "how much would this actually cost
here" honest rather than abstract.

**(a) Fine-tune or continually adapt CrisperWhisper (or another ASR)
toward this project's taxonomy.** Literature precedent: arXiv:2606.14391.
Cost: real — needs a training pipeline (now exists, per the
reassessment's own finding), GPU resources this project has not needed
before, and a paired dataset (transcript-with-disfluencies + audio, at
volume) this project does not yet have. Highest potential ceiling,
highest cost, most speculative until cheaper directions are exhausted.

**(b) Preserve and use richer intermediate representations without
fine-tuning anything.** Literature precedent: arXiv:2311.05203,
arXiv:2409.10704. Cost: **lowest of any option** — this project already
built and validated the infrastructure (`profiling/encoder_embedding.py`,
Stage 1's methodology). The natural, cheapest next experiment (§8, Stage
B) is to point that exact methodology at the specific gap item 19 found,
rather than assume it does or doesn't help.

**(c) Modify decoding objectives/parameters without touching training at
all.** Not yet explored by this project. CrisperWhisper's own verbatim
behavior already comes partly from decoding-side choices (timestamp
alignment via cross-attention DTW) — worth checking whether existing,
already-exposed decoding parameters (beam width, timestamp granularity,
suppression of blank/low-confidence tokens) affect whether sub-word
fragments ever surface, before assuming a decoding change requires
retraining anything.

**(d) Multitask learning — train transcription and disfluency detection
as coupled objectives.** Literature precedent: arXiv:2211.08726,
arXiv:2409.10177, arXiv:1908.05378, and the Mandarin joint-training
result already in `ROADMAP.md`'s Longer-term section. Cost: comparable to
(a) — needs paired training data and infrastructure this project would
need to build or acquire.

**(e) Hybrid ASR-text + acoustic-representation fusion, made more
principled.** Literature precedent: arXiv:2512.13632. This project
already does a version of this (`ARCHITECTURE.md` §4's weighted fusion,
the repetition classifier's own encoder-distance corroboration) — the
open question is whether the *fusion mechanism itself* (not just adding
more signals to it) deserves the same evidence-gated redesign the
corroboration-mechanism comparison gave the repetition classifier
(`VALIDATION.md` §12).

**(f) Reduce reliance on ASR-decoded text for candidate generation on the
types most damaged by it — extend the acoustic-native precedent further.**
Literature precedent: arXiv:2311.00867. This project's own `block` and
`prolongation` detectors are already a working example of exactly this
pattern (§3's table) and are, not coincidentally, the two types
literature says ASR damages least. This is the most direct, lowest-risk
extrapolation of something already proven to work in this codebase — the
open question is whether `sound_repetition` (and possibly `word_
repetition`) can follow the same path, rather than needing a text-based
candidate at all.

**Not yet possible to rank these** — that is exactly what §8's phased
plan is for.

---

## 7. Research questions this track exists to answer, in the order they
should be answered

- **RQ1** — Systematically, not just for `sound_repetition`: for each
  disfluency type in this project's taxonomy, what fraction of real
  instances survive real ASR transcription into a form the current
  token-based candidate generator can recognize at all? (Extends item
  19's single finding into a full type-by-type information-loss profile.)
- **RQ2** — Where the decoded text loses the signal, does the ASR
  encoder's internal representation still carry it, the way arXiv:2311.05203
  and this project's own Stage 1 found for the signals each already
  tested? (Direct, cheap re-application of proven methodology to the new
  gap.)
- **RQ3** — Is this information loss a property of ASR generally, or
  specific to CrisperWhisper's own training/decoding choices? (Ties
  directly into `ROADMAP.md` item 10's still-open second-ASR-backend
  question — now newly relevant to *this* question too, not only the
  general recall-gap question it was originally scoped for.)
- **RQ4** — Does existing literature (starting with the
  not-yet-deep-read arXiv:2512.02027) already contain a concrete,
  adaptable answer for this project's specific constraints, once actually
  read carefully rather than judged by title?
- **RQ5** — If richer representations alone (direction b/f) recover
  enough signal, is fine-tuning or multitask training (direction a/d)
  even necessary, or does this project's real bottleneck resolve at a
  fraction of that cost?

---

## 8. Phased research plan, evidence-gated at every stage

Mirrors the staged, escalation-gated structure `ROADMAP.md` item 17 used
successfully (a cheap Stage 1 test with an explicit trigger for a more
expensive Stage 1b, never reached because Stage 1 succeeded) — the same
discipline applied at the scope of a whole research track instead of one
decision.

**Stage A — Systematic information-loss audit (no new model, no new
audio, uses data already in hand).** Answers RQ1. Hand-trace a larger
sample of this project's own false negatives from the 120-clip Track B
run (42 `sound_repetition`, 41 `word_repetition`, plus `phrase_repetition`
and `filler` for completeness) using the same method that found item
19's mechanism — for each miss, classify it as: (i) genuinely lost — no
trace survives anywhere in the ASR output; (ii) present but mis-typed —
surfaces as a different event type (the one `block`-instead-of-`sound_
repetition` case already found is exactly this); (iii) present in text
but missed by the current candidate-matching logic — a detector bug, not
an ASR problem. This is the direct continuation of `ROADMAP.md` item 20.
**Exit criterion**: a type-by-type loss profile with enough hand-checked
cases per type to trust the categorization (rule 3 — small samples get
an explicit "too small to trust" caveat, not a confident headline).

**Stage A: done, 2026-08-05.** Systematically categorized all 186
disfluent ground-truth positions in the 120-clip Track B sample (not a
hand-picked few — every `sound_repetition`, `word_repetition`,
`phrase_repetition`, and `prolongation` instance), using the existing
`--verbose` diagnostic output, cross-checked line-by-line against
`track_b.py`'s own alignment/scoring code to reconcile exactly with the
official scored counts (a categorization bug was caught this way before
being trusted — see the alternatives-considered note in `PAPER_DECISION_
LOG.md`). Four categories, applied per instance: **(1)** ASR transcribed
the position correctly, but the detector generated no candidate at all
("normalized away"); **(2)** ASR transcribed the position correctly, but
the detector caught it as a *different* type ("mis-routed"); **(3)** ASR
made a genuine transcription error at that position (substitution or
deletion), no candidate generated; **(4)** genuine ASR error, and
something was predicted at the misaligned word (not credited as a match
by the scorer, correctly).

| Type (n) | (1) Normalized away | (2) Mis-routed | (3) Genuine ASR error | (4) ASR error + coincidental type |
|---|---|---|---|---|
| `sound_repetition` (42) | 19 (45.2%) | 4 (9.5%) | 16 (38.1%) | 3 (7.1%) |
| `word_repetition` (42, 1 TP) | 17 (40.5%) | 5 (11.9%) | 11 (26.2%) | 8 (19.0%) |
| `phrase_repetition` (40, 3 TP) | 20 (50.0%) | 0 | 9 (22.5%) | 8 (20.0%) |
| `prolongation` (62, 0 TP)* | 4 (6.5%) | 0 | 50 (80.6%) | 8 (12.9%) |

\* `prolongation` is included for completeness only — it is already
acoustic-native (Praat-gated, §3's table), so a *text*-alignment-based
categorization like this one is the wrong lens for it; its near-total
"(3) genuine ASR error" share mostly reflects that mismatch, not a
comparable finding to the other three types. Not investigated further
here.

**Findings, in order of how much they change this track's picture:**

1. **For `sound_repetition` and `word_repetition`, roughly half of all
   losses (54.7% and 52.4% respectively) happen even when ASR
   transcribed the position "correctly."** This confirms, at full sample
   scale rather than a handful of anecdotal cases, that item 19's
   original finding generalizes: normalization loss is not a rare edge
   case for these two types, it is roughly as common as ordinary ASR
   transcription error (categories 3+4: 45.2% for `sound_repetition`,
   45.2% for `word_repetition` — coincidentally identical totals, not
   the same cases).
2. **`sound_repetition`'s and `word_repetition`'s "correct-but-lost"
   mechanisms are different, not the same story told twice.**
   `sound_repetition` loses the literal fragment token (§0's original
   finding: "considered-" -> "considered," nothing left to detect).
   `word_repetition` loses the *pair*: a direct, targeted follow-up check
   (not just re-reading the same diagnostic lines) traced every `align=
   correct` `word_repetition` position back through the actual cached ASR
   token sequence and found **22 of 23 such cases (95.7%) have the
   *other* half of the repeated pair deleted, substituted, or displaced**
   — e.g. ground truth "will will be" survives in `hyp_tokens` as
   "...soon. That will be..." (the first "will" gone entirely, the
   second transcribed correctly but now adjacent to nothing that matches
   it). This is consistent with, and a more specific instance of, the
   same fluency-normalizing behavior §4 describes generally: a decoder
   biased toward well-formed output has a direct incentive to treat an
   immediate word repeat as one intended word, not two, and this data
   shows it usually acts on that incentive by dropping the first
   occurrence rather than merging or garbling it. **One exception, flagged
   as an open, separate finding**: a single case (clip
   `2092-145706-0025`) has the full repeated pair intact and adjacent in
   the hyp sequence (`['wolf', 'wolf', 'wolf,']` — a genuine triple
   repeat) yet still wasn't caught — this is a detector-logic question
   (a candidate-matching edge case on runs of 3+ identical words), not an
   ASR-representation question, and is out of this track's scope; noted
   for `ROADMAP.md` separately rather than investigated further here.
3. **The "mis-routed" category (2) is real but modest** (9.5%/11.9% for
   `sound_repetition`/`word_repetition`) — not the dominant recovery
   opportunity a single hand-picked example might have suggested, but
   real enough to be worth Stage C's attention once Stage B is done: a
   `block`, `filler`, or `phrase_repetition` label sometimes already
   fires at exactly the position a `sound_repetition`/`word_repetition`
   should have, meaning the acoustic-native detectors are already, by
   accident, catching some of this signal under the wrong name.
4. **~45% of losses for both types are ordinary ASR transcription
   error** (categories 3+4), unrelated to fragment/pair-normalization —
   a different, more general problem (this project's already-documented
   ASR-fidelity gap, Phase 1) that this track's representation-focused
   questions (RQ2 onward) are not expected to fix, and should not be
   conflated with the normalization-specific mechanism above when this
   track reports progress later.

**Small-sample honesty**: category-level percentages above are stable
enough to trust the *ranking* (normalization-loss and plain-ASR-error are
both large, roughly comparable contributors; mis-routing is real but
smaller) but individual cell counts (e.g. `sound_repetition`'s 3-count
category 4) are still small in absolute terms — treat precise
percentages as directional, not as fixed rates to design a fix against
without re-checking at a larger scale later.

**What this resolves for the track's own plan**: RQ1 is answered for
`sound_repetition`/`word_repetition` — loss is broad (roughly half of all
misses), not isolated to a couple of anecdotal cases, and it has at least
two distinct mechanisms (fragment loss, pair-breaking) that a single fix
is unlikely to address at once. Stage B is next: does CrisperWhisper's
encoder still carry a detectable signature at the ~50% of positions
category (1) identifies as "text says nothing, but ASR heard the position
fine" — directly testable with Stage 1's existing methodology, no new
data collection needed.

**Stage B — Representation-level probe (no training, reuses Stage 1's
exact methodology).** Answers RQ2. For the types Stage A finds are
genuinely lost from decoded text, re-run the Stage 1 encoder-distance
methodology (`VALIDATION.md` §11, `profiling/encoder_embedding.py`)
specifically targeted at those positions: does CrisperWhisper's encoder
still show a detectable signature at a real-ASR position where the
decoded text shows nothing? **Decision gate**: if yes (matching
arXiv:2311.05203's finding) — this project's cheapest, already-built
direction (b/f) is the priority, and Stage C is next. If no — richer
representations alone are insufficient for this gap, and the track moves
toward evaluating (a)/(d)'s higher cost directly, skipping Stage C.

### Stage B — pre-registered protocol (2026-08-05, written before running)

**This is a hypothesis test, not a validation exercise for a foregone
conclusion.** The question is neutral: does CrisperWhisper's encoder
retain discriminative information at positions where transcript-level
evidence has been normalized away, or doesn't it. A positive, negative,
or inconclusive result are all acceptable, correctly-reported outcomes —
none is being aimed for in advance.

**Target population** (the cases under test): every Stage A category-1
position — `sound_repetition`/`word_repetition` ground-truth instances
where real ASR aligned the position "correct" (transcribed accurately)
yet the token-based candidate generator produced no candidate at all.
19 `sound_repetition` + 17 `word_repetition` = **36 target positions**,
re-identified directly from the underlying alignment data (not the
printed diagnostic text) for this stage, spanning **38 distinct clips**
that need a real encoder pass (scoped and counted before running, not
estimated).

- For `sound_repetition`, the target span is the real ASR hyp-token that
  absorbed the fragment (e.g. the "considered" token that stands in for
  ground truth's "considered-"/"considered" pair) — testing whether that
  token's acoustic duration/representation still differs from an
  ordinary occurrence of the same word.
- For `word_repetition`, the target span is the *second* (correctly
  transcribed) occurrence's hyp-token — testing whether its
  representation carries a residual trace of the deleted first
  occurrence, a more indirect and more speculative test than the
  `sound_repetition` case, flagged as such rather than treated with equal
  confidence.

**Metric, reusing Stage 1's exact primitives unmodified**
(`profiling/encoder_embedding.py`'s `extract_last_layer_states`,
`pool_span`, `cosine_distance` — the same functions, not reimplemented):
cosine distance from each target position's mean-pooled last-layer
encoder embedding to a **per-clip fluent centroid**, computed the same
way Stage 1 defined it — the mean pooled embedding over every ref
position in that clip that is *not* ground-truth-disfluent and aligns
"correct" (so a trustworthy real-ASR hyp-token span exists for it).
Unlike Stage 1 (which ran entirely on Track A's ground-truth token
timestamps), every span and every centroid here is built from **real ASR
hyp-token boundaries on real audio** — the same underlying waveform, but
the actual timestamps and word boundaries a real user's transcription
would produce, which is the entire point of testing this on Track B
rather than re-reading Stage 1's original Track A result.

**Control group, added specifically to avoid a circularity risk Stage 1
didn't have to handle**: Stage 1 only ever compared disfluent (TP/FP)
spans against a centroid built purely from clean spans, so a clean span
never had its own distance-to-centroid measured. Stage B needs that
comparison distribution to exist, so: for each clip with at least one
target position, a **matched-size sample of held-out fluent positions**
from the same clip has its distance to a **leave-one-out centroid**
computed (recomputed excluding that one point, so a fluent point is never
compared against a centroid partly built from itself). This control
group's distances are what "genuinely fluent, real ASR, same clips"
looks like — the target group is compared against this, not against an
assumed baseline of zero.

**Success criteria, fixed in advance**:
- **Positive result**: target-group distances are measurably larger than
  control-group distances, Cohen's d >= 0.5 (the same bar Stage 1 used,
  `VALIDATION.md` §11.4) — reported separately for `sound_repetition`
  and `word_repetition` (different mechanisms, per Stage A; not pooled
  into one number that could hide one type working and the other not).
  A positive result here means direction (b/f) — richer representations,
  no fine-tuning — is this track's next priority (Stage C).
- **Negative result**: no measurable difference, or `|d| < 0.2` — encoder
  representations alone do not recover this signal at these positions;
  the track moves toward pricing out (a)/(d) directly (Stage D), skipping
  Stage C for this specific gap. This is reported as a real, useful
  finding, not a setback — it would mean the normalization happens
  upstream of where a frozen encoder can see it, which is itself
  something no prior work reviewed in §5 has confirmed either way.
- **Inconclusive**: anywhere between, or effect size not stable/trustable
  given n — reported as exactly that, not rounded toward whichever answer
  seems more interesting. Given n=19/17 target positions per type, this
  is a real possibility to expect going in, not a failure of the
  experiment design.

**Named limitations, stated before results are known**:
- Same duration/word-identity confound Stage 1 named and never fully
  resolved (`VALIDATION.md` §11.6) — a token that absorbed a fragment is
  very likely *longer* than an ordinary token of the same word, and
  encoder representations are known to be duration-sensitive in ways not
  cleanly separable from a "disfluency signature" at this sample size.
  Reported as a real limitation of the conclusion, not solved by this
  design.
- `word_repetition`'s target is the more indirect of the two tests (the
  representation of the *surviving* word standing in for evidence about
  the *deleted* one) — a negative result there is less informative than
  a negative result for `sound_repetition`, and is analyzed and reported
  separately for exactly this reason.
- Single dataset (LibriStutter), single ASR backend (CrisperWhisper),
  same generalization caveat every result in this project carries until
  `ROADMAP.md` item 10 is addressed.
- n=36 target positions total, split across two types — explicitly a
  small-sample regime; this stage is designed to produce a *direction*
  with an honestly-stated confidence, not a number treated as final.

**Cost, scoped before running**: 38 distinct clips need a real encoder
pass at this project's own previously-measured ~30-90s/clip
(`ARCHITECTURE.md` §3) — roughly 20-55 minutes, bounded and known before
starting, not open-ended.

### Stage B — Results (2026-08-05): a mixed, honestly-reported outcome —
positive for `sound_repetition`, inconclusive for `word_repetition`

**A bug caught before trusting the target-identification pass, exactly
the kind of self-check this project applies to its own new tooling, not
just production code**: the first implementation identified target
positions using `audio_bytes=None` (to keep the classifier gate from
running without needing to touch `config.yaml`). That also silently
disabled every *acoustic-native* detector (`block`, `prolongation`),
which meant a real Stage-A category-2 case ("mis-routed to `block`")
could be miscounted as category-1 ("no candidate at all") purely because
the acoustic detector that would have fired was turned off along with
the classifier. Caught by reconciling the identification pass's counts
against Stage A's already-trusted numbers before running the (expensive)
encoder step: the first pass found 19 `sound_repetition` / 18 `word_
repetition` targets against Stage A's known 19/17 — a 1-count mismatch,
investigated rather than shrugged off. Fixed by passing real
`audio_bytes` and forcing the classifier gate off via an explicit
`config` override (`profiling/detect.py`'s own supported per-call
override, never touching `config.yaml`) instead of removing audio
entirely — re-verified it reproduces Stage A's exact 19/17 split across
31 distinct clips before any encoder time was spent.

**Real cost, as scoped**: 31 clips (fewer than the pre-registration's
conservative 38-clip estimate, which included category-2 cases dropped
once identification was corrected), 1026s (~17 min) total encoder time,
~33s/clip — consistent with this project's previously-measured range.

**Results, per the pre-registered metric (cosine distance to each clip's
own leave-one-out-controlled fluent centroid)**:

| Type | n (target) | target mean distance | n (control) | control mean distance | Cohen's d |
|---|---|---|---|---|---|
| `sound_repetition` | 19 | 0.545 | 966 | 0.466 | **0.894** |
| `word_repetition` | 17 | 0.504 | 966 | 0.466 | 0.428 |

(Control group is the same pooled 966-position fluent baseline for both
rows — every clean, correctly-aligned position across the 31 clips, each
scored against a centroid that excludes itself.)

**Against the pre-registered success criteria, read exactly as
written**:
- **`sound_repetition`: positive.** Cohen's d = 0.894 clears the
  pre-registered d >= 0.5 bar clearly, and is close in magnitude to
  Stage 1's own original TP-vs-FP effect (d ≈ 1.05, `VALIDATION.md`
  §11.6) despite testing a completely different population (real-ASR
  "normalized away" positions vs. Track A candidates) and a different
  comparison (fluent controls vs. FP events). CrisperWhisper's encoder
  retains a measurable trace of the sound-repetition fragment at the
  position where the *decoded text* shows nothing at all.
- **`word_repetition`: inconclusive, not negative.** Cohen's d = 0.428
  falls between the pre-registered thresholds (below the 0.5 "positive"
  bar, above the 0.2 "negative" bar) — exactly the outcome the
  pre-registration flagged as plausible in advance, precisely because
  this test is the more indirect one (probing the *surviving* word's
  representation for a trace of the *deleted* partner, not a direct
  fragment-in-place test). The direction is still positive (target mean
  > control mean), so this is not evidence against a signal existing —
  it is evidence this specific, indirect test doesn't establish one with
  confidence at n=17. Per the pre-registration's own instruction, this is
  reported as exactly that: inconclusive, not rounded toward either a
  confirmation or a refutation.

**Limitations, both the ones named in advance and one found while
interpreting the result**:
- The duration/word-identity confound named before running remains
  unresolved: a token that absorbed a `sound_repetition` fragment is very
  likely longer than an ordinary token of the same word, and this design
  cannot yet separate "the encoder detected a disfluency" from "the
  encoder detected an unusually long token" — both produce the same
  measured effect here. This does not make the `sound_repetition` result
  uninterpretable, but it does mean "the encoder carries recoverable
  *disfluency* signal" is a slightly stronger claim than "the encoder
  carries a recoverable acoustic-duration anomaly that correlates with
  where a disfluency happened" — the data collected so far cannot fully
  distinguish these, and future work (Stage C or a dedicated follow-up)
  should test duration-matched controls before treating this as settled.
- **A statistical caveat not in the original pre-registration, worth
  naming honestly rather than glossing over now that real numbers exist**:
  the 966-position control group pools multiple positions from the same
  31 clips, which are not fully independent observations (shared
  recording conditions, speaker, and centroid quality per clip) — the
  standard Cohen's d/pooled-variance calculation used here treats every
  point as independent, which likely overstates the effective sample size
  somewhat. This doesn't change the direction of either result, but a
  future, more rigorous pass (e.g. a clip-level bootstrap or per-clip
  aggregation before computing the effect size) would be a stronger
  version of this same test before it carries real architectural weight.
- Single dataset, single ASR backend — the same standing caveat every
  result in this document carries (§8, `ROADMAP.md` item 10).

**What this resolves for the track's decision gate**: the pre-registered
gate was written as a binary ("if yes... Stage C is next; if no...
skip Stage C") because a clean split hadn't been considered fully in
advance for two types disagreeing. The honest reading of a *mixed*
result: proceed to **Stage C scoped specifically to `sound_repetition`**,
where the signal is real and clears the bar — do not yet extend Stage C
to `word_repetition` on the strength of this test; that type's question
stays open, either for a larger sample, a less indirect test design, or
folded into whatever Stage C or D eventually addresses `word_repetition`
with. This is a genuine, useful, non-obvious finding either way: **not
every type this track cares about behaves the same way**, which is
itself evidence against treating "does the encoder help" as one
project-wide yes/no question.

### Interpretation: what remains uncertain after Stages A+B, and why Stage C is the right next experiment (2026-08-05)

**What Stages A and B have actually established, precisely stated.**
Stage A: roughly half of `sound_repetition`/`word_repetition` losses on
real ASR happen even at correctly-transcribed positions, and the
surface-level mechanism differs by type (fragment loss vs. pair-breaking).
Stage B: for `sound_repetition`, there is a real, measurable, *aggregate*
statistical difference — as a group, the 19 "normalized away" positions
sit farther from the fluent centroid than the 966 genuinely fluent
control positions, at an effect size (d=0.894) too large to dismiss as
noise. That is what has been shown. It is not yet the same thing as
"this is usable evidence for detection."

**The scientific uncertainty that remains, stated as open questions, not
assumed answers:**

1. **Is the Stage B signal a genuine disfluency signature, or a duration
   artifact wearing a disfluency's clothes?** A token that absorbed a
   fragment is very likely longer than an ordinary token of the same
   word — Stage B named this confound before running and could not
   resolve it with the design used. The result is consistent with either
   "the encoder detected the disfluent production itself" or "the
   encoder detected an unusually long span, which happens to correlate
   with where disfluencies occur." Both produce the same group-level
   effect size; nothing measured so far distinguishes them.
2. **Is a real aggregate effect strong enough at the instance level to
   build anything on?** Cohen's d=0.894 describes a *population*
   difference — two overlapping distributions with different means. It
   does not by itself say whether a threshold or classifier operating on
   one position at a time could separate individual disfluent positions
   from individual fluent ones at a precision/recall this project could
   actually ship. A real, sizeable group effect and a usable per-instance
   detector are related but different claims, and only the first has been
   tested.
3. **`word_repetition` remains genuinely open.** Not negative, not
   positive — undetermined whether a real, smaller effect exists there or
   whether the more indirect test design simply can't see one at n=17.

**Why Stage C is the correct next experiment, not a premature jump to
implementation.** Stage C is the cheapest available step that can
actually discriminate between the possibilities above, because it
requires the signal to do something a group-mean comparison never
tested: separate *individual* real candidates from individual non-
candidates, scored against real ground truth (Track B, not just an
aggregate distance comparison) — exactly the standard this project holds
every other detector-side claim to (`ROADMAP.md` item 19's own lesson:
never trust a Track-A-style or aggregate-only number for a claim about
real-world detection value). Building and evaluating a small,
representation-native candidate mechanism for `sound_repetition` is a
direct, falsifiable test of whether Stage B's aggregate result survives
contact with the same "does this actually help" standard the repetition
classifier was held to before it shipped.

**Competing hypotheses Stage C is designed to distinguish:**

- **H1 — Duration confound.** The signal is mainly token duration, not a
  disfluency signature. Predicts: a representation-native detector built
  on this signal performs little better than a naive "flag unusually
  long words" baseline, with poor precision (many long-but-fluent words
  falsely flagged).
- **H2 — Genuine acoustic disfluency signature.** The encoder captures
  something about the disfluent production itself (a residual trace of
  the aborted repetition, altered voicing/energy/pitch at the boundary)
  that is separable from duration alone. Predicts: the detector
  meaningfully beats a duration-only baseline and holds up across
  different words and durations, not just long ones.
- **H3 — Real but not (yet) actionable.** The group-level effect is real
  (Stage B stands either way) but individual disfluent and fluent
  positions overlap too much for a usable per-instance decision rule with
  this signal alone. Predicts: no threshold or simple classifier reaches
  acceptable precision/recall, even though the aggregate difference is
  genuine — a different, more specific conclusion than H1, and one that
  would point toward combining this signal with others (Stage C's own
  fusion-style precedent, §6e) rather than abandoning it.

**One concrete design consequence for Stage C's own pre-registration,
noted here rather than deferred silently**: because H1 vs. H2 is exactly
the confound question Stage B couldn't resolve, Stage C's protocol should
include an explicit duration-only baseline as a comparison arm (flag
positions whose real-ASR token duration is anomalous for that word,
using no encoder signal at all) — not just a "does the new detector work
in isolation" evaluation. Beating that baseline, not just beating chance,
is what would separate H2 from H1 with real evidence rather than
continuing to carry the same unresolved confound forward.

**Stage C — Build a representation-native (not decoded-text-dependent)
candidate path for the types Stage B confirms carry recoverable signal.**
Extends this project's own `block`/`prolongation` precedent (§6f) to
`sound_repetition`/`word_repetition` where justified. No fine-tuning, no
new training data — an architecture change using representations already
accessible. **Decision gate**: benchmark against Track A *and* Track B
(this track's central lesson from item 19: never trust a Track-A-only
number for this kind of change again) — proceed to shipping only if the
Track B improvement is real and non-trivial, per the same standard item
19 applied to the classifier gate.

### Stage C — pre-registered protocol (2026-08-05, written before running), scoped to `sound_repetition` only

Per Stage B's mixed result and the Interpretation above: Stage C is
scoped to `sound_repetition` only (`word_repetition` did not clear the
bar and is not extended here). Its job, precisely: distinguish H1
(duration confound) / H2 (genuine signature) / H3 (real but not
instance-actionable) — not to finalize a shipping decision.

**Design, deliberately the cheapest version that can still distinguish
the three hypotheses**: with only 19 positive (`sound_repetition`
target) instances in the entire sample, training a real classifier
(escalating past a threshold, mirroring §12's own M1-vs-M3 comparison)
is not yet justified by data volume — a **threshold/ranking-based
comparison**, not a trained model, matches this project's own precedent
for what a small sample can support. Two candidate-scoring arms, each
producing one score per position, evaluated identically:

- **Encoder arm**: Stage B's own `encoder_distance` (cosine distance to
  the clip's fluent centroid) — no new encoder passes needed, this data
  already exists (`eval_results/20260805T211000_stage_b_representation_
  probe.json`).
- **Duration-only baseline arm**: each position's real-ASR token
  duration, z-scored against that same word's duration wherever else it
  appears as a *clean* (non-disfluent) token in the sample — cheap to
  extract from the same cached `hyp_tokens`, no new ASR or encoder cost.
  This is the arm the Interpretation section named as necessary to
  separate H1 from H2 — without it, a positive result would be exactly
  as consistent with "detected a long token" as "detected a disfluency."

**Population**: the exact same 19 target (`sound_repetition`, Stage-A
category 1) and 966 control (genuinely fluent) positions Stage B already
collected — no new data collection for either arm.

**Metric**: ROC AUC for each arm (threshold-free, appropriate given the
19-vs-966 class imbalance — a single hand-picked threshold would hide how
sensitive any conclusion is to where it's set), plus precision at two
fixed, pre-declared recall points (0.5 and 0.7) so a concrete, realistic
operating point is reported alongside the summary statistic.

**Success criteria / how each hypothesis is read from the result, fixed
in advance**:
- **H2 (genuine signature) supported**: encoder-arm AUC is both
  meaningfully above chance (0.5) *and* meaningfully above the
  duration-arm AUC (not just numerically higher — a difference judged
  small enough to plausibly be noise at n=19 does not count).
- **H1 (duration confound) supported**: encoder-arm and duration-arm AUC
  are close enough that the encoder signal adds nothing distinguishable
  from duration alone.
- **H3 (real but not actionable) supported**: both arms are near chance
  or achieve only poor precision at both declared recall points, despite
  Stage B's real, statistically supported group-level effect.

**A limitation named before running, not after**: this evaluation is
**in-sample** — the same 19+966 positions that produced Stage B's effect
size are being reused to evaluate a threshold/ranking rule here, which is
the same kind of optimism this project's own `VALIDATION.md` §13.1
explicitly rejected for the repetition classifier ("naively applying the
final shipped model back to that same data would give an optimistic,
in-sample result"). At n=19 positives, a genuine held-out split would be
too small to be stable either way, so this stage is explicitly scoped as
**exploratory hypothesis-distinguishing, not a validated deployment
estimate** — a result here that supports H2 is evidence to invest in a
properly out-of-fold-validated follow-up (more data, a real train/test
split), not evidence ready to ship on its own. This limitation is
identical in kind to Stage 1's original pilot-vs-scaled-validation
pattern (§0, item 17's own history) — a small first pass earns a larger,
honest follow-up, not an immediate shipping decision.

### Stage C — Results (2026-08-05): H1 refuted, H2 supported, H3 also
supported — a genuine signal that isn't yet practically usable alone

**Three real bugs caught by this stage's own safety checks before any
result was trusted, each investigated rather than shrugged off**:
1. A `TypeError` crash on the first run — some real-ASR tokens have
   `start`/`end` of `None` (a real, if rare, property of real ASR output
   not previously encountered by this exact code path). Fixed by
   excluding positions with missing timestamps from the duration arm,
   counted and reported explicitly, not silently dropped.
2. A large, real mismatch (3347 vs. Stage B's 966 control positions) —
   traced to iterating all 120 clips instead of only the 31 clips that
   actually contain a target, unlike Stage B's own population. Fixed by
   applying the identical `if not targets: continue` filter Stage B
   used, restoring an exact target-count match.
3. A residual 1-position mismatch (967 vs. 966) after that fix — turned
   out to be the *same* missing-timestamp position from bug 1: `pool_
   span()` also returns `None` for a missing start/end (checked inline
   in `encoder_embedding.py`), so Stage B's own encoder arm silently
   excluded this exact position too. Once accounted for, both arms
   compute over the same 966-position control population — not a real
   discrepancy, a consistent exclusion on both sides once traced fully.

**Results**:

| Arm | AUC | Precision @ Recall>=0.5 | Precision @ Recall>=0.7 |
|---|---|---|---|
| Encoder (Stage B's `encoder_distance`) | **0.723** | 0.047 (achieved R=0.526) | 0.029 (achieved R=0.737) |
| Duration-only baseline | 0.483 | 0.018 (achieved R=0.526) | 0.020 (achieved R=0.737) |

**Against the pre-registered criteria — and a genuine correction to a
pre-registered assumption, checked directly rather than assumed**: the
pre-registration expected fragment-absorbing tokens to be *longer* than
ordinary tokens. The actual data shows the opposite of that specific
assumption: target positions' mean duration z-score is **-0.139** (very
slightly *shorter* than the control population's mean of 0, not
longer), and the duration arm's AUC (0.483) sits essentially at chance
— duration alone carries no meaningful discriminative signal in *either*
direction at this sample size. This is recorded as a correction to the
pre-registered assumption's specific direction, checked before writing
up the result, not smoothed over because the final verdict still came
out favorably.

**H1 (duration confound): refuted, not just "not confirmed."** If the
encoder signal were substantially explained by duration, the duration
arm would show elevated AUC too. It doesn't — it's indistinguishable
from chance. The encoder arm's real discrimination (AUC=0.723) is not
attributable to the confound this track named before running Stage B.

**H2 (genuine acoustic disfluency signature): supported.** The encoder
arm clears chance by a real margin and clearly outperforms the duration
baseline — exactly the pattern H2 predicted, on the pre-registered
metric, decided before this run.

**H3 (real but not yet instance-actionable): also supported —
simultaneously, not as a contradiction.** The pre-registration treated
H1/H2/H3 as if they'd point to a single answer; the actual result shows
two of them can both be true at once, which is itself the finding worth
reporting precisely rather than forcing a single checkbox. AUC=0.723
means the encoder ranks a random true instance above a random clean one
about 72% of the time — real, but the *absolute* precision achievable at
a useful recall is still low (4.7% precision to catch about half of true
instances, meaning roughly 19 false candidates for every true one at
that operating point) — a direct consequence of the extreme, realistic
class imbalance (19 positives vs. 966 negatives) that a single-signal,
threshold-only mechanism cannot overcome on its own, however genuine the
underlying signal is.

**What this means for next steps, stated precisely rather than as a
blanket "Stage C worked" or "Stage C failed"**: this signal should
**not** be shipped as a standalone, primary candidate generator for
`sound_repetition` on this evidence — the false-positive rate at any
recall worth having is too high. But it is real, evidence-backed
confirmation that CrisperWhisper's encoder carries genuine,
duration-independent disfluency information, which is exactly the kind
of signal this project has previously used as a *corroborating* input
alongside other signals (the fusion pattern already shipped for
`filler`/`stutter_marker`, and the gating role this exact encoder-
distance idea already plays for the repetition classifier itself,
§13) rather than a sole decision-maker. The natural, evidence-justified
next direction is **not** Stage D (fine-tuning) — the confound this
track worried about is refuted, and Stage D's own gate requires
richer-representation approaches to have failed, which they haven't —
it is a **fusion-style Stage C revision**: combine this signal with
other available evidence (e.g. the acoustic mis-routing lead from Stage
A, or additional acoustic features) rather than relying on it alone,
following this project's own §6(e) precedent, before concluding richer
representations "don't work" for `sound_repetition`. Not implemented in
this session — a scoped next step, not started here.

### Stage C2 — Fusion with acoustic voice-quality evidence: pre-registered protocol (2026-08-05, written before running)

**Re-scoping note, checked before writing any code — the handoff's
originally-proposed fusion candidate turns out not to be quantitatively
testable as first framed.** The handoff (and §8's own Stage C write-up)
named Stage A's "mis-routed" finding — positions where an existing
`block`/`filler`/`phrase_repetition` detector already fired instead of
`sound_repetition` — as the clearest first fusion candidate. Checking
this directly before building anything: Stage A's own categories 1
("normalized away," Stage C's target population) and 2 ("mis-routed")
are mutually exclusive by construction — category 1 is defined as
positions where *no* detector produced any prediction at all. This means
the mis-routing signal is constant (always "no") across every one of
Stage C's 19 target positions and cannot add discriminative information
there; and testing it on its own tiny population (n=4 mis-routed
`sound_repetition` cases, §8's table) would not be a meaningful
quantitative test at all — a single case flipping changes any measured
rate by 25 percentage points. **Re-scoped**: the 4 mis-routed cases stay
a qualitative observation (already reported in §8's Stage A findings),
not a quantitative fusion test. The signal actually tested here is a
different, still-available, still-well-powered one: **Praat-derived
voice-quality features** (`profiling/acoustic.py`'s existing
`_praat_features` — pitch stability, jitter, shimmer, HNR), already used
elsewhere in this codebase as `prolongation` corroboration, applied to
the exact same n=19/966 population Stage C used, so the comparison stays
apples-to-apples.

**Hypothesis under test**: does voice-quality irregularity (jitter,
shimmer, pitch instability, reduced harmonics-to-noise ratio —
physiologically plausible correlates of a disfluent, effortful, or
interrupted production, conceptually distinct from both the encoder's
learned representation and the already-refuted duration signal) carry
information the encoder-distance signal doesn't, and does combining them
improve on Stage C's own precision/recall?

**Design**:
1. For each of Stage C's 19 target + 966 control positions, extract
   `pitch_hz`, `pitch_std_hz`, `jitter`, `shimmer`, `hnr` via the
   existing `_praat_features(samples, sr, start, end)`, over the same
   real-ASR hyp-token span already used for the encoder-distance
   measurement. No new audio, no encoder pass — CPU-only signal
   processing, reusing infrastructure this project already ships.
2. **Screen each feature individually first** (same discipline Stage 1
   applied to the encoder signal before it was trusted): compute AUC for
   each of the 5 features alone against the same target/control labels
   (jitter/shimmer/`pitch_std_hz` scored so *higher* = more anomalous;
   `hnr` scored so *lower* = more anomalous, i.e. evaluated as `-hnr`).
   Missing values (Praat failure on short/unvoiced segments — a
   documented, expected mode of this function) excluded from that
   feature's own AUC computation, not imputed; missingness rate reported
   per feature.
3. **Only features clearing AUC >= 0.55** (a deliberately low screening
   bar, not the 0.5-vs-chance "positive result" bar Stage C used — this
   step exists to avoid combining pure-noise features into an ensemble
   and mistaking the resulting inflation for a real fusion effect) are
   carried into the combination step.
4. **Combination rule, fixed before results are seen**: if at least one
   feature clears the screening bar, combine it with encoder-distance via
   the **max of both signals' z-scores** (standardized against the
   control population) — an OR-like, training-free rule, chosen over a
   trained classifier because n=19 remains below what this project's own
   precedent (§12.6.2's nested-CV comparison) required before trusting a
   trained model on a signal like this, and because staying training-free
   keeps this stage in the same "cheapest version that can still test the
   hypothesis" register as Stage C itself.
5. Evaluate the combined score exactly as Stage C evaluated single
   signals: AUC, precision at recall>=0.5 and recall>=0.7.

**Success criteria, fixed in advance**:
- **Fusion helps**: combined-score AUC is meaningfully above Stage C's
  encoder-only AUC (0.723) *and* precision at recall>=0.5 is meaningfully
  above Stage C's 0.047 — not just numerically higher, given n=19 is
  small enough that a small gain is within plausible noise.
- **Fusion doesn't help**: no meaningful improvement over encoder-distance
  alone — reported as a real finding (H-fusion-insufficient), not a
  failure of the experiment.
- **No Praat feature clears the screening bar at all**: reported as its
  own distinct finding — voice-quality features carry no additional
  signal for this population, a different conclusion from "combining two
  informative signals didn't help," and one that would point away from
  acoustic-feature fusion specifically (not fusion in general) as this
  track's next lever.

**Named limitations, stated before running**:
- Still in-sample, same caveat as Stage C's own limitation section —
  exploratory hypothesis-testing, not a validated deployment estimate.
- n=19 positives is now being asked to support screening 5 features *and*
  a combination decision — a real, named risk of overfitting to noise
  even under the pre-registered screen-then-combine discipline. Any
  positive result here is evidence for a larger-sample follow-up, not a
  final answer on its own.
- Praat's own documented failure mode (short/unvoiced segments return
  `None`) may hit the 19-position target set harder or softer than the
  966-position control set by chance — missingness will be reported
  per-population, not just per-feature, so an uneven failure rate is
  visible rather than silently absorbed into the AUC computation.

### Stage C2 — Results (2026-08-06): no Praat voice-quality feature clears
the screening bar — a clean, specific negative result

**Cost, as it actually ran**: 82s to scan all 120 clips (CPU-only, no
model download) — Praat feature extraction is genuinely cheap, matching
the pre-registration's expectation.

**Screening results (AUC vs. chance=0.5, n=19 target / n=967 control
before missingness):**

| Feature | AUC | Target missing | Control missing |
|---|---|---|---|
| `pitch_hz` | 0.549 | 0/19 | 24/966 |
| `pitch_std_hz` | 0.471 | 0/19 | 24/966 |
| `jitter` | 0.527 | 0/19 | 29/966 |
| `shimmer` | 0.507 | 0/19 | 48/966 |
| `hnr` | 0.452 | 0/19 | 1/966 |

**None cleared the pre-registered AUC >= 0.55 screening bar** — every
feature sits close to chance (0.452-0.549), well below even this
deliberately low bar, let alone Stage C's own encoder-only AUC of 0.723
on the identical population. Per the pre-registered protocol, this
specific outcome — no feature passing screening — is a distinct finding
from "fusion didn't help": **the fusion combination step was correctly
not attempted at all**, since combining pure-noise signals with the
encoder-distance signal would only have added noise, not tested anything.

**Interpretation, labeled as a hypothesis, not a confirmed explanation**:
one plausible reason Praat voice-quality features work for this
project's `prolongation` detection (`ARCHITECTURE.md` §4a) but not here —
`prolongation` involves a sustained, voiced segment long enough for
reliable pitch/jitter/HNR tracking, while a `sound_repetition`-absorbing
token is typically an ordinary-length single word, exactly the short-
segment regime Praat's own pitch-tracking algorithms are known to be
least reliable in. Consistent with, but not proven by, the missingness
pattern (`hnr` and `pitch_hz`/`pitch_std_hz` fail on 24-48 of 966 control
positions, presumably shorter/less-voiced ones) — not independently
verified this session, stated as a plausible explanation only.

**What this resolves**: Praat-derived voice-quality features are ruled
out as this track's next fusion signal for `sound_repetition` — a real,
specific, useful negative result that narrows the search rather than
leaving it open. It does not touch Stage C's own encoder-distance
conclusion (H1 refuted, H2 supported, H3 also supported, §8 above), and
it does not rule out fusion in general — only this particular candidate
second signal. The mis-routing lead (Stage A category 2, n=4) remains a
real but small, qualitative-only observation (§8's Stage A findings),
not something this or any statistical test at this sample size can
confirm further. With both readily-available fusion candidates now
tried (mis-routing: too small to test; Praat: tested and ruled out),
the next-lowest-cost options are largely exhausted for `sound_
repetition` at this sample size — see the updated end-of-session
handoff below for what this implies.

**Stage D — If B/C are insufficient**: this is the evidence threshold
for seriously costing out (a) fine-tuning/continual adaptation or (d)
multitask training. Not attempted before this point. Requires first
addressing this project's real, named prerequisites: GPU access, a
paired dataset at sufficient volume (ties to `ROADMAP.md` items 14-16),
and a full cost/risk pre-registration matching the rigor `VALIDATION.md`
§12 applied to the repetition classifier before it was trusted.

**Stage E — Only if Stage D's evidence justifies it**: full
purpose-built-representation or fine-tuned-ASR work. Not started, not
assumed, not the default outcome of this track — the outcome only if
every cheaper stage's evidence points here.

---

## 9. What would justify concluding a purpose-built ASR (or a different
representation entirely) is necessary

All three of the following must hold — any one being false keeps this
project on cheaper alternatives:

1. **Information loss is broad, not isolated** — Stage A finds this
   pattern extends meaningfully beyond `sound_repetition` alone (RQ1).
2. **The loss is not recoverable from existing representations** — Stage
   B finds the encoder itself, not just the decoded text, has lost the
   signal (RQ2) — i.e. direction (b/f) has been tried and genuinely
   isn't enough, not merely untried.
3. **A real, sufficient paired dataset and the infrastructure to use it
   exist or are acquirable** — otherwise this is a correct conclusion
   with no way to act on it yet, which should be recorded as exactly
   that (a validated future-work item, not a stalled implementation).

---

## 10. Non-goals, stated explicitly

- This document does not commit to building a new ASR. It commits to
  finding out, in order of cost, whether one is needed.
- This document does not reopen `PHASE_3_ARCHITECTURE_REVIEW.md`'s
  two-stage-architecture conclusion (§2 above).
- Nothing in this track authorizes a config, threshold, or architecture
  change on `main` — findings here get evidence-gated the same way every
  other decision in this project has been (standing rules 4 and 8), and
  land on `main` only once a stage's evidence supports it.
- `main` stays on the currently-shipped, Track-A-and-now-Track-B-validated
  state throughout this track's work — this branch is additive research,
  not a replacement in progress.

---

## 11. Branch charter (condensed, for `asr-research`)

- **Objective**: determine what representation of speech — conventional
  ASR text, richer ASR-internal representations, an adapted/fine-tuned
  ASR, or a hybrid — is actually sufficient to preserve the
  speech-production information this project's seven-type taxonomy
  requires, and build whatever that turns out to require, evidence-gated
  at every step.
- **Research questions**: RQ1-RQ5, §7.
- **Roadmap**: Stages A-E, §8, each with an explicit exit/decision gate.
- **Decision criteria for the track's biggest possible outcome** (a
  purpose-built ASR/representation): §9's three-part test.
- **Governing philosophy**: implementation is never the objective by
  itself, and neither is documentation alone — the product and the
  research record advance together, the same creative, evidence-seeking
  discipline this project's sibling module's `Practice.md` set out
  explicitly: novel ideas without prior literature backing are welcome
  here, provided they're labeled as hypotheses, given a rationale, and
  validated before they're trusted — not gatekept behind "nobody's shown
  this works yet." `CLAUDE.md` rule 8 (evidence-constrained, not
  preservation-constrained) governs every decision this track produces.

---

## End-of-session handoff — 2026-08-05 close

**Read this section first if you are picking up cold.** It is written so
a new session can act on it directly — "continue from the end-of-session
handoff" — without re-deriving anything above.

**Update, 2026-08-06 — the "exact proposed next stage" below was
executed the following session; read this note before acting on the
original plan text further down, which is kept as written (append
discipline) but is now superseded on this one point.** Stage C2 (Praat
voice-quality fusion, §8) ran exactly as this handoff proposed:
pre-registered, then run. **Result: a clean negative** — none of five
Praat features (pitch, pitch stability, jitter, shimmer, HNR) cleared
even the low AUC>=0.55 screening bar (all near chance, 0.452-0.549), so
the fusion combination step was correctly not attempted at all. This
rules out Praat voice-quality features specifically as this track's next
signal — it does not touch Stage C's own encoder-distance conclusion.
With the mis-routing lead (n=4, too small to test statistically) and now
Praat (tested, ruled out) both explored, the readily-available low-cost
fusion candidates for `sound_repetition` are largely exhausted at this
sample size. **The evidence-grounded options from here, in order of
cost, updating this handoff's original "exact proposed next stage"
section below**:
1. **Scale up the sample** before trying further fusion candidates —
   more real-ASR clips would both sharpen Stage C's own encoder-distance
   estimate (n=19 is small) and make a mis-routing-style recovery
   statistically testable for the first time (ties to `ROADMAP.md` item
   10/14-16, real acquisition work, not a quick step).
2. **Try the mis-routing recovery as a small, separate, rule-based
   addition** (not a statistical fusion test — a direct rule: relabel an
   existing `block`/`filler`/`phrase_repetition` prediction as `sound_
   repetition` when encoder-distance is also high) — cheap, but only
   ever recoverable-in-principle for ~4 cases in the current sample, so
   its value is more about correctness than about moving a headline
   number.
3. **Formally cost out Stage D** (§9's three-part test) — two of its
   three conditions now have real evidence behind them (loss is broad,
   confirmed Stage A; encoder representations alone have been tried more
   than once and found insufficient alone, Stage C/C2) — the missing
   third condition (a real, sufficient paired dataset and infrastructure)
   is the actual open question worth pricing out next, rather than
   trying more cheap fusion candidates that keep coming back small or
   negative.

This update does not pick one of these three — it is recorded here as
the honest state of the decision, for whoever (human or Claude) continues
next to decide with, not decided unilaterally in the middle of a
session-close note.

### What was completed today (full session, not just this track)

1. Re-ran the shipped repetition-classifier gate against real ASR output
   for the first time (`ROADMAP.md` item 19, `VALIDATION.md` §14/§14.1).
   Mechanism confirmed safe; real-world impact found negligible because
   real ASR starves both gated types of candidates.
2. Opened this research track from that finding: reframed core question,
   a formal problem statement, a real 13-source literature review, six
   architectural directions explored without commitment, five research
   questions, and a phased, evidence-gated plan (Stages A-E).
3. **Stage A** (done): systematically categorized all 186 disfluent
   ground-truth positions in the 120-clip Track B sample. Found ~53% of
   `sound_repetition`/`word_repetition` losses happen even at
   correctly-transcribed positions, via two distinct mechanisms (fragment
   loss vs. pair-breaking).
4. Pushed `main` and `asr-research` to GitHub, verified in sync by direct
   hash comparison (not just trusting command output).
5. Wrote an explicit Interpretation section before Stage C: named the
   real remaining uncertainty (aggregate effect vs. instance-level
   actionability vs. confound), three competing hypotheses (H1/H2/H3),
   and the concrete design consequence (a duration-only baseline arm)
   that shaped Stage C's actual protocol.
6. **Stage B** (done): encoder representation-level probe. `sound_
   repetition` positive (Cohen's d=0.894); `word_repetition` inconclusive
   (d=0.428). One real identification bug caught and fixed before
   trusting the numbers.
7. **Stage C** (done): encoder-distance arm vs. a duration-only baseline
   arm, scoped to `sound_repetition`. H1 (duration confound) refuted; H2
   (genuine signature) supported; H3 (real but not yet instance-
   actionable) also supported, simultaneously. Three real bugs caught and
   fixed via the script's own safety-check assertions before trusting any
   number, including a corrected pre-registered assumption (duration
   direction).
8. This final pass: a full documentation/consistency audit (this
   section, plus edits to `CLAUDE.md`, `ARCHITECTURE.md`, `README.md`,
   `HANDOFF.md`, `DOCS.md`, `VALIDATION.md`'s status header) to make sure
   the objective hierarchy (user audio → ASR is one subsystem → transcript
   is one evidence source, not ground truth → representations are
   complementary, evidence-gated) is stated consistently, and that every
   major doc accurately reflects today's conclusions, not just the state
   at the start of the day.

Every stage was pre-registered before running, every result — positive,
negative, mixed, or inconclusive — was reported as measured, and every
bug found was caught by a safety check built into the work itself, not
discovered later by accident.

### Current research state and the strongest conclusions the evidence actually supports

- **Real ASR normalization is a real, measured, two-mechanism phenomenon
  for `sound_repetition`/`word_repetition`**, not a hypothesis anymore —
  Stage A traced it precisely (fragment loss; pair-breaking, 22/23 hand-
  checked cases).
- **CrisperWhisper's encoder retains genuine, duration-independent
  disfluency signal for `sound_repetition`** even where decoded text
  shows nothing — the strongest single conclusion from today's work, and
  the confound named at the very start of Stage B has now been directly
  refuted, not merely left unresolved.
- **That signal alone is not yet precise enough to ship as a standalone
  candidate generator** — the honest, load-bearing caveat on the
  conclusion above. Real (AUC=0.723) is not the same claim as usable
  alone (4.7% precision at 52.6% recall).
- **`word_repetition` is a genuinely separate, still-open question** —
  do not assume the `sound_repetition` conclusion transfers to it.
- **No production code has changed.** `main` reflects only the Track B
  validation of the already-shipped classifier (item 19); the entire
  research-track arc (Stages A-C) lives on `asr-research` and has not
  been merged, by design.

### Remaining uncertainties and open research questions

- Whether a fusion-style combination of the encoder signal with other
  evidence (Stage A's mis-routing lead, acoustic features) actually
  closes Stage C's precision gap — a real, untested hypothesis.
- `word_repetition`'s question, unresolved (inconclusive at n=17, not a
  negative result).
- Stage B/C's control-group non-independence (positions pooled across
  clips, not fully i.i.d.) — named, not yet addressed; a more rigorous
  clip-level analysis would strengthen any result built on top of this
  one before it carries real architectural weight.
- Whether this generalizes beyond CrisperWhisper/LibriStutter (`ROADMAP.md`
  item 10 — unaddressed by anything done today).
- RQ3 (is the loss ASR-general or CrisperWhisper-specific) and RQ4 (the
  not-yet-deep-read arXiv:2512.02027) — both still open, listed in §7,
  neither touched this session.
- Whether Stage A's incidentally-found triple-repeat detector bug
  (`ROADMAP.md` item 21) is an isolated case or a broader pattern — not
  investigated, flagged for `main`, independent of this track.

### The exact proposed next stage

**A fusion-style revision of the Stage C candidate mechanism for
`sound_repetition`** — not Stage D (fine-tuning), because Stage D's own
pre-registered gate (§9) requires richer-representation approaches to
have *failed*, and today's evidence shows the opposite: the confound is
refuted and the signal is genuine, just not sufficient alone. Concretely,
this means combining the encoder-distance signal with at least one other
already-available source of evidence — the clearest first candidate is
Stage A's own "mis-routed" finding (§8, category 2: ~10% of true
`sound_repetition` instances already surface as a `block`/`filler`/
`phrase_repetition` prediction from existing detectors) — rather than
relying on the encoder-distance signal as a sole decision-maker.

**Why this logically follows, not just "what's left on the list"**: it
is the cheapest next step consistent with everything measured today. It
does not require new data collection, new model training, or GPU access
— the two ingredients (encoder distances, existing detector outputs) both
already exist. It directly targets Stage C's own diagnosed weakness
(insufficient precision from a single signal under realistic class
imbalance) rather than re-testing something already resolved (H1) or
escalating past evidence that doesn't yet justify it (Stage D).

**What hypotheses it is intended to test**:
- **H-fusion-positive**: combining the encoder-distance signal with the
  mis-routing signal (and/or other acoustic evidence) measurably improves
  precision at a useful recall over the encoder-distance-only arm from
  Stage C, without requiring a trained classifier (a rule-based or simple
  weighted combination, matching this project's existing fusion
  precedent, `ARCHITECTURE.md` §4).
- **H-fusion-insufficient**: combining available signals still does not
  reach a precision/recall operating point worth shipping — a genuine,
  reportable negative result that would sharpen §9's evidence-threshold
  question (is a *trained* combination, i.e. escalating within Stage C
  rather than to Stage D, now justified by having tried the cheaper
  rule-based version first).

### Detailed plan for the next working session

Execute in this order; each step is a prerequisite for the next being
trustworthy, matching every other stage's own discipline this session:

1. **Pre-register the fusion protocol** (before any code): exact signals
   to combine (encoder-distance + Stage A's mis-routing predictions, at
   minimum), exact combination rule proposed (start with the simplest —
   OR-combine or a weighted sum — before considering a trained
   combination), exact evaluation population (extend beyond the 19/966
   Stage C used if feasible at reasonable cost — check first, per Stage
   B's own "cost, scoped before running" precedent), and success
   criteria fixed in advance (a concrete precision-at-recall bar that
   would justify shipping, and what would count as H-fusion-insufficient).
   **Deliverable**: a pre-registered protocol section appended to this
   document's §8, dated, before any implementation.
2. **Implement and run**, reusing existing infrastructure
   (`profiling/encoder_embedding.py`, Stage A's categorization logic,
   `profiling/evaluation/`'s established script conventions) — new
   research code only, still on `asr-research`, still not touching
   `main` or `profiling/detect.py` directly.
   **Deliverable**: a new `profiling/evaluation/stage_c_fusion_*.py`
   script (or equivalent), with the same self-audit discipline every
   prior stage's script used (known-answer sanity checks before trusting
   real output; count/population reconciliation against prior stages'
   saved data before pairing anything).
3. **Report the result exactly as it came out** — positive, negative, or
   mixed, per this session's own standing discipline — in a dated
   results subsection, updating this handoff's "current research state"
   accordingly.
4. **Update `ROADMAP.md`, `PAPER_DECISION_LOG.md`, `CHANGELOG.md`** with
   the same granularity every stage this session received.
5. **Decide the next branch action** based on the result: if
   H-fusion-positive and the improvement is real and non-trivial per the
   pre-registered bar, that's the evidence Stage C's own original
   decision gate asked for ("benchmark against Track A and Track B...
   proceed to shipping only if the Track B improvement is real and
   non-trivial") — the next step becomes preparing this for a `main`
   merge, which is its own deliberate decision, not automatic. If
   H-fusion-insufficient, the next step is deciding between: (a) trying a
   trained (not just rule-based) combination before concluding richer
   representations are exhausted for `sound_repetition`, or (b) formally
   costing out Stage D per §9's three-part test, now that two of its
   three conditions have real evidence behind them.

**Success criteria for the next session overall**: a pre-registered
fusion protocol exists, a real result (any direction) is measured and
reported with the same rigor as every stage this session, and a
concrete, evidence-grounded decision about what comes after it is made
and recorded — not left as an open question a third time.

**Stopping conditions** (when to end that session, matching this one's
own pattern): once the fusion result is measured, documented across all
the standard files, committed and pushed to `asr-research`, and this
handoff section is updated to reflect the new state — do not
automatically continue into Stage D or a `main` merge in the same
session without an explicit go-ahead, per standing rule 6 and this
project's consistent pattern of pausing at exactly these decision points
today.

### Recommendations, risks, and decisions deserving attention before further implementation

- **Do not merge anything from `asr-research` to `main` without an
  explicit decision to do so.** Nothing here has cleared this project's
  own bar for a production change yet (Stage C's own limitation section
  says so directly — in-sample, exploratory, not a validated deployment
  estimate).
- **The control-group independence caveat (Stage B/C) is worth resolving
  before this result is cited in anything higher-stakes** (a paper draft,
  a architecture decision write-up) — cheap to address (a clip-level
  bootstrap) relative to the risk of overstating confidence in the
  effect size.
- **Do not let the fusion step quietly turn into Stage D.** If the first,
  cheap rule-based fusion attempt looks disappointing, the temptation
  will be to jump straight to a trained combination or to fine-tuning —
  resist that without re-checking §9's three-part test explicitly; a
  disappointing cheap result is itself informative and should be reported
  as such before escalating cost.
- **`word_repetition` should not be forgotten** — it's easy for a project
  to quietly narrow to "the type that worked." If a larger real-ASR
  sample ever becomes available (ties to `ROADMAP.md` items 10/14-16),
  re-running Stage B for `word_repetition` at higher n is a cheap,
  valuable use of it.
- **Item 21** (the triple-repeat detector bug) is small, `main`-side, and
  unrelated to this track — a reasonable thing to fix in a spare moment
  on `main` without waiting for this track to reach any particular
  milestone first.
