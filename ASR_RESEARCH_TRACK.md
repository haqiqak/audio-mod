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
  an implicit or explic't language-model prior favors higher-probability,
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
