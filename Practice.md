# Practice.md — Research & Engineering Methodology for the Speech AI Module

**Read this before making any architectural, implementation, or research
decision in this repository.** It does not describe what the code
currently does — `README.md`, `changes.md`, and the code itself do that,
and drift between this document and the code over time is expected (see
§14 for how to handle that drift, not avoid it). What this document
defines is *how decisions in this repository get made and recorded*, so
that the answer to "why does this exist" and "what evidence supports it"
is always recoverable — by a teammate, or by a Claude instance starting
completely cold, with no memory of any prior conversation and no access
to any other repository, months or years from now.

**Every example in this document is self-contained.** Where a lesson is
illustrated with a scenario ("consider a project where...", "in one
engineering effort..."), that scenario is written to be understood on its
own, without requiring access to any other codebase, conversation, or
file this document doesn't itself quote. Read those as patterns worth
recognizing, not as pointers to go look something up elsewhere — because
whoever reads this next may have nothing else to go look up.

---

## 0. How to use this document

1. Read this document in full before touching the Speech AI codebase.
2. Do **not** begin implementing, refactoring, or "fixing" anything on
   the strength of this document alone. This document is a lens, not a
   punch list. §19 defines the actual protocol for a first pass over this
   repository, and that protocol ends in *"only then begin validation,
   experimentation, implementation, and benchmarking where justified"* —
   not before.
3. Where this document's own principles turn out to be wrong, incomplete,
   or actively counterproductive once real research on this repository is
   underway, they should change — see §20. This document is itself
   subject to the evidence-constrained discipline it asks everything else
   in this repository to follow.
4. This methodology exists to make research rigorous, not to make it
   cautious. §4 states this directly: original ideas and untested
   architectures are not something this document tolerates — they are
   something it actively wants, provided they're labeled honestly and
   validated before they're trusted (§4, §5).

---

## 1. Where this module fits: the two-repository system

The complete product is two repositories working in sequence:

1. **Audio Module** — receives speech from a speaker, transcribes it, and
   detects, classifies, and localizes disfluencies within it. Outputs:
   transcribed text, structured disfluency information, and the
   speaker's evolving speech profile.
2. **Speech AI Module** (this repository) — receives transcribed text
   together with a speaker profile (today edited manually; the intended
   direction is that it's derived and continuously updated from the Audio
   Module's own output over time) and rewrites the text. Its job is
   **not** "replace hard words." Its job is to produce alternative
   wording such that:
   - the original **meaning** is preserved,
   - the original **intent** is preserved,
   - the **narrative** remains intact,
   - the **natural flow** of the text remains intact,
   - while reducing the likelihood that *this specific speaker* will
     experience speech difficulty producing it.

**The research objective, stated precisely** (this is the sentence every
architectural decision in this repository should be checked against):

> Given a piece of text and a speaker profile, generate alternative
> wording that remains as faithful as possible to the author's intended
> meaning while making the text easier for that specific speaker to
> produce naturally, based on a profile of difficulties derived from
> audio input.

The current profile-generation mechanism (self-report, hand-edited JSON)
is scaffolding for that objective, not the object of study — transcribed
text and disfluency data are themselves scaffolding for the Audio
Module's own goal in exactly the same way. Do not let engineering effort
drift into optimizing the profile format or the manual-edit UX as if
*that* were the research question. **The purpose of building any of this
— the profiling, the rewriting, all of it — is to help people speak more
easily and more naturally. That is the standard every downstream metric
ultimately has to answer to, even the ones that are easier to compute.**

---

## 2. Core philosophy: implementation is an experiment, not the destination

State this plainly, because it is easy to lose sight of under deadline
pressure: **implementation is never the end goal.** Every implementation
in this repository exists to answer a research question, and every
research question exists in service of the product. This is not a
tension to manage — it's one loop:

```
 implementation  ─── answers ───▶  a research question
       ▲                                    │
       │                             strengthens or
     informs                          weakens our
       │                             current understanding
       │                                    │
 documentation   ◀── preserves both ────────┘
       │
       └─── the record from which the product AND the eventual
            paper are both built, continuously, not retroactively
```

Three things should be improving together, not competing for the same
hours:

- **The product** should be getting more capable — better rewrites, more
  faithful to meaning, more genuinely easier for the speaker who asked
  for them.
- **Scientific understanding** should be getting deeper — a clearer
  picture of what actually reduces stutter-relevant difficulty in
  rewritten text, and what doesn't, and why.
- **Documentation** should be getting more complete — every decision, why
  it was made, and what it measured, recorded at the time it happened.

If a change only makes the code "cleaner" without answering a question
about what makes text easier to speak, it's refactoring, not research —
fine to do, but don't dress it up as a finding. If a change is proposed
"because it's the more sophisticated approach" without evidence it
outperforms the simpler one on this repository's actual objective, that's
not evidence-driven either (§3). And if a real, well-designed experiment
produces a negative or null result, that is not a failure to hide — it is
exactly as valuable a contribution to the documented record as a positive
one (§13), because it's information a future engineer would otherwise
have to rediscover the hard way.

**We are not building a paper, and we are not building a prototype. We
are building a real product.** The paper and the documentation are not a
separate objective competing with the product for effort — they are the
permanent scientific record of why the product turned out the way it
did. By the point this project is "done," it should have produced a
strong product, a complete research record, and a paper that falls out
of that record almost by transcription rather than by having to be
reconstructed from memory.

---

## 3. Architectural philosophy: evidence-constrained, not preservation-constrained

State this as plainly as §2, because it governs every "should we keep X
or replace it with Y" question this repository will ever face:

**Architectures are not preserved because they already exist. They are
not replaced because a newer technique appeared. Every architectural
decision is constrained by evidence, in both directions.**

Concretely, for this repository, that means:

- The current pipeline — rule-based grammar correction, WordNet/Datamuse
  candidate generation, SBERT semantic filtering, frequency/similarity
  ranking, phoneme-onset gating, the profile-aware soft-rewrite score
  (`similarity - λ·difficulty + μ·frequency`) — is a hypothesis that has
  earned its place so far by being buildable and by producing plausible
  output, not by having been proven to be the best available approach to
  the stated research objective.
- If evidence shows a hybrid approach, a learned re-ranker, a different
  semantic-preservation model, or a wholesale different architecture
  produces measurably better speaker-appropriate, meaning-faithful
  rewrites, that change should be **welcomed**, not resisted because the
  current pipeline is already working code.
- If the current architecture remains the best-supported option once
  actually tested against alternatives, it should **remain** — not
  because it's already there, but because it earned it.
- Simplicity, interpretability, rule-based logic, and learned/pretrained
  components are all *engineering choices*, not virtues in themselves.
  None gets automatic priority. The only question that governs a decision
  here is: **which approach most effectively produces text that is
  faithful to the author's meaning and genuinely easier for this speaker
  to say, and what evidence supports that conclusion?**
- Autonomy to make an architectural call is granted on this basis, but it
  is *evidence-constrained*: reach a decision once real evidence supports
  it confidently, and record the reasoning so a future reader sees why,
  not just what — or, if the evidence isn't there yet, say so explicitly,
  name exactly what's uncertain, pre-register the validation that would
  resolve it (§8), run it, and only then decide. Don't guess, and don't
  default to "keep what's there" or "swap in what's newer" just because
  one of those is less effort to justify.

**A pattern worth watching for explicitly**: an old constraint can
quietly outlive the evidence that once justified it. Consider a project
that defers a promising direction — say, a learned/trained component —
because the infrastructure to train and validate it doesn't exist yet. A
legitimate reason, at the time. Months later, unrelated work happens to
build exactly that infrastructure (a training pipeline, built for a
different feature entirely). If nobody revisits the original deferral,
the old reasoning keeps blocking a decision it no longer applies to — not
because anyone re-evaluated it and confirmed it still holds, but simply
because nobody re-checked. The fix is procedural, not clever: whenever a
constraint is invoked to defer or reject a direction, write down *why*
(§14), so a future reader — including a future instance of you — can
check whether that specific reason still holds before reusing it as a
justification.

**One caution about how to read this section.** "Evidence-constrained"
describes what it takes for a decision to *ship* — it is not a
requirement that an idea must already be backed by evidence before it
can be *proposed*. That distinction matters enough to get its own
section, next.

---

## 4. Creative, evidence-seeking engineering: propose first, validate before it ships

This methodology exists to make research rigorous, not to make it
cautious. Nothing in §3 should be read as "only propose an architecture,
mechanism, or design if the literature or this repository's own history
already supports it" — that would quietly optimize for caution over
discovery, which is backwards for a project whose stated purpose
includes deepening scientific understanding, not just avoiding mistakes.

**Original ideas, alternative architectures, and approaches with no
direct precedent in the literature or in this repository's own history
are explicitly encouraged whenever there is a plausible mechanistic
reason to believe they might better serve the stated objective.**
Absence of prior evidence is not a reason to withhold a proposal — it's a
reason to be honest about what kind of claim it is (§5) and to design the
experiment that would tell whether it's right.

The discipline that keeps this from becoming undisciplined guessing is
simple, and it has three parts:

1. **Say what kind of claim it is, out loud, at the moment it's
   proposed.** A novel idea is a hypothesis or an engineering judgment
   (§5) — never presented with the confidence of an established fact, and
   never quietly written into the codebase as though literature or
   precedent already backed it.
2. **Give the rationale, briefly.** Why is this plausible? What
   mechanism, analogy, or reasoning makes it worth the cost of testing?
   This doesn't need to be a literature citation — "this seems likely to
   help because X, based on how Y behaves in a related setting" is a
   legitimate, sufficient rationale, as long as it's stated as reasoning
   rather than dressed up as settled fact.
3. **Validate before it becomes permanent.** A novel idea earns a place
   in the shipped system the same way any other architectural claim does
   (§3): through a real experiment, pre-registered where the stakes
   justify it (§8), not through having sounded convincing when proposed.
   Until that validation happens, it stays labeled as exactly what it is
   — a hypothesis being tested, not a decision that's been made.

None of this changes §3's actual standard — a decision still has to earn
its place through evidence before it ships. What it changes is where
that evidence is allowed to *come from*: not only "what the literature
already established," but also "a new idea, proposed on its own
reasoning, and then actually tested." A methodology that only ever
validates ideas the literature already suggested will never discover
anything the literature doesn't already know. The goal is a system that
gets better than its starting assumptions, not one that only ever
confirms them.

**Example**: suppose no published work has directly tested whether a
signal this project already computes for an unrelated reason might also
serve as a useful proxy for a difficulty judgment nothing else currently
captures well. There's no literature precedent for it, but there's a
plausible mechanistic reason it might correlate. That is exactly the
kind of thing this methodology wants proposed — not gatekept behind
"nobody's shown this works yet." Propose it, label it clearly as a
hypothesis with its rationale, design the cheapest experiment that would
tell you whether it's right, and let the result — not the absence of
precedent, and not the persuasiveness of the pitch — decide whether it
ships.

This applies as much to entirely new architectures as it does to a small
mechanism swap. If a genuinely different way of approaching the rewrite
problem seems, on reasoning alone, like it could better serve the stated
objective — propose it. Say clearly that it's a hypothesis, not a
finding. Then find the cheapest real test that would tell you if you're
right.

---

## 5. The vocabulary of evidence

Every claim in this repository's documentation should be legible as
exactly one of the following. Mixing them — stating a hypothesis with the
confidence of a fact, or a limitation with the vagueness of a "known
issue" — is the single most common way documentation quietly becomes
untrustworthy. These categories exist to make novelty *legible*, not to
gatekeep it (§4): an idea with no precedent behind it is not a lesser
citizen of this repository than one with a citation behind it, as long as
it's labeled honestly.

| Category | Definition | Example (Speech AI-shaped) |
|---|---|---|
| **Fact** | Directly verifiable from code or data, not an interpretation. | "`semantic.py` rejects a candidate if cosine similarity to the original sentence falls below the configured threshold (default 0.85)." |
| **Observation** | A measured result from a specific run, with its exact conditions stated. | "On a 40-sentence pilot, SBERT-filtered candidates preserved the annotated 'core meaning' in 92% of accepted substitutions — n=40, single annotator, not yet inter-rater checked." |
| **Hypothesis** | A proposed explanation or prediction, not yet confirmed — including a genuinely novel idea with no literature precedent, as long as it's labeled as one. | "We hypothesize that phoneme-onset-only filtering under-corrects for polysyllabic-word difficulty, since it only screens the first sound." |
| **Engineering decision** | A choice made and its stated reasoning, whether or not it was preceded by a formal experiment. | "Chose Datamuse + WordNet over a single source because WordNet alone returned zero candidates for ~15% of test nouns (informally observed, not yet a pre-registered measurement)." |
| **Limitation** | A known, named boundary of what current evidence supports — not swept into "known issues" with no severity or scope attached. | "The difficulty formula (`0.4·onset + 0.3·syllables + 0.3·rarity`) has never been validated against actual speaker-reported or observed difficulty — its weights are engineering defaults, not fitted or measured." |
| **Future work** | An identified, not-yet-started direction, with enough reasoning attached that a reader knows why it's there and roughly how urgent it is. | "Validate the difficulty formula's weights against real disfluency-rate data from the Audio Module's output, once enough paired (utterance, disfluency) data exists per speaker." |

Every entry in this repository's decision log (§14) should make clear
which of these six categories each sentence belongs to. A sentence that
can't be classified into one of them is usually a sign it's either
overclaiming (a hypothesis dressed as a fact) or underspecified (a vague
gesture instead of a checkable claim).

---

## 6. Evidence-driven engineering

The operating rule: **a change ships because evidence supports it, not
because it seems reasonable.** "Seems reasonable" is where every
plausible-but-wrong assumption in a project starts. Bugs that silently
change a result — a scoring approximation that quietly overstates a
number, a data-handling bug that zeros out real input while a synthetic
test keeps passing — tend to survive precisely because they seemed
reasonable and nobody had yet built the check that would have caught
them.

In practice, for this repository, this means:

- A metric or benchmark should exist, or be built, before a change that
  claims to improve something is trusted to have improved it. "This
  candidate ranking feels better" is a hypothesis (§5), not evidence —
  propose it freely (§4), just don't ship it on the strength of feeling
  alone.
- When a metric is cheap to compute and a "real" outcome is expensive to
  measure, that gap is itself a risk to name explicitly, not a reason to
  quietly optimize the cheap one forever (§12 — the single highest-value
  lesson in this document).
- Surprising or dramatic-looking results get checked harder, not reported
  faster. A synonym-substitution change that looks like it doubled
  acceptance rate deserves scrutiny before it's trusted — verify the
  measurement pipeline itself before trusting the number it produced.
- Config values, thresholds, and weights (the SBERT similarity gate, the
  `λ`/`μ` weights in the rewrite score, the difficulty formula's
  coefficients) do not get retuned in response to an evaluation result
  without an explicit, separate go-ahead. Findings get recorded as
  evidence first; acting on them is a distinct, deliberate step. This
  matters especially here because several of this repository's current
  weights (`0.4/0.3/0.3` for difficulty, `0.65/0.35` for ranking) read as
  round, hand-picked numbers rather than fitted or validated ones — that
  is a limitation (§5) worth naming precisely, not a reason to change
  them without a measurement backing the change.

---

## 7. Literature review methodology

Before implementing a change to how this repository selects or ranks
alternative wording, check whether the question has already been studied
— in speech-language pathology, in lexical-substitution/paraphrase NLP,
in stutter-therapy and augmentative-communication research, in
readability/simplification literature. A literature pass isn't a
formality; it exists to catch two things early: (a) a taxonomy or
assumption this repository has built in that the field considers wrong or
incomplete, and (b) a technique that's already been tried and found not
to work, so the same dead end isn't rediscovered from scratch. It is a
tool for sharpening a proposal, not a gate a proposal must clear before
it's allowed to exist (§4) — plenty of good ideas here will have no
literature precedent at all, and that's expected, not a problem.

**How to record it:**

- Write the review as its own document: what was searched, what was
  found, what it confirms or contradicts about the current
  implementation, and a structured plan for what to do about it —
  written *before* the implementation that follows from it, not as an
  afterword justifying work already done.
- State explicitly which of this repository's current assumptions the
  literature corroborates, which it leaves untested, and which it
  actively challenges. All three are useful outcomes; a review that finds
  "everything already checks out" is a legitimate, reportable result, not
  a wasted pass.
- Cite specific sources, not general impressions of "the field." If a
  claim can't be traced to a specific source, it's a hypothesis (§5), and
  should be labeled as one until it is.

**Concrete question this repository will need a literature answer to,
sooner rather than later**: is a fixed phoneme-onset match (this
repository's current stutter-trigger model) actually the dominant
predictor of spoken difficulty for the population this is meant to help,
or does the literature suggest syllable structure, word length, semantic
load, or sentence position matter comparably or more? The current
`difficulty = 0.4·onset + 0.3·syllables + 0.3·rarity` formula bakes in an
implicit answer to this question already, without (as far as this
document's author can tell) a literature pass or a validation run behind
those specific weights. That gap is exactly the kind of thing §19's
review protocol should surface and either resolve or explicitly
pre-register.

---

## 8. Experiment design and pre-registration

**Write the exact metric, protocol, and success criteria down before
writing the code that produces the result.** This is the rule that does
the most work in this whole methodology — and it is exactly the
mechanism that turns a proposed hypothesis (§4) into a validated
decision (§3). The pattern:

1. State the question precisely (not "is the new ranker better?" but "on
   dataset X, does ranker B produce a higher rate of human-judged
   meaning-preservation than ranker A, at matched or better difficulty
   reduction, measured how, on how many sentences, judged by whom?").
2. Decide the success criteria — the exact bar a result has to clear to
   count as "this direction wins" — *before* seeing the result. If a
   result later requires deviating from the pre-registered protocol
   (e.g. the planned sample size turns out too small, or a metric turns
   out not to discriminate), record that as a dated addendum, not a
   silent edit to the original plan.
3. Only then implement and run it.
4. Report the result exactly as it came out, including when it
   contradicts the pre-registered prediction. **A contradicted
   prediction is a finding, not a failed experiment.** For example: a
   team predicts, before running a comparison, that a large measured
   effect from a simple signal will leave little room for a more complex
   mechanism to improve on it. The comparison runs, and the complex
   mechanism wins decisively anyway — the opposite of the pre-registered
   prediction. The honest move is to report that the prediction was
   wrong and explain what that implies, not to quietly reframe the win as
   though it had been expected all along.

**Speech AI-shaped example of what this looks like in practice**: before
building a learned re-ranker to replace the current
similarity/frequency-weighted candidate score, pre-register: the exact
comparison dataset, the exact meaning-preservation metric (human
judgment? SBERT similarity to source? both, reported separately, never
blended — see §10), the exact difficulty-reduction metric, and the exact
bar ("re-ranker must not regress meaning-preservation and must improve
difficulty-reduction by at least X, measured on a held-out speaker
profile set") that would justify shipping it over the current approach.
Write that down first. Then build it. This applies identically whether
the re-ranker is a well-precedented technique or a novel idea proposed
under §4 — the pre-registration discipline is what makes either one
trustworthy.

---

## 9. Implementation methodology and testing

- Every change that can be tested without a heavy model (SBERT, T5,
  CrisperWhisper) should be. This repository already has a real cost
  asymmetry here — the rephrase model (~890 MB) and CrisperWhisper
  (~3 GB) are expensive to load — so a fast, model-free test suite is not
  a nicety, it's load-bearing for development speed the moment any
  model-backed component enters the picture.
- New capability should ship with tests that would have caught the bugs
  a naive first implementation is likely to have. **Illustrative
  example**: a new model-backed component is added, wrapped in a class
  that loads its (heavy) model inside its constructor. The moment any
  test exercises that class — even one that never actually needed the
  model's output — the fast test suite silently becomes slow, or hangs
  entirely. Isolated unit tests, written specifically to run *without*
  the real model, catch this immediately: the fix is to make loading
  lazy (only triggered the first time the model's output is actually
  needed), and to make sure the code path that decides "do we need the
  model here" is itself cheap and correct — a related bug in the same
  family is a check that gets evaluated unconditionally for every input
  instead of only for inputs that actually need it, which can turn a
  rare, expensive operation into one that runs on every call and hangs
  the whole process. Diagnosing that specific failure mode is easy once
  you know to look for it: near-zero CPU time over a long wall-clock
  delay means the process is blocked waiting on something (a model
  download, a network call, a lock), not stuck computing — a useful
  general diagnostic, not specific to any one project.
- A "smoke test on one example" is not validation (§8/§12) — it's a
  sanity check that the code runs, useful before spending time on a real
  evaluation, not a substitute for one.

---

## 10. Benchmarking philosophy

- **Never report a single blended score if the underlying measurement
  covers genuinely different questions.** This repository already has at
  least two distinct axes that must never be collapsed into one number:
  *meaning/intent preservation* and *difficulty reduction for this
  speaker*. A rewrite that is very easy to say but has drifted from the
  original meaning is not a win, and a rewrite that preserves meaning
  perfectly but doesn't reduce difficulty at all hasn't done its job
  either — report both, always, side by side, never averaged into a
  single "quality" figure that could hide either one failing.
- Report per-condition results, not just aggregates: results should be
  broken down by profile type (a speaker who blocks on plosives vs. one
  who repeats sounds), by sentence complexity, and by how a candidate was
  sourced (WordNet vs. Datamuse vs. a future learned re-ranker) — an
  aggregate can hide that a change helps one subgroup while quietly
  hurting another.
- State what a benchmark does *not* cover as clearly as what it does. If
  a metric only measures candidate-level acceptability and not whether
  the *final rewritten sentence* still reads naturally as connected
  prose, say so — don't let a component-level win imply a
  system-level one it hasn't actually demonstrated.

---

## 11. Ablation studies

When multiple components jointly produce an effect (grammar correction +
candidate generation + SBERT filtering + phoneme gating + ranking, all
stacked), isolate which one is actually doing the work before assuming
they all matter equally. **Illustrative example**: a project runs a full
ablation across many independently-toggled configuration variants,
specifically because intuition about which component mattered had turned
out wrong before. One component that was widely believed to matter
(a corroborating signal meant to boost confidence when two independent
checks agreed) turned out, once properly isolated, to have a
near-zero measurable effect on the actual outcome — and that null result
got reported as a real, useful finding rather than quietly dropped
because it wasn't the expected answer.

**Speech AI-shaped candidates for this repository's first ablation, once
a real evaluation harness exists**: does the phoneme-onset gate actually
change which candidates get accepted, holding the SBERT filter and
frequency ranking fixed? Does removing the frequency term from the
ranking formula measurably change output quality, or is SBERT similarity
alone already doing most of the discriminating work? These are cheap to
test once a benchmark exists and are exactly the kind of question that's
easy to *assume* the answer to and never actually check.

---

## 12. Validation methodology — the proxy-metric trap

This is the single most important lesson in this document, because it is
the kind of mistake that survives a genuinely careful, honest, well-
documented process for a long time before anyone notices — and it is
exactly as likely to happen here as anywhere else if it isn't named
explicitly up front.

**The pattern, described generally**: a project with a two-stage
pipeline — say, stage one converts raw input into an intermediate
representation, stage two does the real analysis on that representation
— needs to evaluate stage two's quality. Building a fast, cheap
evaluation track is tempting and useful: feed stage two a *known-correct*
intermediate representation directly, skip stage one's real (slow,
imperfect) output entirely, and measure stage two in isolation. This is a
legitimate, valuable thing to measure — but it answers the question
*"how good is stage two, given a perfect input?"*, not *"how good is the
whole system, given real input?"* Those two questions can have very
different answers whenever stage one's real output is meaningfully
imperfect.

The trap: because the cheap track is so much easier to iterate against,
it becomes the default thing every subsequent experiment gets checked
against — not through a deliberate decision to stop caring about the
realistic case, but simply because it's the path of least resistance,
one experiment at a time. It is entirely possible for a team to run
several rounds of genuinely rigorous, honestly-reported work — including
its most sophisticated piece of engineering yet — validated exclusively
against the cheap track, and not notice that the expensive, realistic
track hasn't been re-checked in a long time, because nothing about any
individual experiment looked wrong. Every individual finding was true
*of the cheap track*. What went unexamined was whether progress on the
cheap track was still moving the number that actually matters. The way
this tends to surface is not through any single experiment failing, but
through a deliberate, retrospective, "if I'd never seen any of this
work, what would the highest-leverage next step actually be" review —
which is exactly why that kind of review is worth doing periodically
(§15), not just once at a project's start.

**Why this matters here specifically**: this repository has an almost
structurally identical trap available to it. SBERT cosine similarity,
Zipf frequency, the phoneme-onset gate, and the hand-picked difficulty
formula are all cheap, fast, offline-computable proxies for the two
things that actually matter — *does the rewritten text still mean what
the author meant*, and *does it actually become easier for this real
speaker to say*. Neither of those two real questions can be answered by
SBERT similarity or a formula alone; they require, respectively, human
or careful automated judgment of meaning fidelity, and — the harder one —
either real behavioral data (does this speaker actually produce the
rewritten version more fluently?) or at minimum a validated,
literature-grounded proxy for it. It is entirely possible to spend
significant engineering effort raising SBERT similarity scores or
lowering a formula-computed difficulty number without ever confirming
either one moves the real, human outcome at all.

**The concrete operating rule this produces**: whenever a change in this
repository is validated only against a fast/cheap/offline metric
(SBERT score, frequency, formula-computed difficulty), that must be
stated as an explicit, named limitation of the result — not folded
silently into "improved quality." And periodically — not just once at
the start — this repository's methodology should include a deliberate,
roadmap-blind self-audit asking exactly the question that exposes this
trap: *has anything we've shipped and validated on the cheap metric
actually been re-checked against the expensive, realistic one recently?
If not, that gap — not whatever is next on the roadmap — is probably the
highest-leverage thing to close.* See §15 for how this becomes a
recurring practice rather than a one-time realization.

---

## 13. Handling negative and null results

A negative or inconclusive result gets exactly the same documentation
rigor as a positive one — same write-up structure, same permanence, same
visibility in the roadmap. **Illustrative example**: a candidate fix is
built, benchmarked against the metric it's meant to improve, and found
to actually make it worse. The right response is to revert it — and to
record the reversion, the exact size of the regression, and a regression
test that locks in the correct (reverted) behavior, all with the same
visibility a successful change would have gotten, rather than deleting
the branch quietly and never mentioning it happened. Similarly, a
planned validation sometimes turns out to be genuinely inconclusive — a
dataset's documentation doesn't confirm what was needed, or a source
can't be verified — and the honest record is to say exactly that
("inconclusive, here's what was checked and what remains unverified"),
not to round it up to a success or down to a silent non-event.

This applies with equal force to a novel idea proposed under §4: a bold,
well-reasoned hypothesis that fails its validation is not an
embarrassment to bury — it is exactly the kind of result that keeps the
next proposal honest, and it belongs in the record with the same weight
a successful one would get.

For this repository, the discipline is the same: if a learned re-ranker
is tried and it doesn't beat the current weighted score, that is a
result to write up in full, not a branch to delete quietly. If the
phoneme-onset-only stutter model turns out — once actually tested against
real difficulty data — not to correlate with observed speaker difficulty
at all, that is one of the more valuable things this project could
learn, and it should be documented with the same weight as a successful
new feature.

---

## 14. Documentation standards and the document set

Documentation is written **continuously, at the time a decision is made**,
not reconstructed afterward from memory. A decision's reasoning is most
accurate and most complete the day it's made; every day after that, it
degrades.

**Every non-trivial decision, result, bug, or finding gets a dated,
append-only log entry** with four parts: *what was done*, *alternatives
considered*, *why this choice*, *measured result* (or "not yet measured"
if only tested, not run for real). Append-only means exactly that — a
later entry can correct an earlier one explicitly, but the earlier entry
is never edited or deleted. This is what makes the log trustworthy months
later: nothing has been quietly smoothed over in hindsight.

**Recommended document set for this repository** — names can be
whatever fits the team's convention, but each of these *roles* is worth
having filled by something:

| Role | Suggested filename | Purpose |
|---|---|---|
| Orientation / standing rules | `CLAUDE.md` or `AGENTS.md` | Short. Points to everything else; states the handful of rules that govern *how* work happens here (this document largely fills that role for the Speech AI module). |
| Primary entry point | `HANDOFF.md` | Curated reading order, what's proven vs. still hypothesis, practical "how to get productive," and — critically — a running list of pitfalls a past session hit, so the next one doesn't rediscover them. |
| File map | `DOCS.md` | One line per file: what it's for, who reads it, how often it should be updated, and an explicit warning where docs are known to drift from code. |
| Living evaluation record | `VALIDATION.md` | Pre-registered protocols, results, ablations, and — crucially — each result's own stated limitations, kept in one place so a number is never cited without its caveats nearby. |
| Forward-looking priority list | `ROADMAP.md` | One list, in priority order, each item linked to the specific finding that justifies it; items move to "done" or "explicitly rejected" with a link to the decision, never silently deleted. |
| Full decision history | `DECISION_LOG.md` | The append-only four-part record described above. This is the file a future paper's methods section gets written from. |
| Fast-scan index | `CHANGELOG.md` | One line per change, reverse-chronological, each pointing into the full decision log — for someone who needs the "what changed" answer without reading every entry's full reasoning. |

**Why a set of files rather than one big document, illustrated**: a
project starts with all of this reasoning living in one growing file.
Two problems show up quickly. First, "what changed recently" and "why
was this specific decision made six months ago" are genuinely different
questions a reader asks at different moments, and a single
chronologically-ordered document is bad at answering the fast-scan
version of the question without making the reader wade through the deep
version. Second, "what's the plan" and "what's the permanent record of
what already happened" also turn out to be different questions — a
roadmap needs to be edited and reprioritized as evidence changes, while a
decision log needs to never be edited, only appended to, or it stops
being trustworthy. Splitting into the roles above (a living roadmap that
changes, an append-only log that doesn't, a fast index that summarizes
the log, a file map that says what's where) resolves both problems at
once, at the cost of needing a short "how these relate" note (that's what
a `DOCS.md`-equivalent file is for) so a new reader isn't left guessing
which file answers which question.

---

## 15. Roadmap evolution

A roadmap is not a static plan; it is a living document whose priorities
change *because new evidence changed them*, not on a schedule. Three
practices worth adopting directly:

- **Every roadmap item cites the specific finding that justifies its
  priority.** "High priority" without a linked reason is an opinion; "high
  priority — see the finding recorded on [date] that X" is traceable.
- **A roadmap item can originate from a proposed hypothesis, not only
  from a completed finding** (§4) — as long as it's labeled as exactly
  that ("proposed direction, not yet validated — see §4") rather than
  written with the same confidence as an item a finding already
  justifies.
- **Periodically, run a deliberate, roadmap-blind reassessment** —
  literally set the existing roadmap aside and ask, from the stated
  research objective alone, what the highest-leverage next step actually
  is, then compare that answer against what the roadmap currently says.
  This is not paranoia; it is exactly the kind of check that catches the
  proxy-metric trap described in §12 before it costs months of
  compounding effort rather than one afternoon of re-examination. Do this
  on a recurring basis, not only once at the start of a project.

---

## 16. Reproducibility

Every real evaluation run should be saved, not just printed: the exact
config used, the dataset/profile version, the git commit, and a
timestamp — so a claim like "meaning preservation improved from X to Y"
can be checked against an actual artifact months later, not just trusted
because it's written down. This matters more here than it might first
appear, because this repository's evaluation inputs (speaker profiles)
are expected to change over time as the Audio Module feeds them real
data — a result measured against one profile snapshot is not
automatically valid against a later one, and the saved run record is
what makes that distinction checkable rather than assumed.

---

## 17. Repository organization

Keep evaluation/research code separate from the application's live path:
code the live app depends on should stay lean and dependency-light; code
that exists to produce a validated research result can afford to be
heavier and slower, and should be clearly marked as "not needed to run
the app itself." This is a common, effective split in research-heavy
codebases — a lean live-application directory, and a separate, heavier
evaluation/research directory that the app never imports from. When a
research result graduates into something the live app actually uses (a
trained re-ranker, a validated difficulty formula), the *artifact* moves
into the app's live path deliberately, but the *harness* that produced
and validated it stays in the research tree as the reproducibility record
for that artifact.

---

## 18. Researcher / Claude handoff

A handoff document should let a new researcher, or a Claude instance with
zero conversation history, become productive without reconstructing
anything from memory. Concretely, that means it states: what's proven
(with its evidence) vs. what's still a hypothesis; what the fast,
model-free test suite currently covers; the exact commands to reproduce
the last validated result; and — this is the part most handoff docs skip
— a running list of specific pitfalls already hit, described concretely
enough to actually prevent a repeat (e.g. "a component that eagerly loads
a heavy model in its constructor broke the fast test suite the moment
any test touched it — new model-backed components must load lazily and
be tested with the real model absent" or "a long-running data-collection
job lost all progress when it was interrupted, because it only wrote
output at the very end — long jobs must checkpoint incrementally").

---

## 19. How to use this document to review the Speech AI repository

This is the protocol this document exists to set up. **Do not skip
straight to implementation.** The order matters — each step is a
prerequisite for the next one being trustworthy:

1. **Read and understand the repository from first principles.**
   Understand its *purpose* — the research objective stated in §1 —
   before evaluating anything about how it's currently implemented.
2. **Study the existing documentation and codebase**, treating stale
   claims as a signal (per §5: docs drift — verify against the running
   code before trusting a claim in any `.md` file, this one included).
3. **Review relevant scientific literature** per §7 — lexical
   substitution, paraphrase generation, readability/simplification, and
   speech-language-pathology research on what actually predicts spoken
   difficulty.
4. **Compare the implementation against current research** — where does
   the current pipeline match established practice, and where does it
   diverge, knowingly or not?
5. **Identify assumptions** the current implementation carries — starting
   with the ones §3, §7, and §12 above already name explicitly (the
   difficulty formula's un-validated weights; the proxy-metric risk in
   SBERT/frequency-based evaluation; whatever the literature pass
   surfaces) — **and propose new ones freely** where reasoning suggests a
   better approach exists, per §4.
6. **Determine which assumptions require validation**, and which are
   either already well-supported or genuinely low-stakes enough not to
   need it yet.
7. **Document what currently exists** — a clear-eyed account of the
   present state, using the fact/observation/hypothesis/decision/
   limitation/future-work vocabulary from §5, not a narrative that
   already assumes a direction.
8. **Build a structured roadmap driven by evidence rather than
   preference** (§15), each item linked to the specific finding or gap
   that justifies it, including proposed-but-unvalidated directions
   labeled as such.
9. **Only then** — after all of the above — **begin validation,
   experimentation, implementation, and benchmarking where the roadmap
   says it's justified.**

This same protocol should be repeated whenever the module goes through a
comparable inflection point in the future (a new phase closing, a major
architectural question arising) — it is a recurring discipline, not a
one-time onboarding step.

---

## 20. This document's own governance

This document is itself subject to the principle stated in §3: it earns
its authority through evidence, not through having been written first.
If research on this repository — literature review, experiments,
real-world validation — produces a finding that a principle stated here
should change, it should change. The requirement is the same one this
document asks of every other decision in this repository: record what
changed, why, and what evidence justified it, as a dated addendum below,
rather than silently rewriting a prior version of this document. Treat
disagreement with this document, backed by evidence, as the discipline
working as intended, not as a deviation from it. This includes proposing
entirely new principles this version doesn't anticipate (§4) — this
document does not claim to be complete, only to be the current
best-supported statement of the methodology.

### Amendment log

*(Append dated entries here as this document's own principles are
revised. Do not edit or remove a past entry — correct it explicitly in a
new one, per the same append-only discipline this document asks of the
decision-log records described in §14.)*

- **2026-08-05** — Initial version, establishing this methodology before
  research begins on this repository. The principles here are distilled
  from general evidence-driven engineering and research practice, not
  from a specific external project's history — no prior version of this
  document existed for the Speech AI module to amend.
- **2026-08-05** — Added §4, "Creative, evidence-seeking engineering,"
  clarifying that §3's evidence-constrained standard governs what ships,
  not what may be proposed. Original ideas, alternative architectures,
  and approaches without direct literature or prior-evidence backing are
  explicitly welcomed, provided they are labeled as hypotheses or
  engineering judgments (not facts), given a concise rationale, and
  validated through appropriate experiments before becoming part of the
  system. Subsequent sections renumbered (old §4-19 became §5-20) and
  cross-references updated throughout to keep pointers accurate.
