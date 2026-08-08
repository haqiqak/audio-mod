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

## PROJECT OBJECTIVE — read this before anything else in this document (added 2026-08-07, governs everything above and below it)

**Every section below this one, including all research, literature
review, and novelty analysis, exists to serve the objective stated
here — none of it is the objective.** This section was added after the
project owner explicitly flagged a real risk: that a long research
track chasing "is this genuinely unexplored" can quietly drift into
treating novelty itself as the destination. It hadn't yet — nothing
below was retracted — but the risk was real enough to warrant making
the actual goal impossible to miss.

**The objective, stated precisely**: build an application that takes a
speaker's audio and reliably extracts the actual speech problems and
hindrances occurring in that speech — stuttering and other disfluencies
— so that these patterns can be **localized, classified, understood,
and used downstream** to help the speaker (e.g., replacing a disfluent
span with a fluent alternative or synonym phrasing, once that
downstream capability exists). This project (`audio-mod`) owns the
*extraction* half — detection, classification, localization from audio.
A separate module (this project's own "Speech AI Module," per the
broader product this project is one half of) is responsible for acting
on that extraction. This document's job is to make audio-mod's
extraction as reliable and complete as it can be — nothing more, and
nothing less.

**What this means concretely, and why it changes how the ASR question
should be read**: an ASR system, a fine-tuned representation, an
acoustic detector, or any other technical component discussed in this
document is a **means**, evaluated only by whether it gets this project
closer to reliable extraction. "Is this a novel research finding" is a
question worth answering honestly wherever it comes up (this project
documents its work rigorously and does not want to overclaim or
underclaim what it found) — but it is never the criterion for choosing
a technical direction. The criterion is always: *does this get the
application the information it needs, reliably, with the resources this
project actually has?*

**The single most consequential reframing this produces, stated
directly**: much of this track's later research (Phase 2, direction
(g), the combined-signal classifier, and especially the L1-L4 novelty
analysis) implicitly treated **true verbatim fragment-content
preservation** (an ASR that literally outputs "c-c-cat") as the target.
Re-read against the actual objective above, this was likely the wrong
target. The application needs to know **that** a disfluency occurred,
**where** (precise timing), and **what type** it is — it does not
obviously need the ASR to spell out the disfluent fragment's own
phonetic content as text, because the *intended* word (the clean
replacement target) is typically already present in the transcript
immediately after the disfluency, whether or not the disfluency itself
was preserved. This distinction — extraction/detection vs. verbatim
fragment transcription — is the organizing idea behind the full
decision analysis at the end of this document
("Application-Objective Decision Analysis — 2026-08-07"), which
should be read as the current, governing interpretation of everything
above it, superseding (not deleting) the novelty-centered framing of
the sections that follow immediately below.

---

## First-principles reassessment (2026-08-06): are we still on the right trajectory, or attached to CrisperWhisper?

**What this section is.** After Stage C2, the project owner asked a
direct question this track had not yet asked of itself explicitly: are
we still pursuing the objective (the best possible speech-analysis
system for disfluency-aware rewriting, regardless of what architecture
that requires), or have we quietly become attached to CrisperWhisper as
the answer rather than one hypothesis among several? This section
answers that, structured the way it was asked: evidence, then inference,
then judgment, kept separate and labeled, ending in a recommendation —
not a survey of options.

### The evidence (directly measured, not interpreted)

- Phase 1: detector recall given a perfect transcript is ~99%; given real
  ASR, ~6-15%. At full speaker diversity, 35.1% of that gap is
  detector-attributable, 64.9% ASR-attributable.
- Item 19 (this session): the shipped repetition classifier's real-world
  impact on real ASR is negligible, because real ASR produces almost no
  candidates for it to act on — `sound_repetition`: zero, in the 120-clip
  sample, either gate setting.
- Stage A: ~53% of `sound_repetition`/`word_repetition` losses happen at
  positions ASR transcribed *correctly* (not a transcription-accuracy
  problem) — two distinct mechanisms, fragment loss and pair-breaking.
  The remaining ~45% of losses are ordinary transcription error.
- Stage B: CrisperWhisper's own last-layer encoder retains a real signal
  at those same "text says nothing" positions (Cohen's d=0.894), and that
  signal is not explained by token duration (the named confound was
  tested directly and refuted, Stage C: duration-arm AUC=0.483, chance).
- Stage C: that same signal's absolute precision at a useful recall is
  low (4.7% at 52.6% recall, realistic ~19:966 class imbalance) — real,
  but not sufficient alone.
- Stage C2: two independent additional signals tried on the same
  population — token duration (Stage C) and five Praat voice-quality
  features (jitter, shimmer, pitch stability, HNR) — both returned
  results indistinguishable from chance. Neither adds anything beyond
  what the encoder-distance signal alone already provides.
- **Not yet measured, anywhere in this project**: whether a different
  encoder layer (only the last layer has ever been used), a different
  decoding configuration (no decoding parameter has ever been varied),
  or a different ASR backend (item 10/RQ3) changes any of the above.

### Inference (reasoned from the evidence, not directly measured)

- The information loss for `sound_repetition`/`word_repetition` is most
  consistent with a **decoding-stage** phenomenon, not an encoder-stage
  one: the encoder (Stage B/C) still carries the signal at exactly the
  positions where the decoded text does not. This is an inference, not a
  direct observation — no experiment here has directly opened the decoder
  and shown fluency-bias in action; it is the most consistent explanation
  of two measured facts (text loses it, encoder doesn't) taken together.
- CrisperWhisper's own documented design (retokenization + attention-based
  timestamp alignment, aimed at word-level verbatim accuracy and
  timestamp precision) plausibly was never optimized to preserve
  sub-word fragments as literal output tokens specifically — consistent
  with, not proven by, Stage A's finding. Nothing in this project has
  read CrisperWhisper's training objective in enough depth to state this
  as fact rather than a reasonable inference from its documented behavior.
- Two failed cheap-fusion attempts (Stage C2) suggest, but do not prove,
  that the "bolt an existing, already-available signal onto the current
  encoder-distance measurement, at this sample size, with no training"
  space is running dry. n=2 attempts is a real but modest basis for this
  inference — it says more about *this specific kind* of cheap addition
  than about the encoder-representation direction as a whole.

### Engineering judgment (my own call, explicitly labeled as such — not fact, not proven inference)

**We are not yet at the point where evidence justifies moving to a
different or purpose-built ASR, and doing so now would repeat the exact
mistake this reassessment is being asked to guard against — just in the
opposite direction.** The evidence above supports "CrisperWhisper's
*decoded text* is an insufficient representation for two disfluency
types" — a real, well-supported conclusion. It does **not** support "a
different ASR would do better," because that has never been tested
(RQ3, item 10 — still fully open). Jumping to a new architecture on the
strength of two negative fusion results and zero cross-backend evidence
would be exploration driven by frustration with diminishing returns, not
by evidence — precisely what standing rule 8 and this track's own charter
exist to prevent, symmetrically in both directions (not preserving
CrisperWhisper by default, and not abandoning it by default either).

**Is the heuristic space "sufficiently explored"? Partially, and it
matters which part.** The narrow slice actually tested so far —
*existing infrastructure, one encoder layer, threshold-based (not
trained) combination, current small sample* — has returned diminishing
and now negative results (Stage C2) and is reasonably considered explored
at this scope. But that is a much narrower claim than "the richer-
representation direction is exhausted." Three concrete, cheap,
well-motivated experiments **within the current architecture** have never
been run:

1. **Encoder layer depth** — every measurement in this track (Stage 1,
   B, C) used only CrisperWhisper's default last-hidden-state. Literature
   already reviewed in this document (arXiv:2311.05203) found *deeper*
   layers carry *more* disfluency-relevant signal for a comparable task —
   a directly applicable, never-tested hypothesis, using infrastructure
   already built (`profiling/encoder_embedding.py` + Stage B/C's own
   pipeline, extended to sweep layers instead of hardcoding one).
2. **Decoding-parameter sensitivity** — no decoding setting (beam width,
   suppression tokens, the implicit LM/fluency bias) has ever been
   varied. This is the most direct available test of the inference above
   (that the loss is decoder-side, not encoder-side) — if adjusting
   decoding parameters measurably changes whether fragments survive to
   text, that's a real, actionable lever within the current ASR choice;
   if it doesn't, that's evidence *for* needing a training-level change
   (Stage D) rather than a configuration-level one.
3. **RQ3, a second ASR backend** — the one experiment that would actually
   tell us whether any of this is CrisperWhisper-specific or general to
   ASR encoders, which is the direct evidence a "should we consider a
   different ASR" conclusion would need and does not yet have.

**Recommendation, stated plainly**: stay within the current architecture
for the next concrete step — extending, not abandoning it — because real,
cheap, well-motivated headroom remains untested there, and the evidence
for leaving it is not yet evidence, only a plausible inference from two
negative side-experiments. Do (1) and (2) next, in that order (both
touch the actual ASR/encoder mechanism directly, both reuse proven
infrastructure, neither needs new data or training); treat their results
as the evidence that would finally justify — or further rule out — (3)
and, eventually, Stage D. This is not a defense of CrisperWhisper for its
own sake — if (1) and (2) also come back flat, that combined with Stage
C2's negatives would be a real, mounting case for RQ3 and Stage D, and
this document should say so plainly when that evidence exists. It
doesn't yet.

### What changes as a result

`ROADMAP.md`'s pointer to this track is updated to reflect this
conclusion. §8's original Stage D framing (evidence threshold: "the loss
is not recoverable from existing representations... direction (b/f) has
been tried and genuinely isn't enough, not merely untried") is now sharper: as of Stage C2, direction (b) has been tried at exactly one
layer, one pooling method, no decoding variation, and one modest sample —
genuinely tried, but not yet tried *enough* to call it insufficient. The
two new experiments above are added as the immediate next steps, ahead
of RQ3 and Stage D, in the plan below.

**Update, 2026-08-06, later the same day — the first of the two
experiments is done.** The encoder layer-depth sweep (below) found the
last layer is uniquely informative — no other layer comes close, and
several sit mildly below chance. This closes that specific question
rather than leaving it open, and per this section's own stated logic
("if that also comes back flat, toward RQ3/Stage D" — referring to
*both* experiments, not either alone), **the honest next step is still
the decoding-parameter experiment, not yet RQ3 or Stage D**: one of the
two recommended in-architecture levers has now been tried and found not
to help; the other has not been tried at all. Only after both return
results should this document's weight shift toward RQ3/Stage D.

### Encoder layer-depth sweep — pre-registered protocol (2026-08-06, written before running)

**Hypothesis under test**: does a layer other than CrisperWhisper's
default last encoder layer carry a stronger `sound_repetition` signal at
Stage A's category-1 ("normalized away") positions — matching arXiv:
2311.05203's finding that deeper layers carry more disfluency-relevant
information for a comparable Whisper-encoder classification task?

**Design**: re-run the encoder pass on the identical 31 clips / 19
target / 966 control positions Stage B and C already established, but
with `output_hidden_states=True` — **one forward pass per clip, not one
per layer**, since `output_hidden_states=True` returns every
intermediate layer's activations from a single call. For each layer
(CrisperWhisper's encoder has the same depth as whisper-large-v3's: 32
transformer layers, so 33 hidden-state tensors including the embedding
output), compute the exact same metric Stage B/C already validated:
mean-pooled span embedding, cosine distance to a per-clip leave-one-out
fluent centroid, then AUC against the target/control labels (matching
Stage C's metric directly, so every layer is comparable to the
already-known last-layer result of 0.723 on the same population).

**Cost, scoped before running**: the same 31 clips already used for
Stage B/C, at approximately the same per-clip cost Stage B measured
(~33s/clip, ~17 minutes total) — `output_hidden_states=True` returns
already-computed intermediate activations from the same forward pass,
not additional passes, so this does not multiply Stage B's original cost
by the number of layers.

**Success criteria, fixed in advance**:
- **A deeper (or other) layer meaningfully beats the last layer**: AUC
  notably above 0.723 (not a difference plausibly explained by noise at
  n=19) — a real, actionable finding: that layer becomes the new signal
  source for any future stage in this track, and this itself is evidence
  the representation direction had more headroom than Stage C alone
  showed.
- **No layer meaningfully beats the last layer**: the last layer was
  already close to the best available in this network — narrows this
  specific lever further, and shifts weight toward the decoding-
  parameter experiment (also proposed above) and, if that also comes
  back flat, toward RQ3/Stage D.
- **Some layers are notably worse**: expected and informative on its own
  — helps map where in the network this signal emerges, useful context
  even if it doesn't change which layer to use going forward.

**Named limitations, stated before running**:
- Same in-sample, small-n (19 target positions) caveat every stage in
  this track has carried since Stage B — screening 33 layers at this
  sample size raises the same overfitting-to-noise risk Stage C2's
  multi-feature screening named, mitigated the same way: report every
  layer's AUC, not just the best one, so a single outlier layer doesn't
  get treated as a confirmed finding without acknowledging how many
  comparisons were made.
- `sound_repetition` only (matching Stage C's scope) — `word_repetition`
  remains a separate, still-open question per Stage B's own result.
- Does not test decoding-parameter sensitivity (the second recommended
  experiment) — a separate, later step.

### Encoder layer-depth sweep — Results (2026-08-06): the last layer is uniquely informative — no other layer comes close

**A real deviation from the pre-registered population, caught and
verified before trusting the comparison, recorded here rather than
silently absorbed.** The pre-registration said "the identical 31 clips /
966 control positions Stage B and C used." The implementation instead
filtered to clips containing a `sound_repetition` target specifically
(18 clips, 551 control positions) — Stage B/C's 31-clip, 966-control
pool was built from clips with *either* `sound_repetition` or `word_
repetition` targets, pooled together; this run only drew from the
`sound_repetition`-relevant subset. **Verified this doesn't change the
answer before trusting anything further**: the last layer's AUC on this
smaller, different population (0.721, n=551 control) came out almost
identical to Stage C's own last-layer AUC on the larger population
(0.723, n=966 control) — an unplanned but genuine partial-replication
check, not just a coincidence to wave away, and it passed.

**Cost, as it actually ran**: 689s (~11.5 min) for 18 clips x 33 layers
each — one forward pass per clip, all layers extracted simultaneously,
confirming the pre-registration's cost estimate (extracting every layer
does not multiply the per-clip cost by the number of layers).

**Results — AUC at every layer, `sound_repetition`, n=19 target / n=551
control:**

| Layer range | AUC range | Pattern |
|---|---|---|
| 0 (embedding output) | 0.544 | Near chance |
| 1-23 | 0.338-0.365 | Consistently *below* chance |
| 24-31 | 0.360-0.383 | Still below chance, mild upward drift |
| **32 (last, = Stage B/C's layer)** | **0.721** | The only layer with real signal |

Full per-layer table in the saved run
(`eval_results/20260806T144057_stage_layer_sweep.json`).

**Against the pre-registered success criteria**: **no layer meaningfully
beats the last layer — the last layer *is* the best layer**, and by a
wide margin (the runner-up, layer 26, reaches only 0.383). This answers
the pre-registered question decisively: Stage B/C's choice to use the
default last-hidden-state was already the right one; there was no
untapped signal sitting in an earlier layer.

**A secondary, unplanned observation, stated as exactly that — an
observation, not a tested hypothesis**: layers 1-31 don't merely fail to
show a positive-direction signal, most sit *below* 0.5 (as low as 0.338),
meaning `sound_repetition` target positions are, at those layers,
mildly *closer* to the fluent centroid than control positions are — the
opposite direction from "anomalous." Layer 0 sits near chance (0.544),
and there's a gradual drift upward through layers 24-31 before the sharp
jump at layer 32. This is consistent with a coherent story — early/mid
encoder layers represent lower-level acoustic detail without much
task-specific structure yet, and the disfluency-relevant signal
crystallizes sharply only in the final layer, immediately upstream of
the decoder — but this project has not independently verified that
explanation (no attention analysis, no probing beyond this single
AUC sweep); it is offered as a plausible reading, not a finding.

**What this resolves**: encoder layer depth is **not** an untapped
lever — the reassessment's first recommended experiment came back with
a clean, decisive answer, closing that specific question rather than
leaving it open. This narrows, not widens, the remaining options: the
decoding-parameter experiment (the reassessment's second recommendation)
is now the last untested lever within the current architecture before
RQ3 (a second ASR backend) or Stage D become the honest next move on
richer-representation grounds specifically.

### Decoding-parameter sensitivity (`num_beams`) — pre-registered protocol (2026-08-06, written before running)

**A scoping fact, checked before designing anything**: `profiling/
asr.py` forces `num_beams=1` in the live app — not a free architectural
choice, but a confirmed workaround for a real `transformers` bug
(`WhisperGenerationMixin._extract_token_timestamps()` mis-shapes
`beam_indices` when word-level timestamps are requested together with
beam search; huggingface/transformers #28007/#36093). CrisperWhisper's
own `generation_config.json` default is `num_beams=5`; the live app
overrides it to 1 purely to avoid this crash. This means the live app is
already running the *least* beam-search-influenced decoding the model
supports — informative in itself (§ below), and it means testing
`num_beams=5` **cannot use the same word-timestamp extraction path**
without hitting the same confirmed crash.

**Design consequence**: this experiment does not need word-level
timestamps at all — only the decoded *text*, to test whether it contains
the literal fragment (`sound_repetition`) or the intact repeated pair
(`word_repetition`) at each of the 36 positions Stage A already
identified as lost under the live app's current `num_beams=1` decoding.
`align()` (used throughout this track, including Stage A itself)
operates on word sequences alone — no timestamps required. So: call
`model.generate()` directly (bypassing `pipeline()` and `return_
timestamps="word"` entirely, sidestepping the crash rather than hitting
it), decode to plain text, split to words, and run the identical
alignment-based check Stage A used.

**Hypothesis under test**: does `num_beams=5` (the model's own trained
default) recover any of the 36 positions lost under the live app's
`num_beams=1`? Directly tests this track's own inference (§ above) that
beam search's cross-hypothesis fluency bias is a plausible contributor
to the normalization loss Stage A found — the app currently runs the
setting that inference would predict is *least* affected by that
specific mechanism, so a null result here would be informative too (it
would mean the loss survives even at the most literal decoding setting
available, pointing away from beam search specifically as the
mechanism).

**Population**: the same 31 clips / 36 target positions (19 `sound_
repetition` + 17 `word_repetition`) every stage since Stage B has used.

**Metric**: count of the 36 positions "recovered" (fragment present /
pair intact) under `num_beams=5` vs. the known 0/36 baseline (`num_
beams=1`, already established — these positions are in the target set
*because* they were lost at baseline). Also: whole-clip word error rate
under each condition, as a broad side-effect check — a decoding change
that recovers disfluency evidence at the cost of meaningfully worse
general transcription accuracy would be a real, reportable trade-off,
not a clean win.

**Success criteria, fixed in advance**:
- **Beam width matters**: a non-trivial fraction of the 36 positions
  (not just 1-2, which could be noise at this n) are recovered under
  `num_beams=5` — real evidence the current forced setting is costing
  disfluency evidence, motivating the separately-scoped timestamp-bug
  fix (`ARCHITECTURE.md`'s known-limitations section already names this
  as unresolved) as newly higher-value than previously assessed.
- **Beam width doesn't matter**: recovery count stays near 0/36 —
  real evidence the loss survives even the most literal decoding
  setting available, meaning no simple decoding-parameter change is
  going to fix this, and localizes the mechanism more specifically to
  the model's learned behavior rather than the search strategy.

**Named limitations, stated before running**:
- `num_beams=5` is **not deployable as-is** in the live app (the
  timestamp-extraction crash is real and unrelated to this experiment) —
  a positive result here identifies a real trade-off worth investigating
  further, not an immediately shippable fix.
- Real, new ASR cost: unlike every other experiment in this track since
  Stage B, this one cannot reuse cached transcription — decoding
  parameters only take effect during generation itself. 31 clips at this
  project's measured ~54-102s/clip range (possibly slower with 5 beams
  than the cached 1-beam runs — timed and reported, not assumed) is the
  real, scoped cost.
- Tests only `num_beams`, not other decoding knobs (`repetition_penalty`,
  suppression settings) — checked directly against the model's actual
  `generation_config` before scoping this experiment: `no_repeat_ngram_
  size=0` and `repetition_penalty=1.0` are already neutral (not actively
  suppressing anything) in the current default, so they are not
  candidate explanations for the *current* observed loss the way beam
  width plausibly is; not tested further here.

**Scoping addendum (2026-08-06, before the full run, once real per-clip
cost was known) — the pre-registered 31-clip cost estimate was
undersold, same pattern as §12.6.1's own precedent.** A 4-clip dry run
(needed anyway, to catch bugs before the real run — and it did: see
below) measured **~316s/clip for both conditions combined** (num_beams=5
is inherently ~5x the compute of num_beams=1, not simply additive with
it) — far above the ~54-102s/clip single-condition range this project's
other benchmarks measured. At that rate, the full 31-clip population
would cost **~2.7 hours**, not the ~55-105 minutes estimated before any
real timing existed. Per the same discipline §12.6.1 applied when this
happened before (scope down, state the reason, commit to a bound *before*
seeing more results, not after): **scoped to the first 40 raw clips
scanned** (not 31) — expected, not guaranteed, to yield a comparable
target-position count to Stage B/C's original population, at a bounded,
known cost (~65-85 min) decided before running, not adjusted afterward
based on how the results looked.

**A real bug caught by the dry run itself, before any full-scale
result was trusted**: the first version of the `sound_repetition`
"recovered" check flagged `"a"` as a recovered fragment before
`"apple"` purely because `"apple".startswith("a")` — a real false-
positive class (any single-letter word trivially prefix-matches many
following words by coincidence, unrelated to genuine fragment
preservation). Caught by a hand-constructed unit test run before any
real audio was processed, not by inspecting a real result after the
fact. Fixed with a minimum fragment length of 2 characters.

### Decoding-parameter sensitivity — Results (2026-08-06): a clean, decisive negative — `num_beams` is not the mechanism

**Cost, as it actually ran**: 40 raw clips scanned yielded 12 clips
containing a `sound_repetition`/`word_repetition` target (14 positions:
6 `sound_repetition`, 8 `word_repetition`) — smaller than Stage B/C's
31-clip/36-position population (expected, since this run scanned fewer
raw clips, per the scoping addendum above), but real, freshly-generated
data under both conditions. 2926s (~49 min) total for 12 clips x 2
conditions (~244s/clip combined) — close to the dry run's own ~316s/clip
estimate, confirming that number wasn't a fluke.

**Result**: **0 of 14 target positions recovered** under `num_beams=5` —
not one of the 6 `sound_repetition` fragments or 8 `word_repetition`
pairs that were lost under the live app's `num_beams=1` reappeared under
the model's own trained-default beam width. Mean word error rate was
**identical** between conditions (0.187 vs. 0.187) — beam width didn't
even measurably change general transcription accuracy on this sample,
let alone recover disfluency-specific evidence.

**Against the pre-registered success criteria**: **beam width does not
matter** — the "doesn't matter" branch, cleanly. This is not a weak or
ambiguous null (0/14, not e.g. 2/14 with wide uncertainty) — combined
with the WER identity, it reads as a genuine, decisive absence of effect
at this sample size, not a small effect this experiment was too
underpowered to see.

**What this resolves**: this directly sharpens the track's own
decoder-stage inference (§ above, "First-principles reassessment"). The
live app already runs `num_beams=1` — the beam-search-influenced
mechanism this experiment tested for — and yet the loss is identical
whether beam search is even in play or not. This is real evidence
*against* "beam search's cross-hypothesis fluency bias" as the
mechanism, specifically, not just an untested gap closing. The loss
survives at the most literal decoding setting the model supports,
pointing more specifically toward the model's own learned token-level
preferences (independent of search strategy) as the likely mechanism —
consistent with, and now more evidence-backed than, the original
"decoder-stage, not encoder-stage" inference, but narrower: it is
apparently not *decoding strategy* specifically, it is something in what
the model was trained to predict, regardless of how thoroughly that
prediction is searched.

**With this result, both of the reassessment's recommended in-
architecture experiments are now done, both negative for their specific
mechanism**: encoder layer depth (no layer beats the last one) and
decoding-parameter sensitivity (beam width doesn't matter). Per the
reassessment's own stated logic, this is the trigger condition for the
full integrative reassessment that follows.

---

## Integrative first-principles reassessment (2026-08-06, after both in-architecture experiments): have we exhausted this architecture?

**What this section is.** The first reassessment (above) was written
before the two experiments it recommended had run. This one integrates
everything this track has produced — Track B, Stages A/B/C, Praat/C2,
literature, layer-depth, decoding — and answers one explicit question,
as the project owner asked: *have we now extracted essentially all
meaningful evidence from the current architecture, or is there still a
justified reason to continue researching within it?* The rule stated
alongside the request is the standard this section is held to: **we do
not abandon an architecture because it disappoints us, and we do not
preserve an architecture because we have invested in it — we move on
only when the remaining uncertainty within the current architecture is
smaller than the uncertainty outside it.**

### Complete evidence inventory (every directly measured fact this track has produced)

1. Phase 1 (pre-track): Track A recall ~99% (perfect transcript), Track B
   ~6-15% (real ASR). At full speaker diversity: 35.1% detector-
   attributable, 64.9% ASR-attributable.
2. Item 19: the shipped repetition classifier's mechanism transfers
   safely to real ASR, but real-world impact is negligible — real ASR
   produced **zero** `sound_repetition` candidates across 120 clips,
   either gate setting.
3. Stage A: ~53% of `sound_repetition`/`word_repetition` losses occur at
   positions ASR transcribed *correctly* — two distinct mechanisms
   (fragment loss; pair-breaking, 22/23 hand-checked cases). ~45% of
   losses remain ordinary transcription error, unrelated to this track's
   question.
4. Stage B: CrisperWhisper's last-layer encoder retains a real signal at
   exactly those "text says nothing" positions (Cohen's d=0.894 for
   `sound_repetition`; d=0.428, inconclusive, for `word_repetition`).
5. Stage C: that signal's duration confound is refuted (duration-only
   AUC=0.483, chance); the signal itself reaches AUC=0.723 but only 4.7%
   precision at 52.6% recall — genuine, not sufficient alone.
6. Stage C2: five Praat voice-quality features (pitch, pitch stability,
   jitter, shimmer, HNR) all near chance (0.452-0.549) — ruled out as a
   fusion signal.
7. Encoder layer-depth sweep: **all 33 layers tested.** Only the last
   layer carries real signal (AUC 0.721-0.723, verified consistent
   across two different control populations); every other layer sits at
   or below chance (0.338-0.383). No untapped depth.
8. Decoding-parameter sensitivity: `num_beams=5` (the model's own
   trained default) recovers **0 of 14** tested positions lost under the
   live app's forced `num_beams=1`; WER identical between conditions
   (0.187 = 0.187).
9. Literature (13 verified sources): real field-level ASR bias against
   disfluent speech is documented independently (arXiv:2405.06150).
   CrisperWhisper's own verbatim behavior comes from retokenization +
   attention-based timestamp alignment (arXiv:2408.16589), not
   necessarily fragment preservation as a design goal. **A real,
   unresolved discrepancy with this track's own result, stated plainly
   rather than smoothed over**: arXiv:2311.05203 found *deeper* Whisper-
   encoder layers carry *more* disfluency signal for a comparable
   classification task — this track's own layer sweep found the exact
   opposite for this specific task on this specific model (only the
   *last* layer carries signal). Not reconciled — plausible explanations
   (different task framing, different base model/scale, CrisperWhisper's
   own fine-tuning reorganizing its representations) are inferences
   below, not established.
10. RQ3 (a second ASR backend) has **never been run** — zero evidence, in
    either direction, on whether any of items 2-8 is CrisperWhisper-
    specific or general to ASR systems.
11. Implementation experience: every single stage in this track caught
    at least one real bug via a pre-flight check, dry run, or unit test
    before trusting its result (Stage A's categorization bug, Stage B's
    `audio_bytes=None` bug, Stage C's population-mismatch and missing-
    timestamp bugs, the layer-sweep's population deviation, the decoding
    experiment's false-positive heuristic bug) — none slipped through
    to a trusted conclusion. The methodology and infrastructure built
    (`profiling/encoder_embedding.py`, the alignment-based categorization
    approach, the AUC/precision-recall tooling) is now mature and,
    notably, **not CrisperWhisper-specific in its design** — it could be
    pointed at a different ASR's encoder with modest changes, meaning
    this investment carries forward regardless of what comes next.

### Inference (reasoned from the evidence above, not directly measured)

- The information-loss mechanism is most consistent with something in
  CrisperWhisper's **learned behavior** — what the model was trained to
  predict — rather than in *how* that prediction is searched or *which*
  layer's representation is consulted. This inference is now
  meaningfully stronger than in the first reassessment: two independent,
  cheap, targeted tests (layer depth, beam width) both came back
  negative for their specific mechanism, while the underlying encoder
  signal (Stage C) remains real. Two negative results narrowing toward
  the same conclusion is more informative than either alone, though it
  remains an inference — no experiment here has intervened on the
  model's actual training.
- The unreconciled literature discrepancy (item 9) is itself weak,
  suggestive evidence that CrisperWhisper's specific fine-tuning may
  behave differently from the base Whisper models most published
  disfluency-classification work uses — which argues for treating this
  track's findings as possibly CrisperWhisper-specific rather than
  general ASR-encoder behavior, strengthening the case for RQ3
  specifically (not just as a generic "nice to have" generalization
  check, but as evidence-motivated by a real, observed anomaly).
- The "free," inference-time investigation of CrisperWhisper's own
  representations and decoding behavior is very likely close to its own
  ceiling — every cheap, non-training lever this track could identify
  (which layer, which pooling comparison, which decoding width, which
  second signal to fuse) has been tried. This does not mean the
  *architecture* is exhausted — it means the *cheap, representation-only
  investigation of this one architecture* is close to exhausted.

### Judgment (this document's own call, explicitly labeled — not fact, not proven inference)

**Answering the question asked, directly: yes — we have essentially
extracted the meaningful evidence obtainable from CrisperWhisper's
existing representations and decoding behavior, at the current sample
size, without training anything.** Every lever available at that
scope — encoder layer choice, decoding configuration, duration, Praat
acoustic features, the mis-routing lead — has been tried. Two came back
genuinely positive-but-limited (the encoder-distance signal itself);
five came back negative or inconclusive. That is a coherent, closed
picture for *this specific slice* of investigation, not a scattered set
of unrelated null results.

**This is not the same as "abandon CrisperWhisper" or "start building a
different ASR."** Per the standard stated at the top of this section: the
remaining uncertainty within the current architecture is genuinely small
now (the cheap levers are tried; what's left — fine-tuning — is a
qualitatively different, much larger commitment, not another cheap
experiment). But the remaining uncertainty *outside* the current
architecture is not yet small either — RQ3 has produced *zero* evidence
either way. Recommending a move to a new ASR now would be exactly as
evidence-free as the concern the project owner opened this reassessment
with. The honest reading is that this track has reached a genuine
decision point, not a call to keep digging in the same place *or* a
license to jump elsewhere on faith.

### Remaining work — the recommended path, in order, none skipped

1. **RQ3 — a second ASR backend, immediately, before anything else.**
   Not because it is glamorous, but because it is the cheapest possible
   step that converts the current genuine uncertainty (is this
   CrisperWhisper-specific, or how ASR systems generally behave?) into
   evidence. No training required — run a second pretrained ASR (e.g.
   stock `whisper-large-v3`, already named as an option in `ROADMAP.md`
   item 10) through the same Track B alignment pipeline this track
   already has, on the same audio already downloaded, and re-run this
   track's own Stage A categorization against its output. **This is the
   literal bridge between "more of this track" and "the next research
   phase"** — its result directly answers whether continuing to invest in
   CrisperWhisper specifically (fine-tuning, Stage D) or broadening to a
   different pretrained model's representation is the evidence-motivated
   next move, rather than guessing at either.
2. **Only after RQ3 has a result**: formally cost out Stage D (§9's
   three-part test) — infrastructure (GPU, a paired transcript+
   disfluency-labeled dataset at real volume) is the one condition of
   that test genuinely unaddressed by anything in this track; pricing it
   out honestly (what exists, what would need acquiring, realistic
   timeline) is itself real, valuable next work, whether or not
   fine-tuning is ultimately pursued.
3. **Not recommended now**: expanding the current sample size for more
   of the same cheap experiments (layer depth, decoding, fusion) at
   higher n. The *direction* of every result in this track is already
   clear enough (not marginal, not contradictory) that more data would
   sharpen confidence intervals, not change conclusions — a lower-value
   use of effort than RQ3 or the Stage D cost assessment.
4. **Not recommended now**: committing to a purpose-built ASR or a
   specific alternative pretrained model. Nothing measured so far
   identifies which alternative (if any) would actually do better —
   that is exactly what RQ3 exists to start answering.

### What changes as a result

`ROADMAP.md`'s pointer to this track is updated with this section's
conclusion: item 10 (second ASR backend) is elevated from "open
question" to **the explicit next step for this track**, ahead of
everything else, including the fusion-style Stage-C revision floated
earlier (superseded by this more complete picture — that revision's own
motivation, a starved signal population, would not be resolved by more
fusion attempts at the current sample size, only by more data or a
different representation source, which RQ3 and Stage D address more
directly).

---

## Phase 2 of this research track: comprehensive design-space investigation (opened 2026-08-06)

**Governing statement, per the project owner's explicit framing:** *we
are no longer optimizing for the best research process — we are
optimizing for the best product through rigorous research.* Everything
below exists to answer one question with evidence: what speech
representation actually gives this application the best foundation for
detecting, classifying, and localizing disfluencies — not to defend
CrisperWhisper, and not to replace it for novelty's sake. Product and
paper advance together; this phase's deliverable is a decision the
product can act on, backed by a record the paper can cite.

**Explicit instruction honored**: comparing pretrained ASRs (item 10)
was elevated as *the strongest current hypothesis*, not assumed as the
correct next step. This section re-opens the full design space from
first principles before committing to that or any other direction.

### The complete design space, considered with an open mind

Every realistic direction identified, none pre-selected:

| Direction | What it means concretely | Status before this phase |
|---|---|---|
| (a) A different pretrained ASR (full system) | Swap CrisperWhisper for another ASR's transcription output, run through Track B | item 10 — proposed, never run |
| (b) A different pretrained *representation* (encoder only, not necessarily an ASR at all) | Extract encoder states from a model not fine-tuned for verbatim ASR — stock Whisper, or a genuinely self-supervised model (WavLM, HuBERT) | Named in §6(b)/§5's literature review, never tested |
| (c) Hybrid architecture | Keep CrisperWhisper for transcription; source the *corroboration signal* from a different model | This project already does a version of this informally (encoder-distance from CrisperWhisper's own encoder); never tried a different signal source |
| (d) Decoder/decoding modifications | Change *how* CrisperWhisper decodes, not what model produces the representation | `num_beams` tested (Stage-decoding, negative); other levers (logit/entropy inspection) unexplored |
| (e) Fine-tuning / continual adaptation | Adapt CrisperWhisper's (or another model's) weights toward this project's taxonomy | Stage D, gated on infrastructure this project doesn't have (§9) |
| (f) Purpose-built disfluency-preserving ASR/representation | Train something new, from scratch or via heavy adaptation, specifically for this objective | The track's own eventual "if everything else fails" tier (§9) |
| (g) Reduce reliance on ASR text further (extend the acoustic-native precedent) | More of what `block`/`prolongation` already do — detect directly from audio, no token candidate required | Partially explored (the superseded fusion-style revision); still a live, cheap option |

### Research questions for this phase

- **RQ-A** (extends item 10's original framing): does a different
  pretrained ASR's **decoded text** preserve `sound_repetition`/`word_
  repetition` evidence better than CrisperWhisper's, run through the
  same Track B alignment pipeline this track already has? Answers
  whether item 2's/Stage A's findings are CrisperWhisper-specific or
  general to ASR systems, at the text level.
- **RQ-B**: does a different pretrained model's **encoder
  representation** carry a stronger, or more depth-distributed, disfluency
  signal than CrisperWhisper's own — specifically testing (i) whether
  CrisperWhisper's own fine-tuning is what concentrated the signal into
  only the last layer (stock `whisper-large-v3`, same architecture, no
  verbatim fine-tuning), and (ii) whether a model never trained for ASR
  at all, explicitly designed for paralinguistic sensitivity (WavLM),
  carries a cleaner signal than any Whisper-family encoder can.
- **RQ-C**: is the "only the last layer matters" pattern this track's
  own layer sweep found a property of CrisperWhisper's *fine-tuning*
  specifically, or of the Whisper *architecture* generally? (Answered
  directly by RQ-B(i)'s stock-Whisper arm, reusing the exact layer-sweep
  methodology already built.)
- **RQ-D** (the product-facing question this phase ultimately serves):
  would a **hybrid** architecture — CrisperWhisper for transcription,
  a different, paralinguistic-specialized model for corroboration —
  outperform either model alone? Not fully testable until RQ-A/B/C have
  results; this phase lays the groundwork, does not claim to answer it
  yet.
- **RQ-E** (a **reasoned proposal**, explicitly speculative, no literature
  or prior evidence backing it — offered per this project's own standing
  discipline that novel ideas are welcome when labeled honestly): does
  the model's own **decoding-time uncertainty** (token-level entropy or
  top-k probability spread at each generation step) carry information
  neither the decoded text nor the encoder's spatial representation
  does? A disfluency smoothed into a fluent token might still leave a
  trace in *how confidently* the model chose that token, independent of
  both signals this track has tested so far. Not scoped for
  implementation this phase — recorded here so it isn't lost, and so a
  future reader knows it was considered, not overlooked.

### Literature review for this phase

**Reused from this track's existing 13-source review** (§5 above,
`ASR_RESEARCH_TRACK.md`'s original literature pass): CrisperWhisper's own
design (retokenization + attention-alignment, not necessarily
fragment-level preservation, arXiv:2408.16589); WavLM already identified
as the best-published word-level stuttering-detection representation
found in this project's research (F1=0.554, arXiv:2409.10704); Whisper
encoder layers shown to carry disfluency signal for a *different* task in
a study whose finding this track's own layer sweep did not replicate
(arXiv:2311.05203) — the discrepancy driving RQ-C.

**New for this phase, checked directly this session, not assumed:**

- **Catastrophic forgetting / representational drift under fine-tuning
  is a real, independently documented phenomenon**, both Whisper-specific
  and general to transformer fine-tuning: Whisper fine-tuning on
  low-resource tasks is documented to cause catastrophic forgetting
  (multiple sources found), and general transformer fine-tuning research
  finds *intermediate layers* — not the first or last — undergo the most
  substantial representational drift, as pretrained features are
  reorganized to emphasize what the fine-tuning objective needs and
  de-emphasize what it doesn't. This directly supports (as an inference,
  not a proven mechanism for this specific model) why CrisperWhisper's
  own fine-tuning — narrowly optimized for verbatim word-level accuracy
  and timestamp precision — could plausibly have concentrated
  disfluency-relevant signal into only the last layer, unlike whatever
  base model arXiv:2311.05203 studied.
- **WavLM's own paper** (arXiv:2110.13900) states its design directly: two
  joint pretraining objectives, masked speech *prediction* (phonetic/ASR-
  relevant content) and masked speech *denoising* (explicitly for noise
  robustness and **paralinguistic and speaker-identity sensitivity**),
  deliberately built to serve "full-stack" speech tasks beyond ASR. This
  is a directly-cited design fact, not an inference. Whisper's own
  training (weak-supervised transcription accuracy at scale) has no
  comparable explicit paralinguistic objective — a real, citable
  architectural difference in what each model was built to represent
  well, though a specific paper directly quantifying "Whisper
  under-represents paralinguistic cues relative to WavLM" could not be
  independently confirmed this session (one candidate source, checked
  directly, did not contain that specific comparative claim on inspection
  — flagged honestly rather than cited with unearned confidence, per this
  project's own standing practice).
- **A real, verified, directly relevant empirical result**: "A Semi-
  Supervised Framework for Speech Confidence Detection using Whisper"
  (arXiv:2605.12387) — a genuinely different paralinguistic task (speaker
  confidence level, not disfluency), but structurally comparable (a
  paralinguistic state detected from a mix of semantic and acoustic
  signal). Found a **hybrid** model (Whisper embeddings + explicit
  acoustic features) beat both a pure-Whisper baseline (Macro-F1 0.751 vs.
  0.736) *and* pure self-supervised baselines including WavLM, HuBERT, and
  wav2vec2 — direct, verified support for the hybrid-architecture
  direction (c) specifically, not just "try a different model in
  isolation." **A real tension worth naming, not smoothing over**: this
  result (explicit acoustic features + Whisper helps) sits in some
  tension with this track's own Stage C2 (Praat acoustic features +
  CrisperWhisper's encoder-distance did *not* help). Plausible reconciling
  factors — different task, much larger sample in the cited paper, a
  different acoustic feature set — are inferences, not confirmed; the
  honest reading is that hybrid approaches *can* work, not that they
  always do, and Stage C2's negative result is real and stands on its own.

### Direction justification: what this phase actually tests, and why

**Not selected for this phase, with reasons**: (e) fine-tuning and (f)
purpose-built systems remain gated on infrastructure (GPU, paired data at
volume) this project does not have — attempting either now would not be
"ambitious," it would be under-resourced. (d) decoder/decoding
modifications beyond `num_beams` (e.g. RQ-E's entropy idea) are recorded
as a reasoned proposal but not scoped for implementation this phase —
worth a dedicated pass once (a)/(b)/(c) have results to build on. (g)
extending the acoustic-native precedent further remains a real, cheap,
live option, but its own prerequisite (a healthier candidate population,
per the earlier superseded fusion-style revision) is exactly what
(a)/(b) would help establish — sequenced after, not instead of.

**Selected for this phase**: directions (a) and (b), run together, since
they share nearly all their infrastructure and jointly answer RQ-A
through RQ-C in one coordinated pre-registration:

1. **Arm 1 (RQ-A, full pipeline)**: run **stock `whisper-large-v3`**
   (same architecture as CrisperWhisper, *not* fine-tuned for verbatim
   transcription) through the same Track B alignment pipeline this track
   already has, on the same audio already downloaded. Answers RQ-A and
   is the direct execution of item 10.
2. **Arm 2 (RQ-B(i)/RQ-C, representation-level)**: repeat the exact
   layer-depth-sweep methodology already built and validated, pointed at
   **stock `whisper-large-v3`'s** encoder instead of CrisperWhisper's.
   Isolates whether CrisperWhisper's fine-tuning specifically narrowed
   the signal to one layer (stock Whisper shows a different, more
   literature-consistent layer profile) or whether this is a Whisper-
   architecture-general property (stock Whisper shows the same
   last-layer-only pattern).
3. **Arm 3 (RQ-B(ii), representation-level)**: repeat the Stage-B/C
   methodology (last-layer encoder-distance, Cohen's d, AUC) pointed at
   **WavLM-Large's** encoder — a genuinely different architecture, never
   fine-tuned for ASR, explicitly designed for paralinguistic sensitivity.
   Tests whether a purpose-different pretrained representation
   out-performs anything in the Whisper family, independent of the
   fine-tuning question Arm 2 addresses.

Arms 1-2 are the cheaper, more directly diagnostic pair (same model
class already integrated, only the checkpoint changes for Arm 1; Arm 2
reuses the layer-sweep script verbatim with a different model_id) and
should run first as a fast, honest gate. Arm 3 requires new integration
work (a different model class, a different frame-rate/pooling
convention — see confounders below) and is scoped as the second,
slightly more expensive step, justified by Arm 1/2's results rather than
run blind alongside them.

### Pre-registered protocol

**Population**: the same 31-clip / 36-position (`sound_repetition`/
`word_repetition`) set this track has used since Stage B, for direct
comparability across every arm and every prior stage.

**Arm 1 (stock whisper-large-v3, full pipeline)**:
- Metric: the same Stage-A-style categorization (normalized-away /
  mis-routed / genuine ASR error / ASR error + coincidental type) applied
  to stock Whisper's decoded text at the same 36 positions, plus overall
  WER for a broad side-effect check.
- Success: a meaningfully lower "normalized-away" rate than CrisperWhisper's
  measured 45.2%/40.5% (`sound_repetition`/`word_repetition`) — real
  evidence CrisperWhisper's own choices (not ASR generally) are the
  driver.
- Failure: a comparable or higher normalized-away rate — evidence this is
  a general property of large-scale weakly-supervised ASR, not a
  CrisperWhisper-specific gap, shifting weight toward representation-
  level (Arm 2/3) or training-level (Stage D) directions over "just pick
  a different off-the-shelf ASR."

**Arm 2 (stock whisper-large-v3, layer sweep)**:
- Metric: AUC per layer, identical to the existing layer-sweep script,
  new model_id only.
- Success: a materially different layer profile than CrisperWhisper's
  (signal present at multiple layers, or concentrated at a different
  depth) — evidence for the fine-tuning-narrowed-the-signal inference.
- Failure: the same last-layer-only pattern — evidence this is a
  Whisper-architecture property, not a CrisperWhisper-specific one,
  narrowing RQ-C to "not fine-tuning specifically" and shifting weight
  toward Arm 3 (a genuinely different architecture) as the more
  informative remaining question.

**Arm 3 (WavLM-Large, representation-level)**:
- Metric: Cohen's d and AUC at WavLM's own last layer (and, resources
  permitting, a layer sweep matching Arm 2's), same target/control
  population, re-pooled to WavLM's own frame rate (see confounders).
- Success: d/AUC meaningfully above CrisperWhisper's own Stage-B/C result
  (d=0.894, AUC=0.723) — real evidence a purpose-different representation
  is worth adopting, at least as a hybrid corroboration signal (RQ-D).
- Failure: comparable or weaker than CrisperWhisper's own signal —
  evidence the "different pretraining objective helps" inference,
  however well-motivated by WavLM's own stated design, does not hold for
  *this specific task* — a real, valuable negative result either way.

### Confounders and costs, named before running

- **Model size confound (Arm 3 specifically)**: WavLM-Large (~300M
  params) is smaller than CrisperWhisper/whisper-large-v3's encoder
  (~600M-1B-parameter range). Any difference found cannot be cleanly
  attributed to "pretraining objective" alone — size is an uncontrolled
  variable this design cannot fully separate without a same-size
  self-supervised alternative, which may not exist. Named explicitly,
  not resolved.
- **Frame-rate / pooling confound (Arm 3)**: this track's existing
  `pool_span`/`FRAME_SECONDS` machinery (`profiling/encoder_embedding.py`)
  is built around Whisper's fixed 20ms-per-frame, 30s-window convention.
  WavLM operates on raw waveform at a different internal frame rate —
  **this is real, non-trivial engineering, not a drop-in checkpoint
  swap**, corrected here from an earlier looser characterization of this
  work as cheap. Scoped as its own small module, not a one-line change.
- **Preprocessing confound (Arm 1)**: stock `whisper-large-v3` may
  benefit from or be hurt by the same `num_beams=1` constraint
  CrisperWhisper needs (the timestamp-extraction bug is in `transformers`
  generally, not CrisperWhisper's fine-tune specifically) — Arm 1 should
  use the same decoding configuration as the live app's CrisperWhisper
  calls, for a fair comparison, not the model's own unconstrained default.
- **Cost, scoped from this session's own measured rates**: Arm 1 needs
  real, fresh ASR inference (not cached) — at this project's measured
  ~54-102s/clip range, 31 clips is a real, bounded cost (~30-55 min,
  single condition, cheaper than the decoding-sensitivity experiment
  since only one condition is needed here, not two). Arm 2 reuses the
  layer-sweep's own measured cost profile (~33-40s/clip x 31 clips,
  roughly 20-25 min, one forward pass per clip regardless of layer
  count). Arm 3's cost is not yet known — a new model, a new extraction
  path — and should get its own small timing check (a 2-4 clip dry run,
  matching every other stage's own established discipline) before a full
  run is committed to.

### Outcome-to-conclusion mapping

| Arm 1 | Arm 2 | Arm 3 | What this would support |
|---|---|---|---|
| Success | Success | — | CrisperWhisper's fine-tuning is the driver; a hybrid using stock Whisper (or an unfine-tuned checkpoint) as the corroboration source is the near-term product move — cheapest real product win this phase could produce. |
| Failure | Failure | Success | The problem is general to Whisper-family ASR; WavLM (or a similar self-supervised model) is the evidence-motivated representation source for a hybrid architecture — the strongest case yet for direction (c), still no fine-tuning required. |
| Failure | Failure | Failure | Neither swapping the ASR nor swapping the representation family helps at this sample size — real evidence to formally cost out Stage D (fine-tuning/data acquisition, §9) as the next real step, not another representation-shopping round. |
| Success | Failure | — | CrisperWhisper's specific behavior differs from general Whisper behavior in the *decoded text*, but not in its *encoder's layer structure* — points toward decoding/fine-tuning-level intervention on CrisperWhisper specifically, not a wholesale model swap. |
| Mixed / inconclusive | — | — | Reported as exactly that, per this track's standing discipline — not rounded toward whichever reading is more convenient, and treated as grounds for a larger sample before further architectural conclusions, per the same discipline Stage B applied to `word_repetition`. |

### Self-critique — an adversarial review of this plan, before proceeding

**Is this just "test another representation source" again, after three
of those already failed (duration, Praat, more CrisperWhisper layers)?**
A fair challenge, answered directly: Praat and duration are low-
dimensional, hand-engineered features with a low ceiling by construction;
more CrisperWhisper layers tested the *same* fine-tuned network's
capacity, already shown to concentrate signal in one place. Arms 1-3 test
a categorically different class of signal (high-dimensional, learned
representations from models with a *different* training objective) —
not the same idea repeated, but this document does not get to declare
that distinction meaningful by assertion; Arm 2 specifically exists to
test whether "different training objective" actually produces a
different result, rather than assuming it will.

**Does this phase quietly ignore the ~45% of Stage A's losses that are
ordinary ASR transcription error, not normalization?** Yes, and that
should be stated plainly rather than left implicit: this phase, like
every stage before it, scopes itself to the normalization-specific
mechanism (Stage A categories 1/2). General ASR transcription accuracy
on disfluent speech remains a separate, unaddressed problem — arguably
this project's original Phase 1 finding, still not directly targeted by
any experiment in this track. Not a flaw in this specific plan, but a
real, standing limitation of the track's scope that should not be
allowed to fade from view.

**Is recommending WavLM specifically just re-adopting an old roadmap item
under new language, rather than genuinely re-derived?** Checked
honestly: WavLM was already named as a candidate months ago
(`PHASE_3_ARCHITECTURE_REVIEW.md`'s original Stage 1b, never triggered).
What is new this session is the *reasoning path* — this track's own
layer-sweep anomaly (contradicting arXiv:2311.05203) motivated asking
*why*, which led to the catastrophic-forgetting/representational-drift
literature and WavLM's own stated design objective, independently
arriving at the same candidate via a different, evidence-driven route.
Worth naming this explicitly so the recommendation reads as re-derived,
not merely recycled — and worth the honest caveat that arriving at the
same answer twice by different paths is reassuring, not proof.

**Is the "hybrid architecture" framing (RQ-D) getting ahead of the
evidence — presenting a conclusion before Arms 1-3 have run?** Checked:
RQ-D is explicitly marked as "not fully testable until RQ-A/B/C have
results," and the outcome-mapping table above does not assume a hybrid
wins in every branch (the all-failure row recommends Stage D, not a
hybrid). The literature cited for direction (c) supports it as a
plausible, evidence-backed *candidate*, not a foregone conclusion.

**Cost discipline**: every arm's cost is estimated from this session's
own measured rates, not guessed fresh — consistent with this track's
own repeated practice of pricing things out before running them, after
underestimating cost twice already this session (Stage B/C's original
38-clip estimate, and the decoding-sensitivity experiment's ~5x
underestimate). Arm 3's cost is explicitly flagged as unknown and
requiring its own dry run, not assumed comparable to Arms 1-2.

### What changes as a result

`ROADMAP.md` item 10 is updated to point at this phase's specific,
pre-registered Arm 1/2 design (not a generic "run a second ASR backend"
note). A new pointer is added noting Arm 3 (WavLM) as the evidence-
motivated escalation if Arms 1-2 don't resolve things. This phase does
not authorize any change to `main` — per this track's own standing
non-goals (§10), findings here get evidence-gated the same way every
other decision in this project has been, and land on `main`, if ever,
only once real evidence supports it.

### Phase 2 results (2026-08-06) — all three arms run, exactly as pre-registered

Implementation: `profiling/evaluation/stage_arm1_stock_whisper.py` (new),
`profiling/evaluation/stage_layer_sweep.py` (extended with `--model-id`,
Arm 2 reuses it unmodified otherwise), `profiling/evaluation/
stage_arm3_wavlm.py` (new). All three ran against the identical 31-clip/
36-position population (or its 18-clip/19-target `sound_repetition`-only
subset, for the two layer sweeps — same subset Arm 2's own pre-registration
already established as population-verified-benign).

**Arm 1 (stock `whisper-large-v3`, full pipeline) — Failure.** 0/36
positions recovered (`recovered_tp` = 0% for both types). The "still
normalized away" rate came back *higher* than CrisperWhisper's own
full-audit baseline, not lower: 89.5% vs. 45.2% (`sound_repetition`,
17/19 still lost) and 88.2% vs. 40.5% (`word_repetition`, 15/17 still
lost). Two positions per type (4 total) didn't stay in category 1 — they
moved to category 3 (genuine ASR error), meaning the bigger model
introduced a *new* transcription error at a position CrisperWhisper had
transcribed correctly (even while losing the disfluency itself). Overall
mean WER (0.177) was comparable to CrisperWhisper's own range on this
sample, so this is not a "the second model is just worse at everything"
artifact — a materially larger, more capable model from the *same*
architecture family still normalizes these exact disfluencies away.
Real cost: 2533s (42 min) for 31 fresh transcriptions, plus an 873s
one-time download/load — within the pre-registered 30-55 min band once
the one-time load is excluded. This is the pre-registered **Failure**
outcome: evidence this is a general property of large-scale weakly-
supervised ASR, not a CrisperWhisper-specific choice.

**Arm 2 (stock `whisper-large-v3`, layer sweep) — Failure.** Same
last-layer-only pattern CrisperWhisper's own sweep found: layer 32 (last)
AUC=0.680, every other layer 0.336-0.378 (near or below chance) — no
signal distributed across depth the way arXiv:2311.05203's different task
found. Same-population comparison (both on the identical 18-clip/19-target
subset): CrisperWhisper's last layer scored 0.721 here; stock Whisper's
scored *lower*, 0.680 — a materially different, non-fine-tuned checkpoint
does not show a better or more depth-distributed signal, it shows a
slightly *weaker* one with the identical shape. Cost: 555s (9 min) for 18
clips, in line with the layer-sweep's established rate. This is the
pre-registered **Failure** outcome: the "last layer only" pattern is a
Whisper-*architecture* property, not something CrisperWhisper's
fine-tuning introduced — narrows RQ-C's answer to "not fine-tuning
specifically" and, per the pre-registration, shifts the open question
entirely onto Arm 3.

**Arm 3 (WavLM-Large, representation-level) — Failure on the primary
metric, with one genuinely new nuance worth recording honestly.**
`sound_repetition` (the type with a directly comparable CrisperWhisper
number): last-layer Cohen's d=-0.061, AUC=0.474 — indistinguishable from
chance, clearly *weaker* than both CrisperWhisper's own result (d=0.894,
AUC=0.723) and Arm 2's stock-Whisper result (AUC=0.680). This is a clean,
unambiguous **Failure**: WavLM's explicit paralinguistic-sensitivity
training objective (arXiv:2110.13900), however well-motivated in the
literature, does not produce a usable `sound_repetition` signal on this
task at this sample size. Cost was low and matched the dry run: 302s (5
min) for 31 clips plus a 357s one-time download — cheaper than either
Whisper arm, confirming the earlier "non-trivial engineering" concern
about frame-rate/pooling was more caution than the situation warranted
once checked directly (WavLM-Large's conv stride gives exactly 20ms/frame
at 16kHz, identical to `FRAME_SECONDS`, and LibriStutter's audio is
natively 16kHz — `pool_span`/`cosine_distance` needed zero modification).

Two things worth recording precisely, neither of which changes the
Failure verdict on the pre-registered metric, both flagged per this
track's own honesty discipline rather than smoothed over:
- **The layer-depth *profile* is genuinely different**, not just weaker.
  WavLM's per-layer AUC rises smoothly from the embedding layer (0.422)
  to a mid-network peak at layer 10 (AUC=0.617, layers 9-13 all in the
  0.58-0.62 range) before declining again toward the last layer
  (AUC=0.474) — the opposite shape from both Whisper variants, whose
  signal is concentrated *only* in the final layer with everything else
  near or below chance. This is a real, literature-consistent
  observation (masked-prediction pretraining is known to distribute
  task-relevant information differently across depth than a
  narrowly-supervised decoder objective) — but WavLM's *best* layer
  (0.617) still underperforms both Whisper variants' peak (0.723 / 0.680),
  so "differently distributed" does not mean "better" here. RQ-C is now
  answered as fully as this track's evidence allows: the concentration-
  in-the-last-layer pattern is Whisper-architecture-general (Arm 2), and
  a genuinely different pretraining objective does redistribute the
  signal across depth (Arm 3) without increasing its peak strength.
- **`word_repetition` showed a small positive signal WavLM alone found**:
  d=0.259, AUC=0.576 (n=17 target, n=966 control). CrisperWhisper's own
  Stage B never produced a usable `word_repetition` number at all (too
  few informative positions at the time). This is the first non-null
  `word_repetition` representation-level signal this track has measured
  — genuinely new, not previously known. **Explicitly flagged as
  too small to trust as a standalone finding** (rule 3): d=0.259 is a
  small effect by conventional standards, n=17 is the same small
  population every stage in this track has used, and this is the single
  positive cell among five arms x two types this phase tested — exactly
  the kind of result that needs replication at a larger sample before
  being treated as real, not a green light to act on. Recorded so it
  isn't lost, not elevated into a claim this evidence doesn't support.

### Integrative conclusion: Failure / Failure / Failure

Per the pre-registered outcome-to-conclusion mapping table, this is the
third row: **"Neither swapping the ASR nor swapping the representation
family helps at this sample size — real evidence to formally cost out
Stage D (fine-tuning/data acquisition, §9) as the next real step, not
another representation-shopping round."** This is not a hedge or a
partial result — all three pre-registered success criteria were checked
against real, freshly-run evidence and all three came back negative,
including the two (Arm 1, Arm 2) whose costs landed within the
pre-registered estimate and the one (Arm 3) whose cost came in
comfortably under it. No cherry-picking, no re-scoping after seeing
results, no "just one more arm" — the plan named this exact branch before
any of the three ran, and the evidence landed there.

**What this rules out, stated plainly**: "just pick a different
pretrained ASR" (direction a) and "just pick a different pretrained
representation, off the shelf" (direction b, in both its
fine-tuning-isolating and architecture-diversifying forms) are now both
evidence-closed for this specific problem (`sound_repetition`/
`word_repetition` evidence lost even at ASR-correct positions), not just
untested. Five real, pre-registered probes across this track (duration
baseline, Praat fusion, CrisperWhisper's own layer depth, `num_beams`,
and now three more model-swap arms) have each independently found the
same thing: nothing cheap, off-the-shelf, and representation-only closes
this gap.

**What remains, per the pre-registration's own "Remaining work" list**:
direction (e)/(f) — fine-tuning or a purpose-built representation — is
the only direction in the original 7-item design space this track has
not yet evidence-tested, precisely because it was gated on infrastructure
(GPU, paired disfluent-speech training data at volume) this project does
not currently have (`ASR_RESEARCH_TRACK.md` §9). The evidence-motivated
next step is **not** to attempt fine-tuning today — that would repeat
this track's own standing discipline violation (acting past what
resources support) — but to formally *cost it out*: what data would be
needed, what it would take to acquire or construct it, what compute a
minimal fine-tuning experiment would require, and what a pre-registered
success criterion for that experiment would look like, before deciding
whether to pursue it. Direction (g) (extending the acoustic-native
precedent further, matching what `block`/`prolongation` already do)
remains the one still-live, still-cheap option this phase deliberately
sequenced *after* (a)/(b) rather than instead of them (§ "Direction
justification" above) — with (a)/(b) now closed, (g) is the nearer-term,
lower-cost option worth a real look before committing to Stage D's much
larger investment.

### What changes as a result (Phase 2 results)

`ROADMAP.md` item 10 is updated to record all three arms as run and
Failed, with the integrative conclusion above, and to name Stage D
costing (not execution) and direction (g) as the two live next steps.
No change to `main` — consistent with this entire track's standing
non-goal, this phase's real, negative, well-evidenced result is exactly
the kind of finding this track exists to produce and record, not a
result that itself ships anything.

## Direction (g): an acoustic-native `sound_repetition` candidate generator — pre-registered protocol (2026-08-06, written before any code)

Chosen over Stage D costing as the immediate next step (project owner,
2026-08-06): cheaper, needs no new infrastructure, and was always
sequenced ahead of Stage D in Phase 2's own "Direction justification."
Pre-registered here before any implementation, per rule 1 and this
track's own two-step cadence (plan, then a separate go-ahead to build).

**This is not the earlier "fusion-style Stage C revision" idea, and the
difference matters.** That idea (§ "The exact proposed next stage,"
above, superseded 2026-08-06) combined the encoder-distance signal with
Stage A's mis-routing predictions — both of which only exist at
ASR-correct, ASR-mis-routed positions, a genuinely starved population
(n=19 targets, n=4 mis-routed cases). Phase 2 (Arms 1-3) has now closed
off the hope that a different ASR/representation would enlarge that
population. **This proposal sidesteps the starvation problem entirely
by not depending on ASR output at all.** `LibriStutter`'s ground-truth
labels carry each disfluent token's own acoustic timestamp
(`LabeledClip.tokens[ref_idx]["start"/"end"]`, from the dataset's own
CSV, independent of any ASR run) — so evaluation can use the *full*
ground-truth `sound_repetition` population in the sample (up to all 42
instances Stage A originally traced, not just the 19 that happen to
align correctly under one specific ASR), and could scale further via
`ROADMAP.md` item 14 (expanding the LibriStutter sample) at zero
additional ASR cost, since no ASR is involved in this detector at all.

**Hypothesis.** `sound_repetition` has a recoverable acoustic signature
independent of any transcript: 2+ short, spectrally self-similar voiced
bursts in immediate succession (the repeated fragment, e.g. "c-c-cat"),
each shorter than a typical syllable, followed shortly by a longer
voiced segment (the completed word) — detectable directly from the
waveform the same way `block` (silence duration) and `prolongation`
(sustained single-segment duration) already are, per `profiling/
acoustic.py`'s existing, shipped pattern.

**Design.**
1. Reuse `segment_voiced()` (`profiling/acoustic.py`) unmodified to get
   each clip's voiced/silent segment sequence — the same RMS/ZCR
   segmentation `block`/`prolongation` already build on.
2. New candidate-generation logic (new function, not touching
   `detect_prolongations`/`detect_blocks`): for each run of 2+
   consecutive voiced segments each under a duration threshold (start at
   350ms — a syllable-scale upper bound, tunable, not tuned against
   results per rule 4), compute pairwise spectral similarity between
   consecutive short segments. Start with the cheapest signal already in
   this codebase's dependency set — RMS/ZCR envelope cross-correlation —
   and escalate to MFCC cosine similarity (via `librosa` or a
   hand-rolled DCT-of-log-mel, checking what's already a dependency
   first) only if the cheap version is inconclusive, per this project's
   own repeated "cheapest first" discipline (Stage C's own duration
   baseline before the richer encoder-distance signal). If similarity
   clears a threshold AND a longer voiced segment follows within a short
   gap, emit a `sound_repetition` acoustic candidate anchored at the
   short-burst run's start.
3. **Evaluation is Track-A-style (ASR bypassed entirely)**, reusing the
   distinction the project's own (separately proposed, not yet built)
   dataset-evaluation methodology already names: score candidates
   directly against `LabeledClip.tokens[ref_idx]`'s ground-truth
   start/end, at a fixed time-tolerance window (start at ±200ms,
   consistent with the general convention in the localization
   literature this project has already reviewed), not against any ASR
   hypothesis index.

**Population.** All ground-truth `sound_repetition` instances in the
existing 120-clip Track B sample (up to 42, per Stage A's own original
audit — not gated by ASR-alignment correctness) as targets; every other
short-voiced-segment run in the same clips (i.e., positions this
mechanism *could* fire on but shouldn't) as the control/false-positive
population — explicitly including ordinary short function words ("the",
"a", "is", "it"), which are the most likely source of false positives
and must be represented in the control set, not just "everything else."

**Metric.** Precision/recall/F1 for candidate generation only (does a
candidate get proposed at the right time and place — not yet
type-classification or full pipeline integration, mirroring how `block`/
`prolongation` themselves started as pure candidate generators before
`detect.py`'s fusion layer touched them).

**Success criterion.** Recall and precision both meaningfully above a
naive duration-only baseline (flag every short-voiced-segment run,
regardless of similarity) — demonstrating the *similarity* check
specifically carries information, not just segment duration (the same
confound Stage C's duration baseline already ruled out for the
encoder-distance signal, checked again here because this is a genuinely
different signal source, not assumed to inherit that earlier result).
A concrete bar: precision at >=50% recall clears whatever `block`/
`prolongation`'s own candidate-generation stage achieves before their
own fusion/corroboration layers refine it (measure that number from
existing code/tests first, so the bar is calibrated to this project's
own shipped precedent, not an arbitrary target).

**Failure criterion.** Precision/recall indistinguishable from the
duration-only baseline — evidence the short-burst-similarity idea
doesn't separate `sound_repetition` from ordinary short-word speech
rhythm, which would itself be a genuine, reportable negative result
(and would suggest `sound_repetition`'s acoustic signature is less
distinctive at the waveform level than `block`/`prolongation`'s,
consistent with — though not proof of — why encoder-level
representations were worth trying in the first place).

**Confounders, named before running.**
- **LibriStutter's synthetic splicing may have its own acoustic
  signature** (a splice artifact at the repeated-fragment boundary)
  that could make this detector look better than it would on real
  stuttered speech — a structural risk of this dataset this track has
  flagged before (Track A/B's own split exists partly because of it).
  Any positive result here should be treated as "worth testing on real
  speech" (SEP-28k/FluencyBank, `ROADMAP.md` items 15/16), not as
  final, for exactly this reason.
- **No natural word-boundary anchor**: unlike Track B's hyp-index-based
  scoring, this evaluates against raw timestamps — a new, small
  alignment module (time-window match, not word-index match) is needed
  and should get its own hand-verification pass before trusting its
  output, matching every prior stage's discipline.
- **Threshold choices (350ms, similarity cutoff, 200ms tolerance) are
  starting points, not tuned values** — per rule 4, they may not be
  adjusted in response to how results look without a separate,
  explicit go-ahead; if the first pass is close but not clearly
  Success/Failure, that itself gets reported as inconclusive, not
  quietly re-thresholded until it clears the bar.

**Cost.** Cheapest experiment this track has run: no ASR inference, no
model download, pure signal processing on audio already downloaded.
Expect implementation + a first pass over the existing sample to be
well under an hour of compute, dominated by engineering time (the new
candidate-generation function and the timestamp-based scorer), not
runtime.

**Not yet started** — this is the pre-registration only, per the
project owner's explicit choice to scope before building. Implementation
requires a separate go-ahead.

### Direction (g) results (2026-08-06) — Failure, with an honest mechanistic diagnosis

Implementation: `profiling/evaluation/stage_g_acoustic_sound_repetition.py`
(new). Built with the same hand-verified-before-real-audio discipline
every prior stage in this track used: 7 self-test cases (candidate
grouping across short/long gaps, similarity scoring, scoring math)
written and passing before any real clip was processed.

**A real bug caught before the result could be trusted (rule 3
applied to the process, not just the outcome).** The first real run
returned an implausible recall=1.000/precision=0.966 — a number this
track's own standing discipline treats as a reason to check harder, not
report faster. Traced directly: the scoring function pooled every
clip's ground-truth target and candidate timestamps into two flat lists
before matching, so a candidate in one clip could "match" a target in a
*different* clip purely because both clips' independent, zero-based
timelines happened to pass through overlapping second-offsets (every
clip is ~10-15s; with 120 clips in the sample, this collision is
frequent, not rare). Fixed by scoring strictly within each clip; a new
self-test (`"a candidate in one clip never matches a target in a
different clip, even with identical raw timestamps"`) now guards this
specifically, matching this track's established pattern (the decoding-
sensitivity experiment's single-letter-word false-positive fix followed
the identical discipline: catch it with a hand-built case before
trusting real audio, not after).

**The real, corrected result**: baseline (every qualifying 2+-short-
burst run counts, no similarity gate) — recall=0.824 (42/51 ground-truth
`sound_repetition` instances matched), precision=0.081 (62/766
candidates were real hits). Similarity gating across the full
pre-registered threshold sweep (0.30-0.90) never meaningfully separates
signal from noise: precision stays pinned between 0.081 and 0.094 at
every threshold, while recall only *drops* as the threshold tightens
(0.824 -> 0.569 at threshold=0.90, since some genuine repetitions score
lower pairwise similarity than the noise floor). Best-F1 operating point
(threshold=0.90, F1=0.161) is not meaningfully above the ungated
baseline (F1=0.147) — a ~1.4-point F1 difference on a 0-1 scale, well
inside what the threshold sweep's own noise would produce by chance.
**This is the pre-registered Failure outcome, exactly as defined in
advance**: "precision/recall indistinguishable from the duration-only
baseline."

**A pre-registered check that was not performed, named honestly rather
than silently dropped**: the original success criterion also named a
"concrete bar" — precision at >=50% recall clearing whatever `block`/
`prolongation`'s own candidate-generation stage achieves before their
fusion layer refines it. That comparison number was never computed this
session. It doesn't change the verdict here (the mechanism failed the
primary duration-only-baseline criterion outright, so the secondary bar
was never reached regardless), but it is a real, acknowledged gap in
what was pre-registered versus what was actually measured — worth
computing directly (via `detect_prolongations`/`detect_blocks` scored
against their own ground-truth labels in this same sample) before this
arm is cited as fully closed out, rather than left as an implicit,
unstated omission.

**Mechanistic diagnosis, checked directly rather than left as a bare
number** (matching this track's standing preference for explaining
*why*, not just reporting *what*): inspected the burst-count and
duration distribution of both true-positive and false-positive
candidates directly. Two findings, together explaining the low
precision:
- Candidate *duration* is not the problem — median candidate span is
  0.81s (mean 1.01s), and no candidate exceeds even 50% of the median
  clip's own duration (15.14s). This rules out the initial concern
  (raised on first inspecting one clip's output, where a single
  32-burst, 4.6s-wide candidate looked like a degenerate "the whole
  clip is one giant run" artifact) as the dominant explanation — that
  case exists but is a tail (only 80/766 candidates exceed 2s), not the
  typical one.
- The real cause is a **base-rate / feature-specificity problem**: the
  true-positive candidates' own `n_bursts` values (2 to 32, median in
  the low single digits) overlap almost entirely with the false-positive
  candidates' `n_bursts` values (2 to 31, median 4) — there is no clean
  count-based separator between "a genuine repeated fragment" and "an
  ordinary short-word sequence," and capping burst count post-hoc would
  cost real recall roughly as fast as it would cost false positives
  (checked directly, not assumed — a large fraction of true hits have
  `n_bursts` well above any plausible "genuine stutter repeat count"
  cutoff). Ordinary fluent speech is simply rich in short, similarly-
  shaped voiced segments (function words, fast syllables spoken by the
  same voice in the same prosodic context) — the RMS/ZCR envelope-shape
  similarity feature this arm tried cannot tell "a word deliberately
  repeated" from "two different short words spoken in the same voice,"
  because both look alike on this specific feature. The base rate makes
  this expensive: ~751 plausible-looking short-burst-run positions exist
  across the sample against only 51 real instances, so even a feature
  with real but modest discriminative power would struggle to reach
  useful precision at this class imbalance.

**Per rule 4, no threshold was retuned in response to this result** —
the reported numbers are the pre-registered threshold sweep's own
output, not a search for a flattering operating point.

**What this does and does not establish**: it does **not** show
`sound_repetition` has no acoustic signature — recall of 0.824 with a
completely naive, ungated mechanism is itself evidence that a run of
short, energetic voiced segments reliably co-occurs with real
`sound_repetition` instances (consistent with the underlying hypothesis).
What it shows is that **this specific, cheap feature (RMS/ZCR envelope
shape) cannot discriminate that co-occurrence from the ordinary
background rate of short-word speech rhythm** — a narrower, more precise
negative result than "acoustic detection doesn't work," and one that
directly points at what a next attempt would need to fix: either a
richer per-burst feature (e.g. MFCC/spectral-shape similarity, the
escalation this arm's own pre-registration named but did not yet need,
since a decision was reachable without it) or a different discriminating
signal entirely (e.g. pitch/formant continuity across the repeated
bursts, which plain RMS/ZCR energy envelope does not capture).

### What changes as a result (direction (g))

`ROADMAP.md` item 10 is updated to record this result: candidate-
generation recall is real but precision is not usable at this feature's
current specificity — Failure per the pre-registered criterion, with a
named, evidence-grounded reason (feature specificity, not an absence of
acoustic signal) rather than a bare negative number. No change to
`main`, no change to `profiling/acoustic.py` or `profiling/detect.py` —
this result lives entirely in `profiling/evaluation/`, exactly as
pre-registered. The two live options this leaves: (1) a richer acoustic
feature (MFCC-based similarity) as a cheap, bounded next iteration on
the *same* mechanism, explicitly not yet tried; (2) treat this as the
track's second and third independently-converging pieces of evidence
(after Phase 2's Arms 1-3) that the remaining paths cheap enough to try
without new infrastructure are increasingly narrow, strengthening the
case for formally costing out Stage D.

### Direction (g), MFCC escalation — pre-registered addendum (2026-08-06, written before implementing)

Per the project owner's explicit choice, executing the escalation the
original pre-registration already named as the fallback if the cheap
RMS/ZCR-envelope feature proved insufficient — which it did. This is
not a new arm; it is the same candidate-generation mechanism
(`generate_candidates()`'s run-detection logic is unchanged) with only
the per-burst similarity feature replaced. The original pre-registration
named "MFCC cosine similarity" without specifying parameters — specified
here, before implementation, per rule 1.

**Design.** Hand-rolled MFCC extraction (no new dependency — `librosa`
is not installed in this project's environment, and this project's own
evaluation-script convention already favors minimal dependencies, see
`stage_c_duration_baseline.py`'s pure-Python `_auc()`; `scipy` is
already a transitive dependency and provides the DCT primitive needed).
Standard pipeline: windowed FFT (reusing `AcousticConfig`'s existing
`frame_seconds=0.025`/`hop_seconds=0.010`, the same frame/hop this
project's segmentation already uses) -> power spectrum -> a 26-filter
mel filterbank (20Hz-8000Hz, standard triangular-filter construction) ->
log energies -> DCT-II, keeping the first 13 coefficients (the standard
MFCC configuration in the speech-processing literature). Per-burst
feature: the mean MFCC vector across that burst's own frames (replacing
the RMS/ZCR resampled-envelope vector the first pass used). Similarity:
cosine similarity between two bursts' mean MFCC vectors, replacing
`_burst_similarity()`'s RMS/ZCR-based computation — everything else
(run detection, gap tolerance, scoring, the pre-registered threshold
sweep, the Track-A-style timestamp-based evaluation) stays exactly as
in the first pass, so any difference in result is attributable to the
feature change alone, not a confounded re-design.

**Population and metric**: identical to the first pass (same 120-clip
sample, same 51 ground-truth `sound_repetition` instances, same
duration-only baseline for comparison, same precision/recall/F1 at the
same threshold sweep) — a direct, controlled comparison against the
RMS/ZCR result already measured.

**Success criterion**: precision meaningfully above the RMS/ZCR pass's
best result (F1=0.161) at comparable or better recall — real evidence
spectral shape (not just energy envelope) is what this candidate class
needs to be separable from background speech rhythm.

**Failure criterion**: no meaningful improvement over the RMS/ZCR
pass — evidence the limiting factor is not *which* cheap acoustic
feature is used, but the deeper class-imbalance/genuine-ambiguity
problem this arm's first-pass diagnosis already named (ordinary short
words and genuine repeated fragments may simply not be separable by
any single-frame-level spectral-shape feature at this population size),
strengthening the case for Stage D costing rather than a further
feature search.

**Implementation self-check before trusting real output**: the MFCC
extractor itself will be validated on a synthetic pure-tone pair
(two instances of the identical synthetic tone must score high MFCC
similarity; two clearly different frequencies must score low) before
being trusted on real audio — the same discipline this project's own
`tests/test_acoustic.py` already applies to `segment_voiced()`.

### MFCC escalation results (2026-08-06) — Failure, closer to the bar, with a second real bug caught first

Implementation: `generate_candidates()` in `stage_g_acoustic_sound_
repetition.py` was refactored to accept a pluggable `similarity_fn`
(run-detection logic byte-identical to the first pass — only the
per-burst feature changes, so any result difference is attributable to
the feature alone, per the addendum's own design requirement). A
hand-rolled MFCC extractor (26-mel filterbank, 13 coefficients,
`scipy`'s DCT, no new dependency) was added and validated on synthetic
pure tones before touching real audio — both synthetic self-tests
passed (same tone -> high similarity; different frequency -> meaningfully
lower).

**A second real bug, caught the same way as the first (rule 3, "a
dramatic-looking number is a reason to check harder, not report
faster")**: the first real MFCC run returned a suspiciously *flat*
result — precision/recall barely moved across the entire 0.30-0.90
threshold sweep (766 candidates survived even the strictest gate,
nearly unchanged from the ungated baseline). This was as much a red
flag as the earlier implausibly-perfect result, just in the opposite
direction: a similarity feature that fails to gate out *anything* is as
suspicious as one that gates out everything correctly by chance.
Diagnosed directly rather than accepted: measured the actual similarity
distribution across all 3,708 burst pairs in the sample and found mean
similarity 0.961, with 90% of ALL pairs (not just true repeats) scoring
>=0.9. Root cause: MFCC coefficient 0 is overall log-energy, not
spectral shape, and it dominates the vector's norm — any two voiced
(energetic) segments score high cosine similarity on this feature
regardless of whether they sound alike, which is standard, well-known
MFCC practice to exclude (not a novel discovery, a known pitfall this
implementation fell into and then caught). Excluding coefficient 0
dropped the mean similarity to 0.433 with a real spread (min -0.955) —
confirmed the feature now measures spectral shape, not just "is this
voiced." Fixed in `_burst_similarity_mfcc()`; the synthetic-tone
self-tests still pass unchanged (spectral shape discrimination was
never the issue - only real speech's high, dominant overall energy
happened to expose the c0 masking problem).

**The real, corrected MFCC result**: unlike the RMS/ZCR pass's nearly
flat curve, this shows genuine precision/recall trade-off structure
across the threshold sweep — recall falls smoothly from 0.824
(ungated) to 0.157 (threshold=0.90) as precision rises from 0.081 to a
peak around 0.095-0.100 (thresholds 0.60-0.80) before collapsing at the
strictest setting. Best-F1 operating point: threshold=0.40, F1=0.170
(recall=0.686, precision=0.097).

**Two related but distinct comparisons were made against this number,
worth stating precisely rather than conflating, since the addendum's
pre-registered wording and the evaluation script's automated verdict
use different reference points**:
1. **The script's own automated verdict** (the same mechanism used to
   judge the RMS/ZCR pass, applied identically here for methodological
   consistency across both feature passes) compares the gated best
   against *this run's own ungated baseline* (F1=0.147, identical to
   the RMS/ZCR run's baseline, since candidate generation is
   feature-independent — see "Design" above) times 1.2: F1=0.170 falls
   short of that 0.176 bar by a small, real margin.
2. **The addendum's literally-worded success criterion** ("precision
   meaningfully above the RMS/ZCR pass's *best result*... at comparable
   or better recall") is a different, more direct comparison: MFCC's
   best-F1 point (precision=0.097, recall=0.686) against RMS/ZCR's own
   best-F1 point (precision=0.094, recall=0.569, threshold=0.90).
   Precision is only marginally higher (+0.003, ~3% relative) despite
   meaningfully better recall — not "meaningfully above" by any
   reasonable reading of that criterion either.

**Both comparisons agree: Failure.** Neither is reclassified as a
Success by picking the more favorable framing after seeing the result —
stated together here specifically so a future reader can see both
numbers were checked, not just the one that was easiest to compute, per
rule 4's discipline against post-hoc criterion selection.

**What this changes about the overall picture**: direction (g) has now
had a real, honest two-step escalation, exactly as pre-registered —
cheap feature (Failure), richer feature (closer, still Failure). Both
failures are explained mechanistically, not just reported as numbers,
and each failure was reached only after a real implementation bug was
caught and fixed first (cross-clip scoring pollution; MFCC coefficient-0
energy masking) — the negative results are trustworthy specifically
*because* the process that produced them survived two independent,
serious "this number is suspiciously good/flat, check it" audits. This
is the strongest, most carefully verified negative evidence this track
has produced for any single mechanism.

### What changes as a result (MFCC escalation)

`ROADMAP.md` item 10 is updated with this result. Direction (g)'s cheap-
feature search is now complete per its own pre-registration (one
escalation step, as named in advance) — no further feature variants are
recommended without a new, separately-justified reason, per this
track's standing discipline against open-ended tuning. No change to
`main`, `profiling/acoustic.py`, or `profiling/detect.py`.

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

**Correction, 2026-08-08, from `EXTERNAL_REVIEW_2026-08-07.md` §4.5,
independently re-verified against the paper's own abstract before
accepting the correction**: the characterization above is wrong on
substance, preserved here rather than silently edited per this track's
own convention. StutterFuse is a **retrieval-augmented classifier** — a
Conformer encoder conditioned on a non-parametric memory bank of clinical
examples ("classify by reference rather than memorization"), not a
text+acoustic gated fusion model. "Modality Collapse" / "Echo Chamber" in
the paper's own abstract refers to naive retrieval boosting recall while
degrading precision — an effect between acoustic evidence and *retrieved
neighbors*, not between text and audio. Confirmed F1=0.65 on SEP-28k,
with the paper's own body text stating this "is approaching the limit of
human agreement (Kappa ~= 0.7)" — directly relevant to this document's
own "Current Project State" discussion of the sound_repetition
inter-annotator ceiling (kappa=0.40, a *different*, lower figure for that
specific type — see the 2026-08-08 reconciliation section at the end of
this document). This paper is not a source of fusion-design ideas for
`ARCHITECTURE.md` §4 as originally claimed; it remains relevant only as a
field-baseline F1 number and as a second, independent confirmation that
SEP-28k's own annotation noise is a real, load-bearing ceiling on
achievable scores in this domain.

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

**Superseded — read "End of Today's Session (2026-08-06)" at the very
end of this document instead, if you are picking up cold.** This entire
section (including its own "Update, 2026-08-06" note immediately below)
describes the state as of 2026-08-05's close, before the first-principles
reassessment, Phase 2 (Arms 1-3), and direction (g) (RMS/ZCR + MFCC) all
ran. None of the three numbered options this section's own update note
proposes ("Scale up the sample" / "Try the mis-routing recovery" /
"Formally cost out Stage D") is what actually happened next — a broader
first-principles reassessment of the whole track ran instead, leading to
Phase 2's 3-arm design and then direction (g)'s two-step escalation, both
now complete. Kept here unedited, per this document's own append
discipline (never rewrite past sections, only mark what supersedes them)
— useful as a historical record of what was known and proposed at that
point in time, not as current guidance.

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

---

## End of Today's Session (2026-08-06)

**Read this section first if you are picking up cold.** This supersedes
the 2026-08-05 handoff above (see the redirect note at its start) as the
current entry point. Written for a reader who was not in today's session
at all — every claim below should be checkable against this document's
own numbered sections and `PAPER_DECISION_LOG.md`'s dated entries, not
taken on faith.

### What we attempted

Two independent lines of work, both fully pre-registered before
implementation, both fully executed and reported today:

1. **Phase 2**: a full re-opening of the 7-direction design space (not
   just "run a second ASR"), narrowing to a 3-arm design — stock
   `whisper-large-v3` through the full pipeline (Arm 1) and through the
   layer-sweep methodology (Arm 2), plus WavLM-Large's representation
   (Arm 3) — each testing whether a different off-the-shelf ASR or
   representation preserves `sound_repetition`/`word_repetition`
   evidence better than CrisperWhisper does.
2. **Direction (g)**: a purpose-built, ASR-independent acoustic
   candidate generator for `sound_repetition` — first with an RMS/ZCR
   envelope-shape similarity feature, then (the pre-registered
   escalation) with MFCC spectral-shape similarity — testing whether the
   disfluency is directly detectable from the waveform the way `block`/
   `prolongation` already are, bypassing ASR text entirely.

### What succeeded

No individual hypothesis was confirmed — every one of the five
sub-experiments (3 arms + 2 feature passes) came back Failure against
its own pre-registered criterion. What *did* succeed, and is worth
stating as a real outcome, not just a preamble to the negative results:

- **The research process itself worked as designed.** Every experiment
  was pre-registered before code existed; every result was reported
  exactly as measured, including two results (Direction (g)'s
  implausibly-perfect first RMS/ZCR run, and its implausibly-flat first
  MFCC run) that would have been easy to accept or discard without
  investigating. Both were investigated, both turned out to hide real
  bugs, both were fixed before being trusted, and both fixes are now
  guarded by dedicated self-tests. Rule 3 ("audit surprising results")
  was applied symmetrically — to results that looked *too good* and to
  ones that looked *too uninformative* — which is a stronger, more
  complete application of that rule than this track had explicitly
  demonstrated before today.
- **Direction (g)'s recall result (0.824, both feature passes) is a
  genuine positive finding**, even though the overall arm is scored
  Failure. It confirms `sound_repetition` reliably co-occurs with a
  detectable run of short voiced bursts in the waveform — the underlying
  acoustic hypothesis behind `block`/`prolongation`'s own detectors
  extends to this type too. What isn't yet solved is discriminating that
  signal from the ordinary background rate of short-word speech rhythm
  at usable precision.
- **WavLM's layer-depth profile genuinely differs from Whisper's** (peaks
  mid-network, not concentrated in the last layer) — a real,
  literature-consistent structural finding, even though its peak
  strength never exceeded either Whisper variant's.

### What failed

- **Arm 1** (stock `whisper-large-v3`, full pipeline): 0/36 known losses
  recovered; normalized-away rate *higher* than CrisperWhisper's own
  baseline (89.5%/88.2% vs. 45.2%/40.5%).
- **Arm 2** (same checkpoint's encoder, layer sweep): same last-layer-
  only concentration pattern, same-population AUC slightly *lower* than
  CrisperWhisper's (0.680 vs. 0.721).
- **Arm 3** (WavLM-Large): `sound_repetition` signal at chance level
  (AUC=0.474) on the one metric with a directly comparable prior number.
- **Direction (g), RMS/ZCR pass**: precision (0.081) indistinguishable
  from a naive duration-only baseline (F1=0.147) across the full
  similarity-threshold sweep.
- **Direction (g), MFCC escalation**: a real, measurably better result
  (F1=0.170, a genuine precision/recall trade-off curve) but still short
  of the pre-registered bar by both ways of measuring "meaningfully
  better" (see "MFCC escalation results" above for both comparisons).

### What we learned

- **CrisperWhisper's own fine-tuning is not the driver of either failure
  mode tested.** Both the text-normalization loss (Arm 1) and the
  last-layer-only signal concentration (Arm 2) are properties of
  large-scale weakly-supervised Whisper-family ASR generally, not
  something CrisperWhisper's specific training introduced. This
  meaningfully narrows where a future fix could live — not "retrain
  CrisperWhisper differently," but either a genuinely different model
  family or a genuinely different detection strategy.
- **A different pretraining objective (WavLM's masked-prediction +
  denoising) does redistribute signal across encoder depth, but does not
  increase its peak strength** for this specific task at this sample
  size — a real, if modest, piece of evidence that "just use a
  self-supervised model instead" is not a free win either.
- **`sound_repetition` has a real acoustic footprint, but no cheap,
  single-frame-level feature tried so far can isolate it from ordinary
  speech rhythm.** Two specific features (RMS/ZCR envelope shape, MFCC
  spectral shape) were tried and both failed on precision, for a
  mechanistically understood reason (ordinary short words look similar
  to genuine repeats on these features) rather than an unexplained one.
- **Two independent implementation bugs were caught this session by the
  same discipline, applied in both directions** — a result that looks
  too good and a result that looks too flat are both grounds to stop and
  check, not just the former. Both bugs (cross-clip timestamp pooling;
  MFCC coefficient-0 energy masking) are exactly the kind of subtle,
  otherwise-invisible errors that would have quietly inflated or
  flattened a reported number if this discipline hadn't been applied.

### What evidence became stronger

The case against "a cheap, off-the-shelf swap closes this gap" is now
built on **seven independent, pre-registered probes** across this
track's full history — Stage C's duration baseline, Stage C2's Praat
fusion, the CrisperWhisper layer-depth sweep, the `num_beams` decoding
experiment, and today's five (3 Phase 2 arms + 2 direction-(g) passes) —
all converging on the same conclusion from different angles (decoding
width, encoder depth, model family, acoustic features). No single result
here is decisive on its own; the *convergence* across genuinely different
mechanisms is what makes the case strong.

### What remains unknown

- **Whether a *combination* of weak signals (not yet tried) would work
  where individual hand-picked features didn't** — e.g., a trained
  classifier over RMS/ZCR + MFCC + duration + pitch/formant continuity
  together, rather than any single feature gated by a hand-set threshold.
  This is meaningfully different from "just try another feature" (rule
  4's concern about open-ended tuning) — it's a different *class* of
  approach (learned combination vs. hand-set threshold), not a fourth
  feature search.
- **The unmeasured "concrete bar"** from direction (g)'s original
  pre-registration (candidate-generation precision compared against
  `block`/`prolongation`'s own) — named honestly as a gap above, not yet
  closed.
- **Whether any of today's findings generalize beyond LibriStutter's
  synthetic splicing** — untested this session, and a real, named
  confound for direction (g) specifically (a splice artifact could make
  synthetic repeats look more or less acoustically distinctive than a
  genuine stutter).
- **The actual cost and feasibility of Stage D** (fine-tuning or a
  purpose-built model) — named as the remaining direction in the
  original 7-item space, but never priced out: what data exists or would
  need to be acquired, what compute/GPU access this project has or
  doesn't, what a realistic timeline looks like. This is explicitly the
  first thing next session should address — see "Next Session Plan"
  below.

### Why we are stopping here

Every experiment named in advance for today's scope is complete: Phase 2
was pre-registered as a 3-arm design and all 3 ran; direction (g) was
pre-registered as a single cheap-feature escalation (RMS/ZCR, then MFCC
if needed) and both steps ran. Continuing to search for a third or
fourth acoustic feature without a new, separately-justified reason would
be exactly the open-ended tuning rule 4 exists to prevent. The project
owner's own explicit instruction was to consolidate before deciding what
comes next, rather than let momentum carry the session directly into
Stage D — consistent with this track's standing pattern of pausing at
decision points rather than auto-continuing.

---

## Next Session Plan

**This section describes how the next session should *begin* — it is
explicitly not a Stage D pre-registration, and Stage D work should not
start from reading this section alone.** Tomorrow's first task is
thinking, not building.

### Step 1 — Re-review the accumulated evidence with fresh eyes

Before deciding anything, read (in order): this document's "End of
Today's Session" section above, then the full Phase 2 and direction-(g)
sections it summarizes (§"Phase 2 results" through §"MFCC escalation
results"), then `ROADMAP.md` item 10's full dated history for the
product-facing framing. The goal of this pass is not to re-verify the
numbers (already done today) but to sit with the *shape* of the evidence
— seven converging negative probes — and check whether it still reads as
convincing after a night's distance, not just in the moment it was
produced.

### Step 2 — Decide, explicitly, whether Stage D is actually justified

This is a real decision, not a formality. Arguments worth weighing
honestly, on both sides, before concluding:

- **For**: seven independent probes across four different mechanisms
  (decoding, encoder depth, model family, acoustic features) all failed;
  the original `ASR_RESEARCH_TRACK.md` §9 three-part gate for Stage D
  named "richer representations tried and found insufficient" as one of
  its three conditions, and that condition now has real, repeated
  evidence behind it — not from one experiment, but from the accumulated
  pattern.
- **Against, or at least worth pausing on**: the "combination of weak
  signals" idea (named above as unknown, not yet tried) is a cheaper,
  faster thing to check first, and skipping it to jump to Stage D would
  repeat the exact "escalate past unexploited cheap options" mistake
  this track's own rule 4 discipline exists to prevent. Similarly, the
  unmeasured `block`/`prolongation` concrete-bar comparison and the
  LibriStutter-generalization question are both real, cheap gaps that
  could change how today's results should be read.
- **The actual question to answer**: is the marginal cost of one more
  cheap experiment (a combined-feature classifier, or closing the two
  named gaps) still lower than the cost of formally pricing out Stage D,
  or has that cost/value calculus flipped? Decide this explicitly and
  record the reasoning — do not default to Stage D just because it's
  "next on the list," and do not default to "one more cheap try" just
  because it's cheaper, without weighing both.

### Step 3 — If Stage D is judged justified: analyze designs before touching code

Should the decision in Step 2 favor moving toward Stage D, the next
session's job is still analysis, not implementation. Questions to work
through and document, in order, before any pre-registration is written:

1. **What would actually be fine-tuned or built?** Options include (a)
   further fine-tuning CrisperWhisper itself specifically toward
   fragment/repeat preservation, (b) fine-tuning a *different* base
   model (e.g. stock `whisper-large-v3`, now that Arm 1/2 show it
   behaves comparably to CrisperWhisper untuned) with the same goal, (c)
   multi-task training that predicts disfluency labels alongside
   transcription rather than modifying transcription itself, or (d) a
   purpose-built model trained from scratch. These are meaningfully
   different projects with different costs and risks — do not conflate
   them under one "Stage D" label without picking one (or explicitly
   comparing a short list).
2. **What data would be required, and does it exist?** A paired
   audio + disfluency-preserving-transcript dataset at real training
   volume. LibriStutter's synthetic splicing may or may not be adequate
   for this (its labels are synthetic insertions, not necessarily
   representative of real disfluent speech acoustically or textually —
   the same confound direction (g) already named). Real datasets
   (SEP-28k, FluencyBank — `ROADMAP.md` items 15/16) would need real
   acquisition work, not just download-and-parse.
3. **What compute/infrastructure does this project actually have access
   to, and what would training realistically cost** (time, money, GPU
   access) — priced from real numbers, not assumed.
4. **What would a pre-registered success criterion for a minimal Stage D
   experiment look like** — matching this track's own standing practice,
   written and reviewed before any training run, with named confounders
   and a stated failure criterion, exactly as every experiment in this
   document has been.

### Step 4 — Document the rationale before implementation, and pre-register exactly as this track always has

Whatever the Step 2 decision is — pursue Stage D, run the combined-
feature idea first, close the named gaps first, or something else not
yet considered — the next session's actual deliverable should be a
**decision recorded with its reasoning**, written into this document
the same way every other decision in this track has been (evidence
separated from inference separated from judgment), followed by a full
pre-registration of whatever experiment comes next, before any code is
written. This mirrors exactly how Phase 2 and direction (g) were both
handled today: plan first, in writing, with a separate go-ahead before
implementation begins.

---

## Research review (2026-08-07) and the decision it produced

Opened today's session with the research-thinking-first pass the "Next
Session Plan" above asked for, before any code — a full evidence/
inference/speculation-separated review answering: what do we know with
high confidence, what's strongly supported, what's uncertain, what's
disproven, what's the strongest remaining hypothesis, have the cheap
directions genuinely been exhausted, and is Stage D actually justified
by the accumulated evidence (as opposed to being the only thing left on
the original list).

**Key finding, checked against this document's own pre-registered §9
gate rather than asserted independently**: §9's second condition for
concluding Stage D is necessary requires that richer-representation
approaches have "been tried and genuinely isn't enough, *not merely
untried*." Every signal this track has measured — CrisperWhisper's own
encoder-distance (Stage C, d=0.894 but 4.7% precision at R>=0.5 alone),
stock Whisper's representation (Arm 2), WavLM's representation (Arm 3),
RMS/ZCR acoustic similarity (direction (g)), MFCC acoustic similarity
(direction (g) escalation) — was tested **alone**, gated by a hand-set
threshold. A **combination** of any of these signals, learned rather
than hand-thresholded, has never been attempted. By the gate's own
language, condition 2 is not yet satisfied — untested, not failed.

**Decision (project owner, 2026-08-07)**: run exactly one more bounded,
low-cost experiment — a combined-signal classifier over signals this
track has already built — before moving to Stage D. Explicitly scoped
as the **final** cheap experiment: if it fails, proceed directly to
Stage D planning without further cheap variants, per the same
discipline that closed out direction (g) after one pre-registered
escalation rather than an open-ended feature search.

### Combined-signal classifier — pre-registered protocol (2026-08-07, written before any code)

**Hypothesis.** Individually, CrisperWhisper's encoder-distance signal
(real, d=0.894, but only 4.7% precision at R>=0.5 alone — Stage C) and
direction (g)'s acoustic candidate signal (real recall=0.824, but only
~8-10% precision alone) are each too weak on precision to ship standalone.
A classifier trained on both together — plus the acoustic candidate's
own already-computed structural features (burst count, duration) — may
separate true `sound_repetition` instances from false candidates better
than either alone, since the two signal families are measuring
genuinely different things (spectral/energy shape of the audio itself,
vs. how anomalous the ASR encoder's internal representation is at that
position) and their errors are not obviously correlated.

**Signals combined, all already built by this track, no new signal
sources**:
1. **MFCC mean similarity** (direction (g)'s escalation — the
   validated, better-performing of its two feature passes).
2. **Burst count** (`n_bursts`) and **candidate duration** — free,
   already computed alongside every acoustic candidate.
3. **Encoder-distance** (Stage C's signal) — recomputed at each acoustic
   candidate's own span (not at ASR-hypothesis-token positions the way
   Stage B/C originally scoped it), using the identical primitives
   (`profiling/encoder_embedding.py`'s `load_encoder`,
   `extract_last_layer_states`, `pool_span`, `cosine_distance`, and the
   same leave-one-out fluent-centroid construction Stage B/C used) — see
   "Population" below for exactly how this candidate-anchored version
   differs from Stage B/C's ASR-token-anchored one, and why that's a
   deliberate, necessary adaptation, not a scope change to the signal
   itself.

**Model.** Plain L2-regularized logistic regression via the exact,
already-built, already-tested infrastructure in `profiling/evaluation/
compare_corroboration_mechanisms.py` (`_fit_logistic_regression`,
`_standardize`, `_select_l2_by_nested_cv`, `_clip_folds`, `_prf1`) —
this project's own established mechanism for exactly this kind of
decision (it is what produced the shipped `word_repetition`/
`sound_repetition` repetition-classifier gate, `ROADMAP.md` item 17).
Reused directly, not reimplemented, and not extended with any other
model family, per the explicit instruction to keep this simple. No
hyperparameter search beyond the L2 grid this infrastructure already
performs *inside* its nested cross-validation (a pre-existing, already-
validated part of the reused code, not a new tuning surface opened for
this experiment).

**Population.** Every acoustic candidate `direction (g)`'s MFCC pass
already generates across the same 120-clip LibriStutter sample (766
candidates in that pass) is the unit of classification. Label = 1 if
the candidate overlaps a ground-truth `sound_repetition` instance
(identical `±200ms` tolerance direction (g) already used), else 0 — the
same definition, not a redefinition.

**A real, necessary adaptation, named explicitly rather than left
implicit**: Stage B/C's encoder-distance signal was originally defined
at *ASR-hypothesis-token* positions (it needs a CrisperWhisper `hyp_
token` span to pool the encoder at). Acoustic candidates are defined at
*waveform* positions with no guaranteed corresponding ASR token. For
each candidate, the encoder-distance feature is computed if a cached
CrisperWhisper `hyp_token` for that clip overlaps the candidate's span
(pooling the encoder there, same leave-one-out centroid); if no such
token exists (no ASR output at that position, or the clip has no cached
Track B transcript), the feature is **missing**, not zero or an error —
matching this project's own established "`None` means no extra evidence,
never a failed check" convention (`profiling/acoustic.py`'s
`vad_coverage`/praat-feature pattern). Missing values are imputed to the
**median** of the candidates that do have the feature (a neutral
placeholder, not one that pushes the classifier toward either label),
with a paired binary "has-encoder-signal" indicator column so the model
can weight imputed and real values differently rather than treating a
median-imputed value as if it were a real measurement.

**Evaluation.** 5-fold, clip-split cross-validation (`_clip_folds`,
deterministic round-robin over sorted clip ids — this project's own
established split, avoiding the control-group-independence leakage
Stage B/C's own limitations section named). Report out-of-fold
precision/recall/F1 for three arms, all under the identical CV
procedure for a fair comparison:
- **(A) MFCC-alone**, threshold selected per training fold
  (`_best_threshold_by_f1`/`_cv_threshold`, already-built) — a
  cross-validated re-statement of direction (g)'s own signal, used here
  as a built-in sanity check as much as a baseline: direction (g)'s
  originally-reported F1=0.170 came from a single in-sample threshold
  sweep (not held out), so this CV'd number is expected to land somewhat
  *lower* than 0.170 (a fairer, not contradictory, restatement) — a
  large, unexplained gap would instead flag a population or logic
  mismatch between this script and direction (g)'s original one, and
  would be investigated before trusting anything else here.
- **(B) Encoder-distance-alone**, same CV threshold procedure, restricted
  to candidates where the feature is real (not imputed) — the fairest
  single-signal comparison for this arm specifically.
- **(C) The combined classifier**, all 5 features (MFCC similarity,
  n_bursts, duration, encoder-distance, has-encoder-signal), via
  `_cv_classifier`'s nested-CV logistic regression, unmodified.

**Success criterion.** Arm (C)'s mean out-of-fold F1 is meaningfully
(>=20% relative — the same bar direction (g)'s own escalation used, for
methodological consistency across this track's decisions) above **both**
(A) and (B) individually. Requiring improvement over both, not just the
weaker one, directly tests the combination hypothesis rather than a
weaker "beats the worse baseline" claim.

**Failure criterion.** Arm (C) does not clear both (A) and (B)
meaningfully — evidence that combining these specific signals, at this
sample size, does not recover what neither provides alone. Per the
project owner's explicit instruction, this is treated as the closing
result for the cheap-direction search: no further feature or model
variant will be tried in response — the next step becomes Stage D
planning (not implementation), directly, without a new cheap-variant
detour.

**Confounders, named before running.**
- **Extreme class imbalance carries over unchanged** (51 positives among
  766 candidates, ~6.6%) — the same imbalance that limited every
  single-signal arm's precision. A learned combination can in principle
  do better than a hand-set threshold at this imbalance, but is not
  guaranteed to; this is exactly the open question being tested, not an
  assumption resolved in advance.
- **Small positive count for 5-fold CV** (~10 positives per fold on
  average) means fold-to-fold variance will be real and should be
  reported (the range across folds, not just the mean), matching
  `compare_corroboration_mechanisms.py`'s own `_summarize()` convention.
- **The encoder-distance feature's missing-value rate is not yet known**
  (depends on how often an acoustic candidate's span actually overlaps a
  cached ASR token) — will be reported directly as part of the result,
  not assumed to be small.
- **This is still LibriStutter's synthetic sample** — the same
  generalization caveat named for direction (g) applies unchanged here.

**Cost.** The acoustic-candidate and MFCC-similarity computation is
already fast (direction (g)'s own measured ~5 min for 120 clips). The
new cost is the encoder-distance pass — one `extract_last_layer_states`
forward pass per clip, at roughly the layer-sweep's own measured rate
(~30s/clip) since the dominant cost is the same full encoder forward
pass either way — estimated ~60 minutes for 120 clips, to be confirmed
with a small dry run (2-3 clips) before committing to the full run,
matching this track's own standing discipline for any new cost.

**Not yet started** — pre-registration only, before any implementation.

**Addendum (2026-08-07, written before seeing the corrected results below,
after the first real run surfaced the issue) — a real deviation from
"reused unmodified," recorded per rule 1 rather than silently absorbed.**
The pre-registration above said `_cv_classifier` (`compare_corroboration_
mechanisms.py`'s existing nested-CV logistic regression) would be
"reused directly, not reimplemented." The first real run instead
returned F1=0.000/0.000/0.000/0.000/0.000 — exactly 0 across all 5 folds
with zero variance — while Arm (B) alone (one of Arm (C)'s own five input
features) independently scored F1=0.244 on the identical data. That
combination (a trained model scoring literally nothing while one of its
own inputs alone scores real, non-trivial F1) is not a plausible "the
combination provides no benefit" result — it was investigated before
being trusted, per rule 3, exactly like both of direction (g)'s bug
catches.

**Root cause, confirmed directly, not assumed**: `_cv_classifier` uses a
hardcoded `proba >= 0.5` decision rule. This population's positive rate
is far more extreme (~8%, 62/766) than whatever population `compare_
corroboration_mechanisms.py`'s original `_cv_classifier` was built and
validated against — the training set of `evaluation.py`'s `compare_
corroboration_mechanisms.py`, whose (S1,M3) arm did not exhibit this
failure mode. Verified the mechanism with a synthetic check (not just
inferred): fit the same `_fit_logistic_regression`/`_standardize`
pipeline on synthetic data at this exact class balance, with a
deliberately real, informative feature planted in it, and found every
predicted probability on held-out data landed below 0.5 anyway (max
observed: 0.234) — a textbook, well-documented property of standard
logistic regression under severe class imbalance (the fitted intercept
correctly reflects the low base rate), not a coding defect in
`_fit_logistic_regression` or `_standardize` themselves.

**The fix, and why it is a fairness correction, not new tuning**: added
`_cv_classifier_optimal_threshold()` to this script — identical to
`_cv_classifier` (same nested-CV L2 selection, same fit) except the
classification threshold is selected on the *training* fold's own
predicted probabilities via `_best_threshold_by_f1()`, the exact same
function Arms (A) and (B) already use for their own thresholds, instead
of a hardcoded 0.5. Reusing a fixed 0.5 cutoff for Arm (C) alone while
Arms (A)/(B) get a train-fold-optimized threshold would have been an
*inconsistent*, not merely simpler, comparison — not calibrated to this
population's base rate at all. This is a methodological correctness fix
required to make the three-arm comparison valid in the first place, not
a new feature, model, or hyperparameter search — the pre-registration's
"no hyperparameter tuning, no feature-engineering rabbit hole"
instruction is about not chasing a better number by trying new inputs or
model families; it does not extend to using an internally consistent
evaluation rule across all three arms. Both the original fixed-0.5
result and the corrected optimal-threshold result are reported below,
not just the one that changed the outcome.

### Combined-signal classifier results (2026-08-07) — Failure: the combination matches, but does not exceed, the stronger individual signal

Implementation: `profiling/evaluation/stage_combined_classifier.py`
(new), reusing `direction (g)`'s MFCC candidate generator, `profiling/
encoder_embedding.py`'s Stage B/C primitives, and `compare_
corroboration_mechanisms.py`'s logistic-regression/nested-CV
infrastructure exactly as pre-registered. 6 self-tests (feature-matrix
construction, imputation, the has-signal indicator) written and passing
before any real audio ran.

**Real dataset**: 766 acoustic candidates across the 120-clip sample (the
identical population direction (g)'s MFCC pass generated), 62 positive
(8.1%), 607/766 (79.2%) with a real (non-imputed) encoder-distance
value — the missing-value rate named as unknown in the pre-registration
turned out to be moderate, not small, confirming the has-signal
indicator column was a necessary design choice, not defensive
over-engineering.

**The bug described in the addendum above reproduced identically on
this full run** (F1=0.000/0.000/0.000/0.000/0.000 with the original
fixed-0.5 threshold, exactly as the first run showed) — confirming it is
a deterministic property of this population's class balance, not
run-to-run noise. With the corrected train-fold-optimal threshold:

| Arm | Mean F1 | Range | Precision | Recall |
|---|---|---|---|---|
| (A) MFCC-alone | 0.147 | 0.121-0.189 | 0.090 | 0.541 |
| (B) Encoder-distance-alone | 0.244 | 0.125-0.300 | 0.343 | 0.234 |
| (C) Combined classifier | 0.242 | 0.105-0.348 | 0.314 | 0.244 |

**Verdict: Failure, exactly as pre-registered.** (C) clears the bar
against (A) (0.242 > 0.177, the 20%-relative bar computed from (A)'s
own mean) but does **not** clear the bar against (B) (0.242 < 0.293) —
and the pre-registration required clearing *both*, specifically to test
whether combination beats the *better* individual signal, not just the
weaker one. In practical terms, the combined classifier's mean F1
(0.242) is statistically indistinguishable from encoder-distance-alone's
(0.244) — a 0.002 difference, well inside the fold-to-fold range both
arms show. **The combination does not hurt, but it does not help
either** — the model appears to have effectively learned to rely on the
encoder-distance feature and treat the three acoustic-candidate features
as close to uninformative on top of it, rather than finding
complementary signal in their combination.

**A real, useful side-finding, worth stating plainly**: encoder-distance
alone (Arm B, precision=0.343) is a meaningfully stronger single signal
at this population and threshold-selection convention than either
acoustic feature. This is not directly comparable to Stage C's original
number (that was scored against a much larger control population,
19 targets vs 966 controls, at ASR-hypothesis-token positions
specifically) — this run scores it at the 607 acoustic-candidate
positions that happen to have a real encoder-distance value, a smaller,
differently-selected population. Both numbers are real and correct for
their own defined population; they answer related but distinct
questions and should not be conflated.

**What this does and does not establish**: it does **not** show that
combining signals never helps, or that the underlying encoder-distance
signal is illusory — Arm (B) alone remains the strongest single number
this entire track has produced against real candidates at usable
precision (0.343). What it shows is that this *specific* combination —
one strong signal (encoder-distance) plus three comparatively weak,
partially-redundant ones (MFCC similarity, burst count, duration, all
three derived from the same acoustic-candidate mechanism and likely
correlated with each other) — does not produce a better classifier than
the strong signal alone, at this sample size (766 candidates, 62
positive, 5-fold CV). A combination of encoder-distance with a
*genuinely different, stronger* second signal remains untested; this
result does not rule that out, it rules out this particular combination.

**Per the project owner's explicit, standing instruction**: this is
treated as the closing result of the cheap-direction search. No further
feature, model, or combination variant will be tried in response — the
next step is Stage D planning, directly, described below.

## Stage D planning (2026-08-07) — design, requirements, risks, and a real infrastructure finding

Per §9, Stage D (fine-tuning or a purpose-built representation) requires
all three of: (1) broad information loss, (2) richer representations
tried and genuinely insufficient, (3) a real, sufficient paired dataset
and the infrastructure to use it, existing or acquirable. Conditions 1
and 2 are now satisfied by this track's accumulated evidence (Stage A;
and, as of today, eight independent single/combined-signal probes
across two sessions — Stage C, Stage C2, the CrisperWhisper layer sweep,
`num_beams`, Phase 2's three arms, direction (g)'s two feature passes,
and today's combined-signal classifier). Condition 3 has never been
checked against this project's actual environment before today — it is
checked directly below, not assumed.

### What Stage D is intended to achieve

Every direction tested so far worked *around* the normalization problem
— trying to recover, from a representation trained for a different
objective, evidence that objective didn't reward preserving. Stage D's
premise is different in kind: adapt a model's weights, via a loss
function that directly rewards preserving `sound_repetition`/`word_
repetition` evidence, so the resulting representation or decoded text
is shaped around this project's own taxonomy rather than general
transcription fluency. This targets the root cause Stage A identified
(normalization is a property of what a model was trained to reward, not
merely information genuinely lost in the audio) rather than extracting
a weak signal from a model that was never asked to preserve it.

### Why previous directions failed, and why Stage D might succeed where they didn't

*(Inference, not proven — general ML priors, not evidence specific to
this problem.)* Every representation tested — CrisperWhisper, stock
Whisper, WavLM — was trained toward an objective that does not
explicitly reward preserving these specific disfluency patterns:
CrisperWhisper's own fine-tuning targeted verbatim transcription
broadly, not this taxonomy specifically; general ASR training (stock
Whisper) is documented to actively reward fluent, normalized output;
WavLM's paralinguistic-sensitivity objective is real but untargeted at
this specific taxonomy. A model fine-tuned with a loss function that
directly penalizes normalizing these patterns away would, in principle
(standard supervised-learning logic — a model optimized for an
objective tends to do better at that objective than one that wasn't),
learn a representation specifically shaped around it. **This is a
reasoned hypothesis grounded in general ML priors, not something this
track's evidence has tested or can currently confirm or refute** — Stage
D would be the first real test of it.

### Major engineering, computational, and data requirements

- **Compute**: fine-tuning a Whisper-scale model (CrisperWhisper/
  `whisper-large-v3`, ~1.5B parameters) requires GPU-scale compute for
  any realistic training run — forward and backward passes across a
  real dataset, likely multiple epochs. Not achievable in a practical
  timeframe on CPU alone (see "A real infrastructure finding" below).
- **Data**: a paired (audio, disfluency-preserving-verbatim-transcript-
  or-label) dataset at real training volume. Real options, honestly
  assessed:
  - **LibriStutter** (already downloaded, this track's own working
    sample): *synthetic* splices, not real disfluent speech. A model
    fine-tuned on it risks learning to recognize splice artifacts
    rather than genuine disfluency patterns — the same generalization
    risk this track has named for direction (g), now more consequential
    for a training target than for an evaluation set. Usable as a cheap
    sanity-check/prototyping set, **not** trustworthy as the sole
    fine-tuning data for a result meant to generalize.
  - **SEP-28k** (real stuttered speech, clip-level labels, no reference
    transcript): needs real audio acquisition — a podcast-URL download
    pipeline, ~32GB raw (`ROADMAP.md` item 15), not yet done, and even
    once acquired has no word-level transcript to fine-tune verbatim
    transcription against directly (only clip-level presence labels).
  - **FluencyBank Timestamped** (real people who stutter, word-level
    timestamps): needs a CHAT-format parser this project doesn't have
    (`ROADMAP.md` item 16), and possibly access-gated (unconfirmed).
  - **A real, uncomfortable observation**: CrisperWhisper's own
    (third-party, undisclosed) training data presumably already
    attempted something like this, and it still normalizes ~45-50% of
    these instances away (Stage A). This doesn't prove Stage D can't
    improve on that — CrisperWhisper's training objective and data are
    unknown to this project, and a taxonomy-specific loss is a different
    intervention than whatever CrisperWhisper's own fine-tuning
    optimized for — but it is a real, sobering prior worth stating
    plainly, not glossed over.
- **Engineering**: a training pipeline (data loading, loss function,
  fine-tuning loop, checkpointing, evaluation-during-training) — this
  project currently has an evaluation/inference harness, not a training
  codebase. Real, non-trivial new infrastructure, not an extension of
  anything that exists today.
- **Evaluation**: reusing the existing Track B pipeline for the final
  measurement is a real advantage (already built, already validated) —
  but requires a held-out test set genuinely uncontaminated by whatever
  data trains the model, which needs to be planned deliberately, not
  assumed automatic.

### A real infrastructure finding, checked directly (not assumed)

Verified against this actual project environment before writing
anything further: `torch.cuda.is_available()` is `False`; the installed
`torch` build is CPU-only (`2.12.1+cpu`); no MPS (not a Mac); the only
GPU present is integrated (Intel UHD Graphics 620, not CUDA-capable, not
suitable for ML training); 17GB RAM; 91GB free disk. **There is no local
GPU compute available for fine-tuning a Whisper-scale model.** This is
not a soft risk — it is a hard, checkable fact about the actual
environment this project runs in, and it directly determines whether
§9's condition 3 is satisfied "as-is" (it is not) versus "acquirable"
(a real, separate question — cloud GPU access — this document cannot
answer unilaterally, since it involves cost and access decisions outside
what's observable from this repository).

### Biggest technical risks

1. **No accessible large-scale, real (non-synthetic) paired training
   data**, independent of the compute question — potentially blocking
   on its own.
2. **LibriStutter's synthetic nature risks training a model that
   overfits to splice artifacts**, producing a result that looks like
   success on this track's own synthetic evaluation set while not
   generalizing to real disfluent speech — the single most insidious
   risk, since it could produce a confidently-wrong positive result.
3. **Catastrophic forgetting**: this track's own literature review
   (Phase 2) already found Whisper fine-tuning on narrow objectives
   documented to degrade general transcription accuracy — a real risk
   of shipping a model that's better at this taxonomy but worse
   overall.
4. **No local compute**, meaning any real attempt depends on cloud GPU
   access this project has not set up or budgeted, adding real cost and
   setup work beyond anything this track has needed so far.
5. **Generalization is unverified even after a nominally successful
   fine-tune**, without real held-out data — the same caveat this track
   has attached to every LibriStutter-based result, now attached to a
   training target rather than an evaluation target, which is a more
   consequential place for it to bite.

### Expected payoff relative to required effort

The scientific case for attempting Stage D is real: two full sessions
and eight independent probes (single-signal and combined) all converge
on the same conclusion that cheap, off-the-shelf, representation-only
approaches are insufficient. But the cost is high and genuinely
uncertain — real data-acquisition work (a multi-hour-to-multi-day
undertaking for either SEP-28k or FluencyBank), a new training
codebase, real cloud-compute cost, and no guarantee of success (Stage D
tests a reasoned hypothesis, not a proven one). The LibriStutter-only
"cheap" path specifically is not a low-risk shortcut here the way it
has been for every evaluation-only stage in this track — for a
*training* target, its synthetic-splice risk could produce a
misleadingly confident false success, which is a materially worse
outcome than a clean negative result. This tempers the "least expensive
Stage D attempt" outcome that might otherwise seem available.

### Simpler alternatives worth considering before committing further

Applying the same discipline that led to today's combined-signal
experiment (never skip a cheaper, still-untested option to jump to the
expensive one): **acquiring a modest amount of real (non-synthetic)
disfluent speech first, without training anything, to test whether
today's combined-classifier signal (or direction (g)'s acoustic
mechanism) generalizes beyond LibriStutter's synthetic splicing** is
strictly cheaper than any Stage D attempt (no GPU, no training
pipeline), directly addresses Stage D's own biggest named risk (data
realism) before committing further effort to it, and would independently
be valuable evidence regardless of which direction comes next. This was
not part of today's authorized experiment and is not undertaken here —
named as a real, credible option for a future decision, not decided
unilaterally.

### What this means for today

**Condition 3 of this track's own §9 gate is not currently satisfied,
and the honest, evidence-disciplined thing to do — exactly what §9
itself anticipates — is to record that directly rather than attempt an
implementation that cannot realistically run**: "otherwise this is a
correct conclusion with no way to act on it yet, which should be
recorded as exactly that (a validated future-work item, not a stalled
implementation)." Stage D is now scientifically well-motivated (§9
conditions 1 and 2 satisfied) but not currently actionable on this
project's existing infrastructure (§9 condition 3 unmet) — a genuine
engineering blocker, not a research dead end, and not something this
document resolves by guessing at a workaround. This is recorded here as
that exact, honest state, pending a decision the project owner needs to
make (cloud GPU access and budget; real-data acquisition scope; or
holding Stage D as validated future work while pursuing the cheaper
real-data-generalization check named above) rather than proceeding
further unilaterally.

---

# Final research audit — 2026-08-07: experimental close-out, literature grounding, and Stage D readiness

**This section is a research-audit and documentation pass only — no
experiment, model, or dataset was touched while writing it.** Its job is
to state, independently and skeptically, exactly what this track's
experiments established, ground that against the published literature
(verified directly against primary sources, not summarized from
memory), and leave a record a new researcher, technical reviewer, or
collaborator could evaluate without asking the authors to explain
anything further.

## Experimental Phase Stopping Point — 2026-08-07

### The full experimental progression, reviewed end to end

| # | Stage | What it tested | Result |
|---|---|---|---|
| 1 | Stage A (§8, 2026-08-05) | Systematically categorized all 186 disfluent ground-truth positions in a 120-clip real-ASR sample into 4 categories (normalized-away / mis-routed / genuine ASR error / ASR error + coincidental type) | Roughly half of `sound_repetition`/`word_repetition` losses (45.2%/40.5%) happen even when CrisperWhisper transcribes the position "correctly" — normalization, not transcription error, is the dominant single mechanism |
| 2 | Stage B (§8) | Does CrisperWhisper's own last-layer encoder retain discriminative information at exactly these normalized-away positions? | Mixed but real: a genuine signal for `sound_repetition`, inconclusive for `word_repetition` at this sample size |
| 3 | Stage C (§8) | Is the Stage B signal explained by a duration confound? Is it precise enough to use alone? | Duration confound refuted (AUC=0.483, chance); genuine signal confirmed (d=0.894, AUC=0.723); precision alone insufficient (4.7% at R>=0.5) |
| 4 | Stage C2 (§8) | Does fusing the encoder signal with Praat voice-quality features (pitch, jitter, shimmer, HNR) help? | Clean negative — all 5 features near chance (AUC 0.452-0.549) |
| 5 | CrisperWhisper layer-depth sweep | Is the signal concentrated in the last encoder layer only, or distributed? | Last-layer-only (AUC 0.721); every other layer near/below chance (0.336-0.378) |
| 6 | Decoding-parameter sensitivity (`num_beams`) | Does beam width (5 vs. the app's forced 1) recover any lost positions? | 0/14 recovered; identical WER |
| 7 | Phase 2 Arm 1 (stock `whisper-large-v3`, full pipeline) | Does a bigger, non-fine-tuned Whisper-family ASR preserve more of this evidence? | Worse, not better — 0/36 recovered, normalized-away rate higher than CrisperWhisper's |
| 8 | Phase 2 Arm 2 (stock `whisper-large-v3`, layer sweep) | Is the last-layer-only concentration CrisperWhisper-specific? | No — same pattern, same-population AUC slightly lower (0.680 vs. 0.721) |
| 9 | Phase 2 Arm 3 (WavLM-Large) | Does a genuinely different (non-ASR) pretraining objective carry a stronger signal? | `sound_repetition` at chance (AUC=0.474); real but unexploited layer-depth-profile difference; small, unreplicated `word_repetition` signal (d=0.259, flagged untrustworthy) |
| 10 | Direction (g), RMS/ZCR acoustic candidates | Does `sound_repetition` have a directly-detectable, ASR-independent acoustic signature? | Real recall (0.824) but unusable precision (0.081); a real cross-clip scoring bug caught and fixed first |
| 11 | Direction (g), MFCC escalation | Does a richer spectral-shape feature do better than RMS/ZCR envelope shape? | Closer (F1=0.170 vs. 0.161) but still short of the pre-registered bar; a real MFCC-coefficient-0 masking bug caught and fixed first |
| 12 | Combined-signal classifier | Does a trained combination of the strongest available signals (encoder-distance + acoustic features) beat the best individual one? | F1=0.242, statistically indistinguishable from encoder-distance alone (0.244); a real fixed-threshold bug caught and fixed first |

**Every stage from 5 onward followed the same discipline**: pre-register
the exact protocol before writing code, run it, treat any surprising
result (too good *or* too flat/bad) as a reason to investigate before
reporting, fix any real bug found, and report the corrected number
alongside an honest account of what was caught and why. Three
independent implementation bugs were caught and fixed this way across
stages 10-12 alone (cross-clip timestamp pooling; MFCC coefficient-0
energy masking; a classification threshold miscalibrated to severe
class imbalance) — each investigated because a result looked
suspiciously good *or* suspiciously uninformative, not accepted either
way without checking.

### Why this is a deliberate stopping point, not "we ran out of ideas"

Three independent, falsifiable lines of evidence converge on the same
conclusion, each addressing a different candidate explanation:

1. **It is not which ASR model.** Arms 1 and 2 tested a materially
   different, larger, non-fine-tuned Whisper-family model and found the
   same or worse behavior on both the decoded-text and encoder-signal
   fronts.
2. **It is not which pretraining objective.** Arm 3 tested a model with
   a genuinely different training objective (WavLM's masked-prediction
   plus denoising, explicitly designed for paralinguistic sensitivity)
   and found no stronger signal on the primary metric.
3. **It is not which cheap acoustic feature, or whether the strongest
   available signals can be combined.** Direction (g) tested two
   different acoustic-similarity features directly on the waveform, and
   the final combined-signal classifier tested whether a trained
   combination of the single best representation-level signal
   (encoder-distance) with the single best acoustic-native signal
   (MFCC/burst structure) could jointly do better than either alone. It
   could not.

This is a closed loop, not an open-ended list that simply ran dry: three
different *kinds* of hypothesis (model choice, pretraining objective,
feature richness/combination) were each tested with a real, falsifiable
experiment, and each was falsified on its own terms. The stopping point
is deliberate because the next remaining direction in the track's
original 7-item design space (fine-tuning/purpose-built training, Stage
D) requires different, non-cheap resources this track does not yet have
verified access to (see "Stage D Requirements / Re-entry Gate" below) —
not because ideas ran out, but because the cheap-direction search
reached a real, evidence-grounded terminus.

## Findings Classification

Separating what these experiments establish from what they don't,
per the project owner's explicit request to be especially careful with
novelty claims.

**A. High-confidence findings, directly supported by our experiments.**
- Roughly half of `sound_repetition`/`word_repetition` ground-truth
  instances are lost even when CrisperWhisper transcribes the position
  "correctly" (Stage A, n=186, large sample).
- CrisperWhisper's last-layer encoder carries a real, duration-
  independent signal for `sound_repetition` at exactly these lost
  positions (Stage B/C, d=0.894, AUC=0.723, duration confound directly
  refuted).
- That signal's last-layer-only concentration is a property of the
  Whisper architecture generally, not CrisperWhisper's fine-tuning
  specifically (Arm 2, direct same-population comparison).
- A bigger, non-fine-tuned Whisper-family model does not recover this
  lost text evidence — it is worse, not better (Arm 1, n=36 well-defined
  positions).
- `sound_repetition` has a genuine, ASR-independent acoustic
  co-occurrence with runs of short voiced bursts (direction (g)'s
  recall=0.824, replicated across two independently-implemented
  similarity features).

**B. Findings supported by multiple experiments but still limited in
scope.**
- That no representation tested so far (CrisperWhisper's own encoder,
  stock Whisper's encoder, WavLM's encoder) carries a *usable-precision*
  `sound_repetition` signal alone — true for the three representations
  actually tested, not established for representations in general.
- That combining the strongest individually-tested signals doesn't beat
  the best one alone — true for this *specific* combination (one strong
  signal, three weaker/correlated ones) at this sample size (766
  candidates, 62 positive); a combination of the strong signal with a
  genuinely different, comparably strong second signal remains untested.
- That two cheap, hand-engineered acoustic features (RMS/ZCR, MFCC)
  cannot discriminate genuine repeats from ordinary speech rhythm — true
  for these two specific features; a *learned* acoustic representation
  (see "Existing Technology & Literature Landscape" below — e.g.
  YOLO-Stutter/SSDM-style trained region detectors) was never tried and
  is a materially different, more capable class of approach.

**C. Open questions.**
- Whether any representation not yet tested (a same-parameter-count
  self-supervised model, a purpose-trained acoustic dysfluency detector)
  could succeed where the three tested ones didn't.
- Whether any of this track's findings generalize beyond LibriStutter's
  synthetic splicing to real disfluent speech.
- The unmeasured comparison against this project's own shipped
  `block`/`prolongation` candidate-generation precision (named, not yet
  closed).
- Whether Stage D (fine-tuning/purpose-built training) would succeed —
  entirely untested by this track.

**D. Stage D hypotheses (reasoned, not evidence-backed).**
- That a model fine-tuned with a loss function directly rewarding
  preservation of this taxonomy's patterns would learn a representation
  better shaped around it than any of the general-purpose representations
  tested so far — a standard supervised-learning prior, not something
  this track's evidence confirms or refutes.
- That such a model would generalize beyond whatever training data it
  used — untested, and a real risk given LibriStutter's synthetic
  splicing (see below).

**E. Claims this track must NOT make, because the evidence does not
support them.**
- **"No existing ASR in the world can solve this."** Not established —
  only three representations (CrisperWhisper, stock `whisper-large-v3`,
  WavLM-Large) and two hand-engineered acoustic features were tested.
  The defensible claim is narrower: *the specific off-the-shelf
  approaches investigated in this project were insufficient to preserve
  or recover the specific disfluency information this task requires, at
  the sample sizes tested.*
- **"This is a groundbreaking / first-ever / world-first problem."** As
  the literature review below shows directly, disfluency-preserving ASR
  is an active, published research area with real, recent (2024-2026)
  work attempting closely related things — this track's own findings sit
  *within* that landscape, not ahead of or outside it. See "Open Research
  Gap and Novelty Assessment" below for the precise, defensible framing.
- **"Combining signals never helps."** Only one specific combination was
  tested; the claim must stay scoped to that combination.
- **"Sound_repetition has no acoustic signature."** Directly contradicted
  by this track's own recall=0.824 result — the defensible claim is that
  the *cheap features tried* couldn't isolate it at usable precision, not
  that no signature exists.

## Existing Technology & Literature Landscape

Every source below was checked directly (fetched and read, not
summarized from a search snippet) before being cited, per this
project's own standing citation discipline. Full bibliographic details
are in the "Bibliography" section at the end of this document.

### Major modern ASR paradigms, and what each is built to preserve or discard

| Paradigm | Representative system(s) | What it's optimized to do | What it tends to preserve | What it can normalize, collapse, or discard |
|---|---|---|---|---|
| **Weakly-supervised encoder-decoder (Whisper family)** | Whisper (Radford et al. 2022/2023 [B1]), `whisper-large-v3`, CrisperWhisper (Zusag et al. 2024 [B2]) | Transcribe fluent, readable text from 680k hours of noisy, weakly-labeled web audio | Word/timestamp-level content when transcribed correctly; CrisperWhisper specifically adds attention-based alignment for accurate word timestamps | Disfluencies generally — the decoder is trained toward fluent output; this track's own Stage A/Arm 1/Arm 2 results are direct, first-party evidence of this for `sound_repetition`/`word_repetition` specifically |
| **CTC (Connectionist Temporal Classification)** | Graves et al. 2006 [B3]; many modern CTC-head systems | Frame-level alignment-free sequence labeling, no explicit language-model decoder | Precise temporal boundaries per emitted symbol (CTC's "blank" mechanism is inherently timing-aware) | No built-in mechanism to represent "this token was said twice" as anything other than two separate label emissions — repetition-as-disfluency is not a first-class concept in the objective |
| **RNN-T / transducers** | Graves 2012 [B4]; modern streaming production systems | Streaming, monotonic-alignment sequence transduction, used heavily in real-time systems (e.g. Zipformer+RNN-T pipelines, per this review's 2025-2026 search [B16]) | Monotonic timing; strong for low-latency partial hypotheses | Same fluency bias as other supervised ASR unless explicitly trained otherwise — no evidence found of RNN-T systems natively preserving disfluency structure |
| **Conformer-based encoders** | Gulati et al. 2020 [B5] | Combine convolution (local) and self-attention (global) for better acoustic modeling, typically paired with CTC or RNN-T decoding | Strong local acoustic detail relative to pure-Transformer encoders | Still trained toward a supervised transcription/decoding objective — the encoder architecture change doesn't by itself change what the decoding objective rewards |
| **wav2vec 2.0** | Baevski et al. 2020 [B6] | Self-supervised contrastive pretraining on raw audio, fine-tuned for ASR or other downstream tasks | A general acoustic representation not shaped by a fluency-rewarding decoder — closer to "what does this audio sound like" than "what would a fluent transcript say" | Nothing disfluency-specific is *encouraged*, but nothing is explicitly *discouraged* either — this is a genuinely different objective than Whisper's, one plausible reason self-supervised encoders were worth testing (Arm 3's WavLM test) |
| **HuBERT** | Hsu et al. 2021 [B7] | Masked prediction of clustered pseudo-labels derived from the model's own earlier representations | Phonetic/acoustic structure needed to predict masked regions — untested in this track directly, but architecturally similar to WavLM | Same caveat as wav2vec 2.0 — no explicit disfluency objective either way |
| **WavLM** | Chen et al. 2022 [B8] | Joint masked speech prediction + masked speech *denoising*, explicitly designed for "full-stack" speech tasks beyond ASR, including paralinguistic/speaker-identity sensitivity | Paralinguistic detail more explicitly than Whisper's objective rewards — this track's own Arm 3 tested exactly this hypothesis | Directly tested by this track (Arm 3): despite the more paralinguistically-aware objective, `sound_repetition` signal at chance on the primary metric at this model size |
| **Neural-codec / discrete acoustic-token language models** | AudioLM, SpeechGPT-style "Chain-of-Modality" systems, Moshi [B9] | Model speech via a hierarchy of *acoustic* tokens (from a neural codec, e.g. SoundStream) alongside or instead of *semantic* tokens, explicitly to preserve enough acoustic detail for high-fidelity reconstruction | By design, acoustic tokens preserve "complete acoustic information" for reconstruction — the closest paradigm reviewed to "don't lossy-compress toward a fluent transcript at all" | Never tested by this track — a genuinely different technical paradigm from everything tried so far, flagged below as a forward-looking, untested direction, not a finding |
| **Forced alignment (non-ASR)** | Montreal Forced Aligner (HMM-GMM-based, still the most widely used [B10]); wav2vec2-CTC-based alignment (e.g. Charsiu) | Align a *known* transcript to audio at high temporal precision, not decode text from scratch | Sub-20ms phone-boundary precision — measurably more precise than end-to-end ASR word-timestamp methods including Whisper-based ones (WhisperX, MMS), per direct comparison in the literature [B10] | Requires the transcript already be correct/verbatim — doesn't solve the normalization problem, it assumes it's already solved |

### Disfluency-aware and disfluency-preserving ASR — dedicated review

This is an active, real, recently-published research area, not an
unexplored one in the broad sense. Representative, directly-verified
work (2021-2026):

- **Kordt et al., "Learning to Hear Hesitation: Continual Learning for
  Disfluency-Aware ASR"** (Interspeech 2026 [B11]) — the single closest
  published precedent to this track's own Stage D hypothesis. Fine-tunes
  `whisper-small.en` with explicit disfluency tokens (FILLER, REP,
  DISRUPT, PAUSE) using four continual-learning methods (EWC, Experience
  Replay, A-GEM, Weight Averaging) on real (non-synthetic) speech from
  three TalkBank/CHAT-format corpora (SME, Pitt/DementiaBank, Delaware —
  ~10-20 hours each, healthy L2 speakers and people with dementia/mild
  cognitive impairment, not stuttering specifically). **Directly
  confirms this track's own catastrophic-forgetting concern with real
  numbers**: a hard, quantified trade-off between marker-prediction F1
  and general transcription WER — methods that best preserved general
  ASR quality (Weight Averaging) failed to emit disfluency markers at
  all (F1=0.00), while methods that reliably emitted markers did so at
  measurable WER cost. **Critically, their `REP` token is a single,
  coarse marker covering both word- and phoneme-level repetitions
  combined** — it does not preserve or distinguish the repeated
  fragment's own content, and does not separate `sound_repetition`
  from `word_repetition` the way this project's taxonomy does. Even at
  that coarser granularity, marker F1 for `REP` ranged only 0.35-0.63
  across methods (their Table 4) — real, independent evidence that even
  a *simpler* version of this track's problem (binary "was there a
  repetition," not fragment-content preservation) is genuinely hard.
- **Lin et al., "Acoustically Precise Hesitation Tagging Is Essential for
  End-to-End Verbatim Transcription Systems"** (SLaTE 2025 [B12]) —
  directly relevant, quantified evidence *for* Stage D's underlying
  premise: fine-tuning Whisper Large V3 Turbo (via LoRA) with
  acoustically-precise filler-word labels (inferred by an LLM from
  audio-transcript pairs) achieved 5.5% WER, an 11.3% relative
  improvement over training with hesitations simply removed (6.2% WER).
  Real, published, verified evidence that precise disfluency labeling
  can improve — not merely not-hurt — transcription quality, on a
  different disfluency type (filler words) than this track's own focus.
- **Mujtaba et al., "Lost in Transcription: Identifying and Quantifying
  the Accuracy Biases of Automatic Speech Recognition Systems Against
  Disfluent Speech"** (NAACL 2024 [B13]) — evaluated multiple ASR
  systems and found "a consistent and statistically significant accuracy
  bias across all ASRs against disfluent speech" — independent,
  published corroboration of this track's own Stage A finding that this
  is not a CrisperWhisper-specific problem, at a broader scale (multiple
  systems, not the three this track tested directly).
- **Gulzar et al., "On the Difficulty of Token-Level Modeling of
  Dysfluency and Fluency Shaping Artifacts"** (ASRU 2025 [B14]) —
  independently confirms the core premise motivating this entire track:
  "dysfluencies and fluency-shaping artifacts are often overlooked,
  resulting in non-verbatim transcriptions with limited clinical and
  research value." Proposes parameter-efficient adaptation with
  multi-step fine-tuning and language-adaptive pretraining — another
  real precedent for what a Stage D attempt's engineering shape could
  look like.
- **YOLO-Stutter** (Zhou et al., Interspeech 2024 [B15]) and **SSDM:
  Scalable Speech Dysfluency Modeling** (NeurIPS 2024 [B16]) — both
  **audio-native, ASR-bypassing** dysfluency detectors, the same general
  strategy this track's own direction (g) pursued, but with genuinely
  more capable machinery: YOLO-Stutter performs end-to-end, time-accurate
  region-wise boundary and class prediction directly from imperfect
  speech-text alignment; SSDM uses articulatory gestures as forced
  alignment plus a "connectionist subsequence aligner" and a large-scale
  *simulated* dysfluency corpus (Libri-Dys — the same "synthetic
  injection" strategy as LibriStutter). **This is a real, honest
  finding for this track's own novelty assessment**: a more
  sophisticated, *trained/learned* acoustic-native detector (as opposed
  to this track's hand-engineered RMS/ZCR/MFCC candidate generator) is
  an actively published, credible approach this track did not test —
  see "Open Research Gap and Novelty Assessment" below for what this
  means for the novelty claim specifically.
- **CrisperWhisper's own paper** (Zusag, Wagner, Thallinger, Interspeech
  2024 [B2]) — explicitly targets accurate verbatim timestamps via
  retokenization and attention-based alignment; this track's own reading
  of it (established in an earlier session) found no claim that it
  specifically preserves sub-word fragment-level repetition evidence —
  consistent with, not contradicted by, this track's own Stage A finding.

### Part 5 — Detecting vs. preserving: a distinction the literature itself treats carefully, which this track must not blur

Checked directly across every system reviewed above; classified by
what it actually does, not what a headline might imply:

| System / paradigm | (A) Clean text only | (B) Preserves disfluencies in transcript | (C) Detects disfluencies separately (not in the transcript) | (D) Preserves acoustic evidence for downstream analysis | (E) Preserves timing/phonetic evidence | (F) Disfluency explicit in the training objective |
|---|---|---|---|---|---|---|
| Stock Whisper / CrisperWhisper (default use) | Yes (stock) / Partial (Crisper verbatim mode) | No (stock) / Partial, word-level (Crisper) | No | No | Word-level only | No |
| Kordt et al. 2026 [B11] | No | Yes, but only a coarse 4-category marker, not fragment content | N/A (markers are in-transcript) | No | No finer than word-level | Yes, explicitly |
| Lin et al. 2025 [B12] | No | Yes, filler-word content specifically | N/A | No | Not specified beyond word-level | Yes |
| SEP-28k/KSoF-style classifiers (clip-level detection) | N/A (no transcript produced) | N/A | Yes | No (clip-level presence only, no fragment boundary) | No | Yes (as a separate detection task, not ASR) |
| YOLO-Stutter [B15] / SSDM [B16] | N/A | N/A | Yes, with time-accurate boundaries | Partially — boundary-level, not full waveform evidence | Yes, region-wise | Yes, explicitly, audio-native |
| Neural-codec acoustic-token LMs (AudioLM/Moshi-style) [B9] | N/A (not built for transcription) | N/A | No (not their objective) | Yes, by design (full acoustic reconstruction) | Yes | No — not disfluency-specific, preserves everything acoustic indiscriminately |
| This track's own `block`/`prolongation` detectors | N/A | N/A | Yes | Yes (waveform-native) | Yes | Yes, by construction (acoustic-native, no ASR text involved) |
| This track's own direction (g) (RMS/ZCR, MFCC) | N/A | N/A | Attempted (Failure on precision) | Yes (waveform-native) | Yes | Yes, by construction |

**The specific distinction the project owner asked to guard against**: a
system that *detects* stuttering (column C) is not automatically
equivalent to an ASR whose internal representation *preserves* what a
downstream audio-native detector needs (column D/E). SEP-28k/KSoF-style
classifiers and YOLO-Stutter/SSDM both do real, published detection
work, but neither is evidence that a general-purpose ASR's own
representation preserves this information as a side effect — that
question is precisely what this track's Stages B/C and Phase 2 tested
directly, and found largely negative.

### Sound_repetition specifically — a dedicated pass

Checked directly whether the literature reports the same "disappears
even when the surrounding transcript looks correct" pattern this
track's own Stage A found:

- No source reviewed reports a direct, controlled experiment isolating
  *sound_repetition specifically* (as opposed to disfluency generally)
  and measuring whether its literal sub-word fragment survives
  real-ASR decoding at positions otherwise transcribed correctly. This
  track's own Stage A appears to be the most granular, controlled
  investigation of exactly this question found in this review.
- The closest indirect evidence: Kordt et al. 2026's `REP` marker
  collapses phoneme-level (sound_repetition-equivalent) and word-level
  repetitions into one token — an implicit acknowledgment that even
  recent, dedicated disfluency-aware fine-tuning work has not yet
  separated these as distinct preservation targets at the granularity
  this project's own taxonomy uses.
- YOLO-Stutter/SSDM's taxonomies (repetition, block, missing,
  replacement, prolongation) are pitched at a similar granularity to
  this project's own 5-7 type taxonomy, and both are audio-native
  (bypass ASR text entirely) — consistent with, not contradicting, this
  track's own inference (from Stage A/Phase 2's combined results) that
  the transcript-text layer specifically is where this information is
  lost, and that audio-native approaches are the more promising
  *general* strategy, even though this track's own specific
  hand-engineered attempt (direction (g)) did not reach usable
  precision.
- **No source reviewed reports evidence of ASR systems *hallucinating*
  spurious repeated fragments** (as opposed to omitting real ones) —
  this track's own decoding-sensitivity experiment (`num_beams`) is
  consistent with omission/normalization being the dominant failure
  mode for this specific type, not fabrication.

## How Our Experimental Findings Fit the Existing Literature

| This track's finding | Relationship to prior work | Basis |
|---|---|---|
| Broad information loss despite apparently correct transcription (Stage A) | **Independently supported**, at a broader scale | Mujtaba et al. 2024 [B13] found a "consistent and statistically significant accuracy bias across all ASRs" against disfluent speech — a multi-system, published finding consistent with this track's own single-system (CrisperWhisper), fragment-granular result |
| `sound_repetition` specifically disappearing even at correctly-transcribed positions | **Not previously investigated at this granularity, as far as this review found** | No source reviewed isolates sub-word fragment preservation specifically; the closest work (Kordt et al. 2026 [B11]) collapses it into a coarser category |
| CrisperWhisper normalizes disfluencies despite verbatim-timestamp design | **Partially supported / not contradicted** | CrisperWhisper's own paper [B2] targets timestamp accuracy, not fragment preservation — this track's finding is a real gap in what the system was built to do, not a bug relative to its own stated goals |
| Stock Whisper behaves the same or worse than CrisperWhisper on this problem | **Not previously investigated directly, consistent with the general "disfluencies are normalized" finding** | No source reviewed directly compares CrisperWhisper against stock Whisper on fragment-level preservation; consistent with [B13]'s general multi-system bias finding |
| Representation-level evidence exists in CrisperWhisper's encoder but is precision-limited alone | **Not previously investigated for this exact task**, but the general phenomenon (self-supervised/ASR encoders carrying more than the decoded text) is well established | This is exactly the premise behind wav2vec2/HuBERT/WavLM's own design and behind YOLO-Stutter/SSDM's audio-native strategy [B6][B7][B8][B15][B16] — this track's finding is a specific instance of a broadly-supported general phenomenon |
| WavLM's representation does not carry a stronger `sound_repetition` signal despite its paralinguistic-sensitivity objective | **Not previously investigated directly** — no source reviewed tests WavLM specifically against this taxonomy; **in tension with** WavLM's own stated design goal [B8], an honest, unresolved tension this track has named rather than smoothed over, confounded by the untested model-size difference |
| Acoustic evidence for `sound_repetition` exists outside the transcript (recall=0.824) but isn't cheaply separable from background speech rhythm | **Consistent with, and less capable than,** the published audio-native detection literature — YOLO-Stutter/SSDM [B15][B16] demonstrate that *trained, learned* acoustic-native detectors can do more than this track's hand-engineered features attempted, without contradicting this track's finding that the *specific simple features tried* were insufficient |
| Limitations of post-hoc recovery (Arms 1-3, direction (g), combined classifier) | **Consistent with** the field's general move toward training-time intervention (explicit disfluency tokens, continual learning) rather than post-hoc recovery from off-the-shelf representations — Kordt et al. 2026 [B11], Lin et al. 2025 [B12], and Gulzar et al. 2025 [B14] all take the training-time-intervention path this track's evidence also points toward |
| The combined-signal classifier's failure to beat the best individual signal | **Not directly comparable to any single prior result** — no source reviewed reports a similar single-strong-signal-plus-weaker-signals combination test for this exact task; the *general* finding that naive feature combination doesn't automatically beat a strong individual signal is unsurprising and not specific to this domain |

**Where this track's results conflict with prior work, stated plainly**:
Arm 3's negative result for WavLM sits in real tension with WavLM's own
stated design rationale [B8] and with this track's own earlier-cited
hybrid-approach precedent (arXiv:2605.12387, a Whisper+acoustic-feature
hybrid beating both pure-Whisper and pure self-supervised baselines on
a different paralinguistic task). Plausible reconciling factors (task
difference, sample size, the untested model-size confound) are named
as inferences in this track's own Phase 2 write-up, not resolved as
fact — the honest reading remains that self-supervised representations
*can* help for paralinguistic tasks generally, without that generalizing
to this specific task at this specific scale.

## Open Research Gap and Novelty Assessment

**Are we standing in an underexplored technical area, or rediscovering
something already solved?**

**Answer, stated directly**: this is a **genuinely narrow, real gap
within a broader, actively-researched area** — not an unsolved area of
technology, and not a solved problem this track failed to find. The
correct framing is precise, not sweeping in either direction.

**What has already been substantially solved** (high confidence, direct
literature evidence):
- Detecting the *presence* of stuttering-type disfluencies from audio,
  at the clip or region level, with real accuracy — SEP-28k/KSoF-trained
  classifiers, YOLO-Stutter, SSDM, and this project's own `block`/
  `prolongation` detectors all demonstrate this works.
- Recognizing that off-the-shelf ASR normalizes disfluencies and that
  this is a real, general problem — extensively documented ([B13],
  [B14], and this track's own Stage A).
- That training-time intervention (explicit disfluency tokens,
  fine-tuning) can improve verbatim preservation for *some* disfluency
  types (fillers specifically — [B12]'s quantified WER improvement) —
  demonstrated, published, real.

**What has been partially solved** (medium confidence):
- Preserving repetitions in ASR output at all — Kordt et al. 2026 [B11]
  does this, but only as a coarse, undifferentiated marker, with a
  documented, real accuracy ceiling (F1 0.35-0.63) and a documented
  catastrophic-forgetting trade-off even at that coarser granularity.
- Audio-native, ASR-bypassing dysfluency detection generally — real,
  published, working systems exist (YOLO-Stutter, SSDM), but none
  reviewed specifically targets or reports results for sub-word
  fragment-level preservation as this project's `sound_repetition` type
  defines it.

**What appears genuinely underexplored** (the actual gap, stated
precisely): **preserving or recovering sub-word, fragment-level
repetition evidence (this project's specific `sound_repetition`
definition) — as distinct from word-level repetition or generic
"a repetition happened here" markers — at positions where surrounding
context is otherwise transcribed correctly.** No source found in this
review isolates this exact sub-problem, tests it directly, or reports a
solution to it. This is a narrow, well-defined gap *within* a real,
active research area, not a claim that the area itself is unexplored.

**What remains completely uncertain**: whether this gap is narrow
because it is genuinely hard (a real scientific open question) or
because it is a small enough slice of the broader disfluency-modeling
problem that no one has prioritized it specifically yet (a research-
attention gap, not necessarily a difficulty gap) — this review cannot
distinguish between these two explanations, and neither should be
assumed.

**Confidence levels, stated explicitly**:
- High confidence: the specific fragment-level preservation gap is real
  and not directly addressed by any single source reviewed.
- Medium confidence: this gap is scientifically meaningful (not just an
  artifact of this project's own specific taxonomy choices) — supported
  by the fact that independent published work (Kordt et al.) also
  collapsed this same distinction rather than resolving it, suggesting
  it's a real, shared modeling difficulty, not an idiosyncratic framing.
- Low confidence / genuinely unknown: whether solving it would require
  a fundamentally different approach (Stage D, or something not yet
  identified) or "merely" more targeted training data and taxonomy
  design within the disfluency-token-fine-tuning paradigm [B11][B12]
  that already exists and already works for coarser categories.

**Explicitly not claimed, per the project owner's instruction**: this
document does not use "groundbreaking," "first-ever," or "world-first."
The defensible claim is: *a real, specific, narrow gap exists within an
active, real research area, evidenced by both this track's own
experiments and by the fact that the closest published precedent
(Kordt et al. 2026) chose not to resolve exactly this distinction.*

## Existing Approaches vs. Our Proposed Stage D

| Existing approach | Objective | What it preserves | What it discards/normalizes | Handles `sound_repetition` specifically? | Suitable for our task as-is? | Relevant limitation |
|---|---|---|---|---|---|---|
| Stock Whisper / CrisperWhisper (as tested, Arms 1-2) | Fluent or verbatim-timestamped transcription | Word-level content and timing when correct | Sub-word fragments, this track's own direct finding | No — directly tested, Failure | No | Trained toward general transcription, not this taxonomy |
| WavLM (as tested, Arm 3) | Masked prediction + denoising, paralinguistic-aware pretraining | General paralinguistic/speaker detail | Not disfluency-specific either way | No — directly tested, chance-level | No | Untargeted at this taxonomy; confounded with model size |
| Kordt et al. 2026's disfluency-token fine-tune [B11] | Verbatim transcription with explicit disfluency markers via continual learning | A coarse repetition marker (word- and phoneme-level combined) | The repeated fragment's own content/boundaries; sound- vs. word-level distinction | No — by design, `REP` doesn't distinguish these | Partially — closest existing precedent, but not this taxonomy's granularity | Real catastrophic-forgetting trade-off, even at coarser granularity |
| Lin et al. 2025's acoustically-precise filler tagging [B12] | Verbatim transcription with acoustically-precise disfluency labels | Filler-word content and acoustic precision | Not evaluated for repetitions at all | No — different disfluency type entirely | No — wrong disfluency type | Real, quantified evidence the *general strategy* (precise labels + fine-tuning) can work |
| YOLO-Stutter / SSDM [B15][B16] | Audio-native, time-accurate dysfluency region detection, bypassing ASR text | Region-level boundaries and class, for a similar type taxonomy to this project's | Full waveform-level acoustic detail beyond the detected region | Plausibly, by taxonomy, but not verified against this project's exact fragment-boundary definition | Unclear — most capable related system found, never tested against this project's own data | A more capable, *learned* version of what direction (g) hand-engineered; the strongest candidate "simpler alternative" this review surfaced |
| Neural-codec acoustic-token LMs [B9] | Full acoustic-detail preservation via discrete codec tokens, not transcription-oriented at all | Everything acoustic, indiscriminately | Nothing disfluency-specific — it isn't disfluency-aware, it's acoustically lossless | Not evaluated — orthogonal paradigm | Unclear — would need a disfluency-detection head on top, untested | A genuinely different technical direction, not directly comparable |
| **Our proposed Stage D** | Fine-tune (or adapt) a model with a loss function that directly rewards preserving `sound_repetition`/`word_repetition` fragment-level evidence specifically | The exact information this track's own Stage A showed is normalized away — targeted at this project's own taxonomy's granularity | Whatever the training data/objective doesn't cover — real risk if data is synthetic (LibriStutter) | Yes, by explicit design intent — untested | N/A — not yet built | Blocked on infrastructure/data (see below), and its core premise is a reasoned hypothesis, not proven |

**What Stage D really is, precisely**: not "a new ASR" in the sense of a
new architecture from scratch — every published precedent reviewed
(Kordt et al., Lin et al., Gulzar et al.) uses the same strategy Stage D
would: **adapt an existing pretrained ASR (most plausibly CrisperWhisper
or stock `whisper-large-v3`, matching this track's own tested
checkpoints) with an auxiliary or modified objective that explicitly
rewards preserving this project's own taxonomy's fragment-level
evidence**, most likely via explicit disfluency/fragment tokens (Kordt
et al.'s approach) combined with continual-learning or parameter-
efficient methods (LoRA, per Lin et al.) to control catastrophic
forgetting (per this track's own Phase 2 literature review and Kordt et
al.'s directly-quantified trade-off).

**The most scientifically accurate terminology, recommended**:
**"disfluency-preserving ASR fine-tuning with a fragment-level auxiliary
objective"** — more precise than "purpose-built ASR" (which implies
architecture-from-scratch, not what any reviewed precedent or this
track's own plan actually proposes) and more specific than "disfluency-
aware ASR" (the term Kordt et al. and others already use for a coarser
granularity than this project's own taxonomy targets). "Multi-objective
ASR" or "ASR with an auxiliary disfluency objective" are both also
defensible, more generic alternatives; "a new ASR" alone should be
avoided as imprecise given the evidence above.

## Assessing Potential Significance

**Only where the evidence above actually supports it**, separated by
category, not marketing language:

- **Scientific novelty**: real but narrow — a specific, well-evidenced
  gap (fragment-level `sound_repetition` preservation) within an active
  field, not a new field or a solved-but-overlooked problem.
- **Technical novelty**: low-to-moderate — the *strategy* Stage D would
  use (auxiliary disfluency tokens + continual/parameter-efficient
  fine-tuning) is already published and working for coarser categories
  [B11][B12][B14]; the novelty would be in the specific taxonomy
  granularity and this project's own accumulated negative-result
  evidence base justifying exactly where to target it, not in a new
  method.
- **Engineering improvement**: real and concrete *if* Stage D succeeds —
  this project's own downstream detector (`detect.py`) already depends
  on exactly this class of evidence, so a model that preserved it would
  be directly, immediately useful to this project's own stated
  objective, not just an abstract research contribution.
- **Potential practical impact, if the gap is closed**: plausible,
  stated as conditional, not asserted — better fragment-level disfluency
  preservation could matter for stuttering detection accuracy, speech-
  fluency assessment tools, assistive/inclusive speech interfaces for
  people who stutter, and speech-language-pathology research tooling
  (echoing SEP-28k's [B17] and KSoF's [B18] own stated motivations for
  exactly these audiences) — a realistic, not overstated, set of
  downstream beneficiaries *if* the technical gap is actually closed,
  which remains untested.

## Stage D Requirements / Re-entry Gate

This formalizes and extends the "Stage D planning (2026-08-07)" section
above (its infrastructure finding — no local GPU, `torch.cuda.is_
available()` is `False` — and its risk analysis are not repeated here;
this section adds the literature-grounded specifics that section
predates) into the concrete readiness specification the project owner
requested.

### The Stage D "wall" — what we have vs. what we don't

**WHAT WE ALREADY HAVE**:
- 12 completed, documented, pre-registered experiments (the full table
  in "Experimental Phase Stopping Point" above), each with real
  measured results, honest negative or mixed outcomes, and — where
  applicable — caught-and-fixed implementation bugs.
- A working evaluation harness (`profiling/evaluation/`) with real,
  reusable primitives: encoder-distance extraction (`profiling/
  encoder_embedding.py`), acoustic candidate generation (`stage_g_
  acoustic_sound_repetition.py`), a nested-CV logistic-regression
  classifier infrastructure (`compare_corroboration_mechanisms.py`),
  Track A/B ASR-vs-ground-truth alignment and scoring (`profiling/
  evaluation/alignment.py`, `track_b.py`).
- Real datasets already on disk: LibriStutter (499/4,736 clips
  downloaded), with cached CrisperWhisper transcriptions and encoder
  states for the working 120-clip sample.
- A validated, shipped downstream detector (`profiling/detect.py`) that
  already consumes exactly the kind of evidence Stage D would aim to
  preserve — meaning a successful Stage D model has an existing,
  real integration point, not a hypothetical one.
- This literature review itself — 18 directly-verified sources spanning
  foundational ASR paradigms through the closest published Stage D
  precedents, now part of the permanent record.

**WHAT WE DO NOT CURRENTLY HAVE**:
- Any local GPU (confirmed directly: no CUDA, CPU-only `torch` build,
  integrated graphics only).
- A large-scale, real (non-synthetic) paired dataset with the specific
  fragment-level `sound_repetition`/`word_repetition` granularity this
  project's taxonomy needs — SEP-28k/KSoF have clip-level labels, not
  word-aligned verbatim transcripts; FluencyBank Timestamped is the
  closest fit but needs an unbuilt CHAT-format parser and possibly
  access approval.
- A training codebase — this project has an inference/evaluation
  harness, not a fine-tuning pipeline (data loaders, loss functions,
  checkpointing, continual-learning machinery of the kind Kordt et al.
  [B11] used).
- Any experience or tooling for the specific engineering the closest
  published precedents used (explicit token introduction into a
  pretrained decoder, continual-learning methods like EWC/Experience
  Replay/A-GEM/Weight Averaging, or LoRA-style parameter-efficient
  adaptation).

**Why this is an infrastructure/data boundary, not evidence the Stage D
hypothesis has failed**: nothing in today's or any prior experiment
tested Stage D's actual premise (a taxonomy-targeted training objective)
— every experiment tested *representations already trained for a
different objective*. The wall is about what resources are needed to
run the real test, not a result from having run it and lost.

### A. Data requirements

- **Type of speech data required**: real (non-synthetic) disfluent
  speech with word-level timestamps and disfluency annotations at
  `sound_repetition`/`word_repetition` granularity — i.e., annotations
  that distinguish a sub-word fragment repeat from a whole-word repeat,
  which none of the datasets reviewed provide off-the-shelf at scale
  (Kordt et al.'s own `REP` category collapses exactly this
  distinction, per the literature review above).
- **Realistic scale**: the closest published precedent (Kordt et al.
  [B11]) achieved real, if imperfect, marker learning with ~10-20 hours
  per dataset (SME: 11.79h, Delaware: 9.72h, a 12.11h subset of Pitt) —
  this recalibrates what "large-scale" plausibly means for a coarse-
  grained disfluency-token fine-tune specifically; a fragment-level
  target likely needs more annotated examples per category given the
  finer granularity, but not necessarily orders of magnitude more.
- **Required annotations**: word-level timestamps (already this
  project's own pipeline's native format) plus a fragment-vs-word
  repetition distinction at each disfluent position — not present in
  SEP-28k, KSoF, or FluencyBank's existing label schemas as reviewed;
  would likely require either new annotation work on an existing corpus
  or a taxonomy-mapping/relabeling pass.
- **Why LibriStutter should not become the primary training corpus**:
  it is synthetic splicing (real words/synthetic-splice audio grafted
  in), not real disfluent speech — training a model to *predict*
  something a splice operation inserted risks the model learning splice
  artifacts (edit-point discontinuities, source-mismatch acoustic cues)
  rather than genuine disfluency production patterns. This is a
  materially worse risk for a training target than it has been for
  this track's own evaluation-only use of it throughout, because a
  training objective actively optimizes toward whatever regularity is
  present in the data, artifact or not.
- **Real datasets that may be suitable**: FluencyBank Timestamped
  [B19] is the best fit identified (real people who stutter, word-level
  timestamps, disfluency labels, explicitly designed to enable "thorough
  analysis of how speech processing models perform when evaluated with
  typical speech versus speech from people who stutter") — but needs a
  CHAT-format parser (`ROADMAP.md` item 16, not built) and possible
  access request. SEP-28k [B17] is the field's most-used benchmark but
  has no word-level transcript, only clip-level presence labels —
  usable for clip-level validation, not directly for fragment-level
  fine-tuning targets without substantial additional work. KSoF [B18]
  is German-language and request-access-gated — a real option only if
  cross-lingual scope becomes an explicit goal.
- **Acquisition/licensing/access work needed**: SEP-28k is CC BY-NC 4.0
  (non-commercial) [B17] — a real licensing constraint to check against
  this project's own intended use before relying on it further.
  FluencyBank access terms need direct confirmation (unconfirmed by this
  review). Any of these paths is real, scoped, multi-hour-to-multi-day
  work, not something resolved by this planning pass.
- **Would custom data collection be necessary?** Plausibly, if no
  existing corpus provides the fragment-vs-word distinction at adequate
  scale — this is a real, open, and expensive possibility this planning
  pass surfaces rather than resolves.

### B. Compute requirements

- **Minimum viable research setup**: a single mid-range cloud GPU
  (e.g. a 16-24GB-VRAM instance) would likely suffice for a `whisper-
  small`-or-`base`-scale fine-tune following Kordt et al.'s own
  demonstrated recipe (10 epochs, batch size 16, lr 2e-5, real but
  modest real-world compute) — this is a real, published existence
  proof that a *scaled-down* Stage D pilot doesn't require frontier-
  scale infrastructure.
- **Recommended setup**: if targeting CrisperWhisper or `whisper-
  large-v3` scale specifically (to stay consistent with this track's
  own already-tested checkpoints), a larger-VRAM instance (24-48GB+) or
  parameter-efficient adaptation (LoRA, as Lin et al. [B12] used) to
  keep the same modest compute envelope while still adapting the
  larger, already-characterized model.
- **GPU/VRAM requirements**: scale-dependent as above; not yet priced
  in dollar terms by this review — a real next-step cost estimate, not
  assumed here.
- **Would cloud GPU rental be sufficient?** Plausibly yes for a
  minimum-viable pilot, based on the published precedent's own
  demonstrated scale — this is an inference from a comparable published
  setup, not a confirmed cost for this project's own specific target.
- **Why the current CPU-only laptop is not sensible for the full Stage D
  workload**: fine-tuning requires forward *and* backward passes across
  many real training examples per epoch, repeated for multiple epochs —
  a categorically heavier workload than this project's own already-slow
  CPU inference (~54s/clip for a single forward pass); training at any
  reasonable dataset scale would take days-to-weeks on this hardware,
  not a viable "implement this session" undertaking.
- **What can still be done locally before cloud infrastructure is
  obtained**: data preparation and taxonomy-mapping work (converting
  whatever corpus is chosen into this project's own word-level format);
  writing and testing the training pipeline's data-loading and
  evaluation code (not the training loop's actual GPU-bound execution)
  against a tiny local subset; a full pre-registration of the Stage D
  experiment itself, exactly as this track has done for every prior
  stage.

### C. Model / engineering requirements

- **Input**: raw audio (matching this project's existing 16kHz mono WAV
  pipeline).
- **Output**: word-level verbatim transcription with an explicit,
  fragment-granular disfluency marker distinguishing `sound_repetition`
  from `word_repetition` at minimum (going beyond Kordt et al.'s
  coarser `REP` category, per this track's own taxonomy).
- **Architecture**: most likely a fine-tune of an existing pretrained
  checkpoint this track has already characterized (CrisperWhisper or
  stock `whisper-large-v3`), not a from-scratch architecture — matching
  every published precedent reviewed and this track's own recommended
  terminology ("disfluency-preserving ASR fine-tuning with a
  fragment-level auxiliary objective," not "a new ASR").
- **Training objective**: an auxiliary/modified loss combining standard
  transcription loss with explicit fragment-level disfluency-token
  prediction, following Kordt et al.'s [B11] general strategy but at
  finer taxonomy granularity, combined with a continual-learning or
  parameter-efficient method (EWC/Experience Replay/A-GEM/Weight
  Averaging, or LoRA) to control the catastrophic-forgetting trade-off
  both Kordt et al. and this track's own earlier literature review
  documented.
- **How disfluency information would be preserved**: as explicit output
  tokens carrying the fragment's own content (not just a presence
  marker), directly targeting what Stage A found is normalized away.
- **How this differs from ordinary ASR**: ordinary ASR (including
  CrisperWhisper, per this track's own direct evidence) is trained
  toward fluent/verbatim-but-not-fragment-preserving output; Stage D's
  objective would explicitly reward the opposite for this taxonomy.
- **What can be reused from this project**: the entire evaluation
  harness (Track A/B scoring, `alignment.py`, encoder-distance
  extraction), the existing 120-clip working sample and its cached
  artifacts (as a cheap prototyping/smoke-test set, explicitly not a
  primary training or claimed-generalization source, per the
  synthetic-data risk above), and the downstream detector
  (`profiling/detect.py`) as the real integration target.
- **What must be built from scratch**: the training pipeline itself
  (data loading, loss function, training loop, checkpointing), any new
  data-loading code for whichever real dataset is chosen, and the
  fragment-level taxonomy-mapping/annotation work.

### D. Evaluation requirements

- **Baseline systems**: this track's own already-measured numbers
  become the baselines to beat, not hypothetical ones — CrisperWhisper's
  own 45.2%/40.5% normalized-away rate (Stage A), stock Whisper's
  89.5%/88.2% (Arm 1), and the combined classifier's F1=0.242 —
  ensuring Stage D's evaluation is directly comparable to everything
  already measured, not a fresh, incomparable metric.
- **Evaluation datasets**: whatever real dataset trains the model must
  have a genuinely held-out, speaker-disjoint test split (matching
  Kordt et al.'s own methodology [B11]); LibriStutter remains usable as
  a secondary, synthetic-data sanity check (explicitly not the primary
  claim of generalization).
- **Metrics**: this track's own established set — Track A/B-style
  normalized-away rate, precision/recall/F1 at the fragment level, plus
  general transcription WER to measure the catastrophic-forgetting
  trade-off directly (matching Kordt et al.'s pWER/marker-F1 pairing).
- **Ablations**: with/without the auxiliary objective; with/without
  continual-learning or parameter-efficient adaptation (isolating
  whether the objective or the adaptation method drives any
  improvement); by disfluency type (`sound_repetition` vs.
  `word_repetition` specifically, not pooled).
- **Transcription-quality tests**: general WER on clean, non-disfluent
  held-out speech (LibriSpeech-style), to directly measure catastrophic
  forgetting, not just assume its absence.
- **Disfluency-preservation tests**: this track's own Stage-A-style
  4-category classification, re-applied to the fine-tuned model's
  output, at the same 36 positions (or a real-data equivalent) this
  track has used throughout — a direct, apples-to-apples comparison to
  every prior arm.
- **`sound_repetition`-specific tests**: recall/precision at fragment-
  content correctness specifically, not just "was a repetition marker
  emitted" (going beyond Kordt et al.'s own coarser evaluation).
- **Criteria for claiming meaningful success**: a pre-registered bar
  (to be set when the experiment is actually pre-registered, not
  guessed here) requiring both a real reduction in the normalized-away
  rate *and* a bounded, explicitly-tolerated WER cost on clean speech —
  mirroring exactly the trade-off Kordt et al.'s own results made
  visible, so this project goes in with eyes open rather than
  discovering the trade-off mid-experiment.

### Stage D Re-entry Gate

**DO NOT START STAGE D UNTIL:**
1. Real GPU compute is secured (cloud rental or otherwise) and its cost
   is known and approved.
2. A real, word-level-annotated, fragment-granular (or a defensible
   plan to derive fragment-granularity from a coarser real corpus)
   dataset is identified and access/licensing is confirmed — FluencyBank
   Timestamped's CHAT parser and access terms, or an equivalent, resolved
   first.
3. A full, dedicated pre-registration exists (exact architecture,
   objective, data split, metrics, success/failure criteria) — this
   section is the readiness specification, not that pre-registration
   itself.
4. The project owner has explicitly reviewed and approved the cost
   (compute + data acquisition + engineering time) against the
   assessed payoff in the "Stage D planning (2026-08-07)" section above.

**READY TO START STAGE D WHEN:**
1. All four items above are satisfied.
2. A minimum-viable pilot scope is agreed (e.g., `whisper-small`-scale,
   FluencyBank-Timestamped-only, `sound_repetition`-only, matching this
   track's own repeated "cheapest first" discipline) rather than
   jumping directly to full CrisperWhisper-scale, full-taxonomy scope.

**First steps once the gate is satisfied** (research thinking before
implementation, exactly as this track has done throughout): re-review
this entire audit with fresh eyes; confirm the minimum-viable pilot
scope explicitly; write the full pre-registration (exact architecture,
loss function, data split, metrics, success/failure criteria,
confounders, cost) *before* writing any training code; only then begin
implementation, incrementally, with the same self-test-before-real-data
discipline every prior stage in this track has used.

## Venture / Research Collaboration Brief

**What did we discover?** Modern ASR systems — including a model
(CrisperWhisper) specifically fine-tuned for verbatim, disfluency-
preserving transcription — reliably normalize away roughly half of a
specific class of stuttering-relevant evidence (`sound_repetition`,
`word_repetition`) even at positions where the surrounding transcript is
otherwise correct. This isn't a transcription-accuracy problem in the
usual sense; it's closer to a design bias — the very thing that makes a
transcript "look clean" removes exactly the evidence this project's
downstream disfluency detector needs.

**Why is the problem difficult?** Every representation we tested —
CrisperWhisper's own text and internal encoder state, a materially
bigger non-fine-tuned Whisper model, and a model built with an explicitly
different, more paralinguistically-aware training objective (WavLM) —
carries the same limitation to some degree. It's not a bug in one
model; it's a consequence of what these models are trained to reward.

**What existing technologies did we investigate?** Foundational ASR
paradigms (Whisper, CTC, RNN-T, Conformer, wav2vec 2.0, HuBERT, WavLM),
forced-alignment systems, streaming ASR, discrete-audio-token speech
foundation models, and — most directly relevant — the published
disfluency-aware and disfluency-preserving ASR literature (2021-2026),
including very recent (2025-2026) work that fine-tunes ASR models with
explicit disfluency tokens.

**What did we test directly?** Three representations (CrisperWhisper,
stock `whisper-large-v3`, WavLM-Large), one decoding parameter (beam
width), two hand-engineered acoustic-only detection features (RMS/ZCR
and MFCC similarity), and one trained combination of the strongest
individual signals — 12 pre-registered experiments in total, each with
a real, honestly-reported result.

**What did they fail to preserve or recover?** All of them, to varying
degrees, failed to preserve or recover the specific sub-word fragment
evidence `sound_repetition` requires. The strongest single result we
found (CrisperWhisper's own encoder-distance signal) ranks a true
instance above a false one about 72% of the time, but only reaches
~34% precision at a useful recall against real candidate positions —
real, but not yet shippable alone, and combining it with our best
acoustic signal didn't improve on it.

**What does the literature say?** This is a real, active, published
research area — not an unexplored one. The closest published precedent
(a 2026 paper fine-tuning Whisper with continual learning and explicit
disfluency tokens) confirms both the general problem (ASR normalizes
disfluencies) and the general difficulty (a real, measured trade-off
between disfluency-marker accuracy and general transcription quality) —
but even that work collapses sub-word and word-level repetitions into
one coarse marker, at F1 0.35-0.63. No source we found isolates and
solves fragment-level `sound_repetition` preservation specifically.

**What exactly remains open?** Whether a model fine-tuned with an
objective targeted specifically at this project's own taxonomy
granularity (not the coarser categories already published) can do
better — a reasoned, literature-consistent hypothesis, not yet tested by
anyone, as far as this review found.

**Why can't we responsibly complete Stage D on the current setup?**
Two hard, directly-verified facts: no local GPU (confirmed via
`torch.cuda.is_available() == False`, a CPU-only PyTorch build, and
integrated-only graphics hardware), and no large-scale real (non-
synthetic) dataset with the specific fragment-level annotation this
taxonomy needs. Attempting it anyway — on inadequate compute, with
synthetic data — risks producing a confidently wrong result, which this
project's own accumulated discipline treats as worse than an honest
"not yet."

**What data do we need?** Real (not synthetic) disfluent speech with
word-level timestamps and a fragment-vs-word repetition distinction —
most plausibly built from FluencyBank Timestamped, which needs an
unbuilt parser and confirmed access, or a comparable real corpus.

**What compute do we need?** Realistically, a single cloud GPU instance
for a minimum-viable pilot at reduced model scale — the closest
published precedent achieved real results with modest compute (10
epochs, ~10-20 hours of audio per dataset) on a small Whisper variant,
which meaningfully lowers the likely entry cost below "frontier-scale
training."

**What would we build?** Not a new ASR architecture — a fine-tune of an
existing, already-characterized checkpoint (CrisperWhisper or stock
`whisper-large-v3`), adapted with an auxiliary training objective that
explicitly rewards preserving fragment-level repetition evidence,
using continual-learning or parameter-efficient methods to control the
same catastrophic-forgetting trade-off the closest published work
already measured. The accurate technical name is "disfluency-preserving
ASR fine-tuning with a fragment-level auxiliary objective," not "a new
ASR."

**What would success look like?** A measurable reduction in the
normalized-away rate for `sound_repetition`/`word_repetition` (below
CrisperWhisper's own 45.2%/40.5% baseline) at a bounded, explicitly
tolerated cost to general transcription accuracy — evaluated against
this project's own existing, already-validated harness, so the result
would be directly comparable to everything measured so far, not a
fresh, incomparable number.

**Why is this scientifically/technically interesting, without
overselling it?** Because the accumulated evidence — twelve independent,
pre-registered, honestly-reported experiments, several of which caught
and fixed real implementation bugs before trusting their own results —
converges cleanly on a narrow, well-defined, real gap that the closest
published work has not yet closed, in a domain (assistive/inclusive
speech technology for people who stutter) where the practical stakes of
closing it are real, if the technical premise holds. That premise is a
reasoned hypothesis grounded in how supervised learning generally works
and in real published precedent for the general strategy — not a proven
result, and this document does not claim otherwise.

## Bibliography

Every source below was fetched and read directly during this or an
earlier session before being cited in this document — not cited from a
search-result snippet alone. `[B#]` markers above refer to this list;
sources cited elsewhere in this document by bare arXiv ID (established
in earlier sessions, per this project's own citation convention) are
also included here for a single, complete index.

**Foundational ASR/speech-representation paradigms**
- [B1] Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., &
  Sutskever, I. (2023). *Robust Speech Recognition via Large-Scale Weak
  Supervision.* Proceedings of the 40th International Conference on
  Machine Learning (ICML 2023), 28492-28518.
  https://arxiv.org/abs/2212.04356
- [B3] Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006).
  *Connectionist Temporal Classification: Labelling Unsegmented Sequence
  Data with Recurrent Neural Networks.* Proceedings of the 23rd
  International Conference on Machine Learning (ICML 2006).
- [B4] Graves, A. (2012). *Sequence Transduction with Recurrent Neural
  Networks.* ICML 2012 Workshop on Representation Learning.
  https://arxiv.org/abs/1211.3711
- [B5] Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y., Yu, J.,
  Han, W., Wang, S., Zhang, Z., Wu, Y., & Pang, R. (2020). *Conformer:
  Convolution-augmented Transformer for Speech Recognition.* Interspeech
  2020, 5036-5040. https://arxiv.org/abs/2005.08100
- [B6] Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). *wav2vec
  2.0: A Framework for Self-Supervised Learning of Speech
  Representations.* NeurIPS 2020. https://arxiv.org/abs/2006.11477
- [B7] Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov,
  R., & Mohamed, A. (2021). *HuBERT: Self-Supervised Speech
  Representation Learning by Masked Prediction of Hidden Units.*
  IEEE/ACM Transactions on Audio, Speech, and Language Processing.
  https://arxiv.org/abs/2106.07447
- [B8] Chen, S., Wang, C., Chen, Z., Wu, Y., Liu, S., Chen, Z., Li, J.,
  Kanda, N., Yoshioka, T., Xiao, X., Wu, J., Zhou, L., Ren, S., Qian,
  Y., Qian, Y., Wu, J., Zeng, M., Yu, X., & Wei, F. (2022). *WavLM:
  Large-Scale Self-Supervised Pre-Training for Full Stack Speech
  Processing.* IEEE Journal of Selected Topics in Signal Processing.
  https://arxiv.org/abs/2110.13900 (already cited in this track's
  earlier sessions; verified again this session)
- [B9] Borsos, Z. et al. (2023). *AudioLM: a Language Modeling Approach
  to Audio Generation.* IEEE/ACM TASLP. https://arxiv.org/abs/2209.03143
  — and Défossez, A. et al. (2024). *Moshi: a speech-text foundation
  model for real-time dialogue.* https://arxiv.org/abs/2410.00037
  (representative of the discrete-acoustic-token paradigm generally,
  not a claim that either directly addresses disfluency)
- [B10] (2026). *Montreal Forced Aligner and the state of speech-to-text
  alignment in 2026.* https://arxiv.org/abs/2606.18466

**CrisperWhisper (this track's primary tested checkpoint)**
- [B2] Zusag, M., Wagner, L., & Thallinger, B. (2024). *CrisperWhisper:
  Accurate Timestamps on Verbatim Speech Transcriptions.* Interspeech
  2024, Kos, Greece, 1265-1269. https://arxiv.org/abs/2408.16589
  (verified directly in an earlier session of this track; re-confirmed
  via this session's literature search)

**Disfluency datasets**
- [B17] Lea, C., Mitra, V., Joshi, A., Kajarekar, S., & Bigham, J. P.
  (2021). *SEP-28k: A Dataset for Stuttering Event Detection From
  Podcasts With People Who Stutter.* ICASSP 2021.
  https://arxiv.org/abs/2102.12394 — dataset:
  https://github.com/apple/ml-stuttering-events-dataset (CC BY-NC 4.0)
- [B18] Bayerl, S. P., von Zeddelmann, D., Riedhammer, K., et al.
  (2022). *KSoF: The Kassel State of Fluency Dataset — A Therapy
  Centered Dataset of Stuttering.* LREC 2022.
  https://arxiv.org/abs/2203.05383
- [B19] (2024). *FluencyBank Timestamped: An Updated Data Set for
  Disfluency Detection and Automatic Intended Speech Recognition.*
  Journal of Speech, Language, and Hearing Research.
  https://doi.org/10.1044/2024_JSLHR-24-00070
- Zhang, H., et al. *LibriStutter* — synthetically-stuttered speech
  derived from LibriSpeech (Panayotov, V., Chen, G., Povey, D., &
  Khudanpur, S. (2015). *Librispeech: An ASR corpus based on public
  domain audio books.* ICASSP 2015). Dataset:
  https://github.com/hhzhang16/LibriStutterData and
  https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/NKVOGQ
  (already the working corpus this entire track has used; formal
  citation confirmed this session)

**Disfluency-aware / disfluency-preserving ASR (closest precedents to
this track's own Stage D hypothesis)**
- [B11] Kordt, H.-L., Pekarek Rosin, T., Lee, J. H., & Wermter, S.
  (2026). *Learning to Hear Hesitation: Continual Learning for
  Disfluency-Aware ASR.* Interspeech 2026 (submitted 2026-06-12).
  https://arxiv.org/abs/2606.14391 — full text fetched and read
  directly this session, including all result tables.
- [B12] Lin, J.-K., Lu, H.-C., Wang, C.-C., Lin, H.-Y., & Chen, B.
  (2025). *Acoustically Precise Hesitation Tagging Is Essential for
  End-to-End Verbatim Transcription Systems.* 10th Workshop on Speech
  and Language Technology in Education (SLaTE 2025), 163-166.
  https://arxiv.org/abs/2506.04076
- [B13] Mujtaba, D., Mahapatra, N. R., Arney, M., Yaruss, J. S.,
  Gerlach-Houck, H., Herring, C., & Bin, J. (2024). *Lost in
  Transcription: Identifying and Quantifying the Accuracy Biases of
  Automatic Speech Recognition Systems Against Disfluent Speech.*
  NAACL 2024, 4795-4809. https://arxiv.org/abs/2405.06150
- [B14] Gulzar, K., Wagner, D., Bayerl, S. P., Hönig, F., Bocklet, T., &
  Riedhammer, K. (2025). *On the Difficulty of Token-Level Modeling of
  Dysfluency and Fluency Shaping Artifacts.* ASRU 2025.
  https://arxiv.org/abs/2512.02027 (previously flagged in this track as
  "not yet deep-read"; fetched and read directly this session)
- arXiv:2606.14391's own cited precedents, not independently re-verified
  this session but relevant context: MacDonald, R. L. et al. (2021).
  *Disordered Speech Data Collection: Lessons Learned at 1 Million
  Utterances from Project Euphonia.* Interspeech 2021. — Akinrintoyo,
  E., Abdelhalim, N., & Salomons, N. (2025). *WhisperD: Dementia Speech
  Recognition and Filler Word Detection with Whisper.* Interspeech 2025,
  1413-1417.
- arXiv:2311.00867 — *Automatic Disfluency Detection from Untranscribed
  Speech* (cited and verified in an earlier session of this track).
- arXiv:2211.08726 — joint speech recognition and disfluency detection,
  two-output-layer architecture with a token-dependency bridge (cited
  in an earlier session).
- arXiv:2409.10177 — *Augmenting Automatic Speech Recognition Models
  with Disfluency Detection* (cited in an earlier session).
- arXiv:1908.05378 — *Multi-Task Self-Supervised Learning for Disfluency
  Detection* (cited in an earlier session).
- arXiv:2512.13632 — multi-representation fusion with gating for
  disfluency-related tasks (cited in an earlier session).

**Audio-native / ASR-bypassing dysfluency detection (the closest
published precedent to this track's own direction (g))**
- [B15] Zhou, X. et al. (2024). *YOLO-Stutter: End-to-end Region-Wise
  Speech Dysfluency Detection.* Interspeech 2024.
  https://arxiv.org/abs/2408.15297
- [B16] (2024). *SSDM: Scalable Speech Dysfluency Modeling.* NeurIPS
  2024. https://arxiv.org/abs/2408.16221

**Whisper-encoder-level disfluency signal (directly relevant to this
track's own Stages B/C and layer-depth sweep)**
- arXiv:2311.05203 — *Whisper in Focus: Enhancing Stuttered Speech
  Classification with Encoder Layer Optimization* — found deeper
  Whisper-encoder layers carry more disfluency signal for a comparable
  task, the opposite of this track's own layer-depth-sweep finding; the
  discrepancy is named and left unresolved, not smoothed over (cited and
  verified in an earlier session).
- arXiv:2409.10704 — self-supervised speech models (WavLM among them)
  for word-level stuttered speech detection, F1=0.554 — this track's
  own basis for testing WavLM in Arm 3 (cited in an earlier session).
- arXiv:2605.12387 — a semi-supervised Whisper-embeddings + explicit-
  acoustic-features hybrid beating both pure-Whisper and pure
  self-supervised baselines on speech-confidence detection, a different
  paralinguistic task — real support for the hybrid-architecture
  direction, in tension with this track's own combined-signal-classifier
  negative result, named explicitly as an unresolved tension (cited in
  an earlier session).

---

# Comprehensive research-positioning analysis — 2026-08-07 (deep pass)

**Explicit instruction honored**: the audit immediately above is treated
as preliminary input, not a final conclusion. This pass independently
re-verifies it against a substantially wider literature search, and
**revises one material claim** made above — flagged explicitly where it
happens, not silently corrected. Nothing above is deleted or rewritten;
this section supersedes specific claims by superseding them in writing,
per this document's own append-and-mark-superseded convention (see the
2026-08-05 handoff section's own precedent for this pattern).

**Nine additional primary sources were fetched and read directly this
pass** (full PDFs for the most load-bearing ones — AS-70 in full,
Kordt et al. already read in full previously), on top of the 19 already
verified. Full citations are in the expanded Bibliography at the end of
this section.

## 1. Reconstructing our research precisely, with exact numbers

**The original problem and motivation.** This project's downstream
disfluency detector (`profiling/detect.py`) needs word-level evidence
of `sound_repetition`/`word_repetition` to fire. That evidence comes
from CrisperWhisper's decoded text. `ROADMAP.md` item 19 first found,
anecdotally, a case where the ASR transcript "normalized away" a
repetition even though nothing else looked wrong with the transcription
— this track exists to determine how general that finding is and
whether it can be fixed cheaply.

**Stage A** (2026-08-05, n=186 hand-traced positions across a 120-clip
real-ASR sample): **we demonstrated experimentally** that 45.2% of
`sound_repetition` losses (19/42) and 40.5% of `word_repetition` losses
(17/42) occur at positions ASR transcribed "correctly" by word-match
standards — a 4-category classification (normalized-away / mis-routed /
genuine ASR error / ASR error + coincidental type), cross-checked
line-by-line against the scoring code before being trusted.

**Stage B** (encoder-distance probe): **we demonstrated experimentally**
that CrisperWhisper's last-layer encoder state at these normalized-away
positions differs measurably from the fluent-speech centroid — a mixed
but real result, strong for `sound_repetition`, inconclusive for
`word_repetition` at n=17.

**Stage C** (duration-confound test, n=19 target / 966 control
positions, leave-one-out centroid): **we demonstrated experimentally**
Cohen's d=0.894, AUC=0.723 for the encoder-distance signal; the
duration-only baseline scored AUC=0.483 (chance), directly refuting the
duration-confound hypothesis this track pre-registered before running
the test. Precision at R>=0.5 was 4.7% — **we demonstrated
experimentally** that the signal is real but not usable alone at this
class imbalance (19 positives : 966 negatives).

**Stage C2** (Praat voice-quality fusion, 5 features): **we demonstrated
experimentally** a clean negative — all 5 features (pitch, pitch
stability, jitter, shimmer, HNR) scored AUC 0.452-0.549, indistinguishable
from chance.

**CrisperWhisper layer-depth sweep** (33-layer forward pass, same
population): **we demonstrated experimentally** the signal is
concentrated in the last layer only (AUC=0.721 on this population's
18-clip subset, matching Stage C's 0.723 on the full 31-clip population
within measurement noise); every other layer scored 0.336-0.378.

**Stock Whisper comparison — Arm 1** (n=36 positions, `whisper-large-v3`
full pipeline, `num_beams=1` matching the live app): **we demonstrated
experimentally** 0/36 positions recovered; normalized-away rate 89.5%
(`sound_repetition`) and 88.2% (`word_repetition`) — higher than
CrisperWhisper's own 45.2%/40.5%, not lower. Mean WER 0.177, comparable
to CrisperWhisper's own range.

**Stock Whisper comparison — Arm 2** (layer sweep, same 18-clip
subset): **we demonstrated experimentally** the identical last-layer-
only pattern (AUC=0.680 at layer 32, all others 0.336-0.378-range
territory), same-population comparison showing CrisperWhisper's own
0.721 is slightly higher, not lower.

**WavLM comparison — Arm 3** (n=19 `sound_repetition` / 966 control;
n=17 `word_repetition`): **we demonstrated experimentally** `sound_
repetition` at chance (d=-0.061, AUC=0.474). **Our results suggest**
(not demonstrated at usable confidence — n=17, single-cell) a small
`word_repetition` signal (d=0.259, AUC=0.576) WavLM alone showed;
explicitly flagged as too small to trust standalone. **We demonstrated
experimentally** a real layer-depth-profile difference (peak at layer
10, AUC=0.617) that never exceeds either Whisper variant's peak.

**Beam-width experiment** (n=14 positions, `num_beams=1` vs. 5): **we
demonstrated experimentally** 0/14 recovered, identical WER (0.187 vs.
0.187).

**Direction (g), RMS/ZCR acoustic candidates** (51 ground-truth
`sound_repetition` instances, 766 total candidates, 120-clip sample):
**we demonstrated experimentally** recall=0.824 (42/51), precision=
0.081 (62/766), best gated F1=0.161 (not meaningfully above the 0.147
ungated baseline). A real cross-clip scoring bug (pooling raw
timestamps across all 120 clips' independent timelines) was caught via
an implausibly-perfect first result (recall=1.000/precision=0.966) and
fixed before this number was trusted.

**MFCC escalation** (identical population and candidate-generation
logic, only the similarity feature changed): **we demonstrated
experimentally** best F1=0.170 (recall=0.686, precision=0.097) — real
precision/recall trade-off structure, unlike RMS/ZCR's near-flat curve,
but still short of the pre-registered 0.176 bar. A second real bug
(MFCC coefficient 0 — overall energy, not spectral shape — masking 90%
of all burst pairs into >=0.9 "similarity" regardless of match) was
caught via an implausibly *flat* first result and fixed before trusting
the corrected number.

**Combined-signal classifier** (766 candidates, 62 positive, 5-fold
clip-split nested-CV logistic regression): **we demonstrated
experimentally** F1=0.242 (P=0.314, R=0.244), statistically
indistinguishable from encoder-distance-alone's F1=0.244 — clearing the
bar against the weaker MFCC-alone baseline (0.177) but not against the
stronger encoder-distance-alone baseline (0.293), which the
pre-registration required clearing both to count as Success. A real
bug (a fixed `proba>=0.5` decision threshold miscalibrated to this
population's ~8% positive rate) produced an implausible F1=0.000 first
result and was caught, confirmed via a synthetic check, and fixed
before trusting the corrected number.

**What every major failure actually ruled out** (stated precisely, not
inflated): Arm 1 ruled out "a bigger, non-fine-tuned Whisper-family
model recovers this text evidence." Arm 2 ruled out "CrisperWhisper's
own fine-tuning caused the last-layer-only concentration." Arm 3 ruled
out "WavLM's representation carries a stronger `sound_repetition`
signal at this scale" — it did **not** rule out self-supervised
representations in general (only one was tested), and did **not**
rule out the model-size confound (WavLM-Large is smaller than
CrisperWhisper's encoder; this was named, not resolved). Direction (g)
ruled out "RMS/ZCR and MFCC similarity alone separate genuine repeats
from background speech rhythm at usable precision" — it did **not**
rule out a *learned* (not hand-engineered) acoustic feature (see the
literature review below — YOLO-Stutter/SSDM are real, more capable
existing examples never tested against this project's own data). The
combined classifier ruled out "this specific 4-feature combination
beats the strongest individual signal" — it did **not** rule out
combining the strong signal with a different, comparably strong second
signal, which remains untested.

**What was pre-registered vs. discovered afterward, stated explicitly**:
every success/failure criterion above was written before the
corresponding experiment ran (see this document's own "pre-registered
protocol" subsections throughout). What was *not* pre-registered and
was discovered only afterward: all three implementation bugs (cross-clip
pooling, MFCC coefficient-0 masking, the classification threshold); the
real, useful side-finding that encoder-distance-alone (P=0.343) is the
strongest single number this track has produced against real
candidates; and everything in this literature review itself.

**Strong findings** (large sample, direct measurement, no confound
left unaddressed): Stage A's normalization-loss rate; the duration-
confound refutation (Stage C); Arm 1's "bigger model doesn't help"
result; Arm 2's "not CrisperWhisper-specific" result.

**Weak findings** (small sample, single-cell, or with a named,
unaddressed confound): WavLM's `word_repetition` signal (n=17, single
cell, explicitly flagged); the model-size confound in Arm 3's overall
verdict; the control-group non-independence caveat (positions pooled
across clips, not fully i.i.d.) named in Stage B/C and never resolved.

**Hypotheses only, not evidence** (Category D from the prior audit,
restated): that a taxonomy-targeted fine-tuning objective (Stage D)
would succeed where representation-level approaches didn't; that such a
model would generalize beyond its training data.

## 2. Where information about a disfluency can disappear — an architectural deep-dive

The pipeline, stated explicitly: **audio waveform -> acoustic feature
extraction (log-mel spectrogram, for Whisper-family models) -> encoder
(stacked self-attention/conv blocks, producing a continuous hidden-state
sequence) -> decoder (autoregressive, conditioning on encoder output and
its own previously-generated tokens) -> token probability distribution
at each step -> greedy or beam-search selection -> discrete text
tokens.**

**Stage-by-stage, what can and cannot survive**:

- **Waveform -> log-mel spectrogram**: lossy in a well-understood,
  bounded way (phase discarded, frequency resolution bounded by the
  filterbank) but not linguistically selective — a repeated fragment's
  acoustic energy is still present in the spectrogram, full stop. This
  is not where disfluency information is lost.
- **Log-mel -> encoder hidden states**: this is where this track's own
  evidence is most direct. The encoder is trained (for Whisper-family
  models) as part of an end-to-end system whose loss is defined on the
  *decoded token sequence*, not on any intermediate representation
  directly — nothing in the training objective explicitly rewards the
  encoder for keeping disfluency-specific information distinct from
  general phonetic/linguistic content, but nothing explicitly removes
  it either. This is consistent with the information-bottleneck framing
  found directly in this pass's search: "the encoder is sufficient if
  it preserves the full predictive information for the target;
  otherwise, the mutual information loss quantifies the irreducible
  performance gap" — the encoder's job is to preserve what the decoder
  needs to hit its own training target, and if that target (a fluent-
  ish transcript) doesn't need fragment-level information, there is no
  guarantee — but also no structural prohibition — against the encoder
  retaining it anyway as a side effect of solving a broader phonetic-
  modeling problem. This track's own Stage B/C result (d=0.894, AUC=
  0.723) is a direct, measured instance of exactly this: the encoder
  retained *some* real signal as an unintended side effect, without
  being asked to, and without that signal being complete enough to use
  alone.
- **Encoder states -> decoder, autoregressively**: this is the
  structurally decisive step, and the step every reviewed source
  agrees is where a fluency bias becomes an active *choice*, not just a
  passive information bottleneck. The decoder is trained with
  teacher-forcing against reference transcripts, and for Whisper-family
  models those reference transcripts come from large-scale, weakly-
  supervised web data whose *own* transcription conventions already
  skew toward readable, normalized text (this is a documented,
  general property of how large ASR training corpora are curated, not
  a claim this project verified about Whisper's specific training set,
  which is not public). At generation time, the decoder's own learned
  language-model-like prior (conditioning on its own prior tokens)
  actively favors likely, fluent continuations — this is precisely the
  mechanism Lin et al. 2025 [B12] exploited in reverse (labeling
  hesitations *increased* their likelihood under the model, reducing
  WER) and precisely what Mujtaba et al. 2024 [B13] and Gulzar et al.
  2025 [B14] describe as ASR being "optimized to omit disfluencies."
  **This is the answer to "why would a decoder have no incentive to
  emit information the encoder has": because nothing in a standard
  cross-entropy training objective against normalized-ish reference
  text rewards emitting it, and the decoder's own autoregressive prior
  actively penalizes low-probability, disfluent-looking continuations
  at generation time — a live, per-token bias, not just an absence of
  reward.**
- **Token probabilities -> discrete selection (greedy/beam)**: this
  track's own `num_beams` experiment (0/14 recovered, identical WER)
  is direct, first-party evidence that this specific late-pipeline
  lever does *not* meaningfully change the outcome — the normalization
  has already happened by the time decoding-width choices matter,
  consistent with the mechanism above (the bias lives in what the
  decoder was trained to predict, not in how many alternatives beam
  search considers among near-equally-likely continuations).

**Why a model can "know" about an event internally while never emitting
it textually, stated as directly as this review can make it**: the
encoder's representation is shaped by everything needed to predict the
*correct* text token sequence, which for disfluent speech may include
implicit cues (energy bursts, spectral repetition, timing irregularities)
that correlate with what the correct fluent output *should* be — the
encoder doesn't need to "throw away" the disfluency to do this, it just
never needs to *use* it for anything the decoder's training target
requires it to emit. The decoder, separately, is trained toward — and
at generation time actively biased toward — exactly the normalized
target the encoder's representation was never asked to protect. Both
things can be true simultaneously, and this track's own results (a
real encoder-distance signal, combined with a training-target/decoder
bias fully explaining why it's never emitted) are a direct, applied
instance of a phenomenon the wider probing/information-theory literature
already frames in general terms (e.g. the "modality collapse" /
mismatched-decoding framing found in this pass's search — encoder-side
sufficiency does not imply decoder-side accessibility).

## 3-4. Structured, analyzed entries for every major relevant study

Each entry below was verified from the primary source directly this
session (full-text PDF for the most load-bearing ones), not from a
search snippet alone. "Solves our exact problem?" means specifically:
*fine-tunes or modifies an ASR system to emit fragment-level
`sound_repetition` evidence in its output, distinct from word-level
repetition, evaluated against real speech* — the precise gap this
track's own experiments point at.

**Kordt et al., "Learning to Hear Hesitation" (Interspeech 2026, [B11])**
- Objective: introduce explicit disfluency tokens into Whisper via
  continual learning without catastrophic forgetting.
- Dataset: SME (11.79h, L2 English speakers), Pitt/DementiaBank (12.11h
  used, dementia), Delaware (9.72h, MCI) — all real, CHAT-format,
  **not stuttering-specific**.
- Task: verbatim transcription with 4 coarse disfluency-token
  categories (FILLER, REP, DISRUPT, PAUSE).
- Model: `whisper-small.en`, fine-tuned with EWC/Experience
  Replay/A-GEM/Weight Averaging.
- Problem solved: catastrophic forgetting during disfluency-token
  introduction.
- Discovered: a real, quantified trade-off between marker-F1 and
  general WER; a shared cross-attention-head mechanism for marker
  emission across CL methods.
- Preserved: presence of a repetition (REP token fires).
- Lost: which fragment repeated, and the sound-vs-word distinction
  (both collapse into REP).
- Modified the ASR model itself: **yes**, fine-tuned.
- Used acoustic information separately: no — text-token-only.
- Used explicit disfluency labels: yes, from TalkBank/CHAT annotations.
- Required paired data: yes, real, ~10-20h/dataset.
- Limitations: coarse granularity; not stuttering-specific data; real
  ASR-quality trade-off.
- Comparison to our work: closest published precedent to a Stage D
  attempt; independently confirms our own catastrophic-forgetting
  concern with real numbers.
- **Solves our exact problem? No** — by construction, `REP` cannot
  distinguish `sound_repetition` from `word_repetition`.

**Lin et al., "Acoustically Precise Hesitation Tagging" (SLaTE 2025,
[B12])**
- Objective: test whether acoustically-precise (vs. generic) disfluency
  labels improve verbatim ASR.
- Dataset: Speak & Improve 2025 corpus (L2 English speech).
- Task: verbatim transcription, filler-word focus.
- Model: Whisper Large V3 Turbo, LoRA fine-tuned.
- Discovered: acoustically-precise labels (LLM-inferred) beat generic
  tags beat removal — 5.5% WER vs. 6.2% WER (Pure), 11.3% relative
  improvement.
- Preserved: filler-word content specifically, with acoustic precision.
- Lost: not evaluated for repetitions at all.
- Modified the ASR model: yes, LoRA fine-tune.
- Acoustic info used separately: yes, an LLM inferred acoustically-
  grounded labels from audio-transcript pairs as a labeling step.
- Explicit disfluency labels: yes.
- Paired data required: yes.
- Limitations: fillers only, not repetitions; L2-speech corpus, not
  stuttering.
- Comparison: real, quantified evidence *for* Stage D's general premise
  (precision-labeled fine-tuning measurably helps, not just avoids
  harm) — on a different disfluency type than this project's own focus.
- **Solves our exact problem? No** — wrong disfluency type.

**Mujtaba et al., "Lost in Transcription" (NAACL 2024, [B13])**
- Objective: quantify ASR accuracy bias against disfluent speech across
  multiple systems.
- Dataset: real stuttering samples + synthetic disfluent speech.
- Discovered: "a consistent and statistically significant accuracy bias
  across all ASRs against disfluent speech."
- **Solves our exact problem? No** — measures bias, doesn't propose or
  test a fix. Directly, independently corroborates this track's own
  Stage A premise at a broader (multi-system) scale.

**Gulzar et al., "On the Difficulty of Token-Level Modeling" (ASRU
2025, [B14])**
- Objective/discovered: independently states "dysfluencies and
  fluency-shaping artifacts are often overlooked, resulting in
  non-verbatim transcriptions" — proposes parameter-efficient
  adaptation + multi-step fine-tuning + language-adaptive pretraining.
- Dataset: German stuttering-therapy speech (fluency-shaping context).
- **Solves our exact problem? No** — targets fluency-shaping artifacts
  broadly and multilingual tokenizer bias, not fragment-level
  `sound_repetition` specifically; real precedent for the general
  engineering strategy (parameter-efficient adaptation).

**YOLO-Stutter (Zhou et al., Interspeech 2024, [B15]) / SSDM (NeurIPS
2024, [B16])**
- Objective: audio-native, time-accurate dysfluency region detection,
  bypassing ASR text.
- Dataset: YOLO-Stutter introduces VCTK-Stutter/VCTK-TTS (simulated);
  SSDM introduces Libri-Dys (simulated, LibriSpeech-derived — the same
  synthetic-injection strategy as LibriStutter).
- Model: end-to-end trained detectors (YOLO-Stutter: spatial feature
  aggregator + temporal dependency extractor; SSDM: articulatory
  gestures as forced alignment + a "connectionist subsequence aligner"
  + LLM-based end-to-end system).
- Preserved: region-level boundaries and class, time-accurate.
- Modified ASR: no — bypasses it entirely, by design.
- Acoustic info used separately: yes, exclusively.
- Explicit labels: yes, region-wise class + boundary.
- Paired data: simulated corpora built for this purpose.
- Limitations: simulated training/eval data (same generalization risk
  this track has named for LibriStutter); neither reports results
  against this project's own exact fragment-boundary definition.
- Comparison: the most *capable* (learned, not hand-engineered) version
  of this track's own direction (g) strategy — a real, credible
  "simpler alternative" this track did not test.
- **Solves our exact problem? Partially/unclear** — plausibly capable
  by taxonomy (both include repetition as a class), never verified
  against this project's own data or exact fragment definition.

**AS-70 (Gong et al., Interspeech 2024, [B20]) — read in full this
pass, see the "material revision" below**
- Objective: provide a real, large-scale, verbatim, character-level-
  annotated Mandarin stuttering dataset, and establish ASR + SED
  baselines on it.
- Dataset: 48.8 hours, 70 adults who stutter (72 including 2 PWS
  interviewers), Mandarin, conversational + voice-command speech, real
  (not synthetic).
- **Annotation, confirmed directly from the paper's own examples**:
  five types — `[]` word/phrase repetition, `/b` block, `/p`
  prolongation, **`/r` sound repetition ("repeated phoneme that do
  not constitute an entire character")**, `/i` interjection — verbatim
  markup embedded inline in the transcript (e.g. "小/r明" marks a
  sound repetition on one phoneme of the character "小").
  **This is a real, distinct sound-vs-word repetition distinction on
  real speech, at scale** — the exact taxonomy split this project's own
  work has repeatedly found no dataset provides.
- ASR baselines (Table 4, CER%): Whisper-large-v2 direct inference
  27.20% CER (all); fine-tuned on AS-70, 8.75% CER — a real, large
  improvement. **Critical caveat, confirmed by direct reading of
  Section 3.2**: "the annotations undergo preprocessing to exclude
  stuttering event labels, stuttering characters, and punctuation" —
  their published ASR fine-tuning experiment targets ordinary
  (fluent-text) CER, not disfluency-preserving output. It does **not**
  test whether fine-tuning on AS-70 improves the model's ability to
  *emit* the `/r`/`[]` markup — that specific experiment, despite ideal
  data sitting right there, is not what AS-70's own authors ran.
- SED baselines (Table 5, F1% by type, wav2vec2.0 best overall): `/r`
  (sound repetition) F1=65.76%, `[]` (word/phrase repetition) F1=78.48%,
  `/b` (block) F1=42.51% — real, substantial, type-specific detection
  accuracy on real speech, via a *separate classifier*, not ASR decoder
  output.
- Modified ASR itself: yes, for the (fluent-target) CER experiments; no
  disfluency-preserving-output experiment was run.
- Comparison to our work: the closest thing found in this entire review
  to "the right data already exists" — but for Mandarin, not English,
  and even its own authors did not run the fragment-preservation
  fine-tuning experiment this track's own Stage D would need.
- **Solves our exact problem? No — but removes the "no adequate real
  data exists at all" objection for the Mandarin case specifically**,
  and demonstrates real detection accuracy (F1=65.76% for `/r`
  specifically) is achievable on real data with this exact taxonomy
  split, which this track's own English-language, ASR-independent
  attempt (direction (g)) did not come close to matching (~8-10%
  precision). This is a materially important, precise finding — see
  "A material revision" below.

**Huang et al., "Leveraging LLM for Stuttering Speech" (Interspeech
2025, [B21])**
- Objective: unify ASR and stuttering event detection (SED) in one
  architecture rather than running them separately.
- Dataset: AS-70 (the same real Mandarin dataset above).
- Model: CTC-generated soft prompts feed an LLM for ASR; a separate SED
  branch outputs stutter embeddings that also feed the LLM —
  bidirectional information flow between the two tasks.
- Discovered: CER 5.45% (37.71% relative reduction vs. baseline), SED
  F1 73.63% (46.58% relative improvement) — both tasks improve when
  trained jointly, real evidence for the multitask/hybrid direction
  (solution family 4/7 below) on real stuttering data.
- Preserved: stutter event detection improves substantially; whether
  the ASR *transcript itself* now includes fragment-level markup is not
  specified in the reviewed abstract — likely not, given the
  architecture separates an SED *branch* from ASR *output*.
- **Solves our exact problem? Unclear/likely no** — real, strong,
  recent (2025) evidence that joint training helps both tasks, on the
  exact data (AS-70) that has the right taxonomy, but not confirmed to
  produce a disfluency-preserving *transcript* as opposed to a separate
  detection signal.

**Lea et al., "From User Perceptions to Technical Improvement" (CHI
2023, [B22])**
- Objective: improve ASR usability for people who stutter via decoder-
  level modification.
- Discovered: WER improves most for whole-word repetitions, part-word
  repetitions, and interjections; least for prolongations/blocks —
  **and word-insertion errors correlate strongly with part-word
  repetitions** — independent, real evidence that `sound_repetition`
  (this project's own term for "part-word repetition") is specifically
  and disproportionately implicated in ASR errors, not an
  undifferentiated part of "disfluency" generally.
- **Solves our exact problem? No** — a decoding-level (language-model
  weight / insertion-penalty) intervention, not a fragment-preservation
  objective; real precedent for solution family 1 (better decoding).

**Shonibare et al., "Detect and Pass" (2022, [B23])**
- Objective: improve ASR for stuttered speech with limited data via a
  frame-level stuttering classifier feeding an RNN-T model.
- Discovered: 12.18%-71.24% relative WER reduction across multiple ASR
  systems.
- Modified ASR: yes, architecturally (RNN-T-specific integration).
- Acoustic info used separately: yes, a dedicated frame-level
  classifier.
- **Solves our exact problem? No** — targets WER reduction via
  detect-then-adjust-decoding, not fragment-content preservation in the
  output; real, direct precedent for solution family 6 (frame-level
  supervision) integrated with an ASR model specifically (not just a
  standalone detector).

**Kouzelis et al., "Weakly-supervised forced alignment of disfluent
speech" (Interspeech 2023, [B24])**
- Objective: make forced alignment robust to disfluencies (repetitions,
  omissions) that break standard aligners.
- Dataset: corrupted TIMIT, UCLASS (real stuttering data).
- Discovered: 23-25% relative improvement over baseline aligners,
  particularly in recall.
- **Solves our exact problem? No** — aligns a *given* transcript to
  audio despite disfluency-caused mismatch; doesn't generate or
  preserve fragment-level content itself. Relevant precedent for
  solution family 6/7 (frame-level/hybrid alignment machinery).

**SEP-28k [B17], KSoF [B18], FluencyBank/FluencyBank Timestamped
[B19], LibriStutter** — dataset papers, not methods; already fully
characterized in the prior audit's literature-landscape section above.
**None of these four provides a sound-vs-word repetition distinction on
real speech at AS-70's scale** — SEP-28k and KSoF's `/r`/sound-
repetition-equivalent labels exist but at clip level, without AS-70's
character-level verbatim markup; FluencyBank Timestamped's exact
granularity was not independently confirmed at this level of detail in
either research pass (a real, still-open gap in this review, not
assumed either way).

### A material revision to the prior audit's data-requirements claim

**The "Final research audit" section above states**: "no accessible
large-scale real (non-synthetic) paired dataset with the specific
fragment-level `sound_repetition`/`word_repetition` granularity this
project's taxonomy needs" exists. **This is revised, not retracted —
the historical reasoning is preserved above, and the correction is
stated here explicitly, per the project owner's instruction not to
silently overwrite prior conclusions.**

**The corrected claim**: AS-70 (real, 48.8 hours, Mandarin, verbatim,
character-level, with an explicit `/r` sound-repetition vs. `[]`
word/phrase-repetition distinction, and a demonstrated F1=65.76% SED
result for `/r` specifically on real speech) directly satisfies the
data-granularity requirement this track named as missing — **for
Mandarin**. The English-language gap this project's own pipeline
depends on (CMU phonetic dictionaries, English filler-word lists, the
existing detector's own English-specific logic) remains real and
unaddressed: no English-language dataset reviewed provides this same
granularity at this scale on real speech. This sharpens, rather than
resolves, the data question for Stage D as this project would actually
need to build it — the "no adequate real data exists anywhere" framing
was too strong; the "no adequate real *English* data exists" framing
is the corrected, defensible one.

## 5. Comparing the field with our findings — narrowest defensible claim per finding

| Our finding | Already known? | Our result relative to prior work | Narrowest defensible claim |
|---|---|---|---|
| ASR normalizes disfluencies broadly | Yes, well-established | Replication, single-system depth | Prior work (Mujtaba et al. [B13]) demonstrated a cross-system bias; we replicate this for CrisperWhisper specifically at fragment-level granularity no prior source measured directly. |
| Roughly half of `sound_repetition`/`word_repetition` losses happen at "correctly transcribed" positions | Not previously measured at this granularity, as far as this review found | New failure-mode analysis | Prior work established that ASR is biased against disfluent speech generally; we found, specifically, that a large fraction of that loss is a *normalization* phenomenon (not a transcription-accuracy phenomenon) at word-correct positions — a distinction not directly measured elsewhere in this review. |
| CrisperWhisper's encoder retains a real `sound_repetition` signal (d=0.894) despite the decoder not emitting it | Not previously demonstrated for this exact model/task | Architectural insight, new evidence | The general phenomenon (encoder sufficiency without decoder accessibility) is theoretically well-understood (information-bottleneck/probing literature); we provide a first-party, directly-measured instance of it for CrisperWhisper and this specific disfluency type. |
| A bigger, non-fine-tuned Whisper model does not recover this evidence | Not previously tested, as far as this review found | New comparative result | No source reviewed directly compares CrisperWhisper against stock `whisper-large-v3` on fragment-level preservation; this is a new, controlled comparison, not a replication. |
| The last-layer-only signal concentration is Whisper-architecture-general, not CrisperWhisper-specific | Not previously tested for this task | New comparative result, partially in tension with prior work | Arm 2 directly tests and finds no support for the depth-distribution pattern arXiv:2311.05203 reported for a different stuttering-classification task — a real, named, unresolved discrepancy, not smoothed into agreement. |
| WavLM's representation does not carry a stronger `sound_repetition` signal | Not previously tested for this task | New negative result, in tension with WavLM's own design rationale [B8] | WavLM's stated design targets paralinguistic sensitivity generally; we found no advantage for this specific task at this specific model size — a real, named tension, not a refutation of WavLM's general utility. |
| `sound_repetition` has a real, ASR-independent acoustic co-occurrence with short voiced-burst runs (recall=0.824) | Consistent with, not novel relative to, block/prolongation's own established acoustic-native precedent and the broader audio-native detection literature (YOLO-Stutter/SSDM) | Extension to a new type, weaker method | This project's own `block`/`prolongation` detectors already demonstrated the general strategy; we extend it to `sound_repetition` with a much simpler (hand-engineered, not learned) method, and find real recall but not usable precision — less capable than the field's own more sophisticated learned detectors (YOLO-Stutter/SSDM), not a novel finding about the underlying phenomenon. |
| Combining the strongest available signals doesn't beat the best one alone | Not directly tested elsewhere for this task | New negative result, narrow scope | No source reviewed reports a comparable single-strong-plus-weaker-signals combination test for fragment-level disfluency preservation; the general statistical fact that naive combination doesn't automatically beat a strong individual signal is unsurprising and not domain-specific. |
| The fragment-level `sound_repetition`-vs-`word_repetition` distinction is not resolved even by the closest published disfluency-token fine-tuning work | **Newly established this session**, via direct verification of Kordt et al.'s `REP` category definition | This is this track's own most defensible novel observation | Prior work (Kordt et al. [B11]) fine-tunes ASR with disfluency tokens and reports real results, but its own token taxonomy collapses exactly the distinction this project's taxonomy treats as two separate types — we identify this specific, narrow gap by direct comparison of taxonomies, not by assumption. |

## 6. Scale and credibility of the evidence — an honest assessment, not encouragement

| Finding | n (targets/controls) | Independent eval? | Replicated? | Pre-registered? | Baseline? | Confound tested? | Synthetic data? | Classification |
|---|---|---|---|---|---|---|---|---|
| Stage A normalization rate | 186 positions, hand-traced | Yes (cross-checked vs. scoring code) | No (single pass) | Yes | N/A (descriptive) | N/A | Yes (LibriStutter) | **High-confidence** — large n, careful methodology, but single-dataset |
| Stage C encoder-distance signal | 19 target / 966 control | Yes | Partially (Stage B->C re-derivation matched) | Yes | Yes (duration) | Yes, explicitly (duration confound refuted) | Yes | **Moderate-confidence** — real effect size and refuted confound, but n=19 targets is small, and control-group independence (positions pooled across clips) was never formally tested |
| Arm 1 (stock Whisper, full pipeline) | 36 positions | Yes | No | Yes | Yes (CrisperWhisper's own rate) | N/A | Yes | **High-confidence** for its narrow claim (this specific comparison), small-n caveat still applies |
| Arm 3 WavLM `sound_repetition` | 19/966 | Yes | No | Yes | Yes | Model-size confound named, not resolved | Yes | **Moderate-confidence** — clean negative, but the unresolved model-size confound genuinely limits what can be concluded about "self-supervised representations in general" |
| Arm 3 WavLM `word_repetition` signal | 17 target, single cell | Yes | No | Yes | No prior comparable number | No | Yes | **Preliminary finding only** — explicitly flagged by this track itself as too small to trust |
| Direction (g) recall/precision | 51 targets, 766 candidates | Yes | Yes (two independent feature implementations both ~0.82 recall) | Yes | Yes (duration-only) | Yes (burst-count distribution checked directly) | Yes (LibriStutter) | **Moderate-confidence** for recall (replicated across features); **high-confidence negative** for precision (consistent, mechanistically explained failure) |
| Combined classifier | 766 candidates, 62 positive, 5-fold CV | Yes | No | Yes | Yes (both individual arms) | N/A | Yes | **Moderate-confidence** — real cross-validation discipline, but 62 positives across 5 folds (~12/fold) is a genuinely small sample for a trained classifier's stability |
| AS-70's `/r` SED F1=65.76% (external) | Not independently re-verified by this project | Published, peer-reviewed (Interspeech 2024) | Not by us | Not applicable (external) | Yes (random-guess baseline) | Not assessed by this review | **No** — real speech | **High-confidence as a published external result**; not independently replicated by this project |

**Is any of this project's own evidence mechanistic or merely
correlational?** Mostly correlational at the representation level
(encoder-distance correlates with ground-truth labels; the acoustic
candidate mechanism correlates burst-count/similarity with ground truth)
— **with one partial exception**: Stage C's duration-confound test is a
genuine mechanistic check (does distance correlate with *duration*
specifically, ruling out a specific alternative causal story), not just
an association with the outcome. The architectural deep-dive in section
2 above is inferential/theoretical, not something this track's own
experiments directly measured (no probing-classifier experiment was run
to test the information-bottleneck framing directly against this
project's own encoder states — a real, named gap, not filled here).

**Is the result likely to generalize?** Genuinely uncertain for every
finding in this table — every experiment ran against LibriStutter's
synthetic splicing. This is the single largest, most consistently
named limitation across this entire track, repeated here rather than
softened: **none of this project's own quantitative findings have been
tested against real (non-synthetic) disfluent speech.**

## 7. Current problem statement — four versions

**One-sentence problem statement**: State-of-the-art ASR systems,
including one fine-tuned specifically for verbatim transcription
(CrisperWhisper), reliably normalize away sub-word repetition evidence
(`sound_repetition`) even at positions otherwise transcribed correctly,
and no cheap, off-the-shelf representation, decoding change, or
combination of acoustic signals tested recovers it at usable precision.

**Technical research problem**: Given an audio signal containing a
sub-word phonetic repetition (e.g. a stuttered "c-c-cat"), and an ASR
system (CrisperWhisper) whose decoded output correctly transcribes the
surrounding words, does the system's decoded text, its internal encoder
representation, an alternative pretrained ASR's decoded text or
representation, or an ASR-independent acoustic feature derived directly
from the waveform, preserve sufficient information to reconstruct the
presence and boundaries of that repetition at a precision usable by a
downstream classifier — and if not, what minimal training-time
intervention would be required to make it so, at a taxonomy granularity
(sound- vs. word-level) that the closest published disfluency-token
fine-tuning work has not yet resolved?

**Paper-style problem statement**: We investigate whether commonly-used
speech representations — a fine-tuned verbatim-transcription ASR model,
a stock large-scale ASR model, and a self-supervised speech
representation — preserve sufficient information to recover sub-word
repetition disfluencies (`sound_repetition`) that are systematically
normalized away during decoding, even at word positions the same system
transcribes correctly. We find that neither the decoded text nor the
internal representations of any of the three systems tested support
recovery at usable precision, that two hand-engineered acoustic
features derived independently of ASR achieve real recall but
insufficient precision, and that combining the strongest representation-
level and acoustic-level signals does not improve on the stronger
signal alone. We argue this constitutes evidence that the gap is not
readily closed by representation selection or post-hoc signal
combination, and requires either a training-time intervention targeted
at this specific taxonomy granularity or richer, learned acoustic
detection methods not yet tested against this task.

**Venture/non-technical explanation**: When someone who stutters
repeats a sound (like saying "c-c-cat"), today's best speech-to-text
systems — even ones built to write down exactly what was said —
routinely clean that up and just write "cat," as if the stutter never
happened. We spent two research sessions checking every cheap way we
could think of to recover that lost information — trying different,
bigger AI transcription models, trying models built differently from
scratch, trying to spot the stutter directly in the sound wave, and
trying to combine our best signals together. None of it reliably
worked. The evidence increasingly points to needing to actually
retrain a model specifically to notice and preserve this kind of
speech pattern — which is a real, bigger undertaking we have not yet
attempted, and have now carefully scoped out.

## 8. Searching for the exact same problem formulation

**The precise phenomenon searched for**: *a disfluency exists
acoustically; the ASR system normalizes/removes it from the textual
hypothesis; information about the event may still exist in the
acoustic/encoder representation.*

Findings, checked against each specific proposed recovery mechanism the
project owner asked about:

- **Explicitly identified as a problem, in exactly this framing?**
  Partially. Mujtaba et al. [B13] and Gulzar et al. [B14] both state the
  general "ASR normalizes disfluencies" half of this framing directly.
  **No source found in this review explicitly frames the second half**
  (information surviving in the encoder despite decoder omission) as a
  named problem for disfluency specifically — this track's own Stage
  B/C appears to be the most direct, controlled test of exactly that
  half found in this review.
- **Proposed recovering information from encoder states specifically for
  disfluency?** Not found directly — the general probing/information-
  bottleneck literature (section 2 above) frames this mechanism
  abstractly, and this track's own Stage B/C is a direct, applied test
  of it, but no source reviewed proposes it as a recovery *method* for
  disfluency.
- **Modified decoding to preserve it?** Yes — Lea et al. 2023 [B22]
  (language-model weight/insertion-penalty tuning) and this track's own
  `num_beams` experiment both test decoding-level levers; both found
  limited-to-no effect on fragment-level preservation specifically
  (Lea et al.'s own finding that word-insertion errors correlate with
  part-word repetitions is about error *causation*, not about a
  decoding fix that resolves it).
- **Trained an auxiliary/multitask objective?** Yes, multiple times —
  Kordt et al. [B11] (disfluency tokens), Huang et al. [B21] (ASR+SED
  unified via LLM), Shonibare et al. [B23] (frame-level classifier
  feeding RNN-T), and the general multitask-ASR literature (auxiliary
  sequence-labeling tasks for disfluency detection, joint encoder-
  decoder objectives). **None resolves this project's own sound-vs-word
  fragment distinction specifically.**
- **Added a disfluency-preservation or acoustic-reconstruction loss?**
  Not found directly for this specific purpose — the discrete-acoustic-
  token paradigm (AudioLM/Moshi-style [B9]) has an acoustic-
  reconstruction objective by design, but not one aimed at disfluency
  specifically, and never tested against this task by anyone reviewed.
- **Used frame-level or alignment supervision?** Yes — Shonibare et al.
  [B23] (frame-level classifier), Kouzelis et al. [B24] (disfluency-
  aware forced alignment), and a CTC-based alignment-gap approach
  ([B26], "Augmenting Automatic Speech Recognition Models with
  Disfluency Detection") reporting 74.13% coverage of untranscribed
  words via a modified alignment-gap classifier — real, direct
  precedent for recovering *some* omitted material via alignment,
  though not evaluated against fragment-level `sound_repetition`
  specifically.
- **Jointly trained ASR + disfluency detection, or used a separate
  disfluency decoder/auxiliary head?** Yes, extensively — this is the
  single most well-populated solution family found (see the taxonomy
  below).
- **Used contrastive learning specifically to preserve these events?**
  Yes, for stuttering *detection* (FGCL, [B25]) — not for ASR output
  preservation.

**Conclusion for this section**: every individual *mechanism* the
project owner asked about has real, published prior work behind it —
**but no single source combines the exact mechanism (encoder-to-decoder
information recovery), the exact target (fragment-level `sound_
repetition`, distinct from word-level), and the exact evaluation
(against real speech, ideally in English given this project's own
pipeline) that this project's own problem statement requires.** This is
a more precise, better-evidenced version of the novelty claim than the
prior audit's pass — the gap is not "nobody has tried anything like
this," it is "every individual piece has been tried by someone, but not
assembled and evaluated together at this specific taxonomy
granularity."

## 9. Taxonomy of solution families

| # | Family | Who has done it | What they achieved | Limitations | Could it solve our problem? | Data needed | Compute needed | Small prototype feasible? | Requires Stage D? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Better decoding | Lea et al. 2023 [B22]; this track's own `num_beams` experiment | Lea et al.: real WER gains via LM-weight/insertion-penalty tuning. This track: 0/14 recovered via beam width. | Doesn't target fragment content directly; this track's own test of the one lever tried was a clean negative | Unlikely alone, per this track's own direct test | None new | None | Yes, already done (negative) | No |
| 2 | Better representation (different pretrained encoder) | This track's Arms 1-3 | Both alternatives tested (stock Whisper, WavLM) — clean negatives | Only 2 of many possible encoders tested; model-size confound unresolved for WavLM | Unlikely for the 2 tested; open for others | None new (reuse existing checkpoints) | Low (inference only) | Yes, already done (negative) | No |
| 3 | Better supervision (explicit disfluency labels) | Kordt et al. [B11]; AS-70's own SED track [B20] | Real marker-F1/detection accuracy on real data | Coarse taxonomy (Kordt) or detection-only, not ASR-output (AS-70) | Partially — a real, necessary ingredient, not sufficient alone | Real labeled data (exists for AS-70/Mandarin; gap for English) | Moderate (fine-tuning) | Only with adequate data | Yes, if pursued at our taxonomy's granularity |
| 4 | Multitask learning (ASR + disfluency jointly) | Huang et al. 2025 [B21]; Streaming Joint Speech Recognition and Disfluency Detection (arXiv:2211.08726) | Real, large improvements on both tasks jointly (CER -37.71% rel., SED F1 +46.58% rel.) | Real precedent, but on Mandarin/AS-70; not confirmed to produce fragment-preserving *transcripts* specifically | Yes, plausibly, the strongest existing precedent for the general strategy | Real paired data | Moderate-high | Only with adequate data | Yes |
| 5 | Auxiliary objectives (encoder retains disfluency-useful info) | General multitask-ASR literature (auxiliary sequence-labeling tasks) | Consistent improvements from lower-level auxiliary losses in encoder-decoder ASR | Not disfluency-specific in the general literature | Plausible, unconfirmed for this exact task | Real paired data | Moderate | Only with adequate data | Yes |
| 6 | Frame-level supervision | Shonibare et al. [B23] (Detect-and-Pass, RNN-T); Kouzelis et al. [B24] (disfluency-aware alignment) | Real WER reductions (12-71% relative); real alignment-recall gains (23-25% relative) | Targets WER/alignment robustness, not fragment-content emission | Plausible ingredient for a Stage D architecture, not a full solution alone | Frame/phoneme-level annotations | Moderate | Partially (the alignment piece could be prototyped without full retraining) | Partially |
| 7 | Hybrid architecture (ASR + acoustic branch + fusion) | This track's own combined-signal classifier; Huang et al. [B21]'s dual-branch design; arXiv:2605.12387's Whisper+acoustic hybrid | Mixed — this track's own attempt found no improvement over the stronger single signal; Huang et al. and arXiv:2605.12387 both found real gains on their own (different) tasks | Real tension between this track's own negative result and the literature's positive ones — task/scale-dependent, not resolved | Plausible, but this track's own most directly comparable test (the combined classifier) was a clean negative | Varies | Varies | This track already ran a version | Not necessarily |
| 8 | Purpose-built ASR (fine-tune on annotated disfluent speech) | AS-70's own ASR track [B20] (fluent-target only); Zhang et al.'s Stutter-TTS [B25] (synthetic-data fine-tuning) | Real CER improvements from fine-tuning on real (AS-70) or synthetic (Stutter-TTS) stuttering data | AS-70's own experiment targets fluent CER, not fragment preservation; Stutter-TTS uses synthetic data (the same generalization risk this track has named) | This is essentially what Stage D proposes, retargeted at fragment preservation specifically — no one reviewed has run exactly this | Real (preferred) or synthetic (real, named risk) paired data | Real (GPU) | No — this is Stage D | Yes, by definition |
| 9 | New objective (explicitly penalize normalization) | Not found directly in this review | N/A | N/A — appears to be the most conceptually novel item in this taxonomy, not because it's untried by accident but because it requires designing a new loss term, not just applying an existing one | Theoretically the most direct fit; entirely unproven | Real paired data + loss-function design work | Real (GPU) | No | Yes |
| 10 | Separate, richer recognition target (not ordinary text) | Discrete-acoustic-token speech foundation models (AudioLM/Moshi-style [B9]); SSDM's LLM-based end-to-end dysfluency system [B16] | Real acoustic-fidelity preservation (by design, for reconstruction, not disfluency specifically); SSDM's own end-to-end dysfluency modeling | Orthogonal paradigm, never tested against this project's exact taxonomy or evaluated for downstream disfluency-detector compatibility | Plausible, most architecturally different option; entirely untested for this purpose | Real paired data, likely at large scale (codec-LM training is typically data-hungry) | Very high (this is closer to training a new foundation model than fine-tuning one) | No | Yes, and the most resource-intensive version of it |

## 10. Why current ASR design may hinder our objective — the technical core

**"Recognize what the speaker meant" vs. "preserve what the speaker
physically produced"** — these are different objectives, and every
piece of evidence in this track and this literature review is
consistent with mainstream ASR training pursuing the first, not the
second, by default:

- **Training data curation**: large ASR training corpora (Whisper's
  680k hours of weakly-supervised web audio [B1]; general industry
  practice) are assembled from *existing* transcripts, which themselves
  were produced by humans or systems whose own convention is typically
  to write down intended meaning, not verbatim acoustic production —
  this is a property of how transcription has conventionally been done
  for readability, not a claim this project independently verified
  about Whisper's specific training set (which is not public).
- **Tokenization**: subword tokenizers (BPE and similar) are built to
  efficiently represent common linguistic units; a repeated phonetic
  fragment that isn't itself a common subword unit has no efficient,
  native representation in the vocabulary — Gulzar et al. 2025 [B14]
  independently found exactly this kind of tokenizer bias limiting
  cross-lingual disfluency-token performance.
- **Language-model priors in the decoder**: an autoregressive decoder
  conditions each token on its own prior outputs, which — especially
  after pretraining on mostly-fluent text — assigns higher probability
  to fluent continuations. Lin et al. 2025 [B12] directly demonstrates
  the practical consequence of this in reverse: adding hesitation
  labels to the training target *reduces* WER, because the labels
  become expected, high-probability continuations rather than
  surprising ones the decoder's own prior fights against.
- **The loss function itself**: standard cross-entropy against a
  reference transcript rewards matching that specific reference exactly
  — if the reference is itself normalized (per the training-data point
  above), the loss function has no mechanism to reward preserving
  information the reference doesn't contain, regardless of what the
  encoder's internal representation retains.

**"If the training target itself does not represent the disfluency
faithfully, why would the model learn to preserve it?"** It has no
reason to, and this track's own Arm 1 result (a bigger, non-fine-tuned
Whisper model doing no better, or worse) is directly consistent with
this: scale alone doesn't fix a training-target problem, because a
bigger model trained on the same kind of target has no more reason to
preserve fragment-level detail than a smaller one.

**"Even if the acoustic encoder contains the information, why might the
decoder have no incentive to emit it?"** Because the decoder's training
signal comes entirely from matching the reference text, and (per the
above) the reference text is where the normalization already happened
— the encoder can carry information the decoder is never asked to use,
which is exactly the information-bottleneck/probing framing in section
2, and exactly what this track's own Stage B/C result measures directly
(real encoder signal, zero decoder utilization of it).

**This leads directly to the architectural solutions in the taxonomy
above**: any fix has to intervene either at the *training-target* level
(supply a reference that faithfully represents the disfluency — families
3, 8, 9) or the *objective* level (add a loss term that doesn't depend
on the reference transcript containing the information at all —
families 4, 5, 6, 7, 10). Decoding-level interventions (family 1) and
representation-swapping (family 2) — the two cheapest, already-tested
families — do not touch either of these root causes, which is a
coherent, mechanistic explanation for why this track's own experiments
in exactly those two families came back negative.

## 11. Critically re-evaluating Stage D itself

**Not assumed correct because the roadmap says so — re-derived from
this session's evidence.**

**What would Stage D actually train?** Per the taxonomy above, the most
evidence-consistent answer is **family 8 (purpose-built fine-tuning) or
family 4 (multitask ASR+disfluency)** — not family 9 (a wholly new
objective, which is more speculative and has no direct precedent found)
and not family 10 (a new recognition target/foundation model, which is
the most resource-intensive option by far and not warranted before
cheaper, better-precedented options are tried). Concretely: a fine-tune
of CrisperWhisper or stock `whisper-large-v3` (both already
characterized by this track) with either (a) explicit fragment-level
disfluency tokens distinguishing `sound_repetition` from `word_
repetition` (extending Kordt et al.'s [B11] approach to finer
granularity), or (b) a joint ASR+SED architecture in Huang et al.'s
[B21] style, adapted to emit fragment content rather than (or in
addition to) a separate detection signal.

**What would the training target actually look like?** The project
owner's own example is the right one to formalize: not "I went to the
store" (normalized) but something carrying explicit, timestamped
fragment structure — **AS-70's own verbatim markup convention is a real,
working example of exactly this shape** ("小/r明" — a phoneme-level
repetition marker embedded inline, timestamped by construction since it
sits within the verbatim transcript), directly informing what an
English-language equivalent would need to look like: not just "the
event happened" (Kordt et al.'s coarser `REP`) but which fragment, and
by extension its own span.

**Which representation is scientifically most appropriate?** Per this
review, embedding the fragment's own content inline in the verbatim
transcript (AS-70's approach) rather than a separate marker token
(Kordt et al.'s approach) is the more information-preserving choice,
and the one most directly compatible with this project's own downstream
detector, which already expects word/fragment-level tokens with
timestamps (`profiling/detect.py`'s existing input contract) — not a
new format this project would need to build new consumers for.

**Does the AS-70 finding change whether Stage D is justified?** **Yes,
materially, but not by removing the infrastructure blocker — by
changing what the *minimum viable* Stage D pilot could look like.** The
prior audit's Re-entry Gate (still valid, not revoked) required (1) real
GPU compute, (2) a real, fragment-granular dataset, (3) a full
pre-registration, (4) explicit owner approval. AS-70 satisfies condition
(2) directly — for Mandarin. This means a **minimum-viable Stage D
pilot could, in principle, be run entirely in Mandarin against AS-70
first, as a pure feasibility/architecture test** (does *this* strategy —
family 8/4, fine-tuning with fragment-level tokens — work at all, on
data that unambiguously has the right structure), **before** committing
to the harder, still-unresolved English-data-acquisition problem this
project's own pipeline actually needs. This is a real, newly-identified,
cheaper-than-previously-thought path to de-risking Stage D's *method*
independently of its *data* problem — named here as a live option, not
decided or started.

**Is Stage D unnecessary?** No — nothing in this deeper pass changes
the section-9-gate conclusion that conditions 1 and 2 are satisfied.
**Is Stage D exactly as scoped in the prior audit?** No — the AS-70
finding is a real, material update to how the *data* requirement should
be sequenced, not a reason to abandon the plan.

## 12. Is this research paper-worthy now?

**Brutally honest, per instruction — assessed against each of the five
proposed paper types:**

**Paper A — Empirical characterization ("We systematically demonstrate
that...").** **Yes, supportable now, with real evidence.** Claims
available without Stage D: the normalization-loss rate (Stage A, n=186);
that it's not CrisperWhisper-specific (Arms 1-2); that a genuinely
different pretraining objective doesn't help at this scale (Arm 3);
that cheap acoustic features achieve real recall but not precision
(direction (g)); that combining the strongest signals doesn't beat the
best one alone (combined classifier). **What reviewers would attack**:
single-dataset (LibriStutter, synthetic) evaluation throughout; small n
for several sub-claims (WavLM `word_repetition`, n=17); no real-speech
validation anywhere. **What would make it stronger**: at minimum, a
real-speech validation pass (even a small one, e.g. against UCLASS or
a FluencyBank subset) before submission — this is the single most
impactful, and most feasible without Stage D, addition.

**Paper B — Failure analysis ("Why existing ASR systems lose disfluency
information...").** **Yes, this is arguably this track's strongest
paper angle**, combining the empirical results above with the
mechanistic explanation in section 10 (training-target/decoder-prior
framing) and the literature grounding (section 2/10's citations). This
paper type doesn't need Stage D to succeed or fail — it needs the
negative results this track already has, explained well, which this
track's own documentation already does. **What would make it stronger**:
the real-speech validation above, plus a probing-classifier experiment
directly testing the information-bottleneck framing against this
project's own encoder states (named as a real, currently-missing piece
of evidence for the mechanistic claim in section 2).

**Paper C — Representation study ("Acoustic/encoder evidence survives
despite textual normalization...").** **Yes, directly supported** by
Stage B/C's d=0.894/AUC=0.723 result and Arm 2/3's comparative
representation results — this is close to already being paper C's
central contribution. **What reviewers would attack**: the same
generalization concern, plus the modest absolute precision (4.7% at
R>=0.5) limiting how strong a "survives" claim can be made — the
honest framing is "detectable above chance, not yet practically
recoverable," which is a real, defensible, if less dramatic, claim.

**Paper D — Proposed architecture ("We propose a disfluency-preserving
ASR architecture...").** **No, not yet** — this requires Stage D itself
(or at least a scoped, executed pilot, e.g. the AS-70-first minimum-
viable pilot named in section 11) to have real results to report. The
architecture *design* (section 11 above, informed by the taxonomy in
section 9) could be a paper's proposal section, but a paper claiming to
propose an architecture without any results testing it is a weak
submission by any venue's standards.

**Paper E — Full system paper (architecture + training + evaluation).**
**No** — requires Stage D fully executed, which requires resources this
project doesn't currently have (per the Re-entry Gate). Premature to
target this paper type before Paper A/B/C exist and Stage D's minimum-
viable pilot has run.

**Overall verdict**: **Papers A, B, and C are supportable now, without
Stage D**, provided at least a small real-speech validation pass is
added first — this is the one piece of additional evidence this review
identifies as essential (not optional) before submission, because every
current quantitative claim rests on synthetic data. Papers D and E
require Stage D (or its minimum-viable pilot) to exist first. **The
single strongest, most defensible paper this track could write today is
Paper B (failure analysis) combined with Paper C's representation-level
evidence** — a negative/explanatory result paper, honestly framed, is
both this track's strongest evidence and a genuinely useful contribution
(it saves other researchers from re-testing the same cheap directions
this track has already, carefully, ruled out).

**On venues, without letting this distract from the science**: this
work's natural venue family is the same one every closely-related paper
in this review was published in — Interspeech, ICASSP, ASRU, NAACL/ACL
(for the more NLP-flavored framing), or a SLaTE-style workshop for the
verbatim-transcription-specific angle. The evidentiary standard those
venues hold (real speaker-exclusive splits, real datasets, honest
negative results) is consistent with this track's own established
methodology, not a bar this project would need to change its practices
to meet.

## 13. Explicit "unexplored terrain" verdict matrix

| Claim | Already known? | Partially known? | Apparently underexplored? | Our evidence | Confidence |
|---|---|---|---|---|---|
| ASR struggles with stuttering | **Yes** | | | Consistent (Stage A, Arms 1-3) | High |
| ASR normalizes disfluencies generally | **Yes** | | | Consistent (Stage A; matches [B13]) | High |
| CrisperWhisper attempts to preserve them (verbatim design) | **Yes** (its own stated design goal [B2]) | | | Consistent — but Stage A shows this goal is unmet for fragment-level `sound_repetition` specifically | High |
| Acoustic features can detect stuttering | **Yes**, extensively (SEP-28k/KSoF/AS-70 classifiers, YOLO-Stutter, SSDM) | | | Direction (g): real recall, weak precision — less capable than the field's best learned detectors | High (field); Moderate (our own attempt) |
| SSL representations contain stuttering information | **Yes**, generally (WavLM's own design rationale; wav2vec2/HuBERT stuttering classifiers reviewed) | | | Arm 3: not confirmed for `sound_repetition` at this model size — a real, named exception to the general pattern | Moderate |
| ASR encoders can retain information the decoder doesn't emit | | **Partially** (general information-bottleneck/probing theory; not disfluency-specific in the literature reviewed) | **Yes**, for this specific disfluency type | Stage B/C: direct, measured instance (d=0.894, AUC=0.723) | Moderate-high for our own instance; the general phenomenon is well-established theoretically |
| Changing the pretrained ASR can recover the information | **Tested and refuted**, this session (Arm 1) | | | Arm 1: worse, not better | High |
| Changing representation can recover it | **Tested and refuted**, this session (Arm 3) | | | Arm 3: chance-level | Moderate (model-size confound named) |
| Decoding changes can recover it | **Tested and refuted**, this session and by Lea et al. [B22] | | | 0/14 recovered; Lea et al.'s own decoder-tuning targeted WER broadly, not fragment recovery specifically | High |
| Acoustic candidate generation can recover it | **Tested, mixed** (real recall, weak precision) | | Whether a *learned* (not hand-engineered) version could is genuinely open | Direction (g): recall=0.824, precision=0.081-0.097 | Moderate (recall); High-confidence negative (precision, this specific method) |
| Combining weak signals can recover it | **Tested and refuted for this specific combination** | | Whether a different combination could remains open | Combined classifier: F1=0.242 vs. 0.244 alone | Moderate |
| A purpose-built training objective may be necessary | | | **This track's own reasoned conclusion**, now grounded in real precedent (families 3/4/8 in the taxonomy) | Not tested by this project; strongly precedented elsewhere | Inference, literature-supported |
| A purpose-built disfluency-preserving ASR/representation has been *proposed* | **Yes**, multiple times (Kordt et al., Gulzar et al., Huang et al., AS-70's own framing) | | | This track's own Stage D design overlaps substantially with existing proposals | High |
| A purpose-built disfluency-preserving ASR has been *demonstrated at the required level* (fragment-level, sound-vs-word distinction, in a form usable by a downstream detector) | | | **Yes, apparently underexplored** — the narrowest, most defensible gap this review identifies | No source reviewed demonstrates this exact thing | High confidence that this specific combination is undemonstrated; cannot rule out an unreviewed source |

**The narrowest genuinely defensible research gap, stated one final
time for precision**: *fine-tuning or otherwise training an ASR system
to preserve fragment-level `sound_repetition` evidence, distinguished
from word-level `word_repetition`, in a form a downstream detector can
consume, evaluated against real (not synthetic) speech* — every piece
of this has real precedent individually; the combination does not
appear to have been demonstrated by anyone reviewed.

## 14. The wall, independently re-verified

**Every claim below is either directly checked against this project's
own environment, or grounded in a real, cited published number — no
cost is invented.**

- **Local GPU**: re-confirmed unchanged from the prior audit —
  `torch.cuda.is_available()` is `False`, CPU-only `torch` build,
  integrated-only graphics. Not re-tested a second time this pass since
  nothing about the local machine changed; carried forward as an
  already-verified fact.
- **Data scale, grounded in two real published examples rather than
  guessed**: Kordt et al. [B11] achieved real (if imperfect) marker
  learning with ~10-20 hours per dataset (SME 11.79h, Delaware 9.72h,
  a 12.11h Pitt subset) on a *smaller* model (`whisper-small.en`). AS-70
  [B20] used 48.8 hours to fine-tune `whisper-large-v2` down from
  27.20% to 8.75% CER (on the fluent-target task, not fragment
  preservation — see the material revision above) — a *larger* model,
  larger dataset. **Reasonable range for a minimum-viable pilot**:
  10-50 hours of appropriately-annotated speech, depending on model
  size and target scope, bounded by these two real examples, not
  invented.
- **Annotations needed**: word/fragment-level timestamps plus a
  sound-vs-word repetition distinction — AS-70's own annotation process
  is a real, working template ("approximately three times more time
  compared to annotating fluent speech," per its own paper), useful as
  a real effort-multiplier reference if new annotation work is ever
  needed.
- **Hardware**: Kordt et al.'s `whisper-small.en` runs are the more
  conservative comparison point (10 epochs, batch size 16, lr 2e-5) —
  consistent with a single modern cloud GPU instance (16-24GB VRAM
  class) being plausible for a *reduced-scale* pilot; AS-70's own
  Whisper-large-v2 fine-tune implies a larger-VRAM requirement (not
  specified in their paper) for full-scale-checkpoint work. **Neither
  number is this project's own measurement** — both are inferences from
  published setups training comparable model classes, not confirmed
  costs for this project's specific target.
- **GPU memory, training time, expected experiments**: not stated
  precisely by either reference paper reviewed, and this review does
  not invent numbers neither paper provides — the honest answer is
  "plausible on a single mid-range cloud instance for a reduced-scale
  pilot, unconfirmed for full CrisperWhisper-scale," stated as a range
  and an inference, not a quote.
- **Minimum-viable prototype**: per section 11's re-derived plan, an
  AS-70-first, Mandarin-only, `whisper-small`-scale pilot testing
  whether fragment-level (not just coarse REP-style) tokens can be
  learned at all — cheaper and more de-risked than this project's own
  original English-first framing, because it reuses AS-70's existing,
  real, right-granularity data instead of waiting on new English data
  acquisition.
- **What cannot realistically be tested on this project's current
  machine**: any GPU-bound training run at any of the scales referenced
  above — confirmed, not re-litigated, from the prior audit.
- **Cloud infrastructure**: a single rented GPU instance (exact
  provider/tier not specified here, since actual pricing changes over
  time and this review does not want to assert a number that will be
  wrong by the time it's read) is very plausibly sufficient for the
  minimum-viable pilot in section 11 — a real, bounded claim, not "we
  don't know."
- **What a venture/research lab would need to provide**: (1) cloud GPU
  budget/access for the scale above; (2) either direct engagement with
  AS-70 (Mandarin pilot) or real English-language data acquisition work
  (FluencyBank Timestamped access + an unbuilt CHAT parser, or new
  annotation work); (3) engineering time for the training pipeline this
  project does not currently have; (4) time for the pre-registration
  and evaluation this track's own discipline requires before any claim
  is trusted.
- **Approximate resource categories, without inventing exact costs**:
  compute (low-to-moderate for a reduced-scale pilot, per the two
  real published comparisons above; higher and less certain for
  full-scale CrisperWhisper fine-tuning); data (low additional cost if
  AS-70-first; moderate-to-high if English-first, given real
  acquisition/annotation work); engineering (moderate — a real but
  bounded new codebase, substantially aided by this project's own
  already-built evaluation harness); time (a real, multi-week-scale
  undertaking at minimum, not a single-session task, consistent with
  every published precedent reviewed taking a dedicated research effort
  to produce its own results).

## 15. Scientific gap, or resource gap? An explicit distinction

**Not a scientific gap in the sense of "nobody knows how to solve
this."** Every individual technique this review found evidence for
(explicit disfluency tokens, multitask ASR+SED, frame-level
classifiers, disfluency-aware forced alignment, LoRA/parameter-
efficient adaptation, continual learning to control forgetting) is a
known, published, working method *somewhere* in this space. The field
knows how to build disfluency-aware ASR in general.

**It is, narrowly, an engineering-and-assembly gap**: nobody reviewed
has assembled these known pieces at this project's own specific
taxonomy granularity (sound- vs. word-level, distinctly), on real
speech, in a form a downstream detector can consume. This is real
engineering work, not a research mystery — the components exist, they
have not been put together this way.

**It is also, distinctly, a data gap — but a narrower one than the
prior audit stated**: real, right-granularity data exists (AS-70), but
not in English, and this project's own pipeline (CMU phonetic
dictionaries, English filler-word lists, an English-specific detector)
is not trivially portable to Mandarin. The data gap is real and
specific to *this project's own language requirement*, not a gap in
the field's data resources generally.

**It is unambiguously a compute gap for this project specifically**:
directly verified, no GPU locally, and every real published comparison
point (Kordt et al., AS-70) required GPU-scale training this project's
current machine cannot provide.

**Combined verdict**: **a real engineering-and-assembly gap, compounded
by a project-specific (not field-wide) data gap for English, compounded
by a project-specific compute gap.** Not a scientific unknown. This
distinction matters directly for how this should be presented to a
venture, lab, or collaborator: the ask is "resource and engineering
support to assemble and adapt known methods to a specific, well-
evidenced, narrow gap," not "fund open-ended research into an unsolved
problem" — a materially different, more concrete, more fundable pitch.

## 16. Final research-positioning verdict

**What did the field know before us?** That ASR is biased against
disfluent speech generally; that disfluencies can be detected from
audio with real accuracy; that fine-tuning ASR with explicit disfluency
markers can improve verbatim transcription for some disfluency types
(fillers) and coarse repetition categories; that joint ASR+detection
architectures can improve both tasks together on real stuttering data
(AS-70/Huang et al.); that real, right-granularity annotated stuttering
data exists — in Mandarin.

**What did we discover?** That this specific problem — fragment-level
`sound_repetition` loss at otherwise-correctly-transcribed positions —
is large (roughly half of instances, Stage A) even for a model
fine-tuned specifically for verbatim transcription; that neither model
scale, model family, nor pretraining objective (of the three tested)
fixes it; that a real, duration-independent representation-level signal
exists but is precision-limited alone; that a hand-engineered acoustic
signal exists independently of ASR entirely, with real recall but not
usable precision; and that combining the strongest signals available
doesn't beat the stronger one alone.

**What did we confirm rather than discover?** That ASR normalizes
disfluent speech generally (already well-published); that catastrophic
forgetting is a real risk for disfluency-token fine-tuning (Kordt et
al. already measured this; we confirmed it's consistent with our own
literature-review reasoning, not with a new experiment of our own).

**What did we rule out?** Direction (a) (different pretrained ASR),
direction (b) (different pretrained representation, for the two tested),
decoding-width changes, two specific hand-engineered acoustic features
alone, and one specific 4-feature combination — each ruled out by a
real, pre-registered, honestly-reported experiment, not by assumption.

**What remains unknown?** Whether any untested representation, a
*learned* (not hand-engineered) acoustic detector, a combination
involving a different second signal, or — most directly — a taxonomy-
targeted fine-tune (Stage D) would succeed. Whether any of this
project's findings generalize to real (non-synthetic) speech at all —
genuinely untested, the single largest open question this entire track
carries.

**What is genuinely novel?** The narrowest defensible claim, restated
one final time: *fine-tuning or otherwise training an ASR system to
preserve fragment-level `sound_repetition` evidence, distinguished from
word-level `word_repetition`, evaluated against real speech, has not
been demonstrated by any source this review found* — every component
technique has precedent; this specific assembly and evaluation does
not.

**How strong is the evidence?** Real and internally consistent for what
it claims (see section 6's credibility table), but bounded by two
honest, load-bearing limitations named repeatedly rather than once:
small sample sizes at the sub-claim level (WavLM `word_repetition`,
n=17), and — the larger one — zero validation against real, non-
synthetic disfluent speech anywhere in this track's own experiments.

**What is the exact current problem?** Section 7's four problem
statements, above — most precisely, the technical research-problem
version.

**Has anyone already solved it?** No, per this review's own search —
the closest (Kordt et al.) collapses exactly the distinction this
project's taxonomy treats as two separate types; AS-70's own authors
had the right data and did not run the fragment-preservation fine-tune
experiment.

**Has anyone proposed a plausible solution?** Yes — assembling
families 3/4/8 from the taxonomy in section 9 is a direct, literature-
grounded, non-speculative proposal, not a novel invention.

**Why haven't we tested it?** Two real, verified reasons: no local GPU,
and (until this pass) an assumed absence of adequate real data that
this pass found to be partially wrong (AS-70 exists, for Mandarin) —
both reasons are infrastructure/data/language-scope reasons, not
scientific ones.

**Is the barrier scientific, data-related, computational, or
engineering?** Per section 15: primarily engineering-and-assembly,
compounded by a project-specific (English-language) data gap and a
project-specific compute gap — not a field-wide scientific unknown.

**Is Stage D justified?** Yes, per this track's own pre-registered §9
gate, whose conditions 1 and 2 are satisfied by the accumulated
evidence and whose condition 3 is now understood more precisely (real
data exists, in the wrong language; compute does not exist locally,
plausibly does in the cloud at bounded, literature-grounded cost).

**What exactly should Stage D attempt?** Per section 11: a fine-tune
(CrisperWhisper or stock `whisper-large-v3`) with fragment-level
disfluency tokens (extending Kordt et al.'s approach to finer
granularity) or a joint ASR+SED architecture (extending Huang et al.'s
approach to emit fragment content).

**What is the minimum viable Stage D?** An AS-70-first, Mandarin-only,
reduced-model-scale pilot testing whether fragment-level tokens can be
learned at all, before committing to English-language data acquisition
— a real, newly-identified, cheaper starting point than this track's
own original framing assumed.

**Could the current work already form a research paper?** Yes — Papers
A, B, and C (section 12), with one essential addition (a real-speech
validation pass) before submission.

**What would the paper's central contribution be?** Most likely Paper
B's framing: a mechanistically-explained, literature-grounded failure
analysis of why cheap, off-the-shelf approaches don't close this
specific gap, combined with Paper C's direct representation-level
evidence that the information is there but inaccessible.

**What additional evidence would make the paper substantially
stronger?** A real-speech validation pass (even small-scale); a direct
probing-classifier test of the information-bottleneck framing in
section 2; and, if pursued, the AS-70-first Stage D pilot's own result,
positive or negative.

**What is the single strongest defensible novelty claim?** *We provide
direct, controlled, multi-angle evidence (representation swapping,
model-family swapping, decoding-parameter variation, and acoustic-
signal combination) that no cheap, off-the-shelf intervention recovers
fragment-level `sound_repetition` evidence lost during ASR
normalization — narrowing, with real evidence rather than assumption,
exactly which of the field's many disfluency-preservation strategies
remains genuinely untested for this specific taxonomy granularity.*

---

## Research Position as of 2026-08-07

**Current problem statement**: Off-the-shelf ASR systems, including one
fine-tuned specifically for verbatim transcription, normalize away
sub-word repetition evidence (`sound_repetition`) even at otherwise-
correctly-transcribed positions, and no cheap representation, decoding,
or acoustic-signal-combination strategy tested recovers it at usable
precision.

**Established facts**: ~45% of `sound_repetition` and ~40% of `word_
repetition` losses occur at word-correct positions (Stage A, n=186);
CrisperWhisper's own encoder carries a real, duration-independent
signal for this (d=0.894, AUC=0.723) that its decoder never emits;
this pattern is Whisper-architecture-general, not CrisperWhisper-
specific (Arm 2); a bigger, non-fine-tuned model doesn't fix it (Arm 1).

**Strongest experimental evidence**: Stage A's normalization-rate
finding (large n, careful methodology) and the Stage C duration-
confound refutation (a genuine mechanistic test, not just an
association).

**Literature consensus**: ASR's bias against disfluent speech is
well-established and independently replicated by this project;
disfluency-preserving fine-tuning is an active, real research area with
working methods, none yet demonstrated at this project's specific
fragment-level taxonomy granularity.

**Unresolved research gap**: fine-tuning or training an ASR system to
preserve fragment-level `sound_repetition` evidence, distinct from
word-level `word_repetition`, evaluated against real speech, in a form
a downstream detector can consume.

**Current proposed solution**: Stage D — a taxonomy-targeted fine-tune
of CrisperWhisper or stock `whisper-large-v3`, most likely combining
explicit fragment-level disfluency tokens with a joint ASR+detection
objective, informed directly by Kordt et al. [B11] and Huang et al.
[B21]'s published methods.

**Why it has not yet been tested**: no local GPU (verified directly);
no adequate real *English*-language fragment-granular dataset (AS-70
resolves this for Mandarin, not English) — both infrastructure/data
gaps, not scientific ones.

**Exact Stage D requirements**: real GPU compute (cloud, bounded cost
per two real published comparisons); a real dataset at the right
granularity (AS-70 now, for a Mandarin pilot; FluencyBank Timestamped
or new annotation work, for English); a training pipeline this project
does not yet have; a full pre-registration before any code, per this
track's own standing discipline.

**Paper potential**: real, now, for an empirical-characterization/
failure-analysis/representation-study paper (Papers A/B/C), pending one
real-speech validation pass; a proposed-architecture or full-system
paper (D/E) requires Stage D's own results first.

**Next decision point**: whether to pursue the AS-70-first, Mandarin,
minimum-viable Stage D pilot (cheaper, faster, tests the *method*
independent of the English-data problem) or the English-first path
(slower, but directly serves this project's own production language) —
a real, live decision for the project owner, not resolved by this
review.

**Revised by the novelty-falsification pass immediately below, marked
explicitly rather than silently corrected**: the Mandarin pilot can
validate the *training recipe* (verbatim-target fine-tuning,
catastrophic-forgetting control) using StammerTalk's own directly-
demonstrated approach, but per that pass's own structural finding
(Chinese orthography likely cannot represent a sound-level fragment as
literal text at all), it cannot test this project's actual L4
hypothesis for `sound_repetition` — the English-language data question
is not just a practical inconvenience, it is plausibly load-bearing for
the specific scientific question Stage D exists to answer. See "Novelty-
falsification pass — 2026-08-07" below for the full reasoning.

## Additional Bibliography — sources verified in the deep pass (2026-08-07)

Continues the numbering from the Bibliography section above. Every
source below was fetched and read directly this session (full PDF, for
the two most load-bearing — AS-70 and the Kordt et al. paper cited
above — read completely, table by table, not summarized from a search
snippet).

- [B20] Gong, R., Xue, H., Wang, L., Xu, X., Li, Q., Xie, L., Bu, H.,
  Wu, S., Zhou, J., Qin, Y., Zhang, B., Du, J., Bin, J., & Li, M.
  (2024). *AS-70: A Mandarin Stuttered Speech Dataset for Automatic
  Speech Recognition and Stuttering Event Detection.* Interspeech 2024.
  https://arxiv.org/abs/2406.07256 — full text fetched and read
  directly, including all five result tables; dataset download:
  https://www.aishelltech.com/AISHELL_6A
- [B21] Huang, S., Deng, J., Kang, J., & Zheng, R. (2025). *Leveraging
  LLM for Stuttering Speech: A Unified Architecture Bridging Recognition
  and Event Detection.* Interspeech 2025.
  https://arxiv.org/abs/2505.22005
- [B22] Lea, C., Huang, Z., Narain, J., Tooley, L., Yee, D., Tran, D. T.,
  Georgiou, P., Bigham, J. P., & Findlater, L. (2023). *From User
  Perceptions to Technical Improvement: Enabling People Who Stutter to
  Better Use Speech Recognition.* CHI 2023.
  https://arxiv.org/abs/2302.09044
- [B23] Shonibare, O., Tong, X., & Ravichandran, V. (2022). *Enhancing
  ASR for Stuttered Speech with Limited Data Using Detect and Pass.*
  https://arxiv.org/abs/2202.05396
- [B24] Kouzelis, T., Paraskevopoulos, G., Katsamanis, A., & Katsouros,
  V. (2023). *Weakly-supervised forced alignment of disfluent speech
  using phoneme-level modeling.* Interspeech 2023.
  https://arxiv.org/abs/2306.00996
- [B25] Zhang, X., Vallés-Pérez, I., Stolcke, A., Yu, C., Droppo, J.,
  Shonibare, O., Barra-Chicote, R., & Ravichandran, V. (2022).
  *Stutter-TTS: Controlled synthesis and improved recognition of
  stuttered speech.* NeurIPS 2022 Workshop on SyntheticData4ML. (cited
  via AS-70's own reference list [B20], not independently fetched this
  session — flagged as secondary sourcing, not directly verified.)
- [B26] *Augmenting Automatic Speech Recognition Models with Disfluency
  Detection.* https://arxiv.org/abs/2409.10177 — the 81.62%-accuracy /
  74.13%-coverage alignment-gap-classification finding cited in Part 8
  above comes from this paper per search-result attribution; the exact
  page/section was not independently re-fetched this session for
  page-level confirmation — flagged as a slightly lower-confidence
  citation than the fully-read sources above, consistent with this
  project's own citation-honesty discipline.
- [B27] Li, J., Li, Q., Gong, R., Wang, L., & Wu, S. (2025). *Our
  Collective Voices: The Social and Technical Values of a Grassroots
  Chinese Stuttered Speech Dataset.* Proceedings of the 2025 ACM
  Conference on Fairness, Accountability, and Transparency (FAccT '25).
  https://doi.org/10.1145/3715275.3732179 — full text (16 pages,
  including appendix) fetched and read directly this session via the
  authors' own open preprint: https://www.shaomei.info/pdfs/FAccT25_Our_
  collective_voices_preprint.pdf. This is the technical paper behind
  the `AImpower/StutteredSpeechASR` Hugging Face model
  (https://huggingface.co/AImpower/StutteredSpeechASR) and the
  `AImpower/MandarinStutteredSpeech` dataset — the same underlying
  data-collection effort as AS-70 [B20] (overlapping authors, identical
  scale and annotation schema, referred to as "StammerTalk" in this
  paper).

Also referenced narratively in this deep pass without a dedicated
citation number (general search-result characterization, not
individually fetched and verified this session): general multitask-ASR
auxiliary-objective literature (Auxiliary Sequence Labeling Tasks for
Disfluency Detection, arXiv:2011.04512; Multitask Learning with
Low-Level Auxiliary Tasks, arXiv:1704.01631); FGCL fine-grained
contrastive learning for Mandarin stuttering event detection,
arXiv:2410.05647; the information-bottleneck/probing framing (general
characterization from search results discussing encoder-decoder mutual
information, not a single specific paper independently verified this
session). These are marked explicitly as narrative context, not
load-bearing citations for any specific numeric claim in this document.

---

# Novelty-falsification pass — 2026-08-07: operationalizing L1-L4, and a further, material revision

**Explicit instruction honored, again**: the deep pass above is treated
as something to stress-test, not defend. This pass — prompted by the
project owner's own direct verification of AS-70's primary source and
a request for an even stricter operationalization — traces the
`StutteredSpeechASR`/AS-70 lineage to its actual technical paper (not
just a model card) and re-examines Kordt et al. specifically for
whether its `REP` token preserves fragment *content*, not just
presence. **One further material finding emerges, revising — not just
extending — this document's most recent recommendation** (the
"AS-70-first, Mandarin pilot" from the deep pass above). Marked
explicitly below, not silently corrected.

## The operationalized question, exactly as posed

Four progressively stricter levels:

- **L1**: Has anyone trained ASR specifically for stuttered speech?
- **L2**: Has anyone trained ASR to produce verbatim/disfluency-aware
  output rather than normalized semantic transcription?
- **L3**: Has anyone explicitly represented different disfluency types
  in the ASR output?
- **L4 (this track's actual question)**: Has anyone trained/evaluated
  ASR to preserve the actual content/form of a repeated sound or
  fragment (e.g. "c-c-cat"), rather than merely emitting a generic
  `<sound_repetition>` marker or detecting that one occurred?

## Tracing `StutteredSpeechASR` to its primary source

The Hugging Face model card alone (fetched first) does not specify
output format precisely enough to answer L4 — it lists "word
repetitions and interjections" as preserved (**notably not sound
repetitions**), gives no example input/output pair, and does not
formally define "verbatim." It links to Li, Li, Gong, Wang, & Wu,
**"Our Collective Voices: The Social and Technical Values of a
Grassroots Chinese Stuttered Speech Dataset"** (FAccT 2025, [B27]),
fetched and read in full (16 pages, including all tables and the
appendix) — not summarized from the model card or a search snippet.

**Confirmed directly**: `StammerTalk` (this paper's name for the
dataset) and `AS-70` (Gong et al.'s Interspeech 2024 paper, [B20]) are
the same underlying data-collection effort — overlapping authors (Gong,
Wang), identical scale (48.8-50h, 70-72 speakers), identical five-type
annotation schema (`[]`/`/b`/`/p`/`/r`/`/i`). `StutteredSpeechASR` is
this paper's own fine-tuned Whisper-large-v2 (LoRA/AdaLora), trained on
what this paper calls "literal transcription" — **the disfluency
markup symbols are stripped for scoring, but the underlying repeated
*text* is real, verbatim content, not a marker** (e.g., "My, my, my
name" is the literal fine-tuning target, kept as three real words).

**Real, quantified L2 result, directly verified**: deletion-error rate
(the error type that fires when ASR omits/smooths a repetition) dropped
from 26.56% to 2.29% (severe stuttering) and 15.77% to 1.27% (moderate)
after fine-tuning on literal transcriptions. This is a genuine,
first-party-verified demonstration that fine-tuning toward a verbatim
target measurably stops a model from deleting repeated content — L2,
solved, on real speech, not inferred.

**The critical, precise evidence for L4, read directly from the paper's
own Tables 3-4** (baseline-model qualitative examples, given to
illustrate the "smoothing" problem, not fine-tuned-model outputs — this
distinction matters and is preserved below): one example shows a
genuine `/r` (sound-repetition) annotation on a single character —
"自**/r**卑的" (roughly: the character 卑, in 自卑/"feel inferior," marked
as having a sound-level repeat). The baseline wav2vec2.0 output at that
position is a single, unrelated homophone-substituted character — **no
repeat structure survives at all for this specific sound-level event**,
even in the same utterance where a *word*-level repeat elsewhere
partially survives (garbled into wrong homophones, but still
structurally repeated). **A real, honest gap in this paper's own
demonstrated evidence, not filled by inference**: no corresponding
example of the *fine-tuned* model's output for a `/r` event was found —
the paper reports aggregate CER/deletion-rate improvements by severity,
not a qualitative before/after comparison isolated to sound-repetition
events specifically. This means the paper demonstrates L2 in aggregate
and demonstrates the *baseline* problem for `/r` events concretely, but
does **not** demonstrate that its own fine-tuned model recovers
sound-level fragment content specifically.

## A structural finding — clearly marked as this review's own inference, not a claim either paper makes

Comparing the concrete annotation convention across both papers
("小/r明", "自/r卑的" — a `/r` marker always attaches to one existing,
whole character) against what L4 actually requires (writing out the
repeated fragment's own content, as English does with "c-c-"): **Chinese
orthography has no sub-character grapheme available to represent a
phonetic fragment as literal text.** A word-level repeat can be written
as genuinely repeated whole characters (我我我 is three real characters,
each a complete linguistic unit) — Chinese script handles this natively,
which is exactly why StammerTalk's word-level L4-equivalent preservation
above is real and demonstrated. A sound-level repeat (part of a single
character's pronunciation) has no comparable sub-character written
form to spell out. **This is this review's own reasoned inference from
directly-verified facts (the annotation convention; the general,
independently-verifiable property of Chinese as a logographic/syllabic
script), not a claim stated by Gong et al., Li et al., or any other
source reviewed** — flagged explicitly as inference, per the project
owner's own instruction not to make a novelty claim the primary sources
don't support.

**If this inference is correct, its consequence is significant**: even
a hypothetically perfect Mandarin verbatim ASR could mark that a
sound-repetition occurred (L3) but likely cannot spell out the
fragment's own content (true L4) — not because nobody has tried, but
because the script itself may lack the representational unit. This
would mean AS-70/StammerTalk, despite being the best real, right-
taxonomy-granularity data this entire review has found, may be
**structurally unable to serve as a testbed for this project's own
actual L4 question for `sound_repetition` specifically** — a limitation
of the language/script pairing, not of the dataset's quality or the
researchers' effort.

## Re-examining Kordt et al. specifically for L3 vs. L4

Confirmed directly (Methodology, already quoted in the deep pass above):
"we aggregate similar disfluency events into four distinct token types:
filler words (FILLER...), repetitions/revisions (REP; **word- and
phoneme-level**)..." — `REP` is explicitly a single, generic marker
across both granularities. This is **L3 at best (event-type
representation), not L4** — confirmed, not inferred, directly from
their own stated aggregation choice. **A genuine, unresolved gap in
this review's own verification, stated honestly**: whether the CHAT-
format source transcripts (SME/Pitt/Delaware) contained finer-grained
phonological-fragment markup *before* Kordt et al.'s own aggregation
step — the CHAT standard does have conventions for this in principle —
was not confirmed one way or the other from the portions of the paper
read. This is left as an open question, not resolved by assumption in
either direction.

## The revised three-level verdict

| Level | Verdict | Evidence, precisely |
|---|---|---|
| **L1** — ASR trained specifically for stuttered speech | **Already solved** | AS-70/StammerTalk, Kordt et al., and others in this review's own literature landscape — directly demonstrated, real data |
| **L2** — verbatim/disfluency-aware output vs. normalized semantic transcription | **Already solved** | StammerTalk's fine-tune: deletion-error rate 26.56%->2.29% (severe), 15.77%->1.27% (moderate) — real, quantified, real speech, directly verified this pass |
| **L3** — explicit disfluency-type representation in output | **Partially solved / closely related** | Kordt et al.: 4 coarse types, `REP` conflates word- and phoneme-level, confirmed directly. StammerTalk: real word-content preservation *without* explicit typing |
| **L4** — actual fragment *content* preserved (`c-c-cat`), specifically for `sound_repetition` | **Exact problem remains unaddressed** | No source found demonstrates this for sound-level fragments specifically. Word-level content preservation is demonstrated (StammerTalk) — a different, already-solved sub-problem. The one available concrete `/r` example shows no repeat structure surviving even in a verbatim-oriented pipeline's baseline output, and (this review's own inference) Mandarin's script may structurally preclude testing this at all |

## What this means for Stage D, revised precisely

**Assembling L1+L2+L3 techniques would not be a new contribution** —
directly demonstrated, repeatedly, on real speech, by sources this
review read in full. **A genuinely new contribution requires the
L4-for-`sound_repetition` combination specifically** — and, per the
structural finding above, **this most plausibly needs to be attempted
in an alphabetic-script language (most directly, English, this
project's own working language) rather than Mandarin**, since Mandarin's
best-available real data may be unable to represent the target
information at all, independent of model or training-method choice.

**This revises, not merely extends, the deep pass's "AS-70-first,
Mandarin pilot" recommendation above.** That pilot remains real and
useful for de-risking the *engineering approach* (verbatim-target
fine-tuning, controlling catastrophic forgetting, per StammerTalk's own
directly-demonstrated recipe) — but it **cannot test this project's
actual L4 hypothesis for `sound_repetition`**, which is the specific
question Stage D exists to answer. The corrected sequencing: a Mandarin
pilot could still be a cheap, real way to validate the *training
recipe* before committing GPU time to the harder case, but it is not,
on its own, a substitute for eventually testing the fragment-content-
preservation objective in English (or another alphabetic-script
language) — the English-language data gap named in the first "Final
research audit" section above is, per this pass, not just a practical
inconvenience but plausibly load-bearing for the specific scientific
question this project cares about.

**Would Stage D, scoped this way, constitute a meaningful new
contribution, or merely reassemble an existing technique?** Per this
now-exhaustively-checked review: **a meaningful new contribution**,
specifically because L4-for-sound-level-fragments has not been
demonstrated by any source found, and because the one body of work with
adequate real data and the right taxonomy granularity (AS-70/
StammerTalk) may be structurally unable to test it. Reassembling known
techniques (verbatim fine-tuning + explicit typing + catastrophic-
forgetting control) is real, necessary engineering — but applying them
to English-language sound-fragment content preservation specifically
is the part this review could not find prior art for, after genuinely
trying.

---

# Application-Objective Decision Analysis — 2026-08-07

**This section reinterprets everything above through the actual
product objective (see "PROJECT OBJECTIVE" at the top of this
document), not through "is this a novel research gap."** Nothing below
is a new experiment — per the project owner's own instruction, this is
synthesis of evidence already gathered, reread through the correct
lens. Every citation, number, and finding from the sections above is
reused, not rerun. Distinctions are marked throughout: **experimentally
demonstrated** (this project's own measured result), **supported by
literature** (a cited, verified external source), **inferred**
(reasoned from evidence, not directly measured), **unresolved** (a real
open question), and **proposed** (a recommendation, not a finding).

## 1. What does the application actually need?

Precisely, separated by category — and explicitly **not** assuming any
of this must appear in the ASR transcript itself:

- **Detection** (does a disfluency exist at this point in the audio?) —
  **required, the foundational need**. Everything downstream depends on
  this firing reliably.
- **Localization** (precise start/end timing of the disfluent span) —
  **required**. A downstream replacement/rephrasing step needs to know
  exactly which audio/text span to act on. Word-level timestamp
  precision is very likely sufficient — **inferred**, not measured;
  nothing in this project's own evidence suggests sub-phoneme timing
  precision is needed for a span-replacement operation, as opposed to a
  clinical/research display use case.
- **Classification** (which type: `sound_repetition`, `word_
  repetition`, `phrase_repetition`, `prolongation`, `block`,
  `interjection`/filler) — **required**. Different types plausibly need
  different downstream handling (a filler is likely simply deleted; a
  repetition's redundant copies need collapsing to one; a block's
  silence needs different treatment than a prolongation's stretched
  sound) — **proposed as a design assumption**, not yet validated
  against actual downstream requirements, since the downstream
  (Speech AI Module) side of this is out of this document's own scope.
- **The intended/clean linguistic content** (what the speaker meant to
  say, usable as the replacement target) — **required**, and — this is
  the key reframing — **typically already available** even when the
  disfluency itself is normalized away. Stage A's own finding
  (`ASR_RESEARCH_TRACK.md` §8, this document, 45.2%/40.5% normalized-
  away rates) is, read this way, evidence that the decoder usually
  *does* correctly produce the intended word (e.g., "cat") — the loss
  is specifically in the *disfluency evidence preceding it* (the
  "c-c-" or the first "want"), not in the clean target itself.
- **The actual disfluent realization's literal content** (e.g., the
  precise phonetic shape of a repeated fragment, spelled out as text)
  — **not obviously required** for the stated downstream goal (replace
  with fluent/synonym phrasing). This would matter for a verbatim-
  display or clinical-analysis feature, which is not what the project
  owner's own restated objective asks for. Distinguishing this from
  "detection + localization + classification" is the single most
  important separation this reframing makes.
- **Acoustic evidence as a detection/classification mechanism** (not as
  an end in itself) — required as a *means*: this project's own
  `block`/`prolongation` detectors already prove acoustic-native
  detection can work in production (**experimentally demonstrated**,
  shipped); this track's own Stage B/C and direction (g) tested whether
  the same strategy extends to `sound_repetition`/`word_repetition`.

## 2. What does the current pipeline already provide?

Mapped directly against the requirements above, using only this
project's own already-measured results:

| Requirement | `block`/`prolongation` (shipped) | `filler` (shipped) | `sound_repetition`/`word_repetition` via ASR text (shipped) | `sound_repetition`/`word_repetition` via encoder/acoustic (this track) |
|---|---|---|---|---|
| Detection | **Solved** — acoustic-native, no ASR text needed | **Solved** — text-based, low normalization risk (fillers are rarely fully deleted the way repetitions are, per this project's own detector design) | **Real gap** — Stage A: ~45%/~40% of true instances produce no text-based candidate at all | **Partial** — real signal (encoder d=0.894; acoustic recall=0.824) but insufficient precision alone (best combined: P=0.314) |
| Localization | Solved (waveform-native timing) | Solved (word timestamps) | Solved *when* a candidate exists | Solved *when* a candidate exists (acoustic-native candidates carry their own timing) |
| Classification | Solved | Solved | Solved *when* a candidate exists | N/A — this track's mechanisms only address `sound_repetition` specifically, not full classification |
| Clean/intended content | N/A (not applicable to these types) | N/A | **Usually already correct** even when the disfluency itself is lost (this document's own reframing above) | Same — this track's work targets detection, not content recovery |

**What this table makes explicit that the novelty-centered framing
did not**: the current pipeline's real, measured gap is narrowly
**detection** for `sound_repetition`/`word_repetition` specifically —
not localization, not classification (once detected), and not recovery
of the intended replacement content (which is usually fine already).
This narrows the actual engineering target substantially compared to
"build an ASR that preserves verbatim fragments."

## 3. What has the literature already solved, reread for our application's actual requirement

Re-examining every major system from the two prior passes, now scored
against **detection/localization/classification**, not verbatim
fragment-content preservation:

| System | Objective | Input/output | Preserves verbatim? | Detects disfluency? | Preserves fragment *content*? | Distinguishes sound- vs. word-level? | Preserves timing? | Dataset / language | Solves *our* requirement? |
|---|---|---|---|---|---|---|---|---|---|
| CrisperWhisper (this project's own, as used) | Verbatim-timestamped transcription | Audio -> text + word timestamps | Partially (word-level) | Only via downstream text-candidate matching, which Stage A shows fails ~45% of the time | No (this track's own finding) | N/A | Yes, word-level | N/A (general) | **Partially** — strong on localization/timing when a candidate exists; weak on detection specifically |
| Stock `whisper-large-v3` (Arm 1) | Same as above, no verbatim fine-tune | Same | No, worse | Worse than CrisperWhisper (**experimentally demonstrated**, this track) | No | N/A | Yes | N/A | **No**, strictly worse than what's already shipped |
| WavLM (Arm 3) | Paralinguistic-aware self-supervised representation | Audio -> embeddings, no decoding | N/A (not a transcription system) | Weak for `sound_repetition` at this scale (**experimentally demonstrated**, chance-level AUC) | N/A | N/A | Frame-level, if paired with a decoder | N/A | **No**, for this specific detection task, at this scale |
| StammerTalk/AS-70 fine-tuned Whisper (**StutteredSpeechASR**) [B20][B27] | Verbatim Mandarin transcription for stuttered speech | Audio -> literal (verbatim) text | **Yes, real, demonstrated** — deletion-error rate dropped 26.56%->2.29% (severe) | **Yes, indirectly** — a model that stops deleting repeated words produces exactly the kind of text-based candidate this project's own detector needs to fire | Partially — word-level repeats preserved as real text; sound-level fragments not demonstrated (this pass's own finding) | No (annotation exists in training data but not confirmed emitted distinctly) | Yes, word-level | Real Mandarin speech, 48.8h | **Yes, for detection via decoder-output normalization specifically** — the closest real precedent to what this project's own gap actually needs |
| Kordt et al. [B11] | Verbatim transcription with explicit disfluency tokens | Audio -> text + marker tokens | Yes, with markers | Yes, directly — a `REP` token is itself a detection signal | No (confirmed, merges word/phoneme level) | No (confirmed) | Token-level, not sub-word | Real (non-stuttering) disfluent speech, English | **Yes, for detection specifically** — an explicit marker token is arguably *more* directly useful for this project's own detection need than spelled-out fragment content would be |
| Huang et al. [B21] (joint ASR+SED) | Joint transcription + stutter-event detection | Audio -> text + separate SED signal | Not the focus | **Yes, strongly** — SED F1 73.63%, +46.58% relative over separate training | No (SED is a detection signal, not fragment text) | Not confirmed | Detection-level | Real Mandarin (AS-70), stuttering-specific | **Yes, very directly** — this is almost exactly the shape of solution this project's application needs: strong joint detection alongside transcription |
| YOLO-Stutter [B15] / SSDM [B16] | Audio-native dysfluency region detection | Audio -> region + type, no transcript needed | N/A | **Yes, directly, by design** | N/A (detection-only) | Yes, by taxonomy (repetition is a labeled class) | Yes, region-level, time-accurate | Simulated (VCTK-Stutter/Libri-Dys) | **Plausibly yes** — architecturally the most direct match to "detect + localize + classify without needing verbatim text," never tested against this project's own data |
| This track's own encoder-distance signal (Stage B/C) | N/A (a measured signal, not a system) | Frozen encoder state -> distance-to-centroid | N/A | Real but precision-limited (**experimentally demonstrated**, d=0.894, AUC=0.723, ~34% precision in the combined-classifier's own re-scoring) | N/A | No | Frame-level, inherited from encoder | Synthetic (LibriStutter) | **Partially** — real detection signal, not yet precise enough alone |
| This track's own acoustic candidates (direction (g)) | N/A | Waveform -> candidate span + similarity score | N/A | Real recall, weak precision (**experimentally demonstrated**) | N/A | N/A (sound_repetition only) | Yes, waveform-native | Synthetic (LibriStutter) | **Partially** — same shape of result as the encoder signal |

**The single most important re-reading this table produces**: **Kordt
et al.'s explicit-marker-token approach and Huang et al.'s joint-
detection approach are, read against this project's *actual*
requirement (detection, not fragment-content preservation), closer to
"solving our problem" than this track's earlier L4 framing gave them
credit for.** A `REP` token or a strong SED signal is exactly the kind
of detection event this project's own downstream pipeline needs — the
earlier framing marked these "No, solves our exact problem" only
because it was scoring them against verbatim fragment-content
preservation, which — per the objective restated at the top of this
document — was never actually the requirement.

## 4. What is the best existing solution we could use — not "what's most novel"

Compared directly on one criterion: **which architecture gives the
application the information it needs most reliably, with the resources
this project actually has?**

- **A. Current CrisperWhisper pipeline (as shipped)**: strong on
  localization/classification/clean-content once a candidate exists;
  real, measured, ~45%/~40% detection gap for `sound_repetition`/
  `word_repetition` specifically. Zero additional cost — already built.
- **B. Existing stutter-aware ASRs (StammerTalk/AS-70-style fine-tune)**:
  **the strongest single lever identified in this entire two-session
  review**, reread against the actual objective. A real, published,
  replicable recipe (LoRA fine-tune on verbatim/literal targets) that
  **directly and measurably fixes the deletion problem** Stage A
  identified — not by preserving exact fragment phonetics, but by
  training the decoder to stop smoothing repeated content away at all,
  which is precisely what this project's own text-based candidate
  matcher needs to fire. Cost: real (GPU + English-language data, per
  the Stage D requirements already scoped above), but now justified by
  a *directly on-target* objective (stop deleting) rather than a harder,
  not-strictly-necessary one (spell out fragments phonetically).
- **C. ASR + acoustic detector hybrid (this project's own architecture,
  and this track's own Stage B/C, direction (g), and combined-signal
  classifier)**: real, partial signal, insufficient precision alone at
  the specific features tried — **but this project's own already-
  shipped precedent (`ROADMAP.md` item 17's corroboration classifier)
  used the *full* raw encoder embedding, not just a distance scalar,
  and reached real, shippable performance for a related task** (see
  Part 8 below — this is a genuine, cheap, unexplored variant of C).
- **D. Explicit-disfluency-token approaches (Kordt et al.)**: real,
  demonstrated detection value (a marker token *is* a detection
  signal), real catastrophic-forgetting cost, not stuttering-specific
  data.
- **E. Joint ASR+SED (Huang et al.)**: the single strongest published
  result found for this shape of problem (CER -37.71% relative, SED F1
  +46.58% relative, jointly) — on real stuttering data, close to this
  project's own architecture in spirit (ASR + a corroborating signal),
  never tested in English or against this project's own taxonomy.
- **F. Audio-native learned detectors (YOLO-Stutter/SSDM)**: real,
  published, more capable than this track's own hand-engineered
  acoustic attempt (direction (g)) — a genuine, cheap-relative-to-
  Stage-D "try a better model, not just better hand-picked features"
  option, never tested against this project's own data.

## 5. What did our experiments actually discover — separated precisely

- **Findings about Whisper/CrisperWhisper**: normalization is real and
  large (Stage A, **experimentally demonstrated**, n=186); it is a
  Whisper-architecture-general property, not CrisperWhisper-specific
  (Arm 2, **experimentally demonstrated**); a bigger, non-fine-tuned
  model doesn't fix it (Arm 1, **experimentally demonstrated**).
- **Findings about information loss**: roughly half of `sound_
  repetition`/`word_repetition` losses happen at otherwise-correctly-
  transcribed positions (Stage A) — this is a **detection** loss
  specifically, not (per the reframing above) necessarily evidence that
  fragment *content* is unrecoverable or needed.
- **Findings about acoustic evidence**: `sound_repetition` has a real,
  ASR-independent acoustic co-occurrence with short voiced-burst runs
  (direction (g), recall=0.824, **experimentally demonstrated**,
  replicated across two feature implementations) — directly useful for
  the **detection** requirement, regardless of the fragment-content
  question.
- **Findings about representation**: CrisperWhisper's frozen last-layer
  encoder carries real, duration-independent detection signal (Stage
  B/C, d=0.894, AUC=0.723, **experimentally demonstrated**) — again,
  directly a **detection** signal, not a claim about fragment-content
  recoverability (which this track never actually tested — no
  experiment attempted to *decode* fragment content from the encoder,
  only to *detect* anomalous positions).
- **Findings about detection specifically (the actual requirement)**:
  the best single detection number this track produced is the encoder-
  distance signal's own precision (~34%, combined-classifier arm B);
  the best combined attempt (5 hand-picked features, simple logistic
  regression) reached P=0.314, not meaningfully better.
- **Findings that are merely limitations of the particular experiments,
  not the underlying question**: the combined classifier used only 5
  scalar features, not the full raw embedding this project's own
  shipped classifier (item 17) uses for a related task — this is a
  **limitation of that specific experiment's design**, not evidence
  that encoder-based detection has a hard ceiling. Similarly, direction
  (g)'s hand-engineered acoustic features are a limitation of *that*
  specific attempt, not evidence that *no* acoustic detector could do
  better (YOLO-Stutter/SSDM's own published results are real,
  independent evidence pointing the other way).

## 6. What is still actually unsolved for our application

The precise gap, stated once, directly: **reliable *detection* of
`sound_repetition`/`word_repetition` at positions where CrisperWhisper's
decoded text currently gives no signal at all.** Not an ASR problem in
the "need a new ASR" sense — **a detection/candidate-generation
problem**, for which this track has already tried and found real-but-
insufficient signal via two different mechanisms (frozen-encoder
distance; hand-engineered acoustic features), and has **not yet
tried** two real, concrete, evidence-backed variants: (1) a richer
classifier over the full encoder embedding (this project's own item-17
precedent, not yet applied to this harder candidate-generation task),
and (2) a learned (not hand-engineered) acoustic detector matching
YOLO-Stutter/SSDM's published architecture class. Both are cheaper than
Stage D and directly targeted at the actual, narrow gap identified
above — not previously proposed with this framing because the prior
passes were scoring options against fragment-content preservation, not
detection.

**Is the gap an ASR problem, acoustic detection problem, alignment
problem, output-representation problem, a combination, or already
solved?** Per this reframing: **primarily a detection-classifier
problem** (can a classifier, given richer inputs than this track has
tried, separate true positives from the class-imbalanced background at
usable precision), **secondarily and optionally an ASR-decoder problem**
(would a StammerTalk-style verbatim fine-tune, which directly targets
*not deleting* content rather than spelling out fragments, close the
gap more directly by making text-based candidates fire in the first
place) — **not** an alignment problem (localization already works when
a candidate exists) and **not** an output-representation problem (the
clean/intended content is usually already correct).

## 7. Re-evaluating Stage D against the actual objective

**Is Stage D necessary?** Not immediately, and not in the form the
L1-L4 novelty analysis scoped it. **Two cheaper, more directly-targeted
options (Part 6 above) have not yet been tried and are evidence-backed,
not speculative.** Forcing the answer toward Stage D before trying them
would repeat exactly the mistake the project owner's own message
warned against — reaching for the more novel/interesting-sounding
option instead of the one the evidence actually points to next.

**If Stage D does become necessary** (i.e., if the cheaper options in
Part 8 below are tried and still leave a real gap), **it should be
re-scoped from what the L1-L4 analysis proposed**: not a fine-tune
targeting true L4 fragment-content preservation (harder, unproven, and
per this session's own script-structural finding, possibly untestable
in the one language with adequate real data), but an **L2-level
fine-tune** — adapt CrisperWhisper (or stock `whisper-large-v3`) on
real English disfluent speech (most plausibly FluencyBank Timestamped,
pending its own access/parser work) with the direct, narrower objective
of **not deleting repeated content**, following StammerTalk's own
directly-demonstrated, real recipe (LoRA fine-tuning on literal
transcriptions). This is better-precedented (a real, published,
quantified success on a structurally analogous problem), cheaper to
specify (no need to solve fragment-phonetic-spelling), and more
directly aligned with what the application actually needs (a text-based
candidate that fires, not a phonetically precise transcript).

**Why this is genuinely why, not just asserted**: StammerTalk's own
result (deletion-error rate 26.56%->2.29% for severe stuttering) is
direct, quantified, real-speech evidence that this exact mechanism —
fine-tuning toward a verbatim target stops the decoder from deleting
disfluent content — works. This project's own text-based candidate
matcher needs exactly that: *something* present at the disfluent
position, not a phonetically exact transcription of it.

**What would be built, what it needs, why the current machine can't run
it, minimum viable experiment, and what would justify investing further**
— unchanged from the Stage D Requirements/Re-entry Gate section earlier
in this document, with one adjustment: the training objective is
"reduce the deletion/normalized-away rate," evaluated with this
project's own already-built Stage-A-style 4-category classification
(re-applied to the fine-tuned model's output), not a fragment-content
accuracy metric that doesn't yet have a defined ground truth. The
GPU/data/engineering requirements already scoped (a single cloud GPU
instance for a reduced-scale pilot; FluencyBank Timestamped or
equivalent real English data; a training pipeline this project doesn't
yet have) stand unchanged.

## 8. Ranked implementation roadmap — shortest path to a working system

**Rank 1 — cheapest, no new data, no GPU, directly reuses proven
internal technology (proposed, not yet run).** Retrain the encoder-
based classifier for the `sound_repetition`/`word_repetition`
candidate-**generation** problem (not the corroboration problem item 17
already solves) using the **full raw last-layer embedding** as the
feature space — reusing `compare_corroboration_mechanisms.py`'s exact,
already-shipped (S1, M3) methodology (nested-CV logistic regression,
clip-split folds) unmodified, the same infrastructure that produced
this project's one other trained, shipped model. This directly tests
whether the combined-signal classifier's negative result (Part 5 above)
was a feature-poverty problem — 5 hand-picked scalars — rather than a
ceiling on what the encoder contains. **Why this ranks first**: zero
new infrastructure, zero new data, zero GPU, and a direct, concrete,
evidence-motivated hypothesis (item 17's own shipped success on a
related task used exactly this richer feature space and worked).

**Rank 2 — cheap, no GPU required for inference/evaluation, tests a
better-architected acoustic detector.** Evaluate whether a learned (not
hand-engineered) acoustic-native detector, matching YOLO-Stutter/SSDM's
general architecture class (a small trained region/boundary detector
over spectral features, not a hand-set RMS/ZCR or MFCC threshold),
improves on direction (g)'s own recall/precision — a real, published,
more-capable-in-general class of method this project has not yet tried
against its own data. Training a small detector of this class is
plausibly CPU-feasible at this project's own data scale (LibriStutter's
120-clip working sample), though this should be confirmed with a real
dry-run cost check before committing, per this track's own standing
discipline — not assumed cheap without checking.

**Rank 3 — if Ranks 1-2 leave a real, honestly-reported gap: cost out
and pursue the re-scoped (L2-level, not L4-level) Stage D fine-tune**,
per Part 7 above — real GPU, real English-language data acquisition
(FluencyBank Timestamped or equivalent), a training pipeline this
project doesn't yet have.

**What to use from existing research, not rebuild**: this project's own
`compare_corroboration_mechanisms.py` infrastructure (Rank 1); the
general architecture pattern from YOLO-Stutter/SSDM as a design
reference, not a required reimplementation (Rank 2); StammerTalk/AS-70's
own real, published LoRA fine-tuning recipe as the starting point if
Rank 3 is reached, rather than designing a training pipeline from
scratch.

**What requires new data**: only Rank 3 (English-language real
disfluent speech). Ranks 1-2 reuse this project's own existing
LibriStutter working sample and cached artifacts.

**What requires GPU/compute**: only Rank 3, and only for training — a
learned detector at Rank 2's scale is plausibly trainable on CPU at
this project's own current data scale, to be confirmed with a real
timing check before assuming so.

**What can be done entirely on the current setup, today**: Ranks 1 and
2 in full — no new infrastructure, no new data acquisition, no GPU.

## 9. Research novelty — kept explicitly secondary, reported accurately either way

Per the project owner's own instruction: do not stretch the evidence to
create a novelty claim, and do not discard a real one either.

- **Already known, not novel**: ASR normalizes disfluent speech
  generally ([B13]); explicit disfluency-token fine-tuning works for
  coarse categories ([B11]); verbatim fine-tuning reduces deletion
  errors ([B20][B27], directly re-verified this session); joint
  ASR+detection training improves both tasks ([B21]).
- **A real, narrow, accurately-scoped novel finding, if this track
  reports it**: direct, controlled, multi-angle evidence (representation
  swapping, model-family swapping, decoding-parameter variation,
  acoustic-signal combination) that no cheap, off-the-shelf intervention
  recovers fragment-level `sound_repetition` *detection* signal at
  usable precision for this specific taxonomy, at this sample size —
  narrower and more defensible than "nobody has solved disfluency-aware
  ASR," and **not** dependent on the L4 fragment-content question at
  all, since detection (not content preservation) is what this track's
  own experiments actually measured throughout.
- **This project's actual contribution, if Rank 1/2 above succeed**: an
  engineering combination — reusing a proven internal technique
  (item 17's full-embedding classifier methodology) on a new, harder
  task (candidate generation, not corroboration) — not a research
  novelty claim in the academic sense, and does not need to be framed
  as one to be valuable to this project.
- **No novelty claim is asserted for the roadmap in Part 8** — it is an
  engineering plan, correctly labeled as such.

## 10. The decision

**Given the actual application objective and everything discovered so
far, the best next technical direction is: extend this project's own
already-shipped, already-proven encoder-based classifier methodology
(item 17's full-embedding logistic regression) to the `sound_
repetition`/`word_repetition` candidate-*generation* problem, evaluated
against direction (g)'s and Stage C's own already-established
population — because this is the cheapest, most directly evidence-
motivated untried option, requires no new data or GPU, reuses proven
internal technology, and directly targets this project's own precisely-
identified real gap (detection precision, not fragment-content
preservation, not a new ASR).**

This is **not** an existing off-the-shelf ASR (Arm 1/2 already ruled
that out); **not** the current pipeline alone (it has the measured
detection gap this whole track exists to close); **not** an existing
stutter-aware model as-is (none is in English, at this project's
taxonomy granularity, for the detection-not-content task); **not** a
modified/fine-tuned ASR (Rank 3, held in reserve, re-scoped and
justified, but not yet warranted before cheaper options are tried); and
**not** Stage D in the form the L1-L4 analysis proposed. If Rank 1/2
are tried and both leave a real, honestly-reported gap, Stage D — re-
scoped to L2 (stop deleting), not L4 (spell out fragments) — becomes
the evidence-justified next step, exactly as scoped in Part 7 above.

---

## Decision for Next Session

**[SUPERSEDED, same day, 2026-08-07 EOD — preserved for the historical
record, not deleted.]** Everything below in this subsection was written
immediately after the Application-Objective Decision Analysis, before
Rank 1, Rank 2, and Rank 3 were actually run. All three have since been
pre-registered, implemented, run, and decided (see "Rank 1 results,"
"Rank 2 results," "Rank 3 results," and "Final synthesis: Rank 1 + Rank
2 + Rank 3, and the Stage D decision," all below). The concrete "start
with Rank 1" instruction below is now complete, not the current plan —
**the authoritative, current next-session plan is in "Current Project
State — 2026-08-07 EOD" at the very end of this document.** This
subsection is kept exactly as originally written so the reasoning that
led to Rank 1 remains part of the record.

**Do not open with more literature research or novelty analysis.** Open
with implementation, per the ranked roadmap above:

1. **Read this "Application-Objective Decision Analysis" section first**
   — it is the current, governing interpretation of this entire
   document, and reframes what "the gap" actually is (detection
   precision, not fragment-content preservation).
2. **Pre-register, then implement, Rank 1**: a full-raw-embedding
   logistic-regression classifier for `sound_repetition`/`word_
   repetition` candidate generation, reusing `compare_corroboration_
   mechanisms.py`'s exact existing methodology, evaluated against
   Stage B/C's own population (the real, ASR-text-normalized-away
   candidate-generation gap), not direction (g)'s acoustic-candidate
   population — corrected here from this line's earlier, imprecise
   wording. No new data, no GPU, no new infrastructure.
3. **If Rank 1 shows real improvement**: scope how to integrate it into
   the live detection pipeline as a genuine engineering task — a
   separate, deliberate decision, not automatic.
4. **If Rank 1 does not show real improvement**: move to Rank 2 (a
   learned, not hand-engineered, acoustic detector), with its own
   pre-registered dry-run cost check first, per standing practice.
5. **Only if both Rank 1 and Rank 2 leave a real, honestly-reported
   gap**: begin costing out the re-scoped (L2-level) Stage D — real
   GPU, real English-language data acquisition — as the evidence-
   justified next real step, not before.

**Standing reminder for whoever opens this document next, human or
Claude**: the objective is a working application that extracts speech
disfluencies reliably enough to act on — not a research paper, not a
novel finding, not a new ASR. Everything in this document, including
its own most interesting-sounding findings, is a means to that end.

---

# Rank 1: full-embedding candidate-generation classifier — pre-registered protocol (2026-08-07)

Per the project owner's explicit instruction to proceed with Rank 1, and
to plan it precisely before running anything. This section is written and
committed to the record *before* the implementation below it is run —
this project's own standing rule 1.

## 1. What information Rank 1 consumes

CrisperWhisper encoder last-layer hidden states (the same primitive
`profiling/encoder_embedding.py` already exposes — `extract_last_layer_
states()` + `pool_span()`), pooled at the span of every ASR hypothesis
token that Track B's alignment marks `correct` (i.e., every position
where the real ASR transcript and the ground-truth reference agree on
which word is there — both disfluent-but-detected and clean words are
eligible, exactly as Stage B/C already scoped it). Per position, this is
a single ~1280-dimensional vector (CrisperWhisper large-v3's encoder
hidden size) — the **full** raw embedding, not the single scalar
(distance-to-centroid) Stage B/C reduced it to.

No new ASR inference is required: this reuses the already-cached Track B
hypothesis tokens (`eval_datasets/_track_b_cache`, populated across
earlier sessions) and the already-downloaded real LibriStutter audio
(`eval_datasets/libristutter_sample[_audio]`, 499 clips). One new encoder
pass is required, over the same small clip subset Stage B already
identified as relevant (see §4) — this is the only new compute cost, and
it is the same cost profile already measured directly (~33s/clip,
Stage B's own log).

## 2. What it is trying to recover

Whether **any position** in the real ASR transcript — not just positions
where the transcript already contains a repetition-shaped surface
pattern — is a `sound_repetition` or `word_repetition` site that
CrisperWhisper's own decoded text gives **zero** surface evidence for
(Stage A's "category 1": nothing predicted there at all, gate off). This
is a **detection/candidate-generation** task: flag a position as
disfluent or not, from encoder representation alone, independent of
whatever the decoded text says. It is explicitly *not* attempting to
recover the disfluent fragment's literal content (L4) — per the
Application-Objective Decision Analysis above, that is not believed
necessary for this project's actual product objective.

## 3. Why this is different from the experiments already run and failed

- **Different from Stage B/C themselves**: Stage B/C tested exactly one
  hand-picked scalar reduction of the embedding (cosine distance to the
  clip's fluent centroid) with a single-threshold decision rule (M1).
  They never asked whether the *full* embedding, given to a classifier
  that can weight all ~1280 dimensions independently (M3), separates the
  two classes better than that one particular projection does. Standard
  dimensionality-reduction theory says a good learned projection can only
  match or beat a fixed hand-picked one on the same information — Rank 1
  tests whether it actually does, on this population, not whether it
  could in principle.
- **Different from the combined-signal classifier** (`stage_combined_
  classifier.py`, Failure, F1=0.242): that experiment combined 5 hand-
  picked *scalar* features (MFCC similarity, burst count, duration, and
  the same single encoder-distance scalar Stage B/C used) — it never gave
  the classifier the raw embedding either. It also ran over direction
  (g)'s acoustic-candidate population (positions an acoustic heuristic
  already flagged as repetition-shaped), not Stage B/C's true candidate-
  generation-gap population (positions where nothing at all was flagged).
  These are different populations answering different questions, and
  conflating them was this project's own imprecise wording in last
  session's "Decision for Next Session," corrected above.
- **Different from item 17's own shipped classifier**: item 17's
  classifier already uses the full raw embedding + M3 — but only ever
  sees positions the app's own candidate generator already flagged as
  `word_repetition`/`sound_repetition` (a **corroboration** task: is this
  already-proposed candidate real or a false positive?). Rank 1 asks the
  logically prior question — should a candidate have been proposed here
  at all? — over positions where today's candidate generator proposes
  nothing. This is the exact gap item 20/Stage A's finding named (45%/
  40% of real repetitions leave no surface trace for any text-based
  candidate generator to find), and item 17's classifier structurally
  cannot help with it, since it never sees these positions.

## 4. Population and ground truth

Identical to Stage B/C's own pre-registered and already-run population,
reused unmodified via `stage_b_representation_probe.py`'s
`_identify_positions()` — chosen deliberately over defining a new
population, to keep this a strict, single-variable methodological
comparison (classifier arm only) against already-published numbers,
rather than introducing a second simultaneous change:

- **Targets (label=1)**: real-ASR positions where the ground-truth
  reference token is `sound_repetition` or `word_repetition`, the
  alignment marks the position `correct` (i.e. this specific position
  really is in the real, real-condition, real-audio ASR hypothesis, not
  a deletion), and the live detector — classifier gate explicitly forced
  off — predicts **nothing** there (Stage A's category 1). Real counts,
  already measured (`eval_results/20260805T211000_stage_b_representation_
  probe.json`): **19** `sound_repetition` targets, **17** `word_
  repetition` targets, **36** combined, across **31** of the 499 real
  LibriStutter clips (the only 31 that contain at least one such
  position at all).
- **Controls (label=0)**: every other `correct`-aligned, non-disfluent
  position in those same 31 clips. Real count: **966**.
- **Honest, load-bearing limitation, stated up front, not discovered
  after the fact**: this is the entire available population for this
  exact task on this dataset — Stage B/C already scanned all 499 real
  clips to find it. 36 positives (19/17 split by type) is a genuinely
  small sample. Under 5-fold clip-split CV this is roughly 7 positive
  test examples per fold combined, 3-4 per fold for either type alone.
  Per this project's own standing rule 3, any result from this
  experiment — positive or negative — gets reported with this sample
  size stated plainly alongside it, not as an unqualified headline
  number. This ceiling cannot be raised without acquiring more labeled
  real (non-synthetic) disfluent speech data, which is explicitly out of
  Rank 1's scope (that is what would make it Rank 3/Stage D, not Rank 1).
- **Provenance note, fixed before running**: `_identify_positions()` is
  re-run fresh against whatever is currently on disk in `eval_datasets/
  _track_b_cache` (a cache that has grown/changed across sessions), not
  against a frozen snapshot of Stage B's own inputs. If the re-derived
  counts differ from the 19/17/36/966 figures above, that will be
  reported explicitly (cause: cache-population difference, a known,
  legitimate reason, not a silent discrepancy) before any result is
  trusted — the same discipline `stage_c_duration_baseline.py` already
  applied when it re-derived Stage B's population and asserted the counts
  matched (they did, that day).

## 5. Baseline

Stage C's own (S1, M1) mechanism — cosine distance to the fluent
centroid, threshold selected by F1 on the training data — but
**recomputed under identical 5-fold clip-split cross-validation** to the
new (S1-full, M3) arm below, using `compare_corroboration_mechanisms.
py`'s own `_cv_threshold()` unmodified. This is a stricter baseline than
Stage C's originally-reported AUC=0.723 (sound_repetition) /
precision-at-recall figures, which were computed over the whole
population at once, not cross-validated — the CV baseline will very
likely score lower than 0.723 AUC's implied ceiling, and that is
expected and correct, not a bug: it is the fair, generalization-honest
number to compare Rank 1's own cross-validated result against.

## 6. Metrics and success/failure criteria

Primary population: **Any** (`word_repetition` + `sound_repetition`
combined, n_pos=36) — the least-small-sample cut available. Per-type
(`sound_repetition` alone, n_pos=19; `word_repetition` alone, n_pos=17)
reported alongside, explicitly labeled exploratory/too-small-to-trust-
standalone per rule 3, not used alone to decide the gate below.

For each population: 5-fold clip-split CV, nested-L2 logistic regression
over the full standardized embedding, **train-fold-optimal threshold**
(reusing `stage_combined_classifier.py`'s already-fixed
`_cv_classifier_optimal_threshold()` convention from the start — not
`compare_corroboration_mechanisms.py`'s hardcoded `proba>=0.5`, which
this project already found silently fails on populations this imbalanced,
first-hand, this same track, two sessions ago). Report mean
precision/recall/F1 across folds, same format as every other stage in
this track.

**Decision gate, fixed before running (rule 1):**
- **Success → integrate/validate toward the app** (§9): (S1-full, M3) on
  the **Any** population beats the CV baseline by **≥20% relative F1**
  (the same bar `stage_combined_classifier.py` used, kept consistent
  across this track's classifier experiments) **AND** clears an absolute
  usability floor of **precision ≥ 0.15 at recall ≥ 0.3** on that same
  population — a technically-passing-but-still-unusable result (e.g. 3%
  precision) does not count as Success even if it beats the baseline
  relatively, because it would not be deployable.
- **Failure, proceed to Rank 2**: does not clear both bars above.
- **Inconclusive** (explicitly a third, honest outcome, not forced into
  Success or Failure), operationalized concretely before running: compare
  per-fold F1 (classifier vs. baseline) across the 5 outer folds; if the
  classifier does not win (strictly higher F1) on at least 4 of 5 folds
  AND does not lose on at least 4 of 5 folds — i.e. the per-fold direction
  is mixed, neither side winning a clear majority — the result is
  Inconclusive regardless of what the mean-F1 comparison alone would
  suggest. Given n_pos=36, this is a real possible outcome and must be
  reported as such, not smoothed into a false Success or Failure.

## 7. What would justify Rank 2 vs. Stage D

- **Rank 2** (learned acoustic detector, YOLO-Stutter/SSDM architecture
  class): justified by Failure or Inconclusive here — this result alone
  does not touch acoustic-only signal at all (it is encoder-representation-
  only, matching Stage B/C's own scope), so a Failure here says nothing
  about whether a purpose-built acoustic detector would do better, only
  that this specific representation-side lever (full embedding vs. one
  scalar) does not close the gap enough to ship.
- **Stage D** (re-scoped to L2, per the Decision Analysis above): only
  justified once **both** Rank 1 and Rank 2 are tried and Failure/
  Inconclusive on both — not by this result alone, per the pre-registered
  roadmap order.

## 8. Downstream integration path, if Success

The live app already runs a second CrisperWhisper encoder-only pass at
inference time for item 17's corroboration classifier (`ROADMAP.md` item
18 tracks removing this redundancy as a separate, not-yet-done
optimization). A shipped Rank 1 classifier would extend that existing
pass from "encoder features at already-flagged candidate positions only"
to "encoder features at every correctly-aligned transcript position" —
a real, already-named, already-accepted-as-open cost (`ROADMAP.md` item
18/20: added live-app latency), not a new unknown. Integration would mean:
run the classifier over every ASR token span during inference, emit a new
candidate wherever it fires, then route that candidate through the
existing type/confidence/downstream-event pipeline exactly as any other
candidate is today. This is a scoped engineering task if and when Success
is reached — not attempted in this pass.

## 9. Novelty (kept explicitly secondary, per the governing objective)

Not a novelty claim. This is this project's own already-proven classifier
methodology (item 17), applied to a different, already-precisely-
identified population within the same project. Whether it succeeds or
fails, the finding is an internal engineering result about this specific
pipeline, not a claim about the broader field.

## Rank 1 results (2026-08-07)

Implemented as `profiling/evaluation/stage_h_candidate_generation_
classifier.py`, self-tested (8/8 checks pass — decision-gate logic on
hand-constructed win/loss/floor cases, row-packing round-trip), then run
for real against the pre-registered protocol above, no changes to the
protocol after seeing results.

**Provenance note triggered, as anticipated in §4**: the Track B cache
has grown since Stage B/C's session (189 clips cached now vs. whatever
was cached then) — re-derivation found **58** clips with at least one
category-1 target position (vs. Stage B's 31), **37** `sound_repetition`
targets (vs. 19), **29** `word_repetition` targets (vs. 17), **1780**
control positions (vs. 966). The ~1.87x scale-up is consistent across
every count (58/31 = 1.87, matching 189/roughly-101-clips-then), which is
the expected signature of more of the same corpus having been ASR'd and
cached since, not a bug — reported per the pre-registered handling, not
silently accepted or treated as alarming.

**Real result, primary population (Any, n_pos=66, n_neg=1780):**

| Arm | F1 | Precision | Recall |
|---|---|---|---|
| Baseline (S1, M1): distance-to-centroid + CV threshold | 0.097 | 0.055 | 0.462 |
| Rank 1 (S1-full, M3): full embedding + nested-CV logistic regression | **0.230** | **0.580** | 0.147 |

Per-fold win/loss: 4W/1L (stable direction, gate proceeds past the
Inconclusive check). Relative bar (F1 > 0.097×1.2 = 0.116): **cleared**
(0.230). Absolute floor (precision ≥0.15 **and** recall ≥0.3): **not
cleared** — precision clears it by a wide margin (0.580, ~10x the
baseline's 0.055) but recall (0.147) falls well short of the 0.3 floor.

**Verdict per the pre-registered gate: FAILURE** — on the absolute-floor
half of the criterion specifically, not the relative-improvement half.

**Per-type cuts (both explicitly too-small-to-trust standalone, per §6,
reported for completeness, not used to override the primary verdict):**
`sound_repetition` alone (n_pos=37): baseline F1=0.068, classifier
F1=0.120, 2W/1L of 5 folds — **Inconclusive** (neither side reaches the
pre-registered 4-fold stability threshold). `word_repetition` alone
(n_pos=29): baseline F1=0.040, classifier F1=0.000 (every fold), 0W/3L —
also **Inconclusive** by the same rule, despite looking like a clear loss
on the raw numbers; the stability rule is deliberately conservative here
and that is working as designed, not a loophole.

**What this actually shows, stated precisely (rule 3 — audit before
trusting, small samples get a caveat, not a headline):**

This is a genuine, informative negative result, not a null one. The full
embedding *does* contain more separable signal than the single hand-
picked scalar projection — precision improved roughly 10x and mean F1
more than doubled, and that gain survived cross-validation (4 of 5
folds agree on the direction). That confirms §3's underlying hypothesis
(a learned projection can do more with the same information than one
fixed scalar) was correct, as far as it goes. But at n_pos=66 (and far
smaller per-type), fitting a ~1280-dimensional decision boundary produces
a classifier that is highly conservative — precise when it fires, but it
fires on only ~15% of real instances. This experiment **cannot
distinguish** between two different explanations for that low recall:
(a) the encoder's own representation genuinely doesn't separate most of
these positions well enough, or (b) 66 positive examples is too few to
fit a stable, well-calibrated decision boundary in this dimensionality,
independent of whether the representation itself is separable. Both are
live possibilities; this experiment does not adjudicate between them, and
claiming it does would be exactly the kind of overclaim rule 3 and the
governing objective both warn against.

**Per the pre-registered roadmap (§7): this result is Failure on the
primary population → Rank 2 is next**, not Stage D. Stage D remains a
reserve option per the project owner's own explicit instruction, not the
automatic destination — and this result does not by itself justify it,
since Rank 1's Failure is specifically about representation/sample-size,
not about acoustic signal, which Rank 2 tests independently.

---

# Rank 2: learned acoustic candidate-generation classifier — pre-registered protocol (2026-08-07)

Per the project owner's explicit go-ahead to run the full cycle (plan,
implement, test, run, document, decide) in one pass. This section is
written and committed to the record before the implementation below it
exists, per this project's own standing rule 1.

## Hypothesis

Direction (g) and its MFCC escalation (`stage_g_acoustic_sound_
repetition.py`) already tested whether `sound_repetition` has a
recoverable acoustic signature — they found real signal (recall 0.69 at
best) but a **hand-designed** similarity metric (cosine similarity
between two bursts' *mean-pooled, single* MFCC vectors, collapsed to one
scalar) could not turn that signal into useful precision (best F1=0.170,
uncross-validated; F1=0.147 under proper CV). The combined-signal
classifier then combined that one scalar with 3 other hand-picked scalars
and a *separate, ASR-encoder-derived* signal — still Failure (F1=0.242,
short of beating the encoder-distance arm alone). Neither experiment ever
let a model **learn** how to weight the individual MFCC coefficients or
how much inter-burst variation is tolerable — both hard-coded that via a
single fixed cosine-similarity formula.

**Rank 2's hypothesis**: a classifier that learns, from the full
per-coefficient MFCC statistics of the bursts a candidate is built from
(not one hand-collapsed similarity number), which spectral properties
distinguish a real repeated fragment from an incidental run of short
voiced bursts, will separate the two classes better than the fixed
cosine-similarity heuristic does — evaluated purely on **acoustic**
signal, with **no ASR and no encoder involved at all** (unlike Rank 1,
which was encoder-representation-only). This is the genuinely distinct
question direction (g) never tested: not "is there acoustic signal"
(already found: yes) and not "does combining acoustic + encoder help"
(already tested: no) — but "does a *learned* combination of the *raw*
acoustic features beat a *hand-designed* one," acoustic-only.

## Acoustic inputs / features

Reuses `stage_g_acoustic_sound_repetition.py`'s exact, unmodified
primitives: `segment_voiced()` (voiced/silent segmentation),
`compute_mfcc()` (the hand-rolled, already-validated 13-coefficient MFCC
extractor), and the exact same candidate-*span* detection logic inside
`generate_candidates()` (a maximal run of ≥2 consecutive short voiced
bursts, short-gap tolerant) — candidate spans depend only on the voiced/
silent segmentation and duration/gap thresholds, not on which similarity
feature is computed from them, so this experiment's candidate population
is **identical** to direction (g)'s own (same spans, same audio, same
thresholds), only the per-candidate feature vector changes.

For each candidate (a run of `n_bursts` ≥ 2 short bursts), instead of one
hand-designed cosine-similarity scalar between consecutive bursts, extract:
- **Mean MFCC vector across the run's bursts** (12 dims, coefficient 0
  excluded — same established exclusion as direction (g)'s own MFCC
  escalation, for the same measured reason: coefficient 0 is overall
  energy and swamps genuine spectral-shape information).
- **Per-coefficient standard deviation across the run's bursts** (12
  dims) — how much the bursts disagree with each other, coefficient by
  coefficient. This generalizes direction (g)'s pairwise-cosine-
  similarity idea (low inter-burst variation ≈ similar ≈ repetition-like)
  to any `n_bursts` ≥ 2 and, critically, lets the classifier weight each
  coefficient's variability independently instead of forcing all 12 into
  one fixed-geometry similarity formula.
- `n_bursts`, `duration` (2 scalars — the same weak context features
  already used in the combined-signal classifier, included for
  consistency, not expected to carry the result alone).

26 dimensions total. Pure numpy, no new dependency (the MFCC extractor
and `scipy.fftpack.dct` it uses are already in this project's
environment). A new function, `_generate_candidates_with_bursts()`, is
added to Rank 2's own script (not to `stage_g_acoustic_sound_
repetition.py`) — a deliberate, small duplication of that module's
run-detection loop, because `AcousticCandidate` itself does not expose
the individual burst segments its aggregate `similarities` were computed
from, and modifying an already-validated, already-shipped module's return
type to serve one new caller is not warranted here.

## Target population

`sound_repetition` only — matching direction (g)'s own established scope
exactly (the acoustic burst mechanism is specific to sub-word fragment
repetition; `word_repetition`'s acoustic signature, if any, is a
different question this project has not scoped). Ground truth comes
directly from LibriStutter's own labels (Track-A-style, no ASR involved
anywhere in this experiment, matching direction (g)'s own methodology).

**Scaled up from direction (g)'s own n=120 clips to all 499 real
LibriStutter clips with usable audio** — a deliberate, real difference
from every acoustic-population experiment so far in this track. This is
possible specifically *because* Rank 2 needs no CrisperWhisper encoder
pass (unlike Rank 1, Stage B/C, or the combined-signal classifier's Arm
B/C) — the entire per-clip cost is voicing segmentation + a hand-rolled
MFCC pass, already measured directly at 97s/120 clips (`eval_datasets/
_stage_g_mfcc_run_output.txt`), giving an expected total cost of
**~400s (under 7 minutes)** for all 499 clips — cited from an
already-logged run of the identical computation, not a fresh guess.
Scaling up directly addresses Rank 1's own most-flagged limitation (a
sample too small to distinguish "no signal" from "not enough data") —
if the same ~51-target-per-120-clip rate holds, this gives roughly
**~210 `sound_repetition` targets**, a meaningfully larger and more
trustworthy sample than Rank 1's 66. Real counts are reported as
measured, not assumed.

## Model / architecture

The exact same nested-CV L2-regularized logistic-regression mechanism
this track has used for every classifier so far (item 17,
`stage_combined_classifier.py`, Rank 1) — `compare_corroboration_
mechanisms.py`'s `_fit_logistic_regression`/`_select_l2_by_nested_cv`/
`_standardize`, plus `stage_combined_classifier.py`'s already-fixed
train-fold-optimal-threshold convention (`_cv_classifier_optimal_
threshold`), all unmodified. This is a **deliberate, honest scoping-
down** from YOLO-Stutter/SSDM's own architecture class (deep CNN/
transformer models trained end-to-end on raw audio, typically on GPU) —
this project has no GPU, and a full replication is out of scope for a
bounded, single-session experiment. What Rank 2 preserves from that
architecture class is the *spirit* (a model that **learns** the
detection function from richer acoustic features, rather than a fixed
hand-designed formula) at a cost this project can actually afford. This
scoping-down is stated plainly here so the result is never described as
"we tried a YOLO-Stutter-class detector" — it is a CPU-feasible,
much smaller approximation of that idea, and is reported as such.

## Baseline

Direction (g)'s own mechanism — mean pairwise MFCC cosine similarity
(coefficient 0 excluded) + a cross-validated threshold (S1, M1) —
**recomputed fresh on this experiment's own n=499 population and folds**
(not reusing the old n=120 cached number), for the same reason Rank 1's
baseline was recomputed rather than reused: a fair comparison requires
the identical sample and fold split for both arms.

## Evaluation protocol

5-fold clip-split cross-validation (`_clip_folds`, unmodified — same
deterministic round-robin over sorted clip ids used throughout this
track). Per fold: baseline arm scored by `_cv_threshold` (train-fold-
optimal similarity threshold); classifier arm scored by `_cv_classifier_
optimal_threshold` (nested-CV L2 selection + train-fold-optimal decision
threshold). Report mean precision/recall/F1 across folds, matching this
track's established format.

## Success / failure criteria (fixed before running)

The same decision-gate structure as Rank 1's, reused unchanged for
cross-experiment consistency, not re-derived per experiment:
- **Stability check first**: per-fold win/loss (classifier F1 vs.
  baseline F1) across the 5 outer folds. If the classifier does not win
  on ≥4 of 5 folds **and** does not lose on ≥4 of 5 folds (a genuinely
  mixed direction), the result is **Inconclusive**, regardless of the
  mean-F1 comparison.
- **Success**: (only if the stability check resolves to a clear
  direction) mean F1 beats the baseline by **≥20% relative**, **AND**
  clears an absolute usability floor of **precision ≥0.15 at recall
  ≥0.3** — the identical floor Rank 1 used, kept fixed across this
  track's classifier experiments rather than tuned per experiment.
- **Failure**: stable direction, but does not clear both bars.

## What would justify Stage D vs. another low-cost direction

Per the project owner's explicit instruction that Stage D is a reserve
option, not the automatic destination, and that the chain should stop
at genuine dead ends rather than spawn open-ended new variants:
- **If Rank 2 succeeds**: validate/integrate toward the live pipeline,
  same as Rank 1's own success path would have been — not a new research
  question.
- **If Rank 2 fails or is Inconclusive**: this is the point named in the
  Application-Objective Decision Analysis's §7/§8 as "Rank 1 and Rank 2
  both tried and leave a real gap" — at that point Stage D (re-scoped to
  L2, per that section) becomes the evidence-justified next step **only
  if** no further low-cost, evidence-motivated direction is identifiable
  at that time; this document's own honest assessment of the two
  results together, not a default, decides which. This pre-registration
  does not commit in advance to "Stage D no matter what" — per the
  owner's explicit instruction to stop the experimental chain at the
  pre-registered stopping point and reassess, not keep spawning variants.

## Novelty (kept explicitly secondary, per the governing objective)

Not a novelty claim. A bounded, CPU-feasible test of whether this
project's own existing acoustic pipeline benefits from a learned rather
than hand-designed similarity function — an internal engineering
question, answered honestly either way.

## Rank 2 results (2026-08-07)

Implemented as `profiling/evaluation/stage_i_learned_acoustic_classifier.
py`, self-tested (12/12 checks pass — candidate/burst extraction, feature
construction on synthetic identical-vs-degenerate bursts, decision-gate
logic, row-packing round-trip), then run for real against the
pre-registered protocol above, no changes to the protocol after seeing
results.

**Scale-up worked as predicted**: 358s (under 6 minutes) for all 499
clips, no encoder involved — confirming the pre-registered cost estimate.
**200** ground-truth `sound_repetition` instances across 168 clips,
**3215** acoustic candidates generated, **245** positive — a substantially
larger, more trustworthy sample than either direction (g)'s original
n=120 run (51 targets) or Rank 1's n=66, exactly the intended benefit of
not needing the encoder.

**Real result (n_pos=245, n_neg=2970):**

| Arm | F1 | Precision | Recall |
|---|---|---|---|
| Baseline: mean MFCC similarity + CV threshold | 0.133 | 0.077 | 0.531 |
| Rank 2: raw per-coefficient MFCC stats + nested-CV logistic regression | 0.163 | 0.114 | 0.308 |

Per-fold win/loss: 4W/1L (stable direction). Relative bar (F1 >
0.133×1.2 = 0.160): **cleared, but only barely** (0.163 vs. 0.160 — a
margin smaller than the fold-to-fold F1 range either arm shows: baseline
0.088-0.159, classifier 0.111-0.215). Absolute floor (precision ≥0.15
**and** recall ≥0.3): recall clears it narrowly (0.308), **precision does
not** (0.114, roughly 24% short of the 0.15 floor).

**Verdict per the pre-registered gate: FAILURE** — on the absolute-floor
half again, this time on precision rather than recall, and by a
narrower margin than Rank 1's failure.

**Audit before trusting (rule 3) — this is a genuinely marginal result,
not a clean one, and is reported as such rather than rounded toward
either "basically failed" or "almost succeeded":** the classifier's mean
F1 improvement over the baseline (0.133→0.163) clears the pre-registered
+20%-relative bar by less than half a percentage point of raw F1 — a
margin smaller than ordinary fold-to-fold noise in this experiment. The
4-of-5 per-fold win record is real evidence the effect isn't pure chance,
but the *size* of the effect is small. Compared to Rank 1's result, the
character is different, not just the numbers: Rank 1's classifier traded
most of its recall for a large (~10x) precision gain; Rank 2's classifier
traded roughly half its recall (0.531→0.308) for a much smaller (~1.5x)
precision gain (0.077→0.114) — closer to what direction (g)'s own
already-published threshold sweep on the single similarity scalar alone
already showed (higher threshold -> lower recall, roughly flat
precision/F1, `stage_g`'s own table above). This experiment cannot fully
rule out that the 26-dimensional classifier is doing little more than
re-deriving a well-calibrated version of the same underlying "how similar
are these bursts" concept the one hand-designed scalar already
captures, rather than genuinely exploiting the richer per-coefficient
information it was given — an honest limitation, not glossed over.

**Per the pre-registered roadmap: this is Failure**, meaning — per this
section's own pre-registered §"What would justify Stage D vs. another
low-cost direction" — the next decision is whether Stage D is now
justified, or whether a further low-cost, evidence-motivated direction
remains identifiable. See the synthesis section immediately below, which
addresses both Rank 1 and Rank 2 together, as the project owner
requested.

---

# Rank 1 + Rank 2 synthesis and next-step decision (2026-08-07)

Both pre-registered, bounded experiments the project owner authorized are
now complete. Per their explicit instruction, this section states the
decision plainly rather than opening new variants.

**What both results say, taken together**: every representation this
track has tried at the candidate-*generation* level — the ASR encoder's
full embedding (Rank 1) and raw per-coefficient acoustic features
(Rank 2) — independently contains **real, non-random, cross-validated
signal** for `sound_repetition`/`word_repetition` positions the current
text-based pipeline misses entirely. Neither, on its own, reaches a
precision/recall combination this project judged deployable in advance.
Critically, **the two failures are shaped differently**: Rank 1 gave up
recall for a large precision gain; Rank 2 gave up recall for a small
precision gain. That difference in shape is itself evidence the two
representations are not redundant with each other — if they were
capturing the exact same information, their failure modes would look
alike.

**Does this move the project closer to a working application?** Yes, in
a specific, honest sense: two more real detection levers now exist and
are quantified (real precision/recall numbers, not vague plausibility),
where before this session there was only the single, weaker hand-picked
similarity scalar and the combined classifier that mixed weak scalars
from both domains together (F1=0.242, using neither Rank 1's full
embedding nor Rank 2's raw MFCC stats). No, in the sense that neither
result is shippable as-is — the application still cannot reliably
recover most of these missed positions today.

**Is a further low-cost, evidence-motivated direction identifiable
(per this document's own pre-registered exit condition)? Yes — one real,
specific, not-yet-tried option**: a classifier combining Rank 1's *full*
encoder embedding with Rank 2's *full* 26-dimensional raw acoustic
feature vector, together, on one population. This is meaningfully
different from the already-failed combined-signal classifier
(`stage_combined_classifier.py`, F1=0.242), which only ever combined four
*hand-picked scalars* (one MFCC-similarity number, burst count, duration,
one encoder-distance number) — never the rich representations either
Rank 1 or Rank 2 just tested individually. Given the two signals' failure
modes look different (precision-favoring vs. a smaller, flatter effect),
there is a real, stated reason — not just optimism — to expect a
combination of the *rich* features might do better than either alone,
the same reasoning that motivated trying the weak-scalar combination in
the first place, now applied to the versions of each signal that
actually contain more information. This remains genuinely cheap: no new
data, no GPU — only an encoder pass over Rank 2's acoustic-candidate
population (scoped to the clips that actually contain a candidate,
matching every prior stage's own cost-scoping discipline).

**Decision**: per the project owner's own explicit hierarchy (existing
components first; cheapest reliable recovery method second; Stage D only
if those demonstrably fail) and this document's own pre-registered exit
condition, **Stage D is not yet justified — one further low-cost,
untried, evidence-motivated combination remains, and should be tried
before Stage D, not after it.** This is a decision to name and record for
the next session, not to implement in this one, per the project owner's
own instruction to stop the experimental chain at the pre-registered
stopping point rather than keep spawning variants inside a single pass.
If that combined-rich-feature classifier is tried and also fails to
clear a deployable floor, that would be the point — for the first time —
where every cheap, evidence-motivated representation-side option this
project can identify has actually been exhausted, and Stage D (re-scoped
to L2, per the Application-Objective Decision Analysis) would become the
honestly-earned next step, not a default.

**Update, same day**: per the project owner's explicit instruction to
treat this as the final bounded low-cost experiment and run it now rather
than defer it, the combined-rich-feature classifier is pre-registered and
run below ("Rank 3").

---

# Rank 3: combined rich-feature classifier — pre-registered protocol (2026-08-07)

Per the project owner's explicit instruction: run this as the final
bounded low-cost experiment in this same pass, without stopping to ask
first, diagnosing genuine bugs if found, and ending with a clear
evidence-based Stage D decision either way — not another new variant if
this one also fails.

## Hypothesis

Rank 1 (full ASR-encoder embedding alone) and Rank 2 (full raw acoustic
MFCC statistics alone) each found real, cross-validated, but individually
insufficient signal, with *differently shaped* failure modes. The
already-failed combined-signal classifier (Phase A, F1=0.242) tested
combination, but only ever combined *hand-picked scalars* from each
domain (one MFCC-similarity number, one encoder-distance number, plus
burst count and duration) — never the rich representations Rank 1/2 just
tested individually. **Rank 3's hypothesis**: a classifier given the
*full* encoder embedding and the *full* raw acoustic feature vector
together, on the same population the original combined classifier used,
separates true `sound_repetition` candidates from false ones better than
either rich signal alone, or than the earlier scalar-only combination.

## Population

**Deliberately the exact same population the original combined-signal
classifier used** — the first 120 real LibriStutter clips with usable
audio (`load_libristutter_dir_with_audio(...)[:120]`, same selection
logic, no re-sorting), scored via `stage_g`'s own acoustic-candidate
generator, `sound_repetition` only. Chosen over Rank 2's larger 499-clip
population **specifically for cost** (§ below) and **specifically for
comparability**: this lets Rank 3 be judged directly against Phase A's
own already-measured Arm A (MFCC-alone, F1=0.147) and Arm B
(encoder-distance-alone, F1=0.244) results on the identical population,
the cleanest possible "did richer features help" test — not a new
population that would confound the comparison. Real counts from the
original run: 766 candidates, 62 positive (~8.1%). Baselines are
recomputed fresh in this run (not reused from the old cached numbers),
same discipline as Rank 1/2, with the same provenance caveat: if the
Track B ASR cache has changed, the re-derived counts may differ slightly
from 766/62, and that will be reported, not hidden.

## Features

- **Acoustic (26 dims)**: Rank 2's own feature exactly — mean + std of
  MFCC coefficients 1-12 across the candidate's bursts, plus n_bursts and
  duration. Reused via direct import from `stage_i_learned_acoustic_
  classifier.py`, unmodified.
- **Encoder (full ~1280-dim embedding + 1 has-signal indicator)**: the
  same pooling logic `stage_combined_classifier.py`'s `_encoder_
  distance_for_candidate` already uses (find the cached ASR hyp token
  overlapping the candidate's span, pool the encoder's last-layer state
  there) — but returning the **raw vector**, not a collapsed distance
  scalar, matching Rank 1's own full-embedding approach. Not every
  candidate has a coverable ASR token (the original run found ~79.2%
  coverage) — missing embeddings are median-imputed per dimension with a
  paired has-signal indicator column, the exact convention `stage_
  combined_classifier.py`'s `_to_feature_matrix()` already established
  and validated.
- **Deliberately excluded**: the derived encoder-distance-to-centroid
  scalar itself is *not* added as an extra input feature (it would be
  redundant with the raw embedding it's derived from) — it is computed
  separately, only as Arm B's own baseline value for comparison.

Total classifier input: 26 + 1280 + 1 ≈ **1307 dimensions**, over 766
candidates (~613 per outer-CV training fold). **Stated honestly, before
running**: this is a substantially worse dimensions-to-samples ratio than
either Rank 1 (1280-dim / 66 positives) or Rank 2 (26-dim / 245
positives) individually — a real, foreseeable risk of instability that
nested-CV L2 selection is designed to mitigate but cannot eliminate. This
is flagged here, not discovered after seeing results, exactly per rule 3.

## Model

The identical nested-CV L2-regularized logistic regression + train-fold-
optimal-threshold mechanism used throughout this track (item 17,
`stage_combined_classifier.py`, Rank 1, Rank 2) — no new model class
introduced for this experiment.

## Baseline

Both of Phase A's own original arms, recomputed fresh on this run's
population: **Arm A** (mean MFCC similarity + CV threshold) and **Arm B**
(encoder-distance-to-centroid + CV threshold) — the same dual-baseline
structure the original combined-signal classifier was judged against, not
a new, easier single baseline.

## Evaluation protocol

5-fold clip-split CV (`_clip_folds`, unmodified). Baselines scored by
`_cv_threshold`; the combined classifier by `_cv_classifier_optimal_
threshold`. Same reporting format as every prior stage.

## Success / failure criteria (fixed before running)

Extends Rank 1/2's stability-then-bars structure to the dual-baseline
case:
- **Stability check**: per-fold win/loss, classifier F1 vs. **Arm B**
  (the stronger of the two baselines, and the most relevant "did this
  beat the single best existing signal" comparison). If the classifier
  does not win on ≥4 of 5 folds **and** does not lose on ≥4 of 5 folds,
  the result is **Inconclusive**.
- **Success**: (only if stable) mean F1 beats **both** Arm A and Arm B by
  **≥20% relative** (the identical dual-bar Phase A itself used) **AND**
  clears the same absolute usability floor used throughout this session
  (precision ≥0.15, recall ≥0.3).
- **Failure**: stable direction, but does not clear all of the above.

## Compute cost

The only new cost is one CrisperWhisper encoder pass over the 120-clip
population's acoustic candidates — **already measured directly** for this
exact shape of computation: Phase A's original combined-signal classifier
ran this same encoder pass over this same population in **3068s (~51
minutes)**. That number is cited directly as this run's cost estimate,
not re-derived from a fresh dry run, since it is the identical
computation. Acoustic feature extraction and the CV/classifier fitting
itself add negligible time (seconds, pure numpy).

## What this experiment settles either way

Per the project owner's explicit instruction: if Rank 3 also fails, this
document does **not** propose a further new feature/variant — the
verdict moves directly to a Stage D go/no-go decision, stated with the
evidence this and the two prior experiments together provide. If Rank 3
succeeds, the result is validated toward the live pipeline as the next,
separate engineering step, not treated as license to keep researching.

## Novelty (kept explicitly secondary, per the governing objective)

Not a novelty claim — a direct, same-population re-test of Phase A's own
already-failed hypothesis, now with the two richer feature sets Phase A
never had access to, both developed later in this same track.

## Rank 3 results (2026-08-07)

Implemented as `profiling/evaluation/stage_j_combined_rich_classifier.py`,
self-tested (11/11 checks pass after one self-test authoring bug was
caught and fixed before trusting the suite — two hand-constructed arms
were given identical F1 values, making a "beats A, not B" case
collapse into a tied, correctly-Inconclusive case instead; the
underlying decision-gate logic was right, the test data was wrong, fixed
by giving the two arms distinct values), then run for real against the
pre-registered protocol, no changes to the protocol after seeing results.

**Population reproduced exactly**: 766 candidates, 62 positive — an
exact match to Phase A's original 120-clip run (unlike Rank 1's
population, which had grown since its own baseline session), and Arm A/
Arm B recomputed fresh here landed on the **identical** F1 numbers Phase
A originally reported (F1=0.147 / F1=0.244) — a clean confirmation this
run's population and methodology are the same ones being compared
against, not a confound.

**Real result (n_pos=62, n_neg=704):**

| Arm | F1 | Precision | Recall |
|---|---|---|---|
| Arm A: MFCC-similarity-alone + CV threshold | 0.147 | 0.090 | 0.541 |
| Arm B: encoder-distance-alone + CV threshold | 0.244 | 0.343 | 0.234 |
| Rank 3: full acoustic (26d) + full embedding (1280d) combined | 0.254 | 0.330 | 0.257 |

Mean F1 is nominally higher for the combined classifier than either
individual arm. But the pre-registered **stability check** (per-fold win/
loss against Arm B, the stronger baseline) came back **2W/3L of 5
folds** — the combined classifier actually did *worse* than the single
encoder-distance scalar on 3 of the 5 outer folds, despite winning on
mean. Per the pre-registered gate, this is **Inconclusive**, not a pass
on the mean-F1 comparison alone.

**Audit before trusting (rule 3)**: this is exactly the failure mode the
pre-registration flagged as a real risk *before* running — 1307 input
dimensions over ~613 training rows per fold. The combined classifier's
own fold-to-fold F1 range (0.091-0.400) is far wider than either
individual arm's (Arm A: 0.121-0.189; Arm B: 0.125-0.300), and its worst
fold (0.091) is worse than Arm A's worst fold — a concrete symptom of an
unstable, high-variance fit, not a robust improvement that happens to
look noisy. The mean-F1 "win" (0.254 vs. 0.244) is well within the
range this instability alone could produce by chance; it should not be
read as evidence the combination helped.

**Verdict per the pre-registered gate: INCONCLUSIVE** — this experiment
does not provide reliable evidence that combining the full embedding and
full acoustic features beats the single best existing signal (Arm B)
already established in this track.

---

# Final synthesis: Rank 1 + Rank 2 + Rank 3, and the Stage D decision (2026-08-07)

Per the project owner's explicit instruction: if Rank 3 also failed (or,
as it did, came back Inconclusive with a real instability signature, not
a clean win), do not invent a further feature/variant — give the
evidence-based Stage D decision directly.

**What the full arc of representation-side experiments now shows,
together:**
- **Rank 1** (full ASR-encoder embedding alone, on the ASR-alignment-gap
  population): real signal, ~10x precision gain, but recall collapsed
  (0.147) — Failure on deployability.
- **Rank 2** (full raw acoustic MFCC statistics alone, on the acoustic-
  candidate population, 4x the sample size): real but small, boundary-
  case signal — Failure on deployability, by a narrow margin.
- **Rank 3** (both together, on Phase A's original acoustic-candidate
  population): no reliable improvement over the single strongest
  individual signal already known (Arm B) — Inconclusive, with an
  instability signature (wide fold-to-fold variance, a worse worst-case
  fold than the weaker baseline) that is itself informative: it is the
  signature of a classifier running out of data to fit its own
  dimensionality, not the signature of a signal waiting to be found by
  yet more feature engineering.

**Every cheap, no-GPU, no-new-data representation-side option this
project could identify has now been tried**: the encoder representation
alone, the acoustic representation alone, and their combination. None
reached the pre-registered deployability floor, and the richest attempt
(Rank 3) shows a pattern consistent with a genuine small-sample ceiling —
adding more parameters (going from Rank 2's 26 dims or Rank 1's 1280 dims
to their 1307-dim union) made the fit *less* stable, not more powerful.
This is a structurally different situation from "we haven't looked hard
enough" — it is evidence that the limiting factor is the quantity of real
labeled `sound_repetition`/`word_repetition` examples this project can
currently access (62-245 depending on population, an order of magnitude
smaller than what a ~1000+-dimensional classifier needs to fit reliably),
not the choice of representation.

**Does this move the project closer to a working application?** Yes, in
an honest, bounded sense: three real, quantified, negative-to-marginal
results now exist where before this session there was only a single
weaker hand-picked-scalar attempt — the project knows, with evidence, that
representation choice alone will not close this gap at the current data
scale, which is itself a real, load-bearing finding that prevents future
sessions from re-trying the same kind of cheap feature-engineering
variant expecting a different outcome.

## The Stage D decision

**Stage D (re-scoped to L2 — stop deleting disfluency information during
ASR decoding, not spell out fragment content, per the Application-
Objective Decision Analysis) is now judged the evidence-justified next
step.**

This is a considered judgment over the full three-experiment arc, not a
mechanical trigger from Rank 3's Inconclusive verdict alone — stated
plainly per standing rule 8's requirement to record the reasoning, not
just the verdict:
- Rank 1 and Rank 2 are clean Failures on their own pre-registered
  criteria, not Inconclusive results being stretched into a verdict.
- Rank 3's Inconclusive result is not a "we don't know, try again"
  outcome — its specific *shape* (worse-than-baseline worst-case fold,
  wide variance despite a nominally higher mean) is itself evidence of a
  sample-size ceiling, not an open question a fourth cheap variant could
  resolve. Per the project owner's own explicit instruction, a fourth
  variant is not proposed.
- Stage D is mechanistically a different kind of intervention than
  anything Rank 1-3 tried: those three fit a classifier on top of the
  *existing, frozen* ASR/encoder representation; Stage D would change
  what the ASR itself produces via fine-tuning on more real data —
  addressing the actual bottleneck this arc identified (data quantity)
  rather than repeating the same small-sample classifier-fitting exercise
  a fourth time with yet another feature combination.
- This is consistent with, not a departure from, the project owner's own
  explicit decision hierarchy: existing components first (tried, and
  found lossy — the original ASR-research-track finding); cheapest
  reliable recovery method second (now tried three ways, all falling
  short); Stage D only once those demonstrably fail (met).

**What remains true and unchanged from earlier sessions**: Stage D's own
real, already-documented infrastructure requirements (no local GPU, real
English-language disfluent-speech data acquisition) have not changed and
are not resolved by this decision — this is a recommendation that Stage D
is now the *evidence-justified* next step, not a claim that it is
immediately actionable without the project owner's separate decision on
budget/infrastructure, exactly as this track's own "Stage D Requirements
/ Re-entry Gate" section already specified.

---

# Current Project State — 2026-08-07 EOD

**This is the end-of-day consolidation checkpoint for 2026-08-07.** It is
an *additive* synthesis, not a replacement for anything above it — every
section above this one remains the full, unedited historical record of
how this project arrived here, with its own citations, methodology, and
reasoning intact. This section's job is narrower: give a reader who has
never seen this conversation a complete, accurate, properly cross-
referenced picture of where the project actually stands right now,
without asking them to reconstruct it from ~7,600 lines of chronological
research log. Read this section first; use the pointers in it to go
deeper into any specific claim.

## The primary objective — restated, because it governs everything below

Per the "PROJECT OBJECTIVE" section at the very top of this document
(added 2026-08-07, morning) and reaffirmed explicitly by the project
owner at the start of this evening's session: **the objective is to
build an application that takes a speaker's audio and reliably extracts
the speech problems/hindrances actually occurring in it — including
stuttering and other disfluencies — so they can be localized, classified,
understood, and used downstream for assistance such as rephrasing,
synonym substitution, or alternative phrasing.** Restated one more time
in the project owner's own words tonight: *"Speaker audio → preserve
useful disfluency evidence → detect/classify/localize problems → provide
useful feedback/intervention. The ASR is a component serving that
application. It isn't the product by itself."*

The objective is **not**: proving nobody has solved this before, finding
an unexplored research territory, creating a novel ASR for its own sake,
or writing a paper for its own sake. An ASR (or any component this track
investigates) earns a place in the pipeline only if it is actually
necessary and useful for the application objective — never by default,
never because it is interesting. This governs every conclusion below.

## A. The complete chronological experimental story

Each item below states the real, measured result and points to the full
section for methodology, citations, and reasoning — nothing here
replaces those sections.

**Stage A — large-scale evidence of information loss/normalization.**
CrisperWhisper's decoded text normalizes away real, ground-truth
`sound_repetition`/`word_repetition` instances at a high rate (45.2%/
40.5% — figures established and cited throughout this document, e.g. the
"Stage D Requirements / Re-entry Gate" §D baseline list above) — the
finding that opened this entire research track. Full detail: the
document's own opening sections and the repeated citations of this exact
figure throughout (search "45.2%/40.5%").

**Stage B/C — the encoder retains information the decoder discards, but
not always precisely enough to act on alone.** Stage B found a real,
Cohen's-d-confirmed signal (`sound_repetition` d=0.894) in CrisperWhisper's
own last-layer encoder states at exactly the positions where the decoded
text gives zero evidence. Stage C confirmed this signal survives a
duration-confound check (encoder arm AUC=0.723 vs. a duration-only
baseline AUC=0.483) — real, not an artifact of "longer disfluent bursts
are just louder/longer." See "Stage B — pre-registered protocol" (line
2100) / "Stage B — Results" (2205) and "Stage C — pre-registered
protocol" (2406) / "Stage C — Results" (2474) above for full methodology
and numbers, including Stage C2's negative Praat voice-quality fusion
result (2564/2661).

**Phase 2 — alternative ASR/representation/decoding investigations.**
A pre-registered 3-arm comparison (stock `whisper-large-v3` full
pipeline; the same checkpoint's encoder via layer sweep; WavLM-Large's
representation) to test whether CrisperWhisper specifically, or the
Whisper/general-ASR architecture more broadly, is the bottleneck. All
three arms: Failure (see "Phase 2 results" at line 1022 and "Integrative
conclusion: Failure / Failure / Failure" at 1117) — the normalization
pattern is architecture-general (Whisper-family), not a CrisperWhisper-
specific fine-tuning artifact, and WavLM's differently-distributed signal
was never stronger than either Whisper variant. This result is what
motivated moving to acoustic-native and combined-signal directions rather
than continuing to shop among existing ASR/representation choices.

**Direction (g) — acoustic-native `sound_repetition` candidate
generation, RMS/ZCR pass.** A waveform-only detection mechanism (short,
spectrally self-similar voiced bursts), tested with no ASR dependency at
all, using RMS/ZCR envelope-shape similarity as the first, cheapest
per-burst feature. Real result: recall=0.824 but precision=0.081 — high
recall, unusable precision, because ordinary fluent speech (function
words, fast syllables) also produces short, similarly-shaped voiced
segments the RMS/ZCR envelope-shape feature cannot distinguish from a
genuine repeated fragment — a feature-specificity problem, not evidence
against an acoustic signature existing (the high recall is itself real,
positive evidence a signature exists). A real cross-clip scoring bug
(pooling raw per-clip timestamps globally, producing an implausible
recall=1.000/precision=0.966) was caught per rule 3 and fixed before this
result was trusted. Full detail: "Direction (g): an acoustic-native
`sound_repetition` candidate generator — pre-registered protocol" (line
1174) and "Direction (g) results" (1306).

**MFCC escalation — corrected implementation, real trade-off, still
short of the bar.** Escalated from RMS/ZCR to a hand-rolled MFCC spectral-
shape similarity feature. A second real bug was caught first: MFCC
coefficient 0 (overall log-energy, not spectral shape) was dominating
similarity scores — 90% of all burst pairs scored ≥0.9 similarity
including it, because any two voiced (energetic) segments look similar
on energy alone, silently defeating the point of escalating past RMS/ZCR
— fixed by excluding coefficient 0 (standard MFCC practice), then
re-run. Corrected result: a genuine precision/recall trade-off curve
(unlike RMS/ZCR's flat one) with best F1=0.170 (recall=0.686,
precision=0.097) vs. RMS/ZCR's F1=0.161 — a real, modest improvement,
but short of the pre-registered ≥20%-relative-improvement bar over the
duration-only baseline (needed F1≥0.176) by a small margin. Full detail:
"Direction (g), MFCC escalation — pre-registered addendum" (line 1433)
and "MFCC escalation results" (1491).

**Combined-signal classifier — weak scalar signals, combined.** Tested
whether a plain logistic regression over 4 hand-picked *scalar* features
(MFCC similarity, burst count, duration, and Stage C's encoder-distance
scalar) separates true `sound_repetition` instances better than any
single signal alone. A real bug (F1=0.000 across all 5 folds despite one
input feature independently scoring F1=0.244) was caught and fixed —
diagnosed via a synthetic check as a hardcoded 0.5 decision threshold
miscalibrated to this population's ~8% positive rate, not a model that
learned nothing — by giving the classifier the same train-fold-optimal
threshold convention its baseline arms already used. Corrected result:
combined F1=0.242, MFCC-alone F1=0.147, encoder-distance-alone F1=0.244
— the combination **matched but did not exceed** the strongest individual
signal (needed to beat both by ≥20% relative to count as Success) —
**Failure**. This result specifically showed that combining *weak,
hand-picked scalars* did not help; it did **not** test whether combining
the *rich* representations each signal is derived from would help (that
question is exactly what Rank 3, below, was designed to answer). Full
detail: "Combined-signal classifier — pre-registered protocol" (line
3358) and "Combined-signal classifier results" (3547).

**Literature review, in parallel** — see item C below; not repeated here.

**Rank 1 — full ASR-encoder embedding, candidate-generation
classifier.** Tested whether item 17's own shipped classifier mechanism
(the full ~1280-dim CrisperWhisper encoder embedding, nested-CV
L2-regularized logistic regression, train-fold-optimal threshold) —
previously only ever applied to *corroborating* already-flagged
candidates — separates true candidate-*generation*-gap positions (real
ASR-normalized-away `sound_repetition`/`word_repetition` sites) from
clean positions, compared against Stage C's own single-scalar
distance-to-centroid baseline, both cross-validated identically. Real
result (n_pos=66, n_neg=1780): baseline F1=0.097 (P=0.055, R=0.462) vs.
classifier **F1=0.230 (P=0.580, R=0.147)** — precision improved roughly
10x and mean F1 more than doubled, clearing the pre-registered relative
bar by a wide margin, with a stable per-fold direction (4 of 5 folds).
**But recall (0.147) missed the pre-registered absolute usability floor
(≥0.3) — Failure on deployability specifically**, not on whether the
full embedding carries more signal than the scalar reduction (it clearly
does). This experiment could not distinguish whether the low recall
reflects a genuine representation ceiling or simply too few positive
examples (66) to fit a stable ~1280-dimensional decision boundary — left
explicitly unresolved, not guessed at. Full detail, including the
provenance note on why this run's population (66/1780) differs slightly
from Stage B/C's original (36/966) due to Track B cache growth between
sessions: "Rank 1: full-embedding candidate-generation classifier —
pre-registered protocol" (line 6767) and "Rank 1 results" (6980).

**Rank 2 — learned acoustic classifier over richer MFCC statistics.**
Tested whether a classifier that *learns* how to weight per-coefficient
MFCC statistics (mean + std across a candidate's bursts, 26 dims) beats
direction (g)'s *hand-designed* single-scalar cosine-similarity metric —
acoustic-only, no ASR/encoder involved, scaled to all 499 real
LibriStutter clips (affordable specifically because this arm needs no
encoder pass, ~6 minutes total). Real result (n_pos=245, n_neg=2970):
baseline F1=0.133 (P=0.077, R=0.531) vs. classifier **F1=0.163 (P=0.114,
R=0.308)** — clears the relative bar only barely (a margin smaller than
either arm's own fold-to-fold F1 variance), recall clears the absolute
floor narrowly (0.308) but **precision does not (0.114 vs. ≥0.15) —
Failure**, honestly flagged as possibly just a well-calibrated version of
the same underlying similarity concept rather than genuinely new
information (this experiment could not rule that out). Provides a
**differently shaped** signal than Rank 1 (small, flatter effect vs.
Rank 1's large precision gain) — evidence the two representations are
not redundant with each other, which is exactly what motivated Rank 3.
Full detail: "Rank 2: learned acoustic candidate-generation classifier —
pre-registered protocol" (line 7056) and "Rank 2 results" (7239); see
also "Rank 1 + Rank 2 synthesis and next-step decision" (7305), which
first named the Rank 3 idea.

**Rank 3 — full encoder embedding + full acoustic representation,
combined.** Tested whether concatenating Rank 1's full embedding with
Rank 2's full 26-dim acoustic features, together, beats either alone —
on Phase A's *original* 120-clip acoustic-candidate population (a
deliberate choice for direct comparability to the already-published
Arm A/B baselines, not Rank 2's larger population). Population and
baselines reproduced Phase A's original numbers exactly (766 candidates,
62 positive; MFCC-alone F1=0.147; encoder-distance-alone F1=0.244) —
confirming a clean, unconfounded comparison. Combined classifier:
**F1=0.254** — nominally the highest of the three arms. But the
pre-registered per-fold stability check (combined vs. the stronger
encoder-distance baseline) came back **2 wins / 3 losses of 5 folds** —
the combined classifier actually lost to the single scalar signal on a
majority of folds despite a higher mean, with a fold-to-fold F1 range
(0.091-0.400) wider than either individual arm's and a worst-case fold
(0.091) worse than even the *weaker* MFCC-alone baseline's worst fold
(0.121). **This is why the mean F1 alone must not be read as success**:
it is exactly the signature of an unstable, high-variance fit at
~1307 dimensions over ~613 training rows per fold — a risk this track
pre-registered explicitly *before* running, not one discovered
after the fact. Per the pre-registered gate: **Inconclusive**, not
Success. One self-test authoring bug (two hand-constructed comparison
arms accidentally given identical F1 values, collapsing an intended test
case into a trivial, already-correct tie) was caught and fixed before
trusting the 11-test suite; the decision-gate logic itself was correct
throughout. Full detail: "Rank 3: combined rich-feature classifier —
pre-registered protocol" (line 7378) and "Rank 3 results" (7512).

**The cumulative conclusion**, stated once, precisely: three
independent, cheap, no-GPU, no-new-data representation-side experiments
(encoder alone, acoustic alone, both combined) each found real,
cross-validated, non-random signal for this exact candidate-generation
gap — but none reached a pre-registered deployable operating point, and
the richest combination became *less* stable, not more powerful, when
given more information. Taken together with the two clean Failures
(Rank 1, Rank 2) and Rank 3's own diagnostic instability signature, this
is judged — as a considered whole, not a mechanical trigger — sufficient
evidence that the limiting factor is the *quantity* of real labeled
examples available (62-245 depending on population), not the choice of
representation. See "Final synthesis: Rank 1 + Rank 2 + Rank 3, and the
Stage D decision" (line 7565) for the full reasoning and the resulting
Stage D decision, summarized again in section F below.

## B. What is EXPERIMENTALLY DEMONSTRATED, STRONGLY SUPPORTED, SUPPORTED BY LITERATURE, INFERRED, UNRESOLVED, NOT TESTED, and PROPOSED

Kept as precise, separate categories per the project owner's explicit
instruction — not blurred into each other, and specifically avoiding two
overclaims the project owner named directly: **do not write "ASR cannot
solve this"** (the correct, narrower claim is below), and **do not write
"nobody has solved this"** (already falsified by this track's own
literature review — see section C).

**EXPERIMENTALLY DEMONSTRATED** (this track's own measured results,
real audio or real experiments, cited above):
- CrisperWhisper's decoded text normalizes away a large fraction (45.2%/
  40.5%) of real `sound_repetition`/`word_repetition` ground truth (Stage
  A).
- The encoder retains a real, duration-confound-refuted signal at
  exactly the positions the decoder discards (Stage B/C: d=0.894,
  AUC=0.723).
- This normalization pattern is Whisper-architecture-general, not a
  CrisperWhisper-specific artifact (Phase 2, all 3 arms Failure in the
  same direction).
- `sound_repetition` has *some* recoverable acoustic signature
  independent of ASR entirely (direction (g)'s real recall=0.824,
  even though precision was unusable at that stage).
- A learned classifier over the full encoder embedding recovers real
  signal at candidate-generation-gap positions that a single hand-picked
  scalar does not (Rank 1: ~10x precision improvement, stable across
  folds).
- A learned classifier over raw acoustic features modestly beats a
  hand-designed acoustic similarity heuristic, but only marginally
  (Rank 2).
- Combining the full embedding and full acoustic features does **not**
  reliably improve over the best single existing signal at the sample
  sizes available, and shows a concrete instability signature when it is
  tried (Rank 3).

**STRONGLY SUPPORTED** (very likely true given the pattern of evidence,
short of being independently re-verified):
- The limiting factor for further cheap, frozen-representation
  candidate-generation work is real-labeled-example quantity (62-245
  positives depending on population), not representation choice — this
  is the load-bearing inference behind tonight's Stage D decision,
  supported by three independently-shaped results (two Failures plus one
  instability-flagged Inconclusive) rather than any single result alone.

**SUPPORTED BY LITERATURE** (established by sources this track verified
directly, not by this track's own experiments):
- Real precedent exists for fine-tuning reducing an ASR's own
  deletion/normalization rate on disfluent speech (StammerTalk: 26.56%→
  2.29% deletion-error rate for severe stuttering) — the direct
  evidentiary basis for Stage D's L2-level scope, see section D below and
  section C for full citation detail.
- Real precedent exists for explicit disfluency-marker fine-tuning at
  modest real-world compute scale (Kordt et al., ~10-20 hours of training
  data per corpus) — recalibrates what "large-scale" plausibly means for
  a Stage D pilot.
- No dataset reviewed provides real speech with word-level, fragment-
  vs-word-granular repetition annotation at scale, off-the-shelf (the
  literature-verified data gap that makes Stage D's data requirement
  real, not assumed).

**INFERRED** (this track's own reasoned conclusions from evidence, not
directly measured, and explicitly labeled as such throughout the
document body):
- A taxonomy-targeted fine-tuning objective would likely do better at
  this specific taxonomy than a representation trained for a different
  objective (general ML priors — supervised learning tends to improve at
  what it's optimized for — not something this track's evidence has
  tested or can currently confirm or refute; Stage D would be the first
  real test).
- Mandarin orthography likely cannot represent a sub-character phonetic
  fragment as literal text at all, meaning AS-70/StammerTalk may be
  structurally unable to test true L4 fragment-content preservation even
  in principle — this review's own reasoned inference, not a claim any
  source makes directly (see "A structural finding" at line 6237).

**UNRESOLVED** (this track's evidence explicitly cannot answer these,
stated as open, not guessed at):
- Whether Rank 1's low recall (0.147) reflects a genuine encoder-
  representation ceiling for this task, or is simply an artifact of too
  few positive examples (66) to fit a stable high-dimensional decision
  boundary.
- Whether Rank 2's classifier is exploiting genuinely new per-coefficient
  information, or is mostly rediscovering a calibrated version of the
  same similarity concept the one hand-designed scalar already captured.
- What real precision/recall an L2-scoped Stage D fine-tune would
  actually achieve on this project's own taxonomy — no experiment in
  this track has tested Stage D's actual premise (a taxonomy-targeted
  training objective); every experiment so far tested representations
  already trained for a different objective.

**NOT TESTED** (explicitly out of scope for every experiment run so
far):
- Any actual model fine-tuning/training (Stage D itself — every
  experiment in this track evaluated frozen, pretrained representations).
- L4-level fragment-content preservation (e.g. literally transcribing
  "c-c-cat") in any language — the literature review found no source that
  has attempted this for `sound_repetition` specifically, and this
  track's own experiments never targeted it either (see section D on why
  L4 is not currently believed necessary for the application anyway).
- Generalization of any of this track's results beyond LibriStutter's
  synthetic splicing to real, non-synthetic disfluent speech — a real,
  named, unresolved risk for any future training effort.

**PROPOSED** (named as options, not committed to, not yet built):
- The re-scoped L2-level Stage D fine-tune (section D/F below).
- A "cheap real-data generalization check" (acquire a modest amount of
  real disfluent speech, without training anything, to test whether
  today's signals generalize beyond LibriStutter before committing to
  Stage D) — named in "Stage D planning (2026-08-07)" (line 3771) as a
  real, credible, cheaper-than-Stage-D option; not decided on tonight,
  named here again for completeness since section F below judges Stage D
  itself as now justified without first requiring this check, a
  judgment call explained there.

## C. The literature position — what existing technology can we use, not "can we claim novelty"

This section's purpose, restated per the project owner's explicit
instruction: **"What existing technology can we use to build the
application?"** — not "can we claim nobody has done this?" The full,
detailed literature review (every source directly fetched and read, not
summarized from search snippets) remains completely intact above — the
"Final research audit" (line 3808), "Comprehensive research-positioning
analysis (deep pass)" (4826), and "Novelty-falsification pass" (6157)
sections, with two full bibliographies (B1-B19 at 4676, B20-B26 at 6082,
plus B27). This subsection is a short, application-focused index into
that material, not a replacement for it.

| Source | What it solves | What our application needs from it |
|---|---|---|
| **AS-70** (Gong et al., Interspeech 2024) [B20/B27-adjacent] | A real, 48.8-hour, verbatim, character-level-annotated Mandarin stuttering dataset with a genuine sound-repetition (`/r`) vs. word/phrase-repetition (`[]`) distinction — the right taxonomy granularity, on real speech. | The clearest real precedent that fragment-granular real-speech annotation is achievable at all; not directly usable as training data for an English-language application, and — per this track's own structural inference — may be unable to represent true L4 sound-fragment *content* even in principle, though L4 is not what our application needs anyway (see below). |
| **StammerTalk / StutteredSpeechASR** (Li et al., FAccT 2025) [B21] | A real, published Whisper-large-v2 LoRA fine-tune on AS-70, with a directly quantified, large deletion-error-rate improvement (26.56%→2.29% severe stuttering) — i.e., a real demonstration that fine-tuning stops an ASR from deleting disfluent content. | **This is the single closest real precedent for exactly the L2-level intervention our application needs** — not fragment-content spelling, just "stop deleting evidence that something disfluent happened here." Directly cited as Stage D's own L2 scoping justification (section D below). |
| **Kordt et al.** (Interspeech 2026 target) [B11] | Explicit disfluency-marker (`REP`, etc.) fine-tuning of Whisper, with real continual-learning methods (EWC, Experience Replay, A-GEM, Weight Averaging) to control catastrophic forgetting, at modest (~10-20h/corpus) real compute scale. Collapses word- and phoneme-level repetition into one coarse marker. | The best real precedent for *how* to fine-tune without wrecking general transcription quality, and for realistic compute/data scale — but not fine-grained enough on its own for our `sound_repetition`-vs-`word_repetition` distinction. |
| **Whisper family** (stock `whisper-large-v3`, CrisperWhisper) | General, fluent/verbatim-leaning transcription; CrisperWhisper specifically is disfluency-preserving-by-design relative to stock Whisper, which is why it was chosen as this project's ASR in the first place. | The current pipeline's ASR component — real, working, chosen for good reason, but demonstrably normalizes away 45.2%/40.5% of the exact evidence this application needs (Stage A) — the finding that opened this whole track. Not being replaced; being extended. |
| **WavLM-Large** | A paralinguistic-sensitivity-oriented encoder, real published design objective. | Tested directly in Phase 2 (Arm 3): no stronger than either Whisper variant for this specific taxonomy, though its signal distributes differently across depth — a real, literature-consistent finding, not adopted further. |
| **HuBERT** and related self-supervised encoders | General-purpose speech representation learning. | Reviewed in the literature passes above as part of the broader representation landscape; not separately experimented on in this track — no claim made about how they'd perform here (NOT TESTED). |
| **Explicit disfluency-token / joint ASR+SED approaches** (Kordt et al.; Huang et al., reviewed in the deep pass and the Application-Objective Decision Analysis §3-4) | Detecting/marking that a disfluency occurred, sometimes jointly with ASR. | Read against the actual application requirement (detection, not fragment-content preservation), these are **closer to solving our real problem than the earlier novelty-centered framing credited them for** — see Application-Objective Decision Analysis §3-4 (lines 6421-6496) for the full re-scoring. |
| **YOLO-Stutter / SSDM** and related audio-native learned detectors | Deep, GPU-trained, end-to-end learned detection directly from audio, no ASR involved. | The architecture-*class* Rank 2 deliberately, honestly scoped down from (stated explicitly in Rank 2's own pre-registration, line 7158) to something CPU-feasible — Rank 2 is not a claim of parity with this class, only a bounded approximation of its spirit. |

**Where existing technology already fully solves a piece of what the
application needs**: block/prolongation/filler detection — already
solved, shipped, in `profiling/detect.py`, no research gap (see
Application-Objective Decision Analysis §2, line 6401, for the exact
per-type mapping table). **Where existing technology partially solves
it**: `sound_repetition`/`word_repetition` detection — the encoder and
acoustic signals this track measured are real partial solutions
(non-zero, cross-validated signal), just not yet deployable ones.
**Where a real, narrow gap remains, accurately documented but kept
secondary to the application**: fragment-level (`sound_repetition`
specifically) candidate-*generation* at production-usable precision/
recall, from either frozen representation, at this project's current
data scale — this is the actual, precise, narrowly-scoped gap, not
"detecting stuttering" broadly and not "novel ASR" generally.

## D. The application requirement, reassessed with everything learned today

Restating and lightly extending the Application-Objective Decision
Analysis's own §1 (line 6353) and §2 (6401) — not replacing them, both
remain the fuller treatment:

- **Intended/clean linguistic content**: required, and — the key,
  repeatedly-reconfirmed finding across today's work — typically already
  available even where the disfluency *evidence itself* is lost, because
  the "clean" word CrisperWhisper eventually settles on is usually
  correct even when it silently dropped the repeated fragment that
  preceded it.
- **Actual disfluent realization (that a repetition/block/prolongation/
  filler occurred, and roughly where)**: required — this is what
  "detection" means for this application, and is the actual target of
  Rank 1/2/3 and everything upstream of them today.
- **Sound repetitions, word/phrase repetitions**: the two types this
  entire day's experimental arc targeted — real, quantified, partial
  progress, not yet deployable (section A/B above).
- **Prolongations, blocks**: already solved, shipped, not touched by any
  of today's work (Application-Objective Decision Analysis §2).
- **Interjections/fillers**: already solved, shipped, not touched by any
  of today's work (same).
- **Timing/localization**: required, and — per the Application-Objective
  Decision Analysis's own §1 — word-level precision is likely sufficient
  (an inference, not independently re-tested today).
- **Acoustic evidence**: a *means*, not an end — every acoustic feature
  this track has built (direction (g)'s bursts, Rank 2's MFCC statistics)
  exists to serve detection, never valued for its own sake.
- **ASR evidence**: also a means, not an end — the encoder's internal
  representations (Rank 1, Rank 3) are used precisely because they
  contain information the *decoded text* alone does not, not because ASR
  itself is the goal.
- **Not every piece of information needs to appear in the ASR transcript
  itself** — this is the single most important reframing to carry
  forward: the current and every proposed architecture in this document
  is fundamentally `audio → ASR + audio-native evidence → alignment/
  fusion → disfluency events → downstream assistance`, exactly the shape
  the project owner named tonight, and exactly what the shipped detector
  (`profiling/detect.py`) already implements structurally (word-level ASR
  text plus acoustic corroboration, fused) — today's work is about
  improving the fusion's candidate-generation half specifically, not
  about replacing this architecture.
- **Downstream semantic/rephrasing requirements**: out of this repo's
  own scope — `audio-mod` owns extraction; a separate "Speech AI Module"
  (per this project's own established two-repo product structure) owns
  the downstream rephrasing/substitution. Nothing in today's work changes
  that boundary.

## E. Stage D — retrieved and updated, not reinvented

**Superseded-in-place notice, 2026-08-08**: this section's conclusion
that Stage D is "the evidence-justified next step" rested on an
exhaustion premise (section F below) that a 2026-08-07 external review
falsified — a cheaper, untried, better-matched mechanism
(reference-text-conditioned phonetic realignment, e.g. Dysfluent-WFST)
existed and was never considered. Stage D is now judged **premature, not
rejected** — see "External Review Reconciliation — 2026-08-08" at the
end of this document for the full re-analysis and the current decision
tree. Everything below in this section remains intact and still
describes what Stage D *would be* if/when it becomes justified; only the
"is it justified now" verdict changed.

**Everything Stage D means in this project was already developed across
two sections earlier today**: "Stage D planning (2026-08-07) — design,
requirements, risks, and a real infrastructure finding" (line 3622) and
"Stage D Requirements / Re-entry Gate" (4296) — both remain complete and
authoritative for *what Stage D is, what it needs, and what blocks it*.
Nothing in this subsection replaces them; it summarizes them and states
what changed tonight.

- **What Stage D means**: fine-tuning an *existing* pretrained checkpoint
  (CrisperWhisper or stock `whisper-large-v3`) — **not training a new ASR
  from scratch** — with a taxonomy-targeted auxiliary objective, most
  plausibly LoRA-style parameter-efficient adaptation (matching
  StammerTalk's own real, published recipe) combined with a continual-
  learning method (EWC/Experience Replay/A-GEM/Weight Averaging, per
  Kordt et al.) to control catastrophic forgetting.
- **What it is NOT**: not a from-scratch architecture; not (per the L2
  re-scoping below) an attempt to spell out literal fragment content;
  not something this project can start tonight or without a separate,
  explicit infrastructure decision.
- **L2 vs. L4 — explicitly distinguished, and the scope actually
  targeted**: **L2** means preventing useful disfluency *evidence* from
  being normalized/deleted during decoding (StammerTalk's own directly-
  demonstrated objective: deletion-error rate 26.56%→2.29%) — something
  *is* output at the disfluent position, even if not phonetically exact.
  **L4** is the much stronger objective of preserving the literal
  repeated fragment's *content* (e.g. "c-c-cat" as actual output text) —
  found, by this track's own literature-falsification pass, to be
  unaddressed by any source, and, per the Application-Objective Decision
  Analysis, **not currently believed necessary for this application**,
  since the intended/clean word is typically already recoverable even
  when the disfluency evidence is lost (section D above). **Stage D, as
  currently scoped, targets L2, not L4** — this is a real, explicit,
  dated revision from the "Novelty-falsification pass"'s own original
  L4-oriented Stage D framing (see "What this means for Stage D, revised
  precisely," line 6296, and the Application-Objective Decision
  Analysis's §7, line 6569, which performs and explains this exact
  revision) — preserved above, not deleted, exactly as the project
  owner's preservation rule requires.
- **Data required**: real (non-synthetic) disfluent speech with word-
  level timestamps and a fragment-vs-word repetition distinction — not
  provided off-the-shelf at scale by SEP-28k, KSoF, or FluencyBank's
  existing schemas (literature-verified gap). FluencyBank Timestamped is
  the best-identified fit (real people who stutter, word-level
  timestamps) but needs an unbuilt CHAT-format parser and possible access
  approval. **LibriStutter (this track's own working sample) is
  synthetic splicing and is explicitly judged insufficient as the sole
  training corpus** — a model fine-tuned on it risks learning splice
  artifacts rather than genuine disfluency production patterns, a
  materially worse risk for a *training* target than it has been for
  this track's own repeated evaluation-only use of it.
- **Annotation structure required**: word-level timestamps (already this
  project's native format) plus a fragment-vs-word distinction at each
  disfluent position.
- **Compute required**: GPU-scale, for any realistic fine-tuning run. A
  single mid-range cloud GPU (16-24GB VRAM) would plausibly suffice for a
  `whisper-small`/`base`-scale pilot following Kordt et al.'s own
  demonstrated recipe; CrisperWhisper/`whisper-large-v3`-scale would want
  24-48GB+ or LoRA-style adaptation to stay in the same modest compute
  envelope.
- **Why the current CPU-only machine is insufficient**: verified
  directly, not assumed — `torch.cuda.is_available()` is `False`, the
  installed `torch` build is CPU-only, the only GPU present is
  integrated (not CUDA-capable). Fine-tuning requires forward *and*
  backward passes across many real examples per epoch, across multiple
  epochs — categorically heavier than this project's own already-slow
  CPU inference (~54s/clip for one forward pass); training at any
  reasonable scale would take days-to-weeks on this hardware.
- **Minimum viable Stage D experiment**: `whisper-small`-scale,
  FluencyBank-Timestamped-only (or equivalent), `sound_repetition`-only,
  L2-scoped (reduce the deletion/normalized-away rate, evaluated by this
  project's own already-built Stage-A-style 4-category classification
  re-applied to the fine-tuned model's output) — not full CrisperWhisper-
  scale, full-taxonomy scope, matching this track's own repeated
  "cheapest first" discipline.
- **Evaluation required**: this track's own already-measured baselines
  (Stage A's 45.2%/40.5%, stock Whisper's 89.5%/88.2%, the combined
  classifier's F1=0.242) as the numbers to beat; a genuinely held-out,
  speaker-disjoint real-data test split; general clean-speech WER
  measured directly to catch catastrophic forgetting rather than assume
  its absence.
- **Main risks**: no accessible large-scale real paired training data
  (potentially blocking on its own); LibriStutter's synthetic-splice risk
  producing a confidently-wrong "success" if used as the primary training
  corpus; catastrophic forgetting (literature-documented risk); no local
  compute (cloud-GPU cost/setup, not yet budgeted); unverified
  generalization even after a nominally successful fine-tune, absent real
  held-out data.
- **What success/failure would mean**: success = a real, measured
  reduction in the normalized-away rate on held-out real speech, with a
  bounded, explicitly-tolerated WER cost on clean speech (mirroring
  Kordt et al.'s own visible trade-off); failure = no meaningful
  reduction, or a reduction that only shows up on LibriStutter's
  synthetic data and does not generalize.
- **What changed tonight, specifically**: earlier today (Application-
  Objective Decision Analysis §7, written before Rank 1/2/3 were run),
  Stage D was judged **not yet justified** — two cheaper, untried options
  (Rank 1, Rank 2) existed and had to be tried first. Tonight, after
  Rank 1 (Failure), Rank 2 (Failure), and Rank 3 (Inconclusive, with the
  instability signature described in section A above), Stage D is now
  judged **the evidence-justified next step** — see "Final synthesis:
  Rank 1 + Rank 2 + Rank 3, and the Stage D decision" (line 7565) for the
  full, considered reasoning. This is a real, dated evolution of the same
  day's own earlier conclusion, stated explicitly as such, not a silent
  contradiction.
- **What this decision does NOT claim**: that Stage D *will* solve the
  problem (untested premise, see section B's UNRESOLVED/NOT TESTED
  entries), or that it is immediately actionable (the infrastructure gate
  above is unchanged and unresolved).

## F. Current research stopping point — 2026-08-07

**We are stopping experimental exploration here, deliberately, because
of what the evidence shows — not because of fatigue or an arbitrary time
limit.** Three independent, cheap, no-GPU, no-new-data representation-
side directions (full encoder embedding alone; full raw acoustic features
alone; both combined) have now been tried against the real
candidate-generation gap this track identified. Two came back clean
Failures against pre-registered deployability criteria; the third came
back Inconclusive with a specific, diagnostic instability signature
(higher mean, less stable, worse worst-case than a weaker baseline) that
is itself evidence of a small-sample ceiling, not an invitation to try a
fourth feature combination. Per the project owner's own explicit
instruction, **no Rank 4, no further classifier, no further threshold or
feature-engineering variant is proposed tonight** — the next real
question is resource/infrastructure feasibility for Stage D (and/or
identifying the single best architecture using entirely existing,
off-the-shelf systems, if Stage D's own infrastructure gate cannot be
cleared), not another round of frozen-representation experimentation.

## G. Decision for next session — supersedes the stale one above

**Superseded-in-place notice, 2026-08-08**: this plan (steps 1-10 below)
assumed Stage D was the right next focus. See "External Review
Reconciliation — 2026-08-08" at the end of this document for the current
plan, which inserts a cheaper, evidence-gap-closing step (a
Dysfluent-WFST trial) before any of steps 4-10 below. Steps 1-3 remain
useful groundwork and are not contradicted.

The "Decision for Next Session" section earlier in this document (line
6731) was written before Rank 1/2/3 existed and is now marked superseded
in place (not deleted). **This is the current plan:**

1. **Review today's final evidence** — this "Current Project State" section, plus the full "Rank 1 results" / "Rank 2 results" / "Rank 3 results" / "Final synthesis" sections it points to.
2. **Confirm the application-level target** — re-read the "PROJECT OBJECTIVE" section (line 18) and section D above before making any architecture decision; do not let Stage D's own technical interest override it.
3. **Review the already-costed Stage D design and requirements** — "Stage D planning" (3622), "Stage D Requirements / Re-entry Gate" (4296), and section E above — do not re-derive this from scratch.
4. **Decide the concrete Stage D implementation architecture** — confirm: CrisperWhisper vs. stock `whisper-large-v3` as the fine-tuning base; LoRA vs. full fine-tune; which continual-learning method.
5. **Identify the exact model to fine-tune/use** — a specific checkpoint, not a class of models.
6. **Identify the exact data path** — resolve FluencyBank Timestamped's CHAT-parser and access-terms question first (named, unbuilt, in "Stage D Requirements" §A), or name a concrete alternative.
7. **Identify the minimum resource/GPU requirement** — price a specific cloud GPU option against the "Compute requirements" figures in section E above; this needs the project owner's explicit budget approval before anything is provisioned.
8. **Define evaluation before implementation** — reuse this track's own established baselines and metrics (section E above), not a fresh, incomparable set.
9. **Pre-register the Stage D experiment** — exact architecture, objective, data split, metrics, success/failure criteria — in this same document, before writing any training code, exactly as every prior stage in this track has done.
10. **Only then, once the plan is agreed with the project owner, begin implementation.**

**Do not begin Stage D implementation, or any other new experiment,
without first completing steps 1-9 above and getting explicit
project-owner sign-off on cost/scope** — this is a resource-committing
decision (cloud GPU cost, data acquisition/access work), categorically
different from every prior stage in this track, which needed nothing
beyond local compute and already-available data.

## H. Which file should I share with collaborators?

**`ASR_RESEARCH_TRACK.md` is the primary, comprehensive research
document** — every experiment, every pre-registration, every literature
citation, every bug caught and fixed, every decision and its reasoning,
in full chronological detail, is here. Share this file (or this "Current
Project State" section specifically, for a fast orientation, with
pointers into the rest for depth) with anyone who needs to understand the
research itself.

`ROADMAP.md`, `PAPER_DECISION_LOG.md`, and `CHANGELOG.md` are supporting
**project-control** documents — `ROADMAP.md` tracks prioritized
implementation/research items with dated status updates and pointers
back into this document; `PAPER_DECISION_LOG.md` is the append-only
decision ledger (what was done / alternatives considered / why / measured
result) for every non-trivial choice; `CHANGELOG.md` is a fast-scan,
one-line-per-change index. All three point back into this document for
full detail rather than duplicating it — useful for tracking project
status at a glance, not a substitute for the research record itself.

## I. Consolidation report — what this pass did to the document

Per the project owner's explicit preservation rule, stated precisely:
- **Preserved unchanged**: every section that existed before tonight's
  session — the full "PROJECT OBJECTIVE" section, the entire "First-
  principles reassessment," "Integrative first-principles reassessment,"
  "Phase 2 of this research track," "Direction (g)" (protocol and
  results), "MFCC escalation," Stage B/C/C2, the "End-of-session
  handoff," "End of Today's Session," "Next Session Plan," "Research
  review (2026-08-07)," "Stage D planning (2026-08-07)," the "Final
  research audit," the "Comprehensive research-positioning analysis
  (deep pass)," the "Novelty-falsification pass," the "Application-
  Objective Decision Analysis," and the Rank 1/Rank 2/Rank 1+2-synthesis/
  Rank 3 sections written earlier this evening. None of these were
  shortened, summarized in place, or rewritten.
- **Received additions**: one short, clearly-marked superseded-notice was
  inserted at the top of the "Decision for Next Session" subsection
  (line ~6731) — its own original content is fully intact below that
  notice, unedited.
- **Revised because later evidence superseded an earlier conclusion**:
  none of tonight's edits altered any prior section's own text. The one
  real, dated revision this evening reflects — Stage D moving from "not
  yet justified" (Application-Objective Decision Analysis §7, written
  this morning) to "evidence-justified" (tonight's Final synthesis,
  written after Rank 1-3 ran) — was already recorded, at the time it
  happened, as a new dated section ("Final synthesis: Rank 1 + Rank 2 +
  Rank 3, and the Stage D decision") rather than an edit to §7's own
  text; this consolidation pass only cross-references that already-
  correctly-recorded evolution, it does not re-perform it.
- **Deleted**: nothing. No section, citation, number, or piece of
  reasoning was removed from this document during this consolidation
  pass.

---

# External Review Reconciliation — 2026-08-08

**What this section is.** On 2026-08-07 an independent, adversarial
external review (`EXTERNAL_REVIEW_2026-08-07.md`, commissioned by the
project owner) was received, arguing that the "Current Project State"
consolidation above reached its Stage D decision on an incomplete
literature map, and specifically that an off-the-shelf system
(Dysfluent-WFST) had been missed. Per the project owner's explicit
instruction, this section does **not** accept either document at face
value: every consequential claim in the external review was independently
re-verified against primary sources (the actual papers, the actual
GitHub repo, current cloud-GPU pricing) before being used to revise
anything here. Full verification detail (per-claim VERIFIED /
PARTIALLY VERIFIED / CONTRADICTS labels) lives in the session that
produced this section; this section records the reconciled conclusions
and the resulting decision, not the raw verification transcript.
Nothing above this section is edited or deleted — this is an additive,
dated correction, exactly this document's own established convention.

## 1. Verification outcome, by claim

**Confirmed accurate** (independently re-verified against primary
sources, not just re-quoted from the review):
- Dysfluent WFST (Guo, Lian, Zhou et al., Interspeech 2025,
  arXiv:2505.16351) is real, requires a **reference transcript** at
  inference (confirmed from the paper's own Algorithm 1 and its
  `data/gt_text/` input directory — "zero-shot" means no additional
  *training*, not no reference text), decodes audio against it via a
  WFST over a wav2vec2/WavLM-CTC emission lattice, and detects
  repetition/insertion/deletion as distinct path behaviors
  (revisit-state / skip-state / lower-index-state) at the
  **phoneme level** — i.e., sub-word, directly on-target for
  `sound_repetition`.
- The paper's own results: 100% repetition-detection accuracy is a
  **simulated-data, count-level** figure; the only **real-speech**
  evaluation is **nfvPPA (aphasia)**, not stuttering (PER 44.75% vs.
  10.71% simulated — a large sim-to-real gap on a *different*
  population than this project's target). The paper states directly
  that WFST decoding "lacks robustness" and is "highly sensitive" to
  emission noise (WPER 10.19% -> 29.77% at a modest synthetic noise
  level in their own ablation).
- This track's own prior characterization of StutterFuse (arXiv:2512.13632)
  was wrong (corrected in §5 above, 2026-08-08 annotation): it is a
  retrieval-augmented classifier, not a text/acoustic fusion model.
- Zayats et al. 2019's human fragment miss/insert rates (12.4%/7.2%),
  the CHI 2026 SEP-28k inter-annotator kappa-by-type figures (sound
  repetitions **0.40**, word repetitions 0.62, blocks 0.25,
  prolongations 0.11), Gulzar et al.'s confirmation that LibriStutter's
  ground truth is a non-verbatim `STUTTER` placeholder token (not the
  literal fragment content — Section II of that paper, read directly),
  and Shih et al.'s F1=0.554 word-level baseline are all confirmed
  accurate at the primary source.
- Current cloud-GPU economics confirm a `whisper-small`-scale LoRA pilot
  (Stage D's own minimum-viable scope, per section E above) costs
  **approximately $0 to ~$20** (Kaggle's free ~30h/week GPU quota,
  Colab's free tier, or ~$0.10-0.70/hr spot rental) — not a
  budget-gated commitment.
- Switchboard's MsState transcripts + the open
  `vickyzayats/switchboard_corrected_reannotated` reannotation genuinely
  mark sub-word fragments (trailing-hyphen convention, e.g. "you kn-")
  with real word-level time alignment, at ~260h scale — a real English
  data point this track had not surfaced.
- RQ-E (Phase 2's own decoding-entropy proposal) was indeed proposed and
  never run (confirmed by this document's own text, line ~745/825) —
  the "every cheap lever has been tried" language in section F above is
  literally false on this track's own record, independent of anything
  external.

**Confirmed overstated by the external review** (independently
re-verified, and found to disagree with the review's framing):
- **Dysfluent-WFST does not output timestamps.** Read directly: timing
  was a feature of the *prior* work (UDM), not this WFST decoder. The
  review's own architecture diagram (§1) shows "... + timings ->
  detect.py" flowing out of the WFST step — that is not supported by
  what the paper describes as output. Timing would need to be recovered
  as separate engineering from the underlying frame-aligned lattice, not
  assumed as a free capability.
- **"Off-the-shelf" overstates integration readiness.** The GitHub repo
  is real (genuine code, an inference notebook, a WPER metric utility,
  the `data/audio` + `data/gt_text` input structure) but its README is
  thin, with no confirmed install/dependency-pinning path verified.
- **SEP-28k-SW does not add sub-word fragment annotation** — confirmed
  directly from its own paper (arXiv:2602.10403): it only relabels
  whole-clip stutter-type categories (>25% of 2,621 clips adjusted).
  Listing it among datasets that close the fragment-annotation gap is
  wrong.
- **"The English data gap is not real" is too strong.** Switchboard
  MsState's underlying audio/transcripts remain LDC-license-gated
  (unverified whether this project has institutional access), and it is
  general conversational-repair disfluency, not clinical stuttering — a
  real, unresolved distribution-transfer risk for a fine-tune targeting
  stuttering specifically. Boli (2.8h, 28 speakers) and FluencyBank
  Timestamped (5.3h, 37 speakers) are real, open, and stuttering-specific,
  but combined (~8h) sit well under Kordt et al.'s own 10-20h/corpus
  precedent. The corrected framing: **the gap is narrower than this
  track previously believed, but it is not closed.**
- **Shih et al.'s F1=0.554** is a general word-level stuttering F1 across
  five combined disfluency types on a smaller, separately-curated
  dataset — not a `sound_repetition`-specific number. Using it as a
  direct ceiling comparison for this track's own `sound_repetition` work
  overstates comparability; it remains a legitimate general field
  baseline this project has never benchmarked against, just not a
  narrowly-scoped one.

## 2. What this changes

- **The exhaustion premise behind "Stage D is the evidence-justified next
  step" (section E/F above) is false.** A cheaper, never-tried,
  better-matched mechanism class (reference-text-conditioned phonetic
  realignment — Dysfluent-WFST or equivalent) exists and was not
  considered before that decision was made. Per this track's own
  standing "cheapest first" discipline (already applied to every prior
  stage), that mechanism must be tried before Stage D, not after.
- **Stage D's status changes from "evidence-justified next step" to
  "premature, not rejected."** Nothing about Stage D's own design (L2
  scope, LoRA on an existing checkpoint, per section E above) was found
  to be wrong — only its *sequencing* relative to a cheaper untried
  alternative.
- **The "bottleneck is quantity of labeled examples" conclusion (Final
  synthesis, section A of the Current Project State) needs revising to
  "quantity *and* annotation reliability."** The now-confirmed
  `sound_repetition` inter-annotator kappa=0.40 means trained human
  experts agree only "fair" on this category — Ranks 1-3's F1=0.17-0.25
  results sit closer to a real, human-annotation-noise-bounded ceiling
  than the original framing implied, independent of representation
  choice or sample size.
- **Ranks 1-3 remain valid, unchanged negative/inconclusive results
  about one specific mechanism class** — post-hoc classification over
  frozen, general-purpose representations. They say nothing about a
  categorically different mechanism (reference-text-conditioned
  phonetic re-decoding) that was never tested. This is not a reason to
  discard Ranks 1-3; it is a reason not to treat them as having closed
  off that different mechanism class too.
- **The Stage D "budget approval" gate (section E above, "Compute
  required") should be removed** — confirmed near-zero actual cost, not
  a categorically different resource commitment from this track's other
  CPU-only experiments.

## 3. Revised decision tree (supersedes section G's 10-step plan above
for near-term sequencing; that plan's steps 1-3 remain useful groundwork)

**Superseded-in-place notice, 2026-08-08 (round 2)**: a second review
pass (section 5 below) refined several of the specifics in this tree
(real-audio acquisition moved earlier and made parallel rather than
gated behind two failures; the timestamp-recovery assumption in step 2
downgraded from "decide how" to "budget as a real, separately-scoped
sub-task of unverified size"; a new Libri-Dys diagnostic step added; the
Stage D data strategy in step 5 revised). The tree below remains a
correct *first-pass* plan and is not wrong, only superseded by the more
complete one in section 5.

1. **Today, zero compute.** Recompute PR/AUPRC curves and clip-level
   bootstrap CIs for Ranks 1-3 from already-saved predictions. Separately
   measure the actual end-to-end Track B detection-F1 effect of
   deploying Rank 1 as a candidate generator (never yet computed, despite
   being the single most product-relevant number available). Either
   could already change a Failure/Inconclusive verdict.
2. **If step 1 does not resolve the gap to a shippable bar**: clone
   `Berkeley-Speech-Group/DysfluentWFST`, pre-register an experiment
   exactly as every prior stage in this track has, and run it against
   the existing 120-clip real-ASR (Track B) sample, using
   CrisperWhisper's own clean transcript as the required reference text.
   Score against the *existing* baselines (Stage A's 45.2%/40.5%, Rank
   1's F1=0.230) — not a fresh, incomparable metric set. Explicitly test
   robustness on this project's own actual audio quality, given the
   paper's own documented noise sensitivity, not just on clean input.
   Explicitly decide and pre-register how (if at all) approximate timing
   will be recovered, since the base system does not provide it.
3. **If Dysfluent-WFST transfers** (meets a pre-registered bar: stable,
   bootstrapped-CI-supported precision/recall improvement over Rank 1 on
   real ASR output): integrate as a third fusion signal alongside the
   existing encoder-distance corroboration, matching `detect.py`'s
   existing block/prolongation fusion pattern. Stop — Stage D is not
   needed for this gap.
4. **If Dysfluent-WFST does not transfer** (fails on real audio quality,
   or its only-real-validation-is-aphasia limitation turns out to apply
   to stuttering too): acquire Boli + FluencyBank Timestamped (~8h
   combined, real, open, stuttering-specific) as a real-audio
   generalization check for whichever signal (Rank 1, the WFST arm, or
   both) looked most promising, before any fine-tuning commitment.
5. **Only after 1-4**: Stage D becomes the evidence-justified next step,
   using the same L2-scoped design as section E above, Switchboard
   MsState (LDC access permitting) and/or Boli/FluencyBank as training
   data, at the now-confirmed near-zero compute cost — no budget
   approval gate required.

## 4. What is unchanged

The two-stage ASR + fusion-detector architecture
(`PHASE_3_ARCHITECTURE_REVIEW.md`), the L2-vs-L4 scoping of Stage D
(section E above), Stage A's core finding, and this track's own process
discipline (pre-registration, rule 3 auditing, superseded-not-deleted)
are all unaffected by this reconciliation — nothing found during
verification argues against any of them.

**No production code has changed as a result of this reconciliation.**
This section is analysis and decision-state only, exactly as every prior
stage in this track has been before code was written.

---

## 5. Second-pass reconciliation — 2026-08-08 (round 2)

**What prompted this round.** After section 1-4 above were written, a
second AI reviewer (external to this repo) independently re-checked this
reconciliation, conceded most of it, pushed back on one specific factual
claim (SEP-28k-SW's granularity), made one unverified technical claim
(Dysfluent-WFST timestamp recovery being "an afternoon of work"), and
proposed three design refinements. None of the pushback was accepted at
face value — each checkable claim was independently re-verified against
primary sources before being folded in here, same discipline as round 1.
Separately, the project owner supplied the full text of the paper this
track has only ever cited secondhand as "Kordt et al. [B11]" —
Kordt, Pekarek Rosin, Lee & Wermter, "Learning to Hear Hesitation:
Continual Learning for Disfluency-Aware ASR" (arXiv:2606.14391) — now
read directly for the first time.

### 5.1 Conceded without further verification needed

The second reviewer's self-corrections matched what section 1-2 above
already concluded independently (timestamps not free, "off-the-shelf"
overstated, Shih et al.'s F1 not sound-repetition-specific, the English
data gap being narrower-not-absent) — no new information, cross-checked
and consistent.

### 5.2 A real correction to round 1's own SEP-28k-SW verdict

Round 1's dataset-verification pass (drawing only on the CHI 2026
paper's prose) concluded Sep-28k-SW "does NOT add sub-word fragment
annotation... only relabels whole-clip stutter-type categories." **This
was wrong**, caught only because the second reviewer flagged a
table-vs-prose discrepancy in the same CHI paper and this was chased
down to its actual primary source (Sridhar & Wu, Interspeech 2025, read
in full, not summarized). The real situation: Sep-28k-SW is **both**
things at once — a clip-level relabeling (the >25%-of-2,621-clips figure,
which the CHI paper's prose describes) **and** a set of manually-produced
verbatim transcripts with inline sub-word dysfluency markup (`/b` block,
`/p` prolongation, `/i` interjection, `/r` sound repetition, `[]` word
repetition — e.g. `"o/pne"`, `"a/rll"`, `"It[It]"`), which the CHI paper's
own Table 1 correctly records and its prose simply omits. **SEP-28k
itself has no transcripts at all**; Sridhar & Wu produced these from
scratch. This is real, English, clinical-stuttering (PWS podcast
speech), sub-word-annotated data using almost exactly this project's own
five-type taxonomy including the sound-repetition-vs-word-repetition
distinction that is the actual target gap — at ~2,621 clips (roughly 2h
at SEP-28k's ~3s/clip format), smaller than Boli+FluencyBank combined but
plausibly the best taxonomy-match found in this entire exercise. **Not
yet confirmed**: whether the annotations are publicly downloadable (a
GitHub repo, `aimpowered/stuttered-speech-benchmark`, was referenced in
Sridhar & Wu's paper but its public availability was not independently
checked) — this is the next thing to verify, not to assume, before
counting on it.

This error is recorded here, not smoothed over, because it demonstrates
exactly the failure mode this whole reconciliation exists to guard
against: a verification pass that reads a source's prose but not its own
table, or that stops at a paper's *citation* of another paper instead of
fetching that paper directly, will reproduce the same kind of
incompleteness this track is correcting the original decision for.

### 5.3 Dysfluent-WFST timestamps: structurally plausible, "an afternoon" is unverified optimism

Independently re-checked, including the actual repo code (`utils/decoder.py`),
not just the paper. Confirmed true: the emission is a genuine per-frame
CTC posterior lattice (`Ξx ∈ R^(T×C)`), and k2's `DenseFsaVec` does
preserve frame correspondence as a structural property — the second
reviewer's underlying premise is correct, not a guess. But the shipped
decoder (`_decode_lattice`) extracts only `shortest[0].aux_labels` — the
reference-phoneme-sequence axis — and never touches `shortest[0].labels`,
the frame-synchronous axis that would carry timing; recovering it means
re-deriving alignment from a track the current pipeline actively
discards. The decoding graph is also not a plain CTC lattice — it is
`ctc_topo` composed with a reference FSA that injects extra
epsilon-labeled `<trans>` arcs for repetition/insertion/deletion
annotation, which do not consume a frame, making path-position-to-frame
mapping materially harder than standard forced alignment. The closest
official precedent (`icefall`'s own "k2-based forced alignment" doc page)
is an empty stub, not a documented recipe. **Verdict: timing recovery is
real, structurally plausible engineering, not a fundamental blocker — but
"an afternoon" is an unverified, likely-optimistic estimate with no
existing recipe to adapt.** Any pre-registration of a Dysfluent-WFST
experiment must budget this as its own separately-scoped, effort-uncapped
sub-task, with a named fallback (using CrisperWhisper's own existing
word-level timestamps as a coarser substitute) if it proves large.

### 5.4 The new Kordt et al. paper (arXiv:2606.14391) — primary source for `[B11]`, read directly for the first time

This is the actual paper behind this track's `[B11]` citation, previously
only known secondhand. Its general shape was already cited accurately
(TalkBank CHAT-format data, EWC/Experience-Replay/A-GEM/Weight-Averaging,
~10-20h/corpus scale, 10 epochs / batch 16 / lr 2e-5 — all confirmed
exact matches). One real correction and several new, decision-relevant
findings:

- **Correction**: section E above describes Stage D as "most plausibly
  LoRA-style parameter-efficient adaptation (matching StammerTalk's own
  real, published recipe) combined with a continual-learning method...
  per Kordt et al." This conflates two techniques that have never been
  combined by anyone. Kordt et al. do **full-parameter** fine-tuning of
  `whisper-small.en` — no LoRA. LoRA is StammerTalk's technique, on a
  different (Mandarin, AS-70) checkpoint and dataset. **If Stage D
  combines LoRA with a continual-learning method, that combination is
  this project's own untested step, not a replication of an existing
  recipe** — must be stated as such, not implied as already-proven.
- **A real, causally-confirmed marker-learning-vs-general-ASR trade-off**:
  Weight Averaging gets the best clean-speech retention but *completely
  fails to emit disfluency markers* (F1=0.00); the methods that reliably
  emit markers (FT/A-GEM/ER, F1=0.73-0.75) pay a measurably larger pWER
  cost. This is a hard Pareto frontier, not a minor caveat, for any
  Stage D design that needs both properties at once.
- **A transferable mechanistic-verification method**: marker emission
  concentrates in a small, consistent set of decoder cross-attention
  heads across every CL method tested; ablating the top-5 heads removes
  ~57-62% of FILLER markers while barely moving pWER (+0.42-0.44%) — a
  real causal check, not a correlation. Worth adopting in any future
  Stage D evaluation to verify a genuine learned mechanism rather than
  trusting an aggregate F1 alone.
- **A third, independent population-mismatch data point**: SME (L2
  English learners), Pitt (dementia/Alzheimer's), and Delaware (mild
  cognitive impairment) are none of them clinical stuttering — general-
  or cognitive-impairment-related disfluency, the same category of
  mismatch already found in Switchboard (conversational repair) and
  Buckeye (ordinary cutoffs). This independently reinforces (from a
  completely separate primary source) the train-on-general-disfluency /
  evaluate-on-real-stuttering split named in 5.5 below.
- **Even this closest real precedent doesn't test this project's actual
  taxonomy granularity**: its REP token collapses word- and phoneme-level
  repetition into one marker type — no sound-repetition-vs-word-
  repetition split, same gap found in every other source this track has
  checked. A real Stage D attempt at that specific split would be
  testing something nobody has published, not adapting a proven method.
- **Marker-F1 is bag-of-words**, not position-localized — not directly
  comparable to this project's own event-localization metric; any future
  Stage D evaluation should reuse this project's own localization
  approach, not adopt this metric as-is.
- **Compute confirmed even cheaper than assumed**: full fine-tune of
  `whisper-small` (244M params, no LoRA needed) on ~34h combined data, 10
  epochs — a smaller, cheaper, and now fully-precedented pilot scope than
  the LoRA-on-CrisperWhisper design section E assumed.

### 5.5 Revised data strategy, incorporating 5.2 and 5.4

Train/evaluate role split (independently motivated by both the second
reviewer's Buckeye argument and Kordt et al.'s own real population
mismatch, section 5.4): use abundant, open, general-disfluency corpora
(Buckeye — free, real, ~300k words, literal `<CUTOFF>` fragment-content
tags with phone alignment; Switchboard MsState if institutional LDC
access exists; TalkBank-style CL adaptation per Kordt et al.'s own proven
recipe) for the "don't delete disfluency evidence" (L2) skill itself, and
reserve the small, real, stuttering-specific sets — Boli (2.8h) +
FluencyBank Timestamped (5.3h) + Sep-28k-SW (~2h, pending public-
availability confirmation, section 5.2) — for evaluation and final
calibration, not primary training volume. This resolves the "8h is under
Kordt's 10-20h/corpus precedent" objection: 8-10h is thin for training,
adequate for evaluation.

### 5.6 A new, free diagnostic for Rank 1's own open question

Section B of the Current Project State lists as UNRESOLVED whether
Rank 1's low recall (0.147) reflects a genuine encoder-representation
ceiling or is an artifact of too few positive examples (66). Libri-Dys
(from the SSDM line of work, ~3,983h, public, no licensing, ~200x
LibriStutter's scale) is large enough to re-run Rank 1 at roughly 10x
its original sample size. This settles a currently-open question at zero
new compute and no data-acquisition cost — it is a diagnostic for an
existing unresolved question, not a claim that Libri-Dys's synthetic
data closes the real-data gap (it does not; section 2 above already
established Libri-Dys/VCTK-Stutter remain synthetic regardless of scale).

### 5.7 A limitation that applies to every LibriStutter-scored result in this track

LibriStutter's ground truth is synthetic and therefore has, by
construction, zero annotation noise — unlike SEP-28k's real human
annotations (kappa=0.40 for `sound_repetition`, section 1 above). This
means **every precision/recall figure this track has produced on
LibriStutter (direction (g), the MFCC escalation, the combined
classifier, Ranks 1-3) is optimistic relative to any real deployment** —
the gap between LibriStutter numbers and real-audio numbers should be
expected to exceed what sampling variation alone predicts. This belongs
in this track's own limitations framing going forward, and in any future
paper draft, alongside the kappa=0.40 finding it compounds.

### 5.8 Revised decision tree (supersedes section 3 above for near-term sequencing)

1. **Today, zero compute, and in parallel (not sequential)**: (a)
   recompute PR/AUPRC + bootstrap CIs for Ranks 1-3 from already-saved
   predictions, and measure the end-to-end Track B effect of Rank 1
   (unchanged from section 3); (b) acquire the small real-audio set now,
   since it is cheap and fast and should not gate behind two experiment
   failures — download Boli (CC-BY-4.0) and FluencyBank Timestamped
   (CSV, no CHAT parser needed) today, and separately verify whether
   Sep-28k-SW's annotations (section 5.2) are actually publicly
   downloadable; (c) re-run Rank 1 at Libri-Dys's ~10x-larger scale
   (section 5.6) to resolve its own open representation-ceiling-vs-
   sample-size question, at zero new cost.
2. **Pre-register and run Dysfluent-WFST** against the existing 120-clip
   Track B sample **and**, in the same pre-registration, against
   whichever real-audio set from step 1(b) is confirmed available —
   two explicit audio conditions (clean/synthetic vs. real), scored
   separately, per the paper's own documented noise sensitivity. Budget
   timestamp recovery (section 5.3) as its own separately-scoped,
   effort-uncapped sub-task with a named fallback (CrisperWhisper's
   existing word-level timestamps as a coarser substitute) rather than
   assuming it is quick.
3. **If it transfers** (stable, bootstrapped-CI-supported improvement
   over Rank 1, holding up on real audio, not just synthetic): integrate
   as a third fusion signal. Stop — Stage D not needed for this gap.
4. **If it does not transfer**: Stage D becomes the evidence-justified
   next step, using the revised data strategy (5.5) and an explicit,
   honest choice between LoRA (StammerTalk's proven recipe) and full
   fine-tune + continual learning (Kordt et al.'s proven recipe, now
   confirmed even cheaper) rather than assuming their untested
   combination — adopting Kordt et al.'s cross-attention head-ablation
   method (5.4) as the mechanistic-verification check regardless of
   which is chosen, and explicitly pre-registering the sound-repetition-
   vs-word-repetition granularity split as its own open research question,
   since no source found anywhere in this track has demonstrated it.

### 5.9 What is still unchanged

Everything in section 4 above remains true after this round. Nothing in
5.1-5.8 argues against the two-stage architecture, the L2 scope, Stage
A's finding, or this track's process discipline — it only sharpens the
data strategy, corrects one dataset verdict, calibrates one technical
estimate, and adds one free diagnostic and one limitations note.

**No production code has changed as a result of this round either.**

---

## 6. Third-pass reconciliation — 2026-08-08 (round 3)

**What prompted this round.** The second reviewer's own follow-up
accepted round 2's corrections (including the one made against its own
earlier claim) and raised five further points: a licensing constraint on
the newly-found Sep-28k-SW data; a concrete, checkable way to settle the
WFST timestamp question rather than arguing it further from reading; a
risk this project should name explicitly in Stage D's pre-registration;
a proposed new experiment (probing CrisperWhisper's own cross-attention);
and an apparent contradiction between two marker-F1 figures cited in
round 2. Each was independently checked before being accepted — including
one, the Sep-28k license, checked against the dataset's own README
directly, not a search snippet.

### 6.1 Sep-28k-SW's licence is a real, verified constraint — VERIFIED

Confirmed directly from `apple/ml-stuttering-events-dataset`'s own
README: "The SEP-28k dataset is licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)."
Sep-28k-SW (section 5.2 above) is an annotation layer over Sep-28k's own
audio and inherits this restriction. Per this project's own two-repo
commercial-product context (`audio-mod` is one half of a shipped
product, not a research-only artifact), this is not a footnote — **Sep-
28k-SW may be used to evaluate a model, but must not be used to train,
fine-tune, or derive features for any model shipped in the product
without separately negotiated permission.** This resolves cleanly into
the train/evaluate split already adopted in 5.5: Sep-28k-SW joins Boli
and FluencyBank Timestamped as an evaluation-only real-stuttering set;
Buckeye (free, no such restriction) and any accessible Switchboard/
TalkBank-style data remain the training-side candidates. Still open,
unchanged from 5.2: whether Sep-28k-SW's annotations are actually
publicly hosted (the referenced `aimpowered/stuttered-speech-benchmark`
repo was not independently confirmed).

### 6.2 WFST timestamps: the question is not resolvable by more reading — confirmed by this round's own attempt to resolve it

Round 2 (section 5.3) found the shipped decoder keeps only the
reference-sequence axis (`aux_labels`), not the frame-synchronous axis
(`labels`), and characterized the special repetition/insertion/deletion
transition arcs as "epsilon-labeled." This round re-fetched the same
code specifically to check whether those arcs use a true FSA epsilon
(free transition, does not consume an input frame) or the numeric label
`0` doing double duty as the CTC blank phone (a real, frame-consuming
symbol) — because if it's the latter, paths through the lattice may in
fact be frame-synchronous after all, and timing would be a much smaller
mapping problem than round 2 implied. The result: the arcs are
confirmed, directly from the code, to carry input label `0` literally —
but **this round's own reading disagreed with round 2's reading about
what that means** (round 2 called it "epsilon"; this round's fetch
initially called it "the blank symbol in CTC," a real frame-consuming
label, before its own summary contradicted itself about whether the
system is therefore frame-synchronous). Two independent reading passes
producing two different characterizations of the same code is itself the
finding: **this is not resolvable by further static reading, and no more
of it should be attempted.** The correct next step is the concrete
runtime check the second reviewer proposed — comparing
`len(shortest[0].labels)` against the true frame count `T` for a known
clip — which settles the question in minutes rather than arguably
indefinitely. This is now folded into the Dysfluent-WFST pre-registration
(6.5 below) as its mandatory first step, before any evaluation run, not
executed in this documentation-only session.

### 6.3 Reconciling the two Kordt et al. marker-F1 figures — resolved directly from the primary text already in hand, no further fetch needed

Round 2 cited two REP-related figures without cross-referencing them:
"FT/A-GEM/ER, F1=0.73-0.75" (used to illustrate the marker-learning/
pWER trade-off) and, separately, "REP markers... ranging from 0.35 to
0.63" (used to illustrate per-type difficulty). These are **not the same
measurement and are not in tension** — re-reading the paper's own text
resolves this precisely:
- **0.73-0.75 is Table 1's micro-F1**, combining FILLER+REP together
  (the only two marker types SME predominantly contains), computed only
  on the SME dataset, at the initial disfluency-token-introduction
  checkpoint — *before* any further continual adaptation.
- **0.35-0.63 is Table 4's REP-specific column**, computed on the
  combination of all three datasets, *after* sequential continual
  adaptation on Pitt and Delaware (which also introduces two further
  marker types, DISRUPT and PAUSE, that the model must learn without
  forgetting FILLER/REP).
These are different scopes (combined-2-type vs. per-type-of-4), different
data (SME-only vs. all three combined), and different training stages
(initial introduction vs. after further sequential adaptation) — a stage
progression, not a contradiction. The corrected, more precise reading is
itself decision-relevant and slightly *strengthens* (not weakens) this
track's point: initial marker introduction looks easy (0.73-0.75), but
**sustained retention of that same marker type through continual
adaptation on more difficult, differently-distributed data is
measurably harder and highly method-dependent (0.35-0.63)** — a real risk
for any Stage D design that expects one round of marker training to be
durable.

### 6.4 Two new items accepted for the queue

- **A CrisperWhisper cross-attention probe**, using Kordt et al.'s own
  head-attribution method (Section 2.3 of that paper): check whether
  CrisperWhisper's *existing*, already-trained decoder has cross-
  attention heads that attend to a disfluent span's audio even when the
  decoded text drops it, and whether ablating/inspecting those heads
  shows the information was there but overruled downstream. This is
  cheap (no training, no new data), directly tests this track's own
  central decoder-vs-encoder mechanistic inference — currently resting
  only on the `num_beams` experiment, a weaker proxy that could only ever
  produce a null result — and gives that inference its first real
  positive-evidence test. Added to the decision tree (6.5) between the
  zero-compute reanalysis and the WFST run.
- **A power analysis before any Stage D commitment.** The assembled
  real-stuttering evaluation data (Boli 2.8h + FluencyBank 5.3h +
  Sep-28k-SW ~2h, evaluation-only per 6.1) totals roughly 10h with a few
  hundred `sound_repetition`-type events, against a documented human
  inter-annotator kappa of 0.40 for that type (section 1 above). Small n
  combined with substantial label noise can produce a confidence interval
  on any Stage D success/failure verdict wide enough to swallow the
  effect being measured — pre-registering a criterion the available data
  cannot resolve would waste a fine-tune. **Before committing to Stage D,
  compute the minimum detectable effect on the assembled real evaluation
  set(s), accounting for annotation-noise-inflated effective variance,
  and confirm the planned success/failure gate is resolvable at that
  power.** If it is not, decide in advance — not after seeing results —
  whether to acquire more evaluation data or fall back to a coarser,
  more measurable endpoint (the L2-level deletion-rate reduction already
  named in section E, rather than event-level P/R).

### 6.5 Revised decision tree (supersedes section 5.8 above)

1. **Step 0, today, zero compute, fully parallel** (not sequential with
   anything else): (a) recompute PR/AUPRC + bootstrap CIs for Ranks 1-3
   from already-saved predictions, and measure the end-to-end Track B
   effect of Rank 1 (unchanged); (b) acquire the real-audio evaluation
   set now — Boli (CC-BY-4.0), FluencyBank Timestamped (CSV), and verify
   Sep-28k-SW's actual public availability (evaluation-only, per 6.1's
   licence finding); (c) **rescore Rank 1's and Rank 2's already-trained,
   frozen classifiers on this real audio** — pure inference, no training,
   no GPU needed beyond what's already used — to directly settle this
   track's single largest UNRESOLVED item (does any signal generalize
   past LibriStutter's synthetic splices) without waiting on any other
   experiment's outcome; (d) re-run Rank 1 at Libri-Dys's ~10x-larger
   scale to resolve its own representation-ceiling-vs-sample-size
   question, at zero new cost.
2. **CrisperWhisper cross-attention probe** (6.4): cheap, no training,
   direct test of the decoder-vs-encoder mechanism this track has relied
   on inferring indirectly since Stage B/C.
3. **Pre-register and run Dysfluent-WFST**, with the blank-vs-epsilon
   frame-synchronicity check (6.2) as its mandatory first step, against
   both the existing 120-clip Track B sample and the real-audio set from
   step 1(b), as two explicit, separately-scored conditions.
4. **If it transfers**: integrate as a third fusion signal. Stop — Stage
   D not needed for this gap.
5. **If it does not transfer**: before pre-registering Stage D, run the
   power analysis (6.4) on the assembled real evaluation data and confirm
   the planned success/failure gate is actually resolvable at that power.
   Only then pre-register Stage D, using the revised data strategy (5.5,
   6.1) and an explicit, honest choice between LoRA and full-fine-tune-
   plus-continual-learning (5.4) rather than assuming their untested
   combination, adopting Kordt et al.'s head-ablation method as the
   mechanistic-verification check, and naming the LoRA+CL combination
   itself (if attempted) as this project's own untested methodological
   step, not a replication of a published recipe.

### 6.6 What this round changes in the overall assessment

Nothing here changes the round-2 classification (Stage D premature, not
rejected) or the two-stage architecture. It sharpens three things: a real
licensing constraint on the best-matched dataset found so far, an honest
"unresolved, needs a runtime check" status (rather than a confident
either-direction claim) on WFST timing, and two new queued items (the
CrisperWhisper probe, the power analysis) that make the remaining path to
Stage D — if it is ever reached — actually decidable rather than
optimistically scoped.

**No production code has changed as a result of this round.**

---

# Final Decision-Oriented Reconciliation — 2026-08-08

**What this section is, and is not.** Rounds 1-3 above (the "External
Review Reconciliation" and its two follow-ups) verified individual
claims against primary sources. This section does something different,
per the project owner's explicit instruction: it treats rounds 1-3
themselves, this project's own prior conclusions, and the external
review's own framing all as **inputs to be weighed, not conclusions to
be inherited** — re-anchored purely to the application objective stated
in `CLAUDE.md`'s "The objective, stated precisely" section, restated
here because it governs everything below: *reliably extract the actual
speech problems occurring in a speaker's audio, so they can be
localized, classified, and used downstream — not prove novelty, not
build an ASR for its own sake.* Nothing above this section is edited or
superseded in substance; several of its decision-tree recommendations
are superseded here because a cheaper, better-fitting option was found
by applying the objective more strictly than the sequencing logic in
rounds 1-3 did.

## 1. Re-anchoring: what actually still matters after three rounds of verification

Stripped of research-momentum framing, exactly two real, quantified gaps
remain in this project's own pipeline, and nothing else:

- **`sound_repetition` candidate generation from real ASR output** — 45.2%
  of ground-truth instances lost even at "correctly transcribed"
  positions (Stage A), zero candidates generated across the entire
  120-clip Track B real-ASR sample (item 19/20), and three independent
  frozen-representation classifier attempts (Ranks 1-3, see section 3
  below) each found real-but-insufficient signal.
- **`word_repetition` candidate generation from real ASR output** — 40.5%
  lost at correctly-transcribed positions, but by a **different
  mechanism**: Stage A's own hand-check (22/23 cases) found the loss is
  not a missing fragment but the *entire other half of the repeated
  pair* being deleted or displaced by the ASR.

Everything else in the taxonomy — `prolongation`, `block`, `interjection`
— is **already solved and shipped** (`PHASE_3_ARCHITECTURE_REVIEW.md`,
Application-Objective Decision Analysis §2) and untouched by any of this
research track. No architecture comparison below revisits these; they
are not part of the remaining gap.

## 2. Architecture comparison, scored against the application's actual requirement

| | **A. Current (CrisperWhisper + shipped detectors)** | **B. CrisperWhisper + Dysfluent-WFST** | **C. Existing stutter-aware ASR + downstream detection** | **D. Acoustic/audio-native detector + ASR fusion (YOLO-Stutter/SSDM-class, or this track's own Rank 2/direction-(g))** | **E. Stage D (fine-tune existing ASR)** | **F. Other existing mechanisms found this session** |
|---|---|---|---|---|---|---|
| Information gained | Clean transcript + word timing (shipped, good); `block`/`prolongation`/`interjection` fully solved | Phoneme-level repetition/insertion/deletion labels at exactly the normalized-away positions | N/A for English — **does not currently exist** (see below) | Audio-native region + type, no transcript dependency for detection itself | Whatever the fine-tune learns to preserve, in principle across the whole taxonomy | Alignment-gap: an unexplained duration/silence residual where ASR output doesn't match audio timing |
| Information lost | `sound_repetition`/`word_repetition` surface evidence (Stage A, quantified) | No timestamps as shipped (verified, code-level); real-world validation is aphasia only | — | Historically weak precision at this project's own data scale (Rank 2: F1=0.163) | Unknown until tried; general-ASR WER at risk (mitigated by CL, per Kordt et al.) | Only targets `word_repetition`'s specific loss mechanism, not `sound_repetition`'s |
| Detects `sound_repetition`? | **No** (0 real candidates, Track B) | Yes, by design — but only demonstrated on simulated data + aphasia, never stuttering | N/A | Yes, by taxonomy, but this project's own CPU-feasible approximation (Rank 2) fell short of a usable bar | Untested at this taxonomy granularity by anyone | No (different mechanism, targets `word_repetition`) |
| Detects `word_repetition`? | Partially, candidate-starved (~7x below ground-truth rate) | Plausible, not specifically demonstrated (paper's own example is sound-level) | N/A | Yes, by taxonomy, untested on this project's own data | Untested at this granularity | **Yes, by design — directly targets the diagnosed "other half deleted" mechanism** |
| Prolongation / block / interjection | **Already solved, shipped** | Not the design target | — | Would duplicate, not clearly improve, the existing solution | Could in principle (Kordt et al.'s FILLER/DISRUPT/PAUSE tokens: F1 0.68-0.75) but redundant with what's shipped | N/A |
| Timing/localization | Yes, word-level, already good | **Unresolved** — genuinely unresolved by two rounds of code reading (round 3, §6.2); needs a runtime check | N/A | Yes, natively frame/region-level | Needs redesign — Kordt et al.'s own marker-F1 is bag-of-words, not localized | Yes, this *is* a timing-based signal by construction |
| Intended vs. actual disfluent realization | Intended content usually recoverable already (Stage A); actual realization lost for the two gap types | Structurally distinguishes them (reference text = intended, decode path = actual) — the right shape in principle | N/A | Detection only; still needs ASR for intended content (already this project's own fusion shape) | Design intent (L2 scope), unproven at this granularity | Detection only, same as D |
| Downstream usefulness | High for 3/5 types, near-zero for the 2 real gaps | Potentially high for `sound_repetition` *if* it transfers and timing resolves — two open ifs | None — not an available option today | Real but historically insufficient signal at this project's data scale | Most complete *if* successful; longest, least certain timeline | Directly closes a specifically-diagnosed, real mechanism at low cost |
| Data required | None (already shipped) | None to test (zero-shot); reference text already produced by existing pipeline | N/A | None to test a pretrained checkpoint (if one exists — **unconfirmed**, not yet checked); real training data for retraining | Buckeye (free) for volume + Boli/FluencyBank/Sep-28k-SW (eval-only, licence-gated) for real-stuttering validation | **None new — reuses CrisperWhisper's own already-computed word timestamps** |
| Compute required | Already running in production | Plausibly CPU-feasible for inference (not confirmed by the paper itself); new dependencies (k2, wav2vec2/WavLM-CTC, a grapheme-to-phoneme step) | N/A | Unconfirmed for a pretrained checkpoint; GPU + real training data to build one from scratch | Confirmed near-zero (Kaggle/Colab/spot rental) | **Zero new compute — an analysis over already-computed alignments** |
| Testable on current machine? | Yes, already is | Plausible, unconfirmed | N/A | Unconfirmed (checkpoint availability unknown) | Inference yes; training needs cloud (cheap, confirmed) | **Yes, today, on the existing 120-clip Track B sample, no new dependencies** |
| Biggest uncertainty | None new — this is the precisely-characterized status quo | Whether it transfers from aphasia/clean-audio validation to real, non-clean stuttering audio — a real, undemonstrated, and non-trivial generalization gap, compounded by the paper's own documented severe noise sensitivity | Whether anyone will publish one — not this project's decision to make | Whether a pretrained checkpoint is even publicly available (not yet checked) | Whether the specific sound-vs-word repetition granularity split is learnable at all at the available real-data scale (~10h, kappa=0.40) — untested by anyone anywhere | Whether the 74.13%-coverage precedent (a *different* task — general untranscribed-word recovery) actually transfers to `word_repetition` specifically — real but comparatively small uncertainty |

**Column C does not currently exist as an available option.** No English,
clinical-stuttering ASR checkpoint at this taxonomy's granularity was
found anywhere in three rounds of literature verification — StammerTalk
(the closest real precedent) is Mandarin-only. For English, "C" and "E"
(Stage D) are the same option: if it's wanted, this project would have
to build it.

## 3. Rank 1 / Rank 2 / Rank 3, reconciled against everything found since

**What they actually established** (unchanged, still trustworthy —
measured, bug-audited, cross-validated): post-hoc classification over
*frozen, general-purpose representations* — the full CrisperWhisper
encoder embedding alone (Rank 1), full raw acoustic MFCC statistics
alone (Rank 2), and both concatenated (Rank 3) — does not reach a
pre-registered deployable operating point for `sound_repetition`
candidate generation, at this project's available real-ASR sample scale
(62-245 positives depending on population). This is a real, three-way
convergent negative/inconclusive result about **one specific mechanism
class**, not a general claim.

**What they did not establish, and must not be read as establishing**:
that no method can solve this problem; that representation *choice*
specifically (as opposed to *mechanism*) is the exhausted axis; that a
categorically different mechanism — reference-text-conditioned phonetic
re-decoding (Dysfluent-WFST), or alignment-gap/duration-residual
detection (never tested by Ranks 1-3, which only ever scored spans
against a similarity/distance threshold, never against timing-residual
signal) — would fail too. Neither was tested by Ranks 1-3, and both
target `sound_repetition`/`word_repetition` through a mechanism outside
what "frozen representation + classifier" covers.

**What must now be withdrawn or narrowed**, dated and reasoned, not
silently corrected: the "every cheap, no-GPU, no-new-data direction has
been tried" / exhaustion framing (Current Project State §F, 2026-08-07)
is **withdrawn** — it was already self-contradicted by this track's own
record (RQ-E named, never run) before the external review even arrived,
and is further narrowed by the existence of at least two more cheap,
untried, differently-mechanistic candidates (below). The "bottleneck is
quantity of labeled examples" conclusion (Final synthesis, 2026-08-07) is
**narrowed**, not withdrawn, to: quantity *and* annotation reliability
(kappa=0.40, round 1 §1) are both real limits **on this specific
mechanism class** — this says nothing about whether a mechanism that
does not require fitting a decision boundary from scarce positives at
all (WFST's zero-training decode; an alignment-gap rule, also
zero-training) faces the same ceiling.

## 4. The single highest-value next technical question

Not Dysfluent-WFST. **The evidence, read strictly against application
fit, reliability, and implementability — in that order, novelty last —
points to a cheaper, lower-risk, equally-real mechanism that has been
sitting cited-and-untried since before the external review arrived**
(named in `EXTERNAL_REVIEW_2026-08-07.md` section 6, item 4, as "likely
the fastest available win for `word_repetition` ... cited and never
tried," ranked *below* Dysfluent-WFST in that review's own
value-per-cost list — a ranking this reconciliation disagrees with,
reasoned explicitly below, not accepted merely because the review
proposed a different order).

**The question**: *Does an alignment-gap / duration-residual signal —
built entirely from CrisperWhisper's own already-computed word-level
timestamps, no new dependencies, no new data — recover usable
`word_repetition` candidates at exactly the positions Stage A already
diagnosed (the "other half of the pair" being deleted or displaced),
when fed into this project's existing candidate-generation/fusion
pipeline?*

**Why this, and not Dysfluent-WFST, first**, scored against the
project's own stated priority order (truth -> application fit ->
reliability -> implementability -> evidence -> novelty last):
- **Application fit**: identical tier to `sound_repetition` — both are
  named taxonomy types with a real, quantified loss rate (45.2% vs.
  40.5%). No principled reason to prioritize one over the other from the
  application's perspective; the tie-break falls to whichever fix is
  more reliable and implementable.
- **Reliability**: this mechanism is a general timing/duration-residual
  signal, not a model trained on a different population — it carries
  **no cross-population transfer risk**, unlike Dysfluent-WFST, whose
  only real-world validation is aphasia, a population this project has
  no evidence generalizes to stuttering, compounded by the paper's own
  documented severe noise sensitivity (round 1 §1: WPER 10%->~30-74%
  under modest synthetic noise).
- **Implementability**: reuses data this project's pipeline **already
  computes** (CrisperWhisper's own word-level timestamps — literally
  that model's headline contribution, per its own paper title,
  "Accurate Timestamps on Verbatim Speech Transcriptions"). Dysfluent-
  WFST requires cloning a new repo, installing `k2` (a nontrivial
  build dependency), downloading a separate phone-posterior model
  (wav2vec2/WavLM-CTC), and adding a grapheme-to-phoneme step this
  project does not currently have — a meaningfully heavier setup, before
  even reaching its own genuinely unresolved timestamp question
  (round 3 §6.2).
- **Evidence**: a real, cited, correctly-verified precedent exists (the
  alignment-gap classifier, [B26]/arXiv:2409.10177, 74.13% coverage on a
  related — not identical — task, verified round 1 §1) *and* this
  project has its own, independently hand-checked confirmation that the
  targeted mechanism is real (22/23 cases, Stage A). Dysfluent-WFST's
  evidence base for *this project's specific use case* (stuttering, not
  aphasia; realistic, not clean-studio, audio) is currently zero.
- **Novelty, deliberately last**: Dysfluent-WFST is the more novel,
  more interesting finding. That is exactly the criterion this
  reconciliation is instructed not to optimize for.

This is not a rejection of Dysfluent-WFST — it remains the correct
**second** step, for `sound_repetition` specifically, once the
independent `word_repetition` question is resolved. It is a re-ordering,
made explicitly and for stated reasons, of a value-per-cost ranking this
reconciliation found to be reasoned from novelty and mechanism-interest
rather than strictly from application fit, reliability, and
implementability.

## 5. Finite decision tree with hard stop conditions

**Step 0 (do regardless, zero cost, not a branch)**: recompute PR/AUPRC
+ bootstrap CIs on Ranks 1-3's already-saved predictions; measure the
end-to-end Track B effect of deploying Rank 1 as a candidate generator.
This is data hygiene on results already in hand, not a new experimental
branch, and should happen before or alongside step 1 regardless of its
outcome.

**Step 1 — `word_repetition`, alignment-gap/duration-residual test**,
against the existing 120-clip Track B sample (no new data, no new
dependencies):
- **SUCCESS** (end-to-end Track B `word_repetition` recall improves by a
  pre-registered, non-trivial margin, without a precision collapse in
  the existing fusion pipeline — reusing the same category-error
  correction already adopted in round 1 §5: judge this as a candidate
  generator feeding the existing corroboration/fusion layer, not by a
  standalone deployability floor): **integrate into `detect.py`'s
  existing fusion architecture as a new `word_repetition`
  candidate-generation signal.** `word_repetition`'s gap is closed. Move
  to Step 2.
- **FAILURE** (no meaningful end-to-end improvement): this specific,
  cheap, well-motivated mechanism is closed as a real, tested negative
  result for `word_repetition` — recorded, not silently dropped. Move to
  Step 2 regardless; `sound_repetition` is an independent problem and
  does not wait on this outcome.

**Step 2 — `sound_repetition`, Dysfluent-WFST**, pre-registered, with
the mandatory first sub-step being the runtime frame-synchronicity check
(`len(shortest[0].labels) == T`, round 3 §6.2) before any further
engineering effort, against both the existing 120-clip Track B sample
and the small real-stuttering evaluation set (Boli + FluencyBank +
Sep-28k-SW, evaluation-only per round 3 §6.1's licence finding):
- **SUCCESS** (stable, bootstrapped-CI-supported improvement over the
  current near-zero real detection, holding up on real audio and not
  just synthetic, with a working timing solution or an accepted coarser
  fallback using CrisperWhisper's own word-level timing): **integrate as
  a new `sound_repetition` fusion signal.** `sound_repetition`'s gap is
  closed. **Stop researching this problem — Stage D is not needed.** Move
  into application engineering (hardening, testing, latency, UI) per
  this project's own evidence-driven-not-preservation-driven discipline
  (standing rule 8): do not keep chasing further marginal gains once
  both real gaps are closed.
- **FAILURE** (does not transfer to real audio, or timing recovery
  proves infeasible at reasonable effort): proceed to the Stage D
  precondition below — this is the condition under which Stage D becomes
  justified for `sound_repetition`.

**When Stage D becomes justified — precise condition**: only after
**both** Step 1 and Step 2 have been run and shown insufficient to reach
a real, end-to-end, product-meaningful improvement, **and** a power
analysis on the assembled real evaluation data (round 3 §6.4: ~10h
across Boli/FluencyBank/Sep-28k-SW, a few hundred `sound_repetition`-type
events, kappa=0.40 annotation noise) confirms the fine-tune's own
planned success/failure gate is actually statistically resolvable at
that scale. If underpowered, the decision (more real data, or a coarser
L2 deletion-rate endpoint) must be made in advance, not discovered in
the results.

**Stopping condition for moving into application engineering entirely**:
once both `word_repetition` and `sound_repetition` candidate generation
produce a measured, real (not synthetic-only) improvement in this
project's own Track B / production pipeline — at that point this
research track's job is done; further effort belongs in shipping, not
in chasing incremental research gains on an already-adequate detector.

## 6. Claim classification for everything in this section

- **Experimentally demonstrated** (this project's own measured results):
  Stage A's 45.2%/40.5% normalization rates and the two distinct loss
  mechanisms (fragment loss vs. pair-half deletion); Ranks 1-3's own
  F1/precision/recall numbers and Rank 3's fold-instability signature;
  Track B's zero-`sound_repetition`-candidate finding.
- **Primary-source supported** (independently fetched and read, not
  taken secondhand): Dysfluent-WFST's actual input/output/limitations
  (rounds 1 and 3); the alignment-gap classifier's 74.13% coverage
  figure on its own (different) task; Sep-28k's CC BY-NC 4.0 licence;
  the dataset landscape findings in round 1 §2; Kordt et al.'s actual
  method and results (round 2 §5.4).
- **Inferred** (reasoned from the above, not directly measured): that
  the alignment-gap mechanism will transfer from its published task to
  `word_repetition` specifically; that Dysfluent-WFST's severe noise
  sensitivity on its own reported ablation predicts a similar problem on
  this project's own non-clean audio.
- **Unresolved** (explicitly not decided either way): whether Dysfluent-
  WFST's timestamps are cheaply recoverable (needs the runtime check,
  round 3 §6.2); whether a pretrained YOLO-Stutter/SSDM checkpoint is
  even publicly available (not yet checked — named honestly as a gap,
  not assumed either way); whether the sound-vs-word repetition
  granularity split is learnable by any fine-tuning approach at the
  available real-data scale.
- **Proposed, not yet run**: the Step 1 alignment-gap test; the Step 2
  Dysfluent-WFST pre-registration; the pre-Stage-D power analysis; the
  CrisperWhisper cross-attention probe (round 3 §6.4) — still queued,
  positioned after Step 1/Step 2 in priority since it tests a
  mechanistic question rather than closing either remaining gap
  directly.

**No production code has changed as a result of this reconciliation.**
This section is a decision, not an implementation — per explicit
instruction, nothing above is executed yet.

---

## Step 1 executed — 2026-08-08: alignment-gap test for `word_repetition` — FAILURE

Per the project owner's explicit go-ahead to proceed, Step 1 of the
decision tree above (section 5) was pre-registered
(`VALIDATION.md` §15, written before any code existed), implemented
(`profiling/evaluation/stage_k_alignment_gap_word_repetition.py`,
self-tested first — 9/9 checks pass), and run against the existing
Track B cache (189/499 clips cached, **no new ASR inference**).

**Result: clean FAILURE, not inconclusive — but scoped precisely, not to
the whole mechanism class.** n=29 `word_repetition` category-1 targets,
941 controls. Cohen's d=-0.209 (wrong direction — target positions had
slightly *lower* residual, not higher). AUC=0.485, 95% clip-level
bootstrap CI [0.396, 0.566] — comfortably includes chance and rules out
even a modest hidden effect at this sample size.

**Exactly what was tested, read from the code
(`stage_k_alignment_gap_word_repetition.py`'s `_residual` function), not
inferred from the title**: a single, specific operationalization —
`residual(h) = duration(h) + gap_before(h)`, i.e. the word's *own*
CrisperWhisper-reported duration **plus** the silence gap immediately
*before* it (not gap-after; not a multi-token window; not a *ratio* of
duration to any baseline/expected value — an additive sum), z-scored
against that clip's own mean/SD. This is the ONLY variant tested. The
**FAILURE verdict applies to this exact additive, gap-before,
own-duration, z-scored operationalization — not to "the alignment-gap/
duration-residual mechanism" as a general class.** Explicitly,
**untested, not rejected**: a gap-*after* variant; a multi-token window
spanning more than the one candidate token; and a **duration-ratio**
formulation (e.g. duration(h) normalized as a ratio to a baseline/
expected duration rather than an additive z-scored sum) — none of these
were implemented or run. Full detail, including the honest
interpretation of *why* this one variant failed (CrisperWhisper's own
accurate-timestamp design property may mean a deleted repetition's audio
simply doesn't leave this particular kind of residual on the surviving
word): `VALIDATION.md` §15.8.

**This closes `word_repetition` candidate generation via this one
specific, precisely-defined operationalization as a real, tested
negative result — it does NOT establish that the alignment-gap/
duration-residual mechanism in general is exhausted**, and must not be
read or cited that way. This is the first result this track has produced
with a clip-level bootstrap CI computed from the start, directly
addressing the external review's own critique that no confidence
interval had ever been computed anywhere in this track.

**Decision tree update**: per Step 1's own failure branch (section 5
above), proceeding to Step 2 (Dysfluent-WFST, for `sound_repetition`) —
`word_repetition` is an independent problem and this failure does not
block it. Step 2 is a materially heavier commitment (new dependencies:
`k2`, a phone-posterior model, a grapheme-to-phoneme step) than Step 1
was, so it is being reported as its own checkpoint before further
implementation, rather than proceeding silently — consistent with this
project's own "measure twice, cut once" discipline for larger
commitments, not a change in the decision tree itself.

---

## Step 0 executed — 2026-08-08: zero-compute reanalysis of Ranks 1/2/3

Per the project owner's explicit instruction to run Step 0, review
thoroughly for mistakes, and prepare (not start) Step 2. Full method and
numbers: `VALIDATION.md` §16. Summary:

**Two real mistakes were found and fixed during this work — named here,
not smoothed over**, per this session's own standing rule about
thorough, honest process:
1. Part B's first version called the detector without the gate-off
   config `_identify_positions` itself uses, silently invoking item 17's
   shipped encoder-based classifier gate for every clip — a ~30-90s/clip
   cost (already documented in `VALIDATION.md` §13.2) that turned a
   should-be-fast reanalysis into a 31+-minute run. Caught only because
   the project owner asked how much longer it would take, prompting a
   check of actual CPU time (47+ minutes of CPU work across 31+ minutes
   wall-clock — genuinely busy, not hung) rather than continuing to wait
   on an assumption. This was not just a performance bug — scoring the
   baseline under gate-on would have silently redefined what "baseline"
   meant relative to Stage A's own gate-off category-1 definition. Fixed.
2. A fold-id recomputation in the recall-floor reanalysis relied on an
   assumption (no rows filtered) that happened to hold in every run so
   far but wasn't guaranteed. Hardened so the actual fold ids used during
   fitting are returned directly, not recomputed. Did not change any
   already-reported number.
3. Unlike every other `stage_*` script in this codebase, `stage_l` was
   run for real before it had a self-test — a real process deviation
   from this project's own convention. Added and verified retroactively
   (10/10 pass); the fixes above don't change the previously-obtained
   numbers, verified by inspection.

**Results**:
- **Rank 1's original "FAILURE (recall floor)" verdict was substantially
  understated.** AUPRC=0.556 (~15x its own 0.036 chance rate, 95% CI
  [0.441, 0.681], nowhere near chance) confirms real, strong
  discrimination. An honest, non-leaky, per-fold recall-targeted
  reanalysis (threshold selected on train data only, never on pooled
  test predictions) reaches mean precision **0.783** at mean recall
  **0.289** — dramatically better than the original F1-optimal
  threshold's P=0.580/R=0.147, though just short of the pre-registered
  0.3 recall floor on average. This directly confirms the external
  review's own "category error" critique (deployability floors judged
  on a single F1-optimal point, not the achievable operating range).
- **Rank 2 remains a real Failure** even under the same honest
  reanalysis (mean P=0.118, still below the 0.15 floor).
- **Rank 3's precision improves substantially** (mean P=0.382) but its
  original concern was fold *instability*, not the floor — unresolved
  by this reanalysis.
- **A real but important caveat**: Rank 1's strong "Any" result combines
  `sound_repetition`+`word_repetition` into one label — it detects that
  a gap position exists well, without itself saying which type. A
  practical, plausibly solvable gap, not resolved here.
- **The end-to-end Track B effect (the number this track's own critique,
  and the external review's, both flagged as never computed) is now
  measured, using the recall-targeted threshold: real but modest** —
  overall recall 0.000 → 0.052, `sound_repetition` specifically
  0.000 → 0.095 (P=0.800). Verified by hand-tracing the full scoring
  path (the number initially looked smaller than Part A's per-fold
  recall implied) to confirm this is real, not a bug: most of the
  remaining loss comes from mechanisms Stage A already categorized as
  *not* Rank 1's target (ordinary ASR error, mis-routing), so Rank 1's
  category-1-scoped improvement, while genuine, only reaches a fraction
  of total real-world loss.

**A concrete, zero-new-cost, low-risk action item is now identified,
separate from Steps 1/2**: re-threshold the shipped Rank 1/item-17
classifier from F1-optimal to recall-targeted selection. Not adopted
here — a finding to be acted on only with explicit go-ahead (standing
rule 4), recorded as a candidate for `ROADMAP.md`.

**Step 0 does not close either remaining gap on its own** — the
end-to-end lift is real but small. Steps 1 (FAILURE) and 2
(Dysfluent-WFST) remain the active plan.

---

## Step 2 proposal — Dysfluent-WFST (NOT YET EXECUTED)

Per the project owner's explicit instruction: prepared for review, **not
started**. Nothing below has been run.

**Revised 2026-08-08 (checkpoint pass)** to incorporate four corrections
from review before anything is executed: real-data validation is made
mandatory rather than an optional appendix; the timing fallback is
decided now rather than left to be discovered at run time; scope is
widened to both disfluency types the mechanism could plausibly address,
not narrowed to one without justification; and a specific, known G2P
failure mode is named as something to measure explicitly rather than
let become an unexplained ceiling later. Nothing in this revision
executes anything — it changes the plan, not the repo.

### What this step is

Pre-registered in spirit by the decision tree above and by round 3's
§6.2/§6.5; this section makes the concrete implementation plan explicit
before any of it is executed, so it can be reviewed first.

**Objective**: test whether Dysfluent-WFST (Guo, Lian, Zhou et al.,
Interspeech 2025, arXiv:2505.16351,
`github.com/Berkeley-Speech-Group/DysfluentWFST`) recovers usable
candidates at exactly the positions CrisperWhisper's decoded text
normalizes away — Stage A's category 1 — for **both** `sound_
repetition` (the fragment-loss mechanism) and, contingent on point 5
below, `word_repetition` (the pair-breaking mechanism Step 1's one
tested variant already found FAILURE for, but which is a categorically
different mechanism from what Step 2 tests).

**Scope correction (was: `sound_repetition` only)**: the original
version of this proposal scoped Step 2 to `sound_repetition` alone,
reasoning that Dysfluent-WFST's own worked example is sound-level. That
scoping was never actually justified against the mechanism itself. The
WFST's "return arc" (revisit a previously-visited state in the reference
phoneme graph) is generic to *any* span of the reference phoneme
sequence being revisited — a single phoneme (matching `sound_
repetition`) or a whole word's multi-phoneme span (matching `word_
repetition`) are both, in principle, states the decoding graph can
revisit. There is no stated mechanism-level reason this would only work
for one granularity. **Step 2 must therefore score against both types**,
not assume `word_repetition` is out of scope by default.

### Mandatory prerequisites (must be satisfied before the main run, not optional extras)

1. **Real, non-synthetic stuttering audio acquisition is part of Step 2
   preparation itself, not an optional "if acquired by then" appendix.**
   The prior version of this proposal listed real audio as something to
   include "if acquired by then, not required to start" — that framing
   is corrected here: given Dysfluent-WFST's *only* real-world validation
   anywhere is aphasia (nfvPPA), not stuttering, and its own documented
   severe noise sensitivity, evaluating it on synthetic LibriStutter
   splices alone would be structurally uninformative about the actual
   question (does it transfer to real stuttering audio) — a synthetic-
   only result could not distinguish "the mechanism works" from "it
   works on clean, spliced-real-narration audio specifically." Acquiring
   Boli (CC-BY-4.0, real, open, downloadable today) is therefore a
   **required** preparation step before the main evaluation run, not an
   optional enrichment of it.
2. **The frame-synchronicity/timing fallback is decided now, in advance,
   not left to be interpreted after seeing the result.** The runtime
   check is `len(shortest_path.labels) == T` (number of frames) on a
   known clip. **Pre-registered decision rule, fixed before running**:
   if this check passes (`==`), timing is recovered from the frame-
   aligned path as originally planned. **If it fails (`!=`), detection
   is still evaluated in full (event occurred: yes/no, and type), but
   localization/timing is explicitly marked deferred/not-provided for
   this run** — not silently guessed at, not treated as a reason to
   abandon the whole experiment, and not decided after looking at how
   the numbers came out. This exact branching rule is what gets executed
   when Step 2 runs; it is not to be revised after seeing which branch
   the real check lands in.
3. **CMUdict/G2P coverage must be measured and reported explicitly, not
   allowed to become a silent, unexplained recall ceiling.** Standard
   G2P pipelines built on CMUdict-style dictionaries commonly skip
   out-of-vocabulary words silently (continue past them rather than
   erroring) — if this project's own future G2P step does the same,
   any word CrisperWhisper transcribes that isn't in the dictionary
   would silently vanish from the phoneme reference sequence, and any
   resulting recall shortfall would look like a Dysfluent-WFST failure
   when it was actually a G2P coverage gap. **Required**: report, for
   the actual evaluation corpus used, the fraction of reference words
   successfully mapped to a phoneme sequence vs. silently skipped/OOV,
   as its own explicit number — before attributing any recall shortfall
   to the WFST mechanism itself.

### Concrete steps, in order

1. **Environment setup** (new dependencies — flagged explicitly, since
   this is a materially larger environment change than anything in
   Step 0/1): clone `Berkeley-Speech-Group/DysfluentWFST`; install `k2`
   (a WFST library with a nontrivial build, per round 1's finding that
   its repo's own documentation is thin); install/download a phone-
   posterior model (wav2vec2 or WavLM-CTC, per the paper); add a
   grapheme-to-phoneme (G2P) step to convert CrisperWhisper's word-level
   transcript into the phoneme reference sequence the WFST decoder
   requires (not currently part of this project's pipeline at all) —
   with OOV/coverage measurement built in from the start, per
   prerequisite 3 above, not added after the fact.
2. **Acquire Boli** (prerequisite 1 above) before the main run.
3. **Run the frame-synchronicity check** (prerequisite 2 above) and take
   the pre-registered branch it lands in — do not re-decide the fallback
   after seeing the result.
4. **Pre-register the exact evaluation methodology** in `VALIDATION.md`
   (matching every prior stage's discipline) before running the real
   test: scoring against the *existing* Stage A baseline (45.2%/40.5%
   loss rates for `sound_repetition`/`word_repetition` respectively) and
   Rank 1's own numbers (section 16/17's corrected understanding of
   them) — not a fresh, incomparable metric set — and covering **both**
   `sound_repetition` and `word_repetition` per the scope correction
   above.
5. **Run against two explicit, separately-scored audio conditions**, per
   the paper's own documented severe noise sensitivity (round 1 §1: WPER
   10% → ~30-74% under modest synthetic noise): (a) the existing
   120-clip Track B real-ASR sample (LibriStutter's spliced-real-audio
   condition); (b) the acquired real Boli audio (now mandatory, not
   optional, per prerequisite 1).
6. **Score using this project's own localization/candidate-generation
   convention** (event position + type, not the paper's own phoneme-
   sequence-accuracy metrics), reusing `score_clip`/Track B machinery
   where possible, same as every other stage in this track — with
   localization explicitly marked deferred wherever the timing fallback
   (prerequisite 2) landed in that branch.

### Named risks, not smoothed over

- **No confirmed CPU feasibility.** The paper never states a GPU
  requirement, but this is an inference (round 1 §1), not confirmed.
- **Timing may be unavailable for this run** — the fallback in
  prerequisite 2 is a pre-registered way to still get a detection-only
  result if so, not a claim that timing is actually recoverable.
- **Only real-world validation anywhere ELSE (i.e. in the paper itself)
  is aphasia (nfvPPA), not stuttering** — exactly why real Boli audio is
  now a mandatory prerequisite rather than an optional extra.
- **Repo maturity is uncertain** (thin README, no confirmed install
  path) — nontrivial integration engineering, not "clone and run."
- **G2P is a new pipeline component** this project has never needed
  before — its coverage/OOV behavior is measured explicitly (prerequisite
  3), not assumed complete.
- **`word_repetition` inclusion is untested territory even for this
  mechanism** — the scope correction above is a reasoned argument for why
  it should be in scope, not a claim that it has been shown to work;
  Step 1's FAILURE was for a *different* mechanism (duration/gap
  residual) and says nothing about whether Dysfluent-WFST specifically
  would recover `word_repetition` positions.

### Success/failure criteria (extended from round 3 §6.5 to cover both types)

- **Success (per type, evaluated separately for `sound_repetition` and
  `word_repetition`)**: stable, bootstrapped-CI-supported improvement
  over Rank 1's own now-better-understood operating point (sections
  16-17 above), holding up on the real Boli audio condition specifically
  (not synthetic-only), with either working timing or the pre-registered
  detection-only fallback explicitly reported as such. → integrate as a
  third fusion signal for that type; Stage D not needed for it.
- **Failure (per type)**: does not transfer to real audio, or the
  detection-only result itself does not clear the bar. → that type
  proceeds to the Stage D precondition (both Steps 1 and 2 failed for
  that type, and a power analysis on the assembled real evaluation data
  confirms Stage D's own gate is resolvable) named in the decision tree
  above. Note `word_repetition`'s Step 1 result was for a different
  mechanism — a `word_repetition` Failure here would be the first real
  test of *this* mechanism for that type, not a second confirmation of
  Step 1's finding.

**Nothing in this proposal has been executed.** This is a plan for
review, exactly as instructed.

---

## Rank 1 re-thresholding follow-up validation — executed 2026-08-08 — INSUFFICIENT EVIDENCE, MORE VALIDATION REQUIRED

Per the project owner's explicit decision (Rank 1's original Failure
verdict should not be treated as final, but the evidence was not yet
strong enough to adopt the recall-targeted threshold in production),
ran a focused, decision-closing follow-up. Full pre-registration and
results: `VALIDATION.md` §17. Step 2 was not executed; no production
code (`profiling/repetition_classifier.py`, `config.yaml`) was touched,
per explicit instruction.

**A real methodological bug was found and fixed, with a retroactive
consequence for Step 0 itself**: the first version of this follow-up
selected its recall-targeted threshold from each fold's own in-sample
training predictions rather than genuinely out-of-fold signal — caught
immediately because the result (mean P=0.400, R=0.054) was drastically
different from Step 0's own already-published number (mean P=0.783,
R=0.289) for what should have been the identical computation. Fixed by
reusing Step 0's exact two-stage procedure; the corrected re-run
reproduces Step 0's numbers exactly. **The same bug, it turns out, was
already present in Step 0's own original end-to-end Part B measurement**
— fixed the same way, with a superseded-in-place notice added to
`VALIDATION.md` §16.3 (not edited, per this track's own convention).

**Results**:
- **Fold stability (Q1)**: threshold values range [0.66, 0.80] — moderate,
  not fragile. A leave-one-fold-out jackknife shows no single fold
  drives the result (max shift ±0.07 precision, ±0.035 recall).
- **Recall-floor approach (Q2)**: mean recall (0.289) falls just short of
  the 0.3 floor, but this is not a precision-for-recall trade-off — mean
  precision (0.783) is *higher* than the original F1-optimal point's
  (0.580). Both metrics improve simultaneously.
- **Any-label/type issue (Q3)**: corrected — true positives validly
  attributed by ground-truth type; false positives honestly tracked as
  type-unattributable rather than fabricated as a specific type.
- **End-to-end effect (Q4), corrected for both the type issue and the
  threshold bug**: substantially larger than Step 0's original (now-
  superseded) finding — Any-label recall 0.000→**0.247** (was 0.052) at
  precision **0.559** (was 0.308). Still recovers only a minority (~25%)
  of total ground-truth loss in these clips.
- **Single vs. per-fold threshold (Q5/Q6)**: naively refitting on the
  full dataset produces a threshold (0.982) that **falls outside** the
  validated per-fold range [0.66, 0.80] and is demonstrably an
  overfitting artifact — **no validated, ready-to-ship threshold value
  currently exists**; the per-fold procedure is confirmed to be an
  evaluation device only, not a deployment specification.

**Strict decision-oriented conclusion: INSUFFICIENT EVIDENCE — MORE
VALIDATION REQUIRED.** Not ADOPT: no principled, validated method exists
yet for choosing the actual value to ship (the one concrete attempt at
this — naive full-dataset refit — is shown to be unreliable), and the
type-attribution problem remains unresolved *for deployment* (only
*measurement* was fixed here) — a fired candidate on new, real audio
still doesn't say which type it is. Not DO NOT ADOPT: the underlying
direction (recall-targeted beats F1-optimal) is now well-supported,
simultaneously improving both precision and recall, with a real,
substantially-larger-than-previously-known end-to-end product benefit.
Three concrete, named requirements remain before a production change:
(1) a validated deployable-threshold-selection method; (2) a resolution
for what the product does when type is unknown; (3) validation against
at least some real, non-synthetic audio.

**Scope correction, 2026-08-08 (same day, checkpoint pass): what the
"threshold work" is and is not.** This entire follow-up must not be read
as a production/shipping experiment, and the item it leaves open (a
validated deployable threshold) must not be read as an invitation to an
open-ended threshold-optimization thread. Precisely:

- **What it is**: a *prerequisite* for evaluating Rank 1 on real
  (non-synthetic) audio at all — a classifier needs some operating point
  before it can be run on anything, real or synthetic, and the whole
  point of this follow-up was determining whether the *existing*,
  already-fitted classifier's operating point could be trusted, not
  designing a new one.
- **What it is not**: not a decision to change `config.yaml` or
  `profiling/repetition_classifier.py` (unchanged, confirmed); not a
  license to keep iterating on threshold-selection variants looking for
  a better number.
- **The next threshold task, if and when picked up, is bounded and
  deterministic, not exploratory**: obtain genuine out-of-fold
  predictions across the full cached "Any" dataset (reusing
  `_out_of_fold_scores`, unmodified logic, exactly as this follow-up
  already does), apply the single pre-specified recall-targeting rule
  (`_best_threshold_for_recall_floor`, `min_recall=0.3`, already fixed
  and unchanged since section 15/16), and report the resulting
  cross-validated operating point. That is the entire task — not a
  search over rules, targets, or feature sets.
- **If a future validated result falls outside the previously observed
  per-fold range [0.66, 0.80], that is itself a finding to record**
  (e.g., "the range shifted because X changed") **— not a trigger to
  automatically launch another optimization cycle** looking for a
  "better" threshold. The 0.982 naive-full-refit value already found
  (17.6) stays recorded as an overfitting artifact, not a rejected
  candidate to keep re-trying variations of.
