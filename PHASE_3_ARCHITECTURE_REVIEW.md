# PHASE_3_ARCHITECTURE_REVIEW.md — is ASR-first the right foundation?

**Written 2026-08-04, before any Phase 3 implementation work**, per the
project owner's explicit request: challenge the ASR-first architecture
from first principles before continuing to build on it, using Phase 1/2's
own evidence plus a fresh literature pass, and record the conclusion as a
major architectural decision *before* proposing implementation. This
follows the same pre-registration discipline as `PHASE_2_RESEARCH_PLAN.md`
— a decision document, not itself an implementation.

**The question being asked, precisely** (the owner's own framing, which
this review adopts): not "which ASR should we use," but *"what
representation of speech gives the detector the highest possible accuracy
for disfluency detection, classification, and localization?"*

**Headline conclusion**, expanded and justified in full below:
**keep the two-stage (ASR + detector) architecture as the foundation — the
evidence does not support replacing it — but the evidence *does* support a
real, scoped extension: the audio-native-primary principle Phase 1/2
already validated for `block`/`prolongation` should be extended to
`word_repetition`/`sound_repetition`/`filler`, and the lowest-cost path to
richer acoustic signal for that is reusing CrisperWhisper's own encoder
representations (already computed on every clip), not adding a new
self-supervised model.** This is a Phase 3 candidate, not a Phase 3
mandate — it still needs its own pre-registration and evaluation, per
standing discipline, before any code is written.

---

## 1. What was already decided, and why this isn't re-litigating it blind

This exact question — ASR-first two-stage vs. a fundamentally different
architecture — was reviewed once before, on 2026-08-03, before Phase 1
began (`PAPER_DECISION_LOG.md`, "Vision alignment review + architecture
decision"). That review considered ~25 papers and 6 datasets and reached
three specific conclusions, each with a stated reason:

1. **End-to-end audio→dysfluency-region models** (YOLO-Stutter,
   Stutter-Solver, SSDM-class) were rejected: they still require a
   speech-text alignment as input (they don't eliminate the ASR stage,
   only replace the detector with something heavier), and a 2025
   comparative study (arXiv:2509.00058) found the most complex of these
   (SSDM) **could not be reproduced by the paper's own independent
   authors**, while the simplest (UDM) had the best accuracy/
   interpretability balance.
2. **Discarding the rule-based/acoustic tier for a learned model** was
   rejected: "Revisiting Rule-Based Stuttering Detection" (arXiv:2508.16681,
   2025) found rule-based systems remain near-SOTA specifically for
   prolongation detection (97-99% reported accuracy).
3. **A learned classifier tier (frozen WavLM/wav2vec2) was identified as
   the clear future step, explicitly deferred** until a baseline existed
   to prove the audio-native restructuring itself worked, and until a
   training pipeline existed. Both conditions were unmet at the time.

**What's changed since then, and why this review is warranted now rather
than being a rerun**: the baseline now exists (`VALIDATION.md` §8.3, §9.5.1
— audio-native restructuring measurably works, `Any` F1 0.773→0.888 across
Phase 1+2), and — critically — Phase 1/2 produced a much sharper, *causal*
picture of exactly where the pipeline's real-world accuracy is lost
(§2 below) that wasn't available in 2026-08-03. This review does not
re-open the three conclusions above without new evidence; it re-examines
the open question they left behind ("what's the future step") in light of
evidence that didn't exist when it was first deferred.

## 2. What this project's own evidence says (Phase 1 + Phase 2)

- **Track A (ground-truth transcript) vs. Track B (real ASR)**: recall
  collapses from ~99% to ~6-15% depending on exact subset/definition
  (`VALIDATION.md` §8.4). This is this project's single most consequential
  empirical finding.
- **Once corrected for a real methodological gap** (context-strict
  scoring), most — but not all — of that real-world gap is
  ASR-attributable, not detector-attributable: **35.1% detector / 64.9%
  ASR** at full 40-speaker diversity (`VALIDATION.md` §8.4.3) — down from
  an earlier, less representative 0%/100% read, but still ASR-majority.
  **This number matters a great deal for this review**: it means the
  representation genuinely is the dominant lever, but "dominant" is not
  "only" — a meaningful detector-attributable share remains, so a
  representation change alone would not close the whole gap.
- **The ASR-attributable share is not evenly distributed across
  disfluency types.** `block` (silent-gap-based, already audio-native
  since Phase 1) and `prolongation` (moved to audio-native-gated in Phase
  2, §9.5.1) both improved from being made *less* dependent on the ASR
  token stream. `word_repetition`/`sound_repetition`/`filler` remain
  almost entirely token-text-dependent today (`ARCHITECTURE.md`:
  "`profiling/acoustic.py`'s audio-native detector only derives
  `block`/`prolongation` candidates — it has no repetition or filler
  logic at all"). This is the load-bearing internal evidence for this
  review's recommendation in §5.
- **A directly relevant negative result**: the word-sandwiched-repetition
  extension (`VALIDATION.md` §8.4.4, `PHASE_2_SUMMARY.md` §4) tried to
  paper over an ASR-token-fidelity problem with *more* token-level logic,
  and made things worse. This is a small but real data point against
  "just add more text-side cleverness" as a fix for the ASR-fidelity
  problem — consistent with this review's conclusion that the fix belongs
  on the acoustic side, not the text side.
- **A directly relevant positive result**: the prolongation redesign
  (`VALIDATION.md` §9.5.1) improved *both* aggregate and type-specific F1
  specifically by adding a genuine acoustic hard-gate (pitch/jitter/
  shimmer stability) in place of pure token-duration reasoning. This is
  the clearest existing proof, inside this project's own data, that
  "less token-dependent, more acoustic-native" is a working lever, not
  just a theoretical one.

## 3. Fresh literature review (2024-2026) — every direction the owner named

### 3.1 Newer/stronger Whisper-family models

CrisperWhisper's own paper (arXiv:2408.16589) reports 6.66% average WER
across nine benchmarks and ranks first on the OpenASR Leaderboard for
verbatim transcription specifically, but **underperforms plain
Whisper-Large-v3 on three general-purpose benchmarks** (Earnings22: 12.37%
vs. 11.3%; GigaSpeech: 10.27% vs. 10.02%; LibriSpeech-other: 3.97% vs.
3.91%) — it trades general accuracy for verbatim/disfluency preservation,
which is exactly the trade this project wants. No newer model was found in
this pass that is demonstrated to preserve disfluencies *and* transcribe
them *and* time them more accurately than CrisperWhisper specifically;
"Lost in Transcription" (§3.4 below) tested Whisper broadly (not
CrisperWhisper) and still found severe disfluent-speech WER degradation,
suggesting the underlying problem is not simply "which exact Whisper
checkpoint" — it's a more structural limitation of transcript-token
representations for disfluent speech (§3.5, §3.6). **Conclusion: no
evidence found that swapping to a different or newer Whisper-family
checkpoint would materially close the gap** — consistent with this
project's own prior finding that faster-whisper/OpenVINO backends were
rejected for unrelated compatibility reasons, not superseded on accuracy
grounds.

### 3.2 ASR models trained specifically on disfluent speech

"Lost in Transcription" (arXiv:2405.06150) explicitly recommends
"incorporating speech from people who stutter in ASR training" as a
mitigation, "despite data scarcity challenges" — i.e., this is a
recognized, real direction, not a solved one. No production-ready,
openly-available ASR model fine-tuned specifically for stuttered/disfluent
speech was found in this pass (the Interspeech 2025 paper on personalized
vs. generalized stuttered-speech ASR fine-tuning, isca-archive.org, could
not be fully extracted by this review's tooling, but its existence
confirms this is active, unresolved research territory, not a shelf-ready
solution). **Conclusion: a real direction, but not currently a drop-in
replacement — building or fine-tuning one is itself a research project
this codebase has no training pipeline for.**

### 3.3 Richer timestamp/alignment techniques

Already directly considered and declined in Phase 1 (`ROADMAP.md`,
"Longer-term": CTC forced alignment "considered... and not adopted...
CrisperWhisper's own benchmarked word-timestamp accuracy was the reason it
was chosen in the first place... revisit only if `VALIDATION.md` results
show word-boundary precision is actually a limiting factor"). Phase 2's
own localization finding (`VALIDATION.md` §9.5.1: praat-gating's surviving
TPs have *lower* localization, 0.857→0.500) is about a smaller, different
TP set under a stricter gate, not evidence of a general word-boundary
precision problem. **Conclusion: this project's own prior decision on this
specific question still holds; nothing in this review's evidence
overturns it.**

### 3.4 Self-supervised speech representations (wav2vec2, HuBERT, WavLM)

The current word-level SOTA found in this pass: WavLM Large + a
Hierarchical Convolution Interface + auxiliary CTC loss, pretrained on
LibriSpeech-with-synthetic-disfluencies then fine-tuned on SEP-28k
(arXiv:2409.10704) — **word-level F1 = 0.554** (vs. a prior baseline's
0.411), utterance-level F1 = 0.803 on SEP-28k. Explicitly stated
limitation: "high recall and low precision in most cases... predict[s]
stuttering events more frequently than the speech pathologist" — **the
exact same failure mode this project fought and fixed in Phase 2**
(§9.1/§9.5.1's FP-suppression work). This is meaningful context: the
field's own SOTA fully-learned model, at word-level granularity, is not
dramatically ahead of what a well-tuned rule-based-plus-acoustic-fusion
system can achieve, and shares the identical precision weakness this
project already solved once. Requires: a GPU training pipeline, a
15.6-hour labeled dataset (small relative to LibriSpeech's 960 hours, per
the paper's own framing), and produces a black-box classifier with, per
the paper, no interpretability method provided. **Conclusion: real,
credible, but not a clearly superior result at this project's current
scale/task granularity, and requires infrastructure this project doesn't
have.**

### 3.5 Acoustic embeddings without a full new SSL model — the most
directly actionable finding of this review

"Optimizing Multi-Stuttered Speech Classification: Leveraging Whisper's
Encoder for Efficient Parameter Reduction" (arXiv:2406.05784) uses
**Whisper's own encoder representations (not its decoded transcript)** as
classification features — freezing all but the last encoder layer (3.29M
trainable params, an 83.7% reduction from full fine-tuning) — and reports
micro/macro/weighted F1 of 0.88/0.85/0.87 on SEP-28k, externally validated
on FluencyBank. This is clip-level multi-label classification, not
word-level localization, so it is not directly comparable to §3.4's
number or to this project's own word-level scoring — but it demonstrates
something specific and useful: **a Whisper encoder's internal
representations carry real disfluency-relevant signal beyond what the
decoded transcript captures, and extracting that signal doesn't require a
new model** — this project already runs a Whisper-family encoder
(CrisperWhisper) on every clip. This is the cheapest "richer
representation" lever found in this entire review, because the expensive
part (running the encoder forward pass) already happens today; only a
lightweight classification head over already-computed encoder states
would be new.

### 3.6 Hybrid acoustic + linguistic architectures, and full audio-native

Two independent, converging findings, both supporting the same direction
this project already took for `block`/`prolongation`:

- A Microsoft paper (arXiv:2311.00867, "Automatic Disfluency Detection
  from Untranscribed Speech") states directly: **"an acoustic-based
  approach that does not require transcription as an intermediate step
  outperforms the ASR language approach"** for disfluency detection.
- "Lost in Transcription" (arXiv:2405.06150) found real stuttered speech
  WER roughly doubles vs. fluent speech across every ASR system tested
  (Whisper: 6.3%→12.1%; Google Cloud: 6.9%→27.5%; IBM Watson: 16.7%→47.6%;
  wav2vec2: 10.1%→38.0%), and — the specific, structurally important
  detail — **word repetitions, prolongations, and interjections show the
  highest WER impact (35-47%), while blocks show a smaller effect
  (~20%)**. This maps almost exactly onto this project's own detector
  structure: `block` (audio-native since Phase 1) is the type ASR damages
  *least*; `word_repetition`/`filler` (still token-text-dependent) are
  among the types ASR damages *most*. The literature's per-type pattern
  and this project's own architecture split line up.

A newer, more theoretical paper ("On the Difficulty of Token-Level
Modeling of Dysfluency and Fluency Shaping Artifacts," arXiv:2512.02027)
gives a structural reason why: it found that jointly modeling dysfluency
as tokens *and* fine-grained modifications degrades word error rate
sharply (0.214→0.743) when forced into one token stream, because
"fluency-shaping modifications resist tokenization entirely... [they]
manifest through subtle acoustic modifications rather than distinct
lexical units." **This is an independent, literature-level explanation
for why this project's own Phase 2 finding held** (prolongation improved
specifically when moved off pure token-duration reasoning onto a genuine
acoustic gate) — prolongation is exactly this kind of continuous,
non-discrete phenomenon.

### 3.7 End-to-end / fully audio-native replacement (revisited)

SSDM 2.0 (arXiv:2412.00265), the direct successor to the SSDM architecture
already rejected in this project's 2026-08-03 decision for
irreproducibility, is **heavier still**: it adds a "neural articulatory
flow," a "connectionist subsequence aligner," and an LLM-integration
pipeline with a "mispronunciation prompt" and "consistency learning"
module, evaluated on specialized corpora (nfvPPA, Libri-Dys, Libri-Co-Dys)
this project has no access to or validated relationship with. **This
reinforces, rather than reopens, the original rejection**: the field's
most complex end is moving toward more specialized, heavier, harder-to-
reproduce architectures, not simpler or more accessible ones.

### 3.8 Joint/multi-task ASR + detection training

A genuinely new and different direction not previously considered by this
project: "Leveraging LLM for Stuttering Speech: A Unified Architecture
Bridging Recognition and Event Detection" (arXiv:2505.22005) jointly
trains ASR and stuttering-event-detection as a multi-task LLM-driven
model with bidirectional information sharing (ASR's CTC output feeds the
LLM's context; the detection branch's stutter embeddings feed back into
the LLM's speech comprehension). Reported: **37.71% relative CER
reduction** and **46.58% relative SED F1 improvement** versus separate-
task baselines, on the AS-70 Mandarin stuttering dataset. This is real,
credible evidence that *coupling* the two tasks — rather than treating ASR
as a fixed upstream black box — has genuine synergy, not just additive
value. **This is the one finding in this review that most directly
challenges the two-stage assumption itself**, and deserves to be named
plainly as such. It is not, however, actionable for this project right
now: it requires jointly fine-tuning an LLM-driven ASR model end-to-end
with detection labels, on a dataset with *both* accurate transcripts and
word-level disfluency labels together (AS-70 has this for Mandarin; this
project's own accessible English datasets — SEP-28k, LibriStutter — do
not pair the two the way this training regime needs), and this project has
already found, empirically, that even *swapping* ASR backends without any
retraining causes real compatibility problems (faster-whisper's tokenizer
incompatibility with CrisperWhisper, `ARCHITECTURE.md` §3). Fine-tuning
the ASR model itself jointly with detection is a substantially larger lift
than anything this project has attempted.

## 4. Weighing it: the four practical constraints, applied honestly

- **Engineering practicality**: this project has no GPU/training
  pipeline, confirmed multiple times across its history (the original
  2026-08-03 decision, the deferred-learned-tier note). Every SSL/
  learned-classifier/joint-training direction in §3.4/§3.5/§3.8 requires
  one, to different degrees. §3.5 (Whisper-encoder reuse) requires the
  least: the expensive forward pass already runs; only a small
  classifier head is new, and it could plausibly be trained on CPU given
  its scale (3.29M params in the comparable published result), or even
  prototyped without gradient training at all (e.g., a distance/
  similarity-based classifier over encoder states, extending this
  project's existing zero-training-component philosophy).
- **Explainability**: this project's rule-based detector produces
  human-readable evidence strings for every event (`ARCHITECTURE.md`) and
  its audio-native components are near-SOTA specifically where they've
  been made acoustic (prolongation, per the 2025 rule-based comparative
  study already cited in this project's docs). Every learned/SSL option
  in §3.4/§3.8 is reported as a black box in its own paper (§3.4's paper
  states no interpretability method was provided). This is a real,
  literature-confirmed trade-off this project would be accepting, not a
  hypothetical one.
- **Compute/latency trade-offs**: this project's ASR stage already costs
  54-102s/clip on CPU (`ARCHITECTURE.md` §3, §7) — a documented,
  load-bearing constraint on the current architecture. WavLM-Large-class
  models are comparable in parameter count to Whisper-large-tier models;
  adding one as a second full model pass would plausibly compound this
  latency problem, not solve it, on the same CPU-only deployment target.
  Reusing the *already-running* CrisperWhisper encoder (§3.5) adds no new
  forward pass at all — the only realistic option in this review that
  doesn't worsen the existing latency constraint.
- **Available validation datasets**: this project's whole methodology
  (Phase 1's founding premise) is that nothing ships without real,
  measured validation. LibriStutter and SEP-28k, the datasets this
  project actually has, support evaluating a Whisper-encoder-based
  acoustic corroboration signal exactly the way `block`/`prolongation`'s
  audio-native signals were validated (Track A/B, ablation, the same
  `VALIDATION.md` machinery already built). A from-scratch SSL classifier
  trained on SEP-28k's labels would need a held-out split disciplined
  enough to avoid the exact leakage risk SEP-28k-E's own paper was
  designed to prevent (`VALIDATION.md` §4 point 5) — a real, extra
  methodological burden §3.5's lighter approach doesn't carry to the same
  degree.

## 5. Recommendation

**Keep the two-stage ASR + rule-based/acoustic-fusion detector as the
architectural foundation.** The evidence reviewed here does not support
replacing it: no fully-learned or end-to-end alternative found in this
pass is clearly, decisively more accurate at this project's actual task
granularity (word-level, multi-type, localized) once the same precision
failure modes are accounted for, and every such alternative costs real,
currently-unavailable infrastructure (training pipeline, GPU, joint-
labeled data) plus explainability this project's current results have
specifically benefited from keeping.

**But the evidence strongly supports treating "ASR-first" as a spectrum,
not a binary, and moving further along it than this project currently
has.** The clearest, most directly evidenced next step: **extend the
audio-native-primary principle — already proven twice inside this
project's own data (`block` in Phase 1, `prolongation` in Phase 2) and
now independently corroborated by outside literature's per-type WER
breakdown (§3.6) — to `word_repetition`/`sound_repetition`/`filler`,
which remain almost entirely token-text-dependent today.** The
lowest-cost, infrastructure-realistic mechanism for that: extract and use
CrisperWhisper's own encoder representations (already computed on every
clip, §3.5) as an additional acoustic corroboration/confidence signal for
these types, the same architectural role Silero VAD and Praat features
already play for `block`/`prolongation` — not a token-stream replacement,
an additional signal in the existing weighted-fusion mechanism.

**This is a Phase 3 candidate, explicitly not authorized for
implementation by this review alone.** Per this project's own standing
discipline, it needs its own pre-registration in `VALIDATION.md`
(what exactly gets extracted from the encoder, what the corroboration
rule is, what success looks like) before any code is written, exactly the
process the prolongation redesign went through in Phase 2. This review's
job was to decide the *foundation* question, not to skip that process for
the specific next step it identifies.

### 5.1 Refinement (2026-08-04, same day): exactly which representation, and a staged plan

The recommendation above named "CrisperWhisper's own encoder representations" without specifying which internal representation, how accessible it actually is in this codebase, or how it compares to the alternatives on cost and evidence. That refinement was done as a direct follow-up, grounded in `profiling/asr.py` as it actually exists today, not in the abstract.

**How CrisperWhisper is actually called matters.** This project calls it through `transformers.pipeline("automatic-speech-recognition", ...)` — the high-level wrapper, not direct model calls — and that wrapper already sets `attn_implementation="eager"` specifically because, per the code's own comment, "the pipeline requests `output_attentions` internally for word-timestamp extraction." **Cross-attention tensors are therefore already computed on every real transcription this app runs today**, consumed internally for the DTW word-timestamp alignment CrisperWhisper's own paper describes, then discarded. Encoder hidden states and decoder token probabilities are also produced as a side effect of the same forward pass but are never touched by the pipeline at all. All three candidate representations require the same one-time engineering change — bypassing `pipeline()` for a direct `model.generate(..., output_attentions=True, output_hidden_states=True, return_dict_in_generate=True)` call — and none require a new model, a new forward pass, or touching CrisperWhisper's weights.

**Three candidates compared:**

| Candidate | Accessibility | Direct evidence for disfluency detection | Caveat |
|---|---|---|---|
| Cross-attention weights | Highest — already computed, already load-bearing for this project's timestamps | Proven for *timing* only; no source found validates the attention *pattern* itself as a standalone disfluency signal (a secondary, unverified claim of CrisperWhisper F1=0.90 vs. WhisperX 0.85 on an AMI disfluency subset surfaced but could not be confirmed against the primary paper) | Using it for anything beyond timing is a novel application, not a literature transplant |
| **Encoder hidden states (last layer)** | Moderate — same pipeline-bypass; the encoder pass is unavoidable regardless, so capturing its output is near-free computationally | **Strongest of the three**: a Whisper encoder's last layer alone (all other layers frozen, ~3.3M trainable params) reached F1 0.88/0.85/0.87 (micro/macro/weighted) on SEP-28k, externally validated on FluencyBank (arXiv:2406.05784) | That result is clip-level classification, not word-level localization, and on stock Whisper, not CrisperWhisper specifically |
| Decoder token confidence/entropy | Moderate-high — free to capture, but Whisper's BPE tokens don't map 1:1 to words | Weakest — general ASR-uncertainty literature confirms it's real and meaningful, and independently we know ASR is far less confident on disfluent speech, but no paper found directly validates token confidence as a stuttering-detection feature | This project runs greedy decoding (`num_beams=1`, a forced workaround for a timestamp bug) — confidence from greedy decoding is typically noisier than from beam search |

**Recommendation, staged, not a single leap to a trained classifier:**

1. **Stage 1 (test first, zero training)**: extract the encoder's last-layer hidden states for a candidate event's audio span and use a simple non-parametric measure (e.g. cosine distance to the clip's fluent-region embeddings, or self-similarity across repeated spans for repetition candidates) as an additional corroboration signal in the existing weighted-fusion mechanism — the same architectural role VAD/Praat already play. Zero training, directly testable with the Track A/B machinery already built.
2. **Stage 2 (only if Stage 1 shows real signal, only with explicit go-ahead)**: a small trained classification head over the frozen last layer, following the published recipe. This would be the first real departure from this project's zero-training-component philosophy (VAD/Praat are both pretrained, zero-shot signals) — worth deciding deliberately, not backing into as a side effect of picking a representation.
3. **Cross-attention weights captured in the same engineering pass, as a free secondary signal** to test alongside Stage 1, since the same code change exposes both — but not the primary candidate, since its only proven use is timing.
4. **Decoder confidence deprioritized** — weakest evidence, plus a project-specific reason (greedy decoding) to expect a noisier signal than the literature it would be drawing the inference from.

This refinement did not change §5's foundational conclusion; it made the specific next step concrete and checked it against this project's actual code rather than the abstract idea of "reuse the ASR's internals." See §8 below for an adversarial challenge to this specific refinement, done the same day at the project owner's request.

## 6. What would change this conclusion

Stated explicitly, so this isn't a conclusion immune to future evidence:

- If a training pipeline (even CPU-feasible, e.g., a small distilled
  classifier) becomes available and a word-level-labeled English dataset
  larger than what exists today becomes accessible, §3.4's learned tier
  becomes a much stronger candidate — this was already true in principle
  before this review and remains the recorded long-term direction
  (`ROADMAP.md`, "the deferred learned tier").
- If Track B's ASR-attributable share (currently 64.9%, §2) is
  re-measured against a second ASR backend or real (non-synthetic)
  disfluent speech (`ROADMAP.md` item 10, already planned) and turns out
  to be even higher or lower, that should directly resize how much
  Phase 3 effort goes toward the acoustic-corroboration extension in §5
  vs. other items.
- If the Whisper-encoder-reuse prototype (§5), once actually
  pre-registered and evaluated, fails to show the same kind of
  simultaneous-precision-and-recall improvement the prolongation redesign
  did, that is itself evidence worth recording — not a reason to escalate
  straight to a full SSL model, per this review's own reasoning in §4.

## 7. Explicit non-recommendations, and why

- **Do not adopt an end-to-end audio→region model** (YOLO-Stutter/
  Stutter-Solver/SSDM-class). Unchanged from the 2026-08-03 decision;
  reinforced by SSDM 2.0's increased complexity (§3.7).
- **Do not train a from-scratch or fully fine-tuned self-supervised
  classifier** (wav2vec2/HuBERT/WavLM) as a wholesale replacement right
  now. Real infrastructure gap (§4), and the field's own reported results
  at comparable granularity aren't decisively ahead of this project's
  current approach once the shared precision problem is accounted for
  (§3.4).
- **Do not attempt joint ASR+detector fine-tuning** (§3.8). The most
  scientifically interesting finding in this review, but the largest
  infrastructure and data lift of any option considered, and this
  project has no joint-labeled (transcript + word-level disfluency)
  English dataset to train it on even if the pipeline existed.
- **Do not pursue CTC forced alignment or a different core ASR
  checkpoint.** Already considered and declined on their own merits
  (§3.1, §3.3); nothing in this pass overturns those specific
  conclusions.

## 8. Adversarial self-review (2026-08-04, same day, before any pre-registration)

The project owner explicitly asked for §5.1's recommendation to be
challenged from a clean-slate design stance — not "does this justify our
architecture" but "if maximizing disfluency-detection accuracy were the
only goal, would encoder-hidden-state reuse still be the right call, or
would a purpose-built self-supervised encoder (wav2vec2/HuBERT/WavLM/
SeamlessM4T) win outright." This section does that honestly, including
where it found a real gap in the prior reasoning.

### 8.1 The strongest objection to §5.1, stated as sharply as possible

**Whisper's encoder is trained under a fundamentally different pressure
than a self-supervised encoder, and that pressure plausibly works against
this project's exact goal.** Whisper's encoder exists to help the decoder
produce the correct *token sequence* — its training signal has every
incentive to compress away exactly the kind of continuous, non-lexical
acoustic variation (a prolongation's exact duration and voice quality, a
block's tension, a hesitation's prosody) that doesn't change *what word*
was said, only *how* it was said. Self-supervised objectives (WavLM's
masked prediction with a denoising/speaker component, wav2vec2/HuBERT's
masked contrastive/cluster prediction) have no equivalent pressure to
discard that information — if anything, WavLM's denoising objective
specifically trains it to model the acoustic signal in noisy/degraded
conditions, closer to the "the audio is compromised, model it carefully"
task disfluency detection actually is. **This is not a minor
consideration — it is a real, mechanistic reason encoder-hidden-state
reuse could underperform a purpose-built encoder specifically for the
kind of continuous, paralinguistic signal §3.6's per-type WER findings
say matters most.** §5.1's recommendation did not weigh this seriously
enough; this section corrects that.

### 8.2 Checking the objection against evidence, not just accepting it

- **A direct, controlled, head-to-head comparison was found**: "Residual
  Speech Embeddings for Tone Classification" (arXiv:2502.19387) evaluates
  Whisper, WavLM, HuBERT, and wav2vec2 embeddings on the *same*
  paralinguistic task (tone classification) with identical methodology.
  Result: **WavLM (0.98 logistic regression / 1.00 random forest accuracy)
  and Whisper (0.97 / 0.96) perform comparably, both clearly ahead of
  HuBERT (0.87 / 0.86).** This directly weakens the strong version of
  §8.1's objection: on this evidence, Whisper's encoder is not
  dramatically worse than a purpose-built SSL encoder for paralinguistic
  content in general — it's in the same tier as the best-performing SSL
  option tested, not a clearly inferior class of representation. **Real,
  stated caveat**: this comparison is tone classification, not
  disfluency, on a controlled single-speaker synthetic dataset — it is
  suggestive, not decisive, for this project's actual task.
- **No direct head-to-head on stuttering/disfluency specifically was
  found**, for either the Whisper-encoder result (arXiv:2406.05784,
  clip-level, F1=0.88/0.85/0.87) or the best SSL word-level result
  (arXiv:2409.10704, WavLM Large + HConv + CTC, word-level F1=0.554) —
  **these two numbers are not comparable to each other** (different task
  granularity: clip-level multi-label vs. word-level localization) and
  neither paper benchmarks against the other's exact representation on
  the same task. **This is a genuine, honestly-stated evidence gap that
  this review does not resolve** — anyone tempted to read §8.2's tone-
  classification comparison as settling the disfluency-specific question
  should not.

  **Partially closed, 2026-08-04, by this project's own Stage 1 result**
  (`VALIDATION.md` §11.6): CrisperWhisper's own last-layer encoder
  embedding separates true from false `word_repetition`/`Any` detections
  with a large, stable effect size (Cohen's d = +1.047/+1.116, 90-clip
  sample). This is not the same head-to-head this section wished for —
  it doesn't compare Whisper's encoder against WavLM/wav2vec2/HuBERT on
  the *same* data — but it does directly answer the narrower, more
  practically important question this project actually needed answered:
  whether Whisper's encoder specifically carries *any* disfluency-
  relevant signal at all, on this project's own real data. It does. The
  broader comparative question (would WavLM do even better) remains
  genuinely open and is exactly what Stage 1b was reserved for — but
  Stage 1b's trigger condition (a weak/null Stage 1 result) did not
  occur, so it is not currently justified by this project's own
  evidence-gated plan (`ROADMAP.md` item 17).
- **SeamlessM4T, checked directly per the owner's request**: its speech
  encoder is itself a w2v-BERT 2.0/Conformer self-supervised model
  (pretrained on 4.5M hours), not a fundamentally new representation
  family — architecturally it's closer to "a bigger, translation-oriented
  wav2vec2-style model" than a distinct option. It does have dedicated
  prosody/expressivity adapters, which is theoretically interesting, but
  no evidence of any kind (disfluency-specific or general paralinguistic)
  was found for it, and it is a substantially heavier, translation-first
  model with no track record in this literature. **Not a stronger
  candidate than WavLM given the evidence available**, and not pursued
  further.
- **Newer Whisper-family checkpoints** (large-v3-turbo, distil-whisper):
  these are speed/efficiency optimizations of the same encoder-decoder-
  for-transcription objective §8.1's objection applies to — nothing found
  suggests any of them changes the fundamental training-pressure argument
  either direction. Not a resolution to §8.1's concern in either
  direction.

### 8.3 What this changes, and what it doesn't

**§8.1's objection survives as a real, unresolved, mechanistically
plausible concern — this review does not dismiss it.** But acted on
honestly, given genuine uncertainty and no direct disfluency-specific
head-to-head, the correct response is not to pick a winner by
theoretical argument alone (that would repeat exactly the mistake this
project's whole methodology exists to avoid — see `CLAUDE.md` standing
rule 3), and not to reflexively keep the original answer either. **The
scientifically honest move is to design Stage 1 so that a weak or null
result is itself informative evidence for §8.1's concern, and make the
escalation path explicit and evidence-gated rather than "someday
reconsider."**

**Refined staged plan, replacing §5.1's plan with an explicit escalation
trigger:**

1. **Stage 1 (unchanged from §5.1, test first, zero training, zero added
   latency)**: Whisper's own last-layer encoder hidden states, as a
   non-parametric corroboration signal. Cheapest possible test, and
   specifically the right *first* test because a weak result here is
   real evidence for (not just consistent with) §8.1's mechanistic
   concern — it isn't wasted effort even if it fails.
2. **Explicit escalation trigger, stated before Stage 1 runs (so this
   isn't decided after seeing results)**: if Stage 1 shows no meaningful
   TP/FP separation on the pre-registered metric, that specific,
   measured failure is the evidence that justifies paying the real added
   cost of Stage 1b below — not a vague "consider a learned tier
   someday."
3. **Stage 1b (new, evidence-gated, only triggered by Stage 1's
   failure)**: a frozen **WavLM-Large** forward pass (chosen over
   wav2vec2/HuBERT specifically because it has both the best published
   word-level stuttering result found in this review, arXiv:2409.10704,
   and a comparable-to-Whisper showing in the one direct paralinguistic
   comparison found, §8.2) as a genuinely new, second model pass —
   honestly priced as a real new cost this project doesn't pay today: a
   new model to load and maintain, and added latency on top of the
   already-documented 54-102s/clip CPU ASR cost (`ARCHITECTURE.md` §3,
   §7). This is the point at which the "clean slate" answer and the
   "given this project's real constraints" answer would diverge, and
   that divergence is now named explicitly rather than smoothed over.
4. **Stage 2 (unchanged from §5.1)**: a small trained classification
   head, only over whichever representation (Stage 1's Whisper encoder,
   or Stage 1b's WavLM if triggered) shows real signal, and only with
   explicit go-ahead.

**If designing this system from scratch today, with zero regard for this
project's existing infrastructure, the honest answer is: a frozen
WavLM-Large pass would likely be the theoretically stronger starting
choice**, given it has no competing training objective diluting its
sensitivity to continuous acoustic signal, and the best published
word-level result found in this exact task. **But that is not the same
question as "what should this project do first"** — this project has no
GPU, is already latency-constrained, and its own standing methodology
requires measuring before committing to added cost. Stage 1 is not a
defense of the status quo; it is the cheapest experiment that produces
real evidence either for encoder-reuse being sufficient, or for exactly
the added cost of Stage 1b being justified — which is a stronger basis
for that decision than either theoretical argument reached in this
section alone.

## 9. Corroboration-mechanism review: given Stage 1's signal, how should it actually be used? (2026-08-04)

Stage 1 (`VALIDATION.md` §11) confirmed CrisperWhisper's encoder carries
real signal (Cohen's d = +1.05/+1.12 for `word_repetition`/`Any`). It did
not decide *how that signal should be turned into a detection decision* —
the zero-training distance-to-centroid measure used to *test* the signal
was a measurement choice, not a commitment to how the signal ships. The
project owner explicitly asked this be re-opened from first principles,
against the project's actual objective (accurate detection/classification/
localization feeding a reliable downstream correction module), not
defaulted toward either the simplest or the most sophisticated option.

### 9.1 The real decision, stated precisely

Two questions were conflated in the Stage 1 framing and need separating:

1. **Which signal computation?** Distance-to-fluent-centroid (what Stage 1
   tested) is one specific way to turn an embedding into a scalar. It is
   not the only one, and Stage 1's positive result doesn't validate it as
   *the* right one — only that *some* signal derivable from this
   embedding separates TP from FP.
2. **Which decision mechanism consumes that signal?** A fixed global
   threshold, a per-clip/per-speaker relative threshold, or a trained
   classifier are three different ways to turn a scalar (or a richer
   feature set) into a detection/confidence decision — orthogonal to
   which signal computation feeds them.

Collapsing these into "threshold vs. classifier" undersells the actual
design space. §9.2 below treats them as two separable axes.

### 9.2 The candidate space, honestly enumerated

**Signal computation candidates:**

- **(S1) Distance-to-fluent-centroid** — what Stage 1 measured. Direct
  evidence: the d=+1.05/+1.12 result itself.
- **(S2) Repeat-pair self-similarity** (new, not previously evaluated) —
  for `word_repetition`/`sound_repetition` specifically (not applicable
  to `filler`, which has no second instance to compare against): instead
  of comparing a candidate span to the clip's fluent baseline, compare
  the two repeated instances' embeddings *to each other*. Motivation,
  stated as a hypothesis, not a finding: a genuine disfluent re-attempt
  might be acoustically *more* self-similar to its own repeat (same
  articulatory struggle pattern) or *less* similar (a corrupted second
  attempt) than two coincidental, independently-produced same-word
  occurrences — either direction would be informative, and neither has
  been tested. **Not evaluated in Stage 1**, and not preferred over S1 by
  default — named here specifically because the owner asked not to let
  the first tested design foreclose considering others.

**Decision-mechanism candidates, applicable to either signal:**

- **(M1) Fixed global threshold** — one calibrated constant, the same
  pattern as `prolongation_min_seconds`/`pitch_std_max_hz`/
  `block_gap_seconds` today. Zero training in any sense; the constant
  itself would be chosen by a documented, reproducible procedure (e.g.
  the value maximizing F1 on a held-out split), not hand-picked.
- **(M2) Per-clip/per-speaker relative threshold** — same signal, but the
  threshold adapts to within-clip or within-speaker distribution (e.g. a
  z-score or percentile against the clip's own fluent-token spread),
  the same relative-calibration principle `calibration.py` already
  applies to `block_gap_seconds`/`prolongation_min_seconds` today. Still
  zero gradient-based training; marginally more engineering than M1
  (needs the within-clip distribution, which Stage 1's `fluent_centroid`
  computation already produces as a byproduct).
- **(M3) A small trained classifier** (linear/logistic, matching
  arXiv:2406.05784's own recipe) over the embedding, or over a small
  feature set (distance, possibly combined with existing rule-based
  confidence and/or Praat features). The originally-envisioned "Stage 2."

### 9.3 Evaluated against the dimensions the owner named, without assuming an answer

- **Performance**: genuinely unknown *in advance* which wins — this is
  the reason to measure, not argue. But Stage 1's own effect size gives a
  real, principled reason to test M1/M2 *first*: a Cohen's d this large
  (>1.0) means the two distributions are well-separated with modest
  overlap, the regime where a simple threshold typically captures most
  of the available separation — a classifier's marginal gain *over a
  well-chosen threshold* shrinks as effect size grows. This is a reason
  to measure the threshold's own ceiling before assuming a classifier is
  needed to reach it, not a reason to skip measuring the classifier.
- **Robustness to ASR errors**: does not differentiate M1/M2/M3 directly
  — all three consume the same encoder embedding, so whatever robustness
  (or fragility) the embedding itself has to ASR/acoustic variation
  applies equally to whichever mechanism sits on top of it. It *does*
  differentiate by training-data dependence: M3, trained on LibriStutter's
  reconstructed-timing clips specifically, risks learning
  LibriStutter-specific artifacts this project has repeatedly flagged as
  a real limitation (§8.2, §9.4, §9.5.1's own frozen-baseline caveats) —
  a threshold calibrated with a documented margin is less able to exploit
  (or be fooled by) such dataset-specific quirks, for better or worse.
- **Localization accuracy**: does not differentiate M1/M2/M3 — none of
  the three changes *where* an event's span is (token-path timestamps or
  the acoustic-native span-finder already own that). All three *can*
  change *which* events survive to be scored for localization, the same
  side effect the Praat-gating change already produced (§9.5.1:
  surviving TPs' localization dropped 0.857→0.500) — worth measuring
  again regardless of which mechanism is chosen, not a differentiator
  between them.
- **Computational cost**: near-identical at inference time. M1/M2 are a
  comparison against a constant; M3 (if linear/logistic, matching the
  one directly-evidenced published recipe) is also effectively one dot
  product. The real cost difference is at *build* time: M1/M2 need a
  calibration pass over already-collected data (cheap); M3 needs an
  actual training loop, a disciplined train/val split (with the same
  leakage risk SEP-28k-E's own paper was designed around, `VALIDATION.md`
  §4 point 5), and produces a versioned artifact to maintain.
- **Maintainability**: M1/M2 fit this project's existing maintenance
  model exactly (a config constant, or a small formula, both already
  used throughout `detect.py`/`calibration.py`). M3 would be **the first
  internally-trained, shipped model artifact this project has ever
  had** — CrisperWhisper and Silero VAD are externally pretrained,
  used zero-shot; there is no existing process here for model
  versioning, retraining triggers, or drift monitoring. This is a real,
  categorical, not incremental, new maintenance burden.
- **Interpretability**: a genuine weakness shared by *all three*
  mechanisms relative to this project's existing acoustic signals, worth
  naming plainly rather than only comparing M1/M2/M3 against each other:
  "cosine distance in a 1280-dimensional embedding space = 0.61" has no
  independent physical meaning the way "jitter = 2.1%, exceeds clinical
  threshold" does — unlike Praat/VAD's signals, none of S1/S2 map onto a
  concept a speech-language pathologist already understands. Within that
  shared weakness, M1/M2 are marginally more inspectable than M3 (the
  evidence string can state the actual distance and threshold; a
  classifier's decision is harder to summarize in one sentence), but the
  bigger, shared cost applies regardless of which mechanism is chosen —
  this is a cost of using this signal *at all*, not specifically of
  picking M3.
- **Reproducibility**: M1/M2, calibrated via a documented deterministic
  procedure, are exactly as reproducible as this project's other
  empirically-set thresholds. M3 adds real, if manageable, reproducibility
  surface (random seeds, exact training data/order, framework-version
  pinning) that this project has not needed to manage for anything else
  it ships.
- **Engineering complexity**: M1 < M2 < M3, unambiguously, in new code
  and new failure modes.
- **Long-term scalability**: the one dimension that structurally favors
  M3 *eventually* — if more signals accumulate later (cross-attention,
  decoder confidence, Praat features extended to repetition types), a
  learned combiner scales more gracefully than hand-tuning an increasing
  number of independent thresholds against each other. **This is a real
  consideration for the future, not a reason to build M3 now**: Stage 1
  has validated exactly one new signal, not yet a multi-signal
  combination problem — acting on a scalability concern before the
  problem it addresses actually exists would be exactly the complexity-
  as-default the owner asked this review to avoid.

### 9.4 Should different disfluency types share one strategy?

**The premise that they currently do is already false, worth correcting
explicitly.** `block` uses `gap_is_silent()` (RMS-threshold silence
detection); `prolongation` uses Praat pitch/jitter/shimmer as a hard gate
(since Phase 2, §9.5.1); `filler`/`stutter_marker` use voiced-energy
presence. The *signal source* has always been type-specific; only the
*fusion architecture* (weighted-confidence combination between token-path
and acoustic-native candidates) is shared. The real question Stage 1
raises is narrower than "should types share a strategy" (they already
don't) — it's "does `word_repetition`/`sound_repetition` get a new,
type-appropriate signal source (S1 or S2 above) added to that same
shared fusion architecture," which is a natural extension of the existing
pattern, not a departure from it. `filler` has zero informative Stage 1
samples (§11.6) — extending this signal to `filler` is not supported by
current evidence either way and should not be assumed to transfer.

### 9.5 Conclusion: multiple candidates remain genuinely plausible — resolved by pre-registered comparison, not by this review's argument alone

This review does not pick a winner, deliberately: §9.3's own dimension-by-
dimension analysis shows real, legitimate arguments on more than one
side (M1/M2's fit with existing practice and lower risk vs. M3's
long-term scalability; S1's direct evidence vs. S2's untested but
mechanistically plausible alternative). Picking one now would repeat the
exact mistake §8's adversarial review already corrected once this
session — reaching a conclusion by argument where a measurement is
available instead. **At least three combinations remain worth measuring
before choosing**: (S1, M1) as the cheapest baseline; (S1, M3) to
measure the classifier's actual marginal gain over that baseline, not an
assumed one; and (S2, M1) to test whether the untested alternative
signal is competitive with S1 at the cheapest mechanism before spending
effort on it at a more expensive one. See `VALIDATION.md` §12 for the
pre-registered comparison protocol.

---

This review is a decision record, not a living document — if a later
review revisits this question with new evidence, it gets its own dated
section or file, same append-only spirit as `PAPER_DECISION_LOG.md` and
`PHASE_2_RESEARCH_PLAN.md`. See `PAPER_DECISION_LOG.md`'s 2026-08-04 entry
for this review for the condensed record, and `ROADMAP.md` for where the
Whisper-encoder-reuse candidate now sits in Phase 3's priority list.
