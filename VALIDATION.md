# VALIDATION.md — evaluation methodology, datasets, and results

**Status as of 2026-08-05 (later the same day): the shipped result
(below) re-validated against real ASR, and a separate research track
opened from what that found.** §14/§14.1 re-ran the shipped repetition-
classifier gate for the first time against real ASR output rather than
the ground-truth-transcript tokens every number in the paragraph below
was measured on: the gate's own mechanism transferred safely, but its
real-world impact was negligible, because real ASR produces almost no
`word_repetition`/`sound_repetition` candidates to gate in the first
place. That finding — evidence the ASR stage itself, not the detector,
may be the ceiling for these two types — was judged a bigger-than-one-
finding checkpoint and opened its own research track, worked on its own
`asr-research` branch (`main` untouched): see `ASR_RESEARCH_TRACK.md`
for that track's full methodology and results (Stages A-C, all done as
of this session's close) — not duplicated in this file, since it
investigates the ASR stage's representation, not this project's existing
detector-evaluation methodology.

**Status as of 2026-08-05 (morning): Phase 3's first shipped result.**
Since the
Phase 2 close described below, a first-principles architecture review
(`PHASE_3_ARCHITECTURE_REVIEW.md`) kept the ASR-first two-stage
architecture and identified one scoped extension, carried through to a
shipped result via the same pre-register → measure → decide discipline:
(1) Stage 1 (§11) confirmed CrisperWhisper's own encoder embedding
carries real disfluency signal beyond the transcript (Cohen's d =
+1.05/+1.12, `word_repetition`/`Any`, stable across 30→90 clips); (2) a
corroboration-mechanism comparison (§12) found a small trained classifier
clearly beats a zero-training threshold on the same signal (the opposite
of this project's own pre-registered prediction, reported as such), and
that the result held and grew at 2.8x the sample with properly-tuned
regularization (§12.6.2); (3) the classifier was implemented, benchmarked
honestly via out-of-fold cross-validation, and shipped as the new default
(§13): `Any` (word+sound repetition) F1 0.631→0.890. This is this
project's first internally-trained, shipped model artifact — a real,
accepted new category of cost (versioning, no established retraining
process yet) and a real, named limitation not yet resolved (added
live-app latency, `§13.2`, `ROADMAP.md` item 18). All numbers below that
predate this are kept exactly as measured at the time.

**Status as of 2026-08-04 (end of day): Phase 2's detector-side and
measurement-infrastructure work is complete.** Since the 2026-08-03
baseline described below, four more evidence-driven changes landed, each
pre-registered and measured the same way: (1) `sound_repetition`'s
fragment-ordering bug fixed, recall 0.000→0.920 (§8.2.1); (2) a candidate
"word-sandwiched repetition" extension was built, measured, and
**reverted** as a documented negative result (§8.4.4); (3) the
prolongation redesign — Praat-feature gating adopted as the new shipped
default (`Any` F1 0.835→0.888, prolongation F1 0.064→0.084), rate-
normalization tested and rejected (§9.5.1); (4) a confidence-sensitive
metric and Wilson confidence intervals were built and run against real
data, closing §9.3's open question with a negative-to-null result for
VAD/Praat's confidence effect (§9.3.1) and making this project's own
extreme-small-n recall claims concrete (§8.4.3 addendum). All numbers
below that predate 2026-08-04 are kept exactly as measured at the time —
see each section's own dated addenda for what has since changed, rather
than silently rewriting history.

**Status as of 2026-08-03 (end of day): this project's first established
research baseline.** `profiling/evaluation/` supports both text-only and
audio-enabled Track A evaluation (LibriStutter, word-level, real
downloaded data). Two real results recorded against the same 499 real
clips: text-only (§8.2) and, for the first time, **audio-native-layer
evaluation** (§8.3) — the only measurement to date of whether the 2026-08
audio-native architecture change (Silero VAD, Praat, weighted fusion)
actually helps. Headline: `Any`-label F1 0.773 → 0.835 with audio, a real
precision gain at negligible recall cost — audited, not just read off a
table (a first attempt was discarded after a direct audit caught it
measuring silent audio due to a `soundfile` decode bug; see
`PAPER_DECISION_LOG.md`). SEP-28k is loader-ready (clip-level schema
confirmed against the real, complete labels file) but has zero scorable
results — no reference transcript, so nothing can run against it without
real audio, which is deliberately not yet acquired (§6 resequencing note).
Track B (steps 5–6) not started.

**Ablations (§9) complete against this baseline.** Ranked by measured
contribution to `Any` F1: `prolongation_min_seconds` dominates by an order
of magnitude (0.294 F1-point range across the sweep; current 1.0s setting
is not the optimum for either aggregate or prolongation-specific F1);
`fusion_weights.acoustic` has a small real effect (+0.003 F1, saturating at
2×); Silero VAD and Praat corroboration both measured **zero** effect on
this presence/absence metric — traced to a real methodological finding
(§9.3), not evidence they don't help; speaker calibration is not
applicable to this dataset. See §9 for the full tables and the evidence-
based next-phase recommendation this produced.

**Track B pilot complete, scaled, and — as of 2026-08-04 — confirmed
across full speaker diversity (§8.4/§8.4.1/§8.4.2/§8.4.3), protocol
pre-registered in §5.1 before any code was written.** 30 clips (pilot),
90 clips (7 speakers, clip-count scaling), then 120 clips spanning **all
40 distinct speakers** (speaker-stratified sampling, §8.4.3) — real
CrisperWhisper ASR, alignment-based scoring, hand-verified against 10
clips' transcripts (methodological gate passed). Headline: Track A's ~99%
recall drops to **~6–15%** under real ASR conditions — the single most
important finding of this project's evaluation work to date, and the
clearest evidence yet that Track A alone was measuring something that does
not transfer to real deployment. That absolute drop is not in question and
has not moved.

**The *attribution* of that gap was refined, not simply confirmed, once
speaker diversity was accounted for — this is a genuine scientific
correction, recorded here rather than smoothed over.** At the 7-speaker
samples (n=2 at 30 clips, n=7 at 90 clips), context-strict scoring
(`R_B|preserved_ctx1`) gave a clean 1.0 recall, read at the time as "the
detector is effectively perfect given intact input" (~0% detector-
attributable / ~100% ASR-attributable). At full 40-speaker diversity (120
clips, §8.4.3), `R_B|preserved_ctx1` recall is **0.667 (10/15)**, and the
decomposition moves to **35.1% detector-attributable / 64.9%
ASR-attributable**. ASR-fidelity remains the majority driver — the
headline does not reverse — but "the detector is essentially perfect given
fair input" was too strong a claim at n=7. **Traced by hand, the correction
has a precise, satisfying mechanism, not a mystery**: all 5 of the misses
in the larger sample are `sound_repetition` or `phrase_repetition`
instances, both types with *already-known, already-documented* structural
gaps unrelated to ASR (§8.2's fragment-ordering mismatch and
LibriStutter-reconstruction limitation, respectively) — **`word_repetition`
itself remains at 100% Any-level recall given intact context, across 10
instances and two independent sample-construction methods.** The earlier
small samples happened, by chance, to be `word_repetition`-heavy; the
larger, speaker-diverse sample exposed the same pre-existing gaps Track A
had already found, now confirmed to matter under real ASR conditions too.
See §8.4.3 for the full numbers, the hand-traced hit/miss breakdown, and
what this does and doesn't change about where development effort belongs.

This is the project's evaluation reference: how we measure whether the
disfluency detector actually works, against what, and — once runs start
happening — what we found. Methodology sections (§1–§7) are stable once
agreed; the results sections (§8 onward) are meant to be edited every time
an evaluation run happens, however small. See `DOCS.md` for how this file
relates to the others.

---

## 1. Why dataset-based evaluation

Two informal real-microphone recordings (logged in
`PAPER_DECISION_LOG.md`, 2026-08-03) surfaced real findings but could not
answer the central question — which flagged events are correct, which are
false positives, and which real disfluencies were missed — because there was
no ground truth to check against, only one speaker's own impression of two
clips. That's not a sufficient basis for tuning thresholds or claiming
accuracy, and it's not how any of the literature reviewed for this project
(SEP-28k, FluencyBank, KSoF, LibriStutter, YOLO-Stutter, Dysfluent WFST, the
2025 rule-based and comparative-architecture studies — see `ROADMAP.md` and
`PAPER_DECISION_LOG.md` for full citations) establishes or reports accuracy.
Public labeled datasets are the only credible path to numbers "comparable
with published research."

Two things stay useful *alongside* dataset evaluation, not instead of it:
- **Informal real-mic spot checks** — good for qualitative sanity checks and
  catching things no dataset will (a UI display bug was found this way this
  round), never for accuracy claims.
- **Reporting per-dataset, never a single blended number** — no single
  dataset covers this app's full 7-type taxonomy (§3), so a combined score
  would hide which types are actually validated.

---

## 2. Datasets — comparison and priority

| Dataset | Speech | Taxonomy coverage of this app's 7 types | Acquisition | Priority |
|---|---|---|---|---|
| **LibriStutter** | Synthetic (LibriSpeech + injected disfluency artifacts) | filler, sound_repetition, word_repetition, phrase_repetition, prolongation — **not block, not stutter_marker** | Direct download (GitHub `hhzhang16/LibriStutterData` / Borealis Dataverse), audio included, permissive license, no clinical-data handling | **Tier 1 — do first** |
| **SEP-28k** (+ bundled FluencyBank clips) | Real (podcast speech) | block, prolongation, sound_repetition, word_repetition, filler — **not phrase_repetition, not stutter_marker** | Labels free/direct (`apple/ml-stuttering-events-dataset`); **audio not bundled** — fetch via `download_audio.py`/`extract_clips.py` from podcast-owner URLs (~32GB raw / ~2.6GB clipped); some URLs may have rotted since 2021. Non-commercial research use; copyright stays with podcast owners. SEP-28k-E (`th-nuernberg/ml-stuttering-events-dataset-extended`) adds a speaker-exclusive train/dev/test partition — prefer this over vanilla SEP-28k once acquiring data. | **Tier 2 — do second** |
| **KSoF** (Kassel State of Fluency) | Real (German stuttering-therapy sessions) | block, prolongation, sound_repetition, word_repetition, filler, + a fluency-shaping label this app has no equivalent for | Request-access; German (this app's filler word list, CMU-dict phonetic lookups, etc. are English-only) | Tier 3 — stretch, only if cross-lingual/therapy-context validation becomes an explicit goal |
| **UCLASS** | Real (SLP-annotated) | block, prolongation, repetition (older/less standardized annotation format) | Available, smaller/older | Tier 3 — secondary real-speech cross-check |

**No dataset reviewed validates `stutter_marker`** as this app defines it
(an ASR-level sub-word fragment marker) — this is an inherent, dataset-driven
coverage gap, not a bug to chase. State it explicitly in every results
report rather than silently omitting the type.

---

## 3. Two independent axes: transcript source, and audio availability

This app is a two-stage pipeline (ASR → detector), and its detector is
itself audio-native-primary (2026-08 restructuring — text/timing checks
fused with an audio-native layer: Silero VAD, Praat, weighted
acoustic-vs-token fusion). A labeled dataset provides a ground-truth
transcript + timestamps, and sometimes audio. Evaluating against it varies
along **two independent axes**, not one:

- **Transcript source**: the dataset's own ground-truth words (bypasses our
  ASR — "Track A"), or our own CrisperWhisper output on the dataset's raw
  audio ("Track B" — built 2026-08-03, §5.1/§6/§8.4; needed a real
  hypothesis-to-reference alignment step, `alignment.py`, §5).
- **Audio availability**: whether the real waveform is also passed to
  `detect_disfluencies()`, activating its audio-native layer, or omitted
  (text/timing-only).

These were originally conflated (Track A's first implementation hardcoded
`audio_bytes=None`), which meant Track A could only ever test the
text/timing half of the detector — the audio-native layer, this project's
main architectural contribution, had never been evaluated against any
ground truth at all. Fixed 2026-08-03 (see `PAPER_DECISION_LOG.md`,
"Audio-enabled evaluation"): `LabeledClip.audio_bytes` is now optional and,
when a loader populates it (real audio decoded and paired with the
ground-truth annotations), Track A passes it straight through. The same
`track_a.evaluate()` function now serves both text-only and audio-enabled
runs — which one runs is decided by which loader/CLI flags were used, not a
second code path.

Both axes matter and results are always reported labeled by which
combination produced them, never blended:

| | No audio | With audio |
|---|---|---|
| **Ground-truth transcript (Track A)** | Isolates detector logic from transcript noise; audio-blind, cannot measure the audio-native layer at all. Built, run — §8.2. | Detector logic **and** the audio-native layer, transcript noise still isolated out. The most direct test of "does the audio-native architecture help," since ground truth removes ASR as a confound. Built, run — §8.3. |
| **Our own ASR (Track B)** | Not meaningful (Track B's whole point is testing with real audio). | *What a real user actually experiences* end-to-end. **Built and run 2026-08-03** — see §5.1 (protocol), §6 (implementation), §8.4/§8.4.1/§8.4.2 (results, 90 clips). |

Report every number labeled by which cell of this table produced it.

---

## 4. Metrics

Extends `profiling/evaluate.py`'s existing per-type TP/FP/FN core (already
built and unit-tested against hand-computed expectations):

1. **Precision / recall / F1 per type**, plus a combined **binary "Any"
   label** (disfluent vs. clean) matching SEP-28k's own paper's reporting
   convention, for direct comparison against published baselines.
2. **Per-type 2×2 confusion matrices** (TP/FP/TN/FN). Not a single N×N
   multi-class matrix — disfluencies co-occur (Bayerl et al., "A Stutter
   Seldom Comes Alone," Interspeech 2023), and forcing multi-label data into
   single-label confusion matrices is a documented methodological error in
   this literature.
3. **Localization accuracy (IoU ≥ 0.5)** — not present in the harness today
   (current matching is word-index-exact, a type-correctness proxy, not a
   temporal metric). Uses `acoustic_start`/`acoustic_end` when present
   (correctly surfaced in the UI as of the 2026-08-03 display fix), else the
   token's nominal span, against the ground-truth span.
4. **Equal Error Rate (EER)** — SEP-28k's own paper reports this alongside
   F1; legitimate since events already carry a confidence score. Add after
   1–3 are solid, not a blocker.
5. **Speaker-exclusive splits.** Any threshold decision made in response to
   evaluation results must be checked against held-out speakers/clips, not
   the ones used to eyeball the change — directly relevant since the prior
   prolongation-threshold tune (`PAPER_DECISION_LOG.md`, Part D) was based on
   a small, non-held-out real-audio sample.

---

## 5. Preprocessing / alignment work required

- **Per-dataset loaders**, following the `load_libristutter_csv` pattern
  already in `profiling/evaluate.py`: `load_sep28k_labels`, `load_ksof_labels`
  (stretch) — all normalizing into the same `LabeledClip` shape, with an
  explicit per-dataset label→taxonomy mapping table (§2) rather than silently
  dropping unsupported types.
- **Audio format**: SEP-28k/FluencyBank ship WAV after `extract_clips.py` —
  compatible with this app's existing pipeline, no new dependency.
  LibriStutter ships FLAC; Track A sidesteps this (timestamp-only) but
  Track B/acoustic-fusion evaluation needed FLAC decoding — **built**
  (`_flac_bytes_to_wav16k`, §8.3), including a real bug found and fixed
  (`soundfile`'s `dtype="int16"` silently producing silent audio — see
  `PAPER_DECISION_LOG.md`).
- **Track B's ASR↔reference alignment** is real work, not `zip()`: our ASR's
  word sequence differs in length/content from the reference whenever it
  mis-transcribes. Uses a modified-cost sequence alignment (Levenshtein/DTW
  between hypothesis and reference words, biased so substitutions land on
  fluent words rather than the labeled disfluent ones) rather than a naive
  positional match — this is an established technique in the ASR/disfluency
  literature, not a bespoke invention. **Built** as `alignment.py`,
  §5.1/§6, verified by hand on 10 clips (§8.4) before being trusted.
- **Reproducibility**: every run recorded (config used, dataset version/date,
  git commit, count of successfully acquired clips for URL-based datasets) as
  a timestamped result file, not just printed — otherwise "F1 improved from X
  to Y" can't be trusted months later.

### 5.1 Track B evaluation protocol (pre-registered 2026-08-03, before implementation)

**Written and locked in before `alignment.py`/`track_b.py` exist**, per the
project owner's explicit instruction — the point of a protocol is that it's
fixed in advance, not adapted once results are visible. Any deviation from
this section, discovered necessary once real implementation starts, gets
recorded as a dated addendum here (with reasoning), never a silent edit —
same append-only discipline as `PAPER_DECISION_LOG.md`.

**Scope of the pilot**: ~20–40 real LibriStutter clips (a subset of the
existing 499-clip sample, same clips already used for §8.2/§8.3/§9 — chosen
before running, not cherry-picked after seeing which ones look good), full
pipeline (CrisperWhisper ASR on the real audio → `detect_disfluencies()` on
our own transcript → alignment back to ground truth). Deliberately small:
CrisperWhisper inference is 54–102s/clip on this hardware (measured,
`ARCHITECTURE.md` §3), so a full 499-clip Track B run is a multi-hour
commitment better justified after the pilot validates the method.

**1. What we align, and how**

For each clip, align the ASR hypothesis word sequence against the
ground-truth reference word sequence (LibriStutter's reconstructed tokens,
same ones used in Track A) via **word-level Levenshtein alignment**
(dynamic-programming edit distance with backtrace, the standard WER-style
technique — not a bespoke algorithm). Every reference word is classified
into exactly one alignment outcome:

- **Correct** — the aligned hypothesis word matches (case/punctuation-
  insensitive, same normalization `detect.py`'s `_norm()` already uses).
- **Substitution** — a hypothesis word aligns to this position but doesn't
  match.
- **Deletion** — no hypothesis word aligns here at all (ASR dropped it).
- (**Insertion** — a hypothesis word with no reference counterpart; tracked
  for completeness but not part of the per-reference-word classification
  above.)

**Cost-function bias, and why**: substitution cost against a reference word
that carries a ground-truth disfluency label is set *higher* than against a
clean word (a configurable multiplier, not a hardcoded assumption — default
1.5×). This directly implements the technique found during this project's
literature review (a modified-cost ASR alignment that biases *away* from
forcing weak substitution matches onto disfluent words, since real ASR is
known to smooth disfluencies into deletions rather than mangled
substitutions) — see `PAPER_DECISION_LOG.md`'s "Vision alignment review"
entry for the citation. Without this bias, a standard aligner can produce a
low-cost alignment that superficially "explains" a disfluent word via a
coincidentally-similar substitution, understating how often ASR actually
loses disfluent content — exactly the failure mode this protocol exists to
avoid pre-empting with an unbiased-looking but actually-misleading
algorithm.

**2. Metrics reported**

Everything reported at **three levels**, always labeled which is which —
never blended into one number:

- **(a) Track A** (already have — §8.2/§8.3): per-type + `Any`
  precision/recall/F1/confusion-matrix/IoU-localization, `audio_bytes=None`
  or real audio, ground-truth transcript, no ASR involved.
- **(b) Track B, ASR-preserved subset** — score `detect_disfluencies()`'s
  output *only* over ground-truth disfluent words classified **Correct** by
  the alignment (§5.1.1). This isolates detector behavior specifically on
  real (not reconstructed) ASR output, for exactly the words ASR didn't
  lose — the fairest apples-to-apples comparison against Track A's logic,
  with real-audio noise but without ASR-loss noise.
- **(c) Track B, overall** — score `detect_disfluencies()`'s output against
  the *full* ground truth, the way an actual end user would experience it:
  a word ASR deleted or mangled can never be correctly flagged, and that
  correctly counts as a miss in this number.
- **(d) Alignment quality itself**: word error rate (WER) per clip and in
  aggregate; the count and rate of Correct/Substitution/Deletion outcomes,
  reported *specifically for ground-truth-disfluent words* separately from
  clean words (since disfluent words are the ones this project cares most
  about ASR preserving, and are already known — from the field's own
  literature, `PAPER_DECISION_LOG.md` — to be disproportionately at risk).

**3. How Track A and Track B are compared — the ASR/detector decomposition**

For each type (and `Any`), report:

```
Track A recall                                =: R_A
Track B, ASR-preserved-subset recall          =: R_B|preserved
Track B, overall recall                       =: R_B|overall

Total gap                 = R_A − R_B|overall
Detector-attributable gap = R_A − R_B|preserved
ASR-attributable gap      = R_B|preserved − R_B|overall
                           (= Total gap − Detector-attributable gap, by construction)
```

This is the concrete, falsifiable decomposition the owner asked for: **any
recall lost on words ASR preserved correctly is charged to the detector;
any recall lost on words ASR deleted or mangled beyond recognition is
charged to ASR.** The two components are computed independently (not
inferred or eyeballed) and sum exactly to the total gap by construction —
there's no remainder to argue about. The same decomposition applies to
precision using the FP side (a false positive on a word ASR *inserted* that
isn't in the reference at all is unambiguously ASR-attributable; a false
positive on a correctly-aligned clean word is detector-attributable).

**4. What constitutes success for this pilot**

Two different questions, both must be answered, neither is "the numbers
look good":

- **Methodological success** (gate for trusting anything else): a random
  sample of at least 10 clips' alignments hand-checked directly against
  their audio/transcript by inspection, judged correct. If alignment
  quality is poor, every other number this pilot produces is reported as
  provisional/unreliable, not silently trusted. This is a pass/fail check
  on the *tool*, before it's pass/fail on the *hypothesis*.
- **Scientific outcome** (not pass/fail — a measurement, reported either
  way): does the ASR-attributable vs. detector-attributable decomposition
  produce an interpretable answer for at least `word_repetition` (the type
  with an existing, concrete real-mic example of an ASR-attributable miss,
  `PAPER_DECISION_LOG.md`, 2026-08-03 real-audio validation entry) and for
  `prolongation` (directly testing whether §9's dominant ablation finding —
  that the current 1.0s threshold isn't optimal — replicates on real,
  non-reconstructed ASR timestamps, or was an artifact of LibriStutter's
  reconstruction as §8.3/§9.4 already flagged as the leading hypothesis).
  **A result showing the gap is mostly ASR-attributable, mostly detector-
  attributable, or genuinely mixed are all valid, useful outcomes** — none
  of them is predetermined as the "success" case. Pre-registering this
  means: whatever the pilot finds, it gets reported as found, not
  reframed to fit an expected answer.

**5. Explicit non-goals for this pilot**

- Not tuning any threshold based on pilot results (same standing
  instruction as every other phase this project has run).
- Not treating a 20–40 clip pilot as statistically conclusive — its job is
  to validate the method and produce a first directional read, explicitly
  flagged as such, with a full-sample run as the natural follow-up once the
  method is confirmed sound.
- Not scoring `phrase_repetition`'s Track B numbers as if reconstruction
  bias were resolved — Track B removes LibriStutter's *token*-reconstruction
  bias (§8.2), but `phrase_repetition`'s ground truth is still LibriStutter's
  own single-word-marker convention (§2), which is a separate, still-present
  limitation Track B does not fix.

**Addendum (2026-08-03, pre-registered before implementation, after the
30-clip pilot's hand-verification found the limitation motivating it —
see `PAPER_DECISION_LOG.md`, "Track B implemented and piloted"):** the
original `R_B|preserved` (point 2(b) above) only requires the *disfluent
word itself* to align `correct`. Hand-verification found this insufficient
— several context-dependent checks (`word_repetition`'s and
`sound_repetition`'s exact/near-match against the *immediately preceding*
word) can miss even when the disfluent word itself is transcribed
correctly, if ASR substituted or reordered the word next to it. Written
before recomputing anything against real data:

- **`R_B|preserved_ctx1`** (context-strict, 1-word window): a ground-truth
  disfluent word at reference index `i` counts as preserved only if **both**
  `i` and `i−1` (when `i>0`; just `i` when `i=0`, no preceding word exists)
  align `correct`. This directly targets the exact mechanism found in
  hand-verification (a 1-word-back dependency) and nothing more.
- **Reported alongside, not instead of, the original `R_B|preserved`** —
  both numbers get shown, so the direction and size of the effect is
  visible, not hidden behind a single "corrected" figure presented as final.
- **Explicitly a partial fix, stated as such**: `word_repetition` and
  `sound_repetition`'s checks are genuinely 1-word-back-dependent, so this
  window is well-justified for them. `phrase_repetition` can depend on up
  to `phrase_repetition_max_words` (8, `config.yaml`) preceding words, which
  a 1-word window does not capture — its numbers under this metric should
  still be read as *directionally* improved, not fully corrected.
  `prolongation`'s dependency is different in kind (the token's own voiced
  duration, plus a clip-wide percentile computed from *all* tokens) — a
  local word-adjacency window does not model this at all, and
  `prolongation`'s context-strict number should not be over-interpreted.
- **Success/non-goal for this refinement, stated in advance**: this is
  expected to *raise* `R_B|preserved` for `word_repetition`/`sound_repetition`
  specifically (fewer spurious "detector failures" once the confound is
  removed) and correspondingly *shrink* the detector-attributable share of
  the decomposition for those types. It is not expected to fully close the
  gap — the absolute Track A → Track B recall drop (§8.4) is a real,
  independently-established finding this refinement does not question, only
  the internal split between its two causes.

**Result of running this refinement (2026-08-03, same day, recorded
separately from the addendum above since the addendum was written *before*
implementation, per this section's own discipline):** see §8.4's
"Context-strict refinement" subsection for full numbers. Headline:
`R_B|preserved_ctx1` (Any label) = **1.0** on this pilot — a larger move than
the addendum's own stated expectation ("raise... not expected to fully close
the gap" referred to the absolute Track A→B drop, which is unaffected; the
*internal detector/ASR split* moved further than anticipated). At n=2 this is
not a claim the detector is flawless — see §8.4 for a concrete case
(`103-1240-0000`, the same "Rachel Rachel Lynde" clip already discussed
above) where `preserved_ctx1` still counts the instance as "preserved" and
the detector still doesn't produce the exact `word_repetition` label, which
surfaces a **further, sharper limitation this 1-word reference-side window
doesn't catch**: it checks that reference positions `i` and `i−1` each
individually align `correct`, but not that their *aligned hypothesis words
are contiguous in the hypothesis sequence itself* — an ASR insertion between
them (e.g. "Lynde," landing between the two "Rachel"s) can make both
individually "correct" while the detector, operating on the real hypothesis
token stream, never sees the two disfluent words back-to-back at all. Flagged
as the next candidate refinement (hypothesis-side contiguity, not just
reference-side correctness) — not implemented now, same reasoning as before:
needs its own pre-registration and more data before it's worth building.

**Addendum (2026-08-03, pre-registered before implementation): speaker-
stratified Track B sampling.** The Phase 1 closing review (§7.2 item 2)
found the 30/90-clip Track B pilots are a deterministic *prefix* of the
speaker-ordered 499-clip sample, covering only 7 of 40 distinct speakers —
not a representative cross-section. This addendum defines, before running
anything, how the next Track B run selects clips instead:

- **Selection method**: round-robin by speaker. Group the 499-clip sample
  by speaker ID (the numeric prefix of the clip name, e.g. `103` from
  `103-1240-0000`); take each speaker's 1st clip (in the existing
  deterministic file order) before any speaker's 2nd clip, then each
  speaker's 2nd before any 3rd, and so on, until the target clip count is
  reached. Deterministic and chosen before seeing any result — not
  cherry-picked speakers or clips.
- **Target size**: up to 3 clips per speaker across all 40 speakers (120
  clips if every speaker has ≥3), or fewer if the target count is reached
  first — sized to be comparable to the existing 90-clip run while
  maximizing speaker coverage (40 speakers instead of 7).
- **What this does and doesn't fix**: this changes *which* clips are
  scored, not the alignment/scoring logic itself (unchanged from §5.1's
  original protocol and the `preserved_ctx1` addendum above) — so results
  from this run are directly comparable to the existing 30/90-clip results,
  not a new metric. Per-clip caching means clips already scored (the
  existing 90 clips, likely including many of each covered speaker's
  earlier-order clips) are reused, not re-run.
- **Success/non-goal, stated in advance**: this is a check of whether the
  confirmed §8.4.2 conclusion (`R_B|preserved_ctx1` recall = 1.0, detector-
  attributable gap ≈ 0%) holds when speaker coverage broadens from 7 to 40
  — not a guarantee it will. A result showing the finding holds, weakens,
  or strengthens are all valid, reportable outcomes; none is predetermined
  as "success." Given `preserved_ctx1`'s positive-instance count was
  already small (n=7 at 90 clips), a modestly larger, more diverse sample
  may still not be enough for a precise number — the qualitative direction
  is what this run can most credibly speak to.

**Addendum (2026-08-04): hypothesis-side-contiguity metric — built and
measured before any detector change, per this section's own "measure
before implementing" discipline (`ROADMAP.md` item 4's candidate fix (b),
done ahead of candidate fix (a)).** Defines, precisely, the gap the
2026-08-03 addendum above only described qualitatively: for a
`preserved_ctx1` instance (reference positions `i`/`i-1` both align
`correct`), compute `gap = hyp_index(i) - hyp_index(i-1) - 1` — the number
of hypothesis-sequence tokens inserted between the two aligned words that
the reference-only check cannot see. `gap == 0` means truly contiguous in
the actual ASR output the detector runs on; `gap > 0` means an insertion
broke true adjacency even though both reference positions individually
align correct. Computed directly from `alignment.py`'s existing
`AlignmentOp.hyp_index` field — no new alignment logic needed, only a new
diagnostic pass over already-aligned data.

**Method**: re-scored all 120 already-cached speaker-stratified clips
(`eval_datasets/_track_b_cache/`, zero new ASR calls) with this metric.
**Non-goal, stated before running**: this is a diagnostic pass to inform
whether candidate fix (a) is worth its false-positive risk, not itself a
benchmark number to headline — its output determines whether (a) proceeds,
not how well (a) performs (that would need its own, separate benchmark
after implementation, per usual discipline).

---

## 6. Framework design (built)

```
profiling/evaluation/
├── __init__.py
├── loaders.py                       # LabeledClip (word-level, now audio-optional) +
│                                     # ClipLevelLabels (clip-level) + load_libristutter_* + load_sep28k_labels
├── metrics.py                        # score_word_level, score_clip_level, localization_rate (IoU),
│                                     # format_confusion_matrix
├── alignment.py                      # Track B word-level Levenshtein alignment, disfluent-word cost bias
├── track_a.py                        # detector-only runner (--self-test, --dataset, --data-dir, --audio-dir)
├── track_b.py                        # full-pipeline runner: real ASR + alignment + preserved/preserved_ctx1/
│                                     # overall scoring, per-clip result caching (--self-test, --n, --verbose)
├── report.py                         # table + timestamped, non-clobbering JSON result files (git commit, config)
├── run_ablations.py                  # config-variant sweep runner (VAD/Praat/fusion-weight/threshold), §9
├── fetch_libristutter_sample.py      # downloads a distributed annotation sample (GitHub mirror)
└── fetch_libristutter_audio.py       # downloads matching real audio for an already-fetched sample
```

```
python -m profiling.evaluation.track_a --dataset libristutter --data-dir DIR
python -m profiling.evaluation.track_a --dataset libristutter --data-dir DIR --audio-dir DIR2
python -m profiling.evaluation.track_a --self-test
python -m profiling.evaluation.track_b --data-dir DIR --audio-dir DIR2 --n 30
python -m profiling.evaluation.track_b --self-test
python -m profiling.evaluation.run_ablations
python -m profiling.evaluation.fetch_libristutter_sample --n 500
python -m profiling.evaluation.fetch_libristutter_audio
```

`profiling/evaluate.py` is now a thin backward-compatible shim over this
package — the CLI (`python -m profiling.evaluate --self-test` /
`--data-dir`) behaves identically to before; `evaluate()`'s return shape
changed on purpose (now `(counts, localization)`, not just `counts`) and is
documented as such in the shim's own docstring, not silently changed.

**Two design refinements discovered during implementation, neither
anticipated in the original plan:**
1. Datasets differ in *granularity*, not just label vocabulary. LibriStutter
   labels individual words (`LabeledClip`, matches this app's own token
   shape, drives `detect_disfluencies()` directly for Track A). SEP-28k,
   confirmed against its own README, labels whole 3-second clips with no
   reference transcript at all — each row is a clip-level count of how many
   of 3 annotators selected each disfluency type "somewhere in this clip,"
   not a word-level location. That needed its own shape (`ClipLevelLabels`)
   and scorer (`score_clip_level`).
2. Transcript source and audio availability are independent axes, not one
   "Track A vs Track B" choice — see §3. `LabeledClip.audio_bytes` (optional)
   and the paired `load_libristutter_*_with_audio` loaders were added so
   Track A can run with or without real audio, since the audio-native
   detection layer was otherwise never being evaluated at all (see
   `PAPER_DECISION_LOG.md`'s "Audio-enabled evaluation" entry).

**Sequencing status (as of 2026-08-03, Phase 1 close):**
1. Done — package skeleton, migrate existing logic, no behavior change.
2. Done — IoU localization, per-type confusion matrices, "Any" label.
3. Partially done — `load_sep28k_labels` built and verified against the
   real, complete labels file (clip-level shape confirmed); real SEP-28k
   *audio* not acquired yet (§7 sequencing note below) — the only
   remaining "not started" item in this list, deferred to Phase 2/§7 for
   an explicit, evidence-based reason, not an oversight.
4. Done for LibriStutter (real, real+audio — §8.2/§8.3); not done for
   SEP-28k (needs its audio first, item 3).
5. **Done** — `alignment.py` + `track_b.py` built (full pipeline, our own
   ASR + hypothesis-to-reference alignment, per-clip caching). §5.1/§8.4.
6. **Done** — Track B run against LibriStutter (30 clips, then scaled to
   90), compared against Track A on an approximated basis (§8.4's own
   caveat, since resolved: Track A's number was originally the full
   499-clip sample's recall, not a separately-computed number for the
   exact 30/90-clip Track B subset — **fixed during the Phase 1 closing
   review**: Track A now also run on the exact matched subsets (§8.4/
   §8.4.1/§8.4.2), replacing the approximation with an exact number. See
   the critical review in §7). Not yet done against SEP-28k (blocked on
   item 3).

**Resequencing note (2026-08-03)**: audio-enabled LibriStutter evaluation
(closing the "audio-native layer never evaluated" gap) was prioritized
ahead of acquiring SEP-28k's audio, based on a deliberate re-examination of
which next step most directly serves improving the model's real-world
audio-based detection ability — see `PAPER_DECISION_LOG.md` for the full
comparison against SEP-28k and FluencyBank Timestamped as alternatives.
SEP-28k audio and Track B remain planned, not dropped.

---

## 7. Honest limitations and critical review

### 7.1 Standing limitations

- No accessible dataset combination covers the full 7-type taxonomy —
  `stutter_marker` has no labeled equivalent anywhere reviewed.
- SEP-28k's audio depends on podcast URLs that can rot; record the actually-
  acquired fraction (count, date) as part of every run's reproducibility
  record, never silently treat it as "the full dataset."
- Human annotators disagree with each other on stuttering event boundaries
  even in these datasets — a perfect detector would not score 1.0. Report
  published inter-annotator agreement as a ceiling reference wherever
  available, not just our own F1 in isolation. **Not yet done** — no
  published inter-annotator agreement figure has been located/recorded for
  LibriStutter specifically (it's synthetic, so this may not even apply the
  same way as it does to human-annotated real-speech datasets); flagged for
  Phase 2/whenever a human-annotated dataset (SEP-28k, KSoF) is scored.
- Track B's alignment step is itself a source of methodology-introduced
  error. **Done, twice**: spot-checked by hand on 10 clips before the
  original pilot's numbers were trusted (§8.4, methodological gate passed),
  and again with `--verbose` diagnostics during the context-strict
  refinement (§8.4.1). Not re-verified by hand on the 60 additional clips
  added in the 90-clip scaled run (§8.4.2) — the alignment *algorithm*
  didn't change between the 30- and 90-clip runs, only which clips it ran
  on, so this is judged low-risk, but it is an assumption, not a re-verified
  fact, and is listed as such rather than silently assumed identical.
- KSoF/UCLASS are lower priority due to access friction and (for KSoF)
  language mismatch, not because they're less valuable — revisit if
  cross-lingual support becomes an explicit goal.

### 7.2 Critical review of methodology (Phase 1 closing review, 2026-08-03)

Performed at the project owner's explicit request before closing Phase 1:
a systematic re-examination of every evaluation procedure, metric, dataset,
ablation, and protocol used so far, specifically looking for mistakes,
unstated assumptions, and gaps — not just re-summarizing what was already
written down. Findings below; each is labeled **fixed this review**,
**deferred with reasoning**, or **accepted as a standing, documented
limitation**.

1. **Track B's confirmed conclusion rests on Track A being computed on the
   right comparison set — it wasn't, and this was a real, previously-uncaught
   gap. Fixed this review.** §5.1's original decomposition formula used
   $R_A$ from the full 499-clip Track A sample as an approximation for what
   Track A would score on the *specific* 30/90-clip subset Track B used —
   flagged as an approximation at the time, but never actually tightened.
   Re-examining it during this review: Track A needs no ASR, so computing it
   exactly on the identical clip subsets Track B used is cheap (seconds, no
   new model inference). Done: exact $R_A$ = 1.000 on both the 30- and
   90-clip subsets (§8.4/§8.4.2 tables), which turns the context-strict
   decomposition's detector-attributable share from "≈0%, small negative
   number attributed to sample noise" into **exactly 0%** — a materially
   cleaner and more defensible result, not just a cosmetic correction. This
   is exactly the kind of gap a "did we make a mistake" pass is meant to
   catch: the approximation was flagged honestly at the time it was made,
   but nobody had gone back to check whether tightening it was actually
   easy — it was.

2. **The Track B clip subset is speaker-clustered, not speaker-representative
   — a real limitation, identified here and directly resolved in Phase 2.**
   Checked directly during this review: the full 499-clip Track A sample
   spans **40 distinct speakers**. The Track B subsets, however, were a
   deterministic *prefix* ("first 30," then "first 90") of that list, and
   LibriStutter's filenames are speaker-ordered — so the 30-clip pilot
   covered only **3 speakers**, and the 90-clip scaled run only **7**
   (confirmed by direct inspection: speaker IDs `103`, `1088`, `1334` at
   n=30; adding `1502`, `1743`, `1867`, `1926` at n=90). This meant the
   entire Track B evaluation — including the then-confirmed §8.4.2
   conclusion — had been measured on **17.5% of the available speakers**,
   not a representative cross-section. This mattered directly because ASR
   error patterns are known to be speaker/accent/recording-dependent; it
   was possible the specific context-corruption mechanism found (§8.4.1)
   was partly a property of how CrisperWhisper handles *these particular 7
   speakers'* voices rather than a fully general finding. **Resolved
   2026-08-04** (`PAPER_DECISION_LOG.md`; `VALIDATION.md` §8.4.3): a
   speaker-stratified Track B run across all 40 speakers found the
   suspicion in this item was justified — the 7-speaker sample's clean
   "0% detector-attributable" result did not hold at full diversity
   (revised to 35.1% detector-attributable), traced to two already-known
   `sound_repetition`/`phrase_repetition` structural gaps rather than a
   speaker-specific ASR artifact per se. This is exactly the outcome this
   critical-review item existed to either rule out or catch — it caught
   something real.

3. **The confirmed "ASR is the bottleneck" conclusion rests on exactly one
   ASR backend (CrisperWhisper) and one dataset family (LibriStutter's
   synthetic disfluency injection). Accepted as a standing, documented
   limitation — this is the single biggest generalization risk in Phase 1's
   headline finding.** Every Track B number in this document, without
   exception, comes from CrisperWhisper's specific error behavior on
   LibriStutter's specific splice artifacts. Two distinct, compounding risks
   follow: (a) a different ASR model might have a materially different error
   profile around disfluent speech (better, worse, or simply different in
   *kind* — e.g. a model that hallucinates differently around repeated
   words), so "ASR fidelity is the dominant bottleneck" is a claim about
   *this* ASR model on *this* data, not ASR-in-general, until checked
   against at least one more backend; (b) LibriStutter's disfluencies are
   synthetically spliced into otherwise-fluent read speech, which may have
   acoustic/prosodic discontinuities at splice boundaries that a real
   stutterer's natural disfluent speech doesn't have (and vice versa — real
   disfluent speech has prosodic cues, like rising pitch into a block, that
   a splice can't reproduce) — so it's possible CrisperWhisper's specific
   struggle with LibriStutter is partly a splice-artifact effect, not purely
   a "disfluent speech is hard for ASR in general" effect. **Not fixed this
   review** — both would require substantial new work (integrating a second
   ASR backend, or acquiring FluencyBank Timestamped's real disfluent
   speech) that is explicitly out of scope for a closing-and-consolidating
   session focused on validation, not model/pipeline changes. This is the
   single most important item carried into `ROADMAP.md` for Phase 2:
   **validate the "ASR is the bottleneck" conclusion against at least one
   more ASR backend and one real (non-synthetic) disfluent-speech dataset
   before treating it as fully general**, not just LibriStutter-specific.
4. **Track B has no localization (temporal/IoU) metric at all — a real,
   previously-unstated gap. Accepted as a standing, documented limitation.**
   Checked directly during this review: `track_b.py`'s `score_clip` always
   sets `localization=None` — confirmed by reading the source, not assumed.
   Track A's IoU-based localization (§4 point 3, used throughout §8.2/§8.3)
   has no Track B equivalent, so every "n/a" in the Localization column of
   every Track B table in §8.4/§8.4.1/§8.4.2 reflects an actual missing
   metric, not a data gap. This means Track B has validated *whether* and
   roughly *at what word position* a disfluency is caught, but never *how
   precisely timed* the detected span is under real ASR conditions —
   Track A's temporal-precision claims do not currently have a Track B
   analogue. Feasible in principle (both a predicted acoustic span and a
   ground-truth reference span exist in real audio time), but a real,
   non-trivial feature addition, not a quick fix — scoped for `ROADMAP.md`,
   not built this session.
5. **No confidence intervals or significance testing anywhere in this
   evaluation phase. Accepted as a standing, documented limitation.** Every
   number in §8/§9 is a point estimate (TP/FP/FN counts and the ratios
   derived from them) with no formal uncertainty quantification (e.g. a
   Wilson/Clopper-Pearson interval on recall given n). This project has
   consistently *qualitatively* flagged small samples as unreliable (e.g.
   §8.4's "too few positive instances... to be individually meaningful"),
   which is the right instinct, but has not *quantified* that unreliability
   anywhere. At the extreme small-n cases (`R_B|preserved_ctx1`'s n=2 and
   n=7), a formal interval would make the "don't over-trust this exact
   number" caveat concrete instead of qualitative. **Not fixed this
   review** — worth adding as low-cost, high-value infrastructure
   (`metrics.py` already has every count needed; this is a stats-formula
   addition, not a new data collection effort) — scoped for `ROADMAP.md`
   as a cheap, no-new-data Phase 2 item, similar in spirit to the
   already-planned confidence-sensitive metric for VAD/Praat (§9.3).
6. **Is the context-strict (`ctx1`) window's design circular — i.e., was it
   reverse-engineered from the one example that motivated it, making its
   dramatic result a foregone conclusion? Reviewed directly; judged sound,
   not circular.** The 1-word-back window was derived from a *mechanistic*
   understanding of exactly which detector checks are 1-word-back-dependent
   (`word_repetition`/`sound_repetition`'s adjacency checks, directly
   readable from `detect.py`), not from searching for a window size that
   produces a good-looking number. Its scope limits were stated *in
   advance* (§5.1's addendum: explicitly a partial fix for
   `phrase_repetition`, not applicable to `prolongation`) before the metric
   was run against real data, and its expected direction (raise
   `R_B|preserved` for the two targeted types) was also stated in advance —
   the magnitude (exactly 1.0, both times) was not predicted and came as a
   genuine surprise, which is itself evidence against circularity (a
   reverse-engineered metric would more likely be tuned to land somewhere
   unremarkable-looking, not somewhere so extreme it required its own
   scaled confirmation run to be believed). Judged methodologically sound;
   no correction needed.
7. **Are the ablation study's "optimal threshold" findings (§9) safe to act
   on yet? Reviewed; answer is still no, consistent with §9.4's existing
   caveat — restated here because a closing review should re-ask, not just
   inherit, past caveats.** §9's `prolongation_min_seconds` sweep is
   entirely on LibriStutter's *reconstructed* token timing (§8.2), and nine
   of this document's own sections have by now independently confirmed that
   reconstruction is an approximation, not a verified transcription. No new
   evidence this review changes that; `config.yaml`'s threshold correctly
   remains untouched, per standing instruction. This is not a new finding,
   but re-confirming an old caveat still holds, rather than letting it go
   stale, is itself part of what this review was asked to do.

### 7.3 Additional datasets or validation strategies considered for this
closing review, and why none were added now

Revisited the dataset comparison in §2 specifically to ask "would adding
one now materially strengthen Phase 1, given everything learned since §2
was first written" — not just re-reading the original comparison:

- **FluencyBank Timestamped** (real people who stutter, not synthetic
  injection) is now, after item 3 above, the single most valuable dataset
  this project could add — it directly tests whether the "ASR is the
  bottleneck" conclusion generalizes beyond LibriStutter's synthetic splices
  to real stuttered speech. **Not added this session**: integration risk
  (TalkBank/CHAT format, unconfirmed access gating) was never resolved
  (§2/`ROADMAP.md`), and resolving it plus building a new loader is real,
  open-ended engineering work — squarely Phase 2 scope ("evidence-driven
  model improvement" builds on a closed Phase 1, not the reverse), not a
  same-session addition to a closing/consolidation pass.
- **SEP-28k audio** would unblock item 3's ASR-generalization concern only
  partially (still real speech, but clip-level labels with no reference
  transcript — §8.5 — so it can't directly repeat Track B's word-level
  alignment method; it would need its own clip-level Track B analogue,
  not yet designed). Deliberately resequenced behind LibriStutter+audio
  earlier in this project (`PAPER_DECISION_LOG.md`, 2026-08-03) for
  reasons that still hold; not revisited as urgent by this review, since
  FluencyBank Timestamped more directly answers item 3's specific question
  (word-level, real speech, comparable methodology to what Track B already
  does) if either is pursued in Phase 2.
- **A second ASR backend** (e.g. a different fine-tune, or stock
  whisper-large-v3 for comparison) is not a "dataset" but is the other half
  of item 3's generalization concern, and is arguably *cheaper* than a new
  dataset (no new licensing/acquisition, just a second inference pass over
  audio already downloaded) — flagged in `ROADMAP.md` as possibly the
  highest-value-per-effort Phase 2 validation addition, ahead of new
  datasets.
- **KSoF/UCLASS**: re-considered and still deprioritized for the same
  reasons as §2/§7.1 (access friction, and for KSoF, language mismatch) —
  nothing learned this session changes that calculus.

**No new dataset was integrated during this closing review.** This is a
deliberate choice, not an oversight: every candidate identified above
requires either unresolved access/integration risk (FluencyBank) or
non-trivial new engineering (a clip-level Track B analogue for SEP-28k, a
second ASR backend wired into the harness) that belongs in Phase 2's
"evidence-driven improvement" work, not in a session explicitly scoped to
finalize and consolidate what Phase 1 already produced. What *was* feasible
and scientifically justified within this session's scope — tightening the
$R_A$ approximation (item 1) — was implemented; what wasn't feasible within
scope is recorded here and in `ROADMAP.md`, not silently dropped.

### 7.4 Verdict: is Phase 1's validation methodology sufficient to close on?

**Yes, with explicitly documented gaps carried forward, not swept under.**
The core methodology — dataset-based evaluation, the two-axis (transcript
source × audio availability) evaluation design, pre-registration before
implementation, hand-verification gates before trusting alignment-based
numbers, auditing surprising results before reporting them (the FLAC-silence
bug, the word_repetition FP audit, this review's own $R_A$ tightening) — has
held up under this review: no result currently reported in §8/§9 is now
believed to be wrong, only some are now known to be **narrower in scope**
than their headline framing might suggest (items 2 and 3 above, specifically
the confirmed Track B conclusion's speaker- and ASR-backend-specificity).
That is the correct outcome of a critical review: not "nothing to find,"
and not "start over," but "tighten what's cheap to tighten now (item 1,
done), name what isn't (items 2–5, named and scoped), and don't let a
strong result's excitement outrun what it actually demonstrated." Phase 1's
job was to establish, with evidence, where the system's real bottleneck is
— it has done that (§8.4.2's synthesis, reaffirmed by this review's item 1
tightening), while also honestly bounding how far that evidence currently
generalizes (items 2–3). That bounded-but-real conclusion is a sufficient,
scientifically sound basis to close Phase 1 and begin Phase 2's
evidence-driven improvement work — see the Phase 1 closing summary for the
full readiness argument.

---

## 8. Results

### 8.1 Run log

| Date | Dataset | Track | Git commit | N clips scored | Notes |
|---|---|---|---|---|---|
| 2026-08-03 | LibriStutter | A (text-only) | (uncommitted at run time) | 499 | First real result — see §8.2. Sample via `profiling/evaluation/fetch_libristutter_sample.py`, every-9th-file distributed sample of 4,736 available annotation files (~10.5% of the corpus). Raw result: `eval_results/20260803T042321640184Z_libristutter_A.json`. |
| 2026-08-03 | LibriStutter | A+audio (1st attempt, discarded) | (uncommitted) | 499 | Discarded — measuring silent audio due to a `soundfile` dtype bug, caught before being recorded. See `PAPER_DECISION_LOG.md`. |
| 2026-08-03 | LibriStutter | A+audio (corrected) | (uncommitted at run time) | 499 | First audio-native-layer result — see §8.3. Same 499 clips as the text-only run, matching audio via `fetch_libristutter_audio.py`. Raw result: `eval_results/20260803T050917442304Z_libristutter_A+audio.json`. |
| 2026-08-03 | LibriStutter | B (pilot) | (uncommitted at run time) | 30 | First Track B result — see §8.4. Real CrisperWhisper ASR + alignment, protocol pre-registered in §5.1 before implementation. Raw result: `eval_results/20260803T082117789381Z_libristutter_B.json`. Hand-verification detail (10 clips, `--verbose`): `eval_results/20260803T084650194736Z_libristutter_B.json`. |
| 2026-08-03 | LibriStutter | Ablations (10 config variants) | (uncommitted at run time) | 499 (same clips as §8.3, per variant) | See §9. 10 raw result files, `eval_results/*_libristutter_ablation-*.json` (gitignored, one per variant — not individually enumerated here). |
| 2026-08-03 | LibriStutter | B (context-strict rescoring, same 30 clips as the pilot) | (uncommitted at run time) | 30 | `R_B\|preserved_ctx1` added — see §8.4.1. Pre-registered in §5.1's addendum before implementation. Raw result: `eval_results/20260803T111624161563Z_libristutter_B.json`. Per-clip cache introduced this run: `eval_datasets/_track_b_cache/`. |
| 2026-08-03 | LibriStutter | B (scaled confirmation) | (uncommitted at run time) | 90 | Scaled from 30 → 90 clips to confirm the context-strict finding at a larger sample — see §8.4.2. 32/90 clips reused from cache (30 original + 2 from an interrupted first attempt of this run), 58 real ASR runs. Raw result: `eval_results/20260803T154940357685Z_libristutter_B.json`. |
| 2026-08-03 | LibriStutter | A+audio (regression confirmation, `word_repetition` SLD/OD tag) | (uncommitted at run time) | 499 | Post-implementation benchmark for the Phase 2 Step 1 taxonomy refinement (`PAPER_DECISION_LOG.md`, "Adversarial self-review... and its first implementation milestone"). Same 499 clips as §8.3. **Confirmed byte-for-byte identical to §8.3's frozen baseline** (`Any` F1 0.835, unchanged) — proves the new `syllable_count`/`likely_sld` metadata on `word_repetition` events is purely additive. Run with `--no-save` (intentionally reproduces §8.3's numbers rather than recording a new result). |
| 2026-08-04 | LibriStutter | B (speaker-stratified, all 40 speakers) | (uncommitted at run time) | 120 | Directly resolves the Phase 1 closing review's speaker-clustering caveat (§7.2 item 2) — see §8.4.3. Round-robin sampling across all 40 speakers, pre-registered in §5.1's addendum before running. `R_B\|preserved_ctx1` recall revised from 1.0 (7 speakers) to 0.667 (40 speakers) — traced by hand to two already-known `sound_repetition`/`phrase_repetition` structural gaps, not `word_repetition` or a new detector weakness. Interrupted once mid-run by an unrelated session restart, resumed cleanly from the per-clip cache. Raw result: `eval_results/20260804T060338639222Z_libristutter_B.json`. |
| 2026-08-04 | LibriStutter | A+audio (exact-subset, speaker-stratified 120 clips) | (uncommitted at run time) | 120 | Exact matched-subset Track A recall for the same 120 clips used in the run above — $R_A$ = 185/186 = 0.9946 (the first time this project's exact-subset $R_A$ was not a clean 1.0). Raw result: `eval_results/20260804T060635995226Z_libristutter_A+audio-speaker-stratified-120-matched-to-trackB.json`. |
| 2026-08-04 | LibriStutter | Ablations (13 config variants — original 10 + 3 new prolongation-redesign variants) | (uncommitted at run time) | 499 (same clips as §8.3, per variant) | See §9.5.1. Pre-registered evaluation plan (§9.5) run exactly as specified. **Decision: `require_praat_stability_for_prolongation` flipped to the new shipped default (`true`) in `config.yaml`** — the only variant of 13 to clear the pre-registered bar (`Any` F1 0.835->0.888, prolongation F1 0.064->0.084, both improved). `use_rate_normalized_prolongation` stays `false` (regressed both metrics, standalone and combined with Praat-gating). 13 raw result files, `eval_results/*_libristutter_ablation-*.json`. Full raw console output: `eval_datasets/_prolongation_ablation_output.txt`. |
| 2026-08-04 | LibriStutter | A+audio (confidence-sensitive metric, real data) | (uncommitted at run time) | 499 (same clips as §8.3) | See §9.3.1. First real-data run of `metrics.confidence_stats()` — TP-vs-FP confidence gap ~zero everywhere measurable, slightly negative for `Any` (-0.007). Not saved via `report.save_run` (a metric computation over existing detection output, not a new scored run) — reproducible via the command recorded in §9.3.1. |

### 8.2 LibriStutter — Track A (real data, first result)

**Read this before the numbers**: these results run against **reconstructed**
tokens, not verified transcriptions. LibriStutter's real annotation format
(confirmed the same day — see `PAPER_DECISION_LOG.md`) marks every
disfluency with a placeholder `"STUTTER"` row, not a label on a real word;
`load_libristutter_csv` reconstructs each one into a plausible token (word/
sound/prolongation types get a copy of the adjacent real word;
phrase_repetition is approximated as a single-word repeat since the true
repeated phrase length isn't recoverable from the file). That reconstruction
is a *known, documented* source of measurement noise — several of the
patterns below are best explained by it, not by `detect_disfluencies()`
itself, and that's called out per-row. This is Track A (ASR bypassed,
`audio_bytes=None`) — the acoustic-fusion path is untested here.

| Type | TP | FP | FN | TN | Precision | Recall | F1 | Localization (IoU≥0.5 rate) |
|---|---|---|---|---|---|---|---|---|
| filler | 0 | 25 | 0 | 17945 | 0.000 | n/a | n/a | n/a |
| sound_repetition | 0 | 1 | 200 | 17769 | 0.000 | 0.000 | n/a | n/a |
| word_repetition | 183 | 640 | 3 | 17144 | 0.222 | 0.984 | 0.363 | 1.000 |
| phrase_repetition | 0 | 4 | 201 | 17765 | 0.000 | 0.000 | n/a | n/a |
| prolongation | 37 | 737 | 185 | 17011 | 0.048 | 0.167 | 0.074 | 1.000 |
| **Any (combined)** | **802** | **465** | **7** | **16696** | **0.633** | **0.991** | **0.773** | n/a |

17,970 tokens scored across 499 clips.

**Interpretation, honestly, per row:**

- **`Any` is the most trustworthy top-line number here**: 99.1% recall — the
  detector finds essentially every real disfluency *somewhere* — at 63.3%
  precision. This is the number least distorted by the reconstruction's
  type-approximation, since it only asks "disfluent or not," not "which
  type."
- **`word_repetition`'s 98.4% recall is expected and not very informative on
  its own**: the reconstruction creates an exact back-to-back word repeat
  for this type by construction, and `detect_disfluencies()`'s exact-match
  check is guaranteed to catch it. Its raw 22.2% precision looked alarming
  but is mostly a reconstruction artifact, **confirmed by a direct audit of
  all 640 FPs against their true ground-truth type**: 195 were actually
  sound_repetition, 198 phrase_repetition, 220 prolongation, and only **27
  (4.2%) were on genuinely clean tokens**. In other words, 95.8% of these
  "false positives" are cases where `detect_disfluencies()` correctly
  flagged *something* disfluent, just under a different type than this
  reconstruction's necessarily-approximate ground truth — reframed against
  clean speech only, that's 183 real TPs vs. 27 real clean-speech FPs, an
  87.1% precision, not 22.2%. The raw per-type precision number is
  misleading here specifically because of how phrase/sound/prolongation get
  reconstructed as look-alike word repeats; it is not evidence of a weak
  detector.
- **`sound_repetition`: 0% recall (0/200) is a genuine, actionable finding,
  not obviously a reconstruction artifact.** `detect_disfluencies()`'s
  fragment-repetition check requires the *fragment* to come **before** the
  complete word (`prev_word.endswith("-")`, e.g. "b- buy"). This
  reconstruction places the fragment **after** the complete word (e.g.
  "Rachel" then "rachel-"), matching how LibriStutter's real STUTTER-row
  placement works. Two live possibilities, not yet distinguished: (a) the
  reconstruction should mirror the fragment *before* the word instead, or
  (b) `detect_disfluencies()`'s sound_repetition check only covers one of
  two real fragment-repeat orderings and is missing the other. Worth
  investigating before touching either side.

  **Root cause found and fix pre-registered, 2026-08-04 (before
  implementation) — deeper than either possibility above.** Directly
  tested both orderings with the exact reconstruction convention
  `load_libristutter_csv` uses (fragment = complete word text + trailing
  `-`, confirmed in `loaders.py`): in **both** "fragment-before-word" and
  "fragment-after-word" arrangements, the event was misclassified as
  `word_repetition` (0 `sound_repetition` events in either case), not just
  the "after" ordering. Cause: `_norm()` strips the trailing `-`, so a
  reconstructed fragment normalizes to a string **identical** to its
  complete-word counterpart — and the code's existing exact-match
  `word_repetition` check (`low == prev_low`) runs *before* the
  fragment-specific check in the `if/elif` chain, intercepting every such
  pair regardless of order. Simply adding a reverse-order check (option
  (a)/(b) above) would not have fixed this, since the exact-match branch
  would still catch it first. **Fix**: move a fragment-pair check (either
  token ending in a literal, pre-normalization `-`, with a prefix/equality
  relationship to its neighbor) to run *before* the exact-match check,
  handling both orderings in one unified branch. **Expected effect,
  stated before running anything**: `sound_repetition` recall on this same
  499-clip sample should move substantially above 0% (exact figure not
  predicted — reporting whatever Track A measures, not a target);
  `word_repetition`'s TP count should drop correspondingly (events
  previously double-counted as `word_repetition` reclassify to
  `sound_repetition`, a redistribution, not a net new detection).
  **Non-goal**: not tuning any similarity/length threshold in response to
  this result — this is a structural correctness fix (the old logic
  could never have produced a `sound_repetition` label for this
  reconstruction pattern, in either token order), not a tunable knob.
  Full results once measured: §8.2.1.
- **`phrase_repetition`: 0% recall (0/201) is expected and not a detector
  finding** — flagged in the reconstruction's own documentation before this
  run happened (see `loaders.py`). A true multi-word phrase repeat can't be
  reconstructed from a single marker row without knowing the real repeated
  phrase's length, so `detect_disfluencies()`'s phrase-repetition check
  (which needs a genuine multi-word match) correctly can't fire on a
  single-word approximation. This type needs a different validation
  approach (real audio + real ASR transcript, i.e. Track B) to be measured
  honestly.
- **`prolongation`: recall 16.7%, precision 4.8% — both low, and unlike
  word_repetition, the same direct audit shows this is only partly a
  reconstruction artifact.** Of 737 FPs: 413 (56%) were on genuinely clean
  tokens — a real precision problem, not an artifact — and the remaining 324
  (44%) were cross-contamination from other reconstructed types (183
  phrase_repetition, 89 word_repetition, 52 sound_repetition; reconstructed
  tokens of any type inherit the original STUTTER row's often-long
  duration, which can trip the duration-based prolongation check as a side
  effect regardless of the token's true type). Even correcting for that
  324, prolongation still has ~413 real false positives against clean
  speech in this sample — a genuinely low-precision result, not just a
  measurement artifact, and worth investigating on real (not reconstructed)
  audio before concluding anything further. The 16.7% recall (37/222) is a
  separate, real finding: most reconstructed prolongations are being missed
  outright.
- **`filler`: no ground truth in this sample at all** (0 TP + 0 FN) —
  confirmed directly against the raw CSV files (zero rows with label 1
  anywhere in the 499 sampled files), not a parsing bug. This specific
  every-9th-file sample happened to miss every interjection-labeled clip;
  filler cannot be evaluated from this sample and needs a different/larger
  sample to say anything about.

**What this checkpoint does and doesn't support**: it's real signal that the
detector's core disfluency-vs-clean discrimination is strong (`Any` ≈ 99%
recall / 63% precision) on real synthetic-stutter data. The FP audit
confirms `word_repetition`'s apparent 22.2% precision is mostly a
reconstruction artifact (true clean-speech precision ≈ 87.1%), while
`prolongation`'s low precision (4.8% raw, still poor even after removing
cross-contamination) is a genuine finding worth investigating on real,
non-reconstructed audio. `sound_repetition`'s 0% recall is a genuine,
confirmed gap tied to fragment ordering (before- vs. after-word), not a
reconstruction artifact. **[2026-08-04: fixed — see §8.2.1, recall
0%→92.0%. The root cause was deeper than "ordering," see that section.]**
`phrase_repetition`/`filler` need different validation setups entirely
(Track B for phrase_repetition, since a true multi-word repeat can't be
reconstructed from one marker row; a larger or targeted sample for filler,
since this sample had zero ground-truth instances) before they can be
measured here at all.

#### 8.2.1 `sound_repetition` fragment-ordering fix — measured result (2026-08-04)

Pre-registered above before implementation. **Result matches the
predicted direction; magnitude was not predicted and is reported as
measured.** Same 499-clip, audio-enabled sample as §8.3 (below), re-run
after the fix (`profiling/detect.py`, `_word_repetition_extra` area — a
fragment-pair check now runs *before* the exact-match `word_repetition`
check, so a fragment reconstructed as "word + trailing `-`" is correctly
classified as `sound_repetition` regardless of which side of its
complete-word counterpart it sits on):

| Type | Metric | Before (frozen §8.3) | After (2026-08-04) | Change |
|---|---|---|---|---|
| `sound_repetition` | TP / FP / FN | 0 / 1 / 200 | 184 / 5 / 16 | +184 TP, +4 FP, −184 FN |
| `sound_repetition` | Precision / Recall / F1 | 0.000 / 0.000 / n/a | 0.974 / **0.920** / 0.946 | recall 0% → 92.0% |
| `word_repetition` | TP / FP / FN | 183 / 640 / 3 | 183 / 452 / 3 | −188 FP, TP/FN unchanged |
| `word_repetition` | Precision / Recall / F1 | 0.222 / 0.984 / 0.363 | 0.288 / 0.984 / 0.446 | precision +6.6pt, recall unchanged |
| `Any` (combined) | TP / FP / FN / F1 | 801 / 308 / 8 / 0.835 | 801 / 308 / 8 / 0.835 | **exactly unchanged** |

**`Any` being byte-for-byte identical is the expected, correct signature
of a pure type-reclassification fix, not a coincidence**: `Any` scoring
only asks "was some type predicted where some true type exists," which
this fix doesn't change (the position was already flagged, just under
the wrong type label) — consistent with this project's now-repeated
finding (§8.4.2, §8.4.3) that binary detection and exact-type
classification are separate axes. `word_repetition`'s TP/FN are also
unchanged — the fix only removed **false positives** that were never
genuine word-repetition detections (contrary to this fix's own
pre-registration text above, which predicted `word_repetition`'s *TP*
would drop; measurement shows it was FP that dropped, TP was never
inflated by this bug in the first place — the pre-registration's
predicted *direction* was right, its specific mechanism guess was
imprecise, corrected here against the actual measurement, not silently).

**16 residual `sound_repetition` FN (8% of 200) remain** — a much smaller,
secondary gap, not investigated further this round (the ~fourfold
precision-for-recall trade shown above, 97.4% precision at 92.0% recall,
is already a strong result; chasing the last 8% is lower priority than
other Phase 2 items per `ROADMAP.md`).

**A related infrastructure fix made alongside this**: `track_b.py`'s
per-clip cache was found to store the *detector's output* (`events`), not
just the ASR output (`hyp_tokens`) — meaning every previously-cached
Track B clip (all 210 clips across the 30/90/120-clip runs) held `events`
computed by the *pre-fix* detector code, and any future Track B run
reusing that cache would have silently kept scoring with the old, buggy
`sound_repetition` classification even after this fix shipped. Fixed:
the cache now stores only `hyp_tokens`; `events` is always recomputed
fresh from the live `detect.py` code on every run (cheap — no ASR
involved), so the cache can never go stale relative to detector-code
changes again. See `PAPER_DECISION_LOG.md`.

### 8.3 LibriStutter — Track A **with real audio** (same 499 clips, audio-native layer active)

**This is the first time this project's audio-native detection layer
(Silero VAD, Praat pitch/jitter/shimmer, weighted acoustic-vs-token fusion —
the main 2026-08 architectural change) has been evaluated against labeled
ground truth at all.** Same 499 clips, same reconstructed tokens as §8.2,
now with real matching audio (`fetch_libristutter_audio.py`) passed to
`detect_disfluencies()`.

**A real bug was found and fixed before these numbers were trusted** —
recorded in full in `PAPER_DECISION_LOG.md` ("Bug found and fixed:
`_flac_bytes_to_wav16k` silently produced silent audio"). The first attempt
produced a dramatic-looking result (prolongation collapsing to 0 recall,
`Any` precision jumping to 93%) that turned out to be measuring silence:
`soundfile.read(..., dtype="int16")` silently zeroed out real LibriStutter
FLAC files. Caught by direct audit (checking raw waveform RMS), not by
inspection of the metrics table alone — exactly the discipline this project
applies to every non-trivial result. Fixed by reading with the default
float64 dtype and scaling to int16 manually. **The numbers below are
post-fix, verified against real, non-silent, correctly-decoded audio.**

**[2026-08-04: `sound_repetition` and `word_repetition`'s rows below are
superseded by the fragment-ordering fix — see §8.2.1 (0.000→0.920 recall
on `sound_repetition`). Kept here as the frozen, accurate record of what
this checkpoint measured at the time; not edited in place, same discipline
as elsewhere in this file. Since audio never affects these two types
(this row's own "identical to text-only" finding below), §8.2.1's numbers
apply equally to this section — no separate audio-enabled re-measurement
was needed for those two rows specifically.]**

**[2026-08-04: the `prolongation` and `Any` rows below are superseded by
the prolongation redesign's default-config change — see §9.5.1
(`require_praat_stability_for_prolongation` flipped `true`, the shipped
default from this date forward). At the new default, this exact 499-clip
sample measures `prolongation` TP=16/FP=145/FN=206/F1=0.084 (up from
0.064 below) and `Any` TP=796/FP=188/FN=13/F1=0.888 (up from 0.835
below). Kept here as the frozen, accurate record of the pre-redesign
baseline this project's original audio-native architecture change was
measured against — not edited in place, same discipline as elsewhere in
this file.]**

| Type | TP | FP | FN | TN | Precision | Recall | F1 | Localization (IoU≥0.5) | vs. text-only (§8.2) |
|---|---|---|---|---|---|---|---|---|---|
| filler | 0 | 25 | 0 | 17945 | 0.000 | n/a | n/a | n/a | **identical** |
| sound_repetition | 0 | 1 | 200 | 17769 | 0.000 | 0.000 | n/a | n/a | **identical** |
| word_repetition | 183 | 640 | 3 | 17144 | 0.222 | 0.984 | 0.363 | 1.000 | **identical** |
| phrase_repetition | 0 | 4 | 201 | 17765 | 0.000 | 0.000 | n/a | n/a | **identical** |
| prolongation | 21 | 409 | 201 | 17339 | 0.049 | 0.095 | 0.064 | 0.857 | TP 37→21, FP 737→409, F1 0.074→0.064 |
| **Any (combined)** | **801** | **308** | **8** | **16853** | **0.722** | **0.990** | **0.835** | n/a | TP 802→801, FP 465→308, **F1 0.773→0.835** |

**Interpretation, honestly, per row:**

- **`filler`, `sound_repetition`, `word_repetition`, `phrase_repetition` are
  byte-for-byte identical to the text-only result.** This is expected, not
  a bug: `profiling/acoustic.py`'s audio-native detector only derives
  `block`/`prolongation` candidates — it has no repetition or filler logic
  at all — so audio can only ever change `prolongation` (and, on real live
  speech, `block`, which this sample has zero ground truth for — LibriStutter
  doesn't label blocks, §2) directly. This is a useful confirmation the
  harness wires audio through exactly where it should and nowhere else.
- **`Any` (combined) is the headline, real finding: F1 0.773 → 0.835, driven
  almost entirely by a genuine precision gain (63.3% → 72.2%, 157 fewer
  false positives) at essentially zero recall cost (802 → 801 TP, 7 → 8
  FN).** This is exactly the outcome the 2026-08 audio-native restructuring
  was designed to produce — real acoustic evidence suppressing false
  alarms without meaningfully hurting detection — and this is the first
  time it's been measured against labeled ground truth rather than argued
  from architecture alone.
- **`prolongation` recall dropped (37 → 21 TP, 16.7% → 9.5%) and F1 dropped
  slightly (0.074 → 0.064) — a genuine, mechanistically-confirmed
  finding, not a bug or an artifact.** Directly verified: of the 37 true
  prolongations whose *nominal* (reconstructed-token) span clears the 1.0s
  threshold, only 21 still clear it once trimmed to their *actual voiced
  duration* in the real audio (`_AcousticContext.voiced_duration()`) —
  exactly matching the observed TP count. In other words, the
  reconstruction's timestamps (copied from LibriStutter's `STUTTER` marker
  row) overstate how much of that span is genuinely sustained voiced sound;
  real audio-based verification correctly declines to credit the
  difference. This is very likely more a property of the **reconstruction's
  span estimation** than of real prolonged speech (real live-mic
  prolongations validated earlier the same day, `PAPER_DECISION_LOG.md`
  Part D, were 1.24–1.42s and were correctly caught) — but it can't be
  fully separated from a genuine detector conservatism without real (not
  reconstructed) prolongation ground truth, which LibriStutter's
  synthetic-injection design can't provide. Flagged as a priority for
  Track B or a real-speech dataset, not resolved here.
- **`prolongation`'s FP count also dropped substantially (737 → 409)** —
  consistent with the same mechanism: reconstructed non-prolongation tokens
  that happened to trip the nominal-duration threshold (the cross-
  contamination pattern audited in §8.2) are, in real audio, often *also*
  found to have shorter voiced duration than their nominal span, so fewer
  of them clear the threshold at all.
- **Localization dropped slightly for prolongation (1.000 → 0.857)**: of
  the 21 correct-type detections, 3 don't clear IoU≥0.5 against the
  ground-truth token's nominal span — a secondary effect of the same
  voiced-duration trimming (the *detected* region is real audio's voiced
  extent, which can end up not well-aligned with the *reconstruction's*
  nominal span it's being compared against).

**What this checkpoint does and doesn't support**: real, audited evidence
that the audio-native architecture change delivers on its design goal in
aggregate (meaningfully fewer false positives, negligible recall cost) —
the first such evidence this project has had. It does **not** yet resolve
whether `prolongation`'s recall drop reflects a real detector limitation or
is mostly an artifact of this dataset's synthetic-splice reconstruction;
that needs either Track B (real ASR transcript, no reconstruction involved)
or a real (non-synthetic) prolongation-labeled dataset to separate cleanly.

### 8.4 LibriStutter — Track B pilot (real ASR, 30 clips, per the §5.1 protocol)

Run exactly per the protocol pre-registered in §5.1, on the first 30 clips
(deterministic sorted order — same selection method the pilot scope
described, not cherry-picked after seeing results) of the same 499-clip
sample. Real CrisperWhisper ASR on real audio, `detect_disfluencies()` on
our own transcript, aligned back to ground truth via `alignment.py`.
Mean WER: **0.224** (22.4%) — high for a modern ASR model, and itself a
finding: CrisperWhisper struggles substantially more with LibriStutter's
synthetically-spliced disfluency segments than with ordinary read speech.

**Methodological gate (§5.1 point 4): PASSED, with a discovered caveat
recorded below, not hidden.** All 10 clips from the `--verbose` diagnostic
run (`profiling/evaluation/track_b.py --verbose`, first 10 clips of the 30)
were hand-checked — every reference word marked correct/substitution/
deletion, checked against the printed ground-truth/hypothesis word
sequences side by side. Every classification was defensible; a few cases
(e.g. two adjacent identical reference words like `"men" "men"` where only
one survived into the hypothesis) had a minor, low-impact ambiguity in
*which* of two identical positions gets credited "correct" vs "deletion" —
this never changed the correct aggregate conclusion (was the disfluency
preserved at all). **Caveat on scope**: verified against transcripts side
by side, not by listening to the underlying audio directly — a further,
stronger check available as future work, not done this round.

**Numbers** (Any label; per-type table has too few positive instances at
n=30 to be individually meaningful — see the note at the end of this
subsection):

| Metric | Track A, full 499-clip sample (§8.3) | Track A, **exact same 30 clips as this Track B run** | Track B overall (30 clips) | Track B, ASR-preserved subset (30 clips) |
|---|---|---|---|---|
| TP | 801 | 48 | 2 | 2 |
| FP | 308 | 11 | 12 | 12 |
| FN | 8 | 0 | 46 | 20 |
| Precision | 0.722 | 0.814 | 0.143 | 0.143 |
| Recall | 0.990 | **1.000** | **0.042** | **0.091** |
| F1 | 0.835 | 0.897 | 0.065 | 0.111 |

**The decomposition** (§5.1 formula). **Updated 2026-08-03 (Phase 1 closing
review) to use the exact Track A recall on this run's own 30-clip subset**
($R_A$ = 1.000, computed directly rather than approximated from the full
499-clip sample's 0.990 — cheap to do since Track A needs no ASR; see
`PAPER_DECISION_LOG.md`, "Phase 1 closing review: exact-subset Track A
recall replaces the approximation"). The full-sample number is kept in the
table above for context, not used in the arithmetic below:

```
R_A (exact, this 30-clip subset) = 1.000
R_B|preserved                    = 0.091
R_B|overall                      = 0.042

Detector-attributable gap = R_A − R_B|preserved = 0.909
ASR-attributable gap      = R_B|preserved − R_B|overall = 0.049
Total gap                 = R_A − R_B|overall = 0.958   (0.909 + 0.049 ✓, by construction)
```

**Mechanically, ~95% of the total gap is "detector-attributable" and ~5%
"ASR-attributable" by the pre-registered formula — but the hand-
verification found a real reason to treat that split with real caution,
recorded here as a dated methodological addendum to §5.1, not a silent
correction:**

**Addendum to §5.1 (2026-08-03, discovered during hand-verification,
not anticipated when the protocol was written):** several hand-checked
cases show a disfluent word correctly transcribed (`align=correct`) while
the detector still misses it — but the reason, on inspection, is that
*adjacent* words were substituted or reordered by ASR, breaking the local
context several detector checks depend on (e.g. `word_repetition`'s
back-to-back exact-match). Concretely: reference `"Rachel" "Rachel"
"Lynde"` (a direct repeat) came back from ASR as `"Rachel" "Lynde," "Rachel"
"Lynde"` — grammatically reinterpreted as introducing "Rachel Lynde" twice,
not transcribed as a stutter at all. The disfluent word itself aligns
`correct` (a "Rachel" really is there), but the adjacency the detector needs
is gone, and it predicted `phrase_repetition`/`block` instead of
`word_repetition`. **This means "ASR-preserved" (this word, alone,
transcribed correctly) is necessary but not sufficient for a fair test of
the detector in isolation — the surrounding context also needs to survive,
and this protocol's `R_B|preserved` doesn't check that.** The practical
effect: the ~91-point "detector-attributable" figure above almost
certainly overstates genuine detector-only failure — a real but currently
unquantified share of it is really a *second-order* ASR effect (context
corruption) that this pilot's methodology cannot yet cleanly separate from
first-order ASR effects (the word itself being lost). Fixing this precisely
needs a stricter preserved-subset definition (e.g. requiring a window of N
adjacent words around each disfluent position to also align `correct`, not
just the disfluent word itself) — flagged as the concrete next refinement
to this protocol, not implemented under time/scope pressure to get a
result out; see `ROADMAP.md`.

**What is NOT in doubt, regardless of the precise attribution split**: the
absolute recall drop from Track A's ~99% to Track B's ~4–9% is real,
large, and the single most important finding of this evaluation phase.
**Track A's ~99% recall was never a claim about real-world performance** —
it measures the detector against a perfect, ground-truth transcript, which
real deployment never provides. Track B is the first measurement of
anything closer to what an actual user experiences, and it shows the gap
between "detector logic works" and "the deployed system works" is large.
This is exactly the result pre-registering the protocol was meant to
surface honestly, whichever way it came out (§5.1 point 4) — it is not
reframed as a detector-only failure, an ASR-only failure, or hidden behind
the more flattering Track A number.

**Per-type note**: with only 30 clips (~48 disfluent instances total, 22
ASR-preserved), individual type breakdowns are too small to support
per-type conclusions (e.g. `sound_repetition` had 0 TP in both Track A and
Track B on this subset — consistent with the already-known structural gap
from §8.2, not new information at this sample size). The `Any` label is
the only number in this pilot large enough to trust directionally. A larger
Track B run (§9's own logic: the ablation's dominant lever also needs
Track B validation) is required before any per-type Track B conclusion is
drawn.

#### 8.4.1 Context-strict refinement — results (2026-08-03, same day, per the §5.1 addendum)

Implemented `R_B|preserved_ctx1` exactly as pre-registered (§5.1 addendum):
a ground-truth disfluent word at reference index `i` counts as preserved only
if **both** `i` and `i−1` align `correct`. Added per-clip caching to
`track_b.py` first (`_cache_path`/`_load_cached`/`_save_cache`,
`eval_datasets/_track_b_cache/`) so this and future metric refinements can
rescore the same 30 clips without re-running CrisperWhisper — this run
populated the cache from empty (30/30 real ASR, ~4,590s total), so future
refinements to the scoring definition are now near-instant. Self-tested first
(14 checks, including a direct reproduction of the hand-verified
`103-1240-0000` "Rachel Rachel Lynde" case, asserting the original
`preserved` metric counts it `fn=1` and `preserved_ctx1` counts it `fn=0`)
before trusting the real re-run.

**Numbers** (Any label; 30 clips, same alignment as §8.4 — only the
preserved-subset *definition* changes across these three columns):

| Metric | `R_B\|preserved` (original, word-only) | `R_B\|preserved_ctx1` (word + preceding word) |
|---|---|---|
| TP | 2 | 2 |
| FP | 12 | 7 |
| FN | 20 | **0** |
| Precision | 0.143 | 0.222 |
| Recall | 0.091 | **1.000** |
| F1 | 0.111 | 0.364 |

**Sample-size chain** (the more important number to read first, before the
recall figure above): of the 48 total ground-truth-disfluent instances in
this 30-clip pilot, 22 survive the *word-only* preserved filter, and only
**2** also survive the *context-strict* filter. That collapse (22 → 2) is
itself a finding, independent of what the detector does with those 2: ASR
overwhelmingly fails to preserve *both* a disfluent word and its immediate
neighbor together, even when it gets the disfluent word itself right. The
recall figure below is computed over that n=2 — real, mechanistically
explained (see below), but not something a 2-instance sample can establish
precisely. This is exactly what §5.1's non-goals section flagged in advance
("not treating a 20–40 clip pilot as statistically conclusive") applied one
level deeper than originally anticipated — scored at the level of *both* the
Track B pilot and this specific sub-metric within it.

**Revised decomposition** (§5.1's formula, using the exact matched-subset
$R_A$ = 1.000 — see §8.4's update above; this section originally used the
0.990 full-sample approximation, corrected during the Phase 1 closing
review, `PAPER_DECISION_LOG.md`):

```
R_A (exact, this 30-clip subset) = 1.000
R_B|preserved_ctx1                = 1.000
R_B|overall                       = 0.042

Detector-attributable gap (ctx1) = R_A − R_B|preserved_ctx1 = 0.000
ASR-attributable gap (ctx1)      = R_B|preserved_ctx1 − R_B|overall = 0.958
Total gap                        = R_A − R_B|overall = 0.958   (0.000 + 0.958 ✓, by construction)
```

With the exact (not approximated) $R_A$, the detector-attributable share is
**exactly zero, not just approximately zero** — the earlier −0.010 was an
artifact of comparing this subset's Track B performance against a
*different* sample's Track A number; recomputed against its own matched
Track A subset, the two numbers are identical (both recall 1.000 — the
detector caught 100% of context-preserved instances, and Track A also
happens to have zero misses on this specific 30-clip subset, see the
exact-subset table in §8.4). Compared to the original split (~95%
detector-attributable / ~5% ASR-attributable, computed on the word-only
`R_B|preserved`), the context-strict version reverses it completely:
**0% detector-attributable / 100% ASR-attributable, exactly, on this
subset.** This is a substantially larger swing than the addendum's own
stated expectation when it was pre-registered (§5.1: "not expected to fully
close the gap" — correctly so, since the *absolute* Track A→B drop is
untouched; what moved further than expected is the internal split).
**[2026-08-04: this exact split was measured on n=2 and did not hold at
full speaker diversity — see §8.4.3, which revises this to 35.1%
detector-attributable and explains why by name and mechanism. Kept here
as the accurate historical record of what n=2 showed, not corrected
in place.]**

**A further nuance the same 2 surviving instances surfaced**: neither is
labeled with the *exact* correct type by the detector, even though both are
caught at the binary `Any` level. Both are ground-truth `word_repetition`;
the detector predicted `{phrase_repetition, block}` for one
(`103-1240-0000`, index 4, "Rachel" — the *same* clip already discussed above
in the original hand-verification) and `{phrase_repetition}` for the other
(`1088-129236-0006`, index 24, "the"). Tracing the first case: `preserved_ctx1`
only checks that reference positions `i` and `i−1` each individually align
`correct` — it does not check that their *aligned hypothesis words are
contiguous in the actual hypothesis token sequence* the detector runs on. In
this clip, ASR's insertion of "Lynde," between the two "Rachel"s means both
"Rachel"s individually align `correct`, but the detector never sees them
back-to-back in its actual input — so it reasonably (given what it was
actually fed: "Rachel Lynde, Rachel Lynde") classified the pattern as a
phrase repetition instead. **Arguably this is not a detector error at all
given its actual input** — a stricter metric that also required
hypothesis-side contiguity (not just reference-side correctness) would likely
credit this differently. Flagged as the next candidate refinement, explicitly
not implemented now (n=2 does not justify another round of protocol changes
before more data exists) — see `ROADMAP.md`.

**What this does and doesn't change**: the absolute Track A→B recall drop
(~99% → ~4–9%, §8.4 above) is unaffected and remains this project's most
important evaluation finding — nothing here questions that a real gap
exists. What changes is the *attribution* of that gap: the evidence now
points toward the detector itself performing close to its ground-truth-level
behavior whenever it is genuinely given intact input, and the deployed-system
shortfall being overwhelmingly an ASR-fidelity problem (both losing disfluent
words outright and corrupting their immediate context) rather than a
detection-logic problem. This reframes where future effort is best spent:
improving ASR robustness on disfluent speech (or the alignment/scoring
protocol's precision) now looks like better-justified next work than tuning
`profiling/detect.py`'s detection logic itself — though, per §5.1's
non-goals and the small sample here, this is a directional conclusion to
validate at scale, not a final one.

#### 8.4.2 Scaled confirmation (90 clips, 2026-08-03) — the finding holds and strengthens

§8.4.1's own conclusion was explicit that n=2 could not be trusted as a
precise number and needed validation at scale (`ROADMAP.md` item 7). Scaled
the same pilot from 30 → 90 clips (deterministic first-90 of the same
499-clip sample — the first 30 are the identical clips from §8.4/§8.4.1, not
a new selection). Per-clip caching (§8.4.1) meant only the 60 new clips
needed fresh CrisperWhisper inference (58 actually ran — 32 were already
cached from a partially-completed earlier attempt of this same run that was
interrupted by an unrelated session restart, confirmed via the cache and
resumed from exactly where it stopped, no work lost or duplicated). Total:
14,203s wall-clock for this leg, dominated by ASR inference on the 58 new
clips (one clip took anomalously long, 5,854s vs. a typical ~100–170s — a
one-off system hiccup, not a code issue; the run completed cleanly with no
errors either way, and this outlier does not appear in the scored results,
which use ASR output only, not timing).

**Numbers** (Any label; 90 clips, 127 total ground-truth-disfluent instances
— up from 48 at n=30):

| Metric | Overall | Preserved (word-only) | Preserved, context-strict (ctx1) |
|---|---|---|---|
| TP | 8 | 8 | 7 |
| FP | 34 | 34 | 20 |
| FN | 119 | 55 | **0** |
| Precision | 0.190 | 0.190 | 0.259 |
| Recall | 0.063 | 0.127 | **1.000** |
| F1 | 0.095 | 0.152 | 0.412 |

**Sample-attrition chain, now at 3x the earlier scale**: 127 total disfluent
instances → 63 preserved (word-only, i.e. the disfluent word itself
transcribed correctly) → **7** preserved context-strict (word *and*
immediately preceding word both correct). The same ~5% collapse rate as the
30-clip pilot (7/127 = 5.5%, vs. 2/48 = 4.2% before) — confirming this isn't
a small-sample fluke but a stable, real property of how ASR handles this
dataset's disfluent segments and their immediate surroundings.

**`R_B|preserved_ctx1` recall is still exactly 1.0 — now on n=7 instead of
n=2.** Every context-strict-preserved disfluent instance was still flagged
by the detector at the binary `Any` level. This is the confirmation §8.4.1
called for: the direction was real, not a 2-instance coincidence.

**Revised decomposition**, using the exact matched-subset Track A recall for
these same 90 clips ($R_A$ = 1.000 — computed directly, same Phase 1
closing-review correction as §8.4/§8.4.1; TP=127, FP=32, FN=0 on this exact
subset, see §8.4's exact-subset table for the 30-clip version and
`PAPER_DECISION_LOG.md` for the full note):

```
R_A (exact, this 90-clip subset) = 1.000
R_B|preserved (word-only)         = 0.127
R_B|preserved_ctx1                = 1.000
R_B|overall                       = 0.063

Detector-attributable gap (word-only) = R_A - R_B|preserved      = 0.873  (~93% of total gap)
ASR-attributable gap (word-only)      = R_B|preserved - R_B|overall = 0.064  (~7% of total gap)

Detector-attributable gap (ctx1) = R_A - R_B|preserved_ctx1 = 0.000  (exactly zero)
ASR-attributable gap (ctx1)      = R_B|preserved_ctx1 - R_B|overall = 0.937  (100% of total gap, exactly)
Total gap                        = R_A - R_B|overall = 0.937   (both decompositions sum to this, by construction)
```

At n=90 the word-only decomposition landed at ~93%/~7% (matching n=30's
~95%/~5% closely) and the context-strict decomposition again lands at
**exactly 0% detector-attributable / 100% ASR-attributable** — not an
approximation this time, since both $R_A$ and $R_B|preserved_ctx1$ are
1.000 on this exact subset. This is
now a *confirmed*, not merely *suggested*, research conclusion: **once ASR
preserves both a disfluent word and its immediate context, the detector's
binary disfluent/clean judgment is effectively perfect on this dataset.**
The deployed system's real-world recall shortfall is overwhelmingly
attributable to ASR failing to preserve disfluent words and/or their
immediate surroundings — not to weaknesses in the detection logic itself.
**[2026-08-04: "confirmed" here meant confirmed across two 7-speaker
samples — it did not hold at full 40-speaker diversity. See §8.4.3: the
revised split is 35.1% detector-attributable, traced to
`sound_repetition`/`phrase_repetition`-specific gaps already known from
§8.2, not `word_repetition` or a general weakness. The absolute
ASR-fidelity conclusion stands; "effectively perfect" was too strong.
Recorded here as the accurate account of what n=7 showed, not edited in
place.]**

**A second, equally important finding sharpens at this scale: binary
detection succeeding does not mean exact-type classification succeeds.**
Breaking down the 7 context-strict-preserved instances by type: only **2 of
7 (29%)** got the exact correct type label from the detector
(`phrase_repetition`: TP=2, FN=0). The other 5 (`word_repetition`: FN=4;
`sound_repetition`: FN=1) were *not* labeled with their true type, yet all 5
still counted toward `Any`'s TP=7 — meaning the detector flagged something
disfluent at every one of those positions, just under the wrong type label.
This is the same mechanism §8.4.1 traced by hand on one example
(`103-1240-0000`, "Rachel Rachel Lynde" → predicted `phrase_repetition`/
`block` instead of `word_repetition`, due to an ASR insertion breaking
hypothesis-side word contiguity even though both reference words individually
aligned correct) — now confirmed as a *recurring pattern*, not a one-off,
at a larger sample. **This is the one place in this whole analysis where a
real, evidence-backed detector-side limitation remains**: not a recall
problem (binary detection is robust), but a **type-classification problem**
specifically for `word_repetition`/`sound_repetition` when the surrounding
hypothesis sequence doesn't literally match the expected back-to-back
pattern, even though the reference words are individually transcribed
correctly.

**Conclusion carried into `ROADMAP.md`**: this scaled result is treated as a
major, confirmed research conclusion (not a provisional one), and directly
determines where the next development effort should go — see the synthesis
in `ROADMAP.md`'s "Immediate" section and the reasoning below.

**Implications for future development priorities (synthesis, not just a
number)**: this evaluation phase has now produced two independent, mutually
reinforcing lines of evidence pointing the same direction:

1. **The detector's core "is this disfluent" judgment is not the
   bottleneck.** Given intact input (word + context both correctly
   transcribed), it catches 100% of instances across two independent sample
   sizes (n=2, then n=7). Further tuning `profiling/detect.py`'s detection
   thresholds/logic (e.g. `prolongation_min_seconds`, §9) may still matter
   for precision (§9.1's ablations), but is not where the *recall*
   ceiling is being lost in real deployment.
2. **ASR fidelity on disfluent speech and its immediate context is the
   dominant real-world bottleneck.** ~95% of ground-truth disfluent
   instances lose either the word itself or its immediate neighbor to ASR
   error (127 → 7 survive both). This is a transcription-robustness problem,
   not a detection-logic problem — CrisperWhisper's mean WER on this
   dataset (22.3%) is already known to be high (§8.4), and this section adds
   the specific finding that errors cluster *around* disfluent segments
   rather than distributing evenly.
3. **A secondary, real detector-side issue exists, but it's about type
   labels, not detection.** `word_repetition`/`sound_repetition` get
   frequently mislabeled as `phrase_repetition`/`block` when ASR's hypothesis
   sequence doesn't literally preserve back-to-back word adjacency — a fixable,
   scoped problem (the detector's word-adjacency check operates on the raw
   ASR token stream, which can differ from a "correct-looking" reference
   alignment), and distinct from a recall/miss problem.

**Net implication**: the highest-impact next investments are (a) evaluating
or improving ASR robustness specifically on disfluent/near-disfluent speech
segments — this could mean testing alternative ASR models or fine-tuning
strategies, not necessarily touching this app's own code — and (b) the
already-flagged type-classification refinement for word/sound repetition
under context corruption (§8.4.1's hypothesis-side-contiguity gap). Both are
now evidence-backed, not roadmap-inherited, priorities. Tuning
`profiling/detect.py`'s core detection thresholds purely for recall is, by
contrast, now a *de-prioritized* direction — the evidence says that lever
isn't where the deployed system is losing performance.

**Scope of this conclusion, stated plainly (see §7.2 for the full critical
review this was checked against)**: this is measured on exactly one ASR
backend (CrisperWhisper) and one dataset family (LibriStutter's synthetic
disfluency injection), and the Track B clip subset itself covers only 7 of
the 499-sample's 40 distinct speakers. The absolute finding — real ASR
recall on this app's disfluency detection collapses dramatically versus a
ground-truth transcript — is not in doubt. Whether "ASR fidelity, not
detector logic, is *generally* the bottleneck" (as opposed to specifically
for CrisperWhisper on LibriStutter's synthetic splices, on these 7
speakers) is the natural next thing to check, not something this phase's
evidence already proves in full generality — carried into `ROADMAP.md` as
the top Phase 2 validation item, ahead of further detector or dataset work.
**Checked directly, 2026-08-04 — see §8.4.3: the speaker-limited sample
did matter, and the "~0% detector-attributable" figure above does not hold
at full speaker diversity. Read §8.4.3 before citing this subsection's
percentages as final.**

#### 8.4.3 Speaker-stratified confirmation (120 clips, 40 speakers,
2026-08-04) — the ctx1 finding is REFINED, not simply confirmed

Directly addresses the speaker-clustering limitation named in §7.2 item 2:
re-ran Track B on a speaker-stratified sample (round-robin across all 40
distinct speakers, up to 3 clips each, 120 total — pre-registered in §5.1's
addendum before running, methodology and success criteria fixed in
advance). This is **not** the same 7 speakers scaled up — every one of the
499-sample's 40 speakers is represented.

**Numbers** (Any label; 120 clips, 186 total ground-truth-disfluent
instances):

| Metric | Overall | Preserved (word-only) | Preserved, context-strict (ctx1) |
|---|---|---|---|
| TP | 11 | 11 | 10 |
| FP | 94 | 94 | 62 |
| FN | 175 | 62 | 5 |
| Precision | 0.105 | 0.105 | 0.139 |
| Recall | 0.059 | 0.151 | **0.667** |
| F1 | 0.076 | 0.124 | 0.230 |

**Addendum (2026-08-04, `ROADMAP.md` item 8): Wilson 95% CIs for the three
`R_B|preserved_ctx1` recall figures cited across this project**, computed
with `metrics.wilson_interval()` once that machinery existed — making the
qualitative "too few instances to trust" caveat already attached to these
numbers concrete:

| Sample | k/n | Point estimate | Wilson 95% CI |
|---|---|---|---|
| n=2 (7 speakers, 30-clip pilot, §8.4.1) | 2/2 | 1.000 | [0.342, 1.000] |
| n=7 (7 speakers, 90-clip pilot, §8.4.2) | 7/7 | 1.000 | [0.646, 1.000] |
| n=15 (40 speakers, 120-clip run, this section) | 10/15 | 0.667 | [0.417, 0.848] |

The n=7 and n=15 intervals **overlap substantially** ([0.646, 1.000] vs.
[0.417, 0.848], overlapping on [0.646, 0.848]) — the drop from a 1.0 point
estimate to a 0.667 point estimate is a real, hand-traced, mechanistically
explained finding (see below), not an artifact of these two samples being
statistically incompatible; it did not need to be a "reversal" to be a
genuine finding, and the CIs confirm the n=7 result alone was never
precise enough to have ruled out something like 0.667 in the first place.
This is exactly the concrete version of the standing small-n caveat
`VALIDATION.md` has attached to this metric since §7.2/§8.4.1.

**Headline: `R_B|preserved_ctx1` recall dropped from 1.0 (at both n=2 and
n=7, the 7-speaker samples) to 0.667 (10/15) at 40 speakers.** This is a
real, meaningful change, not noise the pre-registration didn't anticipate —
§5.1's addendum explicitly listed "the finding weakens" as a valid,
non-predetermined outcome, and this is that outcome. **Revised
decomposition**, using the exact matched-subset Track A recall for these
120 clips ($R_A$ = 185/186 = 0.9946 — the first time this project's
exact-subset $R_A$ has not been a clean 1.0, itself expected at a larger,
more representative sample):

```
R_A (exact, this 120-clip subset) = 0.9946
R_B|preserved (word-only)          = 0.1507
R_B|preserved_ctx1                 = 0.6667
R_B|overall                        = 0.0591

Detector-attributable gap (word-only) = R_A - R_B|preserved      = 0.8439  (90.2% of total gap)
ASR-attributable gap (word-only)      = R_B|preserved - R_B|overall = 0.0915  (9.8% of total gap)

Detector-attributable gap (ctx1) = R_A - R_B|preserved_ctx1 = 0.3280  (35.1% of total gap)
ASR-attributable gap (ctx1)      = R_B|preserved_ctx1 - R_B|overall = 0.6075  (64.9% of total gap)
Total gap                        = R_A - R_B|overall = 0.9355
```

The context-strict decomposition moves from "~0% detector-attributable"
(n=7, two samples) to **35.1% detector-attributable / 64.9%
ASR-attributable** at n=15 (the ctx1-preserved subset within these 120
clips). ASR-fidelity is still the *majority* driver of the real-world
recall gap — that headline does not reverse — but the earlier "the
detector is essentially perfect given fair input" framing was too strong,
and this run is exactly why §7.2/§8.4.2 flagged it as unconfirmed pending
broader speaker coverage rather than stating it as settled.

**Why the earlier n=7 result looked cleaner than the truth: traced by
hand, not guessed.** Direct inspection of all 15 context-strict-preserved
instances (using the per-clip cache — no new ASR needed) found the 5
misses are **not spread across types** — all 5 are either `sound_repetition`
(2 instances: `445-123857-0019` "after-", `5456-24741-0023` "itself-") or
`phrase_repetition` (3 instances: `2836-5354-0011` "would",
`289-121652-0010` "boy", `445-123857-0019` "and"). **Zero of the 5
`word_repetition`-true instances in this subset were missed** — every
`word_repetition` instance, across both this run and the earlier n=90 run
(10 total observed to date), was flagged at the `Any` level when given
intact context. Both miss categories trace directly to **already-known,
already-documented structural gaps that have nothing to do with ASR
context**:

- `sound_repetition`'s 0% Track-A recall (§8.2) — the detector's fragment-
  repeat check only catches "fragment-before-word" ordering; LibriStutter's
  reconstruction places the fragment *after* the word. This is the exact
  same mismatch, now confirmed to also cause outright misses under Track
  B's real-ASR conditions, not just Track A's reconstructed-token
  conditions.
- `phrase_repetition`'s LibriStutter reconstruction limitation (§8.2): a
  true multi-word phrase repeat can't be reconstructed from LibriStutter's
  single-marker-row format, so the ground-truth "phrase repeat" is often a
  single-word approximation that doesn't actually present as a genuine
  2+-word repeated span — the detector's phrase-repetition check correctly
  can't fire on a pattern that isn't really there, the same limitation
  §8.2 already flagged as needing Track B to measure honestly. This *is*
  that honest measurement, and it confirms the concern.

**What this means, precisely — not "the detector is worse than we
thought," but "the aggregate figure was measuring the wrong mix":** the
n=2/n=7 samples happened, by chance (not by selection — both were
deterministic prefixes, not chosen after seeing results), to be
`word_repetition`-heavy. `word_repetition` specifically remains **at 100%
Any-level recall given intact context, confirmed now across 10 instances
and two independent sample-construction methods** (clip-count prefix and
speaker-stratified). `sound_repetition` and `phrase_repetition` are not
robust even when context is intact — but their failures are the *same*
structural gaps already tracked in `ROADMAP.md` (item 10 for
`sound_repetition`; the reconstruction caveat for `phrase_repetition`),
not a new, unexplained detector weakness discovered by this run.

**Revised headline, replacing the "~0% detector-attributable" framing**:
once speaker- and type-diversity are both accounted for, roughly a third
of the context-strict gap is detector-attributable — but that third is
fully explained by two pre-existing, already-scoped structural issues
specific to `sound_repetition`/`phrase_repetition`, not `word_repetition`
or a general detector weakness. ASR-fidelity remains the majority driver
(64.9%) and the single largest lever, consistent with the original
headline finding — the correction is in the *composition* of the
remainder, not in which side dominates.

**What is NOT yet resolved by this run**: the ctx1-preserved subset is
still small (n=15) — enough to see a clear, type-clustered pattern by hand
inspection, not enough to report precise per-type percentages within it as
final. The mechanistic explanation (traced to two specific, already-known
gaps, confirmed by name and clip) is what makes this result trustworthy
at this sample size, not the raw n alone — same standard this project has
applied throughout (`VALIDATION.md`'s "audit surprising results" discipline,
`CLAUDE.md` point 3).

Raw result: `eval_results/20260804T060338639222Z_libristutter_B.json`.
Exact-subset Track A: `eval_results/20260804T060635995226Z_libristutter_
A+audio-speaker-stratified-120-matched-to-trackB.json`. Run took 12,725s
(87 real ASR calls, 33 cache hits from the earlier 90-clip run, this leg
interrupted once by an unrelated session restart and resumed cleanly from
cache mid-run, same as the earlier 90-clip run's own interruption).

#### 8.4.4 Hypothesis-side-contiguity metric — results, and the decision
this produced (2026-08-04)

Pre-registered in §5.1's addendum before running. Re-scored all 120
already-cached clips (zero new ASR) with the `gap` metric defined there.

**Headline: recall is 100% (2/2) when the hypothesis sequence is truly
contiguous (`gap == 0`); it drops to 61.5% (8/13) when it isn't
(`gap > 0`).** This is a cleaner, more isolated signal than the raw
`preserved_ctx1` figure (66.7%, §8.4.3), since that figure blends both
groups together — confirming the hypothesis-contiguity distinction is
real and meaningful, not just a plausible-sounding hypothesis.

**But the deeper question this metric was built to answer — would a
detector fix that tolerates a small ASR insertion actually recover the
misses — has a more precise, and more modest, answer than the qualitative
description suggested.** Checked the exact `gap` size (and its literal
inserted words) for all 5 outright misses in the discontiguous group:

| Clip | Word | True type | `gap` | Inserted words |
|---|---|---|---|---|
| `445-123857-0019` | "after-" | `sound_repetition` | **1** | `["life"]` |
| `2836-5354-0011` | "would" | `phrase_repetition` | 5 | `["be","a","dark","course.","It"]` |
| `289-121652-0010` | "boy" | `phrase_repetition` | 5 | `["blue","in","1697,","and","little"]` |
| `445-123857-0019` | "and" | `phrase_repetition` | 3 | `["unfathered","fruit,","like"]` |
| `5456-24741-0023` | "itself-" | `sound_repetition` | 3 | `["is","all","of"]` |

**Only 1 of the 5 misses has a small (`gap=1`) insertion consistent with
"ASR corrupted an otherwise-tight repeat."** The other 4 have `gap` of
3–5 words — far too large to be a plausible ASR-insertion artifact around
a genuine close-proximity stutter, and the inserted content (ordinary
prose spanning what reads as real sentence structure) is consistent
instead with the *already-documented* limitation this project has
tracked since §8.2: `phrase_repetition`'s ground truth, reconstructed from
a single LibriStutter marker row, sometimes anchors to a word that
*also* recurs naturally later in the sentence for unrelated reasons — a
dataset-reconstruction artifact, not a repeat ASR corrupted. Checked the
2 hits with `gap=1` too, for a complete picture (`103-1240-0000` "Rachel",
`1088-129236-0006` "the," both already discussed in §8.4.1/§8.4.2): both
are already caught at the `Any` level (via type-confusion — labeled
`phrase_repetition`/`block` instead of `word_repetition`), so for these 2,
a contiguity-tolerant fix's benefit would be *exact-type accuracy*, not
recall.

**Decision: total addressable evidence is n=3 (2 type-accuracy cases + 1
recall case) out of 120 clips — too thin to justify a general-purpose
detector change on its own, but the mechanism is precise enough (not a
guess) to implement a narrowly-scoped, low-risk version and let a full
499-clip Track A benchmark decide empirically whether it helps or hurts,
rather than deciding from n=3 alone.** Scope, decided *before* looking at
benchmark results: extend the existing "filler-sandwiched repetition"
`word_repetition` check (already tolerates exactly one intervening
*filler* word) to also tolerate exactly one intervening *non-filler* word
— **exact-match only** (no phonetic/edit-distance near-matching through
the gap, to bound false-positive risk), at **lower confidence** than the
immediate-adjacency case. This is a minimal extension of an existing,
already-shipped pattern, not new detection logic.

**Measured result: implemented, benchmarked, and REVERTED — a genuine
negative result, recorded honestly rather than omitted.** Track A (499
clips, the properly-controlled aggregate check): `word_repetition` FP
+106 (452→558), `Any` (combined) FP +102 (308→410), **`Any` F1 dropped
0.835→0.793** — a real, measurable regression, with **zero new true
positives** (TP unchanged at 801/183). Mechanistically expected in
hindsight: Track A has no ASR involved, so it can only ever show this
fix's *cost* (coincidental same-word repeats 2 tokens apart in ordinary
reconstructed text, e.g. two sentences that happen to both start with the
same word) — never its intended *benefit* (recovering genuine
ASR-insertion-corrupted repeats), which by construction only real ASR
output can exercise. Checked Track B too, for completeness (re-scored the
existing 120-clip cache, zero new ASR): **+1 TP at a cost of +24–29 new
FP** — recovered exactly one of the predicted n=3 addressable cases, at a
steep, disqualifying false-positive cost. **Decision: reverted.** The
predicted mechanism was correct and precise (not a wrong guess about
*why* it might help) — the addressable evidence was simply too thin
(n=3/120 clips) to outweigh the real, broad false-positive exposure a
"tolerate any non-filler word" rule creates across ordinary text. Code
reverted in `profiling/detect.py` (a code comment marks exactly what was
tried and why it was removed, pointing here); a regression test
(`test_word_sandwiched_repetition_not_implemented`,
`tests/test_detect_taxonomy_and_fusion.py`) locks in the correct
(non-firing) behavior going forward, so this specific idea is not
silently reintroduced later without new evidence. Full reasoning and
alternatives considered: `PAPER_DECISION_LOG.md`. **This is exactly the
outcome §5.1's addendum pre-registered as a live, unpredetermined
possibility** ("let a full 499-clip Track A benchmark decide empirically
whether it helps or hurts") — the protocol worked as designed, catching a
plausible-sounding fix that real measurement showed was not worth
shipping.

### 8.5 SEP-28k — Track A / Track B

*Not yet run.* `load_sep28k_labels` is built and verified against the real,
complete 28,177-row `SEP-28k_labels.csv` (see `PAPER_DECISION_LOG.md`), but
SEP-28k has no reference transcript at all (confirmed the same day) — it can
only be scored at clip granularity (`score_clip_level`, already built) by
something that runs without a transcript. That's either an acoustic-only
detection pass (not yet wired to a runner) or Track B pointed at SEP-28k's
audio (the alignment/scoring machinery now exists, per §8.4, and would need
adapting to clip-level rather than word-level ground truth) — both need
real SEP-28k audio, which has not been downloaded. See
`PAPER_DECISION_LOG.md`'s 2026-08-03 entries for the sequencing/resequencing
reasoning.

---

## 9. Ablations

**Run 2026-08-03** against the same 499 real LibriStutter clips (with real
audio) used for §8.3, via the new `profiling/evaluation/run_ablations.py`
(loads clips once, re-runs `detect_disfluencies()` per config variant,
everything else held constant — same clips, same reconstruction, same
scoring). Raw results: `eval_results/*_libristutter_ablation-*.json` (10
files, gitignored). Full per-variant tables were printed and archived; the
summary below is what matters for interpretation.

### 9.1 Results

**`Any` (combined) label:**

| Variant | TP | FP | FN | Precision | Recall | F1 | Δ F1 vs. baseline |
|---|---|---|---|---|---|---|---|
| **baseline** (VAD on, Praat on, fusion=1.0, prolong_thr=1.0) | 801 | 308 | 8 | 0.722 | 0.990 | 0.835 | — |
| `vad_off` | 801 | 308 | 8 | 0.722 | 0.990 | 0.835 | **+0.000** |
| `praat_off` | 801 | 308 | 8 | 0.722 | 0.990 | 0.835 | **+0.000** |
| `fusion_weight=0.5` | 801 | 308 | 8 | 0.722 | 0.990 | 0.835 | +0.000 |
| `fusion_weight=2.0` | 801 | 302 | 8 | 0.726 | 0.990 | 0.838 | +0.003 |
| `fusion_weight=5.0` | 801 | 302 | 8 | 0.726 | 0.990 | 0.838 | +0.003 |
| `prolong_threshold=0.65` | 801 | 897 | 8 | 0.472 | 0.990 | 0.639 | **−0.196** |
| `prolong_threshold=0.85` | 801 | 456 | 8 | 0.637 | 0.990 | 0.775 | −0.060 |
| `prolong_threshold=1.2` | 801 | 155 | 8 | 0.838 | 0.990 | 0.908 | **+0.073** |
| `prolong_threshold=1.4` | 800 | 106 | 9 | 0.883 | 0.989 | 0.933 | **+0.098** |

**`prolongation` (the only type any of these ablations can affect —
filler/sound_repetition/word_repetition/phrase_repetition were confirmed
identical across every variant, exactly as expected since none of them are
touched by the acoustic-native layer):**

| Variant | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **baseline** | 21 | 409 | 201 | 0.049 | 0.095 | 0.064 |
| `vad_off` | 21 | 409 | 201 | 0.049 | 0.095 | 0.064 |
| `praat_off` | 21 | 409 | 201 | 0.049 | 0.095 | 0.064 |
| `fusion_weight=2.0`/`5.0` | 21 | 399 | 201 | 0.050 | 0.095 | 0.065 |
| `prolong_threshold=0.65` | 55 | 1056 | 167 | 0.050 | 0.248 | 0.083 |
| `prolong_threshold=0.85` | 35 | 581 | 187 | 0.057 | 0.158 | **0.084** (best) |
| `prolong_threshold=1.2` | 11 | 219 | 211 | 0.048 | 0.050 | 0.049 |
| `prolong_threshold=1.4` | 8 | 143 | 214 | 0.053 | 0.036 | 0.043 |

### 9.2 Ranking — which components actually contribute

1. **`prolongation_min_seconds` (the block/prolongation threshold) — by far
   the dominant lever.** Sweeping it alone moves `Any` F1 across a 0.294-point
   range (0.639 at 0.65s to 0.933 at 1.4s) — an order of magnitude larger
   than every other ablation combined. **The current baseline (1.0s, the
   Part D real-mic tune) is not the optimum for either objective measured
   here**: raising it further (to 1.4s) keeps improving aggregate `Any` F1
   (fewer false positives, essentially unchanged recall), while
   `prolongation`'s *own* F1 actually peaks lower, at 0.85s (F1 0.084 vs.
   0.064 at the current 1.0s setting) — a real tension between optimizing
   the aggregate signal and optimizing prolongation specifically, not
   resolved by this ablation alone (see the reconstruction caveat below).
2. **`fusion_weights.acoustic` — small but real.** `Any` F1 0.835 → 0.838
   (+0.003) moving from 1.0 to 2.0, then flat at 5.0 — the acoustic-vs-token
   replacement mechanism (§3, §6) has a real but modest and quickly-
   saturating effect on this dataset.
3. **Silero VAD corroboration — measured zero effect (tied last).**
   `vad_off` is byte-for-byte identical to baseline across every TP/FP/FN
   count.
4. **Praat pitch/jitter/shimmer corroboration — measured zero effect (tied
   last).** Same as VAD: `praat_off` is byte-for-byte identical to baseline.
5. **Speaker calibration — not applicable, not run.** `calibration.py`
   personalizes thresholds from a *stored per-account baseline*
   (`SpeakerBaseline`, built from a calibration-sentence read); LibriStutter's
   clips have no such account/baseline attached, so there is nothing to turn
   on or off for this dataset. This ablation needs either a dataset with
   repeated-speaker sessions and a fitted baseline, or a real-mic pass —
   not resolvable from LibriStutter's synthetic per-clip structure. Recorded
   here as an honest scope limit, not skipped silently.

### 9.3 An important finding the ablation itself surfaced: VAD and Praat's designed effect is invisible to this metric

VAD and Praat corroboration were **designed** (2026-08 restructuring,
`PAPER_DECISION_LOG.md`) to adjust event *confidence* up or down — they were
never designed to change whether an event fires at all, except indirectly
through the weighted-fusion replace-vs-keep decision (which `fusion_weight`
*does* measurably affect, per §9.2 item 2). `score_word_level` — this
project's only word-level metric so far — scores pure presence/absence
(TP/FP/FN), which is blind to confidence by construction. **A "zero
measured effect" for VAD/Praat is therefore not strong evidence they don't
help** — it's evidence that this specific metric can't see the thing they
were built to do. Confirming or ruling out a real contribution needs a
confidence-sensitive metric (e.g. mean confidence of TPs vs. FPs, or the EER
metric flagged as a stretch goal in §4) — not yet built at the time this
finding was recorded. This is a genuine methodological finding from
running the ablation, not a predicted result. **Built and run against
real data the same day — see §9.3.1 immediately below.**

#### 9.3.1 Confidence-sensitive metric — real-data results (2026-08-04, `ROADMAP.md` item 7)

`metrics.confidence_stats()` (mean predicted `confidence` of TP vs. FP
events, per type and combined) was built specifically to answer §9.3's open
question and unit-tested on synthetic data first; this is its first run
against real data — the full 499-clip LibriStutter Track A (real-audio)
sample, current shipped config (`use_vad=True`, `use_praat=True`,
`prolongation_min_seconds=1.0`, both new prolongation-redesign toggles at
their default `False`):

| Type | TP mean conf | FP mean conf | Gap (TP-FP) | n_TP | n_FP |
|---|---|---|---|---|---|
| filler | n/a | 0.931 | n/a | 0 | 25 |
| sound_repetition | 0.860 | 0.860 | +0.000 | 184 | 5 |
| word_repetition | 0.920 | 0.920 | +0.000 | 183 | 452 |
| phrase_repetition | n/a | 0.880 | n/a | 0 | 4 |
| prolongation | 0.936 | 0.933 | +0.003 | 21 | 409 |
| **Any** | **0.914** | **0.921** | **-0.007** | 974 | 309 |

(`filler`/`phrase_repetition` have n_TP=0 because this 499-clip sample
happens to contain zero ground-truth instances of either — a known sample
composition gap, `ROADMAP.md` item 14, not a detection failure.)

**Finding: the gap is approximately zero everywhere it's measurable, and
slightly *negative* for the combined `Any` label** (FP mean confidence
0.921 > TP mean confidence 0.914). Per this project's own standing rule
(§ CLAUDE.md point 3), this is a surprising-enough result to audit before
accepting, so it was checked directly:

- The per-type gaps for `sound_repetition` and `word_repetition` are
  exactly `+0.000` to 3 decimals, and `prolongation`'s is `+0.003` —
  effectively noise, not a partial signal.
- This was run against the *current production config*, where VAD/Praat
  corroboration are active exactly as designed (§9.1's ablation already
  showed `vad_off`/`praat_off` don't move presence/absence counts) — so
  this is the right condition to test their designed confidence-adjustment
  effect under.
- Spot-checked that `confidence_stats()` itself is not miscomputing: its
  hand-constructed unit test (`track_a.py` self-test section 7) correctly
  reproduces a designed non-zero gap on synthetic data, so the near-zero
  result here is not a metric bug, it's a real measurement.

**Conclusion: this closes out §9.3's open question with a negative-to-null
result, not a positive one.** §9.3 was careful to state that a "zero
measured effect" on presence/absence was *not* evidence VAD/Praat
corroboration don't help, because that metric was structurally blind to
their designed effect (confidence, not presence/absence). This metric is
*not* blind to it — and it also finds ~zero effect. Combined, the two
results now constitute real (if not definitive, see limitation below)
evidence that VAD/Praat corroboration's confidence adjustment is not
currently producing a meaningful TP/FP separation in this pipeline, on
this dataset. **Limitation, stated explicitly**: this is one dataset
(LibriStutter, reconstructed timing, §8.2's documented caveat) and one
run; it does not by itself prove the mechanism is worthless in general,
only that it isn't earning its keep as measured here. Not acted on by
tuning or removing the corroboration logic — per standing rule 4, findings
are recorded as evidence, not auto-applied. Flagged as a candidate for
Phase 3's decision list: revisit whether VAD/Praat confidence-adjustment
weights are worth their complexity, or should be simplified/removed, based
on this evidence plus whatever a future SEP-28k/real-speech run adds.

### 9.4 A caveat on the dominant finding

The `prolongation_min_seconds` sweep's result is measured entirely on
LibriStutter's *reconstructed* tokens (§8.2's documented approximation:
`STUTTER`-marker rows rebuilt from adjacent-word copies, not verified
transcriptions), and §8.3 already found that reconstruction's declared
timing overstates true voiced duration for this exact type. The "optimal
threshold" identified here (0.85s for prolongation-specific F1, 1.4s+ for
aggregate F1) may not transfer directly to real speech — it should be
treated as a strong, real, evidence-based hypothesis to validate against
non-reconstructed data (Track B, or a real prolongation-labeled dataset),
not as a final tuning decision. Per the owner's explicit instruction,
`config.yaml`'s threshold was **not** changed as a result of this finding —
it's recorded as evidence for a future, separately-approved tuning pass.

### 9.5 Prolongation redesign — pre-registered protocol (2026-08-04, before implementation)

`ROADMAP.md` item 5 / `PHASE_2_RESEARCH_PLAN.md` §10.6 identified
`prolongation` as Phase 2's highest-confidence detector-side target: three
independent sources converge (this ablation's own §9.1/§9.2 finding that
the threshold dominates measured performance; Esmaili et al. 2017's
peer-reviewed rate-normalized formula; full dataset support). The project
owner explicitly authorized implementing this redesign in this phase, "if
still warranted by evidence" — this section fixes the exact methodology
*before* writing code, per standing discipline, so "warranted" is decided
by measurement, not by having already built it.

**Two independent, separately-toggleable changes** (not one combined
change — kept separable so each can be ablated on its own, same
discipline as §9's original sweep):

1. **Rate-normalized duration threshold**, replacing the current
   `max(prolongation_min_seconds, 90th-percentile-of-clip's-own-token-
   durations)` mechanism when enabled. Implements Esmaili et al. 2017's
   validated formula directly: `T = rate_alpha / max(speaking_rate,
   rate_floor)`, where `speaking_rate` is estimated as total syllables
   (summed via the same `phonetic._syllable_count()` already used for the
   SLD/OD tag) divided by the clip's total time span, `rate_alpha`
   defaults to 1.2 (the literature's value), and `rate_floor` (default
   1.5 syllables/sec) guards against instability on very sparse/short
   clips. New config keys: `use_rate_normalized_prolongation`,
   `prolongation_rate_alpha`, `prolongation_rate_floor`.
2. **Praat-feature gating**, promoting pitch-stability/jitter/shimmer from
   confidence-only adjustments (§9.3's finding: invisible to this
   project's presence/absence metric) to a *hard gate* specifically for
   the token-path prolongation check — a prolongation candidate must pass
   `pitch_std_hz <= pitch_std_max_hz AND jitter <= jitter_max AND shimmer
   <= shimmer_max` (the same threshold values already in `config.yaml`,
   previously only consulted by the separate acoustic-native fusion path)
   when Praat features are available; graceful no-op (never blocks) when
   they're not, same principle as every other acoustic check in this
   codebase. New config key: `require_praat_stability_for_prolongation`.

**Evaluation plan**: a 4-variant ablation on the same 499-clip LibriStutter
Track A (audio-enabled) sample used throughout §8.3/§9 — baseline
(neither change), rate-normalization only, Praat-gating only, both
together — reusing `run_ablations.py`'s existing harness. Reported against
both `Any` (combined) and `prolongation`-specific F1, matching §9.1's
existing table format, for direct comparability.

**Success criteria, stated before running anything**: no single predetermined
outcome counts as "success" — a result showing either or both changes
help, hurt, or have no effect are all valid, reportable outcomes. The
practical decision this evaluation is *for*: whether to change
`config.yaml`'s shipped default. **Threshold for changing the default**:
only if a variant improves (or is not worse than, within noise) `Any` F1
*and* `prolongation`-specific F1 simultaneously, avoiding the same
aggregate-vs-type-specific tension §9.2 already found in the original
threshold sweep (raising the floor helped `Any` F1 while hurting
`prolongation`-specific F1) — a variant that repeats that tension does not
clear the bar for a default change, even if part of its results looks
good in isolation.

**Explicit limitation carried over from §9.4, unresolved by this
redesign**: still measured on LibriStutter's reconstructed-token timing,
not verified real-speech durations. This redesign can show whether the
*rate-normalized, literature-validated mechanism* outperforms the
*current, empirically-tuned mechanism* on the same (reconstructed) data —
it does not, by itself, resolve whether either transfers to real speech.
That remains open pending Track B validation with real ASR timestamps
(`ROADMAP.md`), not attempted in this step.

### 9.5.1 Results (2026-08-04, run against the full 499-clip real-audio sample)

Ran as part of a full re-run of the standing 13-variant ablation (the
original 10 from §9.1/§9.2 plus the 3 new prolongation-redesign variants
pre-registered above), so every variant is directly comparable to the
existing baseline row. Full raw output: `eval_datasets/
_prolongation_ablation_output.txt`; each variant's JSON saved individually
under `eval_results/` (filenames in the raw output).

| Variant | Any TP/FP/FN | Any F1 | prolongation TP/FP/FN | prolongation F1 |
|---|---|---|---|---|
| baseline (both OFF) | 801/308/8 | 0.835 | 21/409/201 | 0.064 |
| `prolong_rate_normalized` | 802/3016/7 | 0.347 | 84/3225/138 | 0.048 |
| `prolong_praat_gated` | 796/188/13 | **0.888** | 16/145/206 | **0.084** |
| `prolong_rate_and_praat` | 797/539/12 | 0.743 | 24/509/198 | 0.064 |

**Against the pre-registered bar (must improve, or not worsen within
noise, both `Any` F1 *and* `prolongation`-specific F1 simultaneously):**

- **`prolong_rate_normalized`: fails, badly.** Both metrics get *worse*
  (`Any` F1 0.835->0.347, prolongation F1 0.064->0.048) — not a subtle
  regression, a collapse: FP count on `prolongation` alone jumps
  409->3225. **Audited before accepting** (surprising-result rule):
  traced to the rate-normalization formula itself, not a bug in its
  implementation — `speaking_rate` is estimated per-clip as total
  syllables / clip time span, and LibriStutter's short, reconstructed
  clips (§8.2's known timing-approximation caveat) plausibly produce
  unstable, sometimes very high, speaking-rate estimates, which collapses
  `T = rate_alpha / speaking_rate` toward the `rate_floor`-bounded
  minimum — flagging nearly any token that clears a near-zero duration.
  This is a hypothesis about *why*, not confirmed further (out of scope
  for this ablation; matches this section's own already-declared
  limitation that reconstructed timing may not transfer). The formula
  itself (Esmaili et al. 2017) is peer-reviewed and validated on real
  continuous speech — this result says it does not transfer as-is to
  LibriStutter's short reconstructed clips, not that the literature is
  wrong.
- **`prolong_praat_gated`: clears the bar — the only variant in the full
  13-variant ablation (not just the 3 new ones) to improve both metrics
  simultaneously.** `Any` F1 0.835->0.888 (+0.053), prolongation F1
  0.064->0.084 (+0.020) — both real, non-trivial improvements, achieved
  by *removing* false positives (`prolongation` FP 409->145, a 65%
  reduction) at a TP cost (21->16) smaller than the FP reduction's
  benefit to precision. This is the mechanism working exactly as
  designed: many of the original percentile-threshold's false-positive
  "prolongations" apparently have unstable pitch/jitter/shimmer
  signatures inconsistent with genuine prolongation, and the Praat gate
  screens them out. Note (not disqualifying, but recorded): localization
  (IoU>=0.5) on the surviving true positives drops 0.857->0.500 — a
  smaller, different TP set, not evaluated against the pre-registered
  success criteria (which named only `Any`/prolongation F1), flagged here
  for completeness.
- **`prolong_rate_and_praat`: does not clear the bar.** `Any` F1 gets
  *worse* (0.835->0.743) even though prolongation F1 is exactly flat
  (0.064->0.064, tied) — the rate-normalization component's damage isn't
  fully offset by Praat-gating when both are active together. Confirms
  the two changes are not simply additive; the combination inherits
  `prolong_rate_normalized`'s core problem, diluted but not eliminated.
- **Context: comparison against the original percentile-threshold sweep
  (§9.1/§9.2, unchanged by this run — reproduced identically, confirming
  no regression from adding the 3 new variants to the harness).**
  `prolong_threshold_1.2`/`prolong_threshold_1.4` both raise `Any` F1
  further (0.908/0.933) but *worsen* prolongation-specific F1 (0.049/
  0.043, both below baseline's 0.064) — repeating exactly the aggregate-
  vs-type-specific tension §9.2/§9.4 already flagged as disqualifying.
  `prolong_praat_gated` is the only variant across all 13 that avoids
  this tension entirely.

**Decision: `require_praat_stability_for_prolongation` flipped to `true`
as the new shipped default in `config.yaml`. `use_rate_normalized_
prolongation` stays `false`** — it failed on its own and failed combined
with Praat-gating; no config state involving it clears the pre-registered
bar. This decision follows directly and mechanically from the pre-
registered criteria fixed before this ablation ran (§9.5 above) — no new
judgment call was needed. Full regression suite re-run after the config
change: 45/45 pass (unaffected, since these tests set their own
per-test config or rely on Praat's graceful audio-unavailable no-op).

---

## 10. Benchmark comparisons against published baselines

*Not yet run. Once §8 has real numbers, compare here against the published
figures already gathered during this project's literature review (see
`PAPER_DECISION_LOG.md` and `ROADMAP.md` for full citations):*

| System | Dataset | Reported metric | Reported value | Our result |
|---|---|---|---|---|
| StutterNet (TDNN + MFCC) | SEP-28k | F1 (per-type, varies) | *(from paper — fill in when comparing)* | *(TBD)* |
| wav2vec2-embedding classifier | SEP-28k | F1 | *(from paper)* | *(TBD)* |
| Rule-based (2025 comparative study) | UCLASS/FluencyBank/SEP-28k | Prolongation accuracy | 97–99% | *(TBD)* |
| Self-supervised WavLM (Shih et al., SLT 2024) | Word-level curated set | Word-level F1 | *(from paper)* | *(TBD)* |

---

## 11. Phase 3, Stage 1: encoder-representation corroboration signal — pre-registered protocol (2026-08-04, before implementation)

`ROADMAP.md` item 17 / `PHASE_3_ARCHITECTURE_REVIEW.md` §5.1/§8 identified
CrisperWhisper's own last-layer encoder hidden states as the best-evidenced,
lowest-cost next signal for `word_repetition`/`sound_repetition`/`filler`
— the three types that remain almost entirely token-text-dependent today.
This section fixes the exact methodology *before* writing any extraction
code, per standing discipline, including an explicit escalation trigger to
Stage 1b (a frozen WavLM-Large pass) decided in advance, not after seeing
results.

**Question this stage answers, precisely**: does CrisperWhisper's own
last-layer encoder representation, with *zero training*, carry any signal
that separates this project's current true positives from its current
false positives, for the three target types? This is deliberately a
narrower, cheaper question than "build a detector on this representation"
— it's the minimum test that could produce a real negative result (no
separable signal) or a real positive one (proceed to Stage 2) without
committing to a classifier, a new model, or added inference latency.

### 11.1 What gets extracted

CrisperWhisper's encoder hidden states from its **last layer**, matching
the one directly on-point published result this decision rests on
(arXiv:2406.05784). Requires bypassing `transformers.pipeline()`'s
wrapper for a direct model call with `output_hidden_states=True` (the
pipeline does not expose hidden states at all today — confirmed by
reading `profiling/asr.py`, §5.1 of `PHASE_3_ARCHITECTURE_REVIEW.md`).
For a given word/event with timing `[start, end]`, the representation is
the **mean-pooled encoder hidden state** across the encoder frames whose
time range overlaps `[start, end]` (Whisper's encoder produces frames at
a fixed ~20ms resolution after its conv-subsampling front-end, for the
padded 30s window it always processes internally).

**A methodological subtlety, stated explicitly**: Track A's whole design
principle is bypassing ASR entirely (ground-truth transcript, no
CrisperWhisper call at all — `VALIDATION.md` §3). Testing this signal
still requires pushing the real audio through CrisperWhisper's *encoder*
even under Track A, since the signal being tested is specifically "what
does CrisperWhisper's own encoder compute here" — only the *decoded
text/timestamps* are bypassed in favor of ground truth, not the encoder
forward pass itself. This is a legitimate hybrid (ground-truth transcript
+ real-audio encoder embeddings), consistent with how Track A+audio
already mixes ground-truth transcript with real audio for VAD/Praat
(§8.3) — but it means this stage is not a "free" re-use of an existing
cached run; it requires a real (if one-time, cacheable) forward pass of
CrisperWhisper's encoder over the 499-clip sample's audio.

### 11.2 The corroboration signal being tested

**Zero-training, purely a distance measurement** — no classifier, no
labeled reference set beyond what's already computed within each clip:

1. For every event the current detector produces (TP or FP, determined
   against LibriStutter ground truth) of type `word_repetition`,
   `sound_repetition`, or `filler`, compute its mean-pooled last-layer
   encoder embedding (§11.1).
2. For the same clip, compute a **fluent-reference centroid**: the mean
   of the mean-pooled embeddings of every token in the clip *not* part of
   any ground-truth-disfluent span.
3. Compute `distance = 1 - cosine_similarity(event_embedding,
   fluent_reference_centroid)` for every event.
4. This directly reuses the shape of `metrics.confidence_stats()`,
   already built and validated in Phase 2 (§9.3.1) — same TP-vs-FP-gap
   comparison, with `distance` in place of the existing `confidence`
   field.

**Working hypothesis, stated before running anything**: a genuine
disfluency's encoder embedding should sit measurably farther from the
clip's own fluent baseline than a false-positive detection's embedding
(e.g. a coincidental, grammatically ordinary repeated word like "that
that..." used correctly, which should sound acoustically unremarkable
next to the rest of the clip's fluent speech). This is a hypothesis, not
an assumption treated as fact — a null result (no gap) is exactly as
valid and reportable an outcome as a positive one, matching this
project's treatment of the identical question for VAD/Praat in §9.3/
§9.3.1.

### 11.3 Dataset and scope

Same 499-clip real-audio LibriStutter Track A sample used throughout
§8.3/§9 — direct comparability with every other measurement this project
has made, and the current detector's existing TP/FP labels for these
three types are already known from the frozen §9.5.1 baseline. Restricted
to `word_repetition`/`sound_repetition`/`filler` only — matching
`ROADMAP.md` item 17's scope; `prolongation`/`block` are already
audio-native and out of scope for this specific test, and
`phrase_repetition` is excluded for the same reconstruction-limitation
reason it's excluded from other detector-improvement work (§8.2).

### 11.3.1 Implementation addenda (2026-08-04, discovered while building, before any results)

Two things emerged while writing `profiling/evaluation/encoder_features.py`
that revise or sharpen §11.1-§11.3 above — recorded here rather than
silently absorbed, per standing rule 3 ("if reality contradicts
expectations, treat that as a scientific finding").

- **Simplification**: extracting only the last encoder layer does not
  need `output_hidden_states=True` at all. A plain encoder forward pass's
  primary output (`last_hidden_state`) already *is* the last layer — that
  flag is only needed to retrieve *every* layer. `encoder_features.py`
  therefore calls `model.get_encoder()` directly and reads
  `last_hidden_state`, cheaper and simpler than the pre-registration's
  own wording implied.
- **A real, previously-unpriced cost, found by direct measurement, not
  estimation**: a 1-clip smoke test measured the encoder pass itself at
  **37.8s** (model load: 22.3s, one-time). This confirms and quantifies
  what §11.1 only described qualitatively ("a real forward pass") —
  Whisper always pads to a fixed 30s window before the encoder runs, so
  the encoder pass is *not* proportionally cheaper for short clips; it is
  the dominant cost of a full transcription (`ARCHITECTURE.md` §3: ~44s
  of ~54s measured for a 4s clip), and skipping decoding only saves the
  remaining ~10s. **A full 499-clip run is therefore roughly 6 hours of
  CPU time, not a few minutes.** Decision, consistent with this project's
  own Track B precedent (pilot at 30 clips before scaling to 90 then
  120, §8.4.1-§8.4.3): `run_encoder_signal_stage1.py` defaults to a
  30-clip pilot, not the full 499, with an explicit flag to scale up
  once the pilot's result is reviewed rather than committing 6 hours of
  compute to an unvalidated pipeline. This is a scoping decision about
  *how much data to run first*, not a change to the methodology itself —
  §11.1-§11.5's protocol is unchanged; only the sample size run first is
  reduced, exactly as it was for Track B.
- **A real bug caught by the existing ASCII-console lint rule** (`ROADMAP.md`
  item 13, `tests/test_ascii_console_output.py`) on this new script's
  first run: a literal em-dash in a `print()` string. Fixed immediately.
  Recorded here as a small, concrete confirmation that the lint rule
  (built in Phase 2 specifically to stop finding this class of bug
  reactively) is doing its job on genuinely new code, not just the
  files it was written against.

### 11.4 Success criteria, fixed in advance

**This stage is not evaluated as "does it beat the current detector"** —
it is a narrower, cheaper measurement question. Reported outcomes, all
valid and all reportable, per standing discipline (rule 3: audit
surprising results, don't just report the flattering ones):

- **Clear signal**: mean `distance` for TP events is meaningfully higher
  than for FP events, for at least one of the three target types,
  consistently in the expected direction. **This is what triggers moving
  to Stage 2** (a small trained classification head over this same
  frozen representation) — not an automatic green light to implement
  Stage 2 without a separate go-ahead, per standing rule 4.
- **No signal**: the TP/FP gap is near-zero or inconsistent in direction,
  the same shape of result §9.3.1 already found for VAD/Praat confidence.
  **This is the pre-stated trigger for Stage 1b**: a frozen WavLM-Large
  pass, priced honestly as a new model and real added latency
  (`PHASE_3_ARCHITECTURE_REVIEW.md` §8.3) — not a reason to quietly drop
  the whole direction, since §8's adversarial review already established
  this exact outcome as informative evidence about whether an
  ASR-trained encoder is the wrong representation family for this task,
  not just "this one attempt didn't work."
- **Mixed** (signal for some types, not others): reported per-type,
  exactly as `confidence_stats()` already reports per-type gaps — no
  single number is allowed to obscure a per-type split, matching this
  project's standing "never blend results that answer different
  questions" discipline (§1).

### 11.4.1 Methodology addendum (2026-08-04, discovered on the first real result, before drawing any conclusion)

The 30-clip pilot (§11.6 below) produced a real, non-trivial-looking gap
(`Any`: +0.0895, `word_repetition`: +0.0954) that is neither obviously
"clear signal" nor "no signal" against the criteria as originally
written above — those criteria specified a *direction* ("meaningfully
higher... consistently in the expected direction") but not a rigorous way
to judge *magnitude* against sample noise. This is a real, honestly-
identified gap in the pre-registration, caught by trying to apply it to
an actual result rather than found by review beforehand — exactly the
kind of methodological weakness this project's standing discipline asks
to be caught and fixed rather than argued past.

**Fix, added to `metrics.encoder_distance_stats()` before this result was
interpreted, not after**: each group's standard deviation and a pooled
Cohen's d for the TP-vs-FP gap, the same instinct that added Wilson
intervals for small-n recall claims in Phase 2 (`VALIDATION.md` §8.4.3).
`cohens_d` is `None` (not silently `0.0`) when either group has fewer
than 2 samples — an honest "not enough data," not a wrong number.
**Revised, more precise success criteria, replacing the qualitative
wording above**: a "clear signal" requires **|Cohen's d| >= 0.5**
(conventionally a "medium" effect) *and* the direction matching the
hypothesis (TP > FP), not the raw gap number alone. This is a stricter
bar than the original wording, chosen deliberately in the direction of
harder-to-satisfy rather than easier, consistent with "optimize toward
discovering the truth, not proving the idea correct."

### 11.5 What this stage does *not* decide

Even a clear-signal result does not, by itself, authorize implementing
Stage 2 or changing `detect_disfluencies()`'s behavior — per standing
rule 4, that remains a separate, explicitly-approved step once this
stage's measurement is in hand. This section pre-registers the
*measurement*, not the response to it.

### 11.6 Results (2026-08-04): a clear signal, run at two independent sample sizes

Two runs, per §11.3.1/§11.4.1's scoping decisions — a 30-clip pilot,
then a 90-clip run (matching Track B's own 30-then-90 precedent) once
Cohen's d was added to judge the pilot's result rigorously rather than
by eye:

| Type | n (pilot / 90) | TP mean dist (pilot / 90) | FP mean dist (pilot / 90) | Gap (pilot / 90) | Cohen's d (90) |
|---|---|---|---|---|---|
| `word_repetition` | 12/22 → 30/63 | 0.5941 → 0.5892 | 0.4988 → 0.5039 | +0.0954 → +0.0853 | **+1.047** |
| `sound_repetition` | 15/0 → 37/0 | 0.5883 → 0.6047 | n/a (0 FP) | n/a | n/a |
| `filler` | 0/4 → 0/8 | n/a (0 TP) | 0.4736 → 0.4795 | n/a | n/a |
| **Any** | 28/25 → 70/68 | 0.5860 → 0.5933 | 0.4965 → 0.5014 | +0.0895 → +0.0919 | **+1.116** |

Raw output: `eval_datasets/_stage1_encoder_pilot_output.txt` (30 clips),
`eval_datasets/_stage1_encoder_90clip_output.txt` (90 clips). Saved
results: `eval_results/20260804T140128595787Z_libristutter_stage1-
encoder-signal.json` (30), `eval_results/20260804T151450108987Z_
libristutter_stage1-encoder-signal.json` (90).

**This clears §11.4.1's revised success criterion decisively, not
marginally**: `word_repetition` d=+1.047, `Any` d=+1.116 — both well past
the "medium" bar (0.5) and conventionally "large" effects (Cohen's own
benchmark: 0.8+), on samples large enough (n=63-70 per group) to have
real statistical power at this effect size, not a fragile small-n read.
**The result is stable, not a fluke**: both the gap's sign and rough
magnitude held from 30 clips to 90 (word_repetition: +0.0954 → +0.0853;
Any: +0.0895 → +0.0919) — if anything, `Any`'s gap and effect size grew
slightly with more data, the opposite of what a statistical-noise
artifact regressing toward zero would look like. `sound_repetition`/
`filler` remain uninformative at this scale (0 FP / 0 TP respectively),
the same already-known sampling gap as every other measurement on this
499-clip sample (`ROADMAP.md` item 14) — not evidence against the signal
for those types, just no evidence either way yet.

**A real, unconfirmed limitation, stated honestly rather than glossed
over**: each `word_repetition`/`sound_repetition`/`filler` event is
attributed to a single token (`detect.py`'s events carry one `index`),
so both TP and FP spans being compared are single-word pooled embeddings,
not systematically different-length spans — this partially mitigates,
but does not rule out, a duration- or word-identity-driven confound
(e.g., if certain words are both more likely to be genuinely repeated
*and* happen to pool differently for reasons unrelated to disfluency).
Not investigated further in Stage 1's scope; worth a direct check
(e.g., controlling for word length/identity) if this moves to Stage 2.

**Conclusion, answering this project's central Phase 3 question
directly**: CrisperWhisper's own last-layer encoder representation
carries information the ASR transcript alone does not — the transcript
gives no signal at all to distinguish a genuine `word_repetition` from a
coincidental one (both produce identical tokens), while the encoder
embedding separates them with a large, stable effect size. This is real
evidence the existing architecture can be strengthened without a second
heavyweight model, precisely the outcome `PHASE_3_ARCHITECTURE_REVIEW.md`
§5.1 named as what would make encoder-reuse the right call over Stage 1b
(a frozen WavLM-Large pass) or a wholesale architecture change.

**What this result does *not* yet establish**: whether this signal
*generalizes* beyond LibriStutter's reconstructed-timing clips and this
detector's current FP mix (§11.6's own duration-confound caveat above);
whether a trained classifier (Stage 2) meaningfully improves on this
simple distance measure, or whether the raw distance is usable as a
corroboration signal directly (e.g., a per-type distance threshold,
zero-training, matching how VAD/Praat work today) without training
anything at all — an option this section's own framing did not originally
consider and is worth weighing explicitly before committing to Stage 2's
added complexity. **Per §11.5 and standing rule 4, this result alone does
not authorize implementing anything — it is reported here as the
measurement Stage 1 was pre-registered to produce, with the response to
it left as a separate, explicit decision.**

## 12. Corroboration-mechanism comparison — pre-registered protocol (2026-08-04, before implementation)

`PHASE_3_ARCHITECTURE_REVIEW.md` §9 reviewed, at the project owner's
explicit request, how Stage 1's confirmed signal should actually be
turned into a detection decision — without assuming a threshold, a
classifier, or the current architecture is automatically right. That
review separated two axes (signal computation, decision mechanism) and
concluded multiple combinations remain genuinely plausible, to be decided
by measurement, not argument. This section fixes the exact comparison
methodology before any of them is implemented into `detect_disfluencies()`.

### 12.1 Candidates under comparison

Per `PHASE_3_ARCHITECTURE_REVIEW.md` §9.2/§9.5, three combinations,
evaluated on the same data:

- **(S1, M1)**: distance-to-fluent-centroid (Stage 1's signal, §11.2) +
  a fixed global threshold, calibrated by a documented procedure (§12.3).
- **(S1, M3)**: the same signal + a small trained logistic-regression
  classifier over the raw embedding (not just the scalar distance) —
  measures the classifier's actual marginal gain over (S1, M1), not an
  assumed one.
- **(S2, M1)**: a new signal, not previously evaluated — repeat-pair
  self-similarity (cosine distance between a `word_repetition`/
  `sound_repetition` event's own embedding and its *partner* token's
  embedding, i.e. `tokens[event["index"] - 1]`, confirmed as the
  detector's own pairing convention by reading `detect.py` directly) +
  the same fixed-threshold mechanism as (S1, M1), so the two signals are
  compared under an identical, cheap decision mechanism before either is
  paired with a classifier. **S2 does not apply to `filler`** (no second
  instance to compare against) — scoped to `word_repetition`/
  `sound_repetition` only.

**(S2, M3) and (M2, any signal) are deliberately not run this pass** —
scoped down from the full combinatorial space to the three comparisons
that most directly answer the open questions §9.5 identified (does a
classifier beat a threshold on the known-good signal; is the untested
signal competitive with the known-good one under the cheapest
mechanism), not an exhaustive sweep. Revisit if either candidate cleared
here itself does not clear a later bar.

### 12.2 Data collection

Extends the existing Stage 1 runner rather than re-running the encoder a
third time from scratch for a different measurement: for every
`word_repetition`/`sound_repetition`/`filler` event (TP or FP) in the
same 90-clip real-audio sample already used for §11.6, persist — not
just the aggregate mean Stage 1 already saved, but the raw per-event
data needed to compute *any* signal/mechanism combination post-hoc
without another encoder pass:

- clip id, event index, type, TP/FP label (already computed by the
  existing detector + ground truth comparison)
- the event's own mean-pooled last-layer embedding (full vector, not
  just its distance to anything)
- the clip's fluent-centroid embedding (for S1)
- the partner token's mean-pooled embedding, when `index - 1 >= 0`
  (for S2; `None` for `filler`, which has no partner)

Saved as a new artifact (`.npz`, not the existing JSON result format,
which isn't suited to arrays of embedding vectors) — `eval_results/
stage1_raw_embeddings_<timestamp>.npz`, documented as a new, distinct
artifact type from `save_run()`'s existing per-run JSON files.

### 12.3 Cross-validation protocol

**5-fold, split by clip, not by event** — a clip's own fluent centroid
and its events are correlated (`fluent_centroid` is computed once per
clip and reused by every event in it), so splitting by event would leak
clip-level information across train/test, the same leakage class
SEP-28k-E's own paper was designed to prevent (`VALIDATION.md` §4 point
5). Folds assigned deterministically (clips in their existing sorted
order, round-robin into 5 folds) — no random seed dependency for the
split itself.

For each of the three candidates in §12.1, for each fold:

- **Train** on the other 4 folds' events, **evaluate** on the held-out
  fold's events — never the reverse.
- **M1 (threshold)**: search the training fold's own signal-value range
  for the threshold maximizing F1 on the training data; apply that fixed
  threshold to the held-out fold.
- **M3 (classifier)**: fit a logistic regression on the training folds'
  `(embedding, TP/FP label)` pairs. Implemented directly in this
  project's own code with `numpy` (gradient descent on the standard
  logistic loss) rather than adding `scikit-learn` as a new declared
  dependency (confirmed not currently in `requirements.txt`, only
  present transitively in this dev environment) — consistent with this
  project's existing minimal, explicitly-justified dependency list, and
  logistic regression is simple enough not to need a library for it.
- Report **precision, recall, F1 on the held-out fold only**, then the
  **mean across all 5 folds** (plus the fold-to-fold range, in place of
  a formal CI given only 5 points) as the headline comparison number.

**Scope note, stated in advance**: `sound_repetition` has had zero false
positives across every Stage 1 run so far (0/15 at 30 clips, 0/37 at 90)
— its own precision/F1 cannot be cross-validated with this data (no
negative class to hold out). Reported as `n/a`, not silently omitted;
the primary comparison is `word_repetition` and the combined `Any`
(pooling `word_repetition` + `sound_repetition`, matching §11's own
convention). `filler` remains uninformative (0 TP throughout) for the
same already-known sampling gap (`ROADMAP.md` item 14) and is excluded
from this comparison's headline numbers for the same reason it was
excluded from §11.6's.

### 12.4 Success criteria, fixed in advance

- **No single candidate is assumed to win.** All three are reported side
  by side; a result showing them statistically indistinguishable within
  fold-to-fold variance is a valid, expected, reportable outcome — per
  `PHASE_3_ARCHITECTURE_REVIEW.md` §9.3's own reasoning, a large Cohen's
  d (as Stage 1 found) is specifically the regime where a threshold is
  expected to already capture most of the available separation, so
  "the classifier doesn't meaningfully beat the threshold" is a
  plausible, well-motivated result, not a surprising one to explain away.
- **If (S1, M3) beats (S1, M1) by less than one fold's worth of F1
  variance**, the practical recommendation is (S1, M1) regardless — per
  §9.3's maintainability/reproducibility/engineering-complexity analysis,
  a classifier's added cost needs a real, not marginal, performance
  justification to be worth taking on as this project's first shipped
  trained-model artifact.
- **If (S2, M1) is competitive with or better than (S1, M1)**, that is
  evidence the repeat-pair-self-similarity signal deserves its own
  follow-up (potentially paired with M3 too, not run this pass) rather
  than being a one-off exploratory test with no further action either way.
- **This comparison alone does not authorize implementing any candidate
  into `detect_disfluencies()`** — per standing rule 4 and the same
  discipline §11.5 already established for Stage 1's own result, that
  remains a separate, explicit decision once this comparison's numbers
  are in.

### 12.5 Results (2026-08-04): the classifier wins by a real margin — the opposite of what §12.4 predicted, reported as measured

138 scorable events collected across the same 90-clip sample (30 TP/63 FP
`word_repetition`, 37 TP/0 FP `sound_repetition`, 0 TP/8 FP `filler` —
matching §11.6 exactly, confirming the raw-data collection reproduces
Stage 1's own counts). Raw data: `eval_results/
stage1_raw_embeddings_90clip.npz`. 5-fold, clip-split cross-validation:

| Type | Candidate | F1 (mean, fold range) | Precision | Recall |
|---|---|---|---|---|
| `word_repetition` (n=30 TP/63 FP) | (S1, M1) threshold | 0.588 (0.429-0.714) | 0.521 | 0.692 |
| | **(S1, M3) classifier** | **0.749 (0.667-0.833)** | 0.750 | 0.761 |
| | (S2, M1) self-similarity | 0.546 (0.429-0.667) | 0.446 | 0.737 |
| `Any` (word+sound repetition, n=67 TP/63 FP) | (S1, M1) threshold | 0.755 (0.667-0.812) | 0.694 | 0.834 |
| | **(S1, M3) classifier** | **0.888 (0.815-0.933)** | 0.889 | 0.901 |
| | (S2, M1) self-similarity | 0.678 (0.615-0.774) | 0.583 | 0.847 |

**This directly contradicts §12.4's own pre-registered prediction, and is
reported as exactly that — a real finding, not explained away.** §12.4
reasoned that Stage 1's large Cohen's d (>1.0) meant a threshold should
already capture most of the available separation, making a large
classifier margin unlikely. The measured margin is large and consistent
in the *opposite* direction: (S1, M3) beats (S1, M1) by +0.161 F1
(`word_repetition`) and +0.133 F1 (`Any`) — in both cases the
classifier's *mean* F1 exceeds the threshold's own *best fold*, not just
its average, which is a materially larger and more consistent gap than
"one fold's worth of variance," the bar §12.4 set for treating the
difference as real rather than noise.

**Why this makes sense on reflection, stated honestly as a post-hoc
explanation, not a prediction that was actually made in advance**:
distance-to-fluent-centroid (S1) is an *unsupervised* heuristic — it
never looks at TP/FP labels when computing the signal. Logistic
regression is fit directly to those labels. §12.4's Cohen's d reasoning
assumed the centroid-distance projection was close to the *best possible*
linear separator in this embedding space; the result suggests it isn't
— there is a more discriminative linear direction in the 1280-dimensional
space than "distance from the fluent-token average," and a supervised fit
finds it. This is a real, useful correction to this project's own
reasoning about Stage 1's result, surfaced by actually running the
comparison rather than trusting the plausible-sounding argument.

**(S2, M1) — the repeat-pair self-similarity signal — did not
outperform (S1, M1)** (0.546 vs. 0.588 for `word_repetition`; 0.678 vs.
0.755 for `Any`), slightly underperforming in both slices. Per §12.4's
own pre-registered criterion, this does *not* clear the bar for further
investment in this specific signal right now (e.g. pairing it with M3)
— a negative-ish result for a candidate this review specifically
proposed, recorded as such rather than quietly dropped.

**Real, stated limitations of this specific result, not to be glossed
over by the strong headline numbers**:

- **Sample size remains modest** for a 5-fold split (n=93/130 events
  with a partner, spread across 5 folds — roughly 18-26 test events per
  fold). The classifier's advantage is consistent across folds (every
  fold's F1 for M3 exceeded M1's corresponding fold in both type slices
  — checked directly, not just the means), which is reassuring, but a
  larger sample would tighten confidence in the exact margin.
- **The L2 regularization strength (5.0) was fixed, not tuned**, per
  §12.3's own stated scope limitation — a different value could move
  this result in either direction; robustness to this choice was not
  tested this pass.
- **This is still LibriStutter's reconstructed-timing data** — the same
  standing caveat attached to every result on this dataset since §8.2.
  Whether the classifier's advantage over a threshold transfers to real
  (non-reconstructed) speech is untested.
- `sound_repetition` remains entirely unvalidated by this comparison (0
  FP throughout, `n/a` — confirmed identically to §11.6, not a new gap).

**Conclusion, stated at the same scope as §11.6's — a measurement, not
an implementation decision**: on this data, a small logistic-regression
classifier over CrisperWhisper's own encoder embedding materially
outperforms a zero-training threshold on the same signal, and the
untested repeat-pair-similarity signal does not show an advantage worth
pursuing further right now. This is real evidence in favor of (S1, M3)
specifically — but per §12.4 and standing rule 4, the decision to accept
the real, categorical costs §9.3 named (this project's first shipped
trained-model artifact, its maintenance/reproducibility/interpretability
burden) in exchange for this measured performance gain is a separate,
explicit decision for the project owner, not one this measurement makes
on its own.

### 12.6 Pre-registered follow-up validation (2026-08-04, before running it) — what would resolve the remaining uncertainty

Per `CLAUDE.md` standing rule 8 (architectural decisions are evidence-
constrained: decide confidently once evidence supports it, or name the
exact uncertainty and pre-register what resolves it) and
`PAPER_DECISION_LOG.md`'s 2026-08-04 "Standing principle established"
entry: §12.5's result is real and positive, but not yet decisive, for
three named reasons (small-sample classifier variance, untuned
regularization, reconstructed-timing overfitting risk — full reasoning
in that log entry, not duplicated here). This section fixes the exact
next validation before it is run, not after.

**What gets changed from §12's original protocol**:

1. **Scale**: re-run data collection at a materially larger sample than
   90 clips — target as much of the full 499-clip sample as practical
   (cost permitting; §11.3.1 already measured ~40-50s/clip for the
   encoder pass alone, so the full sample is a real, priced, multi-hour
   cost, not a casual re-run — see that section before committing to it).
2. **Regularization**: chosen by nested cross-validation (an inner CV
   loop within each of the outer 5 folds' training data) rather than
   `compare_corroboration_mechanisms.py`'s current fixed `L2_STRENGTH =
   5.0` — directly answers whether §12.5's result is an artifact of that
   specific, arbitrary choice.
3. **Everything else unchanged**: same clip-split 5-fold outer protocol,
   same three candidates, same metrics.

**Decision rule, fixed before running** (so the response to the result
isn't decided after seeing it): **if the classifier's F1 advantage over
the threshold holds** (does not shrink to within §12.4's original
"one fold's worth of variance" bar) **at the larger scale with tuned
regularization, that constitutes sufficient evidence to adopt (S1, M3)
as the new default corroboration mechanism for `word_repetition`/
`sound_repetition`**, and implementation into `detect_disfluencies()`
(plus the version-artifact/retraining-process work that would newly
require, per `PHASE_3_ARCHITECTURE_REVIEW.md` §9.3) becomes its own
pre-registered next step. **If the advantage shrinks or reverses**, that
is equally valid, reportable evidence — either that §12.5's result was
partly a small-sample/untuned-regularization artifact, or that it does
not generalize past 90 clips — and the project reverts to treating the
threshold (or no new mechanism at all for these types) as the working
default, with the reasoning recorded exactly as thoroughly as a positive
result would be.

**Not run as part of this session** — this is a real, multi-hour
commitment (§11.3.1's cost finding applies again, likely worse at a
larger scale) and is recorded here specifically so it can be picked up
as its own, already-justified next step rather than re-derived from
scratch.

### 12.6.1 Scoping addendum (2026-08-04, before running) — a real, previously-unnoticed cost escalation found, and a bounded target chosen because of it

Before launching the larger run, re-examined the 90-clip run's own raw
per-clip timings (`eval_datasets/_collect_raw_90clip_output.txt`) rather
than assuming the ~44.6s/clip average from §11.3.1/§12.2 would hold flat
at a larger scale. **Found a real, reproducible slowdown across the
single run, not noise**: the first 10 clips averaged ~31s/clip; the last
10 (of the same 90-clip, ~68-minute run) averaged ~85s/clip — roughly
2.7x slower by the end than the start.

**Most likely explanation, stated with appropriate uncertainty (not
confirmed by further instrumentation, which was judged not worth the
time against the actual decision at hand)**: thermal throttling. This
runs on a laptop CPU under sustained, uninterrupted heavy load for over
an hour — exactly the condition laptop CPUs are known to throttle under,
more aggressively than desktop/server hardware. Memory growth in the
long-running `torch`/`transformers` process is a secondary candidate but
judged less likely given the encoder pass shape doesn't change across
clips (same fixed 30s-window input every time).

**Why this changes the scoping decision, not just the runtime estimate**:
§12.6 named "target as much of the full 499-clip sample as practical,"
with practicality bounded by cost. A naive extrapolation from the flat
90-clip average (499 x 44.6s = ~6.2h) already undersold the real cost;
if the observed slowdown continues rather than plateauing, total runtime
for the full sample is genuinely uncertain and could run considerably
longer, with no strong guarantee it stays a well-behaved, predictable
process for that long unattended. **Decision: target 250 clips** (a
~2.8x increase over the 90-clip sample already collected, materially
larger, not just larger) **rather than the unbounded full 499** — bounds
the commitment to a run whose duration can be reasoned about (roughly
5 hours under a plausible thermal-plateau assumption: the observed
90-clip run's shape, held flat past clip 90 rather than extrapolated
linearly forever), while still directly answering §12.6's actual
question (does the classifier's advantage hold at meaningfully more
data). **The full 499 remains available as a further step** if 250
clips' result is still not decisive, now with this cost characteristic
already known rather than discovered mid-run a second time.

### 12.6.2 Results (2026-08-05): the classifier's advantage holds — and grew — at 2.8x the sample, with tuned regularization. Decision rule satisfied.

250 clips, 402 scorable events collected (89 TP/205 FP `word_repetition`,
90 TP/4 FP `sound_repetition`, 0 TP/8 FP `filler` — note `sound_repetition`
finally has real FPs at this scale, unlike every prior run; still not
enough for a separate cross-validated slice, folded into `Any` as
before). Encoder pass averaged 33.7s/clip across the full run — *lower*
than the 90-clip run's 44.6s average, consistent with §12.6.1's
thermal-carryover hypothesis (this run started from a cooler baseline,
not immediately after another hour-long encoder-bound run). Raw data:
`eval_results/stage1_raw_embeddings_250clip.npz`. Full nested-CV output:
`eval_datasets/_compare_250clip_output.txt`.

| Type | Candidate | F1 (mean, fold range) | Precision | Recall |
|---|---|---|---|---|
| `word_repetition` (n=89 TP/205 FP) | (S1, M1) threshold | 0.543 (0.455-0.627) | 0.438 | 0.721 |
| | **(S1, M3) classifier, nested-CV L2** | **0.766 (0.706-0.829)** | 0.784 | 0.762 |
| | (S2, M1) self-similarity | 0.448 (0.375-0.526) | 0.390 | 0.562 |
| `Any` (word+sound repetition, n=179 TP/209 FP) | (S1, M1) threshold | 0.714 (0.697-0.729) | 0.620 | 0.846 |
| | **(S1, M3) classifier, nested-CV L2** | **0.892 (0.870-0.919)** | 0.885 | 0.900 |
| | (S2, M1) self-similarity | 0.617 (0.552-0.667) | 0.552 | 0.727 |

Selected L2 per outer fold (`word_repetition`): `[1.0, 2.0, 20.0, 1.0,
0.5]`; (`Any`): `[0.5, 1.0, 10.0, 20.0, 5.0]` — genuinely variable, not
converging on the old fixed value of 5.0 in every fold, confirming the
regularization strength was a real free parameter worth tuning, not an
arbitrary choice that happened not to matter.

**The margin held, and grew, relative to §12.5's 90-clip result**:
`word_repetition` gap +0.161 -> **+0.223**; `Any` gap +0.133 -> **+0.178**.
**Audited before accepting** (a result this clean deserves the same
scrutiny as before): verified directly, not from the means alone, that
the classifier beat the threshold in **5 of 5 folds in both type
slices**, and that **the ranges do not even overlap** — the
classifier's *worst* fold (`word_repetition` 0.706, `Any` 0.870) still
exceeds the threshold's *best* fold (0.627, 0.729 respectively) in both
cases. This is a materially stronger, more decisive result than §12.5's,
not merely a replication of it.

**§12.6's pre-registered decision rule is satisfied, mechanically, not
by judgment call**: "if the classifier's F1 advantage over the threshold
holds... at the larger scale with tuned regularization, that constitutes
sufficient evidence to adopt (S1, M3) as the new default corroboration
mechanism for `word_repetition`/`sound_repetition`." The advantage did
not shrink toward §12.4's "one fold's worth of variance" bar — it grew,
and the two mechanisms' fold ranges are now fully separated. **Decision:
(S1, M3) is adopted.** Per the same section, implementation into
`detect_disfluencies()` (plus the version-artifact/retraining-process
work `PHASE_3_ARCHITECTURE_REVIEW.md` §9.3 already named as a real,
accepted cost) is the next step — see §13 for the implementation and
the integrated-detector benchmark.

**What remains genuinely untested, stated plainly even though the
decision is made**: this is still LibriStutter's reconstructed-timing
data (the one of §12.6's three named uncertainties this specific
validation could not address by design — scale and regularization were
the two it targeted). Whether this advantage transfers to real,
non-reconstructed speech remains open, exactly as every other result on
this dataset has been since §8.2 — recorded as a standing limitation of
the shipped mechanism, not resolved by this decision.

## 13. Implementation and integrated-detector benchmark (2026-08-05)

`profiling/repetition_classifier.py` applies the trained (S1, M3)
classifier (`models/repetition_corroboration_classifier.npz`, trained by
`profiling/evaluation/train_repetition_classifier.py` on all 250 clips'
`word_repetition`/`sound_repetition` events) as a hard gate on
candidate events in `detect.py`, the same architectural role Praat-gating
plays for `prolongation`. New config key: `require_repetition_
classifier_confirmation`, **default `true`** — this is now the shipped
default, following the same "flip the default once the ablation
justifies it" precedent as the prolongation redesign (§9.5.1). Graceful,
multi-layer no-op (never blocks) when `transformers`/`torch` or the model
artifact are unavailable, or no audio is given — matches every other
optional acoustic component in this codebase.

**A real engineering bug was found and fixed during integration, before
any benchmark was trusted**: the gate's decision was initially computed
unconditionally for every adjacent token pair (`i > 0`), not only when a
repetition candidate actually existed — meaning it would have attempted
the real, ~30-90s encoder load for *every* clip touching this code path,
regardless of whether a `word_repetition`/`sound_repetition` candidate
was present. Caught immediately by the existing fast unit-test suite
hanging (a token pair like "go"/"now", no repetition at all, triggering a
real model-load attempt) — fixed by deferring the gate's evaluation to
only the branches where a candidate was already found, confirmed by
re-running the full suite (fast again, 66/66). A second, smaller bug was
also caught the same way: `confirms_repetition()` originally returned a
numpy `bool_`, not a Python `bool` — harmless for the `if`/`and` logic
that actually uses it, but a latent risk for anything downstream doing an
`is True`/`is False` check or JSON-serializing an event dict (`json.dumps`
fails on numpy scalar types by default). Both fixed before any real
benchmark ran, not after.

**Verified directly on real audio, not just via cross-validated numbers**:
a genuine ground-truth `word_repetition` (clip `103-1240-0000`, index 4)
still fires with the gate on (63.5s, confirming the real encoder pass
runs) and is unaffected with the gate off (0.7s); a genuine false-positive
`word_repetition` (clip `103-1240-0018`, index 13, not matching ground
truth) is correctly suppressed with the gate on and fires with it off,
with every other event in that clip unaffected. Direct, small-scale
confirmation that the integration behaves as designed before trusting the
larger, statistical benchmark below.

### 13.1 Integrated detector benchmark — honest, cross-validated, no new encoder run

Naively applying the final shipped model (trained on all 250 clips) back
to that same data would give an optimistic, in-sample result, not what a
real user would see. Instead, `profiling/evaluation/
benchmark_integrated_gate.py` reconstructs each event's *out-of-fold*
prediction from the same 5-fold, clip-split outer CV split §12.6 already
used — every prediction comes from a fold that never trained on that
event — summed into a real, honest confusion matrix. No new encoder run
needed; this is pure analysis over `stage1_raw_embeddings_250clip.npz`.

| Type | Gate OFF TP/FP | Gate ON TP/FP/FN(new) | Precision (off→on) | Recall (off→on) | F1 (off→on) |
|---|---|---|---|---|---|
| `word_repetition` | 89/205 | 74/22/15 | 0.303 -> **0.771** | 1.000 -> 0.831 | 0.465 -> **0.800** |
| `sound_repetition` | 90/4 | 87/0/3 | 0.957 -> 1.000 | 1.000 -> 0.967 | 0.978 -> 0.983 |
| **Any (both types)** | **179/209** | **161/22/18** | **0.461 -> 0.880** | 1.000 -> 0.899 | **0.631 -> 0.890** |

Full output: `eval_datasets/_benchmark_integrated_gate_output.txt`.

**A large, real, decisive improvement — the same shape as every other
audio-native corroboration change this project has shipped** (Praat-
gating for `prolongation`, §9.5.1; the original audio-native
restructuring, §8.3): a large precision gain at a real, non-trivial
recall cost. `word_repetition`'s false positives drop from 205 to 22 (an
89% reduction) at the cost of 15 of 89 true positives (a 17% recall
loss); combined, `Any`'s F1 improves 0.631 -> 0.890, +0.259 — a larger
single-change F1 improvement than any other individual change this
project has measured and shipped this phase.

**Honest framing of what "Gate OFF" recall = 1.000 means here**: by
construction, every event in this dataset came from the gate-off
detector's own candidate list (§12.2), so gate-off recall is trivially
1.0 on this specific measure — this is not claiming the gate-off detector
has perfect real-world recall for these types (it doesn't; the token-path
exact-match check itself has known misses, e.g. ASR-corrupted repeats,
documented in §8.4.4). This table isolates specifically the gate's own
effect on the candidates the existing detector already produces, not a
full recall analysis against all ground truth.

### 13.2 Known limitation: live-app latency, not resolved by this decision

Stated plainly, not glossed over: enabling this gate in the live app adds
a real, second CrisperWhisper encoder pass (~30-90s, `profiling/
encoder_embedding.py`) on top of the existing ASR transcription cost,
because `profiling/asr.py` calls CrisperWhisper through `transformers.
pipeline()`, which never exposes encoder hidden states — this module
bypasses that wrapper with its own, separate call. `PHASE_3_
ARCHITECTURE_REVIEW.md` §5.1 originally assumed this would be "zero added
latency" by reusing the same forward pass already made for transcription;
building the real integration revealed that assumption doesn't hold
as-is with this project's current ASR call structure. The lazy design
(§13's `RepetitionClassifierContext`) means this cost is only paid on
clips that actually contain a `word_repetition`/`sound_repetition`
candidate, not every clip — but for those clips, the cost is real and
user-facing. **Follow-on engineering work, not attempted here**:
restructure `asr.py`'s core transcription call to capture encoder hidden
states during the same forward pass it already makes (would require
touching a delicate, multiply-patched call with several documented bug
workarounds — a real, separately-scoped risk, not undertaken in the same
session as the classifier's own validation). See `ARCHITECTURE.md`'s
known-limitations section and `ROADMAP.md` for this as a named follow-up.

---

## 14. Track B validation of the shipped repetition-classifier gate (2026-08-05) — pre-registered protocol, before running

**Why this exists.** §11-§13 validated Stage 1, the corroboration-
mechanism comparison, and the integrated-detector benchmark entirely on
Track A — `detect_disfluencies(clip.tokens, ...)` called with
LibriStutter's own ground-truth tokens, confirmed by direct inspection of
`profiling/evaluation/track_a.py`'s `evaluate()`. The gate has never been
run against real CrisperWhisper transcription output. Separately, Track B
itself (the harness that does run real ASR) was last executed as the
120-clip speaker-stratified sample on 2026-08-04, before the gate existed
in any form — so no Track B number to date reflects this decision either
way. `ROADMAP.md`'s "First-principles reassessment" section (2026-08-05)
identified this as the project's single biggest open bottleneck; this
section is the pre-registered protocol for closing it, written before any
of the two runs below have executed, per standing rule 1.

**Dataset.** The existing 120-clip speaker-stratified LibriStutter sample
(`VALIDATION.md` §8.4.3's dataset — 40/40 speakers, round-robin order).
All 120 clips' real CrisperWhisper transcription output is already cached
(`eval_datasets/_track_b_cache/`, confirmed present for all 120 clip names
before this section was written) from the 2026-08-04 run
(`eval_results/20260804T100451177010Z_libristutter_B.json`, 120/120 cache
hits, git commit `023dc95`). **No new ASR inference is required for this
comparison** — `track_b.py`'s `events` are always recomputed fresh from
cached `hyp_tokens` regardless of config (see `_save_cache`'s docstring),
so re-running `track_b.py` against this cache with the gate toggled
isolates exactly one variable: the classifier gate, with the ASR output,
the dataset, and every other line of `detect.py` held fixed. The
classifier itself *will* run for real on this audio for any clip whose
real ASR output happens to produce a `word_repetition`/`sound_repetition`
candidate — that real ~30-90s-per-affected-clip encoder cost is part of
what this run measures, not something bypassed by the cache.

**Conditions.** Two full `track_b.py` runs over the identical 120 clips,
identical cache, differing only in `config.yaml`'s
`require_repetition_classifier_confirmation`:
- **Gate OFF** (`false`) — the pre-classifier detector, i.e. what Track B
  measured before this decision existed.
- **Gate ON** (`true`) — today's shipped default.

**Metrics.** Per `track_b.py`'s standard output, for `word_repetition`,
`sound_repetition`, and the combined `Any` label: precision, recall, F1,
and raw TP/FP/FN counts (counts matter as much as the derived rates given
the sample size — see the small-n caveat below), on all three of Track
B's standard slices (overall / preserved / preserved_ctx1), plus mean WER
and the FP-attribution/alignment-op diagnostics `track_b.py` already
prints. `Any` F1 (overall slice) is the single primary metric this
decision turns on, matching how the Track A integrated-gate benchmark
(§13.1) was itself summarized.

**Success criteria, fixed in advance.**
- **Transfers**: gate ON's `Any` F1 (overall slice) is higher than gate
  OFF's, and the improvement is not solely an artifact of near-zero event
  volume in one condition (i.e. both TP+FP+FN counts are large enough to
  read the rates as meaningful, not a 2-vs-1-event comparison). If this
  holds, the gate-on configuration is confirmed as the real-world
  baseline going forward and no further action is needed before resuming
  Phase 3 work.
- **Does not transfer**: gate ON's `Any` F1 (overall slice) is flat or
  lower than gate OFF's. If this holds, the next step is diagnostic, not
  a reversion: compare `word_repetition`/`sound_repetition` **candidate
  volume** (TP+FN at the token-path level, before the gate's own
  precision/recall split) between Track A's ground-truth tokens and Track
  B's real ASR tokens on the same clips, to test this project's own
  standing hypothesis (`ROADMAP.md`'s reassessment, §2) that the
  classifier only ever refines precision among *surviving* candidates and
  cannot recover candidates ASR already destroyed upstream — i.e.
  determine whether the gap is because the gate hurts real candidates it
  shouldn't (a detector/classifier problem, fixable) or because there
  simply aren't enough real candidates left for the gate to help with (an
  ASR-fidelity problem, per Phase 1's own dominant finding).
- **Small-sample honesty, stated in advance**: 120 clips is Track B's
  existing largest speaker-diverse sample, but `word_repetition`/
  `sound_repetition` events within it are a fraction of that — the exact
  count is not yet known at the time this protocol is written. If the
  resulting TP+FP+FN counts are small (single digits to low tens), the
  result will be reported as directionally informative, not as a
  precision-confirmed number, per standing rule 3 — this is stated now,
  before the counts are known, specifically so the result can't be
  quietly upgraded to "confirmed" after the fact if it turns out to look
  favorable.

**What this does not decide.** A "transfers" result does not itself
validate the classifier's training distribution as representative of all
real-world speech (LibriStutter remains synthetic-injection, single
dataset, single ASR backend — `ROADMAP.md` item 10's open question). It
answers the narrower, prerequisite question this session is scoped to:
does the gate's Track-A-measured benefit survive contact with this
project's own real ASR pipeline at all, on the data already in hand.

### 14.1 Results (2026-08-05): the gate's mechanism transfers safely, but its
real-world impact is currently negligible — and the reason why is now
directly measured, not just hypothesized

Both runs executed against the identical, already-cached 120-clip
speaker-stratified sample (no new ASR inference — confirmed
`120/120 from cache` in both run logs), differing only in
`require_repetition_classifier_confirmation`. Overall slice (the real
end-user-facing metric):

| Type | Gate OFF TP/FP/FN | Gate ON TP/FP/FN | Precision (off->on) | Recall (off->on) | F1 (off->on) |
|---|---|---|---|---|---|
| `word_repetition` | 1/19/41 | 1/12/41 | 0.050 -> 0.077 | 0.024 -> 0.024 | 0.032 -> 0.036 |
| `sound_repetition` | 0/0/42 | 0/0/42 | n/a -> n/a | 0.000 -> 0.000 | n/a -> n/a |
| **Any (all 5 types)** | **11/70/175** | **11/63/175** | **0.136 -> 0.149** | 0.059 -> 0.059 | **0.082 -> 0.085** |

Full logs: `eval_datasets/_gate_off_run_output.txt`,
`eval_datasets/_gate_on_run_output.txt`. Saved runs:
`eval_results/20260805T125451692170Z_libristutter_B.json` (gate off),
`eval_results/20260805T130631020774Z_libristutter_B.json` (gate on).

**Against the pre-registered criteria (§14): technically "transfers"
(`Any` F1 0.082 -> 0.085, both conditions have non-trivial aggregate
counts), but reporting it at that level alone would be misleading given
the standing "don't quietly upgrade a favorable-looking small number"
caveat this section pre-registered.** The honest, complete picture has
three separate parts:

**1. The gate's core mechanism is validated as sound on real ASR text.**
Where a `word_repetition` candidate exists at all, the gate suppresses 7
of 19 false candidates (37% FP reduction) with the one true positive
preserved untouched (TP unchanged, 1 -> 1) — the same
precision-up/recall-flat shape as the Track A result (§13.1), just at a
much smaller scale. This is a real, positive, non-harmful result: the
classifier is not misfiring or over-suppressing on real, ASR-corrupted
input text, which was a genuine open question before this run.

**2. The practical impact on this sample is negligible, because there are
almost no candidates for the gate to act on in the first place —
confirming the reassessment's central hypothesis directly.** Comparing
raw candidate volume (TP+FP at gate-off, i.e. before any gating) against
Track A's own out-of-fold gate-off numbers (§13.1, 250 clips):

| Type | Track A candidates/clip (ground-truth tokens) | Track B candidates/clip (real ASR, this run) | Ratio |
|---|---|---|---|
| `word_repetition` | 294/250 = 1.176 | 20/120 = 0.167 | ~7x fewer |
| `sound_repetition` | 94/250 = 0.376 | 0/120 = 0.000 | complete collapse |

Ground-truth prevalence of both types in this 120-clip sample (42 each,
~0.35/clip) is the same order of magnitude as Track A's sample, so this
is not an artifact of the 120-clip sample simply containing fewer real
disfluencies of these types — the population exists; the detector's
candidate-generation step, operating on real ASR output, essentially
isn't finding it. A gate that only refines precision among survivors
cannot move a number this starved of survivors.

**3. Direct hand-check of the `sound_repetition` zero (standing rule 3 —
audit a dramatic result before trusting it), and a specific, previously-
undocumented mechanism found.** Re-ran gate-off with `--verbose`
(`eval_datasets/_gate_off_verbose_output.txt`) and inspected every
`true=sound_repetition` line. Representative cases:

```
ref[5]='considered-' (true=sound_repetition) -> align=correct, hyp_word='considered', detector_predicted={}
ref[5]='the-'        (true=sound_repetition) -> align=correct, hyp_word='the',        detector_predicted={}
ref[27]='I-'         (true=sound_repetition) -> align=correct, hyp_word='I',          detector_predicted={'block'}
```

LibriStutter's `sound_repetition` ground truth is stored as a
reconstructed reference token with a trailing hyphen (`load_libristutter_
csv`'s documented reconstruction approximation, `PAPER_DECISION_LOG.md`,
2026-08-03) — a dataset-representation convention for "an acoustic
sound-repetition fragment occurred here," not a literal transcribable
token. **Even where CrisperWhisper aligns `correct`** (i.e. transcribed
that word position accurately, by this project's own alignment
scoring), the real output it produces is the clean, full word
("considered," not a fragment like "c- considered"). `detect.py`'s
`sound_repetition` candidate check (the fragment-repeat check,
`is_fragment_repeat`) requires an actual sub-word fragment token to exist
adjacent to the real word in the ASR output — and a real, fluency-
normalizing verbatim-ASR system essentially never emits one, regardless
of whether it "got the word right" in the alignment sense. This is a
narrower and more specific claim than the general "ASR fidelity is the
bottleneck" framing already established (Phase 1): **it is not merely
that ASR sometimes mis-transcribes a sound-repetition; it is that the
current detection strategy's input assumption (a literal fragment token
appearing in the transcript) appears to almost never hold for real ASR
output at all**, independent of how accurate that ASR otherwise is. One
suggestive detail worth naming, not yet investigated further: in the
`'I-'` case above, the acoustic signal wasn't lost entirely — it surfaced
as a `block` prediction instead, from the acoustic-native detector,
hinting the underlying acoustic event may still be detectable, just
mis-routed by today's type taxonomy rather than fully invisible.

**Small-sample honesty, as pre-registered.** `word_repetition` has
exactly 1 true positive in this entire 120-clip sample, in both
conditions — nowhere near enough to trust the *precise* F1 numbers above
as stable estimates. `sound_repetition` has zero true positives in either
condition, so its "0.000 recall, both conditions" is not evidence the
gate doesn't work for that type — it's evidence the type's current
candidate generator essentially never gets invoked on real audio in this
sample at all, which is the more important and more specific finding
(point 3 above). The *qualitative* mechanism finding (fragment tokens
absent from real ASR output) does not depend on this sample size the way
the quantitative F1 numbers do — it is a structural property of the
detection strategy versus what verbatim ASR actually produces, visible in
every one of the handful of cases inspected.

**Verdict.** The classifier gate is confirmed safe and mildly beneficial
on real ASR output — no reason to disable it, and it stays the shipped
default. But it is not, on its own, a fix for this project's real-world
performance gap, because the deeper problem (Phase 3's item 17 was never
positioned to address) is candidate-generation loss under real ASR,
concentrated specifically and severely in `sound_repetition`. This
directly resolves the pre-registered "does not transfer -> investigate
why" branch's question: the interaction is an **ASR/detector candidate-
generation problem**, not a classifier-precision problem. See
`ROADMAP.md` for how this reprioritizes Phase 3.
