# VALIDATION.md — evaluation methodology, datasets, and results

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
