# VALIDATION.md — evaluation methodology, datasets, and results

**Status as of 2026-08-03: methodology researched and approved. Implementation
not yet started — the `profiling/evaluation/` package described in §6 does
not exist yet. This document is written ahead of that work so the plan is
recorded before it's built, per this project's documentation philosophy (see
`DOCS.md`).**

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

## 3. Two evaluation tracks — kept separate, never blended

This app is a two-stage pipeline (ASR → detector). A labeled dataset provides
its own ground-truth transcript + timestamps. Two different, both valuable,
evaluation modes follow from that:

- **Track A — Detector-only (ASR bypassed).** Feed the dataset's own
  ground-truth words/timestamps directly into `detect_disfluencies()`,
  skipping CrisperWhisper entirely. Answers: *given a perfect transcript, how
  good is the detection logic itself?* This is what `profiling/evaluate.py`
  (built 2026-08, see `PAPER_DECISION_LOG.md`) already does in
  timestamp-only mode — extending it to real datasets is largely wiring.
- **Track B — Full pipeline (ASR included).** Run CrisperWhisper on the
  dataset's raw audio ourselves, run the detector on *our own* ASR's output,
  then align our ASR's word sequence back to the dataset's labeled reference
  words. Answers: *what does a real user actually experience?* Requires a
  real alignment step (§5) — this is exactly what would have caught, with
  evidence instead of inference, that a real-mic test's missed word
  repetition (2026-08-03) was an ASR-fidelity gap rather than a detector bug.

Report both, always labeled which track produced which number.

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
  LibriStutter ships FLAC; Track A sidesteps this (timestamp-only); extending
  LibriStutter to Track B / acoustic-fusion evaluation needs FLAC decoding —
  a smaller, separate follow-on.
- **Track B's ASR↔reference alignment** is real work, not `zip()`: our ASR's
  word sequence differs in length/content from the reference whenever it
  mis-transcribes. Use a modified-cost sequence alignment (Levenshtein/DTW
  between hypothesis and reference words, biased so substitutions land on
  fluent words rather than the labeled disfluent ones) rather than a naive
  positional match — this is an established technique in the ASR/disfluency
  literature, not a bespoke invention.
- **Reproducibility**: every run recorded (config used, dataset version/date,
  git commit, count of successfully acquired clips for URL-based datasets) as
  a timestamped result file, not just printed — otherwise "F1 improved from X
  to Y" can't be trusted months later.

---

## 6. Proposed framework design

`profiling/evaluate.py`'s `LabeledClip` dataclass and `score_clips` scoring
core get reused, not discarded. Proposed restructure:

```
profiling/evaluation/
├── __init__.py
├── loaders.py     # load_libristutter_csv (moved), load_sep28k_labels, ...
├── metrics.py      # precision/recall/F1, per-type confusion matrix, IoU localization, EER
├── alignment.py    # Track B's ASR-hypothesis <-> reference alignment
├── track_a.py       # detector-only runner
├── track_b.py       # full-pipeline runner
└── report.py        # table + timestamped JSON/CSV result file (config, dataset version, git commit)
```

```
python -m profiling.evaluation.track_a --dataset libristutter --data-dir DIR
python -m profiling.evaluation.track_a --dataset sep28k --data-dir DIR
python -m profiling.evaluation.track_b --dataset sep28k --data-dir DIR
python -m profiling.evaluation.track_a --self-test
```

`profiling/evaluate.py` becomes a thin backward-compatible shim once the
package exists, not a silent breaking change.

**Sequencing** (not started — pending the baseline commit and a separate
go-ahead per `PAPER_DECISION_LOG.md`'s 2026-08-03 entries):
1. Package skeleton, migrate existing logic, no behavior change (verify via
   today's self-test under the new location).
2. Add IoU localization + per-type confusion matrices + the "Any" label to
   `metrics.py`.
3. Add `load_sep28k_labels` + document the download/extract steps.
4. Run Track A against real LibriStutter + SEP-28k — first real numbers.
5. Build `alignment.py` + `track_b.py`; hand-validate alignment on a handful
   of clips before trusting it.
6. Run Track B against the same datasets; compare against Track A on the
   same clips to separate ASR-attributable gaps from detector gaps.

---

## 7. Honest limitations

- No accessible dataset combination covers the full 7-type taxonomy —
  `stutter_marker` has no labeled equivalent anywhere reviewed.
- SEP-28k's audio depends on podcast URLs that can rot; record the actually-
  acquired fraction (count, date) as part of every run's reproducibility
  record, never silently treat it as "the full dataset."
- Human annotators disagree with each other on stuttering event boundaries
  even in these datasets — a perfect detector would not score 1.0. Report
  published inter-annotator agreement as a ceiling reference wherever
  available, not just our own F1 in isolation.
- Track B's alignment step is itself a source of methodology-introduced
  error — must be spot-checked by hand before its numbers are trusted.
- KSoF/UCLASS are lower priority due to access friction and (for KSoF)
  language mismatch, not because they're less valuable — revisit if
  cross-lingual support becomes an explicit goal.

---

## 8. Results

*No evaluation runs have been performed yet. This section is a template —
each future run should add or update a subsection below, not just print to a
terminal and discard the output.*

### 8.1 Run log

| Date | Dataset | Track | Git commit | Config snapshot | N clips scored | Notes |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | *no runs yet* |

### 8.2 LibriStutter — Track A

*Not yet run.*

| Type | TP | FP | FN | TN | Precision | Recall | F1 | Localization (IoU≥0.5 rate) |
|---|---|---|---|---|---|---|---|---|
| filler | — | — | — | — | — | — | — | — |
| sound_repetition | — | — | — | — | — | — | — | — |
| word_repetition | — | — | — | — | — | — | — | — |
| phrase_repetition | — | — | — | — | — | — | — | — |
| prolongation | — | — | — | — | — | — | — | — |
| Any (combined) | — | — | — | — | — | — | — | — |

### 8.3 SEP-28k — Track A

*Not yet run.*

(Same table shape as §8.2, plus `block`; no `phrase_repetition` row —
not in this dataset's taxonomy, see §2.)

### 8.4 SEP-28k — Track B (full pipeline)

*Not yet run. Requires `alignment.py` (§6, sequencing step 5) first.*

### 8.5 Cross-track comparison (Track A vs Track B, same clips)

*Not yet run. The point of this table: isolate how much of any Track B gap
vs. Track A is attributable to ASR transcription errors specifically,
rather than detector logic — directly following up on the 2026-08-03
real-mic test's missed-repetition finding.*

---

## 9. Ablations

*No ablations run yet. Candidates, once a baseline result from §8 exists to
ablate against:*

| Ablation | What it isolates | Status |
|---|---|---|
| Silero VAD corroboration on/off | Whether VAD gating actually improves prolongation precision/recall, vs. RMS/ZCR alone | Not yet run |
| Praat pitch/jitter/shimmer corroboration on/off | Whether voice-quality features improve prolongation accuracy over RMS/ZCR alone | Not yet run |
| `fusion_weights.acoustic` sweep (0.5 / 1.0 / 2.0 / 5.0) | Sensitivity of results to the acoustic-vs-token fusion weighting | Not yet run |
| Speaker calibration on/off | Whether personalized thresholds (`calibration.py`) measurably improve accuracy over the global floor | Not yet run |
| Block/prolongation threshold sweep | Whether `PAPER_DECISION_LOG.md`'s Part D threshold choices hold up against labeled data, not just anecdotal real-audio checks | Not yet run |

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
