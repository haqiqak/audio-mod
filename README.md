# Audio Profiler

A standalone Streamlit app that detects, classifies, and localizes stuttering
disfluencies from a speaker's own audio. Built for speakers who stutter — it
identifies repetitions, fillers, blocks, and prolongations the moment they
happen, builds a personalized profile of which sounds and word types are
most difficult for that speaker, and calibrates detection to that speaker's
own natural speaking tempo instead of a one-size-fits-all threshold.
Verbatim transcription (CrisperWhisper — captures exactly what was said,
including the repetitions and false starts standard ASR silently removes)
is scaffolding toward that detection goal, not the end product: the
transcript is one evidence source the detector uses, not the deliverable
or an assumed ground truth.

> This is one of several project docs — see [`DOCS.md`](DOCS.md) for the
> full map (architecture, decision history, evaluation methodology,
> roadmap). This file covers setup and usage only.

---

## What it does

### 1 — Transcription

Audio is transcribed verbatim using **CrisperWhisper** (`nyrahealth/CrisperWhisper`),
a fine-tuned version of whisper-large-v3 specifically trained to preserve
disfluencies rather than clean them up. Word-level timestamps are returned for
every token.

### 2 — Speaker tempo calibration (one-time, not per-session)

Before relying on fixed disfluency thresholds, a speaker can read one short,
phonetically-neutral sentence once. From that single read the app measures
their natural word duration and pause length and stores it as a **range**
(median + spread, not a single number — the same person's tempo varies run to
run). Block and prolongation thresholds are then personalized to that range:
a naturally slow speaker's normal pauses won't get misread as blocks, and a
naturally fast speaker keeps full sensitivity. Calibration never lowers
detection below the global default — it only ever raises a speaker's own bar
when their measured tempo is slower than that default. Re-read the sentence
any time tempo has visibly changed; recent reads are blended (last 5), so one
odd read doesn't permanently skew the baseline.

### 3 — Disfluency detection

The detector (`profiling/detect.py`) is **audio-native-primary, not just
ASR-transcript-confirmed**: the acoustic-native module (`profiling/acoustic.py`)
independently derives its own candidates straight from the waveform (energy,
zero-crossing rate, Silero VAD voice-activity, and Praat pitch/jitter/shimmer),
and is reconciled with the text/timing-based checks through **weighted-confidence
fusion** — the more confident signal wins per event, not a fixed "transcript
always wins" priority. Filler and stutter-marker events, which previously
trusted the ASR's own flags with no audio grounding at all, are now also
acoustically corroborated. `word_repetition`/`sound_repetition` candidates
are further confirmed by a small trained classifier
(`profiling/repetition_classifier.py`) over CrisperWhisper's own encoder
embedding — the transcript alone can't distinguish a genuine stuttered
repeat from a coincidental one ("that that..."), since both produce
identical tokens; this project's first internally-trained, shipped model
(decided by a pre-registered, cross-validated comparison, not intuition —
see `VALIDATION.md` §11-§13) resolves that ambiguity from the audio itself.

Event types follow the field's standard taxonomy (matching SEP-28k /
FluencyBank / KSoF, so output is directly comparable to public benchmarks —
see `profiling/evaluation/` and `VALIDATION.md`). Checked against the
speech-pathology and computational-detection literature in depth as of
2026-08 (`PHASE_2_RESEARCH_PLAN.md`) — the core 5 (sound/word repetition,
filler, block, prolongation) are confirmed scientifically sound and
benchmarked against real public datasets; `phrase_repetition` and
`stutter_marker` are detected and reported but are **not annotated as a
distinct category in any public benchmark dataset this project validates
against** (LibriStutter/SEP-28k/FluencyBank/KSoF/UCLASS) — treat their
numbers as unvalidated signal, not a benchmarked accuracy claim:

| Type                   | How it's detected                                                                                              | Benchmark status |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- | --- |
| **Sound repetition**    | A sub-word fragment (`word-`) repeated immediately before the word it fragments, e.g. "b- buy" | Validated (SEP-28k/FluencyBank/KSoF/LibriStutter) |
| **Word repetition**     | Same word twice in a row, near-duplicate words (phonetic similarity for short words, edit-distance for longer/OOV words), or the same word recurring after an intervening filler ("I uh I"). Also tagged `likely_sld` (True/False) from syllable count — monosyllabic repeats are stuttering-like per the clinical SLD/OD literature, polysyllabic ones are an ordinary linguistic-planning disfluency, not stuttering (`PHASE_2_RESEARCH_PLAN.md` §2.1). **This sub-tag is a descriptive heuristic, not a validated clinical measure** — no dataset labels this split. | Validated (type only; the SLD/OD sub-tag is not) |
| **Phrase repetition**   | An immediately-repeated multi-word phrase (any length from 2 up to a configurable cap), e.g. "I want to I want to" | **Not validated by any reviewed dataset** |
| **Filler**              | Token matches a known filler list (`uh`, `um`, `er`, `erm`, `like`) or is marked `is_filler` by CrisperWhisper — acoustically corroborated (voiced-energy check) when audio is available | Validated |
| **Stutter marker**      | Token ends with `-` (sub-word fragment) or is marked `is_stutter` by CrisperWhisper — same acoustic corroboration as filler | **Not validated by any reviewed dataset** |
| **Block**               | Silent gap between two consecutive words exceeding the (calibrated, if available) block threshold, confirmed against actual audio silence, then cross-checked against the independent acoustic-native block detector. **Silent-only** — the clinical literature also describes an "audible/struggle" block (sustained tension energy, not silence) this detector has no code path for at all (`PHASE_2_RESEARCH_PLAN.md` §2.2); not yet buildable against any dataset this project has, since none sub-type blocks this way. | Validated (silent sub-type only) |
| **Prolongation**        | Token duration exceeding the (calibrated, if available) prolongation threshold, confirmed against sustained voiced energy (RMS, ZCR) and, when available, stable pitch/low jitter/low shimmer — then cross-checked against the independent acoustic-native prolongation detector | Validated — literature identifies this as the type interpretable/rule-based detection handles best |

Disfluencies at the start of a sentence get a small confidence boost —
stuttering is overwhelmingly sentence-initial, so this is clinically
meaningful, not just a stylistic weighting. All thresholds — including which
detectors run and how much weight the acoustic-native signal gets in fusion —
are configurable in `config.yaml` (`profiling.detection.detectors` and
`profiling.detection.fusion_weights`).

### 4 — Speaker difficulty profile

Each signed-in user has a `SpeakerDifficultyProfile` stored as
`users/<username>.fluency_profile.json`. After each session:

- Detected events are grouped by phoneme onset (e.g. `B`, `P`, `S T R`).
- Each onset's risk score is updated via **EWMA** (α = 0.35 by default), so
  recent sessions have more weight than old ones.
- On first login (cold start), onset risks are seeded from population
  priors (`profiling/default_onset_priors.json`) and any self-reported
  difficult sounds, but real session data always wins once it exists —
  cold-start seeding never overwrites an onset that has observed data.
- A standing **per-word difficulty score** — onset risk (45%) + syllable
  length (25%) + word rarity (20%) + grammatical class (10%) — is available
  via `profile.difficulty(word)` and can be shown as background shading on
  the transcript (a toggle on the Analyse screen), independent of whether
  the detector flagged that specific word in that specific clip. This is
  the speaker's standing risk, not a one-off event.

### 5 — Visualisation

The **Analyse** tab shows:

- Full verbatim transcript with disfluent words highlighted in orange, and
  an optional background-shading overlay for the standing word-risk score.
- Summary stats: total tokens, disfluency count, fluency rate %.
- Per-type event badges and a full timestamped event table.

The **Profile** tab shows:

- Calibration status (word/gap tempo range, how many reads were pooled).
- Onset-risk bar chart (top 12 onsets by current risk score).
- Full session history with event counts per session.

### 6 — Four input modes

- **Demo** — instant fixture (no audio required, no model download).
- **Calibrate** — read the fixed reference sentence once to set your tempo
  baseline. Doesn't run disfluency detection and is never saved as a session.
- **Record** — live microphone capture via `streamlit-mic-recorder`.
- **Upload** — WAV, MP3, FLAC, M4A, JSON fixture, or plain TXT.

---

## File layout

```
audio-mod/
├── app.py                     ← Streamlit entry point
├── auth.py                    ← Login / Register screen
├── user_store.py              ← File-based user accounts (sha256 passwords)
├── semantic.py                ← Stub (protected word list only)
├── phonetic.py                ← CMUdict phoneme-onset utilities
├── freq.py                    ← wordfreq wrapper (memory-safe fallback)
├── paths.py                   ← Cache-path bootstrapper (import FIRST)
├── config.yaml                ← Tuning knobs (EWMA alpha, detection thresholds)
├── requirements.txt
├── README.md
├── ARCHITECTURE.md            ← Implementation deep-dive, data flow, known gaps
├── .gitignore
├── .streamlit/
│   └── config.toml            ← Light theme + file-watcher disabled
├── users/                     ← Runtime only — gitignored
└── profiling/
    ├── __init__.py
    ├── asr.py                 ← CrisperWhisper pipeline + resampler
    ├── benchmark_asr.py       ← ASR latency benchmark (table + RTF, --self-test)
    ├── acoustic.py            ← Audio-native disfluency cues: RMS/ZCR + Silero VAD + Praat pitch/jitter/shimmer/HNR
    ├── detect.py              ← Disfluency detector — audio-native-primary, weighted-confidence fusion with text/timing checks
    ├── evaluate.py            ← Backward-compatible shim over profiling/evaluation/ (see below)
    ├── evaluation/            ← Accuracy evaluation package — see VALIDATION.md for methodology
    │   ├── loaders.py          ← Per-dataset loading (LibriStutter word-level; SEP-28k clip-level labels)
    │   ├── metrics.py          ← Precision/recall/F1, confusion matrices, IoU localization, "Any" label
    │   ├── alignment.py        ← Track B: ASR-hypothesis <-> reference word alignment (Levenshtein, biased)
    │   ├── track_a.py          ← Detector-only runner, ASR bypassed (--self-test)
    │   ├── track_b.py          ← Full-pipeline runner: real ASR + alignment + per-clip caching (--self-test)
    │   ├── run_ablations.py    ← Config-variant sweep runner (VAD/Praat/fusion-weight/threshold)
    │   └── report.py           ← Table rendering + timestamped, reproducible result files
    ├── profile.py             ← SpeakerDifficultyProfile (EWMA + onset risk + difficulty model)
    ├── calibration.py         ← Speaker tempo baseline (calibration sentence + threshold adjustment)
    ├── coldstart.py           ← Population priors + self-report seeding
    ├── config.py              ← Config loader (YAML with hardcoded defaults)
    └── default_onset_priors.json
```

---

## Setup

```
cd audio-mod
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac/Linux replace the activate line with `source venv/bin/activate`.

That's it — no optional accelerator package to install. (An OpenVINO fast
path was attempted but turned out to be incompatible with the word-level
timestamps this app depends on; see
[ARCHITECTURE.md](ARCHITECTURE.md#asr-backend-selection) if you're curious
why, or before considering `pip install optimum[openvino]` yourself.)

### First-run NLTK data (if not already cached)

```
import nltk
nltk.download("cmudict")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt_tab")
```

---

## Run

```
streamlit run app.py
```

Opens at `http://localhost:8501`. Register an account on first visit.

---

## Verify it works (do this first)

1. Register / log in.
2. Go to **Analyse** → select **Demo (instant, no audio)** → click **Run demo now**.

Expected output in under 3 seconds:

- Green log box completing the pipeline steps.
- Transcript with orange-highlighted disfluent words.
- Stats: 9 tokens, 7 disfluencies, 22.2% fluency rate.
- Badges: `word repetition ×1`, `sound repetition ×1`, `stutter marker ×2`, `block ×1`, `filler ×1`, `prolongation ×1`.
- **Profile** tab showing risk bars for `B`, `T`, `S`.

If all of that appears the full pipeline is working end-to-end without needing
the ASR model.

---

## Calibrating your tempo (recommended, one-time)

1. Go to **Analyse** → select **Calibrate**.
2. Read the displayed sentence naturally, at your normal pace.
3. Click **Stop & calibrate**.

This doesn't run disfluency detection and isn't saved as a session — it only
updates your tempo baseline (visible afterwards on the **Profile** tab). You
can skip this entirely; detection uses sensible fixed defaults either way.
Re-calibrate any time your natural tempo has noticeably changed.

---

## Live microphone recording

1. Go to **Analyse** → select **Record**.
2. Click **Start recording**, speak, click **Stop & analyse**.

The first real-audio run downloads CrisperWhisper (~3.2 GB) automatically —
watch the terminal for download progress. All subsequent runs load from the
local cache in `.cache/hf/`.

**Expected inference time on CPU** (measured 2026-06-26, transformers backend,
16 GB CPU machine):

| Clip length | Inference | Real-time factor |
| ----------- | --------- | ---------------- |
| ~4s         | ~54s      | ~13×             |
| ~8s         | ~81s      | ~9.5×            |
| ~15s        | ~94s      | ~6×              |
| ~20s        | ~102s     | ~5×              |

Plus a **one-time ~29s model load** on the first transcription of a session
(it's cached after that — every later clip skips it). Inference scales with
clip length (a fixed encoder pass of ~44s plus ~1.4s per generated word on
CPU), so the real-time factor is *worse* for short clips (the fixed encoder
cost is spread over less audio). This is a real cost of running a ~3.2 GB
seq2seq model on CPU, not a bug. The app logs elapsed time every 4 seconds so
it never looks frozen — let it finish rather than assuming it's stuck. (Numbers
are reproducible with `python -m profiling.benchmark_asr`.)

---

## Configuration (`config.yaml`)

| Key                                            | Default | Effect                                                                                   |
| ----------------------------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `profiling.ewma_alpha`                          | `0.35`  | How fast new sessions overwrite old risk scores (0 = never update, 1 = replace entirely) |
| `profiling.confidence_events`                   | `30`    | Events needed before personal data fully overrides population priors                      |
| `profiling.weights.onset`                       | `0.45`  | Weight of phoneme-onset risk in per-word difficulty score                                 |
| `profiling.weights.length`                      | `0.25`  | Weight of syllable length                                                                  |
| `profiling.weights.frequency`                   | `0.20`  | Weight of word rarity                                                                      |
| `profiling.weights.grammatical_class`           | `0.10`  | Weight of content-word penalty                                                            |
| `profiling.detection.block_gap_seconds`         | `0.55`  | Global-floor minimum silence gap counted as a block (raised per speaker once calibrated)  |
| `profiling.detection.prolongation_min_seconds`  | `1.0`   | Global-floor minimum token duration counted as prolongation (same calibration behaviour). Raised from `0.65` after real-mic false positives (`PAPER_DECISION_LOG.md`, Part D); `VALIDATION.md` §9's ablation later found the aggregate-optimal value on the (synthetic, reconstructed-timing) LibriStutter sample is higher still (1.2–1.4) — not yet re-tuned, see `ROADMAP.md`. |
| `profiling.detection.prolongation_percentile`   | `90`    | Percentile threshold for prolongation detection (superseded per-clip when `use_rate_normalized_prolongation` is on) |
| `profiling.detection.use_rate_normalized_prolongation` | `false` | When `true`, replaces the percentile/floor threshold above with Esmaili et al. 2017's rate-normalized formula (`T = rate_alpha / speaking_rate`). Stays `false`: a 2026-08-04 ablation found it regresses both `Any` and prolongation-specific F1 on the LibriStutter benchmark (`VALIDATION.md` §9.5.1) — kept as a toggleable option, not removed, pending a real-speech dataset to re-test against. |
| `profiling.detection.prolongation_rate_alpha`   | `1.2`   | Numerator of the rate-normalized formula above; only used when that mode is enabled |
| `profiling.detection.prolongation_rate_floor`   | `1.5`   | Minimum speaking-rate (syllables/sec) the rate-normalized formula divides by, guarding against instability on short/sparse clips |
| `profiling.detection.require_praat_stability_for_prolongation` | `true` | **Hard gate** (not just a confidence adjustment) on the token-path prolongation check: candidate must pass pitch-stability/jitter/shimmer thresholds when Praat features are available; graceful no-op when they're not. Flipped to `true` 2026-08-04 — the only variant of a 13-variant ablation to improve both `Any` and prolongation-specific F1 simultaneously (`VALIDATION.md` §9.5.1). Distinct from `acoustic.use_praat` below, which remains confidence-only. |
| `profiling.detection.require_repetition_classifier_confirmation` | `true` | **Hard gate** on candidate `word_repetition`/`sound_repetition` events: a small trained classifier (`models/repetition_corroboration_classifier.npz`) over CrisperWhisper's own encoder embedding must confirm the candidate; graceful no-op when `transformers`/`torch`, the model file, or audio are unavailable. Flipped to `true` 2026-08-05 — a pre-registered, cross-validated comparison found this clears the decision bar decisively (`VALIDATION.md` §12.6.2, Cohen's d > 1.0). **Adds real latency when it engages** (a second CrisperWhisper encoder pass, ~30-90s, only on clips with an actual repetition candidate) — see `ARCHITECTURE.md` §4b's known-limitations note before enabling this in a latency-sensitive context. |
| `profiling.detection.near_repetition_similarity`| `0.75`  | Edit-distance similarity above which two consecutive words count as a near-repetition     |
| `profiling.detection.phrase_repetition_max_words`| `8`    | Longest repeated phrase scanned for (also capped at `len(tokens)//2`)                     |
| `profiling.detection.sentence_initial_boost`    | `0.08`  | Confidence bonus for disfluencies at sentence-initial position                            |
| `profiling.detection.detectors`                 | all 8   | Enable-list of which named checks run (`filler`, `stutter_marker`, `phrase_repetition`, `word_repetition`, `sound_repetition`, `block`, `prolongation`, `acoustic_fusion`) — toggle without touching code |
| `profiling.detection.fusion_weights.rule` / `.acoustic` | `1.0` / `1.0` | Per-source confidence weighting where the token-path and acoustic-native detectors compete for the same event; acoustic only wins on a strictly higher weighted confidence |
| `profiling.detection.acoustic.use_vad`          | `true`  | Silero VAD gates/down-weights acoustic prolongation confidence; self-disabling (no-op) on clips where VAD finds no speech at all (e.g. synthetic test tones) |
| `profiling.detection.acoustic.use_praat`        | `true`  | Praat pitch/jitter/shimmer/HNR as additional prolongation-corroborating evidence (confidence adjustment only, never a hard gate) |

---

## GitHub / version control notes

The `.gitignore` already excludes:

- `users/` — account credentials and profile data, never commit.
- `.cache/` — model weights (~3.2 GB), re-downloaded on first run.
- `venv/`, `__pycache__/`, `.DS_Store`.

Safe to commit: everything else, including `.streamlit/config.toml` (contains
only theme settings, no secrets).

---

## What this app does NOT do

- Synonym suggestion or sentence rephrasing (that lives in the main Speech AI
  pipeline — `profile.difficulty(word)` is the hook the rewrite pipeline is
  meant to call, but the rewrite pipeline itself is a separate codebase).
- Grammar correction.
- Multi-language transcription (English only, hardcoded).
- Real-time / streaming transcription (full clip is processed after recording
  stops — see [ARCHITECTURE.md](ARCHITECTURE.md#streaming-vs-faster-clips-a-deliberate-choice)
  for why, and what a streaming version would actually require).

For implementation details, data flow, and known limitations, see
[ARCHITECTURE.md](ARCHITECTURE.md).
