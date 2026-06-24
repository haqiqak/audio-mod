# Audio Profiler

A standalone Streamlit app for verbatim speech transcription and disfluency
profiling. Built for speakers who stutter — it captures exactly what was said
(including repetitions, fillers, and false starts that standard ASR silently
removes), builds a personalized profile of which sounds and word types are
most difficult for that speaker, and calibrates detection to that speaker's
own natural speaking tempo instead of a one-size-fits-all threshold.

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

The rule-based detector (`profiling/detect.py`) flags five event types:

| Type               | How it's detected                                                                                              |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Repetition**      | Same word twice in a row, near-duplicate words (edit-distance similarity), a trailing fragment (`word-`) before the same word, or the same word recurring after an intervening filler ("I uh I") |
| **Filler**          | Token matches a known filler list (`uh`, `um`, `er`, `erm`, `like`) or is marked `is_filler` by CrisperWhisper |
| **Stutter marker**  | Token ends with `-` (sub-word fragment) or is marked `is_stutter` by CrisperWhisper |
| **Block**           | Silent gap between two consecutive words exceeding the (calibrated, if available) block threshold, confirmed against actual audio silence when the waveform is available |
| **Prolongation**    | Token duration exceeding the (calibrated, if available) prolongation threshold, confirmed against sustained voiced energy when the waveform is available |

Disfluencies at the start of a sentence get a small confidence boost —
stuttering is overwhelmingly sentence-initial, so this is clinically
meaningful, not just a stylistic weighting. All thresholds are configurable
in `config.yaml`.

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
    ├── detect.py              ← Rule-based disfluency detector
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
- Badges: `repetition ×2`, `stutter marker ×2`, `block ×1`, `filler ×1`, `prolongation ×1`.
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

**Expected inference time on CPU:** roughly 30-50s for a short clip (2-5s),
scaling up with clip length — this is a real cost of running a ~3.2 GB
seq2seq model on CPU, not a bug. The app logs elapsed time every 4 seconds
so it never looks frozen; for a calibration read or longer clip, expect it
to take noticeably longer (a few minutes is normal for a clip in the 5-10s
range) — let it finish rather than assuming it's stuck.

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
| `profiling.detection.prolongation_min_seconds`  | `0.65`  | Global-floor minimum token duration counted as prolongation (same calibration behaviour)  |
| `profiling.detection.prolongation_percentile`   | `90`    | Percentile threshold for prolongation detection                                           |
| `profiling.detection.near_repetition_similarity`| `0.75`  | Edit-distance similarity above which two consecutive words count as a near-repetition     |
| `profiling.detection.sentence_initial_boost`    | `0.08`  | Confidence bonus for disfluencies at sentence-initial position                            |

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
