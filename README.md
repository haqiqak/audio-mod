# Audio Profiler

A standalone Streamlit app for verbatim speech transcription and disfluency
profiling. Built for speakers who stutter — it captures exactly what was said
(including repetitions, fillers, and false starts that standard ASR silently
removes) and builds a personalized profile of which sounds and word types are
most difficult for that speaker over time.

---

## What it does

### 1 — Transcription
Audio is transcribed verbatim using **CrisperWhisper** (`nyrahealth/CrisperWhisper`),
a fine-tuned version of whisper-large-v3 specifically trained to preserve
disfluencies rather than clean them up. Word-level timestamps are returned for
every token.

### 2 — Disfluency detection
The rule-based detector (`profiling/detect.py`) flags five event types:

| Type | How it's detected |
|---|---|
| **Repetition** | Same word appears twice in a row, or a trailing fragment (`word-`) immediately precedes the same word |
| **Filler** | Token matches a known filler list (`uh`, `um`, `er`, `erm`, `like`) or is marked `is_filler` by CrisperWhisper |
| **Stutter marker** | Token ends with `-` (sub-word fragment) or is marked `is_stutter` by CrisperWhisper |
| **Block** | Silent gap ≥ 0.55s between two consecutive words |
| **Prolongation** | Token duration ≥ 90th-percentile of all durations in the clip AND ≥ 0.65s |

All thresholds are configurable in `config.yaml`.

### 3 — Speaker difficulty profile
Each signed-in user has a `SpeakerDifficultyProfile` stored as
`users/<username>.fluency_profile.json`. After each session:

- Detected events are grouped by phoneme onset (e.g. `B`, `P`, `S T R`).
- Each onset's risk score is updated via **EWMA** (α = 0.35 by default),
  so recent sessions have more weight than old ones.
- On first login (cold start), onset risks are seeded from population
  priors (`default_onset_priors.json`) and any self-reported difficult sounds.
- Word difficulty is a weighted combination of four factors:
  onset risk (45%), syllable length (25%), word rarity (20%), and
  grammatical class (10%).

### 4 — Visualisation
The **Analyse** tab shows:
- Full verbatim transcript with disfluent words highlighted in orange.
- Summary stats: total tokens, disfluency count, fluency rate %.
- Per-type event badges and a full timestamped event table.

The **My Profile** tab shows:
- Onset-risk bar chart (top 8 onsets by current risk score).
- Full session history with event counts per session.

### 5 — Three input modes
- **Demo** — instant fixture (no audio required, no model download).
- **Record now** — live microphone capture via `streamlit-mic-recorder`.
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
├── .gitignore
├── .streamlit/
│   └── config.toml            ← Light theme + file-watcher disabled
├── users/                     ← Runtime only — gitignored
└── profiling/
    ├── __init__.py
    ├── asr.py                 ← CrisperWhisper pipeline + resampler
    ├── detect.py              ← Rule-based disfluency detector
    ├── profile.py             ← SpeakerDifficultyProfile (EWMA + onset risk)
    ├── coldstart.py           ← Population priors + self-report seeding
    ├── config.py              ← Config loader (YAML with hardcoded defaults)
    └── default_onset_priors.json
```

---

## Setup

```powershell
cd audio-mod
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac/Linux replace the activate line with `source venv/bin/activate`.

### First-run NLTK data (if not already cached)

```python
import nltk
nltk.download("cmudict")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt_tab")
```

---

## Run

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`. Register an account on first visit.

---

## Verify it works (do this first)

1. Register / log in.
2. Go to **Analyse** → select **Demo (instant, no audio)** → click **Run demo now**.

Expected output in under 3 seconds:
- Green log box completing 6 steps.
- Transcript with orange-highlighted disfluent words.
- Stats: 9 tokens, 7 disfluencies, 22.2% fluency rate.
- Badges: `repetition ×2`, `stutter marker ×2`, `block ×1`, `filler ×1`, `prolongation ×1`.
- **My Profile** tab showing risk bars for `B`, `T`, `S`.

If all of that appears the full pipeline is working end-to-end without needing
the ASR model.

---

## Live microphone recording

1. Go to **Analyse** → select **Record now**.
2. Click **Start recording**, speak, click **Stop & analyse**.

The first real-audio run downloads CrisperWhisper (~3.2 GB) automatically —
watch the terminal for download progress. All subsequent runs load from the
local cache in `.cache/hf/`.

**Expected inference time on CPU:** ~40–80 seconds for a short clip (2–5s),
scaling roughly with clip length. This is normal — CrisperWhisper is a
large model and there is no GPU in this setup. The app logs elapsed time
every 4 seconds so it never looks frozen.

---

## Configuration (`config.yaml`)

| Key | Default | Effect |
|---|---|---|
| `profiling.ewma_alpha` | `0.35` | How fast new sessions overwrite old risk scores (0 = never update, 1 = replace entirely) |
| `profiling.confidence_events` | `30` | Events needed before personal data fully overrides population priors |
| `profiling.weights.onset` | `0.45` | Weight of phoneme-onset risk in per-word difficulty score |
| `profiling.weights.length` | `0.25` | Weight of syllable length |
| `profiling.weights.frequency` | `0.20` | Weight of word rarity |
| `profiling.weights.grammatical_class` | `0.10` | Weight of content-word penalty |
| `profiling.detection.block_gap_seconds` | `0.55` | Minimum silence gap counted as a block |
| `profiling.detection.prolongation_min_seconds` | `0.65` | Minimum token duration counted as prolongation |
| `profiling.detection.prolongation_percentile` | `90` | Percentile threshold for prolongation detection |

---

## Changelog

| Issue | Root cause | Fix |
|---|---|---|
| 600–700s inference per clip | BLAS/OpenMP threads hardcoded to 1, single-threading all encoder matmuls | `paths.py` now sets threads to `min(4, cpu_count)` |
| 600–700s inference per clip (real cause) | `transformers < 4.47` lacked WhisperSdpaAttention; 4.47+ ships it and drops inference to ~50s | `requirements.txt` now pins `transformers>=4.47,<5.0` |
| `KeyError: num_frames` crash | `transformers 5.0` dropped the `num_frames` key from WhisperFeatureExtractor output, breaking the ASR pipeline's preprocess step | Upper-bounded to `<5.0` in requirements |
| 10+ minute hang with no output | CrisperWhisper's `generation_config` had `forced_decoder_ids` set, conflicting with `return_timestamps="word"` | `asr.py` passes `language="en"`, `task="transcribe"`, and clears `forced_decoder_ids` |
| `size of tensor a must match tensor b` crash | `num_beams > 1` + `return_timestamps="word"` triggers a confirmed transformers bug in beam-index reshaping | `num_beams=1` forced in pipeline `generate_kwargs` |
| Mic audio producing garbage | Browser captures at 44100 Hz; Whisper expects 16 kHz; `audioop` removed in Python 3.13 | `asr.py` resamples to 16 kHz using numpy only |
| Black boxes on widgets (dark mode) | No `config.toml` — widgets inherited OS dark theme | `.streamlit/config.toml` added with `base = "light"` |
| torchvision spam in terminal | Streamlit file-watcher scans all transformers submodules including vision models | `config.toml` sets `fileWatcherType = "none"` |
| Cold-start profile overwriting trained scores | `onboarding()` seeded all onsets on every page load, inflating previously trained-down scores | Seed now only applies to onsets with no observed session data |

---

## GitHub / version control notes

The `.gitignore` already excludes:
- `users/` — account credentials and profile data, never commit.
- `.cache/` — model weights (~3.2 GB), re-downloaded on first run.
- `venv/`, `__pycache__/`, `.DS_Store`.

Safe to commit: everything else, including `.streamlit/config.toml`
(contains only theme settings, no secrets).

---

## What this app does NOT do

- Synonym suggestion or sentence rephrasing (that lives in the main Speech AI pipeline).
- Grammar correction.
- Multi-language transcription (English only, hardcoded).
- Real-time / streaming transcription (full clip is processed after recording stops).
