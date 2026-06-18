# Stammr Profiler — Standalone

A self-contained Streamlit app that:

1. **Captures audio** three ways — instant demo fixture, live microphone
   recording, or file upload (WAV / MP3 / FLAC / M4A / JSON / TXT).
2. **Transcribes** it verbatim via **CrisperWhisper**
   (`nyrahealth/CrisperWhisper`), preserving repetitions and false starts
   that vanilla Whisper silently deletes.
3. **Detects disfluencies** — repetitions, silent blocks, fillers,
   prolongations, ASR stutter markers — via the rule-based detector in
   `profiling/detect.py`.
4. **Updates** the signed-in user's `SpeakerDifficultyProfile` with
   EWMA-weighted onset-risk scores, stored in
   `users/<username>.fluency_profile.json`.
5. **Visualises** the transcript, stats, onset-risk bars, and full session
   history in the UI.

This module is **completely independent** of the main Speech AI synonym /
grammar pipeline. It does not import `rewrite/`, `grammar.py`, `engine.py`,
or the real `semantic.py`.

---

## File layout

```
profiling_app/
├── app.py                    ← Streamlit entry point (3 input modes)
├── auth.py                   ← Login / Register screen
├── user_store.py             ← File-based user accounts
├── semantic.py                ← Stub (only _PROTECTED_SINGLE needed)
├── phonetic.py                ← Phoneme-onset utilities
├── freq.py                    ← wordfreq wrapper
├── paths.py                   ← Cache-path bootstrapper (import FIRST)
├── config.yaml                ← Tuning knobs
├── requirements.txt
├── README.md                  ← this file
├── .gitignore
├── .streamlit/
│   └── config.toml             ← Forces a light UI theme app-wide
├── users/                     ← User account files + fluency profiles (gitignored)
└── profiling/
    ├── __init__.py
    ├── asr.py                 ← CrisperWhisper wrapper + WAV resampler + fallback
    ├── detect.py               ← Rule-based disfluency detector
    ├── profile.py               ← SpeakerDifficultyProfile (EWMA + onset risk)
    ├── coldstart.py             ← Population priors + self-report seeding
    ├── config.py                ← Config loader
    └── default_onset_priors.json
```

---

## Setup

```bash
cd profiling_app
pip install -r requirements.txt
```

If `pip` complains about externally-managed environments (common on Linux/Mac
system Python), use a virtual environment instead:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

> **First real audio run:** CrisperWhisper (~3.2 GB) downloads automatically.
> JSON/TXT fixtures and the built-in demo skip this entirely and run instantly.

### Optional — NLTK data (only needed if not already cached)

```python
import nltk
nltk.download("cmudict")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("punkt_tab")
```

---

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

---

## How to verify it's working (do this first)

1. Register/log in.
2. Go to the **Analyse** tab.
3. Select **Demo (instant, no audio)**.
4. Click **Run demo now**.

Expected, in under 3 seconds:
- A green log box ticking through 6 steps
- Transcript with disfluent words highlighted in orange
- Stats: 9 tokens, 7 disfluencies, 22.2% fluency rate
- Badges: `repetition x 2`, `stutter marker x 2`, `block x 1`, `filler x 1`, `prolongation x 1`
- A full event table with timestamps and confidence scores
- The **My Profile** tab now shows risk bars for `B`, `T`, `S`

If all of that appears, the entire pipeline — ASR loader, detector, profile
updater, UI — is confirmed working end to end.

---

## Live microphone recording

1. Make sure `streamlit-mic-recorder` is installed (it's in `requirements.txt`).
2. Go to **Analyse** -> select **Record now**.
3. Click **Start recording**, speak, click **Stop & analyse**.
4. The captured clip plays back, then the log box runs, then results appear.

The first real-audio run downloads the CrisperWhisper model (~3 GB) — watch
your terminal for `model.safetensors: XX%` progress. Subsequent runs are
fast because the model is cached locally.

---

## Testing without a GPU / without downloading the model

Upload a **JSON fixture** instead of real audio:

```json
{
  "tokens": [
    {"word": "I",      "start": 0.0, "end": 0.2},
    {"word": "I",      "start": 0.2, "end": 0.4, "is_stutter": true},
    {"word": "want",   "start": 0.4, "end": 0.7},
    {"word": "uh",     "start": 1.5, "end": 1.7, "is_filler": true},
    {"word": "something", "start": 1.7, "end": 3.0}
  ]
}
```

Or a plain `.txt` file — words tokenise automatically; trailing `word-`
fragments and common fillers (`uh`, `um`, `er`...) are flagged.

---

## Configuration (`config.yaml`)

| Key | Default | Effect |
|---|---|---|
| `profiling.ewma_alpha` | `0.35` | How fast new sessions overwrite old risk scores |
| `profiling.confidence_events` | `30` | Events needed before personal data fully overrides population priors |
| `profiling.weights.onset` | `0.45` | Weight of phoneme-onset risk in difficulty score |
| `profiling.detection.block_gap_seconds` | `0.55` | Minimum silence gap counted as a block |
| `profiling.detection.prolongation_min_seconds` | `0.65` | Minimum token duration counted as prolongation |

---

## Fixes applied (changelog)

| Issue | Root cause | Fix |
|---|---|---|
| 10+ minute hang, no transcript | CrisperWhisper ran multilingual language-detection first; `forced_decoder_ids` in its `generation_config` conflicted with `return_timestamps` | `asr.py` now passes `generate_kwargs={"language": "en", "task": "transcribe", "forced_decoder_ids": None}` |
| `size of tensor a (2) must match tensor b (0)` | Missing `chunk_length_s` caused attention-mask shape mismatches on short/mic clips | Added `chunk_length_s=30` to the pipeline call |
| Silent failure when `transformers` missing | Old code fell back to timing-only mode marked `profile_safe=False`, which the detector silently dropped — looked like nothing was happening | Now raises a clear `RuntimeError` with install instructions, shown directly in the UI |
| Mic audio (44100 Hz) crashing or producing garbage | Whisper expects 16 kHz; `audioop` (the usual stdlib resampler) was removed in Python 3.13 | Added `resample_to_16k()` in `asr.py` using **numpy only** — works on Python 3.13 |
| White-on-white text | Sidebar dark-theme CSS leaked into the global selector | CSS scoped strictly: sidebar styling only inside `section[data-testid="stSidebar"]`, body text forced to a dark color explicitly |
| `use_container_width` deprecation warnings | Streamlit deprecated the param in favor of `width=` | All `st.dataframe(...)` calls now use `width="stretch"` |
| `torchvision` `ModuleNotFoundError` spam in terminal | Streamlit's file watcher scans every transformers submodule, including vision models you'll never use | **Harmless — safe to ignore.** Does not affect ASR. Install `torchvision` only if you want to silence it. |
| Selects, the file uploader, and other widgets rendered as solid black boxes — only readable on hover/focus | No `.streamlit/config.toml`, so native widgets inherited the visitor's OS/browser dark-mode theme instead of the app's own light styling | Added `.streamlit/config.toml` (`base = "light"`); added explicit light CSS for the select popover, file-upload dropzone, and checkboxes |
| Run-log / "elapsed time" text invisible (dark text on a dark panel) | CSS specificity bug — a broad `div` text-color rule unintentionally overrode the log panel's own (lighter) text color | Log panel rule rewritten with higher specificity and switched to the same light theme as the rest of the UI |
| Login page felt very long, mostly empty space above the box | Page is set to `layout="wide"`; the login screen never reset Streamlit's default top padding the way the main app does | `auth.py` now resets the top padding and caps the page to 480px wide |
| Login username/password fields, tabs, and buttons unstyled | Only the outer card `<div>` had CSS — the actual `st.text_input` / `st.tabs` / `st.button` widgets fell back to Streamlit's default theme | Explicit light styling added for inputs, tabs, and buttons in `auth.py` |

---

## Putting this on GitHub

A `.gitignore` is included and already excludes the things you don't want
in version control for this project:

- **`users/`** — account files and `*.fluency_profile.json` data. This is
  real (even if test) user data and login credentials; it shouldn't ever
  be pushed, even to a private repo.
- **Model caches** — CrisperWhisper is ~3.2 GB and re-downloads on first
  run anyway; committing it would blow past GitHub's file-size limits.
- **`venv/`, `__pycache__/`, `.DS_Store`**, and other local/OS noise.

A couple of things worth doing before or right after your first push:

- **Keep `.streamlit/config.toml`** — unlike `secrets.toml`, it only holds
  theme colors, not credentials, so it's safe and expected to be committed.
- **Check `user_store.py`** for how passwords are stored before this goes
  anywhere public — if they're hashed, great; if not, that's worth fixing
  before real users touch it, even in testing.
- **Add a `LICENSE`** if you want to make the repo's terms explicit (MIT is
  the common default for small tools like this).
- **Pin versions in `requirements.txt`** (e.g. `streamlit==1.x.x`) so a
  fresh clone doesn't break from an unrelated upstream update.

---

## What this is NOT

- Does not suggest synonyms or rephrase sentences.
- Does not run grammar correction.
- Does not transcribe-then-analyze text as a separate step (that pipeline
  lives in the main project). This app only profiles disfluencies found
  directly in audio.
