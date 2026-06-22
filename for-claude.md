# audio-mod — Full Handoff Document for Claude

> **Purpose of this file:** This is a briefing document for a new Claude session.
> It explains what this project is, what has been built and why, the exact
> current state of every component, what was tried and failed, and concrete
> proposals for what to build next. Read this before touching any file.

---

## 1. What this project is

`audio-mod` is a standalone Streamlit application for verbatim speech
transcription and stuttering disfluency profiling. It is one module of a
larger "Speech AI" / "StammAI" system — the part that handles audio. The
other parts (synonym suggestion, grammar correction, sentence rewriting) live
in a separate codebase and are deliberately not imported here.

**The core user story:** A person who stutters records a short speech sample.
The app transcribes it verbatim (preserving repetitions, fillers, false starts),
detects which words triggered disfluencies, and builds a personalized profile of
which phoneme onsets and word types are most difficult for that speaker. Over
many sessions, the profile converges toward accurate personal difficulty scores
that downstream synonym-replacement logic can use to suggest easier words.

**What makes this non-trivial:** Standard ASR (Whisper, etc.) silently cleans
up disfluencies — it hears "I I want" and outputs "I want". CrisperWhisper
is a fine-tune specifically trained to preserve exactly those events. Getting
it to run correctly and at usable speed on CPU (no GPU available) was the
main engineering challenge in this session.

---

## 2. File structure

```
audio-mod/
├── app.py                     ← Streamlit entry point. Handles UI, 3 input
│                                modes (demo/mic/upload), progress logging,
│                                calls ASR + detector + profile updater.
├── auth.py                    ← Login/Register screen. Light-theme CSS,
│                                tab_login / tab_reg, calls user_store.py.
├── user_store.py              ← File-based accounts in users/<name>.json.
│                                SHA-256 password hashing. No DB dependency.
├── semantic.py                ← STUB only. Contains _PROTECTED_SINGLE frozenset
│                                (function words that should never be replaced).
│                                The real synonym engine is in the other repo.
├── phonetic.py                ← CMUdict phoneme-onset utilities. onset(word)
│                                returns ARPAbet onset cluster. word_difficulty()
│                                returns a [0,1] heuristic. Used by profile.py.
├── freq.py                    ← wordfreq wrapper with MemoryError fallback to
│                                the small wordlist. zipf_frequency(word, lang).
├── paths.py                   ← MUST be imported first in every entry module.
│                                Redirects all caches (.cache/hf, .cache/nltk,
│                                etc.) into the project folder. Sets BLAS/OMP
│                                thread count to min(4, cpu_count) — this was
│                                the first big performance fix (was hardcoded 1).
├── config.yaml                ← All tuning knobs. EWMA alpha, detection
│                                thresholds, difficulty weights. Loaded by
│                                profiling/config.py with hardcoded fallbacks.
├── requirements.txt           ← Pinned to transformers>=4.47,<5.0 — critical.
├── .streamlit/
│   └── config.toml            ← base="light" theme + fileWatcherType="none"
│                                (disables file-watcher to stop the torchvision
│                                spam in terminal).
└── profiling/
    ├── __init__.py
    ├── asr.py                 ← See section 3 — the most complex file.
    ├── detect.py              ← Rule-based detector. See section 4.
    ├── profile.py             ← SpeakerDifficultyProfile. See section 5.
    ├── coldstart.py           ← Population priors + self-report seeding.
    ├── config.py              ← YAML loader with deep-merge + hardcoded defaults.
    └── default_onset_priors.json ← Population-level plosive/fricative risks.
```

---

## 3. asr.py — CrisperWhisper pipeline (most complex file)

### What it does
Wraps `nyrahealth/CrisperWhisper` (a whisper-large-v3 fine-tune, ~3.2 GB) in
a `transformers` ASR pipeline and returns a list of `VerbatimToken` objects —
one per word — with `word`, `start`, `end`, `is_filler`, `is_stutter` fields.

### Class: CrisperWhisperASR
- `model_id = "nyrahealth/CrisperWhisper"`
- `backend` — env var `ASR_BACKEND`, default `"transformers"`.
  Two other backend stubs exist (`faster_whisper`, `openvino`) but both
  proved incompatible — see section 7 (What Was Tried).
- `last_timing` — dict with `load_pipeline_seconds` and `inference_seconds`,
  populated after each call, displayed in the UI log.
- `transcribe(path)` — main entry point. Pads short clips, dispatches to backend.
- `transcribe_bytes(bytes)` — writes bytes to a temp WAV, calls transcribe().
- `resample_to_16k(audio_bytes)` — resamples any sample rate to 16 kHz mono
  using numpy only (audioop was removed in Python 3.13).

### Critical pipeline configuration (do not change without reading this)

```python
pipeline(
    "automatic-speech-recognition",
    model="nyrahealth/CrisperWhisper",
    return_timestamps="word",
    model_kwargs={"low_cpu_mem_usage": True},
    generate_kwargs={
        "language": "en",
        "task": "transcribe",
        "num_beams": 1,          # ← THE critical setting
        "max_new_tokens": 256,
    },
)
```

**Why `num_beams=1` is mandatory:** `return_timestamps="word"` combined with
`num_beams > 1` triggers a confirmed transformers bug (issues #28007, #36093)
where `_extract_token_timestamps()` mis-shapes `beam_indices`. CrisperWhisper's
own `generation_config.json` defaults to multiple beams, which is what triggers
it. `num_beams=1` (greedy decoding) sidesteps the bug entirely. The CrisperWhisper
model card also explicitly recommends beam_size=1 for word timestamps.

**Why NOT to pass `chunk_length_s`:** transformers explicitly marks this as
"very experimental with seq2seq models". Whisper handles its own long-form
chunking internally; `chunk_length_s` replaces that with a less accurate
sliding-window mechanism.

**Why NOT to pass `forced_decoder_ids`:** The pipeline builds its own logits
processors from `language`/`task`. Passing `forced_decoder_ids` on top creates
duplicate `SuppressTokensLogitsProcessor` instances. Just pass `language` and
`task` and let the pipeline handle the rest.

### Timing context
Before this session: ~1300s for a 3-second clip.
After thread fix (paths.py): ~680s.
After upgrading to transformers 4.47.1: ~47s for a 2-second clip.

The final speedup came from transformers 4.47 shipping **WhisperSdpaAttention**
(scaled dot-product attention), which uses PyTorch's optimized `F.scaled_dot_product_attention`
kernel instead of the manual attention implementation. This is where the real
~14x speedup came from, not from any code changes we made.

---

## 4. detect.py — Rule-based disfluency detector

Takes a list of tokens (VerbatimToken or dicts) and returns a list of event
dicts, one per flagged token, with fields: `word`, `index`, `start`, `end`,
`type`, `confidence`, `evidence`.

### Five detection rules (in order of application)

**Filler** — token matches known filler list (`uh`, `um`, `er`, `erm`, `like`)
OR CrisperWhisper marked it `is_filler=True`. Confidence: 0.9.

**Stutter marker** — token ends with `-` (sub-word fragment) OR CrisperWhisper
marked it `is_stutter=True`. Confidence: 0.85.

**Repetition** — two consecutive tokens have the same normalized text (`_norm`
strips non-alpha). Also catches fragment repetition: if previous token ends with
`-` and the current token starts with the same stem. Confidence: 0.92 / 0.86.

**Block** — gap between `prev.end` and `curr.start` ≥ `block_gap_seconds`
(default 0.55s). Confidence: min(0.95, gap/block_gap) — larger gaps = higher confidence.

**Prolongation** — token duration ≥ 90th percentile of all durations in the
clip AND ≥ `prolongation_min_seconds` (default 0.65s). Not flagged for filler
words (fillers are inherently long). Confidence: min(0.95, dur/threshold).

### Known limitation
Prolongation detection is only as good as CrisperWhisper's word-level
timestamps. On short clips or quiet audio, timestamps can be imprecise (±100ms),
which can generate false prolongation events. The 90th-percentile threshold
helps by making it relative to the clip itself, not an absolute value.

---

## 5. profile.py — SpeakerDifficultyProfile

### What it stores (persisted as `users/<username>.fluency_profile.json`)
- `onset_risk` — dict of ARPAbet onset → risk score [0,1]
  e.g. `{"B": 0.62, "P": 0.41, "S T R": 0.78}`
- `onset_observations` — dict of onset → `{events: int, disfluent: int}`
  (raw counts used for confidence estimation)
- `self_reported_sounds` — list of user-typed problem sounds e.g. `["b", "str"]`
- `sessions` — list of up to 100 session records (each with word-level events)
- `event_count` — total events seen across all sessions

### Difficulty formula (per word)

```
difficulty = w_onset   * onset_risk(word's phoneme onset)
           + w_length  * min(syllable_count / 4.0, 1.0)
           + w_freq    * (1 - min(zipf_frequency / 7.0, 1.0))
           + w_class   * is_content_word(word)
```

Weights from `config.yaml`: onset=0.45, length=0.25, frequency=0.20, class=0.10.

### EWMA update rule

```
new_risk = alpha * observed_rate + (1 - alpha) * previous_risk
```

`alpha` = 0.35 (default). `observed_rate` = disfluent_events / total_events
for that onset in the session. So if a speaker had 3 stutters on /B/ words
and 1 fluent /B/ word, `observed_rate = 0.75`.

### Cold-start (no session data yet)
`fused_cold_start()` in `coldstart.py` blends:
- Population priors from `default_onset_priors.json` (plosives highest: B/P/T/D/K/G at 0.40-0.42)
- Self-reported sounds (seeded at 0.82)
- Prior weight decays as `max(0, 1 - observed_events / confidence_events)`
  so personal data gradually takes over after ~30 events.

### Fixed bug (important context)
An earlier version re-applied cold-start priors on every page load via
`onboarding()`, which was inflating trained-down scores back toward the
self-report prior. Fixed: `onboarding()` now only seeds onsets that have
NO observed session data (i.e. `onset not in self.onset_observations`).

---

## 6. Performance — current state (no quality trade-offs made)

### Quality: unchanged from stock CrisperWhisper
- Same model weights, same tokenizer, same decoding logic.
- `num_beams=1` is a quality trade-off on paper (greedy vs beam search)
  but in practice CrisperWhisper's card recommends it for word timestamps
  anyway, and beam search was causing crashes, not improving output.
- No quantization, no weight modification, no backend swap.

### Speed improvements applied
| Change | Where | Effect |
|---|---|---|
| BLAS/OMP threads: 1 → min(4, cpu_count) | `paths.py` | ~1300s → ~680s |
| Duration-proportional `max_new_tokens` | `asr.py` | Marginal on short clips; prevents tail blowup on long ones |
| `transformers>=4.47` (WhisperSdpaAttention) | `requirements.txt` | ~680s → ~47s (the real fix) |

### Current measured performance
~47 seconds for a 2-3 second microphone clip on a mid-range laptop CPU.
Scales roughly linearly with clip duration (Whisper's 30s window means clips
under 30s all pay a similar base cost).

---

## 7. What was tried and why it didn't work (do not retry without reading this)

### faster-whisper / CTranslate2
**Goal:** Replace the transformers encoder forward pass with CTranslate2's
quantized C++ kernels (typically 4-10x faster on CPU).

**What happened:** CrisperWhisper's HF repo ships the *slow* tokenizer format
(`vocab.json` + `merges.txt`, no `tokenizer.json`). We generated `tokenizer.json`
via `AutoTokenizer.from_pretrained(..., use_fast=True)` and converted the
weights successfully with `ct2-transformers-converter`. But faster-whisper's
internal tokenizer wrapper **hardcodes** assumptions about stock Whisper's
special-token layout (`<|startoftranscript|>`, language tokens, timestamp tokens
at exact positions). CrisperWhisper's fine-tune has a different token layout
(confirmed by the "vocabulary contains holes" warning during conversion — it has
deliberately pruned token IDs). This causes:

```
ValueError: <|startoftranscript|> token was not found in the prompt
```

This is NOT a conversion error or a missing-file error. It's a fundamental
incompatibility between faster-whisper's hardcoded tokenizer assumptions and
CrisperWhisper's non-standard token layout.

**Conclusion:** Do not retry faster-whisper with CrisperWhisper unless
faster-whisper adds support for custom special-token layouts.

### OpenVINO / optimum-intel
**Goal:** Accelerate the same transformers model/tokenizer through Intel's
OpenVINO IR quantized inference engine — avoids the tokenizer problem since
it uses the same HF tokenizer.

**What happened:** `OVModelForSpeechSeq2Seq` wraps the model in a non-standard
class that doesn't pass `isinstance(model, WhisperForConditionalGeneration)`.
The `transformers` ASR pipeline's `preprocess()` method checks for this exact
class to decide whether to compute `num_frames` from the feature extractor.
When the check fails, `num_frames` is never populated and the pipeline crashes:

```
KeyError: 'num_frames'
```

This affects both the OpenVINO path and any other path that wraps the model
in a non-WhisperForConditionalGeneration container (including `quantize_dynamic`,
which was also tried and failed for the same reason).

**Conclusion:** Do not retry optimum-intel / OVModelForSpeechSeq2Seq until
the transformers ASR pipeline properly handles non-standard model wrappers
for Whisper.

### torch.quantization.quantize_dynamic
**Goal:** In-place int8 quantization of Linear layers — no class change,
no tokenizer change, should be transparent to the pipeline.

**What happened:** Same `num_frames` KeyError as OpenVINO. The pipeline
apparently checks `type(self.model)` or `self.model.__class__` somewhere in
the preprocessing path, and `quantize_dynamic` wraps the model in a
`QuantizedLinear`-patched version that changes the class identity enough
to fail the check.

**Conclusion:** `quantize_dynamic` on the pipeline's `.model` attribute
breaks the `num_frames` preprocessing step. Do not retry.

---

## 8. What to build next — scaling proposals

### Priority 1 — Improve disfluency detection accuracy

**Current state:** Pure rule-based on word-level timestamps. Works well for
repetitions and fillers. Blocks and prolongations are timestamp-dependent and
can be noisy on short/quiet clips.

**Proposed improvements:**

A. **Confidence calibration** — current confidence scores (e.g. 0.92 for
repetition) are hardcoded constants, not empirically calibrated. A simple
improvement: use edit distance between consecutive tokens (not just exact match)
to catch near-repetitions like "the the" vs "the a" vs "they". Already partially
there with the fragment-repetition rule.

B. **Prosodic features for prolongation** — instead of relying solely on
CrisperWhisper's timestamps (which can be ±100ms), use the raw audio waveform
to compute per-word energy and zero-crossing rate. A prolonged phoneme looks
very different from a normally spoken word in the waveform. This would let you
set much tighter prolongation thresholds.

C. **ML-based event classifier** — train a small classifier (logistic regression
or a tiny LSTM) on features: duration z-score, gap before/after, is_filler flag,
onset type, position in sentence. Labels come from the rule-based detector as
weak supervision, cleaned up by the EWMA profile. This is a natural next step
once you have enough session data accumulated across users.

### Priority 2 — Connect profiling output to synonym replacement

**Current state:** `profile.py` exposes `difficulty(word)` and `sentence_difficulty(text)`.
The synonym/rewrite pipeline in the main repo doesn't call these yet.

**What needs to happen:**
The main Speech AI pipeline's `engine.py` (HybridEngine) currently scores
synonym candidates by phonetic onset matching against a static user-provided
list. The right upgrade is:
1. Load `SpeakerDifficultyProfile.load(username)` at session start.
2. Replace static onset matching with `profile.difficulty(candidate_word)`.
3. Filter/rank synonyms so candidates with lower `difficulty()` scores are
   preferred, and candidates with onsets in the user's top-risk list are
   explicitly penalized.
4. After the user speaks the rewritten sentence, feed the new session's
   disfluency events back into `profile.update()` — closing the learning loop.

This is the core value proposition of the whole system and it's currently
not connected end-to-end.

### Priority 3 — Improve ASR speed further (if still needed after Priority 2)

**Current state:** ~47s for a 2-3s clip. Usable but slow for a real-time
feel. Scales with clip length.

**Options that haven't been tried yet:**

A. **Flash Attention 2** — if the user ever gets a GPU or upgrades to a machine
where CUDA is available, `model_kwargs={"attn_implementation": "flash_attention_2"}`
drops inference time by 2-3x on GPU. Zero code change needed, just a kwarg.

B. **whisper.cpp** — a pure C++ reimplementation of Whisper that runs on CPU
with AVX2 SIMD. It has its own tokenizer implementation so the CrisperWhisper
tokenizer incompatibility we hit with faster-whisper doesn't necessarily apply
(depends on whether its special-token handling is also hardcoded). Worth
investigating if CPU speed is still a bottleneck.

C. **Lighter model** — CrisperWhisper is large-v3 sized (~3.2GB). If a
small or medium CrisperWhisper fine-tune were released (or trained), it would
be 4-8x faster at proportionally lower transcription quality. For disfluency
detection specifically, where you mostly need to catch repetitions and fillers
rather than rare vocabulary, a smaller model may be sufficient.

D. **Server-side inference** — move ASR off the user's machine entirely.
The Streamlit app sends audio bytes to an API endpoint, the server (with GPU)
transcribes in 2-3 seconds, returns tokens. This is the production path for
real deployment.

### Priority 4 — Multi-user and data persistence improvements

**Current state:** Users stored as flat JSON files in `users/`. Fluency profiles
stored alongside. Works fine for single-user or small-group testing but doesn't
scale.

**Proposed:**
- SQLite for user accounts (drop-in, still file-based, proper transactions).
- Profile versioning — keep the last N profiles per user rather than just the
  current one, so you can show "improvement over time" charts.
- Session export — let users download their session data as CSV for external
  analysis.
- Anonymized aggregation — opt-in pooling of session data across users to
  improve the population priors in `default_onset_priors.json`.

### Priority 5 — UI improvements

A. **Real-time waveform display** during recording (streamlit-webrtc or a
custom component) — gives visual feedback that recording is active.

B. **Word-level playback** — click a highlighted word in the transcript to
jump the audio player to that timestamp. Requires storing the audio clip and
building a JS audio player with timestamp seek.

C. **Trend charts** — show per-onset risk score over sessions as a line chart.
Profile tab currently only shows current risk bars; a user can't see improvement.

D. **Session comparison** — side-by-side view of two sessions: "last week vs
today", showing which onsets improved and which got worse.

---

## 9. Dependencies — current pinned state

```
streamlit>=1.38.0
streamlit-mic-recorder>=0.0.7
transformers>=4.47.0,<5.0.0    ← CRITICAL: 4.47+ for SdpaAttention, <5.0 for num_frames
accelerate>=0.26
torch>=2.0
nltk>=3.8
wordfreq>=3.0
numpy>=1.24
PyYAML>=6.0
```

Do not upgrade `transformers` past 5.0 without testing a full mic-record
round-trip. The `num_frames` KeyError will return immediately.

Do not install `faster-whisper`, `ctranslate2`, `optimum`, or `optimum-intel`
without reading section 7 first. They were tried and are incompatible with
CrisperWhisper in the current transformers pipeline.

---

## 10. Quick orientation for a new Claude session

**To understand the codebase:** Start with `app.py` to see the UI flow, then
`profiling/asr.py` for the ASR pipeline, then `profiling/detect.py` for
disfluency logic, then `profiling/profile.py` for the difficulty model.

**To make changes to detection:** Edit `profiling/detect.py` and `config.yaml`.
The thresholds are all configurable; no code changes needed for tuning.

**To make changes to the profile model:** Edit `profiling/profile.py`. The
EWMA update logic is in `update()`, the difficulty formula is in
`factors_for_word()`, and the cold-start seeding is in `onboarding()` +
`coldstart.py`.

**To make changes to ASR:** Edit `profiling/asr.py`. Read the comments in
`_load_pipeline()` carefully before touching `generate_kwargs` — several of
those settings are bug-fixes for specific crashes, not stylistic choices.

**The most impactful next task** is connecting `profile.difficulty(word)` into
the main Speech AI synonym pipeline (Priority 2 above). Everything on the audio
side is working and stable; the value is in closing the loop between "what
words does this person stutter on" and "which synonyms do we suggest".
