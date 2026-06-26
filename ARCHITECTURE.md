# ARCHITECTURE.md

Implementation deep-dive for `audio-mod`. Read this before changing detection
logic, the ASR pipeline, or the profile model — it documents what the code
actually does today (verified by reading and running it), not just what an
earlier design intended. Where an earlier note turned out to be stale relative
to the live code, that's called out explicitly rather than silently dropped.

---

## 1. What this project is

`audio-mod` is a standalone Streamlit app for verbatim speech transcription
and stuttering disfluency profiling. It's one module of a larger "Speech AI"
system — the part that handles audio. A separate codebase (not in this repo)
is meant to consume `profile.difficulty(word)` to drive synonym suggestions;
that consumer doesn't exist in this repo and isn't wired up here.

**Core user story:** a person who stutters records a speech sample. The app
transcribes it verbatim (preserving repetitions, fillers, false starts),
detects which words triggered disfluencies, and builds a personalized profile
of which phoneme onsets and word types are hardest for that speaker — using
detection thresholds calibrated to that speaker's own natural tempo rather
than one global cutoff for everyone.

---

## 2. Data flow, end to end

```
 audio bytes / fixture
        │
        ▼
 profiling/asr.py  (CrisperWhisperASR.transcribe / transcribe_bytes)
        │  → list[VerbatimToken-as-dict]: word, start, end, is_filler, is_stutter
        ▼
 profiling/detect.py  (detect_disfluencies)
        │  takes tokens + optional audio_bytes (acoustic confirmation)
        │  + optional speaker_baseline (profiling/calibration.py)
        │  → list[event]: word, index, start, end, type, confidence, evidence
        ▼
 profiling/profile.py  (SpeakerDifficultyProfile.update)
        │  groups events by phoneme onset, EWMA-updates onset_risk
        │  persists to users/<username>.fluency_profile.json
        ▼
 app.py renders: transcript with flagged words + optional difficulty()
 shading, event table, onset-risk bars on the Profile tab
```

A second, mostly-independent loop runs for calibration:

```
 calibration read (fixed sentence, audio or fixture)
        │
        ▼
 profiling/asr.py  → tokens (same path as above)
        ▼
 profiling/calibration.py  (measure_calibration_read → update_baseline)
        │  pools median + IQR of word duration and inter-word gap
        ▼
 SpeakerDifficultyProfile.speaker_baseline  (persisted alongside onset_risk)
        │
        ▼
 detect_disfluencies(..., speaker_baseline=profile.speaker_baseline)
        personalizes block_gap_seconds / prolongation_min_seconds
        (never below the config/global floor)
```

---

## 3. profiling/asr.py — CrisperWhisper pipeline

Wraps `nyrahealth/CrisperWhisper` (a whisper-large-v3 fine-tune, ~3.2 GB) in a
`transformers` ASR pipeline and returns one `VerbatimToken` per word, with
`word`, `start`, `end`, `is_filler`, `is_stutter`.

### Critical, do-not-change-casually settings

```python
generate_kwargs = {
    "language": "en",
    "task": "transcribe",
    "num_beams": 1,          # ← THE fix, not a style choice
    "max_new_tokens": <duration-proportional, 20-256>,
}
```

`num_beams=1` sidesteps a confirmed transformers bug
(huggingface/transformers #28007, #36093) where `return_timestamps="word"`
combined with beam search (`num_beams > 1`) mis-shapes `beam_indices` and
crashes with `size of tensor a (2) must match tensor b (0)`. CrisperWhisper's
own `generation_config.json` defaults to multiple beams, which is what
triggers it. Do not pass `chunk_length_s` (transformers flags it experimental
for seq2seq models — Whisper already does its own long-form chunking) or
`forced_decoder_ids` alongside `language`/`task` (creates duplicate
`SuppressTokensLogitsProcessor` instances that fight each other).

`transformers` is pinned to `>=4.47.0,<5.0.0`: `<4.47` lacks
`WhisperSdpaAttention` (inference takes 600-700s on CPU); `>=5.0` drops the
`num_frames` key the ASR pipeline's preprocessing step depends on.

### ASR backend selection

`ASR_BACKEND` env var:

| Value | Behaviour |
|---|---|
| `transformers` (default) | The only backend that currently produces correct word-level timestamps, which every feature in this app depends on. CPU latency is ~54s (4s clip) to ~102s (20s clip) plus a one-time ~29s load — see "Measured latency" below. |
| `auto` | Currently identical to `transformers` — kept as an alias for if/when OpenVINO becomes usable again (see below). |
| `openvino` | **Do not use.** Raises immediately with a clear error instead of attempting transcription — see the incident below for why. |
| `faster_whisper` | Tried and ruled out — see below. Not auto-selected. |

**Why `faster_whisper` doesn't work:** confirmed incompatibility, not a
missing-dependency case. CrisperWhisper's HF repo ships the slow tokenizer
format; converting weights with `ct2-transformers-converter` succeeds, but
faster-whisper's internal tokenizer wrapper hardcodes stock-Whisper
special-token positions (`<|startoftranscript|>`, language/timestamp tokens
at fixed offsets). CrisperWhisper's fine-tune has pruned/shifted token IDs
(confirmed by a "vocabulary contains holes" warning during conversion), so
faster-whisper's hardcoded assumptions don't hold, regardless of how the
model is converted. The failure is `ValueError: <|startoftranscript|> token
was not found in the prompt` — not a fixable config issue. Left in the code
for anyone who wants to retry against a future faster-whisper release with
configurable special-token layouts.

**Why OpenVINO doesn't work — a real incident, documented in full because
it's a good example of how "looks like it works" can fail later:**

An earlier round of this project made `optimum-intel`'s
`OVModelForSpeechSeq2Seq` the auto-selected default, reasoning that it
accelerates the matmul backend underneath the *same* `transformers`
model/tokenizer objects already known to work — there's only ever one
tokenizer implementation in play, so the usual "different tokenizer,
different bugs" risk (the exact thing that sank `faster_whisper`) didn't
seem to apply. The reasoning about the tokenizer was correct. It missed a
different problem.

A real calibration recording (not a fixture) hit this in practice: the
encoder pass ran fast as expected, but the run then crashed ~6 minutes in,
deep inside `transformers/models/whisper/generation_whisper.py`:

```
File ".../generation_whisper.py", line 191, in _extract_token_timestamps
    cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))
TypeError: 'NoneType' object is not subscriptable
```

Root cause: `return_timestamps="word"` (which this app always sets — word
timestamps are the foundation of everything downstream) makes transformers'
Whisper generation code compute word-level alignment from the model's
cross-attention weights after generation finishes. The plain PyTorch model
returns those. The OpenVINO-compiled model's `generate()` does not —
`generate_outputs.cross_attentions` comes back `None`. This is a confirmed,
**still open as of this writing** upstream bug:
`github.com/huggingface/optimum-intel` issue #561 ("OVModelForSpeechSeq2Seq
fails with `return_timestamps="word"`"), with no known workaround reported.
It is not something fixable from this codebase, and not something a smarter
`generate_kwargs` choice can route around — the attention weights are simply
never produced by the OpenVINO decoder.

**The lesson, not just the fix:** the earlier reasoning ("same tokenizer
objects, so the usual risk doesn't apply") was true and still missed this,
because the actual failure mode lived one layer deeper — in what the
*compiled model's generate() call returns*, not in tokenization at all. A
fast encoder pass and a correct transcript for *text-only* output would have
made this look fully fixed in a quick smoke test; it only surfaces once
something downstream (here, word-timestamp extraction) depends on output
the fast path doesn't actually produce. Treat "the simple case worked" as
weaker evidence than it feels like, for exactly this kind of integration.

Current state: `_transcribe_openvino()` raises a clear `RuntimeError`
immediately (citing issue #561) instead of letting anyone hit the confusing
`NoneType` crash several minutes into a real transcription. The original
implementation is preserved, renamed to
`_transcribe_openvino_DISABLED_reference_impl`, for whoever revisits this if
optimum-intel ever fixes #561 upstream — at that point it can become the
real default again, but verify the fix actually restores `cross_attentions`
before flipping the default; don't just check that transcription completes
without crashing, since this exact bug doesn't crash the *text* output, only
word-timestamp extraction.

### Timing history (measured during development, kept for context)

| Stage | ~Time for a 2-3s clip |
|---|---|
| Before any fix (BLAS/OMP threads hardcoded to 1) | ~1300s |
| After `paths.py` thread fix (`min(4, cpu_count)`) | ~680s |
| After `transformers>=4.47` (WhisperSdpaAttention) | ~47-50s |
| OpenVINO | not usable — see incident above. The encoder pass itself was fast (a real calibration read's progress log showed it running for several minutes total, but that run included generation/timestamp-extraction time before the crash, not a clean comparable number — don't infer a speed figure from it). |

The thread-count fix matters specifically because OpenBLAS pre-allocates
per-thread scratch buffers at *load* time sized for every CPU core; a low cap
avoids that load-time OOM without forcing every subsequent matmul onto a
single core (which is what the original hardcoded `1` was accidentally doing
to *inference* speed, not just load-time memory).

### Measured latency (2026-06-26, real benchmark — supersedes the figures above)

Run with `python -m profiling.benchmark_asr` (transformers backend, CPU, 16 GB
machine, threads capped at 4 by `paths.py`):

| Clip | Inference | RTF (infer ÷ clip) | Tokens |
|---|---|---|---|
| ~4s  | ~54s  | ~13×  | 7  |
| ~8s  | ~81s  | ~9.5× | 17 |
| ~15s | ~94s  | ~6×   | 26 |
| ~20s | ~102s | ~5×   | 41 |

Two things to take from this, both correcting earlier assumptions:

1. **It's inference-bound, not load-bound.** Model load is a one-time ~29s on
   the first clip of a process and **~0s on every clip after** (the
   `st.cache_resource` path in `app.py` works — confirmed by the benchmark's
   warm-load row reading 0.00s). So the recurring cost the user feels is
   inference, not loading.
2. **Inference scales with clip length — it is *not* a fixed ~30s-window cost.**
   The numbers fit roughly "a ~44s fixed encoder pass + ~1.4s per generated
   word on CPU": a short clip is dominated by the encoder (hence the *worse*
   RTF — fixed cost spread over less audio), and longer clips add decoder time
   per word. The earlier note that "the decode loop barely matters / it's one
   fixed cost regardless of clip length" was wrong; decode time grows visibly
   with token count (7 → 41 tokens added ~48s).

The `~47-50s` development figure below the line was inference-only for a short
clip and is in the right ballpark for the ~4s row; it just omitted the one-time
load and the length-scaling, both now measured.

---

## 4. profiling/detect.py — rule-based disfluency detector

Five event types: filler, stutter marker, repetition (exact / near /
fragment / filler-sandwiched / phrase-level), block, prolongation. All take
`audio_bytes` for acoustic confirmation when available (RMS for
silence/voicing, zero-crossing rate to distinguish voiced sustain from noise)
and degrade gracefully to timestamp-only when it isn't.

### Threshold personalization (calibration.py integration)

`detect_disfluencies(tokens, config=None, audio_bytes=None,
speaker_baseline=None)` — the `speaker_baseline` parameter is optional and
additive. When omitted (or when `speaker_baseline.is_usable` is False), the
function is identical to its pre-calibration behaviour — this is verified by
a regression test that confirms the demo fixture still produces exactly 7
events. When provided and usable, `calibration.adjusted_thresholds()`
recomputes `block_gap_seconds` and `prolongation_min_seconds` as
`max(global_floor, speaker_median + k * speaker_iqr)` — so calibration can
only raise a speaker's own bar above the default, never lower detection
sensitivity below what an uncalibrated speaker gets.

### Edge cases verified by direct testing (not just claimed)

- Empty token list, single-token clips, punctuation-only tokens → all return
  `[]` cleanly, no exceptions.
- Missing `start`/`end` (either or both `None`) on any token → skipped
  safely, never crashes the gap/duration math.
- `<5` tokens: the 90th-percentile prolongation threshold is meaningless
  (every word looks prolonged relative to itself), so the detector falls
  back to a flat `1.5x` the absolute minimum instead.
- Triple repetition ("I I I want") → both repeats correctly flagged
  independently, not just the first pair.
- Repeated sub-word fragments ("str- str- street") → fragment-repetition,
  stutter-marker, and the final completed-word repetition all fire as
  distinct, correctly-attributed events.
- Repetition across what looks like a sentence boundary due to punctuation
  ("Buy." then "buy") → still correctly caught, because normalization
  strips punctuation/case before comparison; sentence-initial detection is
  based purely on timing gap (≥1.5s), not punctuation, so this is correctly
  *not* double-counted as sentence-initial.
- **Leading/trailing silence is trimmed before the prolongation check when
  audio is available** (the §1 "word-timestamp acoustic cross-validation"
  fix). The ASR anchors a word's `start` to the chunk boundary, so clip-initial
  silence gets billed to the first word — making it look prolonged *and*
  inflating the clip-wide 90th-percentile threshold so genuine prolongations
  elsewhere get suppressed. `_AcousticContext.voiced_span()`/`voiced_duration()`
  recover each word's voiced extent (frame-wise RMS, edges only — a mid-word dip
  doesn't shorten a sustained sound), and that voiced duration feeds both the
  percentile and the per-word check. Verified by direct test
  (`tests/test_detect_acoustic.py`): a silence-padded first word is not flagged
  while a genuinely sustained later word still is, and the with-audio vs
  timestamp-only contrast shows the percentile-poisoning the fix removes.
  **Caveat:** this needs the waveform — in timestamp-only mode (fixtures, or
  audio the detector never received) the raw durations are still used, so the
  bug can persist there. That's an inherent limit of cross-checking against
  audio you don't have, not a regression.

### Known, currently-accepted limitations (not yet fixed — listed honestly)

- **Near-repetition similarity is computed on spelling (edit distance), not
  phonetics**, despite ARPAbet onsets already being available via
  `phonetic.py`. This unfairly penalizes short words (`"a"` vs `"I"` reads as
  0% similar) and can't distinguish a real stutter-repeat from two
  genuinely different but similarly-spelled words ("strawberry" vs
  "strawberries"). A phonetic-distance version for short words, falling back
  to edit distance for longer ones, would be the natural fix — not yet done.
- **Phrase-repetition only checks 2-3 word windows**
  (`phrase_repetition_min_words` + 1). Longer repeated phrases fall through
  silently. Not yet generalized to a sliding window over arbitrary lengths.
- **The multi-factor `difficulty()` score and the event-based detector are
  two separate signals**, surfaced separately in the UI (event highlighting
  vs. optional background shading) rather than merged into one combined
  confidence number. This is a deliberate choice for this round — conflating
  "this word was flagged this time" with "this speaker tends to struggle
  with this word in general" would lose information either signal carries
  on its own. Revisit if user testing shows the two signals are confusing
  side by side.

---

## 5. profiling/calibration.py — speaker tempo baseline

### Why this exists

Block (`block_gap_seconds`) and prolongation (`prolongation_min_seconds`)
thresholds were pure global constants. A naturally slow, deliberate speaker
trips them on completely normal speech; a naturally fast speaker's real
blocks and prolongations might never clear them. The fix is making the
threshold relative to that speaker's own rate, not finding a smarter shared
constant — there isn't one shared constant that's right for everyone.

### Design

- **One fixed, phonetically-neutral calibration sentence**
  (`CALIBRATION_SENTENCE`) — no plosive clusters or rare words, since this
  measures *tempo*, not difficulty. ~20 words, comfortable in one breath.
- **A range, not a point**: `SpeakerBaseline` stores median + IQR for both
  word duration and inter-word gap, because the same person's tempo varies
  read to read. `update_baseline()` pools the newest read with up to
  `MAX_BASELINE_SAMPLES - 1` "carried" copies of the previous median, so a
  handful of recent reads shape the range together rather than the newest
  read silently overwriting everything before it.
- **Usable after one read.** `is_usable` requires only one calibration
  session with at least 3 timed words — calibration is explicitly meant to
  be a one-time (or rarely-repeated) setup step per the product requirement,
  not something a speaker has to do every session before it does anything.
  Additional reads refine the range; they aren't a gate on using it at all.
- **Reading-pause exclusion**: gaps over 1.2s in a calibration read are
  excluded from the *gap* baseline (treated as a natural pause between
  clauses while reading aloud, not the speaker's word-to-word rhythm) —
  otherwise a single dramatic pause while reading would inflate the personal
  block threshold far above what's clinically meaningful.
- **Floor, never ceiling.** `adjusted_thresholds()` always returns
  `max(global_default, personalized_value)`. A speaker who is naturally
  *faster* than the global default keeps the global default — calibration
  cannot make detection less sensitive than the out-of-the-box behaviour.

### Verified behaviour (tested directly, not just asserted in comments)

- A synthetic "slow speaker" calibration read (longer natural word durations
  and gaps, with realistic variance) correctly raises their personal
  thresholds above the global floor.
- A clip with a 0.6s gap — long enough to trip the global 0.55s default —
  is correctly suppressed as a false-positive block once that speaker is
  calibrated, and correctly still flagged for an uncalibrated or
  naturally-fast speaker on the identical clip.
- The original 9-token/7-event demo fixture is unaffected when no baseline
  is supplied — confirmed by direct regression test.

### Not yet built

- No UI affordance to *re-run* calibration if a speaker's tempo has shifted
  significantly except manually re-selecting the Calibrate input mode —
  there's no automatic "your tempo looks different from your baseline, want
  to recalibrate?" prompt. Worth adding if real usage shows tempo drift is
  common (e.g. fatigue, time of day, emotional state).
- The `gap_k=2.2` / `duration_k=1.8` multipliers in `adjusted_thresholds()`
  are reasonable starting points, not empirically tuned against real speaker
  data. Treat them as a first pass.

---

## 6. profiling/profile.py — SpeakerDifficultyProfile

### What it stores

Persisted as `users/<username>.fluency_profile.json`:

- `onset_risk` — ARPAbet onset → risk score `[0,1]`.
- `onset_observations` — raw `{events, disfluent}` counts per onset.
- `self_reported_sounds` — user-typed problem sounds.
- `sessions` — last 100 session records (word-level events).
- `speaker_baseline` — the calibration range from §5, if any.
- `event_count` — total events across all sessions.

Backward compatible: profiles saved before `speaker_baseline` existed load
correctly with an unusable (all-zero) baseline rather than failing — verified
directly by loading a hand-built pre-calibration profile dict.

### Difficulty formula — now actually wired into the UI

```
difficulty(word) =
    0.45 * onset_risk(word's phoneme onset)
  + 0.25 * min(syllable_count / 4.0, 1.0)
  + 0.20 * (1 - min(zipf_frequency / 7.0, 1.0))
  + 0.10 * is_content_word(word)
```

**This was previously dead code.** `factors_for_word()`, `difficulty()`, and
`sentence_difficulty()` were fully implemented but nothing in `app.py` called
them — only the raw EWMA `onset_risk` dict reached the UI, via the Profile
tab's bar chart. As of this round, `app.py`'s Analyse screen has an optional
"Show personalized word-risk shading" toggle that calls `profile.difficulty()`
per word and shades the transcript accordingly, independent of the discrete
event highlighting. Verified: function words ("the") score meaningfully
lower than rare content words with hard onsets ("strawberry", onset `S T R`).

### EWMA update

`new_risk = alpha * observed_rate + (1 - alpha) * previous_risk`, alpha=0.35
by default. `observed_rate` is the fraction of that onset's tokens in the
session that were disfluent.

### Cold start

`fused_cold_start()` blends population priors (`default_onset_priors.json`,
19 onsets, plosives highest at 0.40-0.42) with self-reported sounds (seeded
at 0.82), weighted by `max(0, 1 - observed_events/confidence_events)` so
personal data takes over after ~30 events. `onboarding()` only seeds onsets
with *no* observed session data — fixed in an earlier round after a bug where
it was re-applying priors on every page load and inflating trained-down
scores back up.

**Known dormant edge case in `fused_cold_start`**: once `prior_weight` decays
to 0 (after `confidence_events` observations), the blend formula collapses to
just the population prior, silently discarding the self-reported value
entirely rather than blending it down gracefully. Currently harmless because
`onboarding()` only ever uses this as a `max()` floor for onsets with zero
observed data — but if this function's output is ever used more directly
elsewhere, that collapse will quietly drop user self-reports the moment they
have session history. Not fixed this round; flagged for whoever touches
`coldstart.py` next.

### Bugs fixed this round

- **`_onset_key()` had dead, duplicated logic**: it computed the
  `is_arpabet_code` check once, then immediately recomputed it with a
  "simpler guard" comment and overwrote the first result. The first
  computation never affected behaviour — removed, only the live logic
  remains. (This function is also where an earlier bug lived: vowel-initial
  words like "I" were being misbucketed as the phantom onset `I` because
  `"I".isupper()` is `True`. That bug is fixed at the source in the current
  guard — `fix_profile.py`, a one-time cleanup script for profiles saved
  before that fix, is no longer needed and has been removed; its logic is
  preserved here for context in case an old profile JSON ever resurfaces
  with phantom `I`/`A`/`E`/`O`/`U` onset keys.)
- **`_guess_tag()` called `nltk.download(...)` on every invocation.** Cheap
  in isolation (local cache check), but now that `difficulty()` is actually
  called per-word from the live UI loop, that's a per-word tax across every
  transcript. Moved to a one-time module-level check (`_ensure_pos_tagger`).

---

## 7. Streaming vs. faster clips — a deliberate choice

The stated goal is detecting disfluencies in "continuous real-world speech
without performance lag." Two different things could satisfy that:

1. **Faster processing of a single recorded clip** — partially solved.
   `transformers>=4.47`'s `WhisperSdpaAttention` got CPU inference for a
   short clip down to ~54s (4s clip; measured 2026-06-26, §3), scaling to
   ~102s for a 20s clip. OpenVINO was meant to be the next step in this
   direction but turned out not to be usable at all (§3's incident writeup) —
   so the realistic next lever for "faster," if needed, is a different one:
   profiling `_load_pipeline()`/`generate()` itself for further CPU-side wins,
   or accepting the current ~54-102s range as the working baseline rather than
   assuming a quick backend swap will improve it.
2. **True streaming transcription** — processing audio incrementally as
   someone talks, rather than waiting for a full clip to be recorded and
   then transcribed. This is architecturally a different system: it needs
   sliding-window inference with overlap-stitching, a state machine for
   partial-vs-final transcripts, and a UI that updates incrementally instead
   of showing one result block at the end.

**Decision for this round: still don't build (2), but the calculus is now
less comfortable than it was.** Streaming is a large, separate engineering
effort (different threading model, different UI, different correctness
story for word timestamps near window boundaries), and this is presently a
Streamlit research/clinical tool, not a phone app with a hard real-time
latency requirement — that part of the reasoning hasn't changed. What has
changed is that (1) is no longer "mostly solved": ~54s for a short clip on CPU
(scaling to ~102s at 20s; measured 2026-06-26) is the real current floor, not a
stopgap on the way to a few seconds.
If that latency turns out to be a genuine adoption blocker rather than a
tolerable wait, it's worth treating as its own investigation (CPU
profiling, a smaller/distilled model, or a GPU path) before reaching for
streaming as the fix — streaming a slow per-window inference doesn't
solve slowness, it just changes where the wait is felt.

**If streaming becomes a real requirement later**, the practical starting
points are: a sliding window (e.g. 5-10s) with ~1-2s overlap, word-level
timestamp reconciliation across overlapping windows (dedupe/stitch words that
appear in both), and a UI that appends finalized words incrementally instead
of replacing the whole transcript per clip. None of this exists yet — treat
it as a separate project, not an incremental patch on `run_pipeline()`.

---

## 8. Dependencies — current pinned state

```
streamlit>=1.38.0
streamlit-mic-recorder>=0.0.7
transformers>=4.47.0,<5.0.0    ← 4.47+ for SdpaAttention, <5.0 for num_frames
accelerate>=0.26
torch>=2.0
nltk>=3.8
wordfreq>=3.0
numpy>=1.24
PyYAML>=6.0
```

Do **not** install `optimum[openvino]` expecting a working speedup — see §3's
incident writeup. `ASR_BACKEND=openvino` raises immediately with a clear
error rather than attempting transcription with it.

Do not upgrade `transformers` past 5.0 without testing a full mic-record
round-trip — the `num_frames` KeyError returns immediately. Do not install
`faster-whisper`/`ctranslate2` expecting it to work with CrisperWhisper — see
§3's tokenizer-incompatibility explanation; it's a confirmed dead end, not an
untried option.

---

## 9. Quick orientation for whoever (human or Claude) touches this next

**To understand the codebase:** `app.py` (UI flow) → `profiling/asr.py` (ASR)
→ `profiling/detect.py` (disfluency logic) → `profiling/calibration.py`
(tempo baseline) → `profiling/profile.py` (difficulty model + persistence).

**To tune detection:** edit `config.yaml` — no code changes needed for
threshold tuning. To change detection *logic* (new event types, different
repetition heuristics), edit `profiling/detect.py`.

**To change the difficulty model:** edit `profiling/profile.py`. EWMA update
is in `update()`, the difficulty formula is in `factors_for_word()`, cold-
start seeding is in `onboarding()` + `coldstart.py`.

**To change calibration behaviour:** edit `profiling/calibration.py`. The
calibration sentence, the usability gate, and the `gap_k`/`duration_k`
multipliers are all there and isolated from the rest of detection.

**To change ASR:** edit `profiling/asr.py`. Read the comments in
`_load_pipeline()` and `_transcribe_openvino()`'s docstring carefully first —
several settings there are bug-fixes for specific crashes, not style
choices, and the OpenVINO path is deliberately disabled (raises immediately)
rather than silently routed around — don't re-enable it without confirming
optimum-intel issue #561 is actually fixed upstream, not just that
transcription completes without crashing (see §3 for why that check alone
isn't enough).

**Before trusting any claim in an `.md` file (including this one) over the
code**: run the code. Several specific contradictions between earlier
planning docs and the actual shipped implementation were found and resolved
while writing this version — docs drift, especially across long iterative
sessions; treat them as a starting hypothesis to verify, not ground truth.
