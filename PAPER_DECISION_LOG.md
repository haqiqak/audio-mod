# PAPER_DECISION_LOG.md

A chronological, append-only record of engineering decisions made on
`audio-mod`, one entry per verified step. It exists so that *why* a change was
made — and what it actually measured — survives even when the code and the
prose docs (`README.md`, `ARCHITECTURE.md`) drift apart.

This file was created fresh on 2026-06-26. The earlier planning/roadmap docs
(`improve.md`, `future.md`, `for-claude.md`) and their inline "Results log"
were removed in commit `7c7e808` ("Update project") and consolidated into
`ARCHITECTURE.md`; this log replaces that scattered convention with one place.

**Conventions used here**

- Entries are append-only and ordered oldest → newest. Never rewrite a past
  entry; if something is later found wrong, add a new entry correcting it.
- Each entry follows the same four-part shape:
  - **What was done** — the concrete change, with file references.
  - **Alternatives considered** — what else was on the table.
  - **Why this choice** — the reasoning, including trade-offs accepted.
  - **Measured result** — the actual numbers/test output, or "not yet
    measured" if a step is verified only by tests and a real run is pending.
- "§1 / §2" referenced in entries below mean: **§1** = the accuracy track
  (word-timestamp acoustic cross-validation — the leading-silence /
  sub-word-stutter problems); **§2** = the streaming-vs-faster-clips latency
  direction (`ARCHITECTURE.md` §7). These section numbers are local to this
  log because the original `IMPROVEMENT_PLAN.md`/improve.md that defined them
  no longer exists in the repo.

---

## 2026-06-26 — Step 1a: make `last_timing` self-describing

**What was done**
`CrisperWhisperASR.last_timing` ([profiling/asr.py](profiling/asr.py)) now
records, in addition to the pre-existing `load_pipeline_seconds` and
`inference_seconds`: `clip_duration_seconds`, `max_new_tokens`,
`audio_size_bytes`, and `backend`. Added `_clip_duration_seconds()` and
`_audio_size_bytes()` helpers and refactored `_max_new_tokens_for()` to reuse
the duration helper (single WAV-header read). Both the `transformers` (live)
and `faster_whisper` (dormant) paths populate the full schema;
`faster_whisper` records `max_new_tokens=None` because it stops via
`beam_size`/VAD rather than a token cap. New unit test
[tests/test_asr_timing.py](tests/test_asr_timing.py) verifies the schema using
a **stub pipeline** injected into `asr._pipe` — no real ~3.2 GB model load.

**Alternatives considered**
- Leave `last_timing` as the two-field dict and compute duration/RTF only in
  the benchmark script. Rejected: the goal is for the *app itself* (app.py's
  log line) and any future caller to read honest, self-describing timing
  without re-deriving clip duration separately.
- Record peak RSS / memory. Deferred: not needed for the latency question this
  round, and harder to measure portably on Windows.

**Why this choice**
`max_new_tokens` and `clip_duration_seconds` are exactly the two numbers needed
to (a) compute real-time factor and (b) check whether the decode-token budget
is reasonable against real token counts (Step 3b). Putting them on
`last_timing` keeps one source of truth. Testing with a stub keeps the unit
test fast and CI-friendly instead of gating on a 3.2 GB download + minutes of
CPU inference.

**Measured result**
`python tests/test_asr_timing.py` → 3/3 pass (schema present and correctly
typed; `max_new_tokens` budget = 32 for a 2.0 s clip = `int(2*6)+20`; floors at
20, ceilings at 256; non-WAV → `clip_duration_seconds=None`, budget falls back
to 256). Demo-fixture regression unchanged: **9 tokens / 7 disfluencies**.

---

## 2026-06-26 — Step 1b: benchmark harness for honest latency

**What was done**
Added [profiling/benchmark_asr.py](profiling/benchmark_asr.py): runs the ASR
pipeline over a folder of WAVs (default `./benchmark_clips/`) and prints one
table — `File | Duration(s) | Load(s) | Infer(s) | RTF | Tokens` — where
RTF = inference ÷ clip duration. Clips are run shortest-first (so the
model-load-bearing first row is the cheapest to wait on), and a summary reports
RTF range plus first-clip vs warm-clip load (to show whether the cached-model
path holds). All numbers are read straight from `CrisperWhisperASR.last_timing`
(Step 1a) — the harness adds no timing of its own. A built-in `--self-test`
verifies the table layout and RTF math against a stub pipeline + generated
silence WAVs, with no real model load. Errors per clip are captured into the
row so one bad file doesn't abort the batch; an empty/missing clips folder
prints a clear "drop real recordings" message and exits non-zero.

**Alternatives considered**
- Put the table/RTF logic only in an ad-hoc script. Rejected: making
  `format_table`/`_rtf`/`benchmark_clip` importable + self-testable means the
  format is trusted before a slow real run, per the measurement-first rule.
- Time the clips inside the harness with its own `perf_counter`. Rejected:
  that would double-measure and could disagree with what the app reports;
  reading `last_timing` keeps a single source of truth.
- Use pytest. Rejected for now: pytest isn't installed in the venv; a
  `--self-test` flag + the plain-assert `tests/` file keep this dependency-free.

**Why this choice**
The whole point of this batch is to replace guessed latency figures with
measured ones. A harness whose own math is verified by a mock-model self-test
lets us trust the table format and RTF column before spending minutes on real
CPU inference, and lets us re-run cheaply whenever clips change.

**Measured result**
`python -m profiling.benchmark_asr --self-test` → all checks pass (RTF math;
two rows produced and sorted shortest-first; `token_count` wired from the stub;
`rtf == infer/duration` for each row; clip duration read from WAV header;
table contains all six headers and both filenames). Real-clip numbers are
recorded in the Step 1c entry below. Demo regression unaffected (harness adds
no detection-path code).

---

## 2026-06-26 — Step 1c: real benchmark BLOCKED on this machine (OOM at model load)

**What was done**
Prepared the harness for a real run and attempted it:
- `profiling/benchmark_asr.py` now `import paths` first (before numpy/torch are
  pulled in via `profiling.asr`), so a real run uses the same BLAS/OpenMP
  thread caps and cache routing as the app — otherwise the benchmark would use
  all CPU cores and report a latency the app never delivers. `paths.py` uses
  `setdefault`, so an already-set `HF_HOME` (the on-disk model) is preserved.
- Added `benchmark_clips/` to `.gitignore` (keep the folder via `.gitkeep`,
  ignore its contents — synthetic throwaway clips and, later, real voice data).
- Generated three synthetic faint-noise clips (3 s / 8 s / 15 s, 16 kHz mono)
  as agreed (synthetic dry-run; token counts not meaningful, latency is).

The real run **could not complete on this machine**: every attempt to load the
CrisperWhisper weights crashes with a native **segmentation fault (exit 139)**,
in both venvs (Python 3.13 `venv` and 3.14 `.venv`), via `transformers.pipeline`,
plain `AutoModelForSpeechSeq2Seq.from_pretrained`, and even `safetensors`'
`safe_open(..., framework="pt").get_tensor(...)`.

**Diagnosis (what was ruled in/out)**
- Not a code bug in this repo: it reproduces on a stock `from_pretrained`.
- Not a corrupt download: the file is the expected size (3,219,908,024 B ≈
  3.2 GB, whisper-large-v3 fp16); the safetensors **header reads fine** (1260
  tensors) and reading tensors via the **numpy** framework works. Only the
  **torch** materialization path crashes.
- Not a broken torch/safetensors install per se: `torch.zeros(fp16)`,
  `torch.from_numpy(fp16)`, and a small `safetensors.torch` save/load round-trip
  all succeed.
- **Most likely cause: out of memory.** The machine has 7.8 GB total RAM with
  ~2.2 GB free at test time; a plain CPU `from_pretrained` materializes weights
  in fp32 (~6.4 GB), and even an fp16-resident load is ~3.2 GB — both exceed
  free RAM. On Windows a failed native allocation during weight load surfaces
  as SIGSEGV rather than a clean `MemoryError`, which matches the symptom (crash
  while materializing the *first* tensor on the torch path).

**Alternatives considered**
- Force a minimal-footprint load (`torch_dtype=float16`, `low_cpu_mem_usage`)
  on this machine. Not pursued: 3.2 GB resident still exceeds ~2.2 GB free, and
  the owner opted to run the measurement on a 16 GB device instead.
- Fabricate/estimate the latency numbers to fill the docs. Explicitly rejected
  — this batch is measurement-first; unverified figures are exactly what it's
  meant to remove.

**Why this choice**
The harness is correct and self-test-verified; the only missing ingredient is
hardware that can hold the model. Rather than guess, the real run is deferred
to a 16 GB machine. **To produce the table there:**
```
# (model auto-downloads to the project .cache/hf on first run unless HF_HOME is set)
python -m profiling.benchmark_asr --clips-dir benchmark_clips    # synthetic clips, or
# drop ~3s/8s/15s real recordings into benchmark_clips/ and run the same command
```
Then paste the table back and Step 3 (load-vs-inference, token-budget check)
and the doc latency-number updates can be completed against real figures.

**Measured result**
No latency table yet — model load OOM/segfaults at ~2.2 GB free RAM (needs
3.2–6.4 GB). Harness + synthetic clips are ready; real numbers pending a
16 GB run. Demo-fixture regression still **9 tokens / 7 disfluencies**
(unaffected — detection path untouched).

---

## 2026-06-26 — Docs: fix stale OpenVINO comment (latency numbers deferred)

**What was done**
Corrected a stale comment in `CrisperWhisperASR.__init__`
([profiling/asr.py](profiling/asr.py)) that claimed *"openvino is now the
default fast path instead… so there's no separate tokenizer implementation to
disagree with the model."* This directly contradicted the code: the default
backend is `transformers`, and `_transcribe_openvino()` raises immediately
(optimum-intel issue #561 — no `cross_attentions` for word timestamps). The
comment now states the real situation. No behaviour change (comment only).

The **latency-number** updates to `README.md`/`ARCHITECTURE.md`/`asr.py`
(reconciling the three conflicting figures — `~650-680s`, `~47-50s`, `~30-50s`)
are **deferred**: they must reflect the real benchmark, which is blocked on this
machine (see Step 1c) and will run on a 16 GB device. Replacing them now would
just swap one unverified figure for another.

**Alternatives considered**
- Also rewrite the latency figures now using the historical `~47-50s` from
  ARCHITECTURE.md §3. Rejected: that number is itself unverified against the
  current torch/transformers versions and clip-length scaling; this batch is
  measurement-first.
- Leave the OpenVINO comment as-is until the 16 GB run. Rejected: it's a pure
  code/comment contradiction (flagged during orientation) and fixing it needs
  no measurement.

**Why this choice**
Fix what can be verified from the code now (the contradiction); defer what
requires hardware we don't have (the numbers). Keeps the docs honest without
guessing.

**Measured result**
`python tests/test_asr_timing.py` → 3/3 pass; `asr.py` parses; demo regression
**9 tokens / 7 disfluencies**. Comment-only change, no measurable behaviour
delta.

---

## 2026-06-27 — Step 1c (real run) + Step 3: measured latency & profiling

**What was done**
The benchmark was run on a 16 GB CPU machine (`venv`, Python 3.13, transformers
backend) over four real recordings, and the README/ARCHITECTURE/asr.py latency
claims were updated to match. Measured:

| Clip | Duration | Load | Inference | RTF | Tokens | `max_new_tokens` budget | % of budget |
|---|---|---|---|---|---|---|---|
| 4sec  | 4.09 s  | 28.57 s | 54.00 s  | 13.19× | 7  | 44  | 16% |
| 8sec  | 8.53 s  | 0.00 s  | 81.04 s  | 9.50×  | 17 | 71  | 24% |
| 15sec | 15.53 s | 0.00 s  | 94.26 s  | 6.07×  | 26 | 113 | 23% |
| 19sec | 19.71 s | 0.00 s  | 102.25 s | 5.19×  | 41 | 138 | 30% |

**Step 3a — load-bound vs inference-bound.** **Inference-bound.** Model load is
a one-time **28.57 s** on the first clip and **0.00 s** on every clip after —
the `st.cache_resource` / `self._pipe` caching works exactly as intended
(confirmed by the benchmark's warm-load row). The recurring cost the user feels
is inference (54–102 s), not loading.

**Correction to prior docs:** inference is **not** a fixed ~30 s-window cost
"regardless of clip length." It scales: ~54 s (4 s clip) → ~102 s (20 s clip),
fitting roughly **~44 s fixed encoder + ~1.4 s per generated word** on CPU
(7→41 tokens added ~48 s). RTF is therefore *worse* for short clips (13×) than
long ones (5×) — the fixed encoder cost is amortized over less audio. The old
asr.py docstring ("~650-680 s regardless of clip length"; "decode loop barely
matters") was a stale pre-transformers-4.47 figure and a wrong inference about
where the cost lives; both fixed.

**Step 3b — is the 6 tok/s `max_new_tokens` budget over-budgeted?** **No — the
prior "reasonable, not over-budgeted" conclusion holds, with a sharper reason.**
Real speech ran at **~1.7–2.1 tokens/sec**, so actual token counts (7/17/26/41)
used only **16–30%** of the `int(dur*6)+20` budget (44/71/113/138). The cap is
**never the binding constraint** — the model hits EOS and stops well below it,
so the budget does not inflate normal-clip latency at all; it's purely a
runaway-generation safety ceiling. Lowering it (e.g. to 3 tok/s) would *not*
speed up normal transcription (clips already stop early); it would only truncate
a pathological no-EOS run sooner. Left at 6 tok/s.

**Doc updates made**
- `README.md` "Live microphone recording": replaced "~30-50s for a short clip"
  with the measured table + one-time load + scaling note.
- `ARCHITECTURE.md`: new "Measured latency (2026-06-26 …)" subsection in §3
  (table + the two corrections); updated the `transformers` backend row and the
  two §7 inline `~47-50s` figures. The historical dev-timing table is kept and
  explicitly labelled as superseded by the measured block.
- `profiling/asr.py`: header note (RTF 5-13× not 2-8×; inference scales, not
  fixed; old 650-680s figure marked pre-4.47) and the dormant faster_whisper
  docstring's stale 650-680s reference.

**Alternatives considered**
- Lower `max_new_tokens` to claw back latency. Rejected: the cap isn't binding
  on normal clips (3b), so there's nothing to claw back.
- Keep the synthetic-clip plan. Superseded: the owner ran four real recordings
  on a 16 GB device, so the table uses real token counts, not synthetic ones.

**Why this choice**
Numbers now come from a real run on representative hardware; the docs state what
the code actually does today, and the two stale assumptions (fixed-cost encoder;
2-8× RTF) are corrected at the source.

**Measured result**
See the table above. Cache confirmed (warm load 0.00 s). After the doc edits:
tests 3/3 pass, benchmark `--self-test` passes, demo regression **9 tokens /
7 disfluencies**.

---

## 2026-06-27 — §1 Option A: acoustic cross-validation of word timestamps

**What was done**
Implemented the lighter of the two §1 options (chosen over Option B per owner's
"whatever you think is better" go-ahead, because it's a real accuracy fix that's
locally testable without the 3.2 GB model and builds the energy-envelope
primitives Option B / a future realtime acoustic detector will reuse).
- `profiling/detect.py`: added `_AcousticContext.voiced_span()` and
  `voiced_duration()` — frame-wise RMS trimming of leading/trailing silence
  (edges only, so a mid-word energy dip doesn't shorten a sustained sound). A
  new `_effective_duration()` in `detect_disfluencies` uses the voiced duration
  when audio is available (else the raw timestamp duration), and feeds it to
  **both** the 90th-percentile threshold and the per-word prolongation check.
  Flagged prolongations now carry a `voiced_duration` field and say "voiced
  duration …" in their evidence.
- `tests/test_detect_acoustic.py`: 5 tests (no ASR model — WAV bytes built from
  silence + a 150 Hz tone).
- `ARCHITECTURE.md` §4: documented the fix as verified behaviour, with the
  audio-required caveat.

**The bug being fixed** (from the deleted `improve.md`): the ASR anchors a
word's `start` to the chunk boundary, so clip-initial silence is billed to the
first word. That (1) makes the first word look prolonged, and (2) — the subtler,
more damaging half — inflates the clip-wide 90th-percentile prolongation
threshold, so genuine prolongations *elsewhere* get suppressed.

**Alternatives considered**
- Option B (a parallel waveform-native detector module). Deferred: bigger, and
  Option A's voiced-region work is its prerequisite anyway.
- Only suppress the per-word false positive (leave the percentile alone).
  Rejected: the test shows the percentile poisoning is the half that silently
  hurts recall elsewhere; fixing both is the point.
- Trim inside `word_is_prolonged` only. Rejected: the threshold is computed from
  *all* tokens' durations, so the trim has to happen at the duration source to
  reach the percentile too.

**Why this choice**
Highest-value, lowest-risk, fully testable here, and it advances the realtime
goal indirectly (the energy-envelope/voiced-region code is the first brick of an
acoustic-native detector). No behaviour change without audio, so fixtures and
timestamp-only paths are untouched.

**Measured result**
`python tests/test_detect_acoustic.py` → 5/5 pass:
- `voiced_duration(0..1.5)` over 1.0 s silence + 0.5 s tone ≈ 0.5 s; fully-silent
  span ≈ 0; no-audio → None.
- Silence-padded first word NOT flagged; genuine sustained word IS flagged
  (`voiced_duration` ≈ 1.20 s).
- With audio the real prolongation is recovered; **without** audio the same clip's
  percentile is poisoned by the raw 1.38 s and the real one is missed — the
  contrast that demonstrates the fix.
Regression intact: `tests/test_asr_timing.py` 3/3, demo fixture **9 tokens /
7 disfluencies** (no-audio path unchanged).

---

## 2026-06-27 — Realtime foundation: ASR-independent acoustic detection (Option B, step 1)

**What was done**
Added `profiling/acoustic.py` — a standalone, pure-NumPy module that derives
disfluency cues straight from a waveform, with no ASR and no model:
- frame-level RMS/ZCR features (`frame_features`),
- voiced/silent segmentation (`segment_voiced`),
- **prolongation candidates** (long, energetic, low-ZCR voiced segments) and
  **block candidates** (long silences *flanked by* voiced segments, so
  leading/trailing dead air isn't mistaken for a block),
- `analyze(wav_bytes_or_array, config)` returning `AcousticAnalysis` with
  time-ordered, serializable `Candidate`s; `AcousticConfig.from_detection_cfg`
  reuses the same `profiling.detection` thresholds as the token detector.
`tests/test_acoustic.py`: 8 tests (synthetic silence + 150 Hz tone).

**Why this is the right next step for the dual goal**
- *Realtime:* the benchmark proved transcription is inference-bound at ~5-13×
  real time, so a realtime path cannot wait on ASR. Acoustic cues can be
  computed on an audio *stream* with no model; the segmentation is windowable,
  so this is the brick a sliding-window/streaming detector is built from.
- *Research/accuracy:* it can catch disfluencies the ASR smooths away (sub-word
  prolongations, silent blocks that never became a token) — the Option B / Tier 2
  idea from the deleted `improve.md`.
- It reuses the voiced-region thinking introduced in §1 Option A, at the level of
  the whole waveform rather than a single ASR word span.

**Deliberately NOT done yet**
Not wired into `detect_disfluencies`. Merging acoustic candidates with the
token-based detector (dedupe/reconcile, confidence fusion) changes live output
and should be validated against **real stutter recordings** (e.g. on the 16 GB
device, or SEP-28k/FluencyBank), not just synthetic tones — so it's a separate,
deliberate step. Keeping this purely additive means zero regression risk now
(demo fixture still 9/7; nothing imports `acoustic.py` yet).

**Alternatives considered**
- Wire it into the detector immediately. Rejected: can't validate fusion quality
  on synthetic audio alone; premature live behaviour change.
- Improve phonetic near-repetition instead. Deferred: valuable but nuanced to
  validate without labelled data, and it doesn't advance the realtime goal.

**Measured result**
`python tests/test_acoustic.py` → 8/8 pass (segmentation boundaries within a
frame of truth; 1.2 s tone → 1 prolongation; 0.3 s tone → none; voiced-silence-
voiced → 1 block; edge silences → no block; all-silence → nothing;
WAV-bytes end-to-end → 2 prolongations + 1 block, ordered & serializable).
Full suite: acoustic 8/8, detect-acoustic 5/5, asr-timing 3/3, benchmark
self-test pass, demo fixture **9 tokens / 7 disfluencies**.

---

## 2026-06-27 — Fuse acoustic cues into the live detector (Option B, step 2)

**What was done** (chosen via the owner's "fuse acoustic into live detector"
answer.) `detect_disfluencies` now, **when audio is available**, runs
`profiling/acoustic.py` over the same waveform and merges its prolongation/block
candidates with the token-based events:
- **Dedupe:** an acoustic candidate that overlaps an already-flagged event of the
  same type is dropped — the token path wins, no double counting.
- **Attribution:** a kept candidate is mapped to a token via
  `_token_index_for_span` (max temporal overlap; else the word starting after the
  region — for a silent block; else nearest by midpoint), so the event carries a
  word/onset and flows into the profile like any other.
- Acoustic-sourced events are tagged `source="acoustic"` with
  `acoustic_start`/`acoustic_end` and an `"[acoustic] …"` evidence string.
- Calibrated floors are honoured: the fused `AcousticConfig` uses the same
  personalized `prolong_min`/`block_gap` the token path uses this run.
`tests/test_detect_fusion.py`: 3 tests.

**Why**
This is the immediate accuracy payoff of the acoustic module: it catches
sustained sounds and blocks the token path can't — e.g. a sustain that lands in a
gap with no token of its own, or one the ASR's word timestamps under-shot. It
keeps the detector a single signal (one event list, deduped) rather than two
parallel outputs the UI would have to reconcile.

**Guardrails / what to validate next**
- **Zero change without audio:** the whole block is under `if ac.available`, so
  fixtures and timestamp-only clips are byte-for-byte identical (demo still
  9 tokens / 7 disfluencies; `source="acoustic"` never appears there).
- **Needs real-audio tuning:** the dedupe-by-overlap and the gap→following-word
  attribution are reasonable defaults validated on synthetic tones only. On real
  recordings, watch for (a) acoustic false positives on noisy/voiced non-speech,
  and (b) attribution landing on the "wrong" neighbouring word. Both are
  threshold/heuristic tweaks, not structural — flagged for the 16 GB real-audio
  pass.

**Alternatives considered**
- Keep acoustic candidates as a separate list surfaced beside the events.
  Rejected: two overlapping signals are harder to read and to feed the profile;
  fusion with dedupe is cleaner.
- Confidence fusion (boost a token event that an acoustic candidate confirms,
  rather than just dropping the duplicate). Deferred: adds a tuning knob better
  set against real data; current behaviour is the conservative "don't double
  count."

**Measured result**
`python tests/test_detect_fusion.py` → 3/3: acoustic catches a 1.4 s sustain in a
token-less gap (attributed to the following word, `source="acoustic"`); an
overlapping token-flagged prolongation is **not** double-counted (one event,
token-sourced, carries `voiced_duration`); no audio → demo stays 7 events with no
acoustic source. Full suite: acoustic 8/8, detect-acoustic 5/5, detect-fusion
3/3, asr-timing 3/3, benchmark self-test pass, demo **9 tokens / 7 disfluencies**.

---

## 2026-06-27 — Quality: generalized phrase-repetition (any length)

**What was done**
The phrase-repetition pre-pass in `detect_disfluencies` checked only 2- and
3-word windows, so longer immediate repeats ("I want to I want to", "please pass
the salt please pass the salt") fell through silently. It now scans windows from
`phrase_repetition_min_words` up to `phrase_repetition_max_words` (new config
key, default 8) — also capped at `len(tokens)//2` to bound the scan — and records
the phrase length so the evidence reads e.g. "4-word phrase repeated starting at
token 4". Longest match wins per start index; `add()` still dedupes by
(index, type). Added `phrase_repetition_max_words` to `config.yaml`, marked the
ARCHITECTURE §4 limitation fixed, and added `tests/test_detect_phrase.py`.

**Alternatives considered**
- Keep the 2-3 cap. Rejected: a listed limitation, and longer repeats are a real
  stuttering/cluttering pattern.
- Unbounded window. Rejected: O(n²) on long transcripts; an 8-word ceiling (and
  the structural `len//2` bound) covers realistic repeats cheaply.

**Why this choice**
Pure recall win for a documented gap, fully testable model-free, zero risk to the
audio path or fixtures (text-only change; demo unchanged).

**Measured result**
`python tests/test_detect_phrase.py` → 4/4: 2-word and (newly) 4-word repeats
flagged at the 2nd occurrence with correct length in the evidence; a non-repeat
sentence yields no phrase event; demo fixture still 7 events. Regression sweep:
detect-acoustic 5/5, detect-fusion 3/3, acoustic 8/8, demo **9 tokens /
7 disfluencies**.

---

## 2026-06-27 — Quality: phonetic near-repetition for short words

**What was done**
Near-repetition compared consecutive words by spelling edit distance only, which
is unreliable for short words (one changed letter is a huge fraction of a 2-3
letter word). Added `phonetic.phonemes()` (full ARPAbet phone sequence, stress
stripped, via CMU; `()` for OOV) and `_phonetic_similarity()` in detect.py
(reuses `_edit_distance`, which works on phoneme tuples). The near-repetition
branch now uses phonetic similarity when both words are ≤ `phonetic_short_max_`
`chars` (new config key, default 4) and both are in CMU; otherwise it keeps the
spelling metric. The evidence names the metric used. Added the config key,
updated ARCHITECTURE §4, and `tests/test_detect_phonetic.py`.

**Alternatives considered**
- Use phonetic for *all* word lengths (or max of phonetic/spelling). Rejected:
  broader behaviour change and more homophone false positives; the short-word
  scope is where spelling is genuinely worst and matches the §4 recommendation.
- Require a shared phonetic onset as an extra guard. Considered but not added:
  it removes some legitimate matches and is better tuned against real data.
- Skip this fix because of the homophone risk. Rejected: it's the same
  false-positive class the spelling metric already carried (look-alike words),
  just shifted to sound-alikes, and it's behind a threshold + a short-word cap +
  tunable config — documented honestly for real-audio tuning.

**Why this choice**
Bounded, config-gated, fully model-free-testable improvement to a documented
limitation, with zero change to the audio path or fixtures.

**Known trade-off (flagged for real-audio tuning)**
Consecutive short homophones used legitimately ("to"/"too", "no"/"know") can now
be flagged as near-repetitions. Mitigated by the ≤4-char cap and the 0.75
threshold; revisit `phonetic_short_max_chars` once there's labelled data.

**Measured result**
`python tests/test_detect_phonetic.py` → 5/5: `_phonetic_similarity` gives 1.0 for
no/know and be/bee, <0.5 for cat/dog, None for OOV; "no"→"know" is flagged with
"phonetic similarity" in the evidence (edit distance alone = 0.5, below
threshold); long "walking"/"walkin" still uses the "edit" metric; dissimilar OOV
shorts aren't flagged. **Full suite: 28 tests pass** (asr-timing 3, acoustic 8,
detect-acoustic 5, detect-fusion 3, detect-phrase 4, detect-phonetic 5),
benchmark self-test pass, demo fixture **9 tokens / 7 disfluencies**.

---

## 2026-06-27 — Real-audio validation + prolongation threshold tune (Part D)

**What was validated** (owner ran the app on the 16 GB device with real mic
recordings — first real-audio test of the fusion + phonetic work):
- ✅ Block on an intentional stuck start ("THIS P…PLACE") → flagged acoustically
  (`[acoustic] silent gap 1.32s`).
- ✅ Homophone check "I want to go **too**" → **no** false repetition (the phonetic
  near-rep change did not misfire on to/too).
- ✅ Exaggerated prolongations ("I…", "want…", "water…") all caught (1.24–1.42 s).
- ⚠️ **False-positive prolongations on fluent speech:** "early" 0.69 s, "already"
  0.69 s, "Later" 0.76 s, "my" 0.84 s, "going" 0.69 s — naturally-emphasised
  vowels flagged as prolongations (most acoustic-sourced; "Later" was the token
  path). Noise itself didn't hurt transcription.
- ⚠️ **Long-recording transcription truncates** (separate issue, see below).

**Decision: raise `prolongation_min_seconds` 0.65 → 1.0.**
The data separates cleanly: every false positive was ≤ 0.84 s, every intended
prolongation ≥ 1.24 s. 1.0 s sits in the middle (0.16 s above the worst FP,
0.24 s below the mildest real one). The acoustic detector's floor *is*
`prolongation_min_seconds` (set from the config/calibrated value), so this one
knob suppresses both the acoustic and token-path FPs at once. In short clips
(<5 tokens) the token path's `prolong_min*1.5` fallback becomes 1.5 s, so the
acoustic floor (1.0 s) becomes the effective detector there and still catches the
1.24 s+ real prolongations.

**Alternatives considered**
- 0.9 s: also clears the observed FPs but only 0.06 s above the worst one — too
  thin a margin against the next emphatic vowel. 1.0 s is more robust; lower it
  toward 0.85–0.9 if real prolongations start getting missed.
- A relative/percentile guard inside the acoustic detector (instead of a flat
  floor). Deferred: more code; the floor bump fixes every observed FP now and the
  percentile already governs the token path on longer clips.

**Why this choice**
Pure config, directly derived from the owner's measured numbers, fixes all
observed prolongation false positives while keeping every intended detection;
demo and tests unaffected.

**Measured result**
Demo fixture still **9 tokens / 7 disfluencies** (its 1.29 s "something"
prolongation still clears the new floor); full suite 28/28 pass. Real-audio
re-validation (owner) pending.

**Open issue — long-recording ASR truncation (NOT yet fixed)**
On a long fluent passage the recording completed but the transcript stopped early
(omitted later words); inference ran ~116 s. Consistent with the "Whisper did not
predict an ending timestamp … audio cut off" warnings: most likely the
`max_new_tokens` ceiling (256 in `_max_new_tokens_for`) and/or long-form
word-timestamp chunking. Needs investigation on a machine that can load the model
(can't be reproduced on the 2.2 GB dev box). Tracked as the next task.

---

## 2026-08-03 — Vision alignment review + architecture decision

**What was done**
The project's owner restated the mission explicitly: audio-mod's primary
purpose is to detect and localize speech disfluencies *from the audio
signal itself*; verbatim transcription is scaffolding for that, not the end
goal. The full codebase was re-read against that framing (every file in
`profiling/` plus `app.py`, `auth.py`, `paths.py`, etc.), and a literature/
dataset/model review was conducted: ~25 papers and 6 datasets (SEP-28k,
FluencyBank Timestamped, LibriStutter, KSoF, UCLASS, plus AS-70/Boli for
multilingual context) covering clip-level classifiers (StutterNet,
wav2vec2-embedding classifiers, multi-task/adversarial learning), region/
word-level localization architectures (YOLO-Stutter, Stutter-Solver,
Self-supervised WavLM word-level detection, Dysfluent WFST, SSDM/SSDM 2.0),
a direct comparative study across four representative architectures, a
comprehensive analysis of rule-based/interpretable systems, and the
end-user-alignment literature (stakeholder-need surveys for stuttering
research broadly).

**Finding**: the existing detector (`profiling/detect.py`) was
architecturally transcript-first with acoustic confirmation bolted on, not
audio-first. Filler and stutter-marker events trusted ASR flags with zero
audio grounding; all repetition detection was pure text comparison; block/
prolongation only used audio as a post-hoc veto on ASR-derived boundaries,
never as the originating signal; an acoustic candidate overlapping a
token-path event was always dropped regardless of which was more confident.
This directly inverted the restated mission.

**Alternatives considered**
- Replace the two-stage (ASR + detector) pipeline with an end-to-end
  audio→dysfluency-region model (YOLO-Stutter/Stutter-Solver/SSDM-class).
  **Rejected**: these architectures still require a speech-text alignment as
  input — they don't eliminate the ASR stage, only replace the detector with
  something heavier. A 2025 comparative study (arXiv:2509.00058) directly
  benchmarking YOLO-Stutter, FluentNet, UDM, and SSDM found the simplest/most
  interpretable approach (UDM) had the best accuracy/interpretability
  balance, while SSDM — the most complex — **could not be reproduced by the
  paper's own independent authors**. Given this project has no GPU/training
  pipeline and is a solo/small-team effort, adopting the field's most complex
  architecture against evidence it isn't even independently reproducible was
  judged an unjustifiable risk.
- Discard the rule-based/acoustic detection tier entirely in favor of a
  learned model. **Rejected**: "Revisiting Rule-Based Stuttering Detection"
  (arXiv:2508.16681, 2025) found rule-based systems remain near-SOTA
  specifically for prolongation detection (97–99% reported accuracy) and
  recommended enhancement (richer acoustic features, hierarchical decision
  structure), not replacement.
- Replace CrisperWhisper. **Rejected**: its own paper (arXiv:2408.16589)
  confirms it as the right tool specifically for verbatim transcription +
  accurate word timestamps, which is exactly the role it plays here; no
  evidence found suggests a better choice for that sub-task.

**Why this choice**
Keep the two-stage design (validated for the ASR stage specifically), but
restructure Stage 2 to be audio-native-primary rather than
transcript-first-with-confirmation — enhance the existing rule/acoustic tier
(more features, corroboration, weighted fusion) using only pretrained/
zero-training components, since no training pipeline exists yet and the
comparative-study evidence favors interpretable-and-simpler over
complex-and-unreproducible for a project at this stage. A learned classifier
tier (frozen WavLM/wav2vec2) is identified as the clear future step, not
this round's work — see `ROADMAP.md`.

**Measured result**
Not applicable — this entry is the analysis and decision; see the next entry
for what was implemented and its verification.

---

## 2026-08-03 — Audio-native-primary detector restructuring

**What was done**
Implementing the decision above:
- `profiling/detect.py`: taxonomy split — the generic `repetition` type
  became `sound_repetition` (sub-word fragment repeats), `word_repetition`
  (exact/near/filler-sandwiched whole-word repeats), `phrase_repetition`
  (multi-word repeats) — matching the SEP-28k/FluencyBank/KSoF standard
  5-class taxonomy. Acoustic corroboration added for `filler` and
  `stutter_marker` (a voiced-energy presence check via the existing
  `_AcousticContext.word_rms` primitive — previously these had zero audio
  grounding). Token-vs-acoustic fusion changed from fixed priority ("token
  path always wins on overlap") to weighted-confidence: the acoustic
  candidate replaces the token-path event only when its
  (`fusion_weights.acoustic`-scaled) confidence is strictly higher; ties
  keep the token-path event deliberately (it carries word-level grounding an
  audio-only candidate doesn't have). Config-driven `detectors` enable-list
  added.
- `profiling/acoustic.py`: enriched with Silero VAD and Praat/Parselmouth
  (pitch, jitter, shimmer, HNR) as corroborating evidence for prolongation
  detection.
- New `profiling/evaluate.py` (v1): the first accuracy harness for this
  detector, timestamp-only, LibriStutter-schema-compatible, with a bundled
  synthetic self-test sample.
- `config.yaml` / `profiling/config.py`: new `detectors`, `fusion_weights`,
  and expanded `acoustic.*` keys, documented in both the YAML and the Python
  hardcoded-fallback defaults.
- `app.py`, `README.md`, `ARCHITECTURE.md` updated to match.

**Alternatives considered**
- Hard-replace RMS/ZCR voiced-segment classification with Silero VAD
  outright. **Rejected after direct testing**: Silero VAD is trained on real
  speech and was confirmed (`get_speech_timestamps` on a pure 150 Hz sine
  tone) to return zero speech regions on this project's entire synthetic-tone
  test suite — a hard swap would have silently broken every existing
  synthetic-audio test. Implemented instead as a corroboration signal that
  self-disables (opts out entirely) on any clip where VAD finds no speech
  anywhere, preserving the project's model-free testing philosophy while
  still gating real recordings against a validated model instead of one
  hand-picked constant.
- Build an elaborate offset-shape/abrupt-energy-drop acoustic check for
  `stutter_marker` specifically (a cut-off fragment's characteristic energy
  profile). **Deferred**: no real recordings existed yet to validate such a
  heuristic against; shipped the simpler voiced-energy-presence check instead
  (still closes the "zero audio grounding" gap) and flagged the more
  elaborate version as future work (`ROADMAP.md`) pending real validation
  data.

**Why this choice**
Closes the audio-grounding gap identified in the prior entry for the three
event families that had none (filler, stutter_marker, and — via the
acoustic tier finally being co-equal rather than subordinate — block/
prolongation), while remaining fully buildable with pretrained/zero-training
components on CPU-only, no-GPU hardware.

**Measured result**
Full test suite: 38/38 passing (28 pre-existing tests + 10 new tests in
`tests/test_detect_taxonomy_and_fusion.py` covering the taxonomy split,
filler/stutter_marker acoustic corroboration in both directions, the
weighted-fusion replacement path — forced deterministically via
`fusion_weights.acoustic=10.0` rather than relying on a naturally-occurring
crossover — and the detector enable-list). Only 2 of 28 pre-existing tests
needed a deliberate update (their type-filter helpers, `"repetition"` →
`"word_repetition"`/`"phrase_repetition"`); every other pre-existing test,
including the demo fixture's exact 9-token/7-event count, passed unmodified.
`python -m profiling.benchmark_asr --self-test` and
`python -m profiling.evaluate --self-test` both pass. The demo fixture run
through the real `config.yaml` (not a test-provided config dict) reproduces
9 tokens / 7 events with `word_repetition ×2` in place of the old
`repetition ×2`.

---

## 2026-08-03 — Real-audio validation pass + event-table display fix

**What was done**
Two real microphone recordings were tested end-to-end by the project owner
against the restructured detector (a self-introduction sentence, 31 tokens;
"I went to the shop and bought a banana," 9 tokens, with deliberate word
repetition). Reviewing the results surfaced:
1. **A genuine, pre-existing UI bug** (predates this round's restructuring):
   `app.py`'s Event table showed, for acoustic-sourced events, the full
   nominal span of whichever word the event got attributed to
   (`rows[index].start/end`) rather than the actual detected region
   (`acoustic_start`/`acoustic_end`, computed but never rendered). Confirmed
   by direct reproduction: one case showed a displayed span that barely
   overlapped the true detected region at all. Fixed: the Event table now
   prefers `acoustic_start`/`acoustic_end` when present.
2. **A missed word repetition** — the speaker repeated "went" and "shop"
   deliberately, but the transcript (9 tokens) contained no repeated words at
   all, meaning CrisperWhisper's own transcription smoothed the repetition
   away before the (purely text-based) repetition detector ever saw it.
3. **Four "block" flags the speaker felt were just normal pauses**, while
   the one disfluency they self-reported (stuttering on the word
   "stuttering") wasn't flagged as anything.

**Alternatives considered**
- Retune `block_gap_seconds` (or add a similar quick threshold change) in
  response to finding #3. **Rejected**: a single 2-clip, one-speaker,
  no-ground-truth anecdotal test cannot distinguish a genuinely oversensitive
  threshold from normal speech variation the speaker didn't consciously
  register as a pause — exactly the ambiguity `VALIDATION.md` exists to
  resolve with actual labeled data. Threshold changes based on this session's
  anecdotal evidence were explicitly deferred to after real evaluation
  numbers exist.
- Attempt an acoustic-native repetition detector as a quick fix for finding
  #2. **Rejected as "quick"**: real acoustic-native repetition detection
  (recognizing that two acoustically similar segments were produced, without
  relying on the transcript agreeing they're the same word) is a nontrivial,
  currently-unbuilt capability, not a small patch — logged as a `ROADMAP.md`
  item instead of rushed.

**Why this choice**
Fix what's unambiguously a bug (display, #1) immediately since it's small,
low-risk, and purely presentational; explicitly do not act on findings #2/#3
without ground truth, per the project's own measurement-first convention
(established in this file's earlier entries) and the owner's explicit
instruction to stop tuning based on anecdotal single-speaker tests.

**Measured result**
Fix #1 verified directly: reproducing the fusion test scenario from
`tests/test_detect_fusion.py` showed the old logic would have displayed
Start=2.40/End=2.80 (the attributed word "now"'s own tiny span) for an event
whose actual detected sustained region was 0.58–2.00s — a materially
different, non-overlapping-in-spirit span. After the fix, the displayed span
matches `acoustic_start`/`acoustic_end` exactly. Full 38/38 test suite still
passes (the fix is `app.py`-only, doesn't touch `detect.py`'s data model).
`app.py` parses cleanly (`ast.parse`).

---

## 2026-08-03 — Evaluation methodology research (VALIDATION.md)

**What was done**
Following the real-audio validation pass above, and per the owner's explicit
request to establish rigorous, reproducible, dataset-based evaluation before
any further detection-algorithm changes, researched and documented (in the
new `VALIDATION.md`) a complete evaluation methodology: dataset comparison
and prioritization (LibriStutter Tier 1, SEP-28k Tier 2, KSoF/UCLASS Tier 3),
a two-track evaluation design (Track A: detector-only against ground-truth
transcripts; Track B: full pipeline including this project's own ASR, with
hypothesis-to-reference alignment), metrics (per-type precision/recall/F1,
per-type binary confusion matrices — not a single multi-class matrix, per the
field's multi-label literature — IoU≥0.5 localization accuracy, EER,
speaker-exclusive split discipline), required preprocessing/alignment work,
a proposed `profiling/evaluation/` package redesign, and honestly-stated
limitations (no dataset covers the full 7-type taxonomy; SEP-28k audio
acquisition is fragile; annotator disagreement sets a non-1.0 ceiling; the
Track B alignment step is itself a source of error requiring hand-validation).

**Alternatives considered**
- Treat the existing `profiling/evaluate.py` (built the same day, prior
  entry) as sufficient. **Rejected**: it validates its own scoring math and
  runs a synthetic smoke-test, but has never touched real labeled data, has
  no localization or confusion-matrix metrics, and only implements one of
  the two meaningfully different evaluation modes (Track A). Explicitly
  judged insufficient for "scientifically rigorous, reproducible, and
  comparable with published research" per the owner's stated goal.
- Build the evaluation package immediately. **Not done**: the owner
  explicitly requested a plan first, with implementation deferred until
  after committing the current state as a baseline — see the sequencing
  question asked and answered (wait for go-ahead) at the end of this session.

**Why this choice**
A rigorous evaluation methodology needs to be agreed and recorded *before*
implementation, not discovered ad hoc while writing code — and the owner's
own reasoning (an anecdotal 2-clip test can't disambiguate detector error
from natural speech variation) is the direct, concrete justification for why
this matters now rather than later.

**Measured result**
Not applicable — this entry is a research/planning step; no code was
written. See `VALIDATION.md` §8 for the (currently empty, templated) results
this methodology is meant to produce once implemented.

---

## 2026-08-03 — Documentation architecture established

**What was done**
Given the project's stated intent to grow into a research-quality system
supporting an eventual paper, restructured the project's documentation set:
added `DOCS.md` (a map explaining every file's purpose, audience, and update
cadence), `VALIDATION.md` (methodology from the prior entry, plus living
Results/Ablations/Benchmark-comparison sections — currently templated,
awaiting real runs), `ROADMAP.md` (a single consolidated forward-looking
priority list), and `CHANGELOG.md` (a terse, reverse-chronological index
reconstructed from this file's full history, for fast scanning without
reading every entry's full reasoning). This file's entries above were added
retroactively for this session's work. `august.md` (the working notes from
this session, written before this restructuring) was retired to a short stub
— its content is now split between this file (reasoning) and `VALIDATION.md`
(methodology).

**Alternatives considered**
- Keep `august.md` as a standalone, permanent record of this session
  alongside this file. **Rejected**: this file already exists, predates
  `august.md`, was purpose-built (per its own header) for exactly this kind
  of record, and is explicitly append-only/chronological — maintaining two
  parallel narrative histories would inevitably drift out of sync. Since
  nothing in this session had been committed to git yet, there was no
  external reference to `august.md` to preserve, making this the right time
  to consolidate rather than after a commit made the file's URL/path
  load-bearing.
- Skip a dedicated `ROADMAP.md`/`CHANGELOG.md` and rely on this file alone.
  **Rejected**: this file is deliberately full-reasoning and append-only,
  which makes it the wrong shape for "what's the fast-scan summary" or
  "what's next" — those are genuinely different queries a reader (or a
  future paper draft) will make, and conflating them would make this file
  worse at its own actual job.

**Why this choice**
Matches the stated goal directly: documentation that lets someone understand
months from now exactly how the project evolved, why, what evidence
supported each change, and how to continue — without reconstructing any of
it from memory or git archaeology.

**Measured result**
Not applicable — documentation-only change. Self-review performed before
recommending a commit checkpoint (see `DOCS.md` and this session's final
message to the owner).

---

## 2026-08-03 — Building the profiling/evaluation package

**What was done**
Per the sequencing agreed in `VALIDATION.md` §6 (and confirmed with the
owner after the baseline commit landed as `c1b7c0a`), built steps 1–2 of the
evaluation package:
- `profiling/evaluation/` package: `loaders.py` (`LabeledClip` for word-level
  ground truth, `ClipLevelLabels` for clip-level ground truth — see below —,
  `load_libristutter_csv`/`load_libristutter_dir`,
  `synthetic_libristutter_sample`), `metrics.py` (`score_word_level` and
  `score_clip_level`, both now tracking TN as well as TP/FP/FN;
  `localization_rate` — IoU-based, preferring `acoustic_start`/
  `acoustic_end` when present, matching the display-fix entry above;
  `format_confusion_matrix` — per-type binary, not multi-class), `track_a.py`
  (the detector-only runner, `--self-test`/`--dataset`/`--data-dir` CLI),
  `report.py` (table rendering + timestamped, git-commit-tagged, non-
  clobbering JSON result files under the new gitignored `eval_results/`).
  `profiling/evaluate.py` (the 2026-08-03 v1) is now a thin shim re-exporting
  the package, with its CLI behavior preserved and its one intentional
  return-shape change (`evaluate()` now returns `(counts, localization)`)
  documented in its own docstring rather than silently changed.

**A genuine finding made while building this, not anticipated in the
original plan**: SEP-28k is clip-level labeled (3-second clips, a count out
of 3 annotators per disfluency type present "somewhere in the clip"), with
**no reference transcript or word-level timestamps anywhere in the
dataset** — confirmed directly against the dataset's own README before
writing any parser code. This means SEP-28k cannot drive
`detect_disfluencies()`'s token-based checks the way LibriStutter's
word-level labels can (there's no ground-truth word sequence to feed them);
it can only ever be scored at clip granularity, and only by something that
itself runs without a reference transcript (an acoustic-only detection pass,
or the not-yet-built Track B). `ClipLevelLabels`/`score_clip_level` were
added to the package specifically to have this shape ready, but
`load_sep28k_labels` itself was **not** written — the exact CSV column names
are still unconfirmed (the README describes what the columns mean, not
their literal header text), and this project's measurement-first convention
means not guessing a parser's schema.

**Alternatives considered**
- Force SEP-28k into the same `LabeledClip`/word-index-matching shape
  LibriStutter uses, e.g. by treating the whole clip as "one token."
  **Rejected**: this would silently misrepresent what's actually being
  measured (clip-level presence, not word-level localization) and would make
  `localization_rate` numbers for SEP-28k meaningless without a clear
  disclaimer — better to have two explicit, correctly-named scoring
  functions than one that quietly means different things per dataset.
- Guess SEP-28k_labels.csv's column names from the paper's prose description
  and write `load_sep28k_labels` now. **Rejected**: exactly the kind of
  unverified-schema code this project's own convention (see `DOCS.md`
  philosophy point 3, and the original `evaluate.py`'s LibriStutter loader
  docstring, which flagged the same category of risk) warns against. Wrong
  now, silently, is worse than admittedly-incomplete now.
- Also build Track B (`alignment.py`) this round, since the package was
  already open. **Deferred, not rejected**: real alignment work needs real
  ASR output to align against, which needs real audio, which is exactly the
  "check before large downloads" gate this entry's next step ran into
  anyway — no point building it before there's data to run it against.

**Why this choice**
Ship what's fully verified (steps 1–2, the scoring core) rather than
padding the package with plausible-looking but unverified pieces (a guessed
SEP-28k parser, an alignment module with nothing real to align yet).

**Measured result**
Full regression: 38/38 pre-existing tests still pass, plus
`python -m profiling.evaluation.track_a --self-test` (26 checks: scoring
math, confusion matrix rendering, IoU localization — including a check that
a poor-overlap prediction correctly fails the IoU≥0.5 threshold, and that
`acoustic_start`/`acoustic_end` is correctly preferred over the nominal
token span — CSV round-trip, end-to-end synthetic-sample scoring, and
result-file writing/non-clobbering) — all pass. `python -m profiling.evaluate
--self-test` and a live `python -m profiling.evaluate` / `python -m
profiling.evaluation.track_a` run both produce correct output through the
shim. One real bug caught and fixed during this work: `report.save_run`'s
original second-resolution timestamp caused two runs within the same second
to silently overwrite each other (caught by a direct test, not inspection —
fixed with microsecond-resolution timestamps). One console-encoding bug
caught and fixed: non-ASCII characters (`≥`, em dashes, `§`) in
`print()`-reachable strings crashed or mangled under this Windows machine's
`cp1252` terminal codepage — fixed by using plain ASCII in all actually-
printed strings (docstrings/comments, never printed, were left as prose).

---

## 2026-08-03 — Real LibriStutter/SEP-28k schemas confirmed; SEP-28k labels loader built

**What was done**
With the owner's go-ahead to proceed with real dataset acquisition, verified
both remaining Tier-1/2 dataset schemas directly against real downloaded
files rather than continuing to defer on guesses:

1. **LibriStutter's real annotation format was directly downloaded (GitHub
   mirror, `hhzhang16/LibriStutterData`) and inspected — the original
   `load_libristutter_csv` schema assumption was wrong.** Disfluencies are
   NOT a label on a real word's own row; every non-clean row instead has the
   literal placeholder word `"STUTTER"` sitting between two real words,
   describing a synthetically-inserted disfluency segment (e.g.
   `Rachel,1.8,2.1,0` / `STUTTER,2.1,3.08,3` / `Lynde,3.08,3.48,0`). Fixed
   `load_libristutter_csv` to reconstruct each `STUTTER` row into a
   plausible real token (adjacent word repeated for word/sound/phrase
   repetition and prolongation types; `"uh"` + `is_filler=True` for
   interjection) rather than feeding the literal word "STUTTER" into
   `detect_disfluencies()` — documented as an approximation, not a verified
   transcription, directly in the loader's docstring.
2. **SEP-28k's real column schema was confirmed** (`SEP-28k_labels.csv`,
   fetched directly): `Show,EpId,ClipId,Start,Stop,Unsure,
   PoorAudioQuality,Prolongation,Block,SoundRep,WordRep,
   DifficultToUnderstand,Interjection,NoStutteredWords,NaturalPause,Music,
   NoSpeech` — confirming the clip-level-with-no-transcript structure
   already anticipated (previous entry) and its `ClipLevelLabels` shape.
   `load_sep28k_labels()` written and run against the real, complete
   28,177-row file: parsed cleanly, matches the dataset's documented total
   clip count exactly, and produces a plausible per-type distribution
   (filler most common at 5,973 clips, sound_repetition least at 2,342, at
   a >=2-of-3-annotator agreement threshold — a stated, changeable
   parameter, not a hardcoded assumption, since published work varies on
   this).

**Alternatives considered**
- Keep deferring both loaders until a "someday" pass. **Rejected**: the
  owner explicitly authorized continuing, and both schemas were cheap and
  fast to confirm directly (small label/annotation files, no large audio
  download needed for either verification step) — measurement-first doesn't
  mean "never," it means "verify before writing the parser," which was now
  actually done rather than remaining a placeholder.
- For LibriStutter's STUTTER-row reconstruction, guess a single universal
  reconstruction rule (e.g. always duplicate the preceding word regardless
  of type). **Rejected**: sound_repetition specifically needed a different
  shape (a trailing `-` fragment marker) to have any chance of matching
  `detect_disfluencies()`'s fragment-detection logic, and phrase_repetition
  genuinely cannot be reconstructed correctly from a single marker row (the
  true repeated phrase length isn't recoverable from this file alone) — a
  uniform rule would have quietly mismeasured both.

**Why this choice**
Both confirmations were low-cost (small files) and high-value (they either
would have blocked or silently corrupted the first real evaluation numbers).
Getting the LibriStutter reconstruction wrong specifically would have meant
this round's "first real accuracy numbers" measured something other than
what they claimed to.

**Measured result**
`load_libristutter_csv` verified against 5 real downloaded files (speaker
103, chapter 1240): correctly reconstructs each `STUTTER` row (e.g. index 4
becomes `word='Rachel'` with `ground_truth[4]='word_repetition'`, forming a
genuine back-to-back exact repeat with the preceding real "Rachel" token).
`load_sep28k_labels` verified against the real, complete 28,177-row
`SEP-28k_labels.csv`: parses without error, clip count matches the
dataset's own documented total exactly, per-type counts are plausible
against the field's known class-imbalance description. Full pre-existing
test suite unaffected (this entry's changes are additive to `loaders.py`,
gated by the specific loader functions called).

---

## 2026-08-03 — First real Track A result (LibriStutter, 499 clips)

**What was done**
Downloaded a real, distributed sample of 499 LibriStutter annotation files
(every 9th file of 4,736 available on the GitHub mirror — one git-trees API
call for the listing, individual `raw.githubusercontent.com` fetches for
the files themselves; script committed as
`profiling/evaluation/fetch_libristutter_sample.py`) and ran Track A
against it: `python -m profiling.evaluation.track_a --dataset libristutter
--data-dir eval_datasets/libristutter_sample`. 17,970 tokens scored across
499 clips. Results, full interpretation, and the caveats below are recorded
in `VALIDATION.md` §8.2 — this is the project's first-ever accuracy number
against public labeled data.

Headline: `Any` (combined) label — 99.1% recall, 63.3% precision, F1 0.773.
Per-type numbers are much more mixed and required real interpretation, not
just reading off the table — see below.

**A direct audit was performed, not left as a hypothesis.** The raw per-type
precision numbers looked alarming for `word_repetition` (22.2%) and
`prolongation` (4.8%). Rather than report those figures uninterpreted (or
guess at an explanation), every false positive for both types was audited
against its actual ground-truth label:
- `word_repetition`'s 640 FPs: 195 were actually sound_repetition, 198
  phrase_repetition, 220 prolongation, and only **27 (4.2%) were on
  genuinely clean tokens**. This confirmed the suspected cause: the
  reconstruction approximates `phrase_repetition` as a single-word repeat
  (documented limitation, previous entry), which is structurally
  indistinguishable from a real `word_repetition` — so a correct detection
  of the former scores as a false positive against the latter's ground
  truth. Corrected for clean speech only: 183 TP vs. 27 FP = **87.1%
  precision**, not 22.2%.
- `prolongation`'s 737 FPs: only 413 (56%) were on clean tokens (a real
  precision problem); the remaining 324 (44%) were the same kind of
  cross-type contamination. Unlike word_repetition, correcting for
  contamination does **not** explain away prolongation's poor precision —
  ~413 real false positives against clean speech remain a genuine finding.
- `sound_repetition`: 0/200 recall. Traced to a structural mismatch, not a
  reconstruction bug: `detect_disfluencies()`'s fragment-repetition check
  requires the fragment *before* the complete word (`prev_word.endswith
  ("-")`), while this reconstruction (matching LibriStutter's real
  STUTTER-row placement, which sits *after* the original word) puts the
  fragment second. Flagged as needing investigation on either side — not
  fixed this round (no ground ready to stand on yet for which side is
  "wrong": the reconstruction's placement choice, or the detector's
  fragment-order assumption, or both are valid patterns and the detector
  should handle both).
- `phrase_repetition` (0/201 recall) and `filler` (0 ground-truth instances
  in this specific sample) were not investigated further — both have
  already-understood causes stated in the previous entry (phrase length
  isn't reconstructable from one marker row; this particular every-9th-file
  sample happened to miss every interjection-labeled clip, confirmed
  directly against the raw CSVs, not a parsing bug).

**Alternatives considered**
- Report the raw per-type precision numbers without the FP audit.
  **Rejected**: would have (mis)read as "word_repetition detection is bad"
  when the audit shows the opposite (87.1% real precision) — exactly the
  kind of uninterpreted-metric mistake `VALIDATION.md` §1 was written to
  avoid, and directly contrary to the owner's stated goal of scientific
  rigor over surface-level numbers.
- Treat all four low-precision/low-recall types the same way (all "probably
  reconstruction artifacts"). **Rejected**: the audit shows they're not the
  same — prolongation's precision problem is substantially real, sound_
  repetition's recall problem is a structural detector/reconstruction
  mismatch, phrase_repetition's is a known reconstruction ceiling. Treating
  them identically would have hidden the one part (prolongation, sound_
  repetition ordering) that's an actual signal worth acting on.
- Download the full ~4,736-file corpus instead of a 499-file sample.
  **Deferred, not rejected**: 499 files (~10.5% of the corpus) was judged a
  reasonable first checkpoint — real enough to produce a stable `Any`-label
  signal (802 TP) and to expose the sound_repetition/prolongation findings
  above, without a much longer download for a first pass. Expanding the
  sample (or running the full corpus) is natural follow-up work, not
  required to get value from this checkpoint.

**Why this choice**
This is exactly what real, labeled-data evaluation is for and what the
owner asked for instead of anecdotal single-speaker testing: it turned two
vague-sounding low-precision numbers into one confirmed non-issue
(word_repetition) and one confirmed real issue worth investigating
(prolongation), plus a specific, actionable structural finding
(sound_repetition's fragment ordering) — none of which anecdotal real-mic
testing could have distinguished.

**Measured result**
See `VALIDATION.md` §8.2 for the full table and per-row interpretation.
Raw result file: `eval_results/20260803T042321640184Z_libristutter_A.json`
(gitignored, not committed — see `VALIDATION.md` §8.2 for the recorded
summary). Full pre-existing test suite (38 tests) plus the evaluation
package's own 36-check self-test still pass; this entry involved no code
changes, only running the already-built harness against real data and
auditing its output.

---

## 2026-08-03 — Audio-enabled evaluation (validation strategy decision)

**What was done**
The owner asked for a deliberate re-examination of the validation strategy,
explicitly against the project's ultimate objective (improving the model's
ability to detect/classify/localize disfluencies from a user's audio
recording, as robustly as possible) and explicitly requiring a decision and
implementation, not just options — with computational cost treated as
secondary to scientific rigor.

Re-examining the strategy from that objective surfaced a self-critical
finding: `profiling/evaluation/track_a.py`'s `evaluate()` had `audio_bytes
=None` hardcoded since the module was first built. That means the entire
audio-native detection layer built earlier this same day (Silero VAD
corroboration, Praat pitch/jitter/shimmer, weighted acoustic-vs-token
fusion — this project's headline 2026-08 architectural change) had **never
once been evaluated against labeled ground truth**, including every
ablation question already sitting unanswered in `VALIDATION.md` §9 (VAD
on/off, Praat on/off, fusion-weight sweep). The previous LibriStutter Track
A result (previous entry) was real and valuable, but it only ever tested
the text/timing-based half of the detector.

Given the owner's explicit priority — "usefulness for improving our own
model" above computational/engineering cost — closing that specific,
self-identified gap was judged the highest-value next step, ahead of
acquiring a new dataset (SEP-28k audio, previously the assumed next step).

**Alternatives considered**
- **SEP-28k audio** (the default next step per the earlier dataset-priority
  table). Has no reference transcript at all, so it can only ever test
  `block`/`prolongation` via a not-yet-built acoustic-only pass; everything
  else needs full Track B (our own ASR + hypothesis-to-reference alignment,
  unbuilt, and at 54–102s/clip on this hardware, evaluating even a few
  hundred clips would take many hours). Does not touch the audio-native
  evaluation gap at all — a user could spend the whole download budget here
  and still not know whether Silero VAD/Praat/fusion actually help.
- **FluencyBank Timestamped** (real people who stutter, not synthetic
  injection — genuinely the strongest option on pure scientific-rigor
  grounds). Researched directly: hosted on TalkBank
  (fluency.talkbank.org/derived/) in CHAT format, a specialized linguistic
  transcription format requiring a dedicated parser (not a simple CSV), and
  TalkBank corpora — especially identifiable clinical/disfluent-speech
  corpora — commonly require a registered account or signed data-use
  agreement, neither confirmed nor ruled out without attempting access.
  **Not chosen for this phase**: real, unconfirmed integration risk (new
  file format, possible access gating) that could consume the phase without
  producing a working result. Recorded in `ROADMAP.md` as the clear next
  dataset to pursue once this phase's audio-enabled baseline exists — it
  remains the scientifically strongest option and should not be dropped.
- **Continue expanding the LibriStutter text-only sample size** (more of
  the same evaluation mode, just more clips). Rejected: more data points in
  the same (audio-blind) mode cannot close the audio-native evaluation gap
  no matter how large the sample — the gap is about *what's tested*, not
  *how much*.

**Why this choice**
Directly serves "improving our own model's ability to take only a user's
speech recording and detect disfluencies" more than any single dataset
choice does: it's the first and only way, so far, to measure whether this
project's actual audio-native architecture change works, using real labeled
ground truth instead of the two anecdotal real-mic recordings from earlier
in the day. It also reuses the exact same 499-clip sample already scored in
text-only mode, making the comparison a genuinely controlled before/after
(same clips, same ground truth, audio on vs. off) rather than a fresh,
harder-to-compare number.

**Implementation**
- `loaders.LabeledClip` gained an optional `audio_bytes: bytes | None`
  field (default `None` — no behavior change for existing text-only
  callers).
- `_flac_bytes_to_wav16k()` added: decodes LibriStutter's real audio format
  (FLAC, 22050 Hz — not one of Silero VAD's supported 8000/16000 Hz rates)
  via the new `soundfile` dependency, then reuses `profiling/asr.py`'s
  already-tested `resample_to_16k` rather than duplicating resampling
  logic. Degrades to `None` (never raises) on any decode failure, so one
  bad file can't abort a batch.
- `load_libristutter_csv_with_audio()` / `load_libristutter_dir_with_audio()`
  added: pair annotation CSVs with matching FLAC files by relative path;
  missing/undecodable audio degrades that one clip to text-only rather than
  being dropped or crashing.
- `track_a.evaluate()` changed from hardcoded `audio_bytes=None` to
  `audio_bytes=c.audio_bytes` — the SAME function now serves both text-only
  and audio-enabled evaluation, driven entirely by which loader populated
  the clip, not a second code path to maintain.
- `track_a.py` CLI gained `--audio-dir`; `report.save_run` records
  `track="A+audio"` and the count of clips that actually had usable audio,
  so a partial-audio run is never silently reported as if every clip had
  audio.
- New `profiling/evaluation/fetch_libristutter_audio.py`: downloads FLAC
  files matching an already-downloaded annotation sample (same acquisition
  pattern as `fetch_libristutter_sample.py`, proven earlier the same day).
- 13 new self-test checks added (FLAC decode correctness, 16kHz resampling,
  graceful degradation on garbage/missing audio, and — critically — a
  direct check that audio-enabled evaluation actually produces different,
  evidence-annotated output from text-only evaluation on the same clip, not
  just plumbing that silently goes unused).

**Why soundfile, not a different FLAC route**: pure-Python, wraps the
well-established `libsndfile` C library, small dependency footprint,
installed and directly verified against a real downloaded LibriStutter FLAC
file before being adopted (not assumed to work).

**Measured result**
`python -m profiling.evaluation.track_a --self-test` → all checks pass,
including the new audio-path checks. Full pre-existing test suite (38
tests) unaffected — this round's changes are additive and gated by which
loader/CLI flag is used. Real-data results (499 clips, now with matching
audio downloaded) are recorded in the next entry.

---

## 2026-08-03 — Bug found and fixed: `_flac_bytes_to_wav16k` silently produced silent audio

**What was done**
Downloaded real matching audio (499/499 FLAC files, zero failures — via new
`profiling/evaluation/fetch_libristutter_audio.py`) for the same 499-clip
sample already scored in text-only mode, then ran the first audio-enabled
Track A evaluation. The result looked dramatic and worth investigating
before trusting it: `prolongation` collapsed from 37 true positives
(text-only) to **zero** — not just fewer, none at all — while the `Any`
combined label's precision jumped sharply (63.3% → 93.4%).

Rather than write that up as "the audio-native layer is extremely
effective at suppressing prolongation false positives" (a plausible-
sounding but unverified story), it was audited directly, the same
discipline applied to the text-only result earlier the same day. Direct
inspection of five real ground-truth prolongation cases showed
`_AcousticContext.voiced_duration()` returning **exactly 0.0** for every
one — and inspecting the raw decoded waveform showed why: **the entire
converted audio was silent** (RMS = 0.0, max amplitude = 0.0), for every
clip, despite the source FLAC files being confirmed to contain real speech
(verified directly with `soundfile.read()` using its default `dtype`).

**Root cause**: `_flac_bytes_to_wav16k()` (previous entry) read FLAC data
with `sf.read(..., dtype="int16")`. Confirmed by direct isolation testing
that this specific call **silently returns an all-zero array** for real
LibriStutter FLAC files — a `soundfile`/`libsndfile` quirk with how these
particular files are encoded, not a corrupted-download or wrong-path
problem (the same bytes decode correctly via `sf.read()`'s default float64
dtype). The bug was invisible at every layer above it: no exception was
raised, `resample_to_16k` ran successfully on the (silent) data, the WAV
bytes were valid and correctly-shaped, and the full evaluation pipeline
executed to completion and produced a plausible-looking (if now-understood-
as-wrong) result.

**Why the self-test didn't catch this**: the audio self-test added in the
previous entry generated its own synthetic FLAC in-process (via `sf.write`
on a numpy tone array) and only checked that decoding produced non-empty
WAV bytes at the right sample rate — it never checked that the *content*
was non-silent, and the synthetic file's particular encoding didn't happen
to trigger the same `dtype="int16"` quirk that real downloaded files did.
**Fixed**: the self-test now also decodes the resulting WAV and asserts its
RMS is meaningfully above zero — the exact check that would have caught
this before it ever reached a real evaluation run.

**Fix**: read with `sf.read(..., dtype="float64")` (the default, confirmed
correct) and do the int16 scaling manually
(`np.clip(data * 32767.0, -32768, 32767).astype(np.int16)`) — the same
scaling pattern already used elsewhere in this codebase (e.g.
`profiling/asr.py`'s `resample_to_16k`), rather than trusting the library's
own dtype-conversion path a second time without re-verifying it.

**Alternatives considered**
- Keep the result and report it as a genuine finding, noting the anomaly as
  a caveat. **Rejected outright**: the voiced_duration=0.0 pattern was
  suspicious enough (exactly zero, not just low, across every case
  examined) to warrant a direct check before drawing any conclusion — and
  the check took minutes, not hours, to run. Reporting an unverified
  dramatic result would have directly contradicted this project's own
  measurement-first convention and the owner's explicit request for
  rigorously audited findings, not raw metrics.

**Why this choice**
This is exactly why every non-trivial result this project produces gets
audited before being written up, not just read off a table — a plausible
and even *desirable-sounding* result (audio dramatically improving
precision) turned out to be measuring something else entirely (audio that
was never really there). Catching this before it entered `VALIDATION.md` as
a recorded baseline result, rather than after, is the whole point of the
project's evaluation discipline.

**Measured result**
Post-fix, direct verification on the same real clip: RMS 0.0 → 0.0222,
max amplitude 0.0 → 0.265, individual word-level RMS values all sensible
and non-zero (0.016–0.042 range). `python -m profiling.evaluation.track_a
--self-test` passes with the new non-silence check included. The corrected
full 499-clip audio-enabled run is recorded in the next entry.

---

## 2026-08-03 — First audio-native-layer result (LibriStutter, 499 clips, corrected) — baseline established

**What was done**
Re-ran audio-enabled Track A against the same 499 real LibriStutter clips
used for the text-only baseline, now with the FLAC-decode bug fixed and
verified. This is this project's **first-ever evaluation of its audio-native
detection layer** (Silero VAD, Praat, weighted acoustic-vs-token fusion)
against labeled ground truth. Full results, per-row interpretation, and
mechanism-level audits are in `VALIDATION.md` §8.3 — summarized here:

- **`Any` (combined) — the headline result: F1 0.773 → 0.835 with audio
  enabled**, driven by a real precision gain (63.3% → 72.2%, 157 fewer false
  positives) at essentially no recall cost (802 → 801 TP). This is the
  first measured evidence — not architectural argument — that the 2026-08
  audio-native restructuring achieves its design goal.
- **`filler`/`sound_repetition`/`word_repetition`/`phrase_repetition`
  identical to text-only**, correctly: `profiling/acoustic.py` only derives
  `block`/`prolongation` candidates, so audio can't affect the other four
  types at all. Confirms the harness wires audio through exactly where it
  should.
- **`prolongation` recall dropped (37 → 21 TP) — mechanistically confirmed,
  not left as a guess**: of the 37 true prolongations whose reconstructed-
  token span nominally clears the 1.0s threshold, exactly 21 still clear it
  once trimmed to real *voiced* duration (`_AcousticContext.voiced_duration
  ()`) — matching the observed TP count precisely when checked directly.
  Most likely a property of how LibriStutter's `STUTTER`-row reconstruction
  estimates timing (its declared span overstates genuinely sustained
  voicing) rather than of real prolonged speech — real live-mic
  prolongations validated earlier the same day (Part D, 1.24–1.42s) were
  correctly caught by the same voiced-duration logic. Not resolvable
  further without Track B or a real (non-reconstructed) prolongation
  dataset — flagged in `ROADMAP.md`, not fixed here (algorithm changes are
  explicitly out of scope for this baseline-establishing phase).

**Why this is the baseline, not just another result**
Per the owner's explicit direction, this checkpoint — the current detector/
acoustic implementation, evaluated both with and without audio against 499
real labeled clips, fully audited — is now the project's first established
research baseline. Every future architectural or algorithmic change is
expected to be compared against these numbers, not against vibes or a
single anecdotal recording. `VALIDATION.md` §8/§9/§10 (results, ablations,
published-baseline comparison) exist specifically to keep accumulating that
comparison as the project evolves.

**Measured result**
See `VALIDATION.md` §8.3 for the full table. Raw result file:
`eval_results/20260803T050917442304Z_libristutter_A+audio.json`
(gitignored). Full pre-existing test suite (38 tests) + the evaluation
package's self-test (47 checks, including the new non-silence check) all
pass. No detection-algorithm code was changed as a result of these
findings — per the owner's explicit instruction, they're recorded as
evidence for future, separately-approved algorithm work, not acted on
immediately.

---

## 2026-08-03 — Full ablation study against the baseline

**What was done**
Per the owner's explicit direction, ran a complete ablation study against
the newly-established baseline (previous entry): added a `use_praat` config
toggle to `profiling/acoustic.py` (infrastructure only — defaults to `True`,
verified to produce byte-for-byte identical behavior to before its
addition, via the full regression suite), then built
`profiling/evaluation/run_ablations.py` — loads the 499 real LibriStutter
clips (+ audio) once and re-runs `detect_disfluencies()` across 10 config
variants, holding everything else constant: baseline, VAD off, Praat off,
`fusion_weights.acoustic` ∈ {0.5, 2.0, 5.0}, `prolongation_min_seconds` ∈
{0.65, 0.85, 1.2, 1.4}. Speaker-calibration on/off was not run — no
calibration baseline exists for LibriStutter's synthetic per-clip
structure, documented as a genuine scope limit rather than skipped
silently.

**Findings, ranked by measured contribution to `Any` F1** (full tables in
`VALIDATION.md` §9):
1. `prolongation_min_seconds` — dominant by an order of magnitude (F1 range
   0.639–0.933 across the sweep vs. 0.835 at the current baseline).
2. `fusion_weights.acoustic` — small, real, saturating effect (+0.003 F1 at
   2×, flat at 5×).
3–4. Silero VAD and Praat corroboration — **both measured zero effect**,
   tied last: `vad_off` and `praat_off` are byte-for-byte identical to
   baseline across every TP/FP/FN count.
5. Speaker calibration — not applicable to this dataset.

**A genuine methodological finding emerged from running the ablation, not
predicted going in**: VAD and Praat corroboration were designed (2026-08
restructuring) to adjust event *confidence*, not whether an event fires —
except indirectly through the fusion replace-vs-keep decision, which
`fusion_weight` *does* show measurably affects results. `score_word_level`
(this project's only word-level metric so far) scores pure presence/
absence, which is structurally blind to confidence. A "zero effect" result
for VAD/Praat is therefore evidence the *metric* can't see their designed
effect, not evidence the effect isn't real. This directly identifies a
measurement-infrastructure gap, not a component to deprioritize.

**Alternatives considered**
- Treat VAD/Praat's zero-effect result as "these components don't help" and
  recommend removing them. **Rejected**: would conflate "this metric can't
  see it" with "it doesn't exist" — exactly the kind of unaudited
  conclusion this project's evaluation discipline exists to prevent (see
  the FLAC-bug entry above for the same discipline applied to a different
  question). The correct conclusion is "build a confidence-sensitive metric
  before concluding anything about VAD/Praat specifically."
- Treat the `prolongation_min_seconds` sweep's optimal value (0.85s for
  prolongation-specific F1) as a tuning recommendation to apply now.
  **Rejected, explicitly, per the owner's standing instruction**: this
  baseline-establishing phase does not change detection algorithms/
  thresholds. Recorded as evidence for a future, separately-approved tuning
  pass, with the caveat (§9.4) that it's measured on reconstructed timing
  and needs validation against non-reconstructed data before being trusted
  as a real optimum.

**Why this choice**
Directly answers what the owner asked: which components contribute most
and least, with evidence, not assumption. The dominant-lever finding
(threshold >> VAD/Praat/fusion-weight) is itself decision-relevant for
prioritizing future work, and the measurement-blindness finding is
arguably as valuable as the ranking itself — it's a concrete, actionable
gap in the evaluation methodology, found by using it rigorously rather than
assumed in advance.

**Measured result**
10 result files saved under `eval_results/*_libristutter_ablation-*.json`
(gitignored). Full regression suite (38 tests) + evaluation package
self-test both still pass after the `use_praat` toggle addition — confirmed
before running any ablation variant. See `VALIDATION.md` §9 for complete
tables and the evidence-based next-phase recommendation this produced.

---

## 2026-08-03 — Track B evaluation protocol pre-registered before implementation

**What was done**
Per the owner's explicit instruction, wrote the full Track B evaluation
protocol into `VALIDATION.md` §5.1 **before** writing `alignment.py` or
`track_b.py` — a real pre-registration, not a description of what was
already built. Defines: the Levenshtein word-alignment method and its
disfluent-word cost bias (substitution cost 1.5× higher against
ground-truth-disfluent reference words, so the aligner doesn't force a
low-cost coincidental substitution match onto a word ASR actually deleted —
implementing the modified-cost technique found in this project's earlier
literature review); the exact metrics to report at three levels (Track A;
Track B on the ASR-preserved subset; Track B overall — never blended); the
precise ASR-attributable/detector-attributable decomposition formula
(`Detector-attributable gap = R_A − R_B|preserved`,
`ASR-attributable gap = R_B|preserved − R_B|overall`, which sum exactly to
the total gap by construction, not by inference); and two separate success
criteria — a pass/fail methodological gate (hand-verify ≥10 alignments
before trusting anything else) and an explicitly non-predetermined
scientific outcome (the decomposition being ASR-heavy, detector-heavy, or
mixed are all valid, reportable results).

**Alternatives considered**
- Build the pipeline first and decide how to score/interpret it once
  results are visible. **Rejected, explicitly, per the owner's instruction**:
  this is exactly the failure mode pre-registration exists to prevent —
  deciding what counts as "the detector's fault" after already seeing which
  framing makes the numbers look better is a real risk with a metric this
  interpretively flexible (attribution between two systems, not a single
  objective ground truth), and this project's whole evaluation discipline
  this far has been about removing exactly that kind of interpretive
  latitude.
- A naive unweighted Levenshtein alignment (no disfluent-word cost bias).
  **Rejected**: would understate ASR-attributable loss whenever a
  disfluent reference word happens to superficially resemble a nearby ASR
  output word — the literature reviewed earlier this project specifically
  flags this as a known failure mode for disfluency-transcript alignment,
  not a hypothetical concern.

**Why this choice**
Matches the owner's stated goal directly — "our methodology is fixed in
advance rather than adapted after seeing the results" — and produces a
decomposition formula that's falsifiable and exact (the two components sum
to the observed total gap by construction), not an eyeballed or
after-the-fact rationalized split.

**Measured result**
Not applicable — this entry is the protocol design step; no code was
written. Implementation against this exact protocol follows in the next
entry.

---

## 2026-08-03 — Track B implemented and piloted exactly per the pre-registered protocol

**What was done**
Built `profiling/evaluation/alignment.py` (Levenshtein word alignment with
the disfluent-word cost bias, `disfluent_cost_multiplier=1.5` default) and
`profiling/evaluation/track_b.py` (full pipeline: real CrisperWhisper ASR →
`detect_disfluencies()` on our own transcript → alignment back to ground
truth → the three-level scoring `VALIDATION.md` §5.1 defined) exactly
against the protocol written in the previous entry — no scoring-logic
decisions were made after seeing results. Verified the alignment bias
directly (a hand-constructed case: `"need" "need"` vs. mis-heard `"nerd"`
correctly produces `deletion` on the disfluent word and `substitution` on
the clean one, with the bias off it's reversed) before trusting it on real
data. Self-tested `score_clip`'s decomposition math (10 checks, hand-
computed expectations, no real ASR) before running anything slow.

Ran a 2-clip smoke test first (caught and fixed two lingering non-ASCII
console-encoding bugs, same class as the earlier `track_a.py` fix — this
project clearly needs a standing rule here, see Alternatives below), then
the full 30-clip pilot (deterministic first-30, chosen before running —
CrisperWhisper's 54–102s/clip made the full 499-clip sample a
multi-hour-scale decision explicitly deferred), then a 10-clip `--verbose`
re-run (same first 10 clips, consistent with the 30-clip aggregate) adding
per-disfluent-word diagnostic printing specifically to satisfy the
pre-registered methodological gate (§5.1 point 4: hand-check ≥10 clips
before trusting anything else).

**Methodological gate: passed.** All 10 clips' disfluent-word alignments
hand-checked against the printed reference/hypothesis word sequences —
every classification defensible; a minor, low-impact ambiguity found in
cases with two adjacent identical reference words (only one survives into
ASR's output, and *which* reference position gets credited "correct" vs.
"deletion" is arbitrary) never changed the correct aggregate conclusion.

**Headline finding**: Track A's ~99% recall (previous entries) drops to
**~4–9%** under Track B's `Any`-label evaluation (overall 4.2%, ASR-
preserved-subset 9.1%). Mean WER 22.4% — CrisperWhisper, despite being a
disfluency-preserving fine-tune, struggles substantially with LibriStutter's
synthetically-spliced disfluency segments specifically. The mechanical
decomposition (§5.1 formula) attributes ~95% of the total recall gap to the
detector and ~5% to ASR word-loss.

**A real methodological limitation was found during hand-verification, not
predicted when the protocol was written** — recorded as a dated addendum
to `VALIDATION.md` §5.1, per that section's own stated discipline
(deviations get an addendum, never a silent edit). Concretely: reference
`"Rachel" "Rachel" "Lynde"` (a direct repeat) came back from ASR as
`"Rachel" "Lynde," "Rachel" "Lynde"` — the disfluent word itself aligns
`correct` (a "Rachel" really is there), but ASR's insertion of "Lynde,"
between the repeats broke the *adjacency* `word_repetition`'s exact-match
check depends on, and the detector predicted `phrase_repetition`/`block`
instead. **"This word was transcribed correctly" is necessary but not
sufficient for "the detector had a fair chance"** — the surrounding context
also has to survive, and `R_B|preserved` as currently defined doesn't check
that. This means the ~95%/~5% split almost certainly overstates genuine
detector-only failure, though by an amount this pilot's methodology can't
yet quantify precisely.

**Alternatives considered**
- Report the mechanical 95%/5% decomposition without the hand-verification
  caveat. **Rejected**: would misrepresent a finding the pre-registered
  protocol's own methodological gate was specifically designed to catch —
  reporting a number known (from direct inspection) to plausibly be
  systematically biased, without saying so, is exactly the failure mode
  pre-registration exists to prevent, even when the bias was found *after*
  writing the protocol rather than anticipated in it.
- Immediately fix `R_B|preserved`'s definition (e.g., require an N-word
  context window around each disfluent position to also align `correct`)
  and re-run. **Deferred, not rejected**: a real, concrete next step
  (`ROADMAP.md`), but changing the scoring definition after seeing exactly
  the case that motivates it, without a fresh pre-registration of the new
  definition, would repeat the mistake pre-registration is meant to avoid —
  better to record the finding now and design the fix as its own dated,
  pre-registered step.
- Treat the 30-clip pilot's absolute numbers as final/precise. **Rejected**:
  explicitly scoped as a pilot in §5.1 — the qualitative conclusion (a
  large, real Track A→B recall gap exists) is trustworthy at this sample
  size; the precise percentages are not, and are reported with that caveat
  throughout `VALIDATION.md` §8.4.
- (Recurring) Non-ASCII characters in print-reachable strings. **Fixed
  again** (third occurrence this project — `track_a.py`, `report.py`,
  now `track_b.py`) via the same plain-ASCII substitution as before. Not
  escalated to a lint rule or pre-commit check this round — flagged in
  `ROADMAP.md` as worth doing given the repeated pattern.

**Why this choice**
Directly delivers what the owner asked for: implement Track B, but only
after fixing the methodology in advance — and when implementation revealed
a real gap in that methodology, report it as found rather than smoothing it
over to produce a cleaner-looking number.

**Measured result**
See `VALIDATION.md` §8.4 for full tables. Raw results:
`eval_results/20260803T082117789381Z_libristutter_B.json` (30-clip pilot),
`eval_results/20260803T084650194736Z_libristutter_B.json` (10-clip verbose
hand-verification run). `python -m profiling.evaluation.track_b --self-test`
→ 10/10 pass. Full pre-existing test suite (38 tests) + `track_a.py`
self-test unaffected — this entry's changes are additive
(`alignment.py`/`track_b.py` are new modules; nothing existing was
modified except the encoding fixes).

---

## 2026-08-03 — Context-strict preserved-subset scoring implemented and run

**What was done**
Implemented and ran the `R_B|preserved_ctx1` refinement that the previous
entry's own addendum flagged (`VALIDATION.md` §5.1) — pre-registered *before*
implementation, same discipline as the original Track B protocol: wrote the
exact definition (a ground-truth disfluent word at reference index `i`
preserved only if *both* `i` and `i−1` align `correct`), its explicit
scope/limits (a real fix for `word_repetition`/`sound_repetition`'s 1-word
dependency, only a partial one for `phrase_repetition`, not applicable to
`prolongation`), and the expected direction of the result, into `VALIDATION.md`
before touching `track_b.py`. Added per-clip result caching
(`_cache_path`/`_load_cached`/`_save_cache`,
`eval_datasets/_track_b_cache/`) at the same time, specifically so this and
future metric-only refinements never need to re-run CrisperWhisper again —
built because this is now the second time a scoring-definition change has
followed a full Track B pilot, and it won't be the last. Refactored
`score_clip` to compute `preserved`, `preserved_ctx1`, and `overall`
simultaneously via a shared `_score_into` helper (avoiding tripling the
scoring logic). Added 4 new self-test checks (14 total, up from 10),
including a direct reproduction of the already-hand-verified
`103-1240-0000` "Rachel Rachel Lynde" case, asserting the original metric
counts it as a miss (`fn=1`) and the context-strict metric excludes it
entirely from the preserved subset rather than crediting it. Re-ran the same
30-clip pilot (cache was empty, so this was a full real-ASR run,
~4,590s/30 clips) computing all three metrics together.

**Headline finding**: `R_B|preserved_ctx1` (Any label) = **1.0** — every one
of the 2 disfluent instances that survived the context-strict filter was
correctly flagged by the detector. This reverses the previous entry's
mechanical decomposition: **~95% detector-attributable / ~5%
ASR-attributable → ~0% detector-attributable / ~100% ASR-attributable.**
Equally important: only 2 of 48 total ground-truth-disfluent instances
survive the context-strict filter at all (down from 22 under the
word-only definition) — ASR overwhelmingly fails to preserve a disfluent
word *and* its immediate neighbor together, even when it gets the word
itself right. A further nuance surfaced on inspection: neither of the 2
surviving instances got the *exact* correct type label from the detector
(both are ground-truth `word_repetition`; the detector said
`phrase_repetition`/`block` instead) — and for the `103-1240-0000` case
specifically (the same clip from the original hand-verification), tracing
it shows this is arguably not a detector error at all: ASR's insertion of
an extra word between the two repeated "Rachel"s means both individually
align `correct` (satisfying `preserved_ctx1`) while the detector, which
operates on the actual hypothesis token stream rather than the aligned
reference, never sees them back-to-back — so classifying the pattern as a
phrase repetition is a defensible read of the *input it actually received*.
This flags a sharper next refinement (requiring hypothesis-side contiguity,
not just reference-side correctness) — see `VALIDATION.md` §8.4.1.

**Alternatives considered**
- Report the `preserved_ctx1` = 1.0 figure as the corrected, final
  detector-attributable share. **Rejected**: n=2 is far too small to state a
  precise number — reported as a strong directional finding, mechanistically
  explained (not just an isolated statistic), with the sample-attrition
  chain (48 → 22 → 2) given equal billing so readers see exactly how thin
  the evidence is.
- Also implement the hypothesis-side-contiguity refinement this run's own
  `103-1240-0000` trace motivates, in the same step. **Deferred, not
  rejected**: same reasoning as the previous entry's own deferral of this
  exact refinement — designing a metric change in direct response to the
  one example that motivates it, without a fresh pre-registration and more
  data, repeats the mistake pre-registration exists to avoid.
- Skip caching and just re-run ASR fresh each time a metric definition
  changes. **Rejected**: this is the second scoring-definition change after
  a completed pilot: paying the ~4,590s CrisperWhisper cost again for a
  scoring-only change is wasteful and actively discourages the kind of
  iterative methodology-strengthening this project's standing instruction
  calls for. Caching decouples "run ASR + detector" from "score the
  results," which is the right boundary for a metric that may keep
  evolving.

**Why this choice**
Directly follows the owner's standing instruction: "if validation reveals
weaknesses in our evaluation methodology, strengthen the evaluation first."
The previous entry found and documented exactly such a weakness
(reference-word-only "preserved" doesn't check adjacent context); this
entry fixes it, pre-registered first, and reports the result honestly even
though it dramatically overturns the previous entry's own mechanical
split — which is the point of pre-registering before seeing results.

**Measured result**
See `VALIDATION.md` §8.4.1 for full tables and reasoning. Raw result:
`eval_results/20260803T111624161563Z_libristutter_B.json`. Cache:
`eval_datasets/_track_b_cache/` (30 files, all populated this run).
`python -m profiling.evaluation.track_b --self-test` → 14/14 pass. Full
pre-existing test suite (38 tests) unaffected — changes are additive
(`preserved_ctx1` scoring path, caching helpers; existing `preserved`/
`overall` scoring logic unchanged, verified by the original 10 self-test
checks still passing unmodified).

---

## 2026-08-03 — Track B scaled 30 → 90 clips: context-strict finding confirmed, treated as a major conclusion

**What was done**
The previous entry's own n=2 result was explicitly flagged as too small to
trust as a precise number, with scaling the pilot listed as top
`ROADMAP.md` priority. Re-ran `track_b.py` with `--n 90` against the same
499-clip sample (first 90, so the original 30 clips are an identical
subset, not resampled). Per-clip caching (built in the previous entry) meant
only the 60 new clips needed fresh CrisperWhisper inference. The run was
interrupted once mid-way by an unrelated session/harness restart (32 clips
cached at that point, 30 original + 2 new); resumed with the identical
command, which picked up from the cache with no lost or duplicated work —
the first real-world validation that the caching investment from the
previous entry pays off exactly as intended. One clip took anomalously long
(5,854s vs. a typical ~100–170s, a one-off system hiccup) but completed
without error and didn't affect the scored output. Total: 14,203s wall-clock
for this leg (58 real ASR runs + 32 cache hits).

**Headline finding: the previous entry's dramatic result is confirmed, not
just replicated by coincidence of a tiny sample.** At 90 clips (127 total
ground-truth-disfluent instances, up from 48), `R_B|preserved_ctx1` recall
is again exactly **1.0** — now on n=7 positive instances instead of n=2. The
sample-attrition rate is also stable across the scale-up (7/127 = 5.5% vs.
2/48 = 4.2% survive both the word-correct and context-correct filter,
consistent within noise). The word-only decomposition also replicated
closely (~93% detector-attributable at n=90 vs. ~95% at n=30). This
promotes the previous entry's finding from "a striking result worth
scaling before trusting" to **a confirmed research conclusion**: the
detector's binary disfluent/clean judgment is effectively perfect whenever
ASR preserves both a disfluent word and its immediate context, and the
real-world Track A→B recall shortfall is overwhelmingly attributable to ASR
losing/corrupting disfluent words and their surroundings, not to detector
weakness.

**A second finding sharpened at this scale, not visible at n=2**: breaking
the 7 context-strict-preserved instances down by type, only 2/7 (29%) got
the *exact* correct type label (`phrase_repetition`); the other 5
(`word_repetition` ×4, `sound_repetition` ×1) were flagged as disfluent
(contributing to `Any`'s perfect recall) but mislabeled — the same
hypothesis-side-word-contiguity mechanism traced by hand in the previous
entry (`103-1240-0000`), now confirmed as a recurring pattern across
multiple clips rather than a single anecdote. This is treated as the one
genuine, scoped, detector-side issue this evaluation phase has surfaced:
not a recall problem, a type-classification problem, specific to
`word_repetition`/`sound_repetition` under ASR-inserted context noise.

**Synthesis for future direction (asked for explicitly by the project
owner, not just a raw number)**: with two independent, mutually-reinforcing
lines of evidence now in hand, the highest-impact next investment is judged
to be ASR robustness on and around disfluent speech (not detector-recall
tuning), plus the scoped word/sound-repetition type-classification fix.
Full reasoning: `VALIDATION.md` §8.4.2's "Implications for future
development priorities." `ROADMAP.md` restructured accordingly — detector
threshold/recall tuning explicitly de-prioritized given this is now
evidence, not a hypothesis.

**Alternatives considered**
- Treat the n=90 result as merely "more of the same pilot" rather than a
  confirmed conclusion. **Rejected**: the entire point of scaling was to
  distinguish a real effect from small-sample noise; getting the identical
  `R_B|preserved_ctx1` = 1.0 result at 3.5x the positive-instance count,
  with a stable attrition rate, is exactly the confirmation the previous
  entry's own stated non-goal ("not treating a 20–40 clip pilot as
  statistically conclusive") was waiting for. Continuing to hedge it as
  "provisional" after this would be under-stating real evidence, which is
  its own kind of inaccuracy this project's documentation discipline exists
  to avoid.
- Immediately implement the hypothesis-side-contiguity metric refinement or
  a word/sound-repetition type-classification fix in this same step.
  **Deferred, not rejected**: consistent with this project's standing
  practice, an algorithmic/metric change gets its own pre-registered,
  separately-scoped step, not bundled into the entry that discovered the
  motivating evidence — now explicitly queued in `ROADMAP.md` as the
  top evidence-backed item.
- Scale further than 90 clips in this same step (e.g. the full 499).
  **Deferred**: 90 clips already produced a stable, confirmed result at
  reasonable cost (~4hr total including the earlier 30-clip leg); the
  marginal evidentiary value of a full 499-clip Track B run is lower than
  the marginal cost (a multi-hour-scale commitment, §5.1) given the last two
  runs already agree closely with each other.

**Why this choice**
Directly follows the standing instruction to let validation findings decide
development priorities rather than the reverse, and to treat a
scale-confirmed dramatic result as a major conclusion rather than just
another logged number — synthesizing what it means for the project, not
stopping at reporting the table.

**Measured result**
See `VALIDATION.md` §8.4.2 for full tables, decomposition, and the
synthesis section. Raw result:
`eval_results/20260803T154940357685Z_libristutter_B.json`. Cache:
`eval_datasets/_track_b_cache/` (90 files). Full pre-existing test suite (38
tests) + both eval self-tests (Track A, Track B 14/14) re-verified
unaffected before this entry was written — no code changed in this step,
only a larger data run and its documentation.

---

## 2026-08-03 — Phase 1 closing review: exact-subset Track A recall, critical methodology review, documentation consolidation

**What was done**
The project owner asked to formally close Phase 1 (Validation, Benchmarking,
Analysis, Scientific Understanding) before starting Phase 2 (evidence-driven
model improvement): audit every Markdown file for completeness, critically
re-examine the validation methodology itself for mistakes/gaps, decide
whether any additional datasets or checks would materially strengthen this
phase and implement them if feasible, reorganize the documentation
hierarchy, and produce a closing summary with an explicit commit
recommendation. This entry covers the methodology-review and
documentation-consolidation portion; the closing summary is recorded
separately (see `PHASE_1_SUMMARY.md`, added this same day).

**Exact-subset Track A recall replaces an approximation (implemented).**
§5.1's Track B decomposition formula always used Track A's full 499-clip
sample recall ($R_A$ = 0.990) as a stand-in for what Track A would score on
the *specific* 30- or 90-clip subset Track B actually used — flagged
honestly as an approximation when first written, but never tightened.
Re-examining it during this review: Track A needs no ASR, so computing it
exactly on Track B's own clip subsets costs seconds, not a new multi-hour
run. Done: ran `track_a.evaluate()` directly against the identical 30- and
90-clip subsets (with audio). Result: exact $R_A$ = **1.000** on both (not
0.990) — TP=48/FN=0 at n=30, TP=127/FN=0 at n=90. This turns the
context-strict decomposition's detector-attributable share from "≈0%, a
small negative number attributed to sample-size noise" into **exactly
0.000, by direct subtraction of two matched numbers**, not an approximation
artifact. Every decomposition table in `VALIDATION.md` §8.4/§8.4.1/§8.4.2
was updated to use the exact number, with the change explained inline (not
a silent overwrite) and the full-sample number kept alongside for context.

**Critical review of the full methodology (documented in `VALIDATION.md`
§7.2–§7.4, not just this log entry).** Systematically re-examined every
evaluation procedure, metric, dataset, ablation, and protocol used in Phase
1, specifically hunting for mistakes and unstated assumptions rather than
re-summarizing existing write-ups. Two new, real findings, neither
previously identified or documented anywhere in this project:

1. **Track B's clip subset is speaker-clustered, not speaker-representative.**
   Checked directly: the full 499-clip Track A sample spans 40 distinct
   speakers; LibriStutter's filenames are speaker-ordered, and Track B's
   "first 30" / "first 90" clip selection is a deterministic prefix of that
   list — so the 30-clip pilot covers only 3 speakers, and the 90-clip
   scaled run only 7 (confirmed by direct inspection of speaker IDs). The
   entire Track B evaluation, including the confirmed §8.4.2 conclusion, has
   so far been measured on 17.5% of the available speakers, not a
   representative cross-section — and ASR error behavior is known to be
   speaker/accent-dependent, so this is a real, not cosmetic, generalization
   concern.
2. **The confirmed "ASR is the bottleneck, not the detector" conclusion has
   been measured against exactly one ASR backend (CrisperWhisper) and one
   dataset family (LibriStutter's synthetic disfluency injection).** Neither
   has been cross-checked against a second ASR model or real (non-synthetic)
   disfluent speech (e.g. FluencyBank Timestamped). This is judged the
   single largest generalization risk attached to Phase 1's headline
   finding — the *absolute* Track A→B recall drop is not in doubt, but
   whether ASR-fidelity is *generally* the bottleneck (vs. specifically for
   this backend on this dataset's synthetic splices) is not yet established
   with the same rigor as the rest of the finding.

Neither of these was fixed in this session — both require real new work
(a speaker-stratified Track B re-sample with fresh ASR; a second ASR
backend or FluencyBank Timestamped integration) that belongs in Phase 2's
scope, not a closing/consolidation session. Both are recorded as the top
two Phase 2 validation priorities in `ROADMAP.md`, ahead of further
detector-side work, specifically *because* this review surfaced them as
real open questions rather than closed ones.

Three additional, smaller gaps were identified and also deferred with
reasoning (not fixed, not silently dropped): Track B has no localization
(IoU) metric at all (`score_clip` hardcodes `localization=None`, confirmed
by reading the source); no confidence intervals or significance testing
exists anywhere in this evaluation phase, only point estimates with
qualitative "too small to trust" caveats; and the ablation study's
"optimal threshold" findings (§9) are re-confirmed to still rest entirely
on LibriStutter's reconstructed-token timing, unchanged from §9.4's
original caveat. Full reasoning for all five findings: `VALIDATION.md`
§7.2.

**One additional check, performed and resolved (not a finding, a
verification): was the context-strict window's design circular** — i.e.,
reverse-engineered from the one hand-verified example that motivated it,
making its dramatic result a foregone conclusion? Reviewed directly:
judged **not circular** — the window was derived from which detector
checks are mechanistically 1-word-back-dependent (readable directly from
`detect.py`), its scope limits and expected direction were stated in
`VALIDATION.md` §5.1's addendum *before* being run against real data, and
its actual magnitude (exactly 1.0, confirmed twice) was not predicted in
advance, which argues against reverse-engineering (a tuned-to-look-good
metric would more plausibly land somewhere unremarkable, not somewhere
extreme enough to need its own confirmation run to be believed).

**Documentation consolidation.** Fixed real doc/code drift found while
cross-checking claims against the running code and current `config.yaml`
(per `DOCS.md`'s own stated discipline, applied to itself): `README.md`'s
config table showed `prolongation_min_seconds` as `0.65`, the pre-Part-D
value — the real current default is `1.0` (git history + `config.yaml`
confirm); the table was also missing `phrase_repetition_max_words`,
`fusion_weights.*`, `detectors`, and all `acoustic.*` keys added across the
2026-08 restructuring and this evaluation phase. `README.md`'s and
`ARCHITECTURE.md`'s file-layout/references were updated: the
`profiling/evaluation/` package listing was missing `alignment.py`,
`track_b.py`, and `run_ablations.py` entirely (added since the listing was
last touched); `ARCHITECTURE.md` still pointed at the now-retired
`august.md` for the 2026-08 restructuring's reasoning trail instead of
`PAPER_DECISION_LOG.md`; a "pending validation on real recordings" note in
`ARCHITECTURE.md` §4a was stale — that validation has since happened
(`VALIDATION.md` §8.3), so the note was corrected to state what's actually
been validated and what hasn't (isolated component contribution, and real
non-synthetic stuttered speech — both still open). `VALIDATION.md` §3/§5/§6
still described Track B as "not yet built" / "not started" in several
places predating its actual construction — corrected throughout. `VALIDATION.md`
§8.1's run log was missing rows for the ablation study and both
context-strict/scaled Track B runs — added. Added `CLAUDE.md` (new) as a
short, stable orientation file read automatically at the start of future
Claude Code sessions in this repo, pointing at `DOCS.md` as the full map —
intended to reduce the chance of a future session re-deriving context that
already exists in these files.

**Alternatives considered**
- Leave the $R_A$ approximation as-is, since it was already honestly
  flagged as an approximation. **Rejected**: the whole point of a closing
  review is to fix what's cheap to fix, not just re-confirm existing
  caveats — and this one was genuinely cheap (no new ASR inference needed).
- Fix the speaker-clustering and single-ASR-backend gaps in this same
  session, since they were found during this review. **Rejected, deferred
  with reasoning**: both require substantial new data collection/inference
  work explicitly out of scope for a session the project owner scoped as
  "finalize Phase 1, do not improve the model" — fixing them now would
  blur exactly the phase boundary this closing session exists to establish.
  Recorded as the top two Phase 2 priorities instead of silently noted or
  skipped.
- Treat the critical review as complete once existing caveats were
  re-read. **Rejected**: explicitly checked for *new* gaps (the speaker
  count, the localization gap, the circularity question) rather than only
  restating what earlier entries already said — a review that only
  confirms prior conclusions without checking for anything new is not
  actually a critical review.

**Why this choice**
Directly follows the project owner's explicit request: challenge prior
assumptions the same way this project has throughout, distinguish
confirmed findings from open generalization questions, implement what's
feasible and justified now, and defer — with explicit reasoning, not
silently — what genuinely belongs in Phase 2.

**Measured result**
`VALIDATION.md` §7.2–§7.4 (critical review), §8.4/§8.4.1/§8.4.2 (updated
decomposition tables), §8.1 (run log additions). New exact-subset Track A
results: `eval_results/20260803T164441410299Z_libristutter_A+audio-
first30-matched-to-trackB.json`,
`eval_results/20260803T164555961127Z_libristutter_A+audio-first90-matched-
to-trackB.json`. `README.md`/`ARCHITECTURE.md` drift fixes (see above).
New `CLAUDE.md`. Full regression suite re-verified after all doc-only
changes (no source code was modified in this entry — `detect.py`,
`acoustic.py`, and the evaluation package's logic are all unchanged from
the previous entry).

---

## 2026-08-03 — Phase 2 opening literature review: is our taxonomy scientifically sound?

**What was done**
Before any Phase 2 implementation, the project owner asked whether the
current 7-type taxonomy (`filler`, `sound_repetition`, `word_repetition`,
`phrase_repetition`, `block`, `prolongation`, `stutter_marker`) is
scientifically optimal, or an artifact of Phase 1's own baseline that
hasn't been checked against the literature. Reviewed: clinical
speech-pathology taxonomy (the stuttering-like vs. other-disfluencies
distinction, Ambrose & Yairi's framework), acoustic/computational detection
literature (per-type detection strategies, a recent rule-based-detection
preprint directly comparable to this project's own architecture), and
re-confirmed dataset annotation conventions (SEP-28k, FluencyBank, KSoF,
UCLASS) against a wider literature sweep than Phase 1's original dataset
comparison. Full write-up, citations, and the resulting structured Phase 2
plan: `PHASE_2_RESEARCH_PLAN.md` (new file — this is a literature review
and research plan, not a results/methodology document, so it doesn't belong
in `VALIDATION.md`).

**Headline conclusion: the core 5-type taxonomy is scientifically sound
and correctly matches the field — no wholesale redesign is justified.**
Every major dataset reviewed (SEP-28k, FluencyBank, KSoF) uses essentially
the same 5 types (block, prolongation, sound repetition, word repetition,
interjection) Phase 1 already aligned with. This review's value is in what
it found *within* and *around* that already-correct core, not in
overturning it:

1. **A real, literature-backed distinction our `word_repetition` type
   ignores**: clinically, monosyllabic word repeats ("her-her-her") are
   stuttering-like (motoric), while polysyllabic word repeats are
   classified as ordinary linguistic-planning disfluencies, not
   stuttering — a distinction with real diagnostic significance that costs
   nothing to add (syllable count is already computed elsewhere in this
   codebase for `profile.difficulty()`) and doesn't break any existing
   dataset benchmark (additive metadata, not a new required label).
2. **A real, previously-undocumented architectural gap**: verified directly
   against `profiling/detect.py`/`acoustic.py` that the `block` detector
   only implements the *silent* block sub-type (`gap_is_silent()` is a pure
   RMS-below-threshold check) — the literature identifies a second,
   acoustically distinct "audible/struggle" block sub-type (sustained
   low-amplitude tension energy, not silence) that this project has no code
   path for at all. Also confirmed as the type where even the literature's
   own rule-based systems perform worst (largest neural-vs-rule-based gap
   of any type) — a genuinely hard problem generally, not unique to this
   codebase.
3. **A convergent, three-source case for prolongation as Phase 2's
   highest-confidence detector-side target**: Phase 1's own ablation study
   already found `prolongation_min_seconds` dominates measured performance
   by an order of magnitude; the literature independently identifies
   prolongation as the type rule-based methods handle best, given
   speaking-rate normalization and multi-feature (spectral-stability +
   F0-stability + HNR) core detection rules instead of duration-threshold
   alone; and prolongation is the type with the strongest, most uniform
   dataset support across every candidate corpus. This project already
   computes most of the needed Praat features but only uses them as
   post-hoc confidence adjustments (confirmed inert to the current metric
   by Phase 1's own §9.3 finding) — using them as *core* detection criteria
   instead is the specific, well-evidenced candidate change identified.
4. **Independent, cross-project corroboration of two things Phase 1 already
   found on its own**: a completely different codebase/dataset's error
   analysis reports the same class of confusion this project's Track B
   analysis found (adjacent disfluency types — sound-repetition-before-
   prolongation, word-repetition-vs-phrase-repetition — getting conflated),
   and a separate synthetic-disfluency-data study independently confirms
   the generalization concern Phase 1's own closing review raised about
   LibriStutter's synthetic splicing (`VALIDATION.md` §7.2 item 3). Neither
   changes a prior conclusion; both raise confidence the prior conclusions
   were identifying real, general phenomena, not project-specific quirks.
5. **Confirmed, not newly found: `phrase_repetition` and `stutter_marker`
   have zero dataset support anywhere in the literature reviewed** (already
   known from Phase 1's own dataset comparison, now confirmed across a
   wider sweep) — and `phrase_repetition` specifically maps to the clinical
   "Other Disfluencies" category, i.e. lower diagnostic significance even
   where clinically recognized at all, not just an annotation gap.
6. **One existing architectural choice validated, not challenged**: the
   field's own literature (Bayerl et al., "A Stutter Seldom Comes Alone")
   argues stuttering detection should be a multi-label problem, not forced
   single-class — this project's event structure already allows multiple
   simultaneous labels per token. No change indicated.

**Recommendation on subset focus**: yes — Phase 2 should concentrate
detector-side effort on a deliberately narrowed, evidence-ranked subset
rather than optimizing all 7 types at once. Full ranked list and reasoning:
`PHASE_2_RESEARCH_PLAN.md` §6.

**Recommendation on Phase 2's actual opening step**: not a single answer —
a specific ordering across taxonomy refinement, ASR validation, and
detector redesign, not one winner-take-all direction. (1) Cheap,
low-risk taxonomy/documentation refinements first (the syllable-count
sub-tag, explicit dataset-validation-status labeling, documenting the
silent-only block gap) — additive, no new Track A/B runs needed to trust.
(2) In parallel, Phase 1's own already-top-ranked `ROADMAP.md` priority
(ASR-backend/speaker-diversity validation) proceeds unchanged — this
review reinforces why it matters rather than superseding it. (3) The
prolongation core-detection redesign, evidence-gated on (2)'s outcome and
pre-registered before implementation, per this project's standing
discipline — the single highest-confidence architecture-level change this
review found. (4) The already-scoped word/sound-repetition
type-classification fix. Full reasoning: `PHASE_2_RESEARCH_PLAN.md` §7.

**Alternatives considered**
- Treat this as confirmation to proceed straight to detector implementation
  (e.g., start building the prolongation redesign immediately).
  **Rejected**: the project owner was explicit that this session is
  philosophy/planning only, and Step 3 is itself explicitly gated on Step
  2's outcome (ASR-backend validation) — implementing now would pre-empt
  evidence not yet gathered.
- Treat the taxonomy as needing a deeper redesign (e.g., splitting
  `word_repetition` into two first-class types instead of a computed
  sub-tag on the existing type). **Rejected for now**: no dataset labels
  this split directly, so a first-class type split would create a type
  with *weaker* dataset validation than the sub-tag approach, for the same
  clinical information — the sub-tag achieves the literature-motivated
  goal without the compatibility cost.
- Build audible/tense block detection now, since it's a real, clearly
  identified gap. **Rejected, deferred with reasoning**: no available
  dataset can validate it (§5 of the research plan), which would repeat
  the exact anecdotal-validation mistake Phase 1 was built to avoid.
  UCLASS's possible finer annotations are flagged as worth verifying
  directly before deciding this is buildable with real validation at all.
- Rely on the two preprints found (arXiv:2508.16681, arXiv:2505.22029) as
  settled findings. **Rejected**: explicitly weighted as suggestive,
  non-peer-reviewed evidence, given the most weight where they
  independently corroborate something this project already found on its
  own rather than standing alone.

**Why this choice**
Directly answers what the project owner asked: do not assume Phase 1's
taxonomy is optimal just because it was the baseline; check it against the
literature; keep any redesign compatible with the existing validation
strategy; decide whether Phase 2 should focus on a subset; and produce a
structured, evidence-justified plan before any implementation begins.

**Measured result**
No code changed — this is a research/planning entry. Deliverable:
`PHASE_2_RESEARCH_PLAN.md` (full literature review, gap analysis,
dataset-compatibility analysis, and structured plan). `ROADMAP.md` updated
to reflect the refined Phase 2 ordering (§ pointer in that file). Full
regression suite not re-run for this entry (no code touched) — last
verified green in the previous entry.

---

## 2026-08-03 — Adversarial self-review of the Phase 2 plan, and its first implementation milestone (Step 1)

**What was done**
Per the project owner's explicit instruction, actively tried to disprove
`PHASE_2_RESEARCH_PLAN.md`'s own conclusions before proceeding to
implementation — not treat the prior literature review as self-evidently
correct. Re-searched specifically for counter-evidence: stronger/
peer-reviewed sources for the prolongation-first claim, alternative
taxonomies (continuous/phonetic dysfluency representations, e.g. SSDM/
Dysfluent-WFST), and the field's dominant deep-learning trend as a
challenge to staying rule-based. Full write-up: `PHASE_2_RESEARCH_PLAN.md`
§9.

**Result: the plan's ordering held up, but its evidentiary grounding was
measurably weak in one place and is now fixed.** The original
prolongation-first case leaned on a single non-peer-reviewed preprint.
Found two independent, peer-reviewed sources that were missing: Esmaili et
al. 2017 (*J. Medical Signals and Sensors*, 99%/97.1% prolongation
accuracy on UCLASS/Persian corpora via the same rate-normalization
technique) and a genuine PMC-indexed systematic review of 14 studies
independently confirming prolongation/interjection are the easiest types
to detect reliably and blocks the hardest. The preprint is demoted to
tertiary evidence; these two are now the primary citations. A real
discrepancy was also caught and flagged rather than quietly resolved
(UCLASS's recording count is reported as 118 by one source, 457 by
another — not yet independently verified by this project). The
rule-based-vs-deep-learning tension was addressed directly rather than
ignored: the field's own systematic review confirms deep learning
dominates raw performance, but explainability is a co-equal, explicitly
stated project objective, so staying interpretable-first is a deliberate,
justified choice for this project specifically, not a universal claim — a
scoped, block-specific future role for pretrained embeddings (as an
auxiliary confidence signal, not a replacement) was identified as the most
targeted place a learned component could eventually go, if a suitable
dataset ever exists to validate it. **No direction found (continuous
taxonomy, deep-learning-first, different dataset priority, different first
step) was strong enough to change the plan's ordering.**

**First implementation milestone (Step 1 of the plan's §7): taxonomy/
documentation refinements, implemented and benchmarked.**

1. **`word_repetition` SLD/OD sub-tag.** `_word_repetition_extra()`
   (`profiling/detect.py`) computes `syllable_count` (via the existing
   `phonetic._syllable_count()`, already used by `profile.py`'s
   `difficulty()`) and `likely_sld` (`syllable_count <= 1`) for every
   `word_repetition` event — monosyllabic repeats tagged stuttering-like
   per the Ambrose & Yairi clinical framework, polysyllabic tagged an
   ordinary linguistic-planning disfluency. Wired into all three
   `word_repetition` call sites (exact back-to-back, near/phonetic repeat,
   filler-sandwiched repeat). Surfaced in `app.py`'s Event table as a
   "Class" column (SLD/OD), with an explicit caption stating it is a
   descriptive heuristic, not a validated clinical measure.
2. **Explicit dataset-validation-status labeling.** `README.md`'s taxonomy
   table now states plainly that `phrase_repetition`/`stutter_marker` are
   not annotated as a distinct category in any public benchmark dataset
   this project validates against — extended from `VALIDATION.md` §2 (where
   this was already noted) to the user-facing docs and the app's own event
   table caption.
3. **Silent-only `block` gap documented.** `ARCHITECTURE.md`'s known-
   limitations section and §4 now state directly (verified against the
   source, not inferred) that `block` detection has no code path for the
   literature's "audible/struggle" sub-type — confirmed by reading
   `_AcousticContext.gap_is_silent()`, a pure RMS-below-threshold check.

**Benchmarked against the frozen Phase 1 baseline, not assumed safe.**
Re-ran Track A on the identical 499-clip LibriStutter+audio sample used for
`VALIDATION.md` §8.3. Result: **byte-for-byte identical** to the frozen
baseline across every type and the `Any` label (`Any` TP=801, FP=308, FN=8,
Precision=0.722, Recall=0.990, F1=0.835 — matches §8.3 exactly). Confirms
the change is purely additive metadata with zero effect on detection or
scoring, as designed. New unit test
(`test_word_repetition_sld_tag_by_syllable_count`,
`tests/test_detect_taxonomy_and_fusion.py`) directly asserts the tag's
behavior on both a monosyllabic ("her her") and polysyllabic ("happy
happy") repeat; full suite now 39/39 (was 38/38), all pass.

**Alternatives considered**
- Treat the original literature review as sufficient without an adversarial
  pass. **Rejected**: the project owner explicitly asked for the plan to be
  actively challenged, and doing so found a real, fixable evidentiary gap
  (the preprint-only sourcing) — confirming the review was worth doing, not
  a formality.
- Make the SLD/OD tag a hard reclassification (e.g. split `word_repetition`
  into two first-class types) instead of additive metadata on the existing
  type. **Rejected**, consistent with `PHASE_2_RESEARCH_PLAN.md` §5's own
  reasoning: no dataset labels this split, so a first-class type split
  would have *weaker* dataset validation than the existing `word_repetition`
  type for the same clinical information: the sub-tag achieves the
  literature-motivated goal without that compatibility cost, and keeps the
  change provably non-breaking (confirmed by the identical-baseline result
  above).
- Skip the Track A re-benchmark since the change is "obviously" additive.
  **Rejected**: "obviously safe" is exactly the kind of unverified
  assumption this project's own discipline exists to catch — the benchmark
  cost minutes and turned an assumption into a measured fact.

**Why this choice**
Directly follows the project owner's instruction: challenge the plan
adversarially, update the documentation to reflect whatever the review
finds (reinforcement or revision), and then continue forward with the
strongest-evidenced first implementation step rather than treating the
review as a stopping point.

**Measured result**
`PHASE_2_RESEARCH_PLAN.md` §9 (adversarial review). Code:
`profiling/detect.py` (`_word_repetition_extra`, 3 call sites),
`app.py` (Event table "Class" column + caption), `README.md` (taxonomy
table), `ARCHITECTURE.md` (§4, known-limitations). Tests:
`tests/test_detect_taxonomy_and_fusion.py`, 11/11 (was 10/10). Full suite:
39/39 (was 38/38). Track A benchmark: identical to `VALIDATION.md` §8.3's
frozen baseline (`Any` F1 0.835, unchanged) — no raw result file saved for
this confirmation run (`--no-save`), since it intentionally reproduces an
already-recorded baseline rather than producing a new one.

---

## 2026-08-03 — Per-type definition audit: literature vs. dataset vs. implementation

**What was done**
The project owner asked for one more foundational check before further
Phase 2 implementation: for each of the 7 disfluency types, does our
code's exact operational trigger condition actually detect the phenomenon
as the clinical/scientific literature defines it, or does it only
approximate the dataset's own operational shortcut for labeling it — and
where these differ, why, and what should be done about it. Per the
owner's explicit "if you already did this, pull it up" instruction,
started by reviewing what `PHASE_2_RESEARCH_PLAN.md` §2–§4 already
established (taxonomy structure, per-type detection strategies, the
dataset-compatibility gap table) before doing new research, to avoid
re-deriving already-settled ground. Then did targeted new research
specifically on what hadn't been pinned down precisely: SEP-28k's exact
annotator-count-to-label convention (re-verified directly against
`profiling/evaluation/loaders.py`'s own schema comment, not just recalled),
LibriStutter's exact synthesis mechanism (Kourkounakis et al.: splicing
onto Google-Cloud-Speech-to-Text-timestamped LibriSpeech audio, "random"
placement — not modeled on real per-speaker disfluency statistics), a
clinical minimum-duration standard for prolongation (none universal found;
the field's dominant approach is speaking-rate-*relative*, not absolute —
Esmaili et al. 2017's validated `T_min = α/speaking_rate` formula), and
SSI-4's block-scoring criteria (duration + "physical concomitants" —
struggle signs like facial grimaces, not present in audio at all). Full
write-up, citations, and the per-type verdicts: `PHASE_2_RESEARCH_PLAN.md`
§10.

**Two real, previously-general findings sharpened into specific,
actionable gaps — both already this project's top two detector-side
priorities, now with precise mechanisms rather than general direction:**

1. **`prolongation`'s threshold is empirically-tuned, not
   literature-derived, and the gap is now quantified.** Computed precisely
   from `detect.py`: the effective threshold is
   `max(prolongation_min_seconds=1.0s, 90th-percentile-of-the-clip's-own-
   token-durations)`, and typical adult word durations rarely push a
   clip's own 90th percentile above 1.0s — so **the flat 1.0s floor is the
   binding threshold in most real clips**, not the percentile term. The
   literature's rate-normalized formula (Esmaili 2017,
   `T_min = 1.2/speaking_rate`) gives ~0.24–0.30s at typical conversational
   rate, ~0.48s even at a slow rate — our effective threshold is roughly
   2–4× higher. Not a mistake: it was a deliberate, documented response to
   real false positives on this project's own real-mic testing (Part D
   tune, this log, 2026-06-27) — but it means the current number is
   calibrated for a specific precision/recall trade-off on limited data,
   not derived from or validated against the rate-normalized standard the
   field treats as default. This sharpens (does not add to) the already-
   planned Step 3 prolongation redesign: specifically test rate
   normalization, evaluated via Track A/B, not just add Praat features as
   core criteria in the abstract.
2. **`block`'s silence-only rule tests a necessary-but-not-sufficient
   proxy for the clinical definition, and this project's own benchmark
   dataset shows the gap directly.** The clinical definition is
   effort/struggle-based (SSI-4 scores "physical concomitants" — signs not
   present in audio at all); SEP-28k's own CSV schema has a **separate
   `NaturalPause` column** distinct from `Block` (confirmed directly from
   the schema comment in `loaders.py`), meaning SEP-28k's trained
   annotators already make a pause-vs-block distinction our detector
   structurally cannot (`gap >= block_gap_seconds and gap_is_silent()` —
   pure duration + silence, no effort signal, no way to tell a thinking
   pause from a struggle). This is the same silent-only gap found in the
   2026-08-03 literature review (`PHASE_2_RESEARCH_PLAN.md` §2.2) — now
   with the specific reason a pure-silence rule is expected to
   underperform, not just the empirical observation that it does.

**Every other type's simplification is one this project's own benchmark
datasets also make** (`filler`'s word-list-vs-discourse-judgment gap,
`sound_repetition`'s lack of iteration counting, `stutter_marker` having
no external definition at all to diverge from) — recorded as known,
honest simplifications, not actioned, since a fix couldn't be validated
against any current dataset either. `word_repetition` was found already
aligned (the §7 Step 1 sub-tag). `phrase_repetition` was found to be an
unusual case where **this project's implementation is arguably more
faithful to the literature's real definition than any available dataset's
own label** (LibriStutter approximates it as a single-word marker; SEP-28k/
KSoF/FluencyBank have no equivalent column at all) — reframing its
"unvalidated" status as a dataset limitation, not an implementation one.

**General principle recorded for future definitional questions**: this
project's discipline is not "always move closer to the scientific
definition" — it is "move closer exactly where doing so remains
benchmarkable against real data, and otherwise document the simplification
honestly." Every gap found either matches a simplification the datasets
themselves already make, or was already a top priority before this audit
(`prolongation`, `block`) and is now sharper rather than newly discovered.

**Alternatives considered**
- Treat this as fully new research, ignoring what §2–§4 of
  `PHASE_2_RESEARCH_PLAN.md` already established. **Rejected**: the
  project owner explicitly asked to pull up prior work first — re-deriving
  settled ground would have wasted effort and risked introducing
  inconsistency with the existing review.
- Conclude uniformly that "our implementation should move closer to the
  scientific definition" as a single blanket recommendation. **Rejected**:
  the evidence genuinely differs by type — three types' simplifications
  match the datasets' own limitations (not fixable in a way that stays
  benchmarkable), two have a real, now-specific gap (already top
  priorities), one is already aligned, and one is arguably ahead of the
  datasets. A single summary verdict would have misrepresented this.
- Add new Phase 2 priorities for the `filler`/`sound_repetition`/
  `stutter_marker` simplifications found. **Rejected, consistent with §5's
  existing discipline**: none are validatable against any dataset this
  project has access to, so acting on them now would repeat the
  anecdotal-validation mistake Phase 1 was built to avoid — documented
  instead.

**Why this choice**
Directly answers the project owner's question: are we detecting what the
literature defines, or approximating what the datasets label — and for
each type, precisely which, with the reasoning made explicit rather than
left as an assumption, since these definitions are the conceptual
foundation the eventual research paper will need to state precisely.

**Measured result**
No code changed — this is a research/documentation entry, consistent with
its role as a foundational audit rather than an implementation step.
Full write-up: `PHASE_2_RESEARCH_PLAN.md` §10. Regression suite not
re-run for this entry (no code touched) — last verified green (39/39) in
the previous entry.

---

## 2026-08-04 — Speaker-stratified Track B: the "~0% detector-attributable" finding revised, not confirmed

**What was done**
Directly resolved the Phase 1 closing review's speaker-clustering caveat
(`VALIDATION.md` §7.2 item 2): the confirmed §8.4.2 conclusion
(`R_B|preserved_ctx1` recall = 1.0, ~0% detector-attributable gap) had
only ever been measured on 7 of the 499-sample's 40 speakers (a
deterministic clip-count prefix, speaker-ordered filenames). Pre-registered
a speaker-stratified sampling method in `VALIDATION.md` §5.1's addendum
before writing any code: round-robin across all 40 speakers, up to 3 clips
each, 120 total, explicitly stating "the finding holds/weakens/strengthens
are all valid outcomes" in advance. Implemented `_speaker_stratified_order()`
+ a `--speaker-stratified` CLI flag in `track_b.py` (4 new self-test
checks, 18/18 total), then ran it — 87 real ASR calls, 33 cache hits,
interrupted once mid-run by an unrelated session restart and resumed
cleanly from the per-clip cache (same resilience pattern as the earlier
90-clip run). Also computed the exact matched-subset Track A recall for
these same 120 clips (same discipline as the Phase 1 closing review's
$R_A$ fix) — $R_A$ = 185/186 = 0.9946, the first time this project's
exact-subset $R_A$ was not a clean 1.0.

**Headline: the finding weakened, exactly as the pre-registration allowed
for.** `R_B|preserved_ctx1` recall dropped from 1.0 (both 7-speaker
samples) to **0.667 (10/15)** at full 40-speaker diversity. Decomposition
revised from "~0% detector-attributable / 100% ASR-attributable" to
**35.1% detector-attributable / 64.9% ASR-attributable**. ASR-fidelity
remains the majority driver — the absolute Track A→B recall collapse and
its ASR-dominant attribution are not reversed — but "the detector is
essentially perfect given fair input" was too strong a claim, resting on
a 7-speaker sample that turned out not to generalize.

**The correction has a precise, hand-traced mechanism — not an
unexplained regression.** Direct inspection of all 15 context-strict-
preserved instances (via the per-clip cache, no new ASR needed) found the
5 misses are exclusively `sound_repetition` (2: `445-123857-0019`
"after-", `5456-24741-0023` "itself-") or `phrase_repetition` (3:
`2836-5354-0011` "would", `289-121652-0010` "boy", `445-123857-0019`
"and"). **Zero of the 5 `word_repetition`-true instances in this subset
were missed** — `word_repetition` remains at 100% Any-level recall given
intact context across 10 instances observed to date, across two
independent sample-construction methods (clip-count prefix and
speaker-stratified). Both miss categories trace to **already-known,
already-documented structural gaps with no ASR-context component at
all**: `sound_repetition`'s fragment-ordering mismatch (`VALIDATION.md`
§8.2, already a `ROADMAP.md` item) and `phrase_repetition`'s LibriStutter
single-word-reconstruction limitation (§8.2, already flagged as needing
Track B to measure honestly — this run *is* that honest measurement, and
it confirms the concern was real). The earlier n=2/n=7 samples happened,
by chance (not selection — both were deterministic prefixes chosen before
seeing results), to be `word_repetition`-heavy; the larger, more
type-diverse sample simply had more opportunity to expose gaps that were
already on record.

**Revised, precise headline**: once speaker- and type-diversity are both
accounted for, roughly a third of the context-strict gap is
detector-attributable — but that third is fully explained by two
pre-existing, already-scoped issues specific to `sound_repetition`/
`phrase_repetition`, not `word_repetition` or a general detector weakness.
This sharpens (does not contradict) `PHASE_2_RESEARCH_PLAN.md`'s existing
priority ordering: `sound_repetition`'s fragment-ordering fix and
`phrase_repetition`'s known reconstruction-caused unvalidatability are now
supported by *both* Track A and Track B evidence, not Track A alone.

**Alternatives considered**
- Treat the original §8.4.2 "confirmed" conclusion as still standing since
  the *absolute* ASR-attributable majority didn't change. **Rejected**:
  the specific claim that was stated as "confirmed" — effectively-perfect
  detector recall given intact input — did change materially (100% → 66.7%
  recall on the relevant subset), and reporting only the part that didn't
  move would understate a real, useful correction.
- Report this as "the detector regressed" or investigate it as a new bug.
  **Rejected, checked directly**: hand-tracing found no new failure mode —
  every miss maps cleanly to a structural gap already in `VALIDATION.md`
  §8.2 and `ROADMAP.md`, confirmed by clip name and mechanism, not
  asserted from aggregate counts alone.
- Silently update §8.4.1/§8.4.2's historical numbers to the new figures.
  **Rejected**, consistent with this project's discipline throughout:
  those sections are kept as the accurate record of what n=2/n=7 showed at
  the time, with inline dated pointers to §8.4.3 added rather than the
  history being rewritten.

**Why this choice**
This is the pre-registration discipline doing exactly what it exists for:
the protocol explicitly allowed for "the finding weakens" as a valid
outcome before any data was collected, and when that outcome occurred, it
is reported as found — not reframed, not buried, and not treated as
invalidating the pre-registration process that surfaced it. The Phase 1
closing review's own critical-review item (§7.2 item 2) existed
specifically to catch exactly this kind of speaker-generalization risk; it
did.

**Measured result**
`VALIDATION.md` §8.4.3 (full numbers, decomposition, hand-traced hit/miss
detail), §7.2 item 2 (updated to reflect resolution), top summary block
(revised). Code: `profiling/evaluation/track_b.py`
(`_speaker_stratified_order`, `_speaker_id`, `--speaker-stratified` CLI
flag, `run()`/`main()` wiring). Self-test: 18/18 (was 14). Raw results:
`eval_results/20260804T060338639222Z_libristutter_B.json` (Track B),
`eval_results/20260804T060635995226Z_libristutter_A+audio-speaker-
stratified-120-matched-to-trackB.json` (exact-subset Track A). Cache:
`eval_datasets/_track_b_cache/` (all 120 clips now present, reusable for
future metric refinements without re-running ASR).

---

## 2026-08-04 — Whisper "did not predict an ending timestamp" warning: investigated, confirmed external

**What was done**
This warning has printed on most clips across every real-ASR run this
project has done (Track B pilots, scaled runs, speaker-stratified run) and
was never investigated — the project owner asked directly whether this
project's own pipeline could be contributing (e.g. via truncation) before
continuing to treat it as background noise. Investigated with a bounded,
evidence-based approach rather than assuming either way:

1. Grepped this project's own codebase for the warning string first — no
   match — then searched the installed `transformers` package directly
   and found it at `transformers/models/whisper/tokenization_whisper.py`
   line ~1101, inside `_decode_asr`'s chunk-stitching/consolidation logic.
   **Confirmed library code, not ours**, by reading the source directly.
2. Identified the one concrete, plausible way this project's own pipeline
   *could* cause it: `profiling/asr.py`'s `_max_new_tokens_for()` caps
   generation at `max(20, min(256, duration_seconds*6 + 20))` — if that
   cap were too tight, generation could be cut off before a clean ending
   timestamp, and the warning's own text ("cut off in the middle of a
   word") would be consistent with that mechanism.
3. **Ruled this out by direct measurement, not assumption**: pulled actual
   generated-token counts from already-cached real transcriptions
   (`eval_datasets/_track_b_cache/`, no new ASR needed) and compared
   against each clip's computed budget. Every clip checked used only
   ~30–50% of its allotted budget regardless of whether that clip showed
   the warning — e.g. a 15.6s clip budgeted 113 tokens generated only 47.
   Generation is not hitting the cap on any clip checked, so the token
   budget is not the cause.
4. Read the triggering code's actual condition: the warning fires when a
   "leftover" token sequence at the very end of `transformers`' internal
   long-form decoding never received a paired closing timestamp token — a
   property of how Whisper's own decoder terminates on that specific
   audio content, not of any generation parameter this project sets (this
   pipeline deliberately does not set `chunk_length_s`, confirmed already
   documented in `ARCHITECTURE.md` §3's "critical settings" note).
   Mechanistically consistent with LibriStutter's own clip construction
   (extracted/spliced windows, not natural utterance boundaries —
   `PHASE_2_RESEARCH_PLAN.md` §10 — so some clips legitimately end mid-word
   by construction).

**Conclusion: external, confirmed by evidence — not this project's
pipeline.** No fix needed. The affected word is not lost (`_decode_asr`
still resolves and returns the leftover tokens immediately after the
warning, in the same source function) — this project's own downstream
code already handles a missing `end` timestamp gracefully (`ARCHITECTURE.md`
§4's documented edge-case: tokens with `None` start/end are skipped safely
in duration/gap math, verified by existing tests). Purely a noisy but
harmless log line.

**Alternatives considered**
- Assume it's benign without checking, since it never crashed anything.
  **Rejected**: the project owner asked for it to be verified, not
  assumed, and "never crashed" doesn't rule out a subtler contribution
  (e.g. silently truncated words) that direct measurement was needed to
  exclude.
- Suppress the `transformers` logger to quiet the output, treating it as
  fixed. **Rejected/deferred**: no functional problem was found to justify
  a change; recorded as a low-risk, optional cosmetic follow-up in
  `ARCHITECTURE.md` if the noise ever becomes a real nuisance, not done
  speculatively now.
- Spend further time cross-referencing public `transformers`/Whisper
  issue trackers for independent confirmation this is a known community-
  reported behavior. **Not done, per explicit scope instruction** ("don't
  waste time on this") — the direct, first-party evidence (reading the
  triggering source, measuring actual token usage against budget) was
  already conclusive without needing external corroboration.

**Why this choice**
Directly answers what was asked: verify, don't assume, whether this
project's pipeline contributes to a repeatedly-observed warning; document
the evidence and conclusion since it came out clearly external; do not
over-invest time once the answer was conclusive.

**Measured result**
No code changed — investigation and documentation only. Full account:
`ARCHITECTURE.md` §3, "A recurring console warning, investigated and
confirmed external." Evidence: per-clip generated-token counts pulled
directly from `eval_datasets/_track_b_cache/`'s existing cached results
(no new ASR run needed); triggering source line identified in the
installed `transformers` package.

---

## 2026-08-04 — Is a second ASR backend still necessary before detector-side work, or does current evidence already justify proceeding?

**What was done**
`ROADMAP.md` item 2 ("validate the ASR-is-the-bottleneck conclusion
against a second ASR backend and/or real disfluent speech") has gated
item 4 (the prolongation redesign) since the Phase 1 closing review. The
project owner asked for an explicit, evidence-based re-examination of
whether that gate is still scientifically necessary now that item 3
(speaker-stratified Track B) is done and has materially changed the
picture — rather than mechanically working through the roadmap in its
original order.

**Re-examined what the gate was actually for, and what has changed.** The
gate's original stated purpose (`ROADMAP.md`, prior wording): "if
ASR-fidelity is confirmed even more dominant across backends/speakers,
that should size how much relative effort [the prolongation redesign]
gets" — a *resource-sizing* judgment, not a claim that the redesign's
underlying evidence was invalid without it. Re-checked what that
redesign's own justification actually rests on
(`PHASE_2_RESEARCH_PLAN.md` §6, §10.6): Phase 1's ablation study
(`VALIDATION.md` §9, Track A — no ASR involved at all) already found
`prolongation_min_seconds` dominates measured performance by an order of
magnitude, and the redesign's specific direction (rate-normalization) is
grounded in peer-reviewed literature (Esmaili et al. 2017) that is itself
backend-agnostic. **Neither piece of evidence depends on which ASR backend
this project eventually uses.** The same check applied to the newly-
elevated `sound_repetition`/`phrase_repetition` fixes (`ROADMAP.md` item
10): both bugs were originally found and are fully explained via Track A
alone (`VALIDATION.md` §8.2, §8.4.3) — a fragment-ordering mismatch and a
dataset-reconstruction limitation, neither of which involves ASR at all,
let alone a *specific* ASR backend.

**Conclusion: the gate is lifted for the already-identified detector-side
fixes — their evidentiary basis does not depend on resolving the
cross-backend question. The cross-backend question itself remains open
and valuable, but is re-scoped from "blocking prerequisite" to "external-
validity/generalization strengthening," and de-prioritized below the
now-concretely-justified detector fixes.** Reasoning, stated precisely so
the boundary of this decision is clear: what remains *unproven* without a
second ASR backend is the *general* claim "ASR fidelity is the dominant
real-world bottleneck for disfluency detection *in general*" — that
specific, broad claim should still carry the existing caveat
(`VALIDATION.md` §7.2 item 3, unchanged by this decision) in any future
paper. What does *not* remain unproven is the narrower, already-actionable
claim "these three specific, named issues (`prolongation`'s threshold,
`sound_repetition`'s fragment-ordering, `phrase_repetition`'s
reconstruction limit) are real, backend-independent, and worth fixing" —
that rests on Track A and literature evidence alone. **Stress-tested this
conclusion directly** (would a dramatically different second-backend
result change anything here?): even in the extreme case where a different
ASR backend showed a completely different ASR-attributable/detector-
attributable split, the three specific bugs above would not disappear or
become less true, since Track A's evidence for them never involved ASR at
all — so no plausible second-backend result invalidates proceeding with
them now.

**Alternatives considered**
- Keep the gate exactly as originally worded and complete the second-ASR-
  backend work before touching any detector code. **Rejected**: this
  would mechanically follow the roadmap's original order rather than the
  evidence's actual dependency structure, which is exactly the instruction
  given — "follow the evidence rather than simply the implementation
  order." The gate's own stated reason (resource-sizing) doesn't support
  blocking work whose justification is already independently established.
- Drop the second-ASR-backend/FluencyBank item entirely, treating it as
  no longer valuable. **Rejected**: it still answers a real, currently-
  open question (does the ASR-fidelity finding generalize beyond
  CrisperWhisper) that matters for how broadly this project can eventually
  claim its central finding — just not one that needs to be answered
  before the three already-justified fixes proceed. Kept on `ROADMAP.md`,
  re-prioritized rather than removed.
- Treat this as license to also un-gate work whose evidence *isn't*
  independently ASR-backend-agnostic. **Rejected — checked explicitly**:
  this decision is scoped narrowly to the three items whose justification
  was directly verified as Track-A/literature-only; it does not blanket-
  lift the gate for anything not specifically checked this way.

**Why this choice**
Directly follows the project owner's explicit instruction to decide based
on scientific merit rather than roadmap order, and to document the
reasoning rather than just the conclusion — the decision here is not "the
second ASR backend doesn't matter," it's "the specific work items
currently gated on it don't actually depend on it, verified by tracing
each one's evidence back to its source."

**Measured result**
No code changed — a scope/prioritization decision, not an implementation
step. `ROADMAP.md` updated: items 4, 5, 10 un-gated and reordered ahead of
the second-ASR-backend item; that item (now later in the list) reworded
to reflect its revised role (generalization strengthening, not a
blocker). Full reasoning duplicated at the point of change in `ROADMAP.md`
itself, not just here, so a reader following the roadmap sees the
justification without needing this log entry.

---

## 2026-08-04 — sound_repetition fragment-ordering fix: root cause deeper than documented, fixed and measured; a related cache-staleness bug found and fixed alongside it

**What was done**
`ROADMAP.md` item 3 (Phase 2's top un-gated priority) called for fixing
`sound_repetition`'s confirmed 0% Track A recall, previously documented as
an ordering problem: the detector's fragment-repeat check only handled
"fragment-before-word," while the LibriStutter reconstruction places the
fragment after the word. Before implementing the previously-proposed fix
(add a reverse-order check), tested the actual reconstruction convention
directly (`loaders.py`: a sound_repetition fragment is reconstructed as
the adjacent complete word's text plus a trailing `-`, not a genuine
partial substring) against both orderings. **Found the real cause is
deeper than documented**: because `_norm()` strips the trailing `-`, a
reconstructed fragment normalizes to a string identical to its
complete-word counterpart, and `detect_disfluencies()`'s existing
exact-match `word_repetition` check ran *before* the fragment-specific
logic in the `if/elif` chain — so it intercepted the pair and produced
`word_repetition`, not `sound_repetition`, in **both** orderings, not just
the undocumented one. A reverse-order check alone would not have fixed
this. Pre-registered the corrected understanding and the fix's predicted
direction in `VALIDATION.md` §8.2 (a dated addendum) before writing any
code.

**Fix**: moved a unified fragment-pair check (either token ending in a
literal, pre-normalization `-`, with a prefix/equality relationship to its
neighbor) to run *before* the exact-match check, handling both orderings
in one branch (`profiling/detect.py`). Two new unit tests added
(fragment-after-word, the previously-undocumented case; fragment-before-
word, confirming no regression on the already-working case) — full suite
now 41/41 (was 39). One existing test's fixture-derived expectation was
stale and updated: the demo fixture's own `"buy"`/`"buy-"` pair had been
silently misclassified as `word_repetition` by the same bug and is now
correctly `sound_repetition` — `README.md`'s "Verify it works" walkthrough
updated to match.

**Measured result — matches the predicted direction, magnitude reported as
measured, not targeted**: same 499-clip, audio-enabled sample as
`VALIDATION.md` §8.3. `sound_repetition` recall **0.000 → 0.920** (184/200
recovered; TP 0→184, FP 1→5, FN 200→16), precision 97.4%. `word_repetition`
FP dropped 640→452 (−188) with TP/FN unchanged — the bug was inflating
`word_repetition` false positives, not its true positives, correcting this
entry's own pre-registration prediction ("TP should drop") against what
was actually measured, not silently. **`Any` (combined) label is
byte-for-byte unchanged** (TP=801/FP=308/FN=8/F1=0.835) — the expected
signature of a pure type-reclassification fix, consistent with this
project's repeated finding that binary detection and exact-type
classification are separate axes (`VALIDATION.md` §8.4.2/§8.4.3). Full
comparison table: `VALIDATION.md` §8.2.1.

**A second, unplanned finding surfaced while running this benchmark**:
`track_b.py`'s per-clip cache (`eval_datasets/_track_b_cache/`) was
discovered to store the *detector's output* (`events`), not just the ASR
output (`hyp_tokens`). This meant every one of the 210 clips already
cached across the 30/90/120-clip Track B runs held `events` computed by
the pre-fix detector — any future Track B run reusing that cache would
have silently kept scoring with the old, buggy `sound_repetition`
classification even after this fix shipped, with no error or warning.
**Fixed alongside this entry, not deferred**: `_save_cache`/`_load_cached`
now handle only `hyp_tokens`; `events` is always recomputed fresh from the
live `detect.py` on every run (cheap — no ASR involved, so this costs
nothing but a little CPU time per run). Old-format cache files (with a
now-unused `events` key) still load cleanly, confirmed by a new self-test
case. Track B self-test now 19/19 (was 18).

**Alternatives considered**
- Implement the originally-proposed fix (add a reverse-order check) without
  first testing the actual reconstruction convention. **Rejected**: would
  have shipped a fix that doesn't actually work, since the exact-match
  branch would still intercept both orderings — caught only because the
  fix was tested against the real convention *before* being written, not
  assumed from the existing (incomplete) diagnosis.
- Leave the Track B cache as-is, since fixing it wasn't the task at hand.
  **Rejected**: a cache that silently serves stale detector output after
  every future `detect.py` change is a correctness bug waiting to corrupt
  the next Track B result without any signal that it happened — cheap to
  fix immediately (no ASR involved) versus expensive to debug later (a
  confusing, silently-wrong Track B result with no obvious cause).
- Investigate and fix the 16 residual `sound_repetition` FN (8% of the
  original 200) in the same step. **Deferred, not rejected**: the primary,
  order-of-magnitude gap is closed (0%→92% recall); the residual is a
  secondary, much smaller problem better scoped separately, consistent
  with not over-extending a single change.

**Why this choice**
Directly follows `ROADMAP.md` item 3's evidence-based priority, this
project's pre-registration discipline (methodology and predicted direction
recorded before implementation), and its measurement-first discipline
(the cache bug was found by scrutinizing infrastructure while trusting a
result, not assumed fine because "it worked before").

**Measured result**
Code: `profiling/detect.py` (fragment-pair check reordering),
`profiling/evaluation/track_b.py` (`_save_cache`/`_load_cached` cache
format change), `tests/test_detect_taxonomy_and_fusion.py` (2 new tests,
1 updated), `README.md` (updated demo-fixture badge list). Full suite:
41/41 (was 39). Track A self-test: unaffected, still passing. Track B
self-test: 19/19 (was 18). Track A benchmark:
`sound_repetition` F1 n/a(0)→0.946, `word_repetition` F1 0.363→0.446,
`Any` F1 0.835 unchanged — no raw result file saved (`--no-save`, a
benchmark run, not a new tracked baseline). Full detail: `VALIDATION.md`
§8.2.1.

---

## 2026-08-04 — Hypothesis-side-contiguity metric built; a narrow detector extension implemented, measured, and reverted (negative result)

**What was done**
Continuing `ROADMAP.md` item 4 (the `word_repetition`/`sound_repetition`
type-classification gap), and per the item's own stated discipline
("candidate fix (b): build the hypothesis-side-contiguity-aware metric...
to confirm the fix actually helps before shipping it" — done before
candidate fix (a)): built a diagnostic metric (`gap =
hyp_index(i) - hyp_index(i-1) - 1`, computed directly from
`alignment.py`'s existing `AlignmentOp.hyp_index`) and re-scored all 120
already-cached speaker-stratified Track B clips with it — zero new ASR
calls. Pre-registered in `VALIDATION.md` §5.1's addendum before running.

**Metric result**: recall is 100% (2/2) when the ASR hypothesis sequence
is truly contiguous around a preserved disfluent word, dropping to 61.5%
(8/13) when an insertion breaks it — confirming the qualitative hypothesis
from the 2026-08-03 addendum was directionally real. Tracing the exact
`gap` size for the 5 outright misses found only **1 of 5** has a small
(`gap=1`) insertion consistent with "ASR corrupted an otherwise-tight
repeat" (`445-123857-0019`, "after-"/"after," one word — "life" —
inserted); the other 4 have `gap` of 3–5 words, too large to plausibly be
an ASR-insertion artifact and instead consistent with the already-
documented `phrase_repetition`/LibriStutter-reconstruction limitation
(§8.2) — a word recurring naturally later in a sentence, not a corrupted
close-proximity stutter. Two `gap=1` *hits* were also found
(`103-1240-0000` "Rachel", `1088-129236-0006` "the") — already caught at
the `Any` level via type-confusion, so their addressable benefit would be
exact-type accuracy, not recall. **Total addressable evidence: n=3 out of
120 clips.**

**Decision (made before seeing any benchmark result): implement a narrow,
conservative version anyway, and let a full 499-clip Track A benchmark
decide empirically** — n=3 is too thin to decide from directly, but the
mechanism is precise (not a guess), so a low-risk implementation plus
real measurement was judged the right way to resolve the question, rather
than either shipping on n=3 alone or refusing to try at all. Implemented:
extended the existing "filler-sandwiched repetition" `word_repetition`
check (already tolerates exactly one intervening *filler* word,
pre-existing code) to also tolerate exactly one intervening *non-filler*
word — exact-match only, length≥2 guard, lower confidence (0.65 vs. 0.89)
than the filler-sandwiched case. Two new unit tests added; full suite
43/43.

**Measured result: a clear, real regression on Track A, with essentially
no offsetting benefit on Track B.** Track A (499 clips): `Any` F1
**0.835→0.793** (`word_repetition` FP +106, `Any` FP +102), **zero new
true positives**. Track B (120-clip cache rescore, zero new ASR): +1 TP
at a cost of +24–29 new FP. **Reverted.** Root cause of the mismatch
between the plausible-sounding mechanism and the poor outcome: Track A
(no ASR) can only ever exercise this fix's *cost* (coincidental same-word
repeats across ordinary reconstructed text, e.g. two unrelated sentences
that happen to both start with the same word) since its *intended
benefit* requires real ASR errors to exist at all — and even on Track B,
where the intended benefit *can* appear, it appeared only once, at a
disqualifying false-positive cost.

**Alternatives considered**
- Skip building the metric and go straight to implementing/measuring
  candidate fix (a). **Rejected**: `ROADMAP.md` item 4 explicitly named
  the metric as the safer, evidence-gathering step to do first — building
  it (free, reused existing cached data) is exactly what let this session
  make a precise, mechanism-level decision (n=3, not n=15) rather than a
  cruder one.
- Ship the extension anyway, since its recall benefit on Track B was
  real (if small) and the false-positive cost is "just precision."
  **Rejected**: the pre-registered success criteria in this project's
  standing discipline treat aggregate F1 regression as disqualifying
  regardless of a small recall gain elsewhere — precision matters for a
  clinically-facing tool, and a 4-point aggregate F1 drop for +1 TP
  system-wide is not a defensible trade.
- Silently drop the idea without recording it, since it didn't pan out.
  **Rejected**: the project owner explicitly asked for negative results
  to be documented, not just positive ones — this is exactly that case,
  and the precise diagnosis (why a plausible mechanism didn't translate to
  a good outcome) is itself useful signal for future work on this area.
- Keep the code but disabled by default via a config flag, instead of a
  full revert. **Rejected**: no evidence suggests any configuration of
  this specific mechanism would help enough to be worth the added
  surface area/config complexity; a clean revert plus a locked-in
  regression test is simpler and equally reversible if new evidence
  emerges later.

**Why this choice**
Directly follows this project's standing discipline: measure before
committing to a change too far ahead of the evidence (the metric first),
implement carefully and validate objectively once a scoped version is
worth trying, and report the result honestly whichever way it comes out —
a reverted, well-documented negative result is exactly as valuable to the
project's research record as a positive one, and prevents the same
plausible-but-unvalidated idea from being tried again without new
evidence.

**Measured result**
`VALIDATION.md` §8.4.4 (metric results, gap table, and the full negative-
result writeup). Code: `profiling/detect.py` (extension added then
reverted, a code comment marks what was tried and points here),
`tests/test_detect_taxonomy_and_fusion.py` (2 tests added, then replaced
with 1 regression test locking in the reverted, correct behavior — net
+1 test vs. before this entry). Full suite: 42/42 (was 41 before this
entry). Track A benchmark: `Any` F1 0.835→0.793→0.835
(implemented, regressed, reverted — back to the sound_repetition-fix
baseline). Track B rescore: `eval_results/20260804T100451177010Z_
libristutter_B.json` (the with-extension result, kept for the record even
though the extension was reverted afterward — an honest snapshot of what
was measured, not deleted because the conclusion was negative).

---

## 2026-08-04 — Confidence-sensitive metric run against real data: VAD/Praat's designed confidence effect is not showing up (negative-to-null result)

**What was done**
`ROADMAP.md` item 7 asked for a confidence-sensitive metric so Silero VAD
and Praat corroboration — designed to adjust event *confidence*, not
presence/absence — could actually be evaluated, after `VALIDATION.md`
§9.3 found the project's only existing metric (`score_word_level`,
presence/absence TP/FP/FN) was structurally blind to that effect by
construction. Built `metrics.confidence_stats()` (mean predicted
`confidence` of TP vs. FP events, per type and combined `Any`), unit-
tested on hand-constructed synthetic data first (`track_a.py` self-test
section 7), then ran it against the full 499-clip real-audio LibriStutter
Track A sample under current production config (`use_vad=True`,
`use_praat=True`, both new prolongation-redesign toggles at their default
`False`) — the first time this metric touched real data.

**Result**: the TP-vs-FP confidence gap is approximately zero everywhere
it's measurable (`sound_repetition`/`word_repetition` exactly `+0.000` to
3 decimals, `prolongation` `+0.003`) and slightly *negative* for the
combined `Any` label (FP mean confidence 0.921 vs. TP mean confidence
0.914, gap `-0.007`). Audited before accepting, per this project's
standing rule 3 (a surprising result is a reason to check harder, not
report faster): confirmed the metric itself is not miscomputing (its
synthetic unit test correctly reproduces a designed non-zero gap), and
confirmed the run used the production config where VAD/Praat corroboration
are actually active (§9.1's ablation already showed disabling them doesn't
move presence/absence counts, so this was the right condition to test
their confidence-adjustment effect under).

**Alternatives considered**
- Treat the near-zero gap as inconclusive and not report it as a finding,
  since it's a negative result rather than a positive confirmation.
  **Rejected**: the project owner has explicitly asked for negative
  results to be documented with the same rigor as positive ones — this
  metric was built specifically to give VAD/Praat corroboration a fair
  chance to show a real effect the old metric couldn't see, and a
  near-zero/negative result *is* the answer to that question, not a
  non-answer.
- Immediately simplify or remove the VAD/Praat confidence-weighting logic
  in response to this finding. **Rejected**: standing rule 4 — findings
  are recorded as evidence, acting on them is a separate, explicitly-
  approved step. One dataset, one run is evidence, not proof the mechanism
  is worthless in general; flagged as a Phase 3 candidate decision instead
  of applied automatically.

**Why this choice**
Directly closes the specific open question `VALIDATION.md` §9.3 flagged:
whether "zero measured effect" on presence/absence meant VAD/Praat
corroboration don't help, or just that the metric couldn't see their
effect. Building the metric that *can* see it and finding it still shows
~zero effect is a materially stronger, more specific finding than leaving
the question open with only the "structurally blind" caveat.

**Measured result**
`VALIDATION.md` §9.3.1 (full table, per-type and `Any` gaps, audit steps,
and the explicit single-dataset/single-run limitation). Code:
`profiling/evaluation/metrics.py` (`confidence_stats()`),
`profiling/evaluation/report.py` (`format_confidence_stats()`), both
unit-tested in `track_a.py`'s self-test (section 7, new). No config or
detection-logic change made as a result. `ROADMAP.md` item 7 marked done.

---

## 2026-08-04 — Wilson confidence intervals applied to the project's own extreme-small-n numbers

**What was done**
`ROADMAP.md` item 8 (from the Phase 1 closing review, `VALIDATION.md`
§7.2 item 5) asked for confidence intervals on reported recall/precision
numbers, specifically naming `R_B|preserved_ctx1`'s repeatedly-cited
n=2/n=7/n=15 recall figures as the case where a formal interval would
make the existing "too few instances to trust" qualitative caveat
concrete. `metrics.wilson_interval()`, `TypeCounts.precision_ci()`/
`.recall_ci()`, and `report.format_table_with_ci()` were already built
and unit-tested earlier this session (paired with the confidence-stats
work above, both cheap/parallel/no-new-data items); this entry is their
first application to real, already-published numbers.

**Result**: computed Wilson 95% CIs for the three cited samples — n=2
(2/2, CI [0.342, 1.000]), n=7 (7/7, CI [0.646, 1.000]), n=15 (10/15, CI
[0.417, 0.848]). The n=7 and n=15 intervals overlap substantially
([0.646, 0.848]) — meaning the earlier "1.0 recall" point estimate at
n=7 was never precise enough to have ruled out something like the later
0.667 point estimate, exactly the concrete version of the caveat this
item asked for.

**Alternatives considered**
- Retrofit CIs onto every historical recall/precision number in
  `VALIDATION.md` §8, not just the specifically-named small-n cases.
  **Rejected**: would mean re-deriving raw k/n for many past runs from
  saved JSON result files for no new decision it would change — scoped
  down to "infrastructure exists, and is demonstrated at the exact point
  it was requested for," matching this project's general preference for
  scoped, evidence-driven work over exhaustive retroactive passes. Every
  *future* run automatically gets CIs via `report.save_run()`, so the
  backfill gap shrinks over time rather than needing a one-time sweep.

**Why this choice**
Turns a repeatedly-invoked qualitative caveat ("too few instances to
trust") into an auditable, reusable number, directly per the Phase 1
closing review's own request — and does so exactly where the review
named it as most valuable, without over-scoping into a full historical
backfill nobody asked for.

**Measured result**
`VALIDATION.md` §8.4.3 (new CI table and overlap discussion, inserted
directly after the existing recall table it explains). Code:
`profiling/evaluation/metrics.py` (`wilson_interval()`,
`TypeCounts.precision_ci()`/`.recall_ci()`), `profiling/evaluation/
report.py` (`format_table_with_ci()`, `counts_to_dict()` now saves CIs
for every future run). `ROADMAP.md` item 8 marked done (infrastructure
complete; full historical backfill explicitly out of scope, see above).

---

## 2026-08-04 — UCLASS annotation-schema verification: inconclusive from public sources, the specific "severity" claim is unsubstantiated

**What was done**
`ROADMAP.md` item 11 / `PHASE_2_RESEARCH_PLAN.md` §5 point 3 flagged a
specific, scoped question left open by the Phase 2 literature review: a
rule-based-detection preprint claims UCLASS has event-level "severity"
annotations that might imply a silent-vs-audible block sub-type split,
which would make audible/tense block detection (item 12) actually
validatable — not yet independently confirmed by this project. Checked
directly, in order: (a) the primary UCLASS archive paper (Howell et al.
2009, open-access PMC copy) — describes only recording-level perceptual
quality ratings (background noise, clarity), not event-level severity,
and points to an external "How We Transcribe" page for the actual
dysfluency-annotation conventions rather than specifying them inline;
(b) that external page (`speech.psychol.ucl.ac.uk`) — unreachable, TLS
certificate no longer matches the domain, link rot; (c) UCLASS's current
raw file-directory listing (`uclass.psychol.ucl.ac.uk/Transcript/
TAligned/Annotation/`) — reachable, lists downloadable SFS/CHAT/PRAAT-
TextGrid files but no accompanying methodology documentation; (d) the
originating rule-based-detection preprint itself — its own "severity"
claim cites only Howell et al. 2009, the same primary paper (a) shows
does not itself describe event-level severity.

**Conclusion: inconclusive from every public secondary source checked,
and the specific claim motivating this check is not substantiated by the
primary source it cites.** This does not prove UCLASS lacks a silent/
audible block distinction — only direct inspection of UCLASS's raw
annotation files (gated behind UCLASS's own access process) could
conclusively answer that, and that inspection was not attempted here
(materially larger effort than this scoped literature check).

**Alternatives considered**
- Attempt to acquire UCLASS's raw annotation files directly to settle the
  question conclusively. **Rejected for this step**: `ROADMAP.md` item 11
  was explicitly scoped as "a direct, cheap check," not a full dataset
  acquisition — UCLASS is Tier 3 (`VALIDATION.md` §2) specifically because
  of this access friction; escalating to a full acquisition is a
  separate, larger decision than this item authorized.
- Treat the preprint's claim as sufficient without independent
  verification, and proceed to build audible/tense block detection.
  **Rejected**: this is exactly the anecdotal-validation mistake Phase 1
  was built to avoid — building a detector with no dataset able to
  measure whether it works, on the strength of one secondary source's
  unverified characterization of a primary source.

**Why this choice**
Directly resolves a specifically-scoped, previously-open item from the
Phase 2 literature review with a dated, concrete negative result instead
of leaving it as an indefinite "not yet independently confirmed" — and
does so at the cheap-check scope the item actually called for, not by
over-investing in a full dataset acquisition the item didn't ask for.

**Measured result**
`PHASE_2_RESEARCH_PLAN.md` §5 point 3 (full addendum with the four-source
trace). `ROADMAP.md` items 11 (marked done) and 12 (audible/tense block
detection — status unchanged: still not started, this investigation found
no evidence it's currently validatable). No code change.

---

## 2026-08-04 — ASCII-console-output lint rule built; caught two real, previously-unnoticed violations immediately

**What was done**
`ROADMAP.md` item 13 flagged that the Windows `cp1252` console has broken
on non-ASCII characters in `print()` output three separate times across
this project (`track_a.py`, `report.py`, `track_b.py`), each fixed
reactively, and named a lint rule as the right fix instead of a fourth
reactive patch. Built `tests/test_ascii_console_output.py`: an AST-based
check (parses each file under `profiling/`, walks every `print()` call's
arguments, flags any string/f-string constant containing a non-ASCII
character) rather than a whole-file byte scan — checked first and
confirmed a whole-file scan would false-positive on 31 files that
legitimately use em-dashes in docstrings/comments (never sent to a
console), none of which caused any of the three prior incidents.

**Result**: running the new check immediately found two real,
previously-unnoticed violations — `profiling/benchmark_asr.py` lines
310-311, an ellipsis (`...`) and an em-dash inside `print()` f-strings,
never caught because that script isn't part of the evaluation-harness
paths exercised most often. Fixed both (replaced with `...` and `--`)
in the same change. Full suite now 45/45 (was 44/44 before this entry).

**Alternatives considered**
- A whole-repo, whole-file byte-level ASCII check (simpler to write).
  **Rejected**, verified empirically before choosing the AST approach:
  a dry run found 31 `.py` files with non-ASCII bytes, all from em-dashes
  in module-docstring headers (e.g. `"""report.py -- table rendering..."""`
  style comments already using em-dashes throughout this very codebase),
  zero of which are print-reachable or have ever caused the console bug.
  A rule that fails on legitimate, harmless comment content trains
  everyone to ignore it, defeating the point.
- Scan the whole repository (including `app.py`, the Streamlit UI) rather
  than just `profiling/`. **Rejected**: `app.py`'s output goes to a
  browser via Streamlit widgets, not this console — the three actual
  incidents were all in `profiling/`'s evaluation-harness scripts, and
  scoping the check to where the real risk lives keeps it precise.
- A pre-commit hook instead of a test-suite entry. **Rejected**: this
  project has no existing pre-commit hook infrastructure, and a test-suite
  entry runs on every full-suite invocation (the thing already run before
  every commit per this project's own discipline) with zero new tooling.

**Why this choice**
Directly implements what the roadmap item asked for ("worth a lint rule...
rather than fixing it reactively a fourth time"), scoped precisely enough
(AST-based, `profiling/`-only) to avoid false positives that would erode
trust in the check, and it proved its value immediately by catching a
real bug on its first run rather than only preventing hypothetical future
ones.

**Measured result**
`tests/test_ascii_console_output.py` (new, 1 test). `profiling/
benchmark_asr.py` (2-line fix). Full suite: 45/45 (was 44/44). `ROADMAP.md`
item 13 marked done.

---

## 2026-08-04 — Explicit scope decisions on remaining lower-priority Phase 2 items (9, 14-16): deferred to Phase 3, not silently dropped

**What was done**
Per the project owner's instruction to make explicit, documented scope
decisions on the remaining lower-priority `ROADMAP.md` items before
declaring Phase 2 complete, reviewed items 9 (Track B localization/IoU
metric) and 14-16 (LibriStutter sample expansion, SEP-28k audio
acquisition, FluencyBank Timestamped integration) against what Phase 2
actually needed to decide.

**Conclusion**: all four remain real and valuable but none was a
prerequisite for any decision Phase 2 made — every detector-side change
actually implemented this phase (prolongation redesign, `sound_repetition`
fix, confidence-stats/CI infrastructure) was fully evaluable on data
already in hand (the existing 499-clip LibriStutter sample, Track A's
existing localization metric). Explicitly deferred to Phase 3, each with
its own stated reasoning written directly into `ROADMAP.md` at the point
of the item (not just here) — mirroring exactly how Phase 1 closed
(`PHASE_1_SUMMARY.md` §6): evaluated and deliberately deferred because
each is real engineering/data-acquisition work, not a same-session fix,
not skipped by oversight.

**Alternatives considered**
- Attempt at least one of the dataset-acquisition items (14 is the
  cheapest) before closing Phase 2, to leave less deferred work.
  **Rejected**: none of these items is gated behind anything Phase 2 owns
  — they're independent, standalone follow-on work whose natural home is
  wherever Phase 3 starts, and starting one without a specific Phase-2
  decision that needs it would be scope creep against the owner's explicit
  instruction to complete Phase 2's own remaining scientifically justified
  improvements first, not open new fronts.
- Leave these items exactly as previously worded (no explicit "deferred"
  note), on the theory that an unstarted roadmap item is already
  self-evidently deferred. **Rejected**: the owner explicitly asked for
  *documented* scope decisions with reasoning, not just an unchanged list
  — the same distinction Phase 1's closing summary already established
  between "silently left implicit" and "explicitly scoped as a named,
  evidence-backed next step."

**Why this choice**
Directly follows the owner's explicit instruction and this project's own
established Phase-1-closing precedent for how to end a phase honestly:
every open item gets a stated, evidence-based reason for its status, not
a silent gap a future reader has to guess about.

**Measured result**
No code changed — a scope/documentation decision. `ROADMAP.md` items 9 and
14-16 updated in place with explicit 2026-08-04 deferral notes and
reasoning.

---

## 2026-08-04 — Prolongation redesign ablation run; Praat-gating adopted as the new default, rate-normalization rejected

**What was done**
Ran the 4-variant evaluation pre-registered directly in `VALIDATION.md`
§9.5 (methodology, predicted variants, and the exact success bar written
*before* the redesign's implementation — the implementation itself
(`use_rate_normalized_prolongation`, `require_praat_stability_for_
prolongation`, both new toggles defaulted `false`/off, unit-tested, full
suite green) is not a separate log entry, consistent with this project's
practice of recording pre-registrations in `VALIDATION.md` itself rather
than duplicating them here) as part of a full re-run of the standing
13-variant ablation harness (`run_ablations.py`), so the 3 new variants
are directly comparable to the existing 10-variant baseline. Full
499-clip real-audio LibriStutter sample, same as every other Track A
ablation this project has run.

**Result**: `prolong_praat_gated` (Praat pitch/jitter/shimmer gating
alone) is the *only* variant across the full 13 — not just the 3 new
ones — to improve both `Any` F1 (0.835->0.888) and prolongation-specific
F1 (0.064->0.084) simultaneously, the exact bar fixed in advance.
`prolong_rate_normalized` (rate-normalization alone) is a severe
regression on both metrics (`Any` F1 0.835->0.347, prolongation F1
0.064->0.048) — audited before accepting (a collapse this dramatic is
exactly the kind of surprising result this project's standing discipline
says to check, not just report): traced to a plausible root cause
(LibriStutter's short reconstructed clips likely destabilize the
per-clip speaking-rate estimate the formula divides by), not a bug in
the implementation, but not confirmed further since it was out of this
ablation's scope. `prolong_rate_and_praat` (both together) does not
clear the bar either — `Any` F1 still regresses (0.835->0.743) even
though prolongation F1 stays exactly flat, showing Praat-gating does not
fully rescue the rate-normalization component's damage when both run
together.

**Decision, mechanical given the pre-registered criteria**:
`require_praat_stability_for_prolongation` flipped to `true` in
`config.yaml` (new shipped default). `use_rate_normalized_prolongation`
stays `false` — it never clears the bar in any tested combination. Full
regression suite re-run after the config change: 45/45 pass unaffected.

**Alternatives considered**
- Adopt `prolong_rate_and_praat` (both changes together) anyway, since it
  still includes the literature-grounded rate-normalization mechanism in
  principle. **Rejected**: fails the pre-registered bar outright (`Any`
  F1 regresses) — the pre-registration explicitly ruled out "looks good
  in isolation" as a sufficient reason before this ablation ran, exactly
  to prevent this kind of post-hoc rationalization.
- Investigate and fix the rate-normalization formula's apparent
  instability on short clips before deciding, rather than rejecting it
  outright this session. **Rejected for now**: out of the scope this
  pre-registration authorized (a 4-variant ablation to decide the
  default, not an open-ended debugging session); the code stays in place,
  fully toggleable and off by default, so this remains available as
  future work if a real-speech (non-reconstructed-timing) dataset ever
  becomes available to re-test it against — noted directly in
  `config.yaml`'s comment for the toggle.
- Treat the `prolong_praat_gated` win as too good to be true and re-run
  before trusting it. **Considered, not needed**: the magnitude (+0.053
  `Any` F1, +0.020 prolongation F1) is large but not implausibly so given
  §9.1's own finding that FP suppression is exactly this architecture's
  designed strength, and the mechanism (screening out false positives
  with acoustically inconsistent pitch/jitter/shimmer) is directly
  traceable in the FP count itself (409->145) — a plausible, explicable
  result, not an unexplained anomaly like the rate-normalization
  collapse was.

**Why this choice**
This is exactly what pre-registration is for: the decision criteria were
fixed in `VALIDATION.md` §9.5 before any variant was run, so accepting
`prolong_praat_gated` and rejecting the other two required no new
judgment call once the numbers came in — only mechanical application of
a rule set in advance, immune to being unconsciously bent toward a
preferred outcome after seeing the results.

**Measured result**
`VALIDATION.md` §9.5.1 (full results table, per-variant analysis,
decision). `config.yaml` (`require_praat_stability_for_prolongation:
true`, comment updated with the result; `use_rate_normalized_
prolongation` stays `false`, comment updated with the result and the
root-cause hypothesis). `VALIDATION.md` §8.3's frozen baseline table
given a dated superseded-by pointer to §9.5.1 (not edited in place, same
discipline as every other historical revision in this file). `ROADMAP.md`
item 5 marked done. Full suite: 45/45 (unchanged by this config-only
change).

---

## 2026-08-04 — Pre-Phase-3 architecture review: is ASR-first the right foundation? Kept, with a scoped extension identified

**What was done**
Before starting Phase 3, the project owner asked for a first-principles
challenge to the ASR-first (two-stage ASR-then-detector) architecture
itself, not just its individual components — explicitly not assuming the
current design is correct, and explicitly not recommending a redesign
just because newer models exist. The question was framed precisely: not
"which ASR," but "what representation of speech gives the detector the
highest possible accuracy for detection, classification, and
localization." This revisits the open end of a question this project
already partially answered once, on 2026-08-03 (the "Vision alignment
review + architecture decision" entry above), which rejected end-to-end
audio-native replacement and deferred a learned (SSL) tier pending a
baseline — that baseline now exists.

Conducted a fresh literature pass (2024-2026 papers, arXiv/ISCA), covering
every direction the owner named: newer Whisper-family models, disfluency-
trained ASR, richer alignment techniques, self-supervised representations
(wav2vec2/HuBERT/WavLM), acoustic embeddings, hybrid acoustic+linguistic
architectures, fully audio-native approaches, and joint ASR+detection
training — cross-referenced against this project's own Phase 1/2 empirical
findings (the Track A/B recall gap and its 35.1%/64.9% detector/ASR
attribution split at full speaker diversity; the prolongation redesign's
proof that moving from token-duration to a genuine acoustic gate improved
both aggregate and type-specific F1; the word-sandwiched-repetition
negative result as a data point against "more text-side cleverness" as a
fix). Full review, citations, and reasoning: `PHASE_3_ARCHITECTURE_
REVIEW.md`.

**Conclusion: the two-stage architecture is kept as the foundation — not
by default, but because every alternative considered costs real,
currently-unavailable infrastructure (training pipeline, GPU, joint-
labeled data) or explainability this project's results have specifically
benefited from, without a decisively better accuracy result at this
project's actual task granularity once each alternative's own reported
precision problems are accounted for.** The field's own SOTA fully-learned
word-level model (WavLM Large + HConv + CTC, arXiv:2409.10704) reports
F1=0.554 with the same "high recall, low precision" failure mode this
project already fought and fixed in Phase 2 — not a decisive win. But the
review did surface a genuinely new, well-evidenced next step, not just a
reaffirmation: **extend the audio-native-primary principle (already
proven twice inside this project's own data, for `block` and
`prolongation`) to `word_repetition`/`sound_repetition`/`filler`, which
remain almost entirely token-text-dependent today** — independently
corroborated by outside literature's finding that ASR damages exactly
these types most severely (arXiv:2405.06150: word repetitions,
prolongations, and interjections show 35-47% WER impact vs. ~20% for
blocks) while `block` (already audio-native) is the type least damaged.
The lowest-infrastructure-cost mechanism identified for this: reuse
CrisperWhisper's own encoder representations (already computed on every
clip; no new model, no new forward pass) as an additional acoustic
corroboration signal, the same architectural role VAD/Praat already play
— evidenced as carrying real disfluency signal by arXiv:2406.05784
(F1=0.88/0.85/0.87 using only a Whisper encoder's last layer, frozen
elsewhere, on SEP-28k+FluencyBank).

**One finding is flagged explicitly as the most direct challenge to the
two-stage assumption found in this review, even though it isn't
actionable now**: arXiv:2505.22005 jointly trains ASR and stuttering-event
detection as a multi-task model with bidirectional information sharing,
reporting a 37.71% relative CER reduction *and* a 46.58% relative SED F1
improvement versus separate-task baselines (AS-70, Mandarin) — real
evidence that coupling the two tasks has genuine synergy, not just
additive value. Not adopted: requires jointly fine-tuning an ASR model
with detection labels, which needs a paired transcript+word-level-
disfluency English dataset this project doesn't have and a training
pipeline it doesn't have, and this project has already found that even
*swapping* ASR backends without retraining causes real compatibility
breakage (faster-whisper's tokenizer incompatibility, `ARCHITECTURE.md`
§3) — fine-tuning the model itself is a substantially larger lift.
Recorded as a longer-term direction, not dismissed.

**Alternatives considered** (full detail in `PHASE_3_ARCHITECTURE_
REVIEW.md` §7 "Explicit non-recommendations"):
- End-to-end audio-region models (YOLO-Stutter/Stutter-Solver/SSDM-class).
  **Rejected again, reinforced by new evidence**: SSDM 2.0 (arXiv:2412.00265),
  the direct successor to the already-rejected SSDM, is *heavier* still
  (adds a neural articulatory flow, an LLM-integration pipeline, and
  needs specialized corpora this project has no access to) — the field's
  most complex end is getting more specialized, not more accessible.
- A from-scratch or fully fine-tuned SSL classifier as a wholesale
  replacement. **Rejected for now**: real infrastructure gap, and not
  decisively ahead of this project's current approach at comparable
  granularity once precision is accounted for (see above).
- CTC forced alignment / a different core ASR checkpoint. **Rejected,
  unchanged from prior decisions** (`ROADMAP.md`'s existing reasoning) —
  nothing in this pass overturns either.
- Doing nothing / reaffirming the status quo with no new action item.
  **Rejected**: the per-type WER evidence (§3.6 of the review) is a real,
  specific, actionable finding this project's own architecture is
  structured to exploit (it already has a working pattern —
  `block`/`prolongation` — to extend), and treating "keep the
  architecture" and "keep the architecture exactly as-is with no further
  audio-native extension" as the same conclusion would have wasted the
  clearest actionable finding this review produced.

**Why this choice**
Directly answers the owner's question on its own terms — not "which ASR"
but "what representation" — by tracing the actual, causal, per-type
evidence (both this project's own and the literature's) to a specific,
scoped, infrastructure-realistic next step, rather than either defensively
reaffirming the status quo or chasing the newest architecture in the
literature without regard for what this project can actually build and
validate. Consistent with this project's standing discipline: the
Whisper-encoder-reuse idea is recorded as a *candidate*, not implemented
by this review — it still needs its own `VALIDATION.md` pre-registration
before any code is written, exactly the process the prolongation redesign
went through.

**Measured result**
Not applicable — this entry is the analysis and decision, no code changed.
`PHASE_3_ARCHITECTURE_REVIEW.md` (new, full review). `ROADMAP.md` updated
with the Whisper-encoder-reuse candidate as a new Phase 3 priority item.
`DOCS.md` updated with a pointer to the new file.

---

## 2026-08-04 — Encoder-reuse refined to a specific, staged mechanism; then adversarially challenged from a clean-slate design stance

**What was done**
Two follow-ups to the pre-Phase-3 architecture review above, both same
day, both requested by the project owner before any pre-registration.

**First**: the review's recommendation ("reuse CrisperWhisper's own
encoder representations") was made concrete by reading `profiling/asr.py`
directly rather than reasoning about it abstractly. Finding: this project
calls CrisperWhisper via `transformers.pipeline(...)`, which already sets
`attn_implementation="eager"` specifically because "the pipeline requests
`output_attentions` internally for word-timestamp extraction" (the code's
own comment) — meaning cross-attention tensors are already computed on
every real transcription this app runs, consumed for CrisperWhisper's own
DTW word-timestamp alignment, then discarded. Encoder hidden states and
decoder token probabilities are produced by the same forward pass but
never touched by the pipeline. Compared three candidates (cross-attention,
encoder hidden states, decoder token confidence) on accessibility and
task-specific evidence; encoder hidden states won on evidence (the only
candidate with a direct, quantified, task-matched published result:
arXiv:2406.05784, F1=0.88/0.85/0.87 using a frozen Whisper encoder's last
layer on SEP-28k+FluencyBank). Recommended a staged plan: a zero-training
non-parametric corroboration signal first (Stage 1), a trained
classification head only later and only with explicit go-ahead (Stage 2)
— deliberately not defaulting to training on day one, since every prior
successful signal in this project (VAD, Praat) has been zero-training.

**Second, immediately after**: the owner explicitly asked for this
specific refinement to be adversarially challenged from a clean-slate
design stance — not "does this justify our architecture" but "if
maximizing disfluency-detection accuracy were the only goal, is
encoder-hidden-state reuse actually the best choice, or would a
purpose-built self-supervised encoder (wav2vec2/HuBERT/WavLM/
SeamlessM4T) win outright." Took the objection seriously rather than
performing a review that reflexively reconfirmed the prior answer:
**Whisper's encoder is trained to help produce the correct token
sequence, which has every incentive to compress away exactly the
continuous, non-lexical acoustic variation (a prolongation's exact
voice quality, a hesitation's prosody) that disfluency detection needs —
a real, mechanistic reason encoder-reuse could underperform a
purpose-built SSL encoder.** Checked this against evidence rather than
accepting or dismissing it by argument alone: found one direct,
controlled head-to-head (arXiv:2502.19387, tone classification, not
disfluency) where Whisper (0.97/0.96) performs comparably to WavLM
(0.98/1.00), both well ahead of HuBERT (0.87/0.86) — weakening the strong
version of the objection, but explicitly *not* resolving it, since no
disfluency-specific head-to-head between Whisper-encoder and any SSL
encoder was found anywhere in this pass. Checked SeamlessM4T directly per
the owner's request: its speech encoder is itself a w2v-BERT 2.0/
Conformer self-supervised model, not a distinct representation family,
and no disfluency-specific evidence was found for it at all.

**Conclusion: the objection survives as real and unresolved, but the
honest response is to design Stage 1 so a weak result is itself evidence
for it, with an explicit, pre-stated escalation path — not to pick a
winner by theoretical argument, and not to reflexively keep the original
answer either.** Refined plan: Stage 1 (Whisper-encoder, unchanged) →
an explicit, fixed-in-advance escalation trigger (a weak/null Stage 1
result specifically justifies, rather than vaguely motivates, the next
stage) → Stage 1b (a frozen WavLM-Large pass — chosen over wav2vec2/
HuBERT for having both the best published word-level stuttering result
found, arXiv:2409.10704, and the strongest paralinguistic showing among
SSL options in §8.2's comparison — honestly priced as a real new model
and real added latency this project doesn't pay today) → Stage 2 (a
trained head, either path, explicit go-ahead only). **Stated plainly,
including the answer to the "from scratch" question directly**: if
designing this system today with zero regard for this project's existing
infrastructure, a frozen WavLM-Large pass would likely be the
theoretically stronger *starting* choice — but that is a different
question from what this project, with no GPU and an already
latency-constrained CPU pipeline, should do *first*, and Stage 1 is the
cheapest experiment that produces real evidence for that decision rather
than settling it by argument.

**Alternatives considered**
- Pick a single winner (Whisper-encoder or a specific SSL model) by
  theoretical argument once the objection was raised, rather than
  designing an evidence-gated staged plan. **Rejected**: this project's
  own standing methodology exists specifically to prevent exactly this —
  deciding a real question by plausible-sounding argument instead of
  measurement (`CLAUDE.md` standing rule 3). The disfluency-specific
  evidence gap identified in §8.2 is real; papering over it with a
  same-domain-adjacent comparison (tone classification) would have been
  dishonest about what's actually known.
- Treat the adversarial review as satisfied once a comparable-performance
  data point (arXiv:2502.19387) was found, and declare the objection
  resolved. **Rejected**: that comparison is not disfluency-specific and
  is on a single-speaker synthetic dataset — explicitly flagged as
  suggestive, not decisive, rather than allowed to quietly settle a
  question it doesn't actually answer.
- Skip Stage 1b's cost analysis and just say "escalate to WavLM if Stage
  1 fails," without pricing what that escalation actually costs.
  **Rejected**: this project's own architecture review (§4 of `PHASE_3_
  ARCHITECTURE_REVIEW.md`) explicitly named compute/latency as one of
  four practical constraints to weigh honestly — a second model pass on
  an already 54-102s/clip CPU pipeline is a real cost, not a footnote,
  and stating it plainly is what makes the escalation trigger meaningful
  rather than a vague promise.

**Why this choice**
Directly follows the owner's explicit instruction: assume a clean-slate
design stance, don't default-defend the existing plan, and if the
original conclusion still holds after real challenge, record *why* it
won with its assumptions and evidence stated — which is what §8 of
`PHASE_3_ARCHITECTURE_REVIEW.md` now does, including the parts that don't
fully resolve in the recommendation's favor.

**Measured result**
Not applicable — analysis and decision only, no code changed.
`PHASE_3_ARCHITECTURE_REVIEW.md` §5.1 (staged mechanism) and §8
(adversarial review) added. `ROADMAP.md` item 17 rewritten to the
evidence-gated staged plan with the explicit escalation trigger.

---

## 2026-08-04 — Stage 1 (encoder-representation corroboration signal) pre-registered before implementation

**What was done**
Following the owner's go-ahead to continue to "the next research-planning
step" once the adversarial review above was recorded, wrote the exact
methodology for `ROADMAP.md` item 17's Stage 1 into `VALIDATION.md` §11,
before any extraction code exists — same discipline as every other
pre-registration in this project (`VALIDATION.md` §5.1's Track B protocol,
§9.5's prolongation redesign). Fixed, in advance: exactly what gets
extracted (CrisperWhisper's last-layer encoder hidden states, mean-pooled
per event span, requiring a direct model call since `pipeline()` doesn't
expose hidden states); the exact zero-training signal being tested (a
cosine-distance-to-fluent-centroid measurement, reusing `metrics.
confidence_stats()`'s existing TP-vs-FP-gap shape from Phase 2); the
dataset and scope (the same 499-clip real-audio LibriStutter sample,
restricted to `word_repetition`/`sound_repetition`/`filler`); and success
criteria that treat a null result as informative (triggering the
already-agreed Stage 1b escalation to a frozen WavLM-Large pass) rather
than as a failure to bury.

**A real methodological subtlety was caught and recorded during this
pass, not glossed over**: Track A's entire design principle is bypassing
ASR (`VALIDATION.md` §3), but testing CrisperWhisper's own encoder
representation requires running its encoder over the real audio even
under Track A — only the decoded text/timestamps are bypassed in favor
of ground truth, not the encoder forward pass the signal itself depends
on. This means Stage 1 is not a free re-use of an existing cached run; it
needs a real (one-time, cacheable) encoder pass over the sample. Recorded
explicitly in `VALIDATION.md` §11.1 rather than left implicit, since
silently assuming this was "free like Track A always is" would have been
a real error to discover only once implementation started.

**Alternatives considered**
- Skip the zero-training distance measurement and go straight to
  training a small classification head (Stage 2), since that's ultimately
  the more decisive test. **Rejected**: exactly the escalation-by-
  argument-instead-of-evidence pattern §8's adversarial review just
  rejected for the Whisper-vs-WavLM question — committing to training
  before confirming the raw representation carries *any* separable
  signal risks conflating "the representation has no signal" with "the
  classifier wasn't trained well enough," which would make a negative
  result uninterpretable.
- Test on SEP-28k (where the one directly on-point published result,
  arXiv:2406.05784, was itself measured) instead of LibriStutter.
  **Rejected for Stage 1 specifically**: SEP-28k has no reference
  transcript and no audio acquired yet (`ROADMAP.md` item 15) — LibriStutter
  is the dataset this project can actually test against today, and Stage
  1's question (does *this* representation carry *any* signal in *this*
  project's pipeline) doesn't require matching the published paper's
  exact dataset to be answerable.

**Why this choice**
Directly continues the pre-registration discipline the owner asked to be
followed throughout — methodology, hypothesis, and success criteria
(including what a negative result means and what it triggers) are fixed
before any extraction code is written, so the eventual result can't be
quietly reinterpreted after the fact.

**Measured result**
Not applicable — methodology only, no code changed. `VALIDATION.md` §11
(new, full pre-registered protocol). No implementation started, per the
owner's explicit instruction to record the research direction before
moving into coding.

---

## 2026-08-04 — Stage 1 implemented exactly as pre-registered; a real cost discovered and priced, not absorbed silently

**What was done**
Implemented `VALIDATION.md` §11's protocol, no more and no less: extract
CrisperWhisper's last-layer encoder hidden states per event span
(`profiling/evaluation/encoder_features.py`), aggregate a TP-vs-FP
distance gap in the same shape as Phase 2's `confidence_stats()`
(`metrics.encoder_distance_stats()`, `report.format_encoder_distance_
stats()`), and a runner (`run_encoder_signal_stage1.py`). Pure-math
functions (span pooling, cosine distance, the fluent centroid) are
unit-tested with synthetic hand-constructed data, no real model needed
(`tests/test_encoder_features.py`, 12 tests) — matching this project's
existing split between fast logic tests and real-model integration runs.
Full suite: 57/57 (was 45/45).

**Verified the implementation for real, not just via unit tests, before
committing to any real evaluation run**: a 1-clip smoke test against real
LibriStutter audio confirmed the whole pipeline (model load, encoder
extraction, span pooling, centroid computation, distance aggregation,
result saving) runs correctly end to end and produces a sensible,
non-degenerate result.

**Two things discovered during implementation, both recorded rather than
absorbed silently, per standing rule 3**:
1. **A simplification**: last-layer-only extraction doesn't need
   `output_hidden_states=True` at all — a plain encoder forward pass's
   primary output already is the last layer. Cheaper and simpler than
   the pre-registration's own wording implied.
2. **A real, previously-unpriced cost, found by direct measurement**: the
   encoder pass alone measured 37.8s for one clip (confirming
   `ARCHITECTURE.md`'s existing finding that Whisper's fixed 30s-window
   encoder pass dominates transcription latency regardless of clip
   length, and that skipping decoding only saves a modest fraction of
   the total). A full 499-clip run is therefore ~6 hours of CPU time —
   not stated or estimated in the original pre-registration, which only
   described this as "a real, one-time, cacheable forward pass" without
   quantifying it.

**Decision in response to the cost discovery**: default the runner to a
30-clip pilot rather than the full 499, explicitly mirroring this
project's own Track B precedent (pilot at 30, scale to 90 then 120 only
once the pilot's result justified it, §8.4.1-§8.4.3) rather than
committing 6 hours of compute to an unvalidated pipeline. This changes
*how much data runs first*, not the pre-registered methodology itself —
`VALIDATION.md` §11.1-§11.5 are unchanged; the addendum recording this is
§11.3.1, not a rewrite of the protocol.

**A real bug caught immediately by existing infrastructure**: the ASCII-
console lint rule (`ROADMAP.md` item 13, built in Phase 2) flagged a
literal em-dash in this new script's first `print()` call on its very
first run. Fixed before any real evaluation run used the script. Small,
but a concrete confirmation the lint rule generalizes to new code, not
just the files it was originally written against.

**Alternatives considered**
- Run the full 499-clip sample immediately, accepting the ~6-hour cost,
  rather than piloting first. **Rejected**: this project's own
  established discipline (Track B) is to pilot before committing large
  compute to an unvalidated pipeline — the smoke test had already
  validated correctness on 1 clip, but a 30-clip pilot is the cheap next
  step to confirm the *aggregate* pattern is stable before spending
  hours confirming it at full scale.
- Skip the unit tests for the pure-math functions and rely on the real
  1-clip smoke test alone. **Rejected**: the smoke test validates
  end-to-end plumbing against one real clip, not edge cases (empty spans,
  missing timestamps, degenerate all-disfluent clips, span-preference
  between nominal and acoustic timing) — exactly the kind of case a fast,
  real-model-free unit test exists to cover cheaply and repeatably.

**Why this choice**
Directly follows the owner's explicit instruction for this stage:
implement only what was pre-registered, validate rigorously before
drawing conclusions, and treat any gap between expectation and reality
(the cost, the simplification) as a finding to document, not a surprise
to quietly work around.

**Measured result**
`profiling/evaluation/encoder_features.py`, `metrics.py`
(`encoder_distance_stats`), `report.py` (`format_encoder_distance_stats`),
`run_encoder_signal_stage1.py` (all new/extended). `tests/
test_encoder_features.py` (new, 12 tests) + `track_a.py` self-test section
8 (new). Full suite: 57/57. `VALIDATION.md` §11.3.1 (new addendum). No
detector behavior changed — this is measurement-only code, exactly as
§11.5 states this stage is scoped to. Real evaluation results not yet in
— the 30-clip pilot is running as this entry is written; its result gets
its own entry once complete, not folded into this one, so the
implementation record and the empirical result stay separately
verifiable.

---

## 2026-08-04 — 30-clip pilot: a real, non-trivial gap found; a real gap in the pre-registration's own success criteria found and fixed before drawing any conclusion

**What was done**
The 30-clip pilot completed: `Any` label TP mean distance 0.5860 vs. FP
mean distance 0.4965 (gap +0.0895, n=28 TP/25 FP); `word_repetition`
+0.0954 (n=12 TP/22 FP) — both in the hypothesized direction (genuine
disfluencies farther from the clip's fluent centroid), and roughly
15-20x larger in magnitude than the near-zero gap (-0.007 to +0.003)
`confidence_stats()` found for VAD/Praat in Phase 2 (`VALIDATION.md`
§9.3.1) — the most recent result measured with the same TP-vs-FP-gap
methodology, and the natural comparison point. `sound_repetition` and
`filler` were uninformative in this pilot (0 FP and 0 TP respectively —
expected, matching already-known sampling gaps in this 499-clip sample,
`ROADMAP.md` item 14).

**Before treating this as "clear signal" and moving toward Stage 2**,
tried to apply §11.4's own pre-registered success criteria to this actual
result and found they don't resolve it: "meaningfully higher... in the
expected direction" specifies a direction, not a way to judge whether a
gap this size, on a sample this small, is a real effect or noise. This
is exactly the discipline the owner asked for explicitly this stage
("do not optimize toward proving the idea correct... if you discover
methodological weaknesses, pause to evaluate them scientifically") —
caught by trying to actually use the criteria, not by review beforehand.

**Fixed before interpreting the result, not after**: added Cohen's d
(pooled SD) and each group's standard deviation to `metrics.
encoder_distance_stats()`, the same instinct that added Wilson intervals
for small-n recall claims in Phase 2. `cohens_d` is `None` (not a
silently-wrong `0.0`) when a group has fewer than 2 samples. Revised
success criterion, stricter than the original wording: a "clear signal"
requires `|Cohen's d| >= 0.5` *and* the hypothesized direction, not the
raw gap number alone. Since the 30-clip pilot only saved aggregate means
(not raw per-event distances, added before this need was known), its
effect size can't be computed retroactively — rather than re-running 30
clips a second time just to add the new number, launched a 90-clip run
directly (matching Track B's own 30-then-90 scaling precedent), which
gets both a larger, more trustworthy sample and the new Cohen's d number
in one run rather than two.

**Alternatives considered**
- Declare "clear signal" from the 30-clip pilot's raw gap alone and move
  toward Stage 2. **Rejected, explicitly**: this is precisely "optimizing
  toward proving the idea correct" rather than discovering the truth —
  the gap being 15-20x the VAD/Praat null result's magnitude is
  suggestive, not sufficient, on n=28/25, and the pre-registration itself
  didn't specify a rigorous enough bar to make that call responsibly.
- Retrofit an effect-size estimate onto the already-completed 30-clip
  pilot using only its saved means (e.g. assume a plausible variance).
  **Rejected**: would be estimating a number this project has the actual
  data to measure properly instead — inventing a number when the real
  one is one more (already-justified) run away is exactly the kind of
  shortcut this project's measurement-first discipline exists to avoid.

**Why this choice**
Directly follows the owner's explicit framing for this stage: the
question is whether the representation carries real information, not
whether the pilot's first number looks encouraging. Finding and fixing a
real gap in the pre-registration's own rigor, before it could bias how
an ambiguous result got read, is the kind of self-correction this
project's methodology is supposed to produce under pressure to conclude
something.

**Measured result**
`profiling/evaluation/metrics.py` (`encoder_distance_stats()` now reports
`tp_stdev`/`fp_stdev`/`cohens_d`), `report.py`
(`format_encoder_distance_stats()` updated), `track_a.py` self-test
section 8b (3 new checks). Full suite: 57/57 (was 57/57 — no test count
change, existing tests extended in place). `VALIDATION.md` §11.4.1 (new
addendum, revised success criterion). 30-clip pilot result saved
(`eval_results/20260804T140128595787Z_libristutter_stage1-encoder-signal.json`)
but not the basis for any conclusion — the 90-clip run with the corrected
metric is in progress as this entry is written.

---

## 2026-08-04 — Stage 1 result: a clear, stable, large-effect-size signal — CrisperWhisper's encoder carries information the transcript alone does not

**What was done**
The 90-clip run (launched to get both a larger sample and the newly-added
Cohen's d in one run, per the previous entry) completed:
`word_repetition` gap +0.0853 (Cohen's d = **+1.047**, n=30 TP/63 FP);
`Any` (combined) gap +0.0919 (Cohen's d = **+1.116**, n=70 TP/68 FP).
Both clear §11.4.1's revised bar (`|d| >= 0.5`) decisively — conventionally
"large" effects, not borderline ones — and both the gap's sign and rough
magnitude held from the 30-clip pilot to the 90-clip run (word_repetition
+0.0954→+0.0853; `Any` +0.0895→+0.0919), the signature of a real, stable
effect, not a small-sample fluke regressing toward zero.
`sound_repetition`/`filler` remain uninformative at this scale (0 FP/0 TP
respectively) — the same already-known LibriStutter sampling gap
(`ROADMAP.md` item 14), not a finding against the signal for those types.

**Audited before accepting, per standing rule 3** (a result this clean is
exactly the kind worth checking harder on, not reporting faster):
confirmed each `word_repetition` event is attributed to a single token
(`detect.py`), so TP and FP spans being compared are both single-word
pooled embeddings, not systematically different-length spans — this
partially mitigates, but does not eliminate, a duration- or word-
identity-driven confound. Recorded explicitly as an unresolved limitation
(`VALIDATION.md` §11.6), not glossed over because the headline result is
positive.

**Conclusion, directly answering this project's stated Stage 1 research
question**: CrisperWhisper's own encoder representation carries
information the ASR transcript alone does not — the transcript is
identical for a genuine repeated word and a coincidental one (both
produce the same tokens), while the encoder embedding separates them
with a large, stable effect size across two independent sample sizes.
This is real evidence the existing architecture can be strengthened
without a second heavyweight model, exactly the outcome that would make
encoder-reuse the right call over Stage 1b (frozen WavLM-Large) or a
larger architecture change, per `PHASE_3_ARCHITECTURE_REVIEW.md` §5.1's
own framing of what each outcome would mean.

**A new option surfaced by this result, not previously weighed**: the
raw distance-to-fluent-centroid measure might be directly usable as a
zero-training corroboration signal (e.g. a per-type threshold, the same
shape VAD/Praat already work in) without ever training a classifier —
Stage 2 was originally framed as "the" next step on a clear signal, but
this result suggests it may not be the *only* next step worth
considering. Recorded as an open option, not decided here.

**Alternatives considered**
- Treat this as sufficient grounds to start implementing Stage 2 (a
  trained classification head) immediately, given how clear the result
  is. **Rejected**: `VALIDATION.md` §11.5 and standing rule 4 both fix
  this in advance — even a clear-signal result requires a separate,
  explicit go-ahead before implementation, precisely so a good result
  doesn't create momentum that substitutes for a deliberate decision.
- Treat the duration/word-identity confound as disqualifying and hold
  the result as inconclusive until it's ruled out. **Rejected**: the
  confound is real and unconfirmed, not confirmed to be driving the
  result — the honest position is to report the effect size as measured
  and name the open question, not to discard a large, stable,
  two-sample-consistent effect on a specific unverified doubt alone.

**Why this choice**
Reports the measurement exactly as Stage 1 was pre-registered to
produce, audited (not just read off a promising-looking table), with
its real limitation named rather than hidden by the headline number —
and stops exactly where §11.5 said this stage would stop, leaving the
response to the project owner rather than treating a good result as
its own authorization.

**Measured result**
`VALIDATION.md` §11.6 (new, full results section with the two-run
comparison table, the audit, and the stated limitation).
`eval_results/20260804T151450108987Z_libristutter_stage1-encoder-
signal.json`. No code or detector behavior changed. `ROADMAP.md` item 17
updated to reflect Stage 1's completed, positive result and the pending
Stage 2 go-ahead decision.

---

## 2026-08-04 — Corroboration-mechanism review: neither "threshold" nor "classifier" assumed — a broader candidate space reviewed and a comparison pre-registered

**What was done**
The project owner explicitly asked that the decision following Stage 1's
positive result not default toward either simplicity or complexity —
"design the most effective architecture... do not assume that a
zero-training threshold, a lightweight classifier, or even the current
transcript-first pipeline is automatically the right answer." Re-opened
the question from first principles (`PHASE_3_ARCHITECTURE_REVIEW.md`
§9): separated it into two axes previously conflated — *which signal*
(distance-to-fluent-centroid, what Stage 1 tested; or a new, not
previously evaluated alternative, repeat-pair self-similarity between a
`word_repetition`/`sound_repetition` event and its partner token,
confirmed as `tokens[event["index"] - 1]` by reading `detect.py`
directly) and *which decision mechanism* (fixed threshold, per-clip
relative threshold, or a trained classifier). Evaluated the mechanism
axis against all nine dimensions the owner named (performance,
robustness to ASR errors, localization accuracy, computational cost,
maintainability, interpretability, reproducibility, engineering
complexity, long-term scalability) without assuming an answer for any of
them — e.g. found that Stage 1's own large effect size (d>1.0) is a
principled reason to test a threshold *first*, not because simpler is
better, but because large effect sizes are exactly the regime where a
threshold typically already captures most of the available separation,
making a classifier's *marginal* gain the thing actually worth measuring
rather than assuming.

**A real correction to the Stage 1 framing, caught by this review**: the
existing architecture does **not** currently use one shared corroboration
strategy across types — `block` (silence-threshold), `prolongation`
(Praat hard-gate), and `filler` (voiced-energy presence) already use
type-specific signal sources under one shared fusion architecture. The
real question Stage 1 raised was never "should types share a strategy"
(they don't) but "does `word_repetition`/`sound_repetition` get a
type-appropriate new signal added to that same existing pattern" — a
narrower, better-grounded question than how it was originally framed.

**Also named plainly, not glossed over**: a genuine interpretability cost
shared by *every* embedding-based candidate (threshold or classifier)
relative to this project's existing Praat/VAD signals — a cosine
distance in a 1280-dimensional space has no independent physical meaning
the way "jitter = 2.1%" does. This is a cost of using this signal family
at all, not a property that differs between the mechanism candidates.

**Conclusion: multiple candidates remain genuinely plausible after this
review, resolved by pre-registered comparison, not by argument** — three
combinations selected as the ones that most directly answer the open
questions this review surfaced (not an exhaustive sweep): (S1, M1) as
the cheapest baseline; (S1, M3) to measure a classifier's actual marginal
gain over that baseline; (S2, M1) to test the newly-identified
alternative signal under the cheapest mechanism before investing further
in it. Full protocol pre-registered in `VALIDATION.md` §12: 5-fold
cross-validation split by clip (not event, to avoid the same leakage
class SEP-28k-E's own paper was designed to prevent), logistic regression
implemented in plain `numpy` rather than adding `scikit-learn` as a new
dependency, with L2 regularization explicitly required and named as a
real methodological choice given a ~1280-dimensional embedding and only
roughly a hundred training events per fold — an underdetermined problem
without it.

**Implemented (measurement infrastructure only, per standing rule 4 —
nothing added to `detect_disfluencies()`)**: `encoder_features.
collect_raw_records()` (persists full embedding vectors, not just Stage
1's scalar distance, so any signal/mechanism combination can be computed
post-hoc without a third encoder pass); `collect_raw_encoder_data.py`
(the collection runner); `compare_corroboration_mechanisms.py` (the
cross-validated analysis, pure `numpy`, no encoder needed, runs in
seconds once data is collected). 2 new unit tests for `collect_raw_
records()`. Full suite: 59/59 (was 57/57). **Smoke-tested on synthetic
data before spending real compute**: built a small synthetic dataset with
a deliberately clean, separable signal and confirmed the pipeline
round-trips correctly end to end (`.npz` save/load, fold split, threshold
search, logistic regression) — and, usefully, previewed exactly the
small-sample/high-dimension risk already flagged for the real run (the
classifier underperformed the threshold on synthetic data too, in the
direction the n<<p concern predicts) — a sign the concern is real, not
just a stated caveat.

**Alternatives considered**
- Skip the broader review and go straight to implementing Stage 2 (a
  classifier), since Stage 1's result was strong. **Rejected, explicitly,
  per the owner's own framing**: a strong Stage 1 result answers "does
  this signal carry information," not "what's the best way to act on
  it" — treating a good result as its own justification for the next
  specific implementation choice is exactly the bias-toward-momentum
  the owner asked this review to avoid.
- Run an exhaustive sweep of every signal x mechanism combination (S1/S2
  x M1/M2/M3, 6 combinations). **Rejected**: scoped to the three
  combinations that directly answer the specific open questions this
  review identified, not maximum coverage for its own sake — matches
  this project's general preference for scoped, question-driven
  evaluations (e.g. §9.5's own explicit "not an exhaustive sweep"
  reasoning) over combinatorial completeness.
- Use raw per-event correlation/exploratory stats instead of a proper
  cross-validated comparison. **Rejected**: this project has repeatedly
  found (Wilson intervals, Cohen's d, the Praat-gating ablation) that
  point estimates without held-out validation overstate confidence —
  cross-validation is the correct standard for a fair mechanism
  comparison, not an unnecessary complication.

**Why this choice**
Directly follows the owner's explicit instruction: treat the mechanism
question as genuinely open, evaluate it against real criteria rather
than a default preference, and let evidence decide among the remaining
plausible candidates rather than argument. The corrected type-strategy
framing and the named interpretability cost are both examples of this
review producing sharper understanding, not just a comparison table.

**Measured result**
`PHASE_3_ARCHITECTURE_REVIEW.md` §9 (new). `VALIDATION.md` §12 (new,
full pre-registered protocol). `profiling/evaluation/encoder_features.py`
(`collect_raw_records`, `REPEAT_PARTNER_TYPES`),
`collect_raw_encoder_data.py`, `compare_corroboration_mechanisms.py`
(new). `tests/test_encoder_features.py` (+2 tests, 14 total). Full suite:
59/59. No detector behavior changed. Real comparison results not yet in —
the 90-clip raw-data collection run (needed because the original Stage 1
runner never persisted raw embeddings, only the aggregate distance) is
running as this entry is written; its result gets its own entry once
complete.

---

## 2026-08-04 — Corroboration-mechanism comparison result: the classifier wins clearly — the opposite of this project's own pre-registered prediction

**What was done**
The 90-clip raw-embedding collection completed (138 scorable events,
counts matching §11.6 exactly — confirms the new collection path
reproduces Stage 1's own numbers, not a divergent measurement) and the
pre-registered 5-fold cross-validated comparison ran: (S1, M1)
`word_repetition` F1=0.588, `Any` F1=0.755; **(S1, M3) F1=0.749/0.888
respectively — a +0.161/+0.133 F1 margin over the threshold**; (S2, M1)
F1=0.546/0.678, not better than (S1, M1).

**This is the opposite of what `VALIDATION.md` §12.4 explicitly
predicted before the comparison ran**: that section reasoned Stage 1's
large Cohen's d meant a threshold should already capture most of the
available separation, so a large classifier margin was framed as
unlikely. **Audited before accepting** (per standing rule 3, and directly
per the owner's instruction this stage that a contradicted expectation
is itself a scientific finding, not something to work around): verified
directly, not just eyeballed from the mean, that the classifier beat the
threshold in **5 of 5 folds in both type slices** — a consistent,
fold-by-fold advantage, not an average pulled up by one lucky split.
Reasoned explanation, stated honestly as post-hoc, not a prediction that
was actually made in advance: distance-to-fluent-centroid is an
*unsupervised* heuristic that never looks at TP/FP labels; logistic
regression is fit directly to them. §12.4's reasoning assumed the
centroid-distance projection was close to the best available linear
separator in the embedding space — the result suggests it isn't, and a
supervised fit finds a better one.

**(S2, M1), the new repeat-pair-self-similarity signal this session's
own architecture review proposed, did not outperform (S1, M1)** in
either type slice — a real, negative-ish result for a candidate this
project specifically went looking for, recorded plainly rather than
quietly dropped because it didn't pan out.

**Real limitations named alongside the strong result, not hidden by
it**: still a modest sample (93/130 events across 5 folds); the L2
regularization strength was fixed, not tuned, and robustness to that
choice wasn't tested; still LibriStutter's reconstructed-timing data,
the same standing caveat attached to every result on this dataset since
§8.2; `sound_repetition` remains entirely unvalidated (0 FP throughout).

**Conclusion, at the same scope as every prior stage in this
investigation — a measurement, not an implementation decision**: real
evidence favors (S1, M3) — a small trained classifier over
CrisperWhisper's own encoder embedding — over both the zero-training
threshold and the untested alternative signal. Per `VALIDATION.md`
§12.4 and standing rule 4, this measurement does not itself authorize
implementing anything into `detect_disfluencies()` — accepting the real,
categorical costs `PHASE_3_ARCHITECTURE_REVIEW.md` §9.3 already named
(this project's first shipped trained-model artifact, and everything
that comes with maintaining one) in exchange for this measured gain is
a separate, explicit decision for the project owner.

**Alternatives considered**
- Revise the write-up to soften or reframe §12.4's contradicted
  prediction so the result reads as expected rather than surprising.
  **Rejected, explicitly**: the owner's standing instruction for this
  entire investigation was to treat a contradicted expectation as a
  finding, not something to smooth over — §12.4's original wording is
  left exactly as written, and this entry states plainly that the result
  went the other way, with the reasoning for why in hindsight kept
  clearly labeled as post-hoc.
- Report only the mean F1 gap and treat "exceeds one fold's variance" as
  self-evidently satisfied without checking the actual per-fold values.
  **Rejected**: checked directly (5/5 folds in both slices) before
  writing that claim into the permanent record — an assertion about
  consistency across folds needed to actually be verified, not inferred
  from a mean and a range alone.

**Why this choice**
Reports the measurement exactly as it came out, including the part that
contradicts this project's own stated prediction, with the surprising
claim specifically verified (not just asserted) before being written
down — precisely the discipline the owner asked this stage to hold to
above all else: discover the truth, not confirm what was expected.

**Measured result**
`VALIDATION.md` §12.5 (new, full results section with the comparison
table, the contradicted-prediction discussion, and the stated
limitations). `eval_results/stage1_raw_embeddings_90clip.npz` (raw data).
No code or detector behavior changed — `detect_disfluencies()` is
untouched. `ROADMAP.md` item 17 to be updated with this result and the
pending implementation decision.

---

## 2026-08-04 — Standing principle established: architectural decisions are evidence-constrained, not preservation-constrained; applied immediately to today's own open question

**What was done**
The project owner established a standing rule governing all future
architectural decisions in this project (recorded in full in `CLAUDE.md`
standing rule 8): the current architecture and every prior design
decision are hypotheses that earned their place through evidence, not
defaults to preserve; simplicity, interpretability, rule-based logic,
ML, hybrid methods, pretrained and newly-trained components are all
engineering choices, none privileged over another by default. Autonomy
to decide is granted, but evidence-constrained — reach a confident,
recorded decision when evidence supports one; when it doesn't, name the
specific remaining uncertainty, pre-register what would resolve it, run
it, and only then decide. Explicitly not license to guess, and equally
not license to default to the simpler/existing option out of habit.

**Applied immediately to the one open architectural question this
project actually has right now**: whether to adopt (S1, M3) — the
learned classifier over CrisperWhisper's encoder embedding — as the new
corroboration mechanism for `word_repetition`/`sound_repetition`, given
§12.5's result (F1 0.749/0.888 vs. 0.588/0.755 for the threshold, 5/5
folds consistent).

**Decision, reasoned explicitly against the new principle's own terms —
not a reflexive fallback to the simpler option, and not a rubber stamp
of the better-looking number**: the evidence is real and positive enough
to change this project's *working expectation* — a learned corroboration
signal is, on current evidence, more likely to be the right direction
for these two types than a hand-calibrated threshold. **It is not yet
strong enough to justify shipping it as the new default today**, for
three reasons that are about the *evidence's own limits*, not a
preference for simplicity — each stated so it's falsifiable, not a vague
hedge:

1. **Statistical asymmetry at this sample size.** A ~1280-dimensional
   classifier fit on ~100 training events per fold carries materially
   more overfitting risk, at this same n, than a 1-dimensional threshold
   — holding sample size constant, the classifier's cross-validated
   estimate is inherently less stable to trust than the threshold's, even
   though today's specific 5-fold result was clean. This is a property
   of comparing a high-capacity model to a low-capacity one at small n,
   not a claim that classifiers are worse in general.
2. **The regularization strength was fixed, not tuned** (§12.3's own
   stated scope limit) — the result hasn't been stress-tested against
   this free parameter, so its stability isn't yet established.
3. **LibriStutter's reconstructed-timing data has fooled a
   higher-capacity mechanism before, in this exact project.** The
   clearest precedent: the original prolongation percentile-threshold
   looked good on reconstructed data and only revealed a real problem
   (`VALIDATION.md` §8.3) once measured against real, non-reconstructed
   audio. A classifier has more capacity than a threshold to fit
   dataset-specific reconstruction artifacts that have nothing to do
   with genuine disfluency signal — a real, mechanistically-grounded risk
   specific to *this* dataset, not a generic anti-ML objection.

**None of the three is "it's more complex" or "it's less interpretable"
stated as a disqualifier** — per the new standing rule, those are real
costs (already named in full in `PHASE_3_ARCHITECTURE_REVIEW.md` §9.3)
to weigh *once the evidence is trustworthy enough to act on*, not
grounds to avoid finding out whether the evidence is there.

**Next step, pre-registered now rather than run now** (end of a long
session — the responsible move per the new rule's own "identify the
uncertainty, pre-register the resolution, then decide" pathway, not a
stall): `VALIDATION.md` §12.6 fixes the exact follow-up validation (a
larger-scale re-run with regularization chosen by nested cross-validation
instead of a fixed value) and the explicit decision rule it resolves to
— see that section for the full pre-registration.

**Alternatives considered**
- Adopt the classifier into `detect_disfluencies()` today, since the
  measured result is clean and the owner granted explicit autonomy to
  decide. **Rejected**: the granted autonomy is evidence-constrained, not
  a mandate to act on the first positive-looking number — the three
  named uncertainties are real, not manufactured to justify inaction, and
  ignoring them to move faster would violate the same rule being invoked
  to justify moving at all.
- Reject the classifier and keep the token-only status quo for these
  types, treating "no trained model yet in this project" as sufficient
  reason on its own. **Rejected, explicitly, per the new standing rule**:
  that would be exactly the preservation-constrained reasoning the owner
  just ruled out — "we've never shipped a trained model before" is a
  fact about this project's history, not evidence about which
  architecture is best, and treating it as a veto would violate rule 8
  as much as rubber-stamping the classifier would.
- Treat this as still-open with no further action specified, leaving the
  next step vague. **Rejected**: the new rule requires naming the exact
  remaining uncertainty and pre-registering what resolves it — done in
  `VALIDATION.md` §12.6, not left as a general "more research needed."

**Why this choice**
This is the new standing rule applied to itself, immediately, honestly —
neither defaulting to the existing architecture out of habit nor
adopting the new one out of momentum from a good result, and saying so
explicitly rather than letting the reasoning stay implicit. The decision
record here is what a future researcher (or a future session) needs to
understand not just *what* was decided, but that a real, principled
process produced it.

**Measured result**
`CLAUDE.md` standing rule 8 (new). `VALIDATION.md` §12.6 (new, the
pre-registered follow-up validation and its decision rule). `ROADMAP.md`
item 17 updated to reflect this reasoned non-decision and its concrete
next step. No code changed — `detect_disfluencies()` remains exactly as
it was before Stage 1 began.

---

## 2026-08-04 — Executing §12.6: nested-CV regularization tuning implemented; a real cost-escalation finding changed the run's scope before launching it

**What was done**
Began executing `VALIDATION.md` §12.6's pre-registered follow-up exactly
as specified: (1) nested cross-validation for the classifier's L2
regularization strength, replacing the fixed `L2_STRENGTH = 5.0` used in
§12.5's result — implemented directly in `compare_corroboration_
mechanisms.py` (`_select_l2_by_nested_cv()`: an inner, clip-split k-fold
CV over each outer fold's own training data only, grid-searching a fixed
set of L2 values, selecting by mean inner F1, then refitting on the full
outer-training set before evaluating on the outer test fold — never
touching the outer test fold during selection). Smoke-tested on
synthetic data before trusting it: confirmed different L2 values get
selected per fold (not a constant, degenerate choice) and the whole
pipeline runs correctly end to end.

**Before launching the larger data collection, re-examined the 90-clip
run's own raw per-clip timings rather than assuming its flat
~44.6s/clip average would hold at a larger scale — and found it
wouldn't**: the first 10 clips of that run averaged ~31s/clip, the last
10 averaged ~85s/clip, a real ~2.7x slowdown within a single ~68-minute
run. Most likely explanation (stated with honest uncertainty, not
confirmed by further instrumentation): thermal throttling on the laptop
CPU under sustained load, a known behavior for that class of hardware
under this kind of workload. **This directly changed the scoping
decision**, not just the time estimate: §12.6 named "the full 499-clip
sample, cost permitting," but a naive flat-rate extrapolation (499 x
44.6s = ~6.2h) already undersold the real cost, and if the slowdown
continues rather than plateaus, the full run's duration and reliability
become genuinely uncertain. **Decided to target 250 clips instead** — a
~2.8x increase over the already-collected 90-clip sample, materially
larger and directly able to answer §12.6's actual question, while
bounding the commitment to a run whose duration can be reasoned about
(~5 hours under a plausible thermal-plateau assumption) rather than an
open-ended one. The full 499 remains available as a further step if 250
clips doesn't resolve the question decisively.

**Alternatives considered**
- Launch the full 499-clip run anyway, since §12.6 named it as the
  target. **Rejected**: the newly-discovered slowdown means "cost
  permitting" — the exact clause §12.6 itself included — is no longer
  satisfied with the same confidence it was when that section was
  written, before this timing pattern was known. Proceeding blindly
  would repeat exactly the mistake this project's methodology exists to
  prevent: committing to a plan without accounting for new evidence that
  complicates it.
- Try to fix the suspected thermal-throttling cause (e.g. inserting
  cooldown pauses between clips) before running at scale. **Rejected**:
  the root cause isn't confirmed, and testing an unvalidated mitigation
  would add its own uncertainty and delay without being asked for —
  bounding the scope to a size whose cost is already reasonably well
  understood is the lower-risk path to actually answering §12.6's
  question today.
- Skip nested-CV regularization tuning and only scale up the sample
  size, treating the fixed-L2 concern as secondary. **Rejected**: §12.6
  named both changes together for a reason — a larger sample alone
  wouldn't rule out that §12.5's result depended on the specific,
  arbitrary L2=5.0 value; both were pre-registered as needed to actually
  resolve the uncertainty.

**Why this choice**
Directly continues the pre-registered protocol while applying the same
"audit before trusting, adjust the plan when new evidence complicates
it" discipline this project has used all session — the scope change is
a documented, reasoned response to a real finding, not a silent
deviation or a shortcut taken to save time.

**Measured result**
`profiling/evaluation/compare_corroboration_mechanisms.py` (nested-CV
L2 selection, smoke-tested on synthetic data). `VALIDATION.md` §12.6.1
(new addendum: the timing finding, the reasoning, and the 250-clip
scoping decision). Full suite: 59/59 (unchanged — no new unit-test-
covered logic beyond what the smoke test already exercised at the
integration level; the nested-CV function is exercised by
`compare_corroboration_mechanisms.py`'s own module-level smoke test, not
a `tests/`-suite unit test, matching this file's existing convention of
being validated via direct runs rather than a formal test file). The
250-clip raw-embedding collection is running as this entry is written;
its result and the final decision get their own entry once complete.

---

## 2026-08-04 — Real cost of no checkpointing: a run was killed mid-way with zero recoverable progress; checkpointing added and verified before restarting

**What was done**
The 250-clip collection launched under the previous entry was killed by
an unrelated session interruption partway through. Confirmed directly
(process list showed no running collection process; no output file
existed at all) that **zero progress from that run was recoverable** —
`collect_raw_encoder_data.py` only wrote its output at the very end, so
an interruption at clip 200 would have lost exactly as much as one at
clip 5. This is precisely the risk named (but not yet acted on) when the
90-clip run's slowdown was investigated a few entries above.

Rewrote `collect_raw_encoder_data.py` to checkpoint after **every**
clip, not periodically — cheap relative to the 30-90s/clip encoder cost
already measured, so there's no real reason to checkpoint less often.
Added genuine resume support, not just periodic saving: if `--out`
already exists when the script starts, it's loaded, already-processed
clips are skipped (tracked in a dedicated `processed_clips` array,
separate from the record rows themselves, so a clip that legitimately
produces zero scorable events is still correctly remembered as done and
not silently re-processed forever), and new records are appended to what
already exists rather than overwriting it.

**Verified before trusting it with a real multi-hour run** — the same
discipline applied to every other piece of this project's evaluation
infrastructure: (1) a synthetic round-trip test covering the tricky edge
case (a clip with zero records must still be marked processed); (2) a
real, 2-clip collection run producing a genuine checkpoint file; (3) a
real resume run against that same file with 2 additional clips
requested, confirming it correctly reported "2 already done," skipped
them, and processed only the 2 new ones, merging to a correct combined
total (6 records across 4 clips). Full suite: 59/59, unaffected.

**Alternatives considered**
- Checkpoint periodically (e.g. every 10 clips) rather than every clip,
  to reduce I/O overhead. **Rejected**: the checkpoint write itself is a
  small, fast compressed-array save, negligible next to a single clip's
  30-90s encoder cost — checkpointing every clip bounds worst-case lost
  work to one clip regardless, at effectively no extra cost.
- Just re-run the full 250 clips from scratch without adding
  checkpointing first, to save implementation time. **Rejected,
  directly per the owner's own instruction this turn**: the owner
  explicitly asked for checkpointing to be added because it helps,
  before continuing — and given today's run already demonstrated the
  failure mode once, shipping the same unprotected script a second time
  would risk losing the next multi-hour run for the same avoidable reason.

**Why this choice**
A real, observed failure (total loss of an in-progress run) with an
identified, fixable cause gets fixed and verified before being relied on
again — not patched by hoping the interruption doesn't recur. Matches
this project's standing pattern of treating an operational failure as
seriously as a statistical one.

**Measured result**
`profiling/evaluation/collect_raw_encoder_data.py` (checkpointing +
resume, rewritten). Verified via synthetic round-trip test and a real
2-then-4-clip collection/resume run (both cleaned up afterward, not
part of the real dataset). Full suite: 59/59. The 250-clip collection is
being relaunched under this fixed version immediately after this entry.

---

## 2026-08-05 — Decision executed in full: (S1, M3) implemented, benchmarked, and shipped as the new default — two real bugs caught by the existing test suite before any result was trusted

**What was done**
Following through on the owner's explicit instruction ("if the
classifier's advantage remains robust... implement it as the chosen
corroboration mechanism, benchmark the integrated detector, update all
relevant documentation, and prepare the repository for commit"), and on
`CLAUDE.md` standing rule 8 (architectural decisions are evidence-
constrained — once the evidence is there, decide and act on it):

1. **Trained the final model.** `profiling/evaluation/
   train_repetition_classifier.py`: nested-CV-selected L2 (20.0) on all
   250 clips' 388 `word_repetition`/`sound_repetition` events, saved to
   `models/repetition_corroboration_classifier.npz` (~17KB, committed —
   this project's first internally-trained, shipped model artifact;
   every other pretrained component here is used zero-shot).
2. **Refactored for a real architectural boundary.** `profiling/
   evaluation/` is documented as "not needed to run the app itself" —
   importing evaluation code into the core detector would violate that.
   Extracted the shared, model-agnostic primitives (`load_encoder`,
   `extract_last_layer_states`, `pool_span`, `cosine_distance`) into a
   new core module, `profiling/encoder_embedding.py`; `profiling/
   evaluation/encoder_features.py` now imports them from there instead
   of duplicating them, re-exported for backward compatibility. Verified
   with a full regression run before proceeding (59/59 unaffected).
3. **Implemented the gate.** `profiling/repetition_classifier.py`
   (`RepetitionClassifierContext`), wired into `detect.py` as a hard
   gate on `word_repetition`/`sound_repetition` candidates — the same
   architectural role Praat-gating plays for `prolongation`. New config
   key `require_repetition_classifier_confirmation`, default `true`.
4. **Benchmarked the integrated detector honestly.** `profiling/
   evaluation/benchmark_integrated_gate.py` reconstructs each event's
   out-of-fold cross-validated prediction (never the final model scored
   on its own training data, which would be optimistic) and translates
   it into real TP/FP/FN counts: `Any` (both types) F1 0.631 -> 0.890,
   driven by an 89% false-positive reduction (209 -> 22) at a 10% recall
   cost (179 -> 161 TP) — the same precision-for-recall trade shape
   every other audio-native corroboration change in this project has
   made.

**Two real bugs caught by the existing test infrastructure, before any
benchmark was trusted — exactly what that infrastructure is for**:

- **An eager-loading design flaw**, caught because the full regression
  suite (which must stay fast and real-model-free) started hanging.
  `RepetitionClassifierContext`'s first implementation loaded the
  encoder in `__init__` whenever `audio_bytes` was given, regardless of
  whether a repetition candidate existed — meaning *any* clip with audio
  would trigger a real, multi-second-plus model load. Fixed by deferring
  loading to the first actual query (`available`/`confirms_repetition`).
- **A second, sharper version of the same bug**: even after making
  loading lazy, the gate's *evaluation* (`rc_ok = ...`) was computed
  unconditionally for every adjacent token pair (`i > 0`) in `detect.py`,
  not only inside the branches where a candidate was actually found —
  so a token pair with no repetition at all (e.g. "go"/"now" in `tests/
  test_detect_fusion.py`) still triggered a real encoder-load attempt.
  Root-caused via direct isolation (`timeout`-wrapped runs narrowing it
  to one file, then `user 0m0.077s` CPU time over 91s wall time —
  near-zero CPU confirmed the process was blocked on I/O, not computing,
  pointing straight at a network-dependent model load rather than a
  computational hang). Fixed by moving the gate check into a lazily-
  invoked closure, only called from inside the two branches
  (`is_fragment_repeat`, the exact-match `elif`) where a candidate
  actually exists — deliberately *not* structured as `if candidate and
  rc_ok:` inside an `elif` chain, because a rejected candidate falling
  through to the near-repetition check below would trivially re-match
  the same identical-after-normalization pair, silently defeating the
  gate.
- **A third, smaller bug, same discipline**: `confirms_repetition()`
  returned a numpy `bool_`, not a Python `bool` — invisible to the
  `if`/`and` logic that actually consumes it, but caught by a unit test
  asserting `is True`/`is False` (which numpy bools fail even when
  "equal"), and a real latent risk downstream (numpy scalar types break
  `json.dumps` by default). Fixed with an explicit `bool(...)` cast.

**Verified on real audio directly, not just via aggregate numbers**,
before trusting the statistical benchmark: a genuine ground-truth
`word_repetition` (clip `103-1240-0000`, index 4) fires with the gate on
(63.5s, confirming the real encoder pass runs) and off (0.7s); a genuine
false positive (clip `103-1240-0018`, index 13) is suppressed with the
gate on and fires with it off, every other event in that clip unaffected.

**Alternatives considered**
- Apply the final trained model back to its own 250-clip training data
  to measure the "integrated benchmark," since the model and data were
  already in hand. **Rejected**: this would be an in-sample,
  optimistically-biased estimate — exactly the kind of number this
  project's measurement-first discipline exists to catch, not produce.
  Reconstructing honest out-of-fold predictions from the already-run CV
  (§12.6.2) gives the same rigor without a fourth encoder run.
- Ship the gate with eager loading and accept that the fast test suite
  would need real-model access going forward. **Rejected**: this
  project's fast, real-model-free unit test suite is a load-bearing
  part of its own development speed and reliability (`HANDOFF.md` §6) —
  breaking that to ship one feature faster would be a worse trade than
  spending the time to make loading properly lazy.
- Leave the live-app latency cost unaddressed and undocumented, since
  fixing it (restructuring `asr.py`'s core transcription call) is out of
  scope for this session. **Rejected the "undocumented" part** — the
  cost is real and stays, but it's now named explicitly in three places
  (`ARCHITECTURE.md`, `VALIDATION.md` §13.2, `ROADMAP.md`) as a specific,
  scoped follow-up, not silently absorbed into "it works."

**Why this choice**
Directly executes the decision the evidence supported, per the owner's
explicit instruction and standing rule 8 — and does so with the same
rigor as every step that led to it: audit before trusting (the two
loading bugs, caught by the test suite doing its job), measure honestly
rather than conveniently (out-of-fold predictions, not in-sample), and
verify on real, individually-inspectable cases before trusting the
aggregate statistics.

**Measured result**
`models/repetition_corroboration_classifier.npz` (new, trained
artifact). `profiling/encoder_embedding.py` (new, core module).
`profiling/evaluation/encoder_features.py` (refactored to import from
it). `profiling/repetition_classifier.py` (new). `profiling/detect.py`
(gate wired in, config key added). `profiling/evaluation/
train_repetition_classifier.py`, `benchmark_integrated_gate.py` (new).
`tests/test_repetition_classifier.py` (new, 7 tests). `config.yaml`,
`README.md` (new config key documented in both). `ARCHITECTURE.md` §4b
(new subsection) + known-limitations entry. `VALIDATION.md` §13/§13.1/
§13.2 (implementation, benchmark, latency limitation). Full suite: 66/66
(was 59/59). `ROADMAP.md` item 17 to be marked fully implemented.

---

## 2026-08-05 — First-principles reassessment of the whole project, written into ROADMAP.md

**What was done**
The project owner asked a deliberately roadmap-blind question: ignoring
everything already planned, and starting only from the project's stated
objective (detect/classify/localize disfluencies in arbitrary microphone
speech), what would the next step actually be? This was treated as a real
audit, not a rhetorical exercise — re-derived by reading code directly
rather than trusting this log's own prior narrative. Direct inspection of
`profiling/evaluation/track_a.py`'s `evaluate()` confirmed
`detect_disfluencies(clip.tokens, audio_bytes=clip.audio_bytes)` is called
with `clip.tokens` sourced from LibriStutter's own ground-truth/
reconstructed annotations (`load_libristutter_dir_with_audio`), never
CrisperWhisper's real ASR output. Because Stage 1 (`run_encoder_signal_
stage1.py`), the raw-embedding collection (`collect_raw_encoder_data.py`),
the corroboration-mechanism comparison (`compare_corroboration_
mechanisms.py`), and the integrated-gate benchmark (`benchmark_integrated_
gate.py`) all reuse this same clip-loading path, **the entirety of item
17 — Phase 3's first and only shipped result — has been validated
exclusively under perfect-transcript conditions.** Separately confirmed:
Track B (the harness that does run real ASR) was last executed as 120
speaker-stratified clips on 2026-08-04, before the `sound_repetition` fix,
the prolongation redesign, or the repetition classifier existed — none of
Phase 2 or Phase 3's shipped detector changes have been re-checked against
it since.

The full reassessment (answering the owner's seven specific questions —
architecture choice, right-problem check, assumptions that shouldn't
survive, roadmap re-ranking, local-vs-global optimization, higher-impact
alternatives, single biggest bottleneck) was written into `ROADMAP.md`
under a new, clearly-scoped "First-principles reassessment" heading,
explicitly marked as a different kind of content from the rest of that
file (analysis, not a chronological log) so a reader knows which lens
they're reading under. A new item 19 (re-run Track B on the existing
120-clip sample, gate on/off, before any further Track-A-only detector
work) was added as the concrete, actionable output of the reassessment;
item 10 and the "deferred learned tier" Near-term bullet were annotated
in place (not renumbered or deleted) noting that their original deferral
reasoning no longer fully applies. `ROADMAP.md`'s "Completed" section was
also trimmed where it duplicated full text already carried by the
numbered Phase 2/3 list (items 1-16), replacing ~5 duplicated paragraphs
with one-line pointers back to the item that already has the detail.

**Alternatives considered**
- Treat this as a rhetorical/framing exercise and write a section that
  restates existing conclusions in the requested voice. **Rejected**: the
  owner's own standing instructions for this project ("optimize toward
  discovering the truth," "treat contradicted expectations as findings")
  apply here as much as to any experiment; a reassessment that doesn't
  risk finding something uncomfortable isn't a reassessment.
- Act immediately on the reassessment's conclusion (e.g., re-run Track B
  in this same session). **Rejected for now**: the owner's immediately
  preceding request explicitly scoped this session to documentation
  ("do not begin any new research, experiments, implementations... unless
  a genuine inconsistency must be corrected first") and the current
  request asked specifically for the reassessment to be written into
  `ROADMAP.md`, not executed. The re-run is recorded as item 19 for a
  deliberate future go-ahead, consistent with standing rule 4.
- Renumber `ROADMAP.md`'s items during the redundancy cleanup for a
  cleaner read. **Rejected**: `VALIDATION.md`, `HANDOFF.md`, and this log
  all cite specific item numbers; renumbering would silently break every
  existing cross-reference for a purely cosmetic gain.

**Why this choice**
Directly answers what was asked, grounded in a re-verified fact (not
re-cited from memory) rather than a repackaging of prior conclusions —
and the fact itself is materially important: it means the project's most
recent and most sophisticated piece of work (the trained classifier) is
also the piece with the least-tested connection to the project's actual
real-world objective, which is exactly the kind of thing standing rule 8
("evidence-constrained, not preservation-constrained") exists to surface
rather than let ride on the strength of Track A's own polish.

**Measured result**
Not a numeric result — a documentation/strategy change.
`ROADMAP.md`: new "First-principles reassessment" section (~7 sub-answers
+ a concrete next-steps list), new item 19, annotations on item 10 and the
Near-term "deferred learned tier" bullet, ~5 duplicated "Completed"
entries trimmed to one-line pointers. No code, config, or test changes;
existing item numbering and every other file's cross-references to it are
unchanged.

---

## 2026-08-05 — Track B validation of the shipped repetition-classifier gate (item 19 executed)

**What was done**
Per the project owner's explicit go-ahead ("temporarily freeze new
feature development and first run Track B on the shipped classifier
[...] measure first, conclude second, implement third"), executed item
19 exactly as pre-registered (`VALIDATION.md` §14, written *before*
either run). Two full `track_b.py` runs over the identical, already-
cached 120-clip speaker-stratified sample (all 120 clips' real
CrisperWhisper output already cached from the 2026-08-04 run — zero new
ASR inference needed for either condition, confirmed `120/120 from
cache` in both logs), differing only in `require_repetition_classifier_
confirmation` (`config.yaml`, toggled and restored to its shipped
default `true` afterward — `git diff --stat config.yaml` confirmed clean
before this entry was written).

Results (overall slice, the real end-user-facing metric):
`word_repetition` TP/FP/FN 1/19/41 -> 1/12/41 (F1 0.032 -> 0.036);
`sound_repetition` 0/0/42 -> 0/0/42 (unchanged, zero candidates either
condition); `Any` (all 5 types) 11/70/175 -> 11/63/175 (F1 0.082 ->
0.085). Compared raw candidate volume (TP+FP at gate-off) against Track
A's own out-of-fold numbers (§13.1, 250 clips): `word_repetition`
0.167 candidates/clip here vs. 1.176/clip there (~7x lower);
`sound_repetition` 0.000/clip here vs. 0.376/clip there (complete
collapse), while ground-truth prevalence of both types in this sample
(0.35/clip each) is the same order of magnitude as Track A's — the
population exists, real ASR's output just isn't producing candidates
from it.

Per standing rule 3 (audit a dramatic result before trusting it — a flat
zero across an entire 120-clip sample, for either condition, is exactly
that kind of result), re-ran gate-off with `--verbose` and hand-inspected
every `true=sound_repetition` case. Found the mechanism: LibriStutter's
`sound_repetition` ground truth is a reconstructed fragment token (a
dataset-representation convention, documented 2026-08-03); real
CrisperWhisper output normalizes disfluent fragments into the clean full
word even at positions it transcribes correctly, so `detect.py`'s
fragment-repeat candidate check (which requires an actual sub-word
fragment token in the transcript) has essentially nothing to match,
independent of ASR accuracy at that position. One case's ground-truth
event surfaced as an acoustic-native `block` prediction instead of
`sound_repetition` — a lead, not yet investigated further, that the
acoustic signal may still be present and simply mis-routed by the
current type taxonomy.

**Alternatives considered**
- Treat the technically-positive `Any` F1 delta (0.082 -> 0.085) as a
  clean "transfers, no further action" result and move on. **Rejected**:
  the pre-registered protocol itself named this exact risk ("can't be
  quietly upgraded to confirmed after the fact if it turns out to look
  favorable") — reporting only the aggregate number without the
  candidate-volume context and the small-n caveat (1 true positive total
  for `word_repetition`, 0 for `sound_repetition`, in either condition)
  would be technically true and substantively misleading.
- Re-run Track B with fresh ASR inference instead of the existing cache,
  to rule out the cache itself as stale or unrepresentative.
  **Rejected**: `track_b.py`'s `events` are always recomputed fresh from
  cached `hyp_tokens` regardless of config (`_save_cache`'s own
  docstring, confirmed by direct code reading before relying on it) — the
  cache holds ASR output only, never detector output, so re-running ASR
  would burn significant time (~30-90s/clip x120) to reproduce identical
  `hyp_tokens` already on disk. Isolating exactly one variable (the gate)
  while holding ASR output fixed is the correct experimental design here,
  not a shortcut that compromises it.
- Immediately start redesigning `sound_repetition`'s candidate generation
  in this same session. **Rejected for now, tracked instead as `ROADMAP.
  md` item 20**: the project owner's request scoped this step to
  "measure first, conclude second" before further implementation: this
  entry documents the measurement and conclusion; the redesign itself
  needs its own scoping (starting with the hand-trace of a larger FN
  sample item 20 proposes) rather than being improvised on top of a
  result just now confirmed.

**Why this choice**
Directly executes the pre-registered protocol with the same rigor as
every other decision this session: measured before concluding, audited
the most surprising individual number (the `sound_repetition` zero)
before trusting it rather than reporting it as-is, and reported the full,
nuanced picture (mechanism validated safe + real-world impact negligible
+ a specific new structural cause found) instead of forcing a binary
transfers/doesn't-transfer verdict the data didn't cleanly support.

**Measured result**
`VALIDATION.md` §14 (pre-registered protocol, written before either run)
and §14.1 (full results, candidate-volume comparison, hand-checked
examples). `ROADMAP.md`: item 19 marked done with the full result
summary; new item 20 (redesign `sound_repetition`/`word_repetition`
candidate generation for real ASR — now the highest-priority open item,
ahead of item 18 and the deferred-learned-tier bullet, both re-flagged
accordingly); the "First-principles reassessment" section's closing
paragraph updated with the outcome. `config.yaml` unchanged from its
shipped state (`require_repetition_classifier_confirmation: true`) —
the gate stays enabled; this result does not call for disabling it, only
for not treating it as sufficient on its own. No code changes; no test
suite impact.

---

## 2026-08-05 — A separate research track opened: `ASR_RESEARCH_TRACK.md`, `asr-research` branch

**What was done**
Per the project owner's explicit framing ("this feels like a major
research checkpoint rather than simply another roadmap item... treat
this as the beginning of a separate architectural research direction"),
wrote `ASR_RESEARCH_TRACK.md` — a charter document (not implementation)
for a new, dedicated research track, developed on its own branch
(`asr-research`, not yet created at the time of this entry — created only
after `main` is committed, per the owner's explicit sequencing) so `main`
stays stable throughout.

The document: (1) restates item 19's finding as a reframed core question
— "how do we preserve the speech-production information that
conventional ASR intentionally removes" rather than "how do we improve
the detector" — including the project owner's own framing of it ("that
dust is now gold to us," attributed); (2) a formal problem statement
that explicitly does *not* reopen `PHASE_3_ARCHITECTURE_REVIEW.md`'s
two-stage-architecture conclusion — a narrower question about
representation richness, not pipeline structure; (3) restates this
project's seven-type taxonomy as an explicit "what must survive in the
representation" checklist, kept in view before any architectural
discussion, per the owner's explicit instruction to keep the target in
hindsight throughout; (4) a real literature review — 13 verified sources
found via web search this session (not assumed or fabricated), covering
field-level ASR bias against disfluent speech (arXiv:2405.06150),
CrisperWhisper's own design (arXiv:2408.16589, re-read for precisely
what its verbatim claim covers), continual-learning ASR adaptation
(arXiv:2606.14391), multitask joint ASR+disfluency training
(arXiv:2211.08726, arXiv:2409.10177, arXiv:1908.05378), bypassing
decoded text entirely (arXiv:2311.00867), SSL/encoder representation
probing (arXiv:2409.10704, and arXiv:2311.05203 — the latter an
independent, external corroboration of this project's own Stage 1 result:
Whisper's encoder layers, not just decoded text, carry disfluency-
relevant signal), and hybrid fusion (arXiv:2512.13632); one title-relevant
paper (arXiv:2512.02027) flagged honestly as not yet deep-read rather
than cited with unearned confidence; (5) six architectural directions
laid out without commitment, each mapped to what this project already
has (e.g. "richer intermediate representations" is the cheapest, since
`profiling/encoder_embedding.py` and Stage 1's methodology already
exist); (6) five research questions in priority order; (7) a phased,
evidence-gated research plan (Stages A-E, mirroring item 17's own
successful staged/gated structure) with an explicit three-part test for
when a purpose-built ASR/representation would actually be justified;
(8) explicit non-goals, stating this does not commit to building a new
ASR and does not authorize any change to `main` on its own.

Cross-referenced from `CLAUDE.md` (new pointer, alongside the existing
`PHASE_3_ARCHITECTURE_REVIEW.md` one), `DOCS.md` (new file-map entry,
both the quick-reference table and the full table), `HANDOFF.md` (new
reading-order item 11), and `ROADMAP.md` (item 20 reframed as this
track's Stage A, with a pointer to the full plan rather than duplicating
it).

**Alternatives considered**
- Add a large new section directly to `ROADMAP.md` instead of a separate
  document. **Rejected**: the scope here (problem statement, a real
  literature review, six unpicked architectural directions, a five-stage
  research plan, branch charter) is categorically larger than any other
  `ROADMAP.md` item, and `ROADMAP.md`'s own stated purpose is a
  priority-ordered list pointing *at* full reasoning elsewhere, not
  containing it — exactly the pattern already established for the phase
  research-plan/architecture-review documents (`PHASE_2_RESEARCH_PLAN.md`,
  `PHASE_3_ARCHITECTURE_REVIEW.md`), which this new document follows.
- Number this as "Phase 4." **Rejected**: the owner explicitly framed
  this as "a separate architectural research track," parallel to and
  independent of the numbered phase sequence, not the next sequential
  phase closing the current one — Phase 3 is not closed, and this track's
  own outcome (possibly "no purpose-built ASR needed, cheaper
  representation work sufficed") is not yet known, unlike a phase number
  which this project has only ever assigned in hindsight at a close.
- Cite the Mandarin joint-training paper's arXiv ID directly as newly
  re-verified. **Rejected**: it was not found via this session's own
  searches (the ID could not be independently located), so it is
  referenced only via `ROADMAP.md`'s existing citation, not re-asserted
  as freshly confirmed — consistent with this project's discipline of
  not upgrading a claim's confidence without actually re-checking it.
- Begin Stage A's hand-trace (already scoped as `ROADMAP.md` item 20) in
  this same session. **Not done**: the owner's explicit sequencing was
  to finalize `main` for commit first, create the branch, and only then
  continue — this entry documents the charter, not the first experiment.

**Why this choice**
Matches the project owner's explicit read of the situation: item 19 was a
different *kind* of finding than a normal roadmap item (evidence the
ASR stage's representation, not the detector, may be the ceiling for
certain types) and deserves the same weight a phase-opening document
gets — a real literature pass, explored-not-committed architecture
options, and a pre-registered, evidence-gated plan — before any
implementation, on a branch that keeps `main`'s currently-validated state
untouched while that investigation happens.

**Measured result**
Not a numeric result — a new charter document and a set of documentation
cross-references. `ASR_RESEARCH_TRACK.md` (new, ~400 lines). `CLAUDE.md`,
`DOCS.md`, `HANDOFF.md`, `ROADMAP.md` updated with pointers, no code or
config changes. `asr-research` branch not yet created — per the owner's
explicit sequencing, that happens after `main` is committed with this
entry's changes included.

---

## 2026-08-05 — `asr-research` branch created; Stage A (systematic information-loss audit) done

**What was done**
Created the `asr-research` branch off the now-committed `main`
(`e7add1b`). Executed Stage A of `ASR_RESEARCH_TRACK.md` §8 (answers
RQ1): categorized all 186 disfluent ground-truth positions in the
existing 120-clip Track B sample — not a hand-picked subset — into four
causes, using the existing `--verbose` diagnostic output already cached
from item 19's work (`eval_datasets/_gate_off_verbose_output.txt`, gate
off, zero new ASR cost). Categories: (1) ASR transcribed the position
correctly, no candidate generated ("normalized away"); (2) transcribed
correctly, caught as a different type ("mis-routed"); (3) genuine ASR
transcription error, no candidate; (4) genuine ASR error, something
predicted at the misaligned word (not scored as a match).

Result table: `sound_repetition` 19/4/16/3 (of 42); `word_repetition`
17/5/11/8 (of 42, 1 TP); `phrase_repetition` 20/0/9/8 (of 40, 3 TP);
`prolongation` included for completeness but flagged as the wrong lens
(already acoustic-native, not text-alignment-dependent). Headline: for
`sound_repetition`/`word_repetition`, categories (1)+(2) — losses that
happen even when ASR transcribed the position correctly — account for
~53% of all misses, confirming item 19's finding generalizes across the
full sample rather than being a handful of anecdotal cases.

A targeted follow-up (not just re-reading the diagnostic text — re-ran
the actual alignment against cached `hyp_tokens` for every `word_
repetition` position where the current-word alignment was "correct")
found the specific mechanism for that type: 22 of 23 such cases have the
*other* half of the repeated pair deleted or displaced by ASR — a
different, more specific mechanism than `sound_repetition`'s fragment-
token loss, not the same story restated. One case (a genuine, fully
intact, adjacent triple repeat, `['wolf', 'wolf', 'wolf,']`) was still
missed despite nothing wrong with the ASR output — a detector-logic bug
unrelated to this track's question, flagged separately as `ROADMAP.md`
item 21.

**A real bug in the analysis itself, caught before trusting the result
(rule 3 applied to this session's own tooling, not just the project's
production code)**: the first version of the categorization script
determined "true positive" by checking `true_type in predicted` without
first requiring `align == "correct"` — `track_b.py`'s own scoring
function only ever credits a prediction as a match when the reference
position aligns "correct" (a coincidental type-label match at a
mis-aligned hyp word is not attributable back to the reference instance,
by design — see `score_clip`'s own code comment). The bug inflated
`word_repetition`'s apparent TP count from the true value of 1 to 4, and
silently dropped one row entirely (a ref word containing an apostrophe,
`"Heaven's"`, broke a single-quote-only parsing regex). Caught by
reconciling the script's totals against the already-trusted, officially-
scored Track B table (item 19) before writing up any conclusion — found
a real discrepancy, fixed the categorization order and the regex, and
re-ran before this entry was written.

**Alternatives considered**
- Trust the first (buggy) categorization pass since its headline
  direction ("losses happen even at correct-alignment positions") didn't
  actually change once fixed. **Rejected**: the specific counts did
  change (word_repetition TP 4 -> 1, one missing row recovered), and this
  project's own standing discipline (audit surprising or newly-computed
  numbers before trusting them, not just the ones that look dramatic) is
  exactly what caught this before it became a wrong citation elsewhere.
- Investigate the single triple-repeat miss (item 21) as part of this
  track. **Rejected**: it's a plain detector-logic bug (a candidate check
  not handling 3+ identical adjacent words), unrelated to ASR
  representation richness — this track's charter explicitly scopes it to
  representation questions; mixing in an unrelated detector fix would
  blur what this branch is actually testing. Logged on `ROADMAP.md`
  instead, for `main`.
- Move straight to Stage B (representation probe) using only the four
  hand-picked examples already in `VALIDATION.md` §14.1, skipping the
  full 186-position categorization. **Rejected**: `ASR_RESEARCH_TRACK.md`
  §8's own pre-registered exit criterion for Stage A requires enough
  hand-checked cases to trust the categorization, not a few anecdotes —
  the full sweep is what makes the ~50% "normalized away" figure a
  measured result rather than an extrapolation from 3-4 examples.

**Why this choice**
Directly executes Stage A exactly as `ASR_RESEARCH_TRACK.md` §8
pre-registered it, at the systematic scale that section's own exit
criterion required, and applies the same audit-before-trusting discipline
to this session's own new analysis tooling that the project applies to
every other result — a bug in a one-off script is exactly as capable of
producing a wrong "measured" number as a bug in production code, and
deserves the same scrutiny before being written down.

**Measured result**
`ASR_RESEARCH_TRACK.md` §8 (Stage A results section, full table + four
findings + small-sample caveats). `ROADMAP.md` item 20 marked done with
the summary; new item 21 (the triple-repeat detector bug, for `main`,
independent of this track). This entry. No code changes; no test suite
impact (analysis-only, run against already-cached data, on the
`asr-research` branch — none of this affects `main`).

---

## 2026-08-05 — Stage B (representation-level probe) done: mixed result, reported as such

**What was done**
Executed Stage B of `ASR_RESEARCH_TRACK.md` §8 (answers RQ2), per the
project owner's explicit framing: "a hypothesis test, not an
implementation task... a positive result, a negative result, or an
inconclusive result are all acceptable outcomes." The exact protocol was
pre-registered in `ASR_RESEARCH_TRACK.md` (target population, per-clip
fluent centroid with leave-one-out controls, Cohen's d success criteria
fixed at d>=0.5/positive, |d|<0.2/negative, matching Stage 1's own bar)
*before* `profiling/evaluation/stage_b_representation_probe.py` was
written or run.

Built the extraction script reusing Stage 1's encoder primitives
unmodified (`profiling/encoder_embedding.py`'s `extract_last_layer_
states`/`pool_span`/`cosine_distance`), with a new span-selection layer
for Track B's real ASR hyp-token boundaries (Stage 1 only ever used
Track A's ground-truth token boundaries). Two real bugs caught and fixed
before trusting any result, both by reconciling new numbers against
already-trusted ones rather than accepting a plausible-looking output:

1. **Target-identification used `audio_bytes=None`** to avoid triggering
   the classifier gate without touching `config.yaml` — but this also
   silently disabled the acoustic-native detectors (`block`,
   `prolongation`), which could misclassify a Stage-A category-2 case
   ("mis-routed to `block`") as category-1 ("no candidate at all") purely
   because the detector that would have caught it was turned off too.
   Caught because the first pass found 19/18 (`sound_repetition`/`word_
   repetition`) targets against Stage A's already-known 19/17 — a
   1-count mismatch, investigated rather than accepted as "close enough."
   Fixed by passing real `audio_bytes` and forcing the classifier gate
   off via `detect_disfluencies`'s own supported per-call `config`
   override (never touching `config.yaml` on disk) — re-verified an
   exact 19/17 match, across 31 distinct clips, before spending any
   encoder time.
2. Left-over dead code from an earlier draft of the audio-loading logic
   (an unreachable `if False` branch) was caught and removed during
   review before the script was run for real, not left in as harmless
   clutter.

Ran the real encoder pass: 31 clips, 1026s (~17 min) total, ~33s/clip —
consistent with this project's previously-measured range, and lower cost
than the pre-registration's conservative 38-clip estimate (which
included category-2 cases before the bug above was found and corrected).

**Result**: `sound_repetition` — positive, Cohen's d=0.894 (n=19 target,
n=966 control), clearing the pre-registered d>=0.5 bar and close in
magnitude to Stage 1's own original effect (d≈1.05) despite testing a
completely different population and comparison. `word_repetition` —
inconclusive, d=0.428 (n=17 target), falling between the pre-registered
thresholds — direction positive, not established with confidence at this
n, exactly the outcome the pre-registration flagged as plausible in
advance given that type's more indirect test design (probing the
*surviving* word's representation for a trace of a *deleted* partner,
not a direct in-place fragment test).

**Alternatives considered**
- Treat `word_repetition`'s positive-but-sub-threshold result as a soft
  confirmation since the direction agreed with `sound_repetition`.
  **Rejected**: the pre-registration fixed the threshold before seeing
  any data specifically to prevent this kind of after-the-fact
  softening; reported as inconclusive, exactly as pre-registered.
- Extend Stage C to both types on the strength of `sound_repetition`'s
  result alone. **Rejected**: the pre-registration evaluates each type
  against its own evidence; `sound_repetition` clearing the bar says
  nothing about `word_repetition`, which is why they were pre-registered
  and analyzed separately rather than pooled into one number from the
  start.
- Treat the confound (token-duration/word-identity, named in the
  pre-registration before running) as resolved because the result came
  out positive. **Rejected** — restated explicitly in the results
  write-up as an open, unresolved limitation, plus one further
  statistical caveat (control-group non-independence across positions
  from the same clip) found only while interpreting the actual numbers,
  disclosed even though it doesn't change either result's direction.

**Why this choice**
Directly executes Stage B exactly as pre-registered, applies the same
audit-before-trusting discipline to this session's own new tooling that
the project applies everywhere else (the 19-vs-18 count mismatch would
have silently biased every downstream number if accepted rather than
investigated), and reports a genuinely mixed result as a mixed result —
per the owner's own explicit framing, an inconclusive or negative outcome
here would have been exactly as valuable and exactly as reportable as the
positive one that came out for `sound_repetition`.

**Measured result**
`profiling/evaluation/stage_b_representation_probe.py` (new, research
code only — not imported by the live app). `ASR_RESEARCH_TRACK.md` §8
(Stage B results section: table, per-criterion verdicts, limitations,
and what this resolves for the decision gate — proceed to Stage C scoped
to `sound_repetition` only). `ROADMAP.md` updated with the outcome.
`eval_results/20260805T211000_stage_b_representation_probe.json` (raw
distances, saved). No production code changed; full test suite unaffected
(new file has no test coverage yet — analysis script, not app code, same
convention as other `profiling/evaluation/` scripts). On the
`asr-research` branch only — `main` untouched.

---

## 2026-08-05 — Interpretation added before Stage C: uncertainty, rationale, competing hypotheses

**What was done**
Per the project owner's explicit request, pushed `main` and
`asr-research` to GitHub before any further work (`git push origin
main`, `git push -u origin asr-research`), then re-fetched and directly
compared local vs. remote commit hashes (not just trusted the push
output) to confirm both branches are backed up: `main` local/remote
`e7add1b`, `asr-research` local/remote `d68de95`, both exact matches,
`git branch -vv` showing no ahead/behind drift.

Added an "Interpretation" section to `ASR_RESEARCH_TRACK.md` §8, written
before any Stage C code, stating precisely what Stages A and B have and
haven't established (a real aggregate group-level effect for `sound_
repetition`, not yet an instance-level, actionable, or confound-resolved
one), three named competing hypotheses Stage C should distinguish
(duration confound / genuine acoustic signature / real-but-not-yet-
actionable), and one concrete design consequence: Stage C's own
pre-registration should include an explicit duration-only baseline
comparison arm, not just "does the new detector work in isolation,"
since that is the direct test that separates the duration-confound
hypothesis from the genuine-signature one.

**Alternatives considered**
- Proceed straight to Stage C's implementation without a written
  interpretation step. **Rejected**: the owner explicitly asked for this
  as a distinct, prerequisite step, and it surfaced a concrete
  methodological improvement (the duration-only baseline arm) that
  Stage C's own pre-registration would otherwise have missed — writing
  the interpretation first paid for itself immediately, not just as
  documentation hygiene.
- Trust the `git push` command's own success output as sufficient
  confirmation of backup. **Rejected**, per the owner's explicit
  instruction to verify, not assume: re-ran `git fetch` and compared
  hashes directly.

**Why this choice**
Matches this track's own standing discipline (§8's own stage-gate
pattern) applied one level up: before spending real effort on Stage C's
implementation, state clearly what is and isn't yet known, so Stage C is
designed to answer the actual open question (is this signal usable and
genuine) rather than just "does a stronger version of Stage B's number
exist."

**Measured result**
Not a numeric result — an interpretive addition and a verified backup.
`ASR_RESEARCH_TRACK.md` §8 (new "Interpretation" subsection). No code
changes. `main`/`asr-research` confirmed pushed and in sync with
`origin` before this entry was written.

---

## 2026-08-05 — Stage C done: H1 refuted, H2 supported, H3 also supported

**What was done**
Executed Stage C of `ASR_RESEARCH_TRACK.md` §8 exactly as pre-registered
(target/control population reused from Stage B's saved data, no new
encoder passes; a duration-only baseline arm added specifically to
separate H1 from H2, per the Interpretation step's own recommendation).
Built `profiling/evaluation/stage_c_duration_baseline.py`, computing ROC
AUC (pure Python, Mann-Whitney-U formula, no scipy/sklearn dependency —
sanity-checked against hand-computed known-answer cases, including a
tie-handling case, before trusting it on real data) and precision at two
pre-declared recall points for both arms.

**Three real bugs caught by this stage's own safety checks, each
investigated before proceeding, not one silently patched over the
result**:
1. A crash on real ASR tokens with `start`/`end` of `None` — fixed by
   excluding those positions from the duration arm, with the count
   reported explicitly.
2. A large mismatch (3347 vs. Stage B's 966 saved control positions) —
   traced to iterating all 120 clips instead of only the 31 Stage B
   actually drew its control pool from. Fixed by applying the identical
   population filter.
3. A residual 1-position mismatch (967 vs. 966) after that fix — traced
   to the exact same missing-timestamp position also making `pool_
   span()` return `None` on the encoder side, meaning Stage B's own
   arm silently excluded it too; once accounted for, both arms compute
   over the identical 966-position population.

**Result**: encoder arm AUC=0.723 (P@R>=0.5=0.047, P@R>=0.7=0.029);
duration-only baseline AUC=0.483 (essentially chance). **A correction to
the pre-registered assumption's direction, checked directly rather than
assumed**: the protocol expected fragment-absorbing tokens to be
*longer*; the data shows target positions are, if anything, very
slightly *shorter* (mean duration z=-0.139) — duration carries no
meaningful signal in either direction at this sample size. H1 (duration
confound) is refuted, not merely unconfirmed. H2 (genuine signature) is
supported — the encoder arm clears chance and clearly beats the duration
baseline. H3 (real but not instance-actionable) is *also* supported,
simultaneously: absolute precision at a useful recall (4.7% at 52.6%
recall) is too low for standalone use given realistic 19-vs-966 class
imbalance, regardless of the signal being genuine.

**Alternatives considered**
- Force H1/H2/H3 into a single selected outcome, since the
  pre-registration was written as if they were mutually exclusive.
  **Rejected**: the actual result shows two can hold simultaneously (a
  genuine signal that isn't yet practically usable alone) — reporting
  that precisely is more useful and more honest than picking the
  single "winning" hypothesis the pre-registration's own framing implied
  would exist.
- Treat the 967-vs-966 count mismatch as close enough to ignore.
  **Rejected**: investigated fully, traced to a specific, understood
  mechanism (a shared missing-timestamp exclusion on both arms) rather
  than assumed benign — the assertion was written specifically so a
  mismatch this size would stop the run and require an explanation, not
  silently pass.
- Proceed to Stage D (fine-tuning) on the strength of H2 being
  supported. **Rejected**: Stage D's own pre-registered gate requires
  richer-representation approaches (Stage B/C) to be *insufficient* —
  H1 being refuted here is the opposite of that condition. The
  evidence-justified next step is a fusion-style Stage C revision, not
  escalation to a materially more expensive intervention this evidence
  doesn't yet call for.

**Why this choice**
Directly executes Stage C exactly as pre-registered, with the duration
baseline arm the Interpretation step identified as necessary, and
applies the same audit-before-trusting discipline to three real bugs in
this session's own new tooling that the project applies to production
code and to every other measured result this track has produced.

**Measured result**
`profiling/evaluation/stage_c_duration_baseline.py` (new, research code
only). `ASR_RESEARCH_TRACK.md` §8 (Stage C results section: bug log,
results table, per-hypothesis verdicts, the corrected duration-direction
assumption, and the recommended next step). `ROADMAP.md` updated with
the outcome. `eval_results/20260805T215037_stage_c_duration_baseline.json`
(raw scores, saved). No production code changed. On the `asr-research`
branch only — `main` untouched.

---

## 2026-08-05 — End-of-session documentation, consistency, and handoff pass

**What was done**
Per the project owner's explicit request to formally close today's
session, performed a full documentation/consistency audit rather than
starting a new experimental stage:

1. **Objective hierarchy stated explicitly and consistently.** Added a
   new section to `CLAUDE.md` ("The objective, stated precisely") stating
   the project's long-term objective is not to optimize Whisper, preserve
   the existing architecture, maximize transcript quality, or pursue any
   component for its own sake — it is the most accurate, explainable,
   scientifically grounded speech-disfluency detector possible from the
   user's audio — and the five-point evidence hierarchy that follows from
   it (audio is fundamental; ASR is one subsystem; the transcript is one
   evidence source, not ground truth; encoder/acoustic/confidence signals
   are complementary, evidence-gated; architecture stays evidence-driven
   in both directions). Reinforced consistently, not just declared once:
   `ARCHITECTURE.md` §1's opening reframed to lead with the detection
   objective (transcription explicitly named as scaffolding, not the
   deliverable); `README.md`'s opening paragraph corrected the same way
   for a general reader.
2. **Internal consistency audit of `ASR_RESEARCH_TRACK.md`** (the
   heaviest-edited file this session): fresh full read start to finish;
   found and fixed one typo and confirmed no other structural
   inconsistencies from today's many sequential edits.
3. **Staleness audit and correction across every major doc**: `CLAUDE.md`
   and `HANDOFF.md` both still described the ASR research track as "not
   yet implemented" / "charter document... nothing implemented yet" —
   stale as of Stage C's completion earlier today. Corrected in both,
   plus `HANDOFF.md` §4's "what's proven" list extended with today's two
   strongest new findings (the traced normalization mechanism; the
   genuine-but-not-yet-actionable encoder signal), `DOCS.md`'s file-map
   entry updated, and a new dated status paragraph added to
   `VALIDATION.md`'s top (append-style, prior paragraphs kept per that
   file's own convention) pointing to `ASR_RESEARCH_TRACK.md` for work
   that doesn't belong in `VALIDATION.md` itself.
4. **Formal end-of-session handoff written into `ASR_RESEARCH_TRACK.md`**
   (new section, end of file): a summary of the full day (not just this
   track), the strongest evidence-backed conclusions, named remaining
   uncertainties, the exact proposed next stage (a fusion-style
   combination, not fine-tuning) with the specific hypotheses it would
   test, a fully-ordered next-session plan (deliverables, success
   criteria, stopping conditions) written so a future session can act on
   it directly without re-planning, and explicit recommendations/risks
   (do not merge to `main` without a deliberate decision; do not let a
   disappointing fusion result cause a silent jump to Stage D; the
   control-group independence caveat should be resolved before this
   result is cited anywhere higher-stakes; `word_repetition` should not
   be quietly dropped).

**Alternatives considered**
- Begin the fusion-style Stage C revision (the evidence-justified next
  step) in this same session, since the reasoning for it was already
  clear. **Rejected**, per the owner's explicit instruction: "do not
  begin the next experimental stage... simply for the sake of progress."
  Scoped fully in the handoff instead, for a deliberate next start.
- Leave the objective-hierarchy statement implicit (it was already
  present in scattered form — `CLAUDE.md`'s original "scaffolding, not
  the end product" line, `HANDOFF.md` §1's "not trusting the ASR
  transcript as ground truth"). **Rejected**: the owner asked for it
  stated *consistently*, not just present somewhere — scattered implicit
  alignment is exactly the kind of thing a full audit is supposed to
  catch and make explicit rather than assume is already sufficient.

**Why this choice**
Directly executes the requested close: a repository another researcher
(or Claude instance) could pick up cold, understand the evidence and the
reasoning without reconstruction, and continue from an unambiguous next
step — the same standard this project has held every individual result
to today, applied once more at the level of the whole session.

**Measured result**
Not a numeric result — a documentation/consistency pass. Files touched:
`CLAUDE.md`, `ARCHITECTURE.md`, `README.md`, `HANDOFF.md`, `DOCS.md`,
`VALIDATION.md`, `ASR_RESEARCH_TRACK.md` (typo fix + new end-of-session
handoff section), `ROADMAP.md` (via `CHANGELOG.md` pointer only, no
further change needed there). No code changes. Full test suite
unaffected (documentation-only). Repository confirmed ready for the
owner's own `git push` of both branches, per their explicit request not
to push in this session.

---

## 2026-08-06 — Stage C2 done: Praat voice-quality fusion — clean negative result

**What was done**
Continued directly from the prior session's end-of-session handoff, per
the owner's go-ahead. Executed the "exact proposed next stage" from that
handoff: a fusion-style revision of Stage C. Before writing any code,
checked the handoff's own first-named fusion candidate (Stage A's
"mis-routed" finding, n=4 `sound_repetition` cases caught as a different
type) directly against Stage A's own category definitions — found it
degenerate for Stage C's target population (category 1, Stage C's n=19,
is defined as having *zero* other-detector predictions, by construction
mutually exclusive with category 2's "mis-routed") and too small (n=4)
to test quantitatively on its own regardless. Re-scoped, documented as a
dated addendum in `ASR_RESEARCH_TRACK.md` before proceeding, to a
different, well-powered, already-available signal: Praat-derived voice-
quality features (`profiling/acoustic.py`'s existing `_praat_features` —
pitch, pitch stability, jitter, shimmer, HNR), already used elsewhere in
this codebase for `prolongation` corroboration.

Pre-registered the full protocol (screen each feature individually at a
low AUC>=0.55 bar before combining any with encoder-distance, to avoid
combining noise features and mistaking the resulting inflation for a
real effect; combination rule fixed in advance as max-of-z-scores, no
trained model, consistent with this track's "cheapest version that can
test the hypothesis" discipline) before writing `profiling/evaluation/
stage_c2_praat_fusion.py`. Sanity-checked on a small clip subset first
(a scope-limited dry run correctly tripped the count-check assertion,
confirming the safety check itself works) before the real 120-clip run.

**Result**: none of the five Praat features cleared the screening bar —
`pitch_hz` 0.549, `pitch_std_hz` 0.471, `jitter` 0.527, `shimmer` 0.507,
`hnr` 0.452, all within noise distance of chance (0.5), well below
Stage C's own encoder-only AUC of 0.723 on the identical population. Per
the pre-registered protocol, the fusion combination step was correctly
never attempted — combining these with encoder-distance would only add
noise. A real, clean, specific negative result: Praat voice-quality
features are ruled out as this track's next signal for `sound_
repetition`, distinct from "fusion doesn't help" in general.

**Alternatives considered**
- Force a fusion combination anyway, using the best-scoring feature
  (`pitch_hz`, AUC=0.549) even though it didn't clear the bar, since
  "some signal is better than none." **Rejected**: this is exactly what
  the pre-registered screening step exists to prevent — a feature at
  0.549 is statistically indistinguishable from chance at n=19/966, and
  combining it would very likely just inflate apparent performance
  through noise, not add real information.
- Treat the mis-routing lead (n=4) as still worth a formal statistical
  test, given it was the handoff's originally-named first candidate.
  **Rejected**: checked directly and found it degenerate on Stage C's
  own population (constant, not discriminative, by construction) and
  too small to test on its own regardless — recorded as a dated
  re-scoping addendum rather than silently swapped for the Praat test.
- Try additional feature engineering (e.g. delta/derivative features,
  or a shorter/longer analysis window) to salvage a Praat-based signal
  before concluding it doesn't work. **Rejected for this session**: the
  pre-registered protocol's success criteria were fixed in advance
  specifically to prevent this kind of post-hoc searching for a result;
  a genuinely different windowing strategy would be its own new,
  separately pre-registered experiment, not a silent extension of this
  one.

**Why this choice**
Directly executes the prior session's own handoff plan, applies the same
pre-register-before-code and screen-before-combine discipline as every
other stage in this track, and reports a negative result with the same
weight and rigor a positive one would have gotten — consistent with the
owner's explicit standing instruction that positive, negative, and
inconclusive results are all acceptable outcomes.

**Measured result**
`profiling/evaluation/stage_c2_praat_fusion.py` (new, research code
only). `ASR_RESEARCH_TRACK.md` §8 (Stage C2 pre-registered protocol +
results, including the mis-routing re-scoping addendum) and an updated
end-of-session handoff section (dated 2026-08-06 update note, three
remaining options recorded, none chosen unilaterally). `ROADMAP.md`
updated with the outcome. `eval_results/20260806T135433_stage_c2_praat_
fusion.json` (raw screening/combination data, saved). No production
code changed. On the `asr-research` branch only — `main` untouched.

---

## 2026-08-06 — First-principles reassessment: still the right trajectory, not yet time to leave CrisperWhisper

**What was done**
Per the project owner's explicit request after Stage C2, conducted a
first-principles reassessment of this track's whole trajectory —
specifically whether continued investment in CrisperWhisper's own
representation is evidence-driven or has become attachment to the
existing architecture. Structured as requested: evidence (directly
measured facts, listed exhaustively across Phase 1 through Stage C2),
inference (reasoned conclusions not directly measured — the loss is
most consistent with a decoder-stage, not encoder-stage, phenomenon),
and engineering judgment (labeled as such, not fact).

**Conclusion**: not yet time to move to a different or purpose-built
ASR — that would be exploration with zero supporting evidence (RQ3,
whether any finding is CrisperWhisper-specific, has never been tested),
symmetrically as ungrounded as clinging to CrisperWhisper by default
would be in the other direction. What the evidence actually supports is
narrower and still-actionable: only one encoder layer, threshold-only
combination, no decoding variation, and a modest sample have been
tried — real headroom remains untested *within* the current
architecture. Two concrete experiments identified as the immediate next
steps, ahead of RQ3/Stage D: encoder layer depth (literature-motivated,
never tested — only the default last layer has ever been used) and
decoding-parameter sensitivity (the most direct available test of the
decoder-stage inference itself).

**Alternatives considered**
- Treat Stage C2's negative result as sufficient grounds to escalate
  toward Stage D (fine-tuning) or RQ3 (a different ASR backend).
  **Rejected**: two negative *fusion* results (duration, Praat features)
  say something about which extra signals help, not about whether
  CrisperWhisper's own representation — at other layers, or under other
  decoding settings — has been exhausted. Escalating past untested,
  cheap, in-architecture options on the strength of unrelated negative
  results would be evidence-free in exactly the way the owner asked this
  reassessment to check for.
- Conclude the heuristic space is fully explored and recommend no further
  work within the current architecture. **Rejected**: explicitly checked
  against what has and hasn't actually been tried — encoder layer depth
  and decoding parameters have literally never been varied anywhere in
  this project, a fact, not an assumption.
- Recommend switching to a different pretrained ASR now, given two
  fusion attempts failed. **Rejected**: no experiment in this project
  has ever run a second ASR backend (item 10/RQ3 remain fully open) — a
  recommendation to switch would rest on zero comparative evidence,
  exactly the sunk-cost-avoidant-but-still-ungrounded move the owner
  explicitly warned against.

**Why this choice**
Directly answers the question asked, using the complete body of evidence
rather than the most recent (negative) results in isolation, and keeps
the evidence/inference/judgment distinction the owner asked for legible
enough that a future reader can independently check whether they agree
with the judgment layer without having to accept it on authority.

**Measured result**
Not a numeric result — a strategic reassessment. `ASR_RESEARCH_TRACK.md`
gets a new "First-principles reassessment" section at the top of the
file (mirroring `ROADMAP.md`'s own established pattern for this kind of
content). `ROADMAP.md`'s pointer to this track updated with the
conclusion. No code changed by this entry itself; the two recommended
experiments are pre-registered and executed in the entry that follows.

---

## 2026-08-06 — Encoder layer-depth sweep done: the last layer is uniquely informative

**What was done**
Executed the first of the two experiments the same-day reassessment
recommended. Verified the required API directly before writing the full
script (a quick, isolated check that `output_hidden_states=True` on the
cached `WhisperEncoder` object returns all 33 layers' activations from a
single forward pass, and that the last one matches `extract_last_layer_
states`'s existing output exactly) — cheap, and avoided discovering an
API mismatch mid-way through an 11-minute run. Pre-registered the full
protocol in `ASR_RESEARCH_TRACK.md` before writing `profiling/
evaluation/stage_layer_sweep.py`, which extracts every layer from one
forward pass per clip (not one pass per layer) and reuses Stage B/C's
exact pool_span/cosine_distance/leave-one-out-centroid methodology at
each layer, so every layer's AUC is directly comparable to Stage C's
already-known last-layer result of 0.723.

**A real deviation from the pre-registered population, caught and
verified before trusting the result**: the implementation filtered to
clips containing a `sound_repetition` target specifically (18 clips, 551
control positions), not Stage B/C's full 31-clip/966-control pool (built
from clips with *either* `sound_repetition` or `word_repetition`
targets). Rather than treat this as invalidating, checked it directly:
the last layer's AUC on this smaller population (0.721) came out almost
identical to Stage C's own (0.723) — an unplanned internal-consistency
check that passed, giving more confidence in both numbers, not less.

**Result**: the last layer (32) is decisively the best — AUC 0.721,
against a runner-up of 0.383 (layer 26) and most other layers sitting
*below* chance (0.338-0.365, meaning target positions are mildly
*closer* to the fluent centroid at those layers, the opposite direction
from anomalous). No layer beats the last layer; Stage B/C's original
choice (the default last-hidden-state) was already the right one. A
secondary observation — the gradual below-chance-to-above-chance
trajectory through depth — is reported as a plausible, unverified
reading (early layers encode lower-level acoustic detail, the signal
crystallizes only at the final layer), not an independently confirmed
finding.

**Alternatives considered**
- Treat the population deviation as disqualifying and re-run on the
  exact pre-registered 31-clip/966-control population before trusting
  anything. **Not done**: the near-identical last-layer AUC across both
  populations (0.721 vs 0.723) is itself strong evidence the deviation
  doesn't matter for this question — re-running would cost ~11 more
  minutes to confirm something already effectively confirmed by the
  overlap; documented as a deviation instead, per standing rule 1, rather
  than silently treated as identical to the pre-registered plan.
- Investigate *why* early/mid layers show a mild below-chance pattern
  before reporting the main result. **Rejected for this session**: a
  real, interesting observation, but investigating it (attention
  analysis, probing) is a separate, unscoped piece of work — reported
  honestly as an unverified observation rather than either ignored or
  chased down without a plan.

**Why this choice**
Directly executes the reassessment's own first recommended step, applies
the same pre-flight-check and population-verification discipline every
prior stage in this track has used, and reports a clean, decisive result
(no layer beats the last one) rather than either overclaiming the
secondary observation or hiding the population deviation that was found.

**Measured result**
`profiling/evaluation/stage_layer_sweep.py` (new, research code only).
`ASR_RESEARCH_TRACK.md` (layer-sweep results section, plus a dated
update to the reassessment's own "what changes as a result" note:
decoding-parameter sensitivity is now the sole remaining untested
in-architecture lever before this track's stated logic would shift
weight toward RQ3/Stage D). `ROADMAP.md` updated with the outcome.
`eval_results/20260806T144057_stage_layer_sweep.json` (full per-layer
AUC table, saved). No production code changed. On the `asr-research`
branch only — `main` untouched.

---

## 2026-08-06 — Decoding-parameter sensitivity (num_beams) done: clean negative

**What was done**
Executed the second and final of the reassessment's two recommended
experiments. Checked the live app's actual constraint first:
`profiling/asr.py` forces `num_beams=1` as a confirmed workaround for a
real `transformers` bug (word-level timestamp extraction crashes under
beam search — huggingface/transformers #28007/#36093), and the model's
own `generation_config.json` default is `num_beams=5` — so the live app
already runs the *least* beam-search-influenced decoding available.
Recognized that testing `num_beams=5` would hit the same crash if word
timestamps were requested, but that this experiment only needs decoded
*text* (the same alignment methodology Stage A used operates on word
sequences alone) — so the crash was sidestepped by design, not
encountered.

A 4-clip dry run (required to catch bugs before the real run, and it
did: a hand-constructed unit test caught a real false-positive class in
the "recovered" heuristic — single-letter words like "a" trivially
prefix-match many following words by coincidence — fixed with a minimum
fragment length before any real audio was processed) also surfaced a
real cost-escalation finding: ~316s/clip for both conditions combined,
far above the pre-registered ~55-105 minute estimate for the full
31-clip population (num_beams=5 is ~5x the compute of num_beams=1, not
simply additive). Per the same discipline §12.6.1 established for
exactly this situation, scoped down to 40 raw clips scanned (not the
full 120), committed to *before* seeing further results, not adjusted
afterward.

**Result**: 0 of 14 tested `sound_repetition`/`word_repetition`
positions were recovered under `num_beams=5`; mean WER identical between
conditions (0.187 vs. 0.187). A clean, decisive negative — not a weak or
ambiguous null.

**Alternatives considered**
- Abandon the experiment once the cost overrun was discovered, given the
  emerging early signal already looked negative. **Rejected**: the
  scope-down decision (40 raw clips) was committed to based on cost
  alone, before the dry run's own 0/5 result could bias that choice —
  stopping early *because the result looked confirmed* would have been
  exactly the kind of premature-stopping bias this project's own
  discipline exists to prevent.
- Test additional decoding parameters (`repetition_penalty`,
  temperature) in the same run, now that the model was already loaded.
  **Rejected**: the pre-registered protocol scoped this experiment to
  `num_beams` specifically, with `repetition_penalty`/`no_repeat_ngram_
  size` already checked and ruled out as *current* contributing factors
  (both neutral in the default config) before this experiment was
  designed — adding untested knobs mid-run would be exactly the kind of
  post-hoc scope creep pre-registration exists to prevent.

**Why this choice**
Directly executes the reassessment's second recommended experiment with
the same rigor as the first (pre-flight API/bug checks, a dry run that
caught a real logic bug, an honest cost-escalation response matching
established precedent), and reports a clean negative exactly as found —
the owner's own standing instruction that positive, negative, and
inconclusive results are all acceptable applies here as much as
anywhere else this session.

**Measured result**
`profiling/evaluation/stage_decoding_sensitivity.py` (new, research code
only). `ASR_RESEARCH_TRACK.md` (results section: cost/bug findings,
0/14 recovery table, WER identity, what this resolves for the decoder-
stage inference). `ROADMAP.md` updated with the outcome.
`eval_results/20260806T161513_stage_decoding_sensitivity.json` (per-
position detail, saved). No production code changed. On the
`asr-research` branch only — `main` untouched. Both of the
reassessment's recommended experiments are now complete; the full
integrative reassessment this triggers follows in the next entry.

---

## 2026-08-06 — Integrative reassessment: current architecture's cheap investigation is exhausted; item 10 elevated as the next step

**What was done**
Per the project owner's explicit instruction to pause after the
decoding experiment (not continue implementing) and reassess the entire
track, wrote a full evidence inventory spanning every stage (Track B
through decoding-parameter sensitivity, 11 numbered evidence points),
followed by an explicit inference layer and a labeled judgment layer,
answering the owner's question directly: has this project extracted
essentially all meaningful evidence from CrisperWhisper's existing
architecture, or is there still justified reason to continue researching
within it.

Named, not smoothed over, a real discrepancy surfaced by cross-checking
this track's own result against the literature it already reviewed:
arXiv:2311.05203 found deeper Whisper-encoder layers carry *more*
disfluency signal for a comparable task; this track's own layer sweep
found the opposite (only the last layer carries signal, for this
specific task on this specific model). Not reconciled — offered as an
inference-level consideration (possible causes: task framing,
CrisperWhisper's own fine-tuning) rather than resolved as fact.

**Conclusion**: yes, the cheap, representation/decoding-only
investigation of CrisperWhisper specifically is essentially exhausted —
every lever available without training something (layer choice, decoding
width, duration, Praat fusion, mis-routing) has been tried, split
between genuinely positive-but-limited (the core encoder signal) and
negative/inconclusive (everything else). This is explicitly *not* a
recommendation to abandon CrisperWhisper or begin building a different
ASR — `ROADMAP.md` item 10 (a second ASR backend) has never been run, so
"a different architecture would do better" remains exactly as
evidence-free now as it was at the start of this reassessment cycle.
Item 10 is elevated to this track's explicit next step — the one
remaining cheap, no-training piece of evidence that actually informs
whether to invest further in CrisperWhisper (fine-tuning, Stage D) or
look elsewhere, rather than guessing at either.

**Alternatives considered**
- Recommend moving directly to Stage D (fine-tuning) or a purpose-built
  representation, given two clean negative experiments in a row.
  **Rejected**: neither experiment tested whether the finding is
  CrisperWhisper-specific — recommending a training-level commitment
  without that evidence would repeat the exact evidence-free pattern
  this reassessment cycle was opened to guard against, just escalated
  further than the original concern (attachment to CrisperWhisper)
  rather than resolved.
- Recommend continuing to test more cheap levers within CrisperWhisper
  (different pooling strategies, more Praat/acoustic features, larger
  samples of the same experiments already run) before considering item
  10. **Rejected**: the evidence inventory shows every readily-
  identifiable cheap lever has already been tried, with a coherent,
  non-contradictory pattern of results (not scattered noise that a
  slightly different variant might flip) — further variation within the
  same scope has a low expected information gain relative to item 10,
  which answers a structurally different, still fully open question.
- Silently drop the fusion-style Stage C revision floated in an earlier
  session as the "next step" without acknowledging the change.
  **Rejected**: explicitly superseded in this entry and in `ROADMAP.md`,
  with the reasoning stated (more fusion attempts at the current sample
  size address a symptom — signal scarcity — that only more data or a
  different representation source can actually fix, which item 10 and
  Stage D address more directly).

**Why this choice**
Directly answers the question asked, using the complete evidence record
built across this entire track rather than the two most recent results
in isolation, and keeps the evidence/inference/judgment layers legible
enough that a future reader can check the reasoning independently rather
than accept the conclusion on authority — the same standard every other
reassessment and stage in this track has been held to.

**Measured result**
Not a numeric result — an integrative reassessment. `ASR_RESEARCH_
TRACK.md` gets a new "Integrative first-principles reassessment" section
(the culminating one, distinct from the earlier "First-principles
reassessment" written before the two experiments ran). `ROADMAP.md`
item 10 elevated with the conclusion; the track's own summary pointer
updated. No code changed by this entry. Full test suite unaffected
(documentation only).

---

## 2026-08-06 — Phase 2 of the ASR research track formally opened: full design-space plan, pre-registered

**What was done**
Per the project owner's explicit instruction to formally open the next
research phase from first principles — and *not* assume comparing
pretrained ASRs was automatically correct, treating it only as "the
strongest current hypothesis" — re-opened the complete design space
before committing to any direction. Enumerated seven realistic
directions (different pretrained ASR, different pretrained
representation, hybrid architecture, decoder/decoding modifications,
fine-tuning, purpose-built ASR, further acoustic-native extension),
stated what each concretely means and its status before this phase,
and only then selected and justified a specific, pre-registered plan.

Did real, targeted literature research for this phase specifically (not
reused blindly from the earlier 13-source pass): confirmed catastrophic
forgetting / representational drift under fine-tuning is a real,
independently documented phenomenon (both Whisper-specific and general
transformer fine-tuning research), read WavLM's own paper directly
(arXiv:2110.13900) to ground its paralinguistic-sensitivity design
objective as a cited fact rather than an assumption, and verified a
directly relevant empirical result (arXiv:2605.12387: a hybrid Whisper+
acoustic-features model beat both pure-Whisper and pure self-supervised
baselines for a comparable paralinguistic task). **One candidate claim
checked and found not independently confirmable** — a search-engine
paraphrase asserted Whisper "prioritizes linguistic invariance over
prosodic variability" as if from a specific paper; fetched that paper's
actual content directly and found it did not contain the claim — dropped
from citation, reported honestly as unconfirmed rather than cited with
borrowed confidence.

Selected a 3-arm pre-registered design: Arm 1 (stock `whisper-large-v3`
through Track B, item 10's original ask), Arm 2 (the same model's
encoder through this track's own layer-sweep methodology, isolating
whether CrisperWhisper's fine-tuning specifically — not Whisper
generally — narrowed the disfluency signal to one layer), Arm 3
(WavLM-Large's representation, motivated by its own stated design
objective). Wrote exact success/failure criteria per arm, named real
confounders before they could be discovered mid-run (model-size
mismatch for Arm 3; a frame-rate/pooling adaptation this session's
earlier characterization had wrongly treated as trivial, corrected here;
decoding-configuration parity for Arm 1), costed every arm from this
session's own measured rates, and built a full outcome-to-conclusion
mapping covering every combination of arm results.

**Performed the requested adversarial self-critique of this plan
itself, not a token gesture**: checked directly whether this phase is
just "test another representation source" after three already failed
(duration, Praat, more CrisperWhisper layers) — argued explicitly why
Arms 1-3 are categorically different (high-dimensional learned
representations from differently-trained models, not more hand-
engineered features), while acknowledging that distinction is itself
untested until Arm 2 runs. Named a real, standing scope limitation this
phase does not fix: ~45% of Stage A's original losses are ordinary ASR
transcription error, unrelated to normalization, and remain unaddressed
by this or any prior stage. Checked whether recommending WavLM is
independently re-derived or just an old roadmap item recycled — traced
the actual reasoning path (this track's own layer-sweep anomaly
motivated the literature search that arrived at WavLM) and named this
explicitly rather than let the recommendation read as inherited
uncritically.

**Alternatives considered**
- Simply execute item 10 (a second ASR backend) as originally scoped,
  without re-opening the full design space. **Rejected**: the owner
  explicitly instructed against assuming this was correct without
  reassessment — doing so anyway would have ignored a direct
  instruction and repeated the exact "unexamined default" pattern
  earlier reassessments in this track exist to catch.
- Recommend fine-tuning or a purpose-built representation now, given
  "be ambitious" and "never remain constrained by an existing
  architecture." **Rejected**: ambition without evidence is not the
  standard this project holds itself to (`CLAUDE.md` rule 8) — both
  options remain genuinely gated on infrastructure this project doesn't
  have; recommending them now would be bold in language, not in
  evidence.
- Run Arm 3 (WavLM) simultaneously with Arms 1-2 rather than sequencing
  it. **Rejected**: real, asymmetric engineering cost (Arm 3 needs new
  model integration and a frame-rate adaptation; Arms 1-2 reuse existing
  infrastructure almost unchanged) — sequencing lets the cheaper,
  faster arms inform whether the more expensive one is worth its own
  dry-run-first discipline.

**Why this choice**
Directly executes what was asked — a complete plan before
implementation, with every requested element (research questions,
literature, alternatives comparison, direction justification,
pre-registered criteria, confounders, costs, outcome mapping,
self-critique) present and none reduced to a formality — while applying
this project's own standing discipline (cite what's verified, flag what
isn't; cost from measured rates, not assumption; name limitations rather
than let them fade from view) to the planning process itself, not just
to results.

**Measured result**
Not a numeric result — a complete, pre-registered research plan.
`ASR_RESEARCH_TRACK.md` gets a new major section, "Phase 2 of this
research track: comprehensive design-space investigation." `ROADMAP.md`
item 10 updated with the specific 3-arm design; the track's summary
pointer updated. No code written, no experiment run — per the request,
this entry is the plan, not its execution.

## 2026-08-06 — Arm 1 done: stock whisper-large-v3, full pipeline — clean negative

**What was done**
Built `profiling/evaluation/stage_arm1_stock_whisper.py` and ran it
against the exact pre-registered 31-clip/36-position population
(re-derived from the existing Track B cache and confirmed to match
exactly, not re-scanned fresh). Re-transcribed the same 31 clips with
stock `openai/whisper-large-v3` (`num_beams=1`, matching the live app's
decoding configuration — a fair comparison, not the model's
unconstrained default), then re-applied Stage A's own 4-category
classification scheme to the same 36 known-loss ref positions under the
new transcript.

**Deviation from the letter of the pre-registration, recorded rather
than silently absorbed**: the pre-registered success criterion cited
CrisperWhisper's full-audit 45.2%/40.5% normalized-away rates (computed
over Stage A's original 42-position audit) as the comparison baseline.
Re-running that full hand-traced audit for stock Whisper was out of
proportion to this arm's own 30-55 min cost estimate. Reported instead
what fraction of the 36 *specific, already-known* losses got recovered
under the new transcript — a more targeted version of the same question,
stated as a substitution rather than left implicit.

**Alternatives considered**
- Use `profiling/asr.py`'s `CrisperWhisperASR` class directly with a
  different `model_id`. **Rejected**: that class deliberately raises on
  any vanilla-Whisper model ID to protect the live app from silently
  losing disfluencies — correct for production, wrong tool for a script
  whose entire purpose is measuring that exact behavior. Called
  `transformers.pipeline()` directly instead, mirroring the class's own
  proven decoding configuration.

**Why this choice**
Directly executes RQ-A / item 10's original ask with the cheapest
available design: no new data, no new engineering beyond a second
transcription pass, direct reuse of Stage A's own categorization logic.

**Measured result**
0/36 positions recovered (0% `recovered_tp`, both types). Normalized-away
rate: `sound_repetition` 89.5% (17/19, vs. CrisperWhisper's 45.2%
full-audit baseline), `word_repetition` 88.2% (15/17, vs. 40.5%) — higher,
not lower. 4 positions (2 per type) moved to genuine ASR error instead of
staying normalized-away — the bigger model introduced new transcription
errors at positions CrisperWhisper got right, without recovering the
disfluency either way. Mean WER 0.177, comparable to CrisperWhisper's own
range (not a general degradation). Cost: 2533s (42 min) for 31 clips plus
an 873s one-time download/load. **Failure**, exactly as pre-registered —
real evidence this is a general property of large-scale weakly-supervised
ASR, not something specific to CrisperWhisper's own choices.

## 2026-08-06 — Arm 2 done: stock whisper-large-v3 encoder, layer sweep — clean negative

**What was done**
Extended `profiling/evaluation/stage_layer_sweep.py` with a `--model-id`
argument (defaulting to `None` = CrisperWhisper, so existing behavior is
unchanged) and re-ran the identical, already-validated layer-sweep
methodology pointed at stock `whisper-large-v3`'s encoder. Position
selection (which ref/hyp indices to pool from) was held fixed to
CrisperWhisper's own cached hyp_tokens, exactly as the pre-registration
specified — only the encoder processing the audio changed.

**Alternatives considered**
- Recompute target/control positions fresh under stock Whisper's own
  transcript for this arm too. **Rejected**: the pre-registration is
  explicit that Arm 2 isolates the *encoder* variable specifically by
  holding position selection fixed — recomputing positions would
  conflate two different questions (does a different transcript surface
  different positions vs. does a different encoder represent the same
  positions differently).

**Why this choice**
Cheapest arm in the design (reuses existing, already-cost-measured
infrastructure with a one-line change), and the most direct test of
RQ-C: is the last-layer-only concentration CrisperWhisper's fine-tuning,
or Whisper's architecture generally.

**Measured result**
Same last-layer-only pattern: layer 32 (last) AUC=0.680, every other
layer 0.336-0.378 (near/below chance) — structurally identical shape to
CrisperWhisper's own sweep. Same-population comparison (both on the
identical 18-clip/19-target subset): CrisperWhisper's last layer scored
0.721 here vs. stock Whisper's 0.680 — slightly *lower*, not higher or
more distributed. Cost: 555s (9 min) for 18 clips, matching the
layer-sweep's established rate. **Failure**, exactly as pre-registered —
the concentration pattern is a Whisper-architecture property, not
something CrisperWhisper's fine-tuning introduced. Narrows RQ-C's answer
to "not fine-tuning specifically" and shifts the open question fully
onto Arm 3 (a genuinely different architecture).

## 2026-08-06 — Arm 3 done: WavLM-Large representation — negative on the primary metric, one genuine nuance

**What was done**
Built `profiling/evaluation/stage_arm3_wavlm.py`, loading WavLM-Large
via `transformers`' `WavLMModel`/`Wav2Vec2FeatureExtractor` and
extracting encoder states at the same target/control positions Arm 2
used (CrisperWhisper's own cached hyp_tokens as the position source).
Computed last-layer Cohen's d/AUC (the pre-registered primary metric,
directly comparable to Stage B/C's own number) plus a full layer sweep
(`sound_repetition` only, matching Arm 2's scope) since the dry run
showed it was cheap enough to include.

**Confounder resolved rather than left as assumed friction**: the
pre-registration flagged WavLM's frame-rate/pooling as "real, non-trivial
engineering, not a drop-in checkpoint swap." Checked directly before
writing any extraction code: WavLM-Large's conv feature encoder has
total stride 320 at 16kHz, giving exactly 20ms/frame — identical to
`profiling/encoder_embedding.py`'s `FRAME_SECONDS` convention — and
LibriStutter's audio files are natively 16kHz (verified directly, not
assumed). `pool_span`/`cosine_distance` needed zero modification. Recorded
as a correction to the earlier, more cautious characterization, not a
silent scope change.

**Alternatives considered**
- Skip the layer sweep, last-layer stats only (the strict minimum the
  pre-registration required). **Rejected once the dry run landed**: at
  ~10s/clip the full sweep cost almost nothing extra (302s total for 31
  clips, including the sweep), and the pre-registration explicitly said
  "resources permitting" — worth the additional RQ-C evidence at
  near-zero marginal cost.

**Why this choice**
Directly executes RQ-B(ii): does a representation from a model with a
genuinely different (not ASR) pretraining objective, explicitly designed
for paralinguistic sensitivity, carry a stronger signal than anything in
the Whisper family. Sequenced last, after Arms 1-2 both failed — exactly
the pre-registered trigger for running it.

**Measured result**
`sound_repetition` (the primary, directly comparable metric): d=-0.061,
AUC=0.474 — indistinguishable from chance, weaker than both CrisperWhisper
(d=0.894, AUC=0.723) and Arm 2's stock Whisper (AUC=0.680). **Failure**,
exactly as pre-registered. Two things recorded honestly alongside that
verdict, neither reversing it: (1) the layer-depth *profile* is genuinely
different from either Whisper variant — WavLM peaks mid-network (layer
10, AUC=0.617) rather than only at the last layer, consistent with
masked-prediction pretraining distributing signal differently across
depth than a narrowly-supervised decoder objective, though its peak still
underperforms both Whisper arms; (2) `word_repetition` showed a small
positive signal (d=0.259, AUC=0.576, n=17) that CrisperWhisper's own
Stage B never found at all — flagged explicitly as too small to trust
standalone (rule 3: small effect, small sample, the only positive cell
among five arms x two types this phase tested), not elevated into a
claim this evidence doesn't support. Cost: 302s (5 min) for 31 clips plus
a 357s one-time download — cheapest of the three arms.

## 2026-08-06 — Phase 2 integrative conclusion: Failure/Failure/Failure — Stage D costing is the evidence-motivated next step

**What was done**
Applied the pre-registered outcome-to-conclusion mapping table to the
three arms' real results (all Failure) rather than re-litigating or
re-scoping after seeing them. Updated `ASR_RESEARCH_TRACK.md` with a
full "Phase 2 results" write-up (per-arm numbers, the integrative
conclusion, and what it does/doesn't rule out) and `ROADMAP.md` item 10
with the same conclusion and pointer.

**Alternatives considered**
- Treat Arm 3's `word_repetition` signal or its layer-depth-profile
  nuance as a partial success, softening the overall verdict.
  **Rejected**: the pre-registered primary metric for Arm 3 was
  `sound_repetition` (the one type with a directly comparable prior
  number), and it came back at chance. Recording the nuances honestly
  is not the same as letting them dilute a clean negative result on the
  metric that was actually pre-registered.
- Treat this as grounds to immediately begin fine-tuning or a
  purpose-built representation. **Rejected**: both remain genuinely
  gated on infrastructure (GPU, paired data at volume) this project does
  not have — the evidence-motivated action is to formally *cost out*
  that direction (what data, what compute, what a pre-registered success
  criterion would look like), not to attempt it uninstrumented.

**Why this choice**
This is exactly the branch the pre-registration named before any arm
ran (§ "Outcome-to-conclusion mapping," row 3) — following it here is
the entire point of pre-registering it in the first place: the
conclusion was fixed before the data that would produce it existed.

**Measured result**
Five independent, real, pre-registered probes across this track (Stage
C's duration baseline, Stage C2's Praat fusion, the CrisperWhisper layer
sweep, the `num_beams` decoding experiment, and now these three
model-swap arms) have each independently found the same thing: nothing
cheap, off-the-shelf, and representation-only closes the
`sound_repetition`/`word_repetition` normalization gap. Directions (a)
and (b) (a different pretrained ASR; a different pretrained
representation, in-family or architecture-diverse) are now
evidence-closed for this specific problem, not merely untested. Two live
next steps remain, named but not yet executed: formally costing out
Stage D (fine-tuning/data acquisition, §9), and giving direction (g)
(extending the acoustic-native precedent further) a real look first,
since it remains cheap and was deliberately sequenced after, not
instead of, (a)/(b). No change to `main`.

## 2026-08-06 — Direction (g) pre-registered: an acoustic-native `sound_repetition` candidate generator

**What was done**
Given the choice between costing out Stage D and pursuing direction (g)
next, the project owner chose (g) (cheaper, no new infrastructure,
already sequenced first). Wrote a full pre-registered protocol in
`ASR_RESEARCH_TRACK.md` before any implementation, per rule 1: a new
acoustic candidate-generation mechanism for `sound_repetition` — 2+
short, spectrally self-similar voiced bursts in immediate succession,
detected directly from the waveform via `profiling/acoustic.py`'s
existing `segment_voiced()` segmentation, evaluated against
LibriStutter's own ground-truth timestamps (Track-A-style, no ASR
involved at all).

**Why this is not the same idea as the earlier superseded fusion
proposal, checked explicitly rather than assumed**: the fusion idea
(encoder-distance + Stage A's mis-routing signal) was starved at n=19/
n=4 because both signals only exist at ASR-correct or ASR-mis-routed
positions — a population ceiling Phase 2 (Arms 1-3) has now confirmed
will not be enlarged by a different ASR or representation. This
proposal depends on no ASR output whatsoever, so it can evaluate against
the *full* ground-truth `sound_repetition` population (up to 42
instances, Stage A's original count) rather than the 19-position
ASR-correct subset — a genuinely different, larger, non-starved
population, not the same starvation problem under new branding.

**Alternatives considered**
- Implement the detector immediately rather than pre-register first.
  **Rejected**: this is new engineering (a candidate-generation
  mechanism this codebase doesn't have yet, plus a new timestamp-based
  scorer), not a parameter change to existing, already-validated
  infrastructure — exactly the case rule 1 exists for, and the project
  owner's own choice this session was explicitly "scope a concrete
  pre-registered next experiment," not "build it."
- Anchor evaluation to ASR hypothesis positions (matching every prior
  stage in this track). **Rejected**: doing so would silently reimport
  the same starved-population ceiling this proposal exists to escape —
  the entire point of an ASR-independent detector is an ASR-independent
  evaluation.

**Why this choice**
Cheapest live option left in the original 7-direction design space
(no model download, no GPU, no new data acquisition), directly extends
a pattern this codebase already ships and validates (`block`/
`prolongation`), and was named as the nearer-term option ahead of Stage
D in Phase 2's own "Direction justification" before today's results
existed — not a direction invented after the fact to have something
positive to try next.

**Measured result**
Not a numeric result — a complete, pre-registered protocol (hypothesis,
design, population, metric, success/failure criteria, confounders,
cost). No code written yet. `ASR_RESEARCH_TRACK.md` gets a new section,
"Direction (g): an acoustic-native `sound_repetition` candidate
generator." Implementation requires a separate go-ahead, per this
track's standing two-step cadence.

## 2026-08-06 — Direction (g) implemented and run: Failure, with an honest mechanistic diagnosis

**What was done**
Given explicit go-ahead to implement (not just plan) direction (g), with
instructions to keep the same research discipline regardless of outcome,
built `profiling/evaluation/stage_g_acoustic_sound_repetition.py`: a new
candidate-generation function (short, spectrally self-similar voiced
bursts, via `profiling/acoustic.py`'s existing `segment_voiced()`/
`frame_features()`, unmodified) and a Track-A-style scorer against
LibriStutter's own ground-truth timestamps. 7 self-test cases written and
passing (candidate grouping across short/long gaps, similarity math,
scoring math) before any real clip was processed — one real bug caught
this way (a `dataclass` constructor call missing a required argument,
fixed before the first real run).

**A second, more serious bug caught after the first real run, not
before** — worth recording honestly rather than only crediting the
self-test process: the first real run against 120 clips returned
recall=1.000/precision=0.966, an implausibly perfect result for a
first-pass, untuned mechanism. Per rule 3 ("a dramatic-looking number is
a reason to check harder, not report faster"), did not report this
number. Traced directly: `score()` pooled every clip's targets and
candidates into flat lists before matching, so a candidate in one clip
could spuriously match a target in a different clip whenever their
independent, zero-based clip timelines happened to overlap in raw
seconds (routine at 120 clips x ~10-15s each). Fixed by scoring strictly
within each clip; added a dedicated self-test
(`"a candidate in one clip never matches a target in a different clip,
even with identical raw timestamps"`) so this class of bug cannot
silently recur. This is the same category of catch as the decoding-
sensitivity experiment's single-letter-word false-positive fix earlier
in this track — a bug that inflates a result into looking better than
it is, caught by treating an unexpectedly good number with the same
suspicion this project applies to unexpectedly bad ones.

**Alternatives considered**
- Cap candidate `n_bursts` (the count of short segments merged into one
  candidate) at a small value, on the theory that a real stutter rarely
  repeats more than 3-4 times, to filter out very broad multi-burst
  candidates. **Rejected, checked directly rather than assumed safe**:
  compared the `n_bursts` distribution of true-positive candidates
  against false-positive candidates directly — they overlap almost
  completely (both range 2-32, similar medians), so a count-based cap
  would cost real recall at roughly the same rate it would remove false
  positives. Applying it anyway, after already seeing the result, would
  have been exactly the kind of post-hoc retuning rule 4 prohibits
  without an explicit go-ahead — not attempted.

**Why this choice**
Directly executes the pre-registered protocol without deviation: same
population (the 120-clip sample's full ground-truth `sound_repetition`
set, 51 instances), same metric (candidate-generation precision/recall
against ground-truth timestamps), same pre-registered threshold sweep,
same success/failure criterion (meaningfully above the duration-only
baseline or not).

**Measured result**
Baseline (ungated): recall=0.824 (42/51), precision=0.081 (62/766).
Similarity-gated sweep (thresholds 0.30-0.90): precision never exceeds
0.094 at any threshold; recall only drops as the threshold tightens
(0.569 at threshold=0.90). Best-F1 operating point (threshold=0.90,
F1=0.161) is not meaningfully above the baseline (F1=0.147) — the
pre-registered **Failure** outcome, exactly as defined in advance.
Mechanistic diagnosis (checked directly, not left as a bare number):
candidate duration is not the problem (median 0.81s, no candidate
exceeds 50% of median clip duration) — the real cause is that ordinary
fluent speech contains abundant short, similarly-shaped voiced segments
(function words, fast syllables), and the RMS/ZCR envelope-shape
similarity feature this arm tried cannot distinguish those from a
genuine repeated fragment; true-positive and false-positive candidates'
`n_bursts` distributions overlap almost completely. **What this does
and does not establish**: recall=0.824 from a completely naive
mechanism is itself real, positive evidence that `sound_repetition`
co-occurs reliably with a detectable run of short voiced segments — this
is not a "no acoustic signature" result. What failed specifically is
this one cheap feature's ability to discriminate that co-occurrence from
background speech rhythm at usable precision, given a real class
imbalance (51 true instances against ~751 plausible-looking candidate
positions in the sample). Two live next options, neither pursued this
session: a richer per-burst feature (MFCC-based similarity, the
escalation this arm's pre-registration named but didn't yet need to
reach a Failure/Success decision), or treating this as a third
independently-converging piece of evidence (alongside Phase 2's Arms 1-3)
that the cheap, no-new-infrastructure options are narrowing, and Stage D
costing is the better-justified next use of effort.

## 2026-08-06 — MFCC escalation run: Failure, closer to the bar, second real bug caught first

**What was done**
Per the project owner's choice to run the MFCC escalation the original
pre-registration already named, pre-registered its exact parameters
(26-mel filterbank, 13 coefficients, standard windowed-FFT/DCT pipeline,
no new dependency) before implementing, then refactored
`generate_candidates()` to accept a pluggable `similarity_fn` so the
run-detection logic stays byte-identical between the RMS/ZCR and MFCC
passes — any result difference is attributable to the feature alone.
Validated the MFCC extractor on synthetic pure tones (same tone -> high
similarity, different frequency -> meaningfully lower) before touching
real audio.

**A second real bug, caught the same way as the first — this time on a
suspiciously flat result, not a suspiciously perfect one**: the first
real MFCC run returned a nearly unchanged precision/recall curve across
the entire threshold sweep (766 of 766 candidates survived even a 0.80
threshold). Rule 3 treats an unexpectedly *uninformative* result with
the same suspicion as an unexpectedly good one — a feature that fails to
discriminate anything at all is as diagnostic of a bug as one that
discriminates everything perfectly. Measured the actual similarity
distribution directly (3,708 real burst pairs): mean 0.961, 90% of ALL
pairs (not just true repeats) scoring >=0.9. Root cause: MFCC
coefficient 0 is overall log-energy, not spectral shape, and dominates
cosine similarity between any two voiced (energetic) segments regardless
of whether they actually sound alike — standard, well-documented MFCC
practice excludes it for exactly this reason. Excluding it dropped mean
similarity to 0.433 with real spread (min -0.955), confirming the fix.

**Alternatives considered**
- Trust the first (flat) result and report "MFCC provides no benefit."
  **Rejected**: a curve with zero discriminative movement across a full
  threshold sweep, on a feature specifically chosen for richer spectral
  information, was itself the anomaly rule 3 exists to catch — reporting
  it without investigating would have been exactly the "report faster"
  failure mode the rule is written against.

**Why this choice**
Directly executes the escalation named in the original pre-registration
as the fallback once the RMS/ZCR feature proved insufficient (it did),
completing direction (g)'s cheap-feature search exactly as scoped in
advance — one escalation step, not an open-ended search.

**Measured result**
Corrected MFCC result: a genuine precision/recall trade-off curve
(recall falls from 0.824 to 0.157 as the threshold tightens from
ungated to 0.90; precision rises from 0.081 to a peak of ~0.10 around
threshold 0.60-0.80) — real discriminative structure, unlike the
RMS/ZCR pass's nearly flat curve. Best-F1 operating point:
threshold=0.40, F1=0.170 (recall=0.686, precision=0.097), better than
the RMS/ZCR pass's F1=0.161. **Still a Failure per the pre-registered
criterion** (precision meaningfully, >=20% relative F1, above the
RMS/ZCR baseline): 0.170 falls short of the 0.176 bar by a small but
real margin, not adjusted or reclassified after seeing how close it
came. Direction (g)'s cheap-feature search is now complete per its own
pre-registration. Both failures in this arm are mechanistically
explained (feature specificity, not absence of acoustic signal) and
each was reached only after a real implementation bug was caught and
fixed first (cross-clip scoring pollution; MFCC coefficient-0 energy
masking) — this is the most carefully verified negative result this
track has produced for any single mechanism. No change to `main`,
`profiling/acoustic.py`, or `profiling/detect.py`.

## 2026-08-07 — Research review: Stage D not yet justified by this track's own §9 gate; one final bounded experiment authorized

**What was done**
Opened today's session with a full research-thinking-first pass, before
any code, per the project owner's explicit request: reviewed the entire
accumulated evidence base (Stage A through yesterday's direction (g))
as an independent researcher would, answering seven questions with
evidence/inference/speculation kept separate — what's known with high
confidence, what's strongly supported, what's uncertain, what's
disproven, the strongest remaining hypothesis, whether cheap directions
are genuinely exhausted, and whether Stage D is scientifically
justified now or just the only thing left on the original list.

**Key finding, checked against this track's own pre-registered gate
rather than asserted independently**: `ASR_RESEARCH_TRACK.md` §9's
second condition for concluding Stage D is necessary requires richer-
representation approaches to have been "tried and genuinely isn't
enough, not merely untried." Every signal this track has measured
(CrisperWhisper's encoder-distance, stock Whisper's representation,
WavLM's representation, RMS/ZCR acoustic similarity, MFCC acoustic
similarity) was tested *alone*, gated by a hand-set threshold — never
combined, never trained together. By the gate's own language, that
condition was not yet satisfied.

**Alternatives considered**
- Recommend Stage D anyway, since seven single-signal probes had all
  failed and Stage D was "the only thing left" from the original
  7-direction list. **Rejected, explicitly**: the project owner's own
  framing for this review warned against exactly this reasoning
  pattern, and the track's own §9 gate — written before any of today's
  or yesterday's results existed — draws the same distinction between
  "tried and failed" and "merely untried."

**Why this choice**
A combined-signal classifier over already-built signals (Stage C's
encoder-distance, direction (g)'s acoustic candidate features) is real,
concrete, and cheap — no new data, no new model integration — and had
never been attempted, unlike every other direction in the original
design space.

**Decision (project owner, 2026-08-07)**: authorized exactly one final,
explicitly bounded low-cost experiment — the combined-signal
classifier — before moving to Stage D, with an explicit instruction not
to let it become an open-ended optimization track: if it fails,
document that honestly and proceed directly to Stage D planning without
revisiting further cheap variants.

**Measured result**
Not a numeric result — a research review and a scoped decision. Full
review recorded in `ASR_RESEARCH_TRACK.md`'s "Research review
(2026-08-07)"; the combined-signal classifier's own pre-registration and
results follow as separate entries below.

## 2026-08-07 — Combined-signal classifier: a real bug caught (F1=0.000 despite a real input signal alone scoring 0.244), then a clean Failure

**What was done**
Built `profiling/evaluation/stage_combined_classifier.py`, reusing this
track's own existing infrastructure exclusively, per the pre-
registration's explicit "no new signal sources" scope: direction (g)'s
MFCC candidate generator for acoustic features (similarity, burst count,
duration), `profiling/encoder_embedding.py`'s Stage B/C primitives for
encoder-distance (recomputed at each acoustic candidate's own span, with
a documented, necessary adaptation from Stage B/C's original ASR-
hypothesis-token anchoring — see the pre-registration for why), and
`compare_corroboration_mechanisms.py`'s existing nested-CV logistic-
regression infrastructure (the same mechanism that shipped this
project's one other trained classifier, `ROADMAP.md` item 17) for the
model itself. 6 self-tests (feature-matrix construction, median
imputation, the has-signal indicator column) written and passing before
any real audio ran. A pre-registered dry run (3 clips, 117s) confirmed
the estimated cost (~78 min for 120 clips) before the full run was
committed to.

**A real bug caught before trusting the first real result, investigated
per rule 3 rather than reported as-is**: the first full run (120 clips,
~62 min) returned F1=0.000/0.000/0.000/0.000/0.000 for the combined
classifier across all 5 folds — while the encoder-distance-alone
baseline (one of the combined model's own five input features)
independently scored F1=0.244 on the same data. A trained model scoring
literally nothing, when one of its own inputs alone scores real,
non-trivial F1, is not a plausible "no benefit from combining" result.
Diagnosed directly: `compare_corroboration_mechanisms.py`'s reused
`_cv_classifier` hardcodes a `proba>=0.5` decision rule. Confirmed the
mechanism with a synthetic check (not just inferred) — fit the identical
logistic-regression pipeline on synthetic data at this population's
exact ~8% positive rate, with a deliberately real, informative feature
planted in it, and found every held-out predicted probability landed
below 0.5 anyway (max observed 0.234) — a well-documented property of
logistic regression under severe class imbalance (the fitted intercept
correctly reflects the low base rate), not a defect in the reused
fitting code itself.

**Alternatives considered**
- Report the F1=0.000 result as-is. **Rejected**: exactly the kind of
  implausible result rule 3 exists to catch before it's trusted,
  especially given a direct, cheap synthetic check was available to
  confirm or rule out the suspected mechanism before spending more real
  compute re-running anything.
- Leave `_cv_classifier`'s fixed 0.5 threshold as the reported Arm (C)
  result, since it's the "unmodified, reused" version literally named
  in the pre-registration. **Rejected**: Arms (A) and (B) both already
  use a train-fold-optimal threshold (`_best_threshold_by_f1`) rather
  than a fixed cutoff — keeping Arm (C) at a fixed 0.5 while (A)/(B) get
  an optimized threshold would make the three-arm comparison internally
  inconsistent, not simpler. Added `_cv_classifier_optimal_threshold()`
  (same nested-CV L2 selection and fit, only the threshold-selection
  step changed to match (A)/(B)'s own convention) and recorded this as
  a dated pre-registration addendum, written before re-running, per
  rule 1 — a fairness correction required to make the comparison valid,
  not a new tuning surface.

**Why this choice**
Directly executes the pre-registered protocol's actual hypothesis (does
combining these signals help) rather than an artifact of one reused
function's threshold convention not matching this population's class
balance.

**Measured result**
Re-ran the full pipeline with the fix (cost: 3068s/~51 min for the
full data-collection pass, this time cached for reuse). The F1=0.000
bug reproduced identically under the original fixed-0.5 rule (confirming
it is deterministic, not run-to-run noise) — and with the corrected
threshold: (A) MFCC-alone F1=0.147 (P=0.090, R=0.541); (B) encoder-
distance-alone F1=0.244 (P=0.343, R=0.234, real values only, 607/766
candidates = 79.2%); (C) combined classifier F1=0.242 (P=0.314,
R=0.244). **Failure, exactly as pre-registered**: (C) clears the bar
against (A) (0.242 > 0.177) but not against (B) (0.242 < 0.293), and
both were required. In practical terms, (C)'s F1 (0.242) is
statistically indistinguishable from (B) alone (0.244) — the
combination neither helps nor hurts meaningfully; the model appears to
lean almost entirely on the encoder-distance feature. A real
side-finding worth recording: encoder-distance-alone (P=0.343) is the
strongest single number this track has produced against real candidates
at usable precision, though scored against a different (smaller,
acoustic-candidate-anchored) population than Stage C's original number,
so the two are not directly comparable.

Per the project owner's explicit, standing instruction, this closes the
cheap-direction search — no further feature, model, or combination
variant will be tried in response. Next step: Stage D planning,
directly, described in a following entry.

## 2026-08-07 — Stage D planned: scientifically motivated, not currently actionable — a real infrastructure blocker

**What was done**
Wrote the complete Stage D planning deliverable requested: what it's
intended to achieve, why previous directions failed and why Stage D
might succeed where they didn't (clearly labeled as inference/reasoned
hypothesis, not evidence), major engineering/computational/data
requirements, biggest technical risks, expected payoff relative to
effort, and simpler alternatives worth considering first — all in
`ASR_RESEARCH_TRACK.md`'s "Stage D planning (2026-08-07)" section,
before writing or attempting any training code.

**Checked §9's third condition directly against this project's actual
environment, rather than assumed** — the first time this track has done
so: `torch.cuda.is_available()` is `False`; the installed `torch` build
is CPU-only; the only GPU present is integrated (Intel UHD Graphics
620), not CUDA-capable; 17GB RAM; 91GB free disk. No local GPU compute
exists for fine-tuning a Whisper-scale model. Separately, no
large-scale, real (non-synthetic) paired disfluent-speech dataset is
currently accessible: SEP-28k needs real audio acquisition never
attempted (`ROADMAP.md` item 15); FluencyBank needs a CHAT-format
parser this project doesn't have (`ROADMAP.md` item 16); LibriStutter
(already available) is synthetic splicing, and using it as the sole
fine-tuning target risks training a model that learns to recognize
splice artifacts rather than genuine disfluency patterns — a
materially worse risk for a *training* target than it has been for an
evaluation-only set throughout this track.

**Alternatives considered**
- Attempt a minimal or token Stage D implementation anyway (e.g., a
  tiny LoRA fine-tune on CPU, or training on LibriStutter alone despite
  its named risk), to have *something* running today. **Rejected**:
  this project's own §9 explicitly names the correct handling for
  exactly this situation — "a validated future-work item, not a stalled
  implementation" — and a token implementation on inadequate compute
  and unverified-generalization data would risk producing a
  confidently-wrong result, the same failure mode named as Stage D's
  single biggest risk in the planning write-up itself. Forcing an
  implementation here would repeat that risk rather than avoid it.
- Treat the infrastructure gap as something to silently work around
  (e.g., assume cloud GPU access without confirming it, or scope down
  Stage D's ambition without saying so). **Rejected**: per this
  project's own standing rule 8 (evidence-constrained decisions,
  reasoning recorded so a future reader can see why) and the explicit
  instruction to stop at a genuine blocker requiring a new decision —
  this is exactly that blocker, named plainly rather than smoothed over.

**Why this choice**
Directly follows §9's own pre-registered, three-part test (written
before any of today's or the accumulated evidence existed) rather than
either forcing an implementation the evidence doesn't yet support
infrastructurally, or quietly declining to check condition 3 at all.

**Measured result**
Not a numeric result — a complete Stage D design document and a
directly-verified, hard infrastructure fact (no local GPU; no
accessible real training data at volume). Stage D is now scientifically
well-motivated (conditions 1 and 2 of this track's own gate satisfied)
but not currently actionable (condition 3 unmet) — recorded as exactly
that, per §9's own guidance, pending the project owner's decision on
how to proceed (cloud GPU/budget and real-data acquisition scope, the
cheaper real-data-generalization check named in the planning document,
or holding Stage D as validated future work). No code written for
Stage D itself. No change to `main`.

## 2026-08-07 — Final research audit: literature-grounded close-out of the experimental phase, no new experiments

**What was done**
Per the project owner's explicit request to close out the experimental
phase with a rigorous audit rather than continue experimenting,
performed a documentation-and-research-review pass only: no experiment,
model, or dataset was touched. Reviewed the full 12-experiment
progression this track has run (Stage A through the combined-signal
classifier) and classified findings into five explicit categories
(high-confidence / limited-scope / open questions / Stage D hypotheses
/ claims that must NOT be made), specifically to guard against
overclaiming novelty — the project owner named "no existing ASR in the
world can solve this" as an example of exactly the kind of claim this
track's own evidence does not support (only three representations and
two acoustic features were tested, not an exhaustive survey).

**Extensive external literature research, verified directly rather
than summarized from search results**: fetched and read 18 additional
primary sources (foundational ASR paradigms — Whisper, CTC, RNN-T,
Conformer, wav2vec 2.0, HuBERT, WavLM; disfluency datasets — SEP-28k,
KSoF, FluencyBank Timestamped, LibriStutter; disfluency-aware/
preserving ASR — most importantly Kordt et al. 2026's full paper,
fetched as a PDF and read completely, table by table; audio-native
dysfluency detectors — YOLO-Stutter, SSDM; forced alignment; discrete-
audio-token speech foundation models), on top of this track's own
13 already-verified citations from earlier sessions, for a combined,
single bibliography of 19+ sources with title/authors/year/venue/URL.

**Key finding that corrected the project owner's own framing, found and
stated explicitly rather than simply agreed with**: the closest
published precedent to this track's own Stage D hypothesis (Kordt et
al., "Learning to Hear Hesitation," Interspeech 2026) already fine-tunes
Whisper with explicit disfluency tokens using continual learning, and
already independently confirms this track's own catastrophic-forgetting
concern with real, quantified numbers (a documented trade-off between
marker-F1 and general WER). Critically, their `REP` token collapses
word- and phoneme-level repetitions into one coarse marker — it does
not preserve or distinguish the repeated fragment's own content, and
does not separate `sound_repetition` from `word_repetition` at this
project's own taxonomy granularity. This means Stage D's core strategy
(disfluency-token fine-tuning) is **not** novel in the broad sense —
it's already published and working for coarser categories — but the
*specific* fragment-level granularity this project's own taxonomy
targets remains, as far as this review found, unaddressed by any
source reviewed. This materially sharpens (and narrows) the novelty
claim this track can defensibly make.

**Alternatives considered**
- Accept the project owner's framing at face value and write the audit
  to confirm it. **Rejected, per explicit instruction**: "I want you to
  independently verify my interpretation rather than simply agreeing
  with it." The audit found the framing needed a real, specific
  correction (see above) and stated it plainly rather than smoothing it
  into agreement.
- Treat YOLO-Stutter/SSDM (audio-native, ASR-bypassing, *trained/
  learned* dysfluency detectors) as equivalent to this track's own
  direction (g) (hand-engineered RMS/ZCR/MFCC features) and conclude the
  problem is already solved by them. **Rejected**: neither reviewed
  system reports results against this project's own exact fragment-
  boundary definition, and both are architecturally more capable than
  this track's own hand-engineered attempt — named explicitly as the
  strongest "simpler alternative" this review surfaced, not treated as
  a solved problem or silently ignored.

**Why this choice**
Directly executes the fifteen-part audit requested: define the
experimental stopping point, separate evidence/inference/speculation
with especial care around novelty, ground every major finding against
directly-verified literature (not memory or assumption), produce a
defensible novelty assessment, formalize Stage D's requirements and
re-entry gate, and leave a venture/collaborator-ready brief — all
without starting a thirteenth experiment.

**Measured result**
Not a numeric result — a complete literature-grounded research audit.
`ASR_RESEARCH_TRACK.md` gets a new major section, "Final research audit
— 2026-08-07," containing: Experimental Phase Stopping Point (the full
12-experiment table and why this is a deliberate, evidence-closed
stopping point, not an exhausted idea list); Findings Classification
(A-E); Existing Technology & Literature Landscape (paradigm-by-paradigm
review plus a detecting-vs-preserving distinction table); a dedicated
`sound_repetition`-specific literature pass; "How Our Experimental
Findings Fit the Existing Literature" (independently-supported /
partially-supported / contradicted / not-previously-investigated,
per finding); "Open Research Gap and Novelty Assessment" (with explicit
confidence levels and an explicit list of claims not to make); an
existing-approaches-vs-Stage-D comparison table with a recommended,
precise terminology ("disfluency-preserving ASR fine-tuning with a
fragment-level auxiliary objective," not "a new ASR"); a realistic,
non-marketing significance assessment separated into scientific/
technical/engineering/practical-impact categories; a formalized Stage D
Requirements specification (data/compute/model/evaluation) and Re-entry
Gate (do-not-start-until / ready-when); a Venture/Research Collaboration
Brief; and a full Bibliography. `ROADMAP.md` item 10 updated with a
pointer and summary. No code, experiment, or dataset touched. No change
to `main`.

## 2026-08-07 — Deep-pass research-positioning analysis: one material revision, a sharpened novelty claim, no new experiments

**What was done**
Per the project owner's explicit instruction to treat the same-day
"Final research audit" (above) as preliminary input, not a final
conclusion, and to independently verify and expand it rather than
simply agree with it, performed a second, deeper research pass: 16
requested sections (precise research reconstruction with exact
numbers and an explicit "demonstrated experimentally" / "results
suggest" / "not tested" distinction throughout; an architectural
deep-dive on exactly where in the ASR pipeline disfluency information
disappears; structured, individually-analyzed entries for every major
relevant study; a field-vs-findings comparison stating the narrowest
defensible claim per finding, not a blanket "novel"; an honest
evidence-credibility classification; four versions of the current
problem statement; a direct search for the exact "encoder retains it,
decoder doesn't emit it" problem formulation; a 10-family solution
taxonomy; a technical explanation of why standard ASR training
structurally disfavors this project's objective; a critical
(not assumed-correct) re-evaluation of Stage D itself; a brutally
honest paper-worthiness assessment; an explicit "already known /
partially known / underexplored" verdict matrix; an independently
re-verified infrastructure "wall"; an explicit scientific-vs-
engineering-vs-data-vs-compute gap distinction; and a final verdict
plus a concise "Research Position as of 2026-08-07" section. Nine
additional primary sources were fetched and read directly (two — AS-70
and, previously, Kordt et al. — as complete PDFs, table by table), on
top of the 19 already verified earlier the same day, for 26 total
cited sources.

**The one material revision, found by treating the prior audit as
something to test rather than defend, and marked explicitly rather
than silently corrected**: the prior audit claimed no accessible
large-scale real (non-synthetic) dataset provides the fragment-level
`sound_repetition`-vs-`word_repetition` distinction this project's
taxonomy needs. Reading AS-70 (Gong et al., Interspeech 2024) in full
found this claim wrong: AS-70 is real, 48.8 hours, verbatim,
character-level-annotated Mandarin stuttered speech with an explicit
`/r` (sound repetition — "repeated phoneme that do not constitute an
entire character") vs. `[]` (word/phrase repetition) distinction, and
a published wav2vec2.0 baseline achieving F1=65.76% specifically for
`/r` on real speech. **Also found, by reading past the headline claim**:
AS-70's own published ASR fine-tuning experiments strip disfluency
markup before training and scoring CER — their own authors did not run
the fragment-preservation experiment despite having exactly the right
data sitting in front of them. This sharpens rather than resolves the
novelty claim: the gap is not "no adequate data exists," it is "every
individual technique needed has real precedent (including, now,
adequate data for at least one language), and nobody has assembled and
evaluated them together at this specific taxonomy granularity, on real
speech."

**Alternatives considered**
- Treat the prior audit's conclusions as settled and use this pass only
  to add breadth (more citations, more detail) without revisiting its
  actual claims. **Rejected, per explicit instruction**: "verify and
  expand it against the full literature, critically test our novelty
  claims... Do not assume that reaching a 'stopping point' means there
  is nothing left to investigate." Finding and reporting the AS-70
  correction is the direct product of not doing this.
- Soften the AS-70 finding to avoid contradicting the same-day prior
  audit. **Rejected**: the project owner's own instruction was explicit
  — "If the literature shows that something we thought was novel has
  already been done, tell me" — stated plainly instead, with the
  historical reasoning preserved above it, not deleted.

**Why this choice**
Directly executes a genuinely independent, skeptical second pass rather
than a restatement of the first — the value of this exercise is
specifically in finding what the first pass missed or got wrong, which
it did (materially, on the data-availability question).

**Measured result**
Not a numeric result — a complete, independently-verified research-
positioning analysis. `ASR_RESEARCH_TRACK.md` gets a new major section,
"Comprehensive research-positioning analysis — 2026-08-07 (deep pass),"
16 subsections plus a closing "Research Position as of 2026-08-07"
summary and an expanded bibliography (7 new sources, B20-B26). Final
verdict, stated precisely: the barrier is an engineering-and-assembly
gap compounded by a project-specific (English-language) data gap and a
project-specific compute gap — not a field-wide scientific unknown.
Stage D remains justified per this track's own pre-registered §9 gate,
now with a cheaper, newly-identified minimum-viable path (an AS-70-
first, Mandarin, reduced-scale pilot, testing the *method* independent
of the English-data problem, before committing to English-language
data acquisition). Papers A (empirical characterization), B (failure
analysis), and C (representation study) are assessed as supportable
now, pending one real-speech validation pass; Papers D (proposed
architecture) and E (full system) require Stage D's own results first.
`ROADMAP.md` item 10 updated with a pointer and summary. No code,
experiment, or dataset touched. No change to `main`.

## 2026-08-07 — Novelty-falsification pass: L1-L4 operationalized, a further material revision, primary sources traced to the actual paper

**What was done**
Per the project owner's own direct verification of the AS-70 primary
source and a request for a stricter, four-level operationalization
(L1: ASR for stuttered speech; L2: verbatim vs. normalized output; L3:
explicit disfluency-type representation; L4: actual repeated-fragment
*content* preserved, e.g. "c-c-cat," not a generic marker), traced the
Hugging Face `StutteredSpeechASR` model card to its actual technical
paper rather than accepting the card's own (notably incomplete —
"word repetitions and interjections," not sound repetitions)
description. Fetched and read Li, Li, Gong, Wang, & Wu's "Our
Collective Voices" (FAccT 2025) in full — 16 pages including the
appendix, via the authors' own open preprint, not a paywalled or
summarized version — and confirmed it is the technical paper behind
both `StutteredSpeechASR` and the `StammerTalk`/`AS-70` dataset
(overlapping authors, identical scale, identical annotation schema).

**Real, quantified L2 confirmation, directly verified**: the paper's
own fine-tuned Whisper deletion-error rate (the error type triggered by
omitting a repetition) dropped from 26.56% to 2.29% (severe stuttering)
and 15.77% to 1.27% (moderate) after fine-tuning on literal (verbatim,
markup-stripped-but-content-real) transcriptions — genuine, first-party
evidence L2 is solved, on real speech.

**The precise L4 evidence, read directly from the paper's own Tables
3-4**: one concrete example shows a `/r` (sound-repetition) annotation
on a single character; the baseline model's output at that position is
a single wrong homophone character with no repeat structure at all —
even in the same utterance where a word-level repeat elsewhere
partially (if garbled) survived. **A real, honestly-stated gap in what
this paper demonstrates**: no corresponding fine-tuned-model output
example for a `/r` event was found in the paper — only aggregate,
severity-level CER/deletion-rate numbers, not a qualitative comparison
isolated to sound-repetition events. This was not filled by assumption
in either direction.

**A further, material, explicitly-marked revision to this same day's
earlier deep pass**: comparing the annotation convention across both
papers ("小/r明," "自/r卑的" — always a marker on one whole existing
character) against what true L4 requires, this review's own reasoned
inference (stated explicitly as inference, not attributed to either
paper) is that Chinese orthography has no sub-character grapheme
available to write a phonetic fragment as literal text — word-level
repeats can be written as genuinely repeated whole characters (which is
exactly why StammerTalk's word-level content preservation is real and
demonstrated), but a sound-level fragment has no comparable written
form. **If correct, this means AS-70/StammerTalk — despite being the
best real, right-granularity data this entire review found — may be
structurally unable to test this project's actual L4 hypothesis for
`sound_repetition`,** independent of model or training-method choice.
This revises, not merely extends, the deep pass's "AS-70-first,
Mandarin pilot" recommendation: that pilot remains real and useful for
de-risking the *training recipe* (verbatim-target fine-tuning,
catastrophic-forgetting control, using StammerTalk's own directly-
demonstrated approach), but cannot substitute for eventually testing
fragment-content preservation in English or another alphabetic-script
language, which is the specific question Stage D exists to answer.

**Alternatives considered**
- Accept the Hugging Face model card's own claims at face value, since
  it explicitly says the model "preserves disfluencies." **Rejected**:
  the card itself doesn't specify sound-repetition preservation (only
  "word repetitions and interjections"), gives no example output, and
  doesn't define "verbatim" precisely enough to answer L4 — exactly the
  kind of secondary-source gap this project's own citation discipline
  exists to close by going to the primary source instead.
- Treat AS-70/StammerTalk's real, demonstrated L2 success as sufficient
  evidence that L4 is also likely solved, by extension. **Rejected**:
  checked directly instead of assumed — L2 (verbatim vs. normalized)
  and L4 (fragment-content preservation, specifically for sound-level
  repetition) are different claims, and the paper's own qualitative
  examples show a real, unresolved gap at exactly the point that
  matters for L4.
- Present the Mandarin-script structural finding as an established fact
  from the literature. **Rejected**: explicitly labeled as this
  review's own inference throughout, since no source read states it
  directly — consistent with the project owner's own instruction to
  distinguish "demonstrated by our experiments" / "supported by
  literature" / "inferred" / "still hypothetical" precisely.

**Why this choice**
Directly executes the requested stricter falsification pass — the
value of operationalizing L1-L4 and tracing every claim to its primary
source is specifically in finding exactly where the evidence stops,
which it did (at L4 for sound-level fragments), and in catching a
further real revision (the script-structural finding) that the
broader, less strict deep pass did not surface.

**Measured result**
Not a numeric result — a precise, primary-source-traced novelty
falsification. `ASR_RESEARCH_TRACK.md` gets a new section, "Novelty-
falsification pass — 2026-08-07," with the operationalized L1-L4
framework, the revised three-level verdict (L1/L2 already solved; L3
partially solved; L4 for `sound_repetition` specifically remains
unaddressed), and a revised (not silently replaced) recommendation:
Stage D at true L4 preservation, most plausibly in English, is a
genuine contribution, not a reassembly of existing techniques — but the
Mandarin pilot named in the deep pass, while still useful for the
training-recipe question, cannot test the actual hypothesis on its own.
`ROADMAP.md` item 10 updated with a pointer and summary. No code,
experiment, or dataset touched. No change to `main`.

## 2026-08-07 — Application-Objective Decision Analysis: reinterpreting two sessions of research through the project's actual product objective

**What was done**
The project owner issued an explicit, direct course correction: the prior
four entries today (final research audit, deep-pass positioning analysis,
novelty-falsification pass) had drifted into treating "finding a novel,
unaddressed ASR research gap" as the destination, when the project's
actual objective — stated by the owner in full — is an application that
takes a speaker's audio and reliably extracts the disfluencies actually
occurring in it, so they can be localized, classified, and used downstream
(rephrasing, synonym substitution) to help the speaker toward more fluent
speech. Novelty is a side effect worth documenting honestly, never a goal
to steer toward. Per this instruction, inserted a "## PROJECT OBJECTIVE"
section near the top of `ASR_RESEARCH_TRACK.md` (before all prior
sections) stating this explicitly and flagging that later sections'
implicit L4 (fragment-content-preservation) framing was likely the wrong
target read against the actual objective. Then wrote a full 10-part
"Application-Objective Decision Analysis" re-examining this track's
entire evidence base — Stage A/B/C, direction (g), the RMS/ZCR and MFCC
failures, the combined-signal classifier failure, and all four literature
passes — against that corrected objective: (1) what the application
actually needs (detection/localization/classification of disfluencies;
the *intended* clean content, not necessarily the disfluent fragment's
literal content); (2) what the current pipeline already provides, mapped
per-type; (3) the literature re-scored against "solves our actual
requirement," not "is novel" (finding Kordt et al.'s marker-token approach
and Huang et al.'s joint ASR+SED result closer to solving the real problem
than the earlier framing credited them for); (4) the best existing
solution by that criterion, not by novelty; (5) what this track's own
experiments actually found, separated by category and not overgeneralized;
(6) what remains genuinely unsolved for the application (narrowed to a
detection/candidate-generation gap, not an alignment or output-
representation problem); (7) Stage D re-evaluated (not currently
necessary; if it becomes necessary, re-scoped from L4 to L2); (8) a
ranked implementation roadmap (Rank 1: extend item 17's own full-raw-
embedding classifier methodology to candidate generation, no new data or
GPU; Rank 2: a learned acoustic detector matching YOLO-Stutter/SSDM's
architecture class; Rank 3, fallback only: re-scoped Stage D); (9)
novelty kept explicitly secondary, reported accurately either way; (10) a
final, explicit decision statement in the owner's requested format,
naming Rank 1 as the concrete next step. Closed with an explicit "Decision
for Next Session" section per the owner's direct request, telling a future
session to open with implementation (Rank 1), not further research.

**Alternatives considered**
- Treat this as a lightweight addendum to the existing novelty-focused
  sections rather than a first-class section of its own. **Rejected**:
  the owner explicitly required the objective to be placed *above* the
  novelty question in a way "impossible for a future session to mistake,"
  which a buried addendum would not achieve — hence the standalone
  "PROJECT OBJECTIVE" section placed before everything else, plus a full
  dedicated decision-analysis section rather than scattered edits.
- Silently revise the prior four entries' conclusions to match the
  corrected framing. **Rejected**, per the owner's own explicit
  instruction ("Preserve all existing experimental evidence and
  citations. Do not rewrite history or silently change conclusions.") and
  this project's own standing rule 2 (append-only decision log) — the
  prior audits stand as written, with the new section explicitly marked
  as the current governing interpretation that supersedes their framing
  without deleting it.
- Let the decision analysis recommend Stage D anyway, since three prior
  passes had built toward it. **Rejected** — re-reading the same evidence
  against the actual requirement (detection, not fragment-content
  preservation) shows two cheaper, untried, evidence-motivated options
  (Rank 1/2) that were never disqualified by any experiment this track
  ran; recommending Stage D first would have been schedule-momentum, not
  evidence, exactly what standing rule 8 warns against.
- Recommend further literature research as the next step, given how much
  the last two research passes turned up. **Rejected**, directly per the
  owner's own explicit instruction: "Do not continue experimenting
  indefinitely... I want the research to converge toward implementation."

**Why this choice**
This entry is not a new experimental result — it is a correction of what
question this track's evidence should be read as answering. The prior
audits' literature findings, bibliography, and experimental results all
remain valid and cited; what changes is which of them the project should
act on next, evaluated against the application's real requirement instead
of against "is this the first time anyone has done this."

**Measured result**
Not a numeric result — a documentation and decision-analysis pass.
`ASR_RESEARCH_TRACK.md` gets a new top-of-file "PROJECT OBJECTIVE" section
and a new "Application-Objective Decision Analysis — 2026-08-07" section
(10 parts plus a "Decision for Next Session" close), superseding, not
deleting, the novelty-centered framing of the four prior same-day entries.
`ROADMAP.md` item 10 updated with a pointer and summary. No code,
experiment, or dataset touched. No change to `main`.

## 2026-08-07 — Rank 1 (full-embedding candidate-generation classifier): pre-registered, implemented, run — Failure on deployability, real signal confirmed

**What was done**
Per the project owner's explicit go-ahead to proceed with Rank 1 (item
10's own ranked roadmap, above), pre-registered a full 9-part protocol in
`ASR_RESEARCH_TRACK.md` before writing any code — population, ground
truth, baseline, metrics, a concrete decision gate (including an exact,
numeric per-fold stability rule for calling a result Inconclusive,
fixed before running), what would justify Rank 2 vs. Stage D, and the
downstream integration path if the result were Success. The core idea:
item 17's own shipped classifier mechanism ((S1-full, M3): the full raw
~1280-dim CrisperWhisper encoder embedding, nested-CV L2-regularized
logistic regression) has only ever been applied to *corroboration*
(is an already-flagged candidate real or a false positive?). Stage B/C
separately found a real, Cohen's-d-confirmed encoder signal at
candidate-*generation*-gap positions (where the ASR transcript shows
zero surface evidence of a real repetition) but only ever tested a
single hand-picked scalar (cosine distance to a fluent centroid, with a
threshold). Rank 1 asks whether giving the *full* embedding to item 17's
own classifier mechanism, on that same gap population, does better than
the one scalar did. Implemented as `profiling/evaluation/stage_h_
candidate_generation_classifier.py`, reusing `stage_b_representation_
probe.py`'s population-identification logic, `compare_corroboration_
mechanisms.py`'s CV/threshold machinery, and `stage_combined_
classifier.py`'s already-fixed train-fold-optimal-threshold convention,
all unmodified. Self-tested (8/8: decision-gate logic on hand-constructed
win/floor/inconclusive cases, row-packing round-trip) before running the
real, ~33-minute encoder pass over the 58 real LibriStutter clips that
contain at least one qualifying position.

**Result**: primary population (`word_repetition` + `sound_repetition`
combined, n_pos=66, n_neg=1780) — baseline (distance-to-centroid +
CV threshold) F1=0.097 (P=0.055, R=0.462); Rank 1 classifier F1=0.230
(P=0.580, R=0.147). Clears the pre-registered relative bar (>=20%
relative F1) by a wide margin — precision improved roughly 10x. Does
**not** clear the pre-registered absolute usability floor (precision
>=0.15 AND recall >=0.3) — recall (0.147) falls well short. Per-fold
direction was stable (4 wins / 1 loss of 5), so the result is a genuine
Failure under the pre-registered gate, not an Inconclusive call. Per-type
cuts (`sound_repetition` n_pos=37, `word_repetition` n_pos=29) both
landed Inconclusive under the pre-registered per-fold stability rule —
too few examples to call either way, exactly the caveat pre-registered
for them (`word_repetition`'s classifier scored F1=0.000 on every fold,
which looks like an obvious loss on the raw number, but the stability
rule correctly refuses to call it Failure given only 29 positives and a
3-of-5, not 4-of-5, loss margin — a deliberately conservative rule
working as designed).

**Alternatives considered**
- Report this as an unqualified negative result ("full embedding doesn't
  help"). **Rejected**: the data does not support that — precision
  improved ~10x and mean F1 more than doubled, a real, cross-validated,
  directionally-stable effect. The correct, honest characterization is
  narrower: the classifier *works* in the sense of finding a real,
  learnable, precise signal, but at this sample size (66 positives) it is
  too conservative (low recall) to be usable on its own, and this
  experiment cannot distinguish a genuine representation ceiling from a
  sample-size artifact — stated as an open, unresolved question, not
  glossed over in either direction.
- Loosen the absolute recall floor after seeing the result, since
  precision so clearly cleared its half. **Rejected**, directly per
  standing rule 4 (never tune criteria in response to a result without
  explicit go-ahead) and rule 1 (the floor was fixed in the pre-
  registration before the run) — the floor was chosen specifically so
  that a "big relative win, still practically unusable" outcome could not
  be reported as Success by construction, and this result is exactly the
  case that floor exists to catch.
- Treat this Failure as sufficient justification for Stage D. **Rejected**
  per the project owner's own explicit instruction and the pre-
  registration's own §7: this result is about representation/sample-size
  on the encoder side specifically; it says nothing about whether a
  purpose-built acoustic detector (Rank 2) would do better, and Stage D
  remains a reserve option gated on both Rank 1 *and* Rank 2 failing, not
  either alone.

**Why this choice**
This is the exact question the project owner asked to have answered
before deciding what comes next, answered with a pre-registered,
cross-validated, self-tested experiment reusing this project's own
proven infrastructure — no new data, no GPU, no new dependency. The
result is genuinely informative either way: it confirms the full
embedding carries more usable signal than the single scalar reduction
(the underlying hypothesis was right), while being honest that this
specific sample size cannot yet turn that signal into a deployable
detector — precisely the kind of "established / promising / failed /
unknown" distinction the project owner asked this pass to preserve.

**Measured result**
Numeric, reported in full above and in `ASR_RESEARCH_TRACK.md`'s "Rank 1
results" section; raw output also saved to `eval_results/20260807T193221_
stage_h_candidate_generation_classifier.json` and the cached row-level
dataset to `eval_results/_stage_h_rows_cache.npz`. Verdict: **Failure**
on the primary population (deployability floor, not the relative-
improvement bar), **Inconclusive** on both per-type cuts. Per the
pre-registered roadmap, next step is Rank 2 (a learned acoustic
detector), not Stage D. `ROADMAP.md` item 10 updated with a pointer and
summary. New file: `profiling/evaluation/stage_h_candidate_generation_
classifier.py`. No change to `main`.

## 2026-08-07 — Rank 2 (learned acoustic candidate-generation classifier): pre-registered, implemented, run — marginal Failure; synthesis decision names the real next step

**What was done**
Per the project owner's explicit instruction to complete Rank 1's own
decision gate immediately (Failure -> Rank 2) and run the *full* Rank 2
cycle — plan, implement, test, run, diagnose, document, decide — in the
same session without stopping to ask permission, pre-registered a second
full protocol in `ASR_RESEARCH_TRACK.md` before writing any code. Rank
2's question is deliberately different from Rank 1's: not "does the ASR
encoder's representation help" (Rank 1's question) but "does *learning*
a combination of raw acoustic features beat the *hand-designed* single
similarity scalar direction (g) already tried" — acoustic-only, no ASR,
no encoder, matching direction (g)'s own scope exactly. Designed a
26-dimensional feature per acoustic candidate (mean + std of each MFCC
coefficient, 1-12, across the candidate's bursts — generalizing direction
(g)'s pairwise-cosine-similarity idea to any burst count while letting a
classifier weight each coefficient independently instead of forcing all
of them into one fixed geometric formula), fed into the exact same
nested-CV logistic-regression machinery this track has used for every
classifier so far (item 17, the combined-signal classifier, Rank 1).
Explicitly scoped down from YOLO-Stutter/SSDM's own deep, GPU-trained
architecture class to a CPU-feasible approximation, stated as such rather
than claimed as parity. Implemented as `profiling/evaluation/stage_i_
learned_acoustic_classifier.py`, reusing `stage_g_acoustic_sound_
repetition.py`'s exact candidate-span-detection logic (re-derived in a
new function only because the existing `AcousticCandidate` doesn't expose
individual burst spans, not because the span logic itself needed
changing) and `compare_corroboration_mechanisms.py`/`stage_combined_
classifier.py`'s unmodified CV/threshold/decision-gate machinery.
Self-tested (12/12: candidate/burst extraction on hand-constructed
segments, feature construction on synthetic identical-vs-degenerate
bursts, decision-gate logic, row round-trip) before running.

**A real, deliberate scale-up**: because Rank 2 needs no CrisperWhisper
encoder pass, it was run over all 499 real LibriStutter clips (not
direction (g)'s original 120) — 358 seconds total, matching the
pre-registered cost estimate exactly. This gave 245 positive
`sound_repetition` candidates, several times larger than either
direction (g)'s original sample (51) or Rank 1's (66).

**Result**: baseline (mean MFCC similarity + CV threshold) F1=0.133
(P=0.077, R=0.531); Rank 2 classifier (raw per-coefficient MFCC stats +
nested-CV logistic regression) F1=0.163 (P=0.114, R=0.308). Per-fold
direction stable (4 wins / 1 loss of 5). Clears the pre-registered
relative bar (+20%) but only barely — the margin (0.163 vs. a 0.160 bar)
is smaller than either arm's own fold-to-fold F1 range, an honesty check
applied and reported rather than left implicit. Clears the recall half of
the absolute usability floor (0.308 vs. ≥0.3) narrowly, but **not** the
precision half (0.114 vs. ≥0.15) — **Failure** under the pre-registered
gate. Audited before trusting (rule 3): this experiment cannot rule out
that the 26-dim classifier is mostly rediscovering a well-calibrated
version of the same "how similar are these bursts" concept the single
hand-designed scalar already captures, rather than genuinely exploiting
the richer per-coefficient information — reported as a real, unresolved
limitation, not glossed over in either direction.

**Synthesis and next-step decision** (per the project owner's explicit
request to explain, after a Rank 2 failure, whether Stage D or another
low-cost direction is now justified): Rank 1 and Rank 2 together show two
independent representations (ASR encoder embedding; raw acoustic MFCC
statistics) each carrying real, cross-validated, non-random signal for
this exact gap, with *differently shaped* failure modes (Rank 1: large
precision gain, big recall loss; Rank 2: small precision gain, moderate
recall loss) — evidence the two are not redundant. A concrete, specific,
genuinely untried option exists that is meaningfully different from the
already-failed combined-signal classifier (which only ever combined weak
*hand-picked scalars* from each domain, F1=0.242): a classifier combining
Rank 1's *full* embedding with Rank 2's *full* 26-dim acoustic features
together. This remains cheap (no new data, no GPU — one more encoder pass
scoped to Rank 2's candidate population). Per the project owner's own
explicit decision hierarchy (existing components first; cheapest reliable
recovery method second; Stage D only once those demonstrably fail) and
this session's own pre-registered exit condition, **Stage D is judged
not yet justified** — this combined-rich-feature classifier is the
identified low-cost direction that should be tried first; Stage D
becomes the honestly-earned next step only if that also fails.

**Alternatives considered**
- Report the relative-bar pass (0.163 > 0.160) as a straightforward win
  on that half of the criterion. **Rejected**: the margin is smaller than
  ordinary fold-to-fold noise in this same experiment — reporting it
  without that context would be technically true but misleading about
  how meaningful the "beats_bar=True" flag actually is. Stated plainly
  instead, per rule 3's audit-before-trusting discipline.
- Treat this Failure (on top of Rank 1's) as sufficient grounds to move
  to Stage D immediately, since two representation-side experiments have
  now both fallen short. **Rejected**: neither experiment tried the *rich*
  combination of both signals together — only each alone, plus an
  earlier, already-documented attempt using weak scalars from both. Per
  standing rule 8 (evidence-constrained, not preservation- or momentum-
  constrained decisions) and the project owner's own explicit tiered
  hierarchy, a real untried cheap option must be exhausted, or at least
  named and reasoned about, before a materially larger commitment
  (GPU, new data acquisition) is called for.
- Implement and run the combined-rich-feature classifier immediately in
  this same session, rather than only naming it. **Rejected**, directly
  per the project owner's explicit instruction to stop the experimental
  chain at the pre-registered stopping point after Rank 2 and not
  introduce additional variants inside the same pass — named precisely
  as the next step for a future session instead.

**Why this choice**
This is the exact question the project owner asked to have answered:
given both pre-registered experiments' real results, does the evidence
point toward Stage D or toward a cheaper alternative? Naming a specific,
reasoned, genuinely-different next option — rather than either defaulting
to Stage D on momentum or silently proposing an open-ended new
experiment — is the discipline the project's own standing rule 8 and the
Application-Objective Decision Analysis both call for: evidence-
constrained decisions, recorded so a future reader sees the reasoning,
not just the verdict.

**Measured result**
Numeric, reported in full above and in `ASR_RESEARCH_TRACK.md`'s "Rank 2
results" and "Rank 1 + Rank 2 synthesis and next-step decision" sections;
raw output also saved to `eval_results/20260807T202145_stage_i_learned_
acoustic_classifier.json` and the cached row-level dataset to
`eval_results/_stage_i_rows_cache.npz`. Verdict: **Failure** (marginal,
on the precision half of the absolute floor). Synthesis verdict: **Stage
D not yet justified**; the combined-rich-feature classifier (Rank 1's
embedding + Rank 2's acoustic stats together) is the named next step.
`ROADMAP.md` item 10 updated with a pointer and summary. New file:
`profiling/evaluation/stage_i_learned_acoustic_classifier.py`. No change
to `main`.

## 2026-08-07 — Rank 3 (combined rich-feature classifier): the final bounded experiment, Inconclusive with an instability signature, Stage D decision made

**What was done**
Per the project owner's explicit instruction to run the combined-
rich-feature classifier immediately as the final bounded low-cost
experiment — without stopping to ask first, diagnosing genuine bugs if
any, and ending with a clear evidence-based Stage D decision either way,
not another new variant if this one also failed — pre-registered a third
full protocol in `ASR_RESEARCH_TRACK.md`. Rank 3's design choice: reuse
the *exact* population Phase A's original combined-signal classifier
used (the first 120 real LibriStutter clips' acoustic candidates, 766
total, `sound_repetition` only) rather than Rank 2's larger 499-clip
population, deliberately trading sample size for a clean, directly
comparable re-test of Phase A's own already-published Arm A (MFCC-alone,
F1=0.147) and Arm B (encoder-distance-alone, F1=0.244) baselines — the
cleanest possible "did richer features help" comparison. Concatenated
Rank 1's full ~1280-dim encoder embedding (pooled at each acoustic
candidate's overlapping ASR token, median-imputed with a has-signal
indicator where uncoverable — reusing `stage_combined_classifier.py`'s
already-validated imputation convention) with Rank 2's full 26-dim raw
per-coefficient MFCC statistics, into one ~1307-dim feature vector per
candidate, fed through the same nested-CV logistic regression this track
has used throughout. Explicitly pre-registered, before running, that this
dimensions-to-samples ratio (1307 / ~613 per training fold) was
foreseeably worse than either Rank 1 or Rank 2 individually — a real risk
stated up front, not discovered after the fact. Implemented as
`profiling/evaluation/stage_j_combined_rich_classifier.py`. Self-tested
(11/11) — one self-test authoring bug was caught and fixed before
trusting the suite: two hand-constructed comparison arms were
accidentally given identical F1 values, collapsing an intended "beats
one baseline but not the other" test case into a tied, already-correct
Inconclusive result; the decision-gate logic itself was right, only the
test's own input data needed fixing (documented here per rule 3's audit
discipline, applied to test code as much as production code).

**Result**: population reproduced Phase A's original counts exactly
(766 candidates, 62 positive) and both baselines recomputed fresh landed
on Phase A's exact original numbers (Arm A F1=0.147, Arm B F1=0.244) —
confirming this is a genuinely clean, unconfounded re-test, not a
different population being silently compared. Combined classifier
F1=0.254 (P=0.330, R=0.257) — nominally the highest mean F1 of the three
arms. But the pre-registered per-fold stability check (combined vs. Arm
B, the stronger baseline) came back 2 wins / 3 losses of 5 folds — the
combined classifier actually lost to the single encoder-distance scalar
on a majority of folds despite winning on the mean. The combined
classifier's own fold-to-fold F1 range (0.091-0.400) is far wider than
either individual arm's, and its worst fold (0.091) is worse than Arm
A's worst fold (0.121) — the concrete, pre-registered-as-a-risk signature
of an unstable, high-variance fit at this dimensionality, not a real
generalizable improvement that happens to look noisy. Per the
pre-registered gate: **Inconclusive**.

**Final synthesis and Stage D decision**: taken together with Rank 1
(clean Failure, recall) and Rank 2 (clean Failure, precision, marginal),
Rank 3's Inconclusive-with-instability result is judged, as a considered
whole, sufficient evidence that this project's representation-side,
no-new-data classifier-fitting approach has reached a genuine small-
sample ceiling — not a "look harder with one more feature" situation.
Three independent representations (encoder alone, acoustic alone, both
combined) have each been tried; the richest one became *less* stable, not
more powerful, exactly the signature of running out of real labeled
examples (62-245 depending on population) to fit an increasingly
high-dimensional decision boundary against. **Decision: Stage D,
re-scoped to L2 (stop deleting disfluency information during decoding,
per the Application-Objective Decision Analysis), is now judged the
evidence-justified next step.** Per the project owner's explicit
instruction, no fourth feature/variant is proposed in its place.

**Alternatives considered**
- Report the combined classifier's higher mean F1 (0.254) as a modest
  win, since it is numerically the best of the three arms. **Rejected**:
  the pre-registered stability check exists precisely to catch this —
  a mean improvement built from 2 wins and 3 losses, with a wider
  variance and a worse worst-case fold than even the weakest baseline,
  is not evidence of a real effect. Reporting the mean alone without the
  per-fold picture would have been technically true and substantively
  misleading.
- Treat Inconclusive as insufficient grounds for any decision and
  recommend a fourth experiment (e.g. a smaller, more regularized feature
  set, or a different combination). **Rejected**, directly per the
  project owner's explicit instruction not to invent another feature/
  variant if this one failed, and because the specific *shape* of the
  Inconclusive result (instability from added dimensionality, not a near-
  miss) is itself informative enough, combined with Rank 1/2's own clean
  Failures, to support a real decision rather than another cheap
  iteration.
- Treat this as insufficient to recommend Stage D, since Rank 3's own
  verdict was Inconclusive, not Failure, and defer the Stage D question
  again. **Rejected**: the project owner explicitly asked for the
  evidence-based Stage D decision now, given this exact outcome — and per
  standing rule 8, a considered decision over the *combined* weight of
  three real experiments (two clean Failures plus one Inconclusive-with-
  a-specific-diagnostic-shape) is exactly the kind of evidence-
  constrained call that rule authorizes, recorded with its reasoning, not
  a default.

**Why this choice**
This is the exact question the project owner asked to have answered at
the end of this bounded three-experiment sequence: given everything
tried, is Stage D now justified? The answer is reasoned from the full
arc, not from Rank 3 in isolation — each experiment's specific failure
shape (not just pass/fail) is what makes the aggregate conclusion
defensible: two clean floor-misses plus a combination that got *less*
stable when given more information, together, are the signature of a
data-quantity ceiling this project's existing representations and no-
new-data classifier-fitting approach cannot get past on their own.

**Measured result**
Numeric, reported in full above and in `ASR_RESEARCH_TRACK.md`'s "Rank 3
results" and "Final synthesis" sections; raw output saved to
`eval_results/20260807T214951_stage_j_combined_rich_classifier.json` and
the cached row-level dataset to `eval_results/_stage_j_rows_cache.npz`.
Verdict: **Inconclusive** (Rank 3 alone); **Stage D judged evidence-
justified** (final synthesis over all three experiments). `ROADMAP.md`
item 10 updated with a pointer and summary. New file: `profiling/
evaluation/stage_j_combined_rich_classifier.py`. No change to `main`.

## 2026-08-07 — End-of-day research consolidation: additive synthesis, no new experiment

**What was done**
Per the project owner's explicit, detailed instruction to close out the
day with an *additive* consolidation (not a rewrite) of the entire
`asr-research` branch's research record, reviewed all four project
documents (`ASR_RESEARCH_TRACK.md`, `ROADMAP.md`, `PAPER_DECISION_LOG.md`,
`CHANGELOG.md`) and confirmed every item the project owner explicitly
asked to be incorporated was already present in full detail somewhere in
`ASR_RESEARCH_TRACK.md` (direction (g), the RMS/ZCR result, the cross-
clip scoring bug, MFCC escalation, the coefficient-0 bug, the combined-
signal classifier, Rank 1/2/3, the literature falsification work covering
AS-70/StammerTalk/Kordt et al./StutteredSpeechASR/Whisper/HuBERT/WavLM,
the application-objective reorientation, the Stage D cost/infrastructure
analysis, and the L2-vs-L4 distinction) — verified by section-header
inventory before writing anything, not assumed. Added one new, clearly-
demarcated section, "Current Project State — 2026-08-07 EOD," at the very
end of `ASR_RESEARCH_TRACK.md`: a complete, properly cross-referenced
synthesis covering the primary objective (restated first, per the project
owner's explicit requirement that it be impossible to miss), the full
chronological experimental story with real numbers and pointers into the
detailed sections, a precise Experimentally-Demonstrated/Strongly-
Supported/Supported-by-Literature/Inferred/Unresolved/Not-Tested/Proposed
classification (deliberately avoiding both "ASR cannot solve this" and
"nobody has solved this" as the project owner explicitly named), a
literature position table reframed around "what can we use" rather than
novelty, a reassessed application requirement list, a retrieved-and-
updated (not reinvented) Stage D summary explicitly noting the L2 rescope
and tonight's "not yet justified" -> "evidence-justified" evolution, an
explicit current-stopping-point statement, a 10-step next-session plan
that supersedes (via an in-place marker, not a deletion) the now-stale
"Decision for Next Session" written earlier the same day, and guidance on
which file to share with collaborators. Added matching, non-duplicative
pointer entries to `ROADMAP.md` item 10 and this same-day `CHANGELOG.md`
entry.

**Alternatives considered**
- Rewrite or compress any existing section for stylistic consistency now
  that the full day's story is known. **Rejected**, directly per the
  project owner's explicit, repeated instruction: "Do not shorten
  existing literature reviews, experiment descriptions, citations,
  methodology, or reasoning just because you can summarize them... Do not
  replace detailed sections with summaries." Every section that existed
  before tonight remains completely unedited.
- Silently correct the now-outdated "Decision for Next Session" section
  (written before Rank 1-3 existed) to reflect tonight's actual plan.
  **Rejected**: marked superseded in place, with a short dated note
  explaining exactly what changed and pointing to the new authoritative
  section, per the project owner's own explicit rule 4 ("If an earlier
  conclusion is superseded, DO NOT erase its history. Mark it as
  superseded/revised and explain what new evidence changed it").
- Duplicate the full Stage D design/requirements detail into the new EOD
  section, in case a reader doesn't follow the pointer. **Rejected**:
  the project owner explicitly asked to "retrieve and update the
  previous Stage D analysis rather than inventing a new one" — the EOD
  section summarizes and cross-references the existing "Stage D planning"
  and "Stage D Requirements / Re-entry Gate" sections rather than
  restating them at full length, avoiding exactly the kind of
  unnecessary duplication the project owner also asked to avoid ("Do not
  duplicate thousands of lines unnecessarily across all four files").

**Why this choice**
This is a documentation-and-synthesis pass, not a research pass — its
value is entirely in making the existing, already-correct research record
navigable and complete for a reader (human or a future Claude session)
who was not present for today's work, without risking the loss or
distortion of any of the historical reasoning, citations, or measured
results that make this track's conclusions trustworthy in the first
place.

**Measured result**
Not a numeric result — a documentation consolidation. `ASR_RESEARCH_
TRACK.md` gains one new section ("Current Project State — 2026-08-07
EOD") and one superseded-marker addition to an existing section; no
existing section's own text was edited, shortened, or removed.
`ROADMAP.md` item 10 and `CHANGELOG.md` gain matching pointer entries. No
code, experiment, or dataset touched. No change to `main`.

---

## 2026-08-08 — External review reconciliation: Stage D re-classified premature, not rejected

**What was done**
Received `EXTERNAL_REVIEW_2026-08-07.md`, an independent adversarial
review of the `asr-research` track arguing its 2026-08-07 "Stage D is
the evidence-justified next step" decision rested on an incomplete
literature map (centrally, a missed paper/system, Dysfluent-WFST,
arXiv:2505.16351). Per the project owner's explicit instruction, neither
document was taken at face value. Ran four independent primary-source
verification passes (as parallel research agents, no code written):
(1) fetched the Dysfluent-WFST paper and GitHub repo directly; (2)
verified every disputed dataset claim (Switchboard/MsState, SEP-28k-SW,
Boli, Buckeye, FluencyBank Timestamped, LibriStutter's construction,
AS-70, Libri-Dys/VCTK-Stutter) against primary sources; (3) verified
every disputed statistical/citation claim (Zayats et al.'s human
transcription-error rates, SEP-28k inter-annotator kappa by type, the
StutterFuse characterization, Gulzar et al.'s LibriStutter non-verbatim
finding, Shih et al.'s F1 baseline, the alignment-gap classifier's
coverage figure); (4) verified current cloud-GPU pricing/free-tier
availability against the review's cost claims. Findings recorded as a
new, dated, additive section in `ASR_RESEARCH_TRACK.md` ("External
Review Reconciliation — 2026-08-08"), a correction annotation on the
StutterFuse citation (previously mischaracterized as text/acoustic
fusion; it is a retrieval-augmented classifier), superseded-in-place
notices on the prior Stage D decision and next-session plan, and a
matching sequencing update to `ROADMAP.md` item 20.

**Alternatives considered**
- Accept the external review's conclusions directly and start
  implementing Dysfluent-WFST. **Rejected** — the project owner
  explicitly required independent verification before acting on either
  document; several of the review's claims (Dysfluent-WFST's output
  including timestamps; SEP-28k-SW adding fragment annotation; "the
  English data gap is not real") did not survive verification intact.
- Dismiss the external review because the internal track's process was
  already rigorous. **Rejected** — rigor in execution does not imply
  completeness of the literature map searched; the review's central
  factual claim (a relevant, on-target paper/system existed and was
  missed) verified true.
- Silently revise the 2026-08-07 Stage D decision to "premature" without
  a dated record of why. **Rejected**, same append-and-supersede
  discipline this track has used throughout: the original decision
  remains fully intact and readable, with a pointer to what changed it.
- Re-run Ranks 1-3's zero-compute reanalysis (PR/AUPRC + bootstrap CIs;
  end-to-end Track B effect of Rank 1) or clone/run Dysfluent-WFST as
  part of this same pass. **Rejected for this pass, explicitly deferred**
  — the project owner's instructions were analysis/decision-making only
  ("do not implement anything until that answer is established"); both
  are now the top two items in the revised decision tree.

**Why this choice**
The external review's own most consequential claim — that a cheaper,
better-matched, previously-untried mechanism (reference-text-conditioned
phonetic realignment against a WFST/CTC emission lattice) exists and
targets this track's exact `sound_repetition` gap — held up under
independent verification of the primary source, and this track's own
document already records (RQ-E, named and never run) that its own
"every cheap lever tried" language was not literally true. Per standing
rule 8 (architectural decisions are evidence-constrained, not
preservation-constrained), a decision reached under a since-falsified
premise does not get protected just because it was already recorded —
it gets re-opened, with the reasoning for both the original call and the
revision kept visible. At the same time, several of the review's own
claims did not survive independent verification (no timestamps from the
WFST decoder as released; SEP-28k-SW does not add fragment annotation;
the English-data-gap correction is real but narrower than claimed — see
the reconciliation section for the full list) — accepting the review
uncritically would have repeated the same error in the opposite
direction.

**Measured result**
Not a numeric result — a documentation reconciliation following a
completed, four-pass, primary-source verification exercise. Stage D
moves from "evidence-justified next step" to "premature, not rejected."
Two zero-compute reanalyses and one cheap, never-tried experiment
(Dysfluent-WFST against the existing 120-clip Track B sample) are now
sequenced ahead of it. No code, experiment, or dataset touched this
session. No change to `main`.

---

## 2026-08-08 (same day, round 2) — Correcting round 1's own SEP-28k-SW verdict; primary-source read of Kordt et al.

**What was done**
A second AI reviewer, external to this repo, independently re-checked
the round-1 reconciliation above: conceded most of it without further
verification needed, disputed one specific factual claim (whether
Sep-28k-SW adds sub-word annotation), made one unverified technical claim
(Dysfluent-WFST timestamps being "an afternoon of work" to recover), and
proposed three design refinements (move real-audio acquisition earlier;
promote Buckeye to a training-corpus role under an explicit train/eval
split; use Libri-Dys as a scale diagnostic for Rank 1's open question).
Separately, the project owner supplied the full text of "Kordt et al."
(arXiv:2606.14391, "Learning to Hear Hesitation"), previously cited in
this track only secondhand as `[B11]`. Ran two more independent
primary-source verification passes (the SEP-28k-SW table-vs-prose
discrepancy, traced to Sridhar & Wu's own Interspeech 2025 paper; the
Dysfluent-WFST timestamp claim, checked against the actual repo code and
k2/icefall documentation) and read the new Kordt et al. paper directly.
Recorded all of this as a new, dated, additive "Second-pass reconciliation
— 2026-08-08 (round 2)" section in `ASR_RESEARCH_TRACK.md`, with a
superseded-in-place notice on round 1's decision tree, plus matching
pointers in `ROADMAP.md` item 20 and `CHANGELOG.md`.

**Alternatives considered**
- Accept the second reviewer's SEP-28k-SW and timestamp claims without
  independent re-verification, since round 1 had already done one
  verification pass. **Rejected** — the whole point of this reconciliation
  exercise is not accepting either side's claims at face value; a claim
  surviving one AI's scrutiny is not the same as being independently
  re-verified against the primary source, and in fact round 1's own
  SEP-28k-SW verdict turned out to be wrong.
- Quietly fix round 1's SEP-28k-SW verdict in place without flagging that
  it was wrong. **Rejected**, same append-and-supersede discipline as
  every other correction in this track — the error is recorded, with why
  it happened (a verification pass that read a paper's prose but missed
  its own table, and stopped at a citation instead of fetching the
  underlying paper), because that failure mode is exactly what this
  reconciliation process exists to guard against, in either direction.
- Accept "an afternoon of work" for WFST timestamp recovery since the
  underlying structural claim (frame-indexed emission, k2 preserves
  frame correspondence) checked out. **Rejected** — checked the actual
  shipped repo code, not just the paper, and found the decoder discards
  the one tensor axis that would carry timing and composes a non-trivial
  graph (CTC topology + annotation arcs) that no existing documented
  recipe (icefall's own forced-alignment doc page is an empty stub)
  covers. The structural premise is correct; the effort estimate is not
  verified.
- Treat Kordt et al.'s CL-methods-on-full-fine-tune recipe and
  StammerTalk's LoRA recipe as the same, already-combined approach, since
  Stage D's design section already cited both together. **Rejected** —
  reading Kordt et al. directly shows no LoRA anywhere in it; the two
  recipes have never been demonstrated combined by anyone, and Stage D's
  design text is corrected to say so explicitly rather than implying an
  already-proven combination.

**Why this choice**
Per standing rule 8, a decision (or a verification finding) does not get
protected from revision just because it was already recorded earlier the
same day — it gets re-opened the moment better evidence exists, with both
the original and the correction kept visible. The SEP-28k-SW correction
in particular is a useful, humbling data point about this reconciliation
process's own failure mode (secondhand citations and prose-only reads
miss things a table or a primary source would have caught) and is worth
keeping visible for exactly that reason, not just for its factual content.

**Measured result**
Not a numeric result. One dataset verdict corrected (Sep-28k-SW: real,
sub-word-annotated stuttering data likely exists, pending a public-
availability check); one technical estimate downgraded from "trivial" to
"plausible but unverified, budget as its own sub-task"; one citation
corrected (Stage D's LoRA+CL framing is an untested combination, not a
proven recipe); one new zero-cost diagnostic identified (Rank 1 re-run at
Libri-Dys's ~10x scale); one new limitation recorded (LibriStutter's
zero-annotation-noise ground truth makes every precision figure this
track has produced optimistic relative to real deployment); revised data
strategy (train on general-disfluency corpora, evaluate on the small real
stuttering sets) and decision tree recorded. No code, experiment, or
dataset touched this session. No change to `main`.

---

## 2026-08-08 (same day, round 3) — Sep-28k-SW licence, WFST timing left genuinely unresolved, marker-F1 reconciled, two new queued items

**What was done**
The second reviewer's own follow-up accepted round 2 in full (including
the correction made against its own earlier claim) and raised five
further points. Checked each independently rather than accepting any of
them, including the one round 2 had already flagged as unresolved
(Sep-28k's licence): fetched `apple/ml-stuttering-events-dataset`'s own
README directly and confirmed CC BY-NC 4.0 (non-commercial) — a real
constraint given `audio-mod`'s place in a two-repo commercial product.
Re-fetched the Dysfluent-WFST decoder code specifically to test the
reviewer's counter-argument that the special transition arcs might use
the CTC blank symbol (frame-consuming) rather than a true epsilon
(non-frame-consuming); found this round's own reading disagreed with
round 2's reading about what the arcs' label actually means — concluded
this is not resolvable by further static reading and requires the
concrete runtime check the reviewer proposed. Resolved an apparent
contradiction between two Kordt et al. marker-F1 figures cited in round 2
directly from the paper's own text already in hand (different tables,
different training stages — not a contradiction). Accepted two new
queued items: a CrisperWhisper cross-attention probe (reusing Kordt et
al.'s head-attribution method) and a required power analysis on the
assembled real evaluation data before any Stage D pre-registration.
Recorded all of this as "Third-pass reconciliation — 2026-08-08
(round 3)" in `ASR_RESEARCH_TRACK.md`, with matching pointers in
`ROADMAP.md` item 20 and `CHANGELOG.md`. Also added a new standing rule
to `CLAUDE.md` (rule 9): read primary sources directly for any claim
that gates a decision — prompted by round 2 catching this project's own
wrong verdict on Sep-28k-SW, which had been reached from a paper's prose
summary of another paper rather than that paper itself.

**Alternatives considered**
- Accept the Sep-28k-NC license claim from the reviewer's assertion
  without an independent fetch, since round 1/2 had already verified
  several other claims from this same reviewer accurately. **Rejected**
  — a claim's source having been reliable before is not the same as this
  specific claim being verified, and this one gates a real legal/product
  constraint, not just a research interpretation.
- Try to resolve the WFST blank-vs-epsilon question with one more,
  more careful code-reading pass. **Rejected** — attempted exactly this,
  and it produced a reading that disagreed with the previous round's own
  reading of the same code. Two AI-generated characterizations of the
  same source disagreeing with each other is itself sufficient evidence
  that more reading will not settle this; recorded as genuinely
  unresolved and deferred to the concrete runtime check instead, per the
  new standing rule (rule 9) this round also added.
- Silently pick one of the two marker-F1 figures as authoritative rather
  than reconciling both. **Rejected** — both are real numbers from the
  same real paper; the resolution (different tables, different stages)
  is more informative than picking one, and is itself a decision-relevant
  finding about marker retention degrading under continued adaptation.
- Add the CrisperWhisper cross-attention probe or run the power analysis
  in this same session. **Rejected, deferred** — this remains a
  documentation/decision-analysis session; both are now queued, correctly
  sequenced, in the revised decision tree.

**Why this choice**
The pattern across all three rounds of this reconciliation is the same:
every claim that gates a real decision — whether it originated inside
this project or from an external reviewer — gets checked against its
actual primary source before being acted on, and a claim's earlier
reliability does not exempt a new claim from the same check. Round 2's
own Sep-28k-SW mistake is the clearest demonstration available that this
discipline has to be procedural, not judgment-based, which is why it is
now a numbered standing rule rather than an implicit practice.

**Measured result**
Not a numeric result. One licensing constraint confirmed and applied
(Sep-28k-SW: evaluation-only); one technical question correctly
downgraded from "plausible engineering guess" to "genuinely unresolved,
needs a runtime check" after this round's own attempt to resolve it by
reading failed to agree with the prior round's reading; one apparent
data contradiction resolved directly from primary text (not a
contradiction — different tables/stages); two new items queued
(CrisperWhisper cross-attention probe; pre-Stage-D power analysis); one
new standing rule added to `CLAUDE.md`. No code, experiment, or dataset
touched this session. No change to `main`.

---

## 2026-08-08 (same day) — Final decision-oriented reconciliation: alignment-gap test for `word_repetition` ranked ahead of Dysfluent-WFST

**What was done**
Per the project owner's explicit instruction, treated rounds 1-3's own
decision tree as an input to re-examine, not a conclusion to inherit —
re-anchored the entire reconciliation to `CLAUDE.md`'s stated application
objective and built an explicit architecture comparison (current
pipeline; CrisperWhisper+Dysfluent-WFST; an existing stutter-aware ASR,
found not to exist for English; acoustic/audio-native detector fusion;
Stage D; and other mechanisms found this session) scored strictly
against information gained/lost, taxonomy coverage, timing, data,
compute, testability, and biggest uncertainty — novelty deliberately
excluded as a ranking factor. Reconciled Ranks 1-3 explicitly: what they
established (frozen-representation classification insufficient, real
result), what they did not (that no mechanism can work, or that a
different mechanism — WFST or alignment-gap — would fail too), and what
must be withdrawn (the exhaustion claim) vs. narrowed (the
quantity-of-labels bottleneck, now quantity-and-reliability, scoped to
one mechanism class only). Concluded the single highest-value next step
is an **alignment-gap/duration-residual candidate generator for
`word_repetition`** — reusing CrisperWhisper's own already-computed word
timestamps, no new data or dependencies — placed ahead of Dysfluent-WFST
in priority, which becomes the immediate next step after it (for
`sound_repetition` specifically). Recorded a precise Stage D
justification condition (both tests fail *and* a power analysis confirms
the available real evaluation data can resolve Stage D's own gate).
Wrote this as a new, dated, top-level "Final Decision-Oriented
Reconciliation" section in `ASR_RESEARCH_TRACK.md`, with a claim
classification (experimentally demonstrated / primary-source supported /
inferred / unresolved / proposed) closing it out.

**Alternatives considered**
- Accept rounds 1-3's own sequencing (Dysfluent-WFST as the immediate
  next step) since it had already been through three rounds of
  verification. **Rejected**, per explicit instruction — verification of
  individual *claims* is not the same as verification of the
  *sequencing decision* built from them; the sequencing itself was never
  re-examined against the application objective until this pass, and
  doing so surfaced a cheaper, lower-risk, equally-real alternative
  (the alignment-gap test) that had been present in the evidence the
  whole time but ranked lower by the external review's own novelty-
  weighted ordering.
- Treat the external review's "Dysfluent-WFST is the experiment" framing
  as authoritative since it was the review's own top-ranked item.
  **Rejected** — re-scored against this project's own stated priority
  order (application fit, reliability, implementability, evidence,
  novelty last, in that order) rather than the review's implicit
  novelty-first ordering.
- Propose parallel branches for both `word_repetition` and
  `sound_repetition` simultaneously. **Rejected**, per explicit
  instruction to avoid a speculative multi-branch tree — sequenced
  instead, with each branch's own hard stop condition.
- Run the alignment-gap test, or check YOLO-Stutter/SSDM checkpoint
  availability, in this same session. **Rejected, deferred** — this
  remains a decision-only session; nothing was implemented or executed.

**Why this choice**
The project's own standing rule 8 (evidence-constrained, not
preservation-constrained architecture decisions) applies as much to this
track's own sequencing choices as to its architecture choices — a
decision tree already written down and already verified in its parts
does not thereby become correct in its ordering. Scoring honestly against
application fit, reliability, and implementability before novelty is the
only way to avoid quietly re-adopting "prove something interesting" as
the real objective, which is precisely the failure mode `ASR_RESEARCH_
TRACK.md`'s own "PROJECT OBJECTIVE" section and this reconciliation's
opening section exist to prevent.

**Measured result**
Not a numeric result. A finite, two-step, hard-stop-conditioned decision
tree replaces the three-round reconciliation's own more open-ended
sequencing. `word_repetition`'s alignment-gap test is now the single
next action item; Dysfluent-WFST remains next after it for
`sound_repetition`; Stage D's justification condition is now precise
and falsifiable rather than "premature, not rejected" without a stated
resolution path. No code, experiment, or dataset touched this session.
No change to `main`.

---

## 2026-08-08 (same day) — Step 1 executed: alignment-gap test for `word_repetition` — FAILURE

**What was done**
Per the project owner's explicit go-ahead, pushed the prior reconciliation
work to `origin/asr-research`, then executed Step 1 of the finalized
decision tree. Pre-registered the exact methodology in `VALIDATION.md`
§15 (hypothesis, data source, signal definition, metrics, success/failure
criteria) *before* writing any code, per standing rule 1. Implemented
`profiling/evaluation/stage_k_alignment_gap_word_repetition.py`, reusing
existing infrastructure unmodified (`stage_b_representation_probe
._identify_positions`/`_cohens_d`, `compare_corroboration_mechanisms
._clip_folds`/`_cv_threshold`/`_summarize`, `stage_c_duration_baseline
._auc`, and the existing Track B ASR cache — no new inference, no new
dependency). Added a clip-level bootstrap 95% CI on the AUC, a
statistical practice this track had never used before despite the
external review naming its absence as a real gap. Self-tested (9/9
checks) before running for real. Ran against the existing 189-clip Track
B cache.

**Result: FAILURE.** n=29 `word_repetition` category-1 targets vs. 941
controls. Cohen's d=-0.209 (wrong direction), AUC=0.485 (95% CI
[0.396, 0.566], comfortably includes chance). Full detail:
`VALIDATION.md` §15.8; decision-tree implications: `ASR_RESEARCH_TRACK.md`
"Step 1 executed — 2026-08-08."

**Alternatives considered**
- Try a different residual definition (gap-after, multi-token window)
  immediately after seeing this result, since the AUC came back so
  close to chance. **Rejected**, directly per standing rule 4 (never
  retune/re-engineer in response to an evaluation result without
  explicit go-ahead) and this track's own repeated "name, don't chase"
  discipline (e.g. RQ-E, the dropped variants named throughout this
  track) — the untested variant is named in `VALIDATION.md` §15.8 as a
  possible future direction, not attempted this pass.
- Treat P=0.031/R=0.567 as a partial success since recall alone cleared
  its floor. **Rejected** — the pre-registered gate (§15.6) is
  conjunctive (AUC CI excludes chance AND clears both precision and
  recall floors), decided before running, and a near-chance AUC with
  inflated recall from a forced threshold is exactly the failure
  pattern the gate exists to catch, not a partial win to round up.
- Proceed directly into Dysfluent-WFST implementation without reporting
  this result first, since the project owner said to "proceed with the
  next steps." **Rejected, deferred** — Step 2 is a materially heavier
  commitment (new dependencies: `k2`, a phone-posterior model, a
  grapheme-to-phoneme step) than Step 1's zero-new-dependency scope;
  reporting this checkpoint first matches the project's own
  "measure twice, cut once" discipline for larger commitments without
  changing the decision tree itself, which already names Step 2 as the
  correct next action regardless of Step 1's outcome.

**Why this choice**
Standing rule 3 (audit surprising results before trusting them) cuts
both ways — a result this close to chance, with a wrong-direction point
estimate, is exactly as worth double-checking as a suspiciously good
one. The self-test suite (including hand-constructed decision-gate
cases) was verified passing before this result was trusted, and the
bootstrap CI (rather than a bare point estimate) is what makes "FAILURE"
a confident call here rather than a guess at a borderline number.

**Measured result**
A real, pre-registered, self-tested, clip-level-bootstrapped negative
result for one specific `word_repetition` candidate-generation mechanism.
`word_repetition`'s alignment-gap approach is now closed. Per the
decision tree, Step 2 (Dysfluent-WFST, for `sound_repetition`) is next,
reported here as a checkpoint before that larger implementation begins.
One new file created (`profiling/evaluation/stage_k_alignment_gap_
word_repetition.py`); no other production code changed. No change to
`main` yet (uncommitted at time of writing).

---

## 2026-08-08 (same day) — Step 0 executed and reviewed; Rank 1's original verdict substantially understated; Step 2 proposed, not started

**What was done**
Per the project owner's explicit instruction, ran Step 0 (zero-compute
reanalysis of Ranks 1-3, per the decision tree in `ASR_RESEARCH_TRACK.md`),
then performed a thorough review pass for mistakes before documenting,
then prepared (without starting) a detailed Step 2 proposal.

Implemented `profiling/evaluation/stage_l_zero_compute_reanalysis.py`:
Part A recomputes AUPRC + clip-level bootstrap 95% CIs for Ranks 1/2/3
from already-cached row-level predictions (no new inference); Part B
measures the end-to-end Track B effect of injecting Rank 1's classifier
as a new candidate-generation trigger. During review, found and fixed two
real mistakes and one process deviation (detailed in `VALIDATION.md`
§16.1): (1) Part B's first version omitted the gate-off config, silently
triggering item 17's shipped encoder-based classifier gate per clip — not
just a ~30-90s/clip performance cost but a methodological inconsistency,
since Stage A's "category 1" population is defined under gate-off
conditions; caught when the project owner asked how much longer the run
would take, prompting a check of actual CPU time (47+ min CPU across
31+ min wall-clock) rather than continuing to wait on an unverified
assumption; (2) a fold-id recomputation relied on an assumption that
happened to hold but wasn't guaranteed, hardened to remove the
assumption entirely; (3) `stage_l` was run for real before it had a
self-test, unlike every other `stage_*` script in this codebase — added
and verified retroactively (10/10 pass), confirmed not to change any
already-reported number.

Real results (full detail: `VALIDATION.md` §16.2-16.3): Rank 1's original
"FAILURE (recall floor)" verdict was substantially understated — AUPRC=
0.556 (~15x its own chance rate), and an honest per-fold recall-targeted
reanalysis reaches mean P=0.783 at mean R=0.289 (vs. the original
P=0.580/R=0.147). Rank 2 remains a real Failure; Rank 3's precision
improves but its fold-instability concern is unresolved. The end-to-end
Track B effect (previously never measured) is real but modest: overall
recall 0.000→0.052. A concrete, zero-new-cost action item is identified
(re-threshold the shipped Rank 1 classifier) but not adopted here.

Prepared a detailed Step 2 (Dysfluent-WFST) proposal in `ASR_RESEARCH_
TRACK.md` — concrete steps, named risks, success/failure criteria — with
nothing executed, per explicit instruction.

**Alternatives considered**
- Keep waiting on Part B's first (buggy) run rather than investigating
  why it was taking so long. **Rejected** — 31+ minutes against an
  expected few minutes (matching Part A's own timing for a comparable
  fit) was a real anomaly, not just slower-than-expected; checking actual
  CPU time confirmed genuine heavy computation rather than a hang, which
  by itself was enough to conclude something was wrong (over-invoking
  work that shouldn't have been happening), not merely slow.
- Silently fix the gate-off bug and re-report only the corrected numbers,
  without recording that the first version was wrong. **Rejected**, same
  append-and-supersede discipline as every other correction in this
  track — the mistake and why it happened are recorded in `VALIDATION.md`
  §16.1 precisely because the project owner asked for thorough review and
  debugging, not just a clean final number.
- Treat Rank 1's improved reanalysis (mean R=0.289, just short of the 0.3
  floor) as a clean SUCCESS given how close it is. **Rejected** — the
  pre-registered floor is what it is; the honest characterization
  ("substantially understated, closer to Success than Failure, not a
  clean strict pass") is reported instead of rounding a near-miss up.
- Immediately act on the re-thresholding finding by changing `detect.py`'s
  shipped configuration. **Rejected, deferred** — per standing rule 4,
  this is a finding to be acted on only with explicit go-ahead, recorded
  as a candidate for `ROADMAP.md`, not a silent config change.
- Start Step 2 (Dysfluent-WFST) implementation in this same session.
  **Rejected, deferred** — explicit instruction was to prepare and
  propose Step 2, not begin it; the proposal names concrete steps, new
  dependencies, and risks for review first.

**Why this choice**
The project owner's request was explicitly for thorough review and
honest debugging, not just a result — per standing rule 3 (audit
surprising results before trusting them), a run taking 8x longer than
its own comparable sibling computation is exactly the kind of anomaly
that needed checking before its output could be trusted, and the
resulting fix changed a real methodological inconsistency (gate state),
not just performance. Recording the mistakes found, and confirming they
don't invalidate the numbers ultimately reported, is what makes those
numbers trustworthy rather than merely fast to produce.

**Measured result**
Real, reviewed, self-tested numbers for Step 0 (detailed above and in
`VALIDATION.md` §16); a concrete but not-yet-adopted re-thresholding
candidate for `ROADMAP.md`; a detailed, unexecuted Step 2 proposal in
`ASR_RESEARCH_TRACK.md`. One new file
(`profiling/evaluation/stage_l_zero_compute_reanalysis.py`); no other
production code changed. Nothing beyond Step 0 was executed. No change
to `main` yet (uncommitted at time of writing).
