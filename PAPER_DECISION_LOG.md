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
