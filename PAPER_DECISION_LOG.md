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
