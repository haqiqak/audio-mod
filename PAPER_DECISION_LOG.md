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
