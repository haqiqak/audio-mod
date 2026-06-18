# Stammr Profiler — Future Roadmap (v2)

This supersedes the latency section of `improve.md`. The model-caching fix
applied in the previous round (`st.cache_resource` around the ASR loader)
did not produce a measurable improvement — 1400 seconds for a 2-second clip
was reported after that fix was already in place. That result is itself
important diagnostic information, and this document treats it as such
instead of guessing again. The instruction going in this time is: **measure
before fixing.** No more "this should be the bottleneck" without a number
attached.

---

## What the failed fix tells us

Two honest possibilities, not mutually exclusive:

**A. The fix never actually got exercised.** `st.cache_resource` only
caches within one running Streamlit process — if each test was done by
restarting `streamlit run` (a fresh process each time), the cache provides
zero benefit, because there is no "second call" within the same process to
benefit from it. This is the most likely explanation if testing looked like
"run app, test, stop app, change something, run app again" rather than
"run app once, click Analyse multiple times in the same browser session."

**B. Model loading was never the dominant cost — inference itself is.**
If (A) is ruled out (i.e. the same running process was used for repeat
clicks and it was still ~1400s every time), then the bottleneck is
somewhere in the actual `pipe(audio)` call, not in constructing the
pipeline object. That would mean the real-time-factor itself is ~700x,
which is wildly outside normal CPU Whisper behavior (published benchmarks
for stock `transformers` Whisper on CPU sit in the 1x-8x RTF range; even
heavily under-optimized setups don't typically reach 700x without something
structurally wrong — e.g. an accidental infinite/near-infinite generation
loop, a CPU thread-count misconfiguration causing massive contention, or
the process accidentally running on a throttled/virtualized CPU with a
fraction of expected performance).

**This round starts by determining which of A or B is actually true**,
because the fix is completely different depending on the answer, and
guessing wrong burns another full test cycle (each of which costs ~25
minutes at current speed).

---

## Step 0 — Instrumentation (do this before any further code changes)

Add timing checkpoints around each phase of a single transcription call,
logged with `time.perf_counter()`, and surfaced directly in the app's log
box (which already exists from the previous round) so the next test run
produces an actual breakdown instead of one opaque "1400s total" number:

```
t0 = perf_counter()
asr = get_asr(device)              # should be near-instant on 2nd+ call
t1 = perf_counter()
pipe = asr._load_pipeline()        # should be near-instant if cached
t2 = perf_counter()
result = pipe(audio_path)          # the actual generate() call
t3 = perf_counter()
```

Log all three deltas (`t1-t0`, `t2-t1`, `t3-t2`) to the UI. This single
change turns every future test into a real diagnosis instead of a guess,
and costs nothing in terms of risk. **This should be the very first change
made, before touching anything else**, specifically so the next test run
tells us definitively whether we're in scenario A or B above.

Also log: which device string was actually used, output of
`torch.get_num_threads()`, and whether `torch.backends.mkldnn.is_available()`
returns true — all three are one-line checks and rule out or confirm
several hypotheses at once for free.

---

## Step 1 — If scenario A (cache wasn't actually exercised)

Confirm by testing correctly: run `streamlit run app.py` **once**, then in
the same browser tab/session click "Analyse" on the same short clip twice
in a row without restarting the server. Compare the two timings from the
Step 0 instrumentation.

- If the second click is dramatically faster than the first → the caching
  fix works exactly as designed, and the only remaining problem is that
  the *first* load is unavoidably slow (reading 3.2GB off disk + Windows
  filesystem overhead). This is a one-time cost, not a per-click cost, and
  the actual UX problem becomes "warm up the model once when the app
  starts, before the user's first click" — solvable by calling
  `get_asr(device)` once at app startup/import time rather than waiting
  for the first analysis click, so the unavoidable load time is paid
  during page load, not during the user's first interaction.
- If the second click is just as slow as the first → scenario A is ruled
  out, move to scenario B.

---

## Step 2 — If scenario B (inference itself is the bottleneck)

This is the more concerning case and needs its own breakdown. Likely
culprits, in order of how cheaply they can be tested:

**2a. Thread configuration.** PyTorch's default CPU thread count can
sometimes cause severe contention rather than speedup, especially on
machines with many logical cores combined with hyperthreading, or when
running inside any kind of virtualization/sandboxing. Test explicitly
setting `torch.set_num_threads(4)` (or similar, well below the logical
core count) at app startup and re-measure. This single line has, in
documented cases elsewhere, changed inference time by multiples — cheap
to test, easy to revert.

**2b. Verify no infinite/runaway generation loop is occurring.** The
`max_new_tokens=256` cap added last round should make this impossible,
but if the actual `t3-t2` delta from Step 0 shows generation running for
the full ~1400s, that means the model is generating the maximum allowed
tokens and still taking that long per token — i.e., per-token latency
itself is catastrophic, not the token count. That would point at:

**2c. The transformers/torch install itself.** A torch wheel without
proper CPU vectorization (no AVX2/AVX512, no MKL-DNN) can be an order of
magnitude slower than a properly-built one, especially combined with an
old or mismatched Python/torch/transformers version trio. Worth running,
as a completely separate sanity check outside the app:
```python
import torch, time
torch.set_num_threads(4)
x = torch.randn(2000, 2000)
t0 = time.perf_counter()
for _ in range(20):
    y = x @ x
print(time.perf_counter() - t0)
```
A basic 2000x2000 matmul benchmark like this taking more than ~1-2 seconds
for 20 iterations on any reasonably modern CPU is a strong signal that the
torch install itself is the problem, independent of anything Whisper- or
transformers-specific — at which point reinstalling torch with the correct
wheel for the actual CPU architecture becomes the highest-priority fix,
above any model-level change.

**2d. Background processes / antivirus / Windows Defender real-time
scanning.** Worth ruling out only after 2a-2c are checked, since it's
outside the code entirely, but on Windows specifically, real-time
antivirus scanning of a 3.2GB model file being repeatedly read, or of the
Python process itself, has been documented elsewhere to add severe
overhead to ML workloads. A quick test: temporarily add an exclusion for
the venv/model-cache folder and re-measure.

---

## Step 3 — Architectural options for a genuinely fast final version

Assuming Steps 0-2 identify and fix whatever pathological slowdown is
occurring, here are the real architectural levers available to reach
"click and results in a couple seconds," ranked by expected impact:

**3a. faster-whisper / CTranslate2 backend.** Already flagged in the prior
roadmap, still the single highest-leverage change not yet attempted.
CTranslate2 uses quantized int8 execution and hand-optimized C++ kernels
rather than PyTorch eager-mode Python, and is documented at 4-8x CPU
speedup over stock `transformers` for the same Whisper weights. This is
likely necessary regardless of what Steps 0-2 find, to get from
"acceptable" to "couple of seconds."

**3b. Persistent server process instead of Streamlit's per-script-rerun
model.** Streamlit reruns the entire script top-to-bottom on every
interaction. Even with `st.cache_resource` correctly preventing model
*reloading*, every rerun still re-executes all the Python above the cached
call, and Streamlit's own overhead per rerun (widget re-registration,
session-state diffing) is non-trivial but should be milliseconds, not
seconds — worth measuring with Step 0's instrumentation rather than
assuming. If this does turn out to be a meaningful chunk of time, the
architectural fix is to move the actual ASR call into a background worker
process (e.g. a small FastAPI/Flask service running alongside Streamlit,
called via HTTP) so the heavy lifting happens outside Streamlit's
rerun cycle entirely.

**3c. Smaller model variant for the speed-critical path.** CrisperWhisper
is built on a Whisper-large-class architecture specifically to preserve
disfluency-relevant acoustic detail — but it may be worth testing whether
a distilled/smaller variant (if one exists, or a more aggressively
quantized int8/int4 build) hits an acceptable accuracy/speed tradeoff for
the profiling use case specifically (where exact word-for-word
transcription matters less than reliable disfluency-pattern detection —
see Track 2 below, which argues this engine's job may shift away from
being "the transcript" toward being "one signal among several").

**3d. Streaming/incremental processing.** For the final product (not this
testing phase), processing audio in small streamed chunks as it's
recorded, rather than waiting for the full clip and then running one
big batch call, can make the *perceived* latency much lower even if total
compute is similar — results start appearing while the user is still
speaking. This is a bigger architectural change appropriate for a later
phase, not the current testing-phase priority.

### What NOT to do (carried over, still true)

- Do not reach for `chunk_length_s` — proven wrong twice now.
- Do not drop `return_timestamps="word"` as a speed hack — breaks the
  entire detection pipeline's ability to locate disfluencies in time.
- Do not add more "plausible-sounding" kwargs fixes without measuring
  first. This entire document exists because that approach already failed
  once this round.

---

## Track 2 — Expanded pattern detection (beyond the existing four types)

The existing detector catches: filler words, ASR-marked stutter tokens,
word-repetition, silent blocks, and prolongation-by-duration. The current
known gaps and additions to investigate, roughly ordered by how distinct a
speech-language-pathology concept they represent (not by implementation
difficulty, which varies independently):

**Sub-word / phoneme-level repetition** ("f-f-f-fine", "b-b-buy") — the
specific gap exposed in the last test. Requires either (a) recovering
disfluency signal already present in CrisperWhisper's raw decode before
our post-processing discards it, or (b) an independent acoustic detector
operating on the waveform directly. Already scoped in `improve.md` Track 2,
carried forward here as still the top priority gap.

**Audible silent blocks vs. true silence** — the current "block" detector
treats any gap over a threshold as a block, but clinically, a "block" in
stuttering specifically refers to an inability to *initiate* phonation
with visible/audible effort (sometimes with audible tension, irregular
breathing, or a brief voiced "uh" right at release) — distinct from
ordinary pausing for normal speech planning. Detecting this distinction
requires looking at what happens at the boundaries of the gap (the audio
immediately before and after), not just the gap's duration.

**Interjected revisions / false starts** ("I want to— I need to go") —
currently only caught if the ASR happens to mark a trailing fragment with
a hyphen. A more robust approach: detect when a word or short phrase is
followed by a different word/phrase with high semantic or phonetic overlap
in a short time window, suggesting a self-correction rather than the
fragment-marker heuristic currently used.

**Prolonged sounds mid-word** (not just whole-word duration outliers, but
elongation of a specific phoneme within an otherwise normal-length word,
e.g. "fffffine" stretched at the front but the whole word still resolving
in normal time) — distinct from the current prolongation rule, which only
looks at total word duration. Needs phoneme-level timing, which likely
requires forced alignment (e.g. via a tool like Montreal Forced Aligner,
or wav2vec2-based phoneme alignment) rather than anything available from
word-level Whisper timestamps.

**Secondary/non-speech behaviors** (audible inhalation, lip/tongue clicks,
filled pauses that aren't standard fillers) — likely out of scope until
the core word/phoneme-level detection is solid, but worth noting because
some clinical fluency assessments weight these meaningfully.

**Rate and rhythm metrics** (articulation rate, pause frequency
independent of any single "block" classification, speech rate variability
across a session) — these are aggregate signals computed across many
tokens/events rather than single-token classifications, and would slot
naturally into `SpeakerDifficultyProfile` as new tracked metrics alongside
the existing onset-risk scores, rather than as new `detect_disfluencies`
event types.

### Suggested order for Track 2

1. Finish the Tier 1 audit from `improve.md` (raw CrisperWhisper chunk
   inspection) — still not started, still the cheapest possible win if it
   pans out, and directly addresses the sub-word repetition gap.
2. Build the acoustic energy-envelope module (also still pending from
   `improve.md`) — this single piece of infrastructure feeds both the
   leading-silence-as-prolongation bug fix AND becomes the foundation for
   audible-block detection and prolonged-mid-word-sound detection above,
   so it should be built once and reused across three separate
   improvements rather than three times.
3. False-start/revision detection — independent of the acoustic work,
   can be prototyped purely against existing word-level tokens with a
   semantic-similarity check between consecutive short phrases.
4. Rate/rhythm aggregate metrics — lowest priority, purely additive to
   the profile, doesn't block or get blocked by anything else here.
5. Phoneme-level forced alignment for mid-word prolongation — highest
   effort, new dependency, do last and only if 1-4 don't already cover
   most real-world cases well enough.

---

## What "done" looks like for this round

- A definitive answer to scenario A vs. B above, backed by actual
  `perf_counter()` numbers logged from a real test run — not assumed.
- If scenario A: model warm-up moved to app startup, first-click latency
  documented as a known, accepted one-time cost; repeat-click latency
  confirmed to be in the single-digit-to-low-double-digit seconds range.
- If scenario B: root cause identified from 2a-2d with a number attached
  (e.g. "thread count fix took it from 1400s to 40s"), and a decision made
  on whether 3a (faster-whisper) is still needed on top of that fix to hit
  the "couple of seconds" target.
- Results appended below, same discipline as `improve.md` — record what
  was tried, the actual measured number, and the decision made, every
  time, before moving to the next item.

---

## Results log

*(empty — append here after each step is attempted, with the actual
perf_counter() breakdown from Step 0, not just a final total.)*
