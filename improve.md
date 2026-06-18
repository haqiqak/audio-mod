# Stammr Profiler — Improvement Roadmap

This document lays out two separate problems observed in testing, why each
one happens, and the order in which to fix them. They are independent —
fixing latency does not fix accuracy, and vice versa — so they're treated as
two tracks that can be worked in either order, though Track 1 should come
first because every accuracy experiment becomes painful to iterate on if
each test costs 1500 seconds.

---

## Test case that exposed both problems

Recording: ~3 seconds, spoken as "...I am fffine" (real stutter on "fine",
real leading silence before "I" starts).

Result:
```
"!"      0.00–0.00
"I"      0.00–1.38   (flagged: prolongation)
"am"     1.38–1.84
"fine."  1.84–3.08   (NOT flagged — stutter missed entirely)
```

Time to result: ~1500 seconds (25 minutes) on CPU.

Two bugs, two different layers of the pipeline:

1. **The "fffine" stutter was never flagged**, because it was never
   *preserved* — CrisperWhisper's decoder output the clean word "fine.",
   not "f-f-fine" or "ffine". The detector can only flag what's in the
   token list; if the disfluency was smoothed away before the detector
   ever runs, there's nothing to catch. This is a transcription-layer
   problem, not a detection-layer problem.

2. **"I" was flagged as a 1.38-second prolongation**, but that 1.38 seconds
   is almost certainly leading silence before you started speaking, not the
   word itself stretched out. The word's `start` timestamp is anchored at
   `0.0` (the beginning of the whole clip) instead of where the word
   actually begins acoustically. This is a timestamp-alignment problem.

3. **1500 seconds for 3 seconds of audio** is a 500x real-time factor,
   which is abnormally slow even for CPU inference on a 3.2GB model
   (normal CPU RTF for Whisper-large-class models is 2-8x, not 500x).
   Something in the current pipeline call is doing far more compute than
   necessary for a 3-second clip.

---

## Track 1 — Latency (1500s → target <15s on CPU for a 3s clip)

### Root cause hypothesis, ranked by likelihood

**1. The pipeline is plausibly being rebuilt from scratch on every recording.**
`CrisperWhisperASR.__init__()` is called fresh inside `run_pipeline()` on
every analysis, and `_load_pipeline()` only caches against `self._pipe` —
but `self` here is a brand-new `CrisperWhisperASR` instance every time,
because `app.py` does:
```python
asr = CrisperWhisperASR(device=device)   # <- new instance every run
```
This means the 3.2GB model is loaded from disk into memory from scratch on
every single click of "Stop & analyse", every single upload. Streamlit's
`st.cache_resource` is the canonical tool for fixing this category of
problem — cache the loaded pipeline object across reruns of the script,
keyed by device + model_id. This is the single highest-likelihood
explanation for a 500x real-time factor: model *loading* (reading 3.2GB
of weights off disk, deserializing safetensors, moving to device) easily
dominates total time on a 3-second clip, dwarfing actual inference.

**2. `low_cpu_mem_usage=True` on every call, including warm calls.**
This flag exists to reduce *peak RAM during model loading*, not to speed up
inference. It's irrelevant after the model is loaded once, but if the model
is reloading every time (per #1), this flag is active on every single one
of those reloads, adding overhead on top of an already-unnecessary reload.

**3. No `max_new_tokens` cap.**
Without bounding generation length, Whisper's decoder can run far more
generation steps than a 3-second clip warrants, especially combined with
the suppress-token logits processors CrisperWhisper installs. A short clip
should need on the order of 10-30 output tokens; if nothing caps it, a
single bad generation step (e.g. repetition without an EOS) can spiral.

**4. Float32 on CPU instead of an optimized build.**
`torch_dtype` isn't currently set explicitly. Default dtype for a model
loaded this way is float32, which is the slowest option on CPU. There's no
fp16 benefit on CPU (most CPUs don't get a speedup from it, sometimes a
regression), but ensuring the install has an MKL/oneDNN-accelerated torch
build matters a lot here — a torch built without AVX2/MKL support on
Windows can be 3-5x slower than one that has it.

**5. faster-whisper / CTranslate2 backend as a wholesale replacement.**
This is the highest-leverage single change available, separate from the
model-caching fix above. `nyrahealth` publish a CTranslate2-converted
variant (`nyrahealth/faster_CrisperWhisper`) built specifically for the
`faster-whisper` library, which is 4-8x faster than the stock
`transformers` pipeline on CPU for the *same* model weights, because
CTranslate2 uses quantized int8 kernels and a far more efficient
beam-search/greedy-decode implementation in C++ rather than Python/PyTorch
eager mode. This was referenced directly on the model's own README.

### Fix order (each step independently testable)

1. **Cache the pipeline with `st.cache_resource`.** Single highest-value
   fix relative to effort — likely turns "model load every time" into
   "model loads once per server process," which alone could explain
   the vast majority of the 1500-second figure. Should be done first
   because it makes every subsequent latency experiment faster to
   iterate on too.
2. **Add `max_new_tokens` (e.g. 128) to `generate_kwargs`.** Cheap,
   bounded-risk change; prevents runaway generation on edge-case audio.
3. **Switch to `faster-whisper` + `nyrahealth/faster_CrisperWhisper`.**
   Bigger change (different library, different model artifact, different
   token/timestamp output shape to adapt `_tokens_from_chunks` to), but
   this is the change with the highest expected speedup on top of the
   caching fix. The model card's own usage example uses this combination
   with `beam_size=1`, which lines up with the num_beams fix already
   applied for the tensor-mismatch bug.
4. **Verify the torch install has CPU acceleration.** A one-line check:
   `torch.backends.mkldnn.is_available()` and `torch.get_num_threads()`. If
   the installed torch wheel lacks MKL-DNN, reinstalling with the correct
   wheel for the CPU architecture can matter as much as any code change.
5. **Re-benchmark.** Only after 1-4, measure real-time-factor on a fixed
   test clip (e.g. the 3-second "I am fine" recording) and record it in
   this file so future regressions are visible.

### What NOT to do

- Do not reach for `chunk_length_s` again — already proven wrong (causes
  the "experimental with seq2seq models" warning and contributed to the
  original tensor-mismatch bug).
- Do not reduce `return_timestamps` to `True` (segment-level) instead of
  `"word"` as a speed hack — this would silently break the entire
  disfluency-detection pipeline, which depends on word-level timing to
  find blocks, prolongations, and repetitions. If timing must be sacrificed
  for speed, that's a product decision to surface explicitly, not a quiet
  fallback.

---

## Track 2 — Detection accuracy (catching "fffine", fixing leading-silence)

### Root cause: CrisperWhisper is necessary but not sufficient

CrisperWhisper's whole value proposition is preserving disfluencies that
vanilla Whisper deletes — but "preserving" here means it tries not to
*delete* a stutter outright (so it shouldn't silently turn "I-I-I want" into
just "I want"). It does **not** guarantee that every repeated-letter or
elongated-onset pattern survives as separate, inspectable tokens. A
sub-word disfluency like "fff-fine" can still get absorbed into a single
clean output token if the acoustic model's confidence resolves it that way.
This means: **fixing the token-level detector cannot fully solve this.**
The fix has to happen partly upstream, at the audio/feature level, and
partly downstream, in how we interpret what CrisperWhisper *does* give us.

### Three-tier fix, in order of effort vs. payoff

**Tier 1 — Use what CrisperWhisper already gives us, more carefully (low effort)**

CrisperWhisper's whole differentiator is verbatim filler/disfluency
tokens — its own paper and model card describe specific stutter-marker
tokens it's trained to emit (the model documentation references
"differentiating fillers" and explicit pause/disfluency awareness). Right
now `_tokens_from_chunks()` only checks `chunk.get("is_stutter")` and
`word.endswith("-")` — it is not actually inspecting whether
CrisperWhisper emitted any special markers in its raw token stream before
the pipeline's chunk-decoding step collapses them into clean words. Before
adding any new model or audio processing, audit exactly what
`result["chunks"]` contains for a known-stutter recording — at the raw
pre-`_tokens_from_chunks` level — to see whether disfluency signal is
already present and just being discarded by our own post-processing.
This is the cheapest possible win if it pans out, and must be checked
first before assuming a bigger fix is needed.

**Tier 2 — Acoustic-level repetition/prolongation detection independent of ASR text (medium effort)**

This is the architecturally correct long-term fix. Word-level disfluency
detection inherently caps out at whatever granularity the ASR's tokenizer
gives us — and tokenizers are built to produce clean words, which is in
direct tension with capturing "fff-fine." The fix is to add a parallel,
audio-native repetition/prolongation detector that runs on the raw
waveform (or its mel-spectrogram / MFCC representation) independent of
what CrisperWhisper transcribes, using signal-level heuristics:
- Onset-repetition detection: looking for repeated short-duration
  spectral patterns within a small time window before a word's accepted
  start time (catches "f-f-f" before "fine" even if ASR text shows just
  "fine").
- Energy-envelope segmentation: detecting silence vs. voiced segments
  directly from the waveform's RMS/energy contour, independent of what
  word boundaries the ASR assigns — this is also the fix for the
  leading-silence bug (see below), since it gives an independent,
  ASR-agnostic measure of when speech actually starts.
- This tier requires a small DSP component (e.g. librosa or torchaudio
  for feature extraction, simple onset-detection algorithms — not a new
  ML model) and would sit as a new module (e.g. `profiling/acoustic.py`)
  that runs alongside, not instead of, the existing text-based detector.
  Results from both would be merged in `detect_disfluencies`.

**Tier 3 — Dedicated disfluency/stutter-classification model (highest effort)**

If Tier 2 still under-detects (likely for genuinely subtle stutters), the
next step is a small purpose-built classifier trained or fine-tuned
specifically for stutter-event detection (separate from general ASR) —
there is existing published research and some open datasets (e.g.
SEP-28k, FluencyBank) for stuttering event classification that could be
used to fine-tune a lightweight audio classifier (e.g. a small
wav2vec2-based or CNN-based model) whose only job is "does this 1-2 second
window contain a repetition/block/prolongation," run as a sliding window
over the whole clip. This is a real ML project in its own right and should
only be scoped after Tiers 1 and 2 are exhausted and shown to be
insufficient — it's the most accurate plausible option but by far the
most expensive to build and maintain.

### The leading-silence-as-prolongation bug specifically

This is a more contained, faster fix than the general accuracy work above,
and should be fixed regardless of which detection tier is pursued, because
it's actively producing false positives right now:

**Root cause:** word `start` timestamps from the pipeline appear to anchor
to the beginning of the audio/chunk rather than to the actual onset of
voiced speech, when there is leading silence. The detector's prolongation
rule (`profiling/detect.py`) trusts `token["start"]`/`token["end"]` at face
value with no independent check for whether that span is mostly silence.

**Newly confirmed, sharper version of the bug (code-review finding, not
yet re-tested against real audio):** the corrupted duration doesn't only
mislabel the one word — `detect_disfluencies()` computes its prolongation
threshold as the 90th percentile of *all* token durations in the clip
(`profiling/detect.py`, `_percentile(durations, prolong_pct)`), and that
list includes the corrupted leading-silence-as-duration value. A
back-of-envelope check using the exact durations from this test's tokens
showed the threshold shifting from ~1.0s to ~1.34s once the corrupted
value is included — meaning the leading-silence bug doesn't just cause
one false positive, it also raises the bar for detecting genuine
prolongations anywhere else in the same clip. This makes the fix slightly
more urgent than "one wrong label," since it can suppress true positives
elsewhere too. Still the same underlying fix applies (see below), but
this raises its priority relative to other Track 2 items.

**Fix, two complementary parts:**

1. **Immediate, detector-side mitigation:** before computing word duration
   for the prolongation check, trim leading/trailing silence from the
   token's time span using a simple energy-threshold check against the
   original audio (this requires passing the audio waveform, or at least
   its energy envelope, into `detect_disfluencies`, which it currently
   does not receive — today it only sees ASR token dicts, no audio). This
   is a real architecture change to the detector's inputs, not a one-line
   patch, but it's much smaller in scope than Tier 2 above (in fact this
   silence-trimming logic and the Tier 2 energy-envelope work overlap
   significantly and should be built together).
2. **Root fix, ASR-side:** investigate whether CrisperWhisper/transformers
   has a configuration option (or whether a newer transformers version
   has a fix) for anchoring word-start timestamps to actual voice onset
   rather than segment/chunk start. This may already be improved by
   switching to `faster-whisper`/CTranslate2 in Track 1, since CTranslate2's
   timestamp computation is a different implementation than the
   `transformers` DTW-on-cross-attention approach and may not have this
   specific anchoring issue. Worth checking before building bespoke
   silence-trimming logic — Track 1 and this fix may resolve together.

---

## Suggested execution order

1. Track 1, steps 1-2 (cache pipeline, cap max_new_tokens) — fast, safe,
   immediately makes iteration faster for everything after.
2. Track 2, Tier 1 audit (inspect raw chunks for already-present
   disfluency signal) — cheap to check, possibly free accuracy win.
3. Track 1, step 3 (faster-whisper switch) — bigger lift, but also may
   incidentally fix the leading-silence timestamp bug per the note above,
   so worth doing before building bespoke silence-trimming logic.
4. Re-test the exact "I am fffine" recording from this session as the
   benchmark case for both latency and the two specific accuracy bugs.
5. Only then decide whether Tier 2 (acoustic-level detection) is still
   needed, based on whether step 3 alone closed the gap.
6. Tier 3 (dedicated classifier) is out of scope until 1-5 are done and
   evaluated against a broader set of test recordings, not just this one
   clip.

---

## What "done" looks like for this round

- A 3-second test clip transcribes and analyzes in under ~15 seconds on
  CPU (from 1500s — two orders of magnitude).
- The "I am fffine" test case either flags the stutter on "fine" directly,
  or — at minimum — the raw token audit from Tier 1 produces clear evidence
  of *why* it can't be flagged from text alone, justifying the move to
  Tier 2.
- The leading-silence-as-prolongation false positive on "I" no longer
  occurs on this same test clip.
- This document gets a results section appended (not rewritten) after each
  tier/step is attempted, recording what was tried, the measured effect,
  and whether to proceed to the next tier — so this stays a living record
  of what's been tested, not just a plan.

---

## Results log

**Step 1+2 applied — pipeline caching + max_new_tokens cap**

What changed:
- `app.py`: added `get_asr(device)` wrapped in `@st.cache_resource`, replacing
  the per-click `CrisperWhisperASR(device=device)` construction inside
  `run_pipeline()`. The model now loads from disk once per server process
  instead of once per analysis click.
- `profiling/asr.py`: added `"max_new_tokens": 256` to `generate_kwargs`
  alongside the existing `num_beams: 1` fix.

Expected effect: the first analysis after a fresh `streamlit run` should
still take the full model-load time (this is unavoidable — reading 3.2GB
off disk takes what it takes), but every analysis *after* the first one
in the same running session should drop to roughly true inference time
(low single-digit seconds to ~15s for a 3-second clip on CPU), since the
model stays resident in memory across reruns.

Not yet measured on real hardware — needs verification on the next test
run: confirm (a) first click after restart is still slow as expected,
(b) second and third clicks in the same session are dramatically faster,
and (c) the previously-reported ~1500s figure does not recur on repeat
analyses within one running app instance.

Still pending from Track 1: faster-whisper/CTranslate2 switch (step 3),
torch CPU-acceleration verification (step 4). Not yet started.

Track 2 (accuracy) not yet started — Tier 1 raw-chunk audit is the next
step once Track 1's caching fix is confirmed working.

