# CHANGELOG.md — fast-scan history

Reverse-chronological, one line per change. This is the "what changed and
when" index; for "why," follow the pointer into `PAPER_DECISION_LOG.md` (its
entries are titled to match). See `DOCS.md` for how these files relate.

---

## 2026-08-03

- **Documentation architecture established.** Added `DOCS.md` (documentation
  map), `VALIDATION.md` (evaluation methodology + results/ablation
  placeholders), `ROADMAP.md` (consolidated forward-looking priorities), and
  this file. Retired `august.md` into a stub — its content is now split
  between `PAPER_DECISION_LOG.md` (reasoning) and `VALIDATION.md`
  (methodology). → *PAPER_DECISION_LOG.md, "Documentation architecture
  established"*
- **Evaluation methodology researched and planned; no implementation yet.**
  Datasets compared (LibriStutter, SEP-28k, KSoF, UCLASS), a two-track
  (ASR-bypass / full-pipeline) methodology designed, metrics selected
  (precision/recall/F1, per-type confusion matrices, IoU localization, EER).
  Explicitly deferred pending the baseline commit. → *VALIDATION.md;
  PAPER_DECISION_LOG.md, "Evaluation methodology research (VALIDATION.md)"*
- **Real-audio validation pass; one bug found and fixed.** Two real
  microphone recordings tested end-to-end. Found and fixed a pre-existing
  display bug: the Event table showed an acoustic-sourced event's full
  attributed-word span instead of the actual detected region
  (`acoustic_start`/`acoustic_end`). Also surfaced (not fixed — no ground
  truth to act on): a missed word-repetition traceable to ASR transcription
  fidelity, and possible block-detection over-sensitivity. →
  *PAPER_DECISION_LOG.md, "Real-audio validation pass + event-table display
  fix"*
- **Audio-native-primary detector restructuring.** `profiling/detect.py`
  taxonomy split (`repetition` → `sound_repetition`/`word_repetition`/
  `phrase_repetition`, matching the SEP-28k/FluencyBank/KSoF standard);
  acoustic corroboration added for `filler`/`stutter_marker` (previously zero
  audio grounding); token-vs-acoustic fusion changed from fixed priority to
  weighted-confidence; config-driven `detectors` enable-list.
  `profiling/acoustic.py` enriched with Silero VAD (self-disabling
  corroboration signal, not a hard replacement — see the entry for why) and
  Praat pitch/jitter/shimmer/HNR. New `profiling/evaluate.py` (v1,
  timestamp-only, LibriStutter-schema, self-tested). 38/38 tests passing
  (28 pre-existing + 10 new); only 2 pre-existing tests needed a deliberate
  taxonomy-label update. → *PAPER_DECISION_LOG.md, "Audio-native-primary
  detector restructuring"*
- **Vision-alignment review + background research: architecture decision.**
  Full codebase re-read against the restated mission (audio-based disfluency
  localization, transcription as scaffolding). Literature/dataset/model
  review across ~25 papers and 6 datasets. Decision: keep the two-stage
  (ASR + detector) design, reject end-to-end region-detection architectures,
  restructure the detector to be audio-native-primary. → *PAPER_DECISION_LOG.md,
  "Vision alignment review + architecture decision"*

## 2026-06-27

- **Real-audio validation + prolongation threshold tune (Part D).**
  `prolongation_min_seconds` raised 0.65 → 1.0 after real fluent-speech false
  positives (naturally emphasized vowels). Long-recording ASR truncation
  found, not yet fixed (open issue). → *PAPER_DECISION_LOG.md*
- **Quality: phonetic near-repetition for short words.** ARPAbet
  phoneme-distance comparison for words ≤4 chars, spelling-based edit
  distance otherwise. → *PAPER_DECISION_LOG.md*
- **Quality: generalized phrase-repetition (any length).** Scan window
  widened from a hardcoded 2–3 words to a configurable
  `phrase_repetition_max_words` (default 8). → *PAPER_DECISION_LOG.md*
- **Fuse acoustic cues into the live detector (Option B, step 2).**
  `profiling/acoustic.py`'s candidates merged into `detect_disfluencies()`
  for the first time, with overlap-based dedup (token path always won —
  changed 2026-08-03, see above). → *PAPER_DECISION_LOG.md*
- **Realtime foundation: ASR-independent acoustic detection (Option B,
  step 1).** `profiling/acoustic.py` added — pure-NumPy waveform
  segmentation and prolongation/block candidate derivation, no ASR. Not
  wired into the live detector yet at this point. → *PAPER_DECISION_LOG.md*
- **§1 Option A: acoustic cross-validation of word timestamps.** Fixed
  clip-initial silence being billed to the first word (falsely inflating
  both that word's duration and the clip-wide prolongation percentile
  threshold). → *PAPER_DECISION_LOG.md*
- **Step 1c (real run) + Step 3: measured ASR latency & profiling.** First
  real benchmark numbers on a 16GB machine: ~54s (4s clip) to ~102s (20s
  clip), one-time ~29s model load. Corrected prior stale "~650-680s
  regardless of clip length" assumption. → *PAPER_DECISION_LOG.md*

## 2026-06-26

- **Docs: fixed a stale OpenVINO comment** contradicting the actual default
  backend (`transformers`). → *PAPER_DECISION_LOG.md*
- **Step 1c: real benchmark blocked** — model load OOM/segfault on a 2.2GB-
  free-RAM dev machine; deferred to a 16GB machine (resolved next day, see
  above). → *PAPER_DECISION_LOG.md*
- **Step 1b: benchmark harness** (`profiling/benchmark_asr.py`) added, with
  a `--self-test` mode verified against a stub model before any real run. →
  *PAPER_DECISION_LOG.md*
- **Step 1a: `last_timing` made self-describing** — added
  `clip_duration_seconds`, `max_new_tokens`, `audio_size_bytes`, `backend` to
  `CrisperWhisperASR.last_timing`. → *PAPER_DECISION_LOG.md*

## Before 2026-06-26

Earlier project history (the original planning docs `improve.md`,
`future.md`, `for-claude.md` and their inline results logs) predates
`PAPER_DECISION_LOG.md` and was consolidated into `ARCHITECTURE.md` at commit
`7c7e808`. Not individually reconstructed here — see `ARCHITECTURE.md` for
what survived as still-accurate technical detail.
