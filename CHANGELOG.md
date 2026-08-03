# CHANGELOG.md — fast-scan history

Reverse-chronological, one line per change. This is the "what changed and
when" index; for "why," follow the pointer into `PAPER_DECISION_LOG.md` (its
entries are titled to match). See `DOCS.md` for how these files relate.

---

## 2026-08-03

- **Phase 1 (Validation, Benchmarking, Analysis) formally closed.** Critical
  review of the full evaluation methodology at the project owner's request:
  fixed a real gap (Track B's decomposition used an *approximated* Track A
  recall from the full 499-clip sample instead of the exact matched-clip
  subset — now computed exactly, turning the context-strict
  detector-attributable share from "≈0%" into **exactly 0%**); identified
  two new, real generalization limits on the confirmed Track B conclusion
  (the clip subset is speaker-clustered — 7/40 speakers at n=90 — and the
  whole finding rests on one ASR backend + one synthetic dataset), neither
  fixed this session but both promoted to top Phase 2 priorities; fixed
  several cases of stale documentation (`README.md`'s config table showed
  the pre-Part-D `prolongation_min_seconds` default; `ARCHITECTURE.md`
  pointed at the retired `august.md`; `VALIDATION.md` described Track B as
  "not yet built" in several places after it was built). Added `CLAUDE.md`
  as a stable session-start orientation file. Closing summary:
  `PHASE_1_SUMMARY.md` (new). → *PAPER_DECISION_LOG.md, "Phase 1 closing
  review: exact-subset Track A recall, critical methodology review,
  documentation consolidation"; full critical review in `VALIDATION.md`
  §7.2–§7.4*
- **Track B scaled 30 → 90 clips — context-strict finding CONFIRMED, now a
  major research conclusion.** `R_B|preserved_ctx1` recall = 1.0 again, this
  time on n=7 positive instances (up from n=2), with a stable ~5.5%
  sample-attrition rate (127 disfluent instances → 7 surviving both the
  word-correct and context-correct filter). **The detector's binary
  disfluent/clean judgment is effectively perfect given intact ASR input;
  the real-world recall shortfall is overwhelmingly an ASR-fidelity
  problem, not a detector problem.** A second, sharpened finding: even when
  binary detection succeeds, exact type labeling for `word_repetition`/
  `sound_repetition` is frequently wrong (2/7 exact matches) — a real,
  scoped detector-side issue distinct from the recall question. Synthesis:
  future development priority shifts toward ASR robustness on disfluent
  speech + this scoped type-classification fix; detector-recall tuning is
  explicitly de-prioritized. `ROADMAP.md` restructured accordingly. →
  *PAPER_DECISION_LOG.md, "Track B scaled 30 → 90 clips: context-strict
  finding confirmed, treated as a major conclusion"; full results in
  `VALIDATION.md` §8.4.2*
- **Context-strict preserved-subset scoring implemented and run — reverses
  the previous detector/ASR attribution split.** Pre-registered in
  `VALIDATION.md` §5.1's addendum before implementing (same discipline as
  Track B itself). `R_B|preserved_ctx1` requires both a disfluent word *and*
  its immediately preceding word to survive ASR intact. Result: **~95%
  detector-attributable / ~5% ASR-attributable → ~0% detector-attributable
  / ~100% ASR-attributable** (`R_B|preserved_ctx1` = 1.0, n=2 — only 2 of 48
  disfluent instances survive the stricter filter, itself a finding about
  how thoroughly ASR corrupts local context). Added per-clip caching to
  `track_b.py` so this and future metric refinements skip re-running
  CrisperWhisper. A further nuance: even the 2 surviving instances got the
  wrong *exact* type label (`word_repetition` predicted as
  `phrase_repetition`/`block`) — traced to a hypothesis-side-contiguity gap
  the new metric still doesn't check, flagged as the next candidate
  refinement, not implemented yet. → *PAPER_DECISION_LOG.md, "Context-strict
  preserved-subset scoring implemented and run"; full results in
  `VALIDATION.md` §8.4.1*
- **Track B implemented and piloted — evaluation protocol pre-registered
  first, exactly as requested.** Wrote the full protocol into
  `VALIDATION.md` §5.1 (Levenshtein alignment with a disfluent-word cost
  bias; exact metrics at three levels — Track A, Track B ASR-preserved
  subset, Track B overall; the ASR-vs-detector decomposition formula; two
  explicit success criteria) *before* writing any Track B code. Built
  `alignment.py` + `track_b.py`, self-tested (10 checks against
  hand-computed expectations), then ran a 30-clip pilot with real
  CrisperWhisper ASR. **Headline finding: Track A's ~99% recall drops to
  ~4–9% under real ASR conditions** — the most important result this
  evaluation effort has produced, and direct proof Track A alone
  overstated real-world performance. Hand-verified alignment quality on 10
  clips (the pre-registered methodological gate — passed) and found a real
  limitation in the process: ASR can correctly transcribe a disfluent word
  while still breaking the *adjacent* context a detector check depends on,
  meaning the mechanical 95%/5% detector/ASR split likely overstates
  genuine detector-only failure — recorded as a dated addendum to §5.1, not
  smoothed over. → *PAPER_DECISION_LOG.md, "Track B evaluation protocol
  pre-registered before implementation" and "Track B implemented and
  piloted exactly per the pre-registered protocol"; full results in
  `VALIDATION.md` §8.4*
- **Complete ablation study against the baseline** (10 variants, same 499
  real clips): `prolongation_min_seconds` dominates by an order of
  magnitude (`Any` F1 range 0.639–0.933 across the sweep); `fusion_weights.
  acoustic` has a small real effect (+0.003 F1, saturating at 2×); Silero
  VAD and Praat corroboration both measured **zero** effect on the
  presence/absence metric used — traced to a real methodological gap (the
  metric can't see confidence adjustments, their designed effect), not
  evidence they're unhelpful; speaker calibration not applicable to this
  dataset. Added a `use_praat` config toggle to `profiling/acoustic.py`
  (defaults `True`, verified no behavior change) as ablation
  infrastructure. → *PAPER_DECISION_LOG.md, "Full ablation study against the
  baseline"; full tables in `VALIDATION.md` §9*
- **First research baseline established: audio-native-layer evaluated
  against labeled ground truth for the first time.** Re-examined the
  validation strategy against the project's core objective and prioritized
  closing a self-identified gap (the audio-native detection layer had never
  been evaluated at all — Track A always ran with `audio_bytes=None`) ahead
  of acquiring SEP-28k's audio, the previously-assumed next step. Built
  audio-enabled Track A support (`LabeledClip.audio_bytes`,
  `load_libristutter_*_with_audio`, FLAC→16kHz-WAV conversion via new
  `soundfile` dependency), downloaded matching real audio for the same
  499-clip sample (499/499, zero failures). **Found and fixed a real bug**
  before trusting the result: `soundfile.read(..., dtype="int16")` silently
  zeroed real LibriStutter FLAC files; caught by direct RMS audit, not by
  the metrics table looking wrong. Corrected result: `Any`-label F1
  0.773 → 0.835 with audio (157 fewer false positives, ~0 recall cost) —
  real evidence the architecture's design goal is being met.
  `prolongation` recall dropped (37→21 TP), mechanistically traced to
  voiced-duration trimming on reconstructed-token timing, not a detector
  bug. This checkpoint is now the project's first established research
  baseline — future algorithm changes are expected to be measured against
  it. → *PAPER_DECISION_LOG.md, "Audio-enabled evaluation", "Bug found and
  fixed: `_flac_bytes_to_wav16k` silently produced silent audio", "First
  audio-native-layer result (LibriStutter, 499 clips, corrected) — baseline
  established"; full results in `VALIDATION.md` §8.3*
- **First real Track A result**: 499 real LibriStutter clips (17,970 tokens),
  scored via `profiling/evaluation/track_a.py`. Headline: `Any` combined
  label 99.1% recall / 63.3% precision. Per-type numbers required real
  interpretation — a direct FP audit confirmed `word_repetition`'s alarming
  22.2% raw precision is mostly a reconstruction artifact (87.1% against
  clean speech only), while `prolongation`'s poor precision (4.8%) is
  substantially real even after the same correction. `sound_repetition`
  found at 0% recall, traced to a structural fragment-ordering mismatch
  between the reconstruction and the detector's fragment check — flagged
  for investigation, not fixed this round. → *PAPER_DECISION_LOG.md, "First
  real Track A result (LibriStutter, 499 clips)"; full results in
  `VALIDATION.md` §8.2*
- **Real LibriStutter/SEP-28k schemas confirmed against downloaded files —
  one loader bug found and fixed, one loader newly built.** LibriStutter's
  real format uses a `"STUTTER"` placeholder row (not a label on a real
  word, as originally assumed) — `load_libristutter_csv` fixed to
  reconstruct it into a plausible token instead of feeding the literal word
  "STUTTER" to the detector. `load_sep28k_labels` written and verified
  against the real, complete 28,177-row file (schema confirmed: clip-level,
  counts-out-of-3-annotators, no transcript). → *PAPER_DECISION_LOG.md, "Real
  LibriStutter/SEP-28k schemas confirmed; SEP-28k labels loader built"*
- **`profiling/evaluation/` package built** (sequencing steps 1–2 of
  `VALIDATION.md` §6): word-level scoring (`score_word_level`) migrated from
  `evaluate.py` v1 with no behavior change, plus new IoU-based localization,
  per-type confusion matrices (TP/FP/FN/TN), and the combined "Any" label.
  `profiling/evaluate.py` is now a thin backward-compatible shim.
  **Finding**: SEP-28k is clip-level labeled with no reference transcript at
  all (confirmed against its README) — added `ClipLevelLabels`/
  `score_clip_level` to handle that shape correctly; its CSV parser itself
  deferred pending the real file (exact column names unconfirmed). Two real
  bugs caught and fixed while building this: a same-second result-file
  timestamp collision, and non-ASCII characters breaking under this
  machine's Windows console codepage. → *PAPER_DECISION_LOG.md, "Building the
  profiling/evaluation package"*
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
