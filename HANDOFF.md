# HANDOFF.md — start here

You are picking up `audio-mod` cold — a new researcher, or a new Claude
Code/Claude session with no memory of any prior conversation about this
project. This file is the front door: read it first, in full, before
touching code or citing a result. It tells you what this project is,
exactly where it stands, what order to read the other docs in, and the
standing rules that keep this project's evidence trustworthy. Everything
here is a pointer to a fuller source, not a replacement for reading that
source before you rely on it.

---

## 1. What this project is, in three sentences

`audio-mod` is a Streamlit app that detects, classifies, and localizes
stuttering disfluencies (fillers, repetitions, blocks, prolongations) from
a speech recording — audio-based detection, not just transcription.
Transcription (CrisperWhisper ASR) is scaffolding for that goal, not the
end product. The detector is **audio-native-primary**: an acoustic module
derives its own candidates straight from the waveform and reconciles them
with transcript/timing-based checks through weighted-confidence fusion,
rather than trusting the ASR transcript as ground truth.

## 2. Where the project stands right now

**Both Phase 1 and Phase 2 are closed.** This is a mature, evidence-audited
codebase, not a fresh start — read the closing summaries before assuming
anything is untested or undecided:

- **Phase 1 (Validation, Benchmarking, Analysis)**, closed 2026-08-03:
  established a real evaluation baseline (`profiling/evaluation/`, real
  LibriStutter data, Track A/B methodology) and confirmed the audio-native
  architecture change actually works, while finding that a perfect-
  transcript evaluation drastically overstates real-world performance. →
  **`PHASE_1_SUMMARY.md`**
- **Phase 2 (evidence-driven improvement)**, closed 2026-08-04: a
  literature-grounded taxonomy review, two real detector-code fixes, a
  redesigned prolongation detector (decided by a pre-registered ablation,
  not intuition), new confidence-sensitive/interval-based measurement
  infrastructure, and several explicitly-documented negative and
  inconclusive results. → **`PHASE_2_SUMMARY.md`**

**If you are about to suggest "what should we do next," don't guess —
read `ROADMAP.md` first.** It is very likely already there, in priority
order, each item linked to the specific finding that justifies it. A
change of plan should update `ROADMAP.md`, not silently diverge from it.

## 3. Reading order

Read `DOCS.md` next for the complete file-by-file map (purpose, update
cadence, who owns what) — this section is just a curated path through it
for a first read:

1. **`README.md`** — what the app does, how to install and run it, the
   full config reference. Read this if you need to get the app running.
2. **`PHASE_1_SUMMARY.md`** then **`PHASE_2_SUMMARY.md`** — the two
   closing snapshots. Together they tell you what's been proven, what's
   been tried and failed (and why that's valuable, not embarrassing), and
   what's still open. Read both before citing any result as settled.
3. **`ARCHITECTURE.md`** — how the code works *right now*. Read this
   before changing `profiling/` or `app.py`.
4. **`VALIDATION.md`** — the evaluation methodology and every measured
   result, including ablations, negative results, and a critical review
   of the methodology itself (§7). This is the largest, most detailed
   file in the project. Read the relevant section before citing any
   specific number, and check its dated addenda — results here are
   sometimes revised by later evidence, recorded transparently rather
   than silently rewritten.
5. **`ROADMAP.md`** — what's next, in priority order, across the whole
   project. Read this before proposing new work.
6. **`PAPER_DECISION_LOG.md`** — the full append-only reasoning trail
   (what was done, alternatives considered, why, and what was measured).
   Read the relevant dated entry when you need the *why* behind something
   `ARCHITECTURE.md` or `VALIDATION.md` states as fact. Don't read it
   front-to-back; use `CHANGELOG.md` to find the entry you need.
7. **`CHANGELOG.md`** — fast-scan, reverse-chronological, one line per
   change, each linking into `PAPER_DECISION_LOG.md`. Use this to answer
   "what changed since [date]" in ten seconds.
8. **`PHASE_2_RESEARCH_PLAN.md`** — the literature-grounded review of
   whether the taxonomy itself is scientifically sound, written before
   any Phase 2 code changed. Read this if you're touching detection logic
   for a specific disfluency type and want to know what the clinical/
   computational literature says about it.
9. **`CLAUDE.md`** — short, stable orientation and the standing rules
   (condensed again in §5 below). Loaded automatically for a Claude Code
   session; read it directly if you're a human or a different tool.

## 4. What's proven vs. what's still a hypothesis

Don't treat everything in this repo as equally certain. The short version
(full detail in the two phase summaries' own "confirmed findings" and
"remains a hypothesis" sections):

**Solid, evidence-backed:**
- The audio-native architecture change (VAD + Praat + weighted fusion)
  measurably improves precision at ~0 recall cost.
- A perfect-transcript evaluation (Track A) drastically overstates
  real-world performance — real ASR conditions (Track B) matter, a lot.
- Once corrected for a real methodological gap, most (not all) of that
  real-world recall shortfall is attributable to ASR, not the detector —
  but the *exact* split has already been revised once as evidence
  improved (35.1%/64.9% at full speaker diversity, not the earlier 0%/
  100% read from a smaller, less diverse sample) and should be treated as
  provisional, not final.
- `sound_repetition`'s fragment-ordering fix and the prolongation
  redesign's Praat-gating default are both measured, pre-registered
  wins, not intuition.

**Explicitly still open, not settled:**
- Whether "ASR fidelity is the dominant bottleneck" generalizes beyond
  one ASR backend (CrisperWhisper) and one dataset family (LibriStutter).
- Whether the prolongation redesign's win transfers to real, non-
  reconstructed-timing speech.
- Whether UCLASS's annotation format actually does or doesn't support
  audible/tense block detection (investigated, inconclusive from public
  sources — not confirmed either way).
- Whether VAD/Praat corroboration's confidence-adjustment mechanism is
  worth its complexity, given a near-zero measured effect on one dataset.

## 5. Standing rules — read before doing anything non-trivial

These are established conventions from how this project has actually been
run (full text: `CLAUDE.md`). They are not aspirational:

1. **Pre-register methodology before implementing it**, for anything
   evaluation-related — write the exact metric/protocol/success-criteria
   into `VALIDATION.md` *before* writing the code that produces the
   result. See `VALIDATION.md` §9.5 for a canonical example: the
   prolongation redesign's decision criteria were fixed before the
   ablation ran, so accepting one variant and rejecting two others
   required no post-hoc judgment call.
2. **Document continuously, not retroactively.** Every non-trivial
   decision, result, bug, or finding gets a `PAPER_DECISION_LOG.md` entry
   the same day, plus a one-line pointer in `CHANGELOG.md`.
   `PAPER_DECISION_LOG.md` is append-only — never edit or delete a past
   entry, even a wrong one; a later entry corrects it explicitly.
3. **Audit surprising results before trusting them.** This project has
   caught multiple real bugs this way (a `soundfile` dtype bug that
   silently zeroed real audio; a rate-normalized prolongation formula
   that collapsed FP counts by 8x) — a dramatic-looking number is a
   reason to check harder, not report faster.
4. **Never tune thresholds/config in response to an evaluation result
   without explicit go-ahead.** Findings get recorded as evidence; acting
   on them is a separate, explicitly-approved step. (The prolongation
   default change was pre-authorized this way — see rule 1's example.)
5. **Docs drift — verify against the running code before trusting a claim
   in any `.md` file**, this one included, for anything consequential.
6. **Never commit or push without being explicitly asked, every time.** A
   past approval doesn't carry forward to the next commit.
7. **ASCII-only in any print-reachable string.** Enforced automatically
   now by `tests/test_ascii_console_output.py` — the Windows `cp1252`
   console has broken on non-ASCII characters in evaluation-harness
   output three separate times before this check existed.

## 6. Getting productive quickly

- **Run the app**: see `README.md`'s setup section.
- **Run the full test suite** (fast, no ASR model needed — synthetic
  audio only): each file under `tests/` is runnable standalone
  (`python tests/test_X.py`) and prints a `N/N passed` summary; there is
  no `pytest` dependency installed in this environment, run them this
  way. Current count: 45/45 across 8 files.
- **Run an evaluation**: `profiling/evaluation/track_a.py` (detector-only,
  ASR bypassed) and `track_b.py` (full pipeline, real ASR) are both
  self-testable (`--self-test`) and runnable against real data
  (`--data-dir`/`--audio-dir`). See `VALIDATION.md` §6 for the package
  layout and §5.1 for Track B's alignment protocol before touching either.
- **Run an ablation**: `python -m profiling.evaluation.run_ablations
  --data-dir eval_datasets/libristutter_sample --audio-dir
  eval_datasets/libristutter_sample_audio` — loads clips once, re-runs
  the detector once per config variant. Takes a long time (Praat feature
  extraction across every variant is the bottleneck) — expect it to run
  for the better part of an hour on a full 499-clip sample across many
  variants; check the process is still consuming CPU before assuming it's
  hung.
- **Where results land**: every scored run is saved as a timestamped JSON
  file under `eval_results/` (gitignored) via `report.save_run()` — never
  silently overwritten, so a history of runs accumulates. Raw console
  output for long-running scripts is worth redirecting to a file (see any
  recent `PAPER_DECISION_LOG.md` entry's "Measured result" section for
  the pattern).

## 7. Common pitfalls (things that have actually gone wrong here)

- **Windows console + non-ASCII characters**: covered by rule 7 above and
  now enforced by a test, but if you're writing a *new* script outside
  `profiling/` (the lint check's current scope), stay ASCII-only in
  anything you `print()`.
- **`soundfile.read(..., dtype="int16")` silently zeroes real FLAC
  audio** on this project's LibriStutter pipeline — always read with the
  default float64 dtype and scale manually; a "too good to be true" or
  "collapsed to zero" result is the symptom (`VALIDATION.md` §8.3).
- **Track B's per-clip cache stores only ASR output, never detector
  output** — by design, so it never goes stale relative to `detect.py`
  code changes. If you're adding a new cache anywhere in this project,
  match this principle: cache the expensive, code-independent part
  (ASR), always recompute the cheap, code-dependent part (detection).
- **A plausible mechanism does not guarantee a good outcome** — the
  word-sandwiched repetition extension (`PHASE_2_SUMMARY.md` §4) looked
  reasonable and was implemented carefully, and still regressed
  aggregate F1 with zero benefit on the track that could actually show
  its cost. Measure before trusting an intuition, even a well-reasoned
  one — and measuring on the *right track* (Track A vs. B) matters,
  since each can only show certain effects.
- **Small-n results deserve a Wilson interval, not just a qualitative
  caveat.** `metrics.wilson_interval()` exists now specifically because
  this project's own "1.0 recall" claims at n=2 and n=7 turned out to
  have much wider true uncertainty than the point estimate suggested
  (`VALIDATION.md` §8.4.3).

---

Once you've read this file and the two phase summaries, you should be able
to answer: what does this app do, what's been proven about it, what's
still open, and what's the evidence-ranked next step. If you can't answer
one of those, go back to §3's reading order before writing any code.
