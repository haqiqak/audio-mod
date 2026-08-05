# CLAUDE.md

Orientation for a Claude Code session starting cold in this repository.
Kept short on purpose — it points at the other files rather than
duplicating them. **Read `HANDOFF.md` next — it's the primary entry
point** (curated reading order, what's proven vs. hypothesis, practical
get-productive instructions); `DOCS.md` is the complete file-by-file
reference to come back to afterward.

## What this project is

`audio-mod` is a Streamlit app that takes a speech recording and detects,
classifies, and localizes stuttering disfluencies (fillers, repetitions,
blocks, prolongations, etc.) — audio-based detection, not just
transcription. Transcription (CrisperWhisper ASR) is scaffolding for that
goal, not the end product. See `README.md` for what it does; `ARCHITECTURE.md`
for how the code implements it today.

## The objective, stated precisely — read this before proposing anything architectural

This project's long-term objective is **not** to optimize Whisper,
preserve the existing architecture, maximize transcript quality, or
pursue any individual component for its own sake. The objective is to
build **the most accurate, explainable, scientifically grounded
speech-disfluency detector possible, using only the user's audio**.
Every subsystem — CrisperWhisper ASR, transcripts, encoder embeddings,
acoustic evidence, classifiers, confidence signals, and any future
representation — exists only to serve that objective, and none of them
*is* the objective. Keep this hierarchy in view whenever a change touches
the ASR/detector boundary:

1. **User audio is the fundamental source of information.** Everything
   else in this pipeline is a derived representation of it.
2. **ASR is one subsystem within the detection pipeline, not the project
   itself.** CrisperWhisper was chosen because it was the best-evidenced
   option for producing word-aligned, disfluency-preserving transcripts —
   a decision that stays open to revision the same way any other does.
3. **The transcript is one evidence source, not ground truth.** Treating
   decoded ASR text as if it faithfully represented what the speaker
   produced is an assumption this project has found real ASR output can
   violate for specific disfluency types — do not assume it holds
   elsewhere without checking.
4. **Encoder representations, acoustic features, confidence signals, and
   any future representations are complementary evidence sources**,
   incorporated only when evidence supports that they help — never
   assumed to help by default, and never dismissed by default either.
5. **Architectural decisions remain evidence-driven, not
   preservation-driven** (this is standing rule 8 below, restated at the
   objective level, not just the implementation level). If future
   research shows another ASR, another representation, or another
   processing strategy objectively improves the final detector, adopt
   it. If an idea fails validation, reject it — regardless of how
   attractive it seemed going in. Neither direction gets a thumb on the
   scale.

## Where the project is right now

**Both Phase 1 (Validation, Benchmarking, Analysis) and Phase 2
(evidence-driven improvement) are closed** (2026-08-03 and 2026-08-04
respectively) — this is a mature, evidence-audited codebase, not a fresh
start. See `PHASE_1_SUMMARY.md` and `PHASE_2_SUMMARY.md` for the full
closing summaries. Phase 2 opened with a literature-grounded review of
whether the disfluency taxonomy itself is scientifically sound —
`PHASE_2_RESEARCH_PLAN.md` — which confirmed the core 5-type taxonomy is
correct, found two real gaps (see that file), and set the actual order of
Phase 2 work; that work is now done (two detector fixes, a prolongation
redesign decided by pre-registered ablation, new confidence/CI measurement
infrastructure, and several documented negative/inconclusive results —
see `PHASE_2_SUMMARY.md`). **Phase 3 (evidence-driven architecture
extension) is in progress**, opened 2026-08-04 with a first-principles
architecture review (`PHASE_3_ARCHITECTURE_REVIEW.md`) that kept the
ASR-first two-stage architecture but identified one scoped,
evidence-backed extension (`ROADMAP.md` item 17); that extension is now
**implemented and shipped** (2026-08-05) — a trained classifier gate on
`word_repetition`/`sound_repetition` (this project's first internally-
trained, shipped model artifact), decided by a pre-registered,
cross-validated comparison, default `true`. Full arc: `VALIDATION.md`
§11 (Stage 1 signal validation) → §12 (mechanism comparison) → §13
(implementation + benchmark) → §14 (Track B validation of that result
against real ASR, 2026-08-05 — see below). One real, accepted cost
remains open and tracked as `ROADMAP.md` item 18 (added live-app
latency, de-prioritized behind item 20, not yet resolved). Read
`PHASE_3_ARCHITECTURE_REVIEW.md` before proposing anything that touches
the ASR/detector boundary. `ROADMAP.md` reflects the current state:
priorities in order, each linked to the specific finding that justifies
it. If you're about to suggest a next step, check `ROADMAP.md` first —
it's very likely already there with reasoning, and a change of plan
should update it, not silently diverge from it.

**A new, separate research track opened 2026-08-05: `ASR_RESEARCH_
TRACK.md`.** Track B validation (`VALIDATION.md` §14/§14.1) found real
ASR output essentially never produces the literal sub-word fragment
tokens `sound_repetition` detection depends on — even at positions
transcribed correctly — evidence that the ASR stage's own output
representation may be discarding information this project's downstream
objective needs, independent of any detector-side fix. This is being
investigated as its own research track, on a dedicated `asr-research`
branch, kept separate from `main` so the shipped state stays stable.
**Read `ASR_RESEARCH_TRACK.md` before proposing anything about ASR
representation richness, fine-tuning, or decoding changes** — it defines
the problem, reviews the literature, and lays out a phased,
evidence-gated plan; nothing there is implemented yet.

## Standing rules for working in this repo

These are established conventions from how this project has actually been
run, not aspirational — follow them by default:

1. **Pre-register methodology before implementing it, for anything
   evaluation-related.** Write the exact metric/protocol/success-criteria
   into `VALIDATION.md` *before* writing the code that produces the result.
   If a deviation turns out to be necessary once implementation starts,
   record it as a dated addendum, not a silent edit. See `VALIDATION.md`
   §5.1 for the canonical example of this pattern.
2. **Document continuously, not retroactively.** Every non-trivial decision,
   result, bug, or finding gets a `PAPER_DECISION_LOG.md` entry (What was
   done / Alternatives considered / Why this choice / Measured result) the
   same day, plus a one-line pointer in `CHANGELOG.md`. `PAPER_DECISION_LOG.md`
   is append-only — never edit or delete a past entry, even a wrong one; a
   later entry corrects it explicitly instead.
3. **Audit surprising results before trusting them.** This project has
   caught multiple real bugs this way (a `soundfile` dtype bug that silently
   zeroed real audio; a scoring-approximation gap in a decomposition
   formula) — a dramatic-looking number is a reason to check harder, not a
   reason to report faster. Small samples get an explicit "too small to
   trust" caveat, not a confident headline.
4. **Never tune thresholds/config in response to an evaluation result
   without explicit go-ahead.** Findings get recorded as evidence
   (`VALIDATION.md`); acting on them is a separate, explicitly-approved step.
5. **Docs drift — verify against the running code before trusting a claim in
   any `.md` file**, this one included, for anything consequential. See
   `DOCS.md`'s documentation philosophy for the full reasoning.
6. **Never commit or push without being explicitly asked, every time.** A
   past approval doesn't carry forward to the next commit. Follow the
   project's own git conventions (new commits, not amends; no
   force-push/`--no-verify` without an explicit ask).
7. **ASCII-only in any print-reachable string.** This Windows/`cp1252`
   console has broken on non-ASCII characters (em-dashes, section signs,
   arrows) in evaluation-harness output three separate times, now enforced
   by `tests/test_ascii_console_output.py` (an AST-based check on `print()`
   calls under `profiling/`) — but the check's scope is that directory
   only, so stay disciplined in anything new outside it too.
8. **Architectural decisions are evidence-constrained, not preservation-
   constrained** (project owner, 2026-08-04). The current architecture,
   implementation style, and every prior design decision are hypotheses
   that earned their place through evidence — not defaults to protect for
   their own sake. Simplicity, interpretability, rule-based logic, machine
   learning, hybrid methods, pretrained components, and newly-trained
   components are all *engineering choices*, not objectives in themselves;
   none gets automatic priority over the others. The one question that
   governs an architectural decision: *which approach most effectively
   serves accurate disfluency detection/classification/localization, and
   what evidence supports that conclusion?* Autonomy to decide is granted,
   but it is evidence-constrained: reach a decision once real evidence
   supports it confidently (and record the reasoning so a future reader
   can see why, not just what), or — if the evidence doesn't yet support
   a confident call — say so explicitly, name exactly what's uncertain,
   pre-register the specific validation that would resolve it, run it, and
   only then decide. Do not guess, and do not default to the
   simpler/existing option just because it's simpler or existing. This
   sits alongside, not in tension with, rule 4: rule 4 is about not
   silently retuning a config value in response to noise; this rule is
   about who has standing to make a *considered, evidenced* architecture
   call once real evidence exists — see `PHASE_3_ARCHITECTURE_REVIEW.md`
   and `VALIDATION.md` §12 for the standard this was first applied to.

## Where to look for what

See `DOCS.md` for the full map. Quick pointers:
- A specific measured number or evaluation result → `VALIDATION.md` §8/§9
  (and check §7 for that finding's known limitations before citing it)
- Why a piece of code is the way it is → `PAPER_DECISION_LOG.md`
- Whether something is already planned → `ROADMAP.md`
- Config keys and defaults → `README.md`'s configuration table, cross-checked
  against `config.yaml` directly (the table has drifted from the real
  defaults before)
