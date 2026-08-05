# Documentation map

`audio-mod` is intended to grow into a research-quality system and
eventually support a paper — that means documentation is written
**continuously, as decisions are made**, not reconstructed afterward from
memory or git history. This file exists so anyone (human or AI assistant)
picking this project up — after a day, a month, or after the original
context has scrolled out of a chat — knows where to look and where to write.

**If you are starting cold — a new researcher, or a session with no memory
of this project — read `HANDOFF.md` first, not this file.** It's the
primary entry point: what this project is, exactly where it stands, a
curated reading order through everything below, and the standing rules
that keep this project's evidence trustworthy. Come back to this file
(`DOCS.md`) once you already have that orientation and need the complete
file-by-file reference — purpose, update cadence, who owns what.

| I want to... | Read |
|---|---|
| Get oriented from scratch — start here if new to this project | `HANDOFF.md` |
| Install and run the app | `README.md` |
| Understand how the code works *right now* | `ARCHITECTURE.md` |
| Understand *why* it works this way — the decision history | `PAPER_DECISION_LOG.md` |
| Get a fast, scannable list of what changed and when | `CHANGELOG.md` |
| Know our evaluation methodology and see results so far | `VALIDATION.md` |
| Know what's planned next and in what order | `ROADMAP.md` |
| Get oriented as an AI assistant picking this project up cold | `CLAUDE.md` |
| Get the one-page state of the project as of the Phase 1 close | `PHASE_1_SUMMARY.md` |
| Get the one-page state of the project as of the Phase 2 close | `PHASE_2_SUMMARY.md` |
| Know whether our disfluency taxonomy is scientifically sound, and why | `PHASE_2_RESEARCH_PLAN.md` |
| Know whether the ASR-first two-stage architecture itself is still the right foundation, and why | `PHASE_3_ARCHITECTURE_REVIEW.md` |
| Know why a new, separate ASR-representation research track opened, and what it's investigating | `ASR_RESEARCH_TRACK.md` |

---

## The files and their purpose

| File | Type | Purpose | Who updates it, and when |
|---|---|---|---|
| **`HANDOFF.md`** | Onboarding, living | The primary entry point for anyone (human or AI) starting cold: what this project is, exactly where it stands (pointers into the phase summaries), a curated reading order through every other doc, what's proven vs. still a hypothesis, the standing rules condensed, and practical get-productive-fast instructions (run tests, run an evaluation, common pitfalls). Distinct from this file (`DOCS.md`): `HANDOFF.md` is a guided *path* for a first read; `DOCS.md` is the complete *reference* you come back to afterward. | Whenever the reading order, the proven/hypothesis split, or a standing rule changes enough to mislead a first-time reader — check it during any future phase close, same as `ROADMAP.md`. |
| **`README.md`** | User-facing, living | Setup, usage, config reference. What a new user or contributor needs to run the app and verify it works. | Whenever setup/usage/config actually changes. Not a place for design reasoning. |
| **`ARCHITECTURE.md`** | Technical, living | A snapshot of *how the implementation works today* — data flow, module responsibilities, critical non-obvious settings, known limitations. Rewritten in place as the code changes; it describes the present, not the history. | Whenever a change to `profiling/` or `app.py` is significant enough that the old description would mislead the next reader. |
| **`PAPER_DECISION_LOG.md`** | Historical, **append-only** | The full reasoning trail: what was done, what alternatives were considered, why this choice over those alternatives, and what was actually measured. One dated entry per significant step, never rewritten — if something is later found wrong, a *new* entry corrects it, the old one stays. This is the primary source for writing the eventual paper's Methods/Related-Decisions narrative. | Every time a non-trivial implementation or architecture decision is made. Entries are added, never edited or deleted. |
| **`CHANGELOG.md`** | Historical, living index | A terse, reverse-chronological, one-line-per-change list — the fast-scan companion to `PAPER_DECISION_LOG.md`. Each line links to the log entry that has the full reasoning. Use this to answer "what changed since [date]" in ten seconds; use the decision log to answer "why." | Every time an entry is added to `PAPER_DECISION_LOG.md` — add the one-line summary here too, same day. |
| **`VALIDATION.md`** | Technical + results, living | The evaluation methodology (datasets, metrics, the two-track ASR-bypass/full-pipeline approach) *and* the actual results as they're produced — per-dataset, per-track, per-metric tables, ablations, comparisons against published baselines, and a critical review of the methodology itself (§7). The methodology sections are stable once agreed; the results sections are append/update targets for every future evaluation run. This is the single largest and most detailed file in the project — if you only have time to read one file deeply before touching evaluation code or citing a number, read this one. | Methodology sections: when the evaluation approach itself changes (rare, should also get a `PAPER_DECISION_LOG.md` entry). Results/ablation tables: after every evaluation run, however small. |
| **`ROADMAP.md`** | Forward-looking, living | One place listing what's next, in priority order, across the whole project — not just evaluation. Points *at* the doc/entry with the full reasoning rather than duplicating it. | Whenever priorities shift, or an item is completed/dropped (move it, don't delete the trail — link to the `PAPER_DECISION_LOG.md`/`CHANGELOG.md` entry that closed it out). |
| **`CLAUDE.md`** | AI-assistant-facing, living | Short, stable orientation for a Claude Code session starting cold in this repo: what this project is, the standing rules that govern how work here gets done (pre-register before implementing, document continuously, never commit without being asked, etc.), and where to look for what. Loaded automatically at session start — deliberately kept short; it points at the other files rather than duplicating them. | Rarely — only when a *standing rule* changes (not for individual findings/results, which belong in the files it points to). |
| **`PHASE_1_SUMMARY.md`** | Historical, **snapshot, not living** | A one-time closing summary of Phase 1 (Validation, Benchmarking, Analysis) written when that phase was formally closed on 2026-08-03: confirmed findings, what's still a hypothesis, the validated bottlenecks driving Phase 2, and the readiness argument for starting it. Frozen at the point Phase 1 closed — later phases get their own summary file rather than this one being rewritten, so it stays a reliable snapshot of "what we knew when we stopped Phase 1," even after Phase 2 changes the picture. | Not updated after Phase 1 closed. |
| **`PHASE_2_SUMMARY.md`** | Historical, **snapshot, not living** | The same kind of one-time closing summary, for Phase 2 (evidence-driven improvement), written when that phase was formally closed on 2026-08-04: what was built, confirmed findings (including revised-not-just-confirmed ones), the one fully-documented negative result, what remains a hypothesis, and the evidence-ranked case for Phase 3. Frozen at the point Phase 2 closed, same discipline as `PHASE_1_SUMMARY.md`. | Not updated after Phase 2 closed. If a Phase 3 close happens later, it gets its own `PHASE_3_SUMMARY.md` rather than editing this one. |
| **`PHASE_2_RESEARCH_PLAN.md`** | Research/planning, **written before implementation, not living** | A literature-grounded review of whether the disfluency taxonomy itself is scientifically sound (clinical speech-pathology framework, computational detection literature, dataset annotation conventions), written *before* any Phase 2 code changed, per this project's pre-registration discipline applied to architecture decisions, not just evaluation protocols. Contains the gap analysis, dataset-compatibility analysis, subset-focus recommendation, and the structured plan for what Phase 2 actually does first. | Not a living results file like `VALIDATION.md` — if a *later* literature review happens (e.g. revisiting this once Phase 2 evidence comes in), it gets its own dated section or file rather than silently rewriting this one, same append-only spirit as `PAPER_DECISION_LOG.md`. |
| **`PHASE_3_ARCHITECTURE_REVIEW.md`** | Research/planning, **written before implementation, not living** | A first-principles challenge to the ASR-first two-stage architecture itself, written before any Phase 3 code changed: what speech representation actually gives the detector the highest accuracy, evaluated against a fresh 2024-2026 literature pass (SSL representations, disfluency-trained ASR, end-to-end and hybrid architectures, joint ASR+detection training) and this project's own Phase 1/2 empirical findings. Concludes the two-stage architecture is kept, and identifies one scoped, evidence-backed extension as the top Phase 3 candidate. | Not living — a later architecture review gets its own dated section or file, same append-only spirit as `PHASE_2_RESEARCH_PLAN.md`. |
| **`ASR_RESEARCH_TRACK.md`** | Research/planning, **charter for a separate branch (`asr-research`), living for the life of that track** | Opened 2026-08-05 after Track B validation (`VALIDATION.md` §14) found real ASR essentially never preserves the sub-word fragment tokens `sound_repetition` detection depends on. Reframes the core question as "how do we preserve the speech-production information conventional ASR intentionally removes," reviews the literature on disfluency-preserving ASR/representations, lays out architectural directions without committing to one, and sets a phased, evidence-gated research plan (Stages A-E) plus explicit criteria for when a purpose-built ASR/representation would be justified. Explicitly does not reopen `PHASE_3_ARCHITECTURE_REVIEW.md`'s two-stage-architecture conclusion — a narrower question about representation richness, not pipeline structure. `main` stays stable; this track's work happens on its own branch. | Updated as each stage in the plan produces results, on the `asr-research` branch — not on `main` until a stage's evidence is ready to ship. |

### Retired

- **`august.md`** — the working notes from the 2026-08 audio-native-primary
  restructuring round. Its content has been migrated into
  `PAPER_DECISION_LOG.md` (the reasoning) and `VALIDATION.md` (the
  evaluation-methodology plan it originated). Kept as a short stub pointing
  here rather than deleted, in case anything already links to it.

---

## Documentation philosophy

1. **Living docs describe the present; the decision log describes the past.**
   `ARCHITECTURE.md` and `README.md` should never say "we changed X to Y" —
   they should just describe the current state as if it always looked this
   way. The *why* and the *history of what it used to be* belong in
   `PAPER_DECISION_LOG.md`. Mixing these up is exactly how docs go stale:
   nobody wants to rewrite a wall of historical narrative just to fix one
   sentence about current behavior.
2. **The decision log is append-only, on purpose.** Never edit or delete a
   past entry, even a wrong one — add a new entry that corrects it, and say
   so explicitly ("Entry corrects the 2026-06-27 threshold claim, which..."
   ). This is what makes it trustworthy as a paper source months later: the
   record of what was believed *at the time*, and how that changed, is
   itself part of the research story.
3. **Every claim should be measured, not asserted, wherever possible.**
   This project's own convention (established well before this doc existed
   — see `PAPER_DECISION_LOG.md`'s early entries) is "measurement-first":
   don't write down a number you didn't actually run. Where a real
   measurement isn't available yet, say "not yet measured" explicitly rather
   than estimating and forgetting to correct it later.
4. **Docs drift — verify before trusting.** `ARCHITECTURE.md` itself says
   this and it's worth repeating here: before relying on any claim in these
   files for something consequential (a paper, a threshold change), check it
   against the running code. Long iterative sessions are exactly when a doc
   and the code it describes quietly diverge.
5. **Placeholders are not failures.** `VALIDATION.md`'s results tables start
   mostly empty ("not yet run") — that's intentional and correct at this
   stage, not a gap to hide. An empty, well-structured table you fill in
   later is worth more than no table at all.
