# Documentation map

`audio-mod` is intended to grow into a research-quality system and
eventually support a paper — that means documentation is written
**continuously, as decisions are made**, not reconstructed afterward from
memory or git history. This file exists so anyone (human or AI assistant)
picking this project up — after a day, a month, or after the original
context has scrolled out of a chat — knows where to look and where to write.

Read this file first. Then read in the order that matches what you're doing:

| I want to... | Read |
|---|---|
| Install and run the app | `README.md` |
| Understand how the code works *right now* | `ARCHITECTURE.md` |
| Understand *why* it works this way — the decision history | `PAPER_DECISION_LOG.md` |
| Get a fast, scannable list of what changed and when | `CHANGELOG.md` |
| Know our evaluation methodology and see results so far | `VALIDATION.md` |
| Know what's planned next and in what order | `ROADMAP.md` |

---

## The files and their purpose

| File | Type | Purpose | Who updates it, and when |
|---|---|---|---|
| **`README.md`** | User-facing, living | Setup, usage, config reference. What a new user or contributor needs to run the app and verify it works. | Whenever setup/usage/config actually changes. Not a place for design reasoning. |
| **`ARCHITECTURE.md`** | Technical, living | A snapshot of *how the implementation works today* — data flow, module responsibilities, critical non-obvious settings, known limitations. Rewritten in place as the code changes; it describes the present, not the history. | Whenever a change to `profiling/` or `app.py` is significant enough that the old description would mislead the next reader. |
| **`PAPER_DECISION_LOG.md`** | Historical, **append-only** | The full reasoning trail: what was done, what alternatives were considered, why this choice over those alternatives, and what was actually measured. One dated entry per significant step, never rewritten — if something is later found wrong, a *new* entry corrects it, the old one stays. This is the primary source for writing the eventual paper's Methods/Related-Decisions narrative. | Every time a non-trivial implementation or architecture decision is made. Entries are added, never edited or deleted. |
| **`CHANGELOG.md`** | Historical, living index | A terse, reverse-chronological, one-line-per-change list — the fast-scan companion to `PAPER_DECISION_LOG.md`. Each line links to the log entry that has the full reasoning. Use this to answer "what changed since [date]" in ten seconds; use the decision log to answer "why." | Every time an entry is added to `PAPER_DECISION_LOG.md` — add the one-line summary here too, same day. |
| **`VALIDATION.md`** | Technical + results, living | The evaluation methodology (datasets, metrics, the two-track ASR-bypass/full-pipeline approach) *and* the actual results as they're produced — per-dataset, per-track, per-metric tables, ablations, and comparisons against published baselines. The methodology sections are stable once agreed; the results sections are append/update targets for every future evaluation run. | Methodology sections: when the evaluation approach itself changes (rare, should also get a `PAPER_DECISION_LOG.md` entry). Results/ablation tables: after every evaluation run, however small. |
| **`ROADMAP.md`** | Forward-looking, living | One place listing what's next, in priority order, across the whole project — not just evaluation. Points *at* the doc/entry with the full reasoning rather than duplicating it. | Whenever priorities shift, or an item is completed/dropped (move it, don't delete the trail — link to the `PAPER_DECISION_LOG.md`/`CHANGELOG.md` entry that closed it out). |

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
