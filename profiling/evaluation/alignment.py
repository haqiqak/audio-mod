"""alignment.py — word-level alignment between an ASR hypothesis and a
ground-truth reference, implementing the protocol pre-registered in
VALIDATION.md §5.1 (written BEFORE this file — read that first if changing
anything here; a change in scoring behavior here should also show up as a
dated addendum in that section, not a silent edit).

Standard Levenshtein (edit-distance) word alignment, with one deliberate
bias: substituting a reference word that carries a ground-truth disfluency
label costs more than substituting a clean word. This makes the aligner
prefer explaining a mismatch near a disfluent word as a deletion (ASR lost
it) rather than forcing a coincidental, low-quality substitution match —
the modified-cost technique this project's literature review found used
for exactly this problem (see PAPER_DECISION_LOG.md, "Vision alignment
review" entry, and the pre-registration entry for this module).
"""

from __future__ import annotations

from dataclasses import dataclass
import re


def _norm(word: str) -> str:
    """Same normalization profiling/detect.py's _norm uses — lowercase,
    alphabetic only — so alignment and detection agree on what "the same
    word" means."""
    return re.sub(r"[^a-z]", "", (word or "").lower())


@dataclass
class AlignmentOp:
    ref_index: int | None   # index into the reference sequence, or None for a pure insertion
    hyp_index: int | None   # index into the hypothesis sequence, or None for a deletion
    kind: str                # "correct" | "substitution" | "deletion" | "insertion"


def align(
    reference: list[str],
    hypothesis: list[str],
    disfluent_indices: set[int] | None = None,
    disfluent_cost_multiplier: float = 1.5,
) -> list[AlignmentOp]:
    """Levenshtein-align `hypothesis` against `reference`, word by word.

    Returns one AlignmentOp per reference word (kind in
    correct/substitution/deletion) plus one per pure insertion (hypothesis
    word with no reference counterpart), in reference order (insertions
    appear at the point in the sequence where they were produced).

    `disfluent_indices`: reference indices with a ground-truth disfluency
    label. Substituting one of these costs `disfluent_cost_multiplier`
    (default 1.5) times as much as substituting a clean word — see the
    module docstring for why. Costs are exact multiples of 1.0/1.5 (no
    floating-point division involved), so the backtrace's cost-equality
    checks are exact, not approximate.
    """
    disfluent_indices = disfluent_indices or set()
    ref_norm = [_norm(w) for w in reference]
    hyp_norm = [_norm(w) for w in hypothesis]
    n, m = len(ref_norm), len(hyp_norm)

    INS_COST = 1.0
    DEL_COST = 1.0

    def sub_cost_base(ref_idx: int) -> float:
        return disfluent_cost_multiplier if ref_idx in disfluent_indices else 1.0

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + DEL_COST
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + INS_COST
    for i in range(1, n + 1):
        base = sub_cost_base(i - 1)
        for j in range(1, m + 1):
            match = ref_norm[i - 1] == hyp_norm[j - 1]
            sub_cost = 0.0 if match else base
            dp[i][j] = min(
                dp[i - 1][j - 1] + sub_cost,
                dp[i - 1][j] + DEL_COST,
                dp[i][j - 1] + INS_COST,
            )

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = ref_norm[i - 1] == hyp_norm[j - 1]
            sub_cost = 0.0 if match else sub_cost_base(i - 1)
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                ops.append(AlignmentOp(
                    ref_index=i - 1, hyp_index=j - 1,
                    kind="correct" if match else "substitution",
                ))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + DEL_COST:
            ops.append(AlignmentOp(ref_index=i - 1, hyp_index=None, kind="deletion"))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + INS_COST:
            ops.append(AlignmentOp(ref_index=None, hyp_index=j - 1, kind="insertion"))
            j -= 1
            continue
        break  # unreachable if dp was filled correctly
    ops.reverse()
    return ops


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Standard WER = (substitutions + deletions + insertions) / len(reference).
    Uses the unbiased (disfluent_cost_multiplier=1.0) alignment — WER is a
    generic transcript-quality figure, not part of the disfluency-specific
    decomposition (which uses the biased alignment deliberately, see above)."""
    if not reference:
        return 0.0 if not hypothesis else float("inf")
    ops = align(reference, hypothesis, disfluent_indices=None, disfluent_cost_multiplier=1.0)
    errors = sum(1 for op in ops if op.kind in ("substitution", "deletion", "insertion"))
    return errors / len(reference)
