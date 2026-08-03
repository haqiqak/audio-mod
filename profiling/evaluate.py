"""evaluate.py — backward-compatible shim over profiling/evaluation/.

This module's logic (built 2026-08-03) moved into the profiling/evaluation/
package the same day, as part of building out the full evaluation
methodology described in VALIDATION.md (dataset comparison, the two-track
approach, IoU localization, per-type confusion matrices, the combined "Any"
label — none of which existed in this file's original version). See
PAPER_DECISION_LOG.md's "Building the profiling/evaluation package" entry
for the full reasoning.

What stays compatible: the CLI.
    python -m profiling.evaluate --self-test
    python -m profiling.evaluate --data-dir DIR
both still work exactly as before (the new --dataset flag defaults to
"libristutter", matching this file's original single-dataset behavior).

What changed, on purpose, not silently: `evaluate()`'s return shape. It used
to return just `dict[str, TypeCounts]`; it now returns
`(dict[str, TypeCounts], dict[str, float | None])` — counts plus the new
IoU-based localization rates. Anyone calling `evaluate()` directly (not just
through the CLI) needs to unpack the tuple. `TypeCounts` also gained a `tn`
field.

New work should target `profiling.evaluation.track_a` directly rather than
this shim.
"""

from __future__ import annotations

from profiling.evaluation.loaders import (  # noqa: F401
    LIBRISTUTTER_LABEL_MAP,
    LabeledClip,
    load_libristutter_csv,
    load_libristutter_dir as load_data_dir,  # old name, kept for compatibility
    synthetic_libristutter_sample as _synthetic_sample,
)
from profiling.evaluation.metrics import (  # noqa: F401
    ANY_LABEL,
    TypeCounts,
    format_confusion_matrix,
    localization_rate,
    score_word_level as score_clips,  # old name, kept for compatibility
)
from profiling.evaluation.report import format_table  # noqa: F401
from profiling.evaluation.track_a import evaluate, main, run_self_test  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
