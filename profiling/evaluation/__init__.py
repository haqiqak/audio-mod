"""profiling/evaluation — accuracy evaluation harness for detect_disfluencies().

See VALIDATION.md (project root) for the full methodology this package
implements: dataset comparison, the two-track (detector-only vs full-pipeline)
approach, metrics, and honest limitations. This module docstring only orients
the code; VALIDATION.md is the living reference.

Layout
──────
    loaders.py   Per-dataset loading into the shared LabeledClip shape.
                 Datasets differ in more than just label vocabulary — they
                 differ in GRANULARITY. LibriStutter labels individual words
                 (word-level ground truth, matches this app's own token
                 shape). Other datasets (e.g. SEP-28k) label whole clips
                 without a reference transcript at all (clip-level ground
                 truth: "did this 3-second clip contain a Block anywhere",
                 not "which word"). metrics.py provides a scorer for each
                 granularity — pick the one that matches what a given
                 loader actually produces, don't force one shape onto both.
    metrics.py    Dataset-agnostic scoring: word-level and clip-level TP/FP/
                 FN/TN, per-type binary confusion matrices, the combined
                 "Any" label, and IoU-based localization accuracy.
    track_a.py    Detector-only runner (ASR bypassed) — feeds a dataset's own
                 ground-truth words/timestamps straight into
                 detect_disfluencies(). Answers "how good is the detector
                 logic itself, given a perfect transcript?"
    report.py     Table rendering + timestamped result files (config,
                 dataset info, git commit) for reproducibility.

track_b.py (full pipeline: our own ASR + hypothesis-to-reference alignment)
is not part of this package yet — see VALIDATION.md §6 sequencing step 5.

profiling/evaluate.py (the pre-package v1) is now a thin backward-compatible
shim over this package — existing callers of
`python -m profiling.evaluate --self-test` keep working unchanged.
"""
