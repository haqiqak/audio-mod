"""ASCII-only-in-print-reachable-strings lint check (`ROADMAP.md` item 13 /
`CLAUDE.md` standing rule 7).

The Windows `cp1252` console has broken on non-ASCII characters (em-dashes,
ellipses, section signs, arrows) inside `print()` output three separate
times across this project's history (`track_a.py`, `report.py`,
`track_b.py`), each fixed reactively after the fact. This is the "lint rule
instead of fixing it reactively a fourth time" item that was flagged but
never built.

Deliberately scoped to string literals passed directly as `print()`
arguments (via AST — not a byte-level file scan), not a whole-file ASCII
check: this codebase's docstrings/comments legitimately use em-dashes
throughout (they are never sent to the console), and a whole-file check
would flag dozens of files that have never caused this bug. Scanning
`profiling/` (the evaluation-harness / detection-pipeline package where
every prior incident happened) rather than the whole repo, for the same
reason: `app.py`/Streamlit output goes to a browser, not this console.

    pytest tests/test_ascii_console_output.py
    python tests/test_ascii_console_output.py
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIR = ROOT / "profiling"


def _non_ascii_print_literals(path: Path) -> list[tuple[int, str]]:
    """(lineno, offending substring) for every non-ASCII string literal
    (including f-string constant parts) passed directly as a `print()`
    argument in the file at `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if not sub.value.isascii():
                        hits.append((sub.lineno, sub.value))
    return hits


def test_no_non_ascii_in_print_calls_under_profiling() -> None:
    violations: list[str] = []
    for path in sorted(SCAN_DIR.rglob("*.py")):
        for lineno, text in _non_ascii_print_literals(path):
            rel = path.relative_to(ROOT)
            violations.append(f"{rel}:{lineno}: {text!r}")
    assert not violations, (
        "Non-ASCII character(s) found in print() string literal(s) under "
        "profiling/ -- this breaks on the Windows cp1252 console (see "
        "CLAUDE.md standing rule 7):\n" + "\n".join(violations)
    )


def _run_all() -> int:
    tests = [test_no_non_ascii_in_print_calls_under_profiling]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
