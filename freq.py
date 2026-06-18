"""
freq.py — single choke-point for word-frequency lookups.

Why this exists: wordfreq's default ("best") loads the *large* English wordlist,
a multi-hundred-MB in-memory dict.  On RAM-constrained machines building it
raises MemoryError, which would otherwise crash the whole synonym pipeline
(engine ranking, semantic scoring, phonetic difficulty all call it).

Behaviour:
  • Default wordlist = "best" (== large when present) → identical numbers to the
    original code when there is enough RAM, so synonym quality is preserved.
  • On MemoryError it transparently and permanently downshifts to the small
    wordlist (a few MB) and keeps running.  For common words small == large.
  • Override the preference explicitly with the WORDFREQ_LIST env var
    (e.g. set WORDFREQ_LIST=small to force low-memory mode from the start).

Public API mirrors wordfreq.zipf_frequency(word, lang) so callers only swap the
import, not the call sites.
"""
import paths  # noqa: F401  — keeps caches in ./.cache, must precede wordfreq
import os
import sys

from wordfreq import zipf_frequency as _wf_zipf

_PREFERRED = os.environ.get("WORDFREQ_LIST", "best")
_active = _PREFERRED
_warned = False


def _downshift():
    global _active, _warned
    _active = "small"
    if not _warned:
        _warned = True
        print(f"[freq] '{_PREFERRED}' wordlist hit MemoryError; "
              f"falling back to the 'small' wordlist (lower RAM, common words "
              f"unaffected).", file=sys.stderr)


def zipf_frequency(word: str, lang: str = "en") -> float:
    """Zipf frequency [~0–8]; degrades to the small wordlist under memory pressure."""
    try:
        return _wf_zipf(word, lang, wordlist=_active)
    except MemoryError:
        if _active != "small":
            _downshift()
            try:
                return _wf_zipf(word, lang, wordlist="small")
            except MemoryError:
                return 0.0
        return 0.0


def active_wordlist() -> str:
    """Which wordlist is currently in use (for UI/status display)."""
    return _active
