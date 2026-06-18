"""
phonetic.py — phoneme-onset utilities for stutter-aware synonym filtering.

The core idea: people who stutter tend to block on specific *starting sounds*
(onsets), e.g. the /str/ in "street" or the /pr/ in "present".  To help, we must
(a) recognise which words begin with a user's trouble onset, and (b) avoid
offering synonyms that begin with that same onset (replacing "present" with
"prestige" is useless — both start /pr/).

Onsets are taken from the CMU Pronouncing Dictionary (ARPAbet), NOT spelling,
so "knee"→N, "school"→S K, "psychology"→S are handled correctly.  Words missing
from CMU fall back to a grapheme→ARPAbet onset guess.

This module is deliberately standalone: it imports only paths/nltk/wordfreq and
is imported BY grammar/app, never the other way around, so it cannot destabilise
the existing synonym pipeline.

Public API
──────────
    onset(word)               -> tuple[str, ...]   ARPAbet onset cluster
    normalize_pattern(text)   -> tuple[str, ...]   user grapheme cluster -> onset
    matches_any(word, pats)   -> bool              word onset starts with any pattern
    word_difficulty(word)     -> float [0,1]
    sentence_difficulty(words)-> float [0,1]
"""
import paths  # noqa: F401  — must precede nltk import; redirects caches into ./.cache

import re
from functools import lru_cache

from freq import zipf_frequency   # memory-safe wrapper (falls back to 'small' wordlist)

# ── CMU dictionary (lazy load; degrade gracefully to spelling fallback) ────────
_CMU = None
_CMU_TRIED = False


def _cmu():
    global _CMU, _CMU_TRIED
    if _CMU_TRIED:
        return _CMU
    _CMU_TRIED = True
    try:
        import nltk
        try:
            from nltk.corpus import cmudict
            _CMU = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict
            _CMU = cmudict.dict()
    except Exception:
        _CMU = None
    return _CMU


# ── ARPAbet helpers ───────────────────────────────────────────────────────────
def _is_vowel_phone(phone: str) -> bool:
    """ARPAbet vowels carry a stress digit (0/1/2) as their last char."""
    return phone[-1].isdigit()


def _onset_from_phones(phones: list[str]) -> tuple[str, ...]:
    """Consonant phones up to (not including) the first vowel, stress stripped."""
    out: list[str] = []
    for p in phones:
        if _is_vowel_phone(p):
            break
        out.append(re.sub(r"\d", "", p))
    return tuple(out)


# ── Grapheme → ARPAbet onset (fallback + user-pattern parsing) ────────────────
# Silent / irregular leading clusters (only valid at the very start of a word).
_SILENT_START = {
    "kn": ("N",), "wr": ("R",), "ps": ("S",), "gn": ("N",),
    "mn": ("N",), "pn": ("N",), "pt": ("T",),
}
# Consonant digraphs (anywhere in the onset).
_DIGRAPH = {
    "sh": ("SH",), "ch": ("CH",), "th": ("TH",), "ph": ("F",),
    "wh": ("W",), "ck": ("K",), "gh": ("G",), "qu": ("K", "W"),
    "sc": ("S", "K"),
}
_SINGLE = {
    "b": ("B",),  "c": ("K",),  "d": ("D",),  "f": ("F",),  "g": ("G",),
    "h": ("HH",), "j": ("JH",), "k": ("K",),  "l": ("L",),  "m": ("M",),
    "n": ("N",),  "p": ("P",),  "q": ("K",),  "r": ("R",),  "s": ("S",),
    "t": ("T",),  "v": ("V",),  "w": ("W",),  "x": ("Z",),  "z": ("Z",),
    "y": ("Y",),
}
_VOWEL_LETTERS = set("aeiou")


def _grapheme_onset(text: str) -> tuple[str, ...]:
    """
    Best-effort onset from spelling: walk leading consonant graphemes until the
    first vowel letter.  Used for OOV words and for parsing user patterns like
    'str', 'pr', 'bo', 'sh'.
    """
    s = re.sub(r"[^a-z]", "", text.lower())
    onset: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in _VOWEL_LETTERS:
            break
        if ch == "y":
            if i == 0:
                onset.extend(_SINGLE["y"]); i += 1; continue
            break  # 'y' acts as a vowel after the first position
        two = s[i:i + 2]
        if i == 0 and two in _SILENT_START:
            onset.extend(_SILENT_START[two]); i += 2; continue
        if two in _DIGRAPH:
            onset.extend(_DIGRAPH[two]); i += 2; continue
        if ch in _SINGLE:
            onset.extend(_SINGLE[ch]); i += 1; continue
        i += 1
    return tuple(onset)


# ── Public API ────────────────────────────────────────────────────────────────
@lru_cache(maxsize=8192)
def onset(word: str) -> tuple[str, ...]:
    """
    ARPAbet onset cluster for *word* (e.g. 'present' -> ('P','R')).
    Uses CMU dict; falls back to a grapheme guess for out-of-vocabulary words.
    """
    if not word:
        return ()
    w = word.strip().lower()
    cmu = _cmu()
    if cmu and w in cmu and cmu[w]:
        return _onset_from_phones(cmu[w][0])
    return _grapheme_onset(w)


@lru_cache(maxsize=2048)
def normalize_pattern(text: str) -> tuple[str, ...]:
    """
    Convert a user-typed grapheme cluster ('str', 'pr', 'bo', 'sh') into the
    ARPAbet onset sequence it represents ('bo' -> ('B',) because the user means
    'the B sound at the start').  Returns () for empty/garbage input.
    """
    return _grapheme_onset(text)


def matches_any(word: str, patterns) -> bool:
    """
    True if *word*'s onset begins with the onset of ANY pattern in *patterns*.

    *patterns* is an iterable of raw user strings (e.g. ['str','pr','bo']).
    Empty/None patterns -> always False, which makes every caller a no-op when
    the user hasn't entered any stutter patterns (preserves legacy behaviour).
    """
    if not patterns:
        return False
    w_onset = onset(word)
    if not w_onset:
        return False
    for pat in patterns:
        target = normalize_pattern(pat)
        if target and w_onset[:len(target)] == target:
            return True
    return False


# ── Phonetic difficulty (for the before/after metric in the UI) ───────────────
def _syllable_count(word: str) -> int:
    cmu = _cmu()
    w = word.strip().lower()
    if cmu and w in cmu and cmu[w]:
        return max(1, sum(1 for p in cmu[w][0] if _is_vowel_phone(p)))
    # spelling fallback: count vowel-letter groups
    groups = re.findall(r"[aeiouy]+", w)
    return max(1, len(groups))


@lru_cache(maxsize=8192)
def word_difficulty(word: str) -> float:
    """
    Heuristic stutter difficulty in [0,1] for a single word:

        base  = 0.4 × onset_cluster_length  +  0.3 × syllable_load  +  0.3 × rarity
        bonus = +0.15 if onset starts with a plosive or affricate

    Plosives (P B T D K G) and affricates (CH JH) are the phoneme classes most
    associated with stuttering blocks — a single-phoneme onset /p/ is harder
    to initiate than a single-phoneme fricative /s/ or sonorant /m/.
    The bonus is capped so the total never exceeds 1.0.
    """
    _PLOSIVE_AFFRICATE = {"P", "B", "T", "D", "K", "G", "CH", "JH"}

    w = word.strip().lower()
    if not w or not re.search(r"[a-z]", w):
        return 0.0
    word_onset     = onset(w)
    onset_score    = min(len(word_onset) / 3.0, 1.0)
    syll_score     = min(_syllable_count(w) / 4.0, 1.0)
    rarity         = 1.0 - min(zipf_frequency(w, "en") / 7.0, 1.0)
    base           = 0.4 * onset_score + 0.3 * syll_score + 0.3 * rarity
    # Plosive/affricate penalty: first phoneme in the onset cluster
    plosive_bonus  = 0.15 if (word_onset and word_onset[0] in _PLOSIVE_AFFRICATE) else 0.0
    return round(min(base + plosive_bonus, 1.0), 4)


def sentence_difficulty(words) -> float:
    """Average word_difficulty over the alphabetic content words; 0 if none."""
    scored = [word_difficulty(w) for w in words if re.search(r"[a-z]", str(w).lower())]
    if not scored:
        return 0.0
    return round(sum(scored) / len(scored), 4)
