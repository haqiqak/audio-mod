"""
semantic.py (stub) — provides only _PROTECTED_SINGLE for the standalone
profiling app.  The full semantic / synonym pipeline is not needed here.
"""

_PROTECTED_SINGLE: frozenset[str] = frozenset({
    "a", "an", "the", "i", "me", "my", "we", "us", "our",
    "you", "your", "he", "him", "his", "she", "her", "it",
    "its", "they", "them", "their", "this", "that", "these",
    "those", "and", "but", "or", "so", "yet", "for", "nor",
    "in", "on", "at", "to", "of", "by", "up", "as", "is",
    "be", "do", "go", "am", "are", "was", "were", "has", "had",
    "not", "no", "if",
})
