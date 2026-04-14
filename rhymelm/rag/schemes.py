"""Rhyme scheme parsing and 4-line window classification.

Self-contained — no model dependencies. Reused by:
- profiler.py (analyze artist corpora)
- index.py (tag chunks at index build time)
- rhyme_processor.py (enforce schemes at generation time)
"""

from __future__ import annotations

# Repeating patterns used by the parser. The pattern string is consumed
# letter-by-letter; each unique letter maps to a sequential group ID.
RHYME_SCHEMES: dict[str, str | None] = {
    "AABB": "AABBCCDDEEFFGGHHIIJJ",
    "ABAB": "ABABCDCDEFEFGHGHIJIJ",
    "ABBA": "ABBACDDCEFFEGHHGIJJI",
    "AAAA": "AAAABBBBCCCCDDDDEEEE",
    "free": None,
}


def parse_rhyme_scheme(scheme: str, num_bars: int) -> list[int]:
    """Map a scheme name (or raw pattern string) to per-line group IDs.

    Example: parse_rhyme_scheme("AABB", 8) -> [0, 0, 1, 1, 2, 2, 3, 3]
    """
    pattern = RHYME_SCHEMES.get(scheme, scheme)
    if pattern is None:
        return list(range(num_bars))

    label_map: dict[str, int] = {}
    groups: list[int] = []
    counter = 0
    for ch in pattern:
        if ch not in label_map:
            label_map[ch] = counter
            counter += 1
        groups.append(label_map[ch])

    while len(groups) < num_bars:
        groups.append(groups[-1] + 1)
    return groups[:num_bars]


def classify_window(suffixes: list[str | None]) -> str:
    """Classify a 4-line window's rhyme pattern from its ending suffixes.

    Returns one of: AABB, ABAB, ABBA, AAAA, irregular, unknown.
    Returns "unknown" when any suffix is None (out-of-vocab word).
    """
    if len(suffixes) < 4 or any(s is None for s in suffixes):
        return "unknown"

    s0, s1, s2, s3 = suffixes[0], suffixes[1], suffixes[2], suffixes[3]

    if s0 == s1 == s2 == s3:
        return "AAAA"
    if s0 == s1 and s2 == s3 and s0 != s2:
        return "AABB"
    if s0 == s2 and s1 == s3 and s0 != s1:
        return "ABAB"
    if s0 == s3 and s1 == s2 and s0 != s1:
        return "ABBA"
    return "irregular"


def get_line_ending_word(line: str) -> str:
    """Extract the last alphabetic word from a line, stripping punctuation."""
    words = line.strip().split()
    if not words:
        return ""
    # Walk backwards to find a word with at least one letter
    for w in reversed(words):
        cleaned = w.strip(".,!?;:'\"()-—…").lower()
        if cleaned and any(c.isalpha() for c in cleaned):
            return cleaned
    return ""


def get_window_suffixes(lines: list[str], cmu, get_rhyme_suffix_fn) -> list[str | None]:
    """For a window of lines, return the rhyme suffix of each line's ending word."""
    suffixes: list[str | None] = []
    for line in lines:
        word = get_line_ending_word(line)
        if not word:
            suffixes.append(None)
            continue
        suffixes.append(get_rhyme_suffix_fn(word, cmu))
    return suffixes
