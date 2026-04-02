"""Phoneme utilities built on CMU Pronouncing Dictionary.

Provides rhyme detection, syllable counting, phoneme vocabularies,
and a trie for constrained rhyme-aware generation.
"""

import re
from typing import Optional

import nltk


def get_cmu_dict() -> dict:
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    return cmudict.dict()


# ARPAbet phoneme inventory (39 phonemes + stress markers)
ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
}
ARPABET_CONSONANTS = {
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",
}
ALL_PHONEMES = sorted(ARPABET_VOWELS | ARPABET_CONSONANTS)


def build_phoneme_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Build phoneme-to-index and index-to-phoneme mappings.

    Includes stress variants (0, 1, 2) for vowels and a PAD token.
    """
    phonemes = ["<PAD>"]
    for p in ALL_PHONEMES:
        if p in ARPABET_VOWELS:
            phonemes.extend([f"{p}0", f"{p}1", f"{p}2"])
        else:
            phonemes.append(p)
    p2i = {p: i for i, p in enumerate(phonemes)}
    i2p = {i: p for i, p in enumerate(phonemes)}
    return p2i, i2p


def strip_stress(phoneme: str) -> str:
    return re.sub(r"\d", "", phoneme)


def get_rhyme_suffix(word: str, cmu: dict | None = None) -> Optional[str]:
    """Get the rhyme suffix: last stressed vowel through end of word."""
    if cmu is None:
        cmu = get_cmu_dict()
    word = word.lower().strip(".,!?;:'\"()-")
    if word not in cmu:
        return None
    phonemes = cmu[word][0]
    for i in range(len(phonemes) - 1, -1, -1):
        if any(c.isdigit() for c in phonemes[i]):
            return " ".join(phonemes[i:])
    return None


def words_rhyme(w1: str, w2: str, cmu: dict | None = None) -> bool:
    """Check if two words rhyme (share rhyme suffix)."""
    if w1.lower() == w2.lower():
        return False
    s1 = get_rhyme_suffix(w1, cmu)
    s2 = get_rhyme_suffix(w2, cmu)
    if s1 is None or s2 is None:
        return False
    return s1 == s2


def count_syllables(word: str, cmu: dict | None = None) -> int:
    """Count syllables using CMU dict (stress markers = vowel nuclei)."""
    if cmu is None:
        cmu = get_cmu_dict()
    word = word.lower().strip(".,!?;:'\"()-")
    if word not in cmu:
        # Fallback: count vowel clusters
        return max(1, len(re.findall(r"[aeiouy]+", word)))
    return sum(1 for p in cmu[word][0] if any(c.isdigit() for c in p))


def line_syllable_count(line: str, cmu: dict | None = None) -> int:
    """Count total syllables in a line."""
    if cmu is None:
        cmu = get_cmu_dict()
    return sum(count_syllables(w, cmu) for w in line.split() if w.strip())


def get_word_phonemes(word: str, cmu: dict | None = None) -> Optional[list[str]]:
    """Get ARPAbet phoneme sequence for a word."""
    if cmu is None:
        cmu = get_cmu_dict()
    word = word.lower().strip(".,!?;:'\"()-")
    if word not in cmu:
        return None
    return cmu[word][0]


def build_rhyme_groups(cmu: dict | None = None) -> dict[str, list[str]]:
    """Group words by their rhyme suffix for fast lookup."""
    if cmu is None:
        cmu = get_cmu_dict()
    groups: dict[str, list[str]] = {}
    for word in cmu:
        suffix = get_rhyme_suffix(word, cmu)
        if suffix:
            groups.setdefault(suffix, []).append(word)
    return groups


class RhymeTrie:
    """Character-level trie of CMU dict words for constrained generation.

    During generation, when approaching a line ending, we can constrain
    sampling to follow paths in the trie that lead to words whose phoneme
    suffix matches a target rhyme.
    """

    def __init__(self, cmu: dict | None = None):
        if cmu is None:
            cmu = get_cmu_dict()
        self.cmu = cmu
        self.rhyme_groups = build_rhyme_groups(cmu)
        self.trie: dict = {}
        self._build()

    def _build(self):
        for word in self.cmu:
            node = self.trie
            for ch in word:
                node = node.setdefault(ch, {})
            node["$"] = word  # terminal marker

    def get_rhyming_words(self, target_suffix: str, max_words: int = 100) -> list[str]:
        """Get words that rhyme with the given suffix."""
        return self.rhyme_groups.get(target_suffix, [])[:max_words]

    def valid_next_chars(self, prefix: str) -> set[str]:
        """Get valid next characters given a prefix in the trie."""
        node = self.trie
        for ch in prefix:
            if ch not in node:
                return set()
            node = node[ch]
        return {k for k in node if k != "$"}

    def is_complete_word(self, word: str) -> bool:
        node = self.trie
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        return "$" in node
