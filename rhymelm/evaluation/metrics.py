"""Evaluation metrics for generated verses: perplexity, diversity, rhyme quality."""

import math
from collections import Counter

import nltk
import torch
import torch.nn.functional as F


def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""
    return math.exp(loss)


def distinct_n(text: str, n: int = 1) -> float:
    """Ratio of unique n-grams to total n-grams. Measures lexical diversity."""
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def rhyme_density(verse: str) -> float:
    """Fraction of line-ending words that rhyme with at least one other line ending.

    Uses CMU pronouncing dictionary phoneme suffixes to determine rhyming.
    Two words rhyme if their last stressed vowel and all following phonemes match.
    """
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict

    cmu = cmudict.dict()

    lines = [ln.strip() for ln in verse.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0

    def get_rhyme_suffix(word: str) -> str | None:
        word = word.lower().strip(".,!?;:'\"()-")
        if word not in cmu:
            return None
        phonemes = cmu[word][0]
        # Find last stressed vowel
        for i in range(len(phonemes) - 1, -1, -1):
            if any(c.isdigit() for c in phonemes[i]):
                return " ".join(phonemes[i:])
        return None

    endings = []
    for line in lines:
        words = line.split()
        if words:
            suffix = get_rhyme_suffix(words[-1])
            endings.append(suffix)
        else:
            endings.append(None)

    rhyming = 0
    for i, s in enumerate(endings):
        if s is None:
            continue
        for j, t in enumerate(endings):
            if i != j and s == t:
                rhyming += 1
                break

    return rhyming / len(endings)


def verse_structure_score(verse: str) -> dict:
    """Evaluate structural quality of a generated verse."""
    lines = [ln for ln in verse.strip().splitlines() if ln.strip()]
    line_lengths = [len(ln) for ln in lines]

    # Repetition: fraction of unique lines
    unique_ratio = len(set(lines)) / max(len(lines), 1)

    # Line length variance (lower is more consistent)
    avg_len = sum(line_lengths) / max(len(line_lengths), 1)
    reasonable_lines = sum(1 for l in line_lengths if 5 <= l <= 120)

    return {
        "num_lines": len(lines),
        "avg_line_length": round(avg_len, 1),
        "unique_line_ratio": round(unique_ratio, 3),
        "reasonable_line_ratio": round(reasonable_lines / max(len(lines), 1), 3),
    }


def evaluate_verse(verse: str) -> dict:
    """Run all metrics on a generated verse."""
    return {
        "distinct_1": round(distinct_n(verse, 1), 4),
        "distinct_2": round(distinct_n(verse, 2), 4),
        "distinct_3": round(distinct_n(verse, 3), 4),
        "rhyme_density": round(rhyme_density(verse), 4),
        **verse_structure_score(verse),
    }
