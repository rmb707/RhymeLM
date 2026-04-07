"""Originality filter: prevent the model from copying training data verbatim.

The core problem: character-level LSTMs can memorize and regurgitate training
lines. This module builds an n-gram index of the training corpus and uses it
to penalize generation that too closely matches known lines.

Two mechanisms:
1. Blocked n-grams: if the last N generated characters match a training n-gram,
   suppress the next character that would continue that match.
2. Line-level dedup: after generating each line, check similarity to training
   lines and regenerate if too similar.
"""

import re
from collections import defaultdict


class OriginalityFilter:
    """Indexes the training corpus and blocks verbatim reproduction during generation."""

    def __init__(self, corpus: str, ngram_size: int = 25):
        """
        Args:
            corpus: the full training corpus text
            ngram_size: character n-gram length to block. Longer = allows
                more overlap before blocking. 25 chars ≈ 4-5 words.
        """
        self.ngram_size = ngram_size
        self.blocked_continuations = defaultdict(set)
        self.training_lines = set()
        self._build_index(corpus)

    def _build_index(self, corpus: str):
        # Index character n-grams → next character
        for i in range(len(corpus) - self.ngram_size):
            prefix = corpus[i : i + self.ngram_size]
            next_char = corpus[i + self.ngram_size]
            self.blocked_continuations[prefix].add(next_char)

        # Index full lines for line-level checking
        for line in corpus.splitlines():
            line = line.strip().lower()
            if len(line) > 15:
                self.training_lines.add(line)

        print(f"Originality filter: {len(self.blocked_continuations):,} n-gram prefixes, "
              f"{len(self.training_lines):,} training lines indexed")

    def get_blocked_chars(self, generated_text: str) -> set[str]:
        """Given text generated so far, return characters that would continue a training n-gram."""
        if len(generated_text) < self.ngram_size:
            return set()
        suffix = generated_text[-self.ngram_size:]
        return self.blocked_continuations.get(suffix, set())

    def is_line_copied(self, line: str, threshold: float = 0.85) -> bool:
        """Check if a generated line is too similar to any training line.

        Uses character-level Jaccard similarity on overlapping 5-grams.
        """
        line_clean = line.strip().lower()
        if len(line_clean) < 15:
            return False

        # Quick exact match check
        if line_clean in self.training_lines:
            return True

        # 5-gram similarity check
        line_grams = set(self._char_ngrams(line_clean, 5))
        if not line_grams:
            return False

        for train_line in self.training_lines:
            train_grams = set(self._char_ngrams(train_line, 5))
            if not train_grams:
                continue
            intersection = len(line_grams & train_grams)
            union = len(line_grams | train_grams)
            if union > 0 and intersection / union > threshold:
                return True

        return False

    def is_line_copied_fast(self, line: str) -> bool:
        """Fast check: is this exact line (lowered, stripped) in training data?"""
        return line.strip().lower() in self.training_lines

    @staticmethod
    def _char_ngrams(text: str, n: int) -> list[str]:
        return [text[i : i + n] for i in range(len(text) - n + 1)]


def apply_originality_penalty(
    logits,  # torch.Tensor (1, vocab_size)
    generated_text: str,
    originality_filter: OriginalityFilter,
    tokenizer,
    penalty: float = 10.0,
):
    """Suppress logits for characters that would continue a memorized n-gram.

    This is applied during generation to steer the model away from
    reproducing training data verbatim while still allowing it to use
    learned patterns and vocabulary.
    """
    blocked = originality_filter.get_blocked_chars(generated_text)
    if not blocked:
        return logits

    for ch in blocked:
        ch_id = tokenizer.stoi.get(ch)
        if ch_id is not None:
            logits[0, ch_id] -= penalty

    return logits
