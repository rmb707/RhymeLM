"""Byte-Pair Encoding tokenizer implemented from scratch.

BPE iteratively merges the most frequent pair of tokens, building a
vocabulary of subword units. This allows the model to "think" in larger
units (common words become single tokens) while still handling unseen
words through character-level fallback.

Key tradeoff: with the same BLOCK_SIZE, BPE sees more semantic context
than character-level (e.g., 512 BPE tokens ≈ 2000+ characters), but
loses fine-grained character-level control over spelling and rhyme.
"""

import json
import re
from collections import Counter


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained from corpus."""

    def __init__(self, merges: list[tuple[str, str]], vocab: dict[str, int]):
        self.merges = merges
        self.vocab = vocab
        self.inverse_vocab = {i: tok for tok, i in vocab.items()}
        self.vocab_size = len(vocab)
        # Build merge priority lookup
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def train(cls, corpus: str, target_vocab_size: int = 1000, verbose: bool = True) -> "BPETokenizer":
        """Train BPE from a corpus by iteratively merging frequent pairs.

        Algorithm:
        1. Start with character-level tokens
        2. Count all adjacent pairs
        3. Merge the most frequent pair into a new token
        4. Repeat until target vocab size is reached
        """
        # Initialize: split corpus into character sequences per word
        # Using whitespace-aware pre-tokenization
        word_freqs: Counter = Counter()
        words = re.findall(r"\S+|\s", corpus)
        for word in words:
            chars = tuple(word)
            word_freqs[chars] += 1

        # Initial vocab: all unique characters
        char_vocab = set()
        for word_tuple in word_freqs:
            for ch in word_tuple:
                char_vocab.add(ch)

        vocab = {ch: i for i, ch in enumerate(sorted(char_vocab))}
        merges = []
        num_merges = target_vocab_size - len(vocab)

        if verbose:
            print(f"BPE: starting with {len(vocab)} chars, target {target_vocab_size}")

        for merge_i in range(num_merges):
            # Count adjacent pairs
            pair_counts: Counter = Counter()
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i + 1])] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]

            # Merge all occurrences
            new_word_freqs: Counter = Counter()
            for word_tuple, freq in word_freqs.items():
                new_word = []
                i = 0
                while i < len(word_tuple):
                    if (
                        i < len(word_tuple) - 1
                        and word_tuple[i] == best_pair[0]
                        and word_tuple[i + 1] == best_pair[1]
                    ):
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_word)] += freq

            word_freqs = new_word_freqs
            vocab[new_token] = len(vocab)
            merges.append(best_pair)

            if verbose and (merge_i + 1) % 200 == 0:
                print(f"  merge {merge_i + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' → '{new_token}'")

        if verbose:
            print(f"BPE: final vocab size = {len(vocab)}")

        return cls(merges, vocab)

    def encode(self, text: str) -> list[int]:
        """Encode text to token indices using trained BPE merges."""
        # Pre-tokenize
        words = re.findall(r"\S+|\s", text)
        all_ids = []

        for word in words:
            tokens = list(word)

            # Apply merges in priority order
            for pair in self.merges:
                i = 0
                new_tokens = []
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1
                        and tokens[i] == pair[0]
                        and tokens[i + 1] == pair[1]
                    ):
                        new_tokens.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            for tok in tokens:
                if tok in self.vocab:
                    all_ids.append(self.vocab[tok])
                else:
                    # Fallback: encode character by character
                    for ch in tok:
                        all_ids.append(self.vocab.get(ch, 0))

        return all_ids

    # Interface parity with CharTokenizer
    @property
    def stoi(self) -> dict[str, int]:
        return self.vocab

    @property
    def itos(self) -> dict[int, str]:
        return self.inverse_vocab

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self.inverse_vocab.get(i, "?") for i in ids)

    def encode_to_tensor(self, text: str):
        """Encode text and return a torch.LongTensor."""
        import torch
        return torch.tensor(self.encode(text), dtype=torch.long)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "merges": self.merges,
                "vocab": self.vocab,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        merges = [tuple(m) for m in data["merges"]]
        return cls(merges, data["vocab"])
