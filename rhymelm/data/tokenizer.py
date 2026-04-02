"""Character-level tokenizer with encode/decode and vocabulary management."""

import json
from pathlib import Path

import torch


class CharTokenizer:
    """Maps characters to integer indices and back."""

    def __init__(self, chars: list[str]):
        self.chars = sorted(set(chars))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_corpus(cls, corpus: str) -> "CharTokenizer":
        chars = sorted(set(corpus))
        tok = cls(chars)
        print(f"Tokenizer: {tok.vocab_size} unique characters")
        return tok

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def decode(self, indices) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return "".join(self.itos[i] for i in indices)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        return cls(data["chars"])
