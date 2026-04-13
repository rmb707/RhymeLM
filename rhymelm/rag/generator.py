"""Retrieval-augmented generator.

Pipeline:
1. User gives topic + artist selection
2. Retriever pulls top-k verse chunks from the target artist matching the topic
3. Context is injected into the prompt as "few-shot" style examples
4. Base LLM generates a new verse conditioned on that style

Key insight: no fine-tuning required. The model imitates the retrieved examples
via in-context learning, which is what GPT-2 XL is actually good at.
"""

import os
import re
from dataclasses import dataclass

import torch

from rhymelm.rag.index import VerseIndex
from rhymelm.rag.retriever import Retriever, RetrievalResult


@dataclass
class GenerationResult:
    verse: str
    retrieved_chunks: list
    retrieval_scores: list[float]
    full_prompt: str


class RAGGenerator:
    """Retrieval-augmented rap verse generator."""

    def __init__(
        self,
        index: VerseIndex,
        model_name: str = "gpt2-xl",
        device: str = "cuda",
    ):
        self.index = index
        self.retriever = Retriever(index)
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self._model.eval()
        params = sum(p.numel() for p in self._model.parameters())
        print(f"Loaded: {params/1e6:.0f}M params")

    def _build_prompt(
        self,
        topic: str,
        artist: str,
        retrieved: RetrievalResult,
        num_examples: int = 3,
    ) -> str:
        """Build a few-shot prompt from retrieved verses.

        Uses a natural-language style: write out the artist's name, show
        examples, then start a new verse with the topic as a seed phrase.
        This leverages GPT-2's natural completion behavior without special
        delimiters that can get echoed back prematurely.
        """
        examples = retrieved.chunks[:num_examples]

        parts = [f"Rap verses by {artist}:\n"]
        for i, chunk in enumerate(examples, 1):
            parts.append(f"Verse {i}:")
            parts.append(chunk.text.strip())
            parts.append("")

        parts.append(f"Verse {len(examples) + 1} — {topic}:")
        return "\n".join(parts)

    def generate(
        self,
        topic: str,
        artist: str,
        num_bars: int = 8,
        temperature: float = 0.85,
        top_p: float = 0.92,
        num_examples: int = 3,
    ) -> GenerationResult:
        """Generate a verse using retrieval-augmented prompting."""
        self._load_model()

        # Retrieve relevant verses from the artist's catalog
        retrieved = self.retriever.retrieve_diverse(
            query=topic,
            artist=artist,
            top_k=num_examples,
            fetch_k=num_examples * 4,
        )

        if not retrieved.chunks:
            # Fall back to no artist filter
            retrieved = self.retriever.retrieve_diverse(
                query=topic,
                artist=None,
                top_k=num_examples,
                fetch_k=num_examples * 4,
            )

        prompt = self._build_prompt(topic, artist, retrieved, num_examples)

        # Generate
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=num_bars * 20,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = output[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Clean: collect non-empty lines, stop at next "Verse N:" header
        # Handles outputs with double-blank-line separators between bars.
        lines = []
        stop_re = re.compile(r"^(Verse\s*\d+|Rap verses by|\[Verse|\[Chorus)", re.I)
        meta_prefixes = ("(Left)", "(Right)", "This is ", "In this ", "These lyrics")

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if stop_re.match(line):
                break
            if line.startswith(meta_prefixes):
                continue
            if len(line) < 2:
                continue
            lines.append(line)
            if len(lines) >= num_bars:
                break

        verse = "\n".join(lines)

        return GenerationResult(
            verse=verse,
            retrieved_chunks=retrieved.chunks,
            retrieval_scores=retrieved.scores,
            full_prompt=prompt,
        )
