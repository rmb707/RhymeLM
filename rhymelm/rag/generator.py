"""Retrieval-augmented generator with per-artist style profiling.

Pipeline:
1. User gives topic + artist selection
2. Generator looks up the artist's style profile (dominant rhyme scheme,
   signature words, syllable density)
3. Retriever pulls top-k verse chunks from the target artist matching the
   topic, re-ranked to prefer chunks that demonstrate the dominant scheme
4. Prompt is built with the profile metadata + retrieved examples as
   in-context anchors
5. Base LLM generates a new verse, optionally with a LogitsProcessor that
   biases token selection toward rhyming completions at line endings

Key insight: no fine-tuning required. The model imitates retrieved examples
via in-context learning, while profiling tells it (and the user) what the
artist's actual signature is.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from rhymelm.rag.index import VerseIndex
from rhymelm.rag.retriever import Retriever, RetrievalResult


@dataclass
class GenerationResult:
    verse: str
    retrieved_chunks: list
    retrieval_scores: list[float]
    full_prompt: str
    profile: dict | None = None


class RAGGenerator:
    """Retrieval-augmented rap verse generator with per-artist profiles."""

    def __init__(
        self,
        index: VerseIndex,
        model_name: str = "gpt2-xl",
        device: str = "cuda",
        profiles_path: str = "artist_profiles.json",
    ):
        self.index = index
        self.retriever = Retriever(index)
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self.profiles: dict[str, dict] = {}
        # Lazy-loaded by generate() when rhyme enforcement is requested
        self._cmu = None
        self._rhyme_groups: dict[str, list[str]] | None = None
        self._load_profiles(profiles_path)

    def _load_profiles(self, path: str):
        if not Path(path).exists():
            print(f"[generator] No profiles file at {path}; running without per-artist style hints")
            return
        with open(path) as f:
            self.profiles = json.load(f)
        print(f"[generator] Loaded {len(self.profiles)} artist profiles from {path}")

    def get_profile(self, artist: str) -> dict | None:
        """Look up an artist's profile, with case-insensitive fallback."""
        if artist in self.profiles:
            return self.profiles[artist]
        # Case-insensitive lookup
        for k, v in self.profiles.items():
            if k.lower() == artist.lower():
                return v
        return None

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

    def _truncate_chunk(self, text: str, max_lines: int) -> str:
        """Trim a verse chunk to its first `max_lines` non-empty lines."""
        lines = [l for l in text.splitlines() if l.strip()]
        return "\n".join(lines[:max_lines])

    def _build_prompt(
        self,
        topic: str,
        artist: str,
        retrieved: RetrievalResult,
        profile: dict | None,
        num_examples: int = 3,
        chunk_line_cap: int | None = None,
    ) -> str:
        """Build a few-shot prompt from retrieved verses + style profile.

        The profile injection acts as an instruction the model can follow via
        in-context learning. Style hints stay grounded — they describe what
        the model is about to see in the examples below.

        `chunk_line_cap` truncates each example to that many lines (used by
        the generator to enforce a context-window budget).
        """
        examples = retrieved.chunks[:num_examples]

        parts = []

        # Profile preamble — only if we have one
        if profile:
            scheme = profile.get("dominant_scheme", "free")
            conf = profile.get("scheme_confidence_named", 0)
            avg_syl = profile.get("avg_syllables", 0)
            sig_words = profile.get("signature_words", [])[:8]

            scheme_label = {
                "AABB": "AABB couplets (lines 1+2 rhyme, 3+4 rhyme)",
                "ABAB": "ABAB alternating rhymes",
                "ABBA": "ABBA enclosed rhymes",
                "AAAA": "AAAA — same rhyme held across multiple bars",
                "free": "free verse",
            }.get(scheme, scheme)

            parts.append(f"{artist} writes in {scheme_label}.")
            parts.append(f"Average bar length: {avg_syl:.0f} syllables.")
            if sig_words:
                parts.append(f"{artist}'s signature vocabulary includes: {', '.join(sig_words)}.")
            parts.append("")

        parts.append(f"Rap verses by {artist}:\n")
        for i, chunk in enumerate(examples, 1):
            parts.append(f"Verse {i}:")
            text = chunk.text.strip()
            if chunk_line_cap:
                text = self._truncate_chunk(text, chunk_line_cap)
            parts.append(text)
            parts.append("")

        # Final header — frame the new verse with the topic and scheme reminder
        if profile and profile.get("dominant_scheme") not in (None, "free", "irregular"):
            scheme_short = profile["dominant_scheme"]
            parts.append(f"Verse {len(examples) + 1} by {artist} ({scheme_short} rhyme scheme) about {topic}:")
        else:
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
        enforce_rhyme: bool = True,
        num_candidates: int = 8,
    ) -> GenerationResult:
        """Generate a verse using retrieval-augmented prompting + style profile.

        When `enforce_rhyme` is True and the artist has a dominant named scheme:
        - If `num_candidates > 1`, generates that many parallel verses in a
          single batched forward pass and re-ranks by scheme-match score.
        - Otherwise attaches a `RhymeBiasLogitsProcessor` for soft per-token
          biasing (kept for the single-candidate fast path).

        Candidate re-ranking is the recommended path: it lets the model
        explore the sample space and we pick the verse whose line endings
        actually match the target scheme. Coherence stays intact because
        no candidate has its logits distorted.
        """
        self._load_model()

        # Look up the artist's style profile
        profile = self.get_profile(artist)
        prefer_scheme = None
        if profile and profile.get("dominant_scheme") not in (None, "free", "irregular"):
            prefer_scheme = profile["dominant_scheme"]

        # Retrieve relevant verses, re-ranked to favor the artist's dominant scheme
        retrieved = self.retriever.retrieve_diverse(
            query=topic,
            artist=artist,
            top_k=num_examples,
            fetch_k=num_examples * 8,
            prefer_scheme=prefer_scheme,
        )

        if not retrieved.chunks:
            # Fall back to no artist filter
            retrieved = self.retriever.retrieve_diverse(
                query=topic,
                artist=None,
                top_k=num_examples,
                fetch_k=num_examples * 4,
            )

        # GPT-2 XL has a 1024-token context window. Reserve room for new tokens
        # plus a small safety margin and shrink the prompt if it overflows.
        model_max = getattr(self._model.config, "n_positions", 1024)
        max_new = num_bars * 20
        budget = model_max - max_new - 16  # safety margin

        chunk_line_cap = None
        prompt = self._build_prompt(topic, artist, retrieved, profile, num_examples, chunk_line_cap)
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
        input_len = input_ids.shape[1]

        # Iteratively shrink: first cap each example to fewer lines, then drop examples
        cap_iter = [12, 8, 6, 4]
        cap_idx = 0
        active_examples = num_examples
        while input_len > budget:
            if cap_idx < len(cap_iter):
                chunk_line_cap = cap_iter[cap_idx]
                cap_idx += 1
            elif active_examples > 1:
                active_examples -= 1
            else:
                # Last resort: cut max_new_tokens too
                max_new = max(max(num_bars * 8, 64), max_new // 2)
                budget = model_max - max_new - 16
                if input_len > budget:
                    # Can't shrink further; truncate prompt from the front
                    excess = input_len - budget
                    input_ids = input_ids[:, excess:]
                    input_len = input_ids.shape[1]
                    break

            prompt = self._build_prompt(
                topic, artist, retrieved, profile, active_examples, chunk_line_cap
            )
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids
            input_len = input_ids.shape[1]

        inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": torch.ones_like(input_ids).to(self.device),
        }

        # Decide between candidate re-ranking and single-candidate logits bias.
        # Re-ranking explores the sample space and picks the best rhymer; the
        # logits processor only nudges per-token. Re-ranking wins on quality
        # but can't run together with the processor (the processor isn't
        # batch-aware).
        use_reranking = enforce_rhyme and prefer_scheme and num_candidates > 1

        logits_processors = None
        if enforce_rhyme and prefer_scheme and not use_reranking:
            from rhymelm.rag.rhyme_processor import RhymeBiasLogitsProcessor
            from rhymelm.data.phonemes import get_cmu_dict, build_rhyme_groups

            if not hasattr(self, "_cmu") or self._cmu is None:
                self._cmu = get_cmu_dict()
            if not hasattr(self, "_rhyme_groups") or self._rhyme_groups is None:
                print("[generator] Building rhyme groups index (one-time, ~5s)...")
                self._rhyme_groups = build_rhyme_groups(self._cmu)

            avg_syl = profile.get("avg_syllables", 10.0) if profile else 10.0
            processor = RhymeBiasLogitsProcessor(
                tokenizer=self._tokenizer,
                scheme_name=prefer_scheme,
                num_bars=num_bars,
                rhyme_groups=self._rhyme_groups,
                cmu=self._cmu,
                prompt_len=input_len,
                avg_syllables=avg_syl,
                boost=4.0,
            )
            logits_processors = [processor]

        # Pre-load CMU dict for re-ranking scoring (cheap if already loaded)
        if use_reranking:
            from rhymelm.data.phonemes import get_cmu_dict
            if not hasattr(self, "_cmu") or self._cmu is None:
                self._cmu = get_cmu_dict()

        n_seq = num_candidates if use_reranking else 1

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=self._tokenizer.eos_token_id,
                logits_processor=logits_processors,
                num_return_sequences=n_seq,
            )

        # Decode each candidate, parse to lines, score by scheme match
        candidate_verses: list[tuple[str, list[str], float]] = []
        for cand_idx in range(output.shape[0]):
            new_tokens = output[cand_idx][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            lines = self._clean_generated_text(text, num_bars)
            verse_text = "\n".join(lines)

            score = 0.0
            if use_reranking:
                score = self._score_verse(lines, prefer_scheme)
            candidate_verses.append((verse_text, lines, score))

        if use_reranking:
            # Pick the highest-scoring candidate; ties broken by line count.
            candidate_verses.sort(key=lambda c: (c[2], len(c[1])), reverse=True)

        verse, _, best_score = candidate_verses[0]

        return GenerationResult(
            verse=verse,
            retrieved_chunks=retrieved.chunks,
            retrieval_scores=retrieved.scores,
            full_prompt=prompt,
            profile=profile,
        )

    def _clean_generated_text(self, text: str, num_bars: int) -> list[str]:
        """Apply the line stop/cleanup rules to a single candidate's decoded text."""
        lines: list[str] = []
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
        return lines

    def _score_verse(self, lines: list[str], scheme: str) -> float:
        """Score a candidate verse by rhyme density + scheme alignment + length.

        Combines three signals so the best candidate is the one that:
          - rhymes the most overall (rewards any rhyming pair within 4 bars)
          - actually matches the target scheme positionally (bonus)
          - is long enough to be a real verse (penalty for early bail)
        """
        if not lines or scheme in (None, "free"):
            return 0.0

        from rhymelm.data.phonemes import get_rhyme_suffix
        from rhymelm.rag.schemes import (
            parse_rhyme_scheme,
            get_line_ending_word,
        )

        suffixes: list[str | None] = []
        for line in lines:
            word = get_line_ending_word(line)
            suffixes.append(get_rhyme_suffix(word, self._cmu) if word else None)

        # ── Signal 1: any-position rhyme density ──
        # Reward every pair of lines (within 4 bars of each other) that share a suffix.
        # This credits the candidate that rhymes the most regardless of strict position.
        any_pair_hits = 0
        any_pair_possible = 0
        for i in range(len(suffixes)):
            for j in range(i + 1, min(i + 5, len(suffixes))):
                if suffixes[i] is None or suffixes[j] is None:
                    continue
                any_pair_possible += 1
                if suffixes[i] == suffixes[j]:
                    any_pair_hits += 1
        any_density = any_pair_hits / any_pair_possible if any_pair_possible else 0.0

        # ── Signal 2: positional scheme alignment bonus ──
        # Extra credit when the rhyme actually lands at the scheme's expected positions.
        groups = parse_rhyme_scheme(scheme, len(lines))
        from collections import defaultdict
        group_to_suffixes: dict[int, list[str | None]] = defaultdict(list)
        for grp, suf in zip(groups, suffixes):
            group_to_suffixes[grp].append(suf)

        positional_hits = 0
        positional_possible = 0
        for grp_suffixes in group_to_suffixes.values():
            if len(grp_suffixes) < 2:
                continue
            anchor = next((s for s in grp_suffixes if s), None)
            if anchor is None:
                continue
            for s in grp_suffixes[1:]:
                positional_possible += 1
                if s == anchor:
                    positional_hits += 1
        positional_density = positional_hits / positional_possible if positional_possible else 0.0

        # ── Signal 3: length ──
        # Penalty for short verses, bonus for hitting the target.
        length_score = min(len(lines) / 8.0, 1.0)

        # Weighted combination — any-pair density dominates because it gives
        # the model partial credit, positional bonus rewards hitting the scheme,
        # length score keeps short verses from winning by accident.
        return (any_density * 1.0) + (positional_density * 0.5) + (length_score * 0.3)
