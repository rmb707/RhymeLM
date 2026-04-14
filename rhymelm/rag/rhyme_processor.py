"""HuggingFace LogitsProcessor that biases generation toward a target rhyme scheme.

State machine: as tokens stream out, we track which line we're building, what
group it belongs to in the scheme, and whether a target rhyme suffix has been
established for that group. When we're near the end of a line that needs to
rhyme with an earlier one, we additively boost logits for tokens that complete
a word matching the target suffix, plus boost newline tokens once we have a
rhyming word in hand.

Crucially this is *additive bias*, not a hard constraint — the model is still
free to produce sensible English. We're just nudging.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from rhymelm.data.phonemes import get_rhyme_suffix
from rhymelm.rag.schemes import get_line_ending_word, parse_rhyme_scheme


class RhymeBiasLogitsProcessor(LogitsProcessor):
    """Additively bias logits toward tokens that produce rhyming line endings."""

    def __init__(
        self,
        tokenizer,
        scheme_name: str,
        num_bars: int,
        rhyme_groups: dict[str, list[str]],
        cmu: dict,
        prompt_len: int,
        avg_syllables: float = 10.0,
        boost: float = 5.0,
    ):
        self.tokenizer = tokenizer
        self.cmu = cmu
        self.rhyme_groups = rhyme_groups
        self.prompt_len = prompt_len
        self.boost = boost
        self.avg_syllables = max(avg_syllables, 6.0)

        # Per-line group assignments, e.g. AABB(8) -> [0,0,1,1,2,2,3,3]
        self.scheme_groups = parse_rhyme_scheme(scheme_name, num_bars)
        self.num_bars = num_bars

        # group_id -> established rhyme suffix (set when first line of group completes)
        self.group_suffixes: dict[int, str] = {}

        # Cached: target_suffix -> set of token IDs that complete a word in that rhyme group
        self._completion_token_cache: dict[str, set[int]] = {}

        # Pre-compute newline-containing token IDs (one-time, small set)
        self._newline_token_ids = self._build_newline_tokens()

    def _build_newline_tokens(self) -> set[int]:
        """Token IDs whose decoded text contains a newline."""
        ids: set[int] = set()
        for tid in range(self.tokenizer.vocab_size):
            try:
                txt = self.tokenizer.decode([tid])
            except Exception:
                continue
            if "\n" in txt:
                ids.add(tid)
        return ids

    def _completion_tokens_for_suffix(self, suffix: str) -> set[int]:
        """For a target rhyme suffix, return token IDs that end a word in that group.

        We tokenize each rhyming word and take its LAST token — that's the token
        the model would emit as the final piece of the rhyming word.
        """
        if suffix in self._completion_token_cache:
            return self._completion_token_cache[suffix]

        good: set[int] = set()
        words = self.rhyme_groups.get(suffix, [])
        # Cap to avoid extreme edge cases
        for word in words[:500]:
            # Encode with a leading space (GPT-2 BPE convention for word-internal tokens)
            for variant in (" " + word, word):
                try:
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                except Exception:
                    continue
                if ids:
                    good.add(ids[-1])
        self._completion_token_cache[suffix] = good
        return good

    def _decode_generated(self, input_ids: torch.Tensor) -> str:
        """Decode the generated portion (everything past the prompt)."""
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() == 0:
            return ""
        return self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        text = self._decode_generated(input_ids)
        if not text:
            return scores

        # Split into completed lines + current line buffer
        parts = text.split("\n")
        completed_lines = parts[:-1]
        current_line = parts[-1]

        line_idx = len(completed_lines)
        if line_idx >= self.num_bars or line_idx >= len(self.scheme_groups):
            return scores

        # Record rhyme suffixes for each completed line's group as we go
        for i, line in enumerate(completed_lines):
            if i >= len(self.scheme_groups):
                break
            grp = self.scheme_groups[i]
            if grp in self.group_suffixes:
                continue
            ending_word = get_line_ending_word(line)
            if not ending_word:
                continue
            suffix = get_rhyme_suffix(ending_word, self.cmu)
            if suffix:
                self.group_suffixes[grp] = suffix

        # What group does the current line belong to?
        current_group = self.scheme_groups[line_idx]
        target_suffix = self.group_suffixes.get(current_group)
        if not target_suffix:
            # No established target yet (this is the first line of the group)
            return scores

        # Most rap bars have ~5–9 words; only fire near the natural endpoint
        # so we don't destabilize sentence structure earlier on.
        word_count = len(current_line.split())
        if word_count < 4:
            return scores

        # If the current line already ends in a rhyming word, commit immediately
        last_word = get_line_ending_word(current_line)
        if last_word:
            last_suffix = get_rhyme_suffix(last_word, self.cmu)
            if last_suffix == target_suffix:
                idx_tensor = torch.tensor(
                    list(self._newline_token_ids),
                    device=scores.device,
                    dtype=torch.long,
                )
                scores[0, idx_tensor] = scores[0, idx_tensor] + (self.boost * 1.6)
                return scores

        # Otherwise gently bias toward tokens that complete a rhyming word.
        # Modest boost so coherence wins ties — we'd rather have grammatical
        # English than a forced rhyme that breaks the bar.
        completion_tokens = self._completion_tokens_for_suffix(target_suffix)
        if not completion_tokens:
            return scores

        idx_tensor = torch.tensor(
            list(completion_tokens), device=scores.device, dtype=torch.long
        )
        scores[0, idx_tensor] = scores[0, idx_tensor] + self.boost
        return scores
