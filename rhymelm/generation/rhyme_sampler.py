"""Rhyme-aware generation with scheme templates and constrained decoding.

Supports rhyme schemes (AABB, ABAB, ABBA, free) by boosting logits for
characters that form words rhyming with target line endings. This is a
form of constrained decoding — imposing structured constraints on
autoregressive generation without retraining the model.
"""

import torch
import torch.nn.functional as F

from rhymelm.models.base import RhymeLMBase
from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.data.phonemes import get_cmu_dict, get_rhyme_suffix, build_rhyme_groups
from rhymelm.generation.sampler import apply_repetition_penalty, sample_next_token


RHYME_SCHEMES = {
    "AABB": "AABBCCDDEEFFFFGGHH",
    "ABAB": "ABABCDCDEFEFEFGHGH",
    "ABBA": "ABBACDCDEFFEGHGHIJ",
    "AABB8": "AABBCCDD",
    "free": None,
}


def parse_rhyme_scheme(scheme: str, num_bars: int) -> list[int]:
    """Convert rhyme scheme string to group indices.

    'AABB' → [0, 0, 1, 1, 2, 2, 3, 3, ...] extended to num_bars.
    Lines with the same index should rhyme with each other.
    """
    if scheme in RHYME_SCHEMES:
        pattern = RHYME_SCHEMES[scheme]
    else:
        pattern = scheme

    if pattern is None:
        return list(range(num_bars))

    label_map = {}
    groups = []
    counter = 0
    for ch in pattern:
        if ch not in label_map:
            label_map[ch] = counter
            counter += 1
        groups.append(label_map[ch])

    # Extend pattern if num_bars exceeds pattern length
    while len(groups) < num_bars:
        groups.append(groups[-1] + 1)

    return groups[:num_bars]


@torch.no_grad()
def generate_rhyming_verse(
    model: RhymeLMBase,
    tokenizer: CharTokenizer,
    device: torch.device,
    prompt: str = " ",
    num_bars: int = 16,
    max_chars: int = 2000,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    rhyme_scheme: str = "AABB",
    rhyme_boost: float = 3.0,
    originality_filter=None,
) -> str:
    """Generate a verse with rhyme scheme constraints.

    At line endings, boosts logits for characters that continue toward
    words rhyming with the target line's rhyme group.
    """
    model.eval()
    cmu = get_cmu_dict()
    rhyme_groups_map = build_rhyme_groups(cmu)
    groups = parse_rhyme_scheme(rhyme_scheme, num_bars)

    tokens = [tokenizer.stoi.get(c, 0) for c in prompt]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    hidden = model.init_state(1, device)

    generated = list(prompt)
    generated_ids = list(tokens)
    bar_count = 0
    current_line = ""

    # Track rhyme suffixes per group
    group_suffixes: dict[int, str] = {}

    while bar_count < num_bars and len(generated) < max_chars:
        output = model(x, hidden)
        logits, hidden = output[0], output[1]
        # (1, seq_len, vocab) -> (1, vocab)
        logits = logits[:, -1, :]

        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

        # Block verbatim reproduction of training data
        if originality_filter is not None:
            from rhymelm.generation.originality import apply_originality_penalty
            logits = apply_originality_penalty(
                logits, "".join(generated), originality_filter, tokenizer,
            )

        # Rhyme boosting near line endings
        if bar_count < len(groups):
            current_group = groups[bar_count]
            target_suffix = group_suffixes.get(current_group)

            if target_suffix and len(current_line) > 20:
                current_word = current_line.split()[-1] if current_line.split() else ""
                if current_word:
                    rhyming_words = rhyme_groups_map.get(target_suffix, [])
                    current_suffix = get_rhyme_suffix(current_word, cmu)
                    if current_suffix == target_suffix:
                        newline_id = tokenizer.stoi.get("\n")
                        if newline_id is not None:
                            logits[0, newline_id] += rhyme_boost

                    for rw in rhyming_words[:50]:
                        if rw.startswith(current_word.lower()) and len(rw) > len(current_word):
                            next_ch = rw[len(current_word)]
                            ch_id = tokenizer.stoi.get(next_ch)
                            if ch_id is not None:
                                logits[0, ch_id] += rhyme_boost * 0.5

        next_token = sample_next_token(logits, temperature, top_k, top_p)
        token_id = next_token.item()
        next_char = tokenizer.itos[token_id]
        generated.append(next_char)
        generated_ids.append(token_id)

        if next_char == "\n":
            # Record the rhyme suffix of this line's last word
            words = current_line.split()
            if words and bar_count < len(groups):
                suffix = get_rhyme_suffix(words[-1], cmu)
                grp = groups[bar_count]
                if suffix and grp not in group_suffixes:
                    group_suffixes[grp] = suffix
            bar_count += 1
            current_line = ""
        else:
            current_line += next_char

        x = next_token.view(1, 1)

    model.train()
    return "".join(generated)
