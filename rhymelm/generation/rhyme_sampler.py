"""Rhyme-aware generation with scheme templates and constrained decoding.

Works with both character-level and BPE tokenizers. Rhyme enforcement
operates on decoded text: after each token, we check if a newline was
produced and whether the line's last word matches the target rhyme group.
"""

import torch
import torch.nn.functional as F

from rhymelm.models.base import RhymeLMBase
from rhymelm.data.phonemes import get_cmu_dict, get_rhyme_suffix, build_rhyme_groups
from rhymelm.generation.sampler import apply_repetition_penalty, sample_next_token


RHYME_SCHEMES = {
    "AABB": "AABBCCDDEEFFFFGGHH",
    "ABAB": "ABABCDCDEFEFEFGHGH",
    "ABBA": "ABBACDCDEFFEGHGHIJ",
    "free": None,
}


def parse_rhyme_scheme(scheme: str, num_bars: int) -> list[int]:
    pattern = RHYME_SCHEMES.get(scheme, scheme)
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
    while len(groups) < num_bars:
        groups.append(groups[-1] + 1)
    return groups[:num_bars]


@torch.no_grad()
def generate_rhyming_verse(
    model: RhymeLMBase,
    tokenizer,
    device: torch.device,
    prompt: str = " ",
    num_bars: int = 16,
    max_chars: int = 2000,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    rhyme_scheme: str = "AABB",
    rhyme_boost: float = 5.0,
    originality_filter=None,
) -> str:
    """Generate a verse with rhyme scheme constraints.

    Tokenizer-agnostic: works with both char-level and BPE by operating
    on decoded text for rhyme logic, and boosting token-level logits for
    tokens that contain newlines or rhyming word fragments.
    """
    model.eval()
    cmu = get_cmu_dict()
    rhyme_groups_map = build_rhyme_groups(cmu)
    groups = parse_rhyme_scheme(rhyme_scheme, num_bars)

    # Encode prompt
    if hasattr(tokenizer, "encode_to_tensor"):
        tokens = tokenizer.encode(prompt)
    else:
        tokens = [tokenizer.stoi.get(c, 0) for c in prompt]

    x = torch.tensor([tokens], dtype=torch.long, device=device)

    is_transformer = hasattr(model, "d_model")
    if is_transformer:
        kv_caches = model.init_state(1, device)
        logits, _ = model(x, kv_caches, offset=0)
    else:
        hidden = model.init_state(1, device)
        output = model(x, hidden)
        logits, hidden = output[0], output[1]

    generated_ids = list(tokens)
    generated_text = prompt
    bar_count = prompt.count("\n")
    current_line = prompt.split("\n")[-1] if "\n" in prompt else prompt

    group_suffixes: dict[int, str] = {}

    # Pre-index: which token IDs decode to strings containing a newline?
    newline_token_ids = set()
    for tid in range(tokenizer.vocab_size):
        decoded = tokenizer.decode([tid])
        if "\n" in decoded:
            newline_token_ids.add(tid)

    while bar_count < num_bars and len(generated_text) < max_chars:
        step_logits = logits[:, -1, :]  # (1, vocab)

        step_logits = apply_repetition_penalty(step_logits, generated_ids, repetition_penalty)

        if originality_filter is not None:
            from rhymelm.generation.originality import apply_originality_penalty
            step_logits = apply_originality_penalty(
                step_logits, generated_text, originality_filter, tokenizer,
            )

        # ── Rhyme boosting ──
        if bar_count < len(groups):
            current_group = groups[bar_count]
            target_suffix = group_suffixes.get(current_group)

            if target_suffix and len(current_line) > 15:
                words = current_line.split()
                current_word = words[-1].lower().strip(".,!?;:'\"()-") if words else ""

                if current_word:
                    # Check if current word already rhymes with target
                    current_suffix = get_rhyme_suffix(current_word, cmu)
                    if current_suffix == target_suffix:
                        # Boost all newline-containing tokens
                        for nl_tid in newline_token_ids:
                            step_logits[0, nl_tid] += rhyme_boost

                    # Boost tokens that continue toward rhyming words
                    rhyming_words = rhyme_groups_map.get(target_suffix, [])
                    for rw in rhyming_words[:30]:
                        if rw.startswith(current_word) and len(rw) > len(current_word):
                            # Find tokens that would continue this word
                            continuation = rw[len(current_word):]
                            for tid in range(tokenizer.vocab_size):
                                tok_str = tokenizer.decode([tid])
                                if continuation.startswith(tok_str) or tok_str.startswith(continuation):
                                    step_logits[0, tid] += rhyme_boost * 0.4
                                    break  # only boost first matching token per word

        next_token = sample_next_token(step_logits, temperature, top_k, top_p)
        token_id = next_token.item()
        generated_ids.append(token_id)

        decoded = tokenizer.decode([token_id])
        generated_text += decoded

        if "\n" in decoded:
            # Line completed — record rhyme suffix
            words = current_line.split()
            if words and bar_count < len(groups):
                suffix = get_rhyme_suffix(words[-1], cmu)
                grp = groups[bar_count]
                if suffix and grp not in group_suffixes:
                    group_suffixes[grp] = suffix
            bar_count += decoded.count("\n")
            # Start tracking new line (text after last newline)
            current_line = decoded.split("\n")[-1]
        else:
            current_line += decoded

        # Next step
        if is_transformer:
            offset = min(len(generated_ids) - 1, model.max_seq_len - 1)
            logits, _ = model(next_token.view(1, 1), kv_caches, offset=offset)
        else:
            output = model(next_token.view(1, 1), hidden)
            logits, hidden = output[0], output[1]

    model.train()
    return generated_text
