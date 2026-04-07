"""Sampling strategies for autoregressive text generation.

Works with both character-level and BPE tokenizers. The key difference:
with BPE, each token may decode to multiple characters, so bar counting
and originality checking operate on the decoded text.
"""

import torch
import torch.nn.functional as F

from rhymelm.models.base import RhymeLMBase


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float = 1.2
) -> torch.Tensor:
    """Penalize tokens that have appeared recently in the generated sequence."""
    if penalty == 1.0 or not generated_ids:
        return logits
    recent = set(generated_ids[-200:])
    for token_id in recent:
        if token_id < logits.size(-1):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Sample next token with temperature, optional top-k and top-p filtering."""
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_verse(
    model: RhymeLMBase,
    tokenizer,
    device: torch.device,
    prompt: str = " ",
    num_bars: int = 16,
    max_chars: int = 2000,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    originality_filter=None,
    artist_id: int | None = None,
) -> str:
    """Generate a verse with the specified number of bars (lines).

    Works with both character-level and BPE tokenizers.
    """
    model.eval()

    # Encode prompt
    if hasattr(tokenizer, "encode_to_tensor"):
        tokens = tokenizer.encode(prompt)
    else:
        tokens = [tokenizer.stoi.get(c, 0) for c in prompt]

    x = torch.tensor([tokens], dtype=torch.long, device=device)

    # Handle different model types
    is_transformer = hasattr(model, "d_model")
    if is_transformer:
        kv_caches = model.init_state(1, device)
        artist_ids = torch.tensor([artist_id or 0], device=device) if hasattr(model, "num_artists") and model.num_artists > 0 else None
    else:
        hidden = model.init_state(1, device)

    generated_ids = list(tokens)
    generated_text = prompt
    bar_count = prompt.count("\n")

    # Prefill for transformer
    if is_transformer:
        logits, _ = model(x, kv_caches, offset=0, artist_ids=artist_ids)
    else:
        output = model(x, hidden)
        logits, hidden = output[0], output[1]

    while bar_count < num_bars and len(generated_text) < max_chars:
        step_logits = logits[:, -1, :]  # (1, vocab)

        step_logits = apply_repetition_penalty(step_logits, generated_ids, repetition_penalty)

        if originality_filter is not None:
            from rhymelm.generation.originality import apply_originality_penalty
            step_logits = apply_originality_penalty(
                step_logits, generated_text, originality_filter, tokenizer,
            )

        next_token = sample_next_token(step_logits, temperature, top_k, top_p)
        token_id = next_token.item()
        generated_ids.append(token_id)

        # Decode token to text
        decoded = tokenizer.decode([token_id])
        generated_text += decoded
        bar_count += decoded.count("\n")

        # Next step
        if is_transformer:
            offset = min(len(generated_ids) - 1, model.max_seq_len - 1)
            logits, _ = model(next_token.view(1, 1), kv_caches, offset=offset, artist_ids=artist_ids)
        else:
            output = model(next_token.view(1, 1), hidden)
            logits, hidden = output[0], output[1]

    model.train()
    return generated_text
