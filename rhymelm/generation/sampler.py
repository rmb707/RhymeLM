"""Sampling strategies for autoregressive text generation."""

import torch
import torch.nn.functional as F

from rhymelm.models.base import RhymeLMBase
from rhymelm.data.tokenizer import CharTokenizer


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float = 1.2
) -> torch.Tensor:
    """Penalize tokens that have appeared recently in the generated sequence."""
    if penalty == 1.0 or not generated_ids:
        return logits
    recent = set(generated_ids[-200:])
    for token_id in recent:
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
    tokenizer: CharTokenizer,
    device: torch.device,
    prompt: str = " ",
    num_bars: int = 16,
    max_chars: int = 2000,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate a verse with the specified number of bars (lines)."""
    model.eval()

    tokens = [tokenizer.stoi.get(c, 0) for c in prompt]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    hidden = model.init_state(1, device)

    generated = list(prompt)
    generated_ids = list(tokens)
    bar_count = 0

    while bar_count < num_bars and len(generated) < max_chars:
        output = model(x, hidden)
        logits, hidden = output[0], output[1]
        # logits: (1, seq_len, vocab) -> take last timestep -> (1, vocab)
        logits = logits[:, -1, :]

        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
        next_token = sample_next_token(logits, temperature, top_k, top_p)

        token_id = next_token.item()
        next_char = tokenizer.itos[token_id]
        generated.append(next_char)
        generated_ids.append(token_id)

        if next_char == "\n":
            bar_count += 1

        x = next_token.view(1, 1)

    model.train()
    return "".join(generated)
