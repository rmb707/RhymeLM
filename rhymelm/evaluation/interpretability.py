"""Mechanistic interpretability tools: probing classifiers, neuron analysis, attention analysis.

These tools investigate what the model has learned internally — whether
its hidden states encode features like word boundaries, syllable counts,
and phonemes, even though it was only trained to predict the next character.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.data.phonemes import get_cmu_dict, count_syllables, get_rhyme_suffix


class LinearProbe(nn.Module):
    """A linear classifier trained on frozen hidden states.

    If a simple linear probe achieves high accuracy predicting some feature
    (word boundary, syllable count, phoneme), it proves the model has learned
    a linearly separable representation of that feature — an emergent internal
    representation that wasn't explicitly trained.
    """

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


@torch.no_grad()
def extract_hidden_states(
    model, tokenizer: CharTokenizer, text: str, device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Extract hidden states for each character position in text."""
    model.eval()
    tokens = [tokenizer.stoi.get(c, 0) for c in text]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    hidden = model.init_state(1, device)

    emb = model.dropout(model.embed(x))
    lstm_out, _ = model.lstm(emb, hidden)

    return lstm_out.squeeze(0).cpu(), list(text)


def build_word_boundary_labels(text: str) -> list[int]:
    """Label each position: 1 if it's a word boundary (space/newline follows), 0 otherwise."""
    labels = []
    for i, ch in enumerate(text):
        if ch in (" ", "\n", "\t"):
            labels.append(1)
        else:
            labels.append(0)
    return labels


def build_syllable_labels(text: str, cmu: dict | None = None) -> list[int]:
    """Label each position with the syllable count of its containing word.

    Capped at 6 for classification (0=boundary, 1-5=syllables, 6=6+).
    """
    if cmu is None:
        cmu = get_cmu_dict()
    labels = []
    words = []
    current = []

    for ch in text:
        if ch in (" ", "\n", "\t"):
            if current:
                word = "".join(current)
                words.append((word, len(current)))
                current = []
            words.append((ch, 1))
        else:
            current.append(ch)
    if current:
        words.append(("".join(current), len(current)))

    for word, length in words:
        if word.strip():
            syl = min(count_syllables(word, cmu), 6)
            labels.extend([syl] * length)
        else:
            labels.extend([0] * length)

    return labels


def train_probe(
    hidden_states: torch.Tensor,
    labels: list[int],
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-3,
) -> tuple[LinearProbe, float]:
    """Train a linear probe and return it with its accuracy.

    Uses 80/20 train/test split for honest evaluation.
    """
    labels_t = torch.tensor(labels, dtype=torch.long)
    assert len(hidden_states) == len(labels_t)

    n = len(labels_t)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    train_h, train_l = hidden_states[train_idx], labels_t[train_idx]
    test_h, test_l = hidden_states[test_idx], labels_t[test_idx]

    probe = LinearProbe(hidden_states.shape[1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    ds = TensorDataset(train_h, train_l)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    probe.train()
    for epoch in range(epochs):
        for batch_h, batch_l in loader:
            logits = probe(batch_h)
            loss = F.cross_entropy(logits, batch_l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(test_h).argmax(dim=-1)
        accuracy = (preds == test_l).float().mean().item()

    return probe, accuracy


@torch.no_grad()
def analyze_lstm_neurons(
    model, tokenizer: CharTokenizer, text: str, device: torch.device,
    top_k: int = 10,
) -> dict[str, list[tuple[int, float]]]:
    """Find LSTM hidden units that activate for specific character categories.

    Returns the top-k neurons most selective for each category:
    vowels, consonants, newlines, spaces, punctuation.
    """
    model.eval()
    hidden_states, chars = extract_hidden_states(model, tokenizer, text, device)
    H = hidden_states.numpy()

    categories = {
        "vowel": set("aeiouAEIOU"),
        "consonant": set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"),
        "newline": {"\n"},
        "space": {" "},
        "digit": set("0123456789"),
    }

    results = {}
    for cat_name, cat_chars in categories.items():
        mask = np.array([c in cat_chars for c in chars])
        if mask.sum() == 0:
            continue

        # For each neuron, compute mean activation on category vs. off category
        on_mean = H[mask].mean(axis=0)
        off_mean = H[~mask].mean(axis=0)
        selectivity = on_mean - off_mean

        top_neurons = np.argsort(-np.abs(selectivity))[:top_k]
        results[cat_name] = [
            (int(n), float(selectivity[n])) for n in top_neurons
        ]

    return results


@torch.no_grad()
def analyze_attention_heads(
    attn_weights: torch.Tensor, chars: list[str],
) -> dict[str, list[tuple[int, float]]]:
    """Classify attention heads by what they specialize in.

    Looks for:
    - Local heads: attend primarily to nearby positions
    - Newline heads: attend heavily to newline characters
    - Distance heads: attend to positions far away (potential rhyme heads)
    """
    # attn_weights: (n_heads, T, T)
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.squeeze(0)
    n_heads, T, _ = attn_weights.shape
    attn = attn_weights.cpu().numpy()

    newline_positions = [i for i, c in enumerate(chars) if c == "\n"]

    results = {}
    head_scores = []
    for h in range(n_heads):
        w = attn[h]  # (T, T)

        # Locality score: average distance of attention
        positions = np.arange(T)
        avg_distances = []
        for t in range(T):
            if w[t].sum() > 0:
                expected_pos = (w[t] * positions).sum()
                avg_distances.append(abs(expected_pos - t))
        locality = np.mean(avg_distances) if avg_distances else 0

        # Newline attention score
        if newline_positions:
            nl_attention = w[:, newline_positions].mean()
        else:
            nl_attention = 0

        head_scores.append({
            "head": h,
            "locality": float(locality),
            "newline_attention": float(nl_attention),
        })

    # Classify
    head_scores.sort(key=lambda x: x["locality"])
    results["local_heads"] = [(s["head"], s["locality"]) for s in head_scores[:3]]
    results["distance_heads"] = [(s["head"], s["locality"]) for s in head_scores[-3:]]

    head_scores.sort(key=lambda x: -x["newline_attention"])
    results["newline_heads"] = [(s["head"], s["newline_attention"]) for s in head_scores[:3]]

    return results
