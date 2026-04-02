"""Attention weight visualization for hybrid and transformer models."""

import numpy as np
import matplotlib.pyplot as plt


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    chars: list[str],
    head: int = 0,
    title: str = "Attention Weights",
    max_len: int = 100,
    save_path: str | None = None,
):
    """Plot attention heatmap for a single head.

    Args:
        attn_weights: (n_heads, seq_len, seq_len) or (seq_len, seq_len)
        chars: list of characters at each position
        head: which head to plot (if multi-head)
    """
    if attn_weights.ndim == 3:
        attn = attn_weights[head]
    else:
        attn = attn_weights

    # Truncate for readability
    n = min(len(chars), max_len)
    attn = attn[:n, :n]
    labels = [repr(c)[1:-1] if c.strip() else "·" for c in chars[:n]]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")

    # Labels every 5th position
    tick_step = max(1, n // 30)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)

    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(f"{title} (head {head})")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_multi_head_summary(
    attn_weights: np.ndarray,
    chars: list[str],
    n_heads: int | None = None,
    save_path: str | None = None,
):
    """Plot a grid of attention patterns across all heads."""
    if attn_weights.ndim == 2:
        attn_weights = attn_weights[np.newaxis]

    if n_heads is None:
        n_heads = attn_weights.shape[0]

    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    n = min(len(chars), 80)
    for h in range(n_heads):
        r, c = divmod(h, cols)
        ax = axes[r, c]
        ax.imshow(attn_weights[h, :n, :n], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for h in range(n_heads, rows * cols):
        r, c = divmod(h, cols)
        axes[r, c].axis("off")

    fig.suptitle("Multi-Head Attention Patterns", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
