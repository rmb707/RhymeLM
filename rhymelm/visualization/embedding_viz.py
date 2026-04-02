"""Character embedding space visualization using dimensionality reduction."""

import numpy as np
import matplotlib.pyplot as plt


VOWELS = set("aeiouAEIOU")
CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
DIGITS = set("0123456789")
WHITESPACE = set(" \t\n\r")


def classify_char(c: str) -> str:
    if c in VOWELS:
        return "vowel"
    if c in CONSONANTS:
        return "consonant"
    if c in DIGITS:
        return "digit"
    if c in WHITESPACE:
        return "whitespace"
    return "punctuation"


COLOR_MAP = {
    "vowel": "#e74c3c",
    "consonant": "#3498db",
    "digit": "#2ecc71",
    "whitespace": "#95a5a6",
    "punctuation": "#f39c12",
}


def plot_embedding_space(
    embedding_weights: np.ndarray,
    chars: list[str],
    method: str = "tsne",
    save_path: str | None = None,
):
    """Project character embeddings to 2D and plot, colored by category.

    Args:
        embedding_weights: (vocab_size, embed_dim) numpy array
        chars: list of characters corresponding to each row
        method: 'tsne' or 'umap'
    """
    if method == "umap":
        try:
            from umap import UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(chars) - 1))
        except ImportError:
            print("umap-learn not installed, falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(chars) - 1))

    coords = reducer.fit_transform(embedding_weights)
    categories = [classify_char(c) for c in chars]
    colors = [COLOR_MAP[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

    for i, ch in enumerate(chars):
        if ch.strip():
            ax.annotate(
                repr(ch)[1:-1],
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                ha="center",
                va="bottom",
                alpha=0.8,
            )

    # Legend
    for cat, color in COLOR_MAP.items():
        ax.scatter([], [], c=color, label=cat, s=60)
    ax.legend(loc="upper right", framealpha=0.9)

    ax.set_title(f"Character Embedding Space ({method.upper()})", fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
