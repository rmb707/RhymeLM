"""Training dynamics visualization: loss, perplexity, gradient norms, LR."""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_dashboard(
    steps: list[int],
    train_losses: list[float],
    val_losses: list[float],
    learning_rates: list[float] | None = None,
    grad_norms: list[float] | None = None,
    save_path: str | None = None,
):
    """Plot a multi-panel training dynamics dashboard."""
    num_panels = 2
    if learning_rates:
        num_panels += 1
    if grad_norms:
        num_panels += 1

    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 4))
    if num_panels == 1:
        axes = [axes]

    panel = 0

    # Loss curves
    axes[panel].plot(steps, train_losses, label="Train", alpha=0.8)
    axes[panel].plot(steps, val_losses, label="Val", alpha=0.8)
    axes[panel].set_xlabel("Step")
    axes[panel].set_ylabel("Loss")
    axes[panel].set_title("Cross-Entropy Loss")
    axes[panel].legend()
    axes[panel].grid(True, alpha=0.3)
    panel += 1

    # Perplexity
    train_ppl = [np.exp(l) for l in train_losses]
    val_ppl = [np.exp(l) for l in val_losses]
    axes[panel].plot(steps, train_ppl, label="Train", alpha=0.8)
    axes[panel].plot(steps, val_ppl, label="Val", alpha=0.8)
    axes[panel].set_xlabel("Step")
    axes[panel].set_ylabel("Perplexity")
    axes[panel].set_title("Perplexity")
    axes[panel].legend()
    axes[panel].grid(True, alpha=0.3)
    panel += 1

    if learning_rates:
        axes[panel].plot(steps[: len(learning_rates)], learning_rates, color="green")
        axes[panel].set_xlabel("Step")
        axes[panel].set_ylabel("Learning Rate")
        axes[panel].set_title("LR Schedule")
        axes[panel].grid(True, alpha=0.3)
        panel += 1

    if grad_norms:
        axes[panel].plot(steps[: len(grad_norms)], grad_norms, color="orange", alpha=0.6)
        axes[panel].set_xlabel("Step")
        axes[panel].set_ylabel("Gradient Norm")
        axes[panel].set_title("Gradient Norms")
        axes[panel].grid(True, alpha=0.3)

    fig.suptitle("RhymeLM Training Dynamics", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
