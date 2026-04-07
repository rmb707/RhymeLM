"""Training loop with gradient accumulation, AMP, warmup+cosine schedule."""

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rhymelm.config import Config
from rhymelm.models.base import RhymeLMBase
from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.evaluation.metrics import perplexity
from rhymelm.utils import get_param_groups


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def estimate_loss(
    model: RhymeLMBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_batches: int = 100,
) -> dict[str, float]:
    """Estimate train and val loss over multiple batches."""
    model.eval()
    results = {}

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        loader_iter = iter(loader)
        for _ in range(eval_batches):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))
            losses.append(loss.item())
        results[split] = np.mean(losses)

    model.train()
    return results


def save_checkpoint(
    model: RhymeLMBase,
    optimizer,
    scheduler,
    tokenizer,
    config: Config,
    step: int,
    path: str,
):
    """Save model checkpoint with all state needed to resume."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Store tokenizer data based on type
    if hasattr(tokenizer, "merges"):  # BPE
        tok_data = {"type": "bpe", "merges": tokenizer.merges, "vocab": tokenizer.vocab}
    else:  # Char
        tok_data = {"type": "char", "chars": tokenizer.chars}

    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "tokenizer_data": tok_data,
            "tokenizer_chars": getattr(tokenizer, "chars", []),  # backward compat
            "config": {
                "model": vars(config.model),
                "training": vars(config.training),
            },
        },
        path,
    )


def train(
    model: RhymeLMBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharTokenizer,
    config: Config,
    device: torch.device,
    generate_fn=None,
) -> dict:
    """
    Main training loop.

    Features:
    - Gradient accumulation for effective batch size scaling
    - Mixed precision training (AMP) when enabled
    - Linear warmup + cosine decay LR schedule
    - Decoupled weight decay (only on weight matrices)
    - Gradient clipping
    - Periodic evaluation, sampling, and checkpointing
    """
    tc = config.training
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    param_groups = get_param_groups(model, tc.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=tc.learning_rate)
    scheduler = get_lr_scheduler(optimizer, tc.warmup_steps, tc.num_steps)

    use_amp = tc.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
        "grad_norms": [],
    }

    print(f"Training: {tc.num_steps} steps, batch={tc.batch_size}, "
          f"accum={tc.grad_accum_steps}, amp={use_amp}")
    print("=" * 60)

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    for step in range(1, tc.num_steps + 1):
        # Gradient accumulation
        accum_loss = 0.0
        for micro_step in range(tc.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V),
                    y.view(B * T),
                    label_smoothing=tc.label_smoothing,
                )
                loss = loss / tc.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # Evaluation
        if step % tc.eval_interval == 0:
            losses = estimate_loss(model, train_loader, val_loader, device)
            lr = scheduler.get_last_lr()[0]

            history["steps"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["learning_rates"].append(lr)
            history["grad_norms"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            train_ppl = perplexity(losses["train"])
            val_ppl = perplexity(losses["val"])
            print(
                f"Step {step:>6} | "
                f"train: {losses['train']:.4f} (ppl {train_ppl:.1f}) | "
                f"val: {losses['val']:.4f} (ppl {val_ppl:.1f}) | "
                f"lr: {lr:.6f} | grad: {grad_norm:.2f}"
            )
            model.train()

        # Sample generation
        if step % tc.sample_interval == 0 and generate_fn is not None:
            print("\n" + "=" * 40)
            print("SAMPLE VERSE:")
            print("=" * 40)
            print(generate_fn())
            print("=" * 40 + "\n")
            model.train()

        # Checkpoint
        if step % tc.checkpoint_interval == 0:
            ckpt_path = str(checkpoint_dir / f"{config.experiment_name}_step{step}.pt")
            save_checkpoint(model, optimizer, scheduler, tokenizer, config, step, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final checkpoint
    final_path = str(checkpoint_dir / f"{config.experiment_name}_final.pt")
    save_checkpoint(model, optimizer, scheduler, tokenizer, config, tc.num_steps, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    return history
