"""Fine-tuning pipeline for user-uploaded lyrics.

Supports both full fine-tuning and LoRA (parameter-efficient).
Uses mixed batching (user data + original corpus) to prevent
catastrophic forgetting. Includes early stopping.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rhymelm.config import FinetuneConfig
from rhymelm.data.dataset import LMDataset
from rhymelm.data.corpus import clean_user_lyrics
from rhymelm.training.lora import apply_lora, merge_lora, get_lora_params
from rhymelm.training.trainer import get_lr_scheduler


def prepare_user_data(
    text: str,
    tokenizer,
    block_size: int,
    batch_size: int = 8,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    """Clean, tokenize, and create DataLoaders from user-uploaded lyrics."""
    cleaned = clean_user_lyrics(text)
    lines = [l for l in cleaned.splitlines() if l.strip()]
    print(f"User data: {len(lines)} lines, {len(cleaned):,} chars")

    if hasattr(tokenizer, "encode_to_tensor"):
        encoded = tokenizer.encode_to_tensor(cleaned)
    else:
        encoded = tokenizer.encode(cleaned)

    split = int(len(encoded) * (1 - val_split))
    train_ds = LMDataset(encoded[:split], block_size)
    val_ds = LMDataset(encoded[split:], block_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size // 2), shuffle=False, drop_last=True)

    print(f"User train: {len(train_ds)} samples | val: {len(val_ds)} samples")
    return train_loader, val_loader


@torch.no_grad()
def _eval_loss(model, loader, device, max_batches=50):
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(min(max_batches, len(loader))):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        B, T, V = logits.shape
        losses.append(F.cross_entropy(logits.view(B * T, V), y.view(B * T)).item())
    model.train()
    return np.mean(losses) if losses else float("inf")


def finetune(
    model,
    user_train_loader: DataLoader,
    user_val_loader: DataLoader,
    base_train_loader: DataLoader | None,
    config: FinetuneConfig,
    device: torch.device,
    generate_fn=None,
) -> dict:
    """Fine-tune a pretrained model on user data.

    Returns training history dict. Yields (step, loss, val_loss) if used as generator.
    """
    # Apply LoRA if configured
    if config.use_lora:
        model = apply_lora(model, rank=config.lora_rank)
        params = get_lora_params(model)
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.01)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, total_steps=config.num_steps)

    history = {"steps": [], "train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0

    model.train()
    user_iter = iter(user_train_loader)
    base_iter = iter(base_train_loader) if base_train_loader else None

    print(f"Fine-tuning: {config.num_steps} steps, lr={config.learning_rate}, "
          f"{'LoRA rank=' + str(config.lora_rank) if config.use_lora else 'full'}")

    for step in range(1, config.num_steps + 1):
        # Mixed batching: user data with probability mix_ratio, base data otherwise
        use_user = (base_iter is None) or (torch.rand(1).item() < config.mix_ratio)

        if use_user:
            try:
                x, y = next(user_iter)
            except StopIteration:
                user_iter = iter(user_train_loader)
                x, y = next(user_iter)
        else:
            try:
                batch = next(base_iter)
            except StopIteration:
                base_iter = iter(base_train_loader)
                batch = next(base_iter)
            x, y = batch[0], batch[1]

        x, y = x.to(device), y.to(device)

        output = model(x)
        logits = output[0] if isinstance(output, tuple) else output
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Evaluation
        if step % config.eval_interval == 0:
            val_loss = _eval_loss(model, user_val_loader, device)
            history["steps"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)

            print(f"  Step {step:>5} | train: {loss.item():.4f} | val: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"  Early stopping at step {step} (patience={config.early_stopping_patience})")
                    break

            model.train()

    # Merge LoRA if used
    if config.use_lora:
        model = merge_lora(model)

    print(f"Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    return history


def save_finetune_checkpoint(model, tokenizer, config, user_name: str, base_dir: str = "checkpoints/finetune"):
    """Save a user's fine-tuned model."""
    path = Path(base_dir) / user_name
    path.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "tokenizer_type": "bpe" if hasattr(tokenizer, "merges") else "char",
        "tokenizer_data": {
            "merges": tokenizer.merges,
            "vocab": tokenizer.vocab,
        } if hasattr(tokenizer, "merges") else {
            "chars": tokenizer.chars,
        },
        "user_name": user_name,
    }, str(path / "model.pt"))

    print(f"Saved fine-tuned model to {path}/model.pt")
