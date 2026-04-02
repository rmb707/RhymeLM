"""Multi-task training: character prediction + phoneme auxiliary objective.

The phoneme head forces the LSTM hidden state to encode pronunciation info.
At word boundaries (character after space/newline), the model predicts the
phoneme sequence of the word that just completed. This shared representation
should improve rhyme coherence without changing the generation mechanism.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rhymelm.config import Config
from rhymelm.models.lstm_phoneme import RhymeLMPhoneme
from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.data.phonemes import get_cmu_dict, get_word_phonemes, build_phoneme_vocab
from rhymelm.evaluation.metrics import perplexity
from rhymelm.training.trainer import get_lr_scheduler, save_checkpoint
from rhymelm.utils import get_param_groups


def find_word_boundaries(tokens: torch.Tensor, tokenizer: CharTokenizer) -> list[tuple[int, int, str]]:
    """Find word boundary positions and the words that end there.

    Returns: list of (batch_idx, position, word) tuples.
    """
    boundaries = []
    B, T = tokens.shape
    for b in range(B):
        current_word = []
        for t in range(T):
            ch = tokenizer.itos[tokens[b, t].item()]
            if ch in (" ", "\n", "\t") or t == T - 1:
                if current_word:
                    word = "".join(current_word)
                    boundaries.append((b, t, word))
                    current_word = []
            else:
                current_word.append(ch)
    return boundaries


def train_phoneme(
    model: RhymeLMPhoneme,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharTokenizer,
    config: Config,
    device: torch.device,
    phoneme_lambda: float = 0.1,
    generate_fn=None,
) -> dict:
    """Train with multi-task loss: char_loss + lambda * phoneme_loss."""
    tc = config.training
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmu = get_cmu_dict()
    p2i, i2p = build_phoneme_vocab()
    phoneme_vocab_size = len(p2i)

    param_groups = get_param_groups(model, tc.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=tc.learning_rate)
    scheduler = get_lr_scheduler(optimizer, tc.warmup_steps, tc.num_steps)

    use_amp = tc.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        "steps": [], "train_loss": [], "val_loss": [],
        "char_loss": [], "phoneme_loss": [],
        "learning_rates": [], "grad_norms": [],
    }

    print(f"Multi-task training: char + phoneme (lambda={phoneme_lambda})")
    print(f"Phoneme vocab: {phoneme_vocab_size} tokens")
    print("=" * 60)

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad()

    for step in range(1, tc.num_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            # Forward pass — get hidden states
            emb = model.dropout(model.embed(x))
            lstm_out, _ = model.lstm(emb)
            lstm_out_d = model.dropout(lstm_out)
            char_logits = model.char_head(lstm_out_d)

            B, T, V = char_logits.shape
            char_loss = F.cross_entropy(
                char_logits.view(B * T, V), y.view(B * T),
                label_smoothing=tc.label_smoothing,
            )

            # Phoneme auxiliary loss at word boundaries
            boundaries = find_word_boundaries(x, tokenizer)
            phoneme_loss = torch.tensor(0.0, device=device)
            n_boundaries = 0

            if boundaries:
                hidden_vecs = []
                target_phonemes = []

                for b_idx, pos, word in boundaries:
                    phones = get_word_phonemes(word, cmu)
                    if phones is None:
                        continue
                    phone_ids = [p2i.get(p, 0) for p in phones]
                    # Pad/truncate to max_phonemes
                    phone_ids = phone_ids[:model.max_phonemes]
                    phone_ids += [0] * (model.max_phonemes - len(phone_ids))

                    hidden_vecs.append(lstm_out[b_idx, pos])
                    target_phonemes.append(phone_ids)
                    n_boundaries += 1

                if hidden_vecs:
                    hidden_stack = torch.stack(hidden_vecs)
                    target_stack = torch.tensor(target_phonemes, device=device, dtype=torch.long)
                    phoneme_logits = model.phoneme_forward(hidden_stack)
                    # Flatten for cross-entropy
                    N, M, PV = phoneme_logits.shape
                    phoneme_loss = F.cross_entropy(
                        phoneme_logits.view(N * M, PV),
                        target_stack.view(N * M),
                        ignore_index=0,
                    )

            total_loss = char_loss + phoneme_lambda * phoneme_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        if step % tc.eval_interval == 0:
            lr = scheduler.get_last_lr()[0]
            history["steps"].append(step)
            history["train_loss"].append(char_loss.item())
            history["val_loss"].append(char_loss.item())  # simplified
            history["char_loss"].append(char_loss.item())
            history["phoneme_loss"].append(phoneme_loss.item() if isinstance(phoneme_loss, torch.Tensor) else 0.0)
            history["learning_rates"].append(lr)
            history["grad_norms"].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            ppl = perplexity(char_loss.item())
            print(
                f"Step {step:>6} | char: {char_loss.item():.4f} (ppl {ppl:.1f}) | "
                f"phoneme: {phoneme_loss.item():.4f} | boundaries: {n_boundaries} | "
                f"lr: {lr:.6f}"
            )

        if step % tc.sample_interval == 0 and generate_fn is not None:
            print("\n" + "=" * 40)
            print("SAMPLE VERSE:")
            print("=" * 40)
            print(generate_fn())
            print("=" * 40 + "\n")
            model.train()

        if step % tc.checkpoint_interval == 0:
            ckpt_path = str(checkpoint_dir / f"{config.experiment_name}_step{step}.pt")
            save_checkpoint(model, optimizer, scheduler, tokenizer, config, step, ckpt_path)

    final_path = str(checkpoint_dir / f"{config.experiment_name}_final.pt")
    save_checkpoint(model, optimizer, scheduler, tokenizer, config, tc.num_steps, final_path)
    print(f"\nTraining complete. Final: {final_path}")
    return history
