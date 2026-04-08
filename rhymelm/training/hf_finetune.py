"""Fine-tune a HuggingFace pretrained model (GPT-2 XL / similar) on rap lyrics.

Uses LoRA for parameter-efficient fine-tuning. The pretrained model already
understands English — we just teach it rap style, flow, and vocabulary.
"""

import os
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rhymelm.training.lora import LoRALinear


class LyricsDataset(Dataset):
    """Tokenize lyrics into training chunks for causal LM fine-tuning."""

    def __init__(self, texts: list[str], tokenizer, block_size: int = 256):
        self.examples = []
        # Concatenate all texts into one stream, then chunk
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)

        # Chunk the concatenated stream
        for i in range(0, len(all_ids) - block_size, block_size // 2):
            chunk = all_ids[i : i + block_size]
            if len(chunk) == block_size:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return x[:-1], x[1:]  # input, target


class HFLoRAWrapper(torch.nn.Module):
    """Wraps a HuggingFace Conv1D/Linear with LoRA adapter."""

    def __init__(self, original, rank: int, alpha: float):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank

        d_in = original.weight.shape[0]
        d_out = original.weight.shape[1]
        dev = original.weight.device
        dtype = original.weight.dtype

        self.lora_A = torch.nn.Parameter(torch.randn(d_in, rank, device=dev, dtype=dtype) / rank)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, d_out, device=dev, dtype=dtype))

        # Freeze original
        original.weight.requires_grad = False
        if hasattr(original, "bias") and original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


def apply_lora_to_hf_model(model, rank: int = 16, alpha: float = 32.0):
    """Apply LoRA to a HuggingFace GPT-2 model's attention layers."""
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0
    replacements = []

    for name, module in model.named_modules():
        for attr in ["c_attn", "c_proj", "c_fc"]:
            if hasattr(module, attr):
                original = getattr(module, attr)
                if hasattr(original, "weight") and original.weight.dim() == 2:
                    replacements.append((module, attr, original))

    for module, attr, original in replacements:
        wrapper = HFLoRAWrapper(original, rank, alpha)
        setattr(module, attr, wrapper)
        lora_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {lora_count} layers adapted, {trainable:,} trainable / {total:,} total ({trainable/total:.2%})")
    return model


def format_lyrics_for_training(
    csv_path: str = "lyrics_cleaned.csv",
    lyrics_col: str = "lyrics_clean",
    artist_col: str = "artist",
) -> list[str]:
    """Format lyrics with artist tags for training.

    Format: <|artist|>Artist Name<|verse|>lyrics here<|end|>
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    texts = []
    for _, row in df.iterrows():
        artist = row.get(artist_col, "Unknown")
        lyrics = row.get(lyrics_col, "")
        if not isinstance(lyrics, str) or len(lyrics) < 100:
            continue

        # Split into verses
        lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
        for i in range(0, len(lines), 16):
            chunk = lines[i : i + 16]
            if len(chunk) >= 8:
                verse = "\n".join(chunk)
                formatted = f"<|artist|>{artist}<|verse|>\n{verse}\n<|end|>"
                texts.append(formatted)

    print(f"Formatted {len(texts)} training verses from {df[artist_col].nunique()} artists")
    return texts


def finetune_hf(
    model_name: str = "gpt2-xl",
    csv_path: str = "lyrics_cleaned.csv",
    lyrics_col: str = "lyrics_clean",
    lora_rank: int = 16,
    num_epochs: int = 3,
    batch_size: int = 2,
    block_size: int = 512,
    learning_rate: float = 2e-5,
    output_dir: str = "checkpoints/hf_finetuned",
    device: str = "cuda",
):
    """Fine-tune a HuggingFace model on rap lyrics with LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_name} on rap lyrics")
    print(f"{'='*60}")

    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens for artist conditioning
    special_tokens = {"additional_special_tokens": ["<|artist|>", "<|verse|>", "<|end|>"]}
    tokenizer.add_special_tokens(special_tokens)

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    params_before = sum(p.numel() for p in model.parameters())
    print(f"Model: {params_before/1e6:.0f}M params")

    # Apply LoRA
    model = apply_lora_to_hf_model(model, rank=lora_rank)

    # Prepare data
    print("\nPreparing lyrics data...")
    texts = format_lyrics_for_training(csv_path, lyrics_col)

    dataset = LyricsDataset(texts, tokenizer, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    # Optimizer — only LoRA params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=learning_rate, weight_decay=0.01)

    # Training loop
    model.train()
    total_steps = num_epochs * len(loader)
    step = 0
    best_loss = float("inf")

    print(f"\nTraining: {num_epochs} epochs, {total_steps} total steps")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss.float()  # compute loss in fp32

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            step += 1

            if step % 50 == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{num_epochs} | Step {step}/{total_steps} | Loss: {avg:.4f}")

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} complete | Avg loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

    # Merge LoRA weights and save as a clean model
    print("Merging LoRA weights and saving clean model...")

    # Load a fresh base model, apply merged weights
    base_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    base_model.resize_token_embeddings(len(tokenizer))

    # Get merged state dict from trained model
    merged_sd = {}
    trained_sd = model.state_dict()

    for key in base_model.state_dict():
        if key in trained_sd:
            merged_sd[key] = trained_sd[key]
        else:
            # Find the corresponding wrapper keys
            # e.g., "transformer.h.0.attn.c_attn.weight" -> check for ".original.weight" and lora
            orig_key = key.replace(".weight", ".original.weight").replace(".bias", ".original.bias")
            lora_a_key = key.rsplit(".", 1)[0] + ".lora_A"
            lora_b_key = key.rsplit(".", 1)[0] + ".lora_B"

            if orig_key in trained_sd and key.endswith(".weight"):
                # Merge: W' = W_orig + A @ B * scaling
                w = trained_sd[orig_key].clone()
                if lora_a_key in trained_sd and lora_b_key in trained_sd:
                    A = trained_sd[lora_a_key]
                    B = trained_sd[lora_b_key]
                    # Find scaling from the module
                    scaling = 32.0 / 16  # alpha / rank (defaults)
                    w = w + (A @ B) * scaling
                merged_sd[key] = w
            elif orig_key in trained_sd and key.endswith(".bias"):
                merged_sd[key] = trained_sd[orig_key]
            else:
                merged_sd[key] = base_model.state_dict()[key]

    base_model.load_state_dict(merged_sd)

    # Save clean model
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSaved to {output_dir}")
    print(f"Best loss: {best_loss:.4f}")

    return model, tokenizer


def generate_bars(
    model,
    tokenizer,
    artist: str = "Eminem",
    num_bars: int = 16,
    temperature: float = 0.85,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """Generate bars in an artist's style using the fine-tuned model."""
    prompt = f"<|artist|>{artist}<|verse|>\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=num_bars * 30,  # ~30 tokens per bar
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.encode("<|end|>")[0] if "<|end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the verse
    if "<|verse|>" in text:
        text = text.split("<|verse|>")[-1]
    if "<|end|>" in text:
        text = text.split("<|end|>")[0]

    # Trim to requested bars
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return "\n".join(lines[:num_bars])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-xl", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0"
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    model, tokenizer = finetune_hf(
        model_name=args.model,
        num_epochs=args.epochs,
        lora_rank=args.rank,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Test generation
    print("\n" + "=" * 60)
    for artist in ["Eminem", "Drake", "Kendrick Lamar"]:
        print(f"\n--- {artist} ---")
        bars = generate_bars(model, tokenizer, artist=artist)
        print(bars)
