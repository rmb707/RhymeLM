"""Fine-tune GPT-2 XL (1.5B) on rap lyrics with LoRA via HuggingFace peft.

Uses artist tags so the model learns per-artist style from training data
rather than relying on prompt engineering.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class LyricsDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, block_size: int = 256):
        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        self.examples = []
        for i in range(0, len(all_ids) - block_size, block_size // 2):
            chunk = all_ids[i : i + block_size]
            if len(chunk) == block_size:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return x[:-1], x[1:]


def format_lyrics(csv_path="lyrics_cleaned.csv", lyrics_col="lyrics_clean"):
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = []
    for _, row in df.iterrows():
        artist = row.get("artist", "Unknown")
        lyrics = row.get(lyrics_col, "")
        if not isinstance(lyrics, str) or len(lyrics) < 100:
            continue
        lines = [l.strip() for l in lyrics.splitlines() if l.strip()]
        for i in range(0, len(lines), 16):
            chunk = lines[i : i + 16]
            if len(chunk) >= 8:
                texts.append(f"<|artist|>{artist}<|verse|>\n" + "\n".join(chunk) + "\n<|end|>")
    print(f"Formatted {len(texts)} verses from {df['artist'].nunique()} artists")
    return texts


def finetune_hf(
    model_name="gpt2-xl",
    csv_path="lyrics_cleaned.csv",
    lyrics_col="lyrics_clean",
    lora_rank=32,
    num_epochs=5,
    batch_size=2,
    block_size=256,
    lr=5e-5,
    output_dir="checkpoints/hf_finetuned",
    device="cuda",
):
    print(f"\n{'='*60}\nFine-tuning {model_name}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|artist|>", "<|verse|>", "<|end|>"]})

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print(f"Base model: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # LoRA via peft — handles Conv1D correctly
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    texts = format_lyrics(csv_path, lyrics_col)
    dataset = LyricsDataset(texts, tokenizer, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = num_epochs * len(loader)
    model.train()

    print(f"Training: {num_epochs} epochs, {total_steps} steps\n{'='*60}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            loss = model(input_ids=x, labels=y).loss.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            step = epoch * len(loader) + batch_idx + 1
            if step % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Step {step}/{total_steps} | Loss: {epoch_loss/(batch_idx+1):.4f}")
        print(f"Epoch {epoch+1} done | Loss: {epoch_loss/len(loader):.4f}")

    # Merge LoRA into base and save clean model
    print("Merging LoRA and saving...")
    model = model.merge_and_unload()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")
    return model, tokenizer


def generate_bars(model, tokenizer, artist="Eminem", num_bars=16, temperature=0.85, top_p=0.95, device="cuda"):
    prompt = f"<|artist|>{artist}<|verse|>\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    end_id = tokenizer.encode("<|end|>")[0] if "<|end|>" in tokenizer.get_vocab() else tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=num_bars * 30,
            temperature=temperature, top_p=top_p, do_sample=True,
            repetition_penalty=1.15, eos_token_id=end_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    if "<|verse|>" in text:
        text = text.split("<|verse|>")[-1]
    if "<|end|>" in text:
        text = text.split("<|end|>")[0]
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return "\n".join(lines[:num_bars])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-xl")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0"
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    model, tokenizer = finetune_hf(
        model_name=args.model, num_epochs=args.epochs,
        lora_rank=args.rank, batch_size=args.batch_size, lr=args.lr,
    )

    print("\n" + "="*60)
    for artist in ["Eminem", "Drake", "Kendrick Lamar", "2Pac", "Nas"]:
        print(f"\n--- {artist} ---")
        print(generate_bars(model, tokenizer, artist=artist))
