"""Master training script: scrape data, augment, fine-tune GPT-2 XL.

Usage:
    HSA_OVERRIDE_GFX_VERSION=12.0.0 python train_big.py --epochs 10 --rank 32
    HSA_OVERRIDE_GFX_VERSION=12.0.0 python train_big.py --scrape --epochs 10  # also scrape new data
"""

import os
import argparse
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "12.0.0")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def main():
    parser = argparse.ArgumentParser(description="Train RhymeLM with maximum data")
    parser.add_argument("--scrape", action="store_true", help="Scrape new lyrics from web")
    parser.add_argument("--model", default="gpt2-xl", help="Base model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--augment", type=int, default=3, help="Data augmentation multiplier")
    args = parser.parse_args()

    print("=" * 60)
    print("RhymeLM Big Training Pipeline")
    print("=" * 60)

    # Step 1: Gather data
    all_verses = []

    # Existing CSV data
    print("\n--- Loading existing data ---")
    from rhymelm.data.augment_data import augment_dataset, format_for_hf_training
    import pandas as pd

    csv_path = "lyrics_cleaned.csv" if os.path.exists("lyrics_cleaned.csv") else "lyrics_raw.csv"
    lyrics_col = "lyrics_clean" if "cleaned" in csv_path else "artist_verses"

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        artist = row.get("artist", "Unknown")
        text = row.get(lyrics_col, "")
        if isinstance(text, str) and len(text) > 100:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            for i in range(0, len(lines), 16):
                chunk = lines[i:i+16]
                if len(chunk) >= 8:
                    all_verses.append(("\n".join(chunk), artist))

    print(f"Existing data: {len(all_verses)} verses")

    # Step 2: Scrape new data if requested
    if args.scrape:
        print("\n--- Scraping new lyrics ---")
        try:
            from rhymelm.data.scrape_web import scrape_artist_songs, TOP_ARTISTS
            for artist in TOP_ARTISTS[:20]:  # Top 20 artists
                try:
                    songs = scrape_artist_songs(artist, max_songs=30)
                    for song in songs:
                        text = song["lyrics"]
                        lines = [l.strip() for l in text.splitlines() if l.strip()]
                        for i in range(0, len(lines), 16):
                            chunk = lines[i:i+16]
                            if len(chunk) >= 8:
                                all_verses.append(("\n".join(chunk), artist))
                    print(f"  {artist}: {len(songs)} songs scraped")
                except Exception as e:
                    print(f"  {artist}: failed ({e})")
        except ImportError:
            print("  Scraper not available, using existing data only")

    print(f"Total verses before augmentation: {len(all_verses)}")

    # Step 3: Augment data
    print(f"\n--- Augmenting data ({args.augment}x) ---")
    augmented = []
    augmented.extend(all_verses)  # originals

    try:
        from rhymelm.data.augment_data import augment_verses
        extra = augment_verses(all_verses, multiplier=args.augment)
        augmented.extend(extra)
    except Exception as e:
        print(f"  Augmentation failed: {e}, using originals only")

    print(f"Total verses after augmentation: {len(augmented)}")

    # Step 4: Format for training
    print("\n--- Formatting for HF training ---")
    texts = [f"<|artist|>{artist}<|verse|>\n{verse}\n<|end|>" for verse, artist in augmented]
    print(f"Training texts: {len(texts)}")

    # Step 5: Fine-tune
    print(f"\n--- Fine-tuning {args.model} ---")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    from rhymelm.training.hf_finetune import LyricsDataset

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|artist|>", "<|verse|>", "<|end|>"]})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = LyricsDataset(texts, tokenizer, block_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    model.train()

    print(f"\nTraining: {args.epochs} epochs, {total_steps} total steps")
    print("=" * 60)

    for epoch in range(args.epochs):
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
            if step % 100 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | Step {step}/{total_steps} | Loss: {epoch_loss/(batch_idx+1):.4f}")
        print(f"Epoch {epoch+1} done | Loss: {epoch_loss/len(loader):.4f}")

    # Save
    print("\nMerging LoRA and saving...")
    model = model.merge_and_unload()
    output_dir = "checkpoints/hf_finetuned"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    # Quick test
    print("\n" + "=" * 60)
    print("TEST GENERATION:")
    print("=" * 60)
    model.eval()
    for artist in ["Eminem", "Drake", "Kendrick Lamar"]:
        prompt = f"<|artist|>{artist}<|verse|>\n"
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=200, temperature=0.85,
                top_p=0.95, do_sample=True, repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0], skip_special_tokens=False)
        if "<|verse|>" in text: text = text.split("<|verse|>")[-1]
        if "<|end|>" in text: text = text.split("<|end|>")[0]
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()][:8]
        print(f"\n{artist}:")
        for l in lines:
            print(f"  {l}")


if __name__ == "__main__":
    main()
