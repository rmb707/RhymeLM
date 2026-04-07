"""CLI entry point for training RhymeLM models."""

import argparse
from pathlib import Path
from functools import partial

import torch

from rhymelm.config import Config
from rhymelm.data import load_dictionary, load_lyrics, extract_verses, build_dual_corpus
from rhymelm.data import CharTokenizer, BPETokenizer
from rhymelm.data.corpus import (
    build_artist_vocab, load_lyrics_with_artists, extract_verses_with_artists,
)
from rhymelm.data.dataset import create_dataloaders, create_artist_dataloaders
from rhymelm.models import RhymeLM, RhymeLMPhoneme, RhymeLMAttention, RhymeLMTransformer
from rhymelm.training.trainer import train
from rhymelm.generation.sampler import generate_verse
from rhymelm.visualization.training_plots import plot_training_dashboard
from rhymelm.utils import get_device, seed_everything, count_parameters


def build_model(arch: str, vocab_size: int, config: Config):
    mc = config.model
    if arch == "lstm":
        return RhymeLM(vocab_size, mc.embed_dim, mc.hidden_dim, mc.num_layers, mc.dropout)
    elif arch == "lstm_phoneme":
        from rhymelm.data.phonemes import build_phoneme_vocab
        p2i, _ = build_phoneme_vocab()
        return RhymeLMPhoneme(vocab_size, len(p2i), mc.embed_dim, mc.hidden_dim, mc.num_layers, mc.dropout)
    elif arch == "lstm_attention":
        return RhymeLMAttention(vocab_size, mc.embed_dim, mc.hidden_dim, mc.num_layers, mc.n_heads, mc.dropout)
    elif arch == "transformer":
        return RhymeLMTransformer(
            vocab_size,
            d_model=mc.embed_dim,
            n_heads=mc.n_heads,
            n_layers=mc.num_layers,
            d_ff=mc.d_ff,
            dropout=mc.dropout,
            max_seq_len=mc.max_seq_len,
            num_artists=mc.num_artists,
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")


def main():
    parser = argparse.ArgumentParser(description="Train a RhymeLM model")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--arch", type=str, default=None, help="Override model architecture")
    parser.add_argument("--csv", type=str, default=None, help="Path to lyrics CSV")
    parser.add_argument("--steps", type=int, default=None, help="Override num_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--block-size", type=int, default=None, help="Override block_size")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    if args.arch:
        config.model.arch = args.arch
    if args.csv:
        config.data.csv_path = args.csv
    if args.steps:
        config.training.num_steps = args.steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.block_size:
        config.training.block_size = args.block_size
    if args.no_amp:
        config.training.use_amp = False

    seed_everything(config.training.seed)
    device = get_device()

    # ── Data Pipeline ──
    print("\n--- Data Pipeline ---")
    use_artists = config.model.arch == "transformer" and config.model.num_artists > 0

    dictionary = load_dictionary()
    lyrics = load_lyrics(config.data.csv_path, config.data.lyrics_column)
    verses = extract_verses(lyrics, config.data.min_bars, config.data.max_bars)
    corpus = build_dual_corpus(
        verses, dictionary,
        dict_ratio=config.data.dict_ratio,
        words_per_block=config.data.dict_words_per_block,
        seed=config.training.seed,
    )

    # ── Tokenizer ──
    if config.data.tokenizer_type == "bpe":
        print(f"\nTraining BPE tokenizer (target vocab: {config.data.target_bpe_vocab})...")
        tokenizer = BPETokenizer.train(corpus, target_vocab_size=config.data.target_bpe_vocab)
        encoded = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    else:
        tokenizer = CharTokenizer.from_corpus(corpus)
        encoded = tokenizer.encode(corpus)

    # ── DataLoaders ──
    if use_artists:
        artist_vocab = build_artist_vocab(config.data.csv_path)
        config.model.num_artists = len(artist_vocab)
        pairs = load_lyrics_with_artists(config.data.csv_path, config.data.lyrics_column)
        artist_verses = extract_verses_with_artists(pairs, config.data.min_bars, config.data.max_bars)

        verses_encoded = []
        for verse_text, artist in artist_verses:
            if config.data.tokenizer_type == "bpe":
                enc = torch.tensor(tokenizer.encode(verse_text), dtype=torch.long)
            else:
                enc = tokenizer.encode(verse_text)
            aid = artist_vocab.get(artist, 0)
            verses_encoded.append((enc, aid))

        train_loader, val_loader = create_artist_dataloaders(
            verses_encoded, config.training.block_size, config.training.batch_size,
            val_split=config.data.val_split,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            encoded, config.training.block_size, config.training.batch_size,
            val_split=config.data.val_split,
        )

    # ── Model ──
    print(f"\n--- Model: {config.model.arch} ---")
    model = build_model(config.model.arch, tokenizer.vocab_size, config).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    print(model)

    # ── Generation callback ──
    gen_fn = partial(
        generate_verse, model=model, tokenizer=tokenizer, device=device,
        num_bars=8, temperature=0.7,
    )

    # ── Train ──
    print("\n--- Training ---")
    if config.model.arch == "transformer":
        from rhymelm.training.transformer_trainer import train_transformer
        history = train_transformer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            tokenizer=tokenizer, config=config, device=device,
            generate_fn=gen_fn,
        )
    elif config.model.arch == "lstm_phoneme":
        from rhymelm.training.phoneme_trainer import train_phoneme
        history = train_phoneme(
            model=model, train_loader=train_loader, val_loader=val_loader,
            tokenizer=tokenizer, config=config, device=device,
            generate_fn=gen_fn,
        )
    else:
        history = train(
            model=model, train_loader=train_loader, val_loader=val_loader,
            tokenizer=tokenizer, config=config, device=device,
            generate_fn=gen_fn,
        )

    # Save tokenizer
    tok_path = str(Path(config.checkpoint_dir) / "tokenizer.json")
    tokenizer.save(tok_path)

    if args.plot and history["steps"]:
        plot_training_dashboard(
            steps=history["steps"],
            train_losses=history["train_loss"],
            val_losses=history["val_loss"],
            learning_rates=history.get("learning_rates"),
            grad_norms=history.get("grad_norms"),
            save_path=str(Path(config.checkpoint_dir) / "training_curve.png"),
        )


if __name__ == "__main__":
    main()
