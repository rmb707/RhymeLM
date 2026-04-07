"""CLI entry point for generating verses with a trained RhymeLM model."""

import argparse

import torch

from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.data.bpe import BPETokenizer
from rhymelm.models import RhymeLM, RhymeLMPhoneme, RhymeLMAttention, RhymeLMTransformer
from rhymelm.generation.sampler import generate_verse
from rhymelm.generation.rhyme_sampler import generate_rhyming_verse
from rhymelm.evaluation.metrics import evaluate_verse
from rhymelm.utils import get_device


def load_model(checkpoint_path: str, device: torch.device):
    """Load any RhymeLM model variant from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]

    # Load tokenizer (supports both old char-only and new format)
    tok_data = checkpoint.get("tokenizer_data", {})
    if tok_data.get("type") == "bpe":
        merges = [tuple(m) for m in tok_data["merges"]]
        tokenizer = BPETokenizer(merges, tok_data["vocab"])
    elif "chars" in tok_data:
        tokenizer = CharTokenizer(tok_data["chars"])
    else:
        # Backward compat with old checkpoints
        tokenizer = CharTokenizer(checkpoint.get("tokenizer_chars", []))

    vocab = tokenizer.vocab_size
    arch = model_cfg.get("arch", "lstm")

    if arch == "lstm_phoneme":
        from rhymelm.data.phonemes import build_phoneme_vocab
        p2i, _ = build_phoneme_vocab()
        model = RhymeLMPhoneme(
            vocab, len(p2i), model_cfg["embed_dim"], model_cfg["hidden_dim"],
            model_cfg["num_layers"], model_cfg["dropout"],
        )
    elif arch == "lstm_attention":
        model = RhymeLMAttention(
            vocab, model_cfg["embed_dim"], model_cfg["hidden_dim"],
            model_cfg["num_layers"], model_cfg.get("n_heads", 8), model_cfg["dropout"],
        )
    elif arch == "transformer":
        model = RhymeLMTransformer(
            vocab,
            d_model=model_cfg["embed_dim"],
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg["num_layers"],
            d_ff=model_cfg.get("d_ff", model_cfg.get("hidden_dim", 1024)),
            dropout=model_cfg["dropout"],
            max_seq_len=model_cfg.get("max_seq_len", 512),
            num_artists=model_cfg.get("num_artists", 0),
        )
    else:
        model = RhymeLM(
            vocab, model_cfg["embed_dim"], model_cfg["hidden_dim"],
            model_cfg["num_layers"], model_cfg["dropout"],
        )

    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.to(device)
    model.eval()

    return model, tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate rap verses with RhymeLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=" ")
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--rhyme-scheme", type=str, default=None)
    parser.add_argument("--artist", type=str, default=None)
    parser.add_argument("--artist-id", type=int, default=None, help="Artist embedding ID")
    parser.add_argument("--csv", type=str, default="lyrics_cleaned.csv")
    parser.add_argument("--num-verses", type=int, default=1)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    step = checkpoint.get("step", "?")
    arch = checkpoint["config"]["model"].get("arch", "lstm")
    print(f"Loaded {arch} model (step {step})\n")

    for i in range(args.num_verses):
        if args.num_verses > 1:
            print(f"--- Verse {i + 1} ---")

        if args.artist:
            from rhymelm.generation.artist_style import generate_artist_verse, get_artist_starters
            starters = get_artist_starters(args.csv, lyrics_column="lyrics_clean")
            verse = generate_artist_verse(
                model, tokenizer, device, starters,
                artist=args.artist, num_bars=args.bars,
                temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                rhyme_scheme=args.rhyme_scheme,
            )
        elif args.rhyme_scheme:
            verse = generate_rhyming_verse(
                model, tokenizer, device,
                prompt=args.prompt, num_bars=args.bars,
                temperature=args.temperature, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                rhyme_scheme=args.rhyme_scheme,
            )
        else:
            verse = generate_verse(
                model, tokenizer, device,
                prompt=args.prompt, num_bars=args.bars,
                temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                artist_id=args.artist_id,
            )

        print(verse)
        if args.eval:
            print(f"\nMetrics: {evaluate_verse(verse)}")
        print()


if __name__ == "__main__":
    main()
