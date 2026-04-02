"""CLI entry point for generating verses with a trained RhymeLM model."""

import argparse

import torch

from rhymelm.config import Config
from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.models import RhymeLM, RhymeLMPhoneme, RhymeLMAttention, RhymeLMTransformer
from rhymelm.generation.sampler import generate_verse
from rhymelm.generation.rhyme_sampler import generate_rhyming_verse
from rhymelm.evaluation.metrics import evaluate_verse
from rhymelm.utils import get_device


def load_model(checkpoint_path: str, device: torch.device):
    """Load any RhymeLM model variant from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint["config"]["model"]
    tokenizer = CharTokenizer(checkpoint["tokenizer_chars"])

    arch = model_cfg.get("arch", "lstm")
    vocab = tokenizer.vocab_size

    if arch == "lstm_phoneme":
        from rhymelm.data.phonemes import build_phoneme_vocab
        p2i, _ = build_phoneme_vocab()
        model = RhymeLMPhoneme(
            vocab, len(p2i),
            model_cfg["embed_dim"], model_cfg["hidden_dim"],
            model_cfg["num_layers"], model_cfg["dropout"],
        )
    elif arch == "lstm_attention":
        model = RhymeLMAttention(
            vocab, model_cfg["embed_dim"], model_cfg["hidden_dim"],
            model_cfg["num_layers"], 8, model_cfg["dropout"],
        )
    elif arch == "transformer":
        model = RhymeLMTransformer(
            vocab, d_model=model_cfg["embed_dim"], n_heads=8,
            n_layers=model_cfg["num_layers"], d_ff=model_cfg["hidden_dim"],
            dropout=model_cfg["dropout"],
        )
    else:
        model = RhymeLM(
            vocab, model_cfg["embed_dim"], model_cfg["hidden_dim"],
            model_cfg["num_layers"], model_cfg["dropout"],
        )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Generate rap verses with RhymeLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=" ", help="Starting text")
    parser.add_argument("--bars", type=int, default=16, help="Number of bars")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--rhyme-scheme", type=str, default=None,
                        help="Rhyme scheme: AABB, ABAB, ABBA, or custom string")
    parser.add_argument("--artist", type=str, default=None,
                        help="Generate in the style of an artist from the dataset")
    parser.add_argument("--csv", type=str, default="lyrics_raw.csv",
                        help="Path to lyrics CSV (needed for --artist)")
    parser.add_argument("--num-verses", type=int, default=1)
    parser.add_argument("--eval", action="store_true", help="Run evaluation metrics")
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
            starters = get_artist_starters(args.csv)
            verse = generate_artist_verse(
                model, tokenizer, device, starters,
                artist=args.artist,
                num_bars=args.bars,
                temperature=args.temperature,
                top_p=args.top_p,
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
            )

        print(verse)

        if args.eval:
            metrics = evaluate_verse(verse)
            print(f"\nMetrics: {metrics}")

        print()


if __name__ == "__main__":
    main()
