"""Ablation study framework: systematic comparison with error bars.

Runs the same configuration across multiple seeds and computes
mean +/- std for all metrics, enabling rigorous comparison between
model variants and training configurations.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rhymelm.config import Config
from rhymelm.data import load_dictionary, load_lyrics, extract_verses, build_dual_corpus
from rhymelm.data import CharTokenizer, create_dataloaders
from rhymelm.models import RhymeLM
from rhymelm.training.trainer import train
from rhymelm.generation.sampler import generate_verse
from rhymelm.evaluation.metrics import evaluate_verse
from rhymelm.utils import get_device, seed_everything, count_parameters


def run_single_ablation(
    config: Config,
    device: torch.device,
    seed: int,
    num_eval_verses: int = 10,
) -> dict:
    """Run a single training + evaluation pass with a given seed."""
    config.training.seed = seed
    seed_everything(seed)

    # Data
    dictionary = load_dictionary()
    lyrics = load_lyrics(config.data.csv_path, config.data.lyrics_column)
    verses = extract_verses(lyrics, config.data.min_bars, config.data.max_bars)
    corpus = build_dual_corpus(
        verses, dictionary,
        dict_ratio=config.data.dict_ratio, seed=seed,
    )

    tokenizer = CharTokenizer.from_corpus(corpus)
    encoded = tokenizer.encode(corpus)
    train_loader, val_loader = create_dataloaders(
        encoded,
        block_size=config.training.block_size,
        batch_size=config.training.batch_size,
        val_split=config.data.val_split,
    )

    # Model
    model = RhymeLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)

    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    # Evaluate
    all_metrics = []
    for _ in range(num_eval_verses):
        verse = generate_verse(model, tokenizer, device, temperature=0.8, top_p=0.95, num_bars=16)
        all_metrics.append(evaluate_verse(verse))

    # Aggregate
    final_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")
    aggregated = {"final_val_loss": final_loss}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    return aggregated


def run_ablation(
    name: str,
    config: Config,
    seeds: list[int] = [42, 123, 456],
    output_dir: str = "experiments/ablations",
    num_eval_verses: int = 10,
) -> dict:
    """Run ablation across multiple seeds and aggregate results."""
    device = get_device()
    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Ablation '{name}' | Seed {seed}")
        print(f"{'='*60}")

        result = run_single_ablation(config, device, seed, num_eval_verses)
        result["seed"] = seed
        all_results.append(result)

    # Aggregate across seeds
    summary = {"name": name, "seeds": seeds, "n_seeds": len(seeds)}
    metric_keys = [k for k in all_results[0] if k != "seed"]
    for key in metric_keys:
        values = [r[key] for r in all_results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    # Save
    with open(out_path / "results.json", "w") as f:
        json.dump({"summary": summary, "runs": all_results}, f, indent=2)

    print(f"\n--- Ablation '{name}' Summary ---")
    for key in metric_keys:
        if "_mean" not in key and "_std" not in key:
            print(f"  {key}: {summary.get(f'{key}_mean', 'N/A'):.4f} +/- {summary.get(f'{key}_std', 'N/A'):.4f}")

    return summary


def format_ablation_table(results: list[dict]) -> str:
    """Format multiple ablation results as a comparison table."""
    if not results:
        return "No results."

    # Get all metric keys
    sample = results[0]
    metric_keys = [k.replace("_mean", "") for k in sample if k.endswith("_mean")]

    header = f"{'Name':<25}"
    for k in metric_keys:
        header += f" | {k:<18}"
    lines = [header, "-" * len(header)]

    for r in results:
        row = f"{r['name']:<25}"
        for k in metric_keys:
            mean = r.get(f"{k}_mean", 0)
            std = r.get(f"{k}_std", 0)
            row += f" | {mean:>7.4f} +/- {std:.4f}"
        lines.append(row)

    return "\n".join(lines)
