"""Model comparison framework: evaluate multiple models on the same metrics."""

import torch
import numpy as np

from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.generation.sampler import generate_verse
from rhymelm.evaluation.metrics import evaluate_verse
from rhymelm.utils import count_parameters


def compare_models(
    models: dict[str, tuple],  # name -> (model, tokenizer)
    device: torch.device,
    prompts: list[str] = None,
    num_verses: int = 10,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict[str, dict]:
    """Compare multiple models on generation quality metrics.

    Args:
        models: dict of name -> (model, tokenizer) pairs
        num_verses: number of verses to generate per model for averaging
    """
    if prompts is None:
        prompts = ["I ", "Yeah, ", "Life ", "Money ", "Back in the day "]

    results = {}

    for name, (model, tokenizer) in models.items():
        print(f"\nEvaluating: {name}")
        model.eval()

        all_metrics = []
        sample_verses = []

        for i in range(num_verses):
            prompt = prompts[i % len(prompts)]
            verse = generate_verse(
                model, tokenizer, device,
                prompt=prompt, temperature=temperature,
                top_p=top_p, num_bars=16,
            )
            metrics = evaluate_verse(verse)
            all_metrics.append(metrics)
            if i < 3:
                sample_verses.append(verse)

        # Aggregate
        summary = {"params": count_parameters(model)}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
            summary[f"{key}_std"] = round(float(np.std(values)), 4)

        summary["samples"] = sample_verses
        results[name] = summary
        print(f"  params: {summary['params']:,}")
        for key in all_metrics[0]:
            print(f"  {key}: {summary[f'{key}_mean']:.4f} +/- {summary[f'{key}_std']:.4f}")

    return results


def format_comparison_table(results: dict[str, dict]) -> str:
    """Format comparison results as a readable table."""
    if not results:
        return "No results."

    first = next(iter(results.values()))
    metric_keys = [k.replace("_mean", "") for k in first if k.endswith("_mean")]

    header = f"{'Model':<25} {'Params':>10}"
    for k in metric_keys:
        header += f" | {k:<16}"
    lines = [header, "-" * len(header)]

    for name, r in results.items():
        row = f"{name:<25} {r['params']:>10,}"
        for k in metric_keys:
            mean = r.get(f"{k}_mean", 0)
            std = r.get(f"{k}_std", 0)
            row += f" | {mean:>6.3f}+/-{std:.3f}"
        lines.append(row)

    return "\n".join(lines)


def side_by_side_generation(
    models: dict[str, tuple],
    device: torch.device,
    prompt: str = "I ",
    temperature: float = 0.8,
    seed: int = 42,
) -> dict[str, str]:
    """Generate from each model with the same prompt and seed for direct comparison."""
    import random
    results = {}

    for name, (model, tokenizer) in models.items():
        torch.manual_seed(seed)
        random.seed(seed)
        model.eval()
        verse = generate_verse(
            model, tokenizer, device,
            prompt=prompt, temperature=temperature, num_bars=16,
        )
        results[name] = verse

    return results
