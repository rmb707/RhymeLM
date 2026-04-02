"""Artist-style generation: condition on artist to generate in their style.

Since the training data includes artist labels, we can build artist-specific
corpora or use artist-tagged prompts. The model implicitly learns different
artists' vocabularies, flow patterns, and rhyme tendencies from the data
distribution — this module exposes that learned knowledge.
"""

import random

import pandas as pd
import torch

from rhymelm.models.base import RhymeLMBase
from rhymelm.data.tokenizer import CharTokenizer
from rhymelm.data.phonemes import get_cmu_dict, get_rhyme_suffix
from rhymelm.generation.sampler import generate_verse
from rhymelm.generation.rhyme_sampler import generate_rhyming_verse
from rhymelm.evaluation.metrics import evaluate_verse


def get_artist_starters(csv_path: str, lyrics_column: str = "artist_verses") -> dict[str, list[str]]:
    """Extract characteristic line openings per artist from the dataset.

    These serve as prompts that prime the model toward a specific artist's style.
    """
    df = pd.read_csv(csv_path)
    artist_starters: dict[str, list[str]] = {}

    for _, row in df.iterrows():
        artist = row.get("artist", "unknown")
        text = row.get(lyrics_column, "")
        if not isinstance(text, str):
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 10]
        # Take first lines of verses as characteristic openings
        starters = []
        for i, line in enumerate(lines):
            if i % 16 == 0 and len(line) < 80:
                starters.append(line[:40])  # Truncate to use as prompt

        if starters:
            artist_starters.setdefault(artist, []).extend(starters)

    # Deduplicate and limit
    for artist in artist_starters:
        seen = set()
        unique = []
        for s in artist_starters[artist]:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique.append(s)
        artist_starters[artist] = unique[:50]

    return artist_starters


def generate_artist_verse(
    model: RhymeLMBase,
    tokenizer: CharTokenizer,
    device: torch.device,
    artist_starters: dict[str, list[str]],
    artist: str,
    num_bars: int = 16,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    rhyme_scheme: str | None = None,
) -> str:
    """Generate a verse in the style of a specific artist.

    Uses a characteristic opening line from that artist as the prompt,
    priming the model toward their vocabulary and flow patterns.
    """
    starters = artist_starters.get(artist, [])
    if not starters:
        available = list(artist_starters.keys())
        raise ValueError(
            f"Artist '{artist}' not found. Available: {available}"
        )

    prompt = random.choice(starters)

    if rhyme_scheme:
        return generate_rhyming_verse(
            model, tokenizer, device,
            prompt=prompt, num_bars=num_bars,
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty,
            rhyme_scheme=rhyme_scheme,
        )
    else:
        return generate_verse(
            model, tokenizer, device,
            prompt=prompt, num_bars=num_bars,
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty,
        )


def artist_style_comparison(
    model: RhymeLMBase,
    tokenizer: CharTokenizer,
    device: torch.device,
    csv_path: str,
    artists: list[str] | None = None,
    num_verses: int = 3,
    temperature: float = 0.8,
    rhyme_scheme: str = "AABB",
) -> dict[str, dict]:
    """Generate and evaluate verses in the style of multiple artists."""
    starters = get_artist_starters(csv_path)

    if artists is None:
        artists = list(starters.keys())

    results = {}
    for artist in artists:
        if artist not in starters:
            print(f"  Skipping '{artist}' — not in dataset")
            continue

        print(f"\n{'='*50}")
        print(f"Artist: {artist}")
        print(f"{'='*50}")

        verses = []
        all_metrics = []
        for i in range(num_verses):
            verse = generate_artist_verse(
                model, tokenizer, device, starters,
                artist=artist,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
                rhyme_scheme=rhyme_scheme,
            )
            verses.append(verse)
            metrics = evaluate_verse(verse)
            all_metrics.append(metrics)

            if i == 0:
                print(verse)
                print()

        # Aggregate metrics
        import numpy as np
        summary = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics]
            summary[f"{key}_mean"] = round(float(np.mean(values)), 4)

        summary["sample_verse"] = verses[0]
        summary["num_verses"] = num_verses
        results[artist] = summary

        print(f"  rhyme_density: {summary.get('rhyme_density_mean', 0):.3f}")
        print(f"  distinct_2: {summary.get('distinct_2_mean', 0):.3f}")
        print(f"  unique_lines: {summary.get('unique_line_ratio_mean', 0):.3f}")

    return results
