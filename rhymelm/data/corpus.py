"""Dual-corpus builder: English dictionary (vocabulary grounding) + rap lyrics (style/flow)."""

import random
from typing import Optional

import nltk
import pandas as pd


def load_dictionary(max_word_len: int = 15) -> list[str]:
    """Load English + CMU pronouncing dictionary words for vocabulary grounding."""
    nltk.download("words", quiet=True)
    nltk.download("cmudict", quiet=True)

    from nltk.corpus import words as nltk_words
    from nltk.corpus import cmudict

    all_english = set(w.lower() for w in nltk_words.words())
    cmu_dict = cmudict.dict()
    rhymeable = set(cmu_dict.keys())

    vocab_words = [w for w in rhymeable if 2 <= len(w) <= max_word_len]
    vocab_words += [w for w in all_english if 2 <= len(w) <= 12 and w not in rhymeable]

    return vocab_words


def load_lyrics(csv_path: str, lyrics_column: str = "artist_verses") -> list[str]:
    """Load raw lyrics texts from CSV."""
    df = pd.read_csv(csv_path)
    texts = df[lyrics_column].dropna().tolist()
    print(f"Loaded {len(df):,} tracks ({df['artist'].nunique()} artists)")
    return texts


def build_artist_vocab(csv_path: str) -> dict[str, int]:
    """Map artist names to integer IDs. ID 0 is reserved for 'generic/no artist'."""
    df = pd.read_csv(csv_path)
    artists = sorted(df["artist"].dropna().unique())
    vocab = {name: i + 1 for i, name in enumerate(artists)}
    print(f"Artist vocab: {len(vocab)} artists")
    return vocab


def load_lyrics_with_artists(
    csv_path: str, lyrics_column: str = "artist_verses",
) -> list[tuple[str, str]]:
    """Load lyrics as (text, artist) pairs."""
    df = pd.read_csv(csv_path)
    pairs = []
    for _, row in df.iterrows():
        text = row.get(lyrics_column, "")
        artist = row.get("artist", "unknown")
        if isinstance(text, str) and len(text) > 50:
            pairs.append((text, artist))
    print(f"Loaded {len(pairs):,} tracks with artist labels")
    return pairs


def extract_verses_with_artists(
    pairs: list[tuple[str, str]], min_bars: int = 8, max_bars: int = 16,
) -> list[tuple[str, str]]:
    """Split lyrics into verse chunks, preserving artist labels."""
    noise_patterns = ["See ", "tickets as low as", "You might also like"]
    verses = []
    for text, artist in pairs:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not any(pat in ln for pat in noise_patterns)]
        for i in range(0, len(lines), max_bars):
            chunk = lines[i : i + max_bars]
            if len(chunk) >= min_bars:
                verses.append(("\n".join(chunk), artist))
    print(f"Extracted {len(verses):,} verse chunks with artist labels")
    return verses


def clean_user_lyrics(text: str) -> str:
    """Clean user-uploaded lyrics using the same pipeline as training data."""
    import re
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\[.*\]$", line):
            continue
        if len(line) < 3 and not line[0].isalpha():
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_verses(
    texts: list[str], min_bars: int = 8, max_bars: int = 16
) -> list[str]:
    """Split lyrics into verse chunks of min_bars to max_bars lines."""
    noise_patterns = ["See ", "tickets as low as", "You might also like"]
    verses = []

    for txt in texts:
        if not isinstance(txt, str):
            continue
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        lines = [
            ln for ln in lines if not any(pat in ln for pat in noise_patterns)
        ]
        for i in range(0, len(lines), max_bars):
            chunk = lines[i : i + max_bars]
            if len(chunk) >= min_bars:
                verses.append("\n".join(chunk))

    print(f"Extracted {len(verses):,} verse chunks ({min_bars}-{max_bars} bars)")
    return verses


def build_dual_corpus(
    verses: list[str],
    dictionary_words: list[str],
    dict_ratio: float = 0.25,
    words_per_block: int = 50,
    seed: Optional[int] = None,
) -> str:
    """
    Interleave verse chunks with dictionary word blocks.

    The dictionary teaches the model what words look like (vocabulary grounding).
    The lyrics teach it how artists flow (style, structure, rhyme schemes).
    """
    if seed is not None:
        random.seed(seed)

    shuffled = dictionary_words.copy()
    random.shuffle(shuffled)

    num_dict_blocks = int(len(verses) * dict_ratio / (1 - dict_ratio))
    dict_blocks = []
    for i in range(0, min(len(shuffled), num_dict_blocks * words_per_block), words_per_block):
        dict_blocks.append("\n".join(shuffled[i : i + words_per_block]))

    parts = []
    dict_idx = 0
    for verse in verses:
        parts.append(verse)
        if random.random() < dict_ratio and dict_idx < len(dict_blocks):
            parts.append(dict_blocks[dict_idx])
            dict_idx += 1

    corpus = "\n\n".join(parts)
    print(f"Corpus: {len(corpus):,} chars ({len(dict_blocks):,} dict blocks interleaved)")
    return corpus
