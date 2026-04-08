"""
Data augmentation for RhymeLM training data.

Turns ~530 songs into 1500+ unique training verses through:
- Line shuffling within 4-bar blocks
- Verse recombination across songs by the same artist
- Synonym swapping with rap-specific vocabulary
- Bar splitting at natural break points
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Rap synonym dictionary: word -> list of replacements
# ---------------------------------------------------------------------------
RAP_SYNONYMS: dict[str, list[str]] = {
    # money
    "money": ["bread", "paper", "guap", "racks", "bands", "cheddar", "cheese", "dough"],
    "cash": ["bread", "paper", "guap", "racks", "bands", "cheddar"],
    "dollars": ["bands", "racks", "bills", "stacks"],
    # weapons
    "gun": ["piece", "strap", "heat", "tool", "blick", "iron"],
    "guns": ["straps", "heaters", "tools", "irons"],
    "pistol": ["strap", "piece", "heat", "tool", "blick"],
    # vehicles
    "car": ["whip", "ride", "foreign", "slab"],
    "cars": ["whips", "rides", "foreigns"],
    # people
    "friend": ["homie", "dawg", "bro", "fam"],
    "friends": ["homies", "dawgs", "bros"],
    "girl": ["shawty", "shorty", "baddie", "queen"],
    "woman": ["shawty", "shorty", "queen"],
    "man": ["dude", "bruh", "dawg", "fam"],
    "enemy": ["opp", "hater", "rival"],
    "enemies": ["opps", "haters", "rivals"],
    "police": ["feds", "cops", "twelve", "one-time", "boys"],
    "cop": ["fed", "pig", "twelve", "one-time"],
    # places
    "house": ["crib", "pad", "spot"],
    "home": ["crib", "pad", "spot"],
    "city": ["town", "hood", "block"],
    "neighborhood": ["hood", "block", "turf", "ends"],
    # actions
    "shoot": ["blast", "pop", "bust", "spray"],
    "run": ["dip", "dash", "flee", "bounce"],
    "leave": ["dip", "bounce", "slide", "ghost"],
    "fight": ["scrap", "beef", "clash"],
    "steal": ["jack", "finesse", "swipe"],
    "drive": ["whip", "cruise", "slide"],
    "rap": ["spit", "flow", "bars"],
    "sing": ["croon", "belt", "vibe"],
    # adjectives / descriptors
    "crazy": ["wild", "insane", "mental", "mad"],
    "cool": ["icy", "cold", "smooth", "fresh"],
    "rich": ["loaded", "paid", "wealthy", "flush"],
    "expensive": ["pricey", "costly", "steep"],
    "real": ["legit", "solid", "true", "genuine"],
    "fake": ["phony", "cap", "fraud", "frontin'"],
    # jewelry / fashion
    "jewelry": ["ice", "drip", "chains", "rocks"],
    "chain": ["rope", "piece", "link"],
    "watch": ["timepiece", "rollie", "wrist"],
    "clothes": ["drip", "fits", "threads", "gear"],
    "shoes": ["kicks", "sneakers", "J's"],
    # drugs (common in rap lexicon)
    "weed": ["gas", "loud", "bud", "tree", "za"],
    "drugs": ["pack", "work", "product"],
    # misc
    "phone": ["line", "cell", "jack"],
    "song": ["track", "joint", "banger", "hit"],
    "album": ["tape", "project", "drop"],
    "love": ["luv", "affection"],
    "hate": ["beef", "animosity"],
    "truth": ["facts", "gospel", "word"],
    "lie": ["cap", "fib", "front"],
    "lies": ["cap", "fibs", "fronts"],
}

# Pre-compile a word-boundary regex for each synonym key (case-insensitive)
_SYNONYM_PATTERNS: dict[str, re.Pattern] = {
    word: re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    for word in RAP_SYNONYMS
}

# Natural break-point pattern used for bar splitting
_SPLIT_PATTERN = re.compile(
    r",\s+|\s+and\s+|\s+but\s+|\s+'cause\s+|\s+then\s+|\s+so\s+|\s+like\s+",
    re.IGNORECASE,
)

# Minimum line length (chars) to consider for bar splitting
_MIN_SPLIT_LENGTH = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str) -> list[dict[str, str]]:
    """Load lyrics_cleaned.csv and return list of row dicts."""
    rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("lyrics_clean") and row.get("artist"):
                rows.append(row)
    return rows


def _lines_from_lyrics(lyrics: str) -> list[str]:
    """Split lyrics text into non-empty lines."""
    return [ln.strip() for ln in lyrics.split("\n") if ln.strip()]


def _chunk_lines(lines: list[str], chunk_size: int) -> list[list[str]]:
    """Break a list of lines into chunks of *chunk_size*."""
    return [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]


# ---------------------------------------------------------------------------
# Augmentation techniques
# ---------------------------------------------------------------------------

def shuffle_within_blocks(lyrics: str, block_size: int = 4) -> str | None:
    """Randomly reorder lines within each 4-bar block.

    Returns None if the lyrics are too short to form a full block.
    """
    lines = _lines_from_lyrics(lyrics)
    if len(lines) < block_size:
        return None
    chunks = _chunk_lines(lines, block_size)
    shuffled: list[str] = []
    changed = False
    for chunk in chunks:
        if len(chunk) == block_size:
            original = list(chunk)
            random.shuffle(chunk)
            if chunk != original:
                changed = True
        shuffled.extend(chunk)
    if not changed:
        return None
    return "\n".join(shuffled)


def verse_recombination(
    songs_by_artist: dict[str, list[list[str]]],
    artist: str,
    chunk_size: int = 8,
    max_chunks: int = 2,
) -> str | None:
    """Combine 8-bar chunks from different songs by the same artist.

    *songs_by_artist* maps artist -> list of (list-of-lines per song).
    Picks *max_chunks* chunks from distinct songs and concatenates them.
    Returns None if the artist has fewer than 2 songs with enough lines.
    """
    song_lines_list = songs_by_artist.get(artist, [])
    # Gather eligible chunks across songs, tagged with song index
    tagged_chunks: list[tuple[int, list[str]]] = []
    for idx, song_lines in enumerate(song_lines_list):
        for chunk in _chunk_lines(song_lines, chunk_size):
            if len(chunk) == chunk_size:
                tagged_chunks.append((idx, chunk))
    if len(tagged_chunks) < max_chunks:
        return None

    # Try to pick chunks from different songs
    selected: list[list[str]] = []
    used_songs: set[int] = set()
    random.shuffle(tagged_chunks)
    for song_idx, chunk in tagged_chunks:
        if song_idx not in used_songs:
            selected.append(chunk)
            used_songs.add(song_idx)
        if len(selected) == max_chunks:
            break

    # Fallback: if we couldn't get chunks from distinct songs, just pick any two
    if len(selected) < max_chunks:
        random.shuffle(tagged_chunks)
        selected = [c for _, c in tagged_chunks[:max_chunks]]

    if len(selected) < max_chunks:
        return None

    combined: list[str] = []
    for chunk in selected:
        combined.extend(chunk)
    return "\n".join(combined)


def synonym_swap(lyrics: str, swap_prob: float = 0.3) -> str | None:
    """For each occurrence of a synonym-able word, randomly swap it.

    *swap_prob* controls how likely each match is to be replaced.
    Returns None if no swaps were made.
    """
    result = lyrics
    swapped = False
    for word, pattern in _SYNONYM_PATTERNS.items():
        def _replacer(m: re.Match) -> str:
            nonlocal swapped
            if random.random() > swap_prob:
                return m.group(0)
            replacement = random.choice(RAP_SYNONYMS[word])
            # Preserve original capitalisation of first letter
            orig = m.group(0)
            if orig[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            swapped = True
            return replacement
        result = pattern.sub(_replacer, result)
    return result if swapped else None


def bar_split(lyrics: str) -> str | None:
    """Split long lines at natural break points (commas, conjunctions, etc.).

    Returns None if no splits were made.
    """
    lines = _lines_from_lyrics(lyrics)
    new_lines: list[str] = []
    changed = False
    for line in lines:
        if len(line) >= _MIN_SPLIT_LENGTH:
            match = _SPLIT_PATTERN.search(line)
            if match:
                pos = match.start()
                left = line[:pos].strip()
                right = line[match.end():].strip()
                if left and right and len(left) > 10 and len(right) > 10:
                    new_lines.append(left)
                    new_lines.append(right)
                    changed = True
                    continue
        new_lines.append(line)
    return "\n".join(new_lines) if changed else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_dataset(csv_path: str, multiplier: int = 3) -> list[str]:
    """Load *csv_path* and return a list of augmented verse texts.

    The returned list includes the original verses plus augmented variants,
    targeting roughly ``len(originals) * multiplier`` total entries.

    Parameters
    ----------
    csv_path : str
        Path to ``lyrics_cleaned.csv``.
    multiplier : int, optional
        Target multiplier for the dataset size (default 3).

    Returns
    -------
    list[str]
        Augmented verse texts (plain strings, newline-separated lines).
    """
    rows = _load_csv(csv_path)

    # Collect per-artist song lines for verse recombination
    songs_by_artist: dict[str, list[list[str]]] = defaultdict(list)
    originals: list[tuple[str, str]] = []  # (artist, lyrics_clean)

    for row in rows:
        artist = row["artist"]
        lyrics = row["lyrics_clean"]
        lines = _lines_from_lyrics(lyrics)
        if not lines:
            continue
        songs_by_artist[artist].append(lines)
        originals.append((artist, lyrics))

    target = len(originals) * multiplier
    verses: list[str] = []

    # 1. Add all originals
    for _, lyrics in originals:
        verses.append(lyrics)

    # 2. Line shuffling pass (one per original)
    for _, lyrics in originals:
        aug = shuffle_within_blocks(lyrics)
        if aug:
            verses.append(aug)

    # 3. Synonym swap pass (one per original)
    for _, lyrics in originals:
        aug = synonym_swap(lyrics, swap_prob=0.3)
        if aug:
            verses.append(aug)

    # 4. Bar splitting pass (one per original)
    for _, lyrics in originals:
        aug = bar_split(lyrics)
        if aug:
            verses.append(aug)

    # 5. Verse recombination to fill up toward target
    artist_list = list(songs_by_artist.keys())
    attempts = 0
    max_attempts = target * 3  # safety cap
    while len(verses) < target and attempts < max_attempts:
        artist = random.choice(artist_list)
        aug = verse_recombination(songs_by_artist, artist)
        if aug:
            verses.append(aug)
        attempts += 1

    # 6. Additional synonym + shuffle combos if still short
    if len(verses) < target:
        random.shuffle(originals)
        for _, lyrics in originals:
            if len(verses) >= target:
                break
            aug = synonym_swap(lyrics, swap_prob=0.5)
            if aug:
                aug2 = shuffle_within_blocks(aug)
                verses.append(aug2 if aug2 else aug)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in verses:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique


def format_for_hf_training(verses: list[tuple[str, str]]) -> list[str]:
    """Format (verse_text, artist) pairs for Hugging Face fine-tuning.

    Parameters
    ----------
    verses : list[tuple[str, str]]
        Each element is ``(verse_text, artist_name)``.

    Returns
    -------
    list[str]
        Formatted strings like::

            <|artist|>Drake<|verse|>
            line 1
            line 2
            <|end|>
    """
    formatted: list[str] = []
    for verse_text, artist in verses:
        text = f"<|artist|>{artist}<|verse|>\n{verse_text}\n<|end|>"
        formatted.append(text)
    return formatted


def augment_dataset_with_artists(
    csv_path: str, multiplier: int = 3
) -> list[tuple[str, str]]:
    """Like *augment_dataset*, but returns ``(verse_text, artist)`` tuples.

    This is handy when you want to pipe directly into *format_for_hf_training*.
    """
    rows = _load_csv(csv_path)

    songs_by_artist: dict[str, list[list[str]]] = defaultdict(list)
    originals: list[tuple[str, str]] = []

    for row in rows:
        artist = row["artist"]
        lyrics = row["lyrics_clean"]
        lines = _lines_from_lyrics(lyrics)
        if not lines:
            continue
        songs_by_artist[artist].append(lines)
        originals.append((artist, lyrics))

    target = len(originals) * multiplier
    verses: list[tuple[str, str]] = []

    # 1. Originals
    for artist, lyrics in originals:
        verses.append((lyrics, artist))

    # 2. Shuffled
    for artist, lyrics in originals:
        aug = shuffle_within_blocks(lyrics)
        if aug:
            verses.append((aug, artist))

    # 3. Synonym swapped
    for artist, lyrics in originals:
        aug = synonym_swap(lyrics, swap_prob=0.3)
        if aug:
            verses.append((aug, artist))

    # 4. Bar split
    for artist, lyrics in originals:
        aug = bar_split(lyrics)
        if aug:
            verses.append((aug, artist))

    # 5. Verse recombination
    artist_list = list(songs_by_artist.keys())
    attempts = 0
    max_attempts = target * 3
    while len(verses) < target and attempts < max_attempts:
        artist = random.choice(artist_list)
        aug = verse_recombination(songs_by_artist, artist)
        if aug:
            verses.append((aug, artist))
        attempts += 1

    # 6. Extra synonym + shuffle combos
    if len(verses) < target:
        random.shuffle(originals)
        for artist, lyrics in originals:
            if len(verses) >= target:
                break
            aug = synonym_swap(lyrics, swap_prob=0.5)
            if aug:
                aug2 = shuffle_within_blocks(aug)
                verses.append((aug2 if aug2 else aug, artist))

    # Deduplicate on verse text
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for verse_text, artist in verses:
        if verse_text not in seen:
            seen.add(verse_text)
            unique.append((verse_text, artist))

    return unique


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Augment RhymeLM lyrics data")
    parser.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[2] / "lyrics_cleaned.csv"),
        help="Path to lyrics_cleaned.csv",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Target multiplier for dataset size (default: 3)",
    )
    parser.add_argument(
        "--format-hf",
        action="store_true",
        help="Print output in HuggingFace fine-tuning format",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't dump all verses",
    )
    args = parser.parse_args()

    if args.format_hf:
        pairs = augment_dataset_with_artists(args.csv, args.multiplier)
        formatted = format_for_hf_training(pairs)
        if args.stats_only:
            print(f"Original tracks: {len(_load_csv(args.csv))}")
            print(f"Augmented verses (with artist): {len(pairs)}")
            print(f"Formatted HF samples: {len(formatted)}")
        else:
            for entry in formatted:
                print(entry)
                print()
    else:
        verses = augment_dataset(args.csv, args.multiplier)
        if args.stats_only:
            print(f"Original tracks: {len(_load_csv(args.csv))}")
            print(f"Augmented verses: {len(verses)}")
        else:
            for v in verses:
                print(v)
                print("---")
