"""Fetch clean lyrics from Genius API.

Uses lyricsgenius to pull lyrics per artist, then cleans and deduplicates.
Requires a Genius API token — get one at https://genius.com/api-clients
"""

import os
import re
import json
import time
from pathlib import Path

import pandas as pd


def scrape_artist(
    token: str,
    artist_name: str,
    max_songs: int = 50,
    sleep_between: float = 0.5,
) -> list[dict]:
    """Fetch lyrics for an artist from Genius."""
    import lyricsgenius

    genius = lyricsgenius.Genius(
        token,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Demo)", "(Skit)"],
        remove_section_headers=True,
        verbose=False,
        timeout=15,
        retries=3,
    )

    print(f"  Fetching {artist_name} ({max_songs} songs)...")
    artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    if artist is None:
        print(f"  Could not find {artist_name}")
        return []

    songs = []
    for song in artist.songs:
        if song.lyrics:
            songs.append({
                "artist": artist_name,
                "title": song.title,
                "lyrics": song.lyrics,
            })
        time.sleep(sleep_between)

    print(f"  Got {len(songs)} songs for {artist_name}")
    return songs


def clean_lyrics(raw: str) -> str:
    """Clean a single song's raw Genius lyrics text."""
    lines = raw.splitlines()

    # Remove first line if it's the song title + "Lyrics" suffix
    if lines and "Lyrics" in lines[0]:
        lines = lines[1:]

    cleaned = []
    for line in lines:
        line = line.strip()

        # Skip empty
        if not line:
            continue

        # Skip section headers: [Verse 1], [Chorus], [Bridge], etc.
        if re.match(r"^\[.*\]$", line):
            continue

        # Skip common Genius artifacts
        skip_patterns = [
            "You might also like",
            "Get tickets as low as",
            "See .* Live",
            r"^\d+$",  # bare numbers
            "Embed$",
            r"^\d+Embed$",
            "Contributors",
            r"^\d+ Contributors",
        ]
        if any(re.search(pat, line) for pat in skip_patterns):
            continue

        # Skip lines that are just punctuation or very short noise
        if len(line) < 3 and not line[0].isalpha():
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def build_dataset(
    token: str,
    artists: list[str],
    max_songs_per_artist: int = 50,
    output_path: str = "lyrics_genius.csv",
) -> pd.DataFrame:
    """Scrape, clean, and save lyrics for multiple artists."""
    all_songs = []

    for artist in artists:
        songs = scrape_artist(token, artist, max_songs_per_artist)
        for song in songs:
            song["lyrics"] = clean_lyrics(song["lyrics"])
        all_songs.extend(songs)

    df = pd.DataFrame(all_songs)

    # Deduplicate by lyrics content (same lyrics = same song)
    df = df.drop_duplicates(subset=["lyrics"])
    # Remove songs with very little content
    df = df[df["lyrics"].str.len() > 100]

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} songs to {output_path}")
    print(f"Artists: {df['artist'].value_counts().to_dict()}")
    return df


def clean_existing_csv(
    input_path: str = "lyrics_raw.csv",
    output_path: str = "lyrics_cleaned.csv",
    lyrics_column: str = "artist_verses",
) -> pd.DataFrame:
    """Clean the existing Kaggle CSV — remove artifacts and deduplicate lines."""
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} tracks from {input_path}")

    cleaned_texts = []
    for _, row in df.iterrows():
        text = row.get(lyrics_column, "")
        if not isinstance(text, str):
            cleaned_texts.append("")
            continue
        cleaned_texts.append(clean_lyrics(text))

    df["lyrics_clean"] = cleaned_texts
    df = df[df["lyrics_clean"].str.len() > 100]

    # Line-level deduplication across songs (remove lines appearing 5+ times)
    from collections import Counter
    all_lines = []
    for text in df["lyrics_clean"]:
        for line in text.splitlines():
            line = line.strip()
            if line:
                all_lines.append(line)

    line_counts = Counter(all_lines)
    spam_lines = {line for line, count in line_counts.items() if count >= 5}
    print(f"Removing {len(spam_lines)} spam lines (appearing 5+ times)")

    def remove_spam(text):
        return "\n".join(
            line for line in text.splitlines()
            if line.strip() and line.strip() not in spam_lines
        )

    df["lyrics_clean"] = df["lyrics_clean"].apply(remove_spam)
    df = df[df["lyrics_clean"].str.len() > 100]

    df.to_csv(output_path, index=False)

    total_lines = sum(len(t.splitlines()) for t in df["lyrics_clean"])
    unique_lines = len(set(
        line.strip() for t in df["lyrics_clean"]
        for line in t.splitlines() if line.strip()
    ))
    print(f"Saved {len(df)} tracks to {output_path}")
    print(f"Lines: {total_lines:,} total, {unique_lines:,} unique ({unique_lines/total_lines:.1%} unique)")
    return df
