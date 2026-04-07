"""Auto-fetch lyrics from multiple sources.

Supports:
1. Genius API (requires token) — highest quality
2. lyricsgenius library wrapper
3. Fallback: use existing CSV data for artists already in dataset
"""

import os
import re
import time
from pathlib import Path

import pandas as pd

from rhymelm.data.genius_scraper import clean_lyrics


def fetch_artist_lyrics(
    artist_name: str,
    max_songs: int = 30,
    genius_token: str | None = None,
    cache_dir: str = "lyrics_cache",
) -> list[dict]:
    """Fetch lyrics for an artist. Uses Genius API if token available."""
    cache_path = Path(cache_dir) / f"{_safe_filename(artist_name)}.csv"

    # Check cache first
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        print(f"Loaded {len(df)} cached songs for {artist_name}")
        return df.to_dict("records")

    # Try Genius API
    token = genius_token or os.environ.get("GENIUS_TOKEN", "")
    if token:
        songs = _fetch_from_genius(artist_name, max_songs, token)
    else:
        print(f"No Genius token — checking existing dataset for {artist_name}")
        songs = _fetch_from_existing(artist_name)

    if songs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(songs)
        df.to_csv(cache_path, index=False)
        print(f"Cached {len(songs)} songs for {artist_name}")

    return songs


def _fetch_from_genius(artist_name: str, max_songs: int, token: str) -> list[dict]:
    """Fetch from Genius API using lyricsgenius."""
    try:
        import lyricsgenius
    except ImportError:
        print("lyricsgenius not installed — run: pip install lyricsgenius")
        return []

    genius = lyricsgenius.Genius(
        token,
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)", "(Demo)", "(Skit)"],
        remove_section_headers=True,
        verbose=False,
        timeout=15,
        retries=3,
    )

    print(f"Fetching {artist_name} from Genius ({max_songs} songs)...")
    artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    if artist is None:
        print(f"  Could not find {artist_name}")
        return []

    songs = []
    for song in artist.songs:
        if song.lyrics:
            cleaned = clean_lyrics(song.lyrics)
            if len(cleaned) > 100:
                songs.append({
                    "artist": artist_name,
                    "title": song.title,
                    "lyrics": cleaned,
                })
        time.sleep(0.3)

    print(f"  Got {len(songs)} songs for {artist_name}")
    return songs


def _fetch_from_existing(artist_name: str) -> list[dict]:
    """Check if artist exists in the existing training CSV."""
    for csv in ["lyrics_cleaned.csv", "lyrics_raw.csv"]:
        if not Path(csv).exists():
            continue
        df = pd.read_csv(csv)
        col = "lyrics_clean" if "lyrics_clean" in df.columns else "artist_verses"
        matches = df[df["artist"].str.lower() == artist_name.lower()]
        if len(matches) > 0:
            songs = []
            for _, row in matches.iterrows():
                text = row.get(col, "")
                if isinstance(text, str) and len(text) > 100:
                    songs.append({
                        "artist": artist_name,
                        "title": f"Track {len(songs)+1}",
                        "lyrics": clean_lyrics(text),
                    })
            print(f"  Found {len(songs)} tracks in existing dataset")
            return songs
    return []


def build_training_corpus(
    artists: list[str],
    max_songs_per_artist: int = 30,
    genius_token: str | None = None,
) -> str:
    """Fetch lyrics for multiple artists and build a training corpus."""
    all_lyrics = []
    for artist in artists:
        songs = fetch_artist_lyrics(artist, max_songs_per_artist, genius_token)
        for song in songs:
            all_lyrics.append(song["lyrics"])

    corpus = "\n\n".join(all_lyrics)
    print(f"\nTraining corpus: {len(all_lyrics)} songs, {len(corpus):,} chars")
    return corpus


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.lower())
