"""
Web-based lyrics scraper for RhymeLM.

Scrapes lyrics from free web sources (no API keys needed):
  1. www.lyrics.com
  2. www.songlyrics.com

Includes rate limiting, caching, error handling, and lyrics cleaning.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from urllib.parse import quote, quote_plus

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent / "lyrics_cache"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_DELAY = 2  # seconds between requests

RAP_ARTISTS: list[str] = [
    "Eminem",
    "Drake",
    "Kendrick Lamar",
    "2Pac",
    "Nas",
    "Jay-Z",
    "Kanye West",
    "Lil Wayne",
    "J. Cole",
    "Nicki Minaj",
    "Future",
    "Travis Scott",
    "21 Savage",
    "Megan Thee Stallion",
    "Tyler the Creator",
    "A$AP Rocky",
    "Pusha T",
    "Freddie Gibbs",
    "JID",
    "Denzel Curry",
    "Joey Bada$$",
    "Logic",
    "Mac Miller",
    "Kid Cudi",
    "Childish Gambino",
    "Run the Jewels",
    "MF DOOM",
    "Rakim",
    "Biggie",
    "Wu-Tang Clan",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, **kwargs) -> requests.Response | None:
    """Issue a GET request with standard headers and error handling."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, **kwargs)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.debug("Request failed for %s: %s", url, exc)
        return None


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug (lowercase, hyphens)."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def clean_lyrics(raw: str) -> str:
    """Clean raw lyrics text.

    - Remove section headers like [Verse 1], [Chorus], [Bridge], etc.
    - Remove ad-related lines and junk.
    - Collapse excessive blank lines.
    - Strip leading/trailing whitespace.
    """
    if not raw:
        return ""

    lines = raw.splitlines()
    cleaned: list[str] = []

    # Patterns to remove
    section_header = re.compile(r"^\[.*?\]\s*$")
    ad_patterns = re.compile(
        r"(commercial|advertisement|ringtone|lyrics licensed|"
        r"lyrics provided|lyrics powered|songlyrics\.com|"
        r"lyrics\.com|all rights reserved|copyright|"
        r"embed|share|print|correct|submit)",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        # Skip section headers
        if section_header.match(stripped):
            continue
        # Skip ad / boilerplate lines
        if stripped and ad_patterns.search(stripped):
            continue
        cleaned.append(stripped)

    text = "\n".join(cleaned)
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Source 1: lyrics.com
# ---------------------------------------------------------------------------

def _lyrics_com_search(artist_name: str, max_songs: int) -> list[dict]:
    """Search lyrics.com for songs by an artist and scrape lyrics."""
    results: list[dict] = []
    search_url = f"https://www.lyrics.com/artist/{quote(artist_name)}"
    logger.info("lyrics.com: searching %s", search_url)

    resp = _get(search_url)
    if resp is None:
        # Try the search endpoint as fallback
        search_url = f"https://www.lyrics.com/serp.php?st={quote_plus(artist_name)}&stype=2"
        logger.info("lyrics.com: fallback search %s", search_url)
        resp = _get(search_url)
    if resp is None:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find song links -- they live in <a> tags inside the artist page or search results
    song_links: list[tuple[str, str]] = []

    # Artist page layout: links in .sec-lyric or .lyric-body or table rows
    for a_tag in soup.select("a[href*='/lyric/']"):
        href = a_tag.get("href", "")
        title = a_tag.get_text(strip=True)
        if href and title and len(title) > 1:
            if not href.startswith("http"):
                href = "https://www.lyrics.com" + href
            song_links.append((title, href))
        if len(song_links) >= max_songs:
            break

    # Also try table-based layout (common for artist discography pages)
    if not song_links:
        for a_tag in soup.select("td a[href]"):
            href = a_tag.get("href", "")
            title = a_tag.get_text(strip=True)
            if "/lyric/" in href and title and len(title) > 1:
                if not href.startswith("http"):
                    href = "https://www.lyrics.com" + href
                song_links.append((title, href))
            if len(song_links) >= max_songs:
                break

    logger.info("lyrics.com: found %d song links for '%s'", len(song_links), artist_name)

    for title, url in song_links[:max_songs]:
        if len(results) >= max_songs:
            break
        time.sleep(RATE_LIMIT_DELAY)
        lyrics = _scrape_lyrics_com_page(url)
        if lyrics and len(lyrics) > 50:
            results.append({
                "artist": artist_name,
                "title": title,
                "lyrics": lyrics,
            })
            logger.info("  + scraped: %s (%d chars)", title, len(lyrics))
        else:
            logger.debug("  - skipped (too short/empty): %s", title)

    return results


def _scrape_lyrics_com_page(url: str) -> str:
    """Scrape lyrics from a single lyrics.com lyric page."""
    resp = _get(url)
    if resp is None:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")

    # The lyrics body is typically in <pre id="lyric-body-text"> or
    # <div id="lyric-body-text">
    lyric_div = soup.find(id="lyric-body-text")
    if lyric_div:
        raw = lyric_div.get_text(separator="\n")
        return clean_lyrics(raw)
    return ""


# ---------------------------------------------------------------------------
# Source 2: songlyrics.com
# ---------------------------------------------------------------------------

def _songlyrics_com_search(artist_name: str, max_songs: int) -> list[dict]:
    """Search songlyrics.com for songs by an artist and scrape lyrics."""
    results: list[dict] = []
    slug = _slugify(artist_name)
    artist_url = f"https://www.songlyrics.com/{slug}-lyrics/"
    logger.info("songlyrics.com: trying %s", artist_url)

    resp = _get(artist_url)
    if resp is None:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")

    song_links: list[tuple[str, str]] = []

    # Song list is usually in <table class="tracklist"> or similar
    for a_tag in soup.select("a[href*='songlyrics.com']"):
        href = a_tag.get("href", "")
        title = a_tag.get_text(strip=True)
        # Song pages look like /artist/song-title-lyrics.html
        if (
            title
            and len(title) > 1
            and href.endswith("-lyrics.html")
            and slug in href.lower()
        ):
            song_links.append((title, href))
        if len(song_links) >= max_songs:
            break

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_links: list[tuple[str, str]] = []
    for title, href in song_links:
        if href not in seen:
            seen.add(href)
            unique_links.append((title, href))

    logger.info("songlyrics.com: found %d song links for '%s'", len(unique_links), artist_name)

    for title, url in unique_links[:max_songs]:
        if len(results) >= max_songs:
            break
        time.sleep(RATE_LIMIT_DELAY)
        lyrics = _scrape_songlyrics_com_page(url)
        if lyrics and len(lyrics) > 50:
            results.append({
                "artist": artist_name,
                "title": title,
                "lyrics": lyrics,
            })
            logger.info("  + scraped: %s (%d chars)", title, len(lyrics))
        else:
            logger.debug("  - skipped (too short/empty): %s", title)

    return results


def _scrape_songlyrics_com_page(url: str) -> str:
    """Scrape lyrics from a single songlyrics.com page."""
    resp = _get(url)
    if resp is None:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")

    lyric_div = soup.find(id="songLyricsDiv")
    if lyric_div:
        raw = lyric_div.get_text(separator="\n")
        # songlyrics.com sometimes puts "We do not have the lyrics for ..."
        if "do not have the lyrics" in raw.lower():
            return ""
        return clean_lyrics(raw)
    return ""


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(artist_name: str) -> Path:
    safe = re.sub(r"[^\w\s-]", "_", artist_name).strip()
    return CACHE_DIR / f"{safe}.json"


def _load_cache(artist_name: str) -> list[dict] | None:
    path = _cache_path(artist_name)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) > 0:
                logger.info("Loaded %d cached songs for '%s'", len(data), artist_name)
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache read failed for '%s': %s", artist_name, exc)
    return None


def _save_cache(artist_name: str, songs: list[dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(artist_name)
    try:
        path.write_text(json.dumps(songs, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Cached %d songs for '%s' -> %s", len(songs), artist_name, path)
    except OSError as exc:
        logger.warning("Cache write failed for '%s': %s", artist_name, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_artist_songs(artist_name: str, max_songs: int = 50) -> list[dict]:
    """Scrape lyrics for an artist from web sources.

    Tries sources in order, falling back if one yields no results.
    Results are cached to disk.

    Args:
        artist_name: Name of the artist to search for.
        max_songs: Maximum number of songs to collect.

    Returns:
        List of dicts with keys: ``artist``, ``title``, ``lyrics``.
    """
    # Check cache first
    cached = _load_cache(artist_name)
    if cached is not None:
        return cached[:max_songs]

    songs: list[dict] = []

    # Source 1: lyrics.com
    try:
        songs = _lyrics_com_search(artist_name, max_songs)
    except Exception as exc:
        logger.warning("lyrics.com failed for '%s': %s", artist_name, exc)

    # Source 2: songlyrics.com (fallback or supplement)
    if len(songs) < max_songs:
        remaining = max_songs - len(songs)
        try:
            extra = _songlyrics_com_search(artist_name, remaining)
            # Avoid duplicates by title
            existing_titles = {s["title"].lower() for s in songs}
            for song in extra:
                if song["title"].lower() not in existing_titles:
                    songs.append(song)
                    existing_titles.add(song["title"].lower())
                if len(songs) >= max_songs:
                    break
        except Exception as exc:
            logger.warning("songlyrics.com failed for '%s': %s", artist_name, exc)

    # Cache whatever we got (even partial results are useful)
    if songs:
        _save_cache(artist_name, songs)

    logger.info("Total songs scraped for '%s': %d", artist_name, len(songs))
    return songs


def build_large_dataset(
    artists: list[str] | None = None,
    max_per_artist: int = 50,
) -> str:
    """Scrape lyrics for multiple artists and return a combined corpus.

    Args:
        artists: List of artist names. Defaults to ``RAP_ARTISTS``.
        max_per_artist: Max songs to scrape per artist.

    Returns:
        A single string with all lyrics concatenated, separated by newlines.
    """
    if artists is None:
        artists = RAP_ARTISTS

    all_lyrics: list[str] = []
    total_songs = 0

    for i, artist in enumerate(artists, 1):
        logger.info("=== [%d/%d] Scraping: %s ===", i, len(artists), artist)
        songs = scrape_artist_songs(artist, max_per_artist)
        for song in songs:
            all_lyrics.append(song["lyrics"])
        total_songs += len(songs)
        logger.info("Running total: %d songs", total_songs)

    corpus = "\n\n".join(all_lyrics)
    logger.info(
        "Dataset complete: %d artists, %d songs, %d characters",
        len(artists),
        total_songs,
        len(corpus),
    )
    return corpus


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape rap lyrics from the web")
    parser.add_argument(
        "--artist",
        type=str,
        default=None,
        help="Scrape a single artist (default: scrape all hardcoded artists)",
    )
    parser.add_argument(
        "--max-songs",
        type=int,
        default=50,
        help="Maximum songs per artist (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write combined corpus to this file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.artist:
        songs = scrape_artist_songs(args.artist, args.max_songs)
        print(f"Scraped {len(songs)} songs for {args.artist}")
        for s in songs:
            print(f"  - {s['title']} ({len(s['lyrics'])} chars)")
    else:
        corpus = build_large_dataset(max_per_artist=args.max_songs)
        print(f"\nCorpus size: {len(corpus):,} characters")
        if args.output:
            Path(args.output).write_text(corpus, encoding="utf-8")
            print(f"Written to {args.output}")
