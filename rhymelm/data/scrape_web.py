"""Aggressive multi-source lyrics scraper. No API keys needed.

Hits every free lyrics site, caches results, builds a massive corpus.
Sources: lyrics.com, AZLyrics, songlyrics.com, Genius (direct scrape)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("lyrics_cache")
CACHE_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RAP_ARTISTS = [
    "Eminem", "Drake", "Kendrick Lamar", "2Pac", "Nas",
    "Jay-Z", "Kanye West", "Lil Wayne", "J. Cole", "Nicki Minaj",
    "Future", "Travis Scott", "21 Savage", "Tyler the Creator",
    "A$AP Rocky", "Pusha T", "Freddie Gibbs", "JID", "Denzel Curry",
    "Joey Badass", "Logic", "Mac Miller", "Kid Cudi", "Childish Gambino",
    "MF DOOM", "Rakim", "Notorious B.I.G.", "Ice Cube", "Snoop Dogg", "50 Cent",
]

DELAY = 1.5


def _get(url, timeout=15):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except:
        pass
    return None


def _clean(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\[.*\]$", line):
            continue
        if any(x in line for x in ["Embed", "You might also like", "See ", "Get tickets",
                                     "Contributors", "Translations", "pyright", "writer"]):
            continue
        if re.match(r"^\d+$", line):
            continue
        if len(line) < 3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _slug(name):
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    return s


def _slug_under(name):
    s = name.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"[\s]+", "_", s)
    return s


# ═══════════════════════════════════════
# SOURCE 1: lyrics.com
# ═══════════════════════════════════════
def _scrape_lyricscom(artist, max_songs=50):
    songs = []
    # Try multiple URL patterns
    for url_pattern in [
        f"https://www.lyrics.com/artist/{quote(artist)}",
        f"https://www.lyrics.com/artist.php?name={quote_plus(artist)}&aid=0&o=1",
    ]:
        html = _get(url_pattern)
        if not html or "no results" in html.lower():
            continue

        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/lyric/") or href.startswith("https://www.lyrics.com/lyric/"):
                title = a.get_text(strip=True)
                if title and len(title) > 1 and not title.isdigit():
                    full = href if href.startswith("http") else "https://www.lyrics.com" + href
                    links.append((title, full))

        if links:
            break

    log.info(f"  lyrics.com: {len(links)} links for '{artist}'")

    for title, link in links[:max_songs]:
        time.sleep(DELAY)
        page = _get(link)
        if not page:
            continue
        s = BeautifulSoup(page, "html.parser")
        body = s.find("pre", id="lyric-body-text")
        if not body:
            body = s.find("div", class_="lyric-body")
        if body:
            text = body.get_text(separator="\n").strip()
            cleaned = _clean(text)
            if len(cleaned) > 100:
                songs.append({"artist": artist, "title": title, "lyrics": cleaned})
                log.info(f"    + {title} ({len(cleaned)} chars)")
    return songs


# ═══════════════════════════════════════
# SOURCE 2: AZLyrics.com
# ═══════════════════════════════════════
def _scrape_azlyrics(artist, max_songs=50):
    songs = []
    slug = re.sub(r"[^a-z0-9]", "", artist.lower())
    if not slug:
        return songs
    url = f"https://www.azlyrics.com/{slug[0]}/{slug}.html"
    html = _get(url)
    if not html:
        return songs

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/lyrics/" in href and href.endswith(".html"):
            if not href.startswith("http"):
                href = "https://www.azlyrics.com" + href
            title = a.get_text(strip=True)
            if title:
                links.append((title, href))

    log.info(f"  azlyrics: {len(links)} links for '{artist}'")

    for title, link in links[:max_songs]:
        time.sleep(DELAY + 1)  # AZ is strict
        page = _get(link)
        if not page:
            continue
        s = BeautifulSoup(page, "html.parser")
        # Lyrics are in a div with no class/id after the comment "Usage of azlyrics"
        divs = s.find_all("div", class_=False, id=False)
        for div in divs:
            text = div.get_text(separator="\n").strip()
            if len(text) > 200 and "\n" in text and "azlyrics" not in text.lower()[:100]:
                cleaned = _clean(text)
                if len(cleaned) > 100:
                    songs.append({"artist": artist, "title": title, "lyrics": cleaned})
                    log.info(f"    + {title} ({len(cleaned)} chars)")
                    break
    return songs


# ═══════════════════════════════════════
# SOURCE 3: songlyrics.com
# ═══════════════════════════════════════
def _scrape_songlyrics(artist, max_songs=50):
    songs = []
    slug = _slug(artist)
    # Try multiple URL patterns
    for url in [
        f"https://www.songlyrics.com/{slug}-lyrics/",
        f"https://www.songlyrics.com/{quote_plus(artist.lower())}-lyrics/",
    ]:
        html = _get(url)
        if html and "not found" not in html.lower()[:500]:
            break
    else:
        return songs

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        title = a.get_text(strip=True)
        if title and "songlyrics.com" in href and href.endswith("-lyrics/") and slug in href:
            if href != url:  # not the artist page itself
                links.append((title, href))

    log.info(f"  songlyrics: {len(links)} links for '{artist}'")

    for title, link in links[:max_songs]:
        time.sleep(DELAY)
        page = _get(link)
        if not page:
            continue
        s = BeautifulSoup(page, "html.parser")
        div = s.find("p", id="songLyricsDiv")
        if div:
            text = div.get_text(separator="\n").strip()
            if "do not have the lyrics" not in text.lower():
                cleaned = _clean(text)
                if len(cleaned) > 100:
                    songs.append({"artist": artist, "title": title, "lyrics": cleaned})
                    log.info(f"    + {title} ({len(cleaned)} chars)")
    return songs


# ═══════════════════════════════════════
# SOURCE 4: Genius direct scrape
# ═══════════════════════════════════════
def _scrape_genius(artist, max_songs=50):
    songs = []
    slug = _slug(artist)

    # Try artist page
    url = f"https://genius.com/artists/{slug}"
    html = _get(url)
    if not html:
        # Try search
        url = f"https://genius.com/search?q={quote_plus(artist)}"
        html = _get(url)
    if not html:
        return songs

    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "genius.com/" in href and "-lyrics" in href:
            title = a.get_text(strip=True)
            if title and len(title) > 2:
                links.append((title, href))

    # Dedupe
    seen = set()
    unique = []
    for t, l in links:
        if l not in seen:
            seen.add(l)
            unique.append((t, l))
    links = unique

    log.info(f"  genius: {len(links)} links for '{artist}'")

    for title, link in links[:max_songs]:
        time.sleep(DELAY)
        page = _get(link)
        if not page:
            continue
        s = BeautifulSoup(page, "html.parser")
        containers = s.find_all("div", attrs={"data-lyrics-container": "true"})
        if containers:
            parts = []
            for c in containers:
                for br in c.find_all("br"):
                    br.replace_with("\n")
                parts.append(c.get_text(separator="\n"))
            text = "\n".join(parts)
            cleaned = _clean(text)
            if len(cleaned) > 100:
                songs.append({"artist": artist, "title": title, "lyrics": cleaned})
                log.info(f"    + {title} ({len(cleaned)} chars)")
    return songs


# ═══════════════════════════════════════
# MASTER
# ═══════════════════════════════════════
def scrape_artist_songs(artist, max_songs=50):
    cache_file = CACHE_DIR / f"{_slug_under(artist)}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        if cached:
            log.info(f"Cache: {artist} ({len(cached)} songs)")
            return cached

    log.info(f"\n{'='*40}\nScraping: {artist}\n{'='*40}")
    all_songs = []
    seen = set()

    for fn, name in [
        (_scrape_lyricscom, "lyrics.com"),
        (_scrape_azlyrics, "azlyrics"),
        (_scrape_songlyrics, "songlyrics"),
        (_scrape_genius, "genius"),
    ]:
        try:
            results = fn(artist, max_songs=max_songs)
            for s in results:
                key = re.sub(r"[^\w]", "", s["title"].lower())
                if key not in seen and len(s["lyrics"]) > 100:
                    seen.add(key)
                    all_songs.append(s)
        except Exception as e:
            log.warning(f"  {name}: {e}")

        if len(all_songs) >= max_songs:
            break

    log.info(f"TOTAL {artist}: {len(all_songs)} songs")

    with open(cache_file, "w") as f:
        json.dump(all_songs, f, indent=2)

    return all_songs


def scrape_all(artists=None, max_per=50):
    artists = artists or RAP_ARTISTS
    all_songs = []
    for a in artists:
        songs = scrape_artist_songs(a, max_per)
        all_songs.extend(songs)
    log.info(f"\nGRAND TOTAL: {len(all_songs)} songs from {len(artists)} artists")
    return all_songs


if __name__ == "__main__":
    songs = scrape_all(max_per=50)
    print(f"\n{len(songs)} songs scraped")
    for a in RAP_ARTISTS:
        c = sum(1 for s in songs if s["artist"] == a)
        if c > 0:
            print(f"  {a}: {c}")
