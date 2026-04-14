"""Per-artist style profiler.

Reads lyrics_cleaned.csv, computes a style fingerprint per artist:
- Dominant rhyme scheme (AABB / ABAB / ABBA / AAAA / irregular)
- Scheme distribution and confidence
- Average syllables per line + stddev
- Rhyme density (fraction of lines that rhyme with a neighbor)
- Contraction rate (proxy for casual / AAVE intensity)
- Signature words (top TF-IDF terms vs other artists)

Output: artist_profiles.json. Run once, commit, load at dashboard startup.

Usage:
    python -m rhymelm.rag.profiler --csv lyrics_cleaned.csv --output artist_profiles.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from rhymelm.data.phonemes import (
    get_cmu_dict,
    get_rhyme_suffix,
    line_syllable_count,
)
from rhymelm.rag.schemes import (
    classify_window,
    get_window_suffixes,
)


CONTRACTION_RE = re.compile(
    r"\b\w+'\w+\b|\b\w+in'\b|\bain't\b|\bgon'\b|\btryna\b|\bfinna\b|\bcuz\b|\bcuh\b",
    re.IGNORECASE,
)


def _clean_lines(text: str) -> list[str]:
    """Strip section headers, blank lines, and noise from raw lyrics."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^\[.*\]$", line):
            continue
        if len(line) < 3:
            continue
        lines.append(line)
    return lines


def _profile_artist(artist: str, all_lines: list[str], cmu) -> dict:
    """Compute the full style profile for one artist."""
    if not all_lines:
        return _empty_profile()

    # ── Scheme distribution via 4-line sliding windows ──
    scheme_counts: Counter[str] = Counter()
    rhyme_pair_count = 0
    total_pair_count = 0

    for i in range(0, len(all_lines) - 3):
        window = all_lines[i : i + 4]
        suffixes = get_window_suffixes(window, cmu, get_rhyme_suffix)
        scheme = classify_window(suffixes)
        if scheme != "unknown":
            scheme_counts[scheme] += 1

        # Rhyme density: fraction of adjacent line pairs that rhyme
        for a, b in zip(suffixes, suffixes[1:]):
            if a is None or b is None:
                continue
            total_pair_count += 1
            if a == b:
                rhyme_pair_count += 1

    total_classified = sum(scheme_counts.values())
    total_named = sum(scheme_counts[s] for s in ("AABB", "ABAB", "ABBA", "AAAA"))
    if total_classified > 0:
        scheme_distribution = {s: round(c / total_classified, 3) for s, c in scheme_counts.items()}
        # Pick dominant from the named patterns only (irregular doesn't count as a "style")
        named = {s: scheme_counts[s] for s in ("AABB", "ABAB", "ABBA", "AAAA") if scheme_counts[s] > 0}
        if named:
            dominant_scheme = max(named, key=named.get)
            # Two confidence measures:
            # - confidence_overall: dominant / all windows (includes irregular)
            # - confidence_named: dominant / windows that had a named pattern
            scheme_confidence = round(named[dominant_scheme] / total_classified, 3)
            scheme_confidence_named = round(named[dominant_scheme] / total_named, 3) if total_named else 0.0
        else:
            dominant_scheme = "free"
            scheme_confidence = 0.0
            scheme_confidence_named = 0.0
    else:
        scheme_distribution = {}
        dominant_scheme = "free"
        scheme_confidence = 0.0
        scheme_confidence_named = 0.0

    rhyme_density = round(rhyme_pair_count / total_pair_count, 3) if total_pair_count else 0.0

    # ── Syllable stats ──
    syllable_counts = [line_syllable_count(l, cmu) for l in all_lines]
    syllable_counts = [s for s in syllable_counts if s > 0]
    if syllable_counts:
        avg_syllables = round(statistics.mean(syllable_counts), 1)
        syllable_stddev = round(statistics.stdev(syllable_counts) if len(syllable_counts) > 1 else 0.0, 2)
    else:
        avg_syllables = 0.0
        syllable_stddev = 0.0

    # ── Char length ──
    char_lengths = [len(l) for l in all_lines]
    avg_line_length_chars = round(statistics.mean(char_lengths), 1) if char_lengths else 0

    # ── Contraction rate ──
    full_text = " ".join(all_lines)
    word_count = len(full_text.split())
    contraction_count = len(CONTRACTION_RE.findall(full_text))
    contraction_rate = round(contraction_count / word_count, 3) if word_count else 0.0

    return {
        "dominant_scheme": dominant_scheme,
        "scheme_distribution": scheme_distribution,
        "scheme_confidence": scheme_confidence,
        "scheme_confidence_named": scheme_confidence_named,
        "scheme_window_count": total_classified,
        "scheme_named_count": total_named,
        "avg_syllables": avg_syllables,
        "syllable_stddev": syllable_stddev,
        "avg_line_length_chars": avg_line_length_chars,
        "rhyme_density": rhyme_density,
        "contraction_rate": contraction_rate,
        "total_lines": len(all_lines),
        # signature_words filled in by _add_signature_words after cross-artist TF-IDF
        "signature_words": [],
    }


def _empty_profile() -> dict:
    return {
        "dominant_scheme": "free",
        "scheme_distribution": {},
        "scheme_confidence": 0.0,
        "scheme_confidence_named": 0.0,
        "scheme_window_count": 0,
        "scheme_named_count": 0,
        "avg_syllables": 0.0,
        "syllable_stddev": 0.0,
        "avg_line_length_chars": 0,
        "rhyme_density": 0.0,
        "contraction_rate": 0.0,
        "total_lines": 0,
        "signature_words": [],
    }


_SIGNATURE_STOPS = {
    # meta tokens
    "verse", "chorus", "intro", "outro", "bridge", "hook", "skit",
    "feat", "remix", "produced", "interlude", "instrumental",
    # common rap fillers
    "yeah", "yo", "uh", "ooh", "huh", "ay", "aye", "haha", "uhh",
    "woah", "oooh", "ahh", "eyh", "yuh", "hol", "hmm", "wha",
    # generic
    "like", "just", "got", "one", "know", "get", "going", "cause",
    "that", "this", "with", "they", "them", "your", "youre", "from",
    "have", "what", "when", "where", "than", "then", "now", "all",
}


def _add_signature_words(
    profiles: dict[str, dict],
    artist_text: dict[str, str],
    top_k: int = 30,
) -> None:
    """Compute signature words per artist via TF-IDF and write into profiles in place."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    artists = list(artist_text.keys())
    if not artists:
        return

    docs = [artist_text[a] for a in artists]
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        stop_words="english",
        # ASCII letters only, length 4+ (filters foreign tokens, slang noise)
        token_pattern=r"\b[a-z]{4,}\b",
        lowercase=True,
        sublinear_tf=True,
        # Term must appear in at least 2 docs (filters one-off names) but not >50% of docs
        min_df=2,
        max_df=0.5,
    )
    matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    for i, artist in enumerate(artists):
        row = matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1]
        sig: list[str] = []
        for idx in top_indices:
            if row[idx] <= 0:
                break
            word = vocab[idx]
            if word in _SIGNATURE_STOPS:
                continue
            # Must look like English (no double-consonant openings that are foreign markers)
            if not word.isascii():
                continue
            sig.append(word)
            if len(sig) >= top_k:
                break
        profiles[artist]["signature_words"] = sig


def build_profiles(
    csv_path: str,
    lyrics_col: str = "lyrics_clean",
    artist_col: str = "artist",
    verbose: bool = False,
) -> dict[str, dict]:
    """Build profiles for every artist in the CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    cmu = get_cmu_dict()

    # Group lines per artist
    lines_by_artist: dict[str, list[str]] = defaultdict(list)
    text_by_artist: dict[str, str] = defaultdict(str)
    for _, row in df.iterrows():
        text = row.get(lyrics_col, "")
        if not isinstance(text, str) or len(text) < 50:
            continue
        artist = row.get(artist_col, "Unknown")
        lines = _clean_lines(text)
        lines_by_artist[artist].extend(lines)
        text_by_artist[artist] += "\n".join(lines) + "\n"

    print(f"Found {len(lines_by_artist)} artists")
    for a, ls in sorted(lines_by_artist.items(), key=lambda x: -len(x[1])):
        print(f"  {a}: {len(ls)} lines")

    # Profile each artist
    profiles: dict[str, dict] = {}
    for artist in sorted(lines_by_artist.keys()):
        all_lines = lines_by_artist[artist]
        profile = _profile_artist(artist, all_lines, cmu)
        profiles[artist] = profile
        if verbose:
            print(f"\n=== {artist} ===")
            print(f"  Dominant scheme: {profile['dominant_scheme']} ({profile['scheme_confidence']:.0%})")
            print(f"  Distribution: {profile['scheme_distribution']}")
            print(f"  Avg syllables/line: {profile['avg_syllables']} (±{profile['syllable_stddev']})")
            print(f"  Rhyme density: {profile['rhyme_density']}")
            print(f"  Contraction rate: {profile['contraction_rate']}")

    # Add signature words via TF-IDF
    print("\nComputing signature words via TF-IDF...")
    _add_signature_words(profiles, text_by_artist)

    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="lyrics_cleaned.csv")
    parser.add_argument("--lyrics-col", default="lyrics_clean")
    parser.add_argument("--artist-col", default="artist")
    parser.add_argument("--output", default="artist_profiles.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    profiles = build_profiles(
        csv_path=args.csv,
        lyrics_col=args.lyrics_col,
        artist_col=args.artist_col,
        verbose=args.verbose,
    )

    with open(args.output, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\nSaved {len(profiles)} profiles to {args.output}")
    print("\nSummary (confidence_named = % among lines that DO follow a clean pattern):")
    for artist, p in profiles.items():
        sig = ", ".join(p["signature_words"][:6]) if p["signature_words"] else "—"
        print(f"  {artist:<20} scheme={p['dominant_scheme']:<5} "
              f"named_conf={p['scheme_confidence_named']:.0%}  "
              f"density={p['rhyme_density']:.0%}  "
              f"syl={p['avg_syllables']:>4.1f}  "
              f"sig={sig}")


if __name__ == "__main__":
    main()
