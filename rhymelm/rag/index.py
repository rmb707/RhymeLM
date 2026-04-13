"""Build a FAISS vector index over verse chunks from the lyrics corpus.

Each chunk is:
- 4-16 lines (roughly a verse or half-verse)
- Tagged with artist
- Embedded using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class VerseChunk:
    artist: str
    title: str
    text: str
    chunk_id: int


class VerseIndex:
    """FAISS-backed vector index over verse chunks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._encoder = None
        self._index = None
        self.chunks: list[VerseChunk] = []

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading encoder: {self.model_name}")
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def add_chunks(self, chunks: list[VerseChunk], batch_size: int = 64):
        """Embed and index a batch of verse chunks."""
        import faiss

        texts = [c.text for c in chunks]
        print(f"Embedding {len(texts)} chunks...")
        vectors = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = vectors.shape[1]
        if self._index is None:
            # Inner product on normalized vectors = cosine similarity
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self.chunks.extend(chunks)
        print(f"Index size: {self._index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 5, artist_filter: str | None = None):
        """Retrieve top-k chunks matching the query. Optionally filter by artist."""
        import faiss

        q_vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")

        # Overfetch if filtering so we have enough after filter
        fetch_k = top_k * 10 if artist_filter else top_k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(q_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if artist_filter and chunk.artist.lower() != artist_filter.lower():
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: str):
        import faiss

        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, f"{path}/index.faiss")
        with open(f"{path}/chunks.json", "w") as f:
            json.dump([asdict(c) for c in self.chunks], f)
        print(f"Saved index to {path}")

    @classmethod
    def load(cls, path: str) -> "VerseIndex":
        import faiss

        self = cls()
        self._index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/chunks.json") as f:
            self.chunks = [VerseChunk(**c) for c in json.load(f)]
        print(f"Loaded index: {self._index.ntotal} vectors, {len(self.chunks)} chunks")
        return self


def _extract_chunks(text: str, artist: str, title: str, min_lines: int = 4, max_lines: int = 16) -> list[VerseChunk]:
    """Split lyrics into overlapping chunks."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Drop section headers and garbage
    lines = [l for l in lines if not re.match(r"^\[.*\]$", l) and len(l) >= 3]

    chunks = []
    chunk_id = 0
    # Sliding window with 50% overlap
    stride = max_lines // 2
    for i in range(0, len(lines), stride):
        window = lines[i : i + max_lines]
        if len(window) >= min_lines:
            chunks.append(VerseChunk(
                artist=artist,
                title=title,
                text="\n".join(window),
                chunk_id=chunk_id,
            ))
            chunk_id += 1
    return chunks


def build_index_from_csv(
    csv_path: str,
    lyrics_col: str = "lyrics_clean",
    artist_col: str = "artist",
    title_col: str | None = None,
) -> VerseIndex:
    """Build a vector index from a CSV file of lyrics."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    all_chunks = []
    for idx, row in df.iterrows():
        text = row.get(lyrics_col, "")
        if not isinstance(text, str) or len(text) < 50:
            continue
        artist = row.get(artist_col, "Unknown")
        title = row.get(title_col, f"Track {idx}") if title_col else f"Track {idx}"
        chunks = _extract_chunks(text, artist, title)
        all_chunks.extend(chunks)

    print(f"Extracted {len(all_chunks)} verse chunks")

    # Show per-artist counts
    from collections import Counter
    counts = Counter(c.artist for c in all_chunks)
    for artist, n in counts.most_common():
        print(f"  {artist}: {n} chunks")

    idx = VerseIndex()
    idx.add_chunks(all_chunks)
    return idx


def build_index_from_cache(cache_dir: str = "lyrics_cache") -> VerseIndex | None:
    """Build an index from scraped JSON files in lyrics_cache/."""
    import glob

    files = glob.glob(f"{cache_dir}/*.json")
    if not files:
        return None

    all_chunks = []
    for f in files:
        with open(f) as fh:
            songs = json.load(fh)
        for song in songs:
            chunks = _extract_chunks(
                song.get("lyrics", ""),
                song.get("artist", "Unknown"),
                song.get("title", "Untitled"),
            )
            all_chunks.extend(chunks)

    if not all_chunks:
        return None

    idx = VerseIndex()
    idx.add_chunks(all_chunks)
    return idx


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="lyrics_cleaned.csv")
    parser.add_argument("--output", default="rag_index")
    args = parser.parse_args()

    idx = build_index_from_csv(args.csv)
    idx.save(args.output)

    # Quick test
    print("\n--- Test query: 'making money on the streets' ---")
    for chunk, score in idx.search("making money on the streets", top_k=3):
        print(f"\n[{chunk.artist}] score={score:.3f}")
        print(chunk.text[:200])
