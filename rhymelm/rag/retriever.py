"""High-level retriever: query the vector index with artist filter."""

from dataclasses import dataclass

from rhymelm.rag.index import VerseIndex, VerseChunk


@dataclass
class RetrievalResult:
    chunks: list[VerseChunk]
    scores: list[float]

    def as_context(self, max_chars: int = 2000) -> str:
        """Format retrieved chunks as a context string for the LLM."""
        parts = []
        total = 0
        for chunk in self.chunks:
            block = f"[{chunk.artist}]\n{chunk.text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)


class Retriever:
    """Retrieval wrapper with artist filtering and diversity."""

    def __init__(self, index: VerseIndex):
        self.index = index

    def retrieve(
        self,
        query: str,
        artist: str | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """Retrieve top-k verse chunks matching query, optionally from a specific artist."""
        results = self.index.search(query, top_k=top_k, artist_filter=artist)
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        return RetrievalResult(chunks=chunks, scores=scores)

    def retrieve_diverse(
        self,
        query: str,
        artist: str | None = None,
        top_k: int = 5,
        fetch_k: int = 30,
        prefer_scheme: str | None = None,
        scheme_boost: float = 0.15,
    ) -> RetrievalResult:
        """Retrieve with MMR-style diversity and optional scheme preference.

        If `prefer_scheme` is given, chunks whose `rhyme_scheme` matches receive
        a `scheme_boost` additive bonus to their similarity score before re-ranking.
        This pulls real examples of the target pattern into the in-context set
        so the LLM has a stylistic anchor to imitate.
        """
        # Overfetch with a wider net when re-ranking by scheme
        results = self.index.search(query, top_k=fetch_k, artist_filter=artist)

        if not results:
            return RetrievalResult(chunks=[], scores=[])

        # Apply scheme boost
        if prefer_scheme:
            results = [
                (chunk, score + (scheme_boost if chunk.rhyme_scheme == prefer_scheme else 0.0))
                for chunk, score in results
            ]
            results.sort(key=lambda r: r[1], reverse=True)

        selected = [results[0]]
        seen_text = {results[0][0].text[:100]}

        for chunk, score in results[1:]:
            if len(selected) >= top_k:
                break
            prefix = chunk.text[:100]
            if prefix in seen_text:
                continue
            is_dup = any(
                _overlap(chunk.text, sel[0].text) > 0.6
                for sel in selected
            )
            if is_dup:
                continue
            selected.append((chunk, score))
            seen_text.add(prefix)

        chunks = [r[0] for r in selected]
        scores = [r[1] for r in selected]
        return RetrievalResult(chunks=chunks, scores=scores)


def _overlap(a: str, b: str) -> float:
    """Character-level Jaccard over sets of 5-grams."""
    if not a or not b:
        return 0.0
    sa = {a[i:i+5] for i in range(len(a) - 4)}
    sb = {b[i:i+5] for i in range(len(b) - 4)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
