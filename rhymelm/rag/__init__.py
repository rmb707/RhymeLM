"""Retrieval-Augmented Generation for rap verses.

Instead of fine-tuning a massive model on limited data, index the lyrics corpus
and retrieve stylistically similar verses at generation time. Real bars, real style.
"""

from rhymelm.rag.index import VerseIndex, build_index_from_csv
from rhymelm.rag.retriever import Retriever
from rhymelm.rag.generator import RAGGenerator
