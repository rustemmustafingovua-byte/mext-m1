"""Predictive zero-RAM RAG for MEXT-M1."""

from .embed import HashingEmbedder
from .index import DocChunk, RagIndex, chunk_text, iter_chunks
from .prefetcher import StreamingQueryPrefetcher

__all__ = [
    "DocChunk",
    "HashingEmbedder",
    "RagIndex",
    "StreamingQueryPrefetcher",
    "chunk_text",
    "iter_chunks",
]
