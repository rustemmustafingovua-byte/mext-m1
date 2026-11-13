"""Dependency-free embedding for the streaming RAG prefetcher.

A hashing bag-of-words + character-trigram vectoriser. It is deterministic,
incremental-friendly (a prefix embedding costs microseconds), and needs no
model download — which matters because the whole point of MEXT is to avoid
holding another 500 MiB of weights in RAM.

Swap it for a sentence-transformer by passing any object with
``embed(text) -> np.ndarray`` into :class:`~mext_ai.rag.index.RagIndex`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np

__all__ = ["HashingEmbedder"]

_WORD_RE = re.compile(r"[a-z0-9_]+")
_SPACE_RE = re.compile(r"\s+")


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    # ------------------------------------------------------------------
    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vector
        lower = text.lower()

        # Whole tokens: strongest signal.
        for token in _WORD_RE.findall(lower):
            self._add(vector, "w:" + token, 1.0)
            if len(token) > 4:
                # Word prefixes let a half-typed word still hit the right bucket.
                for cut in (4, 6, 8):
                    if len(token) > cut:
                        self._add(vector, "p:" + token[:cut], 0.45)

        # Character trigrams: robust to typos and partial words.
        compact = _SPACE_RE.sub(" ", lower).strip()
        if len(compact) >= 3:
            for i in range(len(compact) - 2):
                self._add(vector, "c:" + compact[i : i + 3], 0.30)

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        rows = [self.embed(t) for t in texts]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    # ------------------------------------------------------------------
    def _add(self, vector: np.ndarray, token: str, weight: float) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign * weight
