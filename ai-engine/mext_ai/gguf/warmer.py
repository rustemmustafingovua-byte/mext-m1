"""Predictive page warmer for GGUF weights.

The warmer never allocates the weights into Python/NumPy — it drives the OS
page cache directly: for each transformer layer it issues
``mmap.madvise(MADV_WILLNEED)`` over the byte range that layer's tensors occupy
and then touches one byte per chunk (default 1 MiB) so the pages are demand-
paged in before the runner needs them.

Prediction is deliberately simple: sequential lookahead (``last + 1 ..
last + lookahead``) plus reuse of recently-warmed layers, which covers the
common autoregressive decode loop (``0,1,2,...`` forward, then reuse of the
final layers while the first ones are freed).
"""

from __future__ import annotations

import mmap
import time
from collections.abc import Iterable

from .reader import GgufFile

__all__ = ["LayerPageWarmer"]

MB = 1024 * 1024
DEFAULT_CHUNK = 1 * MB

try:
    _MADV_WILLNEED = mmap.MADV_WILLNEED
except AttributeError:  # pragma: no cover - non-POSIX
    _MADV_WILLNEED = None

try:
    _PAGE_SIZE = mmap.PAGESIZE
except AttributeError:  # pragma: no cover
    _PAGE_SIZE = 4096


class LayerPageWarmer:
    """Warm (page in) the on-disk bytes of selected transformer layers."""

    def __init__(
        self,
        gguf: GgufFile,
        lookahead: int = 2,
        chunk: int = DEFAULT_CHUNK,
    ) -> None:
        self.gguf = gguf
        self.lookahead = max(1, int(lookahead))
        self.chunk = max(1, int(chunk))

        self.total_bytes = 0
        self.total_ms = 0.0
        self.total_layers = 0
        self.total_chunks = 0

        self._warmed: set[int] = set()
        self._recent: list[int] = []
        self._last_id: int | None = None

    # ------------------------------------------------------------------
    def warm_layers(self, ids: Iterable[int], lookahead: int | None = None) -> dict:
        """Page in the byte ranges for ``ids``.

        Returns ``{"bytes", "ms", "MBps", "layers", "chunks"}``.
        """
        if lookahead is not None:
            self.lookahead = max(1, int(lookahead))

        if self.gguf.mm is None:
            raise RuntimeError("GGUF file is closed")

        warmed: list[int] = []
        bytes_touched = 0
        chunks_touched = 0
        started = time.perf_counter()

        for raw in ids:
            layer_id = int(raw)
            if layer_id in self._warmed:
                continue
            layer_range = self.gguf.layer_range(layer_id)
            if layer_range is None:
                continue
            start, end = layer_range
            if end <= start:
                continue
            bytes_touched += end - start
            chunks_touched += self._warm_range(start, end)
            warmed.append(layer_id)
            self._mark_warmed(layer_id)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.total_bytes += bytes_touched
        self.total_ms += elapsed_ms
        self.total_layers += len(warmed)
        self.total_chunks += chunks_touched

        return {
            "bytes": bytes_touched,
            "ms": round(elapsed_ms, 3),
            "MBps": round((bytes_touched / 1e6) / (elapsed_ms / 1000.0), 2) if elapsed_ms > 0 else 0.0,
            "layers": warmed,
            "chunks": chunks_touched,
        }

    def _warm_range(self, start: int, end: int) -> int:
        mm = self.gguf.mm
        assert mm is not None

        # Hint the kernel that the whole range will be needed soon.
        if _MADV_WILLNEED is not None:
            try:
                aligned_start = start - (start % _PAGE_SIZE)
                aligned_len = (end - aligned_start + _PAGE_SIZE - 1) // _PAGE_SIZE * _PAGE_SIZE
                mm.madvise(_MADV_WILLNEED, aligned_start, aligned_len)
            except (OSError, ValueError):
                pass

        # Touch one byte per chunk so the pages are actually faulted in.
        touched = 0
        for off in range(start, end, self.chunk):
            _ = mm[off]
            touched += 1
        return touched

    def _mark_warmed(self, layer_id: int) -> None:
        self._warmed.add(layer_id)
        self._recent.append(layer_id)
        self._last_id = layer_id
        if len(self._recent) > 16:
            self._recent.pop(0)

    # ------------------------------------------------------------------
    def predict_next(self, lookahead: int | None = None) -> list[int]:
        """Sequential lookahead plus recent-layer reuse.

        Before anything has been warmed this predicts the first ``lookahead``
        layers; afterwards it predicts ``last+1 .. last+lookahead`` and appends
        recently-warmed layers (deduplicated) to anticipate reuse.
        """
        lookahead = self.lookahead if lookahead is None else max(1, int(lookahead))
        if self._last_id is None:
            return list(range(lookahead))

        predicted = list(range(self._last_id + 1, self._last_id + 1 + lookahead))
        for recent in reversed(self._recent):
            if recent not in predicted:
                predicted.append(recent)
        return predicted

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        total_seconds = self.total_ms / 1000.0
        return {
            "bytes": self.total_bytes,
            "ms": round(self.total_ms, 3),
            "MBps": round((self.total_bytes / 1e6) / total_seconds, 2) if total_seconds > 0 else 0.0,
            "layers_warmed": self.total_layers,
            "chunks": self.total_chunks,
            "lookahead": self.lookahead,
            "warmed": sorted(self._warmed),
            "last_layer": self._last_id,
        }
