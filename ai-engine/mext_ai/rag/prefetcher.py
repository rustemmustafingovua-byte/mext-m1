"""Streaming query prefetcher.

The user types; on every keystroke burst we embed the *prefix*, find the closest
IVF partitions and asynchronously read those cluster files from SSD into RAM.
When the user finally hits Enter the search runs against a warm page cache plus
a warm cluster cache, and we score whether the prefetch actually helped.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .index import RagIndex

log = logging.getLogger("mext_ai.rag.prefetch")

__all__ = ["StreamingQueryPrefetcher"]


class StreamingQueryPrefetcher:
    def __init__(self, index: RagIndex, nprobe: int = 3) -> None:
        self.index = index
        self.nprobe = max(1, int(nprobe))
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mext-rag-prefetch")
        self._futures: list[Any] = []
        self.prefetched_clusters: set[int] = set()
        self.prefetched_docs: set[str] = set()
        self.prefetches = 0
        self.queries = 0
        self.hits = 0
        self.warm_bytes = 0
        self.latency_ms = 0.0
        self.prefetch_ms = 0.0
        self.last_query = ""
        self.last_prefix = ""
        self.closed = False

    # ------------------------------------------------------------------
    def feed(self, prefix: str) -> dict:
        """Called for every debounced keystroke burst."""
        prefix = (prefix or "").strip()
        if len(prefix) < 2:
            return {
                "available": True,
                "clusters": [],
                "docs": [],
                "prefix_len": len(prefix),
                "skipped": True,
            }
        started = time.perf_counter()
        vector = self.index.embedder.embed(prefix)
        clusters = self.index.nearest_clusters(vector, self.nprobe)
        with self._lock:
            self.prefetches += 1
            self.last_prefix = prefix
            self.prefetched_clusters.update(clusters)
            self._futures = [f for f in self._futures if not f.done()][-4:]
            future = self._pool.submit(self._warm, clusters)
            self._futures.append(future)
        self.prefetch_ms = (time.perf_counter() - started) * 1000.0
        return {
            "available": True,
            "clusters": clusters,
            "docs": [],
            "prefix_len": len(prefix),
            "latency_ms": round(self.prefetch_ms, 2),
            "warm_bytes": self.index.warm_cache_bytes(),
            "nprobe": self.nprobe,
        }

    def _warm(self, clusters: list[int]) -> None:
        try:
            result = self.index.warm_clusters(clusters)
            with self._lock:
                self.warm_bytes += int(result.get("bytes", 0))
                for doc in result.get("docs", []):
                    self.prefetched_docs.add(doc.get("path", ""))
        except Exception as exc:  # noqa: BLE001
            log.debug("cluster warm failed: %s", exc)

    # ------------------------------------------------------------------
    def commit(self, query: str, k: int = 8) -> dict:
        query = (query or "").strip()
        if not query:
            return {"available": True, "results": [], "latency_ms": 0.0, "prefetch_hit": False}
        # Let in-flight warmups finish, but never block the UI for long.
        for future in list(self._futures):
            try:
                future.result(timeout=0.4)
            except Exception:  # noqa: BLE001
                pass
        started = time.perf_counter()
        vector = self.index.embedder.embed(query)
        results = self.index.search(vector, k=k)
        latency = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self.queries += 1
            self.last_query = query
            prefetched_for_results = sum(
                1 for r in results if int(r.get("cluster", -1)) in self.prefetched_clusters
            )
            hit = bool(results) and prefetched_for_results >= max(1, len(results) // 2)
            if hit:
                self.hits += 1
            rate = self.hits / self.queries if self.queries else 0.0
            docs = len(self.prefetched_docs)
            clusters = sorted(self.prefetched_clusters)
        self.latency_ms = latency
        return {
            "available": True,
            "query": query,
            "results": results,
            "latency_ms": round(latency, 2),
            "prefetch_hit": bool(hit),
            "prefetch_hit_rate": round(rate, 4),
            "prefetched_clusters": clusters,
            "prefetched_docs": docs,
        }

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        with self._lock:
            rate = self.hits / self.queries if self.queries else 0.0
            return {
                "prefetch_hit_rate": round(rate, 4),
                "prefetches": self.prefetches,
                "queries": self.queries,
                "hits": self.hits,
                "warm_bytes": int(self.warm_bytes),
                "warm_cache_bytes": self.index.warm_cache_bytes(),
                "latency_ms": round(self.latency_ms, 2),
                "prefetch_ms": round(self.prefetch_ms, 2),
                "last_query": self.last_query,
                "last_prefix": self.last_prefix,
                "prefetched_clusters": sorted(self.prefetched_clusters),
            }

    def close(self) -> None:
        self.closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
