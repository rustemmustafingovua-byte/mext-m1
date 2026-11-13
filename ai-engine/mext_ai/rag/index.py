"""Disk-backed LanceDB index with an IVF-style cluster map.

Everything that could be large stays on SSD:

* vectors and text live in a Lance table (``IVF_PQ`` index when big enough);
* centroids live in ``centroids.npy`` and are opened with ``mmap_mode="r"``;
* per-cluster text files in ``clusters/cN.jsonl`` are the unit the prefetcher
  reads — one small sequential read per predicted cluster.

The manifest (``manifest.json``) tracks indexed paths and their primary
cluster, so a file opened by the user can be mapped to the clusters that should
be warmed next.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import HashingEmbedder

log = logging.getLogger("mext_ai.rag")

MB = 1024 * 1024

TABLE = "chunks"

TEXT_EXT = {
    ".md", ".mdx", ".markdown", ".txt", ".rst", ".org",
    ".py", ".pyi", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".c", ".h", ".cc", ".cpp", ".hpp", ".swift", ".m", ".mm",
    ".java", ".kt", ".kts", ".rb", ".php", ".cs", ".scala", ".ex", ".exs",
    ".hs", ".lua", ".zig", ".nim", ".jl", ".pl", ".r", ".sql",
    ".sh", ".bash", ".zsh", ".fish", ".toml", ".yaml", ".yml", ".json",
    ".jsonc", ".ini", ".cfg", ".conf", ".env.example", ".html", ".htm",
    ".css", ".scss", ".sass", ".less", ".xml", ".graphql", ".gql", ".proto",
    ".tex", ".pdf", ".ipynb",
}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "target", "dist", "build", ".next",
    ".nuxt", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".cache", "DerivedData", ".mext", "Library", ".Trash",
    "site-packages", ".idea", ".vscode", "Pods", ".terraform", ".gradle",
    ".dart_tool", "clusters", "db",
}


@dataclass
class DocChunk:
    path: str  # workspace-relative
    chunk: int
    text: str
    lang: str
    mtime: float


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages[:80]:
                try:
                    pages.append(page.extract_text() or "")
                except Exception:  # noqa: BLE001
                    continue
            return "\n".join(pages)
        except Exception as exc:  # noqa: BLE001
            log.debug("pdf %s skipped: %s", path, exc)
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def chunk_text(text: str, size: int = 1200, overlap: int = 150) -> Iterator[str]:
    text = text.strip()
    if not text:
        return
    if len(text) <= size:
        yield text
        return
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if not piece.strip():
            continue
        # Try not to cut a word in half.
        cut = piece.rfind("\n")
        if cut > size * 0.6:
            piece = piece[:cut]
        yield piece.strip()
        if start + size >= len(text):
            break


def iter_chunks(
    root: Path,
    max_files: int = 2000,
    max_file_bytes: int = 4 * MB,
    max_chunks: int = 20000,
    max_chars_per_file: int = 400_000,
) -> Iterator[DocChunk]:
    root = Path(root)
    files = 0
    emitted = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        for filename in sorted(filenames):
            if filename.startswith(".") and filename != ".env":
                continue
            path = Path(dirpath) / filename
            suffix = path.suffix.lower()
            if suffix not in TEXT_EXT:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0 or stat.st_size > max_file_bytes:
                continue
            # Index at most 400k chars per file: one giant log must not eat the
            # whole chunk budget.
            text = extract_text(path)[:max_chars_per_file]
            if not text.strip():
                continue
            rel = str(path.relative_to(root)).replace(os.sep, "/")
            for index, chunk in enumerate(chunk_text(text)):
                yield DocChunk(rel, index, chunk, suffix.lstrip("."), stat.st_mtime)
                emitted += 1
                if emitted >= max_chunks:
                    return
            files += 1
            if files >= max_files:
                return


# ---------------------------------------------------------------------------
# k-means (used for the IVF partition map)
# ---------------------------------------------------------------------------


def kmeans(vectors: np.ndarray, k: int, iters: int = 8, seed: int = 7) -> np.ndarray:
    n, _dim = vectors.shape
    k = max(1, min(int(k), n))
    if k == 1:
        centroid = vectors.mean(axis=0, keepdims=True)
        return centroid.astype(np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=k, replace=False)
    centroids = vectors[indices].astype(np.float32, copy=True)
    for _ in range(max(1, iters)):
        assignment = np.argmax(vectors @ centroids.T, axis=1)
        for cluster in range(k):
            mask = assignment == cluster
            if mask.any():
                centroids[cluster] = vectors[mask].mean(axis=0)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids /= np.maximum(norms, 1e-9)
    return centroids.astype(np.float32)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class RagIndex:
    def __init__(
        self,
        root: str | Path,
        workspace: str | Path,
        embedder: HashingEmbedder | None = None,
        warm_budget: int = 96 * MB,
    ) -> None:
        self.root = Path(root)
        self.workspace = Path(workspace)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "db"
        self.cluster_dir = self.root / "clusters"
        self.cluster_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.centroids_path = self.root / "centroids.npy"
        self.embedder = embedder or HashingEmbedder()
        self.dim = self.embedder.dim
        self.warm_budget = int(warm_budget)

        self._db = None
        self._table_obj = None
        self._manifest: dict = {"paths": {}, "dim": self.dim}
        self._centroids_cache: np.ndarray | None = None
        self._warm: dict[int, list[dict]] = {}
        self._warm_order: list[int] = []
        self._warm_bytes = 0
        self.index_type = "unknown"
        self._built_at: float | None = None
        self._dir_size_cache: tuple[float, int] = (0.0, 0)
        self.load_manifest()

    # ------------------------------------------------------------------
    # Connection / metadata
    # ------------------------------------------------------------------
    def connect(self) -> None:
        import lancedb

        self._db = lancedb.connect(str(self.db_path))
        self._table_obj = None

    def _table(self):
        if self._db is None:
            self.connect()
        if self._table_obj is not None:
            return self._table_obj
        try:
            self._table_obj = self._db.open_table(TABLE)
        except Exception:  # noqa: BLE001
            return None
        return self._table_obj

    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                self._manifest = json.loads(self.manifest_path.read_text())
            except Exception:  # noqa: BLE001
                self._manifest = {"paths": {}, "dim": self.dim}
        return self._manifest

    def save_manifest(self) -> None:
        self._manifest["dim"] = self.dim
        self._manifest["built_at"] = time.time()
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._manifest))
        tmp.replace(self.manifest_path)

    def indexed(self) -> bool:
        return (self._table() is not None) and bool(self._manifest.get("paths"))

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(
        self,
        force: bool = False,
        max_files: int = 2000,
        max_chunks: int = 20000,
    ) -> dict:
        started = time.perf_counter()
        chunks = list(
            iter_chunks(self.workspace, max_files=max_files, max_chunks=max_chunks)
        )
        if not chunks:
            self.index_type = "empty"
            return {**self.metrics(), "built_ms": (time.perf_counter() - started) * 1000}

        existing = self._table() is not None and not force and self._manifest.get("dim") == self.dim
        manifest_paths: dict = self._manifest.get("paths", {}) if existing else {}
        new_chunks = (
            [c for c in chunks if manifest_paths.get(c.path, {}).get("mtime") != c.mtime]
            if existing
            else chunks
        )

        if existing and not new_chunks:
            self._built_at = self._manifest.get("built_at")
            return {**self.metrics(), "updated": 0, "built_ms": (time.perf_counter() - started) * 1000}

        vectors = self._embed_chunks(new_chunks)
        centroids = self._centroids() if existing else None

        if centroids is None:
            sample = vectors
            if len(sample) > 8192:
                pick = np.linspace(0, len(sample) - 1, 8192).astype(int)
                sample = sample[pick]
            k = int(min(64, max(4, round(len(chunks) ** 0.5))))
            centroids = kmeans(sample, k)
            assignment = np.argmax(vectors @ centroids.T, axis=1)
            self._recreate_table(new_chunks, vectors, assignment)
            self._write_clusters(new_chunks, assignment, append=False)
            np.save(self.centroids_path, centroids)
            self._centroids_cache = centroids
            self.index_type = self._create_vector_index(len(new_chunks))
            self._manifest = {"paths": {}, "dim": self.dim}
        else:
            assignment = np.argmax(vectors @ centroids.T, axis=1)
            self._append_rows(new_chunks, vectors, assignment)
            self._write_clusters(new_chunks, assignment, append=True)

        for chunk, cluster in zip(new_chunks, assignment, strict=True):
            entry = self._manifest["paths"].setdefault(
                chunk.path, {"mtime": chunk.mtime, "cluster": int(cluster), "chunks": 0}
            )
            entry["mtime"] = chunk.mtime
            entry["cluster"] = int(cluster)
            entry["chunks"] = int(entry.get("chunks", 0)) + 1

        self.save_manifest()
        self._built_at = time.time()
        return {
            **self.metrics(),
            "updated": len(new_chunks),
            "built_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _embed_chunks(self, chunks: list[DocChunk]) -> np.ndarray:
        rows = np.zeros((len(chunks), self.dim), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            rows[i] = self.embedder.embed(chunk.text)
        return rows

    def _schema(self):
        import pyarrow as pa

        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("path", pa.string()),
                pa.field("chunk", pa.int32()),
                pa.field("lang", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.dim)),
                pa.field("cluster", pa.int32()),
            ]
        )

    def _rows(self, chunks: list[DocChunk], vectors: np.ndarray, assignment: np.ndarray) -> list[dict]:
        rows = []
        for chunk, vector, cluster in zip(chunks, vectors, assignment, strict=True):
            rows.append(
                {
                    "id": f"{chunk.path}#{chunk.chunk}",
                    "path": chunk.path,
                    "chunk": int(chunk.chunk),
                    "lang": chunk.lang,
                    "text": chunk.text,
                    "vector": np.asarray(vector, dtype=np.float32).tolist(),
                    "cluster": int(cluster),
                }
            )
        return rows

    def _recreate_table(self, chunks: list[DocChunk], vectors: np.ndarray, assignment: np.ndarray) -> None:
        assert self._db is not None
        schema = self._schema()
        try:
            self._db.drop_table(TABLE, ignore_missing=True)
        except Exception:  # noqa: BLE001
            pass
        self._table_obj = None
        rows = self._rows(chunks, vectors, assignment)
        batch = rows[:512]
        table = self._db.create_table(TABLE, data=batch, schema=schema)
        for start in range(512, len(rows), 512):
            table.add(rows[start : start + 512])
        self._table_obj = table

    def _append_rows(self, chunks: list[DocChunk], vectors: np.ndarray, assignment: np.ndarray) -> None:
        table = self._table()
        if table is None:
            self._recreate_table(chunks, vectors, assignment)
            return
        rows = self._rows(chunks, vectors, assignment)
        for start in range(0, len(rows), 512):
            table.add(rows[start : start + 512])

    def _create_vector_index(self, n_rows: int) -> str:
        table = self._table()
        if table is None:
            return "none"
        if n_rows < 512:
            return "flat"
        partitions = int(min(256, max(4, round(n_rows ** 0.5))))
        sub_vectors = 48 if self.dim % 48 == 0 else 32 if self.dim % 32 == 0 else 8
        try:
            table.create_index(
                metric="cosine",
                num_partitions=partitions,
                num_sub_vectors=sub_vectors,
                index_type="IVF_PQ",
                vector_column_name="vector",
            )
            return "IVF_PQ"
        except Exception as exc:  # noqa: BLE001
            log.warning("IVF_PQ index failed (%s); using flat search", exc)
            try:
                table.create_index(metric="cosine", vector_column_name="vector")
                return "IVF_FLAT"
            except Exception:  # noqa: BLE001
                return "flat"

    def _write_clusters(self, chunks: list[DocChunk], assignment: np.ndarray, append: bool) -> None:
        grouped: dict[int, list[dict]] = {}
        for chunk, cluster in zip(chunks, assignment, strict=True):
            grouped.setdefault(int(cluster), []).append(
                {"path": chunk.path, "chunk": chunk.chunk, "text": chunk.text, "lang": chunk.lang}
            )
        for cluster, rows in grouped.items():
            path = self.cluster_dir / f"c{cluster}.jsonl"
            mode = "a" if append and path.exists() else "w"
            with path.open(mode, encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------
    # Centroids / prefetch payload
    # ------------------------------------------------------------------
    def _centroids(self) -> np.ndarray | None:
        if self._centroids_cache is not None:
            return self._centroids_cache
        if not self.centroids_path.exists():
            return None
        self._centroids_cache = np.load(self.centroids_path, mmap_mode="r")  # type: ignore[assignment]
        return self._centroids_cache

    def nearest_clusters(self, vector: np.ndarray, nprobe: int = 3) -> list[int]:
        centroids = self._centroids()
        if centroids is None or len(centroids) == 0:
            return []
        # memmap slices materialise only the pages actually touched
        sims = np.asarray(centroids) @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-sims)[: max(1, int(nprobe))]
        return [int(i) for i in order]

    def warm_clusters(self, clusters: Iterable[int]) -> dict:
        docs: list[dict] = []
        loaded: list[int] = []
        nbytes = 0
        for cluster in clusters:
            cid = int(cluster)
            if cid in self._warm:
                docs.extend(self._warm[cid])
                continue
            path = self.cluster_dir / f"c{cid}.jsonl"
            if not path.exists():
                continue
            data = path.read_bytes()  # the actual SSD → RAM transfer
            rows = [json.loads(line) for line in data.splitlines() if line]
            self._warm[cid] = rows
            self._warm_order.append(cid)
            self._warm_bytes += len(data)
            docs.extend(rows)
            loaded.append(cid)
            nbytes += len(data)
        self._trim_warm()
        return {"clusters": [int(c) for c in clusters], "docs": docs, "bytes": nbytes, "loaded": loaded}

    def _trim_warm(self) -> None:
        while self._warm_bytes > self.warm_budget and self._warm_order:
            cid = self._warm_order.pop(0)
            rows = self._warm.pop(cid, None)
            if rows is None:
                continue
            self._warm_bytes = max(0, self._warm_bytes - sum(len(r.get("text", "")) + 64 for r in rows))

    def warm_cache_bytes(self) -> int:
        return int(self._warm_bytes)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def search(self, vector: np.ndarray, k: int = 8, clusters: Iterable[int] | None = None) -> list[dict]:
        table = self._table()
        if table is None:
            return []
        query = table.search(np.asarray(vector, dtype=np.float32).tolist())
        if clusters:
            ids = ",".join(str(int(c)) for c in clusters)
            try:
                query = query.where(f"cluster IN ({ids})", prefilter=True)
            except Exception:  # noqa: BLE001
                pass
        try:
            rows = query.limit(max(k, 8)).to_list()
        except Exception as exc:  # noqa: BLE001
            log.warning("search failed: %s", exc)
            return []
        results = []
        for row in rows:
            distance = row.get("_distance")
            # cosine distance lives in [0, 2]; map it to a 0..1 similarity so the
            # CLI/dashboard never show negative "scores" for antipodal hashes.
            score = 1.0 - float(distance) / 2.0 if distance is not None else 0.0
            score = max(0.0, min(1.0, score))
            results.append(
                {
                    "path": row.get("path", ""),
                    "chunk": int(row.get("chunk", 0)),
                    "text": row.get("text", ""),
                    "cluster": int(row.get("cluster", -1)),
                    "score": round(score, 4),
                }
            )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]

    def prefetch_paths(self, paths: Iterable[str]) -> dict:
        clusters: set[int] = set()
        entries = self._manifest.get("paths", {})
        for path in paths:
            info = entries.get(path)
            if info:
                clusters.add(int(info.get("cluster", 0)))
        return self.warm_clusters(sorted(clusters))

    # ------------------------------------------------------------------
    def _dir_size(self) -> int:
        now = time.time()
        if now - self._dir_size_cache[0] < 5.0:
            return self._dir_size_cache[1]
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    continue
        self._dir_size_cache = (now, total)
        return total

    def metrics(self) -> dict:
        table = self._table()
        chunks = 0
        if table is not None:
            try:
                chunks = int(table.count_rows())
            except Exception:  # noqa: BLE001
                chunks = 0
        centroids = self._centroids()
        return {
            "available": table is not None,
            "docs": len(self._manifest.get("paths", {})),
            "chunks": chunks,
            "clusters": len(centroids) if centroids is not None else 0,
            "index_type": self.index_type,
            "index_bytes": self._dir_size(),
            "warm_bytes": self.warm_cache_bytes(),
            "warm_clusters": len(self._warm),
            "built_at": self._built_at,
            "dim": self.dim,
        }
