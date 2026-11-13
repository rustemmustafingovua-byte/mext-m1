import numpy as np
import pytest

from mext_ai.rag.embed import HashingEmbedder
from mext_ai.rag.index import RagIndex, chunk_text
from mext_ai.rag.prefetcher import StreamingQueryPrefetcher

MB = 1024 * 1024


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "docs" / "arc.md").write_text(
        "# ARC cache\n\n"
        "The compressed pool evicts pages with madvise DONTNEED when the RAM budget "
        "is exceeded. Recency and frequency are balanced by the adaptive parameter p.\n"
    )
    (root / "docs" / "mlx.md").write_text(
        "# MLX layer streaming\n\n"
        "Cold transformer weights live on SSD and are prefetched for the next layers. "
        "Hot embeddings and norms stay in unified memory.\n"
    )
    (root / "src" / "rag.py").write_text(
        "def prefetch(prefix):\n"
        "    '''Stream IVF clusters from LanceDB while the user is typing a query.'''\n"
        "    return prefix\n"
    )
    (root / "src" / "skip.bin").write_bytes(b"\x00\x01\x02")
    return root


@pytest.fixture()
def index(workspace):
    ridx = RagIndex(workspace / ".mext" / "rag", workspace)
    ridx.connect()
    result = ridx.build(force=True)
    assert result["chunks"] >= 3
    return ridx


def test_embedder_is_normalised_and_stable():
    embedder = HashingEmbedder()
    a = embedder.embed("compressed memory pool")
    b = embedder.embed("compressed memory pool")
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    # a prefix is closer to the full phrase than an unrelated sentence
    prefix = embedder.embed("compressed mem")
    other = embedder.embed("banana smoothie recipe")
    assert float(prefix @ a) > float(other @ a)


def test_chunking_overlaps():
    text = "x" * 5000
    chunks = list(chunk_text(text, size=1000, overlap=100))
    assert len(chunks) >= 5
    assert all(len(c) <= 1000 for c in chunks)


def test_build_indexes_supported_files(index):
    metrics = index.metrics()
    assert metrics["chunks"] >= 3
    assert metrics["clusters"] >= 2
    assert metrics["docs"] == 3
    assert (index.root / "centroids.npy").exists()
    assert any(index.cluster_dir.glob("c*.jsonl"))


def test_build_is_incremental(index):
    again = index.build()
    assert again["updated"] == 0


def test_search_finds_relevant_chunk(index):
    vector = index.embedder.embed("madvise DONTNEED compressed pool budget")
    results = index.search(vector, k=3)
    assert results
    paths = " ".join(r["path"] for r in results)
    assert "arc.md" in paths


def test_nearest_clusters_and_warm(index):
    vector = index.embedder.embed("cold weights pretrained layers")
    clusters = index.nearest_clusters(vector, nprobe=2)
    assert len(clusters) >= 1
    warmed = index.warm_clusters(clusters)
    assert warmed["docs"]
    assert warmed["bytes"] > 0
    assert index.warm_cache_bytes() > 0


def test_prefetch_then_commit_scores_hit(index):
    prefetcher = StreamingQueryPrefetcher(index, nprobe=3)
    try:
        feed = prefetcher.feed("madvise compressed")
        assert feed["clusters"]
        result = prefetcher.commit("madvise compressed page eviction", k=3)
        assert result["results"]
        assert result["prefetch_hit"] is True
        assert result["prefetch_hit_rate"] > 0
        stats = prefetcher.metrics()
        assert stats["prefetches"] >= 1
        assert stats["queries"] == 1
    finally:
        prefetcher.close()


def test_prefetch_paths_maps_documents_to_clusters(index):
    result = index.prefetch_paths(["docs/arc.md"])
    assert result["clusters"]
