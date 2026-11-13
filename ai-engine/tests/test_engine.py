import shutil
import tempfile
import time
from pathlib import Path

from mext_ai.engine import AiEngine
from mext_ai.ipc import RpcClient
from mext_ai.memory import MB


def test_engine_rag_pipeline(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "pool.md").write_text(
        "The compressed pool keeps LZ4 frames in RAM and evicts them with ARC.\n"
    )
    (workspace / "docs" / "layers.md").write_text(
        "Neural layers are streamed from SSD while the prefetcher loads the next ones.\n"
    )

    socket_dir = Path(tempfile.mkdtemp(prefix="me-", dir="/tmp"))
    engine = AiEngine(
        workspace=workspace,
        socket=socket_dir / "ai.sock",
        state_dir=tmp_path / "state",
        ram_budget=512 * MB,
        model="synthetic",
        neural=False,
        rag=True,
        nprobe=2,
    )
    engine.server.start()
    client = RpcClient(socket_dir / "ai.sock", timeout=5.0)
    try:
        deadline = time.time() + 40
        health = {}
        while time.time() < deadline:
            health = client.call("health")
            if health.get("rag_ready"):
                break
            time.sleep(0.1)
        assert health.get("rag_ready"), health
        assert health["pid"] > 0

        metrics = client.call("metrics")
        assert metrics["neural"]["disabled"] is True
        assert metrics["rag"]["available"] is True
        assert metrics["rag"]["chunks"] >= 2
        assert metrics["python"]

        typed = client.call("rag.type", {"prefix": "compressed pool"})
        assert typed["clusters"]

        committed = client.call("rag.commit", {"query": "compressed pool ARC", "k": 2})
        assert committed["results"]
        assert committed["prefetch_hit"] is True

        warmed = client.call("prefetch", {"paths": ["docs/pool.md"]})
        assert warmed["available"] is True

        stats = client.call("rag.stats")
        assert stats["chunks"] >= 2
    finally:
        client.close()
        engine.stop()
        shutil.rmtree(socket_dir, ignore_errors=True)
