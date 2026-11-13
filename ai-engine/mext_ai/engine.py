"""AI engine: JSON-RPC server + supervisor for the neural and RAG layers."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from pathlib import Path
from typing import Any

from .ipc import RpcServer
from .memory import HAS_MLX
from .neural.runner import MextModelRunner
from .rag.index import RagIndex
from .rag.prefetcher import StreamingQueryPrefetcher

log = logging.getLogger("mext_ai.engine")

__all__ = ["AiEngine"]


class AiEngine:
    def __init__(
        self,
        workspace: str | Path,
        socket: str | Path,
        state_dir: str | Path,
        ram_budget: int,
        model: str,
        lookahead: int = 2,
        nprobe: int = 3,
        demo: bool = False,
        neural: bool = True,
        rag: bool = True,
        synthetic_moe: int = 0,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.socket = Path(socket)
        self.ram_budget = int(ram_budget)
        self.demo = bool(demo)
        self.enable_neural = bool(neural)
        self.enable_rag = bool(rag)
        self.started_at = time.time()
        self._stop = threading.Event()

        self.runner: MextModelRunner | None = None
        self.index: RagIndex | None = None
        self.prefetcher: StreamingQueryPrefetcher | None = None
        self.rag_status: dict[str, Any] = {
            "ready": False,
            "building": False,
            "error": None,
        }

        if self.enable_neural:
            self.runner = MextModelRunner(
                model,
                self.state_dir / "neural",
                self.ram_budget,
                lookahead=lookahead,
                moe_layers=synthetic_moe,
            )
            self.runner.load()

        if self.enable_rag:
            self.index = RagIndex(self.state_dir / "rag", self.workspace)
            self.prefetcher = StreamingQueryPrefetcher(self.index, nprobe=nprobe)
            threading.Thread(target=self._prepare_rag, name="mext-rag-build", daemon=True).start()

        self.server = RpcServer(self.socket, self.handle, name="mext-ai")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _prepare_rag(self) -> None:
        assert self.index is not None
        self.rag_status["building"] = True
        try:
            self.index.connect()
            result = self.index.build()
            self.rag_status.update(
                ready=True,
                building=False,
                error=None,
                built_at=time.time(),
                chunks=result.get("chunks", 0),
                docs=result.get("docs", 0),
                clusters=result.get("clusters", 0),
            )
            log.info(
                "RAG ready: %s docs, %s chunks, %s clusters (%s)",
                result.get("docs"),
                result.get("chunks"),
                result.get("clusters"),
                result.get("index_type"),
            )
            if self.prefetcher is not None and result.get("clusters"):
                # Warm the first partition so the very first query is fast.
                self.prefetcher.feed("readme")
        except Exception as exc:
            log.exception("RAG indexing failed")
            self.rag_status.update(ready=False, building=False, error=f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        self.server.start()
        threading.Thread(target=self._tune_loop, name="mext-ai-tune", daemon=True).start()
        if self.demo and self.runner is not None:
            threading.Thread(target=self._demo_loop, name="mext-ai-demo", daemon=True).start()
        log.info(
            "AI engine listening on %s (neural=%s rag=%s demo=%s)",
            self.socket,
            self.enable_neural,
            self.enable_rag,
            self.demo,
        )
        try:
            while not self._stop.wait(0.5):
                pass
        except KeyboardInterrupt:  # pragma: no cover
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        self.server.stop()
        if self.prefetcher is not None:
            self.prefetcher.close()
        if self.runner is not None:
            self.runner.stop()
        log.info("AI engine stopped")

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------
    def _tune_loop(self) -> None:
        while not self._stop.wait(1.5):
            if self.runner is None:
                continue
            try:
                self.runner.tune_cache()
            except Exception as exc:  # noqa: BLE001
                log.debug("tune failed: %s", exc)

    def _demo_loop(self) -> None:
        prompts = [
            "def parse_config(path):",
            "SELECT path FROM chunks",
            "impl Cache for Arc",
            "fn main() {",
        ]
        index = 0
        while not self._stop.wait(2.5):
            if self.runner is None or not self.runner.metrics().get("available"):
                continue
            pressure = self.runner.last_pressure
            if pressure == "red":
                time.sleep(2.0)
                continue
            try:
                self.runner.generate(prompts[index % len(prompts)], max_tokens=12)
                index += 1
            except Exception as exc:  # noqa: BLE001
                log.debug("demo generate failed: %s", exc)

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------
    def handle(self, method: str, params: dict[str, Any]) -> Any:
        if method in ("ping", "health"):
            return self.health()
        if method == "metrics":
            return self.metrics()
        if method == "prefetch":
            return self.prefetch_paths(params.get("paths") or [])
        if method == "neural.prefetch":
            if self.runner is None:
                return {"available": False}
            return self.runner.prefetch_layers([int(x) for x in params.get("layers") or []])
        if method == "neural.generate":
            if self.runner is None:
                return {"available": False}
            return self.runner.generate(
                params.get("prompt", "def fib(n):"),
                int(params.get("max_tokens", 16)),
                float(params.get("temperature", 0.0)),
            )
        if method == "neural.status":
            if self.runner is None:
                return {"available": False, "disabled": True}
            return self.runner.metrics()
        if method == "rag.type":
            if not self.rag_ready():
                return {"available": False, "indexing": bool(self.rag_status.get("building"))}
            assert self.prefetcher is not None
            return self.prefetcher.feed(str(params.get("prefix", "")))
        if method == "rag.commit":
            if not self.rag_ready():
                return {
                    "available": False,
                    "indexing": bool(self.rag_status.get("building")),
                    "error": self.rag_status.get("error"),
                    "results": [],
                }
            assert self.prefetcher is not None
            return self.prefetcher.commit(str(params.get("query", "")), int(params.get("k", 8)))
        if method == "rag.index":
            if self.index is None:
                return {"available": False, "disabled": True}
            return self.index.build(force=bool(params.get("force", False)))
        if method == "rag.stats":
            return self.metrics()["rag"]
        if method == "gguf.inspect":
            path = str(params.get("path") or "")
            if not path:
                raise ValueError("gguf.inspect needs `path`")
            from .gguf import GgufFile

            with GgufFile.open(path) as gguf:
                return gguf.summary()
        if method == "gguf.warm":
            path = str(params.get("path") or "")
            if not path:
                raise ValueError("gguf.warm needs `path`")
            from .gguf import GgufFile, LayerPageWarmer

            layers = [int(v) for v in (params.get("layers") or [])]
            lookahead = int(params.get("lookahead", 2))
            with GgufFile.open(path) as gguf:
                warmer = LayerPageWarmer(gguf, lookahead=lookahead)
                result = warmer.warm_layers(layers)
                result["predict_next"] = warmer.predict_next()
                result["stats"] = warmer.stats()
                result["layers_total"] = len(gguf.layers)
                return result
        if method == "shutdown":
            threading.Thread(target=self._delayed_stop, daemon=True).start()
            return {"stopping": True}
        raise ValueError(f"unknown method `{method}`")

    def _delayed_stop(self) -> None:
        time.sleep(0.15)
        self._stop.set()

    def rag_ready(self) -> bool:
        return bool(self.rag_status.get("ready")) and self.prefetcher is not None

    def prefetch_paths(self, paths: list[str]) -> dict:
        if self.index is None:
            return {"available": False, "disabled": True}
        result = self.index.prefetch_paths(paths)
        return {"available": True, "clusters": result.get("clusters", []), "bytes": result.get("bytes", 0)}

    # ------------------------------------------------------------------
    def health(self) -> dict:
        neural = self.runner.metrics() if self.runner else {"available": False}
        return {
            "available": True,
            "pid": os.getpid(),
            "python": platform.python_version(),
            "mlx": HAS_MLX,
            "uptime_s": round(time.time() - self.started_at, 1),
            "socket": str(self.socket),
            "neural_ready": bool(neural.get("available")),
            "neural_mode": neural.get("mode"),
            "rag_ready": self.rag_ready(),
            "rag_status": self.rag_status,
            "gguf": True,
        }

    def metrics(self) -> dict:
        neural: dict[str, Any]
        if self.runner is None:
            neural = {"available": False, "disabled": True}
        else:
            neural = self.runner.metrics()

        rag: dict[str, Any] = {"available": False, "disabled": not self.enable_rag}
        if self.index is not None:
            try:
                rag = dict(self.index.metrics())
            except Exception as exc:  # noqa: BLE001
                rag = {"available": False, "error": str(exc)}
            if self.prefetcher is not None:
                rag.update(self.prefetcher.metrics())
            rag.update(
                {
                    "ready": self.rag_ready(),
                    "indexing": bool(self.rag_status.get("building")),
                    "error": self.rag_status.get("error"),
                }
            )
            rag["available"] = bool(self.rag_ready())

        return {
            "available": True,
            "python": platform.python_version(),
            "pid": os.getpid(),
            "uptime_s": round(time.time() - self.started_at, 2),
            "mlx": HAS_MLX,
            "ram_budget": self.ram_budget,
            "socket": str(self.socket),
            "workspace": str(self.workspace),
            "neural": neural,
            "rag": rag,
        }
