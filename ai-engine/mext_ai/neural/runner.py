"""``MextModelRunner`` — hot/cold split, predictive layer streaming, memory guard.

All MLX work runs on **one dedicated model thread** (``self._worker``). This is
not cosmetic: MLX arrays carry a reference to the stream of the thread that
created them, and using them from another thread fails with
*"There is no Stream(gpu, 0) in current thread"*. The background cold loader
therefore only reads NumPy buffers from SSD; conversion to ``mx.array`` happens
on the model thread.

Responsibilities:

1. a :class:`~mext_ai.neural.store.ColdStore` with the cold weights on SSD,
2. a hot set (embeddings + norms) that never leaves unified memory,
3. a lookahead window that prefetches layers ``i+1 … i+lookahead`` while layer
   ``i`` is being computed,
4. an :class:`~mext_ai.memory.AdaptiveCacheController` that keeps the residency
   budget in sync with the real free RAM of the Mac.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..memory import (
    HAS_MLX,
    AdaptiveCacheController,
    MemoryProfiler,
    set_metal_cache_limit,
)
from .plan import WeightPlan, build_plan, include_in_layers
from .store import ColdStore

if HAS_MLX:  # pragma: no cover - Apple Silicon only
    import mlx.core as mx

log = logging.getLogger("mext_ai.runner")

MB = 1024 * 1024


class _Job:
    __slots__ = ("args", "done", "error", "fn", "result")

    def __init__(self, fn: Callable[..., Any], args: tuple):
        self.fn = fn
        self.args = args
        self.result: Any = None
        self.error: BaseException | None = None
        self.done = threading.Event()


class MextModelRunner:
    """Predictive MLX model runner with SSD-resident cold weights."""

    def __init__(
        self,
        model_id: str,
        state_dir: str | Path,
        ram_budget: int,
        lookahead: int = 2,
        moe_layers: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        profiler = MemoryProfiler()
        snapshot = profiler.sample(force=True)
        # The Rust pool already owns up to `ram_budget`; the weight cache gets
        # a separate slice that the controller grows/shrinks within.
        neural_cap = max(64 * MB, min(int(ram_budget) // 2, snapshot.total // 3))
        initial = max(64 * MB, min(neural_cap, 256 * MB))

        self.store = ColdStore(self.state_dir / "weights", initial)
        self.profiler = profiler
        self.controller = AdaptiveCacheController(neural_cap, snapshot.total)
        self.controller.current = initial

        self.lookahead_cfg = max(1, int(lookahead))
        self.moe_layers = (
            int(moe_layers)
            if moe_layers is not None
            else int(os.environ.get("MEXT_SYNTH_MOE", "0"))
        )
        self.lookahead = self.lookahead_cfg
        self.layers = 0
        self.layer_index = 0
        self.mode: str | None = None
        self.load_error: str | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.plan: WeightPlan | None = None
        self._switches: list[Any] = []

        self.hot_bytes = 0
        self.cold_bytes = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.tokens_per_s = 0.0
        self.generated_tokens = 0
        self.moe_active = 0
        self.last_action = "hold"
        self.last_pressure = snapshot.pressure
        self.metal: dict[str, int] = {}
        self.started_at = time.time()

        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._stopped = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="mext-model", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    # Model thread
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stopped.is_set():
            try:
                job = self._jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                job.result = job.fn(*job.args)
            except BaseException as exc:  # noqa: BLE001
                job.error = exc
                log.debug("model job failed: %s", exc)
            finally:
                job.done.set()

    def _run(self, fn: Callable[..., Any], *args: Any, timeout: float = 180.0) -> Any:
        if threading.current_thread() is self._worker:
            return fn(*args)
        job = _Job(fn, args)
        self._jobs.put(job)
        if not job.done.wait(timeout):
            raise TimeoutError(f"model thread timed out after {timeout}s")
        if job.error is not None:
            raise job.error
        return job.result

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not HAS_MLX:
            self.mode = "unavailable"
            self.load_error = "mlx is not installed (pip install mlx)"
            return
        self.mode = "loading"
        try:
            self._run(self._load_impl)
        except Exception as exc:
            log.exception("model load failed")
            self.mode = "unavailable"
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _load_impl(self) -> None:
        if self._try_mlx_lm_impl():
            self.mode = "mlx_lm"
            return
        self._load_synthetic_impl()

    def _load_synthetic_impl(self) -> None:
        from .model import MiniTransformer

        self.store.refs.clear()
        moe_layers = max(0, min(self.moe_layers, 6))
        model = MiniTransformer(
            self.store,
            moe_layers=moe_layers,
            num_experts=int(os.environ.get("MEXT_MOE_EXPERTS", "8")),
            top_k=int(os.environ.get("MEXT_MOE_TOP_K", "2")),
        )
        self.moe_active = moe_layers
        self.model = model
        self.layers = model.n_layers

        sizes = {name: ref.nbytes for name, ref in self.store.refs.items()}
        sizes.update(model.hot_tensor_sizes())
        self.plan = build_plan(sizes)
        self.hot_bytes = model.hot_bytes()
        self.cold_bytes = sum(ref.nbytes for ref in self.store.refs.values())

        self.store.pin("lm_head.weight")  # used on every decoding step
        self.store.save_index()
        self.store.set_budget(self.controller.current)
        mx.eval(model.embed, model.norm.weight)
        self.mode = "synthetic"
        self._warm_layers([0, 1])
        log.info(
            "synthetic runner ready: hot %.1f MiB, cold %.1f MiB, %d layers, "
            "lookahead %d, moe layers %d",
            self.hot_bytes / MB,
            self.cold_bytes / MB,
            self.layers,
            self.lookahead,
            moe_layers,
        )

    def _try_mlx_lm_impl(self) -> bool:
        """Wrap a local ``mlx_lm`` checkpoint, fp16 or 4/8-bit quantised."""
        try:
            import mlx_lm
        except Exception:  # noqa: BLE001
            return False
        model_path = Path(self.model_id)
        if not model_path.exists():
            log.info("model %s is not a local dir — using the synthetic demo model", self.model_id)
            return False
        try:
            from .wrap import hot_bytes, wrap_cold_modules

            model, tokenizer = mlx_lm.load(str(model_path))
            result = wrap_cold_modules(model, self.store)
            self._switches = list(result.get("switches", []))
            if result["wrapped"] == 0:
                log.warning("no cold projections to wrap — falling back to synthetic")
                return False
            self.model = model
            self.tokenizer = tokenizer
            self.layers = len(getattr(getattr(model, "model", model), "layers", []) or [])
            self.plan = build_plan(result["sizes"])
            include_in_layers(self.plan, self.store.refs.keys())
            self.cold_bytes = sum(ref.nbytes for ref in self.store.refs.values())
            self.hot_bytes = hot_bytes(model, self.store)
            # The head is touched on every decoding step: pin its parts.
            for suffix in ("weight", "scales", "biases", "bias"):
                name = f"lm_head.{suffix}"
                if name in self.store.refs:
                    self.store.pin(name)
            self.store.save_index()
            self.store.set_budget(self.controller.current)
            log.info(
                "wrapped %d cold projections from %s (hot %.1f MiB, cold %.1f MiB)",
                result["wrapped"],
                model_path,
                self.hot_bytes / MB,
                self.cold_bytes / MB,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("mlx_lm wrapping failed (%s) — falling back to synthetic", exc)
            return False

    # ------------------------------------------------------------------
    # Predictive prefetch
    # ------------------------------------------------------------------
    def _warm_layers(self, indices: list[int]) -> int:
        if not self.plan:
            return 0
        queued = 0
        for index in indices:
            queued += self.store.prefetch(self.plan.names_for_layer(index))
        return queued

    def prefetch_layers(self, indices: list[int]) -> dict:
        indices = [int(i) for i in indices if 0 <= int(i) < max(1, self.layers)]
        queued = self._warm_layers(indices)
        return {"layers": indices, "queued": queued}

    def on_layer(self, index: int) -> None:
        """Called by the forward pass right before layer ``index`` runs."""
        self.layer_index = index
        if not self.plan:
            return
        names = self.plan.names_for_layer(index)
        if names:
            if all(self.store.is_resident(name) for name in names):
                self.prefetch_hits += 1
            else:
                self.prefetch_misses += 1
        stop = min(self.layers, index + 1 + self.lookahead)
        for ahead in range(index + 1, stop):
            self.store.prefetch(self.plan.names_for_layer(ahead))
        pager = self.model.moe_at(index) if hasattr(self.model, "moe_at") else None
        if pager is not None:
            pager.prefetch(pager.predicted_experts())
        for switch in self._switches:
            if switch.layer_index == index:
                switch.prefetch(switch.predicted_experts())

    def on_route(self, layer_index: int, experts: list[int]) -> None:
        """Router-guided expert prefetch for the next MoE layers.

        Routing is sticky across adjacent tokens and layers, so the experts the
        current layer just used are the best cheap predictor for the next one.
        """
        if not self.model or not hasattr(self.model, "moe_at"):
            return
        limit = max(2, self.lookahead * 2)
        for ahead in range(layer_index + 1, min(self.layers, layer_index + 1 + self.lookahead)):
            pager = self.model.moe_at(ahead)
            if pager is None:
                continue
            ids = {int(e) for e in experts} | set(pager.predicted_experts(limit))
            pager.prefetch(sorted(ids))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _encode(self, prompt: str):
        vocab = int(getattr(self.model, "vocab", 8192))
        ids = [2]
        for i, ch in enumerate(prompt[:64]):
            ids.append(3 + (ord(ch) * 31 + i * 7) % max(1, vocab - 5))
        return mx.array([ids], dtype=mx.int32)

    def generate(
        self, prompt: str = "def fib(n):", max_tokens: int = 16, temperature: float = 0.0
    ) -> dict:
        return self._run(self._generate_impl, prompt, max_tokens, temperature)

    def _generate_impl(self, prompt: str, max_tokens: int, temperature: float) -> dict:
        if self.model is None or not HAS_MLX:
            raise RuntimeError(self.load_error or "model is not loaded")
        started = time.perf_counter()
        context = self._encode(prompt)
        produced: list[int] = []
        for _ in range(max(1, int(max_tokens))):
            logits = self.model(context, on_layer=self.on_layer, on_route=self.on_route)
            last = logits[:, -1, :]
            if temperature and temperature > 0:
                token = mx.random.categorical(last / float(temperature)).reshape(1, 1)
            else:
                token = mx.argmax(last, axis=-1).reshape(1, 1)
            produced.append(int(token.reshape(-1)[0]))
            context = mx.concatenate([context, token], axis=1)
            mx.eval(context)
            if context.shape[1] > 128:
                context = context[:, -128:]
        elapsed = max(1e-6, time.perf_counter() - started)
        rate = len(produced) / elapsed
        self.tokens_per_s = rate if self.tokens_per_s == 0 else 0.6 * rate + 0.4 * self.tokens_per_s
        self.generated_tokens += len(produced)
        return {
            "prompt": prompt,
            "generated": len(produced),
            "tokens": produced,
            "tokens_per_s": round(self.tokens_per_s, 2),
            "latency_ms": round(elapsed * 1000.0, 2),
            "layer_index": self.layer_index,
            "text": self._detokenize(produced),
        }

    def _detokenize(self, tokens: list[int]) -> str:
        if self.tokenizer is not None:
            try:
                return self.tokenizer.decode(tokens)
            except Exception:  # noqa: BLE001
                pass
        alphabet = "abcdefghijklmnopqrstuvwxyz .,():_"
        return "".join(alphabet[t % len(alphabet)] for t in tokens)

    # ------------------------------------------------------------------
    # Memory guard
    # ------------------------------------------------------------------
    def tune_cache(self) -> dict:
        return self._run(self._tune_impl, timeout=30.0)

    def _tune_impl(self) -> dict:
        snapshot = self.profiler.sample(force=True, with_metal=True)
        decision = self.controller.update(snapshot)
        self.store.set_budget(decision["target"])
        set_metal_cache_limit(int(snapshot.total * 0.08))
        if snapshot.pressure == "red":
            self.lookahead = 1
        elif snapshot.pressure == "yellow":
            self.lookahead = max(1, min(self.lookahead_cfg, 2))
        else:
            self.lookahead = self.lookahead_cfg
        self.last_action = decision["action"]
        self.last_pressure = snapshot.pressure
        self.metal = dict(snapshot.metal)
        return decision

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        store = self.store.stats()
        prefetch_total = self.prefetch_hits + self.prefetch_misses
        experts = self._expert_metrics()
        return {
            "available": self.mode not in (None, "unavailable", "loading"),
            "mode": self.mode,
            "model": Path(self.model_id).name if self.model_id else "unknown",
            "hot_bytes": int(self.hot_bytes),
            "cold_bytes": int(self.cold_bytes),
            "resident_bytes": int(store["resident_bytes"]),
            "cache_target_bytes": int(store["budget_bytes"]),
            "streamed_bytes": int(store["streamed_bytes"]),
            "store_hit_rate": round(float(store["hit_rate"]), 4),
            "prefetch_hits": int(self.prefetch_hits),
            "prefetch_misses": int(self.prefetch_misses),
            "prefetch_hit_rate": (self.prefetch_hits / prefetch_total) if prefetch_total else 0.0,
            "tokens_per_s": round(float(self.tokens_per_s), 2),
            "generated_tokens": int(self.generated_tokens),
            "layer_index": int(self.layer_index),
            "layers": int(self.layers),
            "lookahead": int(self.lookahead),
            "evictions": int(store["evictions"]),
            "ssd_mbps": round(float(store["ssd_mbps"]), 2),
            "metal_active_bytes": int(self.metal.get("metal_active_bytes", 0)),
            "metal_peak_bytes": int(self.metal.get("metal_peak_bytes", 0)),
            "pressure": self.last_pressure,
            "action": self.last_action,
            "pinned": store["pinned"],
            "moe_layers": int(experts.get("moe_layers", 0)),
            "experts_total": int(experts.get("experts_total", 0)),
            "experts_resident": int(experts.get("experts_resident", 0)),
            "expert_hits": int(experts.get("expert_hits", 0)),
            "expert_misses": int(experts.get("expert_misses", 0)),
            "expert_hit_rate": float(experts.get("expert_hit_rate", 0.0)),
            "expert_prefetches": int(experts.get("expert_prefetches", 0)),
            "error": self.load_error,
        }

    def _expert_metrics(self) -> dict:
        """Aggregate router-guided paging stats (synthetic or real MoE)."""
        if self.model is not None and hasattr(self.model, "expert_metrics"):
            return self.model.expert_metrics()
        if not self._switches:
            return {
                "experts_total": 0,
                "experts_resident": 0,
                "expert_hits": 0,
                "expert_misses": 0,
                "expert_hit_rate": 0.0,
                "expert_prefetches": 0,
                "moe_layers": 0,
            }
        per_switch = [switch.metrics() for switch in self._switches]
        hits = sum(item["expert_hits"] for item in per_switch)
        misses = sum(item["expert_misses"] for item in per_switch)
        return {
            "experts_total": sum(item["experts_total"] for item in per_switch),
            "experts_resident": sum(item["experts_resident"] for item in per_switch),
            "expert_hits": hits,
            "expert_misses": misses,
            "expert_hit_rate": (hits / (hits + misses)) if (hits + misses) else 0.0,
            "expert_prefetches": sum(item["expert_prefetches"] for item in per_switch),
            "moe_layers": len(self._switches),
        }

    def plan_dict(self) -> dict:
        return self.plan.to_dict() if self.plan else {}

    def stop(self) -> None:
        self._stopped.set()
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self.store.stop()
