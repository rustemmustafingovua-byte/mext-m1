"""Cold weight store: tensors live as ``.npy`` files on SSD.

Threading model (important, MLX arrays are bound to the thread that created
them — using them elsewhere raises *"There is no Stream(gpu, 0)"*):

* a **background loader thread** only reads bytes from SSD with NumPy and parks
  them in ``_np_ready`` — it never touches MLX;
* the **model thread** (a single worker inside
  :class:`~mext_ai.neural.runner.MextModelRunner`) converts NumPy buffers into
  MLX arrays, runs the graph and evicts from the MLX cache.

So the residency budget covers MLX arrays (converted) plus NumPy buffers
(prefetched but not yet converted). A miss is always served synchronously by
the model thread, therefore prediction quality can only affect speed.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..memory import HAS_MLX, clear_metal_cache

if HAS_MLX:  # pragma: no cover - Apple Silicon only
    import mlx.core as mx

log = logging.getLogger("mext_ai.store")

MB = 1024 * 1024
MIN_BUDGET = 1 * MB


@dataclass
class TensorRef:
    name: str
    path: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
        }


class ColdStore:
    """Bounded RAM window over a directory of per-tensor ``.npy`` files."""

    def __init__(self, root: str | Path, budget_bytes: int = 256 * MB) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.refs: dict[str, TensorRef] = {}
        self.budget = max(int(budget_bytes), MIN_BUDGET)
        self.np_cap = max(MIN_BUDGET, int(self.budget * 0.4))

        self._mx: "OrderedDict[str, object]" = OrderedDict()
        self._mx_bytes = 0
        self._np_ready: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._np_bytes = 0
        self._pinned: set[str] = set()
        self._need_clear = False

        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.streamed_bytes = 0
        self.evictions = 0
        self.prefetch_loads = 0
        self._io_window: deque[tuple[float, int]] = deque()

        self._queue: "queue.Queue[str | None]" = queue.Queue(maxsize=1024)
        self._pending: set[str] = set()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._prefetch_loop, name="mext-cold-loader", daemon=True
        )
        self._worker.start()
        self._index_path = self.root / "index.json"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def put(self, name: str, array: object) -> TensorRef:
        np_array = array if isinstance(array, np.ndarray) else np.array(array)
        np_array = np.ascontiguousarray(np_array, dtype=np_array.dtype)
        path = self.root / f"{name.replace('/', '_')}.npy"
        with open(path, "wb") as handle:
            np.save(handle, np_array)
        ref = TensorRef(
            name=name,
            path=str(path),
            shape=tuple(int(x) for x in np_array.shape),
            dtype=str(np_array.dtype),
            nbytes=int(np_array.nbytes),
        )
        self.refs[name] = ref
        return ref

    def save_index(self) -> None:
        payload = {
            "tensors": {name: ref.to_dict() for name, ref in self.refs.items()},
            "updated_at": time.time(),
        }
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self._index_path)

    def load_index(self) -> bool:
        if not self._index_path.exists():
            return False
        try:
            payload = json.loads(self._index_path.read_text())
            self.refs = {
                name: TensorRef(
                    name=name,
                    path=item["path"],
                    shape=tuple(item["shape"]),
                    dtype=item["dtype"],
                    nbytes=int(item["nbytes"]),
                )
                for name, item in payload.get("tensors", {}).items()
            }
            return bool(self.refs)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot read store index: %s", exc)
            return False

    def size_bytes(self) -> int:
        return sum(ref.nbytes for ref in self.refs.values())

    # ------------------------------------------------------------------
    # Model-thread API
    # ------------------------------------------------------------------
    def pin(self, name: str) -> bool:
        if name not in self.refs:
            return False
        self.get(name)
        with self._lock:
            self._pinned.add(name)
        return True

    def unpin(self, name: str) -> None:
        with self._lock:
            self._pinned.discard(name)

    def is_resident(self, name: str) -> bool:
        with self._lock:
            return name in self._mx or name in self._np_ready

    def resident_names(self) -> list[str]:
        with self._lock:
            return list(self._mx) + [n for n in self._np_ready if n not in self._mx]

    def get(self, name: str) -> object:
        """Fetch a tensor as an MLX array (call from the model thread)."""
        with self._lock:
            cached = self._mx.get(name)
            if cached is not None:
                self._mx.move_to_end(name)
                self.hits += 1
                return cached
            ref = self.refs.get(name)
            if ref is None:
                raise KeyError(f"tensor not in cold store: {name}")
            np_array = self._np_ready.pop(name, None)
            if np_array is not None:
                self._np_bytes = max(0, self._np_bytes - int(np_array.nbytes))

        if np_array is None:
            with self._io_lock:
                np_array = np.load(ref.path)
            self._note_io(int(np_array.nbytes))

        array = mx.array(np_array) if HAS_MLX else np_array
        with self._lock:
            self.misses += 1
            self._insert_mx_locked(name, ref, array)
            need_clear = self._need_clear
            self._need_clear = False
        if need_clear:
            clear_metal_cache()
        return array

    def _insert_mx_locked(self, name: str, ref: TensorRef, array: object) -> None:
        if name not in self._pinned and ref.nbytes > self.budget:
            return  # too big to cache: stream it straight into the graph
        previous = self._mx.pop(name, None)
        if previous is not None:
            self._mx_bytes = max(0, self._mx_bytes - ref.nbytes)
        self._mx[name] = array
        self._mx.move_to_end(name)
        self._mx_bytes += ref.nbytes
        self._enforce_mx_locked()

    def _enforce_mx_locked(self) -> None:
        evicted = 0
        while self._mx_bytes + self._np_bytes > self.budget:
            victim = next((n for n in self._mx if n not in self._pinned), None)
            if victim is None:
                break
            ref = self.refs[victim]
            self._mx.pop(victim, None)
            self._mx_bytes = max(0, self._mx_bytes - ref.nbytes)
            self.evictions += 1
            evicted += 1
        if evicted:
            self._need_clear = True

    # ------------------------------------------------------------------
    # Prefetch (any thread)
    # ------------------------------------------------------------------
    def prefetch(self, names: Iterable[str]) -> int:
        queued = 0
        with self._lock:
            for name in names:
                if name in self._mx or name in self._np_ready or name in self._pending:
                    continue
                if name not in self.refs:
                    continue
                try:
                    self._queue.put_nowait(name)
                except queue.Full:
                    break
                self._pending.add(name)
                queued += 1
        return queued

    def _prefetch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                name = self._queue.get(timeout=0.4)
            except queue.Empty:
                continue
            if name is None:
                return
            try:
                ref = self.refs.get(name)
                if ref is not None:
                    with self._lock:
                        needed = name not in self._mx and name not in self._np_ready
                    if needed:
                        with self._io_lock:
                            buffer = np.load(ref.path)
                        self._note_io(int(buffer.nbytes))
                        with self._lock:
                            if name not in self._mx and name not in self._np_ready:
                                self._np_ready[name] = buffer
                                self._np_bytes += int(buffer.nbytes)
                                self.prefetch_loads += 1
                                self._trim_np_locked()
            except Exception as exc:  # noqa: BLE001
                log.debug("prefetch %s failed: %s", name, exc)
            finally:
                with self._lock:
                    self._pending.discard(name)
                self._queue.task_done()

    def _trim_np_locked(self) -> None:
        while self._np_bytes > self.np_cap and self._np_ready:
            name, buffer = self._np_ready.popitem(last=False)
            self._np_bytes = max(0, self._np_bytes - int(buffer.nbytes))

    def warm(self, names: Iterable[str], wait: float = 0.0) -> int:
        """Load tensors synchronously (tests / ``prepare``)."""
        loaded = 0
        deadline = time.monotonic() + wait
        for name in names:
            if not self.is_resident(name) and name in self.refs:
                self.get(name)
                loaded += 1
            if wait and time.monotonic() > deadline:
                break
        return loaded

    # ------------------------------------------------------------------
    def set_budget(self, budget_bytes: int) -> None:
        with self._lock:
            self.budget = max(MIN_BUDGET, int(budget_bytes))
            self.np_cap = max(MIN_BUDGET, int(self.budget * 0.4))
            self._enforce_mx_locked()
            self._trim_np_locked()

    def clear(self) -> None:
        with self._lock:
            self._mx.clear()
            self._mx_bytes = 0
            self._np_ready.clear()
            self._np_bytes = 0
        clear_metal_cache()

    def _note_io(self, nbytes: int) -> None:
        now = time.monotonic()
        with self._lock:
            self.streamed_bytes += int(nbytes)
            self._io_window.append((now, int(nbytes)))
            while self._io_window and now - self._io_window[0][0] > 1.5:
                self._io_window.popleft()

    def ssd_mbps(self) -> float:
        with self._lock:
            if not self._io_window:
                return 0.0
            span = max(0.25, time.monotonic() - self._io_window[0][0])
            return sum(b for _, b in self._io_window) / span / 1e6

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "resident_bytes": self._mx_bytes + self._np_bytes,
                "resident_tensors": len(self._mx),
                "prefetched_tensors": len(self._np_ready),
                "budget_bytes": self.budget,
                "cold_bytes": self.size_bytes(),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "streamed_bytes": self.streamed_bytes,
                "evictions": self.evictions,
                "prefetch_loads": self.prefetch_loads,
                "ssd_mbps": self.ssd_mbps(),
                "pinned": sorted(self._pinned),
            }

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
