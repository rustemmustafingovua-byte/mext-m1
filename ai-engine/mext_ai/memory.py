"""Memory profiling and the pressure-aware cache controller.

The controller is the piece that keeps macOS out of the yellow/red Memory
Pressure zone: it never grows the weight cache beyond what is actually free
*after* reserving headroom for the OS, it grows with a step limit (so a sudden
free page does not cause a 2 GiB allocation spike) and it shrinks immediately
when XNU reports pressure.

MLX counters are read explicitly (``metal_stats``) rather than on every sample:
MLX arrays/streams are thread-bound and the sampler is called from several
threads.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

try:  # pragma: no cover - exercised on machines with MLX
    import mlx.core as mx

    HAS_MLX = True
except Exception:  # noqa: BLE001
    mx = None  # type: ignore[assignment]
    HAS_MLX = False

MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Metal / MLX memory counters
# ---------------------------------------------------------------------------


def metal_stats() -> dict[str, int]:
    """``mx.metal.get_active_memory()`` and friends (new and old API names).

    Call only from the thread that owns MLX arrays (the model thread).
    """
    if not HAS_MLX:
        return {}
    out: dict[str, int] = {}
    mapping = {
        "metal_active_bytes": "get_active_memory",
        "metal_peak_bytes": "get_peak_memory",
        "metal_cache_bytes": "get_cache_memory",
    }
    for key, fn_name in mapping.items():
        fn = getattr(mx, fn_name, None)
        if fn is None:
            fn = getattr(getattr(mx, "metal", None), fn_name, None)
        if fn is None:
            continue
        try:
            out[key] = int(fn())
        except Exception:  # noqa: BLE001
            continue
    return out


def set_metal_cache_limit(limit_bytes: int) -> bool:
    if not HAS_MLX:
        return False
    fn = getattr(mx, "set_cache_limit", None) or getattr(
        getattr(mx, "metal", None), "set_cache_limit", None
    )
    if fn is None:
        return False
    try:
        fn(int(max(16 * MB, limit_bytes)))
        return True
    except Exception:  # noqa: BLE001
        return False


def clear_metal_cache() -> None:
    if not HAS_MLX:
        return
    fn = getattr(mx, "clear_cache", None) or getattr(
        getattr(mx, "metal", None), "clear_cache", None
    )
    if fn is not None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Platform snapshot
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    total: int = 0
    available: int = 0
    used_pct: float = 0.0
    swap_used: int = 0
    swap_total: int = 0
    pressure: str = "unknown"
    metal: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "total": self.total,
            "available": self.available,
            "used_pct": self.used_pct,
            "swap_used": self.swap_used,
            "swap_total": self.swap_total,
            "pressure": self.pressure,
        }
        data.update(self.metal)
        return data


class MemoryProfiler:
    """Caches platform readings for ~0.8 s (the dashboard polls faster)."""

    def __init__(self, ttl: float = 0.8) -> None:
        self.ttl = ttl
        self._lock = threading.Lock()
        self._at = 0.0
        self._snap = MemorySnapshot()

    def sample(self, force: bool = False, with_metal: bool = False) -> MemorySnapshot:
        now = time.monotonic()
        with self._lock:
            if not force and self._snap.total and (now - self._at) < self.ttl:
                snapshot = self._snap
            else:
                vm = psutil.virtual_memory()
                swap = psutil.swap_memory()
                snapshot = MemorySnapshot(
                    total=int(vm.total),
                    available=int(vm.available),
                    used_pct=float(vm.percent) / 100.0,
                    swap_used=int(swap.used),
                    swap_total=int(swap.total),
                    pressure=self._pressure(vm, swap),
                )
                self._snap = snapshot
                self._at = now
        if with_metal:
            snapshot.metal = metal_stats()
        return snapshot

    # ------------------------------------------------------------------
    @staticmethod
    def _sysctl_pressure() -> str | None:
        try:
            out = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                timeout=1.0,
                check=False,
            )
            if out.returncode != 0:
                return None
            level = int(out.stdout.strip() or b"0")
        except Exception:  # noqa: BLE001
            return None
        # XNU: DISPATCH_MEMORYPRESSURE_NORMAL=1, WARN=2, CRITICAL=4
        return {0: "green", 1: "green", 2: "yellow", 4: "red"}.get(level)

    def _pressure(self, vm: Any, swap: Any) -> str:
        level = self._sysctl_pressure()
        if level:
            return level
        if vm.percent >= 94 or (swap.total and swap.used / swap.total > 0.9):
            return "red"
        if vm.percent >= 86:
            return "yellow"
        return "green"


# ---------------------------------------------------------------------------
# Adaptive cache sizing
# ---------------------------------------------------------------------------


class AdaptiveCacheController:
    """Chooses how many bytes of cold weights may live in RAM.

    * hard ceiling = ``budget_bytes`` (a slice of ``--ram-limit``)
    * headroom = max(320 MiB, 12% of RAM) reserved for macOS
    * growth is rate-limited to 1.6× per tick, shrink is immediate
    * yellow pressure caps at 85% of current, red at 55%
    """

    def __init__(
        self,
        budget_bytes: int,
        total_bytes: int | None = None,
        min_bytes: int = 48 * MB,
    ) -> None:
        self.budget = int(budget_bytes)
        self.total = int(total_bytes or psutil.virtual_memory().total)
        self.min_bytes = min(int(min_bytes), self.budget)
        self.current = int(min(self.budget, 256 * MB))
        self.history: list[dict[str, Any]] = []

    def update(self, snap: MemorySnapshot) -> dict[str, Any]:
        total = snap.total or self.total
        headroom = max(320 * MB, int(total * 0.12))
        usable = max(0, snap.available - headroom)
        target = min(self.budget, usable)
        target = max(target, self.min_bytes)

        if target > self.current:
            target = min(target, max(int(self.current * 1.6), self.current + 32 * MB))
        else:
            floor = int(self.current * (0.5 if snap.pressure == "red" else 0.8))
            target = max(target, floor)

        if snap.pressure == "yellow":
            target = min(target, int(self.current * 0.85))
        elif snap.pressure == "red":
            target = min(target, int(self.current * 0.55))

        target = max(self.min_bytes, min(self.budget, target))
        if abs(target - self.current) <= self.current * 0.04:
            action = "hold"
        elif target > self.current:
            action = "grow"
        else:
            action = "shrink"
        self.current = target
        result = {
            "target": target,
            "action": action,
            "headroom": headroom,
            "usable": usable,
            "pressure": snap.pressure,
            "available": snap.available,
        }
        self.history.append(result)
        if len(self.history) > 64:
            self.history.pop(0)
        return result
