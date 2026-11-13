"""Router-guided expert paging for MoE layers.

Dense MoE inference touches every expert's weights even though each token is
routed to ``top_k`` of them. On a memory-constrained Mac that is the worst
possible pattern: the model keeps 8–64 expert matrices resident to use two.

:class:`ExpertPager` stores every expert as its own tensor pair on SSD, computes
the router (which is small and stays hot), and pulls in only the experts the
router selected. Because routing is sticky across adjacent tokens and layers
(the same domains recur), it also prefetches the union of the recent routing
history before the next forward pass, which is what gives a measurable hit rate.

The router itself is kept in unified memory: it is `num_experts × dim`, i.e. a
few hundred KiB even for 256 experts.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from ..memory import HAS_MLX

if HAS_MLX:  # pragma: no cover - Apple Silicon only
    import mlx.core as mx

log = logging.getLogger("mext_ai.moe")

__all__ = ["ExpertPager"]


def _require_mlx() -> None:
    if not HAS_MLX:
        raise RuntimeError("MLX is not installed (pip install mlx)")


class ExpertPager:
    """Bounded expert cache with router-guided prefetch."""

    def __init__(
        self,
        store,
        prefix: str,
        dim: int,
        hidden: int,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        _require_mlx()
        self.store = store
        self.prefix = prefix
        self.dim = int(dim)
        self.hidden = int(hidden)
        self.num_experts = max(1, int(num_experts))
        self.top_k = max(1, min(int(top_k), self.num_experts))

        # Hot: the router is needed on every token and is tiny.
        self.router = mx.random.normal((self.num_experts, self.dim)) * 0.02
        for expert in range(self.num_experts):
            self.store.put(
                self.name(expert, "w1"),
                mx.random.normal((self.hidden, self.dim)) * 0.02,
            )
            self.store.put(
                self.name(expert, "w2"),
                mx.random.normal((self.dim, self.hidden)) * 0.02,
            )

        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self._history: deque[set[int]] = deque(maxlen=16)
        self._freq = [0.0] * self.num_experts

    # ------------------------------------------------------------------
    def name(self, expert: int, which: str) -> str:
        return f"{self.prefix}.experts.{expert}.{which}"

    def names(self, expert: int) -> list[str]:
        return [self.name(expert, "w1"), self.name(expert, "w2")]

    def prefetch(self, experts) -> int:
        queued = 0
        for expert in experts:
            queued += self.store.prefetch(self.names(int(expert)))
        self.prefetches += queued
        return queued

    def predicted_experts(self, limit: int | None = None) -> list[int]:
        """Hot set: frequency EMA plus recent routing (strong temporal locality)."""
        scores = list(self._freq)
        for snapshot in self._history:
            for expert in snapshot:
                scores[expert] += 0.5
        keep = limit if limit is not None else max(2, self.top_k * 2)
        ranked = sorted(range(self.num_experts), key=lambda i: -scores[i])
        return sorted(ranked[:keep])

    # ------------------------------------------------------------------
    def route(self, x):
        logits = x @ self.router.T  # (B, L, E)
        indices = mx.argsort(-logits, axis=-1)[..., : self.top_k]
        top = mx.take_along_axis(logits, indices, axis=-1)
        weights = mx.softmax(top, axis=-1)
        return indices, weights

    def __call__(self, x, on_route=None):
        indices, weights = self.route(x)
        host_indices = np.array(indices).reshape(-1)
        selected = sorted({int(i) for i in host_indices})
        if on_route is not None:
            on_route(selected)

        out = mx.zeros_like(x)
        for expert in selected:
            w1 = self.name(expert, "w1")
            w2 = self.name(expert, "w2")
            if self.store.is_resident(w1) and self.store.is_resident(w2):
                self.hits += 1
            else:
                self.misses += 1

            mask = indices == expert
            weight = (weights * mask.astype(mx.float32)).sum(axis=-1, keepdims=True)
            keep = mask.any(axis=-1, keepdims=True).astype(mx.float32)
            h = (x * keep) @ self.store.get(w1).T
            h = h * mx.sigmoid(h)  # SwiGLU-ish activation
            out = out + (h @ self.store.get(w2).T) * weight

        for expert in selected:
            self._freq[expert] = self._freq[expert] * 0.9 + 1.0
        self._history.append(set(selected))
        return out

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        total = self.hits + self.misses
        resident = sum(
            1
            for expert in range(self.num_experts)
            if all(self.store.is_resident(name) for name in self.names(expert))
        )
        return {
            "experts_total": self.num_experts,
            "experts_resident": resident,
            "expert_hits": self.hits,
            "expert_misses": self.misses,
            "expert_hit_rate": (self.hits / total) if total else 0.0,
            "expert_prefetches": self.prefetches,
            "top_k": self.top_k,
        }

    def expert_weights_np(self, expert: int) -> tuple[np.ndarray, np.ndarray]:
        """Reference accessor (tests): materialise both matrices as NumPy."""
        w1 = np.array(self.store.get(self.name(expert, "w1")))
        w2 = np.array(self.store.get(self.name(expert, "w2")))
        return w1, w2
