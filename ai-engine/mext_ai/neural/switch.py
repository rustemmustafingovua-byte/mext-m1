"""Paged drop-in replacements for mlx-lm's ``SwitchGLU`` / ``SwitchLinear``.

Real MoE checkpoints do not use plain ``nn.Linear``: experts are batched into a
single tensor ``(num_experts, out, in)`` and applied with ``mx.gather_mm`` /
``mx.gather_qmm``. That keeps the whole expert bank resident — 64 experts ×
3 projections × 4096² × 4 B ≈ 12 GiB, i.e. exactly what breaks a 8/16 GB Mac.

:class:`PagedSwitchGLU` keeps the router hot, writes every expert projection to
SSD (fp16 or 4/8-bit quantised), and evaluates **only the experts the router
selected**, rebuilding the ``(batch, seq, top_k, out)`` layout mlx-lm expects.

There is no accuracy trade-off: the same weights and the same quantised kernels
are used, so logits match a resident model (covered by tests/test_switch.py).
"""

from __future__ import annotations

import logging
import re
from collections import deque

import numpy as np

from ..memory import HAS_MLX

if HAS_MLX:  # pragma: no cover - Apple Silicon only
    import mlx.core as mx

log = logging.getLogger("mext_ai.switch")

__all__ = ["PagedSwitchGLU", "PagedSwitchProjection"]

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


class PagedSwitchProjection:
    """One projection (gate/up/down) with per-expert tensors on SSD."""

    def __init__(
        self,
        store,
        switch_name: str,
        kind: str,
        in_dim: int,
        out_dim: int,
        num_experts: int,
        quantized: bool = False,
        bits: int | None = None,
        group_size: int = 64,
        bias: bool = False,
    ) -> None:
        self.store = store
        self.switch_name = switch_name
        self.kind = kind
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_experts = int(num_experts)
        self.quantized = bool(quantized)
        self.bits = int(bits or 4)
        self.group_size = int(group_size)
        self.has_bias = bool(bias)

    def names(self, expert: int) -> list[str]:
        base = f"{self.switch_name}.experts.{expert}.{self.kind}"
        names = [base + ".weight"]
        if self.quantized:
            names += [base + ".scales", base + ".biases"]
        if self.has_bias:
            names.append(base + ".bias")
        return names

    def is_resident(self, expert: int) -> bool:
        return all(self.store.is_resident(name) for name in self.names(expert))

    def prefetch(self, experts) -> int:
        queued = 0
        for expert in experts:
            queued += self.store.prefetch(self.names(int(expert)))
        return queued

    def _project(self, x, expert: int):
        base = f"{self.switch_name}.experts.{expert}.{self.kind}"
        if self.quantized:
            y = mx.quantized_matmul(
                x,
                self.store.get(base + ".weight"),
                self.store.get(base + ".scales"),
                self.store.get(base + ".biases"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
            )
        else:
            y = x @ self.store.get(base + ".weight").T
        if self.has_bias:
            y = y + self.store.get(base + ".bias")
        return y

    def __call__(self, x, indices):
        """Apply the selected experts.

        * ``x`` of shape ``(..., in)`` with ``indices`` ``(..., k)`` —
          the projection is shared by the ``k`` slots (gate/up).
        * ``x`` of shape ``(..., k, in)`` — one input per routing slot, i.e. the
          activated output that feeds ``down_proj`` in a SwitchGLU.
        """
        host = np.array(indices).reshape(-1)
        selected = sorted({int(v) for v in host})
        aligned = (
            x.ndim == indices.ndim + 1
            and tuple(x.shape[:-2]) == tuple(indices.shape[:-1])
            and x.shape[-2] == indices.shape[-1]
        )
        if aligned:
            out = mx.zeros((*x.shape[:-1], self.out_dim), dtype=x.dtype)
            for expert in selected:
                mask = (indices == expert)[..., None].astype(x.dtype)
                out = out + self._project(x, expert) * mask
            return out
        top_k = indices.shape[-1]
        out = mx.zeros((*x.shape[:-1], top_k, self.out_dim), dtype=x.dtype)
        for expert in selected:
            mask = (indices == expert)[..., None].astype(x.dtype)
            out = out + self._project(x, expert)[..., None, :] * mask
        return out


class PagedSwitchGLU:
    """Router-guided paging wrapper around mlx-lm's ``SwitchGLU``."""

    def __init__(
        self,
        store,
        name: str,
        gate: PagedSwitchProjection,
        up: PagedSwitchProjection,
        down: PagedSwitchProjection,
        activation=None,
    ) -> None:
        self.store = store
        self.name = name
        self.gate_proj = gate
        self.up_proj = up
        self.down_proj = down
        self.activation = activation
        match = _LAYER_RE.search(name)
        self.layer_index = int(match.group(1)) if match else -1
        self.next_switch: PagedSwitchGLU | None = None

        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self._history: deque[set[int]] = deque(maxlen=16)
        self._freq = [0.0] * gate.num_experts

    # ------------------------------------------------------------------
    @classmethod
    def from_module(cls, store, name: str, module) -> PagedSwitchGLU:
        """Move every expert of an mlx-lm ``SwitchGLU`` to the store."""
        gate_mod, up_mod, down_mod = module.gate_proj, module.up_proj, module.down_proj
        quantized = type(gate_mod).__name__ == "QuantizedSwitchLinear"
        bits = int(getattr(gate_mod, "bits", 4)) if quantized else None
        group_size = int(getattr(gate_mod, "group_size", 64))
        num_experts = int(gate_mod.weight.shape[0])
        in_dim = int(gate_mod.input_dims)
        hidden = int(gate_mod.output_dims)

        def persist(mod, kind: str) -> PagedSwitchProjection:
            has_bias = "bias" in mod
            projection = PagedSwitchProjection(
                store,
                name,
                kind,
                in_dim if kind != "down" else hidden,
                hidden if kind != "down" else in_dim,
                num_experts,
                quantized=quantized,
                bits=bits,
                group_size=group_size,
                bias=has_bias,
            )
            for expert in range(num_experts):
                base = f"{name}.experts.{expert}.{kind}"
                store.put(base + ".weight", mod.weight[expert])
                if quantized:
                    store.put(base + ".scales", mod.scales[expert])
                    if mod.biases is not None:
                        store.put(base + ".biases", mod.biases[expert])
                if has_bias:
                    store.put(base + ".bias", mod.bias[expert])
            return projection

        return cls(
            store,
            name,
            persist(gate_mod, "gate"),
            persist(up_mod, "up"),
            persist(down_mod, "down"),
            activation=getattr(module, "activation", None),
        )

    # ------------------------------------------------------------------
    def _observe(self, selected: list[int]) -> None:
        for expert in selected:
            if all(
                projection.is_resident(expert)
                for projection in (self.gate_proj, self.up_proj, self.down_proj)
            ):
                self.hits += 1
            else:
                self.misses += 1
            self._freq[expert] = self._freq[expert] * 0.9 + 1.0
        self._history.append(set(selected))

    def predicted_experts(self, limit: int | None = None) -> list[int]:
        scores = list(self._freq)
        for snapshot in self._history:
            for expert in snapshot:
                scores[expert] += 0.5
        keep = limit if limit is not None else max(2, 4)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return sorted(ranked[:keep])

    def prefetch(self, experts) -> int:
        queued = 0
        for projection in (self.gate_proj, self.up_proj, self.down_proj):
            queued += projection.prefetch(experts)
        self.prefetches += queued
        return queued

    def __call__(self, x, indices, sorted_indices: bool = False):
        indices = mx.stop_gradient(indices)
        host = np.array(indices).reshape(-1)
        selected = sorted({int(v) for v in host})
        self._observe(selected)

        # Router-guided lookahead: this layer's routing is the best cheap
        # predictor for the next MoE layer of the same model.
        if self.next_switch is not None:
            predicted = set(selected) | set(self.next_switch.predicted_experts())
            self.next_switch.prefetch(sorted(predicted))

        gate = self.gate_proj(x, indices)
        up = self.up_proj(x, indices)
        act = (
            self.activation(up, gate)
            if self.activation is not None
            else up * mx.sigmoid(gate)
        )
        return self.down_proj(act, indices)

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        total = self.hits + self.misses
        resident = sum(
            1
            for expert in range(self.gate_proj.num_experts)
            if all(
                projection.is_resident(expert)
                for projection in (self.gate_proj, self.up_proj, self.down_proj)
            )
        )
        return {
            "experts_total": self.gate_proj.num_experts,
            "experts_resident": resident,
            "expert_hits": self.hits,
            "expert_misses": self.misses,
            "expert_hit_rate": (self.hits / total) if total else 0.0,
            "expert_prefetches": self.prefetches,
            "top_k": 0,
            "layer_index": self.layer_index,
        }
