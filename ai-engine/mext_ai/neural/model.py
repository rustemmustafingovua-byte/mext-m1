"""A small but real transformer whose linear weights live in the cold store.

Why a synthetic model? So that the prefetch machinery can be verified on any
machine without downloading a 4 GiB checkpoint: the forward pass, the MLX
graph, the SSD streaming, the residency window and the memory guard are all
real, only the weights are random.

For production weights, :meth:`MextModelRunner.load` first tries to wrap a real
``mlx_lm`` model (when installed) using exactly the same primitives
(:class:`StreamedLinear`).
"""

from __future__ import annotations

from typing import Callable

from ..memory import HAS_MLX
from .moe import ExpertPager

if HAS_MLX:  # pragma: no cover - exercised on Apple Silicon
    import mlx.core as mx
    import mlx.nn as nn
else:  # pragma: no cover - import-time shim for non-MLX hosts
    mx = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


def _require_mlx() -> None:
    if not HAS_MLX:
        raise RuntimeError("MLX is not installed (pip install mlx)")


class StreamedLinear:
    """``y = x @ W.T + b`` where ``W`` is fetched from :class:`ColdStore`."""

    def __init__(self, store, name: str, in_dim: int, out_dim: int, bias: bool = False):
        _require_mlx()
        self.store = store
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_bias = bias

    def __call__(self, x):
        weight = self.store.get(self.name + ".weight")
        y = x @ weight.T
        if self.has_bias:
            y = y + self.store.get(self.name + ".bias")
        return y


class StreamedQuantizedLinear:
    """4/8-bit MLX quantised projection with weights streamed from SSD.

    MLX stores a quantised tensor as three arrays: packed ``weight`` (uint32),
    ``scales`` and ``biases`` (one pair per group). We keep all three on SSD and
    rebuild the product with :func:`mx.quantized_matmul` — bit-for-bit the same
    kernel ``nn.QuantizedLinear`` uses, so numerics match a resident model.
    """

    def __init__(
        self,
        store,
        name: str,
        in_dim: int,
        out_dim: int,
        group_size: int = 64,
        bits: int = 4,
        bias: bool = False,
    ):
        _require_mlx()
        self.store = store
        self.name = name
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_size = int(group_size)
        self.bits = int(bits)
        self.has_bias = bias

    def __call__(self, x):
        weight = self.store.get(self.name + ".weight")
        scales = self.store.get(self.name + ".scales")
        biases = self.store.get(self.name + ".biases")
        y = mx.quantized_matmul(
            x,
            weight,
            scales,
            biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if self.has_bias:
            y = y + self.store.get(self.name + ".bias")
        return y


class RMSNorm:
    """Hot weight (dim floats) — never streamed."""

    def __init__(self, dim: int, weight=None):
        _require_mlx()
        self.weight = weight if weight is not None else mx.ones((dim,))
        self.eps = 1e-5

    def __call__(self, x):
        variance = (x * x).mean(axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class Block:
    """Pre-norm transformer block; every projection is streamed from SSD.

    With ``moe=True`` the dense FFN is replaced by a router + paged experts,
    so a token only pays RAM for the experts it actually selected.
    """

    def __init__(
        self,
        store,
        index: int,
        dim: int,
        heads: int,
        ffn: int,
        moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        _require_mlx()
        self.index = index
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = (self.head_dim) ** -0.5
        prefix = f"model.layers.{index}"
        self.moe = (
            ExpertPager(store, f"{prefix}.mlp", dim, ffn, num_experts, top_k)
            if moe
            else None
        )

        # Cold weights: create once, persist, then forget (only refs remain).
        specs = {
            f"{prefix}.self_attn.q_proj.weight": (dim, dim),
            f"{prefix}.self_attn.k_proj.weight": (dim, dim),
            f"{prefix}.self_attn.v_proj.weight": (dim, dim),
            f"{prefix}.self_attn.o_proj.weight": (dim, dim),
        }
        if self.moe is None:
            specs.update(
                {
                    f"{prefix}.mlp.gate_proj.weight": (ffn, dim),
                    f"{prefix}.mlp.up_proj.weight": (ffn, dim),
                    f"{prefix}.mlp.down_proj.weight": (dim, ffn),
                }
            )
        for name, shape in specs.items():
            store.put(name, mx.random.normal(shape) * 0.02)

        self.n1 = RMSNorm(dim)
        self.n2 = RMSNorm(dim)
        self.q = StreamedLinear(store, f"{prefix}.self_attn.q_proj", dim, dim)
        self.k = StreamedLinear(store, f"{prefix}.self_attn.k_proj", dim, dim)
        self.v = StreamedLinear(store, f"{prefix}.self_attn.v_proj", dim, dim)
        self.o = StreamedLinear(store, f"{prefix}.self_attn.o_proj", dim, dim)
        if self.moe is None:
            self.gate = StreamedLinear(store, f"{prefix}.mlp.gate_proj", dim, ffn)
            self.up = StreamedLinear(store, f"{prefix}.mlp.up_proj", dim, ffn)
            self.down = StreamedLinear(store, f"{prefix}.mlp.down_proj", ffn, dim)

    def __call__(self, x, mask, on_route=None):
        batch, length, dim = x.shape
        heads, head_dim = self.heads, self.head_dim

        h = self.n1(x)
        q = self.q(h).reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        weights = mx.softmax(scores, axis=-1)
        attn = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, length, dim)
        x = x + self.o(attn)

        h2 = self.n2(x)
        if self.moe is not None:
            sink = (lambda ids: on_route(self.index, ids)) if on_route is not None else None
            x = x + self.moe(h2, on_route=sink)
        else:
            # SwiGLU
            gate = self.gate(h2)
            act = gate * mx.sigmoid(gate)
            x = x + self.down(act * self.up(h2))
        return x


class MiniTransformer:
    """4–8 layer decoder used by the demo / self-test path."""

    def __init__(
        self,
        store,
        vocab: int = 8192,
        dim: int = 384,
        layers: int = 6,
        heads: int = 6,
        ffn: int = 1024,
        moe_layers: int = 0,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        _require_mlx()
        self.vocab = vocab
        self.dim = dim
        self.n_layers = layers
        self.heads = heads
        self.moe_layers = max(0, min(int(moe_layers), layers))
        self.num_experts = max(1, int(num_experts))
        self.top_k = max(1, int(top_k))

        # Hot tensors stay in unified memory for the whole session.
        self.embed = mx.random.normal((vocab, dim)) * 0.02
        self.blocks = [
            Block(
                store,
                i,
                dim,
                heads,
                ffn,
                moe=i >= layers - self.moe_layers,
                num_experts=num_experts,
                top_k=top_k,
            )
            for i in range(layers)
        ]
        self.norm = RMSNorm(dim)
        store.put("lm_head.weight", mx.random.normal((vocab, dim)) * 0.02)
        self.lm_head = StreamedLinear(store, "lm_head", dim, vocab)

    # ------------------------------------------------------------------
    def blocks_iterator(self):
        return enumerate(self.blocks)

    def forward(self, ids, on_layer: Callable[[int], None] | None = None, on_route=None):
        _require_mlx()
        batch, length = ids.shape
        x = self.embed[ids]
        mask = self._causal_mask(length)
        for index, block in self.blocks_iterator():
            if on_layer is not None:
                on_layer(index)
            x = block(x, mask, on_route=on_route)
        x = self.norm(x)
        return self.lm_head(x)

    def __call__(self, ids, on_layer=None, on_route=None):
        return self.forward(ids, on_layer=on_layer, on_route=on_route)

    def moe_at(self, index: int):
        if 0 <= index < len(self.blocks):
            return self.blocks[index].moe
        return None

    def pagers(self):
        return [block.moe for block in self.blocks if block.moe is not None]

    def expert_metrics(self) -> dict:
        pagers = self.pagers()
        if not pagers:
            return {
                "experts_total": 0,
                "experts_resident": 0,
                "expert_hits": 0,
                "expert_misses": 0,
                "expert_hit_rate": 0.0,
                "expert_prefetches": 0,
                "moe_layers": 0,
            }
        totals = [p.metrics() for p in pagers]
        hits = sum(m["expert_hits"] for m in totals)
        misses = sum(m["expert_misses"] for m in totals)
        return {
            "experts_total": sum(m["experts_total"] for m in totals),
            "experts_resident": sum(m["experts_resident"] for m in totals),
            "expert_hits": hits,
            "expert_misses": misses,
            "expert_hit_rate": (hits / (hits + misses)) if (hits + misses) else 0.0,
            "expert_prefetches": sum(m["expert_prefetches"] for m in totals),
            "moe_layers": len(pagers),
        }

    def _causal_mask(self, length: int):
        cache = getattr(self, "_mask_cache", None)
        if cache is not None and cache[0] == length:
            return cache[1]
        neg = mx.full((length, length), -1e9)
        mask = mx.triu(neg, 1).reshape(1, 1, length, length)
        self._mask_cache = (length, mask)
        return mask

    def hot_tensor_sizes(self) -> dict[str, int]:
        """Names and sizes of the tensors that stay in unified memory."""
        sizes = {
            "model.embed_tokens.weight": int(self.embed.nbytes),
            "model.norm.weight": int(self.norm.weight.nbytes),
        }
        for index, block in enumerate(self.blocks):
            sizes[f"model.layers.{index}.input_layernorm.weight"] = int(block.n1.weight.nbytes)
            sizes[f"model.layers.{index}.post_attention_layernorm.weight"] = int(block.n2.weight.nbytes)
            if block.moe is not None:
                sizes[f"model.layers.{index}.moe.router"] = int(block.moe.router.nbytes)
        return sizes

    def hot_bytes(self) -> int:
        total = self.embed.nbytes + self.norm.weight.nbytes
        for block in self.blocks:
            total += block.n1.weight.nbytes + block.n2.weight.nbytes
            if block.moe is not None:
                total += block.moe.router.nbytes
        return int(total)
