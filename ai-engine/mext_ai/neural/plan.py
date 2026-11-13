"""Static hot/cold classification of model tensors.

The rules are deliberately model-agnostic: they work on the *name* and the
*size* of a tensor, so they apply to Llama/Qwen/Mistral/MLP-Mixer equally.

Hot (kept in unified memory for the whole session):

* token embeddings (touched for every token),
* all norms (RMSNorm/LayerNorm — a few hundred KiB in total, and every layer
  needs them),
* MoE routers / gates (tiny but on the critical path),
* rotary / position tables,
* every bias and any tensor <= 64 KiB (streaming a 4 KiB tensor costs more in
  syscalls than it saves),
* the language-model head — the caller pins it explicitly because it is used
  on every decoding step (see :class:`mext_ai.neural.store.ColdStore.pin`).

Everything else (attention projections, MLP up/down/gate projections) is cold
and streamed from SSD by :class:`mext_ai.neural.runner.MextModelRunner`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "COLD",
    "HOT",
    "WeightPlan",
    "build_plan",
    "classify",
    "include_in_layers",
    "summarize",
]

HOT = "hot"
COLD = "cold"

SMALL_TENSOR_BYTES = 64 * 1024

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EMBED_RE = re.compile(r"(embed_tokens|word_embeddings|wte|(^|\.)embedding)")
_NORM_RE = re.compile(r"(layernorm|layer_norm|rmsnorm|(^|\.)norm(\.|$))")
_ROUTER_RE = re.compile(r"\.(router|gate)\.(weight|bias)$")
_POS_RE = re.compile(r"(rotary|inv_freq|position_embed|pos_emb)")
_EXPERT_RE = re.compile(r"\.experts\.\d+\.")


def classify(name: str, nbytes: int) -> str:
    """Return ``"hot"`` or ``"cold"`` for a single tensor."""
    lower = name.lower()
    if nbytes <= SMALL_TENSOR_BYTES:
        return HOT
    if _EMBED_RE.search(lower):
        return HOT
    if _NORM_RE.search(lower):
        return HOT
    if _ROUTER_RE.search(lower):
        return HOT
    if _POS_RE.search(lower):
        return HOT
    if lower.endswith(".bias"):
        return HOT
    return COLD


@dataclass
class WeightPlan:
    hot: dict[str, int] = field(default_factory=dict)
    cold: dict[str, int] = field(default_factory=dict)
    layers: dict[int, list[str]] = field(default_factory=dict)

    @property
    def hot_bytes(self) -> int:
        return sum(self.hot.values())

    @property
    def cold_bytes(self) -> int:
        return sum(self.cold.values())

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def names_for_layer(self, index: int) -> list[str]:
        return self.layers.get(index, [])

    def to_dict(self) -> dict:
        return {
            "hot_tensors": len(self.hot),
            "cold_tensors": len(self.cold),
            "hot_bytes": self.hot_bytes,
            "cold_bytes": self.cold_bytes,
            "layers": self.layer_count,
        }


def build_plan(sizes: dict[str, int]) -> WeightPlan:
    """Split a ``{tensor_name: nbytes}`` mapping into a :class:`WeightPlan`."""
    plan = WeightPlan()
    for name, nbytes in sizes.items():
        bucket = classify(name, int(nbytes))
        if bucket == HOT:
            plan.hot[name] = int(nbytes)
        else:
            plan.cold[name] = int(nbytes)
        # Expert tensors are paged by the router, not by the layer plan: adding
        # them would prefetch every expert and defeat the whole point of MoE
        # paging.
        if _EXPERT_RE.search(name):
            continue
        match = _LAYER_RE.search(name)
        if match:
            index = int(match.group(1))
            plan.layers.setdefault(index, []).append(name)
    # Only cold tensors need to be prefetched per layer.
    for index, names in plan.layers.items():
        plan.layers[index] = [n for n in names if n in plan.cold]
    plan.layers = {i: names for i, names in sorted(plan.layers.items()) if names}
    return plan


def include_in_layers(plan: WeightPlan, names: Iterable[str]) -> None:
    """Make sure every stored tensor is prefetched together with its layer.

    Quantised models keep ``weight`` + ``scales`` + ``biases`` per projection;
    classifying them independently would leave the small side tensors out of the
    layer plan, and a prefetched layer would then still hit the disk for its
    scales.
    """
    for name in names:
        if _EXPERT_RE.search(name):
            continue
        match = _LAYER_RE.search(name)
        if not match:
            continue
        index = int(match.group(1))
        bucket = plan.layers.setdefault(index, [])
        if name not in bucket:
            bucket.append(name)
    plan.layers = {
        index: sorted(set(bucket))
        for index, bucket in sorted(plan.layers.items())
        if bucket
    }


def summarize(plan: WeightPlan) -> str:
    return (
        f"hot {plan.hot_bytes / 2**20:.1f} MiB in {len(plan.hot)} tensors, "
        f"cold {plan.cold_bytes / 2**20:.1f} MiB in {len(plan.cold)} tensors "
        f"across {plan.layer_count} layers"
    )


def iter_layer_names(plan: WeightPlan) -> Iterable[tuple[int, list[str]]]:
    return plan.layers.items()
