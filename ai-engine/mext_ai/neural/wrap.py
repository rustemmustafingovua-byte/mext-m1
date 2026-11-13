"""Turn a loaded MLX model into a streamed one.

The transformation is mechanical: walk ``named_modules()``, and for every
``nn.Linear`` / ``nn.QuantizedLinear`` whose tensor is classified as cold, write
its weights to the :class:`ColdStore` and replace the module with a streamed
equivalent. Everything else (embeddings, norms, routers) stays in unified
memory, untouched.

Quantised checkpoints are supported: MLX stores them as ``weight`` (uint32
packed), ``scales`` and ``biases``; all three are persisted per tensor and the
projection is rebuilt with ``mx.quantized_matmul`` — the same kernel
``nn.QuantizedLinear`` uses.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .plan import COLD, classify
from .store import ColdStore

log = logging.getLogger("mext_ai.wrap")

__all__ = ["hot_bytes", "wrap_cold_modules"]


def wrap_cold_modules(model: Any, store: ColdStore) -> dict:
    """Replace cold projections in ``model`` in place.

    Returns ``{"wrapped": n, "sizes": {...}, "skipped": n}`` where ``sizes`` is
    the name→bytes map used to build the :class:`WeightPlan`.
    """
    from .model import StreamedLinear, StreamedQuantizedLinear

    modules = dict(model.named_modules())
    wrapped = 0
    skipped = 0
    sizes: dict[str, int] = {}
    switches = []

    # MoE first: replace whole SwitchGLU banks (their children are
    # SwitchLinear, which the generic loop below must not touch).
    for name, module in list(modules.items()):
        if not name or type(module).__name__ != "SwitchGLU":
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = modules.get(parent_name) if parent_name else model
        if parent is None:
            continue
        from .switch import PagedSwitchGLU

        paged = PagedSwitchGLU.from_module(store, name, module)
        setattr(parent, attr, paged)
        switches.append(paged)
        wrapped += 1
    switches.sort(key=lambda switch: switch.layer_index)
    for current, upcoming in zip(switches, switches[1:]):  # noqa: B905
        current.next_switch = upcoming

    for name, module in list(modules.items()):
        kind = type(module).__name__
        if not name or kind not in ("Linear", "QuantizedLinear"):
            continue
        # MoE routers are tiny and needed for every token: never page them.
        if re.search(r"\.(gate|router)$", name):
            skipped += 1
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = modules.get(parent_name) if parent_name else model
        if parent is None:
            continue
        has_bias = getattr(module, "bias", None) is not None

        if kind == "QuantizedLinear":
            weight = module.weight
            bits = int(getattr(module, "bits", 4))
            group_size = int(getattr(module, "group_size", 64))
            # Packed uint32 carries 32/bits weights; classification must reason
            # about the fp32-equivalent size, not the packed one.
            effective = int(weight.nbytes * 32 / max(1, bits))
            sizes[name + ".weight"] = effective
            if classify(name + ".weight", effective) != COLD:
                skipped += 1
                continue
            store.put(name + ".weight", weight)
            store.put(name + ".scales", module.scales)
            store.put(name + ".biases", module.biases)
            sizes[name + ".scales"] = int(module.scales.nbytes)
            sizes[name + ".biases"] = int(module.biases.nbytes)
            if has_bias:
                store.put(name + ".bias", module.bias)
            out_dim = int(weight.shape[0])
            in_dim = int(module.scales.shape[1]) * group_size
            setattr(
                parent,
                attr,
                StreamedQuantizedLinear(
                    store,
                    name,
                    in_dim,
                    out_dim,
                    group_size=group_size,
                    bits=bits,
                    bias=has_bias,
                ),
            )
        else:
            weight = module.weight
            nbytes = int(weight.nbytes)
            sizes[name + ".weight"] = nbytes
            if classify(name + ".weight", nbytes) != COLD:
                skipped += 1
                continue
            store.put(name + ".weight", weight)
            if has_bias:
                store.put(name + ".bias", module.bias)
            setattr(
                parent,
                attr,
                StreamedLinear(
                    store,
                    name,
                    int(weight.shape[1]),
                    int(weight.shape[0]),
                    bias=has_bias,
                ),
            )
        wrapped += 1

    return {
        "wrapped": wrapped,
        "skipped": skipped,
        "sizes": sizes,
        "switches": switches,
    }


def hot_bytes(model: Any, store: ColdStore) -> int:
    """Bytes of tensors that remain resident in unified memory."""
    total = 0
    for name, module in dict(model.named_modules()).items():
        weight = getattr(module, "weight", None)
        if weight is not None and (name + ".weight") not in store.refs:
            total += int(getattr(weight, "nbytes", 0))
        scales = getattr(module, "scales", None)
        if scales is not None and (name + ".scales") not in store.refs:
            total += int(getattr(scales, "nbytes", 0))
    return total
