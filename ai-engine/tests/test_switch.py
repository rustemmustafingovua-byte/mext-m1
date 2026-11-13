import time

import numpy as np
import pytest

from mext_ai.memory import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed")

if HAS_MLX:
    import mlx.core as mx
    from mlx import nn

from mext_ai.neural.store import ColdStore
from mext_ai.neural.switch import PagedSwitchGLU
from mext_ai.neural.wrap import wrap_cold_modules

MB = 1024 * 1024


def _switch_glu(in_dim=16, hidden=32, experts=4, bias=False):
    switch_layers = pytest.importorskip("mlx_lm.models.switch_layers")
    return switch_layers.SwitchGLU(in_dim, hidden, experts, bias=bias)


def _inputs(batch=2, length=5, in_dim=16, experts=4, top_k=2, seed=0):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(batch, length, in_dim)).astype(np.float32))
    indices = mx.array(rng.integers(0, experts, size=(batch, length, top_k)).astype(np.uint32))
    return x, indices


def test_paged_switch_matches_resident(tmp_path):
    glu = _switch_glu()
    x, indices = _inputs()
    expected = glu(x, indices)
    mx.eval(expected)

    store = ColdStore(tmp_path / "sw", 16 * MB)
    paged = PagedSwitchGLU.from_module(store, "model.layers.0.mlp.switch_mlp", glu)
    got = paged(x, indices)
    mx.eval(got)

    assert got.shape == expected.shape == (2, 5, 2, 16)
    assert float(mx.max(mx.abs(got - expected))) < 1e-5
    # Only the experts actually selected were touched.
    assert paged.metrics()["expert_hits"] + paged.metrics()["expert_misses"] > 0
    store.stop()


def test_paged_switch_quantized_matches(tmp_path):
    glu = _switch_glu(in_dim=64, hidden=128)
    nn.quantize(glu, group_size=32, bits=4)
    assert type(glu.gate_proj).__name__ == "QuantizedSwitchLinear"
    x, indices = _inputs(in_dim=64)
    expected = glu(x, indices)
    mx.eval(expected)

    store = ColdStore(tmp_path / "swq", 16 * MB)
    paged = PagedSwitchGLU.from_module(store, "model.layers.1.mlp.switch_mlp", glu)
    got = paged(x, indices)
    mx.eval(got)

    assert float(mx.max(mx.abs(got - expected))) < 1e-4
    # 4-bit payload is well below half of the fp32 expert bank.
    fp32_bytes = 4 * (3 * 128 * 64 + 3 * 64 * 128) * 4
    stats = store.stats()
    assert stats["cold_bytes"] < fp32_bytes / 2, stats
    store.stop()


def test_paged_switch_budget_and_prefetch(tmp_path):
    glu = _switch_glu(in_dim=256, hidden=256, experts=8)
    store = ColdStore(tmp_path / "swb", 1 * MB)
    paged = PagedSwitchGLU.from_module(store, "model.layers.2.mlp.switch_mlp", glu)
    budget = store.stats()["budget_bytes"]

    rng = np.random.default_rng(4)
    for _ in range(6):
        x = mx.array(rng.normal(size=(1, 8, 256)).astype(np.float32))
        indices = mx.array(rng.integers(0, 8, size=(1, 8, 2)).astype(np.uint32))
        paged.prefetch(paged.predicted_experts())
        time.sleep(0.05)
        paged(x, indices)
        stats = store.stats()
        assert stats["resident_bytes"] <= budget, stats

    metrics = paged.metrics()
    assert metrics["experts_total"] == 8
    assert metrics["experts_resident"] < 8
    assert metrics["expert_prefetches"] > 0
    store.stop()


def test_mixtral_end_to_end(tmp_path):
    mixtral = pytest.importorskip("mlx_lm.models.mixtral")

    args = mixtral.ModelArgs(
        model_type="mixtral",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_experts_per_tok=2,
        num_key_value_heads=2,
        num_local_experts=4,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )
    model = mixtral.Model(args)
    nn.quantize(model, group_size=64, bits=4)

    ids = mx.array([[1, 4, 7, 11, 13]])
    baseline = model(ids)
    mx.eval(baseline)

    store = ColdStore(tmp_path / "mix", 64 * MB)
    result = wrap_cold_modules(model, store)
    assert result["wrapped"] >= 2
    assert len(result["switches"]) == args.num_hidden_layers

    for layer in range(args.num_hidden_layers):
        block = model.model.layers[layer].block_sparse_moe
        assert type(block.switch_mlp).__name__ == "PagedSwitchGLU"
        # Routers stay hot; they are the prediction key.
        assert type(block.gate).__name__ == "QuantizedLinear" or type(block.gate).__name__ == "Linear"

    out = model(ids)
    mx.eval(out)
    assert out.shape == baseline.shape
    assert float(mx.max(mx.abs(out - baseline))) < 1e-4

    metrics = result["switches"][0].metrics()
    assert metrics["experts_total"] == args.num_local_experts
    assert metrics["layer_index"] >= 0

    # Every expert projection lives on SSD.
    for expert in range(args.num_local_experts):
        for kind in ("gate", "up", "down"):
            assert any(
                f".experts.{expert}.{kind}.weight" in name for name in store.refs
            )
    store.stop()
