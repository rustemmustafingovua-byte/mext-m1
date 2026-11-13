import time

import numpy as np
import pytest

from mext_ai.memory import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed")

if HAS_MLX:
    import mlx.core as mx

from mext_ai.neural.model import MiniTransformer
from mext_ai.neural.moe import ExpertPager
from mext_ai.neural.runner import MextModelRunner
from mext_ai.neural.store import ColdStore

MB = 1024 * 1024


def _reference(pager: ExpertPager, x: np.ndarray) -> np.ndarray:
    """Independent dense reference: every expert computed, top-k gathered."""
    router = np.array(pager.router)
    logits = x @ router.T
    idx = np.argsort(-logits, axis=-1)[..., : pager.top_k]
    top = np.take_along_axis(logits, idx, axis=-1)
    exp = np.exp(top - top.max(axis=-1, keepdims=True))
    weights = exp / exp.sum(axis=-1, keepdims=True)

    ref = np.zeros_like(x)
    for expert in range(pager.num_experts):
        w1, w2 = pager.expert_weights_np(expert)
        h = x @ w1.T
        h = h * (1.0 / (1.0 + np.exp(-h)))
        y = h @ w2.T
        for slot in range(pager.top_k):
            chosen = (idx[..., slot] == expert)[..., None]
            ref += np.where(chosen, weights[..., slot][..., None] * y, 0.0)
    return ref


def test_matches_dense_reference(tmp_path):
    store = ColdStore(tmp_path / "moe", 16 * MB)
    pager = ExpertPager(
        store, "model.layers.0.mlp", dim=32, hidden=48, num_experts=6, top_k=2
    )
    x = np.random.default_rng(3).normal(size=(2, 4, 32)).astype(np.float32)
    out = np.array(pager(mx.array(x)))
    ref = _reference(pager, x)
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-5)
    store.stop()


def test_expert_cache_is_bounded(tmp_path):
    # The store floors its budget at 1 MiB, so use experts large enough to force
    # evictions inside that window (128×192 fp32 = 96 KiB per matrix).
    store = ColdStore(tmp_path / "moe-budget", 1 * MB)
    pager = ExpertPager(
        store, "model.layers.1.mlp", dim=128, hidden=192, num_experts=8, top_k=2
    )
    rng = np.random.default_rng(11)
    budget = store.stats()["budget_bytes"]
    assert budget <= 1 * MB
    for _ in range(10):
        x = mx.array(rng.normal(size=(1, 6, 128)).astype(np.float32))
        pager(x)
        stats = store.stats()
        assert stats["resident_bytes"] <= budget, stats
    metrics = pager.metrics()
    assert metrics["experts_total"] == 8
    assert metrics["experts_resident"] < 8, metrics
    assert metrics["expert_hits"] + metrics["expert_misses"] > 0
    store.stop()


def test_router_guided_prefetch_improves_hits(tmp_path):
    # Two adjacent MoE layers with identical routers, like a model where the same
    # domains activate the same experts layer after layer.
    store = ColdStore(tmp_path / "moe-prefetch", 1 * MB)
    current = ExpertPager(
        store, "model.layers.2.mlp", dim=128, hidden=192, num_experts=8, top_k=2
    )
    upcoming = ExpertPager(
        store, "model.layers.3.mlp", dim=128, hidden=192, num_experts=8, top_k=2
    )
    upcoming.router = current.router
    x = mx.array(np.random.default_rng(5).normal(size=(1, 8, 128)).astype(np.float32))

    # Current layer runs cold and learns the routing; the runner then prefetches
    # the predicted set for the next layer.
    current(x)
    predicted = current.predicted_experts()
    queued = upcoming.prefetch(predicted)
    assert queued > 0, predicted

    deadline = time.time() + 3
    while time.time() < deadline:
        remaining = [n for n in upcoming.names(predicted[0]) if not store.is_resident(n)]
        if not remaining:
            break
        time.sleep(0.02)

    upcoming(x)
    upcoming(x)
    metrics = upcoming.metrics()
    assert metrics["expert_hits"] > 0, metrics
    assert metrics["expert_hit_rate"] > 0.0
    assert metrics["expert_prefetches"] > 0
    store.stop()


def test_moe_transformer_forward(tmp_path):
    store = ColdStore(tmp_path / "moe-tx", 8 * MB)
    model = MiniTransformer(
        store,
        vocab=256,
        dim=64,
        layers=3,
        heads=4,
        ffn=96,
        moe_layers=2,
        num_experts=4,
        top_k=2,
    )
    ids = mx.array([[1, 2, 3, 4, 5]])
    routed = []

    logits = model(ids, on_route=lambda layer, experts: routed.append((layer, experts)))
    mx.eval(logits)
    assert logits.shape == (1, 5, 256)
    assert routed, "no routing events emitted"

    metrics = model.expert_metrics()
    assert metrics["moe_layers"] == 2
    assert metrics["experts_total"] == 8
    assert metrics["experts_resident"] >= 0
    assert model.moe_at(0) is None
    assert model.moe_at(1) is not None and model.moe_at(2) is not None

    # MoE blocks must not carry dead dense-MLP weights on SSD.
    dense_layers = [name for name in store.refs if ".layers.1.mlp.gate_proj" in name]
    assert not dense_layers, dense_layers
    assert any(".layers.0.mlp.gate_proj" in name for name in store.refs)
    store.stop()


def test_runner_wires_moe_metrics(tmp_path):
    runner = MextModelRunner("synthetic", tmp_path / "runner", 512 * MB, moe_layers=2)
    runner.load()
    try:
        assert runner.metrics()["available"], runner.load_error
        out = runner.generate("def fib(n):", max_tokens=6)
        assert out["generated"] == 6
        metrics = runner.metrics()
        assert metrics["moe_layers"] == 2
        assert metrics["experts_total"] >= 8
        assert metrics["hot_bytes"] > 0
        assert "expert_hit_rate" in metrics
    finally:
        runner.stop()
