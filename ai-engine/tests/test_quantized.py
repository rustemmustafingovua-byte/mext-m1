import numpy as np
import pytest

from mext_ai.memory import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed")

if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn

from mext_ai.neural.model import StreamedQuantizedLinear
from mext_ai.neural.plan import build_plan, include_in_layers
from mext_ai.neural.store import ColdStore
from mext_ai.neural.wrap import wrap_cold_modules

MB = 1024 * 1024


def _quantized_reference(in_dim, out_dim, group_size=64, bits=4):
    weight = mx.random.normal((out_dim, in_dim)).astype(mx.float32)
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=bits)
    mx.eval(packed, scales, biases)
    reference = nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=group_size, bits=bits)
    reference.weight = packed
    reference.scales = scales
    reference.biases = biases
    return weight, packed, scales, biases, reference


def test_streamed_quantized_matches_nn_quantized_linear(tmp_path):
    in_dim, out_dim = 256, 128
    weight, packed, scales, biases, reference = _quantized_reference(in_dim, out_dim)

    store = ColdStore(tmp_path / "q", 32 * MB)
    store.put("proj.weight", packed)
    store.put("proj.scales", scales)
    store.put("proj.biases", biases)

    layer = StreamedQuantizedLinear(store, "proj", in_dim, out_dim, group_size=64, bits=4)
    x = mx.random.normal((2, 5, in_dim)).astype(mx.float32)
    expected = reference(x)
    got = layer(x)
    mx.eval(expected, got)

    assert got.shape == expected.shape == (2, 5, out_dim)
    assert float(mx.max(mx.abs(got - expected))) < 1e-5

    # 4-bit payload is smaller than the fp32 tensor it represents.
    stats = store.stats()
    assert stats["cold_bytes"] < weight.nbytes
    # Second call must come from residency, not the SSD.
    layer(x)
    assert store.stats()["hits"] >= 1
    store.stop()


def test_streamed_quantized_with_bias(tmp_path):
    in_dim, out_dim = 128, 64
    weight = mx.random.normal((out_dim, in_dim)).astype(mx.float32)
    packed, scales, biases = mx.quantize(weight, group_size=32, bits=8)
    bias = mx.random.normal((out_dim,)).astype(mx.float32)
    x = mx.random.normal((3, in_dim)).astype(mx.float32)
    expected = mx.quantized_matmul(
        x, packed, scales, biases, transpose=True, group_size=32, bits=8
    ) + bias
    mx.eval(expected)

    store = ColdStore(tmp_path / "qb", 8 * MB)
    store.put("p.weight", packed)
    store.put("p.scales", scales)
    store.put("p.biases", biases)
    store.put("p.bias", bias)
    got = StreamedQuantizedLinear(store, "p", in_dim, out_dim, group_size=32, bits=8, bias=True)(x)
    mx.eval(got)
    assert float(mx.max(mx.abs(got - expected))) < 1e-4
    store.stop()


def test_wrap_tiny_quantized_llama(tmp_path):
    llama = pytest.importorskip("mlx_lm.models.llama")

    # Big enough that projections exceed the 64 KiB "hot" threshold, like a
    # real checkpoint (a 64-dim toy would classify everything as hot).
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=512,
        num_hidden_layers=2,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=512,
        rope_theta=10000.0,
    )
    model = llama.Model(args)
    nn.quantize(model, group_size=64, bits=4)
    ids = mx.array([[1, 5, 9, 13, 21]])
    baseline = model(ids)
    mx.eval(baseline)

    store = ColdStore(tmp_path / "llama", 64 * MB)
    result = wrap_cold_modules(model, store)
    assert result["wrapped"] > 0
    assert result["skipped"] >= 0

    # Streamed modules are deliberately not nn.Module instances, so they are
    # invisible to named_modules(); inspect the attributes directly.
    attention = model.model.layers[0].self_attn
    assert type(attention.q_proj).__name__ == "StreamedQuantizedLinear"
    assert type(attention.k_proj).__name__ == "StreamedQuantizedLinear"
    assert type(model.model.layers[0].mlp.down_proj).__name__ == "StreamedQuantizedLinear"
    # Norms and embeddings stay resident.
    assert type(model.model.layers[0].input_layernorm).__name__ == "RMSNorm"
    assert type(model.model.embed_tokens).__name__ == "QuantizedEmbedding"

    out = model(ids)
    mx.eval(out)
    assert out.shape == baseline.shape
    assert float(mx.max(mx.abs(out - baseline))) < 1e-4

    plan = build_plan(result["sizes"])
    include_in_layers(plan, store.refs.keys())
    layer0 = plan.names_for_layer(0)
    assert any(name.endswith(".weight") for name in layer0)
    assert any(name.endswith(".scales") for name in layer0), layer0
    assert any(name.endswith(".biases") for name in layer0)

    stats = store.stats()
    assert stats["cold_bytes"] > 0
    # Everything we replaced is on SSD, and the store is the only owner.
    assert len(store.refs) >= result["wrapped"] * 3
    store.stop()


def test_wrap_skips_hot_projections(tmp_path):
    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.small = nn.Linear(8, 8)     # 256 B -> hot
            self.large = nn.Linear(256, 256)  # 256 KiB -> cold

    model = Holder()
    store = ColdStore(tmp_path / "mix", 32 * MB)
    result = wrap_cold_modules(model, store)
    assert result["wrapped"] == 1
    assert result["skipped"] == 1
    assert type(model.large).__name__ == "StreamedLinear"
    assert type(model.small).__name__ == "Linear"
    store.stop()
