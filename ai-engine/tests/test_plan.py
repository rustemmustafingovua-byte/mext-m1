from mext_ai.neural.plan import COLD, HOT, build_plan, classify

MB = 1024 * 1024


def test_hot_patterns():
    assert classify("model.embed_tokens.weight", 500 * MB) == HOT
    assert classify("model.layers.0.input_layernorm.weight", 4096) == HOT
    assert classify("model.norm.weight", 4096) == HOT
    assert classify("model.layers.3.mlp.gate.weight", 2048) == HOT  # MoE router
    assert classify("model.layers.0.self_attn.rotary_emb.inv_freq", 128) == HOT
    assert classify("model.layers.0.self_attn.q_proj.bias", 4096) == HOT


def test_cold_patterns():
    assert classify("model.layers.0.self_attn.q_proj.weight", 4 * MB) == COLD
    assert classify("model.layers.0.mlp.down_proj.weight", 16 * MB) == COLD
    assert classify("model.layers.7.mlp.gate_proj.weight", 8 * MB) == COLD


def test_tiny_tensors_are_hot_regardless_of_name():
    assert classify("model.layers.0.self_attn.q_proj.weight", 1024) == HOT


def test_build_plan_groups_layers():
    sizes = {
        "model.embed_tokens.weight": 100 * MB,
        "model.norm.weight": 4096,
        "model.layers.0.self_attn.q_proj.weight": 4 * MB,
        "model.layers.0.self_attn.k_proj.weight": 4 * MB,
        "model.layers.0.input_layernorm.weight": 4096,
        "model.layers.1.self_attn.q_proj.weight": 4 * MB,
        "lm_head.weight": 50 * MB,
    }
    plan = build_plan(sizes)
    assert plan.hot["model.embed_tokens.weight"] == 100 * MB
    assert "model.layers.0.self_attn.q_proj.weight" in plan.cold
    assert plan.layers[0] == ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight"]
    assert plan.layers[1] == ["model.layers.1.self_attn.q_proj.weight"]
    assert plan.hot_bytes > 100 * MB
    assert plan.cold_bytes == 62 * MB
    summary = plan.to_dict()
    assert summary["layers"] == 2
