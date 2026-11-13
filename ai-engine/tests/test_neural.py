import pytest

from mext_ai.memory import HAS_MLX

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed")

MB = 1024 * 1024


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    from mext_ai.neural.runner import MextModelRunner

    state = tmp_path_factory.mktemp("neural")
    r = MextModelRunner("synthetic", state, 512 * MB, lookahead=2)
    r.load()
    yield r
    r.stop()


def test_model_loads(runner):
    metrics = runner.metrics()
    assert metrics["available"], metrics["error"]
    assert metrics["mode"] == "synthetic"
    assert metrics["layers"] == 6
    assert metrics["hot_bytes"] > 0
    assert metrics["cold_bytes"] > 0
    assert "lm_head.weight" in metrics["pinned"]


def test_plan_is_split(runner):
    plan = runner.plan_dict()
    assert plan["hot_tensors"] >= 1
    assert plan["cold_tensors"] > plan["hot_tensors"]
    assert plan["layers"] == 6


def test_generate_tokens(runner):
    out = runner.generate("def fib(n):", max_tokens=8)
    assert out["generated"] == 8
    assert len(out["tokens"]) == 8
    assert out["tokens_per_s"] > 0
    assert out["text"]


def test_prefetch_hits_grow(runner):
    before = runner.metrics()["prefetch_hits"]
    runner.generate("impl Cache for Arc", max_tokens=12)
    after = runner.metrics()["prefetch_hits"]
    assert after > before


def test_prefetch_layers_api(runner):
    result = runner.prefetch_layers([0, 1, 2])
    assert result["layers"] == [0, 1, 2]
    assert result["queued"] >= 0


def test_cache_tuning(runner):
    decision = runner.tune_cache()
    assert decision["target"] > 0
    assert decision["action"] in {"grow", "shrink", "hold"}
    assert runner.metrics()["cache_target_bytes"] == decision["target"]
