import time

import numpy as np
import pytest

from mext_ai.neural.store import ColdStore

MB = 1024 * 1024


def make_store(tmp_path, budget=8 * MB):
    return ColdStore(tmp_path / "weights", budget)


def test_put_get_roundtrip(tmp_path):
    store = make_store(tmp_path)
    data = np.arange(1024, dtype=np.float32).reshape(32, 32)
    ref = store.put("layer.0.weight", data)
    assert ref.nbytes == data.nbytes
    assert store.size_bytes() == data.nbytes
    loaded = store.get("layer.0.weight")
    assert np.allclose(np.array(loaded), data)
    assert store.stats()["misses"] == 1
    store.get("layer.0.weight")
    assert store.stats()["hits"] == 1
    store.stop()


def test_budget_evicts_lru(tmp_path):
    store = make_store(tmp_path)
    store.put("a", np.zeros((512, 512), dtype=np.float32))  # 1 MiB
    store.put("b", np.zeros((512, 512), dtype=np.float32))
    store.set_budget(int(1.5 * MB))
    store.get("a")
    loaded = store.get("b")
    assert loaded is not None
    stats = store.stats()
    assert stats["resident_bytes"] <= 1.5 * MB
    assert stats["evictions"] >= 1
    store.stop()


def test_pinned_survives_eviction(tmp_path):
    store = make_store(tmp_path)
    store.put("lm_head.weight", np.zeros((512, 512), dtype=np.float32))
    store.put("other", np.zeros((512, 512), dtype=np.float32))
    assert store.pin("lm_head.weight")
    store.set_budget(int(1.2 * MB))
    store.get("other")
    assert store.is_resident("lm_head.weight")
    store.stop()


def test_prefetch_loads_in_background(tmp_path):
    store = make_store(tmp_path)
    names = []
    for i in range(6):
        name = f"layer.{i}.weight"
        store.put(name, np.full((64, 64), i, dtype=np.float32))
        names.append(name)
    queued = store.prefetch(names)
    assert queued == 6
    deadline = time.time() + 5
    while time.time() < deadline and not all(store.is_resident(n) for n in names):
        time.sleep(0.02)
    assert all(store.is_resident(n) for n in names)
    stats = store.stats()
    assert stats["streamed_bytes"] > 0
    store.stop()


def test_index_and_stats(tmp_path):
    store = make_store(tmp_path)
    store.put("x", np.zeros((8, 8), dtype=np.float32))
    store.save_index()
    other = ColdStore(tmp_path / "weights", 1 * MB)
    assert other.load_index()
    assert "x" in other.refs
    stats = other.stats()
    assert stats["cold_bytes"] == 8 * 8 * 4
    other.stop()
    store.stop()


def test_unknown_tensor_raises(tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(KeyError):
        store.get("nope")
    store.stop()
