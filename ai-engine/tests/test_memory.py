from mext_ai.memory import MB, AdaptiveCacheController, MemoryProfiler, MemorySnapshot


def test_profiler_reports_platform_truth():
    profiler = MemoryProfiler()
    snap = profiler.sample(force=True)
    assert snap.total > 0
    assert 0 <= snap.available <= snap.total
    assert snap.pressure in {"green", "yellow", "red", "unknown"}
    data = snap.to_dict()
    for key in ("total", "available", "used_pct", "swap_used", "pressure"):
        assert key in data


def test_controller_respects_budget():
    total = 8 * 1024 * MB
    budget = 2 * 1024 * MB
    controller = AdaptiveCacheController(budget, total)
    snap = MemorySnapshot(total=total, available=6 * 1024 * MB, used_pct=0.25, pressure="green")
    target = controller.update(snap)["target"]
    assert target <= budget
    assert target > 0


def test_controller_shrinks_under_red_pressure():
    total = 8 * 1024 * MB
    controller = AdaptiveCacheController(2 * 1024 * MB, total)
    controller.current = 512 * MB
    snap = MemorySnapshot(total=total, available=256 * MB, used_pct=0.97, pressure="red")
    result = controller.update(snap)
    assert result["action"] == "shrink"
    assert result["target"] < 512 * MB


def test_controller_caps_growth_rate():
    total = 8 * 1024 * MB
    controller = AdaptiveCacheController(4 * 1024 * MB, total)
    controller.current = 64 * MB
    snap = MemorySnapshot(total=total, available=7 * 1024 * MB, used_pct=0.1, pressure="green")
    result = controller.update(snap)
    assert result["action"] == "grow"
    # step-limited: at most 1.6x per tick
    assert result["target"] <= int(64 * MB * 1.6) + 1


def test_controller_never_goes_below_minimum():
    controller = AdaptiveCacheController(1 * 1024 * MB, 8 * 1024 * MB, min_bytes=32 * MB)
    controller.current = 32 * MB
    snap = MemorySnapshot(total=8 * 1024 * MB, available=0, used_pct=1.0, pressure="red")
    result = controller.update(snap)
    assert result["target"] >= 32 * MB
