import json
import struct
from pathlib import Path

import pytest

from mext_ai.gguf import (
    GgufError,
    GgufFile,
    GgufFormatError,
    LayerPageWarmer,
)

F32 = 0
F16 = 1
Q4_0 = 2
VALUE_UINT32 = 4
VALUE_STRING = 8
VALUE_ARRAY = 9


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", VALUE_STRING) + _string(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<I", VALUE_UINT32) + struct.pack("<I", value)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    body = b"".join(_string(value) for value in values)
    return (
        _string(key)
        + struct.pack("<I", VALUE_ARRAY)
        + struct.pack("<I", VALUE_STRING)
        + struct.pack("<Q", len(values))
        + body
    )


def _ggml_size(ggml_type: int, dims: tuple[int, ...]) -> int:
    n = 1
    for dim in dims:
        n *= dim
    if ggml_type == Q4_0:
        assert n % 32 == 0
        return (n // 32) * 18
    return n * 4  # F32/F16 payload sizes are enough for this fixture


def build_gguf(path: Path) -> dict:
    tensors = [
        ("token_embd.weight", (16, 8), F32),
        ("blk.0.attn_q.weight", (32, 8), Q4_0),
        ("blk.0.ffn_down.weight", (16, 8), F32),
        ("blk.1.attn_q.weight", (32, 8), Q4_0),
        ("blk.1.ffn_gate_exps.weight", (4, 16, 8), F32),
    ]
    metadata = b"".join(
        [
            _kv_string("general.architecture", "llama"),
            _kv_u32("general.alignment", 32),
            _kv_u32("llama.block_count", 2),
            _kv_string("tokenizer.ggml.model", "gpt2"),
            _kv_string_array("tokenizer.ggml.tokens", ["a", "b", "hello"]),
        ]
    )
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", 5)
        + metadata
    )

    infos = b""
    data = b""
    offset = 0
    sizes = {}
    for name, dims, ggml_type in tensors:
        size = _ggml_size(ggml_type, dims)
        infos += (
            _string(name)
            + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", dim) for dim in dims)
            + struct.pack("<I", ggml_type)
            + struct.pack("<Q", offset)
        )
        data += bytes((index * 37 + 11) & 0xFF for index in range(size))
        sizes[name] = size
        offset += size

    header_end = len(header) + len(infos)
    padding = (-header_end) % 32
    path.write_bytes(header + infos + b"\x00" * padding + data)
    return {
        "sizes": sizes,
        "header_end": header_end,
        "padding": padding,
        "data_bytes": offset,
    }


@pytest.fixture()
def gguf_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.gguf"
    build_gguf(path)
    return path


def test_header_metadata_and_alignment(gguf_path):
    with GgufFile.open(gguf_path) as gguf:
        assert gguf.version == 3
        assert gguf.tensor_count == 5
        assert gguf.kv_count == 5
        assert gguf.alignment == 32
        assert gguf.metadata["general.architecture"] == "llama"
        assert gguf.metadata["llama.block_count"] == 2
        assert gguf.metadata["tokenizer.ggml.tokens"] == ["a", "b", "hello"]
        assert gguf.data_offset % 32 == 0
        assert gguf.data_offset > gguf.header_bytes
        assert gguf.path == gguf_path


def test_tensor_sizes_types_and_offsets(gguf_path):
    with GgufFile.open(gguf_path) as gguf:
        q4 = gguf.tensor_by_name["blk.0.attn_q.weight"]
        # (8*32) elements / 32 per block * 18 bytes per Q4_0 block
        assert q4.size == 144
        assert q4.type_name.upper().startswith("Q4_0")
        f32 = gguf.tensor_by_name["token_embd.weight"]
        assert f32.size == 16 * 8 * 4

        histogram = gguf.type_histogram()
        assert histogram[gguf.tensor_by_name["token_embd.weight"].type_name] == 3

        start, end = gguf.tensor_abs_range(f32)
        assert start == gguf.data_offset + f32.offset
        assert end - start == f32.size
        assert end <= gguf.total_bytes


def test_layers_and_experts(gguf_path):
    with GgufFile.open(gguf_path) as gguf:
        assert sorted(gguf.layers) == [0, 1]
        names0 = {t.name for t in gguf.layers[0]}
        assert "blk.0.attn_q.weight" in names0
        assert "blk.0.ffn_down.weight" in names0
        assert any("ffn_gate_exps" in t.name for t in gguf.expert_tensors)

        start, end = gguf.layer_range(0)
        assert start >= gguf.data_offset
        assert end > start
        assert end <= gguf.total_bytes


def test_summary_is_json_serialisable(gguf_path):
    with GgufFile.open(gguf_path) as gguf:
        summary = gguf.summary()
    text = json.dumps(summary)  # must not raise
    assert "tiny.gguf" in text
    assert summary["tensor_count"] == 5
    assert summary["layer_count"] == 2
    assert summary["padding"] == (-summary["header_bytes"]) % 32
    assert summary["data_end"] <= summary["total_bytes"]


def test_warmer_pages_layers(gguf_path):
    with GgufFile.open(gguf_path) as gguf:
        warmer = LayerPageWarmer(gguf, lookahead=2)
        result = warmer.warm_layers([0])
        assert result["layers"] == [0]
        assert result["bytes"] > 0
        assert result["chunks"] >= 1
        assert result["MBps"] >= 0.0

        predicted = warmer.predict_next()
        assert 1 in predicted  # sequential lookahead
        assert 0 in predicted  # recently used

        # Warming the same layer again is a no-op.
        again = warmer.warm_layers([0])
        assert again["bytes"] == 0
        stats = warmer.stats()
        assert stats["layers_warmed"] >= 1


def test_cli_inspect_and_warm(gguf_path, capsys):
    from mext_ai.cli import main

    assert main(["gguf", "inspect", str(gguf_path)]) == 0
    output = capsys.readouterr().out
    assert "GGUF" in output
    assert "tensors        5" in output

    assert main(["gguf", "inspect", str(gguf_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tensor_count"] == 5

    assert main(["gguf", "warm", str(gguf_path), "--layers", "0,1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["layers"] == [0, 1]
    assert result["bytes"] > 0


def test_cli_prewarm_all_layers(gguf_path, capsys):
    from mext_ai.cli import main

    assert main(["gguf", "prewarm", str(gguf_path), "--pace-ms", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["layers"] == [0, 1]
    assert payload["layers_total"] == 2
    assert payload["bytes"] > 0
    assert payload["pace_ms"] == 0
    assert payload["stats"]["layers_warmed"] == 2


def test_engine_rpc(gguf_path, tmp_path):
    from mext_ai.engine import AiEngine

    engine = AiEngine(
        workspace=tmp_path,
        socket=tmp_path / "ai.sock",
        state_dir=tmp_path / "state",
        ram_budget=64 * 1024 * 1024,
        model="synthetic",
        neural=False,
        rag=False,
    )
    try:
        summary = engine.handle("gguf.inspect", {"path": str(gguf_path)})
        assert summary["tensor_count"] == 5

        result = engine.handle(
            "gguf.warm", {"path": str(gguf_path), "layers": [0], "lookahead": 1}
        )
        assert result["bytes"] > 0
        assert engine.health()["gguf"] is True

        with pytest.raises(Exception):
            engine.handle("gguf.inspect", {"path": str(tmp_path / "missing.gguf")})
    finally:
        engine.stop()


def test_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(GgufFormatError):
        GgufFile.open(bad)

    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(b"GGUF" + struct.pack("<I", 3))
    with pytest.raises(GgufError):
        GgufFile.open(truncated)

    with pytest.raises(GgufError):
        GgufFile.open(tmp_path / "missing.gguf")
