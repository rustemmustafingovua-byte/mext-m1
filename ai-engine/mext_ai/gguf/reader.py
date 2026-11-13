"""Read-only GGUF header / metadata / tensor-table parser.

This module parses the GGUF container format (``v1``/``v2``/``v3``) without
loading a single weight: the file is memory-mapped and only the small header
section (magic, version, KV metadata, tensor info table) is decoded.

Layout of the header (all integers little-endian)::

    "GGUF" (4 bytes, magic)
    uint32  version                      (1, 2 or 3)
    uint64  tensor_count
    uint64  metadata_kv_count
    <metadata_kv_count> key/value pairs
    <tensor_count> tensor infos
    <padding to general.alignment (default 32)>
    <tensor data>

Every tensor info records an *absolute* byte offset into the file plus the
GGML block type, so the byte size of each tensor is derived from the block
size / element count table below — no weights are read to compute sizes.
"""

from __future__ import annotations

import mmap
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "GgufError",
    "GgufFormatError",
    "GgufTruncatedError",
    "TensorInfo",
    "GgufFile",
    "GGML_TYPES",
    "GGUF_VALUE_TYPES",
]

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
SUPPORTED_VERSIONS = (1, 2, 3)

# GGUF metadata value type ids -> (name, struct format or None for special)
GGUF_VALUE_TYPES: dict[int, str] = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

# GGML tensor type id -> (name, block_size, bytes_per_block).
# ``block_size`` is the number of elements one ``bytes_per_block`` payload
# encodes; the tensor byte size is ``(n_elems // block_size) * bytes_per_block``.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    4: ("Q4_2", 32, 18),   # legacy, removed from ggml
    5: ("Q4_3", 32, 18),   # legacy, removed from ggml
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 66),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 82),
    30: ("BF16", 1, 2),
}

_LAYER_RE = re.compile(r"(?:^|\.)(?:blk|layers|layer)\.(\d+)\.")
_EXPERT_RE = re.compile(r"expert|_exp(s)?\b|\.exp(s)?\.", re.IGNORECASE)

_MAX_METADATA_STRING = 256


class GgufError(Exception):
    """Base class for GGUF parsing errors."""


class GgufFormatError(GgufError):
    """The file is not a valid / supported GGUF container."""


class GgufTruncatedError(GgufError):
    """The file ended before the header could be fully parsed."""


@dataclass
class TensorInfo:
    """A single entry in the GGUF tensor info table."""

    name: str
    n_dims: int
    dims: tuple[int, ...]
    ggml_type: int
    offset: int
    type_name: str = "unknown"
    size: int = 0
    n_elems: int = 0

    @property
    def layer(self) -> int | None:
        """Transformer layer index parsed from the tensor name, if any."""
        match = _LAYER_RE.search(self.name)
        return int(match.group(1)) if match else None

    @property
    def is_expert(self) -> bool:
        return bool(_EXPERT_RE.search(self.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_dims": self.n_dims,
            "dims": list(self.dims),
            "type": self.type_name,
            "ggml_type": self.ggml_type,
            "offset": self.offset,
            "size": self.size,
            "n_elems": self.n_elems,
            "layer": self.layer,
            "expert": self.is_expert,
        }


def type_name(ggml_type: int) -> str:
    info = GGML_TYPES.get(ggml_type)
    return info[0] if info else f"unknown_{ggml_type}"


def tensor_byte_size(ggml_type: int, dims: tuple[int, ...]) -> int:
    """Derive the on-disk byte size of a tensor from its GGML type.

    Returns ``0`` for unknown types (the tensor is still listed). IQ types use
    the same ``(n_elems // block_size) * bytes_per_block`` rule (best effort);
    a tensor whose element count is not a multiple of the block size also
    reports ``0`` rather than a wrong number.
    """
    info = GGML_TYPES.get(ggml_type)
    if info is None:
        return 0
    n_elems = 1
    for d in dims:
        n_elems *= d
    _, block_size, bytes_per_block = info
    if block_size > 1 and n_elems % block_size != 0:
        return 0
    return (n_elems // block_size) * bytes_per_block


def _product(dims: tuple[int, ...]) -> int:
    total = 1
    for d in dims:
        total *= d
    return total


class _Reader:
    """Bounds-checked little-endian reader over an mmap / bytes buffer."""

    def __init__(self, data: "mmap.mmap | bytes") -> None:
        self.data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def take(self, n: int, what: str) -> bytes:
        if self.remaining() < n:
            raise GgufTruncatedError(
                f"truncated GGUF: need {n} bytes for {what} at offset {self.pos}, "
                f"only {self.remaining()} left"
            )
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u8(self, what: str) -> int:
        return self.take(1, what)[0]

    def u16(self, what: str) -> int:
        return struct.unpack("<H", self.take(2, what))[0]

    def u32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]

    def u64(self, what: str) -> int:
        return struct.unpack("<Q", self.take(8, what))[0]

    def i8(self, what: str) -> int:
        return struct.unpack("<b", self.take(1, what))[0]

    def i16(self, what: str) -> int:
        return struct.unpack("<h", self.take(2, what))[0]

    def i32(self, what: str) -> int:
        return struct.unpack("<i", self.take(4, what))[0]

    def i64(self, what: str) -> int:
        return struct.unpack("<q", self.take(8, what))[0]

    def f32(self, what: str) -> float:
        return struct.unpack("<f", self.take(4, what))[0]

    def f64(self, what: str) -> float:
        return struct.unpack("<d", self.take(8, what))[0]

    def string(self, what: str) -> str:
        length = self.u64(what)
        raw = self.take(length, what)
        return raw.decode("utf-8", errors="replace")


class GgufFile:
    """A parsed, memory-mapped GGUF file (headers only — no weights loaded).

    Use :meth:`GgufFile.open` to open and parse a file, or the context-manager
    form::

        with GgufFile.open("model.gguf") as gguf:
            print(gguf.summary())
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.mm: mmap.mmap | None = None

        self.version = 0
        self.tensor_count = 0
        self.kv_count = 0
        self.alignment = DEFAULT_ALIGNMENT
        self.metadata: dict[str, Any] = {}
        self.tensors: list[TensorInfo] = []
        self.tensor_by_name: dict[str, TensorInfo] = {}
        self.layers: dict[int, list[TensorInfo]] = {}
        self.expert_tensors: list[TensorInfo] = []

        self.header_bytes = 0
        self.data_offset = 0
        self.total_bytes = 0

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> "GgufFile":
        obj = cls(path)
        obj._open()
        return obj

    def _open(self) -> None:
        try:
            with open(self.path, "rb") as handle:
                self.total_bytes = os.fstat(handle.fileno()).st_size
        except OSError as exc:
            raise GgufError(f"cannot open GGUF file {self.path}: {exc}") from exc

        if self.total_bytes == 0:
            raise GgufFormatError(f"empty file is not a GGUF: {self.path}")

        with open(self.path, "rb") as handle:
            self.mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)

        try:
            self._parse_header()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self.mm is not None:
            try:
                self.mm.close()
            except Exception:  # noqa: BLE001
                pass
            self.mm = None

    def __enter__(self) -> "GgufFile":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    @property
    def open_(self) -> bool:
        return self.mm is not None

    # ------------------------------------------------------------------
    # Header parse
    # ------------------------------------------------------------------
    def _parse_header(self) -> None:
        assert self.mm is not None
        reader = _Reader(self.mm)

        magic = reader.take(4, "magic")
        if magic != GGUF_MAGIC:
            raise GgufFormatError(
                f"not a GGUF file (bad magic {magic!r}, expected {GGUF_MAGIC!r}): {self.path}"
            )

        self.version = reader.u32("version")
        if self.version not in SUPPORTED_VERSIONS:
            raise GgufFormatError(
                f"unsupported GGUF version {self.version} "
                f"(supported: {', '.join(map(str, SUPPORTED_VERSIONS))})"
            )

        self.tensor_count = reader.u64("tensor_count")
        self.kv_count = reader.u64("metadata_kv_count")

        for _ in range(self.kv_count):
            key = reader.string("metadata key")
            vtype = reader.u32("metadata value type")
            if vtype not in GGUF_VALUE_TYPES:
                raise GgufFormatError(
                    f"unknown GGUF metadata value type {vtype} for key {key!r}"
                )
            self.metadata[key] = self._read_value(reader, vtype, key)

        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if self.alignment <= 0:
            self.alignment = DEFAULT_ALIGNMENT

        for _ in range(self.tensor_count):
            name = reader.string("tensor name")
            n_dims = reader.u32("tensor n_dims")
            dims = tuple(reader.u64("tensor dim") for _ in range(n_dims))
            ggml_type = reader.u32("tensor ggml_type")
            offset = reader.u64("tensor offset")
            info = TensorInfo(
                name=name,
                n_dims=n_dims,
                dims=dims,
                ggml_type=ggml_type,
                offset=offset,
                type_name=type_name(ggml_type),
                size=tensor_byte_size(ggml_type, dims),
                n_elems=_product(dims),
            )
            self.tensors.append(info)
            self.tensor_by_name[name] = info
            if info.is_expert:
                self.expert_tensors.append(info)
            layer = info.layer
            if layer is not None:
                self.layers.setdefault(layer, []).append(info)

        self.header_bytes = reader.pos
        # GGUF stores each tensor offset relative to the *tensor data* section,
        # which starts at the first alignment boundary after the tensor table.
        self.data_offset = self._align_up(reader.pos, self.alignment)

    @staticmethod
    def _align_up(offset: int, alignment: int | None = None) -> int:
        alignment = alignment or DEFAULT_ALIGNMENT
        return (offset + alignment - 1) // alignment * alignment

    def _read_value(self, reader: _Reader, vtype: int, key: str) -> Any:
        name = GGUF_VALUE_TYPES[vtype]
        if name == "uint8":
            return reader.u8(key)
        if name == "int8":
            return reader.i8(key)
        if name == "uint16":
            return reader.u16(key)
        if name == "int16":
            return reader.i16(key)
        if name == "uint32":
            return reader.u32(key)
        if name == "int32":
            return reader.i32(key)
        if name == "uint64":
            return reader.u64(key)
        if name == "int64":
            return reader.i64(key)
        if name == "float32":
            return reader.f32(key)
        if name == "float64":
            return reader.f64(key)
        if name == "bool":
            return bool(reader.u8(key))
        if name == "string":
            return reader.string(key)
        if name == "array":
            elem_type = reader.u32(f"{key} array element type")
            count = reader.u64(f"{key} array length")
            return [self._read_value(reader, elem_type, key) for _ in range(count)]
        raise GgufFormatError(f"unhandled GGUF value type {vtype} for key {key!r}")

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------
    def tensor_abs_range(self, tensor: TensorInfo) -> tuple[int, int]:
        """Absolute file range ``[start, end)`` of a tensor."""
        start = self.data_offset + tensor.offset
        return start, start + tensor.size

    def layer_range(self, layer_id: int) -> tuple[int, int] | None:
        """Absolute ``[start, end)`` covered by every tensor in a layer."""
        tensors = self.layers.get(layer_id)
        if not tensors:
            return None
        ranges = [self.tensor_abs_range(t) for t in tensors]
        return min(r[0] for r in ranges), max(r[1] for r in ranges)

    def type_histogram(self) -> dict[str, int]:
        hist: dict[str, int] = {}
        for tensor in self.tensors:
            hist[tensor.type_name] = hist.get(tensor.type_name, 0) + 1
        return hist

    # ------------------------------------------------------------------
    def _sample_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.metadata.items():
            out[key] = self._truncate(value)
        return out

    @classmethod
    def _truncate(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value) > _MAX_METADATA_STRING:
            return value[:_MAX_METADATA_STRING] + "…"
        if isinstance(value, list):
            return [cls._truncate(item) for item in value]
        return value

    def summary(self) -> dict[str, Any]:
        layers = {
            index: sorted(t.name for t in tensors)
            for index, tensors in sorted(self.layers.items())
        }
        return {
            "path": str(self.path),
            "version": self.version,
            "tensor_count": self.tensor_count,
            "kv_count": self.kv_count,
            "total_bytes": self.total_bytes,
            "header_bytes": self.header_bytes,
            "data_offset": self.data_offset,
            "alignment": self.alignment,
            "padding": self.data_offset - self.header_bytes,
            "data_end": self.data_offset
            + max((t.offset + t.size for t in self.tensors), default=0),
            "type_histogram": self.type_histogram(),
            "layer_count": len(self.layers),
            "layers": layers,
            "expert_tensors": sorted(t.name for t in self.expert_tensors),
            "metadata": self._sample_metadata(),
        }
