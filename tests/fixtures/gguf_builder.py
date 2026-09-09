"""Build a minimal but valid GGUF header in memory for tests."""

from __future__ import annotations

import struct

MAGIC = b"GGUF"


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value_type: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + payload


def uint16(key: str, value: int) -> bytes:
    """A UINT16 metadata entry, the type gguf-split writes split.no and split.count in."""
    return _kv(key, 2, struct.pack("<H", value))


def uint32(key: str, value: int) -> bytes:
    """A UINT32 metadata entry."""
    return _kv(key, 4, struct.pack("<I", value))


def int32(key: str, value: int) -> bytes:
    """An INT32 metadata entry, the type gguf-split writes split.tensors.count in."""
    return _kv(key, 5, struct.pack("<i", value))


def string(key: str, value: str) -> bytes:
    """A STRING metadata entry."""
    return _kv(key, 8, _string(value))


def string_array(key: str, values: list[str]) -> bytes:
    """An ARRAY of STRING metadata entry, used for the tokenizer vocabulary."""
    body = struct.pack("<I", 8) + struct.pack("<Q", len(values))
    for value in values:
        body += _string(value)
    return _kv(key, 9, body)


def tensor(name: str, dims: list[int], type_id: int, offset: int) -> bytes:
    """One tensor info record."""
    out = _string(name) + struct.pack("<I", len(dims))
    for dim in dims:
        out += struct.pack("<Q", dim)
    return out + struct.pack("<I", type_id) + struct.pack("<Q", offset)


def build(metadata: list[bytes], tensors: list[bytes], version: int = 3) -> bytes:
    """Assemble a header from encoded metadata and tensor records."""
    head = MAGIC + struct.pack("<I", version)
    head += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))
    return head + b"".join(metadata) + b"".join(tensors)
