"""Parse a GGUF header: magic, version, counts, metadata and tensor infos."""

from __future__ import annotations

import struct

from llamafit.errors import CatalogError
from llamafit.gguf.source import ByteSource
from llamafit.gguf.types import ValueType, is_misaligned, tensor_bytes, type_name
from llamafit.models.gguf import GgufHeader, TensorInfo

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)
_DEFAULT_ALIGNMENT = 32
_SCALARS: dict[int, tuple[str, int]] = {
    ValueType.UINT8: ("<B", 1),
    ValueType.INT8: ("<b", 1),
    ValueType.UINT16: ("<H", 2),
    ValueType.INT16: ("<h", 2),
    ValueType.UINT32: ("<I", 4),
    ValueType.INT32: ("<i", 4),
    ValueType.FLOAT32: ("<f", 4),
    ValueType.BOOL: ("<?", 1),
    ValueType.UINT64: ("<Q", 8),
    ValueType.INT64: ("<q", 8),
    ValueType.FLOAT64: ("<d", 8),
}


class _Cursor:
    """Sequential reader over a ByteSource with a buffered window."""

    def __init__(self, source: ByteSource, window: int = (1 << 20) - 1):
        """Read from ``source`` in windows of ``window`` bytes at a time."""
        self._source = source
        self._window = window
        self._buffer = b""
        self._buffer_start = 0
        self.offset = 0

    def take(self, length: int) -> bytes:
        """Return the next ``length`` bytes, refilling the window as needed.

        Raises:
            CatalogError: If fewer than ``length`` bytes remain.
        """
        end = self.offset + length
        if not (
            self._buffer_start <= self.offset and end <= self._buffer_start + len(self._buffer)
        ):
            fetch = max(length, self._window)
            self._buffer = self._source.read(self.offset, fetch)
            self._buffer_start = self.offset
        start = self.offset - self._buffer_start
        chunk = self._buffer[start : start + length]
        if len(chunk) < length:
            raise CatalogError(
                f"truncated GGUF header: wanted {length} bytes at offset {self.offset}, "
                f"got {len(chunk)}",
                hint="The file may still be downloading, or the URL may not be a GGUF file.",
            )
        self.offset = end
        return chunk


def _scalar(cursor: _Cursor, value_type: int) -> object:
    """Read one fixed-width scalar value of ``value_type``."""
    fmt, size = _SCALARS[value_type]
    return struct.unpack(fmt, cursor.take(size))[0]


def _string(cursor: _Cursor) -> str:
    """Read a length-prefixed UTF-8 string."""
    (length,) = struct.unpack("<Q", cursor.take(8))
    return cursor.take(length).decode("utf-8", errors="replace")


def _value(cursor: _Cursor, value_type: int) -> object:
    """Read one metadata value of ``value_type``, recursing for arrays.

    Raises:
        CatalogError: If ``value_type`` is not a known GGUF value type.
    """
    if value_type == ValueType.STRING:
        return _string(cursor)
    if value_type == ValueType.ARRAY:
        (element_type,) = struct.unpack("<I", cursor.take(4))
        (count,) = struct.unpack("<Q", cursor.take(8))
        return [_value(cursor, element_type) for _ in range(count)]
    if value_type in _SCALARS:
        return _scalar(cursor, value_type)
    raise CatalogError(
        f"unknown GGUF value type {value_type} at offset {cursor.offset}",
        hint="This file may use a newer GGUF metadata type than LlamaFit supports.",
    )


def read_header(source: ByteSource) -> GgufHeader:
    """Parse a GGUF header from ``source``, fetching only the bytes it needs.

    Raises:
        CatalogError: If the magic is wrong, the version is unsupported, the
            header is truncated, or it contains an unknown value type. One
            tensor with an unrecognised type does not fail the whole read;
            its size is recorded as 0 and its name and type are listed under
            the ``_unknown_tensor_types`` metadata key. A tensor whose element
            count is not an exact multiple of its type's block size has its
            size rounded up rather than truncated, and is listed under the
            ``_misaligned_tensors`` metadata key.
    """
    cursor = _Cursor(source)
    magic = cursor.take(4)
    if magic != _MAGIC:
        raise CatalogError(
            f"not a GGUF file: bad magic {magic!r} at offset 0",
            hint="Check that this file is actually a .gguf model.",
        )
    (version,) = struct.unpack("<I", cursor.take(4))
    if version not in _SUPPORTED_VERSIONS:
        raise CatalogError(
            f"unsupported GGUF version {version} at offset 4",
            hint=f"LlamaFit supports GGUF versions {_SUPPORTED_VERSIONS}.",
        )
    (tensor_count,) = struct.unpack("<Q", cursor.take(8))
    (metadata_kv_count,) = struct.unpack("<Q", cursor.take(8))

    metadata: dict[str, object] = {}
    for _ in range(metadata_kv_count):
        key = _string(cursor)
        (value_type,) = struct.unpack("<I", cursor.take(4))
        metadata[key] = _value(cursor, value_type)

    alignment = metadata.get("general.alignment", _DEFAULT_ALIGNMENT)
    if not isinstance(alignment, int) or isinstance(alignment, bool):
        alignment = _DEFAULT_ALIGNMENT

    tensors: list[TensorInfo] = []
    unknown_types: list[str] = []
    misaligned: list[str] = []
    for _ in range(tensor_count):
        name = _string(cursor)
        (n_dims,) = struct.unpack("<I", cursor.take(4))
        dims = [struct.unpack("<Q", cursor.take(8))[0] for _ in range(n_dims)]
        (type_id,) = struct.unpack("<I", cursor.take(4))
        (offset,) = struct.unpack("<Q", cursor.take(8))
        try:
            size = tensor_bytes(dims, type_id)
        except KeyError:
            size = 0
            unknown_types.append(f"{name}:{type_name(type_id)}")
        else:
            if is_misaligned(dims, type_id):
                misaligned.append(f"{name}:{type_name(type_id)}")
        tensors.append(TensorInfo(name=name, dims=dims, type=type_id, offset=offset, bytes_=size))

    if unknown_types:
        metadata["_unknown_tensor_types"] = unknown_types
    if misaligned:
        metadata["_misaligned_tensors"] = misaligned

    return GgufHeader(
        version=version,
        tensor_count=tensor_count,
        alignment=alignment,
        metadata=metadata,
        tensors=tensors,
        header_bytes=cursor.offset,
    )
