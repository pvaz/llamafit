"""Parse a GGUF header: magic, version, counts, metadata and tensor infos.

A model too large for one file is published as a *split*: several ``.gguf`` shards,
each a valid GGUF file in its own right. Only the shard numbered 0 carries the
model's metadata; the others carry three bookkeeping keys and their share of the
tensors. :func:`merge_shard_headers` turns a set of shard headers back into the one
header that describes the whole model, so everything downstream stays shard-unaware.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence

from llamafit.errors import CatalogError
from llamafit.gguf.source import ByteSource
from llamafit.gguf.types import ValueType, is_misaligned, tensor_bytes, type_name
from llamafit.i18n import _
from llamafit.models.gguf import GgufHeader, TensorInfo

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)
_DEFAULT_ALIGNMENT = 32
_SPLIT_NO = "split.no"
_SPLIT_COUNT = "split.count"
_SPLIT_TENSORS_COUNT = "split.tensors.count"
_SPLIT_KEYS = (_SPLIT_NO, _SPLIT_COUNT, _SPLIT_TENSORS_COUNT)
_DIAGNOSTIC_KEYS = ("_unknown_tensor_types", "_misaligned_tensors")
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
                _(
                    "truncated GGUF header: wanted %(wanted)d bytes at offset %(offset)d, "
                    "got %(got)d"
                )
                % {"wanted": length, "offset": self.offset, "got": len(chunk)},
                hint=_("The file may still be downloading, or the URL may not be a GGUF file."),
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
        # `_index` rather than `_`: this module imports the translator under that name,
        # and a loop variable called `_` inside a function rebinds it to an int with
        # nothing to warn you until the next translated call is not callable.
        return [_value(cursor, element_type) for _index in range(count)]
    if value_type in _SCALARS:
        return _scalar(cursor, value_type)
    raise CatalogError(
        _("unknown GGUF value type %(type)d at offset %(offset)d")
        % {"type": value_type, "offset": cursor.offset},
        hint=_("This file may use a newer GGUF metadata type than LlamaFit supports."),
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
            _("not a GGUF file: bad magic %(magic)s at offset 0") % {"magic": repr(magic)},
            hint=_("Check that this file is actually a .gguf model."),
        )
    (version,) = struct.unpack("<I", cursor.take(4))
    if version not in _SUPPORTED_VERSIONS:
        raise CatalogError(
            _("unsupported GGUF version %(version)d at offset 4") % {"version": version},
            hint=_("LlamaFit supports GGUF versions %(versions)s.")
            % {"versions": _SUPPORTED_VERSIONS},
        )
    (tensor_count,) = struct.unpack("<Q", cursor.take(8))
    (metadata_kv_count,) = struct.unpack("<Q", cursor.take(8))

    metadata: dict[str, object] = {}
    for _entry in range(metadata_kv_count):
        key = _string(cursor)
        (value_type,) = struct.unpack("<I", cursor.take(4))
        metadata[key] = _value(cursor, value_type)

    alignment = metadata.get("general.alignment", _DEFAULT_ALIGNMENT)
    if not isinstance(alignment, int) or isinstance(alignment, bool):
        alignment = _DEFAULT_ALIGNMENT

    tensors: list[TensorInfo] = []
    unknown_types: list[str] = []
    misaligned: list[str] = []
    for _tensor in range(tensor_count):
        name = _string(cursor)
        (n_dims,) = struct.unpack("<I", cursor.take(4))
        dims = [struct.unpack("<Q", cursor.take(8))[0] for _dim in range(n_dims)]
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


def _metadata_int(header: GgufHeader, key: str) -> int | None:
    """The integer value of ``key`` in ``header``'s metadata, or ``None``."""
    value = header.metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _merged_diagnostics(shards: Sequence[GgufHeader], key: str) -> list[str]:
    """Every shard's entries for one of the reader's diagnostic metadata keys."""
    collected: list[str] = []
    for shard in shards:
        value = shard.metadata.get(key)
        if isinstance(value, list):
            collected.extend(str(item) for item in value)
    return collected


def _declares_a_split(header: GgufHeader) -> bool:
    """Whether a header says anything at all about belonging to a split model."""
    return any(key in header.metadata for key in _SPLIT_KEYS)


def _declared(headers: Sequence[GgufHeader], key: str) -> int | None:
    """What the shards declare ``key`` to be, taken from wherever it appears.

    Every shard of a real split carries the whole bookkeeping trio, so reading these
    from the metadata shard alone would make the size guards vanish exactly when the
    metadata shard is the one missing, or the one written without them.

    Returns:
        The declared value, or ``None`` when no shard declares it.

    Raises:
        CatalogError: If two shards declare different values, which means they are
            not shards of one split.
    """
    values = {
        value for value in (_metadata_int(header, key) for header in headers) if value is not None
    }
    if len(values) > 1:
        raise CatalogError(
            _(
                "these GGUF files are not one split model: they disagree about "
                "'%(key)s', declaring %(values)s"
            )
            % {"key": key, "values": sorted(values)},
            hint=_("Pass the shards of one model, not of several."),
        )
    return values.pop() if values else None


def merge_shard_headers(headers: Sequence[GgufHeader]) -> GgufHeader:
    """Merge one split model's shard headers into a single header for the whole model.

    A model published as one file declares no split keys at all, and its header is
    returned unchanged, so the common case costs nothing here. Everything else is
    treated as a shard set and checked as one, *however few of it arrived*: one shard
    of four is not a single-file model, it is the emptiest possible set, and it is the
    input that reports a hundred-gigabyte model as zero bytes of weights.

    For a set, the metadata comes from the shard whose ``split.no`` is 0 — the only
    shard that carries the architecture — and the tensor table is every shard's
    tensors concatenated in shard order, which is what the byte buckets are summed
    from. Only tensor sizes are summed and never offsets, so a shard's own offsets do
    not matter and the order the shards arrive in does not either.

    An incomplete set is refused rather than merged. Facts derived from a set with a
    shard missing are quietly too small: a memory budget built on them would claim a
    model fits when it does not, which is the whole reason this function exists. The
    two size guards read ``split.count`` and ``split.tensors.count`` from whichever
    shards declare them, not from the metadata shard alone, so a metadata shard
    written without them does not disarm the checks.

    A set is only ever checked against what it declares about itself. Shards of two
    *different* models will merge without complaint, taking the metadata from one and
    the bytes from the other, whenever their declared counts happen to agree: a tail
    shard carries three bookkeeping keys and nothing that identifies its model, so
    there is nothing to cross-check against. Callers are expected to hand over a set
    they matched from one repository directory.

    Args:
        headers: One header per shard, in any order.

    Returns:
        A header whose metadata describes the model and whose tensors are the union
        of every shard's, with ``tensor_count`` and the reader's diagnostic metadata
        keys covering that union. ``header_bytes`` and ``alignment`` stay those of
        the metadata shard, since they describe one file's layout rather than the
        model's. Every ``TensorInfo.offset`` in the union stays the offset into *its
        own shard's* data section, so in a merged header the offsets repeat across
        shard boundaries and run backwards; only the sizes are meaningful, and only
        sizes are ever summed.

    Raises:
        CatalogError: If no headers are given, if a header is not a shard of a split
            model, if the shards disagree about the split they belong to, if the
            metadata shard is absent, if a shard is duplicated or missing, or if the
            union does not hold the number of tensors the model declares.
    """
    if not headers:
        raise CatalogError(
            _("no GGUF header to read facts from"),
            hint=_("Pass the model's file, or every shard of a split model."),
        )
    if len(headers) == 1 and not _declares_a_split(headers[0]):
        return headers[0]

    numbered: list[tuple[int, GgufHeader]] = []
    unnumbered = 0
    for header in headers:
        shard_no = _metadata_int(header, _SPLIT_NO)
        if shard_no is None:
            unnumbered += 1
        else:
            numbered.append((shard_no, header))
    if unnumbered:
        raise CatalogError(
            _(
                "%(count)d of %(total)d GGUF files carry no '%(key)s' key, so their "
                "place in a split model cannot be established"
            )
            % {"count": unnumbered, "total": len(headers), "key": _SPLIT_NO},
            hint=_("Pass a single unsplit file on its own, or every shard of one split model."),
        )

    numbered.sort(key=lambda pair: pair[0])
    shard_numbers = [shard_no for shard_no, _header in numbered]

    declared_shards = _declared(headers, _SPLIT_COUNT)
    if declared_shards is not None and declared_shards != len(numbered):
        raise CatalogError(
            _(
                "incomplete shard set: the model declares %(declared)d shards and "
                "%(arrived)d arrived (%(key)s %(numbers)s)"
            )
            % {
                "declared": declared_shards,
                "arrived": len(numbered),
                "key": _SPLIT_NO,
                "numbers": shard_numbers,
            },
            hint=_("Pass every shard of the split model."),
        )

    base = next((header for shard_no, header in numbered if shard_no == 0), None)
    if base is None:
        raise CatalogError(
            _(
                "the shard holding the model's metadata is missing: got %(key)s "
                "%(numbers)s, expected one of them to be 0"
            )
            % {"key": _SPLIT_NO, "numbers": shard_numbers},
            hint=_("Pass every shard of the split model, including the first."),
        )
    if shard_numbers != list(range(len(numbered))):
        raise CatalogError(
            _("incomplete shard set: expected %(key)s 0 to %(last)d, got %(numbers)s")
            % {"key": _SPLIT_NO, "last": len(numbered) - 1, "numbers": shard_numbers},
            hint=_("Pass each shard of the split model exactly once."),
        )

    tensors: list[TensorInfo] = []
    for _shard_no, header in numbered:
        tensors.extend(header.tensors)

    declared_tensors = _declared(headers, _SPLIT_TENSORS_COUNT)
    if declared_tensors is not None and declared_tensors != len(tensors):
        raise CatalogError(
            _(
                "incomplete shard set: the model declares %(declared)d tensors and its "
                "%(shards)d shards hold %(held)d between them"
            )
            % {
                "declared": declared_tensors,
                "shards": len(numbered),
                "held": len(tensors),
            },
            hint=_("Pass every shard of the split model."),
        )

    metadata = dict(base.metadata)
    shards = [header for _shard_no, header in numbered]
    for key in _DIAGNOSTIC_KEYS:
        collected = _merged_diagnostics(shards, key)
        if collected:
            metadata[key] = collected
        else:
            metadata.pop(key, None)

    return base.model_copy(
        update={"metadata": metadata, "tensors": tensors, "tensor_count": len(tensors)}
    )
