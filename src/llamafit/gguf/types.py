"""GGUF metadata value types and GGML tensor types.

The tensor table gives, per type id, its name, how many elements share one
quantisation block, and how many bytes that block occupies. The values follow
ggml's own type traits table; Task 3 checks them against real files, because a
wrong row would silently mis-size every model that uses that type.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum


class ValueType(IntEnum):
    """The metadata value types a GGUF header can carry."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
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
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
}


def type_name(type_id: int) -> str:
    """The GGML name of a tensor type, or ``unknown(<id>)``."""
    entry = GGML_TYPES.get(type_id)
    return entry[0] if entry else f"unknown({type_id})"


def tensor_bytes(dims: Sequence[int], type_id: int) -> int:
    """Bytes a tensor of these dimensions occupies in this type.

    Raises:
        KeyError: If the type id is not in the table.
    """
    _, block_elements, block_bytes = GGML_TYPES[type_id]
    elements = 1
    for dim in dims:
        elements *= dim
    return elements // block_elements * block_bytes
