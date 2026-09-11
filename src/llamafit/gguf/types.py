# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""GGUF metadata value types and GGML tensor types.

The tensor table gives, per type id, its name, how many elements share one
quantisation block, and how many bytes that block occupies. The values follow
ggml's own type traits table in ``ggml/src/ggml.c``, whose block sizes are the
``sizeof`` of the structs in ``ggml/src/ggml-common.h``. The rows added after the
first version quote those structs beside them, because a wrong row silently
mis-sizes every model that uses that type, and a quoted struct can be checked
against the source without trusting anybody's memory. Real files check the table
too: :func:`llamafit.gguf.facts.accounts_for_file` compares what the rows sum to
against what a file actually holds.

A type the table does not know is refused by the reader rather than sized: see
:func:`llamafit.gguf.reader.read_header` for why a zero is the one wrong answer.
"""

from __future__ import annotations

import hashlib
import json
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
#
# The ids are ggml's ``enum ggml_type``; the gaps are types llama.cpp has removed
# (4, 5, 31 to 33, 36 to 38), which a current build refuses to load and this table
# therefore refuses to size. The four rows after TQ2_0 are quoted from
# ``ggml/src/ggml-common.h`` at ggml-org/llama.cpp commit 85c5522 (2026-08-31),
# where ``ggml_half`` is a 16-bit float and every struct carries a ``static_assert``
# that its size is the sum of its fields.
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
    #     #define QK_MXFP4 32
    #     typedef struct {
    #         uint8_t e; // E8M0
    #         uint8_t qs[QK_MXFP4/2];
    #     } block_mxfp4;
    # One shared power-of-two scale and sixteen bytes of packed FP4 (E2M1) nibbles:
    # 1 + 32 / 2 = 17 bytes for 32 weights, 4.25 bits each. The format gpt-oss,
    # DeepSeek V4 and Kimi K3 publish their expert weights in, and the one this table
    # lacked while those tensors were being summed as zero bytes.
    39: ("MXFP4", 32, 17),
    #     #define QK_NVFP4 64
    #     #define QK_NVFP4_SUB 16  // sub-block size for per-group scales
    #     typedef struct {
    #         uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per
    #                                           // 16-element sub-block)
    #         uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
    #     } block_nvfp4;
    # 64 / 16 + 64 / 2 = 36 bytes for 64 weights.
    40: ("NVFP4", 64, 36),
    #     #define QK1_0 128
    #     typedef struct {
    #         ggml_half d;           // delta
    #         uint8_t qs[QK1_0 / 8]; // bits / quants
    #     } block_q1_0;
    # 2 + 128 / 8 = 18 bytes for 128 weights.
    41: ("Q1_0", 128, 18),
    #     #define QK2_0 64
    #     typedef struct {
    #         ggml_half d;              // delta (scale)
    #         uint8_t qs[QK2_0 / 4];   // 2 bits per element
    #     } block_q2_0;
    # 2 + 64 / 4 = 18 bytes for 64 weights.
    42: ("Q2_0", 64, 18),
}


def table_digest() -> str:
    """A fingerprint of :data:`GGML_TYPES`, for anything that stores a size computed from it.

    A cached header carries each tensor's size in bytes, computed by whichever version of
    the table was current when the header was parsed. Without this in the cache key, a
    row added or corrected here would go on being contradicted by every header parsed
    before it, the wrongly parsed ones included, for as long as the file's ETag stayed the
    same; for a published model that is forever. It is computed on each call rather than
    once at import so that it follows the table it describes, whoever changed it.
    """
    rows = json.dumps(sorted((type_id, *entry) for type_id, entry in GGML_TYPES.items()))
    return hashlib.sha256(rows.encode("ascii")).hexdigest()


def type_name(type_id: int) -> str:
    """The GGML name of a tensor type, or ``unknown(<id>)``."""
    entry = GGML_TYPES.get(type_id)
    return entry[0] if entry else f"unknown({type_id})"


def tensor_bytes(dims: Sequence[int], type_id: int) -> int:
    """Bytes a tensor of these dimensions occupies in this type.

    Rounds up to a whole number of quantisation blocks: a tensor whose element
    count is not an exact multiple of its type's block size still occupies a
    full trailing block on disk, and an undercount is the dangerous direction
    for a memory budget. Callers that need to know whether a tensor was
    misaligned should call :func:`is_misaligned`.

    Raises:
        KeyError: If the type id is not in the table.
    """
    # Named rather than `_`: nothing here calls the translator today, and the
    # next person to wrap a message in this file must not have to notice first.
    _name, block_elements, block_bytes = GGML_TYPES[type_id]
    elements = 1
    for dim in dims:
        elements *= dim
    blocks = (elements + block_elements - 1) // block_elements
    return blocks * block_bytes


def is_misaligned(dims: Sequence[int], type_id: int) -> bool:
    """True when a tensor's element count is not an exact multiple of its block size.

    Raises:
        KeyError: If the type id is not in the table.
    """
    _name, block_elements, _block_bytes = GGML_TYPES[type_id]
    elements = 1
    for dim in dims:
        elements *= dim
    return elements % block_elements != 0
