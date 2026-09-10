# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""A small least-squares solver that refuses rather than guesses.

Four parameters at most, a handful of rows, and one property that matters more than speed:
**it says when the data does not determine the answer.** A solver that always returns
something turns two measurements of three unknowns into three confident numbers, and
nothing downstream can tell those apart from three that were actually measured.

So a rank-deficient system -- two columns that vary together, or fewer rows than columns --
comes back with no values and the rank that was found, and the caller has to say so to the
user rather than print a number.

No NumPy. It is an optional dependency here, and a calibration that ran on some
installations and not others would be a worse thing to have than a hundred lines of
arithmetic. The columns are scaled to unit length before the normal equations are formed,
which is what keeps a design matrix whose columns run from 10^-3 to 1 from losing its
conditioning to the squaring that step performs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_PIVOT_TOLERANCE = 1e-10
"""How small a pivot may be, relative to the largest, before the column counts as absent.

The normal equations are formed from unit-length columns, so the matrix has ones on its
diagonal and every entry is a cosine between two columns. A pivot at 10^-10 of the largest
means two columns agree to five significant figures, which is not two measurements of two
things.
"""


@dataclass(frozen=True)
class Solution:
    """What came out of a least-squares problem, including the news that nothing did.

    Attributes:
        values: One coefficient per column, in the order the columns were given, or
            ``None`` when the columns did not determine them.
        rank: How many independent directions the columns actually spanned.
        columns: How many were asked for. ``rank < columns`` is the whole story of a
            refusal, and both halves are needed to tell it.
        rows: How many measurements went in.
        residual: Root-mean-square of the residuals, in the units of the target.
        exactly_determined: True when there were exactly as many rows as columns, so the
            fit passes through every point and the residual is zero by construction rather
            than by agreement. A number with no residual has nothing corroborating it, and
            a reader is owed that distinction.
    """

    values: tuple[float, ...] | None
    rank: int
    columns: int
    rows: int
    residual: float = 0.0
    exactly_determined: bool = False

    @property
    def ok(self) -> bool:
        """Whether the problem was determined and there is an answer to read."""
        return self.values is not None


def least_squares(matrix: Sequence[Sequence[float]], targets: Sequence[float]) -> Solution:
    """Solve ``matrix @ x = targets`` in the least-squares sense, or report why not.

    Args:
        matrix: One row per measurement, one column per free parameter.
        targets: One observation per row.

    Returns:
        The solution, or a :class:`Solution` whose ``values`` are ``None`` and whose
        ``rank`` says how many of the parameters the data does determine.

    Raises:
        ValueError: If the shapes disagree, or there are no columns at all. Those are
            programming errors rather than data problems, and hiding them behind a rank
            report would make them look like the data's fault.
    """
    rows = [list(row) for row in matrix]
    values = list(targets)
    if len(rows) != len(values):
        raise ValueError("one target per row is required")
    if not rows or not rows[0]:
        raise ValueError("a least-squares problem needs at least one column")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("every row needs the same number of columns")
    if len(rows) < width:
        return Solution(values=None, rank=min(len(rows), width), columns=width, rows=len(rows))

    scales = [math.sqrt(sum(row[column] ** 2 for row in rows)) for column in range(width)]
    if any(scale == 0.0 for scale in scales):
        # A column of zeros carries no information at all about its parameter.
        alive = sum(1 for scale in scales if scale > 0)
        return Solution(values=None, rank=alive, columns=width, rows=len(rows))
    scaled = [[row[column] / scales[column] for column in range(width)] for row in rows]

    normal = [[sum(row[i] * row[j] for row in scaled) for j in range(width)] for i in range(width)]
    right = [
        sum(row[i] * value for row, value in zip(scaled, values, strict=True)) for i in range(width)
    ]

    solved, rank = _gauss(normal, right)
    if solved is None:
        return Solution(values=None, rank=rank, columns=width, rows=len(rows))

    answer = tuple(solved[i] / scales[i] for i in range(width))
    residuals = [
        sum(coefficient * row[i] for i, coefficient in enumerate(answer)) - value
        for row, value in zip(rows, values, strict=True)
    ]
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return Solution(
        values=answer,
        rank=rank,
        columns=width,
        rows=len(rows),
        residual=rms,
        exactly_determined=len(rows) == width,
    )


def _gauss(matrix: list[list[float]], right: list[float]) -> tuple[list[float] | None, int]:
    """Gaussian elimination with partial pivoting; ``None`` when a pivot vanishes.

    Returns the solution and the rank. The rank is the number of pivots that stayed above
    :data:`_PIVOT_TOLERANCE` relative to the largest entry, which for a matrix built from
    unit-length columns is a direct statement about how many independent directions those
    columns span.
    """
    size = len(matrix)
    work = [[*row, right[index]] for index, row in enumerate(matrix)]
    largest = max(abs(work[i][j]) for i in range(size) for j in range(size)) or 1.0
    rank = 0
    for column in range(size):
        pivot_row = column
        for candidate in range(column + 1, size):
            if abs(work[candidate][column]) > abs(work[pivot_row][column]):
                pivot_row = candidate
        if abs(work[pivot_row][column]) <= _PIVOT_TOLERANCE * largest:
            return None, rank
        work[column], work[pivot_row] = work[pivot_row], work[column]
        rank += 1
        pivot = work[column][column]
        for row in range(column + 1, size):
            factor = work[row][column] / pivot
            if factor == 0.0:
                continue
            for col in range(column, size + 1):
                work[row][col] -= factor * work[column][col]
    answer = [0.0] * size
    for row in range(size - 1, -1, -1):
        total = work[row][size] - sum(work[row][c] * answer[c] for c in range(row + 1, size))
        answer[row] = total / work[row][row]
    return answer, rank
