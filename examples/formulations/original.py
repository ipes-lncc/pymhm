"""Direct coefficient reference for explicitly declared leaf variational forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.multiscale import MultiscaleSystem
from pymhm.core.original import assemble_original_blocks
from pymhm.linalg.linear import factorize


@dataclass(frozen=True)
class OriginalSolution:
    """Original local coefficients and trace from an uncondensed direct solve."""

    fields: tuple[Any, ...]
    trace: Any


def solve_original(
    system: MultiscaleSystem, *, fixed: dict[int, float] | None = None
) -> OriginalSolution:
    """Solve literal A/B/C/D rows directly with declared fixed trace coordinates.

    This reference uses the same approximation spaces, quadrature and physical
    data as the condensed system. It independently assembles and solves the
    original coefficient matrix, without using harmonic responses or Schur
    elimination. It is a coefficient comparison, not an independently
    discretized physical reference. Retained layouts and extra gauges need
    an explicitly declared original layout and are not accepted here.
    """
    if system.layout.constraints:
        raise ValueError("the direct reference needs explicitly declared original gauge rows")
    blocks = assemble_original_blocks(system)
    prescribed = dict(system.layout.fixed_trace or {}) if fixed is None else fixed
    values = np.zeros(len(blocks.load), dtype=blocks.load.dtype)
    indices = np.asarray([blocks.trace_offset + index for index in prescribed], dtype=int)
    values[indices] = list(prescribed.values())
    free = np.setdiff1d(np.arange(len(values)), indices)
    load = blocks.load[free] - blocks.matrix[free][:, indices] @ values[indices]
    with factorize(blocks.matrix[free][:, free]) as factor:
        values[free] = factor.solve(load)
    return OriginalSolution(
        tuple(
            values[start:end].copy()
            for start, end in zip(blocks.local_offsets[:-1], blocks.local_offsets[1:], strict=True)
        ),
        values[blocks.trace_offset :].copy(),
    )
