"""Observe original equations separately from global compatibility diagnostics."""

from __future__ import annotations

from typing import Any

from examples.minimal_scalar_convergence import hybrid_original
from pymhm.core.multiscale import MultiscaleSolution, MultiscaleSystem


def original_recovery_rows(
    system: MultiscaleSystem, coefficients: MultiscaleSolution
) -> dict[str, Any]:
    """Measure each original local volume row and the uncondensed trace equations.

    This observer requires no additional direct global operator. The same
    assembled matrix, local fields, essential data and boundary load are handed
    to the shared original-row integration owner. Euclidean coefficient-row
    residuals remain distinct from physical L2 errors and global compatibility.
    The unchanged acceptance gate is 1e-10 relative to the stated physical load
    or individual equation action scale.
    """
    if system.global_matrix.nnz:
        raise ValueError("the observer requires zero additional direct global operator")
    return hybrid_original(
        system,
        coefficients,
        dict(system.layout.fixed_trace or {}),
        system.global_load[: system.trace_size],
    )
