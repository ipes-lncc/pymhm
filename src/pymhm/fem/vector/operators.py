"""Scalar basis tabulation, vector trace coupling and generic element scatter."""

from typing import Any

from pymhm.core.validation import FloatArray
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh


def _lagrange(
    mesh: TriangleMesh, degree: int, bary: FloatArray
) -> tuple[Any, FloatArray, FloatArray, FloatArray]:
    """Tabulate P1 or P2 scalar nodal bases and physical gradients."""
    dofs, points, basis, gradients, _ = tabulate(mesh, degree, bary)
    return dofs, points, basis, gradients


def _p2_coupling(
    coarse: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
) -> FloatArray:
    """Integrate P2 boundary traces using the shared arbitrary-degree assembler."""
    from pymhm.fem.scalar.triangle import trace_coupling

    return trace_coupling(coarse, cell, fine, skeleton, 2)


def _assemble_blocks(blocks: FloatArray, dofs: Any, size: int) -> Any:
    """Scatter square element matrices to a sparse CSC operator."""
    return assemble_element_blocks(blocks, dofs, dofs, (size, size))
