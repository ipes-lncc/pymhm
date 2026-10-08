"""RT inverse-material mass, DG divergence and scalar load integration."""

from typing import Any

from pymhm.core.validation import FloatArray
from pymhm.fem.hdiv.mixed import (
    triangle_mixed_operators,
    triangle_pressure_basis,
    triangle_pressure_integrals,
)
from pymhm.fem.hdiv.rt import rt_basis, rt_degree, rt_dofs
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.meshes.triangle import TriangleMesh


def pressure_basis(degree: int, bary: FloatArray) -> FloatArray:
    """Tabulate discontinuous nodal Pm pressure, using a constant for RT0."""
    return triangle_pressure_basis(rt_degree(degree), bary)


def rt_operators(
    mesh: TriangleMesh, degree: int, permeability: Any = 1.0, source: Any = 0.0, order: int = 5
) -> tuple[Any, Any, FloatArray, FloatArray]:
    """Assemble inverse-diffusion mass, DG-Pm divergence/load and pressure integral weights."""
    m = rt_degree(degree)
    bary, weights, material = material_triangle_quadrature(mesh, permeability, max(order, m + 3))
    values, divergence = rt_basis(mesh, m, bary)
    pressure = pressure_basis(m, bary)
    dofs = rt_dofs(mesh, m)
    nq = (m + 1) * len(mesh.faces) + m * (m + 1) * len(mesh.cells)
    mass, D, load = triangle_mixed_operators(
        mesh, bary, weights, material, source, values, divergence, pressure, dofs, nq
    )
    moment = triangle_pressure_integrals(mesh, weights, pressure)
    return mass, D, load, moment
