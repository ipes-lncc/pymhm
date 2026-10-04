"""Shared assembly operations for physical normal-flux Darcy formulations.

Element families own tabulation, quadrature, orientation and constant-mode
coordinates. These functions assemble supplied bases without changing them.
"""

from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.materials.evaluation import scalar_values, tensor_values
from pymhm.meshes.triangle import TriangleMesh


def triangle_pressure_basis(degree: int, bary: FloatArray) -> FloatArray:
    """Tabulate complete cardinal DG pressure, with constant P0 equal to one.

    ``degree`` is an already validated nonnegative mathematical degree.
    Barycentric points have shape ``(..., 3)``; the trailing output axis is
    the nodal pressure coefficient order used by the simplex FEM tabulator.
    """
    if degree == 0:
        return np.ones((*bary.shape[:-1], 1))
    values = reference_basis(degree, bary.reshape(-1, 3))[0]
    return values.reshape(*bary.shape[:-1], values.shape[-1])


def triangle_mixed_operators(
    mesh: TriangleMesh,
    bary: FloatArray,
    weights: FloatArray,
    material: Any,
    source: Any,
    flux: FloatArray,
    divergence: FloatArray,
    pressure: FloatArray,
    dofs: IntArray,
    flux_size: int,
) -> tuple[Any, Any, FloatArray]:
    """Assemble mass, positive divergence moments and pressure-tested source.

    Supplied flux bases already use global-oriented local DOF coordinates.
    Arrays begin with ``(fine_cell, quadrature_point)``; flux has trailing
    ``(local_flux_dof, 2)`` and pressure/divergence have a basis axis.
    ``weights`` are normalized triangle weights, so physical integrals also
    multiply by ``mesh.areas``. Material values are scalar or SPD tensors.
    DG pressure coefficients are concatenated by fine cell. The returned
    divergence has positive sign; the Darcy saddle applies its minus sign.
    """
    points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    inverse = np.linalg.inv(tensor_values(material, points.reshape(-1, 2))).reshape(
        *weights.shape, 2, 2
    )
    blocks = np.einsum("tq,tqia,tqab,tqjb,t->tij", weights, flux, inverse, flux, mesh.areas)
    pressure_count = pressure.shape[-1]
    pressure_size = pressure_count * len(mesh.cells)
    mass = assemble_element_blocks(blocks, dofs, dofs, (flux_size, flux_size))
    div_blocks = np.einsum("tq,tqi,tqj,t->tij", weights, pressure, divergence, mesh.areas)
    pressure_dofs = np.arange(pressure_size).reshape(-1, pressure_count)
    div = assemble_element_blocks(div_blocks, pressure_dofs, dofs, (pressure_size, flux_size))
    source_values = scalar_values(source, points.reshape(-1, 2)).reshape(weights.shape)
    load = np.einsum("tq,tqi,tq,t->ti", weights, pressure, source_values, mesh.areas).ravel()
    return mass, div, load


def triangle_pressure_integrals(
    mesh: TriangleMesh, bary_weights: FloatArray, pressure: FloatArray
) -> FloatArray:
    """Integrate each DG pressure basis function in concatenated fine-cell order."""
    return np.einsum("tq,tqi,t->ti", bary_weights, pressure, mesh.areas).ravel()


def normal_flux_blocks(
    mass: Any,
    divergence: Any,
    source: FloatArray,
    boundary_dofs: IntArray,
    trace_mapping: FloatArray,
) -> tuple[Any, FloatArray, FloatArray]:
    """Build the mixed Neumann saddle, trace coupling and signed source load.

    Local coefficients are ``(flux, pressure, boundary_pressure)``. The third
    equation prescribes exterior flux moments by ``trace_mapping @ trace``;
    the second equation is ``divergence @ flux = source``. The first equation
    is the weak law ``K^-1 flux + grad(pressure) = 0`` with auxiliary boundary
    pressure. Exterior DOFs and mapping rows must have the same order.
    Constant-mode kernels, physical pressure means and any scaling remain
    caller-owned because nodal and modal pressure coordinates differ.
    """
    flux_size, pressure_size = mass.shape[0], divergence.shape[0]
    boundary_size = len(boundary_dofs)
    selector = sparse.coo_matrix(
        (np.ones(boundary_size), (boundary_dofs, np.arange(boundary_size))),
        shape=(flux_size, boundary_size),
    ).tocsc()
    zero = sparse.csc_matrix((pressure_size, boundary_size))
    matrix = sparse.bmat(
        [
            [mass, -divergence.T, selector],
            [-divergence, None, zero],
            [selector.T, zero.T, None],
        ],
        format="csc",
    )
    coupling = np.zeros((flux_size + pressure_size + boundary_size, trace_mapping.shape[1]))
    coupling[flux_size + pressure_size :] = -trace_mapping
    load = np.r_[np.zeros(flux_size), -source, np.zeros(boundary_size)]
    return matrix, coupling, load
