"""Cartesian Qk symmetric-strain forms on explicitly resolved integration cuts."""

from typing import Any

import numpy as np

from pymhm.core.validation import positive_int
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.scalar.quadrilateral import (
    qk_basis,
    qk_space,
    quadrilateral_quadrature,
    rectangle_intersection_quadrature,
)
from pymhm.fem.vector.primal import StrainOperators
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.elasticity import constitutive_values
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh


def quadrilateral_strain_operators(
    mesh: CartesianMacroMesh,
    degree: int = 1,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0),
    integration_field: CartesianCellField | None = None,
    order: int = 4,
) -> StrainOperators:
    """Integrate Kelvin strain energy, scalar mass and Cartesian vector forcing.

    Continuous Qk displacement coordinates interleave Cartesian components.
    ``integration_field`` explicitly declares a pixel partition resolving all
    coefficient and forcing discontinuities. Exact rectangle/pixel intersections
    receive separate tensor Gauss rules, without averaging material values or
    requiring approximation cells to fit those pixels. Callbacks otherwise use
    the stated element quadrature; sampling does not certify global ellipticity.
    Each cell is integrated separately to bound memory even on coarse elements
    crossing many pixels. No boundary condition, rigid gauge or solve is chosen.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "order"), degree + 1)
    dofs, nodes = qk_space(mesh, degree)
    width = dofs.shape[1]
    blocks = np.empty((len(mesh.cells), 2 * width, 2 * width))
    masses = np.empty((len(mesh.cells), width, width))
    loads = np.empty((len(mesh.cells), width, 2))
    for cell, vertices in enumerate(mesh.points[mesh.cells]):
        origin = vertices[0]
        if integration_field is None:
            reference, weights = quadrilateral_quadrature(order)
        else:
            cut_points, cut_weights = rectangle_intersection_quadrature(
                origin[None], mesh.spacing, integration_field, order
            )
            reference, weights = cut_points[0], cut_weights[0]
        basis, gradient = qk_basis(degree, reference)
        gradient /= mesh.spacing
        strain = np.zeros((len(reference), 3, 2 * width))
        strain[:, 0, 0::2] = gradient[..., 0]
        strain[:, 1, 1::2] = gradient[..., 1]
        strain[:, 2, 0::2] = gradient[..., 1] / np.sqrt(2)
        strain[:, 2, 1::2] = gradient[..., 0] / np.sqrt(2)
        physical = origin + reference * mesh.spacing
        stiffness = constitutive_values(
            constitutive, physical, lame_lambda=lame_lambda, lame_mu=lame_mu
        )
        measure = weights * mesh.areas[cell]
        blocks[cell] = np.einsum("q,qai,qab,qbj->ij", measure, strain, stiffness, strain)
        masses[cell] = np.einsum("q,qi,qj->ij", measure, basis, basis)
        loads[cell] = np.einsum("q,qi,qa->ia", measure, basis, vector_values(source, physical))
    vector_dofs = (2 * dofs[..., None] + np.arange(2)).reshape(len(mesh.cells), 2 * width)
    size = 2 * len(nodes)
    matrix = assemble_element_blocks(blocks, vector_dofs, vector_dofs, (size, size))
    mass = assemble_element_blocks(masses, dofs, dofs, (len(nodes), len(nodes)))
    load = np.bincount(vector_dofs.ravel(), weights=loads.ravel(), minlength=size)
    return StrainOperators(matrix, mass, load, nodes)
