"""Symmetric-strain volume integration independently of trace and kernel choices."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import linalg

from pymhm.core.validation import FloatArray
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.triangle import (
    element_tabulate,
    multiindices,
    nodal_space,
    reference_basis,
)
from pymhm.fem.vector.elasticity import strain_and_divergence as _strain_and_divergence
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.elasticity import constitutive_values as constitutive_values
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class StrainOperators:
    """Literal vector energy/load and scalar mass in physical nodal coordinates."""

    matrix: Any
    mass: Any
    load: FloatArray
    nodes: FloatArray


def trace_detecting_enrichment(fine: TriangleMesh, degree: int, coupling: FloatArray) -> FloatArray:
    """Enrich Pk by one P(k+1) mode that detects its missing odd-degree trace.

    This realizes the polynomial enrichment hypothesis of Lemma 6.8 in
    [Harder, Madureira and Valentin (2016)](https://doi.org/10.1051/m2an/2015046) on one triangular
    element. The added mode is selected from the null trace's nonzero moment functional and is not a
    zero-boundary bubble. Coordinate projection removes its Pk part without changing its
    missing-trace moment.
    """
    if len(fine.cells) != 1:
        raise ValueError("minimal enrichment requires exactly one local triangle")
    low_dofs, low_nodes = nodal_space(fine, degree)
    high_dofs, high_nodes = nodal_space(fine, degree + 1)
    bary = multiindices(degree + 1) / (degree + 1)
    embedding = np.zeros((len(high_nodes), len(low_nodes)))
    embedding[np.ix_(high_dofs[0], low_dofs[0])] = reference_basis(degree, bary)[0]
    # Resolve the analytically null trace at the scale of its assembled moments;
    # physical edge integration also contributes floating-point roundoff.
    moments = embedding.T @ coupling
    null = linalg.null_space(moments, rcond=64 * np.finfo(float).eps * max(moments.shape))
    if null.shape[1] != 1:
        raise ValueError("minimal enrichment requires exactly one invisible scalar trace mode")
    mode = coupling @ null[:, 0]
    mode -= embedding @ np.linalg.lstsq(embedding, mode, rcond=None)[0]
    mode /= np.linalg.norm(mode)
    return np.kron(np.column_stack((embedding, mode)), np.eye(2))


def triangle_strain_operators(
    fine: TriangleMesh,
    degree: int,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0),
    order: int = 6,
) -> StrainOperators:
    """Integrate symmetric Kelvin strain energy, nodal mass and vector forcing.

    Coefficients are interleaved Cartesian displacement at continuous Pk nodes.
    The constitutive law uses orthonormal Kelvin coordinates or a symmetric
    fourth-order tensor. No trace, boundary, rigid kernel or gauge is selected.
    Returned mass is scalar nodal mass; users tensor it with Cartesian identity.
    """
    bary, weights, material = material_triangle_quadrature(fine, constitutive, order)
    dofs, nodes, basis, gradients, hessian = element_tabulate(fine, degree, bary)
    ns, size = basis.shape[-1], 2 * len(nodes)
    udofs = (2 * dofs[:, :, None] + np.arange(2)).reshape(len(fine.cells), 2 * ns)
    strain, _, _ = _strain_and_divergence(gradients, hessian)
    strain[:, :, 2] /= np.sqrt(2)
    physical = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    stiffness = constitutive_values(
        material, physical.reshape(-1, 2), lame_lambda=lame_lambda, lame_mu=lame_mu
    ).reshape(*weights.shape, 3, 3)
    blocks = np.einsum("tq,tqai,tqab,tqbj,t->tij", weights, strain, stiffness, strain, fine.areas)
    matrix = _assemble_blocks(blocks, udofs, size)
    force = vector_values(source, physical.reshape(-1, 2)).reshape(*weights.shape, 2)
    loads = np.einsum("tq,tqi,tqa,t->tia", weights, basis, force, fine.areas)
    load = np.bincount(udofs.ravel(), weights=loads.ravel(), minlength=size)
    mass = _assemble_blocks(
        np.einsum("tq,tqi,tqj,t->tij", weights, basis, basis, fine.areas), dofs, len(nodes)
    )
    return StrainOperators(matrix, mass, load, nodes)
