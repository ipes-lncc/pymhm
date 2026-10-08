"""Symmetric-strain volume integration independently of trace and kernel choices."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.tetrahedron import tetra_tabulate, tetrahedron_quadrature
from pymhm.fem.vector.elasticity_3d import KELVIN_BASIS_3D as _KELVIN3
from pymhm.fem.vector.elasticity_3d import _component as _component
from pymhm.fem.vector.elasticity_3d import rigid_modes_3d as rigid_modes_3d
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.elasticity import constitutive_values_3d as constitutive_values_3d
from pymhm.materials.evaluation import vector_values_3d
from pymhm.meshes.tetrahedron import TetraMesh


@dataclass(frozen=True)
class StrainOperators:
    """Literal vector energy/load and scalar mass in physical nodal coordinates."""

    matrix: Any
    mass: Any
    load: FloatArray
    nodes: FloatArray


def tetra_strain_operators(
    fine: TetraMesh,
    degree: int,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0, 0.0),
    order: int = 6,
) -> StrainOperators:
    """Integrate symmetric Kelvin strain energy, nodal mass and vector forcing.

    Coefficients are interleaved Cartesian displacement at continuous Pk nodes.
    The constitutive law uses orthonormal Kelvin coordinates or a symmetric
    fourth-order tensor. No trace, boundary, rigid kernel or gauge is selected.
    Returned mass is scalar nodal mass; users tensor it with Cartesian identity.
    """
    bary, weights = tetrahedron_quadrature(order)
    dofs, nodes, basis, gradient = tetra_tabulate(fine, degree, bary)
    udofs = (3 * dofs[:, :, None] + np.arange(3)).reshape(len(fine.cells), -1)
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    stiffness = constitutive_values_3d(
        constitutive, physical.reshape(-1, 3), lame_lambda=lame_lambda, lame_mu=lame_mu
    ).reshape(*physical.shape[:2], 6, 6)
    strain = np.einsum("aij,tqnj->tqani", _KELVIN3, gradient).reshape(*gradient.shape[:2], 6, -1)
    blocks = np.einsum(
        "t,q,tqai,tqab,tqbj->tij",
        fine.volumes,
        weights,
        strain,
        stiffness,
        strain,
        optimize=True,
    )
    size = 3 * len(nodes)
    matrix = _assemble_blocks(blocks, udofs, size)
    mass = _assemble_blocks(
        np.einsum("t,q,qi,qj->tij", fine.volumes, weights, basis, basis), dofs, len(nodes)
    )
    force = vector_values_3d(source, physical.reshape(-1, 3)).reshape(physical.shape)
    load = np.bincount(
        udofs.ravel(),
        weights=np.einsum("t,q,qi,tqa->tia", fine.volumes, weights, basis, force).ravel(),
        minlength=size,
    )
    return StrainOperators(matrix, mass, load, nodes)
