"""Affine Robin coefficient adapter for the general boundary bilinear owner.

The user selects sigma=parameter*(x-origin)/3. Every fine boundary integral
uses the outward macrocell normal and the declared nodal coefficient basis.
"""

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.traces.boundary_forms import BoundaryRule, boundary_bilinear
from pymhm.fem.traces.pressure_3d import boundary_rules
from pymhm.meshes.tetrahedron import TetraMesh


def tetra_robin_boundary_operator(
    mesh: TetraMesh,
    cell: int,
    fine: TetraMesh,
    degree: int,
    order: int,
    nu: float,
    origin: FloatArray,
) -> Any:
    """Integrate sigma.n*u*v by supplied fine facet rules, sigma=nu*(x-origin)/3."""
    size = len(tetra_nodal_space(fine, degree)[1])
    rules = []
    for face, dofs, basis, points, weights, _ in boundary_rules(mesh, cell, fine, degree, order):
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        outward = mesh.signs[cell, side] * mesh.normals[face]
        rules.append(BoundaryRule(face, dofs, basis, points, weights, outward))
    return boundary_bilinear(
        size, tuple(rules), lambda points, normal: nu * ((points - origin) @ normal) / 3
    )
