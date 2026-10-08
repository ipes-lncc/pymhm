"""Physical boundary functionals and nodal data on Cartesian conforming fields."""

from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.quadrilateral import qk_basis, qk_space
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.cartesian import CartesianMacroMesh


def quadrilateral_boundary_data(
    mesh: CartesianMacroMesh,
    degree: int,
    *,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    order: int = 6,
) -> tuple[FloatArray, dict[int, float]]:
    """Return negative outward-flux load and strongly prescribed nodal values.

    Neumann keys identify exterior faces and their physical outflow density.
    Remaining exterior faces prescribe dirichlet in the executed Qk node order.
    Intersections share the same physical callback value. The returned load is
    added to the volume functional; no source, operator or solver is chosen.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "order"), degree + 1)
    dofs, nodes = qk_space(mesh, degree)
    load = np.zeros(len(nodes))
    natural = {} if neumann is None else neumann
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify boundary faces")
    fixed = np.zeros(len(nodes), dtype=bool)
    gauss, weights = leggauss(order)
    t = (gauss + 1) / 2
    for face in mesh.boundary_faces:
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start
        cell = mesh.face_cells[face, 0]
        if face in natural:
            physical = start + t[:, None] * tangent
            reference = (physical - mesh.points[mesh.cells[cell, 0]]) / mesh.spacing
            basis = qk_basis(degree, reference)[0]
            load[dofs[cell]] -= mesh.lengths[face] * (
                basis.T @ (weights / 2 * scalar_values(natural[int(face)], physical))
            )
        else:
            side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
            edge_nodes = (
                np.arange(degree + 1),
                np.arange(degree + 1) * (degree + 1) + degree,
                degree * (degree + 1) + np.arange(degree + 1),
                np.arange(degree + 1) * (degree + 1),
            )[side]
            fixed[dofs[cell, edge_nodes]] = True
    ids = np.flatnonzero(fixed)
    values = scalar_values(dirichlet, nodes[ids])
    return load, dict(zip(ids.tolist(), values.tolist(), strict=True))
