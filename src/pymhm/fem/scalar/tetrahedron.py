"""Basix scalar Pk topology, tabulation and affine tetrahedral operators."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.scalar.operators import _scalar_diffusion_blocks
from pymhm.fem.scalar.tetrahedron_topology import (
    continuous_tetra_nodes,
    tetra_indices,
    tetra_values_gradients,
)
from pymhm.materials.evaluation import (
    scalar_values_3d as scalar_values_3d,
)
from pymhm.materials.evaluation import (
    tensor_values_3d as tensor_values_3d,
)
from pymhm.meshes.tetrahedron import (
    TetraMesh as TetraMesh,
)
from pymhm.meshes.tetrahedron import (
    _dyadic as _dyadic,
)
from pymhm.meshes.tetrahedron import (
    _reference_refinement as _reference_refinement,
)
from pymhm.meshes.tetrahedron import (
    tetra_barycentric_gradients as tetra_barycentric_gradients,
)

_EDGES = tuple(combinations(range(4), 2))


def tetrahedron_quadrature(order: int = 5) -> tuple[FloatArray, FloatArray]:
    """Return positive Duffy-product quadrature with weights normalized to unit volume."""
    order = positive_int(order, "order")
    x, w = leggauss(order)
    x, w = (x + 1) / 2, w / 2
    a, b, c = np.meshgrid(x, x, x, indexing="ij")
    wa, wb, wc = np.meshgrid(w, w, w, indexing="ij")
    xyz = np.array([a, (1 - a) * b, (1 - a) * (1 - b) * c]).reshape(3, -1).T
    bary = np.column_stack((1 - xyz.sum(axis=1), xyz))
    weights = (6 * wa * wb * wc * (1 - a) ** 2 * (1 - b)).ravel()
    return bary, weights


def tetra_nodal_space(mesh: TetraMesh, degree: int) -> tuple[IntArray, FloatArray]:
    """Enumerate continuous Pk nodes using exact topological identities."""
    degree = positive_int(degree, "degree")
    if degree > 2:
        return continuous_tetra_nodes(mesh, degree)
    if degree == 1:
        return mesh.cells, mesh.points
    edges, inverse = np.unique(
        np.sort(mesh.cells[:, _EDGES].reshape(-1, 2), axis=1), axis=0, return_inverse=True
    )
    dofs = np.column_stack((mesh.cells, len(mesh.points) + inverse.reshape(-1, 6)))
    points = np.vstack((mesh.points, mesh.points[edges].mean(axis=1)))
    return dofs, points


def tetra_basis(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return Basix Pk values and canonical four-axis barycentric derivatives.

    The extension and unit-sum point convention follow
    :func:`pymhm.fem.scalar.tetrahedron_topology.tetra_values_gradients` for every degree.
    """
    return tetra_values_gradients(degree, bary)


def tetra_face_basis(degree: int, bary: FloatArray, *, opposite_vertex: int) -> FloatArray:
    """Restrict the executed nodal Pk basis to one exact reference face.

    ``bary`` has four unit-sum coordinates with coordinate ``opposite_vertex``
    exactly zero. Entries whose integer nodal weight on that vertex is nonzero
    have identically zero trace and are excluded through the declared topology.
    Values on the face support retain the native volume table unchanged. The
    volume basis, its coefficients and physical derivative maps are unaffected.
    """
    opposite = positive_int(opposite_vertex, "opposite_vertex", 0)
    if opposite >= 4:
        raise ValueError("opposite_vertex must identify one of four reference vertices")
    values = tetra_basis(degree, bary)[0].copy()
    if np.any(np.asarray(bary)[:, opposite] != 0):
        raise ValueError("face barycentric coordinates must vanish at opposite_vertex")
    values[:, tetra_indices(degree)[:, opposite] != 0] = 0.0
    return values


def tetra_tabulate(
    mesh: TetraMesh,
    degree: int,
    bary: FloatArray,
    *,
    backend: Literal["portable", "basix"] = "basix",
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
    """Return nodal DOFs/coordinates, basis values and physical gradients.

    Basix supplies equispaced Pk values/Cartesian derivatives in the literal
    :func:`tetra_indices` node order. Barycentric points must sum to one.
    PyMHM supplies the affine cell geometry and continuous DOF topology.
    ``backend='portable'`` is a compatibility spelling for this same Basix
    execution; there is no separate cardinal-polynomial implementation.
    """
    from pymhm.fem.reference import physical_simplex_tabulation

    if backend not in ("basix", "portable"):
        raise ValueError("backend must be portable or basix")
    dofs, points = tetra_nodal_space(mesh, degree)
    inverse = tetra_barycentric_gradients(mesh)[:, 1:]
    values, gradient, _ = physical_simplex_tabulation(
        "tetrahedron",
        degree,
        bary,
        nodes=tetra_indices(degree) / degree,
        reference_gradients=inverse,
        nderiv=1,
    )
    return dofs, points, values, gradient


def tetra_element_tabulate(
    mesh: TetraMesh,
    degree: int,
    bary: FloatArray,
    *,
    backend: Literal["portable", "basix"] = "basix",
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Tabulate continuous Pk values, physical gradients and physical Hessians.

    ``backend`` follows :func:`tetra_tabulate`. Basix tabulates both
    derivative orders to the library before applying the affine Jacobian.
    """
    from pymhm.fem.reference import physical_simplex_tabulation

    if backend not in ("basix", "portable"):
        raise ValueError("backend must be portable or basix")
    dofs, points = tetra_nodal_space(mesh, degree)
    inverse = tetra_barycentric_gradients(mesh)[:, 1:]
    values, gradient, hessian = physical_simplex_tabulation(
        "tetrahedron",
        degree,
        bary,
        nodes=tetra_indices(degree) / degree,
        reference_gradients=inverse,
    )
    return dofs, points, values, gradient, hessian


def tetra_operators(
    mesh: TetraMesh,
    degree: int = 2,
    *,
    diffusion: Any = 1.0,
    source: Any = 0.0,
    order: int = 5,
    element_backend: Literal["portable", "basix"] = "basix",
) -> tuple[Any, Any, FloatArray]:
    """Assemble scalar diffusion, consistent mass and source with positive quadrature.

    Basix tabulates equispaced Pk in the declared nodal order. Integration and
    affine geometry retain their conventions. ``element_backend='portable'``
    is a compatibility spelling for this same Basix execution.
    Compiled binary64 diffusion integration compensates both Cartesian
    contractions and quadrature sums; mass and source retain their formulas.
    """
    bary, weights = tetrahedron_quadrature(max(order, degree + 2))
    dofs, points, values, gradients = tetra_tabulate(mesh, degree, bary, backend=element_backend)
    x = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    tensors = tensor_values_3d(diffusion, x.reshape(-1, 3)).reshape(*x.shape[:2], 3, 3)
    force = scalar_values_3d(source, x.reshape(-1, 3)).reshape(x.shape[:2])
    stiffness = _scalar_diffusion_blocks(weights, gradients, tensors, mesh.volumes)
    mass = np.einsum("t,q,qi,qj->tij", mesh.volumes, weights, values, values)
    load = np.einsum("t,q,qi,tq->ti", mesh.volumes, weights, values, force)
    rows = np.broadcast_to(dofs[:, :, None], stiffness.shape).ravel()
    columns = np.broadcast_to(dofs[:, None, :], stiffness.shape).ravel()
    size = len(points)
    matrix = sparse.coo_matrix((stiffness.ravel(), (rows, columns)), shape=(size, size)).tocsc()
    assembled_mass = sparse.coo_matrix((mass.ravel(), (rows, columns)), shape=(size, size)).tocsc()
    assembled_load = np.zeros(size)
    np.add.at(assembled_load, dofs, load)
    return matrix, assembled_mass, assembled_load


def tetra_boundary_nodes(mesh: TetraMesh, degree: int) -> IntArray:
    """Return sorted continuous-Pk nodal coordinates on the exterior boundary.

    Exterior incidence is tested in affine barycentric coordinates against the
    opposite local vertex, with the existing geometric tolerance 1e-12. This
    identifies nodes; it neither projects boundary data nor chooses a PDE sign.
    """
    dofs, nodes = tetra_nodal_space(mesh, degree)
    boundary: set[int] = set()
    for face in mesh.boundary_faces:
        cell = int(mesh.face_cells[face, 0])
        vertices = mesh.points[mesh.cells[cell]]
        reference = (nodes[dofs[cell]] - vertices[0]) @ np.linalg.inv(
            (vertices[1:] - vertices[0]).T
        ).T
        bary = np.column_stack((1 - reference.sum(axis=1), reference))
        opposite = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        boundary.update(dofs[cell][np.abs(bary[:, opposite]) < 1e-12])
    return np.array(sorted(boundary), dtype=np.int64)
