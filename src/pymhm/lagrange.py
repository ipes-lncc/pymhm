"""Continuous nodal triangle topology and Basix Pk tabulation."""

from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.elements import p1_geometry
from pymhm.mesh import FloatArray, IntArray, SkeletonSpace, TriangleMesh, positive_int


def multiindices(degree: int) -> IntArray:
    """Order barycentric lattice nodes by vertices, oriented edges and interior."""
    k = positive_int(degree, "degree")
    indices: list[tuple[int, ...]] = [(k, 0, 0), (0, k, 0), (0, 0, k)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for j in range(1, k):
            node = [0, 0, 0]
            node[a], node[b] = k - j, j
            indices.append(tuple(node))
    indices.extend((i, j, k - i - j) for i in range(1, k) for j in range(1, k - i))
    return np.asarray(indices, dtype=np.int64)


def reference_values(degree: int, bary: FloatArray) -> FloatArray:
    """Evaluate native equispaced triangular Pk values in PyMHM node order.

    ``bary`` is a finite real array of shape (points, 3); evaluation is allowed
    outside the reference triangle on the unit-sum barycentric hyperplane.
    Coordinates and returned values use binary64, with the same node order as
    :func:`multiindices`. Basix tabulates values without allocating derivatives.
    """
    from pymhm.element_backends import simplex_lagrange_tabulation

    nodes = multiindices(degree) / degree
    if np.iscomplexobj(bary):
        raise ValueError("barycentric coordinates must be finite real triples")
    bary = np.asarray(bary, dtype=float)
    if bary.ndim != 2 or bary.shape[1] != 3 or not np.isfinite(bary).all():
        raise ValueError("barycentric coordinates must be finite real triples")
    return simplex_lagrange_tabulation("triangle", degree, bary, nodes=nodes, nderiv=0)[0]


def reference_basis(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return Basix Pk values and derivatives in the canonical barycentric extension.

    Shapes are ``(points, basis)``, ``(points, basis, 3)`` and
    ``(points, basis, 3, 3)``. The extension is ``p(lambda_1,lambda_2)``:
    lambda_0 derivatives are zero, while the other entries are Cartesian
    reference derivatives. Contract with all three physical barycentric
    gradients to obtain physical derivatives. Input coordinates must sum to
    one; off-triangle points on that hyperplane are supported. The node order
    remains :func:`multiindices`; independent off-hyperplane extensions are
    not part of the finite-element field.
    """
    from pymhm.element_backends import barycentric_simplex_tabulation

    return barycentric_simplex_tabulation(
        "triangle", degree, bary, nodes=multiindices(degree) / degree
    )


def nodal_space(mesh: TriangleMesh, degree: int) -> tuple[IntArray, FloatArray]:
    """Enumerate continuous nodal unknowns without merging geometrically close nodes."""
    indices = multiindices(degree)
    points = [mesh.points]
    dofs = np.empty((len(mesh.cells), len(indices)), dtype=np.int64)
    dofs[:, :3] = mesh.cells
    offset = len(mesh.points)
    if degree > 1:
        parameter = np.arange(1, degree) / degree
        edge_points = (
            mesh.points[mesh.faces[:, 0], None] * (1 - parameter[None, :, None])
            + mesh.points[mesh.faces[:, 1], None] * parameter[None, :, None]
        )
        points.append(edge_points.reshape(-1, 2))
        for side, (a, _b) in enumerate(((0, 1), (1, 2), (2, 0))):
            forward = mesh.cells[:, a] == mesh.faces[mesh.cell_faces[:, side], 0]
            local = np.where(forward[:, None], np.arange(degree - 1), np.arange(degree - 2, -1, -1))
            dofs[:, 3 + side * (degree - 1) : 3 + (side + 1) * (degree - 1)] = (
                offset + mesh.cell_faces[:, side, None] * (degree - 1) + local
            )
        offset += len(mesh.faces) * (degree - 1)
    interior = indices[3 * degree :] / degree
    if len(interior):
        points.append(np.einsum("qi,tij->tqj", interior, mesh.points[mesh.cells]).reshape(-1, 2))
        dofs[:, 3 * degree :] = offset + np.arange(len(mesh.cells) * len(interior)).reshape(
            len(mesh.cells), -1
        )
    return dofs, np.concatenate(points)


def tabulate(
    mesh: TriangleMesh,
    degree: int,
    bary: FloatArray,
    *,
    backend: Literal["portable", "basix"] = "basix",
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return DOFs, coordinates, values, physical gradients and physical Hessians.

    Basix supplies equispaced Pk values and Cartesian derivatives in
    :func:`multiindices` order. Points must sum to one. The affine geometry,
    continuous topology and nodal coefficient order are explicit PyMHM data.
    ``backend='portable'`` is a compatibility spelling for this same Basix
    execution; there is no separate polynomial implementation.
    """
    from pymhm.element_backends import physical_simplex_tabulation

    if backend not in ("basix", "portable"):
        raise ValueError("backend must be portable or basix")
    dofs, points = nodal_space(mesh, degree)
    geometry, _ = p1_geometry(mesh)
    basis, first, second = physical_simplex_tabulation(
        "triangle",
        degree,
        bary,
        nodes=multiindices(degree) / degree,
        reference_gradients=geometry[:, 1:],
    )
    return dofs, points, basis, first, second


def element_tabulate(
    mesh: TriangleMesh,
    degree: int,
    bary: FloatArray,
    *,
    backend: Literal["portable", "basix"] = "basix",
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Tabulate Pk at separate barycentric quadrature points in each triangle.

    ``bary`` has shape ``(cells, points, 3)``. All returned basis arrays retain
    the leading cell axis, so material intersections do not require evaluating
    every element's basis at every other element's quadrature points.
    ``backend`` follows :func:`tabulate`; tabulation preserves the declared
    continuous topology and nodal coefficient order.
    """
    if bary.strides[0] == 0:
        dofs, nodes, values, first, second = tabulate(mesh, degree, bary[0], backend=backend)
        return dofs, nodes, np.broadcast_to(values, (len(mesh.cells), *values.shape)), first, second
    from pymhm.element_backends import physical_simplex_tabulation

    if backend not in ("basix", "portable"):
        raise ValueError("backend must be portable or basix")
    dofs, nodes = nodal_space(mesh, degree)
    geometry, _ = p1_geometry(mesh)
    values, first, second = physical_simplex_tabulation(
        "triangle",
        degree,
        bary,
        nodes=multiindices(degree) / degree,
        reference_gradients=geometry[:, 1:],
    )
    return dofs, nodes, values, first, second


def trace_coupling(
    coarse: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace, degree: int
) -> FloatArray:
    """Integrate signed nodal traces over independently subdivided macrofaces."""
    from pymhm.element_backends import simplex_lagrange_tabulation

    _, points = nodal_space(fine, degree)
    width = sum(skeleton.faces[f].size for f in coarse.cell_faces[cell])
    matrix = np.zeros((len(points), width))
    offset = 0
    for side, face in enumerate(coarse.cell_faces[cell]):
        space = skeleton.faces[face]
        start, end = coarse.points[coarse.faces[face]]
        tangent = end - start
        length = np.linalg.norm(tangent)
        gauss, gauss_weights = leggauss(max(space.degrees) + degree + 1)
        for fine_face in fine.boundary_faces:
            nodes = fine.faces[fine_face]
            coords = fine.points[nodes]
            t = (coords - start) @ tangent / length**2
            if (
                not np.allclose(coords, start + t[:, None] * tangent, atol=1e-12, rtol=1e-12)
                or min(t) < -1e-12
                or max(t) > 1 + 1e-12
            ):
                continue
            lo, hi = np.clip(np.sort(t), 0, 1)
            cuts = sorted({lo, hi, *(b for b in space.breaks if lo < b < hi)})
            ids = np.r_[nodes, len(fine.points) + fine_face * (degree - 1) + np.arange(degree - 1)]
            for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                parameter = left + (gauss + 1) * (right - left) / 2
                weights = gauss_weights * (right - left) / 2 * length * coarse.signs[cell, side]
                s = (parameter - t[0]) / (t[1] - t[0])
                nodes_1d = np.r_[0.0, 1.0, np.arange(1, degree) / degree]
                basis = simplex_lagrange_tabulation(
                    "interval",
                    degree,
                    np.column_stack((1 - s, s)),
                    nodes=np.column_stack((1 - nodes_1d, nodes_1d)),
                    nderiv=0,
                )[0]
                matrix[ids, offset : offset + space.size] += basis.T @ (
                    weights[:, None] * space.evaluate(parameter)
                )
        offset += space.size
    return matrix


def scalar_operators(
    mesh: TriangleMesh,
    degree: int,
    *,
    diffusion: Any = 1.0,
    source: Any = 0.0,
    reaction: Any = 0.0,
    advection: Any = (0.0, 0.0),
    skew_advection: bool = False,
    order: int = 6,
    element_backend: Literal["portable", "basix"] = "basix",
) -> tuple[Any, Any, FloatArray]:
    """Assemble continuous Pk scalar diffusion, reaction and advective operators.

    With ``skew_advection=False`` the convection term is ``(beta.grad(u),v)``.
    Its antisymmetric part is selected when ``skew_advection=True``. A variable
    conservative velocity also requires half its divergence in ``reaction``.
    Basix tabulates local Pk values and derivatives in the same nodal
    coefficient order. ``element_backend='portable'`` is a compatibility
    spelling for this same execution.
    """
    from scipy import sparse

    from pymhm.cut_cells import material_triangle_quadrature
    from pymhm.elements import _scalar_diffusion_blocks, scalar_values, tensor_values, vector_values

    bary, weights, material = material_triangle_quadrature(mesh, diffusion, max(order, degree + 2))
    dofs, nodes, basis, gradients, _ = element_tabulate(mesh, degree, bary, backend=element_backend)
    physical = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    flat = physical.reshape(-1, 2)
    coefficient = tensor_values(material, flat).reshape(*weights.shape, 2, 2)
    scalar = scalar_values(reaction, flat).reshape(weights.shape)
    if np.any(scalar < 0):
        raise ValueError("reaction must be nonnegative")
    beta = vector_values(advection, flat).reshape(*weights.shape, 2)
    force = scalar_values(source, flat).reshape(weights.shape)
    mass_blocks = np.einsum("tq,tqi,tqj,t->tij", weights, basis, basis, mesh.areas)
    blocks = _scalar_diffusion_blocks(weights, gradients, coefficient, mesh.areas)
    blocks += np.einsum("tq,tqi,tqj,tq,t->tij", weights, basis, basis, scalar, mesh.areas)
    convection = np.einsum("tq,tqi,tqa,tqja,t->tij", weights, basis, beta, gradients, mesh.areas)
    blocks += (convection - convection.swapaxes(1, 2)) / 2 if skew_advection else convection
    row = np.repeat(dofs, basis.shape[-1], axis=1).ravel()
    column = np.tile(dofs, (1, basis.shape[-1])).ravel()
    matrix = sparse.coo_matrix(
        (blocks.ravel(), (row, column)), shape=(len(nodes), len(nodes))
    ).tocsc()
    mass = sparse.coo_matrix((mass_blocks.ravel(), (row, column)), shape=matrix.shape).tocsc()
    load = np.einsum("tq,tqi,tq,t->ti", weights, basis, force, mesh.areas)
    return matrix, mass, np.bincount(dofs.ravel(), weights=load.ravel(), minlength=len(nodes))
