"""Continuous nodal simplex elements with exact first and second derivatives."""

from math import factorial
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
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


def reference_basis(degree: int, bary: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate cardinal polynomials and their barycentric derivatives analytically."""
    indices = multiindices(degree)
    table = np.empty((3, degree + 1, 3, len(bary)))
    for i in range(3):
        for k in range(degree + 1):
            polynomial = (
                Polynomial.fromroots(np.arange(k) / degree) * degree**k / factorial(k)
                if k
                else Polynomial([1.0])
            )
            for derivative in range(3):
                table[i, k, derivative] = polynomial.deriv(derivative)(bary[:, i])
    values = np.ones((len(bary), len(indices)))
    gradient = np.ones((*values.shape, 3))
    hessian = np.ones((*values.shape, 3, 3))
    for node, index in enumerate(indices):
        for i in range(3):
            values[:, node] *= table[i, index[i], 0]
            for a in range(3):
                gradient[:, node, a] *= table[i, index[i], int(i == a)]
                for b in range(3):
                    hessian[:, node, a, b] *= table[i, index[i], int(i == a) + int(i == b)]
    return values, gradient, hessian


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
    mesh: TriangleMesh, degree: int, bary: FloatArray
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return DOFs, coordinates, values, physical gradients and physical Hessians."""
    dofs, points = nodal_space(mesh, degree)
    basis, derivative, second = reference_basis(degree, bary)
    gradients, _ = p1_geometry(mesh)
    return (
        dofs,
        points,
        basis,
        np.einsum("qin,tna->tqia", derivative, gradients),
        np.einsum("qinm,tna,tmb->tqiab", second, gradients, gradients),
    )


def element_tabulate(
    mesh: TriangleMesh, degree: int, bary: FloatArray
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Tabulate Pk at separate barycentric quadrature points in each triangle.

    ``bary`` has shape ``(cells, points, 3)``. All returned basis arrays retain
    the leading cell axis, so material intersections do not require evaluating
    every element's basis at every other element's quadrature points.
    """
    if bary.strides[0] == 0:
        dofs, nodes, values, first, second = tabulate(mesh, degree, bary[0])
        return dofs, nodes, np.broadcast_to(values, (len(mesh.cells), *values.shape)), first, second
    dofs, nodes = nodal_space(mesh, degree)
    values, first, second = reference_basis(degree, bary.reshape(-1, 3))
    shape = (*bary.shape[:2], values.shape[1])
    geometry, _ = p1_geometry(mesh)
    gradient = np.einsum("tqib,tba->tqia", first.reshape(*shape, 3), geometry)
    hessian = np.einsum("tqibc,tba,tcd->tqiad", second.reshape(*shape, 3, 3), geometry, geometry)
    return dofs, nodes, values.reshape(shape), gradient, hessian


def trace_coupling(
    coarse: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace, degree: int
) -> FloatArray:
    """Integrate signed nodal traces over independently subdivided macrofaces."""
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
                basis = np.ones((len(s), degree + 1))
                for i, node in enumerate(nodes_1d):
                    for j, other in enumerate(nodes_1d):
                        if i != j:
                            basis[:, i] *= (s - other) / (node - other)
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
) -> tuple[Any, Any, FloatArray]:
    """Assemble continuous Pk scalar diffusion, reaction and advective operators.

    With ``skew_advection=False`` the convection term is ``(beta.grad(u),v)``.
    Its antisymmetric part is selected when ``skew_advection=True``. A variable
    conservative velocity also requires half its divergence in ``reaction``.
    """
    from scipy import sparse

    from pymhm.cut_cells import material_triangle_quadrature
    from pymhm.elements import scalar_values, tensor_values, vector_values

    bary, weights, material = material_triangle_quadrature(mesh, diffusion, max(order, degree + 2))
    dofs, nodes, basis, gradients, _ = element_tabulate(mesh, degree, bary)
    physical = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    flat = physical.reshape(-1, 2)
    coefficient = tensor_values(material, flat).reshape(*weights.shape, 2, 2)
    scalar = scalar_values(reaction, flat).reshape(weights.shape)
    if np.any(scalar < 0):
        raise ValueError("reaction must be nonnegative")
    beta = vector_values(advection, flat).reshape(*weights.shape, 2)
    force = scalar_values(source, flat).reshape(weights.shape)
    mass_blocks = np.einsum("tq,tqi,tqj,t->tij", weights, basis, basis, mesh.areas)
    blocks = np.einsum(
        "tq,tqia,tqab,tqjb,t->tij", weights, gradients, coefficient, gradients, mesh.areas
    )
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
