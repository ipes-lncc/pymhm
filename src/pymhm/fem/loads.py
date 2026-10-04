"""Discrete point-source functionals with geometric sharing at mesh interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.fem.scalar.quadrilateral import qk_basis, qk_space
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh


def _sources(values: Any) -> FloatArray:
    """Validate an (n,3) array of x, y and signed source strengths."""
    array = np.asarray(values)
    if array.size == 0:
        return np.empty((0, 3))
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError("point sources require finite real rows (x, y, strength)")
    return np.asarray(array, dtype=float)


def _barycentric(mesh: TriangleMesh, point: FloatArray) -> FloatArray:
    """Return barycentric coordinates of one point in every affine triangle."""
    vertices = mesh.points[mesh.cells]
    jacobian = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
    coordinates = np.linalg.solve(jacobian, (point - vertices[:, 0])[..., None])[..., 0]
    return np.column_stack((1 - coordinates.sum(axis=1), coordinates))


def _rectangle_locations(mesh: CartesianMacroMesh, point: FloatArray) -> tuple[Any, FloatArray]:
    """Locate a point in closed Cartesian cells, preserving both sides of interfaces."""
    origin = np.asarray(mesh.bounds)[[0, 2]]
    grid = (point - origin) / mesh.spacing
    counts = np.array([mesh.nx, mesh.ny])
    tolerance = 128 * np.finfo(float).eps * np.maximum(1, counts)
    if np.any(grid < -tolerance) or np.any(grid > counts + tolerance):
        raise ValueError("point source lies outside the mesh")
    rounded = np.rint(grid)
    grid = np.where(abs(grid - rounded) <= tolerance, rounded, grid)
    axes = [
        np.arange(max(0, int(np.ceil(x)) - 1), min(int(n) - 1, int(np.floor(x))) + 1)
        for x, n in zip(grid, counts, strict=True)
    ]
    x, y = np.meshgrid(*axes)
    indices = np.column_stack((x.ravel(), y.ravel()))
    return indices[:, 1] * mesh.nx + indices[:, 0], grid - indices


def split_point_sources(
    mesh: TriangleMesh | CartesianMacroMesh | PolygonMesh, sources: Any
) -> tuple[FloatArray, ...]:
    """Share source strengths among cells in proportion to their incident angles.

    Triangular, Cartesian and simple polygonal meshes are supported.
    Interior points belong to one cell; interface points are split by the
    incident angular sectors. At an interior edge this gives half to either
    cell, and at a vertex it uses the actual triangle angles. This is the
    limit of a radially symmetric regularization of the load, restricted to
    the domain. The total specified strength is preserved, including at domain
    corners, without duplicating wells shared by several macrocells.
    """
    values = _sources(sources)
    allocated: list[list[FloatArray]] = [[] for _ in mesh.cells]
    if isinstance(mesh, PolygonMesh):
        triangles = np.concatenate(
            [vertices[local] for vertices, local in zip(mesh.cells, mesh._triangles, strict=True)]
        )
        parents = np.repeat(np.arange(len(mesh.cells)), [len(t) for t in mesh._triangles])
        for parent, part in zip(
            parents, split_point_sources(TriangleMesh(mesh.points, triangles), values), strict=True
        ):
            allocated[parent].extend(part)
        return tuple(np.asarray(rows).reshape(-1, 3) for rows in allocated)
    if isinstance(mesh, CartesianMacroMesh):
        for point in values:
            cells, _ = _rectangle_locations(mesh, point[:2])
            for cell in cells:
                allocated[cell].append(np.r_[point[:2], point[2] / len(cells)])
        return tuple(np.asarray(rows).reshape(-1, 3) for rows in allocated)
    tolerance = 128 * np.finfo(float).eps
    for point in values:
        barycentric = _barycentric(mesh, point[:2])
        cells = np.flatnonzero(np.all(barycentric >= -tolerance, axis=1))
        if not len(cells):
            raise ValueError("point source lies outside the mesh")
        angles = []
        for cell in cells:
            bary = barycentric[cell]
            zeros = np.abs(bary) <= tolerance
            if np.count_nonzero(zeros) >= 2:
                vertex = int(np.argmax(bary))
                vertices = mesh.points[mesh.cells[cell]]
                edges = np.delete(vertices, vertex, axis=0) - vertices[vertex]
                cosine = (edges[0] @ edges[1]) / np.prod(np.linalg.norm(edges, axis=1))
                angle = np.arccos(np.clip(cosine, -1, 1))
            else:
                angle = np.pi if np.any(zeros) else 2 * np.pi
            angles.append(angle)
        weights = np.asarray(angles) / sum(angles)
        for cell, weight in zip(cells, weights, strict=True):
            allocated[cell].append(np.r_[point[:2], point[2] * weight])
    return tuple(np.asarray(rows).reshape(-1, 3) for rows in allocated)


def point_load_vector(
    mesh: TriangleMesh | CartesianMacroMesh, degree: int, sources: Any
) -> FloatArray:
    """Assemble ``sum Q_j v(x_j)`` for continuous triangular Pk or Cartesian Qk tests.

    A source on an internal fine edge/vertex is evaluated once; continuity of
    the nodal finite-element test space makes the containing-cell choice
    immaterial. Use :func:`split_point_sources` first to allocate sources across
    a broken macro space. Two-dimensional Dirac sources are not H⁻¹ loads;
    this discrete operation does not imply a finite continuum energy norm.
    """
    values = _sources(sources)
    if isinstance(mesh, CartesianMacroMesh):
        dofs, nodes = qk_space(mesh, degree)
        load = np.zeros(len(nodes))
        for point in values:
            cells, reference = _rectangle_locations(mesh, point[:2])
            basis, _ = qk_basis(degree, reference[:1])
            np.add.at(load, dofs[cells[0]], point[2] * basis[0])
        return load
    dofs, nodes = nodal_space(mesh, degree)
    load = np.zeros(len(nodes))
    tolerance = 128 * np.finfo(float).eps
    for point in values:
        barycentric = _barycentric(mesh, point[:2])
        cells = np.flatnonzero(np.all(barycentric >= -tolerance, axis=1))
        if not len(cells):
            raise ValueError("point source lies outside the mesh")
        cell = int(cells[0])
        basis, _, _ = reference_basis(degree, barycentric[cell][None, :])
        np.add.at(load, dofs[cell], point[2] * basis[0])
    return load
