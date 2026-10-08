"""Scalar boundary quadrature and assembly independent of a physical formulation.

The rules expose physical points, positive surface weights, outward normals and
an executed basis/map. Coefficients may depend on points and normals, allowing
Robin, impedance, reaction and weighted boundary mass terms to use one owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.validation import FloatArray, IntArray, positive_int, real_array
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.quadrilateral import qk_basis, qk_space
from pymhm.fem.scalar.tetrahedron import tetra_face_basis, tetra_nodal_space
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.scalar import edge_basis
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class BoundaryRule:
    """One physical facet rule in the supplied scalar coefficient convention.

    ``basis`` has axes (point, local_dof), ``dofs`` selects the global local-field
    coordinates, and ``normal`` points out of the domain whose boundary is
    integrated. A caller can also supply rules for a different mesh or basis.
    """

    face: int
    dofs: IntArray
    basis: FloatArray
    points: FloatArray
    weights: FloatArray
    normal: FloatArray


def boundary_quadrature(
    mesh: TriangleMesh | TetraMesh | CartesianMacroMesh,
    degree: int,
    *,
    order: int = 6,
) -> tuple[int, tuple[BoundaryRule, ...]]:
    """Tabulate an exterior scalar Pk/Qk nodal boundary with positive physical weights.

    ``order`` counts one-dimensional Gauss points, with the degree+1 floor for
    exact polynomial mass integration. Only exterior facets are returned. Mesh
    normals use their canonical outward boundary orientation. The returned size
    is the complete field dimension, including nodes with zero boundary trace.
    """
    degree = positive_int(degree, "degree")
    order = max(positive_int(order, "order"), degree + 1)
    rules = []
    if isinstance(mesh, TetraMesh):
        dofs, nodes = tetra_nodal_space(mesh, degree)
        bary_face, weights = triangle_quadrature(order)
        for face in mesh.boundary_faces:
            cell = int(mesh.face_cells[face, 0])
            ids = mesh.faces[face]
            bary = np.zeros((len(weights), 4))
            for axis, vertex in enumerate(ids):
                bary[:, np.flatnonzero(mesh.cells[cell] == vertex)[0]] = bary_face[:, axis]
            opposite = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
            basis = tetra_face_basis(degree, bary, opposite_vertex=opposite)
            rules.append(
                BoundaryRule(
                    int(face),
                    dofs[cell],
                    basis,
                    bary_face @ mesh.points[ids],
                    weights * mesh.areas[face],
                    mesh.normals[face],
                )
            )
    else:
        gauss, weights = leggauss(order)
        parameter, weights = (gauss + 1) / 2, weights / 2
        if isinstance(mesh, CartesianMacroMesh):
            dofs, nodes = qk_space(mesh, degree)
        elif isinstance(mesh, TriangleMesh):
            _, nodes = nodal_space(mesh, degree)
        else:
            raise TypeError(
                "boundary quadrature requires triangular, tetrahedral or Cartesian mesh"
            )
        for face in mesh.boundary_faces:
            start, end = mesh.points[mesh.faces[face]]
            points = start + parameter[:, None] * (end - start)
            if isinstance(mesh, CartesianMacroMesh):
                cell = int(mesh.face_cells[face, 0])
                reference = (points - mesh.points[mesh.cells[cell, 0]]) / mesh.spacing
                ids, basis = dofs[cell], qk_basis(degree, reference)[0]
            else:
                ids = np.r_[
                    mesh.faces[face],
                    len(mesh.points) + face * (degree - 1) + np.arange(degree - 1),
                ]
                basis = edge_basis(degree, parameter)
            rules.append(
                BoundaryRule(
                    int(face),
                    ids,
                    basis,
                    points,
                    weights * mesh.lengths[face],
                    mesh.normals[face],
                )
            )
    return len(nodes), tuple(rules)


def _values(coefficient: Any, rule: BoundaryRule) -> FloatArray:
    """Evaluate a supplied finite scalar boundary coefficient at one rule."""
    raw = coefficient(rule.points, rule.normal) if callable(coefficient) else coefficient
    values = real_array(raw, "boundary coefficient")
    return np.broadcast_to(values, (len(rule.weights),))


def boundary_bilinear(
    size: int, rules: tuple[BoundaryRule, ...], coefficient: Any = 1.0
) -> sparse.csc_matrix:
    """Assemble ``integral_boundary coefficient*u*v`` in explicitly supplied coordinates.

    A callback has signature ``coefficient(points, outward_normal)`` and returns
    scalar point values. Negative coefficients are allowed. Rules can represent
    a selected boundary subset or a custom basis; they define all geometry and
    orientation. No essential condition, PDE or solver is selected.
    """
    size = positive_int(size, "field size", 0)
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    for rule in rules:
        block = rule.basis.T @ ((rule.weights * _values(coefficient, rule))[:, None] * rule.basis)
        rows.extend(np.repeat(rule.dofs, len(rule.dofs)))
        columns.extend(np.tile(rule.dofs, len(rule.dofs)))
        entries.extend(block.ravel())
    return sparse.coo_matrix((entries, (rows, columns)), shape=(size, size)).tocsc()


def boundary_functional(size: int, rules: tuple[BoundaryRule, ...], datum: Any = 0.0) -> FloatArray:
    """Assemble ``integral_boundary datum*v`` with the same coordinate/quadrature contract.

    A callback has signature ``datum(points, outward_normal)``. This operation
    interprets neither the data as a physical flux nor its sign in an equation;
    both conventions belong to the formulation written by the user.
    """
    size = positive_int(size, "field size", 0)
    result = np.zeros(size)
    for rule in rules:
        np.add.at(result, rule.dofs, rule.basis.T @ (rule.weights * _values(datum, rule)))
    return result
