"""One-sided polynomial traces and local scalar boundary integration."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.elements import scalar_values, vector_values
from pymhm.lagrange import nodal_space
from pymhm.mesh import FloatArray, IntArray, TriangleMesh


def edge_basis(degree: int, parameter: FloatArray) -> FloatArray:
    """Evaluate edge nodes ordered as endpoints followed by interior nodes."""
    from pymhm.element_backends import simplex_lagrange_tabulation

    nodes = np.r_[0.0, 1.0, np.arange(1, degree) / degree]
    bary = np.column_stack((1 - parameter, parameter))
    return simplex_lagrange_tabulation(
        "interval", degree, bary, nodes=np.column_stack((1 - nodes, nodes)), nderiv=0
    )[0]


def edge_pieces(
    macro: TriangleMesh, fine: TriangleMesh, face: int, degree: int
) -> Iterator[tuple[FloatArray, IntArray]]:
    """Yield oriented macro parameters and nodal DOFs of matching fine edges."""
    start, end = macro.points[macro.faces[face]]
    tangent = end - start
    for edge in fine.boundary_faces:
        ids = fine.faces[edge]
        coords = fine.points[ids]
        parameter = (coords - start) @ tangent / (tangent @ tangent)
        if (
            np.allclose(coords, start + parameter[:, None] * tangent, rtol=0, atol=1e-12)
            and parameter.min() >= -1e-12
            and parameter.max() <= 1 + 1e-12
        ):
            nodes = np.r_[ids, len(fine.points) + edge * (degree - 1) + np.arange(degree - 1)]
            yield parameter, nodes


@dataclass(frozen=True)
class PreparedScalarTrace:
    """Read-only face geometry for repeated one-sided nodal trace evaluations.

    Intervals retain the original fine-edge order and endpoint tolerance.
    A sorted interval index visits only overlapping pieces; the original last
    assignment wins at shared endpoints, including tolerance overlaps. Arrays
    are snapshots of the geometry, while each evaluation reads current field
    coefficients. No process-global geometry cache or field averaging is used.
    """

    degree: int
    pieces: tuple[tuple[FloatArray, IntArray], ...]
    starts: FloatArray
    ends: FloatArray
    prefix_maximum: FloatArray
    order: IntArray

    def selections(self, parameter: FloatArray) -> Iterator[tuple[FloatArray, IntArray, IntArray]]:
        """Yield overlapping sample indices in the original fine-boundary order."""
        groups: dict[int, list[int]] = {}
        for sample, point in enumerate(parameter):
            position = int(np.searchsorted(self.starts, point, side="right")) - 1
            while position >= 0 and self.prefix_maximum[position] >= point:
                if self.ends[position] >= point:
                    groups.setdefault(int(self.order[position]), []).append(sample)
                position -= 1
        for index in sorted(groups):
            positions, ids = self.pieces[index]
            yield positions, ids, np.asarray(groups[index], dtype=np.int64)

    def evaluate(self, coefficients: FloatArray, parameter: FloatArray) -> FloatArray:
        """Evaluate without rebuilding edge geometry or changing endpoint ownership."""
        values = np.empty(len(parameter))
        covered = np.zeros(len(parameter), dtype=bool)
        for positions, ids, selected in self.selections(parameter):
            local = (parameter[selected] - positions[0]) / (positions[1] - positions[0])
            values[selected] = edge_basis(self.degree, local) @ coefficients[ids]
            covered[selected] = True
        if not covered.all():
            raise ValueError("trace points must lie on the supplied macroface")
        return values


def prepare_scalar_trace(
    macro: TriangleMesh, fine: TriangleMesh, face: int, degree: int
) -> PreparedScalarTrace:
    """Prepare immutable oriented edge pieces and their exact tolerance intervals."""
    pieces = tuple(
        (positions.copy(), ids.copy()) for positions, ids in edge_pieces(macro, fine, face, degree)
    )
    starts = np.array([min(positions) - 1e-13 for positions, _ in pieces])
    ends = np.array([max(positions) + 1e-13 for positions, _ in pieces])
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    prefix = np.maximum.accumulate(ends)
    for values in (starts, ends, order, prefix, *(a for piece in pieces for a in piece)):
        values.flags.writeable = False
    return PreparedScalarTrace(degree, pieces, starts, ends, prefix, order)


def scalar_trace(
    macro: TriangleMesh,
    fine: TriangleMesh,
    face: int,
    degree: int,
    coefficients: FloatArray,
    parameter: FloatArray,
) -> FloatArray:
    """Evaluate a continuous local trace without averaging across macrofaces."""
    return prepare_scalar_trace(macro, fine, face, degree).evaluate(coefficients, parameter)


def diffusive_boundary_matrix(
    macro: TriangleMesh,
    fine: TriangleMesh,
    faces: tuple[int, ...],
    degree: int,
    velocity: Any,
    order: int,
) -> Any:
    """Assemble +half beta.n boundary mass for prescribed diffusive flux.

    On these exterior faces the retained multiplier is -K grad(u).n, rather
    than the interior Robin multiplier (-K grad(u)+beta*u/2).n.
    """
    if hasattr(velocity, "advection_boundary_matrix"):
        return velocity.advection_boundary_matrix(degree, order, faces=faces)
    count = len(nodal_space(fine, degree)[1])
    result = sparse.lil_matrix((count, count))
    gauss, weights = leggauss(order)
    for face in faces:
        start, end = macro.points[macro.faces[face]]
        for positions, ids in edge_pieces(macro, fine, face, degree):
            local = (gauss + 1) / 2
            parameter = positions[0] + local * (positions[1] - positions[0])
            points = start + parameter[:, None] * (end - start)
            basis = edge_basis(degree, local)
            normal_velocity = vector_values(velocity, points) @ macro.normals[face]
            measure = abs(positions[1] - positions[0]) * macro.lengths[face] / 2
            block = basis.T @ ((weights * normal_velocity * measure / 2)[:, None] * basis)
            result[np.ix_(ids, ids)] += block
    return result.tocsc()


def strong_boundary_dofs(
    macro: TriangleMesh,
    fine: TriangleMesh,
    faces: tuple[int, ...],
    degree: int,
    value: Any,
) -> tuple[IntArray, FloatArray]:
    """Collect unique local nodal Dirichlet DOFs and evaluate the boundary data."""
    pieces = [ids for face in faces for _, ids in edge_pieces(macro, fine, face, degree)]
    ids = np.unique(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
    points = nodal_space(fine, degree)[1]
    return ids, scalar_values(value, points[ids])
