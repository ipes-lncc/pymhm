"""Recursive local MHM through a second, explicit trace constraint.

The inner hybrid unknowns and their boundary reactions form the local field of
the next level. Inner volume fields are reconstructed only after the outer solve.
No inverse, penalty, or implicit boundary gauge is introduced.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.contracts import LocalProblem, _array
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace


def nested_trace_map(
    outer: SkeletonSpace, cell: int, inner: SkeletonSpace
) -> tuple[IntArray, FloatArray]:
    """Restrict oriented parent face polynomials exactly to inner boundary faces.

    The inner mesh must partition the selected parent cell. Its boundary edges
    must lie on individual straight parent edges, and every restricted outer
    basis function must belong to the inner face space. Extra material cuts or
    smaller inner degrees cannot silently project an unrepresentable trace.
    Endpoint projections agree with parent breakpoints within the declared
    coordinate-roundoff envelope. Corrections smaller than half the child edge
    retain its ordering; genuine interior material cuts remain unrepresentable.
    Components retain the standard component-interleaved ordering.
    """
    if outer.components != inner.components:
        raise ValueError("inner and outer trace components must agree")
    outer_dofs = outer.cell_dofs(cell)
    boundary = np.concatenate([inner.dofs(int(f)) for f in inner.mesh.boundary_faces])
    transform = np.zeros((len(boundary), len(outer_dofs)))
    row = 0
    outer_faces = outer.mesh.cell_faces[cell]
    offsets = np.r_[0, np.cumsum([outer.faces[f].size for f in outer_faces])]
    for face in inner.mesh.boundary_faces:
        endpoints = inner.mesh.points[inner.mesh.faces[face]]
        matches = []
        for side, parent in enumerate(outer_faces):
            start, end = outer.mesh.points[outer.mesh.faces[parent]]
            tangent = end - start
            length = outer.mesh.lengths[parent]
            parameter = (endpoints - start) @ tangent / (tangent @ tangent)
            defect = endpoints - start - parameter[:, None] * tangent
            tolerance = 128 * np.finfo(float).eps
            if (
                np.max(np.abs(defect)) <= tolerance * max(length, np.max(abs(endpoints)))
                and parameter.min() >= -tolerance
                and parameter.max() <= 1 + tolerance
            ):
                # Recognize the same partition vertex after coordinate evaluation.
                # A geometric correction cannot collapse a nonzero child edge.
                breaks = np.asarray(outer.faces[parent].breaks)
                nearest = breaks[np.argmin(abs(parameter[:, None] - breaks), axis=1)]
                envelope = (
                    16
                    * np.finfo(float).eps
                    * (abs(endpoints) + abs(start) + abs(end))
                    @ abs(tangent)
                    / (tangent @ tangent)
                    + tolerance
                )
                correction = abs(parameter - nearest)
                snap = (correction <= envelope) & (correction < abs(np.diff(parameter))[0] / 2)
                parameter = np.where(snap, nearest, parameter)
                matches.append((side, parent, np.clip(parameter, 0, 1)))
        if len(matches) != 1:
            raise ValueError("each inner boundary face must belong to exactly one parent edge")
        side, parent, parameter = matches[0]
        a, b = parameter
        child, coarse = inner.faces[face], outer.faces[parent]
        cuts = sorted(
            {
                *child.breaks,
                *((s - a) / (b - a) for s in coarse.breaks if min(a, b) < s < max(a, b)),
            }
        )
        order = max(*child.degrees, *coarse.degrees) + 2
        points, weights = FaceSpace(tuple(cuts), (0,) * (len(cuts) - 1)).quadrature(order)
        fine_values = child.evaluate(points)
        coarse_values = coarse.evaluate(a + (b - a) * points)
        mapping = np.linalg.solve(
            fine_values.T @ (weights[:, None] * fine_values),
            fine_values.T @ (weights[:, None] * coarse_values),
        )
        if np.max(np.abs(fine_values @ mapping - coarse_values)) > 2e-11:
            raise ValueError("inner face space cannot represent the restricted parent trace")
        sign = float(inner.mesh.normals[face] @ outer.mesh.normals[parent])
        block = np.kron(sign * mapping, np.eye(inner.components))
        columns = slice(offsets[side] * outer.components, offsets[side + 1] * outer.components)
        transform[row : row + len(block), columns] = block
        row += len(block)
    return boundary, transform


@dataclass(frozen=True)
class NestedSolution:
    """Recovered inner hybrid coordinates, leaf fields and boundary reactions."""

    unknowns: FloatArray
    fields: tuple[FloatArray, ...]
    boundary_reactions: FloatArray
    interior_residual: float
    raw_residual: float | None = None
    raw_residual_norm: float | None = None


@dataclass(frozen=True)
class NestedLocalProblem:
    """One local problem whose interior discretization is another MHM system.

    Use ``problem`` in the parent :class:`HybridSystem` and pass the resulting
    local field to :meth:`reconstruct`. The trace map sends outer coefficients to
    selected inner trace coordinates; it includes every orientation and basis
    transformation. A kernel is declared in inner hybrid coordinates and lifted
    to the boundary reactions, so retained constants or rigid modes survive each
    condensation level.
    """

    inner: HybridSystem
    problem: LocalProblem
    boundary_dofs: IntArray
    trace_map: FloatArray

    def reconstruct(self, field: Any) -> NestedSolution:
        """Recover all inner local fields and check unconstrained inner equations."""
        n = len(self.inner.rhs)
        field = _array(field, (n + len(self.boundary_dofs),), "nested field")
        coordinates = field[:n]
        values = tuple(
            response.reconstruct(
                coordinates[response.problem.trace_dofs],
                coordinates[self.inner.kernel_offsets[i] : self.inner.kernel_offsets[i + 1]],
            )
            for i, response in enumerate(self.inner.responses)
        )
        free = np.setdiff1d(np.arange(n), self.boundary_dofs)
        residual = self.inner.matrix @ coordinates - self.inner.rhs
        free_matrix = self.inner.matrix[free][:, free]
        prescribed_action = (
            self.inner.matrix[free][:, self.boundary_dofs] @ coordinates[self.boundary_dofs]
        )
        scale = max(
            float(np.linalg.norm(self.inner.rhs[free])),
            float(np.linalg.norm(self.inner.load_scale[free])),
            float(np.linalg.norm(prescribed_action)),
            float(np.linalg.norm(abs(free_matrix) @ abs(coordinates[free]))),
            np.finfo(float).tiny,
        )
        absolute_residual = float(np.linalg.norm(residual[free]))
        relative_residual = absolute_residual / scale
        raw_residual = relative_residual
        if relative_residual > 1e-8:
            bounds = np.zeros(n)
            for cell, (response, value) in enumerate(
                zip(self.inner.responses, values, strict=True)
            ):
                first, last = self.inner.kernel_offsets[cell : cell + 2]
                bounds[first:last] = response.kernel_roundoff_bound(value)
            certified = np.maximum(abs(residual[free]) - bounds[free], 0)
            relative_residual = float(np.linalg.norm(certified)) / scale
        return NestedSolution(
            coordinates.copy(),
            values,
            field[n:].copy(),
            relative_residual,
            raw_residual,
            absolute_residual,
        )

    def moment(
        self, local_weights: list[FloatArray] | tuple[FloatArray, ...]
    ) -> tuple[FloatArray, float]:
        """Express a physical leaf moment as ``weights @ nested_field + offset``.

        The offset is essential when the inner source responses have nonzero
        moments. Subtract it from a parent physical gauge's prescribed value.
        """
        row, target = self.inner.mean_constraint(local_weights)
        return np.r_[row, np.zeros(len(self.boundary_dofs))], -target


def nest_hybrid_system(
    inner: HybridSystem,
    boundary_dofs: Any,
    trace_map: Any,
    trace_dofs: Any,
    *,
    kernel: Any = None,
    left_kernel: Any = None,
    constraints: Any = None,
    test_constraints: Any = None,
) -> NestedLocalProblem:
    """Promote an inner hybrid discretization to a parent local problem.

    With ``A x=b`` the inner equations, ``E`` selecting its boundary trace and
    ``M=trace_map``, the local equations are
    ``[[-A,E],[E.T,0]] [x,mu] + [0,-M] lambda = [-b,0]``.
    Thus ``E.T x=M lambda`` and ``mu`` is minus the inner potential moment.
    The parent continuity pairing ``-M.T mu`` has the original physical sign.

    ``kernel`` and ``left_kernel`` are arrays with one row per inner hybrid
    unknown, and the same number of independent columns. Their reaction parts
    are calculated from A and A.T. They must vanish in the selected trace
    coordinates and satisfy the interior homogeneous equations. Constraints
    have the same inner shape and are extended by zero boundary reactions.
    The caller supplies the physical constant/rigid modes; numerical nullspace
    detection does not replace this formulation contract. Distinct left and
    right kernels are supported when the inner method is Petrov--Galerkin.
    """
    n = len(inner.rhs)
    selected = np.asarray(boundary_dofs)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or not len(selected)
        or len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= inner.trace_size)
    ):
        raise ValueError("boundary_dofs must be distinct valid inner trace indices")
    selected = selected.astype(np.int64, copy=True)
    dofs = np.asarray(trace_dofs)
    if dofs.ndim != 1:
        raise ValueError("trace_dofs must be a vector")
    transform = _array(trace_map, (len(selected), len(dofs)), "trace_map")
    if not len(dofs) or np.linalg.matrix_rank(transform) != len(dofs):
        raise ValueError("trace_map must be injective on the parent trace space")
    extension = sparse.csc_matrix(
        (np.ones(len(selected)), (selected, np.arange(len(selected)))), shape=(n, len(selected))
    )
    operator = sparse.bmat([[-inner.matrix, extension], [extension.T, None]], format="csc")
    basis = np.empty((n, 0)) if kernel is None else np.asarray(kernel)
    if basis.ndim != 2:
        raise ValueError("kernel must be a matrix of inner hybrid coordinates")
    basis = _array(basis, (n, basis.shape[1]), "kernel")
    left = basis if left_kernel is None else _array(left_kernel, basis.shape, "left_kernel")
    right_lift = np.vstack((basis, (inner.matrix @ basis)[selected]))
    left_lift = np.vstack((left, (inner.matrix.T @ left)[selected]))
    right_moments = (
        basis if constraints is None else _array(constraints, basis.shape, "constraints")
    )
    left_moments = (
        right_moments
        if test_constraints is None and np.array_equal(left, basis)
        else left
        if test_constraints is None
        else _array(test_constraints, basis.shape, "test_constraints")
    )
    zeros = np.zeros((len(selected), basis.shape[1]))
    problem = LocalProblem(
        operator,
        np.vstack((np.zeros((n, len(dofs))), -transform)),
        np.r_[-inner.rhs, np.zeros(len(selected))],
        dofs,
        right_lift,
        np.vstack((right_moments, zeros)),
        left_kernel=left_lift,
        test_constraints=np.vstack((left_moments, zeros)),
    )
    return NestedLocalProblem(inner, problem, selected, transform)
