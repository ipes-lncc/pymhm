"""Reuse L11 RAD local responses in exactly nested discontinuous trace spaces.

The local owner assembles every operator, load, strong Dirichlet equation and
retained constant. The nonsymmetric Schur system is restricted by ``diag(T,I)``
on both sides. Horizontal walls carry homogeneous diffusive flux. Source and
strong Dirichlet data may be nonconstant and nonhomogeneous, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import sparse

from examples.formulations.transport import rad_local_assembly as _rad_local
from examples.unfitted_trace_family import nested_trace_injection
from pymhm.core.contracts import HybridSolution
from pymhm.core.validation import positive_int
from pymhm.execution.cpu import map_local
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import ScalarSolution


def gradient_projection_squared(mesh: TriangleMesh, gradient: Any, order: int) -> float:
    """Integrate distance from an exact gradient to its fine-cell DG0 projection.

    Every affine P1 field has a constant gradient on each fine triangle. This
    is therefore a lower bound for its broken H1 error, independent of trace
    restriction. The DG0 projection need not itself be a gradient of an
    admissible pressure. Numerical integration must separately resolve the
    supplied exact gradient; the returned number is not a certified bound.
    """
    bary, weights = triangle_quadrature(order)
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    values = vector_values(gradient, points.reshape(-1, 2)).reshape(len(mesh.cells), -1, 2)
    mean = np.einsum("q,tqa->ta", weights, values) / np.sum(weights)
    difference = values - mean[:, None, :]
    return float(mesh.areas @ (np.sum(difference**2, axis=2) @ weights))


@dataclass(frozen=True)
class _Cell:
    """Original RAD equations and lifts with only structural coupling zeros removed."""

    mesh: TriangleMesh
    matrix: Any
    coupling: Any
    load: np.ndarray
    source: np.ndarray
    lifts: np.ndarray
    retained: np.ndarray
    trace_dofs: np.ndarray
    physical_size: int
    block: np.ndarray
    rhs: np.ndarray


def _prepare_cell(index: int, *, factory: Any) -> _Cell:
    """Condense one unchanged shared-owner local problem inside its worker."""
    assembly = factory(index)
    response = assembly.problem.condense()
    problem = response.problem
    count = problem.coarse_basis.shape[1]
    _, block, rhs = response.global_contribution(np.arange(count, dtype=np.int64))
    coupling = sparse.csr_matrix(problem.coupling)
    coupling.eliminate_zeros()
    return _Cell(
        assembly.metadata[0],
        problem.matrix,
        coupling,
        problem.load,
        response.source,
        response.lifts,
        response.retained_basis,
        problem.trace_dofs,
        assembly.metadata[4],
        block,
        rhs,
    )


@dataclass(frozen=True)
class TransportTraceFamily:
    """Fixed P1 RAD local spaces and their nested P0 trace restrictions."""

    skeleton: SkeletonSpace
    cells: tuple[_Cell, ...]
    matrix: Any
    rhs: np.ndarray
    offsets: np.ndarray
    natural_faces: tuple[int, ...]

    @classmethod
    def prepare(
        cls,
        mesh: TriangleMesh,
        *,
        segments: int,
        local_refinement: int,
        epsilon: float = 1.0,
        source: Any = 1.0,
        dirichlet: Any = 0.0,
        workers: int = 1,
    ) -> TransportTraceFamily:
        """Assemble the L11 operator once, retaining constants exactly as in ``solve_rad``.

        This example family fixes velocity=(1,0), zero reaction, P1/Galerkin and
        horizontal homogeneous diffusive walls. Every other exterior face has
        strong Dirichlet data. Nonzero prescribed wall flux is outside this
        helper's contract; it is not silently projected by trace restriction.
        """
        segments = positive_int(segments, "segments")
        local_refinement = positive_int(local_refinement, "local_refinement")
        workers = positive_int(workers, "workers")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
        natural = tuple(
            int(face) for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) < 1e-14
        )
        strong = tuple(int(f) for f in mesh.boundary_faces if f not in natural)
        factory = partial(
            _rad_local,
            mesh=mesh,
            skeleton=skeleton,
            degree=1,
            refinement=local_refinement,
            diffusion=epsilon,
            diffusion_divergence=(0.0, 0.0),
            velocity=(1.0, 0.0),
            velocity_divergence=0.0,
            reaction=0.0,
            source=source,
            stabilization="galerkin",
            order=5,
            strong_faces=strong,
            dirichlet=dirichlet,
            diffusive_faces=natural,
            coarse_space="constants",
        )
        cells = tuple(
            map_local(
                partial(_prepare_cell, factory=factory),
                range(len(mesh.cells)),
                backend="process" if workers > 1 else "serial",
                workers=workers,
            )
        )
        offsets = np.r_[0, np.cumsum([cell.retained.shape[1] for cell in cells])]
        size = skeleton.size + int(offsets[-1])
        rows, columns, entries = [], [], []
        rhs = np.zeros(size)
        for index, cell in enumerate(cells):
            coarse = skeleton.size + np.arange(offsets[index], offsets[index + 1])
            indices = np.r_[cell.trace_dofs, coarse].astype(np.int64)
            rows.append(np.repeat(indices, len(indices)))
            columns.append(np.tile(indices, len(indices)))
            entries.append(cell.block.ravel())
            np.add.at(rhs, indices, cell.rhs)
        matrix = sparse.coo_matrix(
            (np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsc()
        matrix.eliminate_zeros()
        return cls(skeleton, cells, matrix, rhs, offsets, natural)

    def solve(self, skeleton: SkeletonSpace) -> tuple[ScalarSolution, dict[str, float]]:
        """Solve ``J.T S J`` and recover fields using original source and trace lifts.

        ``J=diag(T,I)`` preserves every retained coordinate, including cells
        with no retained modes. All prescribed exterior trace coordinates are
        zero, including those removed by strong Dirichlet enforcement. Gates
        check projected original Schur equations and original local equations.
        """
        if skeleton.mesh is not self.skeleton.mesh or skeleton.components != 1:
            raise ValueError("the restricted scalar skeleton must use the prepared mesh")
        trace = sparse.block_diag(
            [
                sparse.csc_matrix(nested_trace_injection(high, low))
                for high, low in zip(self.skeleton.faces, skeleton.faces, strict=True)
            ],
            format="csc",
        )
        injection = sparse.block_diag((trace, sparse.eye(int(self.offsets[-1]))), format="csc")
        matrix = (injection.T @ self.matrix @ injection).tocsc()
        rhs = np.asarray(injection.T @ self.rhs).ravel()
        free = np.ones(len(rhs), dtype=bool)
        for face in skeleton.mesh.boundary_faces:
            free[skeleton.dofs(int(face))] = False
        unknowns = np.zeros(len(rhs))
        unknowns[free] = solve_linear(matrix[free][:, free], rhs[free])
        lifted = np.asarray(injection @ unknowns).ravel()
        defect = np.asarray(injection.T @ (self.matrix @ lifted - self.rhs)).ravel()[free]
        denominator = max(float(np.linalg.norm(rhs[free])), np.finfo(float).tiny)
        residual = float(np.linalg.norm(defect)) / denominator
        if residual > 1e-10:
            raise ValueError("restricted original transport equations fail their residual gate")
        values, full_fields, modes, local_residuals = [], [], [], []
        for index, cell in enumerate(self.cells):
            interval = slice(
                skeleton.size + self.offsets[index], skeleton.size + self.offsets[index + 1]
            )
            coarse = unknowns[interval].copy()
            multiplier = lifted[cell.trace_dofs]
            field = cell.source - cell.lifts @ multiplier + cell.retained @ coarse
            defect = cell.matrix @ field + cell.coupling @ multiplier - cell.load
            action = abs(cell.matrix) @ abs(field) + abs(cell.coupling) @ abs(multiplier)
            scale = max(
                float(np.linalg.norm(action)),
                float(np.linalg.norm(cell.load)),
                np.finfo(float).tiny,
            )
            local_residual = float(np.linalg.norm(defect)) / scale
            if local_residual > 1e-10:
                raise ValueError("original local transport equations fail their residual gate")
            local_residuals.append(local_residual)
            modes.append(coarse)
            full_fields.append(field)
            values.append(field[: cell.physical_size])
        hybrid = HybridSolution(
            unknowns[: skeleton.size], tuple(modes), tuple(full_fields), residual, np.empty(0)
        )
        solution = ScalarSolution(
            skeleton,
            tuple(c.mesh for c in self.cells),
            tuple(values),
            hybrid,
            1,
            True,
            self.natural_faces,
        )
        return solution, dict(
            projected_original_residual=residual,
            maximum_original_local_residual=max(local_residuals),
        )
