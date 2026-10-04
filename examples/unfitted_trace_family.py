"""Reuse scalar MHM responses on exactly nested discontinuous trace spaces.

This campaign helper retains the physical local constant modes. It supports
Dirichlet data on every exterior macroface; no boundary condition
is inferred from a restricted Robin multiplier. All local operators and loads
come from the scalar Darcy owner. Sparse coupling storage discards only zeros.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal import (
    DarcySolution,
    _assembly_quadrature_order,
    _DarcyLocalFactory,
)
from pymhm.core.contracts import HybridSolution, LocalResponse
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.triangle import TriangleMesh


def nested_trace_injection(fine: FaceSpace, coarse: FaceSpace) -> np.ndarray:
    """Embed coarse Legendre pieces exactly into a nested discontinuous space.

    Polynomial moments on each fine segment give its coefficients. The degree
    and geometric nesting are checked independently; no least-squares fitting
    of unresolved jumps is allowed.
    """
    if fine.continuous or coarse.continuous:
        raise ValueError("the scalar family requires discontinuous polynomial traces")
    if any(point not in fine.breaks for point in coarse.breaks):
        raise ValueError("every coarse breakpoint must occur in the prepared partition")
    matrix = np.zeros((fine.size, coarse.size))
    offset = 0
    for left, right, degree in zip(fine.breaks[:-1], fine.breaks[1:], fine.degrees, strict=True):
        midpoint = (left + right) / 2
        parent = np.searchsorted(coarse.breaks, midpoint, side="right") - 1
        if coarse.degrees[parent] > degree:
            raise ValueError("the requested degree exceeds the prepared local trace degree")
        points, weights = leggauss(degree + 1)
        parameter = left + (points + 1) * (right - left) / 2
        basis = fine.evaluate(parameter)[:, offset : offset + degree + 1]
        moments = basis.T @ (weights[:, None] * coarse.evaluate(parameter))
        matrix[offset : offset + degree + 1] = (
            (2 * np.arange(degree + 1) + 1)[:, None] / 2 * moments
        )
        offset += degree + 1
    return matrix


@dataclass(frozen=True)
class _Cell:
    """Original scalar equations and lifts, with sparse zero-free coupling storage."""

    mesh: TriangleMesh
    matrix: Any
    coupling: Any
    load: np.ndarray
    source: np.ndarray
    lifts: np.ndarray
    kernel: np.ndarray
    trace_dofs: np.ndarray

    @classmethod
    def from_response(cls, response: LocalResponse, mesh: TriangleMesh) -> _Cell:
        """Preserve corrected field coordinates for one declared constant kernel mode."""
        problem = response.problem
        if problem.kernel.shape[1] != 1:
            raise ValueError("the scalar family requires exactly one retained constant kernel")
        coupling = sparse.csr_matrix(problem.coupling)
        coupling.eliminate_zeros()
        return cls(
            mesh,
            problem.matrix,
            coupling,
            problem.load,
            response.source,
            np.asfortranarray(response.lifts),
            response.retained_basis,
            problem.trace_dofs,
        )


def _prepare_cell(
    factory: _DarcyLocalFactory, cell: int, trace_size: int, local_solver: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _Cell]:
    """Assemble and condense one local problem with one native algebra thread."""
    with threadpool_limits(1):
        assembly = factory(cell)
        response = assembly.problem.condense(solver=local_solver)
        indices, block, load = response.global_contribution(np.array([trace_size + cell]))
        compact = _Cell.from_response(response, assembly.metadata[0])
    return indices, block, load, compact


def _prepared_cells(
    factory: _DarcyLocalFactory, count: int, trace_size: int, workers: int, local_solver: str
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, _Cell]]:
    """Yield local responses in cell order with at most workers pending transfers."""
    if workers == 1:
        for cell in range(count):
            yield _prepare_cell(factory, cell, trace_size, local_solver)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        waiting = deque()
        submitted = 0
        while submitted < min(workers, count):
            waiting.append(
                executor.submit(_prepare_cell, factory, submitted, trace_size, local_solver)
            )
            submitted += 1
        while waiting:
            result = waiting.popleft().result()
            yield result
            if submitted < count:
                waiting.append(
                    executor.submit(_prepare_cell, factory, submitted, trace_size, local_solver)
                )
                submitted += 1


@dataclass(frozen=True)
class ScalarTraceFamily:
    """Prepared Dirichlet Darcy equations and their nested trace solves."""

    skeleton: SkeletonSpace
    cells: tuple[_Cell, ...]
    matrix: Any
    load: np.ndarray
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int

    @classmethod
    def prepare(
        cls,
        mesh: TriangleMesh,
        *,
        trace_degree: int,
        segments: int,
        local_degree: int,
        local_refinement: int,
        permeability: Any = 1.0,
        source: Any = 0.0,
        dirichlet: Any = 0.0,
        quadrature_order: int = 9,
        workers: int = 1,
        local_solver: str = "scipy",
    ) -> ScalarTraceFamily:
        """Condense unchanged local operators with deterministic cell-ordered collection.

        Multiple workers use spawn and require picklable coefficient/source
        callbacks. Each process uses one BLAS thread; at most workers responses
        await collection. The parent retains all original matrices and lifts.
        The explicitly selected local backend retains its original residual
        criterion. Global reduced systems use the standard SciPy backend.
        ``quadrature_order`` counts Gauss points per Duffy coordinate and boundary
        interval. Darcy's shared ``local_degree+2`` minimum applies, and the
        executed volume order is retained on this family and each reconstructed
        solution. Boundary data also apply the trace degree+2 floor; trace
        coupling retains its separate exact polynomial rule.
        """
        local_degree = positive_int(local_degree, "local_degree")
        local_refinement = positive_int(local_refinement, "local_refinement")
        workers = positive_int(workers, "workers")
        order = _assembly_quadrature_order(local_degree, quadrature_order)
        skeleton = SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(trace_degree, segments) for _ in mesh.faces)
        )
        factory = _DarcyLocalFactory(
            mesh,
            skeleton,
            permeability,
            source,
            tuple(np.empty((0, 3)) for _ in mesh.cells),
            local_refinement,
            None,
            local_degree,
            "primal",
            order,
        )
        size = skeleton.size + len(mesh.cells)
        rhs = np.zeros(size)
        boundary, _ = boundary_data(skeleton, dirichlet, None, order=order)
        rhs[: skeleton.size] -= boundary
        rows, columns, entries, cells = [], [], [], []
        for indices, block, load, compact in _prepared_cells(
            factory, len(mesh.cells), skeleton.size, workers, local_solver
        ):
            rows.append(np.repeat(indices, len(indices)))
            columns.append(np.tile(indices, len(indices)))
            entries.append(block.ravel())
            np.add.at(rhs, indices, load)
            cells.append(compact)
        matrix = sparse.coo_matrix(
            (np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size, size),
        ).tocsc()
        matrix.eliminate_zeros()
        return cls(skeleton, tuple(cells), matrix, rhs, local_degree, permeability, source, order)

    def solve(
        self, trace_degree: int, segments: int
    ) -> tuple[DarcySolution, dict[str, float | bool]]:
        """Restrict the full Schur system and reconstruct in the original local spaces.

        The block injection is diag(T,I): the retained constant coordinate of
        every macroelement remains unchanged. Acceptance checks the projected
        original Schur equations and all original local physical equations.
        """
        skeleton = SkeletonSpace(
            self.skeleton.mesh,
            tuple(FaceSpace.uniform(trace_degree, segments) for _ in self.skeleton.faces),
        )
        trace = sparse.block_diag(
            [
                sparse.csc_matrix(nested_trace_injection(high, low))
                for high, low in zip(self.skeleton.faces, skeleton.faces, strict=True)
            ],
            format="csc",
        )
        injection = sparse.block_diag((trace, sparse.eye(len(self.cells))), format="csc")
        rhs = np.asarray(injection.T @ self.load).ravel()
        matrix = (injection.T @ self.matrix @ injection).tocsc()
        unknowns = solve_linear(matrix, rhs)
        lifted = np.asarray(injection @ unknowns).ravel()
        defect = np.asarray(injection.T @ (self.matrix @ lifted - self.load)).ravel()
        denominator = float(np.linalg.norm(rhs))
        residual = (
            float(np.linalg.norm(defect)) / denominator
            if denominator
            else float(np.linalg.norm(defect))
        )
        if residual > 1e-10:
            raise ValueError("restricted original scalar equations fail the physical residual gate")
        pressure, flux, modes, local_residuals = [], [], [], []
        local_defects, backward_errors, roundoff_bounds, roundoff_accepted = [], [], [], []
        for index, cell in enumerate(self.cells):
            multiplier = lifted[cell.trace_dofs]
            mode = unknowns[skeleton.size + index : skeleton.size + index + 1]
            field = cell.source - cell.lifts @ multiplier + cell.kernel @ mode
            action, boundary = cell.matrix @ field, cell.coupling @ multiplier
            scale = np.linalg.norm(action) + np.linalg.norm(boundary) + np.linalg.norm(cell.load)
            defect = action + boundary - cell.load
            local_defect = np.linalg.norm(defect)
            local_residual = float(local_defect / scale) if scale else float(local_defect)
            # Higham's componentwise dot-product envelope uses the actual row
            # lengths, including two final vector additions. It permits only
            # arithmetic cancellation, including a represented constant mode.
            terms = (
                np.bincount(cell.matrix.indices, minlength=len(field))
                + np.diff(cell.coupling.indptr)
                + 2
            )
            units = terms * np.finfo(field.dtype).eps
            component_scale = (
                abs(cell.matrix) @ abs(field)
                + abs(cell.coupling) @ abs(multiplier)
                + abs(cell.load)
            )
            envelope = units / (1 - units) * component_scale
            resolved_defect = np.linalg.norm(np.maximum(abs(defect) - envelope, 0))
            adjusted = float(resolved_defect / scale) if scale else float(resolved_defect)
            if adjusted > 1e-10:
                raise ValueError("original local scalar equations fail the physical residual gate")
            local_residuals.append(local_residual)
            local_defects.append(float(local_defect))
            component_norm = np.linalg.norm(component_scale)
            backward_errors.append(float(local_defect / component_norm) if component_norm else 0.0)
            roundoff_bounds.append(float(np.linalg.norm(envelope)))
            roundoff_accepted.append(local_residual > 1e-10)
            pressure.append(field)
            modes.append(mode)
            dofs, _, _, gradient, _ = tabulate(cell.mesh, self.degree, np.full((1, 3), 1 / 3))
            grad = np.einsum("ti,tia->ta", field[dofs], gradient[:, 0])
            material = tensor_values(
                self.permeability, cell.mesh.points[cell.mesh.cells].mean(axis=1)
            )
            flux.append(-np.einsum("tab,tb->ta", material, grad))
        hybrid = HybridSolution(
            unknowns[: skeleton.size], tuple(modes), tuple(pressure), residual, np.empty(0)
        )
        solution = DarcySolution(
            skeleton,
            tuple(c.mesh for c in self.cells),
            tuple(pressure),
            tuple(flux),
            hybrid,
            "primal",
            self.permeability,
            self.source,
            self.quadrature_order,
            self.degree,
        )
        return solution, {
            "original_trace_residual": residual,
            "original_local_residual_max": max(local_residuals),
            "original_local_defect_l2_max": max(local_defects),
            "local_componentwise_backward_error_max": max(backward_errors),
            "local_roundoff_envelope_l2_max": max(roundoff_bounds),
            "local_accepted_with_roundoff_envelope": any(roundoff_accepted),
        }
