"""Reuse one Helmholtz condensation on exactly nested polynomial trace spaces.

The local matrices, loads and harmonic responses remain owned by the supplied
solution. Only the global Schur matrix is restricted. Reconstructed fields keep
the original execution basis with zero coefficients in the omitted modes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.helmholtz import HelmholtzSolution
from pymhm.helmholtz_forms import complex_vector, real_vector
from pymhm.hybrid import HybridSolution
from pymhm.mesh import FaceSpace, SkeletonSpace, positive_int
from pymhm.solvers import solve_linear


@dataclass(frozen=True)
class RestrictedHelmholtz:
    """Fields in the prepared basis and effective lower-degree trace coordinates."""

    solution: HelmholtzSolution
    skeleton: SkeletonSpace
    trace: np.ndarray
    injection: Any
    free_dofs: np.ndarray
    restricted_residual: float
    original_trace_residual: float


def polynomial_injection(
    prepared: SkeletonSpace, degree: int
) -> tuple[SkeletonSpace, Any, np.ndarray]:
    """Select nested Legendre modes segmentwise, retaining interleaved Re/Im."""
    degree = positive_int(degree, "degree", 0)
    if prepared.components != 2 or any(
        type(face) is not FaceSpace or face.continuous or min(face.degrees) < degree
        for face in prepared.faces
    ):
        raise ValueError("nested Helmholtz restriction requires discontinuous polynomial traces")
    lower = SkeletonSpace(
        prepared.mesh,
        tuple(FaceSpace(face.breaks, (degree,) * len(face.degrees)) for face in prepared.faces),
        components=2,
    )
    selection: list[int] = []
    for face, data in enumerate(prepared.faces):
        offset = 0
        old = prepared.dofs(face)
        for old_degree in data.degrees:
            selection.extend(old[2 * offset : 2 * (offset + degree + 1)])
            offset += old_degree + 1
    selected = np.asarray(selection, dtype=np.int64)
    injection = sparse.csc_matrix(
        (np.ones(len(selected)), (selected, np.arange(len(selected)))),
        shape=(prepared.size, lower.size),
    )
    return lower, injection, selected


def solve_restricted_coordinates(
    skeleton: SkeletonSpace,
    matrix: Any,
    rhs: np.ndarray,
    prescribed: dict[int, float],
    degree: int,
    *,
    solver: str = "scipy",
) -> tuple[SkeletonSpace, Any, np.ndarray, np.ndarray, float]:
    """Restrict and solve the original trace equations with unchanged boundary data."""
    lower, injection, selected = polynomial_injection(skeleton, degree)
    rhs = np.asarray(injection.T @ rhs).ravel()
    matrix = (injection.T @ matrix @ injection).tocsc()
    inverse = np.full(skeleton.size, -1, dtype=np.int64)
    inverse[selected] = np.arange(lower.size)
    trace = np.zeros(lower.size, dtype=np.result_type(matrix.dtype, rhs.dtype))
    fixed = []
    for old, value in prescribed.items():
        new = inverse[old]
        if new < 0:
            if value != 0:
                raise ValueError("prescribed Neumann trace is not in the restricted space")
        else:
            trace[new] = value
            fixed.append(new)
    free = np.setdiff1d(np.arange(lower.size), fixed)
    forcing = (rhs - matrix @ trace)[free]
    if len(free):
        trace[free] = solve_linear(matrix[free][:, free], forcing, solver=solver)
    defect = (matrix @ trace - rhs)[free]
    scale = max(
        float(np.linalg.norm(rhs[free])),
        float(np.linalg.norm((abs(matrix) @ abs(trace))[free])),
        np.finfo(float).tiny,
    )
    residual = float(np.linalg.norm(defect) / scale)
    if not np.isfinite(trace).all() or not np.isfinite(residual) or residual > 1e-10:
        raise ValueError("restricted original Helmholtz trace equations fail")
    return lower, injection, free, trace, residual


def verify_helmholtz_trace_fields(
    skeleton: SkeletonSpace,
    degree: int | None,
    free_dofs: np.ndarray,
    fields: Iterable[tuple[Any, np.ndarray, np.ndarray, np.ndarray]],
    *,
    dtype: Any = float,
) -> float:
    """Check original weak continuity from streamed executed nodal coefficients.

    Each record contains the original real coupling B, global trace injection
    IDs, Dirichlet functional g and real interleaved local field u. The check
    tests sum(B.T u)-g in the requested Legendre subspace on free coordinates;
    prescribed Neumann coordinates are excluded. Its scale accumulates absolute
    physical actions before cancellation. No Schur blocks or lifts enter it.
    ``degree=None`` tests the entire executed trace basis, including oscillatory
    spans, without changing or recomputing their archived coordinates.
    """
    action = np.zeros(skeleton.size, dtype=dtype)
    scale = np.zeros_like(action)
    count = 0
    for coupling, dofs, boundary, values in fields:
        count += 1
        values = np.asarray(values)
        if (
            values.shape != (coupling.shape[0],)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("original trace check requires finite real interleaved nodal fields")
        current = np.result_type(action.dtype, values.dtype, boundary.dtype)
        if current != action.dtype:
            action, scale = action.astype(current), scale.astype(current)
        np.add.at(action, dofs, coupling.T @ values - boundary)
        np.add.at(scale, dofs, abs(coupling).T @ abs(values) + abs(boundary))
    if count != len(skeleton.mesh.cells):
        raise ValueError("original trace check requires every macrocell field exactly once")
    if degree is None:
        injection = sparse.eye(skeleton.size, format="csc")
    else:
        _, injection, _ = polynomial_injection(skeleton, degree)
    defect = np.asarray(injection.T @ action).ravel()[free_dofs]
    original_scale = np.asarray(injection.T @ scale).ravel()[free_dofs]
    residual = float(
        np.linalg.norm(defect) / max(np.linalg.norm(original_scale), np.finfo(float).tiny)
    )
    if not np.isfinite(residual) or residual > 1e-10:
        raise ValueError("reconstructed fields fail original Helmholtz trace equations")
    return residual


def verify_helmholtz_local_field(
    matrix: Any, coupling: Any, load: np.ndarray, values: np.ndarray, local_trace: np.ndarray
) -> tuple[complex, float]:
    """Check A u+B trace=f using executed real interleaved physical coefficients.

    The returned complex sum tests the macrocell constant equation, while the
    relative defect checks every original local equation. Their scale uses
    absolute matrix action, boundary action and original load, without dividing
    by a cancelled signed sum. These are distinct from fine-cell conservation.
    """
    values, local_trace, load = map(np.asarray, (values, local_trace, load))
    if (
        values.shape != (matrix.shape[0],)
        or load.shape != values.shape
        or local_trace.shape != (coupling.shape[1],)
        or any(
            np.iscomplexobj(value) or not np.isfinite(value).all()
            for value in (values, local_trace, load)
        )
    ):
        raise ValueError("local field check requires finite real interleaved coefficients")
    boundary_action = coupling @ local_trace
    defect = matrix @ values + boundary_action - load
    scale = max(
        np.linalg.norm(abs(matrix) @ abs(values)),
        np.linalg.norm(boundary_action),
        np.linalg.norm(load),
        np.finfo(float).tiny,
    )
    residual = float(np.linalg.norm(defect) / scale)
    if not np.isfinite(values).all() or not np.isfinite(residual) or residual > 1e-10:
        raise ValueError("reconstructed field fails original Helmholtz local equations")
    return complex(np.sum(complex_vector(defect))), residual


def verify_helmholtz_solution(
    solution: HelmholtzSolution, *, degree: int | None = None, free_dofs: np.ndarray | None = None
) -> dict[str, float]:
    """Verify actual physical fields against original local and boundary equations.

    With no degree restriction, the original executed trace basis is tested.
    A restricted family supplies its lower polynomial degree and free trace
    coordinates. Prescribed Neumann/absorbing coordinates are excluded; original
    Dirichlet functionals remain in the weak continuity check. The diagnostic
    consumes the solution's physical pressure arrays and never rebuilds lifts.
    """
    system = solution.system
    fixed = {}
    for response, metadata in zip(system.responses, system.local_metadata, strict=True):
        for scalar in metadata[3]:
            fixed[int(response.problem.trace_dofs[2 * scalar])] = 0
            fixed[int(response.problem.trace_dofs[2 * scalar + 1])] = 0
    if free_dofs is None:
        if degree is not None:
            raise ValueError(
                "restricted physical field checks require their free trace coordinates"
            )
        free_dofs = np.setdiff1d(np.arange(solution.skeleton.size), list(fixed))
    trace = real_vector(solution.trace)
    residuals = []
    for response, pressure in zip(system.responses, solution.pressure, strict=True):
        problem = response.problem
        _, residual = verify_helmholtz_local_field(
            problem.matrix,
            problem.coupling,
            problem.load,
            real_vector(pressure),
            trace[problem.trace_dofs],
        )
        residuals.append(residual)
    original = verify_helmholtz_trace_fields(
        solution.skeleton,
        degree,
        free_dofs,
        (
            (
                response.problem.coupling,
                response.problem.trace_dofs,
                metadata[2],
                real_vector(pressure),
            )
            for response, metadata, pressure in zip(
                system.responses, system.local_metadata, solution.pressure, strict=True
            )
        ),
        dtype=np.result_type(system.matrix.dtype, system.rhs.dtype),
    )
    return {
        "original_field_trace_residual": original,
        "original_local_equation_residual_max": max(residuals),
    }


def restrict_helmholtz_trace(
    prepared: HelmholtzSolution, degree: int, *, solver: str = "scipy"
) -> RestrictedHelmholtz:
    """Solve T.T*S*T and recover from existing lifts without copying local CSCs.

    The local operator and quadrature must be identical across the family.
    Dirichlet functionals restrict by T.T, while prescribed physical Neumann
    traces must belong exactly to the smaller space. Absorbing-face multiplier
    coordinates remain zero. The diagnostic tests the original coarse trace
    equations, not the unrequested enriched trace equations.
    """
    system = prepared.system
    if any(response.problem.coarse_basis.shape[1] for response in system.responses):
        raise ValueError("this Helmholtz family requires no retained local modes")
    high_fixed: dict[int, float] = {}
    for response, metadata in zip(system.responses, system.local_metadata, strict=True):
        prescribed = metadata[3]
        ids = response.problem.trace_dofs
        for scalar, value in prescribed.items():
            high_fixed[int(ids[2 * scalar])] = float(value.real)
            high_fixed[int(ids[2 * scalar + 1])] = float(value.imag)
    lower, injection, free, trace, residual = solve_restricted_coordinates(
        prepared.skeleton, system.matrix, system.rhs, high_fixed, degree, solver=solver
    )
    lifted_trace = np.asarray(injection @ trace).ravel()
    coarse = tuple(np.empty(0) for _ in system.responses)
    fields = tuple(
        response.reconstruct(lifted_trace[response.problem.trace_dofs], empty)
        for response, empty in zip(system.responses, coarse, strict=True)
    )
    hybrid = HybridSolution(lifted_trace, coarse, fields, residual, np.empty(0))
    solution = replace(
        prepared,
        pressure=tuple(complex_vector(values) for values in fields),
        trace=complex_vector(lifted_trace),
        hybrid=hybrid,
    )
    diagnostics = verify_helmholtz_solution(solution, degree=degree, free_dofs=free)
    return RestrictedHelmholtz(
        solution,
        lower,
        complex_vector(trace),
        injection,
        free,
        residual,
        diagnostics["original_field_trace_residual"],
    )
