"""Reuse one Helmholtz condensation on exactly nested polynomial trace spaces.

The local matrices, loads and harmonic responses remain owned by the supplied
solution. Only the global Schur matrix is restricted. Reconstructed fields keep
the original execution basis with zero coefficients in the omitted modes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.helmholtz import HelmholtzSolution
from pymhm.helmholtz_forms import complex_vector
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
    trace = np.zeros(lower.size)
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
    if residual > 1e-10:
        raise ValueError("restricted original Helmholtz trace equations fail")
    return lower, injection, free, trace, residual


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
    return RestrictedHelmholtz(solution, lower, complex_vector(trace), injection, free, residual)
