"""Dimension-independent physical Neumann extension of the Robin MH method."""

from dataclasses import dataclass
from math import fsum
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.hybrid import HybridSolution, HybridSystem
from pymhm.mesh import FloatArray, IntArray
from pymhm.solvers import solve_linear


@dataclass(frozen=True)
class RobinBoundaryFace:
    """Integrated boundary data in one scalar Robin multiplier basis.

    The mass matrix pairs multiplier densities and auxiliary pressure traces.
    Coefficient is sigma.n with the outward exterior normal. Load integrates
    the prescribed physical flux q.n; constant represents one in this basis.
    """

    face: int
    dofs: IntArray
    mass: FloatArray
    coefficient: float
    load: FloatArray
    constant: FloatArray


def solve_robin_neumann(
    system: HybridSystem,
    trace_size: int,
    faces: tuple[RobinBoundaryFace, ...],
    volume_weights: tuple[FloatArray, ...],
    volume: float,
    mean_pressure: float,
    pure_neumann: bool,
    solver: str,
    precision: Literal["double", "extended"],
) -> tuple[HybridSolution, dict[int, FloatArray], Any, FloatArray]:
    """Solve symmetric boundary-pressure equations, checking physical compatibility.

    Rows are S lambda + J rho=t-g_D and J.T lambda+R rho=h_N.
    Local coercive inverses are unchanged. The optional mean fixes the full
    reconstructed volume pressure, not an average of trace coefficients.
    """
    width = sum(len(data.dofs) for data in faces)
    coupling = sparse.lil_matrix((trace_size, width))
    robin, loads = [], []
    offset = 0
    for data in faces:
        count = len(data.dofs)
        coupling[np.ix_(data.dofs, np.arange(offset, offset + count))] = data.mass
        robin.append(sparse.csc_matrix(data.coefficient * data.mass))
        loads.append(data.load)
        offset += count
    matrix = sparse.bmat(
        [[system.matrix, coupling.tocsc()], [coupling.T.tocsc(), sparse.block_diag(robin)]],
        format="csc",
    )
    rhs = np.r_[system.rhs, np.concatenate(loads)]
    physical_matrix, physical_rhs = matrix, rhs
    if pure_neumann:
        source_terms = np.concatenate([response.problem.load for response in system.responses])
        boundary_terms = np.concatenate([data.constant * data.load for data in faces])
        source_total, boundary_total = fsum(source_terms), fsum(boundary_terms)
        magnitude = fsum(abs(source_terms)) + fsum(abs(boundary_terms))
        units = (len(source_terms) + len(boundary_terms)) * np.finfo(float).eps
        rounding = units / (1 - units) * magnitude
        if abs(source_total - boundary_total) > (
            1e-10 * max(abs(source_total), abs(boundary_total), np.finfo(float).tiny) + rounding
        ):
            raise ValueError("incompatible Neumann data: source and outward flux integrals differ")
        row, target = system.mean_constraint(volume_weights, mean_pressure * volume)
        row = np.r_[row, np.zeros(width)]
        matrix = sparse.bmat([[matrix, row[:, None]], [row[None, :], None]], format="csc")
        rhs = np.r_[rhs, target]
    solution = solve_linear(matrix, rhs, solver=solver, refinement_precision=precision)
    physical = solution[: len(physical_rhs)]
    scale = max(
        float(np.linalg.norm(physical_rhs)),
        float(np.linalg.norm(np.r_[system.load_scale, abs(np.concatenate(loads))])),
        float(np.linalg.norm(abs(physical_matrix) @ abs(physical))),
        np.finfo(float).tiny,
    )
    residual = float(np.linalg.norm(physical_matrix @ physical - physical_rhs) / scale)
    if residual > 1e-10:
        raise ValueError("MH original boundary equations fail after the physical pressure gauge")
    trace = physical[:trace_size]
    coarse = tuple(np.empty(0) for _ in system.responses)
    pressure = tuple(
        response.reconstruct(trace[response.problem.trace_dofs], c)
        for response, c in zip(system.responses, coarse, strict=True)
    )
    hybrid = HybridSolution(trace, coarse, pressure, residual, solution[len(physical_rhs) :])
    boundary_pressure = {}
    offset = trace_size
    for data in faces:
        count = len(data.dofs)
        boundary_pressure[data.face] = physical[offset : offset + count]
        offset += count
    return hybrid, boundary_pressure, physical_matrix, physical_rhs
