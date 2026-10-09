"""Run one user-declared heterogeneous Darcy form in different environments.

The canonical mathematical callback is ``define_ufl_local_equations`` from the
introductory forms module. This file owns only reproducible mesh descriptions,
execution selection and physical measurements; assembly, condensation, solving
and field evaluation delegate to their existing owners. MPI callable helpers
are importable and native resources remain rank/worker owned.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from time import perf_counter
from typing import Any

import numpy as np

from examples.introduction.scalar import evaluate_qk, grid_quadrature, physical_errors
from examples.introduction.scaling_forms import (
    PeriodicDarcyData,
    define_ufl_local_equations,
    exact_gradient,
    exact_pressure,
)
from pymhm import (
    Equation,
    ExecutionConfig,
    MeshHierarchy,
    MultiscaleProblem,
    SolverConfig,
    assemble,
    bind_interface,
    bind_problem,
    solve,
)
from pymhm.core.context import BoundProblem
from pymhm.core.equations import compile_local_equations
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh


@dataclass(frozen=True)
class LocalMeshes:
    """Describe square Q1 fine meshes without constructing native resources."""

    macro: CartesianMacroMesh
    refinement: int

    def __call__(self, cell: int) -> CartesianMacroMesh:
        """Create the requested portable fine mesh inside the owning worker."""
        return self.macro.submesh(cell, self.refinement)


def declared_problem(n: int = 2, refinement: int = 8) -> BoundProblem:
    """Bind the same period-0.25 Q1 Darcy problem to normal-flux trace spaces.

    Local Q1 pressures retain their constant kernel and integral moment. P1
    continuous face modes live separately on each complete macroface. The
    manufactured pressure vanishes on the entire boundary, so its global
    pressure load is zero and Dirichlet data remove the global pressure gauge.
    The UFL quadrature degree is six, as in the canonical local callback.
    """
    macro = CartesianMacroMesh(n, n)
    skeleton = SkeletonSpace(
        macro, tuple(FaceSpace.uniform(1, 1, continuous=True) for _ in macro.faces)
    )
    hierarchy = MeshHierarchy(macro, LocalMeshes(macro, refinement))
    provider = partial(define_ufl_local_equations, data=PeriodicDarcyData(0.25))
    return bind_problem(
        hierarchy,
        bind_interface(skeleton, convention="normal"),
        provider,
        global_equation=Equation(0, 0),
        retained=1,
    )


def measure_fields(
    problem: BoundProblem, cells: Any, fields: Any, metadata: Any
) -> dict[str, float]:
    """Integrate physical pressure/flux errors on each owned macrocell.

    Measurements reuse the existing Q1 evaluator and physical error owner with
    tensor Gauss order six on every fine cell. The returned absolute squared
    errors are additive across MPI ranks. A raw primal flux is not H(div).
    Native coefficient permutations come from the executed provider metadata.
    """
    sums = {name: 0.0 for name in ("pressure_L2", "flux_L2", "flux_energy")}
    data = PeriodicDarcyData(0.25)
    for cell, values, record in zip(cells, fields, metadata, strict=True):
        fine = problem.context.hierarchy.local_mesh(int(cell))
        points, weights = grid_quadrature(fine.bounds, (fine.nx, fine.ny), order=6)
        # The callback records, for each native coefficient, its Cartesian slot.
        # Invert that scatter map when gathering coefficients for the evaluator.
        coefficients = values[np.argsort(record["native_to_cartesian"])]
        evaluator = partial(evaluate_qk, fine, 1, coefficients)
        measured = physical_errors(
            evaluator,
            lambda x: (exact_pressure(x), exact_gradient(x)),
            data.permeability,
            points,
            weights,
        )
        for name in sums:
            sums[name] += measured[name] ** 2
    return sums


def classical_reference(n: int) -> tuple[Any, dict[str, float]]:
    """Solve conforming Q1 Darcy on an independently specified fine global grid.

    The common UFL callback supplies identical material and source quadrature on
    a single global fine mesh; its volume matrix/load are used directly, without
    multiscale elimination. Strong homogeneous boundary pressure is imposed in
    the executed native coefficient order. Error quadrature uses order six.
    """
    problem = declared_problem(n=1, refinement=n)
    record = compile_local_equations(problem.local_provider(0))
    fine = problem.context.hierarchy.local_mesh(0)
    gather = np.argsort(record.metadata["native_to_cartesian"])
    exterior = np.flatnonzero(
        np.any(np.isclose(fine.points, 0) | np.isclose(fine.points, 1), axis=1)
    )
    definition = MultiscaleProblem.from_global(
        Equation(record.problem.matrix, record.problem.load),
        len(fine.points),
        fixed={int(gather[node]): 0.0 for node in exterior},
    )
    result = solve(assemble(definition))
    evaluator = partial(evaluate_qk, fine, 1, result.trace[gather])
    points, weights = grid_quadrature(fine.bounds, (n, n), order=6)
    errors = physical_errors(
        evaluator,
        lambda x: (exact_pressure(x), exact_gradient(x)),
        PeriodicDarcyData(0.25).permeability,
        points,
        weights,
    )
    return evaluator, errors


def run_cpu(
    problem: BoundProblem,
    backend: str = "serial",
    workers: int = 1,
    local_solver: str = "scipy",
) -> tuple[Any, Any, dict[str, float]]:
    """Assemble, solve and measure one CPU-policy or CUDA-local run completely.

    The complete timer includes first-use form compilation, setup, worker
    startup, numerical assembly, factorization, transfer where applicable,
    global solving, reconstruction and physical measurements. This small guide
    is a correctness demonstration, not a performance campaign.
    """
    started = perf_counter()
    system = assemble(
        problem,
        execution=ExecutionConfig(
            backend=backend,
            workers=workers,
            batch_size=max(1, 2 * workers),
            pipeline=backend != "serial",
            native_threads=1,
        ),
        solvers=SolverConfig(local_solver=local_solver, global_solver="scipy"),
    )
    solution = system.solve()
    sums = measure_fields(
        problem, problem.context.hierarchy.items, solution.fields, system.local_metadata
    )
    report = {name: float(np.sqrt(value)) for name, value in sums.items()}
    report["original_equations_relative"] = float(solution.raw_residual)
    report["complete_seconds"] = perf_counter() - started
    return system, solution, report


def run_mpi(
    problem: BoundProblem, comm: Any, local_solver: str = "scipy"
) -> tuple[Any, dict[str, float]]:
    """Solve rank-owned cells and collectively measure the same physical fields.

    The public MPI API requires PETSc/MUMPS globally. Native local forms use
    COMM_SELF; a selected CUDA solver affects local algebra only. The complete
    clock ends after collective synchronization and includes field norms.
    """
    from mpi4py import MPI

    from pymhm.execution.mpi import solve_distributed

    cells = problem.context.hierarchy.items[comm.rank :: comm.size]
    comm.Barrier()
    started = perf_counter()
    solution = solve_distributed(
        problem.local_provider,
        cells,
        trace_size=problem.trace_size,
        comm=comm,
        local_solver=local_solver,
    )
    sums = measure_fields(problem, cells, solution.fields, solution.local_metadata)
    report = {
        name: float(np.sqrt(comm.allreduce(value, op=MPI.SUM))) for name, value in sums.items()
    }
    report["original_equations_relative"] = float(solution.raw_residual)
    comm.Barrier()
    report["complete_seconds"] = comm.allreduce(perf_counter() - started, op=MPI.MAX)
    return solution, report
