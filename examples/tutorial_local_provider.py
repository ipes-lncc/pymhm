"""Declare primal and RT0 Darcy providers and verify their original equations.

Run ``pixi run -e test python -m examples.tutorial_local_provider`` or select
``--formulation mixed --boundary neumann --local-solver external --backend
process --workers 2 --batch-size 1``. Only numerical diagnostics are printed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm.assembly import HybridProblem, SolverConfig, assemble_hybrid
from pymhm.darcy import darcy_local_provider
from pymhm.elements import boundary_data, rt0_evaluate, triangle_quadrature
from pymhm.hybrid import (
    HybridSolution,
    HybridSystem,
    LocalAssembly,
    LocalProblem,
    hybrid_mean_constraint,
    solve_hybrid_system,
)
from pymhm.lagrange import tabulate
from pymhm.mesh import FloatArray, SkeletonSpace, TriangleMesh
from pymhm.parallel import ExecutionConfig
from pymhm.variational import GlobalForm, LocalForm, compile_local_forms

Formulation = Literal["primal", "mixed"]
Boundary = Literal["dirichlet", "neumann"]
_DEFAULT_EXECUTION = ExecutionConfig()


def exact_pressure(points: FloatArray) -> FloatArray:
    """Return ``p=1+x^2+y^2`` in the unit square."""
    return 1 + np.sum(points**2, axis=-1)


def exact_flux(points: FloatArray) -> FloatArray:
    """Return the independently differentiated Darcy flux ``q=-2(x,y)``."""
    return -2 * points


def source(points: FloatArray) -> FloatArray:
    """Return ``div(q)=-4`` independently of the discrete operator."""
    return np.full(points.shape[:-1], -4.0)


def coefficient_compiler(forms: LocalForm) -> LocalProblem:
    """Compile already integrated Galerkin forms without changing their basis.

    This example compiler accepts a sparse ``a``, vector ``L`` and one numeric
    column per trace or moment form. Physical assembly belongs to the supplied
    Darcy provider. UFL expressions require their own compiler adapter.
    """
    coupling = np.column_stack(forms.trace_forms)
    moments = None if forms.moment_forms is None else np.column_stack(forms.moment_forms)
    return LocalProblem(
        forms.a,
        coupling,
        forms.L,
        forms.trace_dofs,
        kernel=forms.kernel,
        constraints=moments,
        coarse_basis=forms.coarse_basis,
    )


def declared_local(cell: int, *, provider: Callable[[int], LocalAssembly]) -> LocalAssembly:
    """Wrap the shared physical assembler in an explicit LocalForm contract."""
    supplied = provider(cell)
    local = supplied.problem
    forms = LocalForm(
        a=local.matrix,
        L=local.load,
        trace_forms=tuple(local.coupling[:, column] for column in range(local.coupling.shape[1])),
        trace_dofs=local.trace_dofs,
        kernel=local.kernel,
        moment_forms=tuple(
            local.constraints[:, column] for column in range(local.constraints.shape[1])
        ),
    )
    return LocalAssembly(compile_local_forms(forms, coefficient_compiler), supplied.metadata)


def external_response(matrix: Any, rhs: FloatArray) -> FloatArray:
    """Use an independent dense solver for every source/trace/mode RHS column.

    The coordinator supplies owned copies and verifies all original augmented
    rows at its unchanged tolerance before accepting this numerical response.
    """
    return np.linalg.solve(matrix.toarray(), rhs)


def build_problem(
    *,
    formulation: Formulation = "primal",
    boundary: Boundary = "dirichlet",
    local_refinement: int = 2,
    element_backend: Literal["portable", "basix"] = "basix",
) -> HybridProblem[int]:
    """Declare two macrocells with P2 primal pressure or RT0/P0 mixed locals.

    The scalar skeleton is P0 physical normal Darcy flux. RT0's legacy degree
    parameter is one; its actual vector space is RT0. Pure Neumann data have
    the analytically compatible total flux -4; the solve adds a physical mean.
    """
    if boundary not in ("dirichlet", "neumann"):
        raise ValueError("boundary must be dirichlet or neumann")
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh)
    provider = darcy_local_provider(
        mesh,
        skeleton=skeleton,
        formulation=formulation,
        degree=2 if formulation == "primal" else 1,
        local_refinement=local_refinement,
        permeability=1.0,
        source=source,
        quadrature_order=4,
        element_backend=element_backend,
    )
    neumann = (
        {
            int(face): float(
                exact_flux(mesh.points[mesh.faces[face]].mean(axis=0)) @ mesh.normals[face]
            )
            for face in mesh.boundary_faces
        }
        if boundary == "neumann"
        else None
    )
    load, fixed = boundary_data(skeleton, exact_pressure, neumann, order=4)
    form = GlobalForm(
        skeleton.size,
        (1,) * len(mesh.cells),
        boundary_load=load if formulation == "primal" else -load,
        fixed_trace=fixed,
    )
    return HybridProblem(form, partial(declared_local, provider=provider), range(len(mesh.cells)))


@dataclass(frozen=True)
class TutorialResult:
    """Executed problem, solution and explicit field/runtime conventions."""

    problem: HybridProblem[int]
    system: HybridSystem
    solution: HybridSolution
    formulation: Formulation
    boundary: Boundary
    element_backend: Literal["portable", "basix"]


def run_tutorial(
    *,
    formulation: Formulation = "primal",
    boundary: Boundary = "dirichlet",
    execution: ExecutionConfig = _DEFAULT_EXECUTION,
    local_solver: Literal["scipy", "external"] = "scipy",
    element_backend: Literal["portable", "basix"] = "basix",
) -> TutorialResult:
    """Assemble, impose the physical gauge if needed, solve and reconstruct."""
    if local_solver not in ("scipy", "external"):
        raise ValueError("local_solver must be scipy or external")
    problem = build_problem(
        formulation=formulation, boundary=boundary, element_backend=element_backend
    )
    solver = SolverConfig(local_solver=external_response if local_solver == "external" else "scipy")
    system = assemble_hybrid(problem, execution=execution, solvers=solver)
    gauges = None
    if boundary == "neumann":
        # Integral of 1+x^2+y^2 over the unit square is 5/3.
        physical_weights = [record[1] for record in system.local_metadata]
        gauges = [hybrid_mean_constraint(system, physical_weights, 5 / 3)]
    solution = solve_hybrid_system(
        system, fixed=dict(problem.global_form.fixed_trace or {}), constraints=gauges
    )
    return TutorialResult(problem, system, solution, formulation, boundary, element_backend)


def diagnostics(result: TutorialResult) -> dict[str, Any]:
    """Measure physical field errors and original algebraic rows by field block."""
    bary, weights = triangle_quadrature(8)
    errors = {"pressure": 0.0, "Darcy_flux": 0.0, "divergence": 0.0}
    original: dict[str, float] = {}
    for response, field, record in zip(
        result.system.responses, result.solution.fields, result.system.local_metadata, strict=True
    ):
        local, (mesh, _) = response.problem, record
        points = bary @ mesh.points[mesh.cells]
        if result.formulation == "primal":
            dofs, _, values, gradient, hessian = tabulate(
                mesh, 2, bary, backend=result.element_backend
            )
            pressure = field[dofs] @ values.T
            flux = -np.einsum("tn,tqna->tqa", field[dofs], gradient)
            divergence = -np.einsum("tn,tqnaa->tq", field[dofs], hessian)
            blocks = {"pressure": slice(None)}
        else:
            nq, npres = len(mesh.faces), len(mesh.cells)
            pressure = np.broadcast_to(field[nq : nq + npres, None], (npres, len(bary)))
            flux = rt0_evaluate(mesh, field[:nq], bary)
            divergence = np.broadcast_to(
                (np.sum(field[mesh.cell_faces] * mesh.signs, axis=1) / mesh.areas)[:, None],
                pressure.shape,
            )
            blocks = {
                "Darcy_flux": slice(0, nq),
                "pressure": slice(nq, nq + npres),
                "boundary_flux_constraint": slice(nq + npres, None),
            }
        errors["pressure"] += float(
            mesh.areas @ ((pressure - exact_pressure(points)) ** 2 @ weights)
        )
        errors["Darcy_flux"] += float(
            mesh.areas @ (np.sum((flux - exact_flux(points)) ** 2, axis=-1) @ weights)
        )
        errors["divergence"] += float(mesh.areas @ ((divergence - source(points)) ** 2 @ weights))
        rows = (
            local.matrix @ field
            + local.coupling @ result.solution.trace[local.trace_dofs]
            - local.load
        )
        for name, selection in blocks.items():
            original[name] = max(original.get(name, 0.0), float(np.linalg.norm(rows[selection])))
    return {
        "formulation": result.formulation,
        "boundary": result.boundary,
        "element_backend": result.element_backend,
        "spaces": "P2 pressure / P0 flux skeleton"
        if result.formulation == "primal"
        else "RT0 Darcy flux / P0 pressure / boundary-pressure multipliers / P0 flux skeleton",
        "field_L2_errors": {name: float(np.sqrt(value)) for name, value in errors.items()},
        "maximum_local_original_row_l2_by_block": original,
        "global_original_relative_residual": result.solution.raw_residual,
        "assembly_quadrature_count": 4,
        "field_error_quadrature_count": 8,
    }


def main() -> None:
    """Execute the selected small formulation and print field/block diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", choices=("primal", "mixed"), default="primal")
    parser.add_argument("--boundary", choices=("dirichlet", "neumann"), default="dirichlet")
    parser.add_argument("--backend", choices=("serial", "thread", "process"), default="serial")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--local-solver", choices=("scipy", "external"), default="scipy")
    parser.add_argument("--element-backend", choices=("portable", "basix"), default="basix")
    args = parser.parse_args()
    result = run_tutorial(
        formulation=args.formulation,
        boundary=args.boundary,
        execution=ExecutionConfig(
            args.backend, args.workers, native_threads=1, batch_size=args.batch_size
        ),
        local_solver=args.local_solver,
        element_backend=args.element_backend,
    )
    print(json.dumps(diagnostics(result), indent=2))


if __name__ == "__main__":
    main()
