"""Declare primal and RT0 Darcy providers and verify their original equations.

Run ``python -m examples.tutorial_local_provider`` or select
``--formulation mixed --boundary neumann --local-solver external --backend
process --workers 2 --batch-size 1``. Only numerical diagnostics are printed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.assembly import SolverConfig
from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.system import HybridSystem, hybrid_mean_constraint, solve_hybrid_system
from pymhm.core.validation import FloatArray, positive_int
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.scalar.operators import (
    boundary_data,
    face_integration,
    rt0_evaluate,
    rt0_operators,
    triangle_quadrature,
)
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

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


@dataclass(frozen=True)
class TutorialProvider:
    """User-written primal or RT0 equations, including their global balance rows.

    The FEM kernels integrate volume and boundary terms. This application
    declares A, f, B and C=-B.T rather than selecting a library PDE solver.
    Mixed coefficients are flux, P0 pressure and local boundary pressure;
    the third equation prescribes normal flux from the skeletal variable.
    """

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    formulation: Formulation = "primal"
    local_refinement: int = 2
    element_backend: Literal["portable", "basix"] = "basix"

    def __post_init__(self) -> None:
        """Validate the declared application spaces before local construction."""
        if self.formulation not in {"primal", "mixed"}:
            raise ValueError("formulation must be primal or mixed")
        if self.element_backend not in {"portable", "basix"}:
            raise ValueError("element_backend must be portable or basix")
        positive_int(self.local_refinement, "local_refinement")
        if self.skeleton.mesh is not self.mesh or self.skeleton.components != 1:
            raise ValueError("TutorialProvider requires its own scalar skeleton")

    def __call__(self, cell: int) -> LocalEquations:
        """Supply actual local forms and independently declared trace-test rows."""
        fine = self.mesh.submesh(cell, self.local_refinement)
        if self.formulation == "primal":
            a, mass, load = scalar_operators(
                fine,
                2,
                diffusion=1.0,
                source=source,
                order=4,
                element_backend=self.element_backend,
            )
            b = trace_coupling(self.mesh, cell, fine, self.skeleton, 2)
            kernel = np.ones((a.shape[0], 1))
            physical_mean = mass @ kernel
        else:
            for face in self.mesh.cell_faces[cell]:
                space = self.skeleton.faces[face]
                if any(space.degrees) or not np.allclose(
                    np.asarray(space.breaks) * self.local_refinement,
                    np.round(np.asarray(space.breaks) * self.local_refinement),
                    atol=1e-12,
                    rtol=0,
                ):
                    raise ValueError("RT0 needs degree-zero trace segments aligned with fine edges")
            _, flux_map = face_integration(self.mesh, cell, fine, self.skeleton)
            mass, divergence, force = rt0_operators(fine, 1.0, source, order=4)
            nq, npres, nb = len(fine.faces), len(fine.cells), len(fine.boundary_faces)
            normal = sparse.coo_matrix(
                (np.ones(nb), (fine.boundary_faces, np.arange(nb))),
                shape=(nq, nb),
            ).tocsc()
            zero = sparse.csc_matrix((npres, nb))
            a = sparse.bmat(
                [
                    [mass, -divergence.T, normal],
                    [-divergence, None, zero],
                    [normal.T, zero.T, None],
                ],
                format="csc",
            )
            b = np.zeros((nq + npres + nb, flux_map.shape[1]))
            b[nq + npres :] = -flux_map
            load = np.r_[np.zeros(nq), -force, np.zeros(nb)]
            kernel = np.r_[np.zeros(nq), np.ones(npres + nb)][:, None]
            physical_mean = np.r_[np.zeros(nq), fine.areas, np.zeros(nb)][:, None]
        moments = physical_mean / (kernel.T @ physical_mean).item()
        return LocalEquations(
            a,
            load,
            columns(*b.T),
            rows(*(-b.T)),
            self.skeleton.cell_dofs(cell),
            kernel=kernel,
            moments=moments,
            metadata=(fine, physical_mean[:, 0]),
        )


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
) -> MultiscaleProblem[int]:
    """Declare two macrocells with P2 primal pressure or RT0/P0 mixed locals.

    The scalar skeleton is P0 physical normal Darcy flux. RT0's legacy degree
    parameter is one; its actual vector space is RT0. Pure Neumann data have
    the analytically compatible total flux -4; the solve adds a physical mean.
    """
    if boundary not in ("dirichlet", "neumann"):
        raise ValueError("boundary must be dirichlet or neumann")
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh)
    provider = TutorialProvider(mesh, skeleton, formulation, local_refinement, element_backend)
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
    boundary_rhs = -load if formulation == "primal" else load
    form = Equation(0, np.r_[boundary_rhs, np.zeros(len(mesh.cells))])
    return MultiscaleProblem(
        form,
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )


@dataclass(frozen=True)
class TutorialResult:
    """Executed problem, solution and explicit field/runtime conventions."""

    problem: MultiscaleProblem[int]
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
    system = assemble(problem, execution=execution, solvers=solver)
    gauges = None
    if boundary == "neumann":
        # Integral of 1+x^2+y^2 over the unit square is 5/3.
        physical_weights = [record[1] for record in system.local_metadata]
        gauges = [hybrid_mean_constraint(system, physical_weights, 5 / 3)]
    solution = solve_hybrid_system(system, fixed=dict(problem.fixed or {}), constraints=gauges)
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
    from importlib import import_module

    import_module("examples.tutorial_local_provider").main()
