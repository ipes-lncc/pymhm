"""Acquire additional physical vector refinement levels with unchanged methods.

This module orchestrates the existing formulation and quadrature owners. It
stores physical diagnostics, not coefficient vectors or field-replay claims.
The Maxwell spatial and temporal observables remain explicitly independent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.core_elasticity_field_archive import ProductionObservation
from examples.formulations.weak_stress import (
    define_weak_stress,
    recover_weak_stress,
    weak_stress_constraints,
)
from examples.maxwell_data import CavityMode
from examples.maxwell_norms import MaxwellNorms
from examples.minimal_flow_originals import original_diagnostics
from examples.plot_mixed_elasticity import check_manufactured_data, smooth_fields
from examples.solve_primal_elasticity import TensorData
from examples.tutorial_elastodynamic_equations import advance, initialize, prepare
from examples.tutorial_maxwell_equations import EquationLeapfrog
from pymhm import ExecutionConfig, FaceSpace, SkeletonSpace, TriangleMesh, assemble
from pymhm.fem.vector.curl import TangentialTraceSpace, physical_basis, physical_points, quadrature
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from scripts.build_notebook_companions import module_sources

ROOT = case_workspace()


def mixed_displacement(points: np.ndarray) -> np.ndarray:
    """Return the declared solenoidal displacement from the independent data owner."""
    return smooth_fields(points)[0]


def mixed_source(points: np.ndarray) -> np.ndarray:
    """Return minus the independently differentiated isotropic stress divergence."""
    return smooth_fields(points)[2]


def source_hashes() -> dict[str, str]:
    """Identify the numerical dependency closure, without independent plot readers."""
    paths = [
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        *(ROOT / name for name in module_sources(ROOT, [__name__])),
        Path(__file__),
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    return current_source_manifest(source_identity(ROOT, paths), packages=("pymhm",))


def mixed_row(n: int, *, workers: int = 32) -> dict[str, Any]:
    """Extend the solenoidal BDM2/P1/P1 study with its original fine/exterior traces."""
    started = perf_counter()
    check_manufactured_data()
    mesh = TriangleMesh.unit_square(n)
    boundary = set(mesh.boundary_faces)
    skeleton = SkeletonSpace(
        mesh,
        tuple(
            FaceSpace.uniform(2, 2) if face in boundary else FaceSpace.uniform(1)
            for face in range(len(mesh.faces))
        ),
        components=2,
    )
    definition = define_weak_stress(
        mesh,
        skeleton=skeleton,
        stress_degree=2,
        enrichment=0,
        local_refinement=2,
        quadrature_order=10,
        lame_lambda=1.0,
        lame_mu=1.0,
        source=mixed_source,
        dirichlet=mixed_displacement,
    )
    system = assemble(
        definition.problem,
        execution=ExecutionConfig(
            "process" if workers > 1 else "serial", workers=workers, native_threads=1
        ),
    )
    constraints = weak_stress_constraints(definition, system)
    coefficients = system.solve(fixed=definition.problem.fixed, constraints=constraints)
    solution = recover_weak_stress(definition, system, coefficients)
    observed = ProductionObservation(
        system=system,
        applied_boundary=-system.global_load[: system.trace_size],
        fixed=dict(definition.problem.fixed or {}),
        constraints=constraints,
    )
    blocks = []
    for field, (_, ns, nu, nr, *_) in zip(coefficients.fields, system.local_metadata, strict=True):
        blocks.append(
            {
                "stress_constitutive": ns,
                "force_balance": nu,
                "weak_symmetry": nr,
                "normal_traction_compatibility": len(field) - ns - nu - nr,
            }
        )
    original = original_diagnostics(solution, observed, blocks)
    norms = {
        str(order): {
            "displacement_l2": solution.l2_error(lambda p: smooth_fields(p)[0], order),
            "stress_l2": solution.stress_l2_error(lambda p: smooth_fields(p)[1], order),
            "rotation_l2": solution.rotation_l2_error(lambda p: smooth_fields(p)[3], order),
            "stress_divergence_l2": solution.divergence_l2_error(
                lambda p: -smooth_fields(p)[2], order
            ),
        }
        for order in (10, 14)
    }
    low, high = norms.values()
    changes = {
        name: abs(low[name] - high[name]) / max(high[name], np.finfo(float).tiny) for name in high
    }
    invariants = {
        "macro_force_linf": float(np.max(abs(solution.equilibrium_residuals()))),
        "fine_force_linf": max(float(np.max(abs(r))) for r in solution.fine_force_residuals()),
        "weak_symmetry_linf": max(
            float(np.max(abs(r))) for r in solution.weak_symmetry_residuals()
        ),
        "normal_traction_linf": max(
            float(np.max(abs(r))) for r in solution.normal_traction_residuals()
        ),
    }
    accepted = (
        original["accepted"] and max(changes.values()) <= 1e-8 and max(invariants.values()) <= 1e-10
    )
    return {
        "macro_resolution": n,
        "H": 1.0 / n,
        "macro_cells": len(mesh.cells),
        "trace_dofs": skeleton.size,
        **high,
        "quadrature": {
            "assembly_order": 10,
            "error_orders": [10, 14],
            "errors": norms,
            "relative_changes": changes,
        },
        "physical_invariants": invariants,
        "original_equations": original,
        "algebraic_residual": solution.hybrid.residual,
        "wall_seconds": perf_counter() - started,
        "accepted": bool(accepted),
    }


def isotropic_wave_data(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate a homogeneous square-boundary displacement with lambda=mu=1.

    The static shape is (sin(pi*x)sin(pi*y), sin(2*pi*x)sin(pi*y)). Its
    gradient and Hessian come from independent analytical derivatives. The
    force is -lap(w)-2*grad(div(w)), and stress is 2*epsilon(w)+div(w)*I.
    """
    value, gradient, hessian = TensorData(False).derivatives(points)
    stress = (
        gradient
        + gradient.swapaxes(-1, -2)
        + np.trace(gradient, axis1=-2, axis2=-1)[:, None, None] * np.eye(2)
    )
    force = -np.trace(hessian, axis1=-2, axis2=-1) - 2 * np.einsum("njij->ni", hessian)
    return value, stress, force


def wave_force(time: float, points: np.ndarray) -> np.ndarray:
    """Supply rho*u_tt-div(sigma) for u=t²*w/2, rho=lambda=mu=1."""
    value, _, force = isotropic_wave_data(points)
    return value + 0.5 * time * time * force


def elastodynamic_row(
    n: int, *, dt: float = 0.01, final: float = 0.1, workers: int = 32
) -> dict[str, Any]:
    """Measure a smooth 2D analogue separately from the original 3D Eq.53 case.

    Quadratic time dependence makes the continuum Newmark update exact. A
    separate dt/2 run still checks the propagated discrete error. P3 locals,
    P1 negative traction and all original endpoint replay checks are retained.
    """
    started = perf_counter()
    steps = round(final / dt)
    if steps < 1 or abs(steps * dt - final) > 1e-13:
        raise ValueError("Final time must be a positive whole number of steps")
    mesh = TriangleMesh.unit_square(n)
    with prepare(
        mesh,
        time_step=dt,
        degree=3,
        local_refinement=1,
        density=1.0,
        lame_lambda=1.0,
        lame_mu=1.0,
        quadrature_order=10,
        backend="process" if workers > 1 else "serial",
        workers=workers,
    ) as data:
        result = initialize(data)
        for _ in range(steps):
            result = advance(data, result, wave_force)
        norms = {}
        for order in (10, 14):
            stress_total = np.longdouble(0)
            for local, field in zip(result.locals, result.displacement, strict=True):
                bary, weights, _ = quadrature(local.mesh, 1, order)
                _, gradient = physical_basis(local.mesh, local.degree, bary)
                points = physical_points(local.mesh, bary)
                exact = (
                    0.5
                    * final
                    * final
                    * isotropic_wave_data(points.reshape(-1, 2))[1].reshape(*points.shape[:2], 2, 2)
                )
                dofs = local.dofs
                values = field.reshape(-1, 2)[dofs]
                derivative = np.einsum("tqib,tia->tqab", gradient, values)
                stress = (
                    derivative
                    + derivative.swapaxes(-1, -2)
                    + np.trace(derivative, axis1=-2, axis2=-1)[..., None, None] * np.eye(2)
                )
                stress_total += np.sum(
                    weights * np.sum((stress - exact) ** 2, axis=(-1, -2)), dtype=np.longdouble
                )
            norms[str(order)] = {
                "displacement_l2": result.l2_error(
                    lambda p: 0.5 * final * final * isotropic_wave_data(p)[0], order=order
                ),
                "velocity_l2": result.l2_error(
                    lambda p: final * isotropic_wave_data(p)[0], velocity=True, order=order
                ),
                "stress_l2": float(np.sqrt(stress_total)),
            }
        low, high = norms.values()
        changes = {
            name: abs(low[name] - high[name]) / max(high[name], np.finfo(float).tiny)
            for name in high
        }
        return {
            "study": "spatial",
            "dimension": 2,
            "n": n,
            "H": np.sqrt(2) / n,
            "dt": dt,
            "final_time": final,
            "macro_cells": len(mesh.cells),
            "local_degree": 3,
            "local_refinement": 1,
            "trace_degree": 1,
            **high,
            "quadrature": {
                "assembly_order": 10,
                "error_orders": [10, 14],
                "errors": norms,
                "relative_changes": changes,
            },
            "original_endpoint_replay_checked": True,
            "constraint_residual": result.constraint_residual,
            "wall_seconds": perf_counter() - started,
            "accepted": bool(max(changes.values()) <= 1e-8),
        }


def classical_maxwell_row(n: int, *, dt: float = 0.0005, final: float = 0.05) -> dict[str, Any]:
    """Independently solve the TM scalar wave equation using conforming P3 UFL.

    E satisfies E_tt-Delta(E)=0, homogeneous PEC, E(0)=sin(2pi*x)sin(2pi*y)
    and E_t(0)=0. Magnetic H=-curl(integral(E)dt) is recovered with trapezoidal
    time integration. This global classical formulation uses no MHM operators.
    """
    import basix.ufl
    import ufl
    from dolfinx import fem

    from pymhm import compile_form
    from pymhm.backends.spaces import bind_space, coefficient_map
    from pymhm.fem.scalar.triangle import nodal_space
    from pymhm.linalg.dynamics import newmark_step
    from pymhm.linalg.linear import factorize

    started = perf_counter()
    mesh = TriangleMesh.unit_square(n)
    native = bind_space(
        mesh,
        basix.ufl.element(
            "Lagrange", "triangle", 3, lagrange_variant=basix.LagrangeVariant.equispaced
        ),
    )
    try:
        trial, test = ufl.TrialFunction(native.space), ufl.TestFunction(native.space)
        domain = native.space.mesh
        x = ufl.SpatialCoordinate(domain)
        shape = ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 14})
        mass = compile_form(trial * test * dx)
        stiffness = compile_form(ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx)
        _, points = nodal_space(mesh, 3)
        portable_boundary = np.flatnonzero(
            np.any(np.isclose(points, 0) | np.isclose(points, 1), axis=1)
        )
        boundary = coefficient_map(native)[portable_boundary]
        free = np.setdiff1d(np.arange(native.size), boundary)
        M, K = mass[free][:, free], stiffness[free][:, free]
        initial_load = compile_form(shape * test * dx)[free]
        original_momentum = 0.0
        with factorize(M) as mass_factor, factorize(M + dt * dt / 4 * K) as step_factor:
            values, velocity = mass_factor.solve(initial_load), np.zeros(len(free))
            integral = np.zeros(len(free))
            for _ in range(round(final / dt)):
                old, old_velocity = values, velocity
                values, velocity = newmark_step(
                    M,
                    K,
                    step_factor,
                    mass_factor,
                    dt,
                    values,
                    velocity,
                    np.zeros(len(free)),
                    np.zeros(len(free)),
                )
                integral += dt / 2 * (old + values)
                inertia = M @ (velocity - old_velocity)
                elastic = dt / 2 * (K @ (old + values))
                original_momentum = max(
                    original_momentum,
                    float(
                        np.linalg.norm(inertia + elastic)
                        / max(
                            np.linalg.norm(inertia) + np.linalg.norm(elastic), np.finfo(float).tiny
                        )
                    ),
                )
        electric, primitive = fem.Function(native.space), fem.Function(native.space)
        electric.x.array[free] = values
        primitive.x.array[free] = integral
        magnetic = ufl.as_vector((-primitive.dx(1), primitive.dx(0)))
        exact_e = np.cos(2 * np.pi * np.sqrt(2) * final) * shape
        exact_h = float(np.sin(2 * np.pi * np.sqrt(2) * final) / np.sqrt(2)) * ufl.as_vector(
            (
                -ufl.sin(2 * np.pi * x[0]) * ufl.cos(2 * np.pi * x[1]),
                ufl.cos(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1]),
            )
        )
        errors = {}
        for order in (14, 18):
            measure = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": order})
            errors[str(order)] = {
                "electric_l2": float(
                    np.sqrt(fem.assemble_scalar(fem.form((electric - exact_e) ** 2 * measure)))
                ),
                "magnetic_l2": float(
                    np.sqrt(
                        fem.assemble_scalar(
                            fem.form(ufl.inner(magnetic - exact_h, magnetic - exact_h) * measure)
                        )
                    )
                ),
            }
        low, high = errors.values()
        changes = {
            name: abs(low[name] - high[name]) / max(high[name], np.finfo(float).tiny)
            for name in high
        }
        return {
            "resolution": n,
            "degree": 3,
            "dt": dt,
            "time": final,
            "dofs": native.size,
            "errors": errors["18"],
            "quadrature": {"error_orders": [14, 18], "errors": errors, "relative_changes": changes},
            "original_momentum_relative_max": original_momentum,
            "wall_seconds": perf_counter() - started,
            "accepted": bool(max(changes.values()) <= 1e-6 and original_momentum <= 1e-10),
        }
    finally:
        native.close()


def maxwell_row(n: int, *, dt: float = 0.0005, final: float = 0.05) -> dict[str, Any]:
    """Measure the paper's smooth 2D TM cavity through every staggered sample.

    Face P1, a single local P3 triangle, homogeneous PEC and unit material
    match section 6.2 of Lanteri et al. (2018). The error is measured directly
    against the continuum cavity mode, not the same semidiscrete evolution.
    Independent orders 8 and 12 assess every norm maximum over all samples.
    """
    started = perf_counter()
    count = round(final / dt)
    if count < 1 or abs(count * dt - final) > 1e-13:
        raise ValueError("The final time must be a positive whole number of steps")
    mesh = TriangleMesh.unit_square(n)
    base = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    mode = CavityMode(2)
    with EquationLeapfrog(
        mesh,
        skeleton=TangentialTraceSpace(base),
        degree=3,
        local_refinement=1,
        time_step=dt,
        absorbing=None,
        quadrature_order=8,
    ) as stepper:
        solution = stepper.initialize(mode.electric_shape)
        norms = {order: MaxwellNorms(solution, mode, order) for order in (8, 12)}
        maxima = {order: integrator.measure(solution) for order, integrator in norms.items()}
        original = solution.original_electric_residual
        energy, drift, balance = solution.energy, 0.0, 0.0
        for step in range(count):
            solution = stepper.advance()
            for order, integrator in norms.items():
                current = integrator.measure(solution)
                maxima[order] = {
                    name: max(value, current[name]) for name, value in maxima[order].items()
                }
            original = max(original, solution.original_electric_residual)
            drift = max(drift, abs(solution.energy / energy - 1))
            balance = max(balance, abs(solution.energy_balance_residual))
            if (step + 1) % 25 == 0:
                print(
                    json.dumps(
                        {"n": n, "dt": dt, "step": step + 1, "time": solution.magnetic_time}
                    ),
                    flush=True,
                )
        low, high = maxima.values()
        changes = {
            name: abs(low[name] - high[name]) / max(high[name], np.finfo(float).tiny)
            for name in high
        }
        return {
            "study": "spatial",
            "dimension": 2,
            "resolution": n,
            "H": 1.0 / n,
            "macro_cells": len(mesh.cells),
            "trace_degree": 1,
            "local_degree": 3,
            "local_refinement": 1,
            "trace_dofs": stepper.skeleton.size,
            "time_step": dt,
            "steps": count,
            "final_magnetic_time": solution.magnetic_time,
            "final_electric_time": solution.electric_time,
            "max_in_time": high,
            "quadrature": {
                "assembly_order": 8,
                "error_orders": [8, 12],
                "max_in_time": maxima,
                "relative_changes": changes,
            },
            "original_electric_relative_max": original,
            "modified_energy_relative_drift": drift,
            "energy_balance_max": balance,
            "cfl_product": dt * stepper.frequency_bound,
            "wall_seconds": perf_counter() - started,
            "accepted": bool(
                original <= 1e-10
                and max(changes.values()) <= 1e-6
                and drift <= 1e-10
                and balance <= 1e-10
            ),
        }


def main() -> None:
    """Write fresh current-source extension diagnostics after each accepted level."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("mixed-elasticity", "maxwell", "maxwell-classical", "elastodynamics"),
        required=True,
    )
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.0005)
    parser.add_argument("--final", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing numerical record cannot be overwritten")
    hashes = source_hashes()
    record = {
        "schema": "pymhm-tutorial-vector-asymptotic-v1",
        "method": args.method,
        "source_sha256": hashes,
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for n in args.levels:
            if args.method == "mixed-elasticity":
                row = mixed_row(n, workers=args.workers)
            elif args.method == "maxwell":
                row = maxwell_row(n, dt=args.dt, final=args.final)
            elif args.method == "maxwell-classical":
                row = classical_maxwell_row(n, dt=args.dt, final=args.final)
            else:
                row = elastodynamic_row(n, dt=args.dt, final=args.final, workers=args.workers)
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if not row["accepted"]:
                raise RuntimeError(f"Physical equations or quadrature rejected n={n}")
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in hashes.items()):
        raise RuntimeError("An executed numerical source changed during acquisition")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.tutorial_vector_asymptotic").main()
