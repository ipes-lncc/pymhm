"""Initial short-time spatial refinement of the published Equation (53) wave."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from examples.elastodynamics_campaign import ElasticWave, norms
from examples.minimal_wave_convergence import digest, quadrature_change, require_original, write
from pymhm.elastodynamics import ElastodynamicLocal, ElastodynamicStepper, _stress_from_gradient
from pymhm.element_backends import (
    monomial_tabulation,
    orthogonal_polynomial_tabulation,
    simplex_lagrange_basis,
)
from pymhm.tetra_lagrange import tetra_indices
from pymhm.tetrahedral import (
    TetraMesh,
    tetra_barycentric_gradients,
    tetra_nodal_space,
    tetra_tabulate,
    tetrahedron_quadrature,
)


def executed_basis_arrays(degree: int) -> dict[str, np.ndarray]:
    """Capture the actual cached Basix scalar basis and its literal digest/order.

    The vector field repeats this scalar basis in component-interleaved order.
    Archive replay retains the matrix, native polyset/version and integer
    barycentric node order; it does not construct a new nodal element.
    """
    basis = simplex_lagrange_basis("tetrahedron", degree, nodes=tetra_indices(degree) / degree)
    return {
        "actual_basix_basis_matrix": basis.basis_matrix,
        "actual_basix_basis_sha256": np.asarray(basis.basis_sha256),
        "actual_basix_version": np.asarray(basis.element.backend_version),
        "actual_basix_polyset_type": np.asarray(basis.element.polyset_type),
        "actual_basix_family": np.asarray("P"),
        "actual_basix_cell": np.asarray("tetrahedron"),
        "actual_basix_lagrange_variant": np.asarray("equispaced"),
        "actual_multiindices": tetra_indices(degree),
    }


def saved_tabulation(
    mesh: TetraMesh, degree: int, bary: np.ndarray, arrays: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay archived scalar values and physical gradients without a fresh basis.

    Current archives retain the executed Basix coefficient matrix. Historical
    archives retain already differentiated one-dimensional power coefficients;
    those fixed recipes are evaluated through the shared polynomial provider.
    Neither path changes a shared owner or derives a new finite-element basis.
    """
    from hashlib import sha256

    if not np.array_equal(arrays["actual_multiindices"], tetra_indices(degree)):
        raise ValueError("Archived tetrahedral node order differs")
    dofs, _ = tetra_nodal_space(mesh, degree)
    geometry = tetra_barycentric_gradients(mesh)
    if "actual_basix_basis_matrix" in arrays:
        import basix

        coefficients = arrays["actual_basix_basis_matrix"]
        if (
            coefficients.dtype != np.dtype(float)
            or coefficients.shape != (dofs.shape[1], dofs.shape[1])
            or not np.isfinite(coefficients).all()
            or sha256(coefficients.tobytes(order="C")).hexdigest()
            != str(arrays["actual_basix_basis_sha256"])
            or str(arrays["actual_basix_version"]) != basix.__version__
            or str(arrays["actual_basix_polyset_type"]) != "standard"
            or str(arrays["actual_basix_family"]) != "P"
            or str(arrays["actual_basix_cell"]) != "tetrahedron"
            or str(arrays["actual_basix_lagrange_variant"]) != "equispaced"
        ):
            raise ValueError("Archived Basix basis, digest or native convention differs")
        polynomials = orthogonal_polynomial_tabulation("tetrahedron", degree, bary[:, 1:], 1)
        table = np.einsum("ik,dqk->dqi", coefficients, polynomials)
        values = table[0]
        first = np.stack([table[basix.index(*axis)] for axis in np.eye(3, dtype=int)], axis=-1)
        return dofs, values, np.einsum("qia,taj->tqij", first, geometry[:, 1:])
    recipe = arrays["actual_tetra_cardinal_factor_recipe"]
    if recipe.shape != (degree + 1, 3, degree + 1) or not np.isfinite(recipe).all():
        raise ValueError("Archived cardinal factor recipe differs")
    powers = tuple((power,) for power in range(degree + 1))
    indices = arrays["actual_multiindices"]
    factors = []
    for axis in range(4):
        powers_at_points = monomial_tabulation(bary[:, axis, None], powers, nderiv=0)[0]
        factors.append(np.einsum("wdk,qk->wdq", recipe, powers_at_points))
    values = np.ones((len(bary), len(indices)))
    first = np.ones((*values.shape, 4))
    for axis in range(4):
        selected = factors[axis][indices[:, axis]].transpose(1, 2, 0)
        values *= selected[0]
        for derivative in range(4):
            first[..., derivative] *= selected[int(axis == derivative)]
    return dofs, values, np.einsum("qia,taj->tqij", first, geometry)


def field_norms(
    solution: Any,
    model: ElasticWave,
    order: int,
    *,
    basis_arrays: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Integrate displacement, velocity and Cauchy stress without unused Hessian norms."""
    bary, weights = tetrahedron_quadrature(order)
    totals: np.ndarray = np.zeros(3, dtype=np.longdouble)
    amplitude, velocity, _ = model.amplitudes(solution.time)
    for index, local in enumerate(solution.locals):
        points = np.einsum("qi,tij->tqj", bary, local.mesh.points[local.mesh.cells])
        exact, exact_gradient, _ = model.spatial(points.reshape(-1, 3))
        exact, exact_gradient = (
            exact.reshape(points.shape),
            exact_gradient.reshape(*points.shape, 3),
        )
        if basis_arrays is None:
            dofs, _, basis, gradient = tetra_tabulate(local.mesh, local.degree, bary)
        else:
            dofs, basis, gradient = saved_tabulation(local.mesh, local.degree, bary, basis_arrays)
        u = solution.displacement[index].reshape(-1, 3)[dofs]
        v = solution.velocity[index].reshape(-1, 3)[dofs]
        numerical_u = np.einsum("qi,tia->tqa", basis, u)
        numerical_v = np.einsum("qi,tia->tqa", basis, v)
        stress = _stress_from_gradient(local, points, np.einsum("tqib,tia->tqab", gradient, u))
        exact_stress = _stress_from_gradient(local, points, amplitude * exact_gradient)
        errors = (
            np.sum((numerical_u - amplitude * exact) ** 2, axis=-1),
            np.sum((numerical_v - velocity * exact) ** 2, axis=-1),
            np.sum((stress - exact_stress) ** 2, axis=(-1, -2)),
        )
        for component, error in enumerate(errors):
            totals[component] += np.sum(
                local.mesh.volumes[:, None] * weights * error, dtype=np.longdouble
            )
    return dict(
        zip(
            ("displacement_l2", "velocity_l2", "stress_l2"),
            np.sqrt(totals).astype(float).tolist(),
            strict=True,
        )
    )


def original_update(
    stepper: ElastodynamicStepper,
    old_u: tuple[np.ndarray, ...],
    old_v: tuple[np.ndarray, ...],
    loads: dict[int, list[np.ndarray]],
    old_energy: float,
) -> dict[str, float]:
    """Evaluate original Newmark momentum, kinematics, weak trace and energy/work blocks."""
    dt = stepper.time_step
    momentum, kinematic, work = 0.0, 0.0, 0.0
    work_scale = abs(old_energy) + abs(stepper.solution().energy)
    for local, u0, v0, u1, v1 in zip(
        stepper.locals, old_u, old_v, stepper.displacement, stepper.velocity, strict=True
    ):
        f0, f1 = loads[id(local)]
        inertia = local.mass @ (v1 - v0)
        internal = local.stiffness @ (u0 + u1)
        traction = 2 * (local.coupling @ stepper.trace[local.trace_dofs])
        defect = inertia - dt / 2 * (f0 + f1 - internal - traction)
        scale = np.linalg.norm(inertia) + dt / 2 * (
            np.linalg.norm(f0 + f1) + np.linalg.norm(internal) + np.linalg.norm(traction)
        )
        momentum = max(momentum, float(np.linalg.norm(defect) / max(scale, np.finfo(float).tiny)))
        defect = u1 - u0 - dt / 2 * (v0 + v1)
        scale = (
            np.linalg.norm(u1)
            + np.linalg.norm(u0)
            + dt / 2 * (np.linalg.norm(v0) + np.linalg.norm(v1))
        )
        kinematic = max(kinematic, float(np.linalg.norm(defect) / max(scale, np.finfo(float).tiny)))
        term = np.dot(u1 - u0, f0 + f1 - traction) / 2
        work += float(term)
        work_scale += abs(float(term))
    moments = stepper._moments(stepper.displacement)
    trace_scale = sum(
        float(np.linalg.norm(abs(local.coupling).T @ abs(u)))
        for local, u in zip(stepper.locals, stepper.displacement, strict=True)
    )
    return {
        "momentum_relative": momentum,
        "kinematic_relative": kinematic,
        "weak_displacement_trace_relative": float(
            np.linalg.norm((moments - stepper.boundary)[stepper.free])
            / max(trace_scale, np.finfo(float).tiny)
        ),
        "energy_work_relative": abs(stepper.solution().energy - old_energy - work)
        / max(work_scale, np.finfo(float).tiny),
    }


def acquire(output: Path, *, levels: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    """Use P3/r2-P1, dt=.005 and five steps; explicitly reduce the horizon to .025s."""
    model, rows = ElasticWave(), []
    if not levels or any(n not in (1, 2, 3) for n in levels):
        raise ValueError("Initial spatial levels are n=1,2,3")
    for n in levels:
        started = perf_counter()
        with ElastodynamicStepper(
            TetraMesh.unit_cube(n),
            time_step=0.005,
            degree=3,
            local_refinement=2,
            lame_lambda=0.4,
            lame_mu=0.4,
            quadrature_order=12 if n == 1 else 8,
            backend="serial",
            workers=1,
        ) as stepper:
            result = stepper.initialize()
            maxima: dict[str, float] = {}
            original_load = ElastodynamicLocal.load_at_time
            loads: dict[int, list[np.ndarray]] = {}

            def observed_load(
                local: ElastodynamicLocal,
                source: Any,
                time: float,
                *,
                _original: Any = original_load,
                _records: dict[int, list[np.ndarray]] = loads,
            ) -> np.ndarray:
                """Retain the executed original RHS without changing its arithmetic or owner."""
                value = _original(local, source, time)
                _records.setdefault(id(local), []).append(value.copy())
                return value

            with patch.object(ElastodynamicLocal, "load_at_time", observed_load):
                for _ in range(5):
                    old_u, old_v, energy = stepper.displacement, stepper.velocity, result.energy
                    loads.clear()
                    result = stepper.advance(model.source)
                    checks = original_update(stepper, old_u, old_v, loads, energy)
                    require_original(checks)
                    maxima = {
                        key: max(maxima.get(key, 0.0), value) for key, value in checks.items()
                    }
            error_order = 10
            low, high = norms(result, model, error_order), norms(result, model, error_order + 2)
            sensitivity = quadrature_change(low, high)
            if sensitivity > 1e-6:
                raise ArithmeticError("Elastic-wave physical error quadrature is unresolved")
            field = output / f"n{n}-fields.npz"
            retained_basis: dict[str, Any] = executed_basis_arrays(3)
            np.savez_compressed(
                field,
                displacement=np.asarray(result.displacement),
                velocity=np.asarray(result.velocity),
                trace=result.trace,
                macro_points=stepper.mesh.points,
                macro_cells=stepper.mesh.cells,
                local_points=np.asarray([local.mesh.points for local in stepper.locals]),
                local_cells=np.asarray([local.mesh.cells for local in stepper.locals]),
                **retained_basis,
                local_degree=np.asarray(3),
                local_refinement=np.asarray(2),
                time=np.asarray(result.time),
            )
            rows.append(
                {
                    "level": n,
                    "macro_cells": len(stepper.mesh.cells),
                    "time": result.time,
                    "norms": high,
                    "original_equations": maxima,
                    "quadrature_orders": [error_order, error_order + 2],
                    "quadrature_relative_change": sensitivity,
                    "archive": field.name,
                    "archive_sha256": digest(field),
                    "elapsed_seconds": perf_counter() - started,
                }
            )
        write(output / "progress.json", {"rows": rows})
        print(f"Elastic wave n={n}: {rows[-1]['elapsed_seconds']:.3f}s", flush=True)
    return {
        "reference": "Gomes et al. (2017), doi:10.20906/CPS/CILAMCE2017-0399, Equation (53)",
        "scope": (
            "Initial spatial refinement of the original analytical wave "
            "at an explicitly shortened final time"
        ),
        "configuration": {
            "local_space": "continuous vector P3/r2",
            "trace_space": "vector P1",
            "lambda": 0.4,
            "mu": 0.4,
            "rho": 1.0,
            "dt": 0.005,
            "steps": 5,
            "final_time": 0.025,
            "original_campaign_final_time": 0.5,
            "boundary": "homogeneous weak displacement",
            "initial_displacement": 0,
            "initial_velocity": 0,
        },
        "basis_contract": (
            "Actual cached Basix coefficient matrix/digest and native polyset/version, "
            "topological multiindices, physical geometry and original coefficients retained"
        ),
        "rows": rows,
        "plot_fields": ["displacement_l2", "velocity_l2", "stress_l2"],
        "level_label": "Macro resolution n",
        "norm_label": "Physical L2 error",
        "figure_title": "Equation (53): initial refinement at t=0.025s",
        "exact_solution_available": True,
        "asymptotic_convergence_verified": False,
        "limitations": [
            "Five time steps give an initial spatial study; the original .5s horizon "
            "and asymptotic rates remain outside this acquisition",
            "Temporal error is included in each analytical field error",
        ],
    }


def combine(output: Path, directories: tuple[Path, ...]) -> dict[str, Any]:
    """Reintegrate accepted original fields using their archived executed bases at q12/q14."""
    import json

    rows, sources, template = [], [], None
    for directory in directories:
        path = directory / (
            "study.json" if (directory / "study.json").exists() else "progress.json"
        )
        record = json.loads(path.read_text())
        if "configuration" in record:
            template = record
        sources.append(
            {
                "record": str(path.relative_to(output.parents[2])),
                "sha256": digest(path),
                "executed_sources": str(directory / "executed-sources/manifest.json"),
                "executed_sources_sha256": digest(directory / "executed-sources/manifest.json"),
            }
        )
        for original_row in record["rows"]:
            row = dict(original_row)
            field = directory / row["archive"]
            if digest(field) != row["archive_sha256"]:
                raise ValueError("An accepted original elastic-wave field changed")
            require_original(row["original_equations"])
            with np.load(field, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            if float(arrays["time"]) != 0.025 or int(arrays["local_degree"]) != 3:
                raise ValueError("Elastic-wave field time or approximation space differs")
            locals_ = tuple(
                SimpleNamespace(
                    mesh=TetraMesh(points, cells),
                    degree=3,
                    nodes=points,
                    constitutive=None,
                    lame_lambda=0.4,
                    lame_mu=0.4,
                )
                for points, cells in zip(arrays["local_points"], arrays["local_cells"], strict=True)
            )
            solution = SimpleNamespace(
                locals=locals_,
                time=float(arrays["time"]),
                displacement=tuple(arrays["displacement"]),
                velocity=tuple(arrays["velocity"]),
            )
            low, high = (
                field_norms(solution, ElasticWave(), 12, basis_arrays=arrays),
                field_norms(solution, ElasticWave(), 14, basis_arrays=arrays),
            )
            change = quadrature_change(low, high)
            if change > 1e-6:
                raise ArithmeticError("Saved elastic-wave physical error quadrature is unresolved")
            row.update(
                norms=high,
                quadrature_orders=[12, 14],
                quadrature_relative_change=change,
                archive=str(field),
                original_archive_preserved=True,
            )
            rows.append(row)
    if template is None or sorted(row["level"] for row in rows) != [1, 2, 3]:
        raise ValueError(
            "Exactly three accepted spatial levels and a physical configuration required"
        )
    return {
        key: value
        for key, value in template.items()
        if key not in ("source_sha256", "elapsed_seconds", "native_runtime", "generation_id")
    } | {
        "rows": sorted(rows, key=lambda row: row["level"]),
        "input_acquisitions": sources,
        "quadrature_control": "Same q12/q14 at every level, relative sensitivity<=1e-6",
        "basis_contract": (
            "Literal archived executed coefficient matrix or historical factor recipe, "
            "topological ordering and geometry reused; no new basis or PDE solve"
        ),
    }
