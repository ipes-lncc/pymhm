"""Acquire the explicitly selected L11 Section 5.4 Darcy--transport trajectory.

Default spaces match the stated 512 macrotriangles, 64 fine P3 triangles,
Darcy P2 traces on two segments and transport P2 traces on eight segments.
The actual permeability array and exterior Darcy drive are independently
declared inputs. Original-equation acceptance does not certify positivity,
inf-sup stability, continuum accuracy or the missing historical realization.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from examples.campaign_provenance import file_digest
from examples.transport_checkpoints import checkpoint_field, write_progress
from examples.transport_random_problem import (
    INPUT,
    darcy_pressure,
    load_realization,
    macro_mesh,
    natural_faces,
    require_fitted_partition,
)
from examples.transport_trajectory import (
    coefficient_arrays,
    scalar_geometry,
    transport_trace_geometry,
    uniform_trace_geometry,
    vector_coefficients,
)
from pymhm import FaceSpace, SkeletonSpace, solve_darcy
from pymhm.darcy_transport import solve_darcy_transport
from pymhm.mesh import positive_int
from pymhm.scalar_boundary import strong_boundary_dofs
from pymhm.transport import ScalarSolution

ROOT = Path(__file__).resolve().parents[1]


def time_grid(dt: float, final_time: float) -> np.ndarray:
    """Require an integer number of positive backward-Euler increments from zero."""
    if not np.isfinite([dt, final_time]).all() or min(dt, final_time) <= 0:
        raise ValueError("dt and final_time must be finite and positive")
    steps = round(final_time / dt)
    if steps < 1 or abs(steps * dt - final_time) > 64 * np.finfo(float).eps * max(dt, final_time):
        raise ValueError("final_time must contain an integer number of dt increments")
    times = np.arange(steps + 1, dtype=float) * dt
    times[-1] = final_time
    return times


def source_digests() -> dict[str, str]:
    """Guard the actual input, lockfile, all core sources and acquisition owners."""
    paths = [*sorted((ROOT / "src/pymhm").glob("*.py")), INPUT, ROOT / "pixi.lock"]
    paths += [
        ROOT / "examples" / f"{name}.py"
        for name in (
            "transport_random_campaign",
            "transport_random_problem",
            "transport_trajectory",
            "transport_checkpoints",
            "archive_precision",
            "local_response_cache",
            "campaign_provenance",
        )
    ]
    return {str(path.relative_to(ROOT)): file_digest(path) for path in paths}


def acquire(
    output: Path | None = None,
    *,
    nx: int = 32,
    ny: int = 8,
    refinement: int = 8,
    dt: float = 0.001,
    final_time: float = 7.0,
    output_every: int = 1000,
    quadrature_order: int = 8,
) -> dict[str, Any]:
    """Acquire physical fields and every original step check, with bounded history.

    Saves the initial interpolation, first step, specified observation intervals
    and final step. Each state keeps physical nodal coefficients, executed trace
    basis and strong-boundary reactions. Retained coarse amplitudes are omitted:
    they cannot replace the complete executed fields. Checkpoints describe
    observed fields; this command does not implement integrator restart.
    """
    require_fitted_partition(nx, ny, refinement)
    output_every = positive_int(output_every, "output_every")
    quadrature_order = positive_int(quadrature_order, "quadrature_order")
    times = time_grid(dt, final_time)
    observations = sorted({1, len(times) - 1, *range(output_every, len(times), output_every)})
    material, inputs = load_realization()
    sources = source_digests()
    generation = str(uuid.uuid4())
    output = (
        ROOT / "build/results/transport-random-selected" / generation if output is None else output
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError("acquisition output must be empty; historical checkpoints are not reused")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    mesh = macro_mesh(nx, ny)
    darcy_space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces))
    transport_space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 8) for _ in mesh.faces))
    diffusive = natural_faces(mesh, transport=True)
    essential = tuple(int(face) for face in mesh.boundary_faces if face not in diffusive)
    with threadpool_limits(1):
        darcy = solve_darcy(
            mesh,
            permeability=material,
            source=0.0,
            dirichlet=darcy_pressure,
            neumann=natural_faces(mesh, transport=False),
            skeleton=darcy_space,
            degree=3,
            local_refinement=refinement,
            quadrature_order=quadrature_order,
            hybrid_refinement_steps=1,
        )
        darcy_seconds = time.perf_counter() - started
        geometry, geometry_record = scalar_geometry(transport_space, darcy.local_meshes, 3)
        darcy_basis, darcy_record = uniform_trace_geometry(
            darcy_space, convention="Darcy physical normal flux, positive along global macro normal"
        )
        transport_basis, transport_record = transport_trace_geometry(
            transport_space,
            diffusive_faces=tuple(diffusive),
            strong_faces=essential,
            numerical_normal_velocity=(
                "Executed Darcy physical normal flux multiplier; independent of raw volume normal"
            ),
        )
        boundary_nodes = [
            strong_boundary_dofs(
                mesh,
                fine,
                tuple(face for face in essential if face in mesh.cell_faces[cell]),
                3,
                1.0,
            )[0]
            for cell, fine in enumerate(darcy.local_meshes)
        ]
        geometry["essential_nodal_dofs"] = np.concatenate(
            [ids + geometry["nodal_offsets"][cell] for cell, ids in enumerate(boundary_nodes)]
        )
        geometry["essential_offsets"] = np.r_[0, np.cumsum([len(ids) for ids in boundary_nodes])]
        centroids = geometry["fine_points"][geometry["fine_cells"]].mean(axis=1)
        geometry["permeability"] = material(centroids)
        from examples.local_response_cache import array_identity

        geometry_record["basis_sha256"].update(
            {
                key: array_identity(geometry[key])
                for key in ("essential_nodal_dofs", "essential_offsets", "permeability")
            }
        )
        pressure, pressure_record = coefficient_arrays("darcy_pressure", darcy.pressure, geometry)
        darcy_trace, darcy_trace_record = vector_coefficients(
            "darcy_trace", darcy.hybrid.trace, convention=darcy_record["convention"]
        )
        geometry_bundle = dict(geometry, **pressure, **darcy_trace)
        geometry_bundle.update({f"darcy_{key}": value for key, value in darcy_basis.items()})
        geometry_bundle.update(
            {f"transport_{key}": value for key, value in transport_basis.items()}
        )
        geometry_checkpoint = checkpoint_field(
            output / "geometry.npz",
            geometry_bundle,
            {
                "generation": generation,
                "geometry": geometry_record,
                "darcy_pressure": pressure_record,
                "darcy_trace": darcy_trace_record,
                "darcy_trace_basis": darcy_record,
                "transport_trace_basis": transport_record,
                "darcy_original_check": {"criterion": 1e-10, "correction_limit": 1},
                "source_sha256": sources,
            },
        )
        rows: list[dict[str, Any]] = []
        extrema: list[tuple[float, float]] = []
        zero = tuple(np.zeros(len(p)) for p in darcy.pressure)
        initial, initial_record = coefficient_arrays("concentration", zero, geometry)
        initial_checkpoint = checkpoint_field(
            output / "initial.npz",
            initial,
            {"time": 0.0, "interpolation_only": True, "coefficients": initial_record},
        )
        configuration = {
            "macro_shape": [nx, ny],
            "macro_cells": len(mesh.cells),
            "local_degree": 3,
            "local_refinement": refinement,
            "darcy_trace": {"degree": 2, "segments": 2},
            "transport_trace": {"degree": 2, "segments": 8},
            "requested_assembly_order": quadrature_order,
            "effective_assembly_order": max(quadrature_order, 5),
            "dt": dt,
            "final_time": final_time,
            "steps": len(times) - 1,
            "velocity": "Raw primal Darcy volume vector and Darcy skeletal normal flux pair",
            "stabilization": "galerkin",
            "capacity": 1.0,
            "initial": 0.0,
            "inflow": 1.0,
            "source": 0.0,
            "natural_diffusive_flux": 0.0,
            "molecular": 1e-6,
            "longitudinal": 1e-2,
            "transverse": 1e-3,
        }
        progress = {
            "generation": generation,
            "status": "trajectory in progress; no final acceptance",
            "scope": inputs["scope"],
            "permeability_sha256": inputs["permeability_sha256"],
            "source_sha256": sources,
            "configuration": configuration,
            "geometry_archive_sha256": geometry_checkpoint["archive_sha256"],
            "initial_archive_sha256": initial_checkpoint["archive_sha256"],
            "observations": rows,
        }

        def observe(step: int, now: float, solution: ScalarSolution, balance: np.ndarray) -> None:
            """Persist selected complete physical fields without changing the trajectory."""
            nodal_min = float(min(np.min(values) for values in solution.values))
            nodal_max = float(max(np.max(values) for values in solution.values))
            extrema.append((nodal_min, nodal_max))
            if step not in observations:
                return
            arrays, coefficients = coefficient_arrays("concentration", solution.values, geometry)
            trace, trace_record = vector_coefficients(
                "transport_trace", solution.hybrid.trace, convention=transport_record["convention"]
            )
            reactions = []
            for values, field, ids in zip(
                solution.values, solution.hybrid.fields, boundary_nodes, strict=True
            ):
                if not np.array_equal(values, field[: len(values)]) or len(field) != len(
                    values
                ) + len(ids):
                    raise ValueError("executed physical fields or essential reaction maps differ")
                reactions.append(field[len(values) :])
            reaction, reaction_record = vector_coefficients(
                "essential_reactions",
                np.concatenate(reactions),
                convention=(
                    "Nodal Lagrange reactions dual to strong concentration data; not physical flux"
                ),
            )
            state_path = output / f"state-{step:07d}.npz"
            row = {
                "step": step,
                "time": now,
                "original_equations_checked": True,
                "original_criterion": 1e-10,
                "macro_weak_balance_linf": float(np.max(abs(balance))),
                "nodal_min": nodal_min,
                "nodal_max": nodal_max,
                "elapsed_transient_seconds": time.perf_counter() - transient_started,
                "coefficients": coefficients,
                "trace": trace_record,
                "essential_reactions": reaction_record,
            }
            checkpoint = checkpoint_field(state_path, dict(arrays, **trace, **reaction), row)
            row.update(archive=state_path.name, archive_sha256=checkpoint["archive_sha256"])
            rows.append(row)
            write_progress(output / "trajectory.progress.json", progress)
            print({key: row[key] for key in ("step", "time", "nodal_min", "nodal_max")}, flush=True)

        transient_started = time.perf_counter()
        result = solve_darcy_transport(
            darcy,
            times,
            velocity_representation="primal",
            permeability_gradient=0.0,
            skeleton=transport_space,
            degree=3,
            initial=0.0,
            source=0.0,
            dirichlet=1.0,
            diffusive_flux=diffusive,
            dirichlet_enforcement="strong",
            stabilization="galerkin",
            quadrature_order=quadrature_order,
            output_steps=observations,
            on_step=observe,
            check_original=True,
        )
        transient_seconds = time.perf_counter() - transient_started
        if source_digests() != sources:
            raise ValueError("transport sources or physical input changed during acquisition")
        if result.original_residual_norms is None or result.original_rhs_norms is None:
            raise ValueError("every original time-step verification is required")
        checks = checkpoint_field(
            output / "original-equations.npz",
            {
                "times": times[1:],
                "residual_norms": result.original_residual_norms,
                "physical_rhs_norms": result.original_rhs_norms,
                "nodal_extrema": np.asarray(extrema),
            },
            {"original_criterion": 1e-10, "includes_all_steps": True},
        )
        relative = np.divide(
            result.original_residual_norms,
            result.original_rhs_norms,
            out=np.zeros_like(result.original_residual_norms),
            where=result.original_rhs_norms != 0,
        )
        record = dict(
            progress,
            status="physical trajectory acquired; independent and refinement controls pending",
            original_relative_residual_max=float(np.max(relative)),
            nodal_min_all_steps=float(np.min(np.asarray(extrema)[:, 0])),
            nodal_max_all_steps=float(np.max(np.asarray(extrema)[:, 1])),
            original_check_archive_sha256=checks["archive_sha256"],
            operator_builds=result.operator_builds,
            observation_mass=result.total_mass().tolist(),
            darcy_macro_balance_linf=float(np.max(abs(darcy.conservation_residuals()))),
            timing_seconds={
                "darcy": darcy_seconds,
                "transient": transient_seconds,
                "total": time.perf_counter() - started,
            },
            runtime={
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "native": threadpool_info(),
            },
            limitations=(
                "No positivity, H(div), fine-cell conservation, inf-sup, "
                "reference accuracy or historical-realization certificate"
            ),
        )
        write_progress(output / "trajectory.json", record)
        return record


def main() -> None:
    """Acquire the declared default case or explicit same-input resolution controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--final-time", type=float, default=7.0)
    parser.add_argument("--output-every", type=int, default=1000)
    parser.add_argument("--quadrature-order", type=int, default=8)
    args = parser.parse_args()
    record = acquire(
        args.output,
        nx=args.nx,
        ny=args.ny,
        refinement=args.refinement,
        dt=args.dt,
        final_time=args.final_time,
        output_every=args.output_every,
        quadrature_order=args.quadrature_order,
    )
    print(record["generation"], record["status"], flush=True)


if __name__ == "__main__":
    main()
