"""Measure admissible P5/P2 MHM estimates for a localized analytical Darcy field."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.reconstruction3d_data import fields
from examples.reconstruction3d_resolution import reference_norms
from pymhm import TetraMesh, TriangularSkeleton
from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.estimators.darcy_3d import estimate_darcy_error_3d
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/tetra-pk"


def source(points: np.ndarray) -> np.ndarray:
    """Return the independently differentiated negative Laplacian of the Gaussian field."""
    return fields(points, True)[2]


def source_files() -> list[Path]:
    """Identify executed numerical owners and analytical data for reproducible acquisition."""
    names = (
        "_legacy/models/darcy/primal_3d",
        "fem/scalar/tetrahedron",
        "fem/scalar/tetrahedron_topology",
        "meshes/validation",
        "core/contracts",
        "linalg/linear",
        "execution/cpu",
        "recovery/moments_3d",
        "fem/hdiv/rt_3d",
        "fem/hdiv/family_3d",
        "meshes/mixed",
        "estimators/darcy_3d",
        "fem/conditions",
    )
    return [ROOT / f"src/pymhm/{name}.py" for name in names] + [
        Path(__file__),
        ROOT / "examples/reconstruction3d_data.py",
        ROOT / "examples/reconstruction3d_resolution.py",
    ]


def capture(
    mesh: TetraMesh,
    *,
    name: str,
    refinement: int,
    subdivisions: int,
    assembly_order: int,
    error_orders: tuple[int, int],
    exact_norms: np.ndarray,
    workers: int,
) -> dict[str, Any]:
    """Solve, estimate and archive one field with independent quadrature and balance checks."""
    started = perf_counter()
    skeleton = TriangularSkeleton(mesh, subdivisions, degree=2)
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        degree=5,
        local_refinement=refinement,
        source=source,
        quadrature_order=assembly_order,
        backend="thread",
        workers=workers,
    )
    print(f"Solved {name}: {len(mesh.cells)} macros, {skeleton.size} trace DOFs", flush=True)
    estimate = estimate_darcy_error_3d(solution, degree=2, quadrature_order=assembly_order)
    reconstructed = estimate.reconstruction
    print(f"Reconstructed and estimated {name}", flush=True)
    errors = []
    for order in error_orders:
        errors.append(
            np.array(
                [
                    solution.l2_error(lambda x: fields(x, True)[0], order),
                    solution.flux_l2_error(lambda x: fields(x, True)[1], order),
                    reconstructed.flux_l2_error(lambda x: fields(x, True)[1], order),
                    estimate.energy_error(lambda x: -fields(x, True)[1], order),
                ]
            )
        )
        print(f"Integrated {name}, order {order}: {errors[-1].tolist()}", flush=True)
    change = float(np.max(abs(errors[1] - errors[0])))
    if change > 2e-9 * max(float(np.max(errors[1])), 1):
        raise ArithmeticError(f"{name}: physical error integration is unresolved ({change})")
    continuous = max(float(np.max(abs(r))) for r in reconstructed.continuous_moment_residuals())
    normal = max(float(np.max(abs(r))) for r in reconstructed.normal_flux_residuals())
    macro = float(np.max(abs(solution.conservation_residuals())))
    fine_balance = max(float(np.max(abs(r))) for r in reconstructed.fine_conservation_residuals())
    if max(continuous, normal, macro) > 1e-9:
        raise ArithmeticError(f"{name}: canonical equilibrium is unresolved")
    arrays: dict[str, Any] = dict(
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        trace=solution.hybrid.trace,
        local_degree=5,
        trace_degree=2,
        trace_subdivisions=subdivisions,
        reconstruction_degree=2,
        rt_basis=reconstructed.family.coefficients,
        local_squared=estimate.local_squared,
    )
    for cell, fine in enumerate(solution.local_meshes):
        arrays.update(
            {
                f"points_{cell}": fine.points,
                f"cells_{cell}": fine.cells,
                f"pressure_{cell}": solution.pressure[cell],
                f"rt_flux_{cell}": reconstructed.flux[cell],
            }
        )
    path = OUTPUT / f"{name}.npz"
    np.savez_compressed(path, **arrays)
    result = dict(
        name=name,
        macro_cells=len(mesh.cells),
        trace_dofs=skeleton.size,
        fine_cells=sum(len(fine.cells) for fine in solution.local_meshes),
        local_degree=5,
        trace_degree=2,
        reconstruction_degree=2,
        local_refinement=refinement,
        trace_subdivisions=subdivisions,
        estimator_condition="k >= ell + d: 5 = 2 + 3; ell <= m <= k: 2 <= 2 <= 5",
        assembly_order=assembly_order,
        estimator_order=assembly_order,
        error_orders=error_orders,
        pressure_l2=float(errors[1][0]),
        raw_flux_l2=float(errors[1][1]),
        rt_flux_l2=float(errors[1][2]),
        energy_error=float(errors[1][3]),
        pressure_relative=float(errors[1][0] / exact_norms[0]),
        raw_flux_relative=float(errors[1][1] / exact_norms[1]),
        rt_flux_relative=float(errors[1][2] / exact_norms[1]),
        errors_by_order=[item.tolist() for item in errors],
        error_quadrature_change=change,
        indicator=estimate.total,
        effectivity=estimate.total / float(errors[1][3]),
        global_relative_residual=solution.hybrid.residual,
        continuous_equilibrium=continuous,
        normal_moments=normal,
        macro_balance=macro,
        fine_balance=fine_balance,
        terms={
            key: float(np.linalg.norm(getattr(estimate, key)))
            for key in (
                "flux_defect",
                "nonconformity",
                "divergence_defect",
                "oscillation",
            )
        },
        archive=path.name,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        basis_sha256=hashlib.sha256(np.ascontiguousarray(arrays["rt_basis"]).tobytes()).hexdigest(),
        elapsed_seconds=perf_counter() - started,
    )
    print(json.dumps(result), flush=True)
    return result


def run(suite: str, workers: int) -> None:
    """Acquire five uniform resolutions or two fixed-geometry controls without changing the PDE."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    owners = source_files()
    hashes = current_source_manifest(
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    )
    snapshot = ROOT / "build/results/tetra-pk/acquisition-sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in owners:
        (snapshot / f"{hashes[path.relative_to(ROOT).as_posix()]}-{path.name}").write_bytes(
            path.read_bytes()
        )
    norms = reference_norms()
    record: dict[str, Any] = dict(
        suite=suite,
        reference="analytical",
        local_degree=5,
        trace_degree=2,
        reconstruction_degree=2,
        spatial_dimension=3,
        source_sha256=hashes,
        exact_norms=norms.tolist(),
        workers=workers,
        native_threads_per_worker=1,
        interpretation=(
            "Original analytical Gaussian case, not a historical mesh reproduction. "
            "The L09 degree hypothesis is satisfied. Numerical quadrature is verified "
            "by two rules, not by interval certification. Continuous local test-space "
            "equilibrium does not impose one source balance on every fine tetrahedron. "
            "Uniform refinement and fixed-geometry trace controls are distinct studies."
        ),
        rows=[],
    )
    if suite == "uniform":
        cases = [
            (
                TetraMesh.unit_cube(n),
                f"uniform-{n}",
                1,
                1,
                34 if n == 1 else (18 if n == 2 else 12),
                (34, 38) if n == 1 else ((18, 22) if n == 2 else (12, 14)),
            )
            for n in range(1, 6)
        ]
    else:
        path = ROOT / "examples/data/reconstruction3d-macro.json"
        geometry = json.loads(path.read_text())
        mesh = TetraMesh(np.asarray(geometry["macro_points"]), np.asarray(geometry["macro_cells"]))
        record.update(
            fixed_macro_input=path.relative_to(ROOT).as_posix(),
            fixed_macro_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        cases = [(mesh, f"fixed-s{s}", 2, s, 12, (12, 14)) for s in (1, 2)]
    for mesh, name, refinement, subdivisions, assembly, error_orders in cases:
        record["rows"].append(
            capture(
                mesh,
                name=name,
                refinement=refinement,
                subdivisions=subdivisions,
                assembly_order=assembly,
                error_orders=error_orders,
                exact_norms=norms,
                workers=workers,
            )
        )
        if hashes != current_source_manifest(
            {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in owners
            }
        ):
            raise RuntimeError("campaign sources changed during acquisition")
        record["source_changed_during_run"] = False
        (OUTPUT / f"{suite}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("uniform", "fixed"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.suite, args.workers)
