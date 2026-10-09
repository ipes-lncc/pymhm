"""Acquire analytical AFW3D convergence and bulk-modulus controls with independent norms."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import (
    weak_stress_elasticity as solve_elasticity_mixed_3d,
)
from examples.mixed_elasticity3d_data import SolenoidalElasticity3D
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity, source_label
from pymhm.meshes.hexahedron import cube_quadrature
from pymhm.meshes.mixed import AffineMixedMesh

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/mixed-elasticity3d"


def exact_norms(data: SolenoidalElasticity3D, order: int) -> dict[str, float]:
    """Integrate exact physical norms on an independent tensor-Gauss cube rule."""
    points, weights = cube_quadrature(order, 3)
    fields = (data.displacement(points), data.stress(points), data.rotation(points))
    return dict(
        zip(
            ("displacement_l2", "stress_l2", "rotation_l2"),
            [
                float(np.sqrt(weights @ np.sum(field.reshape(len(points), -1) ** 2, axis=1)))
                for field in fields
            ],
            strict=True,
        )
    )


def acquire(
    n: int,
    degree: int,
    refinement: int,
    lam: float,
    workers: int,
    data: SolenoidalElasticity3D,
    norms: dict[str, float],
    archive: bool,
) -> dict[str, Any]:
    """Solve one physical case and preserve fields only after residual/quadrature checks."""
    started = perf_counter()
    mesh = AffineMixedMesh.unit_cube(n)
    assembly_order = 18 if n == 1 else 12 if n == 2 else 9
    error_orders = (18, 22) if n == 1 else (14, 16) if n == 2 else (11, 13)
    solution = solve_elasticity_mixed_3d(
        mesh,
        stress_degree=degree,
        trace_degree=degree - 1,
        local_refinement=refinement,
        lame_lambda=lam,
        source=data.source,
        quadrature_order=assembly_order,
        backend="thread",
        workers=workers,
    )
    print(f"Solved n={n}, BDM{degree}, r={refinement}, lambda={lam}", flush=True)
    errors = [
        solution.errors(data.displacement, data.stress, data.rotation, order=q)
        for q in error_orders
    ]
    change = max(abs(errors[1][key] - errors[0][key]) for key in norms)
    if change > 2e-9 * max(1.0, *errors[1].values()):
        raise ArithmeticError("analytical AFW3D error quadrature is unresolved")
    force = max(float(np.max(abs(a))) for a in solution.fine_force_residuals())
    symmetry = max(float(np.max(abs(a))) for a in solution.weak_symmetry_residuals())
    traction = max(float(np.max(abs(a))) for a in solution.normal_traction_residuals())
    if max(force, symmetry, traction) > 1e-9:
        raise ArithmeticError(
            "AFW3D force, weak symmetry or physical traction moments are unresolved"
        )
    row = dict(
        n=n,
        stress_degree=degree,
        displacement_degree=degree - 1,
        rotation_degree=degree - 1,
        trace_degree=degree - 1,
        local_refinement=refinement,
        trace_subdivisions=1,
        lame_lambda="infinity" if np.isinf(lam) else lam,
        lame_mu=1.0,
        macro_cells=len(mesh.cells),
        fine_cells=sum(len(f.cells) for f in solution.local_meshes),
        trace_dofs=solution.skeleton.size,
        retained_rigid_dofs=6 * len(mesh.cells),
        errors=errors[-1],
        relative_errors={key: errors[-1][key] / norms[key] for key in norms},
        errors_by_order=errors,
        error_orders=list(error_orders),
        assembly_order=assembly_order,
        error_quadrature_change=change,
        global_relative_residual=solution.hybrid.residual,
        force_moment_max=force,
        weak_symmetry_moment_max=symmetry,
        normal_traction_moment_max=traction,
        elapsed_seconds=perf_counter() - started,
    )
    if archive:
        arrays = dict(
            macro_points=mesh.points,
            macro_cells=mesh.cells,
            stress_degree=degree,
            basis=solution.family.coefficients,
            trace=solution.hybrid.trace,
        )
        for cell, fine in enumerate(solution.local_meshes):
            arrays.update(
                {
                    f"points_{cell}": fine.points,
                    f"cells_{cell}": fine.cells,
                    f"stress_{cell}": solution.stress[cell],
                    f"displacement_{cell}": solution.displacement[cell],
                    f"rotation_{cell}": solution.rotation[cell],
                }
            )
        path = OUTPUT / f"bdm{degree}-n{n}.npz"
        np.savez_compressed(path, **arrays)
        row.update(
            archive=path.name,
            archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            basis_sha256=hashlib.sha256(
                np.ascontiguousarray(solution.family.coefficients).tobytes()
            ).hexdigest(),
        )
    return row


def run(workers: int) -> None:
    """Record two five-level sequences and one fixed-mesh lambda-to-infinity comparison."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    owners = [
        source_file(f"src/pymhm/{name}.py", root=ROOT)
        for name in (
            "_legacy/models/elasticity/stress_3d",
            "_legacy/models/elasticity/stress_forms_3d",
            "_legacy/models/elasticity/boundary",
            "fem/hdiv/family_3d",
            "fem/hdiv/moments_3d",
            "meshes/mixed",
            "_legacy/models/darcy/hdiv_3d",
            "core/contracts",
            "linalg/linear",
            "execution/cpu",
        )
    ]
    owners += [Path(__file__), source_file("examples/mixed_elasticity3d_data.py", root=ROOT)]
    hashes = current_source_manifest(source_identity(ROOT, owners), packages=("pymhm", "examples"))
    snapshot = ROOT / "build/results/mixed-elasticity3d/acquisition-sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in owners:
        (snapshot / f"{hashes[source_label(p, ROOT)]}-{p.name}").write_bytes(p.read_bytes())
    data = SolenoidalElasticity3D()
    norms, control = exact_norms(data, 18), exact_norms(data, 22)
    if max(abs(norms[k] - control[k]) for k in norms) > 2e-11:
        raise ArithmeticError("exact-field tensor quadrature is unresolved")
    record = dict(
        reference="Arnold-Falk-Winther2007 Eq7.1; original 3D MHM integration",
        convention=(
            "Cauchy stress, -div(sigma)=f, multiplier -sigma*n, axial skew-gradient rotation"
        ),
        exact_solution="curl((0,0,prod_i sin(pi*x_i)**2))",
        source="-mu*Laplacian(u)",
        source_independent_of_lambda=True,
        boundary="zero displacement on the unit cube",
        exact_norms=control,
        exact_norm_orders=[18, 22],
        source_sha256=hashes,
        convergence=[],
        locking=[],
    )
    jobs = [("convergence", n, k, r, np.inf) for k, r in ((2, 2), (3, 1)) for n in range(1, 6)]
    jobs += [("locking", 2, 2, 2, lam) for lam in (0.0, 1.0, 1e2, 1e4, 1e6, 1e8, np.inf)]
    for group, n, k, r, lam in jobs:
        row = acquire(n, k, r, lam, workers, data, control, group == "convergence")
        if hashes != current_source_manifest(
            source_identity(ROOT, owners), packages=("pymhm", "examples")
        ):
            raise RuntimeError("AFW3D acquisition sources changed during the run")
        record[group].append(row)
        record["source_changed_during_run"] = False
        (OUTPUT / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)


def main() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    with threadpool_limits(1):
        run(parser.parse_args().workers)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_mixed_elasticity3d").main()
