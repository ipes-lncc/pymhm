"""Acquire uniform and adaptive tetrahedral MHM reconstruction and estimator evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.reconstruction3d_data import fields
from pymhm import TetraMesh
from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.adaptivity.darcy_3d import solve_adaptive_darcy_3d
from pymhm.estimators.darcy_3d import estimate_darcy_error_3d
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/reconstruction3d"


def run(suite: str) -> None:
    """Persist each completed state with independently integrated errors and exact-field data."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    localized = suite == "adaptive"
    modules = [
        "recovery/moments_3d",
        "estimators/darcy_3d",
        "fem/conditions",
        "fem/hdiv/rt_3d",
        "_legacy/models/darcy/primal_3d",
        "meshes/refinement_3d",
        "adaptivity/darcy_3d",
        "fem/scalar/tetrahedron",
        "fem/scalar/tetrahedron_topology",
        "core/contracts",
        "linalg/linear",
    ]
    owners = [ROOT / f"src/pymhm/{name}.py" for name in modules] + [
        Path(__file__),
        ROOT / "examples/reconstruction3d_data.py",
    ]
    hashes = current_source_manifest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    )
    report: dict[str, Any] = dict(
        suite=suite,
        reference="analytical",
        source_sha256=hashes,
        local_degree=3,
        local_refinement=2,
        trace_degree=0,
        reconstruction_degree=1,
        assembly_order=12,
        error_orders=[12, 14],
        rows=[],
    )
    exact_norms = []
    for order in (32, 40):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        points = np.stack(np.meshgrid(*([(nodes + 1) / 2] * 3), indexing="ij"), axis=-1)
        product_weights = np.prod(
            np.stack(np.meshgrid(*([weights / 2] * 3), indexing="ij")), axis=0
        )
        exact_p, exact_q, _ = fields(points.reshape(-1, 3), localized)
        exact_norms.append(
            np.sqrt(
                product_weights.ravel() @ np.column_stack((exact_p**2, np.sum(exact_q**2, axis=1)))
            )
        )
    if np.max(abs(exact_norms[1] - exact_norms[0])) > 2e-13:
        raise ArithmeticError("analytical norm quadrature is unresolved")
    report["exact_norms"] = exact_norms[-1].tolist()
    started = perf_counter()

    def pressure(x: np.ndarray) -> np.ndarray:
        """Evaluate the analytical pressure without numerical differentiation."""
        return fields(x, localized)[0]

    def flux(x: np.ndarray) -> np.ndarray:
        """Evaluate the independent analytical physical vector flux."""
        return fields(x, localized)[1]

    def source(x: np.ndarray) -> np.ndarray:
        """Evaluate the independently differentiated negative Laplacian."""
        return fields(x, localized)[2]

    def capture(step: int, solution: Any, estimate: Any) -> None:
        """Save quadrature-checked errors, conservation diagnostics and replayable coefficients."""
        q = estimate.reconstruction
        errors = []
        for order in (12, 14):
            errors.append(
                np.array(
                    [
                        solution.l2_error(pressure, order),
                        solution.flux_l2_error(flux, order),
                        q.flux_l2_error(flux, order),
                        estimate.energy_error(lambda x: -flux(x), order),
                    ]
                )
            )
        difference = float(np.max(abs(errors[1] - errors[0])))
        if difference > 2e-9 * max(float(np.max(errors[1])), 1):
            raise ArithmeticError("independent error quadrature is unresolved")
        row = dict(
            step=step,
            macro_cells=len(solution.local_meshes),
            trace_dofs=solution.skeleton.size,
            fine_cells=sum(len(f.cells) for f in solution.local_meshes),
            pressure_l2=errors[1][0],
            raw_flux_l2=errors[1][1],
            rt_flux_l2=errors[1][2],
            pressure_relative=errors[1][0] / exact_norms[-1][0],
            raw_flux_relative=errors[1][1] / exact_norms[-1][1],
            rt_flux_relative=errors[1][2] / exact_norms[-1][1],
            energy_error=errors[1][3],
            indicator=estimate.total,
            effectivity=estimate.total / errors[1][3],
            error_quadrature_change=difference,
            continuous_equilibrium=max(
                float(np.max(abs(r))) for r in q.continuous_moment_residuals()
            ),
            fine_balance=max(float(np.max(abs(r))) for r in q.fine_conservation_residuals()),
            normal_moments=max(float(np.max(abs(r))) for r in q.normal_flux_residuals()),
            elapsed_seconds=perf_counter() - started,
            terms={
                name: float(np.linalg.norm(getattr(estimate, name)))
                for name in (
                    "flux_defect",
                    "nonconformity",
                    "divergence_defect",
                    "oscillation",
                )
            },
        )
        mesh = solution.skeleton.mesh
        path = OUTPUT / f"{suite}-{step}.npz"
        archive: dict[str, Any] = dict(
            macro_points=mesh.points,
            macro_cells=mesh.cells,
            trace=solution.hybrid.trace,
            local_squared=estimate.local_squared,
            rt_basis=q.family.coefficients,
            local_degree=solution.degree,
            reconstruction_degree=q.family.pressure_degree,
        )
        for cell, fine in enumerate(solution.local_meshes):
            archive.update(
                {
                    f"points_{cell}": fine.points,
                    f"cells_{cell}": fine.cells,
                    f"pressure_{cell}": solution.pressure[cell],
                    f"rt_flux_{cell}": q.flux[cell],
                }
            )
        np.savez_compressed(path, **archive)
        row.update(archive=path.name, archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report["rows"].append(row)
        (OUTPUT / f"{suite}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    options = dict(
        source=source,
        degree=3,
        local_refinement=2,
        quadrature_order=12,
        backend="thread",
        workers=4,
    )
    if localized:
        solve_adaptive_darcy_3d(
            TetraMesh.unit_cube(2),
            iterations=8,
            theta=0.5,
            maximum_cells=6000,
            estimator_order=12,
            on_state=capture,
            **options,
        )
    else:
        for n in (1, 2, 3, 4, 5):
            solution = solve_darcy_3d(TetraMesh.unit_cube(n), **options)
            estimate = estimate_darcy_error_3d(solution, quadrature_order=12)
            capture(n, solution, estimate)
    if hashes != current_source_manifest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    ):
        raise RuntimeError("campaign sources changed during acquisition")
    report["source_changed_during_run"] = False
    (OUTPUT / f"{suite}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("uniform", "adaptive"))
    with threadpool_limits(1):
        run(parser.parse_args().suite)
