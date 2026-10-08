"""Record RT0/RT1/RT2 MHM convergence with classical mixed references."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import rt_darcy as solve_darcy_rt
from examples.formulations.mixed_darcy import (
    conforming_rt_reference as solve_darcy_rt_conforming,
)
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity, source_label
from pymhm.meshes.triangle import TriangleMesh

ROOT = case_workspace()


def potential(points: np.ndarray) -> np.ndarray:
    """Return a sinusoidal homogeneous Dirichlet pressure on the unit square."""
    return np.sin(2 * np.pi * points).prod(axis=1)


def source(points: np.ndarray) -> np.ndarray:
    """Return the exact divergence of the Darcy flux for identity permeability."""
    return 8 * np.pi**2 * potential(points)


def flux(points: np.ndarray) -> np.ndarray:
    """Return minus the exact pressure gradient."""
    sine, cosine = np.sin(2 * np.pi * points), np.cos(2 * np.pi * points)
    return -2 * np.pi * np.column_stack((cosine[:, 0] * sine[:, 1], sine[:, 0] * cosine[:, 1]))


def digest(path: Path) -> str:
    """Identify an acquisition source or field archive by its SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Solve five refinements per RT order and preserve fine/macro conservation checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/darcy-rt.json")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = [
        Path(__file__),
        *(
            source_file(f"src/pymhm/{name}", root=ROOT)
            for name in (
                "_legacy/models/darcy/mixed_rt.py",
                "fem/hdiv/rt.py",
                "core/contracts.py",
                "linalg/linear.py",
                "meshes/triangle.py",
                "fem/scalar/operators.py",
                "fem/quadrature/material.py",
                "fem/scalar/triangle.py",
            )
        ),
    ]
    hashes = current_source_manifest(source_identity(ROOT, paths), packages=("pymhm", "examples"))
    rows = []
    with threadpool_limits(limits=1):
        for degree in (0, 1, 2):
            for n in (1, 2, 4, 8, 16):
                mesh = TriangleMesh.unit_square(n)
                mhm = solve_darcy_rt(
                    mesh, degree=degree, source=source, local_refinement=2, quadrature_order=6
                )
                classical = solve_darcy_rt_conforming(
                    TriangleMesh.unit_square(2 * n),
                    degree=degree,
                    source=source,
                    quadrature_order=6,
                )
                data = {}
                for name, solution in (("mhm", mhm), ("classical", classical)):
                    data[name] = dict(
                        pressure_l2_error=solution.l2_error(potential, order=8),
                        flux_l2_error=solution.flux_l2_error(flux, order=8),
                        divergence_l2_error=solution.divergence_l2_error(source, order=8),
                        fine_moment_residual=max(
                            float(np.max(np.abs(part)))
                            for part in solution.fine_equilibrium_residuals()
                        ),
                        global_backward_residual=solution.residual,
                    )
                    if data[name]["fine_moment_residual"] > 1e-9 or solution.residual > 1e-9:
                        raise RuntimeError("RT moment/residual validation failed")
                row = dict(
                    degree=degree, n=n, macro_triangles=2 * n * n, fine_triangles=8 * n * n, **data
                )
                if degree == 2 and n == 16:
                    path = args.output.with_suffix(".npz")
                    np.savez_compressed(
                        path,
                        macro_points=mesh.points,
                        macro_cells=mesh.cells,
                        local_points=np.stack([local.points for local in mhm.local_meshes]),
                        local_cells=np.stack([local.cells for local in mhm.local_meshes]),
                        pressure=np.stack(mhm.pressure),
                        flux=np.stack(mhm.flux),
                        classical_points=classical.local_meshes[0].points,
                        classical_cells=classical.local_meshes[0].cells,
                        classical_pressure=classical.pressure[0],
                        classical_flux=classical.flux[0],
                    )
                    row.update(fields=path.name, fields_sha256=digest(path))
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = dict(
        problem="Unit square, K=I, p=sin(2pi x)sin(2pi y), f=8pi^2 p, homogeneous pressure",
        mhm="RTm flux/DG Pm pressure locally; unsegmented Pm macroface flux; refinement2",
        classical="Globally conforming RTm/DG Pm on uniform mesh of the same fine spacing",
        analytical_reference=True,
        rows=rows,
        source_sha256=hashes,
        source_changed_during_run=any(digest(p) != hashes[source_label(p, ROOT)] for p in paths),
        timestamp_utc=datetime.now(UTC).isoformat(),
        native_threads=1,
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("acquisition sources changed")
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_darcy_rt").main()
