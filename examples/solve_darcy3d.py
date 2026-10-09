"""Measure a five-level tetrahedral MHM Darcy convergence family on the unit cube."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import tetrahedral_darcy as solve_darcy_3d
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = case_workspace()


def potential(points: np.ndarray) -> np.ndarray:
    """Evaluate the homogeneous Dirichlet sinusoidal pressure from the 3D literature case."""
    return np.sin(2 * np.pi * points).prod(axis=1)


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate minus the Laplacian of the exact sinusoidal potential."""
    return 12 * np.pi**2 * potential(points)


def flux(points: np.ndarray) -> np.ndarray:
    """Evaluate the exact physical flux minus the potential gradient."""
    sine, cosine = np.sin(2 * np.pi * points), np.cos(2 * np.pi * points)
    return (
        -2
        * np.pi
        * np.column_stack(
            (
                cosine[:, 0] * sine[:, 1] * sine[:, 2],
                sine[:, 0] * cosine[:, 1] * sine[:, 2],
                sine[:, 0] * sine[:, 1] * cosine[:, 2],
            )
        )
    )


def hashes() -> dict[str, str]:
    """Identify the runtime and driver sources that define this acquisition."""
    paths = [
        Path(__file__),
        *(
            source_file(f"src/pymhm/{name}", root=ROOT)
            for name in (
                "fem/scalar/tetrahedron.py",
                "_legacy/models/darcy/primal_3d.py",
                "core/contracts.py",
                "execution/cpu.py",
                "linalg/linear.py",
            )
        ),
    ]
    return current_source_manifest(source_identity(ROOT, paths), packages=("pymhm", "examples"))


def run(output: Path, refinement: int = 4) -> None:
    """Acquire five macro resolutions, strict algebraic checks and quadrature sensitivity."""
    initial = hashes()
    rows: list[dict[str, Any]] = []
    for n in (1, 2, 3, 4, 5):
        mesh = TetraMesh.unit_cube(n)
        skeleton = TriangularSkeleton(mesh, 2)
        begin = time.perf_counter()
        solution = solve_darcy_3d(
            mesh,
            source=source,
            degree=2,
            skeleton=skeleton,
            local_refinement=refinement,
            quadrature_order=5,
        )
        elapsed = time.perf_counter() - begin
        pressure_error = solution.l2_error(potential, order=6)
        flux_error = solution.flux_l2_error(flux, order=6)
        balance = float(np.max(np.abs(solution.conservation_residuals())))
        if balance > 1e-10 or solution.hybrid.residual > 1e-10:
            raise RuntimeError("3D conservation or condensed residual criterion failed")
        row = dict(
            n=n,
            macro_tetrahedra=len(mesh.cells),
            fine_tetrahedra=sum(len(local.cells) for local in solution.local_meshes),
            local_pressure_dofs=len(solution.pressure[0]),
            trace_dofs=skeleton.size,
            retained_modes=len(mesh.cells),
            pressure_l2_error=pressure_error,
            flux_l2_error=flux_error,
            macro_balance_max=balance,
            global_relative_residual=solution.hybrid.residual,
            solve_seconds=elapsed,
        )
        if n == 5:
            row["quadrature_order8_pressure_l2_error"] = solution.l2_error(potential, order=8)
            row["quadrature_order8_flux_l2_error"] = solution.flux_l2_error(flux, order=8)
            fields = output.with_suffix(".npz")
            np.savez_compressed(
                fields,
                macro_points=mesh.points,
                macro_cells=mesh.cells,
                local_points=np.stack([local.points for local in solution.local_meshes]),
                local_cells=np.stack([local.cells for local in solution.local_meshes]),
                pressure=np.stack(solution.pressure),
                trace=solution.hybrid.trace,
            )
            row.update(
                fields=fields.name, fields_sha256=hashlib.sha256(fields.read_bytes()).hexdigest()
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = dict(
        problem=(
            "K=I; p=sin(2pi x)sin(2pi y)sin(2pi z); f=12pi^2 p; "
            "unit cube; homogeneous weak Dirichlet"
        ),
        literature=(
            "Gomes et al., On the implementation of a scalable simulator for "
            "Multiscale Hybrid-Mixed methods, arXiv:1703.10435, section 5"
        ),
        discretization={
            "local": "continuous P2",
            "local_edge_refinement": refinement,
            "trace": "discontinuous P0 on 4 equal subtriangles per macroface",
            "assembly_quadrature_order": 5,
            "error_quadrature_order": 6,
        },
        reproduction_limit=(
            "Same PDE and polynomial families; deterministic red-refined local "
            "tetrahedra and Freudenthal macro meshes differ from the reported TetGen meshes. "
            "This is an analytical convergence study, "
            "not a reproduction of published cluster timings."
        ),
        rows=rows,
        source_sha256=initial,
        source_changed_during_run=initial != hashes(),
        timestamp_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        native_threads=1,
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("acquisition sources changed")
    output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    """Run the original pyMHM campaign without any external comparison solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/darcy3d.json")
    parser.add_argument("--refinement", type=int, default=4)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        run(args.output, args.refinement)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_darcy3d").main()
