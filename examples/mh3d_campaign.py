"""Acquire original three-dimensional MH/MH2M convergence with exact physical fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.robin_3d import solve_mh_3d
from pymhm.methods.three_field_3d import solve_mh2m_3d

ROOT = Path(__file__).resolve().parents[1]


def pressure(points: np.ndarray) -> np.ndarray:
    """Return a smooth nonpolynomial pressure with nonhomogeneous Dirichlet data."""
    return 1 + points @ np.array([1.0, 2.0, 3.0]) + np.sin(np.pi * points).prod(axis=1)


def flux(points: np.ndarray) -> np.ndarray:
    """Return minus the independently differentiated pressure gradient for K=I."""
    s, c = np.sin(np.pi * points), np.cos(np.pi * points)
    gradient = np.column_stack(
        (c[:, 0] * s[:, 1] * s[:, 2], s[:, 0] * c[:, 1] * s[:, 2], s[:, 0] * s[:, 1] * c[:, 2])
    )
    return -np.array([1.0, 2.0, 3.0]) - np.pi * gradient


def source(points: np.ndarray) -> np.ndarray:
    """Return the divergence of the physical flux."""
    return 3 * np.pi**2 * np.sin(np.pi * points).prod(axis=1)


def source_hashes() -> dict[str, str]:
    """Identify the executed assembly, local maps, geometry and norm implementations."""
    paths = [
        "examples/mh3d_campaign.py",
        *(
            f"src/pymhm/{name}.py"
            for name in (
                "methods/robin_3d",
                "methods/three_field_3d",
                "methods/boundary",
                "fem/traces/pressure_3d",
                "methods/three_field",
                "_legacy/models/darcy/primal_3d",
                "fem/scalar/tetrahedron",
                "fem/scalar/tetrahedron_topology",
                "fem/scalar/triangle",
                "core/contracts",
                "linalg/linear",
            )
        ),
    ]
    return current_source_manifest(
        {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}
    )


def run(resolutions: list[int], output: Path) -> None:
    """Acquire two methods using fixed local/interface configurations at every level."""
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "problem": "original smooth unit-cube verification with K=I and nonzero Dirichlet data",
        "source_sha256": source_hashes(),
        "pressure_formula": "1+x+2y+3z+sin(pi*x)*sin(pi*y)*sin(pi*z)",
        "pressure_norm": float(np.sqrt(103 / 6 + 1 / 8 + 64 / np.pi**3)),
        "flux_norm": float(np.sqrt(14 + 3 * np.pi**2 / 8)),
        "assembly_order": 6,
        "error_orders": [6, 8],
        "rows": [],
    }
    for method in ("MH", "MH2M"):
        for n in resolutions:
            start = perf_counter()
            mesh = TetraMesh.unit_cube(n)
            with threadpool_limits(1):
                if method == "MH":
                    degree, refinement = 3, 2
                    result = solve_mh_3d(
                        mesh,
                        source=source,
                        dirichlet=pressure,
                        degree=degree,
                        local_refinement=refinement,
                        skeleton=TriangularSkeleton(mesh, degree=1),
                        quadrature_order=6,
                        robin_parameter=0.25,
                        local_refinement_precision="extended",
                        refinement_precision="extended",
                    )
                    global_unknowns = result.global_matrix.shape[0]
                    residual = result.hybrid.residual
                    trace = result.hybrid.trace
                else:
                    degree, refinement = 2, 4
                    result = solve_mh2m_3d(
                        mesh,
                        source=source,
                        dirichlet=pressure,
                        degree=degree,
                        local_refinement=refinement,
                        pressure_trace=PressureTraceSpace3D(mesh, 2),
                        flux_space=TriangularSkeleton(mesh, 2, degree=1),
                        quadrature_order=6,
                    )
                    global_unknowns = len(result.free_dofs)
                    residual = result.residual
                    trace = result.trace
                solved = perf_counter()
                errors = np.array(
                    [
                        [result.l2_error(pressure, order), result.flux_l2_error(flux, order)]
                        for order in (6, 8)
                    ]
                )
                filename = f"{method.lower()}-n{n}.npz"
                np.savez_compressed(
                    output / filename,
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                    local_points=np.array([fine.points for fine in result.local_meshes]),
                    local_cells=np.array([fine.cells for fine in result.local_meshes]),
                    pressure=np.array(result.pressure),
                    degree=degree,
                    trace=trace,
                )
                row = {
                    "method": method,
                    "resolution": n,
                    "macro_cells": len(mesh.cells),
                    "local_degree": degree,
                    "local_refinement": refinement,
                    "gamma_degree": 2 if method == "MH2M" else None,
                    "gamma_subdivisions": 1 if method == "MH2M" else None,
                    "lambda_degree": 1,
                    "lambda_subdivisions": 2 if method == "MH2M" else 1,
                    "robin_parameter": 0.25 if method == "MH" else None,
                    "refinement_precision": "extended" if method == "MH" else "double",
                    "global_unknowns": global_unknowns,
                    "pressure_l2": float(errors[-1, 0]),
                    "flux_l2": float(errors[-1, 1]),
                    "pressure_relative": float(errors[-1, 0] / record["pressure_norm"]),
                    "flux_relative": float(errors[-1, 1] / record["flux_norm"]),
                    "quadrature_change": float(np.max(abs(errors[1] - errors[0]))),
                    "macro_balance_max": float(np.max(abs(result.conservation_residuals()))),
                    "global_residual": residual,
                    "solve_seconds": solved - start,
                    "elapsed_seconds": perf_counter() - start,
                    "fields": filename,
                    "fields_sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                }
            record["rows"].append(row)
            record["source_changed_during_run"] = record["source_sha256"] != source_hashes()
            (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    if record["source_changed_during_run"]:
        raise RuntimeError("Acquisition sources changed during the MH3D campaign")


def main() -> None:
    """Parse the five-level refinement series and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/mh3d")
    args = parser.parse_args()
    if any(n < 1 for n in args.resolutions):
        parser.error("resolutions must be positive")
    run(args.resolutions, args.output)


if __name__ == "__main__":
    main()
