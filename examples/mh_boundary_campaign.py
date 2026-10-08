"""Measure physical mixed boundaries and polygonal MH/MH2M convergence.

This original analytical experiment uses a nonconvex macro partition and exact
anisotropic data. It verifies the boundary and geometry extensions; it is not
a historical numerical table from either method's paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import robin_diffusion as solve_mh
from examples.formulations.application import three_field_diffusion as solve_mh2m
from examples.mh_campaign import l_mesh
from pymhm import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file

ROOT = case_workspace()
MATERIAL = np.array([[3.0, 0.4], [0.4, 2.0]])


def pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate a nonpolynomial pressure with nonhomogeneous boundary data."""
    x, y = points.T
    return 1 + x + 2 * y + np.sin(np.pi * x) * np.sin(np.pi * y)


def flux(points: np.ndarray) -> np.ndarray:
    """Return the physical anisotropic flux, differentiated analytically."""
    x, y = points.T
    gradient = np.column_stack(
        (
            1 + np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
            2 + np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
        )
    )
    return -gradient @ MATERIAL


def source(points: np.ndarray) -> np.ndarray:
    """Return div(q), including both off-diagonal material derivatives."""
    x, y = np.pi * points.T
    return np.pi**2 * (5 * np.sin(x) * np.sin(y) - 0.8 * np.cos(x) * np.cos(y))


def source_hashes() -> dict[str, str]:
    """Record the executed solver, geometry and boundary-integrator source bytes."""
    paths = (
        "src/pymhm/methods/robin.py",
        "src/pymhm/methods/three_field.py",
        "src/pymhm/fem/traces/integration.py",
        "src/pymhm/fem/traces/scalar.py",
        "src/pymhm/meshes/polygonal.py",
        "src/pymhm/meshes/triangle.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/core/contracts.py",
        "src/pymhm/linalg/linear.py",
        "examples/mh_campaign.py",
        "examples/mh_boundary_campaign.py",
    )
    return current_source_manifest(
        {
            path: hashlib.sha256((source_file(path, root=ROOT)).read_bytes()).hexdigest()
            for path in paths
        },
        packages=("pymhm", "examples"),
    )


def run(resolutions: list[int], output: Path) -> None:
    """Acquire five boundary/solver series with independent physical error rules."""
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "problem": "original anisotropic nonhomogeneous polygonal boundary verification",
        "material": MATERIAL.tolist(),
        "source_sha256": source_hashes(),
        "local_degree": 2,
        "local_refinement": 4,
        "mh_trace_degree": 2,
        "mh_trace_segments": 2,
        "mh2m_pressure_trace_degree": 2,
        "mh2m_pressure_trace_segments": 1,
        "mh2m_flux_degree": 1,
        "mh2m_flux_segments": 2,
        "assembly_order": 8,
        "error_orders": [8, 10],
        "mh_refinement_precision": "extended",
        "pressure_norm": float(np.sqrt(20 / 3 + 1 / 4 + 20 / np.pi**2)),
        "flux_norm": float(np.sqrt(33.8 + 3.33 * np.pi**2)),
        "exact_mean": float(2.5 + 4 / np.pi**2),
        "rows": [],
    }
    for method, boundary in (
        ("MH", "mixed"),
        ("MH", "neumann"),
        ("MH2M", "dirichlet"),
        ("MH2M", "mixed"),
        ("MH2M", "neumann"),
    ):
        for n in resolutions:
            start = perf_counter()
            mesh = l_mesh(n)
            faces = [
                int(face)
                for face in mesh.boundary_faces
                if boundary == "neumann"
                or (boundary == "mixed" and np.all(mesh.points[mesh.faces[face], 0] == 1))
            ]
            natural = {
                face: lambda x, normal=mesh.normals[face]: flux(x) @ normal for face in faces
            }
            options = {
                "permeability": MATERIAL,
                "source": source,
                "dirichlet": pressure,
                "neumann": natural,
                "mean_pressure": record["exact_mean"] if boundary == "neumann" else 0,
                "degree": 2,
                "local_refinement": 4,
                "quadrature_order": 8,
            }
            with threadpool_limits(1):
                if method == "MH":
                    result = solve_mh(
                        mesh,
                        skeleton=SkeletonSpace(
                            mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces)
                        ),
                        robin_parameter=0.25,
                        local_refinement_precision="extended",
                        refinement_precision="extended",
                        **options,
                    )
                    free = result.global_matrix.shape[0] - result.skeleton.size
                    global_size = result.global_matrix.shape[0]
                    residual = result.hybrid.residual
                else:
                    result = solve_mh2m(
                        mesh,
                        pressure_trace=PressureTraceSpace.uniform(mesh, 2),
                        flux_space=SkeletonSpace(
                            mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces)
                        ),
                        **options,
                    )
                    free = 0
                    global_size = len(result.free_dofs)
                    residual = result.residual
                errors = np.array(
                    [
                        [result.l2_error(pressure, order), result.flux_l2_error(flux, order)]
                        for order in (8, 10)
                    ]
                )
                row = {
                    "method": method,
                    "boundary": boundary,
                    "resolution": n,
                    "macro_cells": len(mesh.cells),
                    "global_unknowns_before_mean_gauge": global_size,
                    "neumann_pressure_unknowns": free,
                    "pressure_l2": float(errors[-1, 0]),
                    "flux_l2": float(errors[-1, 1]),
                    "pressure_relative": float(errors[-1, 0] / record["pressure_norm"]),
                    "flux_relative": float(errors[-1, 1] / record["flux_norm"]),
                    "quadrature_change": float(np.max(abs(errors[1] - errors[0]))),
                    "macro_balance_max": float(np.max(abs(result.conservation_residuals()))),
                    "global_residual": residual,
                    "elapsed_seconds": perf_counter() - start,
                }
            record["rows"].append(row)
            print(json.dumps(row), flush=True)
            record["source_changed_during_run"] = record["source_sha256"] != source_hashes()
            (output / "boundary-comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["source_changed_during_run"]:
        raise RuntimeError("Acquisition sources changed during the boundary campaign")


def main() -> None:
    """Parse the independent boundary-campaign mesh series."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/mh")
    args = parser.parse_args()
    if any(n < 1 for n in args.resolutions):
        parser.error("resolutions must be positive")
    run(args.resolutions, args.output)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.mh_boundary_campaign").main()
