"""Measure published PGMHM degree pairs with macro and skeletal refinement.

Section 6.1 of Fernando et al. (2023) supplies the PDE and the pairs P0/P2
and P1/P3. The triangular single-element local problems match that section;
polygon connectivity, space-refinement partitions and alpha=0.1 are declared
choices because their complete historical numerical specification is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.mh_campaign import l_mesh
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.petrov_galerkin import PGMHMSolution, solve_pgmhm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/pgmhm"


def crisscross(resolution: int) -> TriangleMesh:
    """Split each Cartesian square into four triangles meeting at its center."""
    n = resolution
    grid = TriangleMesh.unit_square(n)
    points = grid.points.tolist()
    cells = []
    for j in range(n):
        for i in range(n):
            first = j * (n + 1) + i
            corners = [first, first + 1, first + n + 2, first + n + 1]
            center = len(points)
            points.append([(i + 0.5) / n, (j + 0.5) / n])
            cells.extend([corners[k], corners[(k + 1) % 4], center] for k in range(4))
    return TriangleMesh(points, cells)


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the analytical pressure in Section 6.1."""
    x, y = points.T
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate -Delta(p), independently of discrete derivatives."""
    return 8 * np.pi**2 * exact(points)


def exact_flux(points: np.ndarray) -> np.ndarray:
    """Return the signed physical flux -grad(p)."""
    x, y = 2 * np.pi * points.T
    return -2 * np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def norms(solution: PGMHMSolution, order: int = 10) -> dict[str, float]:
    """Keep base pressure, enriched pressure, and broken divergence norms separate."""
    p = solution.l2_error(exact, order)
    q = solution.flux_l2_error(exact_flux, order)
    ep = solution.l2_error(exact, order, enriched=True)
    eq = solution.flux_l2_error(exact_flux, order, enriched=True)
    bary, weights = triangle_quadrature(order)
    divergence = []
    for fine, field in zip(solution.local_meshes, solution.enriched_pressure, strict=True):
        dofs, _, _, _, hessian = tabulate(fine, solution.degree, bary)
        laplacian = np.einsum("tqi,ti->tq", np.trace(hessian, axis1=-2, axis2=-1), field[dofs])
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        difference = -laplacian - source(points.reshape(-1, 2)).reshape(points.shape[:2])
        divergence.append(np.sum(fine.areas[:, None] * weights * difference**2))
    div = float(np.sqrt(np.sum(divergence)))
    return {
        "pressure_l2": p,
        "flux_l2": q,
        "pressure_h1_broken": float(np.hypot(p, q)),
        "enriched_pressure_l2": ep,
        "enriched_flux_l2": eq,
        "enriched_divergence_l2": div,
        "enriched_hdiv_standard_broken": float(np.hypot(eq, div)),
    }


def diagnostics(solution: PGMHMSolution) -> dict[str, float | int]:
    """Measure the actual algebraic size and both raw/enriched macro balances."""
    return {
        "macro_cells": len(solution.local_meshes),
        "fine_cells": sum(len(mesh.cells) for mesh in solution.local_meshes),
        "global_dofs": len(solution.system.rhs),
        "algebraic_residual": solution.hybrid.residual,
        "enriched_macro_balance_max": float(max(abs(solution.conservation_residuals()))),
        "unenriched_macro_balance_max": float(
            max(abs(solution.conservation_residuals(enriched=False)))
        ),
    }


def archive(solution: PGMHMSolution, path: Path) -> None:
    """Save both broken pressures and physical flux samples with the actual macro edges."""
    data = sample_field(solution.local_meshes, solution.pressure, solution.degree, 3)
    enriched = sample_field(solution.local_meshes, solution.enriched_pressure, solution.degree, 3)
    mesh = solution.skeleton.mesh
    np.savez_compressed(
        path,
        **data,
        flux=-data["gradient"],
        enriched_pressure=enriched["values"],
        enriched_flux=-enriched["gradient"],
        macro_points=mesh.points,
        macro_faces=mesh.faces,
        trace=solution.hybrid.trace,
    )


def source_hashes() -> dict[str, str]:
    """Identify the formulation, FEM, geometry and acquisition sources actually used."""
    names = (
        "src/pymhm/methods/petrov_galerkin.py",
        "src/pymhm/_legacy/models/darcy/primal.py",
        "src/pymhm/methods/robin.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/fem/traces/scalar.py",
        "src/pymhm/fem/scalar/operators.py",
        "src/pymhm/meshes/triangle.py",
        "src/pymhm/meshes/polygonal.py",
        "src/pymhm/core/contracts.py",
        "src/pymhm/linalg/linear.py",
        "examples/pgmhm_campaign.py",
        "examples/mh_campaign.py",
    )
    return current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )


def run(output: Path, resolutions: list[int], segments: list[int], workers: int) -> None:
    """Acquire five macro resolutions and five independent skeleton partitions per pair."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    record: dict[str, Any] = {
        "reference": "Fernando et al. (2023), DOI 10.1007/s40314-023-02304-y, Section 6.1",
        "stabilization_parameter": 0.1,
        "parameter_scope": "alpha is an explicitly selected value; historical alpha is not stated",
        "assembly_quadrature": 8,
        "error_quadrature": 10,
        "hdiv_scope": (
            "ordinary broken H(div) norm, divergence weight one; not a globally conforming flux"
        ),
        "source_sha256": hashes,
        "rows": [],
    }
    for study in ("macro", "skeleton"):
        for kind in ("triangles", "L-polygons"):
            for ell in (0, 1):
                for level in resolutions if study == "macro" else segments:
                    if study == "macro":
                        mesh = (
                            TriangleMesh.unit_square(level)
                            if kind == "triangles"
                            else l_mesh(level)
                        )
                        pieces, refinement = 1, 1
                    else:
                        mesh = crisscross(2) if kind == "triangles" else l_mesh(4)
                        pieces, refinement = level, level
                    skeleton = SkeletonSpace(
                        mesh, tuple(FaceSpace.uniform(ell, pieces) for _ in mesh.faces)
                    )
                    start = perf_counter()
                    solution = solve_pgmhm(
                        mesh,
                        source=source,
                        skeleton=skeleton,
                        degree=ell + 2,
                        local_refinement=refinement,
                        stabilization_parameter=0.1,
                        quadrature_order=8,
                        backend="process" if workers > 1 else "serial",
                        workers=workers,
                    )
                    row = {
                        "study": study,
                        "mesh": kind,
                        "level": level,
                        "trace_degree": ell,
                        "local_degree": ell + 2,
                        "trace_segments": pieces,
                        "local_refinement": refinement,
                        "macro_diameter": float(max(mesh.lengths))
                        if kind == "triangles"
                        else float(np.sqrt(13) / (3 * (level if study == "macro" else 4))),
                        **diagnostics(solution),
                        **norms(solution),
                        "elapsed_seconds": perf_counter() - start,
                    }
                    record["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    if study == "macro" and level == 8 and ell == 1:
                        archive(solution, output / f"{kind}-fields.npz")
                    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    record["source_changed_during_run"] = hashes != source_hashes()
    if record["source_changed_during_run"]:
        raise RuntimeError("a numerical source changed during acquisition")
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    """Run numerical campaigns separately from the lightweight verification suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.resolutions, args.segments, args.workers)


if __name__ == "__main__":
    main()
