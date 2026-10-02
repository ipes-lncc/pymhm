"""Measure the Robin MH method on the smooth problem of Barrenechea et al. (2024).

The equation and degree pairs follow Section 4.1. Triangular and L-shaped
partitions are generated explicitly here; they are not the article's archived
meshes. The vanishing-Robin comparison retains exactly the same local spaces
and skeletal spaces in MH and MHM, with genuine broken energy integration.
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

from examples.campaign_checkpoint import archive_identity, require_sources, verify_checkpoint
from examples.field_sampling import sample_field
from pymhm.darcy import solve_darcy
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import tabulate
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.mh import MHSolution, solve_mh
from pymhm.polygon import PolygonMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/mh"


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate equation (4.1), n=3 and m=7, on the unit square."""
    x, y = points.T
    return np.sin(6 * np.pi * x) * np.sin(14 * np.pi * y)


def exact_flux(points: np.ndarray) -> np.ndarray:
    """Return the independently differentiated physical flux -grad(p)."""
    x, y = points.T
    return -np.column_stack(
        (
            6 * np.pi * np.cos(6 * np.pi * x) * np.sin(14 * np.pi * y),
            14 * np.pi * np.sin(6 * np.pi * x) * np.cos(14 * np.pi * y),
        )
    )


def source(points: np.ndarray) -> np.ndarray:
    """Return -Delta(p)=232*pi**2*p, independently of finite-element operators."""
    return 232 * np.pi**2 * exact(points)


def l_mesh(n: int) -> PolygonMesh:
    """Tile n by n rectangles by two complementary nonconvex L-shaped cells.

    Collinear vertices split every polygon boundary at the common underlying
    rectangular grid. Thus adjacent macrofaces match without hanging vertices.
    Every L has diameter sqrt(13)/(3*n); no historical connectivity is inferred.
    """
    patterns = (
        ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2), (0, 1)),
        ((1, 1), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3), (0, 2), (1, 2)),
    )
    indices: dict[tuple[int, int], int] = {}
    cells = []
    for j in range(n):
        for i in range(n):
            for pattern in patterns:
                keys = [(2 * i + a, 3 * j + b) for a, b in pattern]
                for key in keys:
                    indices.setdefault(key, len(indices))
                cells.append(np.array([indices[key] for key in keys], dtype=np.int64))
    points = np.array(list(indices), dtype=float) / np.array([2 * n, 3 * n])
    return PolygonMesh(points, tuple(cells))


def source_hashes() -> dict[str, str]:
    """Identify the executed formulation, geometry, condensation and campaign sources."""
    names = (
        "src/pymhm/mh.py",
        "src/pymhm/darcy.py",
        "src/pymhm/lagrange.py",
        "src/pymhm/mesh.py",
        "src/pymhm/polygon.py",
        "src/pymhm/hybrid.py",
        "src/pymhm/solvers.py",
        "examples/mh_campaign.py",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def diagnostics(result: MHSolution) -> dict[str, float | int]:
    """Measure macro balance, physical equations and the actual global dimension."""
    return {
        "macro_cells": len(result.local_meshes),
        "global_dofs": result.skeleton.size,
        "residual": result.hybrid.residual,
        "macro_balance_max": float(np.max(abs(result.conservation_residuals()))),
        "robin_parameter": result.robin_parameter,
        "local_equation_max": max(
            float(
                np.max(
                    abs(
                        response.problem.matrix @ pressure
                        + response.problem.coupling
                        @ result.hybrid.trace[response.problem.trace_dofs]
                        - response.problem.load
                    )
                )
            )
            for response, pressure in zip(result.system.responses, result.pressure, strict=True)
        ),
    }


def energy_difference(first: MHSolution, second: Any, order: int = 10) -> float:
    """Integrate broken |grad(p_MH-p_MHM)| squared on identical fine partitions."""
    bary, weights = triangle_quadrature(order)
    pieces = []
    for fine, other, p, q in zip(
        first.local_meshes, second.local_meshes, first.pressure, second.pressure, strict=True
    ):
        if not np.array_equal(fine.points, other.points) or not np.array_equal(
            fine.cells, other.cells
        ):
            raise ValueError("the vanishing-Robin comparison requires identical fine partitions")
        dofs, _, _, gradient, _ = tabulate(fine, first.degree, bary)
        values = np.einsum("tqia,ti->tqa", gradient, (p - q)[dofs])
        pieces.append(float(np.sum(fine.areas[:, None] * weights * np.sum(values**2, axis=2))))
    return float(np.sqrt(np.sum(pieces)))


def save_fields(result: MHSolution, path: Path) -> None:
    """Archive one-sided volume samples and the actual triangular/polygonal macro edges."""
    data = sample_field(result.local_meshes, result.pressure, result.degree, 3)
    mesh = result.skeleton.mesh
    np.savez_compressed(
        path,
        **data,
        flux=-data["gradient"],
        macro_points=mesh.points,
        macro_faces=mesh.faces,
        trace=result.hybrid.trace,
    )


def run(
    output: Path, resolutions: list[int], workers: int = 1, *, resume_smooth: bool = False
) -> None:
    """Acquire two mesh families, two degree pairs, and a same-space MH-to-MHM limit."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    smooth: list[dict[str, Any]] = []
    record: dict[str, Any] = {
        "method": "Multiscale Hybrid Robin formulation",
        "reference": "Barrenechea, Gomes, Paredes (2024), DOI 10.1137/22M1542556",
        "problem": "Section 4.1, equation (4.1), n=3, m=7; K=I; homogeneous Dirichlet",
        "mesh_scope": "Explicit original triangles and L tiling; not historical connectivity",
        "assembly_quadrature": 8,
        "error_quadrature": 10,
        "smooth": smooth,
        "source_sha256": hashes,
    }
    if resume_smooth:
        record = json.loads((output / "comparison.json").read_text())
        smooth = record["smooth"]
        expected = {
            (kind, ell, n if kind == "triangles" else max(1, n // 2))
            for kind in ("triangles", "L-polygons")
            for ell in (1, 2)
            for n in resolutions
        }
        actual = {(row["mesh"], row["trace_degree"], row["resolution"]) for row in smooth}
        if actual != expected or len(smooth) != len(expected):
            raise ValueError("resume requires the complete requested smooth campaign")
        previous = record["source_sha256"]
        require_sources(previous, hashes)
        verify_checkpoint(
            record, {"assembly_quadrature": 8, "error_quadrature": 10}, directory=output
        )
        for row in smooth:
            verify_checkpoint(
                row,
                {
                    "local_degree": row["trace_degree"] + 2,
                    "local_refinement": 2,
                    "robin_parameter": 0.25,
                },
                directory=output,
                metrics=("pressure_error_l2", "flux_error_l2", "residual"),
                archive_required=row["resolution"] == 8,
            )
        record["phase_sources"] = {"smooth": previous, "vanishing_robin": hashes}
    for kind in () if resume_smooth else ("triangles", "L-polygons"):
        levels = resolutions if kind == "triangles" else [max(1, n // 2) for n in resolutions]
        for ell in (1, 2):
            for n in levels:
                start = perf_counter()
                mesh = TriangleMesh.unit_square(n) if kind == "triangles" else l_mesh(n)
                skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(ell) for _ in mesh.faces))
                result = solve_mh(
                    mesh,
                    source=source,
                    skeleton=skeleton,
                    degree=ell + 2,
                    local_refinement=2,
                    robin_parameter=0.25,
                    quadrature_order=8,
                    local_refinement_precision="extended",
                    refinement_precision="extended",
                    backend="process" if workers > 1 else "serial",
                    workers=workers,
                )
                row = {
                    "mesh": kind,
                    "resolution": n,
                    "trace_degree": ell,
                    "local_degree": ell + 2,
                    "local_refinement": 2,
                    "macro_diameter": (np.sqrt(2) if kind == "triangles" else np.sqrt(13) / 3) / n,
                    **diagnostics(result),
                    "pressure_error_l2": result.l2_error(exact, 10),
                    "flux_error_l2": result.flux_l2_error(exact_flux, 10),
                    "elapsed_seconds": perf_counter() - start,
                }
                row["pressure_relative_error"] = 2 * row["pressure_error_l2"]
                row["flux_relative_error"] = row["flux_error_l2"] / (np.pi * np.sqrt(58))
                smooth.append(row)
                print(json.dumps(row), flush=True)
                if n == 8:
                    target = output / f"{kind}-ell{ell}.npz"
                    save_fields(result, target)
                    row.update(archive_identity(target))
                (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    mesh = TriangleMesh.unit_square(4)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    reference = solve_darcy(
        mesh, source=source, skeleton=skeleton, degree=3, local_refinement=4, quadrature_order=8
    )
    limit = []
    for parameter in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
        result = solve_mh(
            mesh,
            source=source,
            skeleton=skeleton,
            degree=3,
            local_refinement=4,
            quadrature_order=8,
            robin_parameter=parameter,
            local_refinement_precision="extended",
            refinement_precision="extended",
        )
        eigenvalues = np.linalg.eigvalsh(np.asarray(result.system.matrix.toarray(), dtype=float))
        local_values = np.linalg.eigvalsh(result.system.responses[0].problem.matrix.toarray())
        row = {
            **diagnostics(result),
            "energy_difference": energy_difference(result, reference),
            "global_condition_2": float(eigenvalues[-1] / eigenvalues[0]),
            "first_local_condition_2": float(local_values[-1] / local_values[0]),
        }
        limit.append(row)
        print(json.dumps(row), flush=True)
    record["vanishing_robin"] = {
        "mesh_resolution": 4,
        "trace_degree": 1,
        "local_degree": 3,
        "local_refinement": 4,
        "mhm_pressure_error_l2": reference.l2_error(exact, 10),
        "mhm_flux_error_l2": reference.flux_l2_error(exact_flux, 10),
        "rows": limit,
    }
    record["source_changed_during_run"] = hashes != source_hashes()
    if record["source_changed_during_run"]:
        raise RuntimeError("a formulation source changed during acquisition")
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    """Run the numerical campaign separately from the lightweight test suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-smooth", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(1):
        run(args.output, args.resolutions, args.workers, resume_smooth=args.resume_smooth)


if __name__ == "__main__":
    main()
