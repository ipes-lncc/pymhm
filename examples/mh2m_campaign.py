"""Reproduce the smooth MH2M spaces and measure independent flux-space enrichment.

The smooth problem and degrees follow Section 8.1 of arXiv:2404.16978v3.
The oscillatory experiment uses equation (61) with explicitly selected gamma=1
and the smooth problem's source, since Section 8.2 does not state these values.
The conforming references are separate P1 assemblies using shared FEM kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.formulations.application import three_field_diffusion as solve_mh2m
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import scalar_operators
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file
from pymhm.linalg.linear import solve_linear
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import MH2MSolution

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/mh2m"


def exact(points: np.ndarray) -> np.ndarray:
    """Return the quartic pressure from the article's Section 8.1."""
    x, y = points.T
    return x * (x - 1) * y * (y - 1)


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Return the exact spatial derivative of the quartic pressure."""
    x, y = points.T
    return np.column_stack(((2 * x - 1) * y * (y - 1), x * (x - 1) * (2 * y - 1)))


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate -Delta(p) independently of any discrete operator."""
    x, y = points.T
    return -2 * x * (x - 1) - 2 * y * (y - 1)


def oscillatory(points: np.ndarray) -> np.ndarray:
    """Evaluate (61), epsilon=1/14 and the declared choice gamma=1."""
    x, y = 28 * np.pi * points.T
    return (2 + np.sin(x)) / (2 + np.cos(y)) + (2 + np.sin(y)) / (2 + np.sin(x))


def source_hashes() -> dict[str, str]:
    """Hash the actually used formulation, FEM, geometry and example sources."""
    names = [
        "src/pymhm/methods/three_field.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/fem/scalar/operators.py",
        "src/pymhm/meshes/triangle.py",
        "src/pymhm/fem/quadrature/material.py",
        "src/pymhm/linalg/linear.py",
        "examples/mh2m_campaign.py",
    ]
    return current_source_manifest(
        {
            name: hashlib.sha256((source_file(name, root=ROOT)).read_bytes()).hexdigest()
            for name in names
        },
        packages=("pymhm", "examples"),
    )


def diagnostics(result: MH2MSolution) -> dict[str, float | int]:
    """Record free/global dimensions and uncondensed physical equations."""
    return {
        "free_trace_dofs": len(result.free_dofs),
        "total_trace_dofs": result.trace_space.size,
        "macro_cells": len(result.local),
        "algebraic_residual": result.residual,
        "macro_balance_max": float(np.max(abs(result.conservation_residuals()))),
        "trace_moment_max": max(float(np.max(abs(x))) for x in result.trace_moment_residuals()),
        "local_equation_max": max(float(np.max(abs(x))) for x in result.local_equation_residuals()),
    }


def archive_p1(result: MH2MSolution, path: Path) -> dict[str, np.ndarray]:
    """Archive physical P1 cell vertices and nodal values with duplicated interfaces."""
    vertices = np.concatenate([data.mesh.points[data.mesh.cells] for data in result.local])
    pressure = np.concatenate(
        [p[data.mesh.cells] for data, p in zip(result.local, result.pressure, strict=True)]
    )
    data = {
        "vertices": vertices,
        "pressure": pressure,
        "macro_points": result.trace_space.mesh.points,
        "macro_cells": result.trace_space.mesh.cells,
        "trace": result.trace,
    }
    np.savez_compressed(path, **data)
    return data


def conforming_reference(n: int, order: int = 8) -> dict[str, np.ndarray]:
    """Solve continuous P1 diffusion with strong homogeneous boundary conditions."""
    mesh = TriangleMesh.unit_square(n)
    matrix, _, load = scalar_operators(mesh, 1, diffusion=oscillatory, source=source, order=order)
    fixed = np.unique(mesh.faces[mesh.boundary_faces])
    free = np.setdiff1d(np.arange(len(mesh.points)), fixed)
    pressure = np.zeros(len(mesh.points))
    pressure[free] = solve_linear(matrix[free][:, free], load[free])
    residual = np.linalg.norm((matrix @ pressure - load)[free]) / np.linalg.norm(load[free])
    return {
        "vertices": mesh.points[mesh.cells],
        "pressure": pressure[mesh.cells],
        "residual": np.array(residual),
        "resolution": np.array(n),
    }


def _owners(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Locate cell interiors with candidate barycentric validation, without interpolation."""
    tree = cKDTree(vertices.mean(axis=1))
    candidates = tree.query(points, k=min(12, len(vertices)))[1]
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    selected = vertices[candidates]
    inverse = np.linalg.inv((selected[:, :, 1:] - selected[:, :, :1]).swapaxes(-1, -2))
    xy = np.einsum("pqij,pqj->pqi", inverse, points[:, None] - selected[:, :, 0])
    score = np.minimum(1 - xy.sum(axis=2), xy.min(axis=2))
    choice = score.argmax(axis=1)
    if np.min(score[np.arange(len(points)), choice]) < -1e-10:
        raise ValueError("common integration cell is outside the candidate fine partitions")
    return candidates[np.arange(len(points)), choice]


def difference(
    reference: dict[str, np.ndarray], other: dict[str, np.ndarray], order: int = 6
) -> dict[str, float]:
    """Integrate P1 pressure/raw flux on a common finer triangular partition.

    Every reference triangle must lie in one triangle of the comparison mesh.
    All its vertices are checked after centroid location. Quadrature retains
    K(x) in the physical flux, and both norms use the same reference denominator.
    """
    vertices = reference["vertices"]
    owners = _owners(other["vertices"], vertices.mean(axis=1))
    bary, weights = triangle_quadrature(order)
    sums = np.zeros(4)
    for start in range(0, len(vertices), 2048):
        v = vertices[start : start + 2048]
        w = other["vertices"][owners[start : start + 2048]]
        inverse = np.linalg.inv((w[:, 1:] - w[:, :1]).swapaxes(1, 2))
        corners = np.einsum("tij,tqj->tqi", inverse, v - w[:, :1])
        if min(float(corners.min()), float((1 - corners.sum(axis=2)).min())) < -1e-10:
            raise ValueError("reference cells must form a common nested fine partition")
        points = np.einsum("qi,tia->tqa", bary, v)
        coordinates = np.einsum("tij,tqj->tqi", inverse, points - w[:, None, 0])
        local_bary = np.concatenate(
            (1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2
        )
        p = reference["pressure"][start : start + len(v)]
        other_p = other["pressure"][owners[start : start + len(v)]]
        values = p @ bary.T
        difference_p = values - np.einsum("tqi,ti->tq", local_bary, other_p)
        vr = np.linalg.inv((v[:, 1:] - v[:, :1]).swapaxes(1, 2))
        grad = np.einsum("tji,tj->ti", vr, p[:, 1:] - p[:, :1])
        grad_other = np.einsum("tji,tj->ti", inverse, other_p[:, 1:] - other_p[:, :1])
        coefficient = oscillatory(points.reshape(-1, 2)).reshape(points.shape[:2])
        flux_norm = coefficient**2 * np.sum(grad**2, axis=1)[:, None]
        flux_difference = coefficient**2 * np.sum((grad - grad_other) ** 2, axis=1)[:, None]
        areas = abs(np.linalg.det(v[:, 1:] - v[:, :1])) / 2
        for i, values_squared in enumerate(
            (difference_p**2, flux_difference, values**2, flux_norm)
        ):
            sums[i] += np.sum(areas[:, None] * weights * values_squared)
    norms = np.sqrt(sums)
    return {
        "pressure_difference_l2": float(norms[0]),
        "flux_difference_l2": float(norms[1]),
        "reference_pressure_l2": float(norms[2]),
        "reference_flux_l2": float(norms[3]),
        "pressure_relative_difference": float(norms[0] / norms[2]),
        "flux_relative_difference": float(norms[1] / norms[3]),
        "quadrature_order": order,
    }


def run(output: Path, resolutions: list[int], reference_levels: list[int]) -> None:
    """Acquire smooth rates and fixed-Gamma enrichment with declared provenance."""
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    smooth: list[dict[str, Any]] = []
    for k in (0, 1, 2):
        for n in resolutions:
            start = perf_counter()
            mesh = TriangleMesh.unit_square(n)
            result = solve_mh2m(
                mesh,
                source=source,
                pressure_trace=PressureTraceSpace.uniform(mesh, k + 1),
                flux_space=SkeletonSpace(mesh, tuple(FaceSpace.uniform(k) for _ in mesh.faces)),
                degree=k + 1,
                local_refinement=2 if k == 1 else 1,
                quadrature_order=6,
            )
            row = {
                "k": k,
                "resolution": n,
                "macro_diameter": np.sqrt(2) / n,
                "local_refinement": 2 if k == 1 else 1,
                **diagnostics(result),
                "pressure_error_l2": result.l2_error(exact, 8),
                "gradient_error_l2": result.gradient_l2_error(exact_gradient, 8),
                "elapsed_seconds": perf_counter() - start,
            }
            row["pressure_relative_error"] = 30 * row["pressure_error_l2"]
            row["gradient_relative_error"] = np.sqrt(45) * row["gradient_error_l2"]
            smooth.append(row)
            print(json.dumps(row), flush=True)
            if n == 8:
                data = sample_field(
                    tuple(local.mesh for local in result.local), result.pressure, k + 1, 3
                )
                np.savez_compressed(
                    output / f"smooth-k{k}.npz",
                    **data,
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                )
    references, previous, reference = [], None, None
    for n in reference_levels:
        reference = conforming_reference(n)
        row = {"resolution": n, "dofs": (n + 1) ** 2, "residual": float(reference["residual"])}
        if previous is not None:
            row.update(difference(reference, previous))
        references.append(row)
        np.savez_compressed(output / f"reference-n{n}.npz", **reference)
        previous = reference
        print(json.dumps(row), flush=True)
    enrichment = []
    if reference is not None:
        mesh = TriangleMesh.unit_square(4)
        for segments in (1, 2, 4, 8, 16):
            start = perf_counter()
            result = solve_mh2m(
                mesh,
                permeability=oscillatory,
                source=source,
                flux_space=SkeletonSpace(
                    mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces)
                ),
                degree=1,
                local_refinement=32,
                quadrature_order=8,
            )
            data = archive_p1(result, output / f"enriched-lambda{segments}.npz")
            row = {
                "flux_segments": segments,
                "pressure_segments": 1,
                "macro_resolution": 4,
                "local_refinement": 32,
                **diagnostics(result),
                **difference(reference, data),
                "elapsed_seconds": perf_counter() - start,
            }
            if segments == 16:
                row["quadrature_check"] = difference(reference, data, 8)
            enrichment.append(row)
            print(json.dumps(row), flush=True)
    if source_hashes() != hashes:
        raise RuntimeError("acquisition sources changed during the numerical campaign")
    record = {
        "paper": "de Barros, Madureira, Valentin, arXiv:2404.16978v3 (2026-08-05)",
        "smooth": smooth,
        "references": references,
        "enrichment": enrichment,
        "oscillatory_convention": {
            "gamma": 1,
            "epsilon": "1/14",
            "source": "-2*x*(x-1)-2*y*(y-1)",
            "boundary": "homogeneous Dirichlet",
            "attribution": (
                "Equation (61); gamma and repeated source explicitly selected, "
                "not identified historical data"
            ),
        },
        "source_hashes": hashes,
        "source_changed_during_run": False,
    }
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    """Run a separately reproducible campaign, outside the portable CI suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--reference-levels", type=int, nargs="+", default=[64, 128, 256])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.resolutions, args.reference_levels)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.mh2m_campaign").main()
