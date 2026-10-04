"""Verify nearly incompressible elasticity and render exact-field comparisons.

Run ``pixi run -e notebooks python examples/plot_elasticity.py --workers 4``.
Use ``--reuse-results`` to redraw the archived results without solving a PDE.
Importing the field helpers preserves the caller's Matplotlib backend; the
command-line entry point selects Agg for file rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from elasticity_data import TrigonometricElasticityData
from plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.elasticity.mixed_pressure import (
    ElasticitySolution,
    solve_displacement_pressure,
)
from pymhm._legacy.models.vector import solve_elasticity
from pymhm.fem.scalar.triangle import nodal_space, reference_basis, tabulate
from pymhm.io.provenance import current_source_manifest
from pymhm.linalg.linear import LinearSolveError

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/results/elasticity.json"
FIELDS = ROOT / "examples/results/elasticity-fields.npz"
FIGURES = ROOT / "docs/figures/elasticity"
METHODS = {"gals-p1": ("gals", 1, 4), "gals-p2": ("gals", 2, 2), "th-p2": ("taylor-hood", 2, 2)}
LABELS = {
    "gals-p1": "GaLS P1/P1",
    "gals-p2": "GaLS P2/P2",
    "th-p2": "Taylor–Hood P2/P1",
    "primal": "Primal P1",
}
COLORS = {"gals-p1": "#097c83", "gals-p2": "#3367b5", "th-p2": "#a53f7f", "primal": "#9b5b25"}


def skeleton(mesh: TriangleMesh, segments: int = 1) -> SkeletonSpace:
    """Use discontinuous linear traction polynomials on each macroface segment."""
    return SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, segments) for _ in mesh.faces), 2)


def metrics(result: ElasticitySolution, exact: TrigonometricElasticityData) -> dict[str, float]:
    """Measure physical fields using independent order-ten error quadrature."""
    return {
        "displacement_l2": result.l2_error(exact.displacement, 10),
        "pressure_l2": result.pressure_l2_error(exact.pressure, 10),
        "gradient_l2": result.h1_seminorm_error(exact.gradient, 10),
        "stress_l2": result.stress_l2_error(exact.stress, 10),
        "compressibility_l2": result.compressibility_l2(10),
        "algebraic_residual": result.hybrid.residual,
        "max_gauge_multiplier": float(np.max(np.abs(result.hybrid.gauge_multipliers), initial=0)),
        "stabilization_min": min(result.stabilization),
        "stabilization_max": max(result.stabilization),
    }


def snapshot_hashes() -> dict[str, str]:
    """Record the public numerical and analytical sources used for acquisition."""
    paths = [
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        Path(__file__).resolve(),
        ROOT / "examples/elasticity_data.py",
    ]
    return current_source_manifest(
        {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def write_report(report: dict[str, Any]) -> None:
    """Persist each completed case as strict JSON, retaining partial run progress."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def field_arrays(
    result: ElasticitySolution, exact: TrigonometricElasticityData
) -> dict[str, np.ndarray]:
    """Sample each broken finite element separately without merging macro traces."""
    template = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, 3)
    bary = np.column_stack((1 - template.points.sum(axis=1), template.points))
    points, cells, displacement, pressure, stress = [], [], [], [], []
    offset = 0
    for macro, mesh in enumerate(result.local_meshes):
        dofs, _, basis, _, _ = tabulate(mesh, result.degree, bary)
        pdofs, _, pbasis, _, _ = tabulate(mesh, result.pressure_degree, bary)
        coordinates = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        points.append(coordinates.reshape(-1, 2))
        cells.extend(template.cells + offset + len(bary) * i for i in range(len(mesh.cells)))
        displacement.append(
            np.einsum("qi,tia->tqa", basis, result.values[macro][dofs]).reshape(-1, 2)
        )
        pressure.append((result.pressure[macro][pdofs] @ pbasis.T).ravel())
        stress.append(result.stress(macro, bary).reshape(-1, 2, 2))
        offset += len(mesh.cells) * len(bary)
    coordinates = np.concatenate(points)
    return {
        "points": coordinates,
        "cells": np.concatenate(cells),
        "displacement": np.concatenate(displacement),
        "pressure": np.concatenate(pressure),
        "stress": np.concatenate(stress),
        "exact_displacement": exact.displacement(coordinates),
        "exact_pressure": exact.pressure(coordinates),
        "exact_stress": exact.stress(coordinates),
        "macro_points": result.skeleton.mesh.points,
        "macro_cells": result.skeleton.mesh.cells,
    }


def local_profile(result: ElasticitySolution, cell: int, points: np.ndarray) -> np.ndarray:
    """Evaluate displacement and pressure from one specified side of a macroface."""
    mesh = result.local_meshes[cell]
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    coordinates = np.einsum("tij,tqj->tqi", inverse, points[None] - vertices[:, None, 0])
    bary = np.concatenate((1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2)
    selected = np.argmax(bary.min(axis=2), axis=0)
    point_bary = bary[selected, np.arange(len(points))]
    if np.min(point_bary) < -2e-10:
        raise ValueError("profile point lies outside the requested macrocell")
    dofs, _ = nodal_space(mesh, result.degree)
    basis = reference_basis(result.degree, point_bary)[0]
    displacement = np.einsum("qi,qia->qa", basis, result.values[cell][dofs[selected]])
    pdofs, _ = nodal_space(mesh, result.pressure_degree)
    pressure = np.sum(
        reference_basis(result.pressure_degree, point_bary)[0]
        * result.pressure[cell][pdofs[selected]],
        axis=1,
    )
    return np.column_stack((displacement, pressure))


def profile_arrays(
    result: ElasticitySolution, exact: TrigonometricElasticityData
) -> dict[str, np.ndarray]:
    """Preserve one-sided values at every true macro crossing of y=0.37."""
    mesh = result.skeleton.mesh
    start, end = np.array([0.0, 0.37]), np.array([1.0, 0.37])
    breaks = macro_profile_breaks(mesh, start, end)
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    parameters, numerical, expected = [], [], []
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        midpoint = start + (left + right) / 2 * (end - start)
        coordinates = np.einsum("tij,tj->ti", inverse, midpoint - vertices[:, 0])
        bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
        cell = int(np.argmax(bary.min(axis=1)))
        parameter = np.linspace(left, right, 61)
        points = start + parameter[:, None] * (end - start)
        parameters.append(parameter)
        numerical.append(local_profile(result, cell, points))
        expected.append(np.column_stack((exact.displacement(points), exact.pressure(points))))
    return {
        "profile_parameter": np.stack(parameters),
        "profile_numerical": np.stack(numerical),
        "profile_exact": np.stack(expected),
        "profile_breaks": breaks,
    }


def generate(workers: int) -> dict[str, Any]:
    """Run bounded-load Poisson sweeps and six independent skeleton/fine refinements."""
    mesh = TriangleMesh.unit_square(4)
    report: dict[str, Any] = {
        "case": "consistent MSL trigonometric elasticity family",
        "pressure_amplitude": 1.0,
        "lame_mu": 1.0,
        "macro_resolution": 4,
        "trace_degree": 1,
        "trace_continuous": False,
        "assembly_quadrature": 8,
        "error_quadrature": 10,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "source_sha256": snapshot_hashes(),
        "poisson_sweep": [],
        "refinement": [],
        "evidence": "Exact manufactured fields; no external solver executed by this script.",
    }
    backend = "process" if workers > 1 else "serial"
    options = {"backend": backend, "workers": workers, "quadrature_order": 8}
    poisson = [0.2, 0.3, 0.4, 0.49, 0.499, 0.4999, 0.49999]
    lam_values = [2 * nu / (1 - 2 * nu) for nu in poisson] + [1e8, np.inf]
    for index, lam in enumerate(lam_values):
        exact = TrigonometricElasticityData(lam)
        for name, (formulation, degree, refinement) in METHODS.items():
            result = solve_displacement_pressure(
                mesh,
                lame_lambda=lam,
                source=exact.source,
                formulation=formulation,
                degree=degree,
                local_refinement=refinement,
                **options,
            )
            row = {
                "method": name,
                "index": index,
                "lame_ratio": None if np.isinf(lam) else lam,
                "poisson_ratio": 0.5 if np.isinf(lam) else lam / (2 * (lam + 1)),
                "local_refinement": refinement,
                "status": "success",
                **metrics(result, exact),
            }
            report["poisson_sweep"].append(row)
            print(json.dumps(row), flush=True)
            write_report(report)
        if np.isfinite(lam):
            row = {"method": "primal", "index": index, "lame_ratio": lam, "local_refinement": 4}
            try:
                primal = solve_elasticity(
                    mesh,
                    formulation="primal",
                    lame_lambda=lam,
                    source=exact.source,
                    skeleton=skeleton(mesh),
                    local_refinement=4,
                )
                row.update(
                    status="success", displacement_l2=primal.l2_error(exact.displacement, 10)
                )
            except LinearSolveError as error:
                row.update(status="residual_or_rank_failure", reason=str(error))
            report["poisson_sweep"].append(row)
            print(json.dumps(row), flush=True)
            write_report(report)
    exact = TrigonometricElasticityData(4999.0)
    for name in ("gals-p1", "gals-p2"):
        formulation, degree, minimum = METHODS[name]
        for segments in (1, 2, 3, 4, 6, 8):
            refinement = minimum * segments
            result = solve_displacement_pressure(
                mesh,
                lame_lambda=exact.lame_lambda,
                source=exact.source,
                formulation=formulation,
                degree=degree,
                local_refinement=refinement,
                skeleton=skeleton(mesh, segments),
                **options,
            )
            row = {
                "method": name,
                "segments": segments,
                "lame_ratio": exact.lame_lambda,
                "local_refinement": refinement,
                "skeleton_size": float(max(mesh.lengths) / segments),
                **metrics(result, exact),
            }
            report["refinement"].append(row)
            print(json.dumps(row), flush=True)
            if name == "gals-p2" and segments == 2:
                arrays = field_arrays(result, exact) | profile_arrays(result, exact)
                np.savez_compressed(FIELDS, **arrays)
                report["field_case"] = row | {
                    "file": FIELDS.name,
                    "sha256": hashlib.sha256(FIELDS.read_bytes()).hexdigest(),
                }
            write_report(report)
    for name in ("gals-p1", "gals-p2"):
        rows = [row for row in report["refinement"] if row["method"] == name]
        if not rows[-1]["displacement_l2"] < rows[0]["displacement_l2"]:
            raise ValueError("the resolved displacement did not improve under refinement")
    report["source_sha256_end"] = snapshot_hashes()
    report["changed_source_files"] = [
        name
        for name, digest in report["source_sha256"].items()
        if report["source_sha256_end"].get(name) != digest
    ]
    report["source_changed_during_acquisition"] = bool(report["changed_source_files"])
    write_report(report)
    return report


def save(figure: Any, name: str) -> None:
    """Save compact publication figures in vector and raster formats."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_sweeps(report: dict[str, Any]) -> None:
    """Compare method errors along the finite-to-incompressible material sweep."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for axis, quantity, title in zip(
        axes,
        ("displacement_l2", "pressure_l2", "stress_l2"),
        ("Displacement L² error", "Herrmann pressure L² error", "Full Cauchy stress L² error"),
        strict=True,
    ):
        for method in LABELS:
            rows = [
                row
                for row in report["poisson_sweep"]
                if row["method"] == method and row["status"] == "success" and quantity in row
            ]
            if rows:
                axis.semilogy(
                    [row["index"] for row in rows],
                    [row[quantity] for row in rows],
                    "o-",
                    color=COLORS[method],
                    label=LABELS[method],
                    markersize=4,
                )
        axis.set(title=title, xlabel="λ/μ (categorical; last point is the limit)")
        axis.set_xticks(
            range(9), ["⅔", "1.5", "4", "49", "499", "4999", "49999", "10⁸", "∞"], rotation=45
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    failures = [
        row
        for row in report["poisson_sweep"]
        if row["method"] == "primal" and row["status"] != "success"
    ]
    if failures:
        axes[0].text(
            0.97,
            0.58,
            "Primal guard rejected\n"
            + ", ".join(f"λ/μ={row['lame_ratio']:.3g}" for row in failures),
            transform=axes[0].transAxes,
            fontsize=8,
            ha="right",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    figure.suptitle("Fixed macro mesh: 32 triangles; linear traction on each macroface")
    save(figure, "material-sweep")


def plot_refinement(report: dict[str, Any]) -> None:
    """Show six measured resolutions with separate pressure and stress norms."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, quantity, title in zip(
        axes.flat,
        ("displacement_l2", "gradient_l2", "pressure_l2", "stress_l2"),
        (
            "Displacement L² error",
            "Broken gradient L² error",
            "Herrmann pressure L² error",
            "Full Cauchy stress L² error",
        ),
        strict=True,
    ):
        for method in ("gals-p1", "gals-p2"):
            rows = [row for row in report["refinement"] if row["method"] == method]
            axis.loglog(
                [row["skeleton_size"] for row in rows],
                [row[quantity] for row in rows],
                "o-",
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.set(title=title, xlabel="Maximum skeleton segment length")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.suptitle("ν=0.4999; fixed 32 macrotriangles; s=1, 2, 3, 4, 6, 8")
    save(figure, "refinement")


def plot_fields(report: dict[str, Any]) -> None:
    """Compare sampled exact and broken finite-element fields on the real macro mesh."""
    data = np.load(FIELDS)
    mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
    triangulation = mtri.Triangulation(*data["points"].T, triangles=data["cells"])
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, name in enumerate(("displacement", "pressure", "stress")):
        actual, exact = data[name], data[f"exact_{name}"]
        if name == "pressure":
            plotted = (exact, actual, np.abs(actual - exact))
            lower, upper = min(exact.min(), actual.min()), max(exact.max(), actual.max())
        else:
            dimensions = tuple(range(1, actual.ndim))
            plotted = tuple(
                np.sqrt(np.sum(value**2, axis=dimensions))
                for value in (exact, actual, actual - exact)
            )
            lower, upper = 0.0, max(plotted[0].max(), plotted[1].max())
        symbols = {
            "displacement": (r"$|u|$", r"$|u_h|$", r"$|u-u_h|$"),
            "pressure": (r"$p$", r"$p_h$", r"$|p-p_h|$"),
            "stress": (r"$\|\sigma\|_F$", r"$\|\sigma_h\|_F$", r"$\|\sigma-\sigma_h\|_F$"),
        }[name]
        titles = (
            f"Exact {symbols[0]}",
            f"GaLS P2/P2 {symbols[1]}",
            f"Pointwise error {symbols[2]}",
        )
        for column, (values, title) in enumerate(zip(plotted, titles, strict=True)):
            axis = axes[row, column]
            artist = axis.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                rasterized=True,
                cmap="magma" if column == 2 else ("coolwarm" if name == "pressure" else "viridis"),
                vmin=0 if column == 2 else lower,
                vmax=None if column == 2 else upper,
            )
            draw_macro_mesh(axis, mesh)
            figure.colorbar(artist, ax=axis, shrink=0.82)
            axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    figure.suptitle("ν=0.4999; two traction segments per macroface; local refinement 4")
    save(figure, "fields")
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    parameter = data["profile_parameter"]
    for component, (axis, label) in enumerate(zip(axes, ("uₓ", "uᵧ", "p"), strict=True)):
        for segment, x in enumerate(parameter):
            axis.plot(
                x,
                data["profile_exact"][segment, :, component],
                color="#222222",
                linewidth=1.5,
                label="Exact" if segment == 0 else None,
            )
            axis.plot(
                x,
                data["profile_numerical"][segment, :, component],
                color=COLORS["gals-p2"],
                linewidth=1.2,
                label="GaLS P2/P2" if segment == 0 else None,
            )
        mark_macro_interfaces(axis, data["profile_breaks"][1:-1], label=True)
        axis.set(title=label, xlabel="x at y=0.37")
        axis.grid(True, alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("One-sided profiles retain jumps at real macro interfaces")
    save(figure, "profiles")


def main() -> None:
    """Generate original numerical results or render the existing public dataset."""
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    with threadpool_limits(limits=1):
        report = json.loads(RESULTS.read_text()) if args.reuse_results else generate(args.workers)
        plot_sweeps(report)
        plot_refinement(report)
        plot_fields(report)


if __name__ == "__main__":
    main()
