"""Verify the analytical Darcy lifts with full source moments and Figure 5 data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.io.workspace import case_workspace

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.formulations.analytic_darcy import analytic_darcy as solve_darcy_analytic
from examples.formulations.application import darcy as solve_darcy
from examples.plot_style import set_refinement_ticks
from pymhm import TriangleMesh
from pymhm.fem.scalar.operators import rt0_evaluate, triangle_quadrature


def pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate the zero-mean exact pressure in Harder--Paredes--Valentin Section 5.1."""
    return np.prod(np.cos(2 * np.pi * points), axis=-1)


def flux(points: np.ndarray) -> np.ndarray:
    """Exact physical Darcy flux, with homogeneous normal boundary data."""
    x, y = (2 * np.pi * points).T
    return 2 * np.pi * np.column_stack((np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)))


def source(points: np.ndarray) -> np.ndarray:
    """Independently differentiated source, minus the pressure Laplacian."""
    return 8 * np.pi**2 * pressure(points)


def measure(n: int, refinement: int = 2, order: int = 10) -> dict[str, float | int]:
    """Integrate each declared pressure/flux convention without fitting ordinates."""
    start = perf_counter()
    mesh = TriangleMesh.unit_square(n)
    natural = {int(face): 0.0 for face in mesh.boundary_faces}
    result = solve_darcy_analytic(
        mesh,
        source=source,
        neumann=natural,
        source_refinement=refinement,
        quadrature_order=order,
    )
    classical = solve_darcy(
        mesh,
        formulation="mixed",
        source=source,
        neumann=natural,
        local_refinement=1,
        quadrature_order=order,
    )
    bary, weights = triangle_quadrature(order + 2)
    points = np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells])
    exact = pressure(points)
    exact_flux = flux(points.reshape(-1, 2)).reshape(points.shape)
    means = exact @ weights
    harmonic, harmonic_flux, integrated = [], [], []
    for cell, space in enumerate(result.spaces):
        p, q = result.pressure_update(cell, points[cell])
        harmonic.append(p)
        harmonic_flux.append(q)
        local, coefficients = classical.local_meshes[cell], classical.flux[cell]
        centroid_flux = rt0_evaluate(local, coefficients, np.full((1, 3), 1 / 3))[0, 0]
        divergence = np.sum(coefficients[local.cell_faces[0]] * local.signs[0]) / local.areas[0]
        potential = space.integrate_rt0(classical.hybrid.coarse[cell][0], centroid_flux, divergence)
        integrated.append(space.evaluate(points[cell])[0] @ potential)

    def scalar_error(values: np.ndarray | list[np.ndarray]) -> float:
        """Physical L2 error on the actual macrotriangles."""
        return float(np.sqrt(mesh.areas @ ((exact - values) ** 2 @ weights)))

    full_pressure, full_flux = result.errors(pressure, flux, order + 2)
    coarse = np.array([value[0] for value in result.hybrid.coarse])
    rtcoarse = np.array([value[0] for value in classical.hybrid.coarse])
    record = {
        "n": n,
        "spacing": 1 / n,
        "macro_diameter": float(mesh.lengths.max()),
        "macro_cells": len(mesh.cells),
        "source_degree": 2,
        "source_refinement": refinement,
        "quadrature_order": order,
        "full_pressure_l2": full_pressure,
        "full_flux_l2": full_flux,
        "pressure_update_l2": scalar_error(harmonic),
        "harmonic_flux_l2": float(
            np.sqrt(
                mesh.areas
                @ (np.sum((np.asarray(harmonic_flux) - exact_flux) ** 2, axis=-1) @ weights)
            )
        ),
        "coarse_pressure_l2": scalar_error(coarse[:, None]),
        "projected_coarse_l2": float(np.sqrt(mesh.areas @ (means - coarse) ** 2)),
        "classical_rt0_quadratic_l2": scalar_error(integrated),
        "classical_rt0_flux_l2": classical.flux_l2_error(flux, order=order + 2),
        "classical_coarse_l2": scalar_error(rtcoarse[:, None]),
        "classical_projected_coarse_l2": float(np.sqrt(mesh.areas @ (means - rtcoarse) ** 2)),
        "residual": result.hybrid.residual,
        "seconds": perf_counter() - start,
    }
    return record


def plots(rows: list[dict[str, float | int]], root: Path) -> None:
    """Compare full-source analytical MHM, classical RT0 and digitized publication data."""
    with (root / "examples/results/published/harder2013_figure5.csv").open() as stream:
        published = [
            {key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)
        ]
    keys = (
        (
            "pressure_update_l2",
            "classical_rt0_quadratic_l2",
            "pressure_l2",
            "Pressure error",
        ),
        ("harmonic_flux_l2", "classical_rt0_flux_l2", "flux_l2", "Flux error"),
        (
            "coarse_pressure_l2",
            "classical_coarse_l2",
            "macroconstant_pressure_l2",
            r"Coarse pressure $p_0$",
        ),
        (
            "projected_coarse_l2",
            "classical_projected_coarse_l2",
            "macroconstant_projection_l2",
            r"Projected coarse error $\Pi_0p-p_0$",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    h = [row["spacing"] for row in rows]
    for ax, (mhm, rt, pub, title) in zip(axes.flat, keys, strict=True):
        ax.loglog(h, [row[mhm] for row in rows], "o-", label="Analytical MHM, full source moments")
        if mhm in {"pressure_update_l2", "harmonic_flux_l2"}:
            full = "full_pressure_l2" if mhm == "pressure_update_l2" else "full_flux_l2"
            ax.loglog(
                h, [row[full] for row in rows], "v-.", label="MHM including local source field"
            )
        ax.loglog(
            h,
            [row[rt] for row in rows],
            "s--",
            color="tab:green",
            label="Classical RT0 + quadratic potential",
        )
        ax.loglog(
            [row["h_nominal"] for row in published],
            [row[pub] for row in published],
            "kx",
            ms=8,
            label="Figure 5, digitized",
        )
        ax.set(
            title=title,
            xlabel="Declared grid spacing / published nominal h",
            ylabel="Absolute L2 error",
        )
        set_refinement_ticks(ax, h, [f"1/{row['n']}" for row in rows])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    folder = root / "docs/figures/analytic"
    folder.mkdir(exist_ok=True, parents=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"source-comparison.{extension}", dpi=180)
    plt.close(fig)


@threadpool_limits.wrap(limits=1)
def main() -> None:
    """Acquire six resolutions and an independent source/quadrature sensitivity check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 6, 8, 16, 32, 64])
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    root = case_workspace()
    path = root / "examples/results/analytic.json"
    if args.plot_only:
        data = json.loads(path.read_text())
    else:
        data = {
            "method": (
                "analytical MHM, full variable-source moments; separate P2 Neumann source lift"
            ),
            "results": [],
        }
        for n in args.levels:
            row = measure(n)
            data["results"].append(row)
            path.write_text(json.dumps(data, indent=2) + "\n")
            print(row, flush=True)
        data["source_refinement_check"] = [measure(4, r) for r in (4, 8)]
        data["quadrature_check"] = measure(4, 2, 12)
        path.write_text(json.dumps(data, indent=2) + "\n")
    plots(data["results"], root)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.verify_analytic").main()
