"""Five-level macro-adaptive energy estimation for the L09 smooth Darcy problem."""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

from pymhm.io.workspace import case_workspace

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
from threadpoolctl import threadpool_limits

from examples.field_sampling import sample_field
from examples.formulations.application import darcy as solve_equations
from examples.plot_mesh import draw_macro_mesh
from pymhm.adaptivity.darcy import solve_adaptive_darcy
from pymhm.meshes.triangle import TriangleMesh

ROOT = case_workspace()
DATA = ROOT / "examples/results/adaptive-darcy"
FIGURES = ROOT / "docs/figures/adaptive-darcy"


def pressure(points: np.ndarray) -> np.ndarray:
    """Exact homogeneous-boundary pressure used in L09 Section 6.1."""
    return np.prod(np.sin(2 * np.pi * points), axis=1)


def gradient(points: np.ndarray) -> np.ndarray:
    """Analytical gradient of the published sine pressure."""
    a, b = 2 * np.pi * points.T
    return 2 * np.pi * np.column_stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)))


def collect() -> list[dict[str, Any]]:
    """Acquire both polynomial families; estimator and error use independent quadrature."""
    records = []
    for ell in (0, 1):
        result = solve_adaptive_darcy(
            TriangleMesh.unit_square(2),
            solve_step=solve_equations,
            iterations=5,
            theta=0.5,
            trace_degree=ell,
            reconstruction_degree=2,
            degree=ell + 2,
            local_refinement=2,
            source=lambda x: 8 * np.pi**2 * pressure(x),
            quadrature_order=10,
            estimator_order=10,
        )
        for level, (solution, estimator, marked) in enumerate(
            zip(result.solutions, result.estimators, result.marked, strict=True)
        ):
            coarse = solution.skeleton.mesh
            error = estimator.energy_error(gradient, order=12)
            independently_integrated = estimator.energy_error(gradient, order=14)
            record = {
                "level": level,
                "trace_degree": ell,
                "local_degree": ell + 2,
                "local_refinement": 2,
                "reconstruction_degree": 2,
                "theta": 0.5,
                "macro_triangles": len(coarse.cells),
                "minimum_shape_quality": float(
                    np.min(
                        4
                        * np.sqrt(3)
                        * coarse.areas
                        / np.sum(coarse.lengths[coarse.cell_faces] ** 2, axis=1)
                    )
                ),
                "marked": int(marked.sum()),
                "global_dofs": solution.skeleton.size + len(coarse.cells),
                "energy_error": error,
                "energy_error_order14": independently_integrated,
                "estimator": estimator.total,
                "effectivity": estimator.total / error,
                "pressure_l2": solution.l2_error(pressure, order=12),
                "macro_balance_linf": float(np.max(abs(solution.conservation_residuals()))),
                "continuous_equilibrium_l2_max": float(np.max(estimator.equilibrium_defect)),
            }
            records.append(record)
            samples = sample_field(solution.local_meshes, solution.pressure, solution.degree, 3)
            np.savez_compressed(
                DATA / f"ell{ell}-level{level}.npz",
                **samples,
                exact=pressure(samples["points"]),
                macro_points=coarse.points,
                macro_cells=coarse.cells,
                local_squared=estimator.local_squared,
                marked=marked,
            )
            print(record, flush=True)
    (DATA / "campaign.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def save(figure: Any, name: str) -> None:
    """Write scientific figures with vector labels and rasterized dense fields."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def plot() -> None:
    """Plot complete refinement histories and broken solution fields with actual macros."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14})
    rows = json.loads((DATA / "campaign.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
    for ell, marker in ((0, "o"), (1, "s")):
        series = [row for row in rows if row["trace_degree"] == ell]
        dofs = [row["global_dofs"] for row in series]
        axes[0].loglog(
            dofs,
            [row["energy_error"] for row in series],
            marker + "-",
            label=f"Energy error, ℓ={ell}",
        )
        axes[0].loglog(
            dofs, [row["estimator"] for row in series], marker + "--", label=f"Estimator, ℓ={ell}"
        )
        axes[1].plot(dofs, [row["effectivity"] for row in series], marker + "-", label=f"ℓ={ell}")
    for axis in axes:
        axis.set_xlabel("Global unknowns (trace + macro means)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Broken energy error / estimator")
    axes[0].set_xticks([30, 60, 120, 240, 480])
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_ylabel("Estimator / energy error")
    save(fig, "refinement")
    for ell in (0, 1):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5), layout="constrained")
        for level, axis in enumerate(axes.flat):
            if level == 5:
                axis.axis("off")
                axis.text(
                    0.05,
                    0.6,
                    "Blue: Dörfler bulk set\n\nEdges: actual macro mesh\n\n"
                    "Last marking is displayed;\nno sixth solve is included.",
                    transform=axis.transAxes,
                    va="top",
                )
                continue
            field = np.load(DATA / f"ell{ell}-level{level}.npz")
            mesh = TriangleMesh(field["macro_points"], field["macro_cells"])
            colors = np.where(field["marked"], 1.0, 0.0)
            axis.tripcolor(
                *mesh.points.T, mesh.cells, facecolors=colors, cmap="Blues", vmin=0, vmax=1
            )
            draw_macro_mesh(axis, mesh)
            axis.set(
                title=f"Level {level}: {len(mesh.cells)} cells",
                xlabel="x",
                ylabel="y",
                aspect="equal",
            )
        save(fig, f"meshes-ell{ell}")
    field = np.load(DATA / "ell1-level4.npz")
    coarse = TriangleMesh(field["macro_points"], field["macro_cells"])
    tri = mtri.Triangulation(*field["points"].T, field["cells"])
    error = field["values"] - field["exact"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4), layout="constrained")
    for axis, values, title, limits in zip(
        axes,
        (field["exact"], field["values"], error),
        ("Exact pressure", "Adaptive MHM\nP3 / trace P1", "Signed pressure error"),
        ((-1, 1), (-1, 1), (-np.max(abs(error)), np.max(abs(error)))),
        strict=True,
    ):
        artist = axis.tripcolor(
            tri,
            values,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=limits[0],
            vmax=limits[1],
            rasterized=True,
        )
        draw_macro_mesh(axis, coarse)
        axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        colorbar = fig.colorbar(
            artist, ax=axis, orientation="horizontal", pad=0.14, label="Pressure"
        )
        colorbar.locator = MaxNLocator(5)
        colorbar.update_ticks()
    save(fig, "fields")


def main() -> None:
    """Collect a separate numerical campaign or render its archived data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    options = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if options.collect:
        with threadpool_limits(1):
            collect()
    plot()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.adaptive_darcy_campaign").main()
