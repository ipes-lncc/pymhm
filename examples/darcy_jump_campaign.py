"""Five-level study of the L02 face indicator with explicitly discretized local lifts."""

from __future__ import annotations

import argparse
import json
from typing import Any

import matplotlib

from pymhm.io.workspace import case_workspace

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import darcy as solve_darcy
from examples.plot_style import set_refinement_ticks
from pymhm.estimators.darcy_jump import estimate_darcy_jumps
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh

ROOT = case_workspace()
DATA = ROOT / "examples/results/darcy-jump"
FIGURES = ROOT / "docs/figures/darcy-jump"


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the published pressure cos(2pi x) cos(2pi y)."""
    return np.prod(np.cos(2 * np.pi * points), axis=1)


def gradient(points: np.ndarray) -> np.ndarray:
    """Exact gradient, derived directly from the pressure."""
    x, y = (2 * np.pi * points).T
    return -2 * np.pi * np.column_stack((np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)))


def norms(solution: Any, order: int = 10) -> dict[str, float]:
    """Evaluate the paper's sum of flux H(div,h) and scaled broken potential terms."""
    macro = solution.skeleton.mesh
    bary, weights = triangle_quadrature(order)
    totals = np.zeros(4)
    for cell, (fine, coefficients) in enumerate(
        zip(solution.local_meshes, solution.pressure, strict=True)
    ):
        dofs, _, values, derivatives, hessians = tabulate(fine, solution.degree, bary)
        physical = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        points = physical.reshape(-1, 2)
        p = coefficients[dofs] @ values.T
        g = np.einsum("ti,tqia->tqa", coefficients[dofs], derivatives)
        laplacian = np.einsum("ti,tqiaa->tq", coefficients[dofs], hessians)
        dp = p - exact(points).reshape(p.shape)
        dg = g - gradient(points).reshape(g.shape)
        df = -laplacian - 8 * np.pi**2 * exact(points).reshape(p.shape)
        diameter = np.max(macro.lengths[macro.cell_faces[cell]])
        integrals = np.array(
            [
                fine.areas @ (dp**2 @ weights),
                fine.areas @ (np.sum(dg**2, axis=2) @ weights),
                fine.areas @ (df**2 @ weights),
            ]
        )
        totals[:3] += integrals
        totals[3] += diameter**2 * integrals[2]
    p_error, grad_error, div_error = np.sqrt(totals[:3])
    scaled_flux = np.sqrt(totals[1] + totals[3])
    error = scaled_flux + p_error / np.max(macro.lengths) + grad_error
    return {
        "pressure_l2": float(p_error),
        "gradient_l2": float(grad_error),
        "flux_divergence_l2": float(div_error),
        "flux_hdiv_h": float(scaled_flux),
        "paper_error_sum": float(error),
        "norm_order": order,
    }


def collect() -> list[dict[str, Any]]:
    """Run ten discretizations and preserve the stated local-source approximation."""
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for ell, calibration in ((0, 3.0), (3, 50.0)):
        for n in (4, 8, 16, 32, 64):
            mesh = TriangleMesh.unit_square(n)
            natural = {int(face): 0.0 for face in mesh.boundary_faces}
            solution = solve_darcy(
                mesh,
                degree=ell + 2,
                local_refinement=2,
                skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(ell) for _ in mesh.faces)),
                source=lambda x: 8 * np.pi**2 * exact(x),
                neumann=natural,
                quadrature_order=10,
            )
            indicator = estimate_darcy_jumps(solution, neumann=natural, calibration=calibration)
            row = {
                "resolution": n,
                "macro_triangles": len(mesh.cells),
                "trace_degree": ell,
                "local_degree": ell + 2,
                "local_refinement": 2,
                "calibration": calibration,
                "indicator": indicator.total,
                "global_dofs": solution.skeleton.size + len(mesh.cells),
                "h": float(np.max(mesh.lengths)),
                **norms(solution),
            }
            row["effectivity"] = row["indicator"] / row["paper_error_sum"]
            if n == 64:
                row["independent_order12"] = norms(solution, order=12)
            rows.append(row)
            np.savez_compressed(
                DATA / f"ell{ell}-n{n}.npz",
                macro_points=mesh.points,
                macro_cells=mesh.cells,
                local_squared=indicator.local_squared,
                face_squared=indicator.face_squared,
            )
            (DATA / "campaign.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(row, flush=True)
    return rows


def plot() -> None:
    """Display norms and effectivity without implying exact historical local solves."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = json.loads((DATA / "campaign.json").read_text())
    plt.rcParams.update({"font.size": 13})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    for ell, marker in ((0, "o"), (3, "s")):
        part = [row for row in rows if row["trace_degree"] == ell]
        h = [row["h"] for row in part]
        axes[0].loglog(
            h, [row["paper_error_sum"] for row in part], marker + "-", label=f"Error sum, ℓ={ell}"
        )
        axes[0].loglog(
            h, [row["indicator"] for row in part], marker + "--", label=f"Indicator, ℓ={ell}"
        )
        axes[1].semilogx(h, [row["effectivity"] for row in part], marker + "-", label=f"ℓ={ell}")
    for axis in axes:
        axis.set_xlabel("Maximum macro diameter H")
        positions = sorted({row["h"] for row in rows})
        set_refinement_ticks(axis, positions, [f"{value:.3g}" for value in positions])
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Error sum / calibrated indicator")
    axes[1].set_ylabel("Indicator / error sum")
    for extension in ("png", "svg"):
        figure.savefig(FIGURES / f"refinement.{extension}", dpi=180)
    plt.close(figure)


def main() -> None:
    """Separate numerical acquisition from archived-data visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    options = parser.parse_args()
    with threadpool_limits(1):
        if options.collect:
            collect()
        plot()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.darcy_jump_campaign").main()
