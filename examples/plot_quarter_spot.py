"""Quarter-five-spot point wells, layered media and an interior-series reference."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from field_sampling import sample_field
from plot_quarter_reference import finish_quarter_panel
from threadpoolctl import threadpool_limits

from pymhm import SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.elements import rt0_evaluate
from pymhm.lagrange import tabulate
from pymhm.visualization import macro_edges

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/quarter-five-spot"
FIGURES = ROOT / "docs/figures/quarter-five-spot"
WELLS = np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]])


@dataclass(frozen=True)
class LayeredPermeability:
    """Scalar K=1000 below the chosen interface and K=1 above it."""

    height: float

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the discontinuous, isotropic layer coefficient."""
        return np.where(points[:, 1] < self.height, 1000.0, 1.0)


def series_reference(points: np.ndarray, terms: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the homogeneous square's Neumann Green series at interior y.

    Unit corner wells have p=y-1/2 + sum_{m>=1} 2/(m*pi) cos(m*pi*x)
    [(-1)^m cosh(m*pi*y)-cosh(m*pi*(1-y))]/sinh(m*pi).
    Exponentially scaled ratios avoid overflow. The zero-mean solution is
    singular at the wells. Truncation is checked by doubling ``terms``;
    evaluation on y=0 or y=1 is excluded because differentiated endpoint
    series need their limiting interpretation.
    """
    points = np.asarray(points)
    if np.any((points[:, 1] <= 0) | (points[:, 1] >= 1)):
        raise ValueError("the series evaluator requires 0 < y < 1")
    modes = np.arange(1, terms + 1)
    frequency = np.pi * modes
    sign = (-1.0) ** modes
    denominator = -np.expm1(-2 * frequency)
    pressure, flux = [], []
    for batch in np.array_split(points, max(1, (len(points) + 255) // 256)):
        x, y = batch[:, 0, None], batch[:, 1, None]
        upper = np.exp(-frequency * (1 - y))
        upper_image = np.exp(-frequency * (1 + y))
        lower = np.exp(-frequency * y)
        lower_image = np.exp(-frequency * (2 - y))
        even = (sign * (upper + upper_image) - lower - lower_image) / denominator
        odd = (sign * (upper - upper_image) + lower - lower_image) / denominator
        cosine, sine = np.cos(frequency * x), np.sin(frequency * x)
        pressure.append(batch[:, 1] - 0.5 + np.sum(2 * even * cosine / frequency, axis=1))
        flux.append(
            np.column_stack(
                (np.sum(2 * even * sine, axis=1), -1 - np.sum(2 * odd * cosine, axis=1))
            )
        )
    return np.concatenate(pressure), np.concatenate(flux)


def sampled_centroids(solution) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate pressure and physical flux at fine-triangle centroids."""
    positions, pressures, fluxes, areas = [], [], [], []
    for mesh, p, q in zip(solution.local_meshes, solution.pressure, solution.flux, strict=True):
        positions.append(mesh.points[mesh.cells].mean(axis=1))
        areas.append(mesh.areas)
        if solution.formulation == "primal":
            dofs, _, basis, _, _ = tabulate(mesh, solution.degree, np.full((1, 3), 1 / 3))
            pressures.append((p[dofs] @ basis.T)[:, 0])
            fluxes.append(q)
        else:
            pressures.append(p)
            fluxes.append(rt0_evaluate(mesh, q, np.full((1, 3), 1 / 3))[:, 0])
    return tuple(np.concatenate(values) for values in (positions, pressures, fluxes, areas))


def sampled_grid(solution) -> pv.UnstructuredGrid:
    """Build independent display triangles, preserving all one-sided fields."""
    if solution.formulation == "primal":
        fields = sample_field(
            solution.local_meshes, solution.pressure, solution.degree, refinement=2
        )
        permeability = np.concatenate(
            [
                np.repeat(
                    solution.permeability(mesh.points[mesh.cells].mean(axis=1))
                    if callable(solution.permeability)
                    else np.full(len(mesh.cells), solution.permeability),
                    6,
                )
                for mesh in solution.local_meshes
            ]
        )
        flux = -permeability[:, None] * fields["gradient"]
        points, cells, pressure = fields["points"], fields["cells"], fields["values"]
    else:
        points = np.concatenate(
            [mesh.points[mesh.cells].reshape(-1, 2) for mesh in solution.local_meshes]
        )
        cells = np.arange(len(points)).reshape(-1, 3)
        pressure = np.concatenate([np.repeat(p, 3) for p in solution.pressure])
        flux = np.concatenate(
            [
                rt0_evaluate(mesh, q, np.eye(3)).reshape(-1, 2)
                for mesh, q in zip(solution.local_meshes, solution.flux, strict=True)
            ]
        )
    grid = pv.UnstructuredGrid(
        np.column_stack((np.full(len(cells), 3), cells)).ravel(),
        np.full(len(cells), pv.CellType.TRIANGLE),
        np.column_stack((points, np.zeros(len(points)))),
    )
    grid.point_data["Pressure"] = pressure
    grid.point_data["Flux magnitude"] = np.linalg.norm(flux, axis=1)
    grid.point_data["Flux"] = np.column_stack((flux, np.zeros(len(flux))))
    return grid


def run_cases() -> None:
    """Compute the published physical cases with explicitly normalized point wells."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "well_functional": "v(1,1)-v(0,0); unit intensity chosen explicitly",
        "boundary": "zero exterior normal flux; domain mean pressure zero",
        "macro_connectivity": "32x32 squares, southwest-to-northeast diagonal",
        "point_sharing": "incident-angle partition, without duplicated strengths",
        "rows": [],
    }
    mesh = TriangleMesh.unit_square(32)
    np.savez_compressed(OUTPUT / "macro.npz", points=mesh.points, cells=mesh.cells)
    for name, coefficient in (
        ("homogeneous", 1.0),
        ("layer-half", LayeredPermeability(0.5)),
        ("layer-offset", LayeredPermeability(0.484375)),
    ):
        for method, degree in (("primal", 2), ("mixed", 1)):
            start = time.perf_counter()
            solution = solve_darcy(
                mesh,
                permeability=coefficient,
                point_sources=WELLS,
                neumann={int(face): 0 for face in mesh.boundary_faces},
                formulation=method,
                degree=degree,
                local_refinement=2,
                skeleton=SkeletonSpace(mesh),
            )
            row = {
                "name": name,
                "formulation": method,
                "degree": degree,
                "macro_triangles": len(mesh.cells),
                "local_refinement": 2,
                "trace_degree": 0,
                "macro_balance_max": float(np.max(abs(solution.conservation_residuals()))),
                "seconds": time.perf_counter() - start,
            }
            if method == "mixed":
                row["fine_balance_max"] = float(
                    max(np.max(abs(r)) for r in solution.fine_conservation_residuals())
                )
            points, p, q, area = sampled_centroids(solution)
            np.savez_compressed(
                OUTPUT / f"{name}-{method}.npz", points=points, pressure=p, flux=q, areas=area
            )
            sampled_grid(solution).save(OUTPUT / f"{name}-{method}.vtu")
            report["rows"].append(row)
            (OUTPUT / "point-wells.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)


def convergence() -> None:
    """Check six refinements against the Green series away from the singular wells."""
    rows = []
    for n in (8, 12, 16, 24, 32, 48):
        mesh = TriangleMesh.unit_square(n)
        solution = solve_darcy(
            mesh,
            point_sources=WELLS,
            neumann={int(face): 0 for face in mesh.boundary_faces},
            degree=2,
            local_refinement=2,
        )
        points, p, q, area = sampled_centroids(solution)
        selected = (
            np.minimum(np.linalg.norm(points, axis=1), np.linalg.norm(points - 1, axis=1)) > 0.125
        )
        reference_p, reference_q = series_reference(points, 2048)
        check_p, check_q = series_reference(points, 4096)
        row = {
            "n": n,
            "macro_triangles": len(mesh.cells),
            "sampled_pressure_rms": float(
                np.sqrt(
                    np.average((p[selected] - reference_p[selected]) ** 2, weights=area[selected])
                )
            ),
            "sampled_flux_rms": float(
                np.sqrt(
                    np.average(
                        np.sum((q[selected] - reference_q[selected]) ** 2, axis=1),
                        weights=area[selected],
                    )
                )
            ),
            "reference_doubling_pressure_max": float(np.max(abs(reference_p - check_p))),
            "reference_doubling_flux_max": float(np.max(abs(reference_q - check_q))),
            "reference_doubling_flux_away_wells_max": float(
                np.max(abs(reference_q[selected] - check_q[selected]))
            ),
            "macro_balance_max": float(np.max(abs(solution.conservation_residuals()))),
        }
        rows.append(row)
        (OUTPUT / "point-convergence.json").write_text(
            json.dumps(
                {
                    "reference_terms": 2048,
                    "check_terms": 4096,
                    "excluded_radius": 0.125,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps(row), flush=True)


def plot_cases() -> None:
    """Render broken pressure and flux on the actual macrotriangulation."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    with np.load(OUTPUT / "macro.npz") as data:
        mesh = TriangleMesh(data["points"], data["cells"])
    for name in ("homogeneous", "layer-half", "layer-offset"):
        grids = [pv.read(OUTPUT / f"{name}-{method}.vtu") for method in ("primal", "mixed")]
        plotter = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1500, 1700))
        for row, scalar in enumerate(("Pressure", "Flux magnitude")):
            limits = (
                min(float(grid[scalar].min()) for grid in grids),
                max(float(grid[scalar].max()) for grid in grids),
            )
            for column, (grid, method) in enumerate(
                zip(grids, ("Primal P2", "Mixed RT0/P0"), strict=True)
            ):
                plotter.subplot(row, column)
                plotter.set_background("white")
                actor = plotter.add_mesh(
                    grid,
                    scalars=scalar,
                    clim=limits,
                    cmap="RdBu_r" if row == 0 else "viridis",
                    lighting=False,
                    show_scalar_bar=False,
                )
                edges = macro_edges(mesh)
                edges.points[:, 2] = 1e-5
                plotter.add_mesh(edges, color="#303943", opacity=0.35, line_width=0.6)
                if name != "homogeneous":
                    height = 0.5 if name == "layer-half" else 0.484375
                    plotter.add_mesh(
                        pv.Line((0, height, 2e-5), (1, height, 2e-5)), color="red", line_width=3
                    )
                finish_quarter_panel(
                    plotter,
                    actor,
                    title=f"{method}: {name}\nUnit point wells\n2048 macrotriangles",
                    scalar=scalar,
                    scientific=False,
                )
        plotter.screenshot(FIGURES / f"{name}.png")
        plotter.close()
    rows = json.loads((OUTPUT / "point-convergence.json").read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in zip(
        axes, ("sampled_pressure_rms", "sampled_flux_rms"), ("Pressure", "Flux"), strict=True
    ):
        axis.loglog(
            [1 / r["n"] for r in rows],
            [r[key] for r in rows],
            "o-",
            label="Primal P2 / constant trace",
        )
        axis.set(xlabel="Macro square width", ylabel="Area-weighted centroid RMS", title=title)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.suptitle(
        "Unit corner wells: Green-series comparison outside radius 0.125\n"
        "Sampled diagnostics; no finite-energy convergence claim for Dirac sources"
    )
    fig.savefig(FIGURES / "point-convergence.svg")
    fig.savefig(FIGURES / "point-convergence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Compute and/or redraw the archived quarter-five-spot experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if not args.reuse_results:
            run_cases()
            convergence()
        plot_cases()


if __name__ == "__main__":
    main()
