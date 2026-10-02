"""Measure MHM--MsHHO equivalence and five-level analytical convergence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from plot_mesh import draw_macro_mesh
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.lagrange import tabulate
from pymhm.mshho import solve_mshho
from pymhm.solvers import LinearSolveError

ROOT = Path(__file__).resolve().parents[1]


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the homogeneous Dirichlet sine solution."""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def flux(points: np.ndarray) -> np.ndarray:
    """Evaluate minus the analytical gradient."""
    x, y = np.pi * points.T
    return -np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def main() -> None:
    """Run research examples outside CI and preserve all measured numerical values."""
    figures = ROOT / "docs/figures/mshho"
    figures.mkdir(parents=True, exist_ok=True)
    rows, equivalence = [], []
    with threadpool_limits(1):
        for degree in (0, 1):
            for n in (1, 2, 4, 8, 16):
                mesh = TriangleMesh.unit_square(n)
                skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree) for _ in mesh.faces))
                result = solve_mshho(
                    mesh,
                    skeleton=skeleton,
                    cell_degree=degree,
                    degree=3,
                    local_refinement=2,
                    source=lambda x: 2 * np.pi**2 * exact(x),
                    quadrature_order=8,
                )
                rows.append(
                    dict(
                        n=n,
                        face_degree=degree,
                        cell_degree=degree,
                        pressure_l2=result.l2_error(exact, 10),
                        flux_l2=result.flux_l2_error(flux, 10),
                        residual=result.residual,
                    )
                )
                print(rows[-1], flush=True)
                if degree == 1 and n == 4:
                    selected = result
        for contrast in (1.0, 1e3, 1e4, 1e6):
            mesh = TriangleMesh.unit_square(2)
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
            options = dict(
                skeleton=skeleton,
                permeability=np.diag([contrast, 1.0]),
                source=4.0,
                dirichlet=lambda x: x[:, 0] - x[:, 1],
                degree=3,
                local_refinement=2,
                quadrature_order=8,
            )
            primal = solve_mshho(mesh, cell_degree=1, **options)
            try:
                dual = solve_darcy(mesh, local_refinement_precision="extended", **options)
            except LinearSolveError:
                equivalence.append(
                    dict(
                        contrast=contrast,
                        status="not_certified",
                        reason="MHM local solve does not meet the physical residual tolerance",
                        primal_residual=primal.residual,
                    )
                )
                continue
            difference = max(
                float(np.max(np.abs(a - b)))
                for a, b in zip(primal.pressure, dual.pressure, strict=True)
            )
            scale = max(float(np.max(np.abs(a))) for a in dual.pressure)
            equivalence.append(
                dict(
                    contrast=contrast,
                    status="verified",
                    field_linf=difference,
                    relative_linf=difference / scale,
                    primal_residual=primal.residual,
                    dual_residual=dual.hybrid.residual,
                )
            )
            assert difference / scale < 2e-8
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for degree, marker in ((0, "o"), (1, "s")):
        selected_rows = [row for row in rows if row["face_degree"] == degree]
        h = 1 / np.array([row["n"] for row in selected_rows])
        for ax, key in zip(axes, ("pressure_l2", "flux_l2"), strict=True):
            errors = np.array([row[key] for row in selected_rows])
            ax.loglog(h, errors, marker=marker, label=f"m=k={degree}")
            rate = np.log(errors[-2] / errors[-1]) / np.log(2)
            selected_rows[-1][key + "_last_rate"] = float(rate)
            assert rate > (1.65 if key == "pressure_l2" else 0.8) + 0.6 * degree
    for ax, name in zip(axes, ("Pressure", "Physical raw flux"), strict=True):
        ax.set(xlabel="Macro spacing H", ylabel="L2 error", title=name)
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(figures / "convergence.png", dpi=200)
    fig.savefig(figures / "convergence.svg")
    plt.close(fig)
    points, cells, values = [], [], []
    reference = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, 3)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    offset = 0
    for data, pressure in zip(selected.local, selected.pressure, strict=True):
        dofs, _, basis, _, _ = tabulate(data.mesh, selected.degree, bary)
        points.append(
            np.einsum("qi,tia->tqa", bary, data.mesh.points[data.mesh.cells]).reshape(-1, 2)
        )
        values.append((pressure[dofs] @ basis.T).ravel())
        for cell in range(len(data.mesh.cells)):
            cells.append(reference.cells + offset + cell * len(bary))
        offset += len(data.mesh.cells) * len(bary)
    xy, numerical = np.vstack(points), np.concatenate(values)
    analytical = exact(xy)
    triangulation = mtri.Triangulation(*xy.T, triangles=np.vstack(cells))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    error = numerical - analytical
    for ax, data, title in zip(
        axes,
        (analytical, numerical, error),
        ("Analytical pressure", "MsHHO m=k=1", "Signed pressure error"),
        strict=True,
    ):
        limits = (-max(abs(error)), max(abs(error))) if ax is axes[2] else (0, 1)
        artist = ax.tripcolor(
            triangulation,
            data,
            shading="gouraud",
            vmin=limits[0],
            vmax=limits[1],
            cmap="RdBu_r" if ax is axes[2] else "viridis",
            rasterized=True,
        )
        draw_macro_mesh(ax, selected.skeleton.mesh)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(artist, ax=ax, shrink=0.85, format="%.2g")
    fig.savefig(figures / "fields.png", dpi=200)
    fig.savefig(figures / "fields.svg")
    plt.close(fig)
    destination = ROOT / "examples/results/mshho.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            dict(
                convergence=rows,
                equivalence=equivalence,
                mhm_local_refinement_precision="extended",
                local_residual_tolerance_unchanged=True,
                accumulation_mantissa_bits=int(np.finfo(np.longdouble).nmant),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
