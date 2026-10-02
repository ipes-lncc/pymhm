"""Five-level polygonal RAD study with the manufactured data of L12 section 5.2.1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from plot_mesh import draw_macro_mesh
from polygon_meshes import polygon_partition
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace
from pymhm.lagrange import tabulate
from pymhm.polygon import solve_transport_polygons

ROOT = Path(__file__).resolve().parents[1]


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate sin(6 pi x) sin(14 pi y), as in the generalized RAD article."""
    x, y = points.T
    return np.sin(6 * np.pi * x) * np.sin(14 * np.pi * y)


def source(points: np.ndarray) -> np.ndarray:
    """Apply -0.1 Delta + partial_x analytically, with zero reaction."""
    x, y = points.T
    return 0.1 * (36 + 196) * np.pi**2 * exact(points) + 6 * np.pi * np.cos(6 * np.pi * x) * np.sin(
        14 * np.pi * y
    )


def main() -> None:
    """Measure convergence and render true broken polynomial fields with macro outlines."""
    destination = ROOT / "examples/results/polygons.json"
    rows = json.loads(destination.read_text())["convergence"] if destination.exists() else []
    selected = None
    with threadpool_limits(1):
        for family in ("triangle", "square", "rhombus", "L", "hexagon"):
            for n in (2, 4, 8, 16, 32):
                old = next((r for r in rows if r["family"] == family and r["n"] == n), None)
                if old is not None and not (family == "hexagon" and n == 8):
                    continue
                mesh = polygon_partition(n, family)
                skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
                result = solve_transport_polygons(
                    mesh,
                    skeleton=skeleton,
                    degree=3,
                    local_refinement=1,
                    diffusion=0.1,
                    velocity=(1.0, 0.0),
                    source=source,
                    dirichlet_enforcement="strong",
                    quadrature_order=12,
                )
                row = dict(
                    family=family,
                    n=n,
                    macros=len(mesh.cells),
                    global_dofs=result.hybrid.trace.size
                    + sum(c.size for c in result.hybrid.coarse),
                    local_dofs=sum(len(v) for v in result.values),
                    error_l2=result.l2_error(exact, 14),
                    residual=result.hybrid.residual,
                )
                if old is not None:
                    rows.remove(old)
                rows.append(row)
                print(row, flush=True)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        dict(
                            reference="10.1016/j.cma.2024.117089, section 5.2.1",
                            trace_degree=1,
                            local_degree=3,
                            local_refinement=1,
                            local_triangulation="boundary-separated centroid fans",
                            note=(
                                "Published PDE and degrees; explicitly constructed polygon meshes "
                                "and local refinements."
                            ),
                            convergence=rows,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                if family == "hexagon" and n == 8:
                    selected = result
    convergence_plot(rows)
    figures = ROOT / "docs/figures/polygons"
    field_plot(selected, figures)


def convergence_plot(rows: list[dict]) -> None:
    """Render the recorded partition refinement levels with explicit sparse ticks."""
    from plot_style import set_refinement_ticks

    figures = ROOT / "docs/figures/polygons"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for family, marker in zip(
        ("triangle", "square", "rhombus", "L", "hexagon"), ("v", "s", "D", "^", "o"), strict=True
    ):
        series = sorted((r for r in rows if r["family"] == family), key=lambda r: r["n"])
        ax.loglog(
            [r["n"] for r in series], [r["error_l2"] for r in series], marker=marker, label=family
        )
    ax.set(
        xlabel="Partition parameter N",
        ylabel=r"Scalar $L^2$ error",
        title="Oscillatory RAD: trace P1, local P3",
    )
    ax.grid(alpha=0.25)
    set_refinement_ticks(ax, [2, 4, 8, 16, 32])
    ax.legend()
    for extension in ("png", "svg"):
        fig.savefig(figures / f"convergence.{extension}", dpi=220)
    plt.close(fig)


def field_plot(selected: object, figures: Path) -> None:
    """Render the calculated polygonal field without averaging macro boundaries."""
    if selected is None:
        raise RuntimeError("the field example must be evaluated before rendering")
    from pymhm import TriangleMesh

    sample = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, 4)
    bary = np.column_stack((1 - sample.points.sum(axis=1), sample.points))
    xy, values, cells, offset = [], [], [], 0
    for fine, coefficients in zip(selected.local_meshes, selected.values, strict=True):
        dofs, _, basis, _, _ = tabulate(fine, 3, bary)
        xy.append(np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 2))
        values.append((coefficients[dofs] @ basis.T).ravel())
        cells.extend(sample.cells + offset + i * len(bary) for i in range(len(fine.cells)))
        offset += len(fine.cells) * len(bary)
    coordinates, numerical = np.vstack(xy), np.concatenate(values)
    analytical = exact(coordinates)
    triangulation = mtri.Triangulation(*coordinates.T, triangles=np.vstack(cells))
    error = numerical - analytical
    limit = float(np.max(abs(error)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, data, title, clim in zip(
        axes,
        (analytical, numerical, error),
        ("Analytical", "Polygonal MHM", "Signed error"),
        ((-1, 1), (-1, 1), (-limit, limit)),
        strict=True,
    ):
        artist = ax.tripcolor(
            triangulation,
            data,
            shading="gouraud",
            cmap="RdBu_r",
            vmin=clim[0],
            vmax=clim[1],
            rasterized=True,
        )
        draw_macro_mesh(ax, selected.skeleton.mesh)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(artist, ax=ax, shrink=0.85, format="%.2g")
    for extension in ("png", "svg"):
        fig.savefig(figures / f"fields.{extension}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
