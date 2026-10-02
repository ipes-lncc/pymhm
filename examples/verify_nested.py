"""Five-level field equivalence for an original recursive MHM construction."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, HybridSystem, SkeletonSpace
from pymhm.conforming import ConformingQuadrilateralSolution
from pymhm.nested import nest_hybrid_system, nested_trace_map
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    _assemble_quad,
    _QuadTask,
    quadrilateral_quadrature,
    solve_darcy_quadrilateral,
)

ROOT = Path(__file__).resolve().parents[1]


def exact(x: np.ndarray) -> np.ndarray:
    """Analytical pressure vanishing on the unit square boundary."""
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def gradient(x: np.ndarray) -> np.ndarray:
    """Analytical gradient, independently differentiated from the source."""
    a, b = np.pi * x.T
    return np.pi * np.column_stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)))


def source(x: np.ndarray) -> np.ndarray:
    """Positive Laplacian source for -Delta(p)=f."""
    return 2 * np.pi**2 * exact(x)


def acquire(n: int) -> tuple[dict, list]:
    """Compare identical leaf spaces before and after a second hybridization."""
    macro = CartesianMacroMesh(n)
    outer = SkeletonSpace(macro, tuple(FaceSpace.uniform(1, 2) for _ in macro.faces))
    children = []
    for cell in range(len(macro.cells)):
        mesh = macro.submesh(cell, 2)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        inner = HybridSystem.from_local_factory(
            _assemble_quad,
            [_QuadTask(mesh, i, (2, 2), skeleton, 2, 1.0, source, 6) for i in range(4)],
        )
        kernel = np.r_[np.zeros(skeleton.size), np.ones(4)][:, None]
        moments = [metadata[1] for metadata in inner.local_metadata]
        constraint = inner.mean_constraint(moments)[0][:, None]
        selected, mapping = nested_trace_map(outer, cell, skeleton)
        children.append(
            nest_hybrid_system(
                inner,
                selected,
                mapping,
                outer.cell_dofs(cell),
                kernel=kernel,
                constraints=constraint,
            )
        )
    parent = HybridSystem([child.problem for child in children])
    result = parent.solve()
    flat_mesh = CartesianMacroMesh(2 * n)
    flat = solve_darcy_quadrilateral(
        flat_mesh,
        skeleton=SkeletonSpace(flat_mesh, tuple(FaceSpace.uniform(1) for _ in flat_mesh.faces)),
        degree=2,
        local_refinement=2,
        source=source,
        quadrature_order=6,
    )
    difference, norm, errors = 0.0, 0.0, np.zeros(2)
    largest_inner_residual = 0.0
    output = []
    quad, weights = quadrilateral_quadrature(6)
    for cell, (child, field) in enumerate(zip(children, result.fields, strict=True)):
        recovered = child.reconstruct(field)
        largest_inner_residual = max(largest_inner_residual, recovered.interior_residual)
        for subcell, values in enumerate(recovered.fields):
            i, j = 2 * (cell % n) + subcell % 2, 2 * (cell // n) + subcell // 2
            comparison = flat.pressure[j * 2 * n + i]
            difference += float(np.sum((values - comparison) ** 2))
            norm += float(np.sum(comparison**2))
            mesh = child.inner.local_metadata[subcell][0]
            local = ConformingQuadrilateralSolution(mesh, 2, values, 1.0, result.residual)
            physical = mesh.points[mesh.cells[:, 0], None] + quad[None] * mesh.spacing
            points = physical.reshape(-1, 2)
            p, g = local.evaluate(points)
            integrand = np.column_stack(
                ((p - exact(points)) ** 2, np.sum((g - gradient(points)) ** 2, axis=1))
            )
            errors += np.prod(mesh.spacing) * np.einsum(
                "q,tqi->i", weights, integrand.reshape(-1, len(weights), 2)
            )
            if n == 4:
                output.append(local)
    return dict(
        n=n,
        macro_cells=n * n,
        inner_macro_cells=4 * n * n,
        fine_cells=16 * n * n,
        outer_global_dofs=len(parent.rhs),
        flat_global_dofs=flat.skeleton.size + 4 * n * n,
        relative_leaf_difference=np.sqrt(difference / norm),
        pressure_l2=float(np.sqrt(errors[0])),
        flux_l2=float(np.sqrt(errors[1])),
        outer_residual=result.residual,
        inner_residual=largest_inner_residual,
    ), output


def plots(rows: list[dict], fields: list) -> None:
    """Render convergence, component fields and both actual macro partitions."""
    output = ROOT / "docs/figures/nested"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    h = 1 / np.array([row["n"] for row in rows])
    for key, label in (("pressure_l2", r"$\|p-p_h\|_{L^2}$"), ("flux_l2", r"$\|q-q_h\|_{L^2}$")):
        axes[0].loglog(h, [r[key] for r in rows], "o-", label=label)
    axes[0].set(xlabel="Outer macro size H", ylabel="Absolute error")
    axes[0].legend()
    axes[1].loglog(h, [r["relative_leaf_difference"] for r in rows], "s-", color="#7a3e9d")
    axes[1].set(xlabel="Outer macro size H", ylabel="Relative difference to one-level MHM")
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    for ext in ("png", "svg"):
        fig.savefig(output / f"convergence.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    titles = ("Pressure", r"Flux $q_x$", r"Flux $q_y$")
    for k, ax in enumerate(axes):
        for field in fields:
            xmin, xmax, ymin, ymax = field.mesh.bounds
            x, y = np.linspace(xmin, xmax, 17), np.linspace(ymin, ymax, 17)
            xx, yy = np.meshgrid(x, y)
            p, g = field.evaluate(np.column_stack((xx.ravel(), yy.ravel())))
            values = p if k == 0 else -g[:, k - 1]
            image = ax.pcolormesh(
                xx,
                yy,
                values.reshape(xx.shape),
                shading="gouraud",
                rasterized=True,
                cmap="viridis" if k == 0 else "RdBu_r",
                vmin=0 if k == 0 else -np.pi,
                vmax=1 if k == 0 else np.pi,
            )
        for n, color, width in ((8, ".6", 0.4), (4, ".1", 1.0)):
            macro = CartesianMacroMesh(n)
            ax.add_collection(
                LineCollection(macro.points[macro.faces], colors=color, linewidths=width)
            )
        ax.set(xlabel="x", ylabel="y", title=titles[k], aspect="equal")
        fig.colorbar(image, ax=ax, shrink=0.8, pad=0.04)
    for ext in ("png", "svg"):
        fig.savefig(output / f"fields.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    """Run research verification and serialize scientific evidence separately from CI."""
    rows, fields = [], []
    with threadpool_limits(1):
        for n in (1, 2, 4, 8, 16):
            row, selected = acquire(n)
            rows.append(row)
            fields.extend(selected)
            print(row, flush=True)
    record = dict(
        method="Original recursive MHM, based on nested local variational decompositions",
        reference="10.1007/978-3-319-41640-3_13, sections 2 and 4.1",
        local_space="Q2, 2x2 fine cells per inner macrocell",
        trace="P1 per inner face; two P1 segments per outer face",
        rows=rows,
    )
    (ROOT / "examples/results/nested.json").write_text(json.dumps(record, indent=2) + "\n")
    plots(rows, fields)


if __name__ == "__main__":
    main()
