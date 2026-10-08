"""Plot the archived RT-family convergence and broken RT2/DG2 physical fields."""

from __future__ import annotations

import hashlib
import json

import matplotlib

from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from examples.plot_mesh import draw_macro_mesh
from pymhm.fem.hdiv.rt import rt_evaluate
from pymhm.meshes.triangle import TriangleMesh

ROOT = case_workspace()


def exact(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact sinusoidal pressure and flux magnitude at physical samples."""
    s, c = np.sin(2 * np.pi * points), np.cos(2 * np.pi * points)
    p = s.prod(axis=1)
    q = -2 * np.pi * np.column_stack((c[:, 0] * s[:, 1], s[:, 0] * c[:, 1]))
    return p, np.linalg.norm(q, axis=1)


def main() -> None:
    """Render five-level norms and flat fine-cell samples with explicit macro boundaries."""
    report = json.loads(read_resource_text(ROOT / "examples/results/darcy-rt.json"))
    output = ROOT / "docs/figures/darcy-rt"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for degree, color in zip((0, 1, 2), ("#3465a4", "#d97706", "#157f58"), strict=True):
        rows = [row for row in report["rows"] if row["degree"] == degree]
        h = 1 / np.array([row["n"] for row in rows])
        for axis, key, label in zip(
            axes,
            ("pressure_l2_error", "flux_l2_error", "divergence_l2_error"),
            ("Pressure L² error", "Physical flux L² error", "Divergence L² error"),
            strict=True,
        ):
            for method, style in (("mhm", "o-"), ("classical", "s--")):
                axis.loglog(
                    h,
                    [row[method][key] for row in rows],
                    style,
                    color=color,
                    label=f"RT{degree} {method}",
                )
            axis.set(xlabel="Macro spacing 1/n", ylabel=label)
            axis.set_xticks(h, labels=["1", "1/2", "1/4", "1/8", "1/16"])
            axis.minorticks_off()
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    for ext in ("png", "svg"):
        fig.savefig(output / f"convergence.{ext}", dpi=180)
    plt.close(fig)
    row = report["rows"][-1]
    path = ROOT / "examples/results" / row["fields"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != row["fields_sha256"]:
        raise ValueError("RT field digest mismatch")
    data = np.load(local_resource(path))
    macro = TriangleMesh(data["macro_points"], data["macro_cells"])
    vertices = []
    triangles = []
    pressure = []
    magnitude = []
    centers = []
    from pymhm.fem.hdiv.rt_forms import pressure_basis

    bary = np.ones((1, 3)) / 3
    for i in range(len(data["local_points"])):
        fine = TriangleMesh(data["local_points"][i], data["local_cells"][i])
        offset = sum(len(v) for v in vertices)
        vertices.append(fine.points)
        triangles.append(fine.cells + offset)
        pressure.extend((data["pressure"][i] @ pressure_basis(2, bary).T).ravel())
        magnitude.extend(
            np.linalg.norm(rt_evaluate(fine, data["flux"][i], 2, bary)[0][:, 0], axis=1)
        )
        centers.extend(fine.points[fine.cells].mean(axis=1))
    vertices = np.vstack(vertices)
    triangles = np.vstack(triangles)
    pe, qe = exact(np.asarray(centers))
    numerical = np.column_stack((pressure, magnitude))
    analytical = np.column_stack((pe, qe))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for i, label in enumerate(("Pressure", "Flux magnitude")):
        bound = max(np.max(abs(numerical[:, i])), np.max(abs(analytical[:, i])))
        shared = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound) if i == 0 else Normalize(0, bound)
        difference = numerical[:, i] - analytical[:, i]
        error_bound = max(np.max(abs(difference)), 1e-15)
        for j, values in enumerate((analytical[:, i], numerical[:, i], difference)):
            axis = axes[i, j]
            norm = shared if j < 2 else TwoSlopeNorm(vmin=-error_bound, vcenter=0, vmax=error_bound)
            cmap = "RdBu_r" if i == 0 or j == 2 else "viridis"
            image = axis.tripcolor(
                vertices[:, 0],
                vertices[:, 1],
                triangles,
                facecolors=values,
                shading="flat",
                cmap=cmap,
                norm=norm,
            )
            draw_macro_mesh(axis, macro)
            axis.set(
                aspect="equal",
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="x",
                ylabel="y",
                title=f"{label}: {('exact', 'MHM RT2', 'MHM − exact')[j]}",
            )
            fig.colorbar(image, ax=axis, shrink=0.85, pad=0.025)
    fig.suptitle("RT2/DG2 · fine-cell centroid samples · actual macro edges", fontsize=16)
    for ext in ("png", "svg"):
        fig.savefig(output / f"fields.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_darcy_rt").main()
