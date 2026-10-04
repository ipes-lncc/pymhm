"""Render independently acquired RAD fields from self-contained numerical archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator

from pymhm.fem.scalar.triangle import reference_basis

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/rad-native"
OUTPUT = ROOT / "docs/figures/rad-layer"


def sample(data: dict[str, np.ndarray]) -> tuple[mtri.Triangulation, tuple[np.ndarray, ...]]:
    """Evaluate both broken cubic fields on identical separate fine-cell display pieces."""
    subdivision = 4
    lattice = [(i, j) for j in range(subdivision + 1) for i in range(subdivision + 1 - j)]
    lookup = {point: i for i, point in enumerate(lattice)}
    local_cells = []
    for j in range(subdivision):
        for i in range(subdivision - j):
            local_cells.append([lookup[i, j], lookup[i + 1, j], lookup[i, j + 1]])
            if i + j + 1 < subdivision:
                local_cells.append([lookup[i + 1, j], lookup[i + 1, j + 1], lookup[i, j + 1]])
    xi = np.asarray(lattice) / subdivision
    bary = np.column_stack((1 - xi.sum(axis=1), xi))
    basis, dbary, _ = reference_basis(int(data["local_degree"]), bary)
    corners = data["fine_points"][data["fine_cells"]]
    transform = np.linalg.inv((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
    dxi = dbary[:, :, 1:] - dbary[:, :, :1]
    gradient = np.einsum("qia,tab->tqib", dxi, transform)
    coordinates = np.einsum("qi,tia->tqa", bary, corners)
    triangles = (
        np.asarray(local_cells)[None] + len(bary) * np.arange(len(corners))[:, None, None]
    ).reshape(-1, 3)
    tri = mtri.Triangulation(*coordinates.reshape(-1, 2).T, triangles)
    fields = []
    for name in ("native_field_canonical", "pymhm_field_canonical"):
        coefficients = data[name][data["element_dofs"]]
        scalar = coefficients @ basis.T
        derivative = np.einsum("ti,tqia->tqa", coefficients, gradient)
        flux = -0.01 * derivative
        flux[:, :, 0] += scalar
        fields.append(np.column_stack((scalar.ravel(), flux.reshape(-1, 2))))
    return tri, tuple(fields)


def main() -> None:
    """Compare the full n=8, r=8 layer fields without executing external reference code."""
    record = json.loads((DATA / "verification.json").read_text())
    row = next(
        item
        for item in record["rows"]
        if item["kind"] == "layer" and item["n"] == 8 and item["local_refinement"] == 8
    )
    path = DATA / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("archived native comparison differs from its recorded digest")
    with np.load(path) as archive:
        data = dict(archive)
    tri, (native, candidate) = sample(data)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    fig = plt.figure(figsize=(12, 12.8), layout="constrained")
    grid = fig.add_gridspec(6, 3, height_ratios=[1, 0.06] * 3, hspace=0.08, wspace=0.08)
    labels = (r"$u$", r"$q_x=-\epsilon\partial_xu+u$", r"$q_y=-\epsilon\partial_yu$")
    for component, label in enumerate(labels):
        values = (
            native[:, component],
            candidate[:, component],
            candidate[:, component] - native[:, component],
        )
        common = max(np.max(abs(values[0])), np.max(abs(values[1])))
        common_minimum = min(0.0, values[0].min(), values[1].min())
        for column, field in enumerate(values):
            axis = fig.add_subplot(grid[2 * component, column])
            limit = max(np.max(abs(field)), np.finfo(float).tiny) if column == 2 else common
            signed = component == 2 or column == 2
            artist = axis.tripcolor(
                tri,
                field,
                shading="gouraud",
                cmap="seismic" if signed else "viridis",
                vmin=-limit if signed else common_minimum,
                vmax=limit,
                rasterized=True,
            )
            axis.add_collection(
                LineCollection(data["macro_segments"], colors="white", linewidths=0.85, alpha=0.7)
            )
            axis.add_collection(
                LineCollection(data["macro_segments"], colors=".2", linewidths=0.35, alpha=0.8)
            )
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="$x$",
                ylabel="$y$",
                title=(
                    f"{('MHM reference', 'PyMHM', 'PyMHM − reference')[column]} · {label}\n"
                    + (
                        "DOLFINx P3 + explicit P1 coupling",
                        "P3 triangles / P1 macrofaces",
                        "Same discrete spaces",
                    )[column]
                ),
            )
            bar = fig.colorbar(
                artist,
                cax=fig.add_subplot(grid[2 * component + 1, column]),
                orientation="horizontal",
            )
            bar.locator = MaxNLocator(3)
            bar.update_ticks()
    fig.suptitle(
        "80 hexagonal macroelements · 28,544 fine triangles\n"
        "Same MHM discretization; reference: UFL local forms + explicit coupling / SuperLU"
    )
    OUTPUT.mkdir(exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"native-comparison.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
