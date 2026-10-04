"""Render archived independent displacement/velocity comparisons on physical cuts."""

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

from examples.elastodynamics_results import evaluate
from examples.tetra_section_samples import section_grid
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/elastodynamics/native"
OUTPUT = ROOT / "docs/figures/elastodynamics"


def samples(
    data: dict[str, np.ndarray],
) -> tuple[mtri.Triangulation, np.ndarray, tuple[np.ndarray, ...]]:
    """Replay saved native nodal coordinates before sampling separate polynomial pieces."""
    mesh = TetraMesh(data["macro_points"], data["macro_cells"])
    common = {"local_degree": np.asarray(3), "local_refinement": np.asarray(2)}
    native, candidate = dict(common), dict(common)
    for field in ("displacement", "velocity"):
        native[field] = np.empty_like(data["canonical_pymhm_" + field])
        candidate[field] = data["canonical_pymhm_" + field]
        for macro, (start, stop) in enumerate(
            zip(data["offsets"][:-1], data["offsets"][1:], strict=True)
        ):
            native[field][macro, data["native_to_pymhm"][macro]] = data["native_" + field][
                start:stop
            ]
    coordinates, triangles, values = [], [], [[], []]
    offset = 0
    for macro in range(len(mesh.cells)):
        fine = mesh.submesh(macro, 2)
        nodes = tetra_nodal_space(fine, 3)[1]
        native_nodes = nodes[data["native_to_pymhm"][macro][::3] // 3]
        if not np.allclose(
            native_nodes, data["native_local_coordinates"][macro], atol=2e-13, rtol=0
        ):
            raise ValueError("archived native nodes differ from their declared polynomial map")
        section = section_grid(fine, height=0.37, refinement=6)
        if not len(section["points"]):
            continue
        coordinates.append(section["points"])
        triangles.append(section["cells"] + offset)
        for index, fields in enumerate((native, candidate)):
            values[index].append(
                evaluate(fields, macro, fine, section["parents"], section["barycentric"])
            )
        offset += len(section["points"])
    points = np.concatenate(coordinates)
    tri = mtri.Triangulation(points[:, 0], points[:, 1], np.concatenate(triangles))
    edges = section_grid(mesh, height=0.37, refinement=1)["segments"]
    return tri, edges, tuple(np.concatenate(value) for value in values)


def main() -> None:
    """Draw independently solved fields with shared scales and explicit macro intersections."""
    record = json.loads((DATA / "verification.json").read_text())
    row = next(item for item in record["rows"] if item["n"] == 2 and item["dt"] == 0.005)
    path = DATA / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("native comparison archive differs from its recorded digest")
    with np.load(path) as archive:
        tri, edges, (native, candidate) = samples(dict(archive))
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    for vector, name, symbol in ((0, "displacement", "u"), (1, "velocity", "v")):
        fig = plt.figure(figsize=(12, 12.8), layout="constrained")
        grid = fig.add_gridspec(6, 3, height_ratios=[1, 0.06] * 3, hspace=0.08, wspace=0.08)
        for component in range(3):
            index = 3 * vector + component
            fields = (
                native[:, index],
                candidate[:, index],
                candidate[:, index] - native[:, index],
            )
            common = max(abs(fields[0]).max(), abs(fields[1]).max())
            for column, field in enumerate(fields):
                axis = fig.add_subplot(grid[2 * component, column])
                limit = max(abs(field).max(), np.finfo(float).tiny) if column == 2 else common
                artist = axis.tripcolor(
                    tri,
                    field,
                    shading="gouraud",
                    cmap="seismic",
                    vmin=-limit,
                    vmax=limit,
                    rasterized=True,
                )
                axis.add_collection(
                    LineCollection(edges, colors="white", linewidths=0.85, alpha=0.7)
                )
                axis.add_collection(LineCollection(edges, colors=".2", linewidths=0.35, alpha=0.8))
                label = ("DOLFINx/UFL", "PyMHM P3/P1", "PyMHM − DOLFINx")[column]
                axis.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    aspect="equal",
                    xlabel="$x$",
                    ylabel="$y$",
                    title=rf"{label} · ${symbol}_{'xyz'[component]}$",
                )
                bar = fig.colorbar(
                    artist,
                    cax=fig.add_subplot(grid[2 * component + 1, column]),
                    orientation="horizontal",
                )
                bar.locator = MaxNLocator(3)
                bar.update_ticks()
        fig.suptitle(
            f"Independent {name} comparison · same discrete spaces\n"
            r"48 macrotetrahedra · $T=0.5$, $\Delta t=0.005$, $z=0.37$"
        )
        OUTPUT.mkdir(exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(OUTPUT / f"native-{name}.{suffix}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
