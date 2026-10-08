"""Render all archived publication comparisons without rerunning finite element solves."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from pymhm import TriangleMesh

ERROR_NAMES = (
    "pressure_l2",
    "flux_l2",
    "macroconstant_pressure_l2",
    "macroconstant_projection_l2",
)


def comparison_field(field: str, formulation: str) -> str:
    """Use the integrated quadratic potential for the published updated-pressure curve."""
    if formulation == "mixed" and field == "pressure_l2":
        return "rt0_integrated_quadratic_pressure_l2"
    return field


def plot_comparison(
    measured: list[dict[str, float | int]],
    published: list[dict[str, float]],
    directory: Path,
    formulation: str = "primal",
) -> None:
    """Plot both datasets without shifting scales or fitting a numerical constant."""
    labels = (
        r"$\|p-p_h\|_0$",
        r"$\|q-q_h\|_0$",
        r"$\|p-p_0\|_0$",
        r"$\|\Pi_0p-p_0\|_0$",
    )
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, field, label in zip(axes.flat, ERROR_NAMES, labels, strict=True):
        axis.loglog(
            [row["h_nominal"] for row in published],
            [row[field] for row in published],
            "ko--",
            label="Published, digitized",
            markersize=5,
        )
        axis.loglog(
            [row["grid_spacing"] for row in measured],
            [row[comparison_field(field, formulation)] for row in measured],
            "s-",
            color="tab:blue",
            markerfacecolor="none",
            label=("pyMHM RT0 + quadratic potential" if formulation == "mixed" else "pyMHM P1"),
        )
        axis.set(
            title=label,
            xlabel="Published h / candidate grid spacing (mesh caveat applies)",
            ylabel="Absolute L2 error",
        )
        axis.set_xticks([1 / n for n in (64, 32, 16, 8, 4)], ["1/64", "1/32", "1/16", "1/8", "1/4"])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(which="both", alpha=0.25)
        axis.legend(frameon=False)
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "darcy-2013-comparison.svg")
    figure.savefig(directory / "darcy-2013-comparison.png", dpi=180)
    plt.close(figure)


def coarse_mesh(n: int, kind: str) -> TriangleMesh:
    """Construct the declared diagonal or criss-cross partition without fitted parameters."""
    mesh = TriangleMesh.unit_square(n)
    if kind == "diagonal":
        return mesh
    points = mesh.points.tolist()
    cells = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            corners = [a, a + 1, a + n + 2, a + n + 1]
            center = len(points)
            points.append([(i + 0.5) / n, (j + 0.5) / n])
            cells.extend((corners[k], corners[(k + 1) % 4], center) for k in range(4))
    return TriangleMesh(np.array(points), np.array(cells, dtype=np.int64))


def plot_results(path: Path) -> None:
    """Render the saved pressure fields and published-error comparison without FEM imports."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
    from matplotlib.tri import Triangulation

    from examples.plot_mesh import draw_macro_mesh

    data = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[1]
    folder = (
        root
        / "docs/figures/reproduction/stokes"
        / (f"{data.get('mesh_kind', 'diagonal')}-p{data['local_degree']}")
    )
    folder.mkdir(parents=True, exist_ok=True)
    summary = {
        **data,
        "results": [
            {key: value for key, value in row.items() if key != "pressure_field"}
            for row in data["results"]
        ],
    }
    (folder / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure_number = 3 + data["trace_degree"]
    reference = root / f"examples/results/published/araya2017_figure{figure_number}.csv"
    with reference.open(newline="") as stream:
        published = [
            {key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)
        ]
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 10.2), constrained_layout=True)
    norms = (
        "velocity_l2",
        "pressure_l2",
        "velocity_h1_seminorm",
        "pseudostress_l2",
        "pseudostress_hdiv_broken",
        "divergence_l2",
    )
    titles = (
        r"Velocity $L^2$ error",
        r"Pressure $L^2$ error",
        "Velocity broken\n" + r"$H^1$ seminorm error",
        r"Pseudostress $L^2$ error",
        "Pseudostress broken\n" + r"$H(\mathrm{div})$ error",
        "Velocity divergence\n" + r"$L^2$ norm",
    )
    for axis, key, title in zip(axes.flat, norms, titles, strict=True):
        numerical_label = "PyMHM + DOLFINx/UFL"
        if key == "pseudostress_l2":
            numerical_label = "Full pseudostress error"
        elif key == "pseudostress_hdiv_broken":
            numerical_label = "Standard H(div)"
        axis.loglog(
            [r["grid_spacing"] for r in data["results"]],
            [r[key] for r in data["results"]],
            "o-",
            label=numerical_label,
        )
        if key == "pseudostress_l2":
            axis.loglog(
                [r["grid_spacing"] for r in data["results"]],
                [r["velocity_h1_seminorm"] for r in data["results"]],
                "s:",
                label="Velocity gradient error\n(distinct norm)",
            )
        if key == "pseudostress_hdiv_broken":
            axis.loglog(
                [r["grid_spacing"] for r in data["results"]],
                [np.sqrt(2 * r[key] ** 2 - r["pseudostress_l2"] ** 2) for r in data["results"]],
                "s:",
                label="Domain-scaled norm\nEq. (8)",
            )
        if key in published[0]:
            axis.loglog(
                [r["H_table1_same_index"] for r in published],
                [r[key] for r in published],
                "x--",
                label=f"Figure {figure_number}; Table 1 H",
            )
        axis.set(
            title=title,
            xlabel="Table 1 H / grid spacing",
            ylabel="Absolute error",
        )
        ticks = [1 / n for n in (128, 64, 32, 16, 8, 4)]
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        axis.xaxis.set_major_formatter(
            FixedFormatter(["1/128", "1/64", "1/32", "1/16", "1/8", "1/4"])
        )
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.title.set_fontsize(13)
        axis.xaxis.label.set_fontsize(12)
        axis.yaxis.label.set_fontsize(12)
        axis.tick_params(axis="both", labelsize=11.5)
        axis.grid(which="both", alpha=0.25)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), frameon=False, fontsize=11.5)
    figure.savefig(folder / "stokes-2017-convergence.svg")
    figure.savefig(folder / "stokes-2017-convergence.png", dpi=180)
    plt.close(figure)
    for row in data["results"]:
        if "pressure_field" not in row:
            continue
        values = row["pressure_field"]
        coordinates = np.array(values["points"])
        mesh = Triangulation(*coordinates.T, np.array(values["cells"]))
        macro_mesh = coarse_mesh(row["n"], row.get("mesh_kind", data.get("mesh_kind", "diagonal")))
        numerical, exact = np.array(values["numerical"]), np.array(values["exact"])
        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        maximum = max(np.abs(exact).max(), np.abs(numerical).max())
        for axis, value, title in zip(
            axes[:2],
            (exact, numerical),
            (
                "Analytical pressure",
                "PyMHM + DOLFINx/UFL\n"
                f"USFEM P{data['local_degree']}/P{data['local_degree']} · n = {row['n']}",
            ),
            strict=True,
        ):
            artist = axis.tripcolor(
                mesh,
                value,
                shading="gouraud",
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
                rasterized=True,
            )
            axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
            axis.title.set_fontsize(11)
            draw_macro_mesh(axis, macro_mesh, label=True)
            axis.legend(loc="upper right", fontsize=7, framealpha=0.8, handlelength=1.5)
        figure.colorbar(artist, ax=axes[:2], shrink=0.75, label="Pressure")
        artist = axes[2].tripcolor(
            mesh,
            np.abs(numerical - exact),
            shading="gouraud",
            cmap="magma",
            vmin=0,
            rasterized=True,
        )
        axes[2].set(title="Sampled absolute pressure error", xlabel="x", ylabel="y", aspect="equal")
        axes[2].title.set_fontsize(11)
        draw_macro_mesh(axes[2], macro_mesh, label=True)
        axes[2].legend(loc="upper right", fontsize=7, framealpha=0.8, handlelength=1.5)
        figure.colorbar(artist, ax=axes[2], shrink=0.75)
        figure.savefig(folder / f"stokes-2017-pressure-n{row['n']}.svg", dpi=240)
        figure.savefig(folder / f"stokes-2017-pressure-n{row['n']}.png", dpi=180)
        plt.close(figure)


def main() -> None:
    """Plot the original and matched discretizations from their measured records."""
    root = Path(__file__).resolve().parents[1]
    results = root / "examples" / "results"
    output = root / "docs" / "figures" / "reproduction"
    with (results / "published" / "harder2013_figure5.csv").open(newline="") as stream:
        published = [
            {key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)
        ]
    for name, folder in (
        ("darcy_2013_comparison.json", output),
        ("darcy_2013_mixed_comparison.json", output / "mixed"),
    ):
        data = json.loads((results / name).read_text())
        plot_comparison(data["results"], published, folder, data["formulation"])
        (folder / "darcy-2013-comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    for name in (
        "stokes_2017_ufl.json",
        "stokes_2017_ufl_crisscross.json",
        "stokes_2017_ufl_p3.json",
    ):
        plot_results(results / name)


if __name__ == "__main__":
    main()
