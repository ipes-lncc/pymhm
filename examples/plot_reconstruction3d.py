"""Publication figures from persisted tetrahedral MHM fields and executed RT bases."""

import hashlib
import json

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter, ScalarFormatter
from threadpoolctl import threadpool_limits

from examples.reconstruction3d_data import fields
from examples.reconstruction3d_replay import evaluate, profile
from examples.tetra_section_samples import section_grid
from pymhm import TetraMesh
from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)

ROOT = case_workspace()
DATA = ROOT / "examples/results/reconstruction3d"
OUTPUT = ROOT / "docs/figures/reconstruction3d"


def save(figure: plt.Figure, name: str) -> None:
    """Export identical vector and raster layouts with dedicated colorbar/legend regions."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def convergence(suite: str, record: dict) -> None:
    """Show all acquired errors, estimator terms and measured effectivity without inferred rates."""
    rows = record["rows"]
    size = [r["macro_cells"] for r in rows]
    figure = plt.figure(figsize=(15, 6.7), layout="constrained")
    grid = figure.add_gridspec(2, 3, height_ratios=[5, 1.6])
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]
    for key, label in (
        ("pressure_l2", "Pressure"),
        ("raw_flux_l2", "Raw flux"),
        ("rt_flux_l2", "RT1 flux"),
        ("indicator", "Energy indicator"),
    ):
        axes[0].loglog(size, [r[key] for r in rows], "o-", label=label)
    for key in rows[0]["terms"]:
        axes[1].loglog(size, [r["terms"][key] for r in rows], "o-", label=key.replace("_", " "))
    axes[2].semilogx(size, [r["effectivity"] for r in rows], "o-", label="Indicator / energy error")
    axes[2].axhline(1, color="black", ls=":", label="Unit ratio")
    for i, axis in enumerate(axes):
        axis.set(xlabel="Macrotetrahedra", ylabel=("Absolute norm" if i < 2 else "Ratio"))
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=0.2)
        legend = figure.add_subplot(grid[1, i])
        legend.set_axis_off()
        legend.legend(*axis.get_legend_handles_labels(), loc="center", frameon=False)
    axes[0].set_title("Physical errors and indicator")
    axes[1].set_title("Four estimator terms")
    axes[2].set_title("Measured effectivity")
    save(figure, f"{suite}-convergence")


def section(suite: str, row: dict, *, localized: bool | None = None) -> None:
    """Replay full broken pressure/flux polynomials on z=0.37 using the archived RT basis."""
    path = DATA / row["archive"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != row["archive_sha256"]:
        raise ValueError("field archive digest differs from its numerical record")
    with np.load(local_resource(path)) as archive:
        coarse = TetraMesh(archive["macro_points"], archive["macro_cells"])
        macro = section_grid(coarse, refinement=1)["segments"]
        samples, cells, actual = [], [], []
        offset = 0
        for cell in range(len(coarse.cells)):
            fine = TetraMesh(archive[f"points_{cell}"], archive[f"cells_{cell}"])
            display = section_grid(fine)
            if not len(display["points"]):
                continue
            bary = display["barycentric"]
            owners = display["parents"]
            rt_degree = (
                int(archive["reconstruction_degree"]) if "reconstruction_degree" in archive else 1
            )
            samples.append(display["points"])
            cells.append(display["cells"] + offset)
            actual.append(evaluate(archive, cell, owners, bary))
            offset += len(display["points"])
    points, connectivity, actual = (
        np.concatenate(samples),
        np.concatenate(cells),
        np.concatenate(actual),
    )
    p, q, _ = fields(points, suite == "adaptive" if localized is None else localized)
    exact = np.column_stack((p, q))
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], connectivity)
    figure = plt.figure(figsize=(13, 18), layout="constrained")
    grid = figure.add_gridspec(8, 3, height_ratios=[1, 0.055] * 4, hspace=0.12, wspace=0.1)
    for component, label in enumerate(("$p$", "$q_x$", "$q_y$", "$q_z$")):
        error = actual[:, component] - exact[:, component]
        lower = min(exact[:, component].min(), actual[:, component].min())
        upper = max(exact[:, component].max(), actual[:, component].max())
        amplitude = max(abs(error).max(), np.finfo(float).eps)
        for column, (values, title) in enumerate(
            (
                (exact[:, component], "Exact"),
                (actual[:, component], "MHM" if component == 0 else f"RT{rt_degree}"),
                (error, "MHM − exact" if component == 0 else f"RT{rt_degree} − exact"),
            )
        ):
            axis = figure.add_subplot(grid[2 * component, column])
            field_amplitude = max(abs(lower), abs(upper), np.finfo(float).eps)
            clim = (
                (-amplitude, amplitude)
                if column == 2
                else ((-field_amplitude, field_amplitude) if component else (lower, upper))
            )
            cmap = "seismic" if column == 2 or component else "viridis"
            artist = axis.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap=cmap,
                vmin=clim[0],
                vmax=clim[1],
                rasterized=True,
            )
            axis.add_collection(LineCollection(macro, colors="white", linewidths=0.9, alpha=0.7))
            axis.add_collection(LineCollection(macro, colors="0.22", linewidths=0.5, alpha=0.8))
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="$x$",
                ylabel="$y$",
                title=f"{title} · {label}",
            )
            colorbar = figure.colorbar(
                artist,
                cax=figure.add_subplot(grid[2 * component + 1, column]),
                orientation="horizontal",
            )
            colorbar.locator = MaxNLocator(3)
            colorbar.update_ticks()
    method = (
        f"Local P{row['local_degree']} · trace P{row['trace_degree']} · "
        f"{row['trace_subdivisions'] ** 2} subfaces per macroface\n"
        if "local_degree" in row
        else "Local P3 · constant macroface trace\n"
    )
    figure.suptitle(
        method + f"{len(coarse.cells)} macrotetrahedra · section z = 0.37\n"
        f"Relative volume errors: pressure {row.get('pressure_relative', float('nan')):.2%}, "
        f"RT flux {row.get('rt_flux_relative', float('nan')):.2%}"
        if "pressure_relative" in row
        else f"{len(coarse.cells)} macrotetrahedra · section z = 0.37"
    )
    save(figure, f"{suite}-components")


def profiles(rows: list[dict]) -> None:
    """Compare exact and numerical profiles, keeping independent one-sided interface values."""
    definitions = (
        (np.array([0.0, 0.413, 0.37]), np.array([1.0, 0.413, 0.37]), "$x$", "$y=0.413, z=0.37$"),
        (np.array([0.317, 0.0, 0.37]), np.array([0.317, 1.0, 0.37]), "$y$", "$x=0.317, z=0.37$"),
    )
    figure = plt.figure(figsize=(12, 14), layout="constrained")
    grid = figure.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.18])
    error_figure = plt.figure(figsize=(12, 14), layout="constrained")
    error_grid = error_figure.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.18])
    for direction, (first, last, coordinate, title) in enumerate(definitions):
        sampled = []
        for row in rows:
            with np.load(local_resource(DATA / row["archive"])) as archive:
                sampled.append(profile(archive, first, last))
        parameter = np.linspace(0, 1, 501)
        p, q, _ = fields(first + parameter[:, None] * (last - first), True)
        exact = np.column_stack((p, q))
        for component, label in enumerate(("$p$", "$q_x$", "$q_y$", "$q_z$")):
            axis = figure.add_subplot(grid[component, direction])
            error_axis = error_figure.add_subplot(error_grid[component, direction])
            axis.plot(parameter, exact[:, component], color="black", lw=1.4, label="Exact")
            for index, (row, sample) in enumerate(zip(rows, sampled, strict=True)):
                name = (
                    f"Local P{row['local_degree']}, trace P{row['trace_degree']}, "
                    f"s={row['trace_subdivisions']}; flux RT{row['reconstruction_degree']}"
                )
                p, q, _ = fields(sample["points"].reshape(-1, 3), True)
                expected = np.column_stack((p, q)).reshape(sample["actual"].shape)
                for segment in range(len(sample["parameter"])):
                    axis.plot(
                        sample["parameter"][segment],
                        sample["actual"][segment, :, component],
                        color=("#757575", "#0072B2")[index],
                        ls=("--", "-")[index],
                        lw=1,
                        label=name if segment == 0 else None,
                    )
                    if index == len(rows) - 1:
                        error_axis.plot(
                            sample["parameter"][segment],
                            (sample["actual"] - expected)[segment, :, component],
                            color="#0072B2",
                            lw=1,
                            label=name if segment == 0 else None,
                        )
            for target in (axis, error_axis):
                for crossing in sampled[-1]["macro_breaks"][1:-1]:
                    target.axvline(crossing, color="0.75", ls=":", lw=0.7)
                target.set(
                    xlabel=coordinate,
                    ylabel=label if target is axis else f"Error in {label}",
                    title=title if component == 0 else None,
                    xlim=(0, 1),
                )
                target.grid(alpha=0.15)
            error_axis.axhline(0, color="black", lw=0.7)
        for target, layout, source in (
            (figure, grid, axis),
            (error_figure, error_grid, error_axis),
        ):
            legend = target.add_subplot(layout[4, direction])
            legend.set_axis_off()
            legend.legend(
                *source.get_legend_handles_labels(), loc="center", frameon=False, fontsize=8
            )
    save(figure, "adaptive-profiles")
    save(error_figure, "adaptive-profile-errors")


def main() -> None:
    """Render complete records without changing any solved coefficient or error norm."""
    for suite in ("uniform", "adaptive"):
        record = json.loads(read_resource_text(DATA / f"{suite}.json"))
        if record.get("source_changed_during_run") is not False:
            raise ValueError(f"{suite} campaign is incomplete or its sources changed")
        convergence(suite, record)
        section(
            "adaptive-constant-trace" if suite == "adaptive" else suite,
            record["rows"][-1],
            localized=suite == "adaptive",
        )
    resolution = json.loads(read_resource_text(DATA / "resolution.json"))
    if resolution.get("source_changed_during_run") is not False:
        raise ValueError("resolution campaign is incomplete or its sources changed")
    section("adaptive-rt2", resolution["rows"][-1], localized=True)
    control = json.loads(read_resource_text(DATA / "reconstruction-order.json"))
    if control.get("source_changed_during_run") is not False:
        raise ValueError("reconstruction-order control is incomplete or its sources changed")
    section("adaptive", control, localized=True)
    profiles([resolution["rows"][0], control])
    for name in ("uniform", "adaptive", "resolution", "reconstruction-order"):
        (OUTPUT / f"{name}.json").write_bytes(read_resource_bytes(DATA / f"{name}.json"))


def cli() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    with threadpool_limits(1):
        main()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_reconstruction3d").cli()
