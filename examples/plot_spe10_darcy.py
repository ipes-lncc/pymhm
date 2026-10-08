"""Replay SPE10 quadrilateral fields and compare profiles with the published raster."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from pymhm.io.workspace import local_resource, read_resource_text, resource_glob

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from examples.plot_mesh import mark_macro_interfaces
from examples.plot_spe10_data import FIGURES, OUTPUT, data_grid, macro_mesh, panel
from pymhm.postprocessing.visualization import structured_cell_grid


def records() -> list[dict]:
    """Read completed MHM cases in increasing local and skeleton resolution."""
    return sorted(
        [
            json.loads(read_resource_text(path))
            for path in resource_glob(OUTPUT, "darcy-q1-r*-s*.json")
        ],
        key=lambda row: (row["local_refinement"][0], row["skeleton_segments"]),
    )


def q1_grid(path: Path) -> pv.UnstructuredGrid:
    """Reconstruct separate Q1 macro-local display grids from archived nodal values."""
    with np.load(local_resource(path)) as data:
        pressure = data["local_pressure"]
        mesh = macro_mesh()
        refinement = int(round(np.sqrt(pressure.shape[1]))) - 1
        x, y = np.meshgrid(np.linspace(0, 200, refinement + 1), np.linspace(0, 200, refinement + 1))
        offsets = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        lower = mesh.points[mesh.cells[:, 0]]
        points = (offsets[None] + np.column_stack((lower, np.zeros(len(lower))))[:, None]).reshape(
            -1, 3
        )
        indices = np.arange(refinement)[:, None] * (refinement + 1) + np.arange(refinement)
        ids = indices.ravel()
        local = np.column_stack((ids, ids + 1, ids + refinement + 2, ids + refinement + 1))
        cells = (local[None] + np.arange(len(lower))[:, None, None] * len(offsets)).reshape(-1, 4)
        grid = pv.UnstructuredGrid(
            np.column_stack((np.full(len(cells), 4), cells)).ravel(),
            np.full(len(cells), pv.CellType.QUAD),
            points,
        )
        grid.point_data["Pressure"] = pressure.ravel()
    return grid


def fields(row: dict) -> None:
    """Render the coefficient, full broken Q1 pressure and sampled raw flux."""
    path = OUTPUT / row["archive"]
    pressure_grid = q1_grid(path)
    with np.load(local_resource(path)) as data:
        flux_grid = structured_cell_grid(
            (60, 220),
            cell_data={"Raw flux magnitude": np.linalg.norm(data["flux"], axis=2)},
            spacing=(20, 10),
        )
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 1000))
    descriptors = (
        (data_grid(36), "ln(Kx / mD)", "Layer 36: unchanged permeability", (-6.2, 9.1)),
        (
            pressure_grid,
            "Pressure",
            f"Broken Q1 pressure / {row['free_trace_plus_retained']} global DOFs\n"
            f"r={row['local_refinement'][0]}, {row['skeleton_segments']} C0-P1 face segments",
            (min(0, row["pressure_nodal_min"]), max(1, row["pressure_nodal_max"])),
        ),
        (flux_grid, "Raw flux magnitude", "Physical flux at material-pixel centers", None),
    )
    for column, (grid, scalar, title, limits) in enumerate(descriptors):
        plotter.subplot(0, column)
        panel(plotter, grid, scalar, title, limits=limits, cmap="jet", mesh=macro_mesh())
    plotter.screenshot(FIGURES / "darcy-fields.png")
    plotter.close()


def conforming_reference() -> dict:
    """Select the article-grid Q3 comparison independently of other reference grids."""
    rows = [
        json.loads(read_resource_text(path))
        for path in resource_glob(OUTPUT, "reference-q3-*.json")
    ]
    return max(
        (row for row in rows if row["article_grid"]), key=lambda row: row["quadrature_order"]
    )


def reference_fields(row: dict, reference: dict) -> None:
    """Compare pressure at identical material-pixel centers without spatial averaging."""
    with (
        np.load(local_resource(OUTPUT / row["archive"])) as mhm,
        np.load(local_resource(OUTPUT / reference["archive"])) as ref,
    ):
        fields = [ref["pressure"], mhm["pressure"], mhm["pressure"] - ref["pressure"]]
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 1000))
    for column, (values, title) in enumerate(
        zip(
            fields,
            (
                f"Continuous Q3 / {reference['degrees_of_freedom']:,} DOFs",
                f"MHM Q1 / {row['free_trace_plus_retained']} global DOFs",
                "MHM pressure minus conforming Q3",
            ),
            strict=True,
        )
    ):
        plotter.subplot(0, column)
        grid = structured_cell_grid(
            (60, 220),
            cell_data={"Pressure" if column < 2 else "Pressure difference": values},
            spacing=(20, 10),
        )
        extent = float(np.max(abs(values)))
        panel(
            plotter,
            grid,
            "Pressure" if column < 2 else "Pressure difference",
            title + "\nValues at identical material-pixel centers",
            limits=(0, 1) if column < 2 else (-extent, extent),
            cmap="jet" if column < 2 else "RdBu_r",
            mesh=macro_mesh(),
        )
    plotter.screenshot(FIGURES / "darcy-reference-fields.png")
    plotter.close()


def profile_values(data: dict, coordinates: np.ndarray) -> np.ndarray:
    """Evaluate profile samples within their own macro intervals without averaging."""
    values = np.empty(len(coordinates))
    owners = np.minimum((coordinates / 200).astype(int), 10)
    for row in range(11):
        mask = owners == row
        values[mask] = np.interp(
            coordinates[mask], data["profile_points"][row, :, 1], data["profile_pressure"][row]
        )
    return values


def plot_profiles(rows: list[dict]) -> None:
    """Compare both full and zoomed x=199 profiles with published curve samples."""
    published = json.loads(read_resource_text(OUTPUT / "published-profile.json"))
    reference = np.asarray(published["points"])
    uncertainty = published["digitization_uncertainty"]["pressure"]
    conforming = conforming_reference()
    with np.load(local_resource(OUTPUT / conforming["archive"])) as data:
        continuous_y = data["profile_points"][:, :, 1].ravel()
        continuous_p = data["profile_pressure"].ravel()
    available = {row["local_refinement"][0] for row in rows}
    complete = [
        r
        for r in available
        if {x["skeleton_segments"] for x in rows if x["local_refinement"][0] == r} >= {8, 16, 32}
    ]
    refinement = max(complete)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    colors = {8: "#37a122", 16: "#2359dd", 32: "#e02d32"}
    for axis in axes:
        axis.plot(
            continuous_y,
            continuous_p,
            color="#444444",
            ls="--",
            lw=1.0,
            label=f"Continuous Q3 computed here ({conforming['degrees_of_freedom']:,} DOFs)",
        )
        axis.plot(
            reference[:, 0],
            reference[:, 1],
            "k.",
            ms=2.5,
            label="Article Q3 reference (raster samples)",
        )
        axis.fill_between(
            reference[:, 0],
            reference[:, 1] - uncertainty,
            reference[:, 1] + uncertainty,
            color="gray",
            alpha=0.22,
            label="Raster pressure uncertainty ±0.0021",
        )
        for row in rows:
            if row["local_refinement"][0] != refinement or row["skeleton_segments"] not in colors:
                continue
            with np.load(local_resource(OUTPUT / row["archive"])) as data:
                for macro in range(11):
                    axis.plot(
                        data["profile_points"][macro, :, 1],
                        data["profile_pressure"][macro],
                        color=colors[row["skeleton_segments"]],
                        lw=1.15,
                        label=f"PyMHM: H/{row['skeleton_segments']} trace, r={refinement}"
                        if macro == 0
                        else None,
                    )
        mark_macro_interfaces(axis, np.arange(0, 2201, 200))
        axis.set(xlabel="y at x=199 [ft]", ylabel="Pressure")
        axis.grid(alpha=0.15)
    axes[0].set(
        xlim=(0, 2200),
        ylim=(-0.025, 1.025),
        title="Same layer, boundary conditions, 66-square macrogrid and continuous P1 skeleton",
    )
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set(
        xlim=(1420, 1800),
        ylim=(0.23, 0.46),
        title="Channel-region detail: independent one-sided macro values are retained",
    )
    fig.savefig(FIGURES / "darcy-profiles.svg")
    fig.savefig(FIGURES / "darcy-profiles.png", dpi=160)
    plt.close(fig)
    measurements = []
    for row in rows:
        with np.load(local_resource(OUTPUT / row["archive"])) as data:
            difference = profile_values(data, reference[:, 0]) - reference[:, 1]
            with np.load(local_resource(OUTPUT / conforming["archive"])) as ref:
                pressure_difference = data["pressure"] - ref["pressure"]
                flux_difference = data["flux"] - ref["flux"]
        measurements.append(
            {
                "refinement": row["local_refinement"][0],
                "segments": row["skeleton_segments"],
                "coefficient_pixels_aligned": row["coefficient_pixels_aligned"],
                "global_dofs": row["free_trace_plus_retained"],
                "profile_rms_against_raster": float(np.sqrt(np.mean(difference**2))),
                "profile_max_against_raster": float(np.max(abs(difference))),
                "pixel_pressure_rms_against_conforming": float(
                    np.sqrt(np.mean(pressure_difference**2))
                ),
                "pixel_flux_rms_against_conforming": float(
                    np.sqrt(np.mean(np.sum(flux_difference**2, axis=-1)))
                ),
                "inlet_flux": -row["exterior_flux_bottom_right_top_left"][0],
                "macro_balance_max": row["macro_balance_max"],
                "pressure_nodal_range": [row["pressure_nodal_min"], row["pressure_nodal_max"]],
            }
        )
    (OUTPUT / "darcy-comparison.json").write_text(
        json.dumps(
            {
                "published_reference": published["doi"],
                "samples": len(reference),
                "pressure_digitization_uncertainty": uncertainty,
                "measurements": measurements,
                "conforming_reference": conforming["archive"],
                "sampling": (
                    "uniform 60x220 material-pixel centers; RMS samples are not integrated L2 norms"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for segments in (8, 16, 32):
        selected = [
            r for r in measurements if r["segments"] == segments and r["coefficient_pixels_aligned"]
        ]
        axes[0].plot(
            [r["refinement"] for r in selected],
            [r["profile_rms_against_raster"] for r in selected],
            "o-",
            color=colors[segments],
            label=f"H/{segments} trace",
        )
        axes[1].plot(
            [r["refinement"] for r in selected],
            [r["inlet_flux"] for r in selected],
            "o-",
            color=colors[segments],
        )
    unfitted = [
        r for r in measurements if r["segments"] == 32 and not r["coefficient_pixels_aligned"]
    ]
    for axis, key in zip(axes, ("profile_rms_against_raster", "inlet_flux"), strict=True):
        if unfitted:
            axis.plot(
                [r["refinement"] for r in unfitted],
                [r[key] for r in unfitted],
                "s--",
                color="#85569d",
                label="H/32, cells crossing material pixels",
            )
    axes[1].axhline(
        -conforming["outward_flux_bottom"],
        color="#444444",
        ls=":",
        label="Continuous Q3 on the article's grid",
    )
    axes[0].axhspan(
        0,
        uncertainty,
        color="gray",
        alpha=0.2,
        label="Raster placement scale (not a PDE error bound)",
    )
    axes[0].set(
        ylabel="Pressure RMS at published curve samples",
        xlabel="Local subdivisions per macro side",
        title="Pixel-aligned locals: solid; unfitted locals: dashed",
    )
    axes[1].set(
        ylabel="Total inlet flux",
        xlabel="Local subdivisions per macro side",
        title="Independent integrated-flow diagnostic",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(FIGURES / "darcy-refinement.svg")
    fig.savefig(FIGURES / "darcy-refinement.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Redraw all completed records without importing any reference execution code."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = records()
    finest = max(
        (r for r in rows if r["skeleton_segments"] == 32 and r["coefficient_pixels_aligned"]),
        key=lambda r: r["local_refinement"][0],
    )
    fields(finest)
    reference_fields(finest, conforming_reference())
    plot_profiles(rows)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_spe10_darcy").main()
