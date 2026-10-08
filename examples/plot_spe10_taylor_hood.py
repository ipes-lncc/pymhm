"""Compare unchanged SPE10 MHM fields with successively refined Taylor--Hood fields.

The default replays archived display samples. ``--sample`` evaluates a selected
reference on the broken P3 display grid and requires its full coefficient file.
Finite-element error norms are computed separately by the reference driver;
the display triangulation is not used for integration.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.ticker import NullFormatter

from examples.plot_mesh import mark_macro_interfaces
from examples.plot_spe10_data import FIGURES, OUTPUT, ROOT, panel
from examples.solve_spe10_taylor_hood import load_field
from pymhm import TriangleMesh
from pymhm.postprocessing.visualization import broken_triangle_grid

MHM = "flow-layer1-n6x11-p3-r10-s10-q8-pointwise-2017.npz"
SAMPLES = OUTPUT / "taylor-hood-display.npz"
METADATA = OUTPUT / "taylor-hood-display.json"


def checked_records() -> list[dict]:
    """Validate every displayed reference and norm against its acquisition record."""
    records = sorted(
        (json.loads(path.read_text()) for path in OUTPUT.glob("taylor-hood-[0-9]*x[0-9]*.json")),
        key=lambda row: row["unknowns"],
    )
    if len(records) < 2:
        raise ValueError("At least two completed Taylor-Hood refinements are required")
    layer_hash = hashlib.sha256((OUTPUT / "layer-1.npz").read_bytes()).hexdigest()
    mhm_hash = hashlib.sha256((OUTPUT / MHM).read_bytes()).hexdigest()
    for record in records:
        if (
            hashlib.sha256((OUTPUT / record["archive"]).read_bytes()).hexdigest()
            != record["sha256"]
            or record["layer_sha256"] != layer_hash
        ):
            raise ValueError(f"Stale reference archive or material: {record['archive']}")
        nx, ny = record["mesh_shape"]
        path = OUTPUT / f"taylor-hood-mhm-{nx}x{ny}.json"
        if path.exists():
            comparison = json.loads(path.read_text())
            if (
                comparison["reference"] != record["archive"]
                or comparison["reference_sha256"] != record["sha256"]
                or comparison["reference_coefficients_sha256"] != record["coefficient_sha256"]
                or comparison["mhm"] != MHM
                or comparison["mhm_sha256"] != mhm_hash
            ):
                raise ValueError(f"Stale integrated comparison: {path.name}")
    return records


def mhm_grid() -> tuple[pv.UnstructuredGrid, TriangleMesh]:
    """Sample the broken local P3 fields without identifying opposite traces."""
    with np.load(OUTPUT / MHM) as data:
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        meshes = tuple(
            TriangleMesh(points, cells)
            for points, cells in zip(data["local_points"], data["local_cells"], strict=True)
        )
        states = np.concatenate(
            (data["local_velocity"], data["local_pressure"][..., None]), axis=-1
        )
        grid = broken_triangle_grid(meshes, states, degree=3, name="State", subdivision=6)
    return grid, macro


def sample_reference(record: dict, grid: pv.UnstructuredGrid) -> None:
    """Archive actual P2/P1 evaluations at the common broken display vertices."""
    coefficient_path = ROOT / record["coefficient_archive"]
    if hashlib.sha256(coefficient_path.read_bytes()).hexdigest() != record["coefficient_sha256"]:
        raise ValueError("Reference coefficient checksum does not match its numerical record")
    reference = load_field(coefficient_path)
    velocity, pressure = reference.evaluate(grid.points[:, :2])
    np.savez_compressed(SAMPLES, velocity=velocity, pressure=pressure)
    METADATA.write_text(
        json.dumps(
            {
                "reference": record["archive"],
                "reference_coefficients_sha256": record["coefficient_sha256"],
                "mhm": MHM,
                "mhm_sha256": hashlib.sha256((OUTPUT / MHM).read_bytes()).hexdigest(),
                "display_subdivision": 6,
                "display_points_sha256": hashlib.sha256(grid.points.tobytes()).hexdigest(),
                "sampling": (
                    "Actual P2/P1 reference and broken P3/P3 MHM evaluations at identical "
                    "display vertices; piecewise linear rendering, no interface averaging"
                ),
                "archive": SAMPLES.name,
                "sha256": hashlib.sha256(SAMPLES.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


def field_comparison(grid: pv.UnstructuredGrid, macro: TriangleMesh, record: dict) -> None:
    """Render common-scale fields and vector/pressure differences on six panels."""
    metadata = json.loads(METADATA.read_text())
    checks = (
        metadata["reference"] == record["archive"],
        metadata["reference_coefficients_sha256"] == record["coefficient_sha256"],
        metadata["mhm_sha256"] == hashlib.sha256((OUTPUT / MHM).read_bytes()).hexdigest(),
        metadata["display_points_sha256"] == hashlib.sha256(grid.points.tobytes()).hexdigest(),
        metadata["sha256"] == hashlib.sha256(SAMPLES.read_bytes()).hexdigest(),
    )
    if not all(checks):
        raise ValueError("Stale Taylor-Hood display samples: regenerate with --sample")
    with np.load(SAMPLES) as data:
        reference_u, reference_p = data["velocity"], data["pressure"]
    state = grid["State"]
    velocity = [np.linalg.norm(reference_u, axis=1), np.linalg.norm(state[:, :2], axis=1)]
    pressure = [reference_p, state[:, 2]]
    velocity.append(np.linalg.norm(state[:, :2] - reference_u, axis=1))
    pressure.append(state[:, 2] - reference_p)
    velocity_limits = (0.0, max(float(values.max()) for values in velocity[:2]))
    pressure_limits = (
        min(float(values.min()) for values in pressure[:2]),
        max(float(values.max()) for values in pressure[:2]),
    )
    pressure_difference = float(np.max(np.abs(pressure[2])))
    nx, ny = record["mesh_shape"]
    headings = (
        f"Taylor-Hood P2/P1\n{nx} x {ny} rectangles, split into triangles",
        "MHM USFEM P3/P3\nPublished 264-macrotriangle configuration",
        "MHM minus Taylor-Hood\nSame physical coordinates and pressure level",
    )
    plotter = pv.Plotter(shape=(2, 3), off_screen=True, window_size=(2100, 2000))
    for row, fields in enumerate((velocity, pressure)):
        for column, values in enumerate(fields):
            plotter.subplot(row, column)
            scalar = (
                ("Velocity magnitude" if row == 0 else "Pressure")
                if column < 2
                else ("Velocity difference" if row == 0 else "Pressure difference")
            )
            name = f"{scalar}:{row}:{column}"
            grid.point_data[name] = values
            # A shallow copy gives each mapper an explicit scalar association.
            display = grid.copy(deep=False)
            display.point_data[scalar] = values
            limits = velocity_limits if row == 0 else pressure_limits
            if column == 2:
                limits = (
                    (0.0, float(values.max()))
                    if row == 0
                    else (
                        -pressure_difference,
                        pressure_difference,
                    )
                )
            panel(
                plotter,
                display,
                scalar,
                headings[column],
                limits=limits,
                cmap="RdBu_r" if row == 1 and column == 2 else "viridis",
                mesh=macro,
            )
    plotter.screenshot(FIGURES / "brinkman-taylor-hood-fields.png")
    plotter.close()


def profiles(records: list[dict]) -> None:
    """Compare continuous reference profiles with independent one-sided MHM values."""
    with np.load(OUTPUT / MHM) as data:
        points = data["profile_points"].copy()
        mhm = np.concatenate(
            (data["profile_velocity"], data["profile_pressure"][..., None]), axis=-1
        )
        breaks = data["profile_breaks"] * 2200
    for suffix, domain in (("profiles", (0, 2200)), ("profiles-detail", (150, 550))):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.79, wspace=0.36)
        for index, record in enumerate(records[-2:]):
            with np.load(OUTPUT / record["archive"]) as data:
                values = np.column_stack((data["profile_velocity"], data["profile_pressure"]))
                for component, axis in enumerate(axes):
                    nx, ny = record["mesh_shape"]
                    axis.plot(
                        data["profile_points"][:, 1],
                        values[:, component],
                        color="#777777" if index == 0 else "black",
                        ls="--" if index == 0 else "-",
                        lw=1.4,
                        label=f"Taylor-Hood {nx} x {ny}",
                    )
        for component, axis in enumerate(axes):
            for index, segment in enumerate(mhm):
                axis.plot(
                    points[index, :, 1],
                    segment[:, component],
                    color="#c34622",
                    ls="--",
                    lw=1.2,
                    label="MHM P3/P3" if index == 0 else None,
                )
            mark_macro_interfaces(axis, breaks)
            axis.set(
                xlabel="y at x=199 [ft]",
                ylabel=(r"Velocity $u_x$", r"Velocity $u_y$", "Pressure")[component],
                xlim=domain,
            )
            if suffix.endswith("detail"):
                visible = (points[..., 1] >= domain[0]) & (points[..., 1] <= domain[1])
                low, high = mhm[..., component][visible].min(), mhm[..., component][visible].max()
                for record in records[-2:]:
                    with np.load(OUTPUT / record["archive"]) as data:
                        mask = (data["profile_points"][:, 1] >= domain[0]) & (
                            data["profile_points"][:, 1] <= domain[1]
                        )
                        values = (
                            data["profile_velocity"][:, component]
                            if component < 2
                            else data["profile_pressure"]
                        )
                        low, high = min(low, values[mask].min()), max(high, values[mask].max())
                padding = max(1e-6, 0.06 * (high - low))
                axis.set_ylim(low - padding, high + padding)
            axis.grid(alpha=0.15)
            axis.xaxis.label.set_fontsize(11)
            axis.yaxis.label.set_fontsize(11)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncols=3, frameon=False
        )
        fig.savefig(
            FIGURES / f"brinkman-taylor-hood-{suffix}.svg", bbox_inches="tight", pad_inches=0.12
        )
        fig.savefig(
            FIGURES / f"brinkman-taylor-hood-{suffix}.png",
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.12,
        )
        plt.close(fig)


def convergence(records: list[dict]) -> None:
    """Show successive integrated reference changes separately from MHM distances."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    differences = [record for record in records if "successive_difference" in record]
    for axis, variable in zip(axes, ("velocity", "pressure"), strict=True):
        axis.loglog(
            [record["unknowns"] for record in differences],
            [
                100 * record["successive_difference"][f"{variable}_relative"]
                for record in differences
            ],
            "o-",
            color="#215b8f",
            label="Successive Taylor-Hood difference",
        )
        comparisons = []
        for record in records:
            nx, ny = record["mesh_shape"]
            path = OUTPUT / f"taylor-hood-mhm-{nx}x{ny}.json"
            if path.exists():
                norms = max(
                    json.loads(path.read_text())["norms"], key=lambda row: row["quadrature_order"]
                )
                comparisons.append((record["unknowns"], 100 * norms[f"{variable}_relative"]))
        if comparisons:
            values = np.array(comparisons)
            axis.loglog(
                values[:, 0],
                values[:, 1],
                "s--",
                color="#c34622",
                label="Fixed MHM vs. Taylor-Hood",
            )
        axis.set(
            xlabel="Taylor-Hood unknowns",
            ylabel=r"Relative $L^2$ difference [%]",
            title=variable.capitalize(),
        )
        ticks = [record["unknowns"] for record in records]
        axis.set_xticks(ticks, [f"{value / 1e6:.3g} M" for value in ticks])
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(which="both", alpha=0.15)
        axis.legend(fontsize=8)
    fig.savefig(FIGURES / "brinkman-taylor-hood-refinement.svg")
    fig.savefig(FIGURES / "brinkman-taylor-hood-refinement.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Replay verified samples or refresh them from the finest available reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="evaluate reference coefficients")
    args = parser.parse_args()
    records = checked_records()
    grid, macro = mhm_grid()
    if args.sample:
        sample_reference(records[-1], grid)
    field_comparison(grid, macro, records[-1])
    profiles(records)
    convergence(records)


if __name__ == "__main__":
    main()
