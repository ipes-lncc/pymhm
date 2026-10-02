"""Plot archived NeoPZ and pyMHM mixed Darcy comparisons without new solves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from analytical_darcy import darcy_fields
from plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from threadpoolctl import threadpool_limits

from pymhm import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "examples/results/neopz/comparison.json"
DEFAULT_OUTPUT = ROOT / "docs/figures/neopz"


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    """Export PNG and SVG while retaining vector labels and macro contours."""
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=190, bbox_inches="tight", pad_inches=0.07)
    plt.close(figure)


def load_archive(row: dict[str, Any], directory: Path) -> dict[str, np.ndarray]:
    """Read the exact comparison archive, checking its recorded checksum."""
    path = directory / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError(f"Reference archive checksum mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def select_row(
    rows: list[dict[str, Any]], case: str, resolution: int, segments: int
) -> dict[str, Any]:
    """Select one recorded case without substituting a different mesh or trace."""
    selected = [
        row
        for row in rows
        if row["case"] == case
        and row["macro_resolution"] == resolution
        and row["trace_segments"] == segments
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one archive for {case}, n={resolution}, segments={segments}; "
            f"found {len(selected)}"
        )
    return selected[0]


def macro_mesh(data: dict[str, np.ndarray]) -> TriangleMesh:
    """Recover the actual macro mesh stored with the reference fields."""
    return TriangleMesh(data["macro_points"], data["macro_cells"])


def broken_fields(
    data: dict[str, np.ndarray], prefix: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover broken vertex samples of P0 pressure and affine RT0 flux.

    Duplicate triangle vertices preserve interelement jumps. The three
    archived barycentric flux samples determine each affine component exactly;
    no smoothing or projection across neighboring cells is introduced.
    """
    vertices = data["points"][data["cells"]]
    barycentric = data["barycentric"]
    if barycentric.shape != (3, 3):
        raise ValueError("RT0 visualization requires three independent barycentric samples")
    count = len(vertices)
    pressure = data[f"{prefix}pressure"].reshape(count, 3)
    if not np.allclose(pressure, pressure[:, :1], rtol=0, atol=1e-13):
        raise ValueError("Archived pressure samples do not represent cellwise P0 values")
    flux = data[f"{prefix}flux"].reshape(count, 3, 2)
    flux_vertices = np.linalg.solve(barycentric, flux)
    return (
        vertices.reshape(-1, 2),
        np.arange(3 * count).reshape(-1, 3),
        np.repeat(pressure.mean(axis=1), 3),
        flux_vertices.reshape(-1, 2),
    )


def color_limits(values: list[np.ndarray]) -> tuple[float, float]:
    """Choose a common physical scale, including a visible constant-field scale."""
    lower = min(float(np.min(value)) for value in values)
    upper = max(float(np.max(value)) for value in values)
    magnitude = max(abs(lower), abs(upper), 1.0)
    if upper - lower <= 1e-10 * magnitude:
        return lower - 0.05 * magnitude, upper + 0.05 * magnitude
    if lower < 0 < upper:
        return -max(abs(lower), abs(upper)), max(abs(lower), abs(upper))
    return lower, upper


def plot_convergence(rows: list[dict[str, Any]], directory: Path, output: Path) -> None:
    """Compare cosine errors and implementation differences for both trace spaces."""
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    colors = {1: "#7b4e99", 2: "#167a86"}
    for segments in (1, 2):
        series = sorted(
            (row for row in rows if row["case"] == "cos" and row["trace_segments"] == segments),
            key=lambda row: row["macro_resolution"],
        )
        if len(series) < 3:
            raise ValueError(f"At least three cosine mesh levels are required for s={segments}")
        diameters = np.array(
            [macro_mesh(load_archive(row, directory)).lengths.max() for row in series]
        )
        trace_label = "s=1: restricted MHM" if segments == 1 else "s=2: full RT0 trace"
        for column, quantity in enumerate(("pressure", "flux")):
            reference = np.array([row[f"reference_{quantity}_error_l2"] for row in series])
            ours = np.array([row[f"{quantity}_error_l2"] for row in series])
            axes[0, column].loglog(
                diameters,
                reference,
                "o-",
                color=colors[segments],
                markerfacecolor="none",
                markersize=7,
                label=f"NeoPZ, {trace_label}",
            )
            axes[0, column].loglog(
                diameters,
                ours,
                "x--",
                color=colors[segments],
                markersize=6,
                label=f"pyMHM, s={segments}",
            )
            difference = np.array([row[f"{quantity}_difference_l2"] for row in series])
            positive = difference > 0
            axes[1, column].loglog(
                diameters[positive],
                difference[positive],
                "o-",
                color=colors[segments],
                label=trace_label,
            )
            if np.any(~positive):
                axes[1, column].text(
                    0.02,
                    0.06 + 0.07 * (segments - 1),
                    f"s={segments}: {np.count_nonzero(~positive)} exact zeros omitted",
                    transform=axes[1, column].transAxes,
                    fontsize=8,
                )
        if segments == 2:
            for column, quantity in enumerate(("pressure", "flux")):
                base = series[-1][f"{quantity}_error_l2"]
                axes[0, column].loglog(
                    diameters[-3:],
                    base * diameters[-3:] / diameters[-1],
                    ":",
                    color="#777777",
                    label="Slope 1 guide",
                )
    for column, quantity in enumerate(("Pressure", "Flux")):
        axes[0, column].set_title(f"{quantity}: error against the analytical solution")
        axes[0, column].set_ylabel("L² error")
        axes[1, column].set_title(f"{quantity}: pyMHM − NeoPZ field difference")
        axes[1, column].set_ylabel("L² difference")
    for axis in axes.ravel():
        axis.set_xlabel("Actual macrotriangle diameter H")
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.23)
        axis.legend(fontsize=8)
    figure.suptitle("Cosine Darcy: matched RT0/P0 spaces and two skeletal partitions")
    figure.supxlabel(
        "Two fine segments per macroedge. The s=1 reference restricts the NeoPZ mixed "
        "operator; s=2 retains its full fine trace.\n"
        "Matching error markers overlap. Slope guides are not fitted convergence claims.",
        fontsize=9,
    )
    save_figure(figure, output, "darcy-convergence")


def plot_fields(row: dict[str, Any], directory: Path, output: Path) -> None:
    """Render exact, independent, pyMHM and difference fields with actual macrofaces."""
    data = load_archive(row, directory)
    mesh = macro_mesh(data)
    points, cells, pressure, flux = broken_fields(data)
    _, _, reference_pressure, reference_flux = broken_fields(data, "reference_")
    exact_pressure, exact_flux = darcy_fields(row["case"])
    expected_pressure = exact_pressure(points)
    if row["case"].startswith("layer"):
        centers = data["points"][data["cells"]].mean(axis=1)
        expected_flux = np.repeat(exact_flux(centers), 3, axis=0)
    else:
        expected_flux = exact_flux(points)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    figure, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)
    titles = ("Analytical", "NeoPZ", "pyMHM", "pyMHM − NeoPZ")
    fields = (
        (expected_pressure, reference_pressure, pressure, "Pressure p"),
        (expected_flux[:, 0], reference_flux[:, 0], flux[:, 0], "Flux qₓ"),
        (expected_flux[:, 1], reference_flux[:, 1], flux[:, 1], "Flux qᵧ"),
    )
    for index, (expected, reference, ours, label) in enumerate(fields):
        limits = color_limits([expected, reference, ours])
        cmap = "RdBu_r" if limits[0] < 0 < limits[1] else "viridis"
        for column, values in enumerate((expected, reference, ours, ours - reference)):
            if column == 3:
                magnitude = max(float(np.max(np.abs(values))), 1e-16)
                minimum, maximum = -magnitude, magnitude
                palette = "RdBu_r"
            else:
                minimum, maximum = limits
                palette = cmap
            artist = axes[index, column].tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap=palette,
                vmin=minimum,
                vmax=maximum,
                rasterized=True,
            )
            axis = axes[index, column]
            axis.set_title(f"{titles[column]}\n{label}", fontsize=10)
            axis.set_aspect("equal")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            draw_macro_mesh(axis, mesh)
            if column in (2, 3):
                attached = axes[index, :3].tolist() if column == 2 else axis
                bar = figure.colorbar(artist, ax=attached, shrink=0.76, pad=0.025)
                bar.formatter.set_powerlimits((-3, 3))
                bar.update_ticks()
    resolution, segments = row["macro_resolution"], row["trace_segments"]
    contrast = " · contrast 1000" if row["case"] == "layer1000" else ""
    figure.suptitle(
        f"{row['case']} Darcy{contrast}: n={resolution}, s={segments}, "
        f"{len(mesh.cells)} macrotriangles / {len(data['cells'])} fine triangles"
    )
    rendering_note = (
        "Analytical cosine fields are interpolated for display only; error norms use quadrature."
        if row["case"] == "cos"
        else "Analytical layers use the exact value on each side of the fitted interface."
    )
    figure.supxlabel(
        "All panels show the actual macro mesh. Pressure remains cellwise P0; RT0 flux is "
        "rendered affinely without smoothing across cells.\n"
        f"{rendering_note} The first three columns share scales; differences use separate scales.",
        fontsize=9,
    )
    save_figure(figure, output, f"{row['case']}-fields-n{resolution}-s{segments}")


def sample_profile(
    data: dict[str, np.ndarray], locations: np.ndarray, prefix: str = ""
) -> np.ndarray:
    """Evaluate the archived P0/RT0 field inside each actual fine triangle."""
    vertices = data["points"][data["cells"]]
    _, _, pressure, flux = broken_fields(data, prefix)
    pressure = pressure.reshape(-1, 3)
    flux = flux.reshape(-1, 3, 2)
    values = np.full((len(locations), 3), np.nan)
    for index, triangle in enumerate(vertices):
        matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
        local = np.linalg.solve(matrix, (locations - triangle[0]).T).T
        barycentric = np.column_stack((1 - local.sum(axis=1), local))
        inside = np.all(barycentric >= -1e-12, axis=1)
        values[inside, 0] = pressure[index, 0]
        values[inside, 1:] = barycentric[inside] @ flux[index]
    if not np.isfinite(values).all():
        raise ValueError("The requested profile leaves the archived fine mesh")
    return values


def plot_layer_profiles(
    rows: list[dict[str, Any]], directory: Path, output: Path, resolution: int
) -> None:
    """Show continuous normal flux and discontinuous tangential flux in both layers."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    x = np.linspace(0, 1, 502)
    locations = np.column_stack((x, np.full_like(x, 0.37)))
    for index, case in enumerate(("layer", "layer1000")):
        row = select_row(rows, case, resolution, 1)
        data = load_archive(row, directory)
        mesh = macro_mesh(data)
        ours = sample_profile(data, locations)
        reference = sample_profile(data, locations, "reference_")
        pressure, flux = darcy_fields(case)
        expected = np.column_stack((pressure(locations), flux(locations)))
        intersections = macro_profile_breaks(mesh, locations[0], locations[-1])[1:-1]
        contrast = 10 if case == "layer" else 1000
        for column, label in enumerate(("Pressure p", "Normal flux qₓ", "Tangential flux qᵧ")):
            axis = axes[index, column]
            axis.plot(x, expected[:, column], color="#20272d", linewidth=1.8, label="Analytical")
            axis.plot(
                x,
                reference[:, column],
                color="#237c91",
                linewidth=1.1,
                marker="o",
                markerfacecolor="none",
                markevery=25,
                markersize=4,
                label="NeoPZ",
            )
            axis.plot(
                x,
                ours[:, column],
                "--",
                color="#cf692f",
                linewidth=1.1,
                marker="x",
                markevery=25,
                markersize=4,
                label="pyMHM",
            )
            mark_macro_interfaces(axis, intersections, label=True)
            axis.set_title(f"K contrast {contrast} · {label}")
            axis.set_xlabel("x along y = 0.37")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.2)
            if column == 1:
                axis.set_ylim(-1.1, -0.9)
            axis.legend(fontsize=8)
    figure.suptitle(f"Fitted layers: n={resolution}, one constant flux trace per macroedge")
    figure.supxlabel(
        "Dotted lines mark intersections with the actual macro mesh. The material interface "
        "is x=0.5.\n"
        "Normal flux is continuous; tangential flux jumps from −2 to −2K. P0 pressure profiles "
        "retain the actual cellwise values.",
        fontsize=9,
    )
    save_figure(figure, output, "layer-profiles")


def main() -> None:
    """Read recorded results and export comparisons without rerunning either solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--field-resolution", type=int, default=8)
    parser.add_argument("--layer-resolution", type=int, default=4)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = report["rows"]
    directory = args.report.parent
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    with threadpool_limits(limits=1):
        plot_convergence(rows, directory, args.output)
        plot_fields(select_row(rows, "cos", args.field_resolution, 1), directory, args.output)
        for case in ("layer", "layer1000"):
            plot_fields(select_row(rows, case, args.layer_resolution, 1), directory, args.output)
        plot_layer_profiles(rows, directory, args.output, args.layer_resolution)
    print(f"Saved five PNG/SVG figure pairs to {args.output}")


if __name__ == "__main__":
    main()
