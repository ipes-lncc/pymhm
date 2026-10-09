"""Render complex acoustic fields, directional accuracy and PML verification."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

from examples.helmholtz_campaign import AcousticWave
from examples.plot_style import set_refinement_ticks
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.io.workspace import case_workspace, local_resource, read_resource_text, resource_glob

ROOT = case_workspace()
RESULTS = ROOT / "examples/results/helmholtz"
OUTPUT = ROOT / "docs/figures/helmholtz"


def save(figure: Any, output: Path, name: str) -> None:
    """Export publication-sized PNG and vector text with rasterized dense fields."""
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.12)
    figure.savefig(output / f"{name}.svg", bbox_inches="tight", pad_inches=0.12)
    plt.close(figure)


def convergence(record: dict[str, Any], output: Path) -> None:
    """Keep pressure, gradient and weighted norms distinct for each analytical wave."""
    figure = plt.figure(figsize=(15, 12), layout="constrained")
    grid = figure.add_gridspec(4, 3, height_ratios=(1, 0.27, 1, 0.27))
    axes = [[figure.add_subplot(grid[2 * i, j]) for j in range(3)] for i in range(2)]
    keys = ("pressure_relative_error", "gradient_relative_error", "energy_relative_error")
    for row, kind in zip(axes, ("plane", "hankel"), strict=True):
        for ell in (2, 3):
            for basis in ("polynomial", "oscillatory"):
                rows = [
                    r
                    for r in record["rows"]
                    if r["study"] == "convergence"
                    and r["wave"] == kind
                    and r["trace_degree"] == ell
                    and r["trace_basis"] == basis
                ]
                h = np.array([r["macro_edge_length"] for r in rows])
                for axis, key in zip(row, keys, strict=True):
                    error = np.array([r[key] for r in rows])
                    rate = np.log(error[-2] / error[-1]) / np.log(h[-2] / h[-1])
                    axis.loglog(
                        h,
                        error,
                        "o-" if basis == "polynomial" else "s--",
                        label=rf"$\ell={ell}$ {basis}; rate {rate:.2f}",
                    )
                    set_refinement_ticks(axis, h, [f"1/{int(round(1 / v))}" for v in h])
                    axis.set(xlabel="Macro edge length", ylabel="Relative physical norm")
                    axis.grid(True, alpha=0.25)
        for axis, title in zip(row, ("pressure", "gradient", "weighted norm"), strict=True):
            axis.set_title(f"{kind.capitalize()} wave: {title}")
    for i, row in enumerate(axes):
        for j, axis in enumerate(row):
            legend_axis = figure.add_subplot(grid[2 * i + 1, j])
            legend_axis.axis("off")
            legend_axis.legend(*axis.get_legend_handles_labels(), loc="center", fontsize=11)
    figure.suptitle("Local Qℓ₊₂ on 2×2 fine squares per macrocell; outgoing impedance")
    save(figure, output, "convergence")


def directions(record: dict[str, Any], output: Path) -> None:
    """Compare bases at the same angular samples and discrete dimensions."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for basis in ("polynomial", "oscillatory"):
        rows = [
            r for r in record["rows"] if r["study"] == "direction" and r["trace_basis"] == basis
        ]
        angles = np.array([r["angle"] for r in rows]) * 180 / np.pi
        for axis, key in zip(
            axes, ("pressure_relative_error", "gradient_relative_error"), strict=True
        ):
            axis.semilogy(angles, [r[key] for r in rows], "o-", label=basis)
            axis.set(xlabel="Propagation angle [degrees]", ylabel="Relative physical norm")
            axis.set_xticks([0, 15, 30, 45, 60, 75, 90])
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, fontsize=11)
    axes[0].set_title("Complex pressure")
    axes[1].set_title("Broken complex gradient")
    figure.suptitle(r"Plane wave: $\omega=20\pi$, macro edge $1/11$, $\ell=2$, local Q4")
    save(figure, output, "directions")


def panel(
    axis: Any,
    data: dict[str, np.ndarray],
    values: np.ndarray,
    title: str,
    limits: tuple[float, float],
    signed: bool,
) -> None:
    """Preserve all incident-side samples and overlay the actual macro mesh."""
    points = data["points"]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], data["cells"])
    artist = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap="RdBu_r" if signed else "viridis",
        vmin=limits[0],
        vmax=limits[1],
        rasterized=True,
    )
    segments = data["macro_points"][data["macro_faces"]]
    axis.add_collection(LineCollection(segments, colors="white", linewidths=0.7, alpha=0.65))
    axis.add_collection(LineCollection(segments, colors="black", linewidths=0.3, alpha=0.6))
    axis.set(xlabel="x", ylabel="y", title=title, aspect="equal", xlim=(0, 1), ylim=(0, 1))
    axis.set_xticks([0, 0.5, 1])
    axis.set_yticks([0, 0.5, 1])
    bar = axis.figure.colorbar(artist, ax=axis, pad=0.025, fraction=0.045)
    bar.ax.tick_params(labelsize=12)


def fields(path: Path, output: Path) -> None:
    """Display exact, numerical and error maps for the complex pressure components."""
    with np.load(local_resource(path)) as archive:
        data = {key: archive[key] for key in archive.files}
    wave = AcousticWave(float(data["omega"]), float(data["angle"]), str(data["kind"]))
    exact, computed = wave.pressure(data["points"]), data["values"]
    figure, axes = plt.subplots(3, 3, figsize=(14, 12.5), layout="constrained")
    for row, transform, label, signed in zip(
        axes,
        (np.real, np.imag, np.abs),
        ("Real part", "Imaginary part", "Magnitude"),
        (True, True, False),
        strict=True,
    ):
        truth, field = transform(exact), transform(computed)
        maximum = float(max(abs(truth).max(), abs(field).max()))
        limits = (-maximum, maximum) if signed else (0.0, maximum)
        panel(row[0], data, truth, f"{label}: analytical", limits, signed)
        panel(row[1], data, field, f"{label}: MHM", limits, signed)
        error = abs(field - truth)
        panel(row[2], data, error, f"{label}: absolute error", (0, float(error.max())), False)
    figure.suptitle(f"{wave.kind.capitalize()} wave — {path.stem.replace('-fields', '')}")
    save(figure, output, path.stem)


def q4_profile(data: dict[str, np.ndarray], y: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replay a horizontal line from the archive's complete 5-by-5 Q4 nodal samples.

    Fine-cell endpoints remain independent. The line must avoid horizontal fine
    faces, whose two incident values otherwise need an explicit side convention.
    """
    points = data["points"].reshape(-1, 25, 2)
    values = data["values"].reshape(-1, 25)
    lower, upper = points[:, 0], points[:, -1]
    if np.any((lower[:, 1] == y) | (upper[:, 1] == y)):
        raise ValueError("profile on a horizontal fine face requires an explicit side")
    selected = (lower[:, 1] < y) & (y < upper[:, 1])
    profiles = []
    for cell in np.flatnonzero(selected):
        parameter = np.linspace(0, 1, 33)
        reference = np.column_stack(
            (
                parameter,
                np.full(len(parameter), (y - lower[cell, 1]) / (upper[cell, 1] - lower[cell, 1])),
            )
        )
        basis = qk_basis(4, reference)[0]
        x = lower[cell, 0] + parameter * (upper[cell, 0] - lower[cell, 0])
        profiles.append((x, basis @ values[cell]))
    return profiles


def pml(record: dict[str, Any], output: Path, archive: Path) -> None:
    """Report PML approximation error without identifying it with truncation error."""
    rows = [r for r in record["rows"] if r["study"] == "pml" and r["resolution"] >= 16]
    h = np.array([r["macro_edge_length"] for r in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4), layout="constrained")
    for key, label in (
        ("pressure_relative_error", "pressure"),
        ("gradient_relative_error", "gradient"),
    ):
        axes[0].loglog(h, [r[key] for r in rows], "o-", label=label)
    set_refinement_ticks(axes[0], h, [f"1/{r['resolution']}" for r in rows])
    axes[0].set(
        xlabel="Macro edge length", ylabel="Relative physical norm", title="PML approximation"
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, fontsize=11)
    axes[0].grid(True, alpha=0.25)
    x = np.linspace(0, 1, 501)
    wave = AcousticWave(10 * np.pi, kind="pml")
    y = 0.473
    values = wave.pressure(np.column_stack((x, x * 0 + y)))
    axes[1].plot(x, values.real, color="black", label="Exact real part")
    axes[1].plot(x, abs(values), color="#009E73", label="Exact magnitude")
    with np.load(local_resource(archive)) as data:
        profiles = q4_profile(dict(data), y)
        faces = data["macro_points"][data["macro_faces"]]
    for index, (coordinate, value) in enumerate(profiles):
        axes[1].plot(
            coordinate,
            value.real,
            "--",
            color="#0072B2",
            label="MHM real part" if index == 0 else None,
        )
        axes[1].plot(
            coordinate,
            abs(value),
            "--",
            color="#D55E00",
            label="MHM magnitude" if index == 0 else None,
        )
    axes[1].axvline(0.6, color="black", linestyle=":", label="PML starts")
    vertical = faces[np.isclose(faces[:, 0, 0], faces[:, 1, 0], atol=0, rtol=0)]
    for position in np.unique(vertical[:, 0, 0]):
        axes[1].axvline(position, color="0.5", alpha=0.22, linewidth=0.6, zorder=0)
    axes[1].set(
        xlabel="x",
        ylabel="Pressure",
        title=f"Attenuation profile, y={y}\nArchived n=24 macro intersections",
    )
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, fontsize=11)
    axes[1].grid(True, alpha=0.25)
    figure.suptitle("PML verification: local Q4, constant face traces, exact Dirichlet data")
    save(figure, output, "pml-convergence")


def main() -> None:
    """Render the immutable acquisition records without assembling a wave operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = json.loads(read_resource_text(args.input / "comparison.json"))
    if record.get("source_changed_during_run") is not False:
        raise ValueError("require a complete acquisition with unchanged recorded sources")
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.titlesize": 17})
    convergence(record, args.output)
    directions(record, args.output)
    pml(record, args.output, args.input / "pml-fields.npz")
    for path in sorted(resource_glob(args.input, "*-fields.npz")):
        fields(path, args.output)
    shutil.copy2(args.input / "comparison.json", args.output / "comparison.json")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_helmholtz").main()
