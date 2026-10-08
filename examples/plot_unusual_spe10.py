"""Replay physical SPE10 reaction layers, separate resolution controls and CG2 norms."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from examples.plot_style import set_refinement_ticks
from examples.solve_unusual_spe10 import OUTPUT, ROOT, UnusualSPE10Field
from examples.solve_unusual_spe10_reference import CG2Field
from examples.spe10_plot_records import CASES, completed_cases, digest, validate_pair

FIGURES = ROOT / "docs/figures/unusual-spe10"
REFERENCE = "classical-cg2-graded-xy-1440x498"


def read(path: Path) -> dict:
    """Require the physical field digest used by the acquisition or norm record."""
    record = json.loads(path.read_text())
    for key in ("archive", "reference"):
        if key in record:
            actual = digest(path.parent / record[key])
            if actual != record[f"{key}_sha256"]:
                raise ValueError(f"{key} checksum mismatch in {path.name}")
    return record


def save(figure: plt.Figure, output: Path, name: str) -> None:
    """Preserve readable vector axes alongside rasterized spatial fields."""
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def overlay(axis: plt.Axes, field: UnusualSPE10Field, height: float) -> None:
    """Highlight actual macro edges, including their intersections with the display window."""
    for face in field.macro.faces:
        points = field.macro.points[face]
        axis.plot(*points.T, color="black", lw=0.65, alpha=0.9)
    axis.set(xlim=(0, 1200), ylim=(0, height), xlabel="x (ft)", ylabel="y (ft)")


def profile(field: UnusualSPE10Field, points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Keep both incident macro values and insert every crossed fine edge explicitly."""
    result = []
    x = float(points[0, 0])
    for mesh, pressure in zip(field.meshes, field.pressure, strict=True):
        edges = mesh.points[mesh.faces]
        selected = (
            (edges[:, :, 0].min(axis=1) <= x)
            & (edges[:, :, 0].max(axis=1) >= x)
            & (edges[:, 0, 0] != edges[:, 1, 0])
        )
        edges = edges[selected]
        if not len(edges):
            continue
        fraction = (x - edges[:, 0, 0]) / (edges[:, 1, 0] - edges[:, 0, 0])
        crossing = edges[:, 0, 1] + fraction * (edges[:, 1, 1] - edges[:, 0, 1])
        values = (1 - fraction) * pressure[mesh.faces[selected, 0]] + fraction * pressure[
            mesh.faces[selected, 1]
        ]
        crossing, indexes = np.unique(crossing, return_index=True)
        values = values[indexes]
        a, b = crossing.min(), crossing.max()
        if a == b:
            continue
        coordinates = np.unique(
            np.r_[crossing, points[(points[:, 1] >= a) & (points[:, 1] <= b), 1]]
        )
        # A P1 restriction to the line is affine between consecutive crossed
        # fine edges. This preserves the physical piecewise polynomial exactly.
        result.append((coordinates, np.interp(coordinates, crossing, values)))
    return result


def spatial(
    field: UnusualSPE10Field, height: float
) -> tuple[mtri.Triangulation, np.ndarray, np.ndarray]:
    """Retain P1 nodal pressure and constant one-sided cell flux without smoothing."""
    points, cells, pressure, flux = [], [], [], []
    offset = 0
    for macro, mesh in enumerate(field.meshes):
        selected = np.flatnonzero(mesh.points[mesh.cells, 1].min(axis=1) <= height)
        if not len(selected):
            continue
        nodes, inverse = np.unique(mesh.cells[selected], return_inverse=True)
        points.append(mesh.points[nodes])
        cells.append(inverse.reshape(-1, 3) + offset)
        pressure.append(field.pressure[macro][nodes])
        centers = mesh.points[mesh.cells[selected]].mean(axis=1)
        flux.append(field.evaluate_local(macro, centers, selected)[2][:, 1])
        offset += len(nodes)
    return (
        mtri.Triangulation(*np.concatenate(points).T, triangles=np.concatenate(cells)),
        np.concatenate(pressure),
        np.concatenate(flux),
    )


def paired_profiles(
    fields: list[UnusualSPE10Field],
    reference: CG2Field,
    comparisons: list[dict],
    crossings: np.ndarray,
    output: Path,
) -> None:
    """Separate local and skeletal refinement on the same resolved reaction-layer geometry."""
    points = np.column_stack((np.full(1201, 33.0), np.r_[0.0, np.geomspace(1e-4, 2200, 1200)]))
    truth = reference.evaluate(points)[0]
    selected = (len(CASES) - 1, len(CASES), len(CASES) + 1)
    labels = ("r32 / s32", "r64 / s32", "r64 / s64")
    fig = plt.figure(figsize=(13.2, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(3.7, 0.6))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    for axis, height in zip(axes, (2200, 20, 1), strict=True):
        axis.plot(points[:, 1], truth, color="black", lw=1.8, label="DOLFINx CG2 reference")
        for crossing in crossings:
            if 0 < crossing < height:
                axis.axvline(crossing, color="0.7", ls=":", lw=0.7)
        axis.set(xlim=(0, height), xlabel="y (ft), x = 33 ft", ylabel="Pressure")
        axis.grid(alpha=0.15)
    for color, (index, label) in enumerate(zip(selected, labels, strict=True)):
        for part, (y, pressure) in enumerate(profile(fields[index], points)):
            for axis in axes:
                axis.plot(
                    y,
                    pressure,
                    color=f"C{color}",
                    lw=1.2,
                    ls=("-", "--", "-.")[color],
                    label=label if part == 0 else None,
                )
    for axis, title in zip(
        axes, ("Complete profile", "First material row", "Reaction-layer scale"), strict=True
    ):
        axis.set_title(title)
    handles, names = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, names, loc="center", ncols=4, fontsize=9, frameon=False)
    save(fig, output, "paired-profiles")

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), layout="constrained")
    for axis, key, title in zip(
        axes,
        ("pressure", "flux", "energy"),
        ("Pressure L²", "Raw flux L²", "Diffusion–reaction energy"),
        strict=True,
    ):
        values = [100 * comparisons[index]["rows"][-1][key + "_relative"] for index in selected]
        axis.plot(range(3), values, "o-")
        set_refinement_ticks(axis, range(3), labels)
        axis.set(title=title, ylabel="Difference from CG2 (%)", yscale="log")
        axis.grid(alpha=0.2)
    save(fig, output, "paired-resolution")


def plot(data: Path, output: Path) -> None:
    """Render current, checksum-verified records and their declared numerical baseline."""
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    reference_record = read(data / f"{REFERENCE}.json")
    reference = CG2Field.load(data / reference_record["archive"])
    rows, comparisons, fields = [], [], []
    cases = completed_cases(data)
    for stem, _ in cases:
        path = data / f"mhm-unusual-{stem}-q5.json"
        rows.append(read(path))
        comparisons.append(read(path.with_stem(path.stem + "-comparison")))
        fields.append(UnusualSPE10Field(path.with_suffix(".npz")))
    if any(
        len(record["rows"]) != 2
        or {value["order"] for value in record["rows"]} != {3, 4}
        or record["reference_sha256"] != reference_record["archive_sha256"]
        or record["archive"] != row["archive"]
        or record["archive_sha256"] != row["archive_sha256"]
        or record.get("source_changed_during_run") is not False
        for row, record in zip(rows, comparisons, strict=True)
    ):
        raise ValueError("all controls require two norm rules and the same CG2 reference")
    if len(cases) > len(CASES):
        validate_pair(rows, fields)

    points = np.column_stack((np.full(1201, 33.0), np.r_[0.0, np.geomspace(1e-4, 2200, 1200)]))
    truth = reference.evaluate(points)[0]
    with np.load(data / rows[0]["archive"]) as arrays:
        crossings = np.unique(arrays["profile_points"][:, [0, -1], 1])
    fig = plt.figure(figsize=(13.2, 5.3), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(3.6, 1.1))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    for axis, height in zip(axes, (2200, 20, 1), strict=True):
        axis.plot(points[:, 1], truth, color="black", lw=2, label="DOLFINx CG2 reference")
        for crossing in crossings:
            if 0 < crossing < height:
                axis.axvline(crossing, color="0.7", ls=":", lw=0.7)
        axis.set(xlim=(0, height), xlabel="y (ft), x = 33 ft", ylabel="Pressure")
        axis.grid(alpha=0.15)
    for index, (field, (_, label)) in enumerate(zip(fields[: len(CASES)], CASES, strict=True)):
        for part, (y, p) in enumerate(profile(field, points)):
            for axis in axes:
                axis.plot(y, p, color=f"C{index}", lw=1.0, label=label if part == 0 else None)
    axes[0].set_title("Complete profile; independent macro sides")
    axes[1].set_title("First material row")
    axes[2].set_title("Reaction-layer scale")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(
        handles,
        labels,
        loc="center",
        ncols=3,
        fontsize=10,
        frameon=False,
        columnspacing=2,
        handlelength=2.5,
    )
    save(fig, output, "profiles")

    fig = plt.figure(figsize=(13.2, 5.0), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(4.1, 0.5))
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    legend_axis = fig.add_subplot(grid[1, :])
    legend_axis.set_axis_off()
    groups = ((0, 1, 2), (2, 3, 4), (5, 6, 7, 8))
    for axis, indexes, title in zip(
        axes,
        groups,
        ("Local refinement, fixed s=8", "Trace refinement, fixed r=32", "Physical-layer controls"),
        strict=True,
    ):
        for key, label in (
            ("pressure", "Pressure L²"),
            ("flux", "Raw flux L²"),
            ("energy", "Diffusion–reaction energy"),
        ):
            axis.plot(
                range(len(indexes)),
                [100 * comparisons[i]["rows"][-1][f"{key}_relative"] for i in indexes],
                "o-",
                label=label,
            )
        labels = [
            CASES[i][1].replace(" / r", "\nr").replace(" + pixel trace", "\n+ pixel trace")
            for i in indexes
        ]
        set_refinement_ticks(axis, range(len(indexes)), labels)
        axis.set(title=title, ylabel="Difference from CG2 (%)", yscale="log")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, fontsize=9, loc="center", ncols=3, frameon=False)
    save(fig, output, "resolution")
    if len(cases) > len(CASES):
        paired_profiles(fields, reference, comparisons, crossings, output)
    coarse_field, finest_field = fields[0], fields[-1]
    del fields, field

    # The CG2 display samples each native quadratic cell on its nodal half-grid.
    # Norms use exact overlays; this rendering does not define a comparison norm.
    height = 20.0
    xa, ya = reference.x_axis, reference.y_axis
    x = np.sort(np.r_[xa, (xa[:-1] + xa[1:]) / 2])
    y = np.sort(np.r_[ya, (ya[:-1] + ya[1:]) / 2])
    y = y[y <= height]
    xx, yy = np.meshgrid(x, y)
    reference_points = np.column_stack((xx.ravel(), yy.ravel()))
    lower = (np.arange(len(y) - 1)[:, None] * len(x) + np.arange(len(x) - 1)).ravel()
    reference_cells = np.concatenate(
        (
            np.column_stack((lower, lower + 1, lower + 1 + len(x))),
            np.column_stack((lower, lower + 1 + len(x), lower + len(x))),
        )
    )
    reference_tri = mtri.Triangulation(*reference_points.T, triangles=reference_cells)
    rp = reference.evaluate(reference_points)[0]
    rq = reference.evaluate(reference_points[reference_cells].mean(axis=1))[2]
    mhm = [spatial(field, height) for field in (coarse_field, finest_field)]
    p_limits = (
        min(rp.min(), *(item[1].min() for item in mhm)),
        max(rp.max(), *(item[1].max() for item in mhm)),
    )
    q_bound = max(abs(rq[:, 1]).max(), *(abs(item[2]).max() for item in mhm))
    # Both windows retain the same samples, one-sided fluxes and linear color limits.
    for display_height, name in ((20.0, "boundary-layer"), (1.0, "reaction-layer")):
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 5.8), layout="constrained")
        image = axes[0, 0].tripcolor(
            reference_tri,
            rp,
            cmap="viridis",
            vmin=p_limits[0],
            vmax=p_limits[1],
            shading="gouraud",
            rasterized=True,
        )
        fig.colorbar(image, ax=axes[0, 0], label="Pressure", shrink=0.8)
        image = axes[1, 0].tripcolor(
            reference_tri,
            facecolors=rq[:, 1],
            cmap="RdBu_r",
            vmin=-q_bound,
            vmax=q_bound,
            rasterized=True,
        )
        fig.colorbar(image, ax=axes[1, 0], label="Flux qᵧ", shrink=0.8)
        for col, (tri, pressure, flux) in enumerate(mhm, 1):
            image = axes[0, col].tripcolor(
                tri,
                pressure,
                shading="gouraud",
                cmap="viridis",
                vmin=p_limits[0],
                vmax=p_limits[1],
                rasterized=True,
            )
            fig.colorbar(image, ax=axes[0, col], label="Pressure", shrink=0.8)
            image = axes[1, col].tripcolor(
                tri, facecolors=flux, cmap="RdBu_r", vmin=-q_bound, vmax=q_bound, rasterized=True
            )
            fig.colorbar(image, ax=axes[1, col], label="Flux qᵧ", shrink=0.8)
        for row in axes:
            for axis, title in zip(
                row,
                (
                    "DOLFINx CG2 reference",
                    "P1/P0 nominal r8/s8",
                    "P1/P0 c=.25 + pixel trace\n"
                    f"r{rows[-1]['nominal_local_refinement']} / s{rows[-1]['trace_segments']}",
                ),
                strict=True,
            ):
                overlay(axis, coarse_field, display_height)
                axis.set_title(title)
        save(fig, output, name)

    for path in data.glob("*.json"):
        shutil.copy2(path, output / path.name)


def main() -> None:
    """Replay all completed controls without changing fields or running a PDE solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path, default=FIGURES)
    args = parser.parse_args()
    plot(args.data, args.output)


if __name__ == "__main__":
    main()
