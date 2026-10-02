"""Replay physical local/trace resolution controls without changing the SPE10 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.plot_spe10_adaptive import display_fields, panel, reference_display, save
from examples.plot_style import set_refinement_ticks
from examples.spe10_adaptive import DATA, StructuredRT
from examples.spe10_adaptive_norms import BrokenP2

FIGURES = Path(__file__).resolve().parents[1] / "docs/figures/spe10-adaptive/resolution"


def material_resolution(archive: Path) -> dict[str, float | int]:
    """Count geometrically pixel-crossing triangles, accounting for coordinate roundoff."""
    with np.load(archive) as stored:
        values = {key: stored[key] for key in ("local_points", "local_cells")}
        if "point_offsets" in stored:
            values.update({key: stored[key] for key in ("point_offsets", "cell_offsets")})
        if "point_offsets" in values:
            vertices = np.concatenate(
                [
                    values["local_points"][
                        values["point_offsets"][i] : values["point_offsets"][i + 1]
                    ][
                        values["local_cells"][
                            values["cell_offsets"][i] : values["cell_offsets"][i + 1]
                        ]
                    ]
                    for i in range(len(values["point_offsets"]) - 1)
                ]
            )
        else:
            vertices = np.concatenate(
                [
                    points[cells]
                    for points, cells in zip(
                        values["local_points"], values["local_cells"], strict=True
                    )
                ]
            )
    uncertainty = 64 * np.finfo(float).eps * np.max(abs(vertices), axis=(1, 2))
    lower = np.floor((vertices.min(axis=1) + uncertainty[:, None]) / [20.0, 10.0])
    upper = np.ceil((vertices.max(axis=1) - uncertainty[:, None]) / [20.0, 10.0]) - 1
    crossing = np.any(lower < upper, axis=1)
    first, second = (vertices[:, 1:] - vertices[:, :1]).transpose(1, 0, 2)
    areas = abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2
    minimum_angle = min(
        np.min(
            np.arctan2(
                2 * areas,
                np.sum(
                    (vertices[:, (corner + 1) % 3] - vertices[:, corner])
                    * (vertices[:, (corner + 2) % 3] - vertices[:, corner]),
                    axis=1,
                ),
            )
        )
        for corner in range(3)
    )
    return {
        "crossing_triangles": int(crossing.sum()),
        "crossing_area_fraction": float(areas[crossing].sum() / areas.sum()),
        "minimum_fine_angle_degrees": float(np.rad2deg(minimum_angle)),
    }


def read_controls(directory: Path) -> list[dict[str, Any]]:
    """Verify every physical archive before collecting its integrated comparisons."""
    published = json.loads((DATA / "published/adaptive.json").read_text())[-1]
    baseline = {
        **published,
        "material_fitted": False,
        "trace_segments": 1,
        "archive_directory": DATA / "published",
        "norm_file": directory / "baseline-r2-s1-norms.json",
    }
    controls = [baseline]
    for fitted, refinement, segments in (
        (False, 4, 1),
        (False, 8, 1),
        (False, 8, 2),
        (False, 8, 4),
        (False, 4, 4),
        (True, 4, 1),
        (True, 4, 4),
    ):
        stem = f"mhm-{'fitted-' if fitted else ''}r{refinement}-s{segments}"
        controls.append(
            {
                **json.loads((directory / f"{stem}.json").read_text()),
                "archive_directory": directory,
                "norm_file": directory / f"{stem}-norms.json",
            }
        )
    joint = directory / "mhm-fitted-r8-s8.json"
    joint_norm = directory / "mhm-fitted-r8-s8-norms.json"
    if (
        joint.exists()
        and joint_norm.exists()
        and len(json.loads(joint_norm.read_text())["rows"]) >= 2
    ):
        controls.append(
            {
                **json.loads(joint.read_text()),
                "archive_directory": directory,
                "norm_file": joint_norm,
            }
        )
    for row in controls:
        norm_record = json.loads(row["norm_file"].read_text())
        if len({entry["quadrature_order"] for entry in norm_record["rows"]}) < 2:
            raise ValueError("two distinct norm quadratures are required before publication")
        archive = row["archive_directory"] / row["archive"]
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != row["archive_sha256"] or digest != norm_record["archive_sha256"]:
            raise ValueError("physical archive checksum differs from its solve or norm record")
        reference = DATA / norm_record["reference"]
        if hashlib.sha256(reference.read_bytes()).hexdigest() != norm_record["reference_sha256"]:
            raise ValueError("classical reference checksum differs from the norm record")
        row["norms"] = norm_record["rows"][-1]
        row["reference"] = norm_record["reference"]
        row["reference_sha256"] = norm_record["reference_sha256"]
        row["material_resolution"] = material_resolution(archive)
        if row["material_fitted"] and row["material_resolution"]["crossing_triangles"]:
            raise ValueError("declared fitted archive contains material-crossing triangles")
        low, high = norm_record["rows"][0], norm_record["rows"][-1]
        row["quadrature_relative_change_linf"] = max(
            abs(low[key] - high[key]) / abs(high[key])
            for key in low
            if key not in ("quadrature_order", "seconds")
        )
    if len({row["reference"] for row in controls}) != 1:
        raise ValueError("all controls in one comparison must use the same reference")
    return controls


def component_map(
    rows: list[dict[str, Any]],
    indices: tuple[int, int],
    labels: tuple[str, str],
    reference_field: dict[str, np.ndarray],
    output: Path,
    filename: str,
) -> None:
    """Render two verified reconstructed fields and a common classical reference."""
    candidates = [
        BrokenP2(rows[index]["archive_directory"] / rows[index]["archive"]) for index in indices
    ]
    fields = [
        reference_field,
        *(
            display_fields(candidate, material_fitted=rows[index]["material_fitted"])
            for candidate, index in zip(candidates, indices, strict=True)
        ),
    ]
    names = ["Classical RT2/P2 reference", *(f"MHM P2: {label}" for label in labels)]
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 11.4), layout="constrained")
    for component in range(2):
        arrays = [fields[0]["values"][:, 1 + component]] + [
            field["values"][:, 3 + component] for field in fields[1:]
        ]
        limit = max(float(np.max(abs(value))) for value in arrays)
        for column, (field, value, name) in enumerate(zip(fields, arrays, names, strict=True)):
            panel(
                axes[component, column],
                field,
                value,
                candidates[0].macro,
                f"{name}\nRT2 {'reconstructed ' if column else ''}flux "
                f"q{'x' if component == 0 else 'y'}",
                (-limit, limit),
                signed=True,
                asinh=True,
            )
    save(figure, filename, output)


def plot(
    directory: Path, output: Path, *, records_only: bool = False, joint_only: bool = False
) -> None:
    """Show independent local/trace changes and unsmoothed physical vector components."""
    rows = read_controls(directory)
    output.mkdir(parents=True, exist_ok=True)
    if records_only:
        export_records(rows, directory, output)
        return
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    if joint_only:
        if len(rows) != 9:
            raise ValueError("joint enrichment requires its completed two-order norm record")
        component_map(
            rows,
            (7, 8),
            ("fitted r=4, four P0 traces", "fitted r=8, eight P0 traces"),
            reference_display(StructuredRT.load(DATA / rows[0]["reference"])),
            output,
            "joint-flux-components",
        )
        export_records(rows, directory, output)
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    quantities = (
        ("pressure_relative_difference", "Pressure", "o"),
        ("raw_flux_relative_difference", "Raw flux", "s"),
        ("reconstructed_flux_relative_difference", "RT2 reconstructed flux", "^"),
    )
    for axis, selected, key, title in (
        (axes[0], rows[:3], "local_refinement", "Local refinement; one P0 trace per macroface"),
        (axes[1], rows[2:5], "trace_segments", "Trace enrichment; fixed local refinement r=8"),
    ):
        values = [row[key] for row in selected]
        for quantity, label, marker in quantities:
            axis.plot(
                values,
                [100 * row["norms"][quantity] for row in selected],
                marker=marker,
                label=label,
            )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Local subdivisions r" if key == "local_refinement" else "P0 segments per face",
            ylabel="Relative physical L2 difference (%)",
            title=title,
        )
        set_refinement_ticks(axis, values)
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=9)
    save(figure, "independent-resolution", output)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for axis, energy in zip(axes, (False, True), strict=True):
        suffix = "energy_relative_difference" if energy else "relative_difference"
        for fitted, selected, style in (
            (False, (rows[1], rows[5]), "--"),
            (True, (rows[6], rows[7]), "-"),
        ):
            for name, label, marker in (
                ("raw_flux", "Raw", "s"),
                ("reconstructed_flux", "RT2 reconstruction", "^"),
            ):
                axis.plot(
                    [1, 4],
                    [100 * row["norms"][f"{name}_{suffix}"] for row in selected],
                    style,
                    marker=marker,
                    label=f"{label}; {'fitted' if fitted else 'unfitted'} local mesh",
                )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="P0 segments per macroface",
            ylabel="Relative difference (%)",
            title="Inverse-permeability norm" if energy else "Physical L2 norm",
        )
        set_refinement_ticks(axis, [1, 4])
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8.5)
    save(figure, "material-fitting", output)

    reference = StructuredRT.load(DATA / rows[0]["reference"])
    reference_field = reference_display(reference)
    for indices, labels, filename in (
        ((0, 4), ("r=2, one P0 trace", "r=8, four P0 traces"), "flux-components"),
        (
            (5, 7),
            ("unfitted r=4, four P0 traces", "fitted r=4, four P0 traces"),
            "material-flux-components",
        ),
    ):
        component_map(rows, indices, labels, reference_field, output, filename)
    if len(rows) == 9:
        component_map(
            rows,
            (7, 8),
            ("fitted r=4, four P0 traces", "fitted r=8, eight P0 traces"),
            reference_field,
            output,
            "joint-flux-components",
        )
    export_records(rows, directory, output)


def export_records(rows: list[dict[str, Any]], directory: Path, output: Path) -> None:
    """Archive all verified controls, including joint enrichment beyond the plotted pairs."""
    summary = [
        {key: value for key, value in row.items() if key not in ("archive_directory", "norm_file")}
        for row in rows
    ]
    (directory / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    for record in directory.glob("*.json"):
        shutil.copy2(record, output / record.name)


def main() -> None:
    """Render validated numerical records, preserving every physical field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA / "resolution")
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--records-only", action="store_true")
    parser.add_argument("--joint-only", action="store_true")
    options = parser.parse_args()
    if options.records_only and options.joint_only:
        parser.error("choose either records-only or joint-only")
    plot(
        options.data,
        options.output,
        records_only=options.records_only,
        joint_only=options.joint_only,
    )


if __name__ == "__main__":
    main()
