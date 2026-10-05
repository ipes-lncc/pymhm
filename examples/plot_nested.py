"""Render current recursive MHM results from executed archived sampling tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter

if __package__:
    from .nested_field_archive import array_digest, display_fields, read_archive
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.nested_field_archive import array_digest, display_fields, read_archive

ROOT = Path(__file__).resolve().parents[1]


def sample_edges(samples: np.ndarray) -> np.ndarray:
    """Bound sample pixels inside each independent leaf, preserving its one-sided edges.

    Adjacent leaves end at their common macroface without overlapping pixels.
    Display colors are the actual saved sample values; they are not averaged
    or interpolated across the macroface.
    """
    if (
        samples.ndim != 1
        or len(samples) < 2
        or not np.all(np.isfinite(samples))
        or np.any(np.diff(samples) <= 0)
    ):
        raise ValueError("strictly increasing physical sample coordinates required")
    return np.r_[samples[0], (samples[:-1] + samples[1:]) / 2, samples[-1]]


def save(figure: plt.Figure, folder: Path, name: str) -> dict[str, str]:
    """Write independently inspectable PNG and SVG representations of one current figure."""
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    for extension in ("png", "svg"):
        path = folder / f"{name}.{extension}"
        figure.savefig(path, dpi=190)
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(figure)
    return result


def convergence(record: dict[str, Any], output: Path) -> dict[str, str]:
    """Plot both boundary acquisitions and physical differences to their executed flat MHM."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    colors = {"pressure": "#225e96", "flux": "#ab4f13"}
    for boundary, marker, style, suffix in (
        ("sine_zero_dirichlet", "o", "-", "zero boundary"),
        ("affine_plus_sine", "s", "--", "affine boundary"),
    ):
        rows = sorted(
            (row for row in record["rows"] if row["boundary_case"] == boundary),
            key=lambda row: row["n"],
        )
        h = 1 / np.asarray([row["n"] for row in rows])
        for field, key, label in (
            ("pressure", "pressure_l2", "Pressure"),
            ("flux", "flux_l2", "Raw flux"),
        ):
            options = dict(
                marker=marker,
                linestyle=style,
                color=colors[field],
                markersize=6,
                markerfacecolor="white" if marker == "s" else colors[field],
            )
            axes[0].loglog(h, [row[key] for row in rows], label=f"{label}; {suffix}", **options)
            difference = (
                "pressure_relative_flat_difference"
                if field == "pressure"
                else "raw_flux_relative_flat_difference"
            )
            axes[1].loglog(
                h,
                [row["norms"]["quadrature_10"][difference] for row in rows],
                label=f"{label}; {suffix}",
                **options,
            )
    for axis in axes:
        axis.set(xlabel="Outer macro size H", xticks=[1, 1 / 2, 1 / 4, 1 / 8, 1 / 16])
        axis.set_xticklabels(["1", "1/2", "1/4", "1/8", "1/16"])
        axis.minorticks_off()
        axis.invert_xaxis()
        axis.grid(alpha=0.2, which="both")
        axis.legend(fontsize=8.5, loc="best")
    axes[0].set(title="Analytical L2 errors", ylabel="Absolute physical error")
    axes[1].set(title="Recursive / flat field agreement", ylabel="Relative physical L2 difference")
    return save(fig, output, "convergence")


def fields(arrays: dict[str, np.ndarray], output: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Compare analytical and replayed fields with separate color scales and both macro meshes."""
    points, pressure, flux = display_fields(arrays)
    x, y = np.pi * points[..., 0], np.pi * points[..., 1]
    exact_pressure = np.sin(x) * np.sin(y)
    exact_flux = -np.pi * np.stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)), axis=-1)
    names = ("Pressure", r"Raw flux $q_x$", r"Raw flux $q_y$")
    numerical = (pressure, flux[..., 0], flux[..., 1])
    analytical = (exact_pressure, exact_flux[..., 0], exact_flux[..., 1])
    fig = plt.figure(figsize=(12.5, 11.0), layout="constrained")
    grid = fig.add_gridspec(
        3, 6, width_ratios=[1, 0.045, 1, 0.045, 1, 0.045], wspace=0.13, hspace=0.12
    )
    for row, (name, exact, computed) in enumerate(zip(names, analytical, numerical, strict=True)):
        shared_limit = max(float(np.max(abs(exact))), float(np.max(abs(computed))))
        columns = (
            ("Analytical", exact),
            ("Recursive MHM", computed),
            ("Absolute error", abs(computed - exact)),
        )
        for column, (heading, values) in enumerate(columns):
            axis = fig.add_subplot(grid[row, 2 * column])
            color_axis = fig.add_subplot(grid[row, 2 * column + 1])
            error = column == 2
            limits = (
                (0, float(values.max()))
                if error
                else (0, shared_limit)
                if row == 0
                else (-shared_limit, shared_limit)
            )
            for coordinates, samples in zip(points, values, strict=True):
                physical = coordinates.reshape(17, 17, 2)
                image = axis.pcolormesh(
                    sample_edges(physical[0, :, 0]),
                    sample_edges(physical[:, 0, 1]),
                    samples.reshape(17, 17),
                    shading="flat",
                    rasterized=True,
                    cmap="magma" if error else "viridis" if row == 0 else "RdBu_r",
                    vmin=limits[0],
                    vmax=limits[1],
                )
            for edges, color, width in (
                (arrays["flat_face_endpoints"], ".50", 0.45),
                (arrays["outer_face_endpoints"], ".12", 1.0),
            ):
                axis.add_collection(LineCollection(edges, colors=color, linewidths=width))
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="x",
                ylabel="y",
                aspect="equal",
                title=f"{heading}\n{name}",
            )
            axis.tick_params(labelsize=9)
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            colorbar = fig.colorbar(image, cax=color_axis, format=formatter)
            colorbar.ax.tick_params(labelsize=8.5)
    hashes = save(fig, output, "fields")
    samples = {
        "points": array_digest(points),
        "pressure": array_digest(pressure),
        "raw_flux": array_digest(flux),
    }
    return hashes, samples


def render(record_path: Path, output: Path) -> dict[str, Any]:
    """Render current numerical and analytical data from the checked executed archive."""
    record = json.loads(record_path.read_text())
    if record["schema"] not in ("pymhm-recursive-study-v2", 2) or len(record["rows"]) != 10:
        raise ValueError("current complete recursive study record required")
    cases = {(row["n"], row["boundary_case"]) for row in record["rows"]}
    if cases != {
        (n, boundary)
        for n in (1, 2, 4, 8, 16)
        for boundary in ("sine_zero_dirichlet", "affine_plus_sine")
    }:
        raise ValueError("all five levels and both physical boundary problems required")
    if record["schema"] == "pymhm-recursive-study-v2":
        selected = record["display_field"]
        source_identity = record["source_identity_sha256"]
        source_scope = record["source_identity_scope"]
    else:
        # A fresh producer record has its own executed-source contract. Never
        # assign the identity of a previous published snapshot to new data.
        row = next(
            row
            for row in record["rows"]
            if row["n"] == 4 and row["boundary_case"] == "sine_zero_dirichlet"
        )
        selected = {"archive": row["archive"], "archive_sha256": row["archive_sha256"], "n": 4}
        sources = row["source_sha256"]
        if any(item["source_sha256"] != sources for item in record["rows"]):
            raise ValueError("one unchanged executed source generation required")
        source_identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        source_scope = "Executed source map recorded by this fresh acquisition"
    archive = record_path.parent / selected["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != selected["archive_sha256"]:
        raise ValueError("selected executed display archive changed")
    acquisition, arrays = read_archive(archive)
    if acquisition["n"] != selected["n"] or acquisition["boundary_case"] != "sine_zero_dirichlet":
        raise ValueError("selected homogeneous physical field archive required")
    figures = convergence(record, output)
    field_figures, samples = fields(arrays, output)
    figures.update(field_figures)
    provenance = dict(
        schema="pymhm-nested-figure-provenance-v1",
        numerical_source_identity_sha256=source_identity,
        numerical_source_identity_scope=source_scope,
        acquisition_uuid=acquisition["acquisition_uuid"],
        input_archive=archive.relative_to(ROOT).as_posix(),
        input_archive_sha256=selected["archive_sha256"],
        input_record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
        render_source_sha256={
            Path(__file__).relative_to(ROOT).as_posix(): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
        actual_display_digests=samples,
        one_sided_sample_pixels=(
            "Bound within each independent leaf; no shared-face averaging or color interpolation"
        ),
        macro_mesh=(
            "Actual archived outer and inner face endpoints on every analytical, numerical "
            "and error panel"
        ),
        colorbars="Nine independent field/axis/colorbar regions",
        figures_sha256=figures,
        mathematical_flux="minus the one-sided Q2 gradient; not an H(div) field",
        physical_norms=(
            "Actual archived q8/q10 tables; pointwise figure errors do not replace "
            "integrated physical norms"
        ),
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    """Render current checked field archives without acquiring or solving a numerical case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "examples/results/nested.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/nested")
    args = parser.parse_args()
    print(json.dumps(render(args.record.resolve(), args.output.resolve()), indent=2), flush=True)


if __name__ == "__main__":
    main()
