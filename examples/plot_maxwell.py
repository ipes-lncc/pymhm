"""Render signed Maxwell fields, actual macro sections and staggered convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection

from examples.maxwell_data import CavityMode
from examples.plot_darcy3d import slice_polygon
from examples.plot_style import set_refinement_ticks
from pymhm.fem.vector.curl import scalar_basis
from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)
from pymhm.meshes.triangle import TriangleMesh

ROOT = case_workspace()
RESULTS = ROOT / "examples/results/maxwell"
OUTPUT = ROOT / "docs/figures/maxwell"


def save(figure: Any, output: Path, name: str) -> None:
    """Save vector annotations and rasterized dense fields at publication resolution."""
    figure.savefig(output / f"{name}.png", dpi=180)
    figure.savefig(output / f"{name}.svg")
    plt.close(figure)


def sampled_fields(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Evaluate each incident DG polynomial separately, including 3D section intersections."""
    dimension, degree = int(data["dimension"]), int(data["degree"])
    template = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    template = template.submesh(0, 4)
    display_bary = np.column_stack((1 - template.points.sum(axis=1), template.points))
    points, cells, electric, magnetic, macros = [], [], [], [], []
    offset, height = 0, 0.37
    macro_vertices = data["macro_points"][data["macro_cells"]]
    if dimension == 2:
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        macros = list(macro.points[macro.faces])
    else:
        for vertices in macro_vertices:
            polygon = slice_polygon(vertices, height)
            if len(polygon):
                macros.extend(zip(polygon[:, :2], np.roll(polygon[:, :2], -1, axis=0), strict=True))
    for coords, connectivity, e, h in zip(
        data["local_points"], data["local_cells"], data["electric"], data["magnetic"], strict=True
    ):
        e = e.reshape(len(connectivity), -1, 1 if dimension == 2 else 3)
        h = h.reshape(len(connectivity), -1, dimension)
        for cell, vertices in enumerate(coords[connectivity]):
            if dimension == 2:
                triangles = [vertices]
            else:
                polygon = slice_polygon(vertices, height)
                triangles = [polygon[[0, i, i + 1]] for i in range(1, len(polygon) - 1)]
            for triangle in triangles:
                physical = display_bary @ triangle
                bary = np.column_stack((np.ones(len(physical)), physical)) @ np.linalg.inv(
                    np.column_stack((np.ones(dimension + 1), vertices))
                )
                basis = scalar_basis(degree, bary)[0]
                points.append(physical)
                cells.append(template.cells + offset)
                electric.append(basis @ e[cell])
                magnetic.append(basis @ h[cell])
                offset += len(physical)
    return {
        "points": np.concatenate(points),
        "cells": np.concatenate(cells),
        "electric": np.concatenate(electric),
        "magnetic": np.concatenate(magnetic),
        "macro_segments": macros,
    }


def panel(axis: Any, sampled: dict[str, Any], values: np.ndarray, title: str, limit: float) -> None:
    """Retain signed values, a centered color scale and all actual macro boundaries."""
    points = sampled["points"]
    tri = mtri.Triangulation(points[:, 0], points[:, 1], sampled["cells"])
    artist = axis.tripcolor(
        tri, values, shading="gouraud", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True
    )
    axis.add_collection(LineCollection(sampled["macro_segments"], colors="white", linewidths=0.9))
    axis.add_collection(LineCollection(sampled["macro_segments"], colors="black", linewidths=0.35))
    axis.set(xlabel="x", ylabel="y", title=title, aspect="equal", xlim=(0, 1), ylim=(0, 1))
    axis.set_title(title, fontsize=16)
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    axis.tick_params(labelsize=15)
    axis.set_xticks([0, 0.5, 1])
    axis.set_yticks([0, 0.5, 1])
    bar = axis.figure.colorbar(artist, ax=axis, fraction=0.045, pad=0.025)
    bar.ax.tick_params(labelsize=15)


def fields(row: dict[str, Any], source: Path, output: Path) -> None:
    """Plot analytical/numerical/difference components at exactly the archived field times."""
    path = source / row["fields"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != row["fields_sha256"]:
        raise ValueError("Maxwell field archive digest mismatch")
    with np.load(local_resource(path)) as stored:
        data = {name: stored[name] for name in stored.files}
    sampled = sampled_fields(data)
    mode = CavityMode(row["dimension"], float(data["wavenumber"]))
    exact_e = mode.electric(float(data["electric_time"]), sampled["points"])
    exact_h = mode.magnetic(float(data["magnetic_time"]), sampled["points"])
    groups = (
        [
            (
                "fields",
                np.column_stack((exact_e, exact_h)),
                np.column_stack((sampled["electric"], sampled["magnetic"])),
                ["E z", "H x", "H y"],
            )
        ]
        if row["dimension"] == 2
        else [
            ("electric", exact_e, sampled["electric"], ["E x", "E y", "E z"]),
            ("magnetic", exact_h, sampled["magnetic"], ["H x", "H y", "H z"]),
        ]
    )
    for suffix, expected, actual, names in groups:
        figure, axes = plt.subplots(3, 3, figsize=(14, 12.3), layout="constrained")
        for index, (row_axes, name) in enumerate(zip(axes, names, strict=True)):
            limit = float(max(abs(expected[:, index]).max(), abs(actual[:, index]).max()))
            error = actual[:, index] - expected[:, index]
            panel(row_axes[0], sampled, expected[:, index], name + ": analytical", limit)
            panel(row_axes[1], sampled, actual[:, index], name + ": MHM", limit)
            panel(row_axes[2], sampled, error, name + ": MHM − analytical", float(abs(error).max()))
        section = "; section z=0.37" if row["dimension"] == 3 else ""
        figure.suptitle(
            f"PEC cavity {row['dimension']}D: ℓ={row['trace_degree']}, P{row['local_degree']}"
            f"; tE={float(data['electric_time']):.5f}, "
            f"tH={float(data['magnetic_time']):.5f}{section}"
        )
        save(figure, output, path.stem + "-" + suffix)


def convergence(record: dict[str, Any], output: Path) -> None:
    """Display genuine maximum-in-time L2 and broken-curl errors separately."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.1), layout="constrained")
    for dimension, ell in ((2, 1), (2, 2), (3, 1)):
        rows = [
            r
            for r in record["rows"]
            if r["study"] == "cavity" and r["dimension"] == dimension and r["trace_degree"] == ell
        ]
        h = 1 / np.array([r["resolution"] for r in rows])
        for axis, key in zip(axes, ("combined_l2", "combined_hcurl"), strict=True):
            error = np.array([r["max_in_time"][key] for r in rows])
            rate = np.log(error[-2] / error[-1]) / np.log(h[-2] / h[-1])
            axis.loglog(h, error, "o-", label=f"{dimension}D ℓ={ell}; final rate {rate:.2f}")
            axis.set(xlabel="Cartesian grid width", ylabel="Maximum physical error norm")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=12)
    for axis in axes:
        values = [1, 0.5, 1 / 3, 0.25, 0.2, 0.125, 1 / 12, 0.0625]
        set_refinement_ticks(axis, values, ["1", "1/2", "1/3", "1/4", "1/5", "1/8", "1/12", "1/16"])
    axes[0].set_title("Combined L² error")
    axes[1].set_title("Combined broken H(curl) error")
    figure.suptitle("Local Pℓ₊₂ on one simplex; 100 steps with Δt=0.0005")
    save(figure, output, "convergence")


def time_and_energy(record: dict[str, Any], output: Path) -> None:
    """Separate second-order temporal consistency from the PEC energy invariant."""
    rows = [r for r in record["rows"] if r["study"] == "time"]
    dt = np.array([r["time_step"] for r in rows])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for key, name in (
        ("electric_error_l2", "electric"),
        ("magnetic_error_l2", "magnetic"),
        ("combined_error_l2", "combined"),
    ):
        axes[0].loglog(dt, [r[key] for r in rows], "o-", label=name)
    set_refinement_ticks(axes[0], dt, [f"{value:g}" for value in dt])
    axes[0].set(
        xlabel="Time step", ylabel="Physical L² error", title="Exact semidiscrete reference"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    cavity = [r for r in record["rows"] if r["study"] == "cavity"]
    for dimension, ell in ((2, 1), (2, 2), (3, 1)):
        selected = [r for r in cavity if r["dimension"] == dimension and r["trace_degree"] == ell]
        axes[1].semilogy(
            [r["resolution"] for r in selected],
            [r["modified_energy_relative_drift"] for r in selected],
            "o-",
            label=f"{dimension}D ℓ={ell}",
        )
    axes[1].set(
        xlabel="Grid cells per direction",
        ylabel="Maximum relative energy drift",
        title="Leapfrog cross-time energy",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    save(figure, output, "time-energy")


def main() -> None:
    """Plot immutable physical records without repeating a time evolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    record = json.loads(read_resource_text(args.input / "comparison.json"))
    if record.get("source_changed_during_run") is not False:
        raise ValueError("require a complete Maxwell acquisition with unchanged sources")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "figure.titlesize": 17})
    convergence(record, args.output)
    time_and_energy(record, args.output)
    for row in record["rows"]:
        if "fields" in row:
            fields(row, args.input, args.output)
    shutil.copy2(args.input / "comparison.json", args.output / "comparison.json")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_maxwell").main()
