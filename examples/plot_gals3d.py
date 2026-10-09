"""Render three-dimensional mixed elasticity fields, refinement and incompressibility controls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

from pymhm.io.workspace import case_workspace, read_resource_bytes, read_resource_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from examples.gals3d_data import GaLS3DData
from examples.plot_flow3d import overlay, slices
from examples.plot_style import set_refinement_ticks

ROOT = case_workspace()
LABELS = {"gals-p1": "GaLS P1/P1", "gals-p2": "GaLS P2/P2", "th-p2": "Taylor–Hood P2/P1"}


def save_fields(row: dict, output: Path) -> None:
    """Render exact, numerical and one-sided error sections with the true macro geometry."""
    path = ROOT / "examples/results/gals3d" / row["fields"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != row["fields_sha256"]:
        raise ValueError("elasticity archive digest mismatch")
    data = GaLS3DData(float(row["lame_lambda"]))
    polygons, actual, exact, macros = slices(path, data, vector_key="displacement")
    numerical = np.column_stack((np.linalg.norm(actual[:, :3], axis=1), actual[:, 3]))
    expected = np.column_stack((np.linalg.norm(exact[:, :3], axis=1), exact[:, 3]))
    error = np.column_stack(
        (np.linalg.norm(actual[:, :3] - exact[:, :3], axis=1), actual[:, 3] - exact[:, 3])
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.6), layout="constrained")
    for index, label in enumerate(["Displacement magnitude", "Herrmann pressure"]):
        limit = max(np.abs(numerical[:, index]).max(), np.abs(expected[:, index]).max(), 1e-15)
        error_limit = max(np.abs(error[:, index]).max(), 1e-15)
        common = Normalize(0, limit) if index == 0 else TwoSlopeNorm(0, vmin=-limit, vmax=limit)
        error_norm = (
            Normalize(0, error_limit)
            if index == 0
            else TwoSlopeNorm(0, vmin=-error_limit, vmax=error_limit)
        )
        for column, values in enumerate((expected[:, index], numerical[:, index], error[:, index])):
            axis = axes[index, column]
            artist = PolyCollection(
                polygons,
                array=values,
                cmap="viridis" if index == 0 else "RdBu_r",
                norm=common if column < 2 else error_norm,
                edgecolors="none",
                rasterized=True,
            )
            axis.add_collection(artist)
            overlay(axis, macros)
            axis.set_title(
                ("Exact", "MHM", "Vector error magnitude" if index == 0 else "MHM − exact")[column],
                fontsize=15,
            )
            colorbar = fig.colorbar(artist, ax=axis, pad=0.025, fraction=0.052)
            colorbar.set_label(
                label
                if column < 2
                else ("Displacement error magnitude" if index == 0 else "Pressure difference"),
                fontsize=13,
            )
    fig.suptitle(f"{LABELS[row['case']]} — λ=10⁸, variable shear, section z=0.37", fontsize=19)
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.86))
    fig.text(
        0.5,
        0.022,
        (
            "One-sided values at cut fine-cell polygon centroids; no interface averaging.\n"
            "Black/white lines show the actual macro-face intersections."
        ),
        ha="center",
        fontsize=12,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"{row['case']}-fields.{extension}", dpi=180)
    plt.close(fig)


def curves(report: dict, output: Path, study: str) -> None:
    """Display physical errors separately for mesh refinement and lambda sensitivity."""
    keys = ("displacement_l2", "pressure_l2", "stress_l2")
    labels = ("Displacement L2 error", "Pressure L2 error", "Cauchy-stress L2 error")
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.5), layout="constrained")
    for case, label in LABELS.items():
        rows = [r for r in report["rows"] if r["case"] == case and r["study"] == study]
        if not rows:
            continue
        x = (
            1 / np.array([r["macro_subdivisions"] for r in rows])
            if study == "refinement"
            else np.arange(len(rows))
        )
        for axis, key in zip(axes, keys, strict=True):
            if study == "refinement":
                axis.loglog(x, [r[key] for r in rows], "o-", label=label)
                set_refinement_ticks(axis, x, [f"1/{r['macro_subdivisions']}" for r in rows])
            else:
                axis.semilogy(x, [r[key] for r in rows], "o-", label=label)
                axis.set_xticks(x, ["1", "10²", "10⁴", "10⁶", "10⁸", "10¹²", "∞"])
    if study == "lambda":
        rows = report["primal_control"]
        for axis, key in [(axes[0], "displacement_l2"), (axes[2], "stress_l2")]:
            axis.semilogy(
                np.arange(len(rows)),
                [r[key] for r in rows],
                "s--",
                color="black",
                label="Primal P2 control",
            )
    for axis, label in zip(axes, labels, strict=True):
        axis.set_xlabel(
            "Macro cube side" if study == "refinement" else "Sampled λ (categorical)", fontsize=13
        )
        axis.set_ylabel(label, fontsize=13)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=10)
    fig.suptitle(
        "Three-dimensional mixed elasticity — "
        + (
            "five macro meshes, λ=10⁸"
            if study == "refinement"
            else "incompressibility at fixed macro n=2"
        ),
        fontsize=18,
    )
    for extension in ("png", "svg"):
        fig.savefig(output / f"{study}.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    """Replay only accepted archived fields and measured numerical records."""
    source = ROOT / "examples/results/gals3d/campaign.json"
    report = json.loads(read_resource_text(source))
    control = ROOT / "examples/results/gals3d/primal/campaign.json"
    report["primal_control"] = json.loads(read_resource_text(control))["primal_control"]
    output = ROOT / "docs/figures/gals3d"
    output.mkdir(parents=True, exist_ok=True)
    for case in LABELS:
        rows = [r for r in report["rows"] if r["case"] == case and r["study"] == "refinement"]
        save_fields(max(rows, key=lambda r: r["macro_subdivisions"]), output)
    for study in ["refinement", "lambda"]:
        curves(report, output, study)
    (output / "campaign.json").write_bytes(read_resource_bytes(source))
    (output / "primal-control.json").write_bytes(read_resource_bytes(control))


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_gals3d").main()
