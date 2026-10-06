"""Replay the mixed-boundary Harder et al. (2015) campaign without solving problems."""

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

from examples.field_sampling import local_values
from examples.plot_mesh import draw_macro_mesh, macro_profile_breaks
from examples.plot_style import set_refinement_ticks
from examples.transport_mixed_campaign import exact
from examples.verify_transport_published import checked_endpoint, checked_rows
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/transport"
FIGURES = ROOT / "docs/figures/transport"


def save(figure: Any, name: str) -> None:
    """Write matching raster/vector figures with reserved axes and legend regions."""
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=180)
    plt.close(figure)


def load(row: dict) -> dict[str, np.ndarray]:
    """Check the exact archived field bytes before evaluating any plot."""
    path = DATA / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError(f"field digest mismatch: {path.name}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def convergence(report: dict) -> None:
    """Separate errors, jump indicators and same-skeleton local resolution."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.79, wspace=0.32)
    for index, key in enumerate(("l2_error", "broken_h1_error")):
        for section, label, marker in (
            ("adaptive", "Face adaptation: fixed 16 macros", "o"),
            ("spatial", "Uniform macro refinement", "s"),
        ):
            rows = report[section]
            axes[index].loglog(
                [row["free_trace_dofs"] for row in rows],
                [row["quadrature"]["12"][key] for row in rows],
                f"{marker}-",
                label=label,
            )
        axes[index].set(
            xlabel="Free interior trace DOFs",
            ylabel="Absolute error",
            title="Scalar L² error" if index == 0 else "Broken H¹ seminorm error",
        )
        axes[index].grid(True, which="both", alpha=0.25)
    rows = sorted(
        [*report["local_controls"], report["adaptive"][-1]], key=lambda r: r["local_refinement"]
    )
    refinement = [row["local_refinement"] for row in rows]
    for key, label in (("l2_error", "L² error"), ("broken_h1_error", "Broken H¹ error")):
        axes[2].loglog(
            refinement, [row["quadrature"]["12"][key] for row in rows], "o-", label=label
        )
    set_refinement_ticks(axes[2], refinement)
    axes[2].set(
        xlabel="Local subdivision r", ylabel="Absolute error", title="Same final trace space"
    )
    axes[2].grid(True, which="both", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.06, 0.015), frameon=False)
    handles, labels = axes[2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.985, 0.015), frameon=False)
    figure.suptitle(
        "Harder et al. (2015) mixed walls\nε = 0.1, P1 Galerkin locals and P0 multipliers"
    )
    save(figure, "mixed-convergence")


def published_comparison(report: dict) -> None:
    """Keep graphical Figure 12 coordinates and computed absolute errors unscaled."""
    published = json.loads((DATA / "published-figure-12.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.33, top=0.85, wspace=0.3)
    for axis, panel, key in zip(
        axes, published["panels"], ("l2_error", "broken_h1_error"), strict=True
    ):
        for series, color in zip(panel["series"], ("tab:green", "tab:red"), strict=True):
            points = series["points"]
            x, y = np.array([(point["N1"], point["value"]) for point in points]).T
            xinterval = np.array([point["N1_graphical_interval"] for point in points])
            yinterval = np.array([point["graphical_interval"] for point in points])
            axis.errorbar(
                x,
                y,
                xerr=np.stack((x - xinterval[:, 0], xinterval[:, 1] - x)),
                yerr=np.stack((y - yinterval[:, 0], yinterval[:, 1] - y)),
                fmt="o--" if series["method"] == "mesh" else "s--",
                color=color,
                markerfacecolor="none",
                capsize=3,
                label=f"Figure 12: {series['method']}",
            )
            for point in points:
                if point["visibility"] != "visible_marker":
                    axis.plot(point["N1"], point["value"], "x", color=color, ms=8)
        for section, marker, label in (
            ("spatial", "o", "ε = 0.1: uniform macro refinement"),
            ("adaptive", "s", "ε = 0.1: fixed 16 macros, face adaptation"),
        ):
            rows = report[section]
            axis.loglog(
                [row["free_trace_dofs"] for row in rows],
                [row["quadrature"]["12"][key] for row in rows],
                f"{marker}-",
                label=label,
            )
        coefficient_path = DATA / "mixed-coefficient-e1.json"
        if coefficient_path.exists():
            controls = json.loads(coefficient_path.read_text())["records"]
            axis.loglog(
                [row["free_trace_dofs"] for row in controls],
                [row["quadrature"]["12"][key] for row in controls],
                "D-",
                color="tab:purple",
                label="ε = 1: separate Figure 7 regime",
            )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="First-level DOFs (published) / free interior multipliers (PyMHM)",
            ylabel="Absolute L² error" if key == "l2_error" else "Absolute broken H¹ error",
        )
        axis.grid(True, which="both", alpha=0.2)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Harder et al. (2015), Figure 12 and explicit coefficient controls\n"
        "P1/P0 local/trace spaces"
    )
    figure.text(
        0.5,
        0.015,
        "Bars: graphical reading intervals. ×: vector endpoint obscured by the original legend. "
        "No axis rescaling.",
        ha="center",
        fontsize=9,
    )
    save(figure, "mixed-published-comparison")


def fields(report: dict) -> None:
    """Show broken initial/final fields with actual macrofaces and trace subdivisions."""
    rows = [report["adaptive"][0], report["adaptive"][-1]]
    data = [load(row) for row in rows]
    low = min(float(min(d["values"].min(), d["exact"].min())) for d in data)
    high = max(float(max(d["values"].max(), d["exact"].max())) for d in data)
    figure, axes = plt.subplots(2, 3, figsize=(12, 8.8), layout="constrained")
    for i, (row, values) in enumerate(zip(rows, data, strict=True)):
        mesh = TriangleMesh(values["macro_points"], values["macro_cells"])
        triangulation = mtri.Triangulation(*values["points"].T, triangles=values["cells"])
        for j, (field, title) in enumerate(
            (
                (values["exact"], "Analytical"),
                (values["values"], "MHM"),
                (values["values"] - values["exact"], "MHM − analytical"),
            )
        ):
            limit = float(np.max(abs(field)))
            artist = axes[i, j].tripcolor(
                triangulation,
                field,
                shading="gouraud",
                cmap="RdBu_r" if j == 2 else "viridis",
                vmin=-limit if j == 2 else low,
                vmax=limit if j == 2 else high,
                rasterized=True,
            )
            draw_macro_mesh(axes[i, j], mesh)
            if j == 1:
                for face, nodes in enumerate(mesh.faces):
                    start, end = mesh.points[nodes]
                    begin, stop = values["trace_break_offsets"][face : face + 2]
                    parameters = values["trace_breaks"][begin + 1 : stop - 1]
                    points = start + parameters[:, None] * (end - start)
                    axes[i, j].plot(*points.T, ".", color="red", ms=3)
            axes[i, j].set(
                title=(title if j == 0 else f"{title}; {row['free_trace_dofs']} free trace DOFs"),
                xlabel="x",
                ylabel="y",
                aspect="equal",
            )
            figure.colorbar(
                artist,
                ax=axes[i, j],
                orientation="horizontal",
                pad=0.12,
                label="Signed scalar error" if j == 2 else "Scalar field u",
            )
    save(figure, "mixed-fields")


def face_resolution() -> None:
    """Keep uniform, marked and local-resolution controls distinct on published axes."""
    published = json.loads((DATA / "published-figure-12.json").read_text())
    controls = {}
    for refinement in (16, 32, 64, 128, 256):
        path = DATA / f"mixed-face-uniform-e1-r{refinement}.json"
        if path.exists():
            rows = checked_rows(path)
            if len(rows) == 5:
                controls[refinement] = rows
    adaptive = checked_rows(DATA / "mixed-face-adaptive-e1-r16.json")
    endpoint_path = DATA / "mixed-face-endpoint-e1-r512.json"
    endpoint = checked_endpoint(endpoint_path)["records"] if endpoint_path.exists() else []
    figure, axes = plt.subplots(1, 2, figsize=(11, 6))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.29, top=0.84, wspace=0.3)
    for axis, panel, key in zip(
        axes, published["panels"], ("l2_error", "broken_h1_error"), strict=True
    ):
        points = next(s for s in panel["series"] if s["method"] == "space")["points"]
        x, y = np.array([(point["N1"], point["value"]) for point in points]).T
        interval = np.array([point["graphical_interval"] for point in points])
        axis.errorbar(
            x,
            y,
            yerr=np.stack((y - interval[:, 0], interval[:, 1] - y)),
            fmt="ks--",
            markerfacecolor="none",
            capsize=3,
            label="Published space curve; caption ε = 0.1",
        )
        for refinement, rows in controls.items():
            axis.loglog(
                [row["free_trace_dofs"] for row in rows],
                [row["quadrature"]["12"][key] for row in rows],
                "o-",
                label=f"Uniform segments; local r = {refinement}",
            )
        if endpoint:
            axis.loglog(
                [endpoint[0]["free_trace_dofs"]],
                [endpoint[0]["quadrature"]["12"][key]],
                "*",
                color="tab:brown",
                markersize=10,
                label="Finest endpoint only; local r = 512",
            )
        axis.loglog(
            [row["free_trace_dofs"] for row in adaptive],
            [row["quadrature"]["12"][key] for row in adaptive],
            "x:",
            color="tab:purple",
            label="Maximum marking, θ = 0.75; local r = 16",
        )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Free interior multiplier DOFs",
            ylabel="Absolute L² error" if key == "l2_error" else "Absolute broken H¹ error",
        )
        set_refinement_ticks(axis, [368, 736, 1472, 2944, 5888])
        axis.grid(True, which="both", alpha=0.22)
    figure.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("Fixed 256 macros: separate ε = 1 controls, P1/Galerkin/P0")
    figure.text(
        0.5,
        0.015,
        "Uniform subdivision does not implement the maximum-indicator rule.",
        ha="center",
        fontsize=9,
    )
    save(figure, "mixed-face-resolution")


def local_resolution() -> None:
    """Measure local error on fixed uniform and maximum-marked skeletal spaces."""
    uniform, adaptive, bounds = [], [], []
    for refinement in (16, 32, 64, 128, 256):
        path = DATA / f"mixed-face-uniform-e1-r{refinement}.json"
        if path.exists() and len(rows := checked_rows(path)) == 5:
            uniform.append((refinement, rows[-1]))
            report = json.loads(path.read_text())
            if "gradient_dg0_projection_error" in report:
                bounds.append((refinement, report["gradient_dg0_projection_error"]["12"]))
            elif refinement == 16:
                bound_path = DATA / "mixed-gradient-bound-r16.json"
                if bound_path.exists():
                    bound = json.loads(bound_path.read_text())
                    if bound["archive_sha256"] != rows[-1]["archive_sha256"]:
                        raise ValueError("gradient bound uses a different physical field archive")
                    bounds.append((refinement, bound["gradient_dg0_projection_error"]["12"]))
        if refinement == 16:
            row = checked_rows(DATA / "mixed-face-adaptive-e1-r16.json")[-1]
            adaptive.append((refinement, row))
        else:
            path = DATA / f"mixed-adaptive-fixed-e1-r{refinement}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if (
                    hashlib.sha256((DATA / row["archive"]).read_bytes()).hexdigest()
                    != row["archive_sha256"]
                ):
                    raise ValueError("fixed-skeleton archive digest differs")
                adaptive.append((refinement, row))
    endpoint = DATA / "mixed-face-endpoint-e1-r512.json"
    if endpoint.exists():
        report = checked_endpoint(endpoint)
        rows = report["records"]
        uniform.append((512, rows[0]))
        bounds.append((512, report["gradient_dg0_projection_error"]["12"]))
    refinements = sorted({r for r, _ in (*uniform, *adaptive)})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.82, wspace=0.3)
    for axis, key in zip(axes, ("l2_error", "broken_h1_error"), strict=True):
        for rows, label, marker in (
            (uniform, "Fixed uniform s = 16: 5888 multipliers", "o"),
            (adaptive, "Fixed maximum-marked space: 1408 multipliers", "s"),
        ):
            axis.loglog(
                [r for r, _ in rows],
                [v["quadrature"]["12"][key] for _, v in rows],
                f"{marker}-",
                label=label,
            )
        set_refinement_ticks(axis, refinements)
        axis.set(xlabel="Local subdivision r", ylabel="Absolute error")
        axis.set_title("Scalar L² error" if key == "l2_error" else "Broken H¹ seminorm error")
        axis.grid(True, which="both", alpha=0.22)
    if bounds:
        axes[1].loglog(
            *np.array(bounds).T, "k^--", label="DG0 gradient lower bound for any local P1 field"
        )
    for axis in axes:
        set_refinement_ticks(axis, refinements)
    figure.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False)
    figure.suptitle("Local approximation controls: same macrogeometry and skeletal moments; ε = 1")
    save(figure, "mixed-local-resolution")


def profiles(report: dict) -> None:
    """Evaluate independent one-sided local fields at every macroface intersection."""
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.8, wspace=0.3)
    height = 0.4375
    x = np.linspace(0, 1, 1001)
    for axis in axes:
        axis.plot(
            x, exact(np.column_stack((x, np.full_like(x, height)))), "k--", label="Analytical"
        )
    for row, label, color in zip(
        (report["adaptive"][0], report["adaptive"][-1]),
        ("Initial MHM", "Final MHM"),
        ("tab:orange", "tab:blue"),
        strict=True,
    ):
        data = load(row)
        mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
        start, end = np.array([0.0, height]), np.array([1.0, height])
        breaks = macro_profile_breaks(mesh, start, end)
        vertices = mesh.points[mesh.cells]
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        for k, (left, right) in enumerate(zip(breaks[:-1], breaks[1:], strict=True)):
            midpoint = start + (left + right) * (end - start) / 2
            local = np.einsum("tij,tj->ti", inverse, midpoint - vertices[:, 0])
            bary = np.column_stack((1 - local.sum(axis=1), local))
            cell = int(np.argmax(bary.min(axis=1)))
            parameter = np.linspace(left, right, 151)
            points = start + parameter[:, None] * (end - start)
            fine = TriangleMesh(data["local_points"][cell], data["local_cells"][cell])
            values = local_values(fine, data["coefficients"][cell], 1, points)
            for axis in axes:
                axis.plot(parameter, values, color=color, label=label if k == 0 else None)
        for axis in axes:
            for value in breaks[1:-1]:
                axis.axvline(value, color="0.6", lw=0.6, alpha=0.7)
    for axis, limits, title in zip(
        axes, ((0, 1), (0.75, 1)), ("Full profile", "Outflow layer"), strict=True
    ):
        axis.set(xlim=limits, xlabel="x", ylabel="Scalar field u", title=title)
        axis.grid(True, alpha=0.2)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    figure.suptitle("y = 0.4375; vertical lines mark actual macroface intersections", y=0.9)
    save(figure, "mixed-profiles")


def main() -> None:
    """Replay archived fields and current numerical comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons-only",
        action="store_true",
        help="Render comparison curves from compact records",
    )
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = json.loads((DATA / "mixed-campaign.json").read_text())
    convergence(report)
    published_comparison(report)
    if args.comparisons_only:
        return
    face_resolution()
    local_resolution()
    fields(report)
    profiles(report)
    (FIGURES / "mixed-campaign.json").write_bytes((DATA / "mixed-campaign.json").read_bytes())


if __name__ == "__main__":
    main()
