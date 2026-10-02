"""Compare all acquired Marmousi trace degrees without recomputing acoustic fields.

The published table and the specified-data experiment remain separate series.
Four incident sampling conventions are retained as a range; physical volume
norms use the common integration records and the fixed point-source cutout.
"""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
COLORS = {20: "#0072B2", 40: "#D55E00", 80: "#009E73"}


def read_rows(source: Path) -> list[dict[str, Any]]:
    """Require all fifteen accepted fields and matching physical-comparison digests."""
    reference = source / "classical-p4.json"
    reference_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    rows = []
    for width in (20, 40, 80):
        for degree in range(5):
            path = source / "family" / f"mhm-H{width}-ell{degree}-q9.json"
            comparison = path.with_name(f"{path.stem}-vs-classical-p4.json")
            record = json.loads(path.read_text())
            norms = json.loads(comparison.read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if record["source_changed_during_run"] or norms["source_changed_during_run"]:
                raise ValueError("a trace-family record has changed acquisition sources")
            if (
                norms["candidate_record_sha256"] != digest
                or norms["reference_record_sha256"] != reference_digest
            ):
                raise ValueError("the physical comparison does not identify the current fields")
            sides = [tuple(row["incident_side"]) for row in norms["sampled_pressure"]]
            if set(sides) != {(-1, -1), (-1, 1), (1, -1), (1, 1)} or len(sides) != 4:
                raise ValueError("four distinct incident sampling conventions are required")
            if record["H_m"] != width or record["trace_degree"] != degree:
                raise ValueError("trace-family filename and discretization disagree")
            rows.append(
                {
                    "H_m": width,
                    "trace_degree": degree,
                    "macro_cells": record["macro_cells"],
                    "global_complex_dofs_free": record["global_complex_dofs_free"],
                    "algebraic_residual": record["algebraic_residual"],
                    "macro_balance_max": record["macro_balance_max"],
                    "local_equation_residual_max": record["local_equation_residual_max"],
                    "sampled_pressure": norms["sampled_pressure"],
                    "norms": norms["norms"]["10"],
                    "quadrature_check": norms["norms"]["8"],
                    "candidate_record": path.name,
                    "candidate_record_sha256": digest,
                    "comparison_record": comparison.name,
                    "comparison_record_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(),
                }
            )
    return rows


def plot_family(
    rows: list[dict[str, Any]], published: dict[str, Any], reference: dict[str, Any], output: Path
) -> None:
    """Separate sampled published-table comparisons from physical pressure and flux norms."""
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 8.1), layout="constrained")
    degrees = np.arange(5)
    for width, color in COLORS.items():
        group = [row for row in rows if row["H_m"] == width]
        sampled = np.array(
            [
                [100 * side["relative_difference"] for side in row["sampled_pressure"]]
                for row in group
            ]
        )
        lower, upper = sampled.min(axis=1), sampled.max(axis=1)
        axis = axes[0, 0]
        axis.fill_between(degrees, lower, upper, color=color, alpha=0.15)
        axis.plot(
            degrees, lower, color=color, marker="o", lw=1.4, label=f"H={width} m; incident range"
        )
        axis.plot(degrees, upper, color=color, lw=1.4)
        printed = [
            row["relative_sampled_pressure_percent"]
            for row in published["mhm_rows"]
            if row["macro_width_m"] == width
        ]
        axis.plot(
            degrees, printed, "s--", color=color, mfc="none", lw=1, label=f"H={width} m; Table 6.1"
        )
        for axis, name in zip(
            (axes[0, 1], axes[1, 0], axes[1, 1]),
            (
                "pressure_relative_difference",
                "flux_relative_difference",
                "graph_relative_difference",
            ),
            strict=True,
        ):
            axis.plot(
                degrees,
                [100 * row["norms"][name] for row in group],
                "o-",
                color=color,
                lw=1.4,
                label=f"H={width} m",
            )
    finest = reference["references"][-1]["increment_quadrature_check"]
    for axis, name in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        ("pressure_relative_difference", "flux_relative_difference", "graph_relative_difference"),
        strict=True,
    ):
        axis.axhline(
            100 * finest[name], color="#444444", ls=":", lw=1.2, label="Classical P3→P4 increment"
        )
    for axis, title in zip(
        axes.flat,
        (
            "513×129 pressure samples\nDeclared crop and published table",
            "Global pressure L² difference\nDeclared crop; P4 denominator",
            "Acoustic flux difference outside source cutout\nq = −ρ⁻¹∇p; P4 denominator",
            "Graph-norm difference outside source cutout\nSame cutout in both positive terms",
        ),
        strict=True,
    ):
        axis.set(title=title, xlabel="Polynomial trace degree ℓ", ylabel="Relative difference (%)")
        axis.set_xticks(degrees)
        axis.set_yscale("log")
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=0.2, which="major")
    legend = [
        Line2D([], [], color=color, marker="o", label=f"H={width} m")
        for width, color in COLORS.items()
    ]
    legend.extend(
        (
            Line2D(
                [],
                [],
                color="#444444",
                marker="s",
                mfc="none",
                ls="--",
                label="Published Table 6.1 samples",
            ),
            Line2D([], [], color="#444444", ls=":", label="Classical P3→P4 increment"),
            Patch(facecolor="#66737a", alpha=0.2, label="Four incident sampling conventions"),
        )
    )
    figure.legend(handles=legend, loc="outside lower center", ncol=3, fontsize=11)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"trace-family.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Publish the complete numerical comparison and render its four distinct measures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "examples/results/marmousi")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/marmousi")
    args = parser.parse_args()
    rows = read_rows(args.source)
    published_path = args.source / "published-table.json"
    reference_path = args.source / "classical-convergence.json"
    published, reference = (
        json.loads(path.read_text()) for path in (published_path, reference_path)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    plot_family(rows, published, reference, args.output)
    record = {
        "rows": rows,
        "reference": "Pixel-conforming classical triangular P4 on the declared Marmousi crop",
        "historical_article_arrays_identified": False,
        "incident_sampling": "All four incident conventions retained; no pressure averaging",
        "gradient_cutout_m": [4975.0, 5025.0, 25.0, 75.0],
        "published_table_sha256": hashlib.sha256(published_path.read_bytes()).hexdigest(),
        "reference_series_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = args.source / "trace-family-comparison.json"
    destination.write_text(json.dumps(record, indent=2) + "\n")
    shutil.copy2(destination, args.output / destination.name)
    mirror = args.output / "family"
    mirror.mkdir(exist_ok=True)
    for row in rows:
        for key in ("candidate_record", "comparison_record"):
            shutil.copy2(args.source / "family" / row[key], mirror / row[key])


if __name__ == "__main__":
    main()
