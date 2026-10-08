"""Compare the declared CAMWA h/p and contrast studies with original plotted markers."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from examples.plot_style import set_refinement_ticks
from pymhm.io.workspace import case_workspace, read_resource_bytes, read_resource_text

ROOT = case_workspace()


def read_campaign(path: Path, count: int) -> dict:
    """Require complete unchanged acquisitions and identify every actual field archive."""
    record = json.loads(read_resource_text(path))
    if (
        not record.get("complete")
        or record["source_changed_during_run"]
        or len(record["cases"]) != count
    ):
        raise ValueError(f"incomplete campaign: {path.name}")
    for row in record["cases"]:
        actual = hashlib.sha256(read_resource_bytes(path.parent / row["archive"])).hexdigest()
        if actual != row["archive_sha256"]:
            raise ValueError(f"field digest mismatch: {row['archive']}")
    return record


def gradient_error(row: dict) -> float:
    """Select the higher recorded norm quadrature without changing the physical quantity."""
    order = max(int(key.removeprefix("quadrature_")) for key in row["norms"])
    return row["norms"][f"quadrature_{order}"]["gradient_absolute"]


def local_differences(data: Path, smooth: dict[int, dict]) -> list[dict]:
    """Validate physical fixed-trace differences against both acquired field digests."""
    selected = [(24, 32, f"ell{ell}-s32") for ell in range(4)]
    selected += [(24, 32, "ell4-s4"), (16, 32, "ell3-s16")]
    result = []
    for first, second, name in selected:
        expected = []
        for refinement in (first, second):
            row = next(row for row in smooth[refinement]["cases"] if row["name"] == name)
            expected.append({"name": row["archive"], "sha256": row["archive_sha256"]})
        records = {}
        digests = {}
        for order in (9, 11):
            path = data / "local-resolution" / f"{name}-r{first}-r{second}-q{order}.json"
            record = json.loads(read_resource_text(path))
            if (
                record["fields"] != expected
                or record["quadrature_order"] != order
                or record["source_changed_during_run"]
                or not record["source_sha256"]
            ):
                raise ValueError(f"local-resolution provenance mismatch: {path.name}")
            values = [record["norms"][key] for key in ("pressure_l2", "broken_gradient_l2")]
            if not np.all(np.isfinite(values)) or np.any(np.asarray(values) < 0):
                raise ValueError(f"invalid physical local-resolution norm: {path.name}")
            records[str(order)] = record["norms"]
            digests[path.name] = hashlib.sha256(read_resource_bytes(path)).hexdigest()
        result.append(
            {
                "name": name,
                "local_refinements": [first, second],
                "fields": expected,
                "physical_difference_by_quadrature": records,
                "record_sha256": digests,
            }
        )
    return result


def printed_comparisons(smooth: dict, contrast: dict, published: dict[int, dict]) -> list[dict]:
    """Match the printed configurations without shifting trace counts or normalizing errors."""
    smooth_rows = {row["name"]: row for row in smooth["cases"]}
    contrast_rows = {row["name"]: row for row in contrast["cases"]}
    result = []
    for figure in (2, 3, 7):
        for series in published[figure]["series"]:
            for point in series["values"]:
                if figure == 2:
                    name = f"ell{series['trace_degree']}-s{point['skeleton_subdivisions']}"
                elif figure == 3:
                    name = f"ell{point['trace_degree']}-s{series['skeleton_subdivisions']}"
                else:
                    name = f"{series['setting']}-contrast{int(point['contrast'])}"
                field = (contrast_rows if figure == 7 else smooth_rows)[name]
                value = gradient_error(field)
                lower, upper = point["graphical_interval"]
                result.append(
                    {
                        "figure": figure,
                        "configuration": name,
                        "archive_sha256": field["archive_sha256"],
                        "computed_absolute_gradient_error": value,
                        "published_absolute_gradient_error": point["value"],
                        "published_graphical_interval": [lower, upper],
                        "difference_relative_to_published": (value - point["value"])
                        / point["value"],
                        "inside_graphical_interval": lower <= value <= upper,
                    }
                )
    return result


def markers(axis: Any, values: list[dict], xkey: str, color: str, marker: str) -> None:
    """Draw the independently extracted values with their graphical reading envelopes."""
    x = np.array([row[xkey] for row in values])
    y = np.array([row["value"] for row in values])
    interval = np.array([row["graphical_interval"] for row in values])
    axis.errorbar(
        x,
        y,
        yerr=np.vstack((y - interval[:, 0], interval[:, 1] - y)),
        fmt=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        color=color,
        markersize=6,
        capsize=3,
        linewidth=1.1,
        zorder=5,
    )


def finish(figure: Any, axis: Any, output: Path, name: str, xlabel: str, subtitle: str) -> None:
    """Reserve room above the axes for method labels and below for the declared scope."""
    axis.set_xlabel(xlabel)
    axis.set_ylabel(r"Absolute broken gradient error $\|\nabla_h(p-p_H)\|_{L^2}$")
    axis.grid(True, which="major", alpha=0.25)
    figure.text(0.5, 0.025, subtitle, ha="center", va="bottom", fontsize=10)
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.79)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=220, facecolor="white")
    plt.close(figure)


def main() -> None:
    """Render all printed comparisons after both independent local-resolution controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "examples/results/unfitted")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/unfitted")
    args = parser.parse_args()
    data = args.source / "convergence"
    smooth = {
        r: read_campaign(data / f"smooth-p8-r{r}.json", 23 if r == 16 else 27) for r in (16, 24, 32)
    }
    contrast = {r: read_campaign(data / f"contrast-p4-r{r}.json", 12) for r in (8, 16)}
    differences = local_differences(data, smooth)
    publication = json.loads(read_resource_text(args.source / "published-convergence.json"))
    published = {row["figure"]: row for row in publication["figures"]}
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12, "legend.fontsize": 11})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    data_markers = ["o", "^", "s", "D"]
    by_name = {row["name"]: row for row in smooth[32]["cases"]}

    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    for degree, color, marker in zip(range(4), colors, data_markers, strict=True):
        values = published[2]["series"][degree]["values"]
        rows = [by_name[f"ell{degree}-s{s}"] for s in (32, 16, 8, 4, 2, 1)]
        axis.plot(
            [row["H"] for row in rows],
            [gradient_error(row) for row in rows],
            color=color,
            linewidth=1.8,
        )
        markers(axis, values, "H", color, marker)
    handles = [
        Line2D([], [], color=c, marker=m, markerfacecolor="white", label=rf"$\ell={ell}$")
        for ell, c, m in zip(range(4), colors, data_markers, strict=True)
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.95), ncol=4)
    figure.suptitle("Figure 2: h-refinement of the trace", y=0.99, fontsize=15)
    figure.text(
        0.5,
        0.83,
        "Lines: declared P8/r32 locals   |   Open markers: published values",
        ha="center",
        fontsize=11,
    )
    positions = [0.5 / s for s in (32, 16, 8, 4, 2, 1)]
    set_refinement_ticks(axis, positions, ["1/64", "1/32", "1/16", "1/8", "1/4", "1/2"])
    finish(
        figure,
        axis,
        args.output,
        "trace-h-convergence",
        r"Nominal skeletal size $H$",
        "Same printed PDE and declared 16-triangle crisscross partition; no error rescaling.",
    )

    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    axis.set_yscale("log")
    handles = []
    for series, color, marker in zip(
        published[3]["series"], colors[:3], data_markers[:3], strict=True
    ):
        segments = series["skeleton_subdivisions"]
        rows = [by_name[f"ell{ell}-s{segments}"] for ell in range(5)]
        axis.plot(range(5), [gradient_error(row) for row in rows], color=color, linewidth=1.8)
        markers(axis, series["values"], "trace_degree", color, marker)
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                markerfacecolor="white",
                label=rf"$H=\mathcal{{H}}/{segments}$",
            )
        )
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.95), ncol=3)
    figure.suptitle("Figure 3: polynomial enrichment of the trace", y=0.99, fontsize=15)
    figure.text(
        0.5,
        0.83,
        "Lines: declared P8/r32 locals   |   Open markers: published values",
        ha="center",
        fontsize=11,
    )
    set_refinement_ticks(axis, range(5))
    finish(
        figure,
        axis,
        args.output,
        "trace-p-convergence",
        r"Trace degree $\ell$",
        "Both figures use the absolute gradient norm; their nominal common point differs.",
    )

    figure, axis = plt.subplots(figsize=(8.4, 5.5))
    axis.set_xscale("log")
    axis.set_yscale("log")
    handles = []
    for series, color, marker in zip(
        published[7]["series"], colors[:2], data_markers[:2], strict=True
    ):
        setting = series["setting"]
        rows = [row for row in contrast[16]["cases"] if row["setting"] == setting]
        axis.plot(
            [row["contrast"] for row in rows],
            [gradient_error(row) for row in rows],
            color=color,
            linewidth=1.8,
        )
        markers(axis, series["values"], "contrast", color, marker)
        handles.append(
            Line2D([], [], color=color, marker=marker, markerfacecolor="white", label=setting)
        )
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.95), ncol=2)
    figure.suptitle("Figure 7: material contrast", y=0.99, fontsize=15)
    figure.text(
        0.5,
        0.83,
        "Lines: declared P4/r16 locals   |   Open markers: published values",
        ha="center",
        fontsize=11,
    )
    set_refinement_ticks(axis, [10.0**i for i in range(1, 7)], [rf"$10^{i}$" for i in range(1, 7)])
    finish(
        figure,
        axis,
        args.output,
        "contrast-convergence",
        r"Contrast $a^\star$",
        r"Declared control: $\ell=2$, S2 $\delta=1/6$; Figure 7 does not restate these choices.",
    )

    # The local-resolution table reports actual changes, not a certified local-error bound.
    local_rows = []
    for row in smooth[32]["cases"]:
        local_rows.append(
            {
                "name": row["name"],
                "gradient_error_by_local_refinement": {
                    str(r): gradient_error(other)
                    for r, record in smooth.items()
                    for other in record["cases"]
                    if other["name"] == row["name"]
                },
            }
        )
    summary = {
        "comparison_scope": "Printed PDE and trace sweeps; local P8/P4 discretizations declared",
        "published_sha256": hashlib.sha256(
            read_resource_bytes(args.source / "published-convergence.json")
        ).hexdigest(),
        "printed_comparisons": printed_comparisons(smooth[32], contrast[16], published),
        "smooth_local_controls": local_rows,
        "smooth_physical_local_differences": differences,
        "contrast_local_controls": [
            {
                "name": row["name"],
                "gradient_error_by_local_refinement": {
                    str(r): gradient_error(other)
                    for r, record in contrast.items()
                    for other in record["cases"]
                    if other["name"] == row["name"]
                },
                "gradient_error_2047_modes": row["series_2047_control"]["gradient_absolute"],
            }
            for row in contrast[16]["cases"]
        ],
        "acquisition_sha256": {
            path.name: hashlib.sha256(read_resource_bytes(path)).hexdigest()
            for path in sorted(
                [data / f"smooth-p8-r{r}.json" for r in smooth]
                + [data / f"contrast-p4-r{r}.json" for r in contrast]
            )
        },
    }
    (args.output / "convergence-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    (data / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in summary["acquisition_sha256"]:
        shutil.copy2(data / name, args.output / name)
    for difference in differences:
        for name in difference["record_sha256"]:
            shutil.copy2(data / "local-resolution" / name, args.output / name)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_unfitted_convergence").main()
