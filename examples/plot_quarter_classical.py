"""Render the classical RT0 reference and the measured effect of MHM trace enrichment."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.plot_quarter_reference import (
    DATA,
    FIGURES,
    compare_components,
    compare_directions,
    compare_fields,
    load_record,
)


def convergence_figure(report: dict) -> None:
    """Plot integrated field differences without treating the reference as exact."""
    comparisons = report["mhm_trace_enrichment"]
    last_change = report["refinement_differences"][-1]
    segments = [row["segments"] for row in comparisons]
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "savefig.dpi": 220}):
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
        for axis, field, title in zip(
            axes, ("pressure", "flux"), ("Pressure", "Flux"), strict=True
        ):
            key = f"{field}_l2_relative"
            axis.semilogy(
                segments,
                np.array([row[key] for row in comparisons]) * 100,
                "o-",
                color="#126b9a",
                label="MHM vs. global RT0 (same fine mesh)",
            )
            axis.axhline(
                last_change[key] * 100,
                linestyle="--",
                color="#bb492c",
                label="Classical reference: finest refinement change",
            )
            axis.set(
                xticks=segments,
                xlabel="Constant trace segments per macroface",
                ylabel="Relative $L^2$ difference (%)",
                title=title,
            )
            axis.grid(True, which="both", alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.08), frameon=False
        )
        figure.text(
            0.5,
            0.03,
            "Differences use integrated fields. "
            "The reference refinement change is not an error bound.",
            ha="center",
            fontsize=10,
        )
        figure.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.33, wspace=0.32)
        for suffix in ("png", "svg"):
            figure.savefig(FIGURES / f"classical-trace-enrichment.{suffix}")
        plt.close(figure)


def main() -> None:
    """Read verified archives and redraw the classical-reference comparisons."""
    report = json.loads((DATA / "classical-convergence.json").read_text())
    row = next(item for item in report["mhm_trace_enrichment"] if item["segments"] == 2)
    record = load_record(row)
    row = {
        **row,
        "reference_discretization": f"{len(record['cells'])} fine triangles; r={row['refinement']}",
    }
    FIGURES.mkdir(parents=True, exist_ok=True)
    label = "NeoPZ global RT0/P0"
    compare_fields(row, label, "classical-vs-mhm-fields.png")
    compare_components(row, label, "classical-vs-mhm-components.png")
    compare_directions(row, label, "classical-vs-mhm-directions.png")
    convergence_figure(report)


if __name__ == "__main__":
    main()
