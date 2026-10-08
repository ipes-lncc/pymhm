"""Compare measured RAD matrix conditioning with published vector markers."""

import csv
import json

import matplotlib

from pymhm.io.workspace import case_workspace, local_resource, read_resource_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.plot_style import set_refinement_ticks

ROOT = case_workspace()
COLORS = ("#cf352e", "#a52a8b", "#168449", "#2166ac", "#9a7016")
MARKERS = ("^", "o", "s", "D", "x")


def main() -> None:
    """Render physical sweeps, mesh sensitivity and measured asymptotic exponents."""
    data = json.loads(read_resource_text(ROOT / "examples/results/rad-conditioning.json"))
    rows = data["records"]
    with (
        local_resource(ROOT / "examples/results/published/araya2024_conditioning.csv")
    ).open() as stream:
        published = list(csv.DictReader(stream))
    target = ROOT / "docs/figures/rad-conditioning"
    target.mkdir(parents=True, exist_ok=True)
    summary = []
    for study, symbol in (("epsilon", r"$\epsilon$"), ("omega", r"$\omega$")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for case, color, marker in zip(range(1, 6), COLORS, MARKERS, strict=True):
            records = sorted(
                (r for r in rows if r["study"] == study and r["case"] == case),
                key=lambda r: r[study],
            )
            for axis, name in zip(axes, ("local", "global"), strict=True):
                key = name + "_condition"
                valid = [r for r in records if key in r]
                x = np.array([r[study] for r in valid])
                y = np.array([r[key] for r in valid])
                axis.loglog(x, y, "-", color=color, linewidth=1.3, label=f"Case {case}")
                reference = sorted(
                    (
                        r
                        for r in published
                        if r["study"] == study and int(r["case"]) == case and r["matrix"] == name
                    ),
                    key=lambda r: float(r["parameter"]),
                )
                axis.loglog(
                    [float(r["parameter"]) for r in reference],
                    [float(r["condition"]) for r in reference],
                    linestyle="none",
                    marker=marker,
                    markersize=5,
                    markerfacecolor="none",
                    color=color,
                )
                summary.append(
                    dict(
                        study=study,
                        case=case,
                        matrix=name,
                        small_parameter_slope=float(np.polyfit(np.log(x[:3]), np.log(y[:3]), 1)[0]),
                        large_parameter_slope=float(
                            np.polyfit(np.log(x[-3:]), np.log(y[-3:]), 1)[0]
                        ),
                    )
                )
                axis.set(xlabel=symbol, ylabel=r"$\kappa_2$", title=f"{name.title()} operator")
                axis.grid(which="major", alpha=0.25)
        axes[0].legend(fontsize=9, ncols=2)
        fig.suptitle("Lines: PyMHM; open markers: published figures", fontsize=11)
        fig.savefig(target / f"{study}.png", dpi=220)
        fig.savefig(target / f"{study}.svg")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    records = sorted((r for r in rows if r["study"] == "local"), key=lambda r: r["refinement"])
    axes[0].semilogx(
        [r["refinement"] for r in records], [r["global_condition"] for r in records], "o-"
    )
    set_refinement_ticks(axes[0], [r["refinement"] for r in records])
    axes[0].set(
        xlabel="Local subdivisions r", ylabel=r"$\kappa_2$", title="Case 2; fixed P0 macrofaces"
    )
    for case, color, marker in zip(range(1, 6), COLORS, MARKERS, strict=True):
        records = sorted(
            (r for r in rows if r["study"] == "skeleton" and r["case"] == case),
            key=lambda r: r["segments"],
        )
        axes[1].loglog(
            [r["segments"] for r in records],
            [r["global_condition"] for r in records],
            marker=marker,
            color=color,
            label=f"Case {case}",
        )
    set_refinement_ticks(axes[1], [1, 2, 4, 8, 16])
    axes[1].set(
        xlabel="Face segments per macroface",
        ylabel=r"$\kappa_2$",
        title="Fixed 4×4 macro partition",
    )
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(target / "mesh.png", dpi=220)
    fig.savefig(target / "mesh.svg")
    plt.close(fig)
    (ROOT / "examples/results/rad-conditioning-slopes.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_rad_conditioning").main()
