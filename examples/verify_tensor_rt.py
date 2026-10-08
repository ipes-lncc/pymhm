"""Reproduce the quadrilateral mixed-space configuration of Duran et al., Fig. 3."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import matplotlib

from pymhm.io.workspace import case_workspace

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import tensor_darcy as solve_darcy_tensor_rt
from pymhm import CartesianMacroMesh

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/tensor-rt.json"
FIGURES = ROOT / "docs/figures/tensor-rt"


def pressure(points: np.ndarray) -> np.ndarray:
    """Evaluate the smooth NeoPZ MHM reference problem TLaplaceExampleSmooth."""
    return np.cos(2 * np.pi * points[:, 0]) * np.cos(2 * np.pi * points[:, 1])


def flux(points: np.ndarray) -> np.ndarray:
    """Evaluate the physical flux -grad(p), with the reference frequency 2*pi."""
    x, y = 2 * np.pi * points.T
    return 2 * np.pi * np.column_stack((np.sin(x) * np.cos(y), np.cos(x) * np.sin(y)))


def source(points: np.ndarray) -> np.ndarray:
    """Evaluate div(q) independently from the mixed discrete divergence."""
    return 8 * np.pi**2 * pressure(points)


def plot(rows: list[dict]) -> None:
    """Plot pressure and flux errors in the same panel order as published axis labels."""
    from examples.plot_style import set_refinement_ticks

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), layout="constrained")
    reference_file = ROOT / "examples/results/tensor-rt-published.json"
    reference = json.loads(reference_file.read_text())["series"] if reference_file.exists() else []
    comparisons = []
    for published in reference:
        row = next(
            (
                r
                for r in rows
                if (r["N"], r["k"], r["enrichment"])
                == (published["N"], published["k"], published["enrichment"])
            ),
            None,
        )
        if row is not None:
            value = row[published["field"]]
            difference = abs(value - published["value"])
            comparisons.append(
                dict(
                    **published,
                    pymhm_value=value,
                    relative_difference=difference / published["value"],
                    uncertainty_units=difference / published["absolute_uncertainty"],
                )
            )
    (ROOT / "examples/results/tensor-rt-comparison.json").write_text(
        json.dumps(dict(comparisons=comparisons), indent=2) + "\n"
    )
    for k in (1, 2, 3):
        for n, marker in ((0, "s"), (1, "^"), (2, "o")):
            selected = sorted(
                (r for r in rows if r["k"] == k and r["enrichment"] == n), key=lambda r: r["N"]
            )
            h = [1 / r["N"] for r in selected]
            for column, key in enumerate(("pressure_l2", "flux_l2")):
                line = axes[k - 1, column].loglog(
                    h, [r[key] for r in selected], marker=marker, label=f"PyMHM k+{n}"
                )[0]
                points = [
                    r
                    for r in reference
                    if r["k"] == k and r["enrichment"] == n and r["field"] == key
                ]
                if points:
                    axes[k - 1, column].errorbar(
                        [r["H"] for r in points],
                        [r["value"] for r in points],
                        yerr=[r["absolute_uncertainty"] for r in points],
                        color=line.get_color(),
                        linestyle="none",
                        marker="x",
                        markersize=8,
                        label=f"Article k+{n}",
                    )
        for axis, name in zip(axes[k - 1], ("Pressure", "Physical flux"), strict=True):
            axis.set(xlabel="Macro size H", ylabel="L2 error", title=f"{name}: face degree k={k}")
            set_refinement_ticks(
                axis,
                [1 / n for n in (2, 4, 8, 16, 32, 64)],
                [f"1/{n}" for n in (2, 4, 8, 16, 32, 64)],
            )
            axis.grid(which="both", alpha=0.2)
            axis.legend(fontsize=8, ncol=2)
    fig.savefig(FIGURES / "convergence.png", dpi=200)
    fig.savefig(FIGURES / "convergence.svg")
    plt.close(fig)


def main() -> None:
    """Run full research levels separately from the lightweight automated tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    rows = json.loads(OUTPUT.read_text())["records"] if OUTPUT.exists() else []
    if not args.plot_only:
        with threadpool_limits(1):
            for k in (1, 2, 3):
                for n in (0, 1, 2):
                    for resolution in args.levels:
                        if any(
                            r["k"] == k and r["enrichment"] == n and r["N"] == resolution
                            for r in rows
                        ):
                            continue
                        start = perf_counter()
                        result = solve_darcy_tensor_rt(
                            CartesianMacroMesh(resolution),
                            degree=k,
                            enrichment=n,
                            source=source,
                            dirichlet=pressure,
                            local_refinement=2,
                            quadrature_order=8,
                        )
                        errors = result.errors(pressure, flux, source, order=10)
                        row = dict(
                            N=resolution,
                            k=k,
                            enrichment=n,
                            local_refinement=2,
                            **errors,
                            equilibrium_linf=max(
                                float(np.max(abs(v))) for v in result.equilibrium_residuals()
                            ),
                            normal_moment_linf=max(
                                float(np.max(abs(v))) for v in result.normal_flux_residuals()
                            ),
                            hybrid_residual=result.hybrid.residual,
                            seconds=perf_counter() - start,
                        )
                        assert row["equilibrium_linf"] < 2e-9
                        assert row["normal_moment_linf"] < 2e-9
                        rows.append(row)
                        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                        OUTPUT.write_text(
                            json.dumps(
                                dict(
                                    reference=(
                                        "Duran et al. (2019), Figure 3; NeoPZ TLaplaceExampleSmooth"
                                    ),
                                    pressure="cos(2*pi*x)*cos(2*pi*y)",
                                    records=rows,
                                ),
                                indent=2,
                            )
                            + "\n"
                        )
                        print(row, flush=True)
    plot(rows)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.verify_tensor_rt").main()
