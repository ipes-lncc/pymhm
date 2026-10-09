"""Plot the recorded analytical-problem errors as standalone scientific figures."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt

from pymhm.io.workspace import case_workspace, read_resource_text


def main() -> None:
    """Render convergence panels with explicit norms and independent method labels."""
    root = case_workspace()
    darcy = json.loads(read_resource_text(root / "examples/results/darcy-audit.json"))
    flow = json.loads(read_resource_text(root / "examples/results/flow-audit.json"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for formulation, label in (("primal", "Darcy: P1 primal"), ("mixed", "Darcy: RT0/P0 mixed")):
        rows = [
            row
            for row in darcy["studies"]["macro_refinement"]
            if row["formulation"] == formulation and row["macro_resolution"] >= 2
        ]
        axes[0].loglog(
            [1 / row["macro_resolution"] for row in rows],
            [row["pressure_l2"] for row in rows],
            "o-",
            label=label,
        )
    for formulation, label in (("taylor-hood", "Stokes: P2/P1"), ("usfem", "Stokes: USFEM P1/P1")):
        rows = [row for row in flow["H_refinement"] if row["formulation"] == formulation]
        axes[1].loglog(
            [1 / row["n"] for row in rows],
            [row["velocity_l2"] for row in rows],
            "o-",
            label=label,
        )
    for axis, norm in zip(axes, ("Pressure L2 error", "Velocity L2 error"), strict=True):
        axis.set_xlabel("Macro grid spacing")
        axis.set_ylabel(norm)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False)
    directory = root / "docs/figures"
    directory.mkdir(exist_ok=True)
    figure.savefig(directory / "convergence.svg")
    figure.savefig(directory / "convergence.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_verification").main()
