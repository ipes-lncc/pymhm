"""Plot archived physical boundary and nonconvex-polygon convergence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.plot_style import set_refinement_ticks

ROOT = Path(__file__).resolve().parents[1]


def render(record_path: Path, output: Path) -> None:
    """Render all five measured series without solving a finite-element problem."""
    data = json.loads(record_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), layout="constrained")
    labels = {
        ("MH", "mixed"): "MH, mixed",
        ("MH", "neumann"): "MH, Neumann",
        ("MH2M", "dirichlet"): "MH²M, Dirichlet",
        ("MH2M", "mixed"): "MH²M, mixed",
        ("MH2M", "neumann"): "MH²M, Neumann",
    }
    for (method, boundary), label in labels.items():
        rows = [r for r in data["rows"] if r["method"] == method and r["boundary"] == boundary]
        n = np.array([r["resolution"] for r in rows])
        for axis, key in zip(axes, ("pressure_relative", "flux_relative"), strict=True):
            axis.loglog(n, [r[key] for r in rows], "o-", label=label, markersize=4)
            set_refinement_ticks(axis, n)
    for axis, title in zip(axes, ("Pressure", "Physical Darcy flux"), strict=True):
        axis.set(xlabel="L-polygon grid resolution n", ylabel="Relative L² error", title=title)
        axis.grid(True, which="major", alpha=0.3)
        axis.legend(fontsize=10, loc="lower left")
    figure.suptitle("Nonhomogeneous anisotropic data; nonconvex macroelements")
    for suffix in ("png", "svg"):
        figure.savefig(output / f"boundary-convergence.{suffix}", dpi=180)
    plt.close(figure)
    shutil.copy2(record_path, output / record_path.name)


def main() -> None:
    """Parse the archived record and destination directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record", type=Path, default=ROOT / "examples/results/mh/boundary-comparison.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/mh")
    args = parser.parse_args()
    render(args.record, args.output)


if __name__ == "__main__":
    main()
