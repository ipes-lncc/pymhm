"""Render archived Oseen fields and convergence; no numerical solve occurs here."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from plot_mesh import draw_macro_mesh

from pymhm import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/oseen"
FIGURES = ROOT / "docs/figures/oseen"


def fields(record: dict, name: str) -> None:
    """Plot exact/numerical fields with common scales and separate physical-error scales."""
    path = DATA / record["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Oseen field archive does not match its recorded checksum")
    with np.load(path) as data:
        triangulation = Triangulation(*data["points"].T, data["cells"])
        mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
        exact_u, numerical_u = data["exact_velocity"], data["velocity"]
        exact_p, numerical_p = data["exact_pressure"], data["pressure"]
        panels = [
            [
                np.linalg.norm(exact_u, axis=1),
                np.linalg.norm(numerical_u, axis=1),
                np.linalg.norm(numerical_u - exact_u, axis=1),
            ],
            [exact_p, numerical_p, np.abs(numerical_p - exact_p)],
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
        labels = [
            ["Exact velocity magnitude", "MHM velocity magnitude", "Velocity error norm"],
            ["Exact pressure", "MHM pressure", "Absolute pressure error"],
        ]
        for row in range(2):
            minimum = min(panels[row][0].min(), panels[row][1].min())
            maximum = max(panels[row][0].max(), panels[row][1].max())
            for column in range(3):
                ax = axes[row, column]
                options = (
                    dict(vmin=minimum, vmax=maximum, cmap="viridis")
                    if column < 2
                    else dict(vmin=0, cmap="magma")
                )
                field = ax.tripcolor(
                    triangulation,
                    panels[row][column],
                    shading="gouraud",
                    rasterized=True,
                    **options,
                )
                macro_edges = draw_macro_mesh(ax, mesh)
                macro_edges.set_linewidth(0.22 if len(mesh.cells) > 256 else 0.6)
                macro_edges.set_alpha(0.3 if len(mesh.cells) > 256 else 0.55)
                ax.set(aspect="equal", title=labels[row][column], xlabel="x", ylabel="y")
                fig.colorbar(field, ax=ax, shrink=0.78, pad=0.02)
        last = record["rows"][-1]
        fig.suptitle(
            f"{record['case'].capitalize()} Oseen — P3/P3, trace P{record['trace_degree']}\n"
            f"physical L² errors: velocity {last['velocity_l2']:.3e}; "
            f"pressure {last['pressure_l2']:.3e}"
        )
        for extension in ("png", "svg"):
            fig.savefig(FIGURES / f"{name}-fields.{extension}", dpi=180)
        plt.close(fig)


def history(record: dict, name: str) -> None:
    """Keep estimator components, physical norms and effectivity visible separately."""
    rows = record["rows"]
    x = np.array([row["trace_dofs"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    for key, label in (("velocity_l2", "velocity L²"), ("pressure_l2", "pressure L²")):
        axes[0].loglog(x, [row[key] for row in rows], "o-", label=label)
    for key, label in (
        ("mixed_error", "exact V×Q error"),
        ("eta1", "η₁: macro traces"),
        ("eta2", "η₂: local residual"),
    ):
        axes[1].loglog(x, [row[key] for row in rows], "o-", label=label)
    axes[2].semilogx(x, [row["effectivity"] for row in rows], "o-", color="tab:purple")
    axes[0].set_title("Integrated physical errors")
    axes[1].set_title("Two-level estimator")
    axes[2].set_title("(η₁ + η₂) / exact error")
    for ax in axes:
        ax.set_xlabel("Trace degrees of freedom")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend()
    axes[1].legend()
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}-history.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    """Render each completed verified archive without changing numerical records."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in sorted(DATA.glob("*.json")):
        record = json.loads(path.read_text())
        fields(record, path.stem)
        history(record, path.stem)


if __name__ == "__main__":
    main()
