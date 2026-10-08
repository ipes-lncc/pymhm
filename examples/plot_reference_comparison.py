"""Plot archived MSL and pyMHM Darcy comparisons without executing new solves."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from examples.plot_mesh import draw_macro_mesh
from pymhm import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures/reference"


def save(figure: plt.Figure, name: str) -> None:
    """Write exportable scientific figures in raster and vector formats."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{suffix}", dpi=170)
    plt.close(figure)


def plot_errors(report: dict) -> None:
    """Compare errors and field differences using distinct marker encodings."""
    rows = report["rows"]
    sizes = np.array([row["H"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for axis, quantity, title in zip(
        axes[:2], ("pressure", "raw_flux"), ("Pressure error", "Raw flux error"), strict=True
    ):
        reference = np.array([row[f"reference_{quantity}_l2"] for row in rows])
        ours = np.array([row[f"pymhm_{quantity}_l2"] for row in rows])
        axis.loglog(sizes, reference, "o-", color="#29647c", label="MSL (msl_mhm)")
        axis.loglog(sizes, ours, "x--", color="#bf632c", markersize=8, label="PyMHM")
        power = 2 if quantity == "pressure" else 1
        axis.loglog(
            sizes[-3:],
            reference[-1] * (sizes[-3:] / sizes[-1]) ** power,
            ":",
            color="#777777",
            label=f"Slope {power} guide",
        )
        axis.set_title(title)
        axis.set_ylabel("Error against the analytic field in L²")
        axis.legend(fontsize=9)
    for quantity, label in (("pressure", "Pressure"), ("raw_flux", "Raw flux")):
        axes[2].loglog(
            sizes, [row[f"{quantity}_field_difference_l2"] for row in rows], "o-", label=label
        )
    axes[2].set_title("PyMHM − MSL field difference")
    axes[2].set_ylabel("Full-field difference in L²")
    axes[2].legend(fontsize=9)
    for axis in axes:
        axis.set_xlabel("Macrotriangle diameter H")
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.22)
    figure.suptitle(
        "MSL and PyMHM: same primal MHM discretization, P1 local pressure/trace, h = H/8"
    )
    figure.supxlabel(
        "4, 16, 64, 256 and 1024 crisscross macrotriangles; weak homogeneous Dirichlet. "
        "Matching markers overlap.\nMSL: msl_mhm + msl_cg + msl_core (IPES/LNCC).",
        fontsize=10,
    )
    save(figure, "darcy-convergence")


def plot_fields() -> None:
    """Show exact and computed signed components on shared physical color scales."""
    with np.load(ROOT / "examples/results/reference-darcy-fields.npz") as data:
        points, cells = data["points"], data["cells"]
        reference_p, ours_p = data["reference_pressure"], data["pymhm_pressure"]
        reference_q, ours_q = data["reference_flux"], data["pymhm_flux"]
    with np.load(ROOT / "examples/results/reference-darcy-64.npz") as data:
        macro_mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    x, y = (2 * np.pi * points).T
    exact_p = np.sin(x) * np.sin(y)
    centers = points[cells].mean(axis=1)
    x, y = (2 * np.pi * centers).T
    exact_q = -2 * np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))
    figure, axes = plt.subplots(3, 4, figsize=(15, 10.5), constrained_layout=True)
    titles = ("Exact (sampled)", "MSL (msl_mhm)", "PyMHM", "PyMHM − MSL")
    for row, (expected, native, ours, label, limit) in enumerate(
        (
            (exact_p, reference_p, ours_p, "Pressure p", 1.0),
            (exact_q[:, 0], reference_q[:, 0], ours_q[:, 0], "Flux qₓ", 2 * np.pi),
            (exact_q[:, 1], reference_q[:, 1], ours_q[:, 1], "Flux qᵧ", 2 * np.pi),
        )
    ):
        for column, values in enumerate((expected, native, ours, ours - native)):
            scale = limit if column < 3 else max(float(np.max(np.abs(values))), 1e-16)
            artist = axes[row, column].tripcolor(
                triangulation,
                values,
                shading="gouraud" if row == 0 else "flat",
                cmap="RdBu_r",
                vmin=-scale,
                vmax=scale,
                rasterized=True,
            )
            axes[row, column].set_title(titles[column] + (f" · {label}" if column == 0 else ""))
            axes[row, column].set_aspect("equal")
            axes[row, column].set_xlabel("x")
            axes[row, column].set_ylabel("y")
            draw_macro_mesh(axes[row, column], macro_mesh)
            figure.colorbar(artist, ax=axes[row, column], shrink=0.8)
    figure.suptitle("64 macrotriangles / 4096 fine triangles: MSL and PyMHM field comparison")
    figure.supxlabel(
        "MSL: msl_mhm + msl_cg + msl_core (IPES/LNCC). Outlined edges show the macro mesh.\n"
        "Pressure uses nodal P1 rendering; raw flux is "
        "cellwise constant. Exact flux is sampled at "
        "cell centroids for display.\nNorms use quadrature, not these samples. Difference panels "
        "have their own roundoff-scale color bars.",
        fontsize=10,
    )
    save(figure, "darcy-fields")


def plot_coarse_cosine() -> None:
    """Display archived fluxes for the diagonal, constant-trace cosine case.

    The archive contains physical vectors at fine-cell centroids. For RT0,
    these samples are only a visualization; full-field L2 differences in the
    report were integrated with a rule exact for squared affine vectors.
    """
    report = json.loads((ROOT / "examples/results/coarse-cosine/comparison.json").read_text())
    coordinates = np.linspace(0, 1, 151)
    x, y = np.meshgrid(coordinates, coordinates)
    exact_magnitude = np.pi * np.sqrt(
        (np.sin(np.pi * x) * np.cos(np.pi * y)) ** 2 + (np.cos(np.pi * x) * np.sin(np.pi * y)) ** 2
    )
    for formulation in ("primal", "mixed"):
        record = report[formulation]
        with np.load(ROOT / f"examples/results/coarse-cosine/{formulation}-fields.npz") as data:
            points, cells = data["points"], data["cells"]
            macro_mesh = TriangleMesh(data["macro_points"], data["macro_cells"])
            native, ours = data["reference_flux"], data["pymhm_flux"]
        centers = points[cells].mean(axis=1)
        triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
        selection = slice(None, None, max(1, len(cells) // 150))
        positions = centers[selection]
        exact_arrows = np.pi * np.column_stack(
            (
                np.sin(np.pi * positions[:, 0]) * np.cos(np.pi * positions[:, 1]),
                np.cos(np.pi * positions[:, 0]) * np.sin(np.pi * positions[:, 1]),
            )
        )
        figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
        physical = axes[0].pcolormesh(
            x,
            y,
            exact_magnitude,
            shading="auto",
            cmap="viridis",
            vmin=0,
            vmax=np.pi,
            rasterized=True,
        )
        axes[0].set_title("Analytical Darcy flux")
        for axis, field, title in zip(
            axes[1:3],
            (native, ours),
            (record["plot_label"], f"pyMHM {formulation}"),
            strict=True,
        ):
            axis.tripcolor(
                triangulation,
                np.linalg.norm(field, axis=1),
                shading="flat",
                cmap="viridis",
                vmin=0,
                vmax=np.pi,
                rasterized=True,
            )
            axis.set_title(title)
        for axis, vectors in zip(
            axes[:3], (exact_arrows, native[selection], ours[selection]), strict=True
        ):
            axis.quiver(*positions.T, *vectors.T, scale=48, width=0.004, color="black")
        difference = np.linalg.norm(ours - native, axis=1)
        artist = axes[3].tripcolor(
            triangulation,
            difference,
            shading="flat",
            cmap="magma",
            vmin=0,
            vmax=max(float(difference.max()), 1e-16),
            rasterized=True,
        )
        axes[3].set_title("Vector difference magnitude\nat fine-cell centroids")
        figure.colorbar(physical, ax=list(axes[:3]), shrink=0.75, label="Physical |q|")
        figure.colorbar(artist, ax=axes[3], shrink=0.75, label="|q(pyMHM) − q(reference)|")
        for axis in axes:
            draw_macro_mesh(axis, macro_mesh)
            axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
        figure.suptitle(
            "32 diagonal macrotriangles / 512 fine triangles / one constant trace per macroface\n"
            f"Relative flux L² error: reference {record['reference_flux_relative_l2']:.8%}; "
            f"pyMHM {record['pymhm_flux_relative_l2']:.8%}"
        )
        figure.supxlabel(
            f"{record['plot_note']}\n"
            "Macro boundaries are outlined. Identical arrow locations and scale; "
            "no field smoothing. "
            "The difference panel uses its own scale.\n"
            f"Full-field flux difference in L²: {record['flux_difference_l2']:.3e}; "
            + (
                "the raw P1 gradient is constant on each fine triangle."
                if formulation == "primal"
                else "RT0 centroid colors do not display its affine within-cell variation."
            ),
            fontsize=9,
        )
        save(figure, f"coarse-cosine-{formulation}")


def main() -> None:
    """Render convergence and signed field comparisons from recorded data."""
    report = json.loads((ROOT / "examples/results/reference-darcy-comparison.json").read_text())
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    plot_errors(report)
    plot_fields()
    plot_coarse_cosine()


if __name__ == "__main__":
    main()
