"""Render archived AFW3D displacement, unsymmetrized stress and weak rotation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from threadpoolctl import threadpool_limits

from examples.mixed_elasticity3d_data import SolenoidalElasticity3D
from examples.plot_style import set_refinement_ticks
from examples.tetra_section_samples import section_grid
from pymhm.hdiv3d_family import HDiv3DFamily
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_dofs, hdiv3d_transform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/mixed-elasticity3d"
OUTPUT = ROOT / "docs/figures/mixed-elasticity3d"


def save(figure: plt.Figure, name: str) -> None:
    """Save matched raster/vector versions with independent axes and colorbar regions."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(OUTPUT / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def convergence(record: dict[str, Any]) -> None:
    """Show all five resolutions for both stress families without implying an exact rate."""
    figure = plt.figure(figsize=(11, 5.5), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1, 0.15))
    for column, degree in enumerate((2, 3)):
        rows = [r for r in record["convergence"] if r["stress_degree"] == degree]
        axis = figure.add_subplot(grid[0, column])
        for key, label in (
            ("displacement_l2", "Displacement"),
            ("stress_l2", "Stress (Frobenius)"),
            ("rotation_l2", "Axial rotation"),
        ):
            axis.loglog(
                [r["macro_cells"] for r in rows],
                [100 * r["relative_errors"][key] for r in rows],
                "o-",
                label=label,
            )
        set_refinement_ticks(axis, [r["macro_cells"] for r in rows])
        axis.set(
            xlabel="Tetrahedral macrocells",
            ylabel="Relative physical L² error (%)",
            title=f"BDM{degree} / P{degree - 1} / P{degree - 1} · λ = ∞",
        )
        axis.grid(alpha=0.2)
        legend = figure.add_subplot(grid[1, column])
        legend.set_axis_off()
        legend.legend(*axis.get_legend_handles_labels(), loc="center", ncol=3, frameon=False)
    figure.suptitle("Weak-symmetry mixed elasticity in 3D · original analytical problem")
    save(figure, "convergence")


def incompressibility(record: dict[str, Any]) -> None:
    """Compare fixed-space error as bulk modulus grows without increasing the body force."""
    rows = record["locking"]
    figure = plt.figure(figsize=(10, 5), layout="constrained")
    grid = figure.add_gridspec(2, 1, height_ratios=(1, 0.2))
    axis = figure.add_subplot(grid[0, 0])
    for key, label in (
        ("displacement_l2", "Displacement"),
        ("stress_l2", "Stress (Frobenius)"),
        ("rotation_l2", "Axial rotation"),
    ):
        axis.plot(
            np.arange(len(rows)), [100 * r["relative_errors"][key] for r in rows], "o-", label=label
        )
    axis.set_xticks(np.arange(len(rows)), ("0", "1", "10²", "10⁴", "10⁶", "10⁸", "∞"))
    axis.set(
        xlabel="λ / μ (categorical positions)",
        ylabel="Relative physical L² error (%)",
        title="Same solenoidal displacement, stress and force · fixed BDM2/P1/P1 space",
    )
    axis.grid(alpha=0.2)
    legend = figure.add_subplot(grid[1, 0])
    legend.set_axis_off()
    legend.legend(*axis.get_legend_handles_labels(), loc="center", ncol=3, frameon=False)
    save(figure, "incompressibility")


def fields(row: dict[str, Any]) -> dict[str, Any]:
    """Replay stored H(div) coordinates on disconnected fine-tetrahedron section triangles."""
    path = DATA / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("mixed-elasticity field archive does not match its acquisition digest")
    points, cells, values = [], [], []
    offset = 0
    degree = row["stress_degree"]
    family = HDiv3DFamily("tetrahedron", degree - 1, degree)
    with np.load(path) as archive:
        coefficients = archive["basis"]
        digest = hashlib.sha256(np.ascontiguousarray(coefficients).tobytes()).hexdigest()
        if digest != row["basis_sha256"]:
            raise ValueError("stored AFW stress basis has changed")
        macro = AffineMixedMesh(archive["macro_points"], archive["macro_cells"], "tetrahedron")
        outlines = section_grid(macro, refinement=1)["segments"]
        for cell in range(row["macro_cells"]):
            fine = AffineMixedMesh(
                archive[f"points_{cell}"], archive[f"cells_{cell}"], "tetrahedron"
            )
            cut = section_grid(fine, refinement=5)
            if not len(cut["points"]):
                continue
            parent = cut["parents"]
            vector, _, scalar = family.tabulate(
                cut["barycentric"][:, 1:], coefficients=coefficients
            )
            transform = hdiv3d_transform(fine, family, coefficients=coefficients)[parent]
            basis = np.einsum(
                "qab,qib,qij->qja", fine.jacobian[parent], vector, transform, optimize=True
            )
            basis /= fine.determinants[parent, None, None]
            stress = np.einsum(
                "qib,qia->qab", basis, archive[f"stress_{cell}"][hdiv3d_dofs(fine, family)[parent]]
            )
            displacement = np.einsum("qi,qia->qa", scalar, archive[f"displacement_{cell}"][parent])
            rotation = np.einsum("qi,qia->qa", scalar, archive[f"rotation_{cell}"][parent])
            values.append(
                np.column_stack(
                    (displacement[:, 0], stress[:, 0, 0], stress[:, 0, 1], rotation[:, 2])
                )
            )
            points.append(cut["points"])
            cells.append(cut["cells"] + offset)
            offset += len(parent)
    xyz, connectivity, numerical = (
        np.concatenate(points),
        np.concatenate(cells),
        np.concatenate(values),
    )
    data = SolenoidalElasticity3D()
    exact_stress = data.stress(xyz)
    analytical = np.column_stack(
        (
            data.displacement(xyz)[:, 0],
            exact_stress[:, 0, 0],
            exact_stress[:, 0, 1],
            data.rotation(xyz)[:, 2],
        )
    )
    triangulation = mtri.Triangulation(xyz[:, 0], xyz[:, 1], connectivity)
    figure = plt.figure(figsize=(13, 18), layout="constrained")
    grid = figure.add_gridspec(8, 3, height_ratios=[1, 0.065] * 4)
    for component, label in enumerate(
        ("Displacement uₓ", "Stress σₓₓ", "Stress σₓᵧ", "Axial rotation r_z")
    ):
        common = float(max(abs(analytical[:, component]).max(), abs(numerical[:, component]).max()))
        difference = numerical[:, component] - analytical[:, component]
        delta = float(max(abs(difference).max(), np.finfo(float).eps))
        for column, (samples, title) in enumerate(
            (
                (analytical[:, component], "Exact"),
                (numerical[:, component], "Mixed MHM"),
                (difference, "Difference"),
            )
        ):
            axis = figure.add_subplot(grid[2 * component, column])
            bound = delta if column == 2 else common
            artist = axis.tripcolor(
                triangulation,
                samples,
                shading="gouraud",
                rasterized=True,
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            axis.add_collection(
                LineCollection(outlines, colors="black", linewidths=0.55, alpha=0.75, zorder=3)
            )
            axis.set(
                aspect="equal",
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="x",
                ylabel="y",
                title=f"{title} · {label}",
            )
            colorbar = figure.colorbar(
                artist,
                cax=figure.add_subplot(grid[2 * component + 1, column]),
                orientation="horizontal",
            )
            colorbar.locator = MaxNLocator(3)
            colorbar.update_ticks()
    figure.suptitle(
        f"BDM{degree}/P{degree - 1}/P{degree - 1} · λ = ∞ · {row['macro_cells']} macrocells\n"
        "Section z = 0.37 · actual macro boundaries · "
        "full polynomial fields without interface averaging"
    )
    save(figure, f"bdm{degree}-fields")
    return dict(
        archive=path.name,
        archive_sha256=row["archive_sha256"],
        basis_sha256=digest,
        height=0.37,
        display_refinement=5,
        display_points=len(xyz),
        display_triangles=len(connectivity),
        sample_difference_max=np.max(abs(numerical - analytical), axis=0).tolist(),
    )


def run() -> None:
    """Render complete checked records and save explicit basis-replay provenance."""
    record = json.loads((DATA / "comparison.json").read_text())
    if len(record["convergence"]) != 10 or len(record["locking"]) != 7:
        raise ValueError("the two five-level sequences and seven Lamé cases must be complete")
    convergence(record)
    incompressibility(record)
    sampling = [
        fields([r for r in record["convergence"] if r["stress_degree"] == k][-1]) for k in (2, 3)
    ]
    (OUTPUT / "field-sampling.json").write_text(json.dumps(sampling, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
