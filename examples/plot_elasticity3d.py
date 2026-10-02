"""Render archived 3D displacement/stress slices and independently integrated norms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from threadpoolctl import threadpool_limits

from examples.solve_elasticity3d import ElasticityData3D
from examples.tetra_section_samples import section_grid
from pymhm.elasticity3d import constitutive_values_3d
from pymhm.tetrahedral import TetraMesh, tetra_basis, tetra_nodal_space

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples/results/elasticity3d"
OUTPUT = ROOT / "docs/figures/elasticity3d"


def load(name: str) -> tuple[dict, dict]:
    """Read one completed campaign and verify the corresponding field archive."""
    report = json.loads((INPUT / f"{name}.json").read_text())
    path = INPUT / report["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["sha256"]:
        raise ValueError("field archive digest mismatch")
    with np.load(path) as archive:
        return report, dict(archive)


def section_fields(degree: int) -> tuple[dict, dict]:
    """Evaluate complete stored polynomials on disconnected fine-cell section grids.

    Every sample retains its owning tetrahedron. Coincident coordinates remain
    separate at interfaces; only the display triangles interpolate colors.
    Stress uses the physical gradient and the spatially varying Kelvin tensor.
    """
    report, data = load(f"anisotropic-p{degree}")
    problem = ElasticityData3D()
    macro = TetraMesh(data["macro_points"], data["macro_cells"])
    height, refinement = 0.375, 6
    outlines = section_grid(macro, height, refinement=1)["segments"]
    points, triangles, displacements, stresses = [], [], [], []
    offset = 0
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    for cell in range(len(macro.cells)):
        mesh = TetraMesh(data["local_points"][cell], data["local_cells"][cell])
        cut = section_grid(mesh, height, refinement)
        if not len(cut["points"]):
            continue
        dofs, _ = tetra_nodal_space(mesh, degree)
        parent = cut["parents"]
        basis, derivative = tetra_basis(degree, cut["barycentric"])
        coefficients = data["values"][cell, dofs[parent]]
        vertices = mesh.points[mesh.cells]
        inverse = np.linalg.inv(
            np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
        )
        gradient = np.einsum(
            "qia,qib,qcb->qac", coefficients, derivative, inverse[parent, 1:], optimize=True
        )
        strain = np.column_stack(
            [
                (gradient[:, i, j] + gradient[:, j, i]) / (2 if i == j else np.sqrt(2))
                for i, j in pairs
            ]
        )
        kelvin = np.einsum(
            "qab,qb->qa", constitutive_values_3d(problem.constitutive, cut["points"]), strain
        )
        sigma = np.zeros((len(parent), 3, 3))
        for index, (i, j) in enumerate(pairs):
            sigma[:, i, j] = sigma[:, j, i] = kelvin[:, index] / (1 if i == j else np.sqrt(2))
        displacements.append(np.einsum("qi,qia->qa", basis, coefficients))
        stresses.append(sigma)
        points.append(cut["points"])
        triangles.append(cut["cells"] + offset)
        offset += len(parent)
    xyz = np.concatenate(points)
    sampled = dict(
        points=xyz,
        cells=np.concatenate(triangles),
        segments=outlines,
        displacement=np.concatenate(displacements),
        stress=np.concatenate(stresses),
        exact_displacement=problem.displacement(xyz),
        exact_stress=problem.stress(xyz),
        height=height,
        display_refinement=refinement,
    )
    return report, sampled


def field_panels(report: dict, sampled: dict, degree: int, *, components: bool = False) -> None:
    """Render shared physical scales and separate error scales on broken display triangles."""
    u, stress = sampled["displacement"], sampled["stress"]
    exact_u, exact_stress = sampled["exact_displacement"], sampled["exact_stress"]
    if components:
        labels = ("Displacement uₓ", "Displacement uᵧ", "Stress σₓₓ", "Stress σₓᵧ")
        actual = np.column_stack((u[:, 0], u[:, 1], stress[:, 0, 0], stress[:, 0, 1]))
        expected = np.column_stack(
            (exact_u[:, 0], exact_u[:, 1], exact_stress[:, 0, 0], exact_stress[:, 0, 1])
        )
        error = actual - expected
    else:
        labels = ("Displacement magnitude", "Stress Frobenius norm")
        actual = np.column_stack((np.linalg.norm(u, axis=1), np.linalg.norm(stress, axis=(1, 2))))
        expected = np.column_stack(
            (np.linalg.norm(exact_u, axis=1), np.linalg.norm(exact_stress, axis=(1, 2)))
        )
        error = np.column_stack(
            (
                np.linalg.norm(u - exact_u, axis=1),
                np.linalg.norm(stress - exact_stress, axis=(1, 2)),
            )
        )
    xy = sampled["points"][:, :2]
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], sampled["cells"])
    fig = plt.figure(figsize=(14, 4.7 * len(labels) + 1.0), layout="constrained")
    grid = fig.add_gridspec(2 * len(labels) + 1, 3, height_ratios=[1, 0.055] * len(labels) + [0.1])
    for row, label in enumerate(labels):
        common = max(abs(actual[:, row]).max(), abs(expected[:, row]).max(), 1e-15)
        for column, values in enumerate((expected[:, row], actual[:, row], error[:, row])):
            axis = fig.add_subplot(grid[2 * row, column])
            limit = common if column < 2 else max(abs(values).max(), 1e-15)
            artist = axis.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                rasterized=True,
                norm=Normalize(-limit if components else 0, limit),
                cmap="RdBu_r" if components else ("viridis" if column < 2 else "magma"),
            )
            for color, width in (("white", 1.5), ("black", 0.55)):
                axis.add_collection(
                    LineCollection(sampled["segments"], colors=color, linewidths=width, zorder=3)
                )
            title = (
                "Exact",
                f"MHM P{degree}",
                "MHM − exact" if components else "Field-error norm",
            )[column]
            axis.set(
                xlim=(0, 1),
                ylim=(0, 1),
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=f"{title} · {label}",
            )
            colorbar = fig.colorbar(
                artist, cax=fig.add_subplot(grid[2 * row + 1, column]), orientation="horizontal"
            )
            colorbar.locator = MaxNLocator(4)
            colorbar.update_ticks()
    caption = fig.add_subplot(grid[-1, :])
    caption.set_axis_off()
    caption.text(
        0.5,
        0.4,
        "Complete local polynomials sampled independently in each fine tetrahedron.\n"
        "Black/white lines: actual macro intersections; no averaging across interfaces.",
        ha="center",
        va="center",
        fontsize=11,
    )
    last = report["rows"][-1]
    fig.suptitle(
        f"General-tensor elasticity · z = 3/8 · {last['macro_tetrahedra']} macro tetrahedra\n"
        f"Physical L² errors: displacement {last['displacement_l2']:.4g}; "
        f"stress {last['stress_l2']:.4g}",
        fontsize=16,
    )
    save(fig, f"{'components' if components else 'fields'}-p{degree}")


def fields(degree: int) -> dict:
    """Render magnitude and signed component panels and record exact archive provenance."""
    report, sampled = section_fields(degree)
    field_panels(report, sampled, degree)
    field_panels(report, sampled, degree, components=True)
    return dict(
        archive=report["archive"],
        archive_sha256=report["sha256"],
        degree=degree,
        height=sampled["height"],
        display_refinement=sampled["display_refinement"],
        display_points=len(sampled["points"]),
        display_triangles=len(sampled["cells"]),
        displacement_sample_error_max=float(
            np.linalg.norm(sampled["displacement"] - sampled["exact_displacement"], axis=1).max()
        ),
        stress_sample_error_max=float(
            np.linalg.norm(sampled["stress"] - sampled["exact_stress"], axis=(1, 2)).max()
        ),
        norm_definition=(
            "Display sample maxima; physical L2 volume errors remain in the acquisition records."
        ),
    )


def save(fig: plt.Figure, name: str) -> None:
    """Write both publication formats from the same figure."""
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=175)
    plt.close(fig)


def main() -> None:
    """Replay two five-level spatial studies and two bounded-force Lamé sweeps."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sampling = [fields(degree) for degree in (2, 3)]
    (OUTPUT / "field-sampling.json").write_text(json.dumps(sampling, indent=2) + "\n")
    for mode in ("anisotropic", "locking"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for degree in (2, 3):
            report, _ = load(f"{mode}-p{degree}")
            rows = report["rows"]
            x = np.array(
                [
                    1 / row["resolution"] if mode == "anisotropic" else row["lame_lambda"]
                    for row in rows
                ]
            )
            for ax, field in zip(axes, ("displacement_l2", "stress_l2"), strict=True):
                ax.loglog(
                    x,
                    [row[field] for row in rows],
                    "o-",
                    label=f"P{degree}, local r={rows[0]['local_refinement']}, P1 trace",
                )
                ax.set(
                    xlabel="Macro spacing 1/n"
                    if mode == "anisotropic"
                    else "First Lamé modulus λ (μ=1)",
                    ylabel=field.replace("_", " ") + " error",
                )
                ax.grid(alpha=0.25)
                ax.legend(fontsize=9)
                if mode == "anisotropic":
                    ax.set_xticks(x, labels=["1", "1/2", "1/3", "1/4", "1/5"])
                    ax.minorticks_off()
        fig.suptitle(
            "Spatially varying anisotropic elasticity"
            if mode == "anisotropic"
            else "Solenoidal finite-Lamé study · no locking-free claim"
        )
        save(fig, "convergence" if mode == "anisotropic" else "lame-sweep")


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
