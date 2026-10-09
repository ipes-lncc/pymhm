"""Plot archived MSL and pyMHM GaLS fields without running either comparison solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

from pymhm.io.workspace import (
    case_workspace,
    local_resource,
    read_resource_bytes,
    read_resource_text,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from threadpoolctl import threadpool_limits

from examples.elasticity_data import TrigonometricElasticityData
from examples.plot_mesh import draw_macro_mesh
from pymhm import TriangleMesh
from pymhm.fem.scalar.triangle import reference_basis

ROOT = case_workspace()
DEFAULT_RECORD = ROOT / "examples/results/elasticity-reference.json"
DEFAULT_OUTPUT = ROOT / "docs/figures/elasticity-reference"


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    """Save scientific figures as PNG and SVG with vector labels and mesh contours."""
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(figure)


def load_archive(record: dict[str, Any], directory: Path) -> dict[str, np.ndarray]:
    """Verify the checksum before loading an archived nodal field."""
    path = directory / record["archive"]
    if hashlib.sha256(read_resource_bytes(path)).hexdigest() != record["archive_sha256"]:
        raise ValueError(f"Reference archive checksum mismatch: {path.name}")
    with np.load(local_resource(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def plot_convergence(rows: list[dict[str, Any]], output: Path) -> None:
    """Compare both implementations on the five recorded P1/P1 macro meshes."""
    selected = sorted(
        (row for row in rows if row["degree"] == 1), key=lambda row: row["macro_diameter"]
    )
    sizes = np.array([row["macro_diameter"] for row in selected])
    metrics = [
        ("u_l2", r"$\|u-u_h\|_{L^2}$"),
        ("u_h1_semi", r"$\|\nabla(u-u_h)\|_{L^2}$"),
        ("p_l2", r"$\|p-p_h\|_{L^2}$"),
        ("stress_l2", r"$\|\sigma-\sigma_h\|_{L^2}$"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for axis, (key, label) in zip(axes.flat, metrics, strict=True):
        for source, style, title in [
            ("msl", "o-", "MSL_MHM + MSL_CG (GaLS)"),
            ("pymhm", "x--", "pyMHM GaLS"),
        ]:
            axis.loglog(
                sizes, [row[source][key] for row in selected], style, label=title, linewidth=1.6
            )
        axis.set(xlabel="Macro diameter H", ylabel=label)
        axis.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(r"GaLS P1/P1, linear macro traces, local subdivision 4, $\nu=0.4999$")
    save_figure(figure, output, "convergence")


def plot_differences(rows: list[dict[str, Any]], output: Path) -> None:
    """Show absolute field differences without interpreting roundoff as convergence."""
    selected = sorted(
        (row for row in rows if row["degree"] == 1), key=lambda row: row["macro_diameter"]
    )
    figure, axis = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    for key, label in [
        ("u_l2", "Displacement"),
        ("p_l2", "Pressure"),
        ("u_h1_semi", "Displacement gradient"),
        ("stress_l2", "Cauchy stress"),
    ]:
        values = [row["difference_l2"][key] for row in selected]
        axis.loglog([row["macro_diameter"] for row in selected], values, "o-", label=label)
    axis.set(
        xlabel="Macro diameter H",
        ylabel="Absolute L2 difference: pyMHM − MSL",
        title="Agreement of exported fields on the same physical quadrature",
    )
    axis.legend(fontsize=9)
    axis.grid(True, which="both", alpha=0.25)
    save_figure(figure, output, "field-differences")


def sample_fields(data: dict[str, np.ndarray]) -> tuple[mtri.Triangulation, dict[str, np.ndarray]]:
    """Evaluate each broken polynomial separately, preserving one-sided stresses."""
    reference = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, 8)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    basis, derivative, _ = reference_basis(int(data["degree"]), bary)
    dofs = data["dof_cells"]
    vertices = data["dof_points"][dofs[:, :3]]
    points = np.einsum("qi,tij->tqj", bary, vertices)
    count = len(reference.points)
    triangles = (reference.cells[None] + np.arange(len(vertices))[:, None, None] * count).reshape(
        -1, 3
    )
    triangulation = mtri.Triangulation(points[..., 0].ravel(), points[..., 1].ravel(), triangles)
    affine = np.concatenate((np.ones((*vertices.shape[:2], 1)), vertices), axis=2)
    bary_gradients = np.linalg.inv(affine)[:, 1:, :].swapaxes(1, 2)
    gradients = np.einsum("qin,tna->tqia", derivative, bary_gradients)
    fields = {}
    for source in ("msl", "pymhm"):
        nodal_u = data[f"{source}_displacement"][dofs]
        displacement = np.einsum("qi,tia->tqa", basis, nodal_u)
        pressure = data[f"{source}_pressure"][dofs] @ basis.T
        gradient = np.einsum("tqia,tic->tqca", gradients, nodal_u)
        fields[f"{source}_u1"] = displacement[..., 0].ravel()
        fields[f"{source}_pressure"] = pressure.ravel()
        fields[f"{source}_stress11"] = (2 * gradient[..., 0, 0] - pressure).ravel()
    exact = TrigonometricElasticityData(lame_lambda=4999, lame_mu=1)
    physical = points.reshape(-1, 2)
    fields["exact_u1"] = exact.displacement(physical)[:, 0]
    fields["exact_pressure"] = exact.pressure(physical)
    fields["exact_stress11"] = exact.stress(physical)[:, 0, 0]
    return triangulation, fields


def plot_fields(record: dict[str, Any], directory: Path, output: Path) -> None:
    """Display exact/reference/native fields, approximation errors and code differences."""
    data = load_archive(record, directory)
    macro = TriangleMesh(data["macro_points"], data["macro_cells"])
    triangulation, fields = sample_fields(data)
    for key, label in [
        ("u1", r"Displacement $u_1$"),
        ("pressure", "Herrmann pressure"),
        ("stress11", r"Cauchy stress $\sigma_{11}$"),
    ]:
        exact, reference, native = (
            fields[f"{source}_{key}"] for source in ("exact", "msl", "pymhm")
        )
        values = [exact, reference, native, reference - exact, native - exact, native - reference]
        titles = [
            "Analytical",
            "MSL_MHM + MSL_CG\nGaLS",
            "pyMHM\nGaLS",
            "MSL − analytical",
            "pyMHM − analytical",
            "pyMHM − MSL",
        ]
        field_limit = max(float(np.max(np.abs(value))) for value in values[:3])
        error_limit = max(float(np.max(np.abs(value))) for value in values[3:5])
        difference_limit = max(float(np.max(np.abs(values[5]))), np.finfo(float).tiny)
        figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained")
        artists = []
        for i, (axis, value, title) in enumerate(zip(axes.flat, values, titles, strict=True)):
            limit = field_limit if i < 3 else error_limit if i < 5 else difference_limit
            artist = axis.tripcolor(
                triangulation,
                value,
                shading="gouraud",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                rasterized=True,
            )
            artists.append(artist)
            draw_macro_mesh(axis, macro, label=i == 0)
            axis.set(xlabel="x", ylabel="y", title=title, aspect="equal", xlim=(0, 1), ylim=(0, 1))
            axis.title.set_fontsize(10)
        figure.colorbar(artists[0], ax=list(axes[0]), shrink=0.83, label=label)
        figure.colorbar(artists[3], ax=list(axes[1, :2]), shrink=0.83, label="Approximation error")
        figure.colorbar(artists[5], ax=axes[1, 2], shrink=0.83, label="Implementation difference")
        figure.suptitle(
            f"{label}: P{record['degree']}/P{record['degree']}, "
            f"n={record['macro_resolution']}, linear macro traces\n"
            "Black lines: actual macroelement boundaries"
        )
        save_figure(figure, output, key)


def main() -> None:
    """Render the complete archived comparison gallery from a JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = json.loads(read_resource_text(arguments.record))
    with threadpool_limits(1):
        plot_convergence(report["rows"], arguments.output)
        plot_differences(report["rows"], arguments.output)
        selected = next(
            row for row in report["rows"] if row["degree"] == 3 and row["macro_resolution"] == 4
        )
        plot_fields(selected, arguments.record.parent, arguments.output)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.plot_elasticity_reference").main()
