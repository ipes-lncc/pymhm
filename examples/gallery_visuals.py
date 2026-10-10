"""Focused Gallery material and field plots using existing application owners."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits


def _digest(path: Path) -> str:
    """Return the SHA-256 digest of literal file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_record(path: Path, record: dict[str, Any], inputs: tuple[Path, ...]) -> None:
    """Bind a plot receipt to its executed helper sources and physical inputs."""
    root = Path.cwd()
    record["source_sha256"] = {
        item.resolve().relative_to(root).as_posix(): _digest(item) for item in inputs
    }
    record["runtime_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("pymhm", "numpy", "scipy", "numba", "fenics-basix", "matplotlib")
    }
    record["figure_sha256"] = _digest(path)
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def photonic_material(output: Path, *, preview: Path | None = None) -> None:
    """Render the declared fifteen-cylinder device's exact pointwise permittivity.

    This material picture is not an electric or magnetic field solution. The
    faint lines identify the acquired 16-by-16 MHM comparison macro partition.
    Permittivities are sampled at cell centres without refractive-index squaring.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.ticker import MaxNLocator

    from examples.maxwell_nanoguide import NanoWaveguide

    coordinate = (np.arange(1000) + 0.5) / 100
    xx, yy = np.meshgrid(coordinate, coordinate)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    model = NanoWaveguide()
    material = model.permittivity(points).reshape(xx.shape)
    figure, axis = plt.subplots(figsize=(6.2, 5), layout="constrained")
    image = axis.imshow(
        material,
        extent=(0, 10, 0, 10),
        origin="lower",
        interpolation="nearest",
        cmap=ListedColormap(("#e8edf2", "#22818e", "#402269")),
        norm=BoundaryNorm((0.75, 1.25, 2.3, 3.5), 3),
    )
    for value in np.linspace(0, 10, 17):
        axis.axvline(value, lw=0.32, color="#233644", alpha=0.3)
        axis.axhline(value, lw=0.32, color="#233644", alpha=0.3)
    axis.set(xlabel="$x$", ylabel="$y$", xlim=(0, 10), ylim=(0, 10), aspect="equal")
    axis.set_title("Photonic device: relative permittivity", fontsize=13, pad=10)
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.yaxis.set_major_locator(MaxNLocator(6))
    colorbar = figure.colorbar(image, ax=axis, ticks=(1, 1.5, 3.14), shrink=0.78, pad=0.04)
    colorbar.ax.set_yticklabels(("1 — air", "1.5 — silica", "3.14 — inclusions"))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    if preview is not None:
        colorbar.remove()
        figure.set_layout_engine(None)
        figure.set_size_inches(5, 5)
        axis.set_position((0, 0, 1, 1))
        axis.set_title("")
        axis.axis("off")
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=180)
    plt.close(figure)
    record = {
        "scope": "Exact declared material geometry; no wave field is solved or shown",
        "bounds": [0, 10, 0, 10],
        "macro_overlay": [16, 16],
        "inclusion_radius": model.radius,
        "sample_grid": [1000, 1000],
        "permittivities": [1, 1.5, 3.14],
        "image_interpolation": "nearest",
        "attribution": "IPES Research Group",
    }
    if preview is not None:
        record["preview_sha256"] = _digest(preview)
    _write_record(
        output,
        record,
        (Path(__file__), Path("examples/maxwell_nanoguide.py")),
    )


def layered_transport(
    output: Path,
    archive: Path,
    *,
    macro_divisions: int = 8,
    steps: int = 128,
    preview: Path | None = None,
) -> dict[str, Any]:
    """Acquire and render the existing layered manufactured transport at time one.

    This reuses the Darcy/transport application, material, source, exact field,
    H(div) velocity and hydrodynamic dispersion owners without redefining their
    numerical formulas. RT0/P0 Darcy drives local P3/SUPG transport with P2
    macroface traces and homogeneous diffusive walls. The default uses 128
    actual macrotriangles with 128 backward-Euler steps, rather than modifying
    only the display partition. Optional preview renders the same field with
    its actual macro edges and no axis/legend text for a small Gallery card.
    The plotted field preserves independent incident values and the macro mesh.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.ticker import MaxNLocator

    from examples.field_sampling import sample_field
    from examples.plot_mesh import draw_macro_mesh
    from examples.transport_campaign import (
        concentration,
        conductivity,
        layered_darcy,
        layered_trajectory,
    )

    with threadpool_limits(1):
        darcy, skeleton = layered_darcy(macro_divisions)
        mesh = skeleton.mesh
        trajectory = layered_trajectory(darcy, skeleton, steps, check_original=True)
        final = trajectory.solutions[-1]

        def exact(points: np.ndarray) -> np.ndarray:
            """Evaluate the independent analytical concentration at final time."""
            return np.exp(-1) * concentration(points)

        error10, error12 = final.l2_error(exact, 10), final.l2_error(exact, 12)
        flux_error = darcy.flux_l2_error(
            lambda points: np.column_stack((conductivity(points), np.zeros(len(points)))), 10
        )
    samples = sample_field(final.local_meshes, final.values, final.degree, 3)
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        **samples,
        exact=exact(samples["points"]),
        macro_points=mesh.points,
        macro_cells=mesh.cells,
    )
    triangulation = mtri.Triangulation(
        samples["points"][:, 0], samples["points"][:, 1], samples["cells"]
    )
    figure, axis = plt.subplots(figsize=(7.6, 3.2), layout="constrained")
    image = axis.tripcolor(
        triangulation,
        samples["values"],
        shading="gouraud",
        cmap="viridis",
        vmin=float(samples["values"].min()),
        vmax=float(samples["values"].max()),
        rasterized=True,
    )
    draw_macro_mesh(axis, mesh)
    axis.set(xlabel="$x$", ylabel="$y$", aspect="equal", xlim=(0, 3), ylim=(0, 1))
    axis.set_title("Layered Darcy-driven transport: concentration at $t=1$", fontsize=13, pad=10)
    axis.xaxis.set_major_locator(MaxNLocator(7))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", shrink=0.8, pad=0.13)
    colorbar.set_label("Concentration", fontsize=11)
    colorbar.locator = MaxNLocator(5)
    colorbar.update_ticks()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)
    if preview is not None:
        figure, axis = plt.subplots(figsize=(6.6, 2.2))
        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        axis.tripcolor(
            triangulation,
            samples["values"],
            shading="gouraud",
            cmap="viridis",
            vmin=float(samples["values"].min()),
            vmax=float(samples["values"].max()),
            rasterized=True,
        )
        draw_macro_mesh(axis, mesh)
        axis.set(xlim=(0, 3), ylim=(0, 1), aspect="equal")
        axis.axis("off")
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=180)
        plt.close(figure)
    record = {
        "scope": "Current-source acquisition of the documented layered analytical transport case",
        "bounds": [0, 3, 0, 1],
        "time": 1,
        "time_step": 1 / steps,
        "steps": steps,
        "macro_divisions": macro_divisions,
        "macro_triangles": len(mesh.cells),
        "local_refinement": 4,
        "darcy_spaces": "RT0/P0",
        "transport_degree": 3,
        "trace_degree": 2,
        "stabilization": "SUPG",
        "l2_error_order10": error10,
        "l2_error_order12": error12,
        "darcy_flux_l2_error": flux_error,
        "maximum_discrete_balance_residual": float(np.max(np.abs(trajectory.balance_residuals))),
        "maximum_original_equation_relative_residual": float(
            np.max(trajectory.original_residual_norms / trajectory.original_rhs_norms)
        ),
        "sample_archive_sha256": _digest(archive),
        "sample_archive_scope": "Broken physical display samples, not restart coefficients",
        "attribution": "IPES Research Group",
    }
    if preview is not None:
        record["preview_sha256"] = _digest(preview)
    _write_record(
        output,
        record,
        (
            Path(__file__),
            Path("examples/transport_campaign.py"),
            Path("examples/formulations/darcy_transport.py"),
            Path("examples/formulations/transient.py"),
        ),
    )
    return record
