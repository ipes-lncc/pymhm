"""Physical replay, temporal refinement and component plots for the elastic wave."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from examples.elastodynamics_campaign import ElasticWave
from examples.reconstruction3d_replay import line_intervals
from examples.tetra_section_samples import section_grid
from pymhm.fem.scalar.tetrahedron import (
    tetra_basis,
    tetra_element_tabulate,
    tetra_nodal_space,
    tetrahedron_quadrature,
)
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/elastodynamics"
OUTPUT = ROOT / "docs/figures/elastodynamics"


def load(path: Path) -> dict[str, np.ndarray]:
    """Load only the explicitly archived mesh and physical polynomial coefficients."""
    with np.load(path) as archive:
        return dict(archive)


def evaluate(
    data: dict[str, np.ndarray], macro: int, fine: TetraMesh, owners: np.ndarray, bary: np.ndarray
) -> np.ndarray:
    """Evaluate displacement and velocity on their declared independent fine-cell sides."""
    degree = int(data["local_degree"])
    dofs, _ = tetra_nodal_space(fine, degree)
    basis = tetra_basis(degree, bary)[0]
    return np.column_stack(
        [
            np.einsum("qi,qia->qa", basis, data[key][macro].reshape(-1, 3)[dofs[owners]])
            for key in ("displacement", "velocity")
        ]
    )


def difference(first: Path, second: Path) -> dict[str, float]:
    """Integrate same-space time-refinement increments using exact polynomial quadrature."""
    a, b = load(first), load(second)
    for key in ("macro_points", "macro_cells", "local_degree", "local_refinement", "time"):
        if not np.array_equal(a[key], b[key]):
            raise ValueError(f"time comparison requires matching {key}")
    mesh = TetraMesh(a["macro_points"], a["macro_cells"])
    degree = int(a["local_degree"])
    bary, w = tetrahedron_quadrature(degree + 3)
    totals = np.zeros(6, dtype=np.longdouble)
    for macro in range(len(mesh.cells)):
        fine = mesh.submesh(macro, int(a["local_refinement"]))
        dofs, _, basis, gradient, hessian = tetra_element_tabulate(fine, degree, bary)
        u = (a["displacement"][macro] - b["displacement"][macro]).reshape(-1, 3)[dofs]
        v = (a["velocity"][macro] - b["velocity"][macro]).reshape(-1, 3)[dofs]
        value = np.einsum("qi,tia->tqa", basis, u)
        velocity = np.einsum("qi,tia->tqa", basis, v)
        du = np.einsum("tqib,tia->tqab", gradient, u)
        dv = np.einsum("tqib,tia->tqab", gradient, v)
        ddu = np.einsum("tqnij,tna->tqaij", hessian, u)
        sigma = 0.4 * (
            du + du.swapaxes(-1, -2) + np.trace(du, axis1=-2, axis2=-1)[..., None, None] * np.eye(3)
        )
        div = ElasticWave.divergence(ddu)
        squares = (
            np.sum(value**2, axis=-1),
            np.sum(velocity**2, axis=-1),
            np.sum(du**2, axis=(-1, -2)),
            np.sum(dv**2, axis=(-1, -2)),
            np.sum(sigma**2, axis=(-1, -2)),
            np.sum(div**2, axis=-1),
        )
        for i, values in enumerate(squares):
            totals[i] += np.sum(fine.volumes[:, None] * w * values, dtype=np.longdouble)
    u, v, du, dv, sigma, div = totals
    return dict(
        zip(
            (
                "displacement_l2",
                "velocity_l2",
                "displacement_h1",
                "velocity_h1",
                "stress_l2",
                "stress_broken_hdiv",
            ),
            np.sqrt([u, v, u + du, v + dv, sigma, sigma + div]).astype(float).tolist(),
            strict=True,
        )
    )


def save(fig: Any, name: str) -> None:
    """Write the same publication layout to raster and vector formats."""
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=220)
    plt.close(fig)


def section(path: Path) -> None:
    """Show all signed displacement and velocity components on a physical tetrahedral cut."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator

    data = load(path)
    mesh = TetraMesh(data["macro_points"], data["macro_cells"])
    macro_segments = section_grid(mesh, height=0.37, refinement=1)["segments"]
    points, triangles, values = [], [], []
    offset = 0
    for macro in range(len(mesh.cells)):
        fine = mesh.submesh(macro, int(data["local_refinement"]))
        cut = section_grid(fine, height=0.37, refinement=6)
        if not len(cut["points"]):
            continue
        points.append(cut["points"])
        triangles.append(cut["cells"] + offset)
        values.append(evaluate(data, macro, fine, cut["parents"], cut["barycentric"]))
        offset += len(cut["points"])
    coordinates = np.concatenate(points)
    actual = np.concatenate(values)
    model = ElasticWave()
    spatial = model.spatial(coordinates)[0]
    u, v, _ = model.amplitudes(float(data["time"]))
    exact = np.column_stack((u * spatial, v * spatial))
    tri = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], np.concatenate(triangles))
    for vector, name, symbol in ((0, "displacement", "u"), (1, "velocity", "v")):
        fig = plt.figure(figsize=(12, 13), layout="constrained")
        grid = fig.add_gridspec(6, 3, height_ratios=[1, 0.065] * 3, hspace=0.1, wspace=0.08)
        for component in range(3):
            index = vector * 3 + component
            error = actual[:, index] - exact[:, index]
            limit = max(abs(actual[:, index]).max(), abs(exact[:, index]).max())
            defect = max(abs(error).max(), np.finfo(float).eps)
            for column, field in enumerate((exact[:, index], actual[:, index], error)):
                ax = fig.add_subplot(grid[2 * component, column])
                scale = limit if column < 2 else defect
                rendered = ax.tripcolor(
                    tri,
                    field,
                    shading="gouraud",
                    cmap="seismic",
                    vmin=-scale,
                    vmax=scale,
                    rasterized=True,
                )
                ax.add_collection(
                    LineCollection(macro_segments, colors="white", linewidths=0.85, alpha=0.65)
                )
                ax.add_collection(
                    LineCollection(macro_segments, colors=".2", linewidths=0.35, alpha=0.8)
                )
                label = ("Exact", "MHM P3/P1", "MHM − exact")[column]
                ax.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    aspect="equal",
                    xlabel="$x$",
                    ylabel="$y$",
                    title=rf"{label}: ${symbol}_{'xyz'[component]}$",
                )
                bar = fig.colorbar(
                    rendered,
                    cax=fig.add_subplot(grid[2 * component + 1, column]),
                    orientation="horizontal",
                )
                bar.locator = MaxNLocator(3)
                bar.update_ticks()
        fig.suptitle(
            f"{name.capitalize()} · t={float(data['time']):g} · z=0.37\n"
            f"{len(mesh.cells)} macrotetrahedra"
        )
        save(fig, f"{name}-components")


def profiles(path: Path) -> None:
    """Retain both traces at every local/macro crossing along x at y=.413,z=.37."""
    import matplotlib.pyplot as plt

    data = load(path)
    mesh = TetraMesh(data["macro_points"], data["macro_cells"])
    first, last = np.array([0, 0.413, 0.37]), np.array([1, 0.413, 0.37])
    intervals = line_intervals(mesh, first, last)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    points = first + np.linspace(0, 1, 1001)[:, None] * (last - first)
    model = ElasticWave()
    spatial = model.spatial(points)[0]
    u, v, _ = model.amplitudes(float(data["time"]))
    for vector, amplitude in enumerate((u, v)):
        for component in range(3):
            ax = axes[component, vector]
            ax.plot(points[:, 0], amplitude * spatial[:, component], "k--", lw=1.2, label="Exact")
            for crossing in np.unique(intervals[:, 1:]):
                ax.axvline(crossing, color=".75", ls=":", lw=0.6)
    for macro in intervals[:, 0].astype(int):
        fine = mesh.submesh(macro, int(data["local_refinement"]))
        for owner, lo, hi in line_intervals(fine, first, last):
            parameter = np.linspace(lo, hi, 25)
            points = first + parameter[:, None] * (last - first)
            vertices = fine.points[fine.cells[int(owner)]]
            xi = (points - vertices[0]) @ np.linalg.inv((vertices[1:] - vertices[0]).T).T
            bary = np.column_stack((1 - xi.sum(axis=1), xi))
            value = evaluate(data, macro, fine, np.full(len(points), int(owner)), bary)
            for vector in range(2):
                for component in range(3):
                    axes[component, vector].plot(
                        parameter, value[:, 3 * vector + component], color="#0072B2", lw=1
                    )
    for vector, symbol in enumerate(("u", "v")):
        for component in range(3):
            ax = axes[component, vector]
            ax.plot([], [], color="#0072B2", label="MHM P3/P1")
            ax.set(xlabel="$x$", ylabel=rf"${symbol}_{'xyz'[component]}$", xlim=(0, 1))
            ax.grid(alpha=0.15)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.suptitle(r"Displacement and velocity profiles · $y=0.413$, $z=0.37$")
    save(fig, "profiles")


def main() -> None:
    """Render completed spatial/time studies, retaining all measured errors and increments."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter

    spatial = [
        json.loads((DATA / f"n{n}-dt0.005-q{12 if n == 1 else 8}-s1.json").read_text())
        for n in (1, 2, 3, 4, 5, 6, 8)
    ]
    temporal = []
    reference = DATA / f"n2-dt{0.00009765625:g}-q8-s1.npz"
    previous_reference = DATA / f"n2-dt{0.0001953125:g}-q8-s1.npz"
    reference_increment = difference(previous_reference, reference)
    for dt in (0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125):
        path = DATA / f"n2-dt{dt:g}-q8-s1.npz"
        temporal.append({"dt": dt, "difference_to_time_reference": difference(path, reference)})
    keys = (
        "displacement_l2",
        "velocity_l2",
        "displacement_h1",
        "velocity_h1",
        "stress_broken_hdiv",
    )
    labels = (
        r"$\|u-u_h\|_0$",
        r"$\|v-v_h\|_0$",
        r"$\|u-u_h\|_{1,h}$",
        r"$\|v-v_h\|_{1,h}$",
        r"$\|\sigma-\sigma_h\|_{\mathrm{div},h}$",
    )
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), layout="constrained")
    for key, label in zip(keys, labels, strict=True):
        axes[0].loglog(
            [np.sqrt(3) / r["n"] for r in spatial],
            [r["norms"][key] for r in spatial],
            "o-",
            label=label,
        )
        axes[1].loglog(
            [r["dt"] for r in temporal],
            [r["difference_to_time_reference"][key] for r in temporal],
            "o-",
            label=label,
        )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.invert_xaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=3)
    h = np.sqrt(3) / np.array([r["n"] for r in spatial[-3:]])
    slope_handles = []
    for power, key, style in (
        (3, "displacement_l2", ":"),
        (2, "displacement_h1", "--"),
        (1, "stress_broken_hdiv", "-."),
    ):
        (handle,) = axes[0].loglog(
            h,
            0.7 * spatial[-1]["norms"][key] * (h / h[-1]) ** power,
            style,
            color="0.35",
            label=rf"$H^{power}$",
        )
        slope_handles.append(handle)
    axes[0].legend(handles=slope_handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=3)
    dt = np.array([r["dt"] for r in temporal[-3:]])
    (handle,) = axes[1].loglog(
        dt,
        0.6 * temporal[-1]["difference_to_time_reference"]["displacement_l2"] * (dt / dt[-1]) ** 2,
        ":",
        color="0.35",
        label=r"$\Delta t^2$",
    )
    axes[1].legend(handles=[handle], loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axes[0].set(
        xlabel="Macro diameter H",
        ylabel="Absolute physical error",
        title="Spatial refinement / exact solution",
    )
    axes[1].set(
        xlabel="Macro time step",
        ylabel="Absolute physical difference",
        title="Temporal refinement / same-space reference",
    )
    axes[0].xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1.0]))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    save(fig, "convergence")
    path = DATA / "n8-dt0.005-q8-s1.npz"
    section(path)
    profiles(path)
    record = {
        "spatial": spatial,
        "temporal": temporal,
        "temporal_reference": reference.name,
        "temporal_reference_refinement": {
            "previous_reference": previous_reference.name,
            "difference": reference_increment,
            "fraction_of_finest_comparison": {
                key: value / temporal[-1]["difference_to_time_reference"][key]
                for key, value in reference_increment.items()
            },
        },
        "field_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob("*.npz")
        },
    }
    (DATA / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    (OUTPUT / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
