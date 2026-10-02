"""Published hexagonal MHM boundary layer against exact and classical P2 fields."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from plot_mesh import draw_macro_mesh, macro_profile_breaks, mark_macro_interfaces
from plot_style import set_refinement_ticks
from polygon_meshes import polygon_partition
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_transport_polygons
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import nodal_space, reference_basis, scalar_operators, tabulate
from pymhm.solvers import solve_linear

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 0.01


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the published layer without exponentially growing intermediate values."""
    x = points[:, 0]
    return x - (np.exp((x - 1) / EPSILON) - np.exp(-1 / EPSILON)) / (-np.expm1(-1 / EPSILON))


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Differentiate the exact layer analytically."""
    return np.column_stack(
        (
            1 - np.exp((points[:, 0] - 1) / EPSILON) / (-EPSILON * np.expm1(-1 / EPSILON)),
            np.zeros(len(points)),
        )
    )


def classical(n: int) -> tuple[TriangleMesh, np.ndarray]:
    """Solve standard continuous P2 Galerkin with strong vertical Dirichlet data."""
    mesh = TriangleMesh.unit_square(n)
    _, nodes = nodal_space(mesh, 2)
    matrix, _, rhs = scalar_operators(mesh, 2, diffusion=EPSILON, source=1.0, advection=(1.0, 0.0))
    free = np.flatnonzero((nodes[:, 0] > 0) & (nodes[:, 0] < 1))
    values = np.zeros(len(nodes))
    values[free] = solve_linear(matrix[free][:, free], rhs[free])
    return mesh, values


def errors(meshes: tuple, fields: tuple, degree: int, order: int = 20) -> dict:
    """Integrate L2 and diffusion-weighted broken gradient errors, without nodal sampling."""
    bary, weights = triangle_quadrature(order)
    l2, energy = 0.0, 0.0
    for fine, values in zip(meshes, fields, strict=True):
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        dofs, _, basis, grad, _ = tabulate(fine, degree, bary)
        delta = values[dofs] @ basis.T - exact(points.reshape(-1, 2)).reshape(points.shape[:2])
        derivative = np.einsum("ti,tqia->tqa", values[dofs], grad)
        delta_grad = derivative - exact_gradient(points.reshape(-1, 2)).reshape(points.shape)
        l2 += float(fine.areas @ (delta**2 @ weights))
        energy += EPSILON * float(fine.areas @ (np.sum(delta_grad**2, axis=-1) @ weights))
    return dict(
        error_l2=float(np.sqrt(l2)),
        error_energy=float(np.sqrt(energy)),
        error_v=float(np.sqrt(energy / EPSILON + 0.5 * l2)),
    )


def profile(
    fine: TriangleMesh, values: np.ndarray, degree: int, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a profile only inside this local mesh, preserving independent macro traces."""
    owner = mtri.Triangulation(*fine.points.T, fine.cells).get_trifinder()(*points.T)
    selected = np.flatnonzero(owner >= 0)
    coordinates, result = [], []
    dofs = nodal_space(fine, degree)[0]
    for cell in np.unique(owner[selected]):
        indices = selected[owner[selected] == cell]
        corners = fine.points[fine.cells[cell]]
        local = np.linalg.solve((corners[1:] - corners[0]).T, (points[indices] - corners[0]).T).T
        bary = np.column_stack((1 - local.sum(axis=1), local))
        basis = reference_basis(degree, bary)[0]
        coordinates.extend(points[indices, 0])
        result.extend(basis @ values[dofs[cell]])
    indices = np.argsort(coordinates)
    return np.asarray(coordinates)[indices], np.asarray(result)[indices]


def solve_case(n: int, refinement: int, segments: int = 1) -> object:
    """Construct the published P3/P1 operator on an explicit hexagonal partition."""
    mesh = polygon_partition(n, "hexagon")
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, segments) for _ in mesh.faces))
    horizontal = [int(f) for f in mesh.boundary_faces if abs(mesh.normals[f, 1]) > 0.99]
    return solve_transport_polygons(
        mesh,
        skeleton=skeleton,
        degree=3,
        local_refinement=refinement,
        diffusion=EPSILON,
        velocity=(1.0, 0.0),
        source=1.0,
        quadrature_order=8,
        diffusive_flux=dict.fromkeys(horizontal, 0.0),
        dirichlet_enforcement="strong",
        coarse_space="kernel",
    )


def main() -> None:
    """Acquire five macro and five local resolutions and render matched physical profiles."""
    target = ROOT / "examples/results/rad-layer.json"
    rows = json.loads(target.read_text())["records"] if target.exists() else []
    selected = None
    with threadpool_limits(1):
        settings = [("macro", n, 2, 1) for n in (2, 4, 8, 16, 32)]
        settings += [("local", 8, r, 1) for r in (1, 2, 4, 8, 16)]
        settings += [("skeleton", 4, s, s) for s in (1, 2, 4, 8, 16)]
        for study, n, refinement, segments in settings:
            old = next(
                (
                    row
                    for row in rows
                    if (row["study"], row["n"], row["local_refinement"], row.get("segments", 1))
                    == (study, n, refinement, segments)
                ),
                None,
            )
            if old is not None:
                if study == "local" and refinement == 8:
                    selected = solve_case(n, refinement)
                continue
            solution = solve_case(n, refinement, segments)
            skeleton = solution.skeleton
            mesh = skeleton.mesh
            norm = errors(solution.local_meshes, solution.values, 3)
            row = dict(
                study=study,
                n=n,
                local_refinement=refinement,
                segments=segments,
                macro_diameter=max(
                    float(
                        np.max(
                            np.linalg.norm(
                                mesh.points[cell, None] - mesh.points[cell][None], axis=2
                            )
                        )
                    )
                    for cell in mesh.cells
                ),
                macros=len(mesh.cells),
                trace_dofs=skeleton.size,
                global_dofs=len(solution.hybrid.trace)
                + sum(v.size for v in solution.hybrid.coarse),
                free_global_dofs=len(solution.hybrid.trace)
                + sum(v.size for v in solution.hybrid.coarse)
                - sum(len(skeleton.dofs(int(face))) for face in mesh.boundary_faces),
                local_dofs=sum(v.size for v in solution.values),
                residual=solution.hybrid.residual,
                **norm,
            )
            if study == "macro":
                fine, values = classical(n)
                row.update(
                    classical_dofs=len(values),
                    free_classical_dofs=4 * n * n - 1,
                    classical=errors((fine,), (values,), 2),
                )
            if study == "local" and refinement == 8:
                selected = solution
            if (study == "macro" and n == 2) or (
                study in ("local", "skeleton") and refinement == 16
            ):
                check = errors(solution.local_meshes, solution.values, 3, 28)
                row["quadrature_relative_change"] = max(abs(check[k] / norm[k] - 1) for k in norm)
            rows.append(row)
            print(row, flush=True)
            target.write_text(json.dumps(dict(records=rows), indent=2) + "\n")
    target.write_text(
        json.dumps(
            dict(
                reference="10.1016/j.cma.2024.117089, section 5.2.2, Figures 5–6",
                epsilon=EPSILON,
                velocity=[1.0, 0.0],
                source=1.0,
                local_degree=3,
                trace_degree=1,
                geometry="clipped staggered Voronoi hexagons; original explicit connectivity",
                records=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    render(rows, selected)


def render(rows: list[dict], selected: object) -> None:
    """Plot exact comparisons, convergence and one-sided macro profiles in separate panels."""
    destination = ROOT / "docs/figures/rad-layer"
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for axis, key, title in zip(
        axes,
        ("error_l2", "error_energy"),
        ("Scalar error", "Diffusion-weighted gradient error"),
        strict=True,
    ):
        values = [r for r in rows if r["study"] == "macro"]
        ns = [r["n"] for r in values]
        axis.loglog(ns, [r[key] for r in values], "o-", label="MHM P3 / P1; r=2")
        axis.loglog(ns, [r["classical"][key] for r in values], "s--", label="Classical P2")
        set_refinement_ticks(axis, ns)
        axis.set(xlabel="Macro resolution n", ylabel="Integrated error", title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(destination / "convergence.png", dpi=220)
    fig.savefig(destination / "convergence.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for study, label, marker in (
        ("macro", "MHM: macro refinement", "o"),
        ("skeleton", "MHM: skeleton refinement", "s"),
    ):
        records = [row for row in rows if row["study"] == study]
        axes[0].loglog(
            [row["free_global_dofs"] for row in records],
            [row["error_v"] for row in records],
            marker + "-",
            label=label,
        )
    records = [row for row in rows if row["study"] == "macro"]
    axes[0].loglog(
        [row["free_classical_dofs"] for row in records],
        [row["classical"]["error_v"] for row in records],
        "^--",
        label="Classical P2",
    )
    axes[0].set(
        xlabel="Global unknowns", ylabel=r"$\|u-u_h\|_V$", title="The norm of published Figure 6"
    )
    records = [row for row in rows if row["study"] == "local"]
    axes[1].semilogx(
        [row["local_refinement"] for row in records],
        [row["error_v"] for row in records],
        "o-",
        label=f"MHM: fixed {records[0]['free_global_dofs']} global unknowns",
    )
    set_refinement_ticks(axes[1], [row["local_refinement"] for row in records])
    axes[1].set(
        xlabel="Local subdivisions", ylabel=r"$\|u-u_h\|_V$", title="Independent local refinement"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(destination / "published-norm.png", dpi=220)
    fig.savefig(destination / "published-norm.svg")
    plt.close(fig)
    field_panels(selected, destination)


def sampled_field(meshes: tuple, fields: tuple, degree: int) -> tuple:
    """Tabulate piecewise polynomials on duplicated triangles without macro averaging."""
    sample = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, 3)
    bary = np.column_stack((1 - sample.points.sum(axis=1), sample.points))
    basis = reference_basis(degree, bary)[0]
    coordinates, values, cells, offset = [], [], [], 0
    for mesh, field in zip(meshes, fields, strict=True):
        coordinates.append(np.einsum("qi,tia->tqa", bary, mesh.points[mesh.cells]).reshape(-1, 2))
        values.append((field[nodal_space(mesh, degree)[0]] @ basis.T).ravel())
        cells.append(
            (
                sample.cells[None] + offset + len(bary) * np.arange(len(mesh.cells))[:, None, None]
            ).reshape(-1, 3)
        )
        offset += len(mesh.cells) * len(bary)
    xy = np.vstack(coordinates)
    return mtri.Triangulation(*xy.T, triangles=np.vstack(cells)), np.concatenate(values)


def field_panels(selected: object, destination: Path) -> None:
    """Compare classical, MHM and exact fields with shared scales and macro outlines."""
    fine, values = classical(8)
    classical_mesh, classical_values = sampled_field((fine,), (values,), 2)
    numerical_mesh, numerical_values = sampled_field(selected.local_meshes, selected.values, 3)
    xy = np.column_stack((numerical_mesh.x, numerical_mesh.y))
    exact_values = exact(xy)
    lower = min(float(classical_values.min()), float(numerical_values.min()), 0.0)
    upper = max(
        float(classical_values.max()), float(numerical_values.max()), float(exact_values.max())
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout="constrained")
    for axis, mesh, values, title in zip(
        axes,
        (classical_mesh, numerical_mesh, numerical_mesh),
        (classical_values, numerical_values, exact_values),
        ("Classical continuous P2; n=8", "MHM P3/P1; n=8, r=8", "Exact"),
        strict=True,
    ):
        artist = axis.tripcolor(
            mesh, values, shading="gouraud", cmap="viridis", vmin=lower, vmax=upper, rasterized=True
        )
        draw_macro_mesh(axis, selected.skeleton.mesh)
        axis.set(xlabel="x", ylabel="y", title=title, aspect="equal")
    fig.colorbar(artist, ax=list(axes), label="Scalar field", shrink=0.85, pad=0.025)
    fig.savefig(destination / "fields.png", dpi=220)
    fig.savefig(destination / "fields.svg")
    plt.close(fig)
    elevation_panels(
        selected.skeleton.mesh,
        classical_mesh,
        classical_values,
        numerical_mesh,
        numerical_values,
        (lower, upper),
        destination,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    xy = np.array([(x, y) for y in np.linspace(0, 1, 200) for x in np.linspace(0, 1, 300)])
    mesh = selected.skeleton.mesh
    artist = axes[0].tripcolor(*xy.T, exact(xy), shading="gouraud", cmap="viridis", rasterized=True)
    draw_macro_mesh(axes[0], mesh)
    axes[0].set(
        xlabel="x", ylabel="y", title="Exact layer and actual macro partition", aspect="equal"
    )
    fig.colorbar(artist, ax=axes[0], label="Scalar field", shrink=0.85)
    y = 0.4375
    breaks = macro_profile_breaks(mesh, (0.0, y), (1.0, y))
    x = np.unique(np.r_[np.linspace(0.0, 1.0, 2500), breaks])
    points = np.column_stack((x, np.full(len(x), y)))
    axes[1].plot(x, exact(points), color="black", linewidth=1.8, label="Exact")
    fine, values = classical(8)
    xs, field = profile(fine, values, 2, points)
    axes[1].plot(xs, field, "--", color="#1976b9", label="Classical P2; n=8")
    first = True
    for fine, values in zip(selected.local_meshes, selected.values, strict=True):
        xs, field = profile(fine, values, 3, points)
        if len(xs):
            axes[1].plot(
                xs,
                field,
                color="#c3422e",
                linewidth=1.1,
                label="MHM P3/P1; n=8, r=8" if first else None,
            )
            first = False
    mark_macro_interfaces(axes[1], breaks)
    axes[1].set(xlabel="x", ylabel="Scalar field", title="Profile at y=0.4375")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="upper left", fontsize=9)
    fig.savefig(destination / "profile.png", dpi=220)
    fig.savefig(destination / "profile.svg")
    plt.close(fig)


def elevation_panels(
    macro: object,
    classical_mesh: object,
    classical_values: np.ndarray,
    numerical_mesh: object,
    numerical_values: np.ndarray,
    limits: tuple[float, float],
    destination: Path,
) -> None:
    """Render the published elevation view with common height and color scales."""
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    fig = plt.figure(figsize=(10.5, 5.2), layout="constrained")
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    floor = limits[0] - 0.06 * (limits[1] - limits[0])
    edges = np.concatenate(
        (macro.points[macro.faces], np.full((len(macro.faces), 2, 1), floor)), axis=2
    )
    for axis, mesh, values, title in zip(
        axes,
        (classical_mesh, numerical_mesh),
        (classical_values, numerical_values),
        ("Classical continuous P2", "MHM P3/P1"),
        strict=True,
    ):
        artist = axis.plot_trisurf(
            mesh,
            np.asarray(values, dtype=float),
            cmap="viridis",
            norm=Normalize(*limits),
            linewidth=0,
            antialiased=False,
            rasterized=True,
        )
        axis.add_collection3d(Line3DCollection(edges, colors="#444444", linewidths=0.45))
        axis.set(
            xlabel="x",
            ylabel="y",
            zlabel="Scalar field",
            title=title,
            xlim=(0, 1),
            ylim=(0, 1),
            zlim=(floor, limits[1]),
        )
        axis.view_init(elev=32, azim=-130)
        axis.set_box_aspect((1, 1, 0.8))
        axis.tick_params(labelsize=8)
    fig.colorbar(
        artist,
        ax=axes,
        location="bottom",
        fraction=0.045,
        pad=0.045,
        label="Scalar field; macro edges projected onto the base plane",
    )
    fig.savefig(destination / "elevation.png", dpi=220)
    fig.savefig(destination / "elevation.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
