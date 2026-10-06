"""Render fields, estimator components and adaptive histories of Araya et al. (2021)."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
from plot_mesh import draw_macro_mesh
from plot_style import set_refinement_ticks

from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/stokes-adaptive"
OUTPUT = ROOT / "docs/figures/stokes-adaptive"


def save(fig: plt.Figure, name: str) -> None:
    """Save publication raster and compact vector figures."""
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def fields(record: dict, name: str) -> None:
    """Keep exact/numerical limits identical and preserve independent macro traces."""
    path = DATA / record["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Stokes archive differs from its recorded digest")
    with np.load(path) as values:
        points, cells = values["points"], values["cells"]
        mesh = TriangleMesh(values["macro_points"], values["macro_cells"])
        triangulation = Triangulation(*points.T, cells)
        actual = [np.linalg.norm(values["velocity"], axis=1), values["pressure"]]
        if "exact_velocity" not in values:
            return
        exact = [np.linalg.norm(values["exact_velocity"], axis=1), values["exact_pressure"]]
        errors = [
            np.linalg.norm(values["velocity"] - values["exact_velocity"], axis=1),
            abs(values["pressure"] - values["exact_pressure"]),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for row, title in enumerate(("Velocity magnitude", "Pressure")):
            lower = min(actual[row].min(), exact[row].min())
            upper = max(actual[row].max(), exact[row].max())
            for col, array in enumerate((exact[row], actual[row], errors[row])):
                ax = axes[row, col]
                artist = ax.tripcolor(
                    triangulation,
                    array,
                    shading="gouraud",
                    rasterized=True,
                    cmap="viridis" if col < 2 else "magma",
                    vmin=lower if col < 2 else 0,
                    vmax=upper if col < 2 else errors[row].max(),
                )
                draw_macro_mesh(ax, mesh).set_linewidth(0.3 if len(mesh.cells) > 256 else 0.65)
                ax.set(
                    aspect="equal",
                    xlabel="x",
                    ylabel="y",
                    title=(f"{title}: " + ("exact", "PyMHM")[col])
                    if col < 2
                    else ("Velocity vector error norm" if row == 0 else "Absolute pressure error"),
                )
                fig.colorbar(artist, ax=ax, shrink=0.8, pad=0.025)
        final = record["rows"][-1]
        fig.suptitle(
            f"Stokes · ν={record['viscosity']:g} · P3/P3 / trace P{record['trace_degree']}"
            f" · {record['strategy']} refinement\n"
            f"Physical L² errors: velocity {final['velocity_l2']:.3e}; "
            f"pressure {final['pressure_l2']:.3e}"
        )
        save(fig, name + "-fields")


def history(record: dict, name: str) -> None:
    """Show estimator components separately from integrated errors and changing mesh sizes."""
    rows = record["rows"]
    x = np.arange(len(rows))
    ticks = x if len(x) <= 12 else np.unique(np.r_[x[:: int(np.ceil(len(x) / 10))], x[-1]])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for key, label in (
        ("eta1", "η₁ macro traces"),
        ("eta2", "η₂ local residual"),
        ("total", "η₁+η₂"),
    ):
        axes[0].semilogy(x, [row[key] for row in rows], "o-", label=label)
    if "mixed_error" in rows[0]:
        for key, label in (
            ("velocity_l2", "Velocity L²"),
            ("pressure_l2", "Pressure L²"),
            ("mixed_error", "Mixed norm"),
        ):
            axes[1].semilogy(x, [row[key] for row in rows], "o-", label=label)
        axes[1].set_title("Analytical physical errors")
    else:
        axes[1].semilogy(x, [row["divergence_l2"] for row in rows], "o-", label="Divergence L²")
        axes[1].set_title("Diagnostic: no exact solution")
    for key, label in (
        ("macro_cells", "Macro triangles"),
        ("fine_cells", "Fine triangles"),
        ("trace_dofs", "Trace DOFs"),
    ):
        axes[2].semilogy(x, [row[key] for row in rows], "o-", label=label)
    axes[0].set_title("Araya et al. (2021)\nTwo-level estimator")
    axes[2].set_title("Actual approximation sizes")
    for ax in axes:
        set_refinement_ticks(ax, ticks)
        ax.set_xlabel("Solved refinement state")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(
        f"{record['case'].replace('L14', 'Araya et al. (2021)')} · {record['strategy']} · "
        f"ν={record['viscosity']:g} · γ={record['drag']:g}"
    )
    save(fig, name + "-history")


def publication() -> None:
    """Compare every printed component without replacing the inconsistent Ei column."""
    published = json.loads((DATA / "published-tables.json").read_text())["rows"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, nu in enumerate((1, 0.01)):
        for degree in (0, 1, 2):
            name = f"polynomial-uniform-nu{nu:g}-g0-l{degree}"
            path = DATA / (name + ".json")
            if not path.exists():
                raise FileNotFoundError("All six viscosity/trace-degree series are required")
            records = json.loads(path.read_text())["rows"]
            h = np.array([2 / np.sqrt(v["macro_cells"]) for v in records])
            paper = sorted(
                [v for v in published if v["viscosity"] == nu and v["trace_degree"] == degree],
                key=lambda v: -v["H"],
            )
            for col, (key, pubkey, title) in enumerate(
                (
                    ("mixed_error", "error", "Mixed error norm"),
                    ("eta1", "eta1", "η₁ macro traces"),
                    ("eta2", "eta2", "η₂ local residual"),
                )
            ):
                ax = axes[row, col]
                ax.loglog(
                    h, [v[key] for v in records], "o-", color=f"C{degree}", label=f"Trace P{degree}"
                )
                ax.loglog(
                    [v["H"] for v in paper], [v[pubkey] for v in paper], "x--", color=f"C{degree}"
                )
                set_refinement_ticks(ax, h)
                ax.set(xlabel="Macro diameter H", title=f"ν={nu:g} · {title}")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=9)
    fig.suptitle(
        "Solid: PyMHM, declared solenoidal data and crisscross grid\n"
        "Dashed: Araya et al. (2021), Tables 1 and 3, printed values\n"
        "Identical historical inputs are not established"
    )
    save(fig, "published-components")


def cavity_fields(record: dict, name: str) -> None:
    """Compare the regularized cavity against the independently refined classical baseline."""
    path = DATA / record["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("cavity comparison archive digest mismatch")
    with np.load(path) as arrays:
        triangle = Triangulation(*arrays["points"].T, arrays["cells"])
        mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
        panels = [
            (
                np.linalg.norm(arrays["reference_velocity"], axis=1),
                np.linalg.norm(arrays["velocity"], axis=1),
                np.linalg.norm(arrays["velocity"] - arrays["reference_velocity"], axis=1),
            ),
            (
                arrays["reference_pressure"],
                arrays["pressure"],
                abs(arrays["pressure"] - arrays["reference_pressure"]),
            ),
        ]
        fine_geometry = (
            (arrays["fine_points"], arrays["fine_cells"]) if "fine_points" in arrays else None
        )
        profile_arrays = {key: arrays[key] for key in arrays.files if key.startswith("profile_")}
        cutout = record.get("corner_cutout", 0.0)
        retained = ~(
            (arrays["points"][:, 1] > 1 - cutout)
            & ((arrays["points"][:, 0] < cutout) | (arrays["points"][:, 0] > 1 - cutout))
        )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for row, values in enumerate(panels):
        selected = retained if row and cutout else np.ones_like(retained)
        lower = min(values[0][selected].min(), values[1][selected].min())
        upper = max(values[0][selected].max(), values[1][selected].max())
        title = "Velocity magnitude" if row == 0 else "Pressure"
        for column, value in enumerate(values):
            ax = axes[row, column]
            artist = ax.tripcolor(
                triangle,
                value,
                shading="gouraud",
                rasterized=True,
                vmin=lower if column < 2 else 0,
                vmax=upper if column < 2 else value[selected].max(),
                cmap="viridis" if column < 2 else "magma",
            )
            draw_macro_mesh(ax, mesh)
            if record.get("corner_cutout"):
                width = record["corner_cutout"]
                for x in (0, 1 - width):
                    ax.add_patch(
                        Rectangle(
                            (x, 1 - width),
                            width,
                            width,
                            fill=bool(row),
                            facecolor="white",
                            edgecolor="0.5" if row else "white",
                            linestyle="--",
                            linewidth=1.0,
                        )
                    )
            ax.set(
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=(f"{title}: " + ("DOLFINx P2/P1", "PyMHM")[column])
                if column < 2
                else "Velocity vector difference norm"
                if row == 0
                else "Absolute pressure difference",
            )
            fig.colorbar(artist, ax=ax, shrink=0.8, pad=0.025)
    error = record["norms"]["28"]
    scope = "Interior" if record.get("corner_cutout") else "Physical"
    lid = record.get("lid", "regularized").capitalize()
    fig.suptitle(
        f"{lid} cavity · γ={record['drag']:g} · {record['strategy']} adaptivity\n"
        f"{scope} L² differences: velocity {100 * error['velocity_relative']:.3g}%; "
        f"pressure {100 * error['pressure_relative']:.3g}%"
    )
    save(fig, name + "-fields")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    upper_values = panels[0]
    upper_bottom = max(0.875, 1 - 4 / np.sqrt(record["drag"])) if record["drag"] else 0.875
    for column, value in enumerate(upper_values):
        ax = axes[column]
        artist = ax.tripcolor(
            triangle,
            value,
            shading="gouraud",
            rasterized=True,
            cmap="viridis" if column < 2 else "magma",
            vmin=0,
            vmax=max(upper_values[0].max(), upper_values[1].max()) if column < 2 else value.max(),
        )
        draw_macro_mesh(ax, mesh).set_linewidth(0.35)
        ax.set(
            xlim=(0, 1),
            ylim=(upper_bottom, 1),
            xlabel="x",
            ylabel="y",
            title=("DOLFINx P2/P1", "PyMHM", "Velocity vector difference norm")[column],
        )
        fig.colorbar(
            artist,
            ax=ax,
            location="bottom",
            pad=0.09,
            shrink=0.9,
            label="Velocity magnitude" if column < 2 else r"$|u_{\mathrm{MHM}}-u_{\mathrm{ref}}|$",
        )
    fig.suptitle(f"{lid} cavity · γ={record['drag']:g} · upper boundary layer (vertical zoom)")
    save(fig, name + "-upper-layer")
    if profile_arrays:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
        for row, (label, coordinate, title) in enumerate(
            (
                ("upper", 0, "Horizontal profile y=0.99"),
                ("vertical", 1, "Vertical profile x=0.5"),
            )
        ):
            x = profile_arrays[f"profile_{label}_points"][:, :, coordinate]
            crossings = profile_arrays[f"profile_{label}_macro_intersections"]
            for component, ax in enumerate(axes[row]):
                for reference, color, display_label in (
                    (True, "C0", "DOLFINx P2/P1"),
                    (False, "C1", "PyMHM one-sided"),
                ):
                    key = f"profile_{label}_{'reference_' if reference else ''}velocity"
                    values = profile_arrays[key][:, :, component]
                    for index, (segment, value) in enumerate(zip(x, values, strict=True)):
                        ax.plot(
                            segment,
                            value,
                            color=color,
                            linewidth=1.25,
                            label=display_label if index == 0 else None,
                        )
                ax.plot(
                    crossings,
                    np.full(len(crossings), 0.025),
                    "|",
                    color="0.45",
                    markersize=5,
                    markeredgewidth=0.6,
                    transform=ax.get_xaxis_transform(),
                )
                ax.set(
                    xlabel="x" if coordinate == 0 else "y",
                    ylabel=f"u{component + 1}",
                    title=f"{title} · component {component + 1}",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=9)
        fig.suptitle(
            f"{lid} cavity · γ={record['drag']:g} · {record['strategy']} adaptivity\n"
            "Gray ticks mark actual macroface intersections"
        )
        save(fig, name + "-profiles")
    if record.get("history"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        rows = record["history"]
        for ax, xkey, xlabel in zip(
            axes,
            ("trace_and_coarse_dofs", "fine_cells"),
            ("Trace and coarse DOFs", "Fine triangles"),
            strict=True,
        ):
            for key, label in (
                ("velocity_relative", "Velocity L²"),
                ("pressure_relative", "Pressure L²"),
                ("velocity_energy_relative", "Velocity energy"),
            ):
                ax.loglog(
                    [r[xkey] for r in rows],
                    [100 * r["norms"]["20"][key] for r in rows],
                    "o-",
                    label=label,
                )
            ax.set(xlabel=xlabel, ylabel="Relative difference (%)")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(
            f"{lid} cavity · γ={record['drag']:g} · {record['strategy']}\n"
            f"{scope} physical differences from refined DOLFINx P2/P1"
        )
        save(fig, name + "-physical-convergence")
    if fine_geometry is not None:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
        points, cells = fine_geometry
        edges = points[cells[:, [[0, 1], [1, 2], [2, 0]]]].reshape(-1, 2, 2)
        for ax, bounds, title in zip(
            axes,
            ((0, 1, 0, 1), (0, 0.125, 0.875, 1), (0.875, 1, 0.875, 1)),
            ("Entire cavity", "Upper left corner", "Upper right corner"),
            strict=True,
        ):
            ax.add_collection(LineCollection(edges, colors="0.55", linewidths=0.2))
            draw_macro_mesh(ax, mesh)
            ax.set(
                xlim=bounds[:2],
                ylim=bounds[2:],
                aspect="equal",
                xlabel="x",
                ylabel="y",
                title=title,
            )
        fig.suptitle(
            f"Constant-lid cavity · γ={record['drag']:g} · {record['strategy']}\n"
            "Actual fine edges (gray) and macro boundaries (black/white)"
        )
        save(fig, name + "-meshes")


def classical_refinement() -> None:
    """Show the reference's own mesh sensitivity separately from the MHM differences."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, drag in zip(axes, (0, 10000), strict=True):
        records = [
            json.loads(path.read_text())
            for path in DATA.glob(f"cavity-classical-gamma{drag}-n*.json")
        ]
        records = sorted(
            (row for row in records if row.get("difference_from_previous")),
            key=lambda row: row["n"],
        )
        levels = [row["n"] for row in records]
        for key, label in (
            ("velocity_relative", "Velocity L²"),
            ("pressure_relative", "Pressure L²"),
            ("velocity_h1_relative", "Velocity H¹ seminorm"),
        ):
            ax.loglog(
                levels,
                [100 * row["difference_from_previous"][key] for row in records],
                "o-",
                label=label,
            )
        set_refinement_ticks(ax, levels)
        ax.set(
            xlabel="Rectangles per coordinate direction",
            ylabel="Successive-reference difference [%]",
            title=f"Regularized cavity · γ={drag:g}",
        )
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("DOLFINx conforming P2/P1 · PETSc/MUMPS · numerical reference refinement")
    save(fig, "cavity-reference-convergence")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for ax, drag in zip(axes, (0, 10000), strict=True):
        rows = sorted(
            (
                json.loads(path.read_text())
                for path in DATA.glob(f"cavity-constant-classical-gamma{drag}-n*.json")
            ),
            key=lambda row: row["n"],
        )
        rows = [row for row in rows if row.get("interior_difference_from_previous")]
        levels = [row["n"] for row in rows]
        for scope, key, label in (
            ("difference_from_previous", "velocity_relative", "Global velocity L²"),
            ("interior_difference_from_previous", "velocity_relative", "Interior velocity L²"),
            ("interior_difference_from_previous", "pressure_relative", "Interior pressure L²"),
            ("interior_difference_from_previous", "velocity_h1_relative", "Interior velocity H¹"),
        ):
            ax.loglog(levels, [100 * row[scope][key] for row in rows], "o-", label=label)
        set_refinement_ticks(ax, levels)
        ax.set(
            xlabel="Rectangles per coordinate direction",
            ylabel="Successive difference (%)",
            title=f"Constant lid · γ={drag:g}",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(
        "DOLFINx P2/P1 reference refinement · fixed upper-corner cutouts of side 1/32\n"
        "Global pressure and gradient norms are excluded for discontinuous lid data"
    )
    save(fig, "cavity-constant-reference-convergence")


def main() -> None:
    """Replay complete native numerical records without acquiring new solutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--histories-only", action="store_true", help="Render estimator curves from compact records"
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(DATA.glob("*.json")):
        record = json.loads(path.read_text())
        if (
            not args.histories_only
            and isinstance(record, dict)
            and "norms" in record
            and "strategy" in record
        ):
            cavity_fields(record, path.stem)
        if not isinstance(record, dict) or "archive" not in record or "rows" not in record:
            continue
        history(record, path.stem)
        if not args.histories_only and (
            record["strategy"] != "uniform" or record["trace_degree"] == 1
        ):
            fields(record, path.stem)
    publication()
    if not args.histories_only:
        classical_refinement()


if __name__ == "__main__":
    main()
