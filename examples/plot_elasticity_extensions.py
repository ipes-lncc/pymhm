"""Replay general-tensor primal and rectangular mixed-elasticity numerical archives."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from examples.plot_mesh import draw_macro_mesh
from pymhm import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]


def save(figure, directory: Path, name: str) -> None:
    """Write vector and raster figures with identical layout."""
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def fields(record: dict, path: Path, directory: Path) -> None:
    """Show exact/numerical/difference fields with common scales and actual macro boundaries."""
    archive = path.parent / record["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Elasticity archive checksum mismatch")
    with np.load(archive) as data:
        triangulation = Triangulation(*data["points"].T, data["cells"])
        macro = (
            SimpleNamespace(points=data["macro_points"], faces=data["macro_faces"])
            if "macro_faces" in data
            else TriangleMesh(data["macro_points"], data["macro_cells"])
        )
        figure, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
        for row, (name, component, label) in enumerate(
            (("displacement", (0,), "Displacement uₓ"), ("stress", (0, 0), "Cauchy stress σₓₓ"))
        ):
            exact = data["exact_" + name][(slice(None), *component)]
            computed = data[name][(slice(None), *component)]
            difference = computed - exact
            maximum = max(np.abs(difference).max(), np.finfo(float).eps)
            for column, value in enumerate((exact, computed, difference)):
                options = (
                    dict(
                        cmap="viridis",
                        vmin=min(exact.min(), computed.min()),
                        vmax=max(exact.max(), computed.max()),
                    )
                    if column < 2
                    else dict(cmap="RdBu_r", vmin=-maximum, vmax=maximum)
                )
                artist = axes[row, column].tripcolor(
                    triangulation, value, shading="gouraud", rasterized=True, **options
                )
                draw_macro_mesh(axes[row, column], macro)
                axes[row, column].set(
                    aspect="equal",
                    xlabel="x",
                    ylabel="y",
                    title=f"{label}: {('exact', 'PyMHM', 'PyMHM − exact')[column]}",
                )
                figure.colorbar(artist, ax=axes[row, column], shrink=0.8, pad=0.02)
        last = record["rows"][-1]
        figure.suptitle(
            f"{path.stem}: L² displacement {last['displacement_l2']:.3e}; "
            f"stress {last['stress_l2']:.3e}"
        )
        save(figure, directory, path.stem + "-fields")


def primal(*, include_fields: bool = True) -> None:
    """Compare six primal spaces and optionally replay their archived spatial fields.

    Convergence curves use compact JSON records; spatial fields additionally
    require the original NPZ payloads. Face jumps remain indicators only.
    """
    data = ROOT / "examples/results/primal-elasticity"
    output = ROOT / "docs/figures/primal-elasticity"
    for variable in (False, True):
        name = "variable" if variable else "constant"
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        for path in sorted(data.glob(name + "-*.json")):
            record = json.loads(path.read_text())
            rows = record["rows"]
            degree = rows[0]["degree"]
            label = (
                "P1, local r=4"
                if degree == 1
                else "P2 + one mode, r=1"
                if degree == 2
                else "P3, r=1"
            )
            h = 1 / np.array([r["resolution"] for r in rows])
            for axis, key, title in zip(
                axes,
                ("displacement_l2", "stress_l2", "face_indicator"),
                (
                    "Displacement L² error",
                    "Raw stress L² error",
                    "Face indicator η (not full error)",
                ),
                strict=True,
            ):
                axis.loglog(h, [r[key] for r in rows], "o-", label=label)
                axis.set(xlabel="Macro spacing 1/n", ylabel=title)
                axis.grid(True, which="both", alpha=0.25)
            if include_fields:
                fields(record, path, output)
        axes[0].legend(fontsize=8)
        axes[2].set_title("Harder et al. (2016)")
        figure.suptitle(
            f"{name.capitalize()} anisotropic stiffness — P1 macro traces, "
            "homogeneous displacement boundary"
        )
        save(figure, output, name + "-convergence")


def mixed_rectangular() -> None:
    """Plot five-level convergence and finite/infinite-modulus behavior of RT families."""
    data = ROOT / "examples/results/elasticity-tensor-rt"
    output = ROOT / "docs/figures/elasticity-tensor-rt"
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    sweep, saxes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for path in sorted(data.glob("*.json")):
        record = json.loads(path.read_text())
        rows = record["rows"]
        label = f"RT{rows[0]['degree']}" + "+" * rows[0]["enrichment"]
        for axis, saxis, key, title in zip(
            axes,
            saxes,
            ("displacement_l2", "stress_l2", "rotation_l2"),
            ("Displacement L² error", "Stress L² error", "Rotation L² error"),
            strict=True,
        ):
            axis.loglog(
                1 / np.array([r["resolution"] for r in rows]),
                [r[key] for r in rows],
                "o-",
                label=label,
            )
            axis.set(xlabel="Macro spacing 1/n", ylabel=title)
            saxis.semilogy(np.arange(5), [r[key] for r in record["sweep"]], "o-", label=label)
            saxis.set(
                xlabel="First Lamé modulus λ (μ=1)",
                ylabel=title,
                xticks=np.arange(5),
                xticklabels=["1", "10²", "10⁴", "10⁸", "∞"],
            )
        fields(record, path, output)
    for axis in (*axes, *saxes):
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend()
    saxes[0].legend()
    figure.suptitle("Rectangular mixed elasticity — Q_s displacement and total-degree P_s rotation")
    sweep.suptitle("Fixed four-rectangle mesh — bounded-force polynomial solution")
    save(figure, output, "convergence")
    save(sweep, output, "incompressible-sweep")


def main() -> None:
    """Render archived results without solving or importing any reference-code runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--convergence-only", action="store_true", help="Render primal curves from compact records"
    )
    args = parser.parse_args()
    primal(include_fields=not args.convergence_only)
    if not args.convergence_only:
        mixed_rectangular()


if __name__ == "__main__":
    main()
