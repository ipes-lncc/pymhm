"""Replay broken Q2 nanoguide fields with common-time, componentwise L2 comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.io.workspace import case_workspace, local_resource, read_resource_bytes

ROOT = case_workspace()
DATA = ROOT / "examples/results/maxwell-nanoguide"


def fields(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read every discontinuous Q2 coefficient, checking the archived physical times."""
    with np.load(local_resource(path)) as data:
        if int(data["degree"]) != 2 or not np.array_equal(data["bounds"], [0, 10, 0, 10]):
            raise ValueError("expected the Q2 nanoguide field contract")
        value = np.concatenate((data["electric"][..., None], data["magnetic"]), axis=-1)
        metadata = {
            "electric_time": float(data["electric_time"]),
            "magnetic_time": float(data["magnetic_time"]),
            "resolution": value.shape[0],
            "sha256": hashlib.sha256(read_resource_bytes(path)).hexdigest(),
            "file": path.name,
        }
    return value, metadata


def sample(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate actual cell polynomials without averaging independent incident traces."""
    n = len(values)
    coordinate = np.asarray(points) * n / 10
    cell = np.clip(np.floor(coordinate).astype(int), 0, n - 1)
    basis = qk_basis(2, coordinate - cell)[0]
    return np.einsum("qi,qia->qa", basis, values[cell[:, 1], cell[:, 0]])


def compare(first: Path, second: Path, order: int = 3) -> dict[str, Any]:
    """Integrate component errors on the common cell partition, with matched physical times."""
    a, ameta = fields(first)
    b, bmeta = fields(second)
    for key in ("electric_time", "magnetic_time"):
        if not np.isclose(ameta[key], bmeta[key], atol=1e-12, rtol=0):
            raise ValueError(f"cannot compare fields at different {key}")
    n = int(np.lcm(len(a), len(b)))
    t, w = leggauss(order)
    t, w = (t + 1) / 2, w / 2
    squared, norm = np.zeros(3, dtype=np.longdouble), np.zeros(3, dtype=np.longdouble)
    h = 10 / n
    for row in range(0, n, 16):
        x, y = np.meshgrid(np.arange(n), np.arange(row, min(row + 16, n)))
        origins = h * np.column_stack((x.ravel(), y.ravel()))
        for i, tx in enumerate(t):
            for j, ty in enumerate(t):
                points = origins + h * np.array([tx, ty])
                actual, target = sample(a, points), sample(b, points)
                squared += (
                    h
                    * h
                    * w[i]
                    * w[j]
                    * np.sum((actual - target) ** 2, axis=0, dtype=np.longdouble)
                )
                norm += h * h * w[i] * w[j] * np.sum(target**2, axis=0, dtype=np.longdouble)
    return {
        "field": ameta,
        "reference": bmeta,
        "components": ["electric", "magnetic_x", "magnetic_y"],
        "absolute_l2": np.sqrt(squared).astype(float).tolist(),
        "reference_l2": np.sqrt(norm).astype(float).tolist(),
        "relative_l2": np.sqrt(squared / norm).astype(float).tolist(),
        "combined_relative_l2": float(np.sqrt(squared.sum() / norm.sum())),
        "integration_grid": n,
        "gauss_order": order,
    }


def plot(paths: list[Path], labels: list[str], output: Path) -> None:
    """Render signed components and errors with shared field scales and actual macro boundaries."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output.mkdir(parents=True, exist_ok=True)
    coordinates = (np.arange(640) + 0.5) * 10 / 640
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    maps = [sample(fields(path)[0], points).reshape(640, 640, 3) for path in paths]
    fig, axes = plt.subplots(3, len(paths), figsize=(4.5 * len(paths), 12), layout="constrained")
    components = (r"$E_z$", r"$H_x$", r"$H_y$")
    for component in range(3):
        maximum = max(float(abs(value[..., component]).max()) for value in maps)
        norm = Normalize(-maximum, maximum)
        for column, (value, label, path) in enumerate(zip(maps, labels, paths, strict=True)):
            ax = axes[component, column]
            rendered = ax.imshow(
                value[..., component],
                origin="lower",
                extent=(0, 10, 0, 10),
                cmap="seismic",
                norm=norm,
                interpolation="nearest",
            )
            with np.load(local_resource(path)) as data:
                if "macro_points" in data:
                    from matplotlib.collections import LineCollection

                    edges = data["macro_points"][data["macro_faces"]]
                    ax.add_collection(
                        LineCollection(edges, colors="0.15", linewidths=0.35, alpha=0.6)
                    )
                else:
                    # Classical fields have no MHM partition; the comparison macro
                    # grid is explicitly shown and named in the caption.
                    for p in np.linspace(0, 10, 17):
                        ax.axvline(p, color="0.15", lw=0.35, alpha=0.6)
                        ax.axhline(p, color="0.15", lw=0.35, alpha=0.6)
            ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="$x$", ylabel="$y$")
            ax.set_title(f"{label}\n{components[component]}", fontsize=12)
        fig.colorbar(
            rendered,
            ax=axes[component].tolist(),
            shrink=0.82,
            pad=0.02,
            label=components[component],
        )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"components.{suffix}", dpi=200)
    plt.close(fig)


def plot_errors(paths: list[Path], labels: list[str], reference: Path, output: Path) -> None:
    """Show signed component differences and discontinuous horizontal profiles."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    target, _ = fields(reference)
    coordinates = (np.arange(640) + 0.5) * 10 / 640
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    target_samples = sample(target, points)
    maps = [
        (sample(fields(path)[0], points) - target_samples).reshape(640, 640, 3) for path in paths
    ]
    fig = plt.figure(figsize=(4.5 * len(paths), 14), layout="constrained")
    grid = fig.add_gridspec(6, len(paths), height_ratios=[1, 0.055] * 3, hspace=0.12)
    components = (r"$E_z$", r"$H_x$", r"$H_y$")
    for component in range(3):
        mhm_maximum = max(float(abs(value[..., component]).max()) for value in maps[:2])
        for column, (value, label, path) in enumerate(zip(maps, labels, paths, strict=True)):
            ax = fig.add_subplot(grid[2 * component, column])
            maximum = mhm_maximum if column < 2 else float(abs(value[..., component]).max())
            rendered = ax.imshow(
                value[..., component],
                origin="lower",
                extent=(0, 10, 0, 10),
                cmap="seismic",
                norm=Normalize(-maximum, maximum),
                interpolation="nearest",
            )
            with np.load(local_resource(path)) as data:
                if "macro_points" in data:
                    ax.add_collection(
                        LineCollection(
                            data["macro_points"][data["macro_faces"]],
                            colors="0.15",
                            linewidths=0.35,
                            alpha=0.6,
                        )
                    )
                else:
                    for p in np.linspace(0, 10, 17):
                        ax.axvline(p, color="0.15", lw=0.35, alpha=0.6)
                        ax.axhline(p, color="0.15", lw=0.35, alpha=0.6)
            ax.set(xlim=(0, 10), ylim=(0, 10), xlabel="$x$", ylabel="$y$")
            ax.set_title(f"{label} − reference\n{components[component]}", fontsize=12)
            bar = fig.colorbar(
                rendered,
                cax=fig.add_subplot(grid[2 * component + 1, column]),
                orientation="horizontal",
            )
            bar.locator = MaxNLocator(3)
            bar.update_ticks()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"errors.{suffix}", dpi=200)
    plt.close(fig)

    colors = ("black", "#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for column, y in enumerate((4.873, 5.137)):
        for path, label, color in zip(
            [reference, *paths], ["Reference", *labels], colors, strict=True
        ):
            coefficients, _ = fields(path)
            n = len(coefficients)
            row = int(np.floor(y * n / 10))
            parameter = np.linspace(0, 1, 9)
            basis = qk_basis(2, np.column_stack((parameter, np.full(9, y * n / 10 - row))))[0]
            value = np.einsum("qi,tia->tqa", basis, coefficients[row])
            x = (np.arange(n)[:, None] + parameter) * 10 / n
            for component, ax in enumerate(axes[:, column]):
                ax.plot(
                    x.T,
                    value[..., component].T,
                    color=color,
                    lw=0.8,
                    ls="--" if path == reference else "-",
                )
                ax.plot([], [], color=color, label=label)
                ax.set(xlabel="$x$", ylabel=components[component], xlim=(0, 10))
                ax.grid(alpha=0.15)
        for ax in axes[:, column]:
            for crossing in np.linspace(0, 10, 17):
                ax.axvline(crossing, color=".7", lw=0.5, ls=":")
            ax.set_title(f"y = {y:g}")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside upper center", ncol=4)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"profiles.{suffix}", dpi=200)
    plt.close(fig)


def plot_material(output: Path) -> None:
    """Render the exact inclusion dimensions and each published macro partition."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    from examples.maxwell_nanoguide import NanoWaveguide

    coordinate = (np.arange(800) + 0.5) * 10 / 800
    xx, yy = np.meshgrid(coordinate, coordinate)
    values = NanoWaveguide().permittivity(np.column_stack((xx.ravel(), yy.ravel())))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    cmap = ListedColormap(["#f2f0eb", "#56B4E9", "#D55E00"])
    for ax, n in zip(axes, (16, 8), strict=True):
        rendered = ax.imshow(
            values.reshape(800, 800),
            extent=(0, 10, 0, 10),
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=BoundaryNorm([0.75, 1.25, 2.32, 3.96], 3),
        )
        for p in np.linspace(0, 10, n + 1):
            ax.axvline(p, color="0.15", lw=0.4, alpha=0.7)
            ax.axhline(p, color="0.15", lw=0.4, alpha=0.7)
        ax.set(
            xlabel="$x$", ylabel="$y$", xlim=(0, 10), ylim=(0, 10), title=f"{n} × {n} macroelements"
        )
    fig.colorbar(
        rendered, ax=axes.tolist(), label="Permittivity", ticks=[1, 1.5, 3.14], shrink=0.75
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"material.{suffix}", dpi=200)
    plt.close(fig)


def plot_recorded_results(comparison: Path, controls: Path, output: Path) -> dict[str, Any]:
    """Plot retained component norms without accessing or reconstructing field arrays.

    Relative differences use each record's named numerical reference, not an
    exact solution. Electric and magnetic fields retain their separate
    physical times. This validates the recorded norm arithmetic and comparison
    contracts; it does not revalidate the unavailable coefficient archives.
    """
    import matplotlib.pyplot as plt

    records = json.loads(read_resource_bytes(comparison))["comparisons"]
    increments = json.loads(read_resource_bytes(controls))
    component_names = ["electric", "magnetic_x", "magnetic_y"]
    for row in [*records, *increments.values()]:
        absolute = np.asarray(row["absolute_l2"], dtype=float)
        denominator = np.asarray(row["reference_l2"], dtype=float)
        relative = np.asarray(row["relative_l2"], dtype=float)
        if (
            row["components"] != component_names
            or any(value.shape != (3,) for value in (absolute, denominator, relative))
            or not np.isfinite([absolute, denominator, relative]).all()
            or np.any(absolute < 0)
            or np.any(denominator <= 0)
            or not np.allclose(relative, absolute / denominator, rtol=1e-12, atol=0)
            or not np.isclose(
                row["combined_relative_l2"],
                np.linalg.norm(absolute) / np.linalg.norm(denominator),
                rtol=1e-12,
                atol=0,
            )
        ):
            raise ValueError("recorded Maxwell norms require consistent physical components")
        for name in ("electric_time", "magnetic_time"):
            if not np.isclose(row["field"][name], row["reference"][name], rtol=0, atol=1e-12):
                raise ValueError(f"recorded Maxwell fields have different {name}")
    if len(records) != 3 or [row["field"]["resolution"] for row in records] != [128, 128, 32]:
        raise ValueError("recorded Maxwell comparison requires the declared three field spaces")
    control_names = {
        "spatial_256_512": "DG spatial: 256 → 512",
        "spatial_512_1024": "DG spatial: 512 → 1024",
        "time_512": "DG time: 0.0025 → 0.00125",
        "material_quadrature_256": "DG material rule: 12 → 20",
        "mhm_time": "MHM time: 0.01 → 0.005",
        "mhm_material_quadrature": "MHM material rule: 20 → 28",
    }
    if set(increments) != set(control_names):
        raise ValueError("recorded Maxwell controls require each declared independent refinement")
    output.mkdir(parents=True, exist_ok=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, axis = plt.subplots(figsize=(9.0, 4.5), layout="constrained")
    x = np.arange(3)
    for index, (row, label, color) in enumerate(
        zip(records, ("MHM 16 × 16", "MHM 8 × 8", "Classical DG 32 × 32"), colors, strict=True)
    ):
        axis.bar(
            x + (index - 1) * 0.24,
            100 * np.asarray(row["relative_l2"]),
            width=0.24,
            color=color,
            label=label,
        )
    axis.set(
        xticks=x,
        xticklabels=(r"$E_z$", r"$H_x$", r"$H_y$"),
        ylabel="Relative physical L2 difference (%)",
        yscale="log",
        title="Comparison with the classical DG Q2 1024 × 1024 reference",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    names = ["recorded-component-errors"]
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{names[-1]}.{suffix}", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 5.8), layout="constrained")
    y = np.arange(len(control_names))
    values = np.array([increments[name]["relative_l2"] for name in control_names])
    for component, (label, color) in enumerate(
        zip((r"$E_z$", r"$H_x$", r"$H_y$"), colors, strict=True)
    ):
        axis.barh(
            y + (component - 1) * 0.24,
            100 * values[:, component],
            height=0.24,
            color=color,
            label=label,
        )
    axis.set(
        yticks=y,
        yticklabels=list(control_names.values()),
        xlabel="Relative physical L2 increment (%)",
        xscale="log",
        xlim=(0.01, 1.0),
        title="Independent spatial, temporal and material integration controls",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    names.append("recorded-refinement-controls")
    for suffix in ("png", "svg"):
        figure.savefig(output / f"{names[-1]}.{suffix}", dpi=200)
    plt.close(figure)
    receipt = {
        "scope": "Plots of retained norm measurements; no field reconstruction or new simulation",
        "electric_time": records[0]["field"]["electric_time"],
        "magnetic_time": records[0]["field"]["magnetic_time"],
        "source_records_sha256": {
            path.name: hashlib.sha256(read_resource_bytes(path)).hexdigest()
            for path in (comparison, controls)
        },
        "plot_owner_sha256": hashlib.sha256(read_resource_bytes(Path(__file__))).hexdigest(),
        "figure_sha256": {
            f"{name}.{suffix}": hashlib.sha256(
                (output / f"{name}.{suffix}").read_bytes()
            ).hexdigest()
            for name in names
            for suffix in ("png", "svg")
        },
    }
    (output / "recorded-results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    """Produce quadrature-checked comparisons from completed source-frozen acquisitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, default=DATA / "dg-q2-n1024-q12-dt0.00125-device.npz"
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--records-only",
        action="store_true",
        help="Plot retained norms and material without acquiring or reading field arrays",
    )
    args = parser.parse_args()
    if args.records_only:
        output = ROOT / "docs/figures/maxwell-nanoguide"
        output.mkdir(parents=True, exist_ok=True)
        plot_material(output)
        plot_recorded_results(DATA / "comparison.json", DATA / "refinement-controls.json", output)
        return
    paths = [
        DATA / "mhm-n16-f128-q12-dt0.01-fields.npz",
        DATA / "mhm-n8-f128-q12-dt0.01-fields.npz",
        DATA / "dg-q2-n32-q12-dt0.01.npz",
    ]
    records = [compare(path, args.reference) for path in paths]
    result = {
        "norm": "physical unweighted L2; broken Q2 integration on common cell partition",
        "comparisons": records,
    }
    (DATA / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    controls = {
        "spatial_256_512": (
            "dg-q2-n256-q12-dt0.0025.npz",
            "dg-q2-n512-q12-dt0.0025.npz",
        ),
        "spatial_512_1024": (
            "dg-q2-n512-q12-dt0.00125-device.npz",
            "dg-q2-n1024-q12-dt0.00125-device.npz",
        ),
        "time_512": (
            "dg-q2-n512-q12-dt0.0025.npz",
            "dg-q2-n512-q12-dt0.00125-device.npz",
        ),
        "material_quadrature_256": (
            "dg-q2-n256-q12-dt0.0025.npz",
            "dg-q2-n256-q20-dt0.0025.npz",
        ),
        "mhm_time": (
            "mhm-n16-f128-q12-dt0.01-fields.npz",
            "mhm-n16-f128-q12-dt0.005-aligned.npz",
        ),
        "mhm_material_quadrature": (
            "mhm-n16-f128-q20-dt0.01-fields.npz",
            "mhm-n16-f128-q28-dt0.01-fields.npz",
        ),
    }
    increments = {name: compare(DATA / a, DATA / b) for name, (a, b) in controls.items()}
    (DATA / "refinement-controls.json").write_text(json.dumps(increments, indent=2) + "\n")
    output = ROOT / "docs/figures/maxwell-nanoguide"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "refinement-controls.json", "dg-device-verification.json"):
        (output / name).write_bytes(read_resource_bytes(DATA / name))
    if args.plot:
        plot(
            [args.reference, *paths],
            [
                "Classical DG Q2 reference",
                "MHM 16 × 16 / Q2",
                "MHM 8 × 8 / Q2",
                "Classical DG Q2 / 32 × 32",
            ],
            ROOT / "docs/figures/maxwell-nanoguide",
        )
        plot_errors(paths, ["MHM 16 × 16", "MHM 8 × 8", "DG 32 × 32"], args.reference, output)
        plot_material(output)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.maxwell_nanoguide_results").main()
