"""Verify enriched Darcy, tensor Brinkman, conservative SUPG and transient heat.

Use ``--sections`` for selected experiments and ``--reuse-results`` to redraw
the archived measurements without numerical solves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as mtri
import numpy as np
import scipy
from field_sampling import sample_field, sample_profile
from manufactured import darcy_flux, darcy_pressure, darcy_source, stokes_pressure, stokes_velocity
from native_extension_data import (
    BoundaryLayer,
    flow_source,
    heat_exact,
    heat_source,
    rad_diffusion,
    rad_reaction,
    rad_source,
    rad_velocity,
    resistance,
    sine,
)
from plot_mesh import draw_macro_mesh, mark_macro_interfaces
from threadpoolctl import threadpool_limits

from pymhm import (
    FaceSpace,
    SkeletonSpace,
    TriangleMesh,
    solve_brinkman,
    solve_darcy,
    solve_heat,
    solve_transport,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/native-extensions"
REPORT = OUTPUT / "report.json"
FIGURES = ROOT / "docs/figures/high-order"
SEGMENTS = (1, 2, 3, 4, 6, 8)


def face_space(
    mesh: TriangleMesh, degree: int, segments: int, components: int = 1
) -> SkeletonSpace:
    """Build independent Legendre traces with the recorded degree and partition."""
    return SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(degree, segments) for _ in mesh.faces), components
    )


def save_report(report: dict[str, Any]) -> None:
    """Write each completed case to strict JSON so numerical work is preserved."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def add_case(report: dict[str, Any], section: str, row: dict[str, Any]) -> None:
    """Persist one completed measurement with its complete discretization data."""
    report[section].append(row)
    print(json.dumps({"section": section, **row}), flush=True)
    save_report(report)


def snapshot() -> dict[str, str]:
    """Record numerical sources and the original analytical-data implementation."""
    paths = [
        *sorted((ROOT / "src/pymhm").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "examples/native_extension_data.py",
        ROOT / "examples/field_sampling.py",
        ROOT / "examples/manufactured.py",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def save_fields(name: str, mesh: TriangleMesh, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    """Archive sampled broken fields together with the actual macrotriangulation."""
    filename = OUTPUT / f"{name}.npz"
    np.savez_compressed(filename, macro_points=mesh.points, macro_cells=mesh.cells, **arrays)
    return {"file": filename.name, "sha256": hashlib.sha256(filename.read_bytes()).hexdigest()}


def run_darcy(report: dict[str, Any], workers: int) -> None:
    """Refine P1 through P4 pressure and their independently integrated physical flux."""
    mesh = TriangleMesh.unit_square(2)
    for degree in (1, 2, 3, 4):
        for segments in SEGMENTS:
            result = solve_darcy(
                mesh,
                degree=degree,
                skeleton=face_space(mesh, degree - 1, segments),
                local_refinement=2 * segments,
                source=darcy_source,
                dirichlet=darcy_pressure,
                quadrature_order=8,
                backend="thread" if workers > 1 else "serial",
                workers=workers,
            )
            row = {
                "degree": degree,
                "trace_degree": degree - 1,
                "segments": segments,
                "macro_resolution": 2,
                "local_refinement": 2 * segments,
                "assembly_quadrature": 8,
                "error_quadrature": 10,
                "pressure_l2": result.l2_error(darcy_pressure, 10),
                "flux_l2": result.flux_l2_error(darcy_flux, 10),
                "macro_balance": float(np.max(np.abs(result.conservation_residuals(10)))),
                "algebraic_residual": result.hybrid.residual,
            }
            if degree == 4 and segments == 2:
                arrays = sample_field(result.local_meshes, result.pressure, degree)
                arrays["exact"] = darcy_pressure(arrays["points"])
                arrays["exact_flux"] = darcy_flux(arrays["points"])
                row["fields"] = save_fields("darcy-p4", mesh, arrays)
            add_case(report, "darcy", row)


def run_flow(report: dict[str, Any], workers: int) -> None:
    """Verify complete variable anisotropic resistance in TH and equal-order USFEM."""
    mesh = TriangleMesh.unit_square(4)
    for method in ("taylor-hood", "usfem"):
        for segments in SEGMENTS:
            result = solve_brinkman(
                mesh,
                degree=2,
                formulation=method,
                drag=resistance,
                source=flow_source,
                skeleton=face_space(mesh, 1, segments, 2),
                local_refinement=2 * segments,
                quadrature_order=8,
                backend="process" if workers > 1 else "serial",
                workers=workers,
            )
            row = {
                "method": method,
                "degree": 2,
                "pressure_degree": result.pressure_degree,
                "trace_degree": 1,
                "segments": segments,
                "macro_resolution": 4,
                "local_refinement": 2 * segments,
                "assembly_quadrature": 8,
                "error_quadrature": 8,
                "velocity_l2": result.l2_error(stokes_velocity, 8),
                "pressure_l2": result.pressure_l2_error(stokes_pressure, 8),
                "divergence_l2": result.divergence_l2(),
                "algebraic_residual": result.hybrid.residual,
            }
            if method == "usfem" and segments == 2:
                arrays = sample_field(result.local_meshes, result.values, 2)
                arrays["exact"] = stokes_velocity(arrays["points"])
                pressure = sample_field(
                    result.local_meshes, result.pressure, result.pressure_degree
                )
                arrays["pressure"] = pressure["values"]
                arrays["exact_pressure"] = stokes_pressure(arrays["points"])
                row["fields"] = save_fields("tensor-brinkman", mesh, arrays)
            add_case(report, "flow", row)


def run_rad(report: dict[str, Any], workers: int) -> None:
    """Verify conservative variable-coefficient Galerkin and SUPG against a smooth field."""
    mesh = TriangleMesh.unit_square(2)
    for stabilization in ("galerkin", "supg"):
        for segments in SEGMENTS:
            result = solve_transport(
                mesh,
                degree=2,
                skeleton=face_space(mesh, 1, segments),
                local_refinement=2 * segments,
                diffusion=rad_diffusion,
                diffusion_divergence=(0.115, 0.065),
                velocity=rad_velocity,
                velocity_divergence=2.0,
                reaction=rad_reaction,
                source=rad_source,
                stabilization=stabilization,
                quadrature_order=8,
                backend="process" if workers > 1 else "serial",
                workers=workers,
            )
            row = {
                "stabilization": stabilization,
                "degree": 2,
                "trace_degree": 1,
                "segments": segments,
                "macro_resolution": 2,
                "local_refinement": 2 * segments,
                "assembly_quadrature": 8,
                "error_quadrature": 10,
                "scalar_l2": result.l2_error(sine, 10),
                "algebraic_residual": result.hybrid.residual,
            }
            if stabilization == "supg" and segments == 2:
                arrays = sample_field(result.local_meshes, result.values, 2)
                arrays["exact"] = sine(arrays["points"])
                row["fields"] = save_fields("variable-rad", mesh, arrays)
            add_case(report, "rad", row)


def run_layer(report: dict[str, Any], workers: int) -> None:
    """Resolve two outflow layers while measuring overshoots independently of L2 error."""
    mesh = TriangleMesh.unit_square(4)
    for epsilon in (0.02, 0.005):
        exact = BoundaryLayer(epsilon)
        for stabilization in ("galerkin", "supg"):
            for segments in SEGMENTS:
                result = solve_transport(
                    mesh,
                    diffusion=epsilon,
                    velocity=(1.0, 0.0),
                    source=1.0,
                    dirichlet=exact.value,
                    degree=1,
                    skeleton=face_space(mesh, 1, segments),
                    local_refinement=4 * segments,
                    stabilization=stabilization,
                    quadrature_order=20,
                    backend="process" if workers > 1 else "serial",
                    workers=workers,
                )
                # P1 extrema on each triangle occur at vertices; this range is
                # exact for the reconstructed field, including weak BC deviations.
                minimum = float(min(np.min(field) for field in result.values))
                maximum = float(max(np.max(field) for field in result.values))
                row = {
                    "epsilon": epsilon,
                    "stabilization": stabilization,
                    "degree": 1,
                    "trace_degree": 1,
                    "segments": segments,
                    "macro_resolution": 4,
                    "local_refinement": 4 * segments,
                    "assembly_quadrature": 20,
                    "error_quadrature": 24,
                    "scalar_l2": result.l2_error(exact.value, 24),
                    "minimum": minimum,
                    "maximum": maximum,
                    "exact_maximum": exact.maximum,
                    "undershoot": max(0.0, -minimum),
                    "overshoot": max(0.0, maximum - exact.maximum),
                    "algebraic_residual": result.hybrid.residual,
                }
                if segments in (1, 8):
                    arrays = sample_field(result.local_meshes, result.values, 1)
                    arrays["exact"] = exact.value(arrays["points"])
                    arrays |= sample_profile(result, result.values, 1)
                    shape = arrays["profile_points"].shape[:-1]
                    arrays["profile_exact"] = exact.value(
                        arrays["profile_points"].reshape(-1, 2)
                    ).reshape(shape)
                    row["fields"] = save_fields(
                        f"layer-{epsilon:g}-{stabilization}-{segments}", mesh, arrays
                    )
                add_case(report, "layer", row)


def run_heat(report: dict[str, Any], workers: int) -> None:
    """Compare P2/P3 spatial refinement for a smooth genuinely time-dependent field."""
    mesh = TriangleMesh.unit_square(2)
    for degree in (2, 3):
        for segments in SEGMENTS:
            result = solve_heat(
                mesh,
                np.linspace(0.0, 0.1, 5),
                degree=degree,
                skeleton=face_space(mesh, degree - 1, segments),
                local_refinement=2 * segments,
                initial=sine,
                source=heat_source,
                quadrature_order=8,
                backend="thread" if workers > 1 else "serial",
                workers=workers,
            )[-1]
            row = {
                "degree": degree,
                "trace_degree": degree - 1,
                "segments": segments,
                "macro_resolution": 2,
                "local_refinement": 2 * segments,
                "assembly_quadrature": 8,
                "error_quadrature": 10,
                "time": 0.1,
                "time_steps": 4,
                "scalar_l2": result.l2_error(lambda x: heat_exact(x, 0.1), 10),
                "algebraic_residual": result.hybrid.residual,
            }
            if degree == 3 and segments == 2:
                arrays = sample_field(result.local_meshes, result.values, degree)
                arrays["exact"] = heat_exact(arrays["points"], 0.1)
                row["fields"] = save_fields("heat-p3", mesh, arrays)
            add_case(report, "heat", row)


def save_figure(figure: Any, name: str) -> None:
    """Save vector labels and rasterized scientific fields in SVG and PNG."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def field_panel(
    axis: Any,
    data: Any,
    values: np.ndarray,
    title: str,
    *,
    limits: tuple[float, float] | None = None,
    error: bool = False,
) -> Any:
    """Show an unglued scalar image with its actual macrocell boundaries."""
    triangulation = mtri.Triangulation(*data["points"].T, triangles=data["cells"])
    artist = axis.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        rasterized=True,
        cmap="magma" if error else "viridis",
        vmin=0 if error else (None if limits is None else limits[0]),
        vmax=None if error or limits is None else limits[1],
    )
    draw_macro_mesh(axis, TriangleMesh(data["macro_points"], data["macro_cells"]))
    axis.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    return artist


def plot_map(name: str, title: str, *, vector: bool = False, quantity: str = "values") -> None:
    """Compare exact and numerical values with their actual scalar/vector error."""
    data = np.load(OUTPUT / f"{name}.npz")
    if quantity == "flux":
        exact, actual = data["exact_flux"], -data["gradient"]
    elif quantity == "pressure":
        exact, actual = data["exact_pressure"], data["pressure"]
    else:
        exact, actual = data["exact"], data["values"]
    if vector:
        values = [
            np.linalg.norm(exact, axis=1),
            np.linalg.norm(actual, axis=1),
            np.linalg.norm(actual - exact, axis=1),
        ]
    else:
        values = [exact, actual, np.abs(actual - exact)]
    limits = min(values[0].min(), values[1].min()), max(values[0].max(), values[1].max())
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    for index, (axis, value, label) in enumerate(
        zip(axes, values, ("Exact", "Numerical", "Pointwise error"), strict=True)
    ):
        figure.colorbar(
            field_panel(axis, data, value, label, limits=limits, error=index == 2),
            ax=axis,
            shrink=0.82,
        )
    figure.suptitle(title)
    save_figure(figure, name if quantity == "values" else f"{name}-{quantity}")


def plot_convergence(report: dict[str, Any]) -> None:
    """Render six-point error series for every enriched formulation family."""
    specifications = [
        (
            "darcy",
            "degree",
            (1, 2, 3, 4),
            (("pressure_l2", "Darcy pressure L² error"), ("flux_l2", "Raw physical flux L² error")),
        ),
        (
            "flow",
            "method",
            ("taylor-hood", "usfem"),
            (
                ("velocity_l2", "Tensor-Brinkman velocity L² error"),
                ("pressure_l2", "Tensor-Brinkman pressure L² error"),
            ),
        ),
        (
            "rad",
            "stabilization",
            ("galerkin", "supg"),
            (("scalar_l2", "Variable conservative RAD, P2"),),
        ),
        ("heat", "degree", (2, 3), (("scalar_l2", "Transient heat, t=0.1"),)),
    ]
    for section, group, methods, norms in specifications:
        if not report.get(section):
            continue
        figure, axes = plt.subplots(
            1, len(norms), figsize=(6 * len(norms), 4.2), constrained_layout=True, squeeze=False
        )
        for axis, (norm, title) in zip(axes[0], norms, strict=True):
            for method in methods:
                rows = [row for row in report[section] if row[group] == method]
                label = (
                    f"P{method}"
                    if group == "degree"
                    else {
                        "taylor-hood": "Taylor–Hood P2/P1",
                        "usfem": "USFEM P2/P2",
                        "galerkin": "Galerkin",
                        "supg": "SUPG",
                    }[method]
                )
                axis.loglog(
                    [1 / row["segments"] for row in rows],
                    [row[norm] for row in rows],
                    "o-",
                    label=label,
                )
            axis.set(title=title, xlabel="Relative skeleton segment length 1/s", ylabel="L² error")
            axis.set_xticks(
                [1 / s for s in SEGMENTS], ["1" if s == 1 else f"1/{s}" for s in SEGMENTS]
            )
            axis.xaxis.set_minor_formatter(ticker.NullFormatter())
            axis.grid(True, which="major", alpha=0.25)
            axis.legend()
        save_figure(figure, f"{section}-convergence")


def plot_layers(report: dict[str, Any]) -> None:
    """Show layer error and true P1 extrema, without claiming a maximum principle."""
    if not report.get("layer"):
        return
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row, epsilon in enumerate((0.02, 0.005)):
        for method, color in (("galerkin", "#3367b5"), ("supg", "#097c83")):
            values = [
                case
                for case in report["layer"]
                if case["epsilon"] == epsilon and case["stabilization"] == method
            ]
            for axis, quantity in zip(
                axes[row], ("scalar_l2", "undershoot", "overshoot"), strict=True
            ):
                axis.plot(
                    [case["segments"] for case in values],
                    [case[quantity] for case in values],
                    "o-",
                    color=color,
                    label="Galerkin" if method == "galerkin" else "SUPG",
                )
                label = {
                    "scalar_l2": "L² error",
                    "undershoot": "Undershoot",
                    "overshoot": "Overshoot",
                }[quantity]
                axis.set(xlabel="Segments per macroface s", title=f"ε={epsilon:g}: {label}")
                axis.set_xticks(SEGMENTS)
                axis.grid(True, alpha=0.25)
                axis.legend()
    save_figure(figure, "layer-errors")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for row, epsilon in enumerate((0.02, 0.005)):
        for column, segments in enumerate((1, 8)):
            axis = axes[row, column]
            for method, color in (("galerkin", "#3367b5"), ("supg", "#097c83")):
                data = np.load(OUTPUT / f"layer-{epsilon:g}-{method}-{segments}.npz")
                for index, parameter in enumerate(data["profile_parameter"]):
                    if method == "galerkin":
                        axis.plot(
                            parameter,
                            data["profile_exact"][index],
                            color="#222222",
                            linewidth=1.5,
                            label="Exact" if index == 0 else None,
                        )
                    axis.plot(
                        parameter,
                        data["profile_values"][index],
                        color=color,
                        linewidth=1.2,
                        label=("Galerkin" if method == "galerkin" else "SUPG")
                        if index == 0
                        else None,
                    )
                if method == "galerkin":
                    mark_macro_interfaces(axis, data["profile_breaks"][1:-1], label=True)
            axis.set(
                title=f"ε={epsilon:g}; s={segments}; one-sided profiles",
                xlabel="x at y=0.37",
                ylabel="u",
            )
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
    save_figure(figure, "layer-profiles")
    plot_map("layer-0.005-supg-1", "Thin outflow layer: SUPG P1, ε=0.005, s=1")


def render(report: dict[str, Any]) -> None:
    """Draw all available measurements and representative exact-reference maps."""
    plot_convergence(report)
    for section, name, title in (
        ("darcy", "darcy-p4", "Primal Darcy P4 pressure"),
        (
            "flow",
            "tensor-brinkman",
            "USFEM P2/P2 velocity magnitude with variable tensor resistance",
        ),
        ("rad", "variable-rad", "Conservative variable-coefficient SUPG P2"),
        ("heat", "heat-p3", "Transient heat P3 at t=0.1"),
    ):
        if report.get(section):
            plot_map(name, title, vector=section == "flow")
    if report.get("darcy"):
        plot_map(
            "darcy-p4", "Primal Darcy P4: physical flux magnitude", vector=True, quantity="flux"
        )
    if report.get("flow"):
        plot_map(
            "tensor-brinkman", "USFEM P2/P2: pressure with tensor resistance", quantity="pressure"
        )
    plot_layers(report)


def main() -> None:
    """Select numerical experiments and preserve their independent source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("darcy", "flow", "rad", "layer", "heat"),
        default=["darcy", "flow", "rad", "layer", "heat"],
    )
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"provenance": {}}
    report["evidence"] = (
        "Native PDE solutions against independently differentiated exact fields; "
        "no external solver execution."
    )
    with threadpool_limits(limits=1):
        if not args.reuse_results:
            report["runtime_environment"] = {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
                "pixi_lock_sha256": hashlib.sha256((ROOT / "pixi.lock").read_bytes()).hexdigest(),
                "workers": args.workers,
                "native_threads": 1,
            }
            runners = {
                "darcy": run_darcy,
                "flow": run_flow,
                "rad": run_rad,
                "layer": run_layer,
                "heat": run_heat,
            }
            for section in args.sections:
                report[section] = []
                report["provenance"][section] = {"start": snapshot()}
                runners[section](report, args.workers)
                provenance = report["provenance"][section]
                provenance["end"] = snapshot()
                provenance["changed_source_files"] = sorted(
                    path
                    for path in set(provenance["start"]) | set(provenance["end"])
                    if provenance["start"].get(path) != provenance["end"].get(path)
                )
                save_report(report)
        render(report)


if __name__ == "__main__":
    main()
