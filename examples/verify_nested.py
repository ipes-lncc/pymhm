"""Five-level field equivalence for an original recursive MHM construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.cartesian_darcy import define_cartesian_darcy
from examples.nested_field_archive import (
    capture_display,
    display_fields,
    executed_arrays,
    field_norms,
    original_checks,
    read_archive,
    replay_responses,
    restore,
    write_archive,
)
from examples.nested_formulation import DiffusionDiscretization, child_problem
from pymhm import FaceSpace, HybridSystem, SkeletonSpace, assemble
from pymhm.core.nested import nest_hybrid_system, nested_trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import qk_basis, quadrilateral_quadrature
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution

ROOT = case_workspace()


def exact(x: np.ndarray) -> np.ndarray:
    """Analytical pressure vanishing on the unit square boundary."""
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def gradient(x: np.ndarray) -> np.ndarray:
    """Analytical gradient, independently differentiated from the source."""
    a, b = np.pi * x.T
    return np.pi * np.column_stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)))


def source(x: np.ndarray) -> np.ndarray:
    """Positive Laplacian source for -Delta(p)=f."""
    return 2 * np.pi**2 * exact(x)


def nonhomogeneous_exact(x: np.ndarray) -> np.ndarray:
    """Pressure with affine Dirichlet lifting and the unchanged sine source."""
    return 1 + x[:, 0] - x[:, 1] + exact(x)


def nonhomogeneous_gradient(x: np.ndarray) -> np.ndarray:
    """Return the derivative of the affine lifting plus the independent sine gradient."""
    return gradient(x) + np.array([1.0, -1.0])


def dirichlet(x: np.ndarray) -> np.ndarray:
    """Exact affine pressure on every exterior side, excluding sine roundoff."""
    return 1 + x[:, 0] - x[:, 1]


def acquire(
    n: int, *, nonhomogeneous: bool = False, archive: Path | None = None
) -> tuple[dict[str, Any], list[ConformingQuadrilateralSolution]]:
    """Acquire actual recursive/flat responses and compare unchanged Q2 leaf spaces.

    Dirichlet pressure fixes the global constant; every Neumann leaf and nested
    parent declares its physical constant mode and volume moment. The archive
    stores these executed bases, including E and both levels' original data.
    Numerical raw flux means minus the polynomial gradient, independently of
    the conservative macro normal-flux multiplier. No fine-cell conservation
    or mesh-uniform inf-sup estimate is inferred from this finite acquisition.
    """
    if type(n) is not int or n not in {1, 2, 4, 8, 16} or type(nonhomogeneous) is not bool:
        raise ValueError("levels1,2,4,8,16 and an explicit boolean boundary selection required")
    started = perf_counter()
    source_files = [
        Path(__file__),
        Path(__file__).with_name("nested_field_archive.py"),
        Path(__file__).with_name("archive_precision.py"),
        Path(__file__).with_name("transport_checkpoints.py"),
        Path(__file__).with_name("campaign_provenance.py"),
        ROOT / "pixi.lock",
        *sorted(source_file("src/pymhm/__init__.py", root=ROOT).parent.rglob("*.py")),
    ]
    executed_sources = current_source_manifest(
        source_identity(
            ROOT,
            (
                path
                for path in source_files
                if path.name not in ("pixi.lock", "pixi.toml", "pyproject.toml") or path.is_file()
            ),
        ),
        packages=("pymhm", "examples"),
    )
    macro = CartesianMacroMesh(n)
    outer = SkeletonSpace(macro, tuple(FaceSpace.uniform(1, 2) for _ in macro.faces))
    children = []
    for cell in range(len(macro.cells)):
        mesh = macro.submesh(cell, 2)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        inner = assemble(child_problem(DiffusionDiscretization(mesh, skeleton, source=source)))
        kernel = np.r_[np.zeros(skeleton.size), np.ones(4)][:, None]
        moments = [metadata[1] for metadata in inner.local_metadata]
        constraint = inner.mean_constraint(moments)[0][:, None]
        selected, mapping = nested_trace_map(outer, cell, skeleton)
        children.append(
            nest_hybrid_system(
                inner,
                selected,
                mapping,
                outer.cell_dofs(cell),
                kernel=kernel,
                constraints=constraint,
            )
        )
    boundary, fixed = boundary_data(outer, dirichlet if nonhomogeneous else 0.0, order=8)
    parent = HybridSystem([child.problem for child in children], boundary_load=boundary)
    result = parent.solve(fixed=fixed)
    recovered = [
        child.reconstruct(field) for child, field in zip(children, result.fields, strict=True)
    ]
    flat_mesh = CartesianMacroMesh(2 * n)
    flat_skeleton = SkeletonSpace(flat_mesh, tuple(FaceSpace.uniform(1) for _ in flat_mesh.faces))
    flat_boundary, flat_fixed = boundary_data(
        flat_skeleton, dirichlet if nonhomogeneous else 0.0, order=8
    )
    flat_definition = define_cartesian_darcy(
        flat_mesh,
        skeleton=flat_skeleton,
        degree=2,
        permeability=1.0,
        source=source,
        dirichlet=dirichlet if nonhomogeneous else 0.0,
        local_refinement=(2, 2),
        quadrature_order=6,
    )
    flat = assemble(flat_definition.problem)
    flat_result = flat.solve(fixed=flat_fixed)
    arrays = executed_arrays(
        n,
        outer,
        parent,
        result,
        children,
        recovered,
        flat,
        flat_result,
        flat_skeleton,
        flat_boundary,
        boundary,
    )
    for order in (6, 8, 10):
        reference, weights = quadrilateral_quadrature(order)
        basis, derivatives = qk_basis(2, reference)
        arrays.update(
            {
                f"q{order}_reference": reference,
                f"q{order}_weights": weights,
                f"q{order}_basis": basis,
                f"q{order}_gradient": derivatives,
            }
        )
    capture_display(arrays)
    analytical = nonhomogeneous_exact if nonhomogeneous else exact
    analytical_gradient = nonhomogeneous_gradient if nonhomogeneous else gradient
    norms = {
        f"quadrature_{q}": field_norms(arrays, analytical, analytical_gradient, q) for q in (8, 10)
    }
    before, after = norms.values()
    sensitivity = max(
        abs(before[k] - after[k]) / max(abs(after[k]), np.finfo(float).tiny)
        for k in ("pressure_l2", "flux_l2")
    )
    difference = restore(arrays, "leaf_pressure_recursive") - restore(arrays, "leaf_pressure_flat")
    relative_difference = float(
        np.linalg.norm(difference) / np.linalg.norm(restore(arrays, "leaf_pressure_flat"))
    )
    checks = original_checks(arrays)
    replay = max(
        float(
            np.linalg.norm(replay_responses(arrays, prefix) - restore(arrays, f"{prefix}_pressure"))
            / max(np.linalg.norm(restore(arrays, f"{prefix}_pressure")), np.finfo(float).tiny)
        )
        for prefix in ("leaf", "flat", "parent")
    )
    if (
        max(checks.values()) > 1e-10
        or relative_difference > 1e-11
        or replay > 1e-11
        or sensitivity > 1e-9
        or max(
            after["pressure_relative_flat_difference"], after["raw_flux_relative_flat_difference"]
        )
        > 1e-9
    ):
        raise RuntimeError(
            "original equations, executed-basis replay or physical field gate failed"
        )
    row: dict[str, Any] = dict(
        schema=2,
        acquisition_uuid=str(uuid4()),
        n=n,
        macro_cells=n * n,
        inner_macro_cells=4 * n * n,
        fine_cells=16 * n * n,
        boundary_case="affine_plus_sine" if nonhomogeneous else "sine_zero_dirichlet",
        outer_global_dofs=len(parent.rhs),
        flat_global_dofs=len(flat.rhs),
        relative_leaf_difference=relative_difference,
        pressure_l2=after["pressure_l2"],
        flux_l2=after["flux_l2"],
        outer_residual=result.residual,
        inner_residual=max(value.interior_residual for value in recovered),
        original_physical_checks=checks,
        executed_response_replay_relative_difference=replay,
        norm_quadrature_relative_sensitivity=sensitivity,
        norms=norms,
        assembly_quadrature_order=6,
        boundary_quadrature_order=8,
        boundary="Dirichlet pressure; no additional physical gauge",
        physical_trace="signed macro normal Darcy flux; parameter basis [1,2t-1]",
        raw_flux="minus the one-sided Q2 polynomial gradient; not an H(div) reconstruction",
        method=(
            "Original two-level recursive MHM and a separately executed flat MHM "
            "in identical leaf spaces"
        ),
        literature="10.1007/978-3-319-41640-3_13; recursive construction and analytical square",
        finite_case_acceptance=True,
        literal_literature_reproduction=False,
        mesh_uniform_inf_sup_estimate_verified=False,
        elapsed_seconds=perf_counter() - started,
        source_sha256=executed_sources,
        precision_convention=(
            "Executed coefficient arithmetic; portable float64 high/correction/tail"
        ),
        local_kernel="Declared physical constant, paired with its actual volume integral",
        recursive_map="Signed exact P1 restriction; parent reactions and one-sided leaves retained",
    )
    if executed_sources != current_source_manifest(
        source_identity(
            ROOT,
            (
                path
                for path in source_files
                if path.name not in ("pixi.lock", "pixi.toml", "pyproject.toml") or path.is_file()
            ),
        ),
        packages=("pymhm", "examples"),
    ):
        raise RuntimeError("executed acquisition source changed")
    if archive is not None:
        row = write_archive(archive, arrays, row)
    fields = []
    if n == 4 and not nonhomogeneous:
        for i, values in enumerate(restore(arrays, "leaf_pressure_recursive")):
            fine = flat.local_metadata[i][0]
            fields.append(ConformingQuadrilateralSolution(fine, 2, values, 1.0, result.residual))
    return row, fields


def plots(rows: list[dict[str, Any]], archive: Path | None = None) -> None:
    """Render current convergence and one-sided fields from executed saved display tables."""
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    output = ROOT / "docs/figures/nested"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    h = 1 / np.array([row["n"] for row in rows])
    for key, label in (("pressure_l2", r"$\|p-p_h\|_{L^2}$"), ("flux_l2", r"$\|q-q_h\|_{L^2}$")):
        axes[0].loglog(h, [r[key] for r in rows], "o-", label=label)
    axes[0].set(xlabel="Outer macro size H", ylabel="Absolute error")
    axes[0].legend()
    axes[1].loglog(h, [r["relative_leaf_difference"] for r in rows], "s-", color="#7a3e9d")
    axes[1].set(xlabel="Outer macro size H", ylabel="Relative difference to one-level MHM")
    for ax in axes:
        ax.grid(alpha=0.2, which="both")
    for ext in ("png", "svg"):
        fig.savefig(output / f"convergence.{ext}", dpi=180)
    plt.close(fig)
    if archive is None:
        return
    record, arrays = read_archive(archive)
    if record["boundary_case"] != "sine_zero_dirichlet":
        raise ValueError("the selected field panel requires the declared sine boundary case")
    points, pressure, flux = display_fields(arrays)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    titles = ("Pressure", r"Flux $q_x$", r"Flux $q_y$")
    for k, ax in enumerate(axes):
        all_values = pressure if k == 0 else flux[..., k - 1]
        limit = float(np.max(abs(all_values)))
        for physical, values in zip(points, all_values, strict=True):
            xx, yy = physical.T.reshape(2, 17, 17)
            image = ax.pcolormesh(
                xx,
                yy,
                values.reshape(xx.shape),
                shading="gouraud",
                rasterized=True,
                cmap="viridis" if k == 0 else "RdBu_r",
                vmin=min(float(all_values.min()), 0) if k == 0 else -limit,
                vmax=float(all_values.max()) if k == 0 else limit,
            )
        for endpoints, color, width in (
            (arrays["flat_face_endpoints"], ".6", 0.4),
            (arrays["outer_face_endpoints"], ".1", 1.0),
        ):
            ax.add_collection(LineCollection(endpoints, colors=color, linewidths=width))
        ax.set(xlabel="x", ylabel="y", title=titles[k], aspect="equal")
        fig.colorbar(image, ax=ax, shrink=0.8, pad=0.04)
    for ext in ("png", "svg"):
        fig.savefig(output / f"fields.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    """Acquire fresh field archives for selected levels and both physical boundary cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/nested-current")
    parser.add_argument(
        "--boundary", choices=("both", "homogeneous", "nonhomogeneous"), default="both"
    )
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--acquire-only", action="store_true")
    args = parser.parse_args()
    if (
        not args.levels
        or len(set(args.levels)) != len(args.levels)
        or any(n not in {1, 2, 4, 8, 16} for n in args.levels)
        or type(args.native_threads) is not int
        or args.native_threads <= 0
    ):
        raise ValueError("distinct supported levels and positive native thread count required")
    if args.output.exists():
        raise ValueError("a fresh acquisition output directory is required")
    args.output.mkdir(parents=True)
    rows = []
    selected_archive = None
    cases = (False, True) if args.boundary == "both" else (args.boundary == "nonhomogeneous",)
    with threadpool_limits(args.native_threads):
        for n in args.levels:
            for nonhomogeneous in cases:
                name = "affine" if nonhomogeneous else "homogeneous"
                path = args.output / f"n{n}-{name}.npz"
                row, _ = acquire(n, nonhomogeneous=nonhomogeneous, archive=path)
                rows.append(row)
                if n == 4 and not nonhomogeneous:
                    selected_archive = path
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "n",
                                "boundary_case",
                                "relative_leaf_difference",
                                "original_physical_checks",
                                "elapsed_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
    record = dict(
        schema=2,
        method="Original recursive MHM in declared Q2/r2 and P1 leaf spaces",
        local_space="Q2, 2x2 fine cells per inner macrocell",
        trace="P1 per inner face; two P1 segments per outer face",
        rows=rows,
    )
    (args.output / "nested.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.acquire_only:
        plots(
            [row for row in rows if row["boundary_case"] == "sine_zero_dirichlet"], selected_archive
        )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.verify_nested").main()
