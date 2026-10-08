"""Acquire the finite-Galerkin MHM--MsHHO analogue and analytical convergence."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import matplotlib

from pymhm.io.provenance import current_source_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.campaign_provenance import positive_integers
from examples.formulations.application import darcy as solve_darcy
from examples.formulations.application import moment_diffusion as solve_mshho
from examples.local_response_cache import array_identity
from examples.mshho_field_archive import attach_mhm, field_arrays, replay, write_field
from examples.plot_mesh import draw_macro_mesh
from examples.transport_checkpoints import write_progress
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.fem.scalar.triangle import tabulate
from pymhm.io.provenance import file_digest
from pymhm.linalg.linear import LinearSolveError

ROOT = Path(__file__).resolve().parents[1]


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate the homogeneous Dirichlet sine solution."""
    return np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])


def flux(points: np.ndarray) -> np.ndarray:
    """Evaluate minus the analytical gradient."""
    x, y = np.pi * points.T
    return -np.pi * np.column_stack((np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)))


def main() -> None:
    """Acquire fresh fields, bases and sources before publishing verified case records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/results/mshho-current")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--contrasts", type=float, nargs="+", default=[1, 1e3, 1e4, 1e6])
    parser.add_argument("--acquire-only", action="store_true")
    args = parser.parse_args()
    levels = positive_integers(args.levels, label="macro levels", increasing=True)
    contrasts = tuple(args.contrasts)
    if (
        not contrasts
        or any(not np.isfinite(value) or value < 1 for value in contrasts)
        or len(set(contrasts)) != len(contrasts)
    ):
        raise ValueError("Material contrasts must be distinct finite values at least one")
    if args.output.exists():
        raise ValueError("MsHHO acquisition requires a fresh output directory")
    files = list((ROOT / "src/pymhm").rglob("*.py")) + [
        ROOT / name
        for name in (
            "pixi.lock",
            "pixi.toml",
            "pyproject.toml",
            "examples/verify_mshho.py",
            "examples/mshho_field_archive.py",
            "examples/archive_precision.py",
            "examples/campaign_provenance.py",
            "examples/local_response_cache.py",
            "examples/transport_checkpoints.py",
            "examples/plot_mesh.py",
        )
    ]
    sources = current_source_manifest(
        {path.relative_to(ROOT).as_posix(): file_digest(path) for path in files}
    )
    args.output.mkdir(parents=True)
    for name in sources:
        destination = args.output / "executed_sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
        if file_digest(destination) != sources[name]:
            raise ValueError("MsHHO numerical source changed while creating its executed snapshot")
    acquisition = str(uuid4())
    metadata = dict(
        schema="pymhm-mshho-study-v1",
        acquisition_uuid=acquisition,
        source_sha256=sources,
        macro_levels=list(levels),
        contrasts=list(contrasts),
        local_degree=3,
        local_refinement=2,
        assembly_quadrature_order=8,
        norm_quadrature_orders=[10, 12],
        source_variant="projected",
        mshho_local_refinement_precision="extended",
        mhm_local_refinement_precision="extended",
        mhm_hybrid_refinement_steps=2,
        mhm_hybrid_refinement_min_steps=2,
        original_residual_tolerance=1e-10,
        scope="Finite Galerkin analogue; not an exact-local-solve theorem reproduction",
        status="acquiring",
    )
    write_progress(args.output / "manifest.json", metadata)
    figures = ROOT / "docs/figures/mshho"
    rows, equivalence = [], []
    selected = None
    started = perf_counter()
    with threadpool_limits(1):
        for degree in (0, 1):
            for n in levels:
                mesh = TriangleMesh.unit_square(n)
                skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(degree) for _ in mesh.faces))
                result = solve_mshho(
                    mesh,
                    skeleton=skeleton,
                    cell_degree=degree,
                    degree=3,
                    local_refinement=2,
                    source=lambda x: 2 * np.pi**2 * exact(x),
                    quadrature_order=8,
                    local_refinement_precision="extended",
                )
                norms = {}
                archives = {}
                for order in (10, 12):
                    archive = args.output / f"analytical-n{n}-moment{degree}-q{order}.npz"
                    archives[str(order)] = write_field(
                        archive,
                        result,
                        dict(n=n, face_degree=degree, cell_degree=degree, source="2*pi^2*sin*sin"),
                        acquisition_uuid=acquisition,
                        source_sha256=sources,
                        order=order,
                    )
                    arrays = field_arrays(result, order)
                    pressure_error = flux_error = 0.0
                    for cell in range(len(result.local)):
                        numerical_p, _, numerical_q = replay(arrays, cell)
                        physical = arrays[f"physical_quadrature_points_{cell}"]
                        pressure_difference = numerical_p - exact(physical.reshape(-1, 2)).reshape(
                            numerical_p.shape
                        )
                        flux_difference = numerical_q - flux(physical.reshape(-1, 2)).reshape(
                            numerical_q.shape
                        )
                        weights = arrays["quadrature_weights"]
                        areas = arrays[f"areas_{cell}"]
                        pressure_error += float(areas @ (pressure_difference**2 @ weights))
                        flux_error += float(areas @ (np.sum(flux_difference**2, axis=-1) @ weights))
                    norms[str(order)] = dict(
                        pressure_l2=float(np.sqrt(pressure_error)),
                        flux_l2=float(np.sqrt(flux_error)),
                    )
                stability = max(
                    abs(norms["10"][key] - norms["12"][key])
                    / max(norms["12"][key], np.finfo(float).tiny)
                    for key in norms["10"]
                )
                if stability > 1e-9:
                    raise ValueError("Independent MsHHO error quadratures do not agree")
                rows.append(
                    dict(
                        n=n,
                        face_degree=degree,
                        cell_degree=degree,
                        **norms["10"],
                        residual=result.residual,
                        norm_quadratures=norms,
                        norm_quadrature_relative_difference=stability,
                        field_archives=archives,
                    )
                )
                print(
                    {key: value for key, value in rows[-1].items() if key != "field_archives"},
                    flush=True,
                )
                if degree == 1 and n == 4:
                    selected = result
        for contrast in contrasts:
            mesh = TriangleMesh.unit_square(2)
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
            options: dict[str, Any] = dict(
                skeleton=skeleton,
                permeability=np.diag([contrast, 1.0]),
                source=4.0,
                dirichlet=lambda x: x[:, 0] - x[:, 1],
                degree=3,
                local_refinement=2,
                quadrature_order=8,
                local_refinement_precision="extended",
            )
            primal = solve_mshho(mesh, cell_degree=1, **options)
            try:
                dual = solve_darcy(
                    mesh, hybrid_refinement_steps=2, hybrid_refinement_min_steps=2, **options
                )
            except LinearSolveError:
                equivalence.append(
                    dict(
                        contrast=contrast,
                        status="not_certified",
                        reason="MHM local solve does not meet the physical residual tolerance",
                        primal_residual=primal.residual,
                    )
                )
                continue
            diagnostics = attach_mhm(field_arrays(primal, 10), dual, options["dirichlet"])
            if any(
                diagnostics[f"{method}_original_relative_residual"] > 1e-10
                for method in ("mhm", "mshho")
            ):
                equivalence.append(
                    dict(
                        contrast=contrast,
                        status="not_certified",
                        reason="Original physical saddle equations exceed the unchanged 1e-10 gate",
                        original_saddle_diagnostics=diagnostics,
                        primal_residual=primal.residual,
                        dual_residual=dual.hybrid.residual,
                    )
                )
                print(equivalence[-1], flush=True)
                continue
            primal_archive = write_field(
                args.output / f"contrast-{contrast:g}-q10.npz",
                primal,
                dict(
                    n=2,
                    face_degree=1,
                    cell_degree=1,
                    permeability=[contrast, 1.0],
                    source=4.0,
                    dirichlet="x-y",
                ),
                acquisition_uuid=acquisition,
                source_sha256=sources,
                order=10,
                mhm=dual,
                dirichlet=options["dirichlet"],
            )
            difference = max(
                float(np.max(np.abs(a - b)))
                for a, b in zip(primal.pressure, dual.pressure, strict=True)
            )
            scale = max(float(np.max(np.abs(a))) for a in dual.pressure)
            equivalence.append(
                dict(
                    contrast=contrast,
                    status="verified",
                    field_linf=difference,
                    relative_linf=difference / scale,
                    primal_residual=primal.residual,
                    dual_residual=dual.hybrid.residual,
                    primal_field=primal_archive,
                    original_saddle_diagnostics=diagnostics,
                )
            )
            assert difference / scale < 2e-8
    if any(file_digest(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError("MsHHO executed numerical sources changed during acquisition")
    metadata.update(
        status="fields-acquired",
        acquisition_seconds=perf_counter() - started,
        convergence=rows,
        equivalence=equivalence,
        mhm_local_refinement_precision="extended",
        mshho_local_refinement_precision="extended",
        mhm_hybrid_refinement_steps=2,
        mhm_hybrid_refinement_min_steps=2,
        local_residual_tolerance_unchanged=True,
        accumulation_mantissa_bits=int(np.finfo(np.longdouble).nmant),
    )
    write_progress(args.output / "study.json", metadata)
    write_progress(args.output / "manifest.json", metadata)
    if args.acquire_only:
        return
    if levels != (1, 2, 4, 8, 16) or contrasts != (1, 1e3, 1e4, 1e6):
        raise ValueError("The public gallery requires the complete declared analytical study")
    if selected is None or any(row["status"] != "verified" for row in equivalence):
        raise ValueError("The public gallery requires every declared equivalence field")
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for degree, marker in ((0, "o"), (1, "s")):
        selected_rows = [row for row in rows if row["face_degree"] == degree]
        h = 1 / np.array([row["n"] for row in selected_rows])
        for ax, key in zip(axes, ("pressure_l2", "flux_l2"), strict=True):
            errors = np.array([row[key] for row in selected_rows])
            ax.loglog(h, errors, marker=marker, label=f"$m=k_F={degree}$")
            rate = np.log(errors[-2] / errors[-1]) / np.log(2)
            selected_rows[-1][key + "_last_rate"] = float(rate)
            assert rate > (1.65 if key == "pressure_l2" else 0.8) + 0.6 * degree
    for ax, name in zip(axes, ("Pressure", "Physical raw flux"), strict=True):
        ax.set(xlabel="Macro spacing H", ylabel="L2 error", title=name)
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(figures / "convergence.png", dpi=200)
    fig.savefig(figures / "convergence.svg")
    plt.close(fig)
    points, cells, values = [], [], []
    rendering: dict[str, np.ndarray] = {
        "macro_points": selected.skeleton.mesh.points,
        "macro_faces": selected.skeleton.mesh.faces,
    }
    reference = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, 3)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    rendering["reference_barycentric"] = bary
    rendering["reference_cells"] = reference.cells
    offset = 0
    for macro, (data, pressure) in enumerate(zip(selected.local, selected.pressure, strict=True)):
        dofs, _, basis, _, _ = tabulate(data.mesh, selected.degree, bary)
        rendering[f"executed_cardinal_values_{macro}"] = basis
        rendering[f"nodal_dofs_{macro}"] = dofs
        rendering[f"fine_points_{macro}"] = data.mesh.points
        rendering[f"fine_cells_{macro}"] = data.mesh.cells
        rendering.update(precision_fields(f"pressure_{macro}", pressure))
        points.append(
            np.einsum("qi,tia->tqa", bary, data.mesh.points[data.mesh.cells]).reshape(-1, 2)
        )
        values.append((pressure[dofs] @ basis.T).ravel())
        for cell in range(len(data.mesh.cells)):
            cells.append(reference.cells + offset + cell * len(bary))
        offset += len(data.mesh.cells) * len(bary)
    xy, numerical = np.vstack(points), np.concatenate(values)
    analytical = exact(xy)
    connectivity = np.vstack(cells)
    rendering.update(points=xy, triangles=connectivity, analytical_pressure=analytical)
    rendering.update(precision_fields("numerical_pressure", numerical))
    rendering_archive = args.output / "selected-field-rendering.npz"
    np.savez_compressed(rendering_archive, allow_pickle=False, **rendering)
    rendering_record = dict(
        schema="pymhm-mshho-field-rendering-v1",
        acquisition_uuid=acquisition,
        source_sha256=sources,
        archive=rendering_archive.name,
        archive_sha256=file_digest(rendering_archive),
        array_sha256={name: array_identity(value) for name, value in rendering.items()},
        n=4,
        face_degree=1,
        cell_degree=1,
        local_degree=3,
        local_refinement=2,
        interface_convention="Independent one-sided macrocell values",
    )
    write_progress(rendering_archive.with_suffix(".json"), rendering_record)
    metadata["field_rendering"] = rendering_record
    triangulation = mtri.Triangulation(*xy.T, triangles=connectivity)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    error = numerical - analytical
    for ax, data, title in zip(
        axes,
        (analytical, numerical, error),
        ("Analytical pressure", "MsHHO m=k_F=1", "Signed pressure error"),
        strict=True,
    ):
        limits = (-max(abs(error)), max(abs(error))) if ax is axes[2] else (0, 1)
        artist = ax.tripcolor(
            triangulation,
            data,
            shading="gouraud",
            vmin=limits[0],
            vmax=limits[1],
            cmap="RdBu_r" if ax is axes[2] else "viridis",
            rasterized=True,
        )
        draw_macro_mesh(ax, selected.skeleton.mesh)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        fig.colorbar(artist, ax=ax, shrink=0.85, format="%.2g")
    fig.savefig(figures / "fields.png", dpi=200)
    fig.savefig(figures / "fields.svg")
    plt.close(fig)
    if any(file_digest(ROOT / name) != digest for name, digest in sources.items()):
        raise ValueError("MsHHO executed numerical sources changed while rendering")
    metadata["status"] = "verified-gallery"
    write_progress(args.output / "study.json", metadata)
    write_progress(args.output / "manifest.json", metadata)
    destination = ROOT / "examples/results/mshho.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            dict(
                **metadata,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
