"""Acquire explicit SPE10 PGMHM material alternatives and refined references.

Figure 11 identifies Kz; Figure 13 pressure and the supplied layer data support Kx.
These alternatives remain distinct. The two NW--SE macrotriangles and 2048 local
triangles follow Figure 13;
alpha=0.1, P2 local elements, zero source and regular local connectivity are
explicitly declared where the historical executable specification is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields, restore_precision
from examples.field_sampling import local_values
from examples.pgmhm_campaign import diagnostics
from examples.spe10_adaptive import DOMAIN, ROOT, StructuredRT, mesh_rectangle, natural_faces
from examples.spe10_adaptive_norms import overlay_quadrature
from pymhm.cut_cells import fit_material_faces, fit_material_mesh
from pymhm.darcy_rt import solve_darcy_rt_conforming
from pymhm.elements import p1_geometry, triangle_quadrature
from pymhm.lagrange import nodal_space, reference_basis
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.pgmhm import solve_pgmhm
from pymhm.reservoir import CartesianCellField

OUTPUT = ROOT / "examples/results/pgmhm-spe10"


def load_material(component: str = "kz") -> CartesianCellField:
    """Return the selected layer-one component as an explicit isotropic scalar field."""
    if component not in ("kx", "kz"):
        raise ValueError("component must be kx or kz")
    with np.load(ROOT / "examples/results/spe10/layer-1.npz") as arrays:
        return CartesianCellField(
            arrays["permeability"][..., 0 if component == "kx" else 2], tuple(arrays["spacing"])
        )


def pressure_boundary(points: np.ndarray) -> np.ndarray:
    """Impose bottom pressure one and top pressure zero; side values are unused."""
    return 1 - points[:, 1] / DOMAIN[1]


def macro_mesh() -> TriangleMesh:
    """Use the northwest--southeast diagonal visible in Figure 13."""
    return TriangleMesh(
        np.array([[0.0, 0.0], [1200.0, 0.0], [1200.0, 2200.0], [0.0, 2200.0]]),
        np.array([[0, 1, 3], [1, 2, 3]]),
    )


def fingerprint() -> dict[str, str]:
    """Identify the executed operators, geometry, reader and physical layer data."""
    names = [
        f"src/pymhm/{name}.py"
        for name in (
            "pgmhm",
            "darcy",
            "darcy_rt",
            "rt",
            "mh",
            "lagrange",
            "cut_cells",
            "elements",
            "hybrid",
            "solvers",
            "mesh",
            "reservoir",
            "scalar_boundary",
        )
    ]
    names += [
        "examples/solve_pgmhm_spe10.py",
        "examples/spe10_adaptive.py",
        "examples/spe10_adaptive_norms.py",
        "examples/archive_precision.py",
        "examples/results/spe10/layer-1.npz",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def write_record(path: Path, row: dict[str, Any], hashes: dict[str, str]) -> None:
    """Write current provenance only after checking all acquisition sources."""
    row["source_hashes"] = hashes
    row["source_changed_during_run"] = hashes != fingerprint()
    if row["source_changed_during_run"]:
        raise RuntimeError("a numerical acquisition source changed")
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)


def acquire_mhm(
    output: Path,
    segments: int,
    order: int,
    alpha: float,
    component: str = "kx",
    *,
    refinement: int = 32,
    fitted: bool = False,
    trace_fitted: bool = False,
) -> None:
    """Retain P2/P0 and the physical coefficient while explicitly varying resolution."""
    hashes = fingerprint()
    mesh = macro_mesh()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
    material = load_material(component)
    if trace_fitted:
        skeleton = fit_material_faces(skeleton, material)
    local = (
        tuple(
            fit_material_mesh(mesh.submesh(i, refinement), material) for i in range(len(mesh.cells))
        )
        if fitted
        else None
    )
    started = perf_counter()
    solution = solve_pgmhm(
        mesh,
        stabilization_parameter=alpha,
        skeleton=skeleton,
        permeability=material,
        source=0.0,
        dirichlet=pressure_boundary,
        neumann=natural_faces(mesh),
        degree=2,
        local_refinement=refinement,
        local_meshes=local,
        quadrature_order=order,
        local_refinement_precision="extended",
    )
    elapsed = perf_counter() - started
    save_mhm_solution(
        output,
        solution,
        segments,
        order,
        alpha,
        component,
        refinement=refinement,
        fitted=fitted,
        trace_fitted=trace_fitted,
        elapsed=elapsed,
        hashes=hashes,
    )


def save_mhm_solution(
    output: Path,
    solution: Any,
    segments: int,
    order: int,
    alpha: float,
    component: str,
    *,
    refinement: int,
    fitted: bool,
    trace_fitted: bool,
    elapsed: float,
    hashes: dict[str, str],
    reduced_trace: np.ndarray | None = None,
    reduced_skeleton: SkeletonSpace | None = None,
    reuse: dict[str, Any] | None = None,
) -> None:
    """Archive one accepted field, with explicit coordinates for an exactly restricted trace."""
    mesh = solution.skeleton.mesh
    skeleton = solution.skeleton if reduced_skeleton is None else reduced_skeleton
    archived_trace = solution.hybrid.trace if reduced_trace is None else reduced_trace
    name = f"pgmhm-s{segments}-q{order}"
    if refinement != 32 or fitted or trace_fitted:
        label = "fitted" if fitted else "uniform"
        label += "-tracefit" if trace_fitted else ""
        name = f"pgmhm-{label}-r{refinement}-s{segments}-q{order}"
    profile_points = np.stack(
        [
            np.column_stack((np.full(501, 600.0), np.linspace(0, 1100, 501))),
            np.column_stack((np.full(501, 600.0), np.linspace(1100, 2200, 501))),
        ]
    )
    profiles = {}
    for label, fields in (
        ("pressure", solution.pressure),
        ("enriched_pressure", solution.enriched_pressure),
    ):
        profiles[label] = np.stack(
            [
                local_values(fine, field, 2, points)
                for fine, field, points in zip(
                    solution.local_meshes, fields, profile_points, strict=True
                )
            ]
        )
    path = output / f"{name}.npz"
    np.savez_compressed(
        path,
        component=component,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.concatenate([fine.points for fine in solution.local_meshes]),
        local_cells=np.concatenate([fine.cells for fine in solution.local_meshes]),
        point_offsets=np.r_[0, np.cumsum([len(fine.points) for fine in solution.local_meshes])],
        cell_offsets=np.r_[0, np.cumsum([len(fine.cells) for fine in solution.local_meshes])],
        coefficient_offsets=np.r_[0, np.cumsum([len(value) for value in solution.pressure])],
        **precision_fields("pressure", np.concatenate(solution.pressure)),
        **precision_fields("enriched_pressure", np.concatenate(solution.enriched_pressure)),
        trace=archived_trace,
        trace_breaks=np.concatenate([np.asarray(face.breaks) for face in skeleton.faces]),
        trace_break_offsets=np.r_[0, np.cumsum([len(face.breaks) for face in skeleton.faces])],
        profile_points=profile_points,
        profile_pressure=np.asarray(profiles["pressure"], dtype=float),
        profile_enriched_pressure=np.asarray(profiles["enriched_pressure"], dtype=float),
    )
    row = {
        "method": "PyMHM PGMHM, base and residual-enriched P2 pressure",
        "article": "10.1007/s40314-023-02304-y, Section 6.3, Figure 13",
        "material": f"SPE10 layer 1 {component.upper()}, isotropic",
        "component": component,
        "units": "original numerical feet and millidarcies; pressure difference one",
        "degree": 2,
        "trace_degree": 0,
        "segments": segments,
        "local_refinement": refinement,
        "material_fitted_local_meshes": fitted,
        "material_fitted_trace": trace_fitted,
        "trace_segments_per_face": [len(face.degrees) for face in skeleton.faces],
        "quadrature_order": order,
        "alpha": alpha,
        "source": 0.0,
        "local_refinement_precision": "extended",
        "free_global_dofs": (
            skeleton.size
            + len(mesh.cells)
            - sum(skeleton.faces[face].size for face in natural_faces(mesh))
        ),
        **diagnostics(solution),
        "solve_seconds": elapsed,
        "archive": path.name,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodal_pressure_min": float(min(p.min() for p in solution.pressure)),
        "nodal_pressure_max": float(max(p.max() for p in solution.pressure)),
        "nodal_enriched_pressure_min": float(min(p.min() for p in solution.enriched_pressure)),
        "nodal_enriched_pressure_max": float(max(p.max() for p in solution.enriched_pressure)),
    }
    if reuse is not None:
        row["prepared_global_dofs"] = row["global_dofs"]
        row["global_dofs"] = skeleton.size + len(mesh.cells)
        row["response_reuse"] = reuse
    write_record(output / f"{name}.json", row, hashes)


def reference_norms(
    current: StructuredRT, previous: StructuredRT | None, order: int, component: str = "kz"
) -> dict[str, float]:
    """Integrate RT2/P2 norms exactly on nested, pixel-aligned triangles."""
    if current.nx % 60 or current.ny % 220:
        raise ValueError("reference norm meshes must align with every physical pixel")
    if previous is not None and (current.nx != 2 * previous.nx or current.ny != 2 * previous.ny):
        raise ValueError("reference increments require isotropic dyadic nested triangulations")
    bary, weights = triangle_quadrature(order)
    partials = []
    areas = current.mesh.areas
    material = load_material(component)
    for first in range(0, len(current.mesh.cells), 256):
        ids = np.arange(first, min(first + 256, len(current.mesh.cells)))
        points = np.einsum("qi,tia->tqa", bary, current.mesh.points[current.mesh.cells[ids]])
        points = points.reshape(-1, 2)
        pressure, flux, divergence = current.evaluate(points)
        coefficient = material(points)
        weight = (areas[ids, None] * weights).ravel().astype(np.longdouble)
        old_p, old_q = (pressure, flux) if previous is None else previous.evaluate(points)[:2]
        flux_delta = np.sum((flux - old_q) ** 2, axis=1)
        values = np.stack(
            (
                pressure**2,
                np.sum(flux**2, axis=1),
                np.sum(flux**2, axis=1) / coefficient,
                divergence**2,
                (pressure - old_p) ** 2,
                flux_delta,
                flux_delta / coefficient,
            )
        )
        partials.append(np.sum(values * weight, axis=1, dtype=np.longdouble))
    norms = np.sqrt(np.sum(partials, axis=0, dtype=np.longdouble))
    return dict(
        zip(
            (
                "pressure_l2",
                "flux_l2",
                "flux_energy",
                "divergence_l2",
                "pressure_difference_l2",
                "flux_difference_l2",
                "flux_difference_energy",
            ),
            map(float, norms),
            strict=True,
        )
    ) | {
        "pressure_relative_difference": float(norms[4] / norms[0]),
        "flux_relative_difference": float(norms[5] / norms[1]),
        "flux_energy_relative_difference": float(norms[6] / norms[2]),
        "order": order,
    }


def acquire_reference(output: Path, nx: int, solver: str, component: str = "kx") -> None:
    """Solve a classical global RT2/P2 reference with strong zero side flux."""
    if nx % 60:
        raise ValueError("nx must be a multiple of 60 to resolve the material pixels")
    ny = 11 * nx // 3
    mesh = mesh_rectangle(nx, ny)
    hashes, started = fingerprint(), perf_counter()
    solution = solve_darcy_rt_conforming(
        mesh,
        degree=2,
        permeability=load_material(component),
        source=0.0,
        dirichlet=pressure_boundary,
        neumann=natural_faces(mesh),
        quadrature_order=5,
        solver=solver,
    )
    elapsed = perf_counter() - started
    current = StructuredRT(nx, ny, solution.pressure[0], solution.flux[0])
    path = output / f"classical-rt2-{nx}x{ny}.npz"
    np.savez_compressed(
        path, component=component, nx=nx, ny=ny, pressure=current.pressure, flux=current.flux
    )
    previous_path = output / f"classical-rt2-{nx // 2}x{ny // 2}.npz"
    previous = StructuredRT.load(previous_path) if previous_path.exists() else None
    moments = current.flux[3 * mesh.cell_faces] * mesh.signs
    bottom = np.array(
        [face for face in mesh.boundary_faces if np.all(mesh.points[mesh.faces[face], 1] == 0)]
    )
    row = {
        "method": "classical global RT2/P2; no macro trace restriction",
        "nx": nx,
        "ny": ny,
        "cells": len(mesh.cells),
        "dofs": current.pressure.size + current.flux.size,
        "material": f"SPE10 layer 1 {component.upper()}, isotropic",
        "component": component,
        "residual": solution.residual,
        "fine_balance_linf": float(np.max(abs(moments.sum(axis=1)))),
        "inflow": -float(current.flux[3 * bottom].sum()),
        "solver": solver,
        "solve_seconds": elapsed,
        "archive": path.name,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    write_record(path.with_suffix(".json"), row, hashes)
    row["norms"] = [reference_norms(current, previous, order, component) for order in (4, 5)]
    write_record(path.with_suffix(".json"), row, hashes)


class PGMHMField:
    """Restore independent macro-local P2 fields in their executed nodal coordinates."""

    def __init__(self, path: Path) -> None:
        """Load the portable wider-precision components without gluing macrointerfaces."""
        with np.load(path) as arrays:
            self.component = str(arrays["component"]) if "component" in arrays else "kz"
            local_points, local_cells = arrays["local_points"], arrays["local_cells"]
            if "point_offsets" in arrays:
                points, cells = arrays["point_offsets"], arrays["cell_offsets"]
                self.meshes = tuple(
                    TriangleMesh(
                        local_points[points[i] : points[i + 1]],
                        local_cells[cells[i] : cells[i + 1]],
                    )
                    for i in range(len(points) - 1)
                )
            else:
                self.meshes = tuple(
                    TriangleMesh(p, c) for p, c in zip(local_points, local_cells, strict=True)
                )
            self.fields = {}
            offsets = arrays.get("coefficient_offsets")
            for name in ("pressure", "enriched_pressure"):
                field = restore_precision(
                    arrays[name], arrays[f"{name}_correction"], arrays[f"{name}_tail"]
                )
                if offsets is not None:
                    self.fields[name] = tuple(
                        field[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
                    )
                else:
                    self.fields[name] = tuple(field)
        self.dofs = [nodal_space(mesh, 2)[0] for mesh in self.meshes]
        self.geometry = [p1_geometry(mesh)[0] for mesh in self.meshes]
        self.material = load_material(self.component)

    def evaluate(self, macro: int, cell: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate both pressures and physical raw fluxes inside one specified fine cell."""
        vertices = self.meshes[macro].points[self.meshes[macro].cells[cell]]
        inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
        coordinate = (points - vertices[0]) @ inverse.T
        bary = np.column_stack((1 - coordinate.sum(axis=1), coordinate))
        basis, derivative, _ = reference_basis(2, bary)
        coefficients = np.stack(
            [
                self.fields[name][macro][self.dofs[macro][cell]]
                for name in ("pressure", "enriched_pressure")
            ]
        )
        pressure = np.einsum("qi,bi->bq", basis, coefficients)
        gradient = np.einsum("qin,na,bi->bqa", derivative, self.geometry[macro][cell], coefficients)
        return pressure, -self.material(points)[None, :, None] * gradient


_COMPARISON_FIELDS: tuple[PGMHMField, StructuredRT] | None = None
_COMPARISON_THREAD_LIMIT: Any = None


def initialize_comparison(mhm: str, reference: str) -> None:
    """Load validated immutable reference coefficients once in each spawn worker."""
    global _COMPARISON_FIELDS, _COMPARISON_THREAD_LIMIT
    _COMPARISON_THREAD_LIMIT = threadpool_limits(1)
    _COMPARISON_FIELDS = PGMHMField(Path(mhm)), StructuredRT.load(Path(reference))


def integrate_comparison_group(task: tuple[int, int, int, int]) -> np.ndarray:
    """Return per-cell squared norms in their original deterministic reduction order."""
    if _COMPARISON_FIELDS is None:
        raise RuntimeError("initialize comparison fields before integrating cells")
    field, reference = _COMPARISON_FIELDS
    macro, first, last, order = task
    mesh = field.meshes[macro]
    partials = []
    for cell in range(first, last):
        vertices = mesh.points[mesh.cells[cell]]
        points, weights = overlay_quadrature(vertices, reference, order)
        p, q = field.evaluate(macro, cell, points)
        rp, rq, _ = reference.evaluate(points)
        coefficient = field.material(points)
        delta = np.sum((q - rq) ** 2, axis=-1)
        values = np.vstack(
            (
                rp**2,
                np.sum(rq**2, axis=1),
                np.sum(rq**2, axis=1) / coefficient,
                (p - rp) ** 2,
                delta,
                delta / coefficient,
            )
        )
        partials.append(np.sum(values * weights.astype(np.longdouble), axis=1, dtype=np.longdouble))
    return np.asarray(partials)


def compare(output: Path, mhm_path: Path, reference_path: Path, workers: int = 1) -> None:
    """Compute physical intersection norms with an unchanged ordered cell reduction."""
    if workers < 1:
        raise ValueError("workers must be positive")
    hashes = fingerprint()
    field = PGMHMField(mhm_path)
    with np.load(reference_path) as arrays:
        component = str(arrays["component"]) if "component" in arrays else "kz"
    if component != field.component:
        raise ValueError("MHM and reference material components must agree")
    rows = []
    for order in (4, 5):
        tasks = [
            (macro, first, min(first + 32, len(mesh.cells)), order)
            for macro, mesh in enumerate(field.meshes)
            for first in range(0, len(mesh.cells), 32)
        ]
        partials = []
        if workers == 1:
            initialize_comparison(str(mhm_path), str(reference_path))
            for values in map(integrate_comparison_group, tasks):
                partials.extend(values)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_comparison,
                initargs=(str(mhm_path), str(reference_path)),
            ) as pool:
                for index, values in enumerate(pool.map(integrate_comparison_group, tasks)):
                    partials.extend(values)
                    if (index + 1) % 16 == 0 or index + 1 == len(tasks):
                        print(dict(order=order, groups=index + 1, total=len(tasks)), flush=True)
        norms = np.sqrt(np.sum(partials, axis=0, dtype=np.longdouble))
        row = {
            "order": order,
            "reference_pressure_l2": float(norms[0]),
            "reference_flux_l2": float(norms[1]),
            "reference_flux_energy": float(norms[2]),
        }
        for variant, offset in (("base", 0), ("enriched", 1)):
            for key, position, denominator in (
                ("pressure_l2", 3, 0),
                ("flux_l2", 5, 1),
                ("flux_energy", 7, 2),
            ):
                row[f"{variant}_{key}_difference"] = float(norms[position + offset])
                row[f"{variant}_{key}_relative_difference"] = float(
                    norms[position + offset] / norms[denominator]
                )
        rows.append(row)
    report = {
        "method": "physical intersection integrals; raw -K grad(p), no H(div) reconstruction",
        "component": field.component,
        "workers": workers,
        "reduction": "original fine-cell order and per-cell quadrature sums",
        "mhm": mhm_path.name,
        "mhm_sha256": hashlib.sha256(mhm_path.read_bytes()).hexdigest(),
        "reference": reference_path.name,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "norms": rows,
    }
    write_record(output / f"{mhm_path.stem}-comparison.json", report, hashes)


def main() -> None:
    """Keep acquisitions and comparisons explicit and restartable outside CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("mhm", "reference", "compare"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--refinement", type=int, default=32)
    parser.add_argument("--material-fitted", action="store_true")
    parser.add_argument("--trace-fitted", action="store_true")
    parser.add_argument("--component", choices=("kx", "kz"), default="kx")
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--nx", nargs="+", type=int, default=[60, 120, 240])
    parser.add_argument("--solver", default="scipy")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mhm", type=Path)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.output = args.output or (OUTPUT / "kx" if args.component == "kx" else OUTPUT)
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(args.native_threads):
        if args.kind == "mhm":
            for segments in args.segments:
                acquire_mhm(
                    args.output,
                    segments,
                    args.order,
                    args.alpha,
                    args.component,
                    refinement=args.refinement,
                    fitted=args.material_fitted,
                    trace_fitted=args.trace_fitted,
                )
        elif args.kind == "reference":
            for nx in args.nx:
                acquire_reference(args.output, nx, args.solver, args.component)
        else:
            if args.reference is None or args.mhm is None:
                parser.error("compare requires --mhm and --reference")
            compare(args.output, args.mhm, args.reference, args.workers)


if __name__ == "__main__":
    main()
