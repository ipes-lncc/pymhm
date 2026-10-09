"""Acquire material-resolved P1/P0 UNUSUAL controls for the SPE10 reaction case.

The 128 macrotriangles, eight P0 pieces per macroface and nominal local
refinement eight follow the stated Section 4.2 resolution. Material fitting
adds local triangles explicitly: quadrature cuts do not define the strong
residual across a coefficient jump. The zero source is a declared data choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.field_sampling import local_values
from examples.formulations.application import transport as solve_rad
from examples.solve_spe10 import load_layer, pressure_boundary
from examples.unusual_spe10_refinement import prepare_local
from pymhm.execution.cpu import map_local
from pymhm.fem.quadrature.material import fit_material_faces, fit_material_mesh
from pymhm.fem.scalar.operators import p1_geometry
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import ScalarSolution

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/unusual-spe10"


def hashes() -> dict[str, str]:
    """Fingerprint the executed numerical owners independently of reference acquisitions."""
    names = [
        "examples/solve_unusual_spe10.py",
        "examples/field_sampling.py",
        "examples/unusual_spe10_refinement.py",
        "examples/solve_spe10.py",
        "examples/results/spe10/layer-36.npz",
    ]
    names += [
        f"src/pymhm/{name}.py"
        for name in (
            "_legacy/models/transport/rad",
            "_legacy/models/transport/stabilization",
            "fem/scalar/triangle",
            "fem/quadrature/material",
            "meshes/roundoff",
            "io/reservoir",
            "_legacy/models/darcy/cartesian",
            "meshes/refinement",
            "fem/scalar/operators",
            "meshes/triangle",
            "meshes/longest_edge",
            "core/contracts",
            "linalg/linear",
            "execution/cpu",
            "fem/traces/scalar",
        )
    ]
    return current_source_manifest(
        {
            name: hashlib.sha256((source_file(name, root=ROOT)).read_bytes()).hexdigest()
            for name in names
        },
        packages=("pymhm", "examples"),
    )


class UnusualSPE10Field:
    """Independent one-sided P1 coefficients on explicitly archived macro-local meshes."""

    def __init__(self, path: Path) -> None:
        """Restore variable-sized local partitions without interpolation or averaging."""
        with np.load(path) as arrays:
            self.macro = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
            point_offsets, cell_offsets = arrays["point_offsets"], arrays["cell_offsets"]
            points, cells, pressure = (
                arrays["local_points"],
                arrays["local_cells"],
                arrays["pressure"],
            )
            self.meshes = tuple(
                TriangleMesh(
                    points[point_offsets[i] : point_offsets[i + 1]],
                    cells[cell_offsets[i] : cell_offsets[i + 1]],
                )
                for i in range(len(self.macro.cells))
            )
            self.pressure = tuple(
                pressure[point_offsets[i] : point_offsets[i + 1]] for i in range(len(self.meshes))
            )
        self.material = load_layer()
        self.gradients = tuple(p1_geometry(mesh)[0] for mesh in self.meshes)
        self.vertices = tuple(mesh.points[mesh.cells] for mesh in self.meshes)
        self.inverse = tuple(
            np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
            for vertices in self.vertices
        )

    def evaluate_local(
        self, macro: int, points: np.ndarray, owners: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return pressure, gradient and physical flux on stated incident fine cells."""
        coordinate = np.einsum(
            "qab,qb->qa", self.inverse[macro][owners], points - self.vertices[macro][owners, 0]
        )
        bary = np.column_stack((1 - coordinate.sum(axis=1), coordinate))
        if np.any(bary < -1e-10):
            raise ValueError("evaluation points do not belong to their declared fine cells")
        coefficients = self.pressure[macro][self.meshes[macro].cells[owners]]
        value = np.einsum("qi,qi->q", coefficients, bary)
        gradient = np.einsum("qi,qia->qa", coefficients, self.gradients[macro][owners])
        flux = -np.einsum("qab,qb->qa", self.material(points), gradient)
        return value, gradient, flux


def make_skeleton(
    macro: TriangleMesh, material: Any, segments: int, *, trace_fitted: bool
) -> SkeletonSpace:
    """Optionally add material-intersection moments without changing degree or stabilization."""
    skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(0, segments) for _ in macro.faces))
    return fit_material_faces(skeleton, material) if trace_fitted else skeleton


def acquire(
    refinement: int,
    *,
    segments: int = 8,
    workers: int = 8,
    order: int = 5,
    layer_resolution: float | None = None,
    trace_fitted: bool = False,
    max_local_cells: int = 400_000,
) -> dict[str, Any]:
    """Retain the stated macro trace and vary only the explicitly fitted local resolution."""
    fingerprint = hashes()
    unit = TriangleMesh.unit_square(8)
    macro = TriangleMesh(unit.points * [1200.0, 2200.0], unit.cells)
    material = load_layer()
    skeleton = make_skeleton(macro, material, segments, trace_fitted=trace_fitted)
    started = perf_counter()
    refinement_history = []
    if layer_resolution is None:
        local = tuple(
            fit_material_mesh(macro.submesh(cell, refinement), material)
            for cell in range(len(macro.cells))
        )
    else:
        prepared = map_local(
            partial(
                prepare_local,
                macro=macro,
                material=material,
                refinement=refinement,
                layer_resolution=layer_resolution,
                max_cells=max_local_cells,
            ),
            range(len(macro.cells)),
            backend="process" if workers > 1 else "serial",
            workers=workers,
        )
        local = tuple(item[0] for item in prepared)
        refinement_history = [item[1] for item in prepared]
    fitting_seconds = perf_counter() - started
    natural = {int(face): 0.0 for face in macro.boundary_faces if abs(macro.normals[face, 0]) > 0.9}
    started = perf_counter()
    solution = solve_rad(
        macro,
        diffusion=material,
        diffusion_divergence=(0.0, 0.0),
        reaction=1.0,
        source=0.0,
        dirichlet=pressure_boundary,
        neumann=natural,
        skeleton=skeleton,
        degree=1,
        local_refinement=refinement,
        local_meshes=local,
        quadrature_order=order,
        stabilization="unusual",
        backend="process" if workers > 1 else "serial",
        workers=workers,
    )
    elapsed = perf_counter() - started
    return save_solution(
        solution,
        refinement=refinement,
        segments=segments,
        order=order,
        layer_resolution=layer_resolution,
        trace_fitted=trace_fitted,
        max_local_cells=max_local_cells,
        refinement_history=refinement_history,
        fitting_seconds=fitting_seconds,
        elapsed=elapsed,
        workers=workers,
        fingerprint=fingerprint,
    )


def save_solution(
    solution: ScalarSolution,
    *,
    refinement: int,
    segments: int,
    order: int,
    layer_resolution: float | None,
    trace_fitted: bool,
    max_local_cells: int,
    refinement_history: list[Any],
    fitting_seconds: float,
    elapsed: float,
    workers: int,
    fingerprint: dict[str, str],
    output: Path = OUTPUT,
    reuse: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Archive a solved field with its actual trace coordinates and acquisition provenance.

    A restricted prepared response must supply a ``ScalarSolution`` whose skeleton
    and trace coefficients already use the restricted coordinates. Physical local
    coefficients are preserved without another projection or solve. Optional reuse
    metadata describes the exact injection and the prepared global dimension.
    """
    macro, skeleton = solution.skeleton.mesh, solution.skeleton
    local = solution.local_meshes
    output.mkdir(parents=True, exist_ok=True)
    label = "fitted" if layer_resolution is None else f"layer{layer_resolution:g}"
    if trace_fitted:
        label += "-tracefit"
    stem = f"mhm-unusual-{label}-r{refinement}-s{segments}-q{order}"
    archive = output / f"{stem}.npz"
    point_offsets = np.r_[0, np.cumsum([len(fine.points) for fine in local])]
    cell_offsets = np.r_[0, np.cumsum([len(fine.cells) for fine in local])]
    # Each macro side retains its own endpoint value on the vertical profile.
    profile_points, profile_values, profile_macros = [], [], []
    for cell, fine in enumerate(local):
        vertices = macro.points[macro.cells[cell]]
        x = 33.0
        intersections = []
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            if min(a[0], b[0]) <= x <= max(a[0], b[0]) and a[0] != b[0]:
                intersections.append(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))
        if len(intersections) < 2 or max(intersections) == min(intersections):
            continue
        points = np.column_stack(
            (np.full(101, x), np.linspace(min(intersections), max(intersections), 101))
        )
        profile_points.append(points)
        profile_values.append(local_values(fine, solution.values[cell], 1, points))
        profile_macros.append(cell)
    np.savez_compressed(
        archive,
        macro_points=macro.points,
        macro_cells=macro.cells,
        local_points=np.concatenate([fine.points for fine in local]),
        local_cells=np.concatenate([fine.cells for fine in local]),
        pressure=np.concatenate(solution.values),
        point_offsets=point_offsets,
        cell_offsets=cell_offsets,
        trace=solution.hybrid.trace,
        profile_points=np.asarray(profile_points),
        profile_pressure=np.asarray(profile_values),
        profile_macros=np.asarray(profile_macros),
    )
    row = {
        "method": "MHM-UNUSUAL, continuous P1 locals and piecewise-constant macro traces",
        "article": "10.55592/cilamce2025.v5i.14270, Section 4.2",
        "material": "SPE10 Model 2 layer 36 Kx=Ky, original numerical coordinate/material units",
        "domain": [1200.0, 2200.0],
        "reaction": 1.0,
        "advection": [0.0, 0.0],
        "source": 0.0,
        "source_status": "explicit declared control; the heterogeneous section does not restate f",
        "boundary": "bottom p=1; top p=0; zero physical side flux; weak Dirichlet moments",
        "macro_cells": len(macro.cells),
        "trace_degree": 0,
        "trace_segments": segments,
        "trace_segments_per_face": [len(face.degrees) for face in skeleton.faces],
        "material_fitted_trace": trace_fitted,
        "local_degree": 1,
        "nominal_local_refinement": refinement,
        "local_material_fitting": True,
        "reaction_layer_resolution": layer_resolution,
        "max_local_cells": max_local_cells,
        "local_refinement_history": refinement_history,
        "fine_cells": int(cell_offsets[-1]),
        "fine_cells_per_macro": np.diff(cell_offsets).tolist(),
        "trace_dofs": skeleton.size,
        "coarse_dofs": sum(len(value) for value in solution.hybrid.coarse),
        "residual": float(solution.hybrid.residual),
        "pressure_min": float(min(value.min() for value in solution.values)),
        "pressure_max": float(max(value.max() for value in solution.values)),
        "quadrature_order": order,
        "fitting_seconds": fitting_seconds,
        "solve_seconds": elapsed,
        "workers": workers,
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_hashes": fingerprint,
        "source_changed_during_run": hashes() != fingerprint,
    }
    if reuse is not None:
        row["response_reuse"] = reuse
    if row["source_changed_during_run"]:
        raise RuntimeError("numerical source changed during acquisition")
    (output / f"{stem}.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    """Run declared local-space controls while preserving the published macro trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--segments", type=int, nargs="+", default=[8])
    parser.add_argument("--layer-resolution", type=float)
    parser.add_argument("--trace-fitted", action="store_true")
    parser.add_argument("--max-local-cells", type=int, default=400_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--order", type=int, default=5)
    args = parser.parse_args()
    with threadpool_limits(1):
        for refinement in args.refinement:
            for segments in args.segments:
                acquire(
                    refinement,
                    segments=segments,
                    workers=args.workers,
                    order=args.order,
                    layer_resolution=args.layer_resolution,
                    trace_fitted=args.trace_fitted,
                    max_local_cells=args.max_local_cells,
                )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_unusual_spe10").main()
