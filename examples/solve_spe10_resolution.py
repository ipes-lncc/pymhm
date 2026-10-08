"""Separate local and skeletal resolution on one fixed SPE10 macro partition.

These are declared resolution controls for the L09 P2/RT2 configuration.
Segmented P0 skeletons are enrichment experiments, not the article's single
constant per macroface. Every field retains the unchanged permeability and
physical boundary conditions of the archived adaptive state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.formulations.application import darcy as solve_darcy
from examples.solve_spe10 import load_layer, pressure_boundary
from examples.solve_spe10_balanced import offsets
from examples.spe10_adaptive import DATA, hashes, natural_faces, source_path
from pymhm.core.validation import positive_int
from pymhm.execution.cpu import map_local
from pymhm.fem.quadrature.material import fit_material_mesh
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh
from pymhm.recovery.moments import reconstruct_darcy_moments


@dataclass(frozen=True)
class MaterialFittedLocals:
    """Fit a fixed uniform local partition while retaining all trace breakpoints."""

    mesh: TriangleMesh
    material: CartesianCellField
    refinement: int

    def __call__(self, cell: int) -> TriangleMesh:
        """Intersect the local trial partition with each physical material pixel."""
        return fit_material_mesh(self.mesh.submesh(cell, self.refinement), self.material)


def acquire(
    macro_archive: Path,
    output: Path,
    refinement: int,
    segments: int,
    *,
    workers: int = 8,
    fit_material: bool = False,
) -> dict[str, Any]:
    """Solve and checkpoint one explicitly aligned P2/P0/RT2 resolution control."""
    for value, name in (
        (refinement, "refinement"),
        (segments, "segments"),
        (workers, "workers"),
    ):
        positive_int(value, name)
    if refinement % segments:
        raise ValueError("trace segments must divide the local refinement")
    with np.load(macro_archive) as values:
        mesh = TriangleMesh(values["macro_points"], values["macro_cells"])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
    fingerprint = hashes()
    for name in (
        "src/pymhm/meshes/roundoff.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/core/contracts.py",
        "src/pymhm/execution/cpu.py",
        "examples/solve_spe10_resolution.py",
        "examples/archive_precision.py",
    ):
        fingerprint[name] = hashlib.sha256(source_path(name).read_bytes()).hexdigest()
    started = perf_counter()
    material = load_layer()
    local_meshes = (
        tuple(
            map_local(
                MaterialFittedLocals(mesh, material, refinement),
                range(len(mesh.cells)),
                backend="process",
                workers=workers,
            )
        )
        if fit_material
        else None
    )
    prepared = perf_counter()
    solution = solve_darcy(
        mesh,
        degree=2,
        permeability=material,
        dirichlet=pressure_boundary,
        neumann=natural_faces(mesh),
        skeleton=skeleton,
        local_refinement=refinement,
        local_meshes=local_meshes,
        quadrature_order=6,
        local_refinement_precision="extended",
        parallel_assembly=True,
        backend="process",
        workers=workers,
    )
    solved = perf_counter()
    print(
        dict(stage="solved", refinement=refinement, segments=segments, seconds=solved - started),
        flush=True,
    )
    reconstructed = reconstruct_darcy_moments(
        solution, degree=2, quadrature_order=6, backend="process", workers=workers
    )
    finished = perf_counter()
    fine = solution.local_meshes
    output.mkdir(parents=True, exist_ok=True)
    stem = f"mhm-{'fitted-' if fit_material else ''}r{refinement}-s{segments}"
    archive = output / f"{stem}.npz"
    balance = solution.conservation_residuals()
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.concatenate([part.points for part in fine]),
        point_offsets=offsets(tuple(part.points for part in fine)),
        local_cells=np.concatenate([part.cells for part in fine]),
        cell_offsets=offsets(tuple(part.cells for part in fine)),
        **precision_fields("pressure", np.concatenate(solution.pressure)),
        pressure_offsets=offsets(solution.pressure),
        **precision_fields("reconstructed_flux", np.concatenate(reconstructed.flux)),
        flux_offsets=offsets(reconstructed.flux),
        trace=solution.hybrid.trace,
        macro_balance=balance,
    )
    bottom = [face for face in mesh.boundary_faces if np.all(mesh.points[mesh.faces[face], 1] == 0)]
    inflow = -sum(
        mesh.lengths[face] * np.mean(solution.hybrid.trace[skeleton.dofs(face)]) for face in bottom
    )
    record = {
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "macro_archive": macro_archive.name,
        "macro_archive_sha256": hashlib.sha256(macro_archive.read_bytes()).hexdigest(),
        "macro_triangles": len(mesh.cells),
        "fine_triangles": sum(len(part.cells) for part in fine),
        "local_degree": 2,
        "local_refinement": refinement,
        "trace_degree": 0,
        "trace_segments": segments,
        "reconstruction_degree": 2,
        "global_dofs": skeleton.size + len(mesh.cells),
        "material_fitted": fit_material,
        "material_integration": "exact Cartesian pixel intersections",
        "boundary_conditions": "bottom pressure 1; top pressure 0; zero side flux",
        "local_refinement_precision": "extended",
        "assembly_order": 6,
        "reconstruction_order": 6,
        "residual": solution.hybrid.residual,
        "macro_balance_linf": float(np.max(abs(balance))),
        "continuous_equilibrium_moment_linf": float(
            max(np.max(abs(part)) for part in reconstructed.continuous_moment_residuals())
        ),
        "normal_moment_defect_linf": float(
            max(np.max(abs(part)) for part in reconstructed.normal_flux_residuals())
        ),
        "inflow": float(inflow),
        "local_geometry_seconds": prepared - started,
        "solve_seconds": solved - prepared,
        "reconstruction_seconds": finished - solved,
        "timing_scope": "solve and reconstruction only; diagnostics and export excluded",
        "source_hashes": fingerprint,
        "source_changed_during_solve": any(
            hashlib.sha256(source_path(name).read_bytes()).hexdigest() != digest
            for name, digest in fingerprint.items()
        ),
    }
    if record["source_changed_during_solve"]:
        raise RuntimeError("numerical source changed during acquisition; preserve and review")
    (output / f"{stem}.json").write_text(json.dumps(record, indent=2) + "\n")
    print({key: value for key, value in record.items() if key != "source_hashes"}, flush=True)
    return record


def main() -> None:
    """Run one control without changing the archived published adaptive sequence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-archive", type=Path, default=DATA / "published/mhm-level6.npz")
    parser.add_argument("--output", type=Path, default=DATA / "resolution")
    parser.add_argument("--refinement", type=int, required=True)
    parser.add_argument("--segments", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fit-material", action="store_true")
    options = parser.parse_args()
    with threadpool_limits(1):
        acquire(
            options.macro_archive,
            options.output,
            options.refinement,
            options.segments,
            workers=options.workers,
            fit_material=options.fit_material,
        )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_spe10_resolution").main()
