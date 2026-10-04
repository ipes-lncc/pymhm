"""Literal L09 indicators with P2/P0 SPE10 spaces and isotropic metric remeshing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.campaign_checkpoint import require_sources, verify_checkpoint
from examples.solve_spe10 import load_layer, pressure_boundary
from examples.solve_spe10_balanced import macro_quality, offsets
from examples.spe10_adaptive import DATA, hashes, mesh_rectangle, natural_faces, source_path
from pymhm.adaptivity.darcy import solve_adaptive_darcy
from pymhm.adaptivity.metric import remesh_freefem, residual_mesh_size
from pymhm.core.validation import positive_int
from pymhm.estimators.darcy_local import estimate_darcy_local_refinement
from pymhm.meshes.triangle import TriangleMesh


def validate_state(
    row: dict[str, Any], directory: Path, current: dict[str, str], remesher_sha256: str | None
) -> None:
    """Verify physical data, metric policy and field bytes before remeshing a checkpoint."""
    require_sources(row.get("source_hashes", {}), current)
    verify_checkpoint(
        row,
        dict(
            local_degree=2,
            local_refinement=2,
            trace_degree=0,
            reconstruction_degree=2,
            material_fitted=False,
            assembly_order=6,
            estimator_order=6,
            estimator_convention="published",
            source_changed_during_solve=False,
            local_refinement_precision="extended",
            metric_coefficient=1.0,
            remesher_max_vertices=10000,
            macro_refinement="FreeFEM/BAMG isotropic residual metric",
            remesher_executable_sha256=remesher_sha256,
        ),
        directory=directory,
        archive_required=True,
        metrics=(
            "estimator",
            "local_indicator",
            "flux_defect",
            "nonconformity",
            "divergence_defect",
            "oscillation",
            "macro_balance_linf",
        ),
    )


def acquire(
    output: Path,
    *,
    levels: int = 24,
    target_cells: int = 3786,
    workers: int = 8,
    executable: str = "FreeFem++",
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Preserve the published four-local-triangle P2/P0 choice and indicator factors.

    Each state uses equation (5.6), the FreeFEM documented residual metric
    with coefficient one, and BAMG remeshing. The author's public call identifies
    that metric, but its modified wrapper and final mesh are not supplied.
    Local nested energy differences are measured but do not change the spaces.
    The target is a macrocell budget, not an accuracy or estimator certificate.
    """
    for value, name in ((levels, "levels"), (target_cells, "target_cells"), (workers, "workers")):
        positive_int(value, name)
    output.mkdir(parents=True, exist_ok=True)
    mesh = mesh_rectangle(16, 16)
    fingerprint = hashes()
    for name in (
        "src/pymhm/adaptivity/metric.py",
        "src/pymhm/estimators/darcy_local.py",
        "src/pymhm/estimators/darcy.py",
        "src/pymhm/core/contracts.py",
        "src/pymhm/execution/cpu.py",
        "examples/solve_spe10_published.py",
        "examples/solve_spe10_balanced.py",
        "examples/archive_precision.py",
        "examples/solve_spe10.py",
        "src/pymhm/fem/conditions.py",
        "src/pymhm/recovery/equilibrated.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/fem/scalar/operators.py",
    ):
        fingerprint[name] = hashlib.sha256(source_path(name).read_bytes()).hexdigest()
    program = shutil.which(executable)
    binary_hash = hashlib.sha256(Path(program).read_bytes()).hexdigest() if program else None
    material, started = load_layer(), perf_counter()
    records: list[dict[str, Any]] = []
    if resume:
        records = json.loads((output / "adaptive.json").read_text())
        if not records:
            raise ValueError("resume requires a completed published state")
        for record in records:
            validate_state(record, output, fingerprint, binary_hash)
        previous = records[-1]
        with np.load(output / previous["archive"]) as values:
            mesh = TriangleMesh(values["macro_points"], values["macro_cells"])
            metric = residual_mesh_size(mesh, values["local_squared"])
        mesh = remesh_freefem(mesh, metric.requested, executable=executable)
    for level in range(len(records), levels):
        solve_started = perf_counter()
        state = solve_adaptive_darcy(
            mesh,
            iterations=1,
            degree=2,
            local_refinement=2,
            trace_degree=0,
            reconstruction_degree=2,
            estimator_order=6,
            estimator_convention="published",
            estimator_backend="process",
            estimator_workers=workers,
            theta=0.5,
            permeability=material,
            dirichlet=pressure_boundary,
            neumann=natural_faces(mesh),
            quadrature_order=6,
            local_refinement_precision="extended",
            parallel_assembly=True,
            backend="process",
            workers=workers,
        )
        solution, estimate = state.solutions[0], state.estimators[0]
        # A metric acts on all cells; no binary bulk-marked set enters remeshing.
        marked = np.zeros(len(mesh.cells), dtype=bool)
        metric = residual_mesh_size(mesh, estimate.local_squared)
        solve_seconds = perf_counter() - solve_started
        local_started = perf_counter()
        local = estimate_darcy_local_refinement(
            solution, refinement_precision="extended", backend="process", workers=workers
        )
        local_seconds = perf_counter() - local_started
        fine = solution.local_meshes
        archive = f"mhm-level{level}.npz"
        np.savez_compressed(
            output / archive,
            macro_points=mesh.points,
            macro_cells=mesh.cells,
            local_points=np.concatenate([part.points for part in fine]),
            point_offsets=offsets(tuple(part.points for part in fine)),
            local_cells=np.concatenate([part.cells for part in fine]),
            cell_offsets=offsets(tuple(part.cells for part in fine)),
            **precision_fields("pressure", np.concatenate(solution.pressure)),
            pressure_offsets=offsets(solution.pressure),
            **precision_fields(
                "recovered_pressure", np.concatenate(estimate.potential.local_values)
            ),
            **precision_fields(
                "reconstructed_flux", np.concatenate(estimate.reconstructed_flux.flux)
            ),
            flux_offsets=offsets(estimate.reconstructed_flux.flux),
            trace=solution.hybrid.trace,
            local_squared=estimate.local_squared,
            flux_defect=estimate.flux_defect,
            nonconformity=estimate.nonconformity,
            divergence_defect=estimate.divergence_defect,
            oscillation=estimate.oscillation,
            local_refinement_squared=local.local_squared,
            marked=marked,
            mesh_size=metric.current,
            requested_mesh_size=metric.requested,
            mesh_size_factors=metric.factors,
        )
        bottom = [
            face for face in mesh.boundary_faces if np.all(mesh.points[mesh.faces[face], 1] == 0)
        ]
        inflow = -sum(
            mesh.lengths[face] * solution.hybrid.trace[solution.skeleton.dofs(face)][0]
            for face in bottom
        )
        reason = (
            "macro_budget"
            if len(mesh.cells) >= target_cells
            else "iterations"
            if level + 1 == levels
            else "continue"
        )
        row = dict(
            level=level,
            archive=archive,
            archive_sha256=hashlib.sha256((output / archive).read_bytes()).hexdigest(),
            archive_encoding="float64 high + correction + tail",
            macro_triangles=len(mesh.cells),
            fine_triangles=sum(len(part.cells) for part in fine),
            global_dofs=solution.skeleton.size + len(mesh.cells),
            local_degree=2,
            local_refinement=2,
            trace_degree=0,
            reconstruction_degree=2,
            material_fitted=False,
            material_integration="exact Cartesian pixel intersections",
            local_refinement_precision="extended",
            assembly_order=6,
            estimator_order=6,
            estimator_convention=estimate.convention,
            metric_coefficient=1.0,
            metric_threshold=metric.threshold,
            requested_size_min=float(metric.requested.min()),
            requested_size_max=float(metric.requested.max()),
            macro_refinement="FreeFEM/BAMG isotropic residual metric",
            remeshing_scope="documented metric; modified historical wrapper unavailable",
            remesher_executable_sha256=binary_hash,
            remesher_max_vertices=10000,
            marked=int(marked.sum()),
            estimator=estimate.total,
            flux_defect=float(np.linalg.norm(estimate.flux_defect)),
            nonconformity=float(np.linalg.norm(estimate.nonconformity)),
            divergence_defect=float(np.linalg.norm(estimate.divergence_defect)),
            oscillation=float(np.linalg.norm(estimate.oscillation)),
            local_indicator=local.total,
            local_indicator_definition=(
                "nested local energy difference at fixed trace; diagnostic only"
            ),
            local_indicator_backend="process",
            local_indicator_workers=workers,
            parallel_assembly=True,
            estimator_backend="process",
            estimator_workers=workers,
            solve_and_estimator_seconds=solve_seconds,
            local_diagnostic_seconds=local_seconds,
            local_resolution_policy="four uniform triangles per macro, unchanged throughout",
            macro_balance_linf=float(np.max(abs(solution.conservation_residuals()))),
            continuous_equilibrium_l2_max=float(np.max(estimate.equilibrium_defect)),
            inflow=float(inflow),
            decision="macro" if reason == "continue" else "stop",
            stop_reason=reason,
            target_macro_cells=target_cells,
            source_hashes=fingerprint,
            source_changed_during_solve=any(
                hashlib.sha256(source_path(name).read_bytes()).hexdigest() != digest
                for name, digest in fingerprint.items()
            ),
            campaign_seconds=perf_counter() - started,
            **macro_quality(mesh),
        )
        records.append(row)
        target = output / "adaptive.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(target)
        print({key: value for key, value in row.items() if key != "source_hashes"}, flush=True)
        if reason != "continue":
            break
        mesh = remesh_freefem(mesh, metric.requested, executable=executable)
    return records


def main() -> None:
    """Acquire a separate multilevel campaign without executing it in the light CI suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "published")
    parser.add_argument("--levels", type=int, default=24)
    parser.add_argument("--target-cells", type=int, default=3786)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executable", default="FreeFem++")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    with threadpool_limits(1):
        acquire(
            options.output,
            levels=options.levels,
            target_cells=options.target_cells,
            workers=options.workers,
            executable=options.executable,
            resume=options.resume,
        )


if __name__ == "__main__":
    main()
