"""Material-fitted local/macro refinement for the SPE10 L09 energy estimator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.campaign_checkpoint import require_sources, verify_checkpoint
from examples.formulations.application import darcy as solve_equations
from examples.solve_spe10 import load_layer, pressure_boundary
from examples.spe10_adaptive import DATA, hashes, mesh_rectangle, natural_faces, source_path
from pymhm.adaptivity.darcy_balanced import solve_balanced_adaptive_darcy
from pymhm.estimators.darcy_local import estimate_darcy_local_refinement
from pymhm.fem.quadrature.material import fit_material_mesh
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.longest_edge import refine_longest_edge
from pymhm.meshes.refinement import refine_triangles
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class FittedLocals:
    """Fit once to each macro geometry, then uniformly red-refine that partition."""

    material: CartesianCellField

    def __call__(self, mesh: TriangleMesh, cell: int, refinement: int) -> TriangleMesh:
        """Preserve material edges and shape quality through a nested dyadic hierarchy."""
        if refinement < 1 or refinement & (refinement - 1):
            raise ValueError("fitted local refinement must be a power of two")
        fine = fit_material_mesh(mesh.submesh(cell, 1), self.material)
        for _ in range(refinement.bit_length() - 1):
            fine = refine_triangles(fine, np.ones(len(fine.cells), dtype=bool)).mesh
        return fine


def offsets(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    """Locate variable-sized local arrays in a non-object archive."""
    return np.r_[0, np.cumsum([len(part) for part in parts])]


def local_difference(solution: Any, *, workers: int = 8) -> float:
    """Measure local refinement at fixed trace, without using divergence as its proxy."""
    return estimate_darcy_local_refinement(
        solution, refinement_precision="extended", backend="process", workers=workers
    ).total


def acquisition_hashes() -> dict[str, str]:
    """Fingerprint the physical data, estimator and acquisition owners for continuation."""
    fingerprint = hashes()
    for name in (
        "examples/solve_spe10_balanced.py",
        "src/pymhm/adaptivity/darcy_balanced.py",
        "src/pymhm/estimators/darcy_local.py",
        "src/pymhm/estimators/darcy.py",
        "src/pymhm/core/contracts.py",
        "src/pymhm/execution/cpu.py",
        "src/pymhm/meshes/longest_edge.py",
        "examples/archive_precision.py",
        "examples/solve_spe10.py",
        "src/pymhm/fem/conditions.py",
        "src/pymhm/recovery/equilibrated.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/fem/scalar/operators.py",
    ):
        fingerprint[name] = hashlib.sha256(source_path(name).read_bytes()).hexdigest()
    return fingerprint


def acquire(
    levels: int,
    output: Path,
    *,
    initial_checkpoint: Path | None = None,
    resume: Path | None = None,
    macro_refinement: str = "longest-edge",
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Archive fitted P2 fields and the explicit local/macro balance decisions."""
    if levels < 1 or workers < 1 or macro_refinement not in {"longest-edge", "red-green"}:
        raise ValueError("require positive levels/workers and a supported macro refinement")
    if initial_checkpoint is not None and resume is not None:
        raise ValueError("initial_checkpoint and resume are mutually exclusive")
    output.mkdir(parents=True, exist_ok=True)
    refiner = refine_longest_edge if macro_refinement == "longest-edge" else refine_triangles
    fingerprint = acquisition_hashes()
    mesh = mesh_rectangle(16, 16)
    material = load_layer()
    started = perf_counter()
    records = []
    local_refinement, elapsed_offset = 1, 0.0
    continued_from = None
    if resume is not None:
        records, mesh, local_refinement = resume_checkpoint(
            resume, output, refiner, macro_refinement
        )
        continued_from = records[-1]["level"]
        elapsed_offset = records[-1]["campaign_seconds"]
        if levels <= len(records):
            return records
    if initial_checkpoint is not None:
        records, mesh = reuse_initial_checkpoint(initial_checkpoint, output, mesh, refiner)
        records[0]["next_macro_refinement"] = macro_refinement
        (output / "adaptive.json").write_text(json.dumps(records, indent=2) + "\n")
        if levels == 1:
            return records

    def checkpoint(balanced: Any) -> None:
        """Save one validated physical state before beginning the next solve."""
        level = len(records)
        solution = balanced.result.solutions[-1]
        estimate = balanced.result.estimators[-1]
        marked = balanced.result.marked[-1]
        coarse = solution.skeleton.mesh
        points = tuple(fine.points for fine in solution.local_meshes)
        cells = tuple(fine.cells for fine in solution.local_meshes)
        flux = estimate.reconstructed_flux.flux
        archive = f"mhm-level{level}.npz"
        np.savez_compressed(
            output / archive,
            macro_points=coarse.points,
            macro_cells=coarse.cells,
            local_points=np.concatenate(points),
            point_offsets=offsets(points),
            local_cells=np.concatenate(cells),
            cell_offsets=offsets(cells),
            **precision_fields("pressure", np.concatenate(solution.pressure)),
            pressure_offsets=offsets(solution.pressure),
            **precision_fields(
                "recovered_pressure", np.concatenate(estimate.potential.local_values)
            ),
            **precision_fields("reconstructed_flux", np.concatenate(flux)),
            flux_offsets=offsets(flux),
            trace=solution.hybrid.trace,
            local_squared=estimate.local_squared,
            flux_defect=estimate.flux_defect,
            nonconformity=estimate.nonconformity,
            divergence_defect=estimate.divergence_defect,
            oscillation=estimate.oscillation,
            marked=marked,
        )
        bottom = [
            face
            for face in coarse.boundary_faces
            if np.all(coarse.points[coarse.faces[face], 1] == 0)
        ]
        inflow = -sum(
            coarse.lengths[face] * solution.hybrid.trace[solution.skeleton.dofs(face)][0]
            for face in bottom
        )
        row = {
            "level": level,
            "archive": archive,
            "macro_triangles": len(coarse.cells),
            "fine_triangles": sum(len(fine.cells) for fine in solution.local_meshes),
            "local_degree": 2,
            "local_refinement": balanced.local_refinements[-1],
            "material_fitted": True,
            "nested_local_refinement": True,
            "local_refinement_precision": "extended",
            "trace_degree": 0,
            "reconstruction_degree": 2,
            "global_dofs": solution.skeleton.size + len(coarse.cells),
            "theta": 0.5,
            "local_error_ratio": 0.25,
            "local_indicator": balanced.local_indicators[-1],
            "local_indicator_backend": "process",
            "local_indicator_workers": workers,
            "macro_refinement": macro_refinement,
            **macro_quality(coarse),
            "archive_encoding": "float64 high + correction + tail",
            "archive_sha256": hashlib.sha256((output / archive).read_bytes()).hexdigest(),
            "local_indicator_definition": "nested local energy difference at fixed trace",
            "assembly_order": 6,
            "estimator_order": 6,
            "estimator_convention": "energy",
            "decision": balanced.decisions[-1],
            "stop_reason": balanced.stop_reason,
            "marked": int(marked.sum()),
            "estimator": estimate.total,
            "flux_defect": float(np.linalg.norm(estimate.flux_defect)),
            "nonconformity": float(np.linalg.norm(estimate.nonconformity)),
            "divergence_defect": float(np.linalg.norm(estimate.divergence_defect)),
            "oscillation": float(np.linalg.norm(estimate.oscillation)),
            "macro_balance_linf": float(np.max(abs(solution.conservation_residuals()))),
            "continuous_equilibrium_l2_max": float(np.max(estimate.equilibrium_defect)),
            "inflow": float(inflow),
            "source_hashes": fingerprint,
            "source_changed_during_solve": any(
                hashlib.sha256(source_path(name).read_bytes()).hexdigest() != digest
                for name, digest in fingerprint.items()
            ),
            "campaign_seconds": elapsed_offset + perf_counter() - started,
            **({"continued_from_level": continued_from} if continued_from is not None else {}),
        }
        records.append(row)
        print({key: value for key, value in row.items() if key != "source_hashes"}, flush=True)
        target = output / "adaptive.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(target)

    solve_balanced_adaptive_darcy(
        mesh,
        solve_step=solve_equations,
        iterations=levels - len(records),
        local_refinement=local_refinement,
        maximum_local_refinement=16,
        local_error_ratio=0.25,
        theta=0.5,
        maximum_cells=20000,
        trace_degree=0,
        reconstruction_degree=2,
        estimator_order=6,
        degree=2,
        local_mesh_factory=FittedLocals(material),
        local_error_indicator=partial(local_difference, workers=workers),
        macro_refiner=refiner,
        permeability=material,
        dirichlet=pressure_boundary,
        neumann=natural_faces(mesh),
        quadrature_order=6,
        local_refinement_precision="extended",
        backend="process",
        workers=workers,
        on_state=checkpoint,
    )
    return records


def macro_quality(mesh: TriangleMesh) -> dict[str, float]:
    """Record physical angle and diameter/area ratios independently of refinement templates."""
    vertices = mesh.points[mesh.cells]
    angles, lengths = [], []
    for index in range(3):
        a = vertices[:, (index + 1) % 3] - vertices[:, index]
        b = vertices[:, (index + 2) % 3] - vertices[:, index]
        lengths.append(np.linalg.norm(a, axis=1))
        cosine = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        angles.append(np.arccos(np.clip(cosine, -1, 1)))
    return {
        "minimum_macro_angle_degrees": float(np.min(angles) * 180 / np.pi),
        "maximum_macro_diameter_squared_over_area": float(
            np.max(np.max(lengths, axis=0) ** 2 / mesh.areas)
        ),
    }


def validate_state(row: dict[str, Any], directory: Path, current: dict[str, str]) -> None:
    """Verify one fitted energy-estimator state before reusing its refinement decision."""
    require_sources(row.get("source_hashes", {}), current)
    verify_checkpoint(
        row,
        dict(
            local_degree=2,
            trace_degree=0,
            reconstruction_degree=2,
            material_fitted=True,
            theta=0.5,
            local_error_ratio=0.25,
            assembly_order=6,
            estimator_order=6,
            estimator_convention="energy",
            source_changed_during_solve=False,
            local_refinement_precision="extended",
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


def reuse_initial_checkpoint(
    source: Path, output: Path, expected: TriangleMesh, refiner: Any
) -> tuple[list[dict[str, Any]], TriangleMesh]:
    """Reuse one verified initial field and apply only the selected next mesh refinement.

    The stored macro decision is required. P0 traces restrict to P0 on every
    inherited face, so the next scalar skeleton is reconstructed without an
    approximation-space change. The old physical coefficients and their source
    hashes are preserved byte for byte; this operation does not resolve the PDE.
    """
    row = json.loads((source / "adaptive.json").read_text())[0]
    validate_state(row, source, acquisition_hashes())
    if any(
        row.get(key) != value
        for key, value in {
            "level": 0,
            "local_degree": 2,
            "local_refinement": 1,
            "trace_degree": 0,
            "reconstruction_degree": 2,
            "material_fitted": True,
            "decision": "macro",
        }.items()
    ):
        raise ValueError("initial checkpoint must use the declared P2/P0 fitted initial state")
    archive = source / row["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("initial checkpoint field digest differs from its numerical record")
    material_name = "examples/results/spe10/layer-36.npz"
    if (
        row["source_hashes"][material_name]
        != hashlib.sha256(source_path(material_name).read_bytes()).hexdigest()
    ):
        raise ValueError("initial checkpoint material differs from the prescribed layer")
    with np.load(archive) as values:
        if not np.array_equal(values["macro_points"], expected.points) or not np.array_equal(
            values["macro_cells"], expected.cells
        ):
            raise ValueError("initial checkpoint geometry differs from the prescribed mesh")
        refined = refiner(expected, values["marked"])
    target = output / row["archive"]
    if archive.resolve() != target.resolve():
        shutil.copyfile(archive, target)
    row = {**row, **macro_quality(expected), "reused_initial_field": True}
    return [row], refined.mesh


def resume_checkpoint(
    source: Path, output: Path, refiner: Any, macro_refinement: str
) -> tuple[list[dict[str, Any]], TriangleMesh, int]:
    """Continue the last recorded decision without resolving or changing earlier fields.

    Check all archive digests, the material and the declared approximation and
    policy. A recorded macro decision applies its saved marking once; a local
    decision doubles the common local resolution. An iteration-budget stop can
    be continued using that same recorded balance rule. Physical data, polynomial
    spaces and the macro-refinement rule cannot change through this operation.
    """
    records = json.loads((source / "adaptive.json").read_text())
    material_name = "examples/results/spe10/layer-36.npz"
    material_digest = hashlib.sha256(source_path(material_name).read_bytes()).hexdigest()
    if not records:
        raise ValueError("resume requires at least one completed state")
    current = acquisition_hashes()
    for level, row in enumerate(records):
        validate_state(row, source, current)
        expected = {
            "level": level,
            "local_degree": 2,
            "trace_degree": 0,
            "reconstruction_degree": 2,
            "material_fitted": True,
            "theta": 0.5,
            "local_error_ratio": 0.25,
            "assembly_order": 6,
            "estimator_order": 6,
            "source_changed_during_solve": False,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("resume requires the unchanged fitted P2/P0 estimator policy")
        if row["source_hashes"][material_name] != material_digest:
            raise ValueError("resume material differs from the completed states")
        archive = source / row["archive"]
        if hashlib.sha256(archive.read_bytes()).hexdigest() != row["archive_sha256"]:
            raise ValueError("resume archive digest differs from its numerical record")
    last = records[-1]
    if last.get("macro_refinement", last.get("next_macro_refinement")) != macro_refinement:
        raise ValueError("resume cannot change the macro refinement policy")
    refinement = last["local_refinement"]
    decision = last["decision"]
    if decision == "stop" and last["stop_reason"] == "iterations":
        other = np.hypot(last["flux_defect"], last["nonconformity"])
        decision = "local" if last["local_indicator"] > 0.25 * other else "macro"
    with np.load(source / last["archive"]) as arrays:
        mesh = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
        if decision == "macro":
            mesh = refiner(mesh, arrays["marked"]).mesh
        elif decision == "local" and 2 * refinement <= 16:
            refinement *= 2
        else:
            raise ValueError("the recorded stopping decision does not permit continuation")
    for row in records:
        archive, target = source / row["archive"], output / row["archive"]
        if archive.resolve() != target.resolve():
            shutil.copyfile(archive, target)
    (output / "adaptive.json").write_text(json.dumps(records, indent=2) + "\n")
    return records, mesh, refinement


def main() -> None:
    """Acquire the declared fitted hierarchy separately from CI and visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DATA / "longest-edge")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--macro-refinement", choices=("longest-edge", "red-green"), default="longest-edge"
    )
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()
    with threadpool_limits(1):
        acquire(
            options.levels,
            options.output,
            initial_checkpoint=options.initial_checkpoint,
            resume=options.resume,
            macro_refinement=options.macro_refinement,
            workers=options.workers,
        )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_spe10_balanced").main()
