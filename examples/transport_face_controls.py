"""Separate uniform face subdivision from the literal maximum-indicator L11 rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import transport as solve_transport
from examples.pgmhm_campaign import crisscross
from examples.transport_campaign import natural_horizontal
from examples.transport_coefficient_controls import norm_contribution
from examples.transport_mixed_campaign import SOURCES
from pymhm.adaptivity.transport import (
    TransportBounds,
    estimate_transport_faces,
    refine_skeleton_faces,
)
from pymhm.execution.cpu import map_local
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file

ROOT = case_workspace()
DATA = ROOT / "examples/results/transport"


def acquire(skeleton: SkeletonSpace, name: str, *, refinement: int, workers: int) -> tuple:
    """Measure one declared epsilon=1/P1/P0 control and its exact marking decisions."""
    start = perf_counter()
    paths = (
        *SOURCES,
        "examples/transport_face_controls.py",
        "examples/transport_coefficient_controls.py",
        "src/pymhm/execution/cpu.py",
    )
    hashes = current_source_manifest(
        {p: hashlib.sha256(source_file(p, root=ROOT).read_bytes()).hexdigest() for p in paths},
        packages=("pymhm", "examples"),
    )
    mesh = skeleton.mesh
    with threadpool_limits(1):
        solution = solve_transport(
            mesh,
            skeleton=skeleton,
            diffusion=1.0,
            velocity=(1, 0),
            source=1,
            dirichlet=0,
            diffusive_flux=natural_horizontal(mesh),
            dirichlet_enforcement="strong",
            degree=1,
            local_refinement=refinement,
            stabilization="galerkin",
            quadrature_order=5,
            coarse_space="constants",
            backend="process" if workers > 1 else "serial",
            workers=workers,
        )
        print("assembled", name, solution.hybrid.residual, flush=True)
        measurements = {}
        for order in (8, 12):
            tasks = [
                (fine, coef, 1.0, order)
                for fine, coef in zip(solution.local_meshes, solution.values, strict=True)
            ]
            contributions = map_local(
                norm_contribution,
                tasks,
                backend="process" if workers > 1 else "serial",
                workers=workers,
            )
            squared = np.zeros(2)
            for value in contributions:
                squared += value
            measurements[str(order)] = dict(
                l2_error=float(np.sqrt(squared[0])), broken_h1_error=float(np.sqrt(squared[1]))
            )
        indicator = estimate_transport_faces(solution, TransportBounds(1, 1, 0))
        marked = indicator.mark(0.75)
    archive = DATA / f"{name}.npz"
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.stack([m.points for m in solution.local_meshes]),
        local_cells=np.stack([m.cells for m in solution.local_meshes]),
        coefficients=np.stack(solution.values),
        trace=solution.hybrid.trace,
        trace_breaks=np.concatenate([np.asarray(f.breaks) for f in skeleton.faces]),
        trace_break_offsets=np.r_[0, np.cumsum([len(f.breaks) for f in skeleton.faces])],
        segment_indicators=np.concatenate(indicator.values),
        marked_segments=np.concatenate(marked),
    )
    assert all(
        hashlib.sha256(source_file(p, root=ROOT).read_bytes()).hexdigest() == h
        for p, h in hashes.items()
    )
    row = dict(
        name=name,
        epsilon=1.0,
        macro_resolution=8,
        macro_triangles=len(mesh.cells),
        local_refinement=refinement,
        local_degree=1,
        trace_degree=0,
        free_trace_dofs=sum(
            len(skeleton.dofs(int(f))) for f in np.flatnonzero(mesh.face_cells[:, 1] >= 0)
        ),
        retained_constant_coordinates=sum(len(v) for v in solution.hybrid.coarse),
        face_indicator=indicator.total,
        marked_segment_count=sum(np.count_nonzero(m) for m in marked),
        maximum_segments=max(len(f.degrees) for f in skeleton.faces),
        quadrature=measurements,
        original_hybrid_residual=solution.hybrid.residual,
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_hashes=hashes,
        source_changed_during_run=False,
        seconds=perf_counter() - start,
    )
    row["marked_segment_count"] = int(row["marked_segment_count"])
    archive.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    print(name, row["free_trace_dofs"], measurements["12"], flush=True)
    return row, marked


def main() -> None:
    """Preserve local meshes and distinguish uniform and maximum-marked face spaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["uniform", "adaptive"], required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--local-refinement", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    if min(options.iterations, options.local_refinement, options.workers) < 1:
        parser.error("iteration count, local refinement and workers must be positive")
    DATA.mkdir(parents=True, exist_ok=True)
    macro = crisscross(8)
    skeleton = SkeletonSpace(macro)
    report = dict(
        strategy=options.strategy,
        theta=0.75,
        interpretation=(
            "epsilon=1 is a separate Figure7 regime; Figure12 prints epsilon=0.1. "
            "Uniform subdivision is a control, not the literal maximum-indicator rule."
        ),
        records=[],
    )
    sequence = (1, 2, 4, 8, 16) if options.strategy == "uniform" else range(options.iterations)
    for step in sequence:
        if options.strategy == "uniform":
            skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(0, step) for _ in macro.faces))
        name = f"mixed-face-{options.strategy}-e1-{step}-r{options.local_refinement}"
        row, marked = acquire(
            skeleton, name, refinement=options.local_refinement, workers=options.workers
        )
        row["step"] = step
        report["records"].append(row)
        (DATA / f"mixed-face-{options.strategy}-e1-r{options.local_refinement}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if options.strategy == "adaptive":
            skeleton = refine_skeleton_faces(skeleton, marked)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.transport_face_controls").main()
