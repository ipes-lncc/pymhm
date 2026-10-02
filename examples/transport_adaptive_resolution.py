"""Measure local P1 resolution on one fixed, literally marked L11 face space."""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.transport_checkpoints import checkpoint_field, checkpoint_norm
from examples.transport_coefficient_controls import norm_contribution
from examples.transport_face_resolution import local_bound
from examples.transport_mixed_campaign import SOURCES, continuity_moments
from examples.transport_trace_family import TransportTraceFamily
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.parallel import map_local
from pymhm.scalar_adaptive import TransportBounds, estimate_transport_faces

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/transport"


def acquire(refinement: int, workers: int) -> dict:
    """Keep all final r16 adaptive breakpoints while changing only local subdivision."""
    input_record = DATA / "mixed-face-adaptive-e1-r16.json"
    initial = json.loads(input_record.read_text())["records"][-1]
    input_field = DATA / initial["archive"]
    if hashlib.sha256(input_field.read_bytes()).hexdigest() != initial["archive_sha256"]:
        raise ValueError("the fixed adaptive field archive differs from its record")
    with np.load(input_field) as values:
        mesh = TriangleMesh(values["macro_points"], values["macro_cells"])
        breaks = values["trace_breaks"].copy()
        offsets = values["trace_break_offsets"].copy()
    skeleton = SkeletonSpace(
        mesh,
        tuple(
            FaceSpace(tuple(breaks[a:b]), (0,) * (b - a - 1))
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
        ),
    )
    paths = (
        *SOURCES,
        "examples/transport_adaptive_resolution.py",
        "examples/transport_checkpoints.py",
        "examples/transport_trace_family.py",
        "examples/transport_face_resolution.py",
        "examples/transport_coefficient_controls.py",
        "examples/unfitted_trace_family.py",
        "src/pymhm/parallel.py",
    )
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    backend = "process" if workers > 1 else "serial"
    with threadpool_limits(1):
        # All accepted adaptive knots lie on eighths; the exact injection checks this.
        family = TransportTraceFamily.prepare(
            mesh, segments=8, local_refinement=refinement, workers=workers
        )
        solution, physical = family.solve(skeleton)
        print("recovered", refinement, physical, flush=True)
        indicator = estimate_transport_faces(solution, TransportBounds(1, 1, 0))
        moment = continuity_moments(solution)
        name = f"mixed-adaptive-fixed-e1-r{refinement}"
        archive = DATA / f"{name}.npz"
        checkpoint = checkpoint_field(
            archive,
            dict(
                macro_points=mesh.points,
                macro_cells=mesh.cells,
                local_points=np.stack([m.points for m in solution.local_meshes]),
                local_cells=np.stack([m.cells for m in solution.local_meshes]),
                coefficients=np.stack(solution.values),
                trace=solution.hybrid.trace,
                trace_breaks=breaks,
                trace_break_offsets=offsets,
                segment_indicators=np.concatenate(indicator.values),
            ),
            dict(
                epsilon=1.0,
                local_refinement=refinement,
                free_trace_dofs=initial["free_trace_dofs"],
                source_hashes=hashes,
                physical_checks=physical,
            ),
        )
        measurements, bounds = {}, {}
        for order in (8, 12):
            contributions = map_local(
                norm_contribution,
                [
                    (fine, coefficients, 1.0, order)
                    for fine, coefficients in zip(
                        solution.local_meshes, solution.values, strict=True
                    )
                ],
                backend=backend,
                workers=workers,
            )
            squared = np.zeros(2)
            for contribution in contributions:
                squared += contribution
            measurements[str(order)] = dict(
                l2_error=float(np.sqrt(squared[0])),
                broken_h1_error=float(np.sqrt(squared[1])),
            )
            checkpoint_norm(archive, checkpoint, order, measurements[str(order)])
            values = map_local(
                partial(local_bound, order=order),
                solution.local_meshes,
                backend=backend,
                workers=workers,
            )
            bounds[str(order)] = float(np.sqrt(sum(values)))
    if not all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items()):
        raise RuntimeError("a source changed during the local-resolution acquisition")
    report = dict(
        epsilon=1.0,
        local_refinement=refinement,
        local_degree=1,
        trace_degree=0,
        free_trace_dofs=initial["free_trace_dofs"],
        fixed_skeleton_from=initial["archive"],
        fixed_skeleton_archive_sha256=initial["archive_sha256"],
        interpretation=(
            "Local refinement of the same final maximum-marked r16 face space; "
            "this is not a rerun of the adaptive sequence or a historical mesh identification."
        ),
        quadrature=measurements,
        gradient_dg0_projection_error=bounds,
        face_indicator=indicator.total,
        maximum_continuity_moment=moment,
        physical_checks=physical,
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_hashes=hashes,
        source_changed_during_run=False,
    )
    archive.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, measurements["12"], flush=True)
    return report


def main() -> None:
    """Acquire one explicitly chosen fixed-skeleton local control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-refinement", type=int, choices=(32, 64, 128), required=True)
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("workers must be positive")
    acquire(options.local_refinement, options.workers)


if __name__ == "__main__":
    main()
