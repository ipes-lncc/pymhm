"""Separate finite P1 local resolution from nested L11 face enrichment."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.pgmhm_campaign import crisscross
from examples.transport_checkpoints import checkpoint_field, checkpoint_norm
from examples.transport_coefficient_controls import norm_contribution
from examples.transport_mixed_campaign import SOURCES
from examples.transport_trace_family import TransportTraceFamily, gradient_projection_squared
from pymhm.execution.cpu import map_local
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/transport"


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Return the gradient of the independent section 5.1 solution with epsilon=1."""
    gradient = np.zeros_like(points)
    gradient[:, 0] = 1 - np.exp(points[:, 0] - 1) / -np.expm1(-1)
    return gradient


def local_bound(mesh: TriangleMesh, *, order: int) -> float:
    """Evaluate the DG0 gradient lower bound on one macrocell's actual fine partition."""
    return gradient_projection_squared(mesh, exact_gradient, order)


def acquire(refinement: int, workers: int, *, endpoint_only: bool = False) -> dict:
    """Measure five nested restrictions or only their common finest s=16 endpoint.

    An endpoint record is separate from the five-field family. Both modes use
    the same P1/Galerkin local operator, P0 trace and physical norm quadratures.
    """
    paths = (
        *SOURCES,
        "examples/transport_face_resolution.py",
        "examples/transport_checkpoints.py",
        "examples/campaign_provenance.py",
        "examples/transport_trace_family.py",
        "examples/unfitted_trace_family.py",
        "examples/transport_coefficient_controls.py",
        "src/pymhm/execution/cpu.py",
    )
    hashes = current_source_manifest(
        {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    )
    start = perf_counter()
    macro = crisscross(8)
    backend = "process" if workers > 1 else "serial"
    with threadpool_limits(1):
        family = TransportTraceFamily.prepare(
            macro, segments=16, local_refinement=refinement, workers=workers
        )
        print("prepared", refinement, perf_counter() - start, flush=True)
        bounds = {}
        for order in (8, 12):
            values = map_local(
                partial(local_bound, order=order),
                [cell.mesh for cell in family.cells],
                backend=backend,
                workers=workers,
            )
            bounds[str(order)] = float(np.sqrt(sum(values)))
        report = dict(
            epsilon=1.0,
            macro_resolution=8,
            local_refinement=refinement,
            local_degree=1,
            trace_degree=0,
            prepared_segments=16,
            gradient_dg0_projection_error=bounds,
            interpretation=(
                "Uniform face subdivision is a control, not the literal maximum-indicator rule. "
                "Epsilon=1 is the published Figure7 regime; Figure12 prints epsilon=0.1. "
                "The DG0 gradient projection gives a separately integrated lower bound for "
                "P1 error on these local meshes; it does not identify historical local meshes."
            ),
            source_hashes=hashes,
            records=[],
        )
        for segments in (16,) if endpoint_only else (1, 2, 4, 8, 16):
            skeleton = SkeletonSpace(
                macro, tuple(FaceSpace.uniform(0, segments) for _ in macro.faces)
            )
            solution, checks = family.solve(skeleton)
            print("recovered", refinement, segments, checks, flush=True)
            name = f"mixed-face-uniform-e1-{segments}-r{refinement}"
            archive = DATA / f"{name}.npz"
            checkpoint = checkpoint_field(
                archive,
                dict(
                    macro_points=macro.points,
                    macro_cells=macro.cells,
                    local_points=np.stack([m.points for m in solution.local_meshes]),
                    local_cells=np.stack([m.cells for m in solution.local_meshes]),
                    coefficients=np.stack(solution.values),
                    trace=solution.hybrid.trace,
                    trace_breaks=np.concatenate([np.asarray(f.breaks) for f in skeleton.faces]),
                    trace_break_offsets=np.r_[
                        0, np.cumsum([len(f.breaks) for f in skeleton.faces])
                    ],
                ),
                dict(
                    epsilon=1.0,
                    local_refinement=refinement,
                    segments=segments,
                    source_hashes=hashes,
                    physical_checks=checks,
                    retained_constant_coordinates=int(family.offsets[-1]),
                ),
            )
            measurements = {}
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
            assert all(
                hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items()
            )
            row = dict(
                name=name,
                epsilon=1.0,
                local_refinement=refinement,
                segments=segments,
                free_trace_dofs=368 * segments,
                retained_constant_coordinates=int(family.offsets[-1]),
                quadrature=measurements,
                physical_checks=checks,
                archive=archive.name,
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                source_hashes=hashes,
                source_changed_during_run=False,
            )
            archive.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
            report["records"].append(row)
            report["source_changed_during_run"] = False
            report["seconds"] = perf_counter() - start
            family_name = "endpoint" if endpoint_only else "uniform"
            (DATA / f"mixed-face-{family_name}-e1-r{refinement}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(name, measurements["12"], flush=True)
    return report


def main() -> None:
    """Acquire explicit local resolutions independently of the original finite-r16 study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-refinement", type=int, choices=(16, 32, 64, 128, 256, 512), required=True
    )
    parser.add_argument(
        "--endpoint-only",
        action="store_true",
        help="Solve only s=16, preserving its field and two physical norm quadratures",
    )
    parser.add_argument("--workers", type=int, default=4)
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("workers must be positive")
    DATA.mkdir(parents=True, exist_ok=True)
    acquire(options.local_refinement, options.workers, endpoint_only=options.endpoint_only)


if __name__ == "__main__":
    main()
