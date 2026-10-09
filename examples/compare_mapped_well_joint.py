"""Integrate several mapped-well comparisons jointly without changing their quadrature.

Run with ``python -m examples.compare_mapped_well_joint``. Results use a separate
output from the original pairwise campaign and checkpoint each integration order.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from examples.mapped_well_comparison import differences
from examples.mapped_well_fields import MappedWellField
from pymhm.execution.cpu import map_local
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import (
    case_workspace,
    read_resource_bytes,
    source_file,
    source_identity,
    source_label,
)

ROOT = case_workspace()
DIRECTORY = ROOT / "examples/results/mapped-well-oscillatory"
SOURCE_PATHS = (
    Path(__file__),
    source_file("examples/mapped_well_comparison.py", root=ROOT),
    source_file("examples/mapped_well_fields.py", root=ROOT),
    source_file("src/pymhm/_legacy/models/darcy/mapped.py", root=ROOT),
)


def _source_hashes() -> dict[str, str]:
    """Record the actual analysis and Piola-map implementation bytes."""
    return current_source_manifest(
        source_identity(ROOT, SOURCE_PATHS), packages=("pymhm", "examples")
    )


def _order_job(job: tuple) -> dict:
    """Evaluate one tensor Gauss order with native threads bounded to one."""
    order, reference, candidates, output, source_hashes, group_workers, group_backend = job
    started = perf_counter()
    fine, ref_record = MappedWellField.load(DIRECTORY / f"{reference}.json")
    loaded = [MappedWellField.load(DIRECTORY / f"{name}.json") for name in candidates]
    integration_started = perf_counter()

    def progress(done: int, count: int) -> None:
        """Expose bounded progress without inspecting or changing numerical sums."""
        if done == 1 or done % 128 == 0 or done == count:
            print(
                f"q{order}: {done}/{count} offset groups; {perf_counter() - started:.1f}s",
                flush=True,
            )

    with threadpool_limits(1):
        norms = differences(
            fine,
            [field for field, _ in loaded],
            (order, order, 3),
            pressure_offset=25e6,
            progress=progress,
            workers=group_workers,
            backend=group_backend,
        )
        pools = [{k: v for k, v in pool.items() if k != "filepath"} for pool in threadpool_info()]
    if source_hashes != _source_hashes():
        raise RuntimeError("physical-norm sources changed during acquisition")
    result = dict(
        order=order,
        reference=reference,
        reference_sha256=ref_record["sha256"],
        candidates=list(candidates),
        candidate_sha256=[record["sha256"] for _, record in loaded],
        norms=norms,
        integration_seconds=perf_counter() - integration_started,
        group_workers=group_workers,
        group_backend=group_backend,
        total_seconds=perf_counter() - started,
        native_threadpools=pools,
        source_hashes=source_hashes,
        source_changed_during_run=False,
    )
    checkpoint = output.with_name(f"{output.stem}-q{order}.json")
    checkpoint.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Acquire separately checkpointed, jointly integrated norms and their provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="classical-invariant-xy128-z1-q8")
    parser.add_argument(
        "--candidates", nargs="+", default=[f"fine8-macro{m}-s1-q40z10" for m in (1, 2, 4)]
    )
    parser.add_argument("--orders", type=int, nargs="+", default=[6, 8])
    parser.add_argument(
        "--kind",
        choices=(
            "MHM versus refined reference",
            "macro trace restriction",
            "classical spatial refinement",
            "classical fine-resolution gap",
            "material quadrature",
            "classical material quadrature",
            "MHM local refinement",
            "local-control trace restriction",
            "local-control refined reference",
        ),
        default="MHM versus refined reference",
        help="Approximation spaces compared; this label does not alter the physical norm.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--group-workers", type=int, default=4)
    parser.add_argument("--group-backend", choices=("thread", "process"), default="process")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/results/mapped-well-joint.json"
    )
    args = parser.parse_args()
    if (
        min(args.workers, args.group_workers) < 1
        or min(args.orders) < 1
        or len(set(args.orders)) != len(args.orders)
    ):
        parser.error("positive workers and distinct positive quadrature orders required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = _source_hashes()
    snapshots = args.output.parent / "mapped-well-joint-sources"
    snapshots.mkdir(exist_ok=True)
    for path in SOURCE_PATHS:
        (snapshots / f"{sources[source_label(path, ROOT)]}.py").write_bytes(
            read_resource_bytes(path)
        )
    jobs = [
        (
            order,
            args.reference,
            tuple(args.candidates),
            args.output,
            sources,
            args.group_workers,
            args.group_backend,
        )
        for order in args.orders
    ]
    acquired = map_local(
        _order_job, jobs, backend="serial" if args.workers == 1 else "process", workers=args.workers
    )
    if sources != _source_hashes():
        raise RuntimeError("physical-norm sources changed during acquisition")
    rows = [
        dict(
            kind=args.kind,
            reference=args.reference,
            candidate=candidate,
            reference_sha256=acquired[0]["reference_sha256"],
            candidate_sha256=acquired[0]["candidate_sha256"][index],
            norms={str(item["order"]): item["norms"][index] for item in acquired},
            denominator="physical norm of the explicitly named reference field",
            geometry="exact nested tensor hierarchy; integration respects every fine hexahedron",
        )
        for index, candidate in enumerate(args.candidates)
    ]
    result = dict(
        method="Native Piola RT1/Q1 physical L2 norms with joint geometry/reference reuse",
        pressure_offset=25e6,
        reference=args.reference,
        units="m, Pa, s",
        rows=rows,
        acquisitions=acquired,
        source_hashes=sources,
        source_changed_during_run=False,
        workers=args.workers,
        group_workers=args.group_workers,
        group_backend=args.group_backend,
        python=platform.python_version(),
        numpy=np.__version__,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.compare_mapped_well_joint").main()
