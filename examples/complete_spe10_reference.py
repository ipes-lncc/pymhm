"""Complete physical refinement norms from an archived SPE10 reference acquisition."""

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

from examples.solve_spe10 import load_layer
from examples.spe10_adaptive import (
    DATA,
    ROOT,
    StructuredRT,
    reference_norm_record,
    reference_squared_batch,
)
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import local_resource, read_resource_bytes, read_resource_text, source_file

_FIELDS: tuple[StructuredRT, StructuredRT, Any] | None = None
_LIMIT: Any = None


def initialize(current: str, previous: str) -> None:
    """Load immutable physical fields and the material once in each spawn worker."""
    global _FIELDS, _LIMIT
    _LIMIT = threadpool_limits(1)
    _FIELDS = StructuredRT.load(Path(current)), StructuredRT.load(Path(previous)), load_layer()


def integrate(task: tuple[int, int, int]) -> np.ndarray:
    """Use the original 512-cell batch reduction with no change to quadrature."""
    if _FIELDS is None:
        raise RuntimeError("initialize reference fields before integrating")
    current, previous, material = _FIELDS
    first, last, order = task
    return reference_squared_batch(current, previous, order, np.arange(first, last), material)


def complete(nx: int, previous: Path, output: Path, workers: int) -> dict[str, Any]:
    """Preserve acquisition attribution and separately fingerprint physical postprocessing."""
    if workers < 1:
        raise ValueError("workers must be positive")
    ny = 11 * nx // 3
    archive = DATA / f"reference-rt2-{nx}x{ny}.npz"
    acquisition = archive.with_name(archive.stem + "-acquisition.json")
    row = json.loads(read_resource_text(acquisition))
    if hashlib.sha256(read_resource_bytes(archive)).hexdigest() != row["archive_sha256"]:
        raise ValueError("reference acquisition checksum mismatch")
    with np.load(local_resource(previous)) as arrays:
        if nx != 2 * int(arrays["nx"]) or ny != 2 * int(arrays["ny"]):
            raise ValueError("reference meshes must form the stated isotropic dyadic hierarchy")
    sources = [
        "examples/complete_spe10_reference.py",
        "examples/spe10_adaptive.py",
        "examples/solve_spe10.py",
        "src/pymhm/fem/hdiv/rt.py",
        "src/pymhm/fem/quadrature/material.py",
        "src/pymhm/meshes/triangle.py",
        "examples/results/spe10/layer-36.npz",
    ]
    hashes = current_source_manifest(
        {
            name: hashlib.sha256(read_resource_bytes(source_file(name, root=ROOT))).hexdigest()
            for name in sources
        },
        packages=("pymhm", "examples"),
    )
    row["norm_source_hashes"] = hashes
    row["previous_archive"] = previous.name
    row["previous_archive_sha256"] = hashlib.sha256(read_resource_bytes(previous)).hexdigest()
    row["norm_reduction"] = "original ordered 512-cell batches, same quadrature and wider sums"
    row["norm_workers"] = workers
    output.parent.mkdir(parents=True, exist_ok=True)
    for order in (5, 6):
        started = perf_counter()
        tasks = [
            (first, min(first + 512, row["cells"]), order) for first in range(0, row["cells"], 512)
        ]
        partials = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(archive), str(previous)),
        ) as executor:
            for index, values in enumerate(executor.map(integrate, tasks)):
                partials.append(values)
                if (index + 1) % 128 == 0 or index + 1 == len(tasks):
                    print(dict(order=order, groups=index + 1, total=len(tasks)), flush=True)
        norms = reference_norm_record(np.sqrt(np.sum(partials, axis=0, dtype=np.longdouble)), order)
        if order == 5:
            row.update(norms)
        else:
            row["independent_order6"] = norms
        row[f"order{order}_norm_seconds"] = perf_counter() - started
        row["norm_source_changed_during_run"] = any(
            hashlib.sha256(read_resource_bytes(source_file(name, root=ROOT))).hexdigest() != digest
            for name, digest in hashes.items()
        )
        if row["norm_source_changed_during_run"]:
            raise RuntimeError("physical norm source changed during acquisition")
        output.write_text(json.dumps(row, indent=2) + "\n")
        print(norms, flush=True)
    return row


def main() -> None:
    """Evaluate physical norms without repeating assembly or factorization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=480)
    parser.add_argument("--previous", type=Path, default=DATA / "reference-rt2-240x880.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    complete(args.nx, args.previous, args.output, args.workers)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.complete_spe10_reference").main()
