"""Integrate fixed-macro SPE10 resolution controls against the archived RT2 reference."""

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

from examples.spe10_adaptive import DATA, StructuredRT
from examples.spe10_adaptive_norms import BrokenP2, integrated_squared_norms, norm_record
from pymhm.io.workspace import local_resource, read_resource_bytes, source_file

_FIELDS: tuple[BrokenP2, StructuredRT, int] | None = None
_THREAD_LIMIT: Any = None


def initialize(mhm: str, reference: str, order: int) -> None:
    """Load immutable physical coefficient arrays once inside each spawn worker."""
    global _FIELDS, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(1)
    _FIELDS = (BrokenP2(Path(mhm)), StructuredRT.load(Path(reference)), order)


def integrate(cells: np.ndarray) -> np.ndarray:
    """Use the shared exact-overlay integrator on one disjoint macrocell group."""
    if _FIELDS is None:
        raise RuntimeError("initialize physical fields before integration")
    mhm, reference, order = _FIELDS
    return integrated_squared_norms(mhm, reference, order, cells=cells)


def acquire(
    archive: Path,
    reference: Path,
    output: Path,
    *,
    orders: tuple[int, ...] = (4,),
    workers: int = 8,
) -> dict[str, Any]:
    """Save integrated norms with input digests and per-order progress checkpoints."""
    if workers < 1 or not orders or min(orders) < 4:
        raise ValueError("require positive workers and norm orders at least four")
    with np.load(local_resource(archive)) as values:
        count = len(values["macro_cells"])
    groups = [np.arange(start, min(start + 32, count)) for start in range(0, count, 32)]
    sources = (
        Path(__file__),
        Path(__file__).with_name("spe10_adaptive_norms.py"),
        Path(__file__).with_name("spe10_adaptive.py"),
        source_file("src/pymhm/fem/hdiv/rt.py"),
    )
    fingerprint = {
        path.name: hashlib.sha256(read_resource_bytes(path)).hexdigest() for path in sources
    }
    result: dict[str, Any] = {
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(read_resource_bytes(archive)).hexdigest(),
        "reference": reference.name,
        "reference_sha256": hashlib.sha256(read_resource_bytes(reference)).hexdigest(),
        "integration": "exact intersections of local triangles, reference triangles and pixels",
        "denominator": "corresponding norm of the stated classical RT2 reference",
        "source_hashes": fingerprint,
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for order in orders:
        started = perf_counter()
        partials = []
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(archive), str(reference), order),
        ) as executor:
            for index, value in enumerate(executor.map(integrate, groups)):
                partials.append(value)
                if (index + 1) % 16 == 0 or index + 1 == len(groups):
                    print(
                        dict(order=order, completed_groups=index + 1, total_groups=len(groups)),
                        flush=True,
                    )
        totals = np.sum(partials, axis=0, dtype=np.longdouble)
        result["rows"].append({**norm_record(totals, order), "seconds": perf_counter() - started})
        result["source_changed_during_run"] = any(
            hashlib.sha256(read_resource_bytes(path)).hexdigest() != fingerprint[path.name]
            for path in sources
        )
        if result["source_changed_during_run"]:
            raise RuntimeError("integration source changed during acquisition")
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(result["rows"][-1], flush=True)
    return result


def main() -> None:
    """Compare one saved numerical control without resolving either PDE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", type=Path, default=DATA / "reference-rt2-240x880.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--orders", type=int, nargs="+", default=[4])
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()
    with threadpool_limits(1):
        acquire(
            options.archive,
            options.reference,
            options.output or options.archive.with_name(options.archive.stem + "-norms.json"),
            orders=tuple(options.orders),
            workers=options.workers,
        )


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.compare_spe10_resolution").main()
