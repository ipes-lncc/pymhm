"""Pixel-aligned refinement hierarchy for the classical RT2 SPE10 reference."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from examples.formulations.mixed_darcy import conforming_rt_reference as solve_darcy_rt_conforming
from examples.solve_spe10 import load_layer, pressure_boundary
from examples.spe10_adaptive import (
    DATA,
    ROOT,
    StructuredRT,
    hashes,
    mesh_rectangle,
    natural_faces,
    reference_integrals,
)


def peak_resident_memory_kib() -> int | None:
    """Report native process peak RSS where the platform provides resource statistics."""
    try:
        import resource
    except ImportError:
        return None
    import sys

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value / 1024 if sys.platform == "darwin" else value)


def collect(
    resolutions: list[int], *, solver: str = "scipy", previous_archive: Path | None = None
) -> list[dict[str, Any]]:
    """Acquire a dyadic series whose 60x220 and finer grids resolve every pixel edge."""
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    previous = None if previous_archive is None else StructuredRT.load(previous_archive)
    record = DATA / "references-aligned.json"
    if previous is not None and record.exists():
        rows = [row for row in json.loads(record.read_text()) if row["nx"] <= previous.nx]
    for nx in resolutions:
        if nx % 3:
            raise ValueError("nx must be divisible by three for the declared pixel hierarchy")
        ny = 11 * nx // 3
        mesh = mesh_rectangle(nx, ny)
        fingerprint = hashes()
        fingerprint["examples/spe10_adaptive_aligned.py"] = hashlib.sha256(
            (ROOT / "examples/spe10_adaptive_aligned.py").read_bytes()
        ).hexdigest()
        started = perf_counter()
        solution = solve_darcy_rt_conforming(
            mesh,
            degree=2,
            permeability=load_layer(),
            dirichlet=pressure_boundary,
            neumann=natural_faces(mesh),
            quadrature_order=5,
            solver=solver,
        )
        elapsed = perf_counter() - started
        current = StructuredRT(nx, ny, solution.pressure[0], solution.flux[0])
        archive = f"reference-rt2-{nx}x{ny}.npz"
        np.savez_compressed(
            DATA / archive,
            nx=nx,
            ny=ny,
            pressure=current.pressure,
            flux=current.flux,
            points=mesh.points,
            cells=mesh.cells,
        )
        moments = current.flux[3 * mesh.cell_faces] * mesh.signs
        bottom = [
            face for face in mesh.boundary_faces if np.all(mesh.points[mesh.faces[face], 1] == 0)
        ]
        row = {
            "nx": nx,
            "ny": ny,
            "cells": len(mesh.cells),
            "dofs": current.pressure.size + current.flux.size,
            "degree": 2,
            "assembly_order": 5,
            "solver": solver,
            "material_aligned": nx % 60 == 0,
            "solve_seconds": elapsed,
            "peak_process_rss_kib": peak_resident_memory_kib(),
            "residual": solution.residual,
            "archive": archive,
            "archive_sha256": hashlib.sha256((DATA / archive).read_bytes()).hexdigest(),
            "fine_balance_linf": float(np.max(abs(moments.sum(axis=1)))),
            "inflow": -float(current.flux[3 * np.array(bottom)].sum()),
            "native_threadpools": [
                {**entry, "filepath": Path(entry["filepath"]).name} for entry in threadpool_info()
            ],
            "source_hashes": fingerprint,
            "source_changed_during_solve": any(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                for name, digest in fingerprint.items()
            ),
        }
        # Preserve the solved physical fields and their provenance before postprocessing.
        checkpoint = DATA / f"reference-rt2-{nx}x{ny}-acquisition.json"
        checkpoint.write_text(json.dumps(row, indent=2) + "\n")
        print(f"Archived {archive}; integrating physical refinement norms", flush=True)
        row.update(reference_integrals(current, previous))
        if previous is not None:
            row["independent_order6"] = reference_integrals(current, previous, order=6)
        rows.append(row)
        record.write_text(json.dumps(rows, indent=2) + "\n")
        print({key: value for key, value in row.items() if key != "source_hashes"}, flush=True)
        previous = current
    return rows


def main() -> None:
    """Run reference refinement separately from light tests and the adaptive MHM solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[15, 30, 60, 120, 240])
    parser.add_argument(
        "--solver",
        choices=("scipy", "pypardiso", "pypardiso-symmetric", "pypardiso-symmetric-matching"),
        default="scipy",
    )
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--previous-archive", type=Path)
    options = parser.parse_args()
    if options.native_threads < 1:
        parser.error("--native-threads must be positive")
    with threadpool_limits(options.native_threads):
        collect(
            options.resolutions, solver=options.solver, previous_archive=options.previous_archive
        )


if __name__ == "__main__":
    main()
