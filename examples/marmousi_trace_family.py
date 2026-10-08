"""Acquire all five polynomial traces from one Marmousi local condensation.

For a fixed H, every degree uses the same Q3 local mesh, coefficient samples,
point source and quadrature order nine. Only the global skeleton is restricted;
the original local matrices and response maps are shared throughout.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.helmholtz_compact_family import CompactFamily
from examples.helmholtz_field_store import write_coefficients
from examples.helmholtz_response_store import ResponseStore
from examples.marmousi_campaign import evaluate_fields, source_hashes
from examples.marmousi_data import load_marmousi_crop
from examples.tutorial_helmholtz_equations import AcousticAssemblyProvider
from pymhm.fem.loads import split_point_sources
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.io.provenance import file_digest
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]


def hashes() -> dict[str, str]:
    """Include the response-restriction owner alongside the acquisition kernels."""
    result = source_hashes()
    for name in (
        "examples/marmousi_trace_family.py",
        "examples/helmholtz_trace_family.py",
        "examples/helmholtz_compact_family.py",
        "examples/helmholtz_response_store.py",
        "examples/helmholtz_field_store.py",
        "examples/local_response_cache.py",
        "examples/campaign_checkpoint.py",
        "examples/campaign_provenance.py",
    ):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def main() -> None:
    """Condense once, save each complete field, and release each restricted field in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--H", type=int, choices=(20, 40, 80), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--solver", choices=("scipy", "pypardiso"), default="scipy")
    parser.add_argument("--local-solver", choices=("scipy", "pypardiso"), default="scipy")
    parser.add_argument(
        "--local-refinement-precision", choices=("double", "extended"), default="double"
    )
    parser.add_argument("--response-store", type=Path)
    parser.add_argument("--response-batch-size", type=int, default=128)
    parser.add_argument("--field-store", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/marmousi/family")
    args = parser.parse_args()
    material = load_marmousi_crop(args.data)
    mesh = CartesianMacroMesh(10240 // args.H, 2560 // args.H, (0, 10240, 0, 2560))
    skeleton = helmholtz_skeleton(mesh, 40 * np.pi, degree=4)
    absorbing = {
        int(face): 0j
        for face in mesh.boundary_faces
        if not np.all(mesh.points[mesh.faces[face], 1] == 0)
    }
    before, started = hashes(), time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        factory = AcousticAssemblyProvider(
            mesh,
            skeleton,
            40 * np.pi,
            3,
            args.H // 5 * 2,
            9,
            material.density,
            material.bulk_modulus,
            0j,
            split_point_sources(mesh, ((5000, 50, 1.0),)),
            0j,
            absorbing,
            {},
            None,
        )
        execution = dict(
            backend="process" if args.workers > 1 else "serial",
            workers=args.workers,
            local_solver=args.local_solver,
            local_refinement_precision=args.local_refinement_precision,
        )
        if args.response_store is None:
            prepared = CompactFamily.prepare(factory, **execution)
            local_storage_bytes = sum(item.storage_bytes for item in prepared.local)
            response_acquisition_seconds = None
        else:
            store = ResponseStore.prepare(
                factory,
                args.response_store,
                sources=before,
                configuration={
                    "material": material.provenance,
                    "H_m": args.H,
                    "bounds_m": [0, 10240, 0, 2560],
                    "local_degree": 3,
                    "local_refinement": args.H // 5 * 2,
                    "trace_degree": 4,
                    "assembly_order": 9,
                    "omega": float(40 * np.pi),
                    "point_source": [5000, 50, 1.0],
                    "source_allocation": "incident angle",
                    "boundary": "Top Dirichlet zero; other sides outgoing absorption zero",
                },
                batch_size=args.response_batch_size,
                **execution,
            )
            prepared = CompactFamily.from_locals(skeleton, store)
            local_storage_bytes = store.storage_bytes
            response_acquisition_seconds = store.acquisition_seconds
        preparation_seconds = time.perf_counter() - started
        print(json.dumps({"prepared_H": args.H, "seconds": preparation_seconds}), flush=True)
        x, y = np.meshgrid(np.linspace(0, 10240, 513), np.linspace(0, 2560, 129))
        points = np.column_stack((x.ravel(), y.ravel()))
        sides = ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for degree in (4, 0, 1, 2, 3):
            stage_started = time.perf_counter()
            stored = {}
            if args.field_store:
                solved = prepared.solve_trace(degree, solver=args.solver)
                coefficient_path = args.output / f"mhm-H{args.H}-ell{degree}-coefficients.npy"
                stored = write_coefficients(
                    prepared, solved, coefficient_path, cell_count=len(mesh.cells)
                )
                coefficients = np.load(coefficient_path, mmap_mode="r", allow_pickle=False)
                trace = solved.trace
                residual = solved.residual
                macro_balance = stored["macro_balance_max"]
                local_residual = stored["local_equation_residual_max"]
            else:
                solution = prepared.solve(degree, solver=args.solver)
                trace = solution.trace
                coefficients = np.asarray(solution.pressure)
                residual = solution.residual
                macro_balance = float(np.max(abs(solution.balance)))
                local_residual = solution.local_residual_max
                stored["original_field_trace_residual"] = solution.original_trace_residual
            archive = args.output / f"mhm-H{args.H}-ell{degree}-q9.npz"
            try:
                samples = np.array(
                    [
                        evaluate_fields(coefficients, mesh, args.H // 5 * 2, 3, points, side=side)
                        for side in sides
                    ]
                )
                np.savez_compressed(
                    archive,
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                    pressure=coefficients,
                    trace=trace,
                    sample_points=points,
                    sample_pressure=samples,
                    incident_sides=np.asarray(sides),
                )
            finally:
                if args.field_store:
                    coefficients._mmap.close()
            if before != hashes():
                raise RuntimeError("Marmousi trace-family acquisition sources changed")
            record = {
                "method": (
                    "MHM Helmholtz2020; Q3 local fields; exact nested polynomial trace restriction"
                ),
                "doi": "10.1137/19M1255616",
                "material": material.provenance,
                "H_m": args.H,
                "macro_shape": [mesh.nx, mesh.ny],
                "macro_cells": len(mesh.cells),
                "local_refinement": args.H // 5 * 2,
                "local_degree": 3,
                "trace_degree": degree,
                "prepared_trace_degree": 4,
                "assembly_order": 9,
                "requested_assembly_order": 9,
                "omega": 40 * np.pi,
                "point_source": [5000, 50, 1.0],
                "point_source_allocation": [
                    {"macro_cell": cell, "sources": values.tolist()}
                    for cell, values in enumerate(factory.point_sources)
                    if len(values)
                ],
                "point_source_convention": (
                    "Incident-angle allocation; total unit strength preserved"
                ),
                "boundary": (
                    "Top weak Dirichlet zero; other sides outgoing first-order absorption zero"
                ),
                "global_complex_dofs_total": len(mesh.faces) * (degree + 1),
                "global_complex_dofs_free": (2 * len(mesh.cells) - mesh.ny) * (degree + 1),
                "algebraic_residual": residual,
                "residual_equations": (
                    "Original skeleton equations tested in the stated lower-degree space"
                ),
                "macro_balance_max": macro_balance,
                "local_equation_residual_max": local_residual,
                "local_storage_bytes": local_storage_bytes,
                "response_storage": "memory" if args.response_store is None else "verified batches",
                "preparation_seconds_shared": preparation_seconds,
                "response_acquisition_seconds_total": response_acquisition_seconds,
                "field_storage": "streamed NPY/memory-map" if args.field_store else "memory",
                **stored,
                "restriction_and_export_seconds": time.perf_counter() - stage_started,
                "workers": args.workers,
                "local_solver": args.local_solver,
                "local_refinement_precision": args.local_refinement_precision,
                "global_solver": args.solver,
                "archive": archive.name,
                "archive_sha256": file_digest(archive),
                "source_sha256": before,
                "source_changed_during_run": False,
                "sampling": "513 by129 nodes; four incident macro values, without averaging",
                "historical_article_arrays_identified": False,
                "response_reuse": "T.T S T; original local CSCs and lifts shared; "
                "boundary coupling stored sparsely after original Schur calculation",
            }
            record_path = archive.with_suffix(".json")
            temporary = record_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(record_path)
            print(json.dumps(record), flush=True)
            if args.field_store:
                del coefficients, samples, solved
            else:
                del coefficients, samples, solution
            gc.collect()


if __name__ == "__main__":
    main()
