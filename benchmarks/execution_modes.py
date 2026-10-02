"""Measure distributed, offline/online and resident batched local computations.

Examples: ``pixi run -e fem mpiexec -n 4 python benchmarks/execution_modes.py mpi``;
``pixi run -e test python benchmarks/execution_modes.py offline``;
``pixi run -e gpu python benchmarks/execution_modes.py gpu``.
Every mode verifies numerical fields before writing a report. Timings include
the operations named in each report and never assume a positive acceleration.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm import HybridSystem, LocalProblem, SkeletonSpace, TriangleMesh
from pymhm.elements import boundary_data, face_integration, p1_geometry, p1_operators
from pymhm.hybrid import LocalAssembly
from pymhm.offline import OfflineHybridSystem


def hashes() -> dict[str, str]:
    """Capture public numerical-source digests without machine-specific paths."""
    names = (
        "hybrid",
        "parallel",
        "solvers",
        "mesh",
        "elements",
        "lagrange",
        "offline",
        "distributed",
        "gpu",
    )
    files = [Path("src/pymhm") / f"{name}.py" for name in names]
    files.append(Path("benchmarks/execution_modes.py"))
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in files}


def exact(points: Any) -> Any:
    """Define the affine Darcy field shared by all distributed runs."""
    return 1 + points[:, 0] + 2 * points[:, 1]


def local(task: tuple[SkeletonSpace, int, int]) -> LocalAssembly:
    """Assemble a P1 Neumann cell entirely on its owning worker/rank."""
    skeleton, index, refinement = task
    fine = skeleton.mesh.submesh(index, refinement)
    matrix, mass, source = p1_operators(fine)
    coupling, _ = face_integration(skeleton.mesh, index, fine, skeleton)
    mode = np.ones((len(fine.points), 1))
    return LocalAssembly(
        LocalProblem(matrix, coupling, source, skeleton.cell_dofs(index), mode, mass @ mode), fine
    )


def mpi_campaign(arguments: argparse.Namespace) -> dict[str, Any] | None:
    """Measure full distributed solves and verify affine pressure and flux locally."""
    from mpi4py import MPI
    from petsc4py import PETSc

    from pymhm.distributed import solve_distributed

    comm = MPI.COMM_WORLD
    mesh = TriangleMesh.unit_square(arguments.mesh)
    skeleton = SkeletonSpace(mesh)
    boundary, _ = boundary_data(skeleton, exact, None)
    items = [
        (skeleton, index, arguments.refinement)
        for index in range(comm.rank, len(mesh.cells), comm.size)
    ]
    samples, errors = [], []
    for repetition in range(arguments.repeats + 1):
        comm.Barrier()
        started = time.perf_counter()
        result = solve_distributed(
            local,
            items,
            trace_size=skeleton.size,
            comm=comm,
            boundary_load=(np.arange(skeleton.size), boundary) if comm.rank == 0 else None,
        )
        elapsed = comm.allreduce(time.perf_counter() - started, op=MPI.MAX)
        pressure_error = flux_error = 0.0
        for fine, pressure in zip(result.local_metadata, result.fields, strict=True):
            pressure_error = max(
                pressure_error, float(np.max(np.abs(pressure - exact(fine.points))))
            )
            gradients, _ = p1_geometry(fine)
            flux = -np.einsum("ti,tia->ta", pressure[fine.cells], gradients)
            flux_error = max(flux_error, float(np.max(np.abs(flux + [1.0, 2.0]))))
        error = comm.allreduce(max(pressure_error, flux_error, result.residual), op=MPI.MAX)
        if error > 1e-9:
            raise RuntimeError(f"MPI physical-field verification failed: {error}")
        if repetition:
            samples.append(elapsed)
            errors.append(error)
    owned_rows = comm.gather(len(result.owned_coefficients), root=0)
    owned_cells = comm.gather(len(items), root=0)
    return (
        {
            "mode": "distributed-mpi",
            "ranks": comm.size,
            "host": platform.node(),
            "mpi_library": MPI.Get_library_version().splitlines()[0],
            "petsc": PETSc.Sys.getVersion(),
            "global_solver": "distributed PETSc AIJ + MUMPS LU",
            "macro_triangles": len(mesh.cells),
            "fine_refinement": arguments.refinement,
            "owned_global_rows": owned_rows,
            "owned_local_cells": owned_cells,
            "global_unknowns": result.global_size,
            "seconds": samples,
            "median_seconds": float(np.median(samples)),
            "max_coefficient_flux_residual_error": max(errors),
            "scope": (
                "rank-local assembly, condensation, distributed global assembly/solve "
                "and owned reconstruction; coarse mesh metadata replicated"
            ),
            "hardware_limit": (
                "all ranks share this host; these measurements do not establish inter-node scaling"
            ),
        }
        if comm.rank == 0
        else None
    )


def offline_campaign(arguments: argparse.Namespace) -> dict[str, Any]:
    """Compare changing-source queries with independent refactorization of identical operators."""
    mesh = TriangleMesh.unit_square(arguments.mesh)
    skeleton = SkeletonSpace(mesh)
    started = time.perf_counter()
    assembled = [local((skeleton, index, arguments.refinement)) for index in range(len(mesh.cells))]
    assembly_seconds = time.perf_counter() - started
    cells = [entry.problem for entry in assembled]
    data = []
    for index in range(arguments.queries):
        a, b = 1 + index / arguments.queries, (-1.0) ** index * (index + 1) / arguments.queries
        loads = [-4 * b * p.constraints[:, 0] for p in cells]
        boundary, _ = boundary_data(
            skeleton, lambda x, a=a, b=b: a * exact(x) + b * (x[:, 0] ** 2 + x[:, 1] ** 2), None
        )
        data.append((loads, boundary))
    setup, online_times, fresh_times, error = [], [], [], 0.0
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        prepared = OfflineHybridSystem(cells)
        setup.append(time.perf_counter() - started)
        with prepared:
            for loads, boundary in data:
                started = time.perf_counter()
                result = prepared.solve(loads, boundary_load=boundary)
                online_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                rebuilt = HybridSystem(
                    [p.with_load(f) for p, f in zip(cells, loads, strict=True)],
                    boundary_load=boundary,
                ).solve()
                fresh_times.append(time.perf_counter() - started)
                error = max(
                    error,
                    max(
                        float(np.max(np.abs(u - v)))
                        for u, v in zip(result.fields, rebuilt.fields, strict=True)
                    ),
                )
    if error > 1e-10:
        raise RuntimeError(f"offline/online field mismatch: {error}")
    return {
        "mode": "offline-online",
        "macro_triangles": len(cells),
        "fine_refinement": arguments.refinement,
        "queries_per_preparation": arguments.queries,
        "assembly_seconds": assembly_seconds,
        "offline_seconds": setup,
        "online_query_seconds": online_times,
        "fresh_query_seconds": fresh_times,
        "median_offline_seconds": float(np.median(setup)),
        "median_online_query_seconds": float(np.median(online_times)),
        "median_fresh_query_seconds": float(np.median(fresh_times)),
        "max_field_difference": error,
        "scope": (
            "fixed matrices and source/BC moments prepared once; online includes local source "
            "solves, RHS assembly, global factor application, checks and reconstruction"
        ),
        "problem": (
            "p=a(1+x+2y)+b(x²+y²), f=-4b, K=I, weak full Dirichlet; "
            "P1 discrete fields compared, not asserted exact for quadratic p"
        ),
    }


def gpu_campaign(arguments: argparse.Namespace) -> dict[str, Any]:
    """Measure resident P1 volume assembly, batched LU and repeated device RHS solves."""
    import cupy as cp

    from pymhm.gpu import BatchedFactorization, assemble_p1_batch

    fine = TriangleMesh.unit_square(1).submesh(0, arguments.refinement)
    count = arguments.batch
    coordinates = np.repeat(fine.points[None], count, axis=0)
    cpu_started = time.perf_counter()
    a, mass, source = p1_operators(fine, diffusion=[[3.0, 0.0], [0.0, 1.0]], source=1.0)
    n = len(source)
    coupling = np.eye(n)[:, :3]
    problem = LocalProblem(
        a, coupling, source, np.arange(3), np.ones((n, 1)), mass @ np.ones((n, 1))
    )
    matrix, rhs = problem.condensation_system()
    expected = problem.condense()
    reference = np.column_stack((expected.source, expected.lifts))
    cpu_reference_seconds = time.perf_counter() - cpu_started
    records, error = [], 0.0
    for repetition in range(arguments.repeats + 1):
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        points_device, cells_device = cp.asarray(coordinates), cp.asarray(fine.cells)
        cp.cuda.Stream.null.synchronize()
        uploaded = time.perf_counter()
        stiffness, masses, forcing = assemble_p1_batch(
            points_device, cells_device, diffusion=cp.diag(cp.asarray([3.0, 1.0])), source=1.0
        )
        constrained = cp.zeros((count, n + 1, n + 1))
        constrained[:, :n, :n] = stiffness
        # Use the same physical moment normalization as the shared condensation contract.
        scaling = float(matrix[:n, n].max()) / float((mass @ np.ones(n)).max())
        means = cp.sum(masses, axis=2) * scaling
        constrained[:, :n, n], constrained[:, n, :n] = means, means
        loads = cp.asarray(np.repeat(rhs[None], count, axis=0))
        loads[:, :n, 0] = forcing
        cp.cuda.Stream.null.synchronize()
        assembled = time.perf_counter()
        with BatchedFactorization(constrained) as batch:
            cp.cuda.Stream.null.synchronize()
            factored = time.perf_counter()
            solved = batch.solve(loads)
            cp.cuda.Stream.null.synchronize()
            first = time.perf_counter()
            for factor in np.linspace(0.5, 1.5, arguments.queries):
                repeated = batch.solve(loads * factor)
            cp.cuda.Stream.null.synchronize()
            online = time.perf_counter()
            actual = cp.asnumpy(solved[:, :n])
            last = cp.asnumpy(repeated[:, :n])
            cp.cuda.Stream.null.synchronize()
            ended = time.perf_counter()
        error = max(
            error,
            float(np.max(np.abs(actual - reference))),
            float(np.max(np.abs(last - factor * reference))),
        )
        if repetition:
            records.append(
                {
                    "upload": uploaded - started,
                    "resident_volume_assembly_and_boundary_upload": assembled - uploaded,
                    "factorization": factored - assembled,
                    "first_solve": first - factored,
                    "repeated_rhs_total": online - first,
                    "download": ended - online,
                    "total": ended - started,
                }
            )
    if error > 1e-9:
        raise RuntimeError(f"GPU resident field mismatch: {error}")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    return {
        "mode": "resident-batched-gpu",
        "device": properties["name"].decode(),
        "cupy": cp.__version__,
        "batch": count,
        "local_unknowns": n,
        "fine_refinement": arguments.refinement,
        "queries": arguments.queries,
        "seconds": records,
        "cpu_single_local_reference_seconds": cpu_reference_seconds,
        "max_lift_difference": error,
        "scope": (
            "P1 affine volume assembly, constrained dense matrix creation, batched pivoted LU, "
            "residuals and multiple source/trace solves on GPU; skeleton coupling supplied from CPU"
        ),
        "comparison_limit": (
            "CPU single-local time is a correctness/reference cost, "
            "not a speedup denominator for the GPU batch"
        ),
    }


def main() -> None:
    """Run one explicit workload and serialize measured scope, checks and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["mpi", "offline", "gpu"])
    parser.add_argument("--mesh", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if (
        min(
            arguments.mesh,
            arguments.refinement,
            arguments.repeats,
            arguments.queries,
            arguments.batch,
        )
        < 1
    ):
        parser.error("all workload sizes and repeat counts must be positive")
    before = hashes()
    with threadpool_limits(limits=1):
        result = {"mpi": mpi_campaign, "offline": offline_campaign, "gpu": gpu_campaign}[
            arguments.mode
        ](arguments)
    if result is None:
        return
    result.update(
        {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "native_threads": 1,
            "source_sha256": before,
            "source_changed_during_run": before != hashes(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"source_sha256", "online_query_seconds", "fresh_query_seconds"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
