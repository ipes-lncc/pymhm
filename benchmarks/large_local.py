"""Profile substantial local Darcy problems and isolated prepared-matrix condensation.

Run ``pixi run -e test python benchmarks/large_local.py``. The measured public
Darcy path is unchanged. Scoped call wrappers only record phase durations; no
matrix, tolerance, solver, discretization or execution backend is replaced.
Every process measurement creates a fresh portable spawn pool. Diagnostics and
reference construction are outside the timers. Use ``--pilot`` before a full run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from scaling import exact_pressure, machine_information, source_fingerprint, verify_solution
from threadpoolctl import threadpool_limits

import pymhm._legacy.models.darcy.primal as darcy_module
import pymhm.core.system as hybrid_module
from pymhm._legacy.models.darcy.primal import DarcySolution
from pymhm.core.contracts import LocalAssembly, LocalProblem, LocalResponse
from pymhm.core.system import HybridSystem
from pymhm.execution.cpu import map_local
from pymhm.fem.scalar.operators import boundary_data, face_integration, p1_geometry, p1_operators
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.triangle import TriangleMesh

PHASES = (
    "local_assembly",
    "local_condensation",
    "global_assembly",
    "global_solve",
    "reconstruction",
    "parent_bookkeeping",
    "total",
)


FACTORY_PHASES = (
    "boundary_setup",
    "local_assembly_and_condensation",
    "global_assembly",
    "global_solve",
    "reconstruction",
    "parent_bookkeeping",
    "total",
)


@dataclass(frozen=True)
class DarcyCell:
    """Picklable input containing geometry and a local refinement specification."""

    skeleton: SkeletonSpace
    cell: int
    refinement: int


def assemble_darcy_cell(task: DarcyCell) -> LocalAssembly:
    """Assemble one P1 cell with public element kernels and retain its physical mesh."""
    coarse = task.skeleton.mesh
    fine = coarse.submesh(task.cell, task.refinement)
    coupling, _ = face_integration(coarse, task.cell, fine, task.skeleton)
    matrix, mass, load = p1_operators(fine)
    kernel = np.ones((len(fine.points), 1))
    problem = LocalProblem(
        matrix, coupling, load, task.skeleton.cell_dofs(task.cell), kernel, mass @ kernel
    )
    return LocalAssembly(problem, fine)


def factory_solve(mesh: TriangleMesh, refinement: int, backend: Any, workers: int) -> DarcySolution:
    """Use the public worker-factory API for the same affine Dirichlet Darcy problem."""
    skeleton = SkeletonSpace(mesh)
    boundary, _ = boundary_data(skeleton, exact_pressure, None)
    system = HybridSystem.from_local_factory(
        assemble_darcy_cell,
        (DarcyCell(skeleton, cell, refinement) for cell in range(len(mesh.cells))),
        boundary_load=boundary,
        backend=backend,
        workers=workers,
    )
    result = system.solve()
    fluxes = []
    for fine, pressure in zip(system.local_metadata, result.fields, strict=True):
        gradients, _ = p1_geometry(fine)
        gradient = np.einsum("ti,tia->ta", pressure[fine.cells], gradients)
        tensors = tensor_values(1.0, fine.points[fine.cells].mean(axis=1))
        fluxes.append(-np.einsum("tab,tb->ta", tensors, gradient))
    return DarcySolution(
        skeleton, system.local_metadata, result.fields, tuple(fluxes), result, "primal", 1.0, 0.0
    )


def profiled_solve(
    mesh: TriangleMesh, refinement: int, backend: str, workers: int, *, factory_path: bool = False
) -> tuple[DarcySolution, dict[str, float], tuple[LocalProblem, ...]]:
    """Time existing assembly/condensation/solve/reconstruction calls with scoped wrappers."""
    phase_names = FACTORY_PHASES if factory_path else PHASES
    times = dict.fromkeys(phase_names, 0.0)
    prepared: tuple[LocalProblem, ...] = ()
    original_map = hybrid_module.map_local
    original_assemble = HybridSystem._assemble_global
    original_solve = HybridSystem.solve
    original_reconstruct = LocalResponse.reconstruct
    finished_global = 0.0
    started = 0.0

    def timed_map(*args: Any, **kwargs: Any) -> Any:
        """Measure the actual public parallel mapper including pool startup and serialization."""
        start = time.perf_counter()
        times["boundary_setup" if factory_path else "local_assembly"] = start - started
        result = original_map(*args, **kwargs)
        phase = "local_assembly_and_condensation" if factory_path else "local_condensation"
        times[phase] += time.perf_counter() - start
        return result

    def timed_assemble(
        self: HybridSystem, responses: Any, metadata: Any, boundary_load: Any
    ) -> None:
        """Time the shared global assembler without repeating any local factorization."""
        nonlocal prepared
        start = time.perf_counter()
        original_assemble(self, responses, metadata, boundary_load)
        times["global_assembly"] = time.perf_counter() - start
        prepared = tuple(response.problem for response in responses)

    def timed_reconstruct(self: LocalResponse, *args: Any, **kwargs: Any) -> Any:
        """Record actual response evaluation, retaining its original numerical implementation."""
        start = time.perf_counter()
        result = original_reconstruct(self, *args, **kwargs)
        times["reconstruction"] += time.perf_counter() - start
        return result

    def timed_solve(self: HybridSystem, *args: Any, **kwargs: Any) -> Any:
        """Include global boundary handling, factorization, residual checks and gauge work."""
        nonlocal finished_global
        start = time.perf_counter()
        result = original_solve(self, *args, **kwargs)
        finished_global = time.perf_counter()
        times["global_solve"] = finished_global - start - times["reconstruction"]
        return result

    with (
        threadpool_limits(limits=1),
        patch.object(HybridSystem, "_assemble_global", timed_assemble),
        patch.object(hybrid_module, "map_local", timed_map),
        patch.object(HybridSystem, "solve", timed_solve),
        patch.object(LocalResponse, "reconstruct", timed_reconstruct),
    ):
        started = time.perf_counter()
        if factory_path:
            result = factory_solve(mesh, refinement, backend, workers)
        else:
            result = darcy_module.solve_darcy(
                mesh,
                dirichlet=exact_pressure,
                local_refinement=refinement,
                backend=backend,
                workers=workers,
            )
        end = time.perf_counter()
        times["total"] = end - started
        times["reconstruction"] += end - finished_global
    times["parent_bookkeeping"] = times["total"] - sum(times[key] for key in phase_names[:-1])
    if min(times.values()) < 0 or not prepared:
        raise AssertionError("Invalid phase timing or no observed local assembly")
    return result, times, prepared


def condense(problem: LocalProblem) -> LocalResponse:
    """Expose a spawn-pickleable call to the original constrained local factorization."""
    return problem.condense("scipy")


def statistics_record(values: list[float]) -> dict[str, Any]:
    """Preserve all repetitions and summarize observed wall-clock variation."""
    return {
        "seconds": values,
        "median_seconds": statistics.median(values),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
        "population_stddev_seconds": statistics.pstdev(values),
    }


def verify_responses(actual: list[LocalResponse], reference: list[LocalResponse]) -> float:
    """Check both inhomogeneous and harmonic local responses against serial execution."""
    error = 0.0
    for actual_response, reference_response in zip(actual, reference, strict=True):
        for field in ("source", "lifts"):
            values, expected = getattr(actual_response, field), getattr(reference_response, field)
            np.testing.assert_allclose(values, expected, rtol=1e-10, atol=1e-10)
            error = max(error, float(np.max(np.abs(values - expected), initial=0.0)))
    return error


def measure_workload(n: int, refinement: int, repeats: int, *, pilot: bool) -> dict[str, Any]:
    """Compare full solves and prepared local systems over serial/thread/process workers."""
    mesh = TriangleMesh.unit_square(n)
    with threadpool_limits(limits=1):
        reference = darcy_module.solve_darcy(
            mesh, dirichlet=exact_pressure, local_refinement=refinement
        )
        warmup, warmup_times, prepared = profiled_solve(mesh, refinement, "serial", 1)
        verify_solution(warmup, reference)
        factory_warmup, factory_warmup_times, _ = profiled_solve(
            mesh, refinement, "serial", 1, factory_path=True
        )
        verify_solution(factory_warmup, reference)
        local_reference = map_local(condense, prepared)
    configurations = (
        [("serial", 1)]
        if pilot
        else [
            ("serial", 1),
            *((backend, workers) for backend in ("thread", "process") for workers in (1, 2, 4, 8)),
        ]
    )
    full_records, local_records, factory_records = [], [], []
    for backend, workers in configurations:
        full_times, errors, factory_times, factory_errors = [], [], [], []
        local_times, local_errors = [], []
        for _ in range(repeats):
            result, durations, _ = profiled_solve(mesh, refinement, backend, workers)
            full_times.append(durations)
            factory_result, factory_durations, _ = profiled_solve(
                mesh, refinement, backend, workers, factory_path=True
            )
            factory_times.append(factory_durations)
            with threadpool_limits(limits=1):
                factory_errors.append(verify_solution(factory_result, reference))
                errors.append(verify_solution(result, reference))
                start = time.perf_counter()
                responses = map_local(condense, prepared, backend=backend, workers=workers)
                local_times.append(time.perf_counter() - start)
                local_errors.append(verify_responses(responses, local_reference))
        profile = {key: statistics_record([t[key] for t in full_times]) for key in PHASES}
        total = profile["total"]["median_seconds"]
        full = {
            "backend": backend,
            "workers": workers,
            "native_threads": 1,
            "phases": profile,
            "speedup_over_serial_median": (
                full_records[0]["phases"]["total"]["median_seconds"] / total
                if full_records
                else 1.0
            ),
            "worst_verification_error": {key: max(e[key] for e in errors) for key in errors[0]},
        }
        local = {
            "backend": backend,
            "workers": workers,
            **statistics_record(local_times),
            "maximum_response_coefficient_difference": max(local_errors),
            "speedup_over_serial_median": (
                local_records[0]["median_seconds"] / statistics.median(local_times)
                if local_records
                else 1.0
            ),
        }
        factory_profile = {
            key: statistics_record([t[key] for t in factory_times]) for key in FACTORY_PHASES
        }
        factory_total = factory_profile["total"]["median_seconds"]
        factory_record = {
            "backend": backend,
            "workers": workers,
            "native_threads": 1,
            "phases": factory_profile,
            "speedup_over_serial_median": (
                factory_records[0]["phases"]["total"]["median_seconds"] / factory_total
                if factory_records
                else 1.0
            ),
            "speedup_over_original_serial_median": (
                full_records[0]["phases"]["total"]["median_seconds"] / factory_total
                if full_records
                else total / factory_total
            ),
            "worst_verification_error": {
                key: max(e[key] for e in factory_errors) for key in factory_errors[0]
            },
        }
        full_records.append(full)
        local_records.append(local)
        factory_records.append(factory_record)
        print(
            f"n={n} r={refinement} {backend:7s} workers={workers}: "
            f"total={total:.3f}s; assembly={profile['local_assembly']['median_seconds']:.3f}s; "
            f"condensation={profile['local_condensation']['median_seconds']:.3f}s; "
            f"prepared={local['median_seconds']:.3f}s; factory={factory_total:.3f}s; "
            f"speedup full={full['speedup_over_serial_median']:.3f}, "
            f"prepared={local['speedup_over_serial_median']:.3f}",
            flush=True,
        )
    return {
        "macro_resolution": n,
        "macrocells": len(mesh.cells),
        "local_refinement": refinement,
        "fine_triangles": len(mesh.cells) * refinement**2,
        "local_pressure_dofs": prepared[0].matrix.shape[0],
        "local_trace_rhs": prepared[0].coupling.shape[1],
        "local_matrix_nonzeros": prepared[0].matrix.nnz,
        "repeats": repeats,
        "warmup_profile_seconds": warmup_times,
        "factory_warmup_profile_seconds": factory_warmup_times,
        "complete_solve": full_records,
        "prepared_condensation": local_records,
        "factory_complete_solve": factory_records,
    }


def main() -> None:
    """Save full raw timings, verification results and source/machine provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=["2:64", "2:96", "4:64"])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/large-local.json"))
    args = parser.parse_args()
    if args.repeats < 3 and not args.pilot:
        parser.error("At least three repetitions are required for the reported benchmark")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    cases = [tuple(map(int, item.split(":"))) for item in args.cases]
    if any(len(case) != 2 or min(case) < 1 for case in cases):
        parser.error("Each case must be positive macro-resolution:local-refinement")
    root = Path(__file__).resolve().parents[1]
    before = source_fingerprint(root)
    report = {
        "schema_version": 1,
        "machine": machine_information(root),
        "isolation": (
            "Nonexclusive workstation, without CPU affinity pinning. Competing agent "
            "scientific jobs were paused, but existing user processes remained running, "
            "including the VS Code C++ language server and documentation server."
        ),
        "source_sha256_before": before,
        "problem": "Darcy primal P1, unit square, K=I, f=0, p=1+x+2y, constant macroface traces",
        "local_solver": "scipy",
        "global_solver": "scipy",
        "parallel_scope": (
            "Existing solve_darcy: condensation only. Factory path: local assembly and "
            "condensation in the same worker; global work and reconstruction stay in parent."
        ),
        "timing_policy": (
            "One untimed uninstrumented reference and one serial warmup per measured path. "
            "Complete solve includes assembly, pool creation, condensation, global solve "
            "and reconstruction. "
            "Prepared condensation excludes assembly but includes pool creation, input and output "
            "serialization and thread limits. Native pools are limited to one thread. "
            "Scoped wrappers record unchanged public calls; no runtime source modification. "
            "Verification is outside timers. Repetitions alternate original, factory "
            "and prepared work."
        ),
        "phase_definitions": {
            "boundary_setup": "factory-path boundary integrals and task construction",
            "local_assembly_and_condensation": (
                "factory-path worker wall time including fresh pool and metadata serialization"
            ),
            "parent_bookkeeping": "remaining parent-side dispatch and result bookkeeping",
            "local_assembly": (
                "boundary data, local meshes, local variational matrices and validation"
            ),
            "local_condensation": (
                "actual map_local including fresh pool, factorization and trace/source lifts"
            ),
            "global_assembly": "HybridSystem construction excluding its local condensation",
            "global_solve": (
                "boundary elimination, LU, gauge/residual work excluding local reconstruction"
            ),
            "reconstruction": (
                "response evaluation plus final Darcy physical pressure/flux extraction"
            ),
        },
        "pilot": args.pilot,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for n, refinement in cases:
        report["cases"].append(measure_workload(n, refinement, args.repeats, pilot=args.pilot))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    after = source_fingerprint(root)
    report["source_sha256_after"] = after
    report["source_changed_during_run"] = before != after
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if before != after:
        raise RuntimeError("Numerical source fingerprint changed; these timings need a fresh run")


if __name__ == "__main__":
    main()
