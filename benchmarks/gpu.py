"""Measure complete MHM solves with CPU and device solver placements.

The benchmark includes GPU transfer, factorization, planning, reconstruction,
and synchronization costs. It does not measure a resident device implementation.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cupy
from scaling import exact_pressure, machine_information, source_fingerprint, verify_solution
from threadpoolctl import threadpool_limits

from pymhm.darcy import solve_darcy
from pymhm.mesh import TriangleMesh


def measure_gpu_case(
    resolution: int, refinement: int, repeats: int, include_amgx: bool
) -> dict[str, Any]:
    """Compare solver placement while preserving the same discretization."""
    mesh = TriangleMesh.unit_square(resolution)
    with threadpool_limits(limits=1):
        reference = solve_darcy(mesh, dirichlet=exact_pressure, local_refinement=refinement)
    configurations = [
        ("cpu", "scipy", "scipy"),
        ("global-cupy", "cupy", "scipy"),
        ("global-cudss", "cudss", "scipy"),
        ("local-cudss", "scipy", "cudss"),
        ("local-global-cudss", "cudss", "cudss"),
    ]
    if include_amgx:
        configurations.append(("local-amgx", "scipy", "amgx"))
    measurements = []
    for name, global_solver, local_solver in configurations:
        durations = []
        errors = []
        for repetition in range(repeats + 1):
            cupy.cuda.Device().synchronize()
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                solution = solve_darcy(
                    mesh,
                    dirichlet=exact_pressure,
                    local_refinement=refinement,
                    solver=global_solver,
                    local_solver=local_solver,
                )
            cupy.cuda.Device().synchronize()
            elapsed = time.perf_counter() - start
            error = verify_solution(solution, reference)
            if repetition > 0:
                durations.append(elapsed)
                errors.append(error)
        median = statistics.median(durations)
        result = {
            "name": name,
            "global_solver": global_solver,
            "local_solver": local_solver,
            "backend": "serial",
            "native_threads": 1,
            "seconds": durations,
            "median_seconds": median,
            "minimum_seconds": min(durations),
            "maximum_seconds": max(durations),
            "population_stddev_seconds": statistics.pstdev(durations),
            "speedup_over_cpu_median": measurements[0]["median_seconds"] / median
            if measurements
            else 1.0,
            "worst_verification_error": {
                key: max(error[key] for error in errors) for key in errors[0]
            },
        }
        measurements.append(result)
        print(
            f"macro={resolution} local={refinement} {name:18s}: {median:.6f} s "
            f"speedup={result['speedup_over_cpu_median']:.3f}",
            flush=True,
        )
    return {
        "macro_grid": [resolution, resolution],
        "macrocells": len(mesh.cells),
        "local_refinement": refinement,
        "total_fine_triangles": len(mesh.cells) * refinement**2,
        "local_pressure_dofs_per_macrocell": (refinement + 1) * (refinement + 2) // 2,
        "global_dofs": len(mesh.faces) + len(mesh.cells),
        "repeats": repeats,
        "measurements": measurements,
    }


def main() -> None:
    """Validate GPU availability, run repeated solves, and record provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use one small smoke-test workload")
    parser.add_argument(
        "--amgx", action="store_true", help="Require and measure the native AmgX adapter"
    )
    parser.add_argument("--amgx-revision", help="Optional Git revision of the native AmgX build")
    parser.add_argument("--pyamgx-revision", help="Optional Git revision of the PyAMGX binding")
    parser.add_argument(
        "--amgx-build", help="Optional native build version and compiler description"
    )
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions, minimum three")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/gpu.json"))
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("repeats must be at least three")
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise SystemExit("No compatible CUDA device is available")
    amgx_api = importlib.import_module("pyamgx").get_api_version() if args.amgx else None
    root = Path(__file__).resolve().parents[1]
    before = source_fingerprint(root)
    properties = cupy.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    report = {
        "schema_version": 1,
        "measurement": "complete Darcy primal P1 MHM solve with staged GPU algebra",
        "problem": "unit square, K=I, f=0, p=1+x+2y on the boundary",
        "timing_scope": (
            "assembly, transfers, planning, factorization, solve, reconstruction, synchronization"
        ),
        "startup_policy": "one untimed solve for every workload and solver placement",
        "coefficient_comparison": {"rtol": 1e-9, "atol": 1e-10},
        "maximum_manufactured_error": 1e-9,
        "quick": args.quick,
        "include_amgx": args.amgx,
        "machine": machine_information(root),
        "gpu": {
            "name": name.decode() if isinstance(name, bytes) else name,
            "total_memory_bytes": properties["totalGlobalMem"],
            "compute_capability": [properties["major"], properties["minor"]],
            "cuda_runtime_version": cupy.cuda.runtime.runtimeGetVersion(),
            "cuda_driver_api_version": cupy.cuda.runtime.driverGetVersion(),
            "cupy": cupy.__version__,
            "nvmath-python": importlib.metadata.version("nvmath-python"),
            "nvidia-cudss-cu12": importlib.metadata.version("nvidia-cudss-cu12"),
        },
        "source_sha256_before": before,
        "cases": [
            measure_gpu_case(n, refinement, args.repeats, args.amgx)
            for n, refinement in ([(2, 4)] if args.quick else [(4, 8), (8, 8)])
        ],
    }
    if args.amgx:
        report["gpu"]["amgx"] = {
            "api_version": amgx_api,
            "source_revision": args.amgx_revision,
            "binding_revision": args.pyamgx_revision,
            "build_description": args.amgx_build,
        }
    after = source_fingerprint(root)
    report["source_sha256_after"] = after
    report["source_changed_during_run"] = before != after
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results written to {args.output}")
    if before != after:
        print("Source changed during measurement; rerun before using the timings as a baseline.")


if __name__ == "__main__":
    main()
