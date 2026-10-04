"""Measure complete Darcy MHM solves with verified CPU execution equivalence.

Run ``pixi run -e test python benchmarks/scaling.py --output result.json``.
Timings include local assembly, pool startup, condensation, global assembly and
solve, reconstruction, and native thread-limit setup. Error diagnostics are
outside the timer. Each process-mode solve constructs a new spawn pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

import pymhm
from pymhm._legacy.models.darcy.primal import DarcySolution, solve_darcy
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh


def exact_pressure(points: np.ndarray) -> np.ndarray:
    """Return the affine manufactured pressure exactly representable by P1."""
    return 1 + points[:, 0] + 2 * points[:, 1]


def exact_flux(points: np.ndarray) -> np.ndarray:
    """Return the corresponding constant physical Darcy flux for unit K."""
    return np.broadcast_to(np.array([-1.0, -2.0]), points.shape)


def source_fingerprint(root: Path) -> dict[str, str]:
    """Hash numerical sources and the lockfile to identify the measured state."""
    paths = [
        *sorted((root / "src/pymhm").rglob("*.py")),
        *sorted((root / "benchmarks").rglob("*.py")),
        root / "pixi.lock",
    ]
    return current_source_manifest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    )


def machine_information(root: Path) -> dict[str, Any]:
    """Collect portable software, CPU, affinity, BLAS, and revision metadata."""
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            model,
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    keys = ("user_api", "internal_api", "prefix", "version", "num_threads", "architecture")
    with threadpool_limits(limits=1):
        libraries = [{key: library.get(key) for key in keys} for library in threadpool_info()]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
        "pymhm": pymhm.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "native_libraries_during_timing": libraries,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def verify_solution(solution: DarcySolution, reference: DarcySolution) -> dict[str, float]:
    """Compare coefficients with serial execution and check exact PDE identities."""
    np.testing.assert_allclose(solution.hybrid.trace, reference.hybrid.trace, rtol=1e-9, atol=1e-10)
    for field, baseline in zip(solution.pressure, reference.pressure, strict=True):
        np.testing.assert_allclose(field, baseline, rtol=1e-9, atol=1e-10)
    for field, baseline in zip(solution.flux, reference.flux, strict=True):
        np.testing.assert_allclose(field, baseline, rtol=1e-9, atol=1e-10)
    errors = {
        "pressure_l2": solution.l2_error(exact_pressure),
        "flux_l2": solution.flux_l2_error(exact_flux),
        "macrocell_conservation_linf": float(np.max(np.abs(solution.conservation_residuals()))),
        "global_residual": float(solution.hybrid.residual),
    }
    if any(error > 1e-9 for error in errors.values()):
        raise AssertionError(f"Manufactured affine case failed verification: {errors}")
    return errors


def measure_case(
    resolution: int,
    refinement: int,
    repeats: int,
    workers: int,
    local_solver: str,
    *,
    parallel_assembly: bool = False,
) -> dict[str, Any]:
    """Time three ordered execution backends on an identical complete problem."""
    mesh = TriangleMesh.unit_square(resolution)
    with threadpool_limits(limits=1):
        reference = solve_darcy(mesh, dirichlet=exact_pressure, local_refinement=refinement)
    results: list[dict[str, Any]] = []
    for backend in ("serial", "thread", "process"):
        durations = []
        errors = []
        for _ in range(repeats):
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                solution = solve_darcy(
                    mesh,
                    dirichlet=exact_pressure,
                    local_refinement=refinement,
                    backend=backend,
                    workers=workers,
                    local_solver=local_solver,
                    parallel_assembly=parallel_assembly,
                )
            durations.append(time.perf_counter() - start)
            errors.append(verify_solution(solution, reference))
        median = statistics.median(durations)
        result = {
            "backend": backend,
            "workers": 1 if backend == "serial" else workers,
            "native_threads": 1,
            "seconds": durations,
            "median_seconds": median,
            "minimum_seconds": min(durations),
            "maximum_seconds": max(durations),
            "population_stddev_seconds": statistics.pstdev(durations),
            "speedup_over_serial_median": results[0]["median_seconds"] / median if results else 1.0,
            "worst_verification_error": {
                key: max(error[key] for error in errors) for key in errors[0]
            },
        }
        results.append(result)
        print(
            f"macro={resolution} local={refinement} {backend:7s}: {median:.6f} s "
            f"speedup={result['speedup_over_serial_median']:.3f}",
            flush=True,
        )
    return {
        "macro_grid": [resolution, resolution],
        "macrocells": len(mesh.cells),
        "local_refinement": refinement,
        "total_fine_triangles": len(mesh.cells) * refinement**2,
        "local_pressure_dofs_per_macrocell": (refinement + 1) * (refinement + 2) // 2,
        "skeleton_dofs": len(mesh.faces),
        "coarse_kernel_dofs": len(mesh.cells),
        "repeats": repeats,
        "local_solver": local_solver,
        "parallel_assembly": parallel_assembly,
        "measurements": results,
    }


def main() -> None:
    """Parse the benchmark CLI and write a versioned, machine-readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use two smaller smoke-test workloads")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results/scaling.json"))
    parser.add_argument("--local-solver", choices=("scipy", "pyamg"), default="scipy")
    parser.add_argument(
        "--parallel-assembly",
        action="store_true",
        help="Build and condense each local problem inside its worker",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Thread/process workers (default: 2)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Measurements per backend, minimum 3"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.repeats < 3:
        parser.error("workers must be positive and repeats must be at least three")
    root = Path(__file__).resolve().parents[1]
    before = source_fingerprint(root)
    report = {
        "schema_version": 1,
        "measurement": "complete Darcy primal P1 MHM solve",
        "problem": "unit square, K=I, f=0, p=1+x+2y on the boundary",
        "timing_scope": "assembly, pool startup, local condensation, global solve, reconstruction",
        "parallel_scope": (
            "independent local assembly and condensation"
            if args.parallel_assembly
            else "independent local condensation only"
        ),
        "startup_policy": "one untimed serial warmup per workload; fresh process pool per solve",
        "coefficient_comparison": {"rtol": 1e-9, "atol": 1e-10},
        "maximum_manufactured_error": 1e-9,
        "quick": args.quick,
        "machine": machine_information(root),
        "source_sha256_before": before,
        "cases": [
            measure_case(
                n,
                refinement,
                args.repeats,
                args.workers,
                args.local_solver,
                parallel_assembly=args.parallel_assembly,
            )
            for n, refinement in ([(2, 4), (4, 4)] if args.quick else [(4, 8), (8, 8)])
        ],
    }
    after = source_fingerprint(root)
    report["source_sha256_after"] = after
    report["source_changed_during_run"] = before != after
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Results written to {args.output}")
    if before != after:
        print(
            "Source files changed during the run; rerun before using these timings as a baseline."
        )


if __name__ == "__main__":
    main()
