"""Verify and measure reusable block-preconditioned solves of actual MHM saddle matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.block import SaddleBlockSolver
from pymhm.elements import boundary_data, face_integration, p1_operators
from pymhm.hybrid import HybridSystem, LocalProblem
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.solvers import factorize

ROOT = Path(__file__).resolve().parents[1]


def build(n: int) -> HybridSystem:
    """Build P1 Darcy with exact affine pressure and a real indefinite skeleton/coarse matrix."""
    mesh = TriangleMesh.unit_square(n)
    skeleton = SkeletonSpace(mesh)
    problems = []
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, 8)
        A, M, f = p1_operators(fine)
        B = face_integration(mesh, cell, fine, skeleton)[0]
        Z = np.ones((len(fine.points), 1))
        problems.append(LocalProblem(A, B, f, skeleton.cell_dofs(cell), Z, M @ Z))
    boundary, _ = boundary_data(skeleton, lambda x: 1 + x[:, 0] + 2 * x[:, 1], None)
    return HybridSystem(problems, boundary_load=boundary)


def main() -> None:
    """Record separate setup and repeated-RHS costs, verifying every field against sparse LU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pyamg", "amgx"), default="pyamg")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        Path(__file__),
        *(
            ROOT / "src/pymhm" / name
            for name in ("block.py", "solvers.py", "hybrid.py", "elements.py", "mesh.py")
        ),
    ]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    rows = []
    with threadpool_limits(limits=1):
        for n in (2, 4, 8):
            system = build(n)
            expected = system.solve()
            samples = []
            direct_samples = []
            error = 0.0
            residual = 0.0
            for repetition in range(4):
                start = time.perf_counter()
                with factorize(system.matrix) as direct:
                    setup = time.perf_counter() - start
                    start = time.perf_counter()
                    for multiplier in (0.5, 1.0, 1.5):
                        direct.solve(multiplier * system.rhs)
                    direct_elapsed = time.perf_counter() - start
                start = time.perf_counter()
                with SaddleBlockSolver(
                    system.matrix, system.trace_size, backend=args.backend
                ) as block:
                    block_setup = time.perf_counter() - start
                    start = time.perf_counter()
                    for multiplier in (0.5, 1.0, 1.5):
                        block.solve(multiplier * system.rhs)
                    block_elapsed = time.perf_counter() - start
                    iterations = block.iterations.copy()
                    actual = system.solve(factorization=block.as_factorization())
                    error = max(
                        error,
                        max(
                            float(np.max(np.abs(a - b)))
                            for a, b in zip(actual.fields, expected.fields, strict=True)
                        ),
                    )
                    residual = max(residual, actual.residual)
                if repetition:
                    samples.append(
                        dict(
                            setup_seconds=block_setup,
                            three_rhs_seconds=block_elapsed,
                            iterations=iterations,
                        )
                    )
                    direct_samples.append(
                        dict(setup_seconds=setup, three_rhs_seconds=direct_elapsed)
                    )
            if error > 1e-8 or residual > 1e-10:
                raise RuntimeError("MHM block verification failed")
            row = dict(
                macro_triangles=2 * n * n,
                trace_size=system.trace_size,
                global_unknowns=len(system.rhs),
                block=samples,
                direct=direct_samples,
                max_pressure_coefficient_difference=error,
                max_global_backward_residual=residual,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(
        backend=args.backend,
        outer="CPU GMRES on original symmetric saddle",
        primal_preconditioner="one retained AMG V-cycle on A+alpha B B.T",
        schur_preconditioner="sparse LU of C+B.T diag(P)^-1 B",
        operator_unchanged=True,
        rows=rows,
        python=platform.python_version(),
        numpy=np.__version__,
        native_threads=1,
        measurement_limit=(
            "One untimed warmup and three repetitions; setup includes block "
            "construction, positive-definiteness checks and AMG hierarchy. Assembly is excluded. "
            "Other numerical campaigns were active; wall times do not establish isolated "
            "comparative speedup."
        ),
        source_sha256=hashes,
        source_changed_during_run=any(
            hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))]
            for path in paths
        ),
        timestamp_utc=datetime.now(UTC).isoformat(),
    )
    if report["source_changed_during_run"]:
        raise RuntimeError("source changed during acquisition")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
