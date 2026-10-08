"""Solve an archived native FEM matrix in a separate optional solver environment.

The finite-element assembly and physical verification remain in the calling
DOLFINx example. This process boundary keeps the PETSc and Intel runtimes
independent while using the package's shared factorization and equilibration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from pymhm.linalg.linear import factorize


def _digest(path: Path) -> str:
    """Hash large algebra archives without reading them into one memory buffer."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _peak_rss() -> float | None:
    """Return the platform resource peak when available, otherwise leave it unspecified."""
    if sys.platform == "win32":
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024**3 if sys.platform == "darwin" else 1024**2
    )


def relative_residual(matrix: sparse.csr_matrix, vector: np.ndarray, rhs: np.ndarray) -> float:
    """Accumulate the original RHS-relative residual in bounded extended-precision blocks."""
    dtype = np.longdouble if np.finfo(np.longdouble).eps < np.finfo(float).eps else np.float64
    state = np.asarray(vector, dtype=dtype)
    right = np.asarray(rhs, dtype=dtype)
    squared = dtype(0)
    for first in range(0, len(right), 32768):
        last = min(first + 32768, len(right))
        defect = right[first:last] - matrix[first:last].astype(dtype) @ state
        squared += np.sum(defect * defect, dtype=dtype)
    denominator = np.linalg.norm(right)
    return float(np.sqrt(squared) / denominator) if denominator else float(np.sqrt(squared))


def solve_archive(
    matrix_path: Path,
    rhs_path: Path,
    solution_path: Path,
    *,
    threads: int = 4,
    solver: str = "pypardiso-symmetric-matching",
    equilibration: Literal["none", "symmetric"] = "symmetric",
) -> dict[str, Any]:
    """Factor the archived CSR equations and validate the serialized float64 solution.

    The acceptance criterion is the original, unscaled RHS-relative residual
    ``1e-10``. Congruence changes algebraic coordinates only; no physical row
    or right-hand-side term is omitted from this check. Thread counts are
    applied after the selected native solver library has been loaded.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    matrix = sparse.load_npz(matrix_path).tocsr()
    rhs = np.load(rhs_path, allow_pickle=False)
    if rhs.ndim != 1 or np.iscomplexobj(rhs) or not np.isfinite(rhs).all():
        raise ValueError("one finite real right-hand-side vector is required")
    if solver.startswith("pypardiso"):
        importlib.import_module("pypardiso")
    with threadpool_limits(limits=threads):
        started = perf_counter()
        with factorize(matrix, solver=solver, equilibration=equilibration) as decomposition:
            factored = perf_counter()
            solution = np.asarray(decomposition.solve(rhs), dtype=float)
            solved = perf_counter()
        residual = relative_residual(matrix, solution, rhs)
        if not np.isfinite(residual) or residual > 1e-10:
            raise ValueError(f"original RHS-relative residual exceeds 1e-10: {residual}")
        pools = [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in threadpool_info()
        ]
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(solution_path, solution, allow_pickle=False)
    solver_source = Path(importlib.import_module("pymhm.linalg.linear").__file__)
    return dict(
        solver=solver,
        equilibration=equilibration,
        original_rhs_relative_residual=residual,
        factor_seconds=factored - started,
        solve_seconds=solved - factored,
        requested_threads=threads,
        native_threadpools=pools,
        peak_rss_gib=_peak_rss(),
        source_sha256=_digest(Path(__file__)),
        solver_source_sha256=_digest(solver_source),
        matrix_sha256=_digest(matrix_path),
        rhs_sha256=_digest(rhs_path),
        solution_sha256=_digest(solution_path),
    )


def main() -> None:
    """Read explicit matrix/vector paths and write checked coefficients and diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("rhs", type=Path)
    parser.add_argument("solution", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--solver", default="pypardiso-symmetric-matching")
    parser.add_argument("--equilibration", choices=("none", "symmetric"), default="symmetric")
    args = parser.parse_args()
    record = solve_archive(
        args.matrix,
        args.rhs,
        args.solution,
        threads=args.threads,
        solver=args.solver,
        equilibration=args.equilibration,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_hpc4e_algebra").main()
