"""Acquire current-source Q5 periodic references with explicit field provenance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

if __package__:
    from .verify_periodic import ARTIFACTS, ROOT, factor_x, factor_y, fingerprint
else:
    from verify_periodic import ARTIFACTS, ROOT, factor_x, factor_y, fingerprint

from pymhm.mesh import positive_int
from pymhm.quadrilateral import CartesianMacroMesh
from pymhm.separable import SeparableField
from pymhm.separable_krylov import solve_separable_krylov


def sources() -> dict[str, str]:
    """Fingerprint the executed acquisition and all numerical dependency owners."""
    paths = [Path(__file__), ROOT / "examples/verify_periodic.py"]
    paths += [
        ROOT / "src/pymhm" / f"{name}.py"
        for name in (
            "separable_krylov",
            "separable",
            "conforming",
            "quadrilateral",
            "solvers",
            "mesh",
            "elements",
            "cut_cells",
            "hybrid",
            "lagrange",
        )
    ]
    return {str(p.relative_to(ROOT)): fingerprint(p) for p in paths}


def run(n: int, order: int, native_threads: int = 1) -> None:
    """Solve one reference without borrowing coefficients from an earlier acquisition."""
    native_threads = positive_int(native_threads, "native_threads")
    original = sources()
    path = ARTIFACTS / f"reference-q5-{n}-current-order{order}-lor.npz"
    record_path = ROOT / "examples/results" / f"periodic-reference-q5-{n}-order{order}.json"
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record["source_sha256"] != original or record["archive_sha256"] != fingerprint(path):
            raise ValueError("existing reference differs from its acquisition contract")
        return
    start = time.perf_counter()
    with threadpool_limits(native_threads):
        solution = solve_separable_krylov(
            CartesianMacroMesh(n),
            degree=5,
            permeability=SeparableField(((1.0, 1.0), (factor_x, factor_y))),
            source=SeparableField(((np.sin, np.sin),)),
            quadrature_order=order,
            refinement_precision="extended",
        )
        native_libraries = [
            {key: value for key, value in info.items() if key != "filepath"}
            for info in threadpool_info()
        ]
    if sources() != original:
        raise RuntimeError("reference sources changed during acquisition")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pressure=solution.pressure, residual=solution.residual)
    record = {
        "n": n,
        "degree": 5,
        "quadrature_order": order,
        "dofs": len(solution.pressure),
        "residual": solution.residual,
        "relative_equation_residual": solution.relative_equation_residual,
        "iterations": solution.iterations,
        "preconditioner_levels": solution.preconditioner_levels,
        "solver": "low-order-refined-pyamg-cg",
        "native_threads_requested": native_threads,
        "native_libraries_during_solve": native_libraries,
        "source_sha256": original,
        "archive": path.name,
        "archive_sha256": fingerprint(path),
        "elapsed_seconds": time.perf_counter() - start,
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


def main() -> None:
    """Acquire the requested nested Q5 resolutions under the same quadrature rule."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--native-threads", type=int, default=1)
    args = parser.parse_args()
    if args.native_threads < 1:
        parser.error("native threads must be positive")
    for n in args.sizes:
        run(n, args.order, args.native_threads)


if __name__ == "__main__":
    main()
