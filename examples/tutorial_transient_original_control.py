"""Observe original concentration rows in the unchanged spatial teaching march.

The numerical equations come from the tagged UFL notebook cell. This observer
delegates every solve, then applies the common independent original-row checker;
it distinguishes reduced global compatibility from uncondensed local equations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from threadpoolctl import threadpool_limits

from examples.core_elasticity_field_archive import ProductionObservation
from examples.minimal_flow_originals import original_diagnostics
from examples.tutorial_transient_asymptotic import NOTEBOOK, exported_definitions
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from scripts.build_notebook_companions import module_sources

ROOT = case_workspace()


def observed_row(equations: Any, n: int, dt: float, final: float, workers: int) -> dict[str, Any]:
    """Execute the identical march and check every original concentration block."""
    numerical_solve = MultiscaleSystem.solve
    steps = []

    def solve_and_observe(system: MultiscaleSystem, **options: Any) -> Any:
        """Delegate the original solve and independently inspect its executed rows."""
        state = numerical_solve(system, **options)
        if system.global_matrix.nnz:
            raise ValueError("The scalar original observer requires zero direct global operator")
        observation = ProductionObservation(
            system=system,
            applied_boundary=-system.global_load[: system.trace_size],
            fixed=dict(system.layout.fixed_trace or {}),
        )
        diagnostics = original_diagnostics(
            SimpleNamespace(hybrid=state),
            observation,
            [{"concentration": len(values)} for values in state.fields],
        )
        if not diagnostics["accepted"]:
            raise RuntimeError("The unchanged transient solve fails its original physical rows")
        steps.append({key: value for key, value in diagnostics.items() if key != "blocks"})
        return state

    with patch.object(MultiscaleSystem, "solve", solve_and_observe):
        row = equations.spatial_row(n, dt, final, workers=workers)
    row["original_equations"] = {
        "steps": len(steps),
        "criterion": 1e-10,
        "full_uncondensed_relative_max": max(
            item["full_uncondensed_relative_to_physical_rhs"] for item in steps
        ),
        "concentration_block_backward_error_max": max(
            item["maximum_original_block_backward_error"] for item in steps
        ),
        "global_weak_row_absolute_residual_max": max(
            item["global_weak_row_absolute_residual_norm"] for item in steps
        ),
        "steps_diagnostics": steps,
        "norm_convention": "Original coefficient rows; physical concentration/gradient L2 separate",
        "accepted": all(item["accepted"] for item in steps),
    }
    return row


def main() -> None:
    """Write current-source same-method controls without altering notebook equations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[4, 16, 48])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--final", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing numerical control cannot be overwritten")
    sources = current_source_manifest(
        source_identity(
            ROOT,
            [
                *sorted((ROOT / "src/pymhm").rglob("*.py")),
                *(
                    ROOT / name
                    for name in module_sources(
                        ROOT, ["examples.tutorial_transient_original_control"]
                    )
                ),
                Path(__file__),
                NOTEBOOK,
                ROOT / "pixi.lock",
                ROOT / "pixi.toml",
                ROOT / "pyproject.toml",
            ],
        ),
        packages=("pymhm",),
    )
    record = {
        "schema": "pymhm-transient-original-physical-row-control-v1",
        "source_sha256": sources,
        "dt": args.dt,
        "final_time": args.final,
        "execution": {"backend": "process", "workers": args.workers, "native_threads": 1},
        "case": "Same literal tagged spatial teaching definitions; c=t*sin(pi*x)*sin(pi*y)",
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="pymhm-original-transient-") as directory, threadpool_limits(1):
        module_path = Path(directory) / "pymhm_transient_spatial_equations.py"
        module_path.write_text(exported_definitions())
        sys.path.insert(0, directory)
        try:
            equations = importlib.import_module("pymhm_transient_spatial_equations")
            for n in args.levels:
                row = observed_row(equations, n, args.dt, args.final, args.workers)
                record["rows"].append(row)
                args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                print(
                    json.dumps({k: v for k, v in row.items() if k != "original_equations"}),
                    flush=True,
                )
        finally:
            sys.path.remove(directory)
            sys.modules.pop("pymhm_transient_spatial_equations", None)
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in sources.items()):
        raise RuntimeError("An executed original-row control source changed during acquisition")


if __name__ == "__main__":
    main()
