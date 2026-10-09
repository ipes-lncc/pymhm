"""Acquire Maxwell time errors and original-equation diagnostics separately.

The existing semidiscrete comparator owns the exact exponential reference.
An identical second march records its original electric equations and modified
energy, without changing the field-error definition or numerical equations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from examples.maxwell_campaign import CavityMode, temporal_row
from examples.tutorial_maxwell_equations import EquationLeapfrog
from pymhm import TriangleMesh
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from scripts.build_notebook_companions import module_sources

ROOT = case_workspace()


def time_row(dt: float) -> dict[str, Any]:
    """Retain the exact constrained-ODE errors and observe the identical march."""
    row = temporal_row(dt)
    mode = CavityMode(2)
    with EquationLeapfrog(TriangleMesh.unit_square(), time_step=dt, quadrature_order=8) as runtime:
        initial = runtime.initialize(mode.electric_shape)
        original = initial.original_electric_residual
        drift, balance = 0.0, 0.0
        for _ in range(round(0.04 / dt)):
            state = runtime.advance()
            original = max(original, state.original_electric_residual)
            drift = max(drift, abs(state.energy / initial.energy - 1))
            balance = max(balance, abs(state.energy_balance_residual) / initial.energy)
        if abs(state.electric_time - row["electric_time"]) > 1e-13:
            raise RuntimeError("The observed march must use the comparator's staggered times")
    row.update(
        original_electric_equation_relative_max=float(original),
        modified_energy_relative_drift_max=float(drift),
        energy_balance_relative_max=float(balance),
        accepted=bool(max(original, drift, balance) <= 1e-10),
    )
    return row


def main() -> None:
    """Write a fresh four-level time comparison with frozen source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("An existing numerical record cannot be overwritten")
    paths = [
        *sorted((ROOT / "src/pymhm").rglob("*.py")),
        *(
            ROOT / name
            for name in module_sources(ROOT, ["examples.tutorial_maxwell_time_acquisition"])
        ),
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    sources = current_source_manifest(source_identity(ROOT, paths), packages=("pymhm",))
    record = {
        "schema": "pymhm-maxwell-semidiscrete-time-convergence-v1",
        "source_sha256": sources,
        "spaces": {"dimension": 2, "macro_resolution": 1, "trace_degree": 1, "local_degree": 3},
        "reference": "Exact scipy.linalg.expm evolution of the same constrained DG ODE",
        "scope": "Temporal integration only; physical spatial error is assessed separately",
        "observer": "A second identical march records original electric rows and modified energy",
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "physical_equation_tolerance": 1e-10,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for dt in (0.002, 0.001, 0.0005, 0.00025):
            row = time_row(dt)
            record["rows"].append(row)
            args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if not row["accepted"]:
                raise RuntimeError(f"The original equations rejected dt={dt}")
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in sources.items()):
        raise RuntimeError("An executed temporal source changed during acquisition")


if __name__ == "__main__":
    main()
