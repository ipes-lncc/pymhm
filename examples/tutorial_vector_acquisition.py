"""Acquire current vector tutorial norms through existing formulation owners.

The records contain scalar diagnostics only. They do not persist coefficients
or claim field replay, and the original physical-equation criterion is 1e-10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.core_elasticity_field_archive import ProductionObservation
from examples.formulations.primal_elasticity import (
    define_primal_elasticity,
    primal_constraints,
    recover_primal_elasticity,
)
from examples.maxwell_campaign import temporal_row
from examples.minimal_flow_originals import original_diagnostics
from examples.solve_primal_elasticity import STIFFNESS, TensorData
from pymhm import ExecutionConfig, FaceSpace, SkeletonSpace, TriangleMesh, assemble
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from scripts.build_notebook_companions import module_sources

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/tutorial-methods"


def source_hashes() -> dict[str, str]:
    """Identify package owners, transitive acquisition dependencies and locked input.

    Independent result readers and plotters are outside this computation's
    dependency graph. The shared companion resolver includes package initializers
    and literal dynamic imports, rather than watching unrelated examples.
    """
    paths = [
        *sorted(source_file("src/pymhm/__init__.py", root=ROOT).parent.rglob("*.py")),
        Path(__file__),
        source_file("examples/solve_primal_elasticity.py", root=ROOT),
        source_file("examples/formulations/primal_elasticity.py", root=ROOT),
        source_file("examples/minimal_flow_originals.py", root=ROOT),
        source_file("examples/core_elasticity_field_archive.py", root=ROOT),
        source_file("examples/maxwell_campaign.py", root=ROOT),
        source_file("examples/tutorial_maxwell_equations.py", root=ROOT),
        source_file("scripts/build_notebook_companions.py", root=ROOT),
        *(ROOT / name for name in module_sources(ROOT, ["examples.tutorial_vector_acquisition"])),
        ROOT / "pixi.lock",
        ROOT / "pixi.toml",
        ROOT / "pyproject.toml",
    ]
    return current_source_manifest(source_identity(ROOT, paths), packages=("pymhm",))


def primal_row(n: int, *, workers: int = 32) -> dict[str, Any]:
    """Measure homogeneous P3/r1 elasticity with unsplit vector P1 tractions.

    This is the existing anisotropic trigonometric problem, with identical
    material, source, boundary data, assembly order and approximation spaces.
    Error orders 12 and 16 independently assess quadrature stability.
    """
    data = TensorData(False)
    mesh = TriangleMesh.unit_square(n)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    started = perf_counter()
    with threadpool_limits(1):
        definition = define_primal_elasticity(
            mesh,
            degree=3,
            minimal_enrichment=False,
            local_refinement=1,
            skeleton=skeleton,
            constitutive=data.constitutive,
            source=data.source,
            dirichlet=data.displacement,
            quadrature_order=10,
        )
        system = assemble(
            definition.problem,
            execution=ExecutionConfig("process", workers=workers, native_threads=1),
        )
        constraints = primal_constraints(definition, system)
        result = system.solve(fixed=definition.problem.fixed, constraints=constraints)
        solution = recover_primal_elasticity(definition, system, result)
        # Original diagnostics use the hybrid convention -C.T. This problem
        # has no D/direct-local g terms, so its physical boundary is -global g.
        if system.global_matrix.nnz or any(
            record.equations.matrix.any() or record.equations.load.any() for record in system.cells
        ):
            raise ValueError("The original-equation observer requires pure primal coupling")
        observed = ProductionObservation(
            system=system,
            applied_boundary=-system.global_load[: system.trace_size],
            fixed=dict(definition.problem.fixed or {}),
            constraints=constraints,
        )
        original = original_diagnostics(
            solution,
            observed,
            [{"displacement_equilibrium": len(field)} for field in result.fields],
        )
        norms = {
            str(order): {
                "displacement_l2": solution.l2_error(data.displacement, order),
                "stress_l2": solution.stress_l2_error(data.stress, order),
            }
            for order in (12, 16)
        }
    low, high = norms.values()
    changes = {
        field: abs(low[field] - high[field]) / max(high[field], np.finfo(float).tiny)
        for field in high
    }
    accepted = bool(original["accepted"] and max(changes.values()) <= 1e-8)
    return {
        "n": n,
        "H": float(np.sqrt(2) / n),
        "macro_cells": len(mesh.cells),
        "trace_dofs": skeleton.size,
        "global_coordinates": len(system.rhs),
        "local_coordinates": len(result.fields[0]),
        **high,
        "quadrature": {
            "assembly_order": 10,
            "error_orders": [12, 16],
            "errors": norms,
            "relative_change": changes,
            "relative_change_limit": 1e-8,
        },
        "original_equations": original,
        "algebraic_residual": result.residual,
        "wall_seconds": perf_counter() - started,
        "accepted": accepted,
    }


def acquire_primal(resolutions: list[int], path: Path, *, workers: int = 32) -> None:
    """Write current scalar-only convergence data without replacing old records."""
    if path.exists():
        raise ValueError("A fresh current-source record is required")
    hashes = source_hashes()
    record: dict[str, Any] = {
        "schema": "pymhm-tutorial-current-vector-norms-v1",
        "case": "Homogeneous anisotropic trigonometric primal elasticity",
        "dimension": 2,
        "spaces": {"local_displacement_degree": 3, "local_refinement": 1, "trace_degree": 1},
        "tensor_kelvin": STIFFNESS.tolist(),
        "boundary": "Exact displacement; homogeneous on the square boundary",
        "source": "Independent TensorData(False) analytic minus div(C epsilon(u))",
        "physical_stress": "C epsilon(u_h); broken raw symmetric stress, not H(div)",
        "execution": {"backend": "process", "workers": workers, "native_threads": 1},
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "source_sha256": hashes,
        "references": ["https://doi.org/10.1051/m2an/2015046"],
        "rows": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    for n in resolutions:
        row = primal_row(n, workers=workers)
        record["rows"].append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "original_equations"}),
            flush=True,
        )
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if not row["accepted"]:
            raise RuntimeError(f"The original equations or error quadrature rejected n={n}")
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in hashes.items()):
        raise RuntimeError("An executed numerical source changed during acquisition")


def acquire_maxwell(path: Path) -> None:
    """Measure the existing exact semidiscrete Maxwell temporal comparison."""
    if path.exists():
        raise ValueError("A fresh current-source record is required")
    hashes = source_hashes()
    with threadpool_limits(1):
        rows = [temporal_row(dt) for dt in (0.001, 0.0005)]
    fields = ("electric_error_l2", "magnetic_error_l2", "combined_error_l2")
    rates = {field: float(np.log2(rows[0][field] / rows[1][field])) for field in fields}
    record = {
        "schema": "pymhm-tutorial-current-vector-norms-v1",
        "case": "TM Maxwell temporal error against the exact constrained semidiscrete flow",
        "source_sha256": hashes,
        "rows": rows,
        "last_interval_rates": rates,
        "original_electric_equation_acceptance": 1e-10,
        "reference": "scipy.linalg.expm of the executed mass/curl constrained ODE",
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "references": ["https://doi.org/10.1137/16M110037X"],
    }
    if any(file_digest(source_file(name, root=ROOT)) != digest for name, digest in hashes.items()):
        raise RuntimeError("An executed numerical source changed during acquisition")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"maxwell_temporal_rates": rates}), flush=True)


def main() -> None:
    """Acquire the requested current-source primal and Maxwell tutorial controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=Path, default=OUTPUT / "primal-elasticity-current.json")
    parser.add_argument("--maxwell-only", action="store_true")
    args = parser.parse_args()
    if args.maxwell_only:
        acquire_maxwell(OUTPUT / "maxwell-current.json")
    else:
        acquire_primal(args.resolutions, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
