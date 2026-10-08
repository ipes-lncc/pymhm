"""Acquire scalar initial AFW3D convergence records with the published finite spaces.

BDM2/P1/P1 uses r2 and BDM3/P2/P2 uses r1, with normal trace degrees one and
two. No locking sweep, coefficient archive or replay assertion is included.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import weak_stress_elasticity as solve_elasticity_mixed_3d
from examples.minimal_flow_originals import (
    capture_sources,
    observe_originals,
    original_diagnostics,
    sources_unchanged,
    write_record,
)
from examples.mixed_elasticity3d_data import SolenoidalElasticity3D
from pymhm.fem.hdiv.family_3d import cell_quadrature
from pymhm.meshes.mixed import AffineMixedMesh

ROOT = Path(__file__).resolve().parents[1]


def physical_errors(solution: Any, data: Any, order: int) -> dict[str, float]:
    """Integrate displacement, full stress, weak rotation and div(stress)+force separately."""
    result = solution.errors(data.displacement, data.stress, data.rotation, order=order)
    reference, weights = cell_quadrature("tetrahedron", order)
    total = np.longdouble(0)
    for cell, fine in enumerate(solution.local_meshes):
        points = fine.geometry(reference)
        divergence = solution.evaluate(cell, reference)[3]
        force = data.source(points.reshape(-1, 3)).reshape(divergence.shape)
        total += np.sum(
            fine.determinants[:, None] * weights * np.sum((divergence + force) ** 2, axis=-1),
            dtype=np.longdouble,
        )
    result["divergence_l2"] = float(np.sqrt(total))
    return result


def acquire(n: int, degree: int, output: Path) -> dict[str, Any]:
    """Solve one unchanged analytical finite space and persist only scalar diagnostics."""
    if output.exists() or n not in (1, 2, 3) or degree not in (2, 3):
        raise ValueError("Fresh output, n1/n2/n3 and BDM2/BDM3 are required")
    output.mkdir(parents=True)
    hashes = capture_sources(output, [Path(__file__), ROOT / "examples/mixed_elasticity3d_data.py"])
    refinement = 2 if degree == 2 else 1
    assembly = 18 if n == 1 else 12 if n == 2 else 9
    orders = (18, 22) if n == 1 else (14, 16) if n == 2 else (11, 13)
    data = SolenoidalElasticity3D()
    started = perf_counter()
    with threadpool_limits(1), observe_originals() as observed:
        solution = solve_elasticity_mixed_3d(
            AffineMixedMesh.unit_cube(n),
            stress_degree=degree,
            trace_degree=degree - 1,
            local_refinement=refinement,
            lame_lambda=np.inf,
            source=data.source,
            quadrature_order=assembly,
            backend="serial",
            workers=1,
        )
        blocks = []
        for cell, field in enumerate(solution.hybrid.fields):
            stress, displacement, rotation = (
                solution.stress[cell].size,
                solution.displacement[cell].size,
                solution.rotation[cell].size,
            )
            blocks.append(
                {
                    "constitutive_stress": stress,
                    "force_balance": displacement,
                    "weak_symmetry": rotation,
                    "negative_traction_compatibility": len(field)
                    - stress
                    - displacement
                    - rotation,
                }
            )
        original = original_diagnostics(solution, observed, blocks)
        norms = {str(q): physical_errors(solution, data, q) for q in orders}
        moments = {
            "fine_force_moment_max": max(
                float(np.max(abs(a))) for a in solution.fine_force_residuals()
            ),
            "weak_symmetry_moment_max": max(
                float(np.max(abs(a))) for a in solution.weak_symmetry_residuals()
            ),
            "normal_traction_moment_max": max(
                float(np.max(abs(a))) for a in solution.normal_traction_residuals()
            ),
        }
    lower, upper = norms.values()
    changes = {key: abs(lower[key] - upper[key]) for key in upper}
    criterion = 2e-9 * max(1.0, *upper.values())
    accepted = bool(
        original["accepted"]
        and max(changes.values()) <= criterion
        and max(moments.values()) <= 1e-9
    )
    if not sources_unchanged(hashes):
        raise ValueError("An original numerical owner changed during execution")
    origins = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (
                path.is_relative_to(ROOT)
                and path.suffix == ".py"
                and path.relative_to(ROOT).parts[0] in ("src", "examples")
            ):
                origins[name] = path.relative_to(ROOT).as_posix()
    record: dict[str, Any] = {
        "schema": "pymhm-initial-analytical-scalar-record-v1",
        "family": "mixed-elasticity3d",
        "case": f"bdm{degree}",
        "resolution": n,
        "acquisition_uuid": str(uuid4()),
        "source_sha256": hashes,
        "stress_degree": degree,
        "displacement_degree": degree - 1,
        "rotation_degree": degree - 1,
        "trace_degree": degree - 1,
        "local_refinement": refinement,
        "lame_lambda": "infinity",
        "lame_mu": 1.0,
        "macro_cells": len(solution.local_meshes),
        "fine_cells": sum(len(fine.cells) for fine in solution.local_meshes),
        "assembly_order": assembly,
        "norm_orders": list(orders),
        "norms_by_order": norms,
        "terminal_norms": upper,
        "physical_error_quadrature_absolute_changes": changes,
        "physical_error_quadrature_criterion": criterion,
        "original_equations": original,
        "physical_moment_diagnostics": moments,
        "accepted": accepted,
        "elapsed_seconds": perf_counter() - started,
        "conventions": (
            "Physical row-wise H(div) Cauchy stress, -div(stress)=force; "
            "canonical multiplier -stress*n; axial skew-gradient weak rotation; "
            "zero mean hydrostatic pressure at incompressibility"
        ),
        "source": "Independently differentiated solenoidal trigonometric AFW2007 Eq7.1 data",
        "workers": 1,
        "native_threads": 1,
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "historical_literature_reproduction": False,
        "uniform_inf_sup_proven": False,
        "independent_native_whole_case_verified": False,
        "source_changed": False,
        "python_version": platform.python_version(),
        "runtime_module_origins": origins,
    }
    if any(origin not in hashes for origin in record["runtime_module_origins"].values()):
        raise ValueError("An actual executed numerical module was not captured")
    write_record(output / "record.json", record)
    print(
        json.dumps(
            {
                "case": record["case"],
                "resolution": n,
                "accepted": accepted,
                "norms": upper,
                "original": original["full_uncondensed_relative_to_physical_rhs"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    if not accepted:
        raise ArithmeticError("Original/physical norm criteria failed; diagnostic record preserved")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--degree", type=int, choices=(2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.resolution, args.degree, args.output)
