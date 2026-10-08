"""Acquire an initial L18 oscillatory-modulus study in the selected BDM2/P1/P1 space.

The macro mesh is the original unit_square(4), with interior P1 negative
traction, full exterior P2 trace and h_in=h_sk/2. This selected three-level
study does not claim to reproduce every historical table entry or enrichment.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import weak_stress_elasticity as solve_elasticity_mixed
from examples.minimal_flow_originals import (
    capture_sources,
    observe_originals,
    original_diagnostics,
    sources_unchanged,
    write_record,
)
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh

ROOT = Path(__file__).resolve().parents[1]


def acquire(segments: int, output: Path) -> dict[str, Any]:
    """Run unchanged oscillatory analytical physics and persist scalar field errors only."""
    if segments not in (1, 2, 4) or output.exists():
        raise ValueError("Segments 1/2/4 and a fresh output directory are required")
    sys.path.insert(0, str(ROOT / "examples"))
    data = importlib.import_module("examples.plot_mixed_elasticity")
    output.mkdir(parents=True)
    hashes = capture_sources(
        output,
        [
            Path(__file__),
            ROOT / "examples/plot_mixed_elasticity.py",
            ROOT / "examples/plot_mesh.py",
        ],
    )
    data.check_manufactured_data()
    mesh = TriangleMesh.unit_square(4)
    boundary = set(mesh.boundary_faces)
    refinement, poisson, assembly, orders = 2 * segments, 0.3, 16, (16, 20)
    skeleton = SkeletonSpace(
        mesh,
        tuple(
            FaceSpace.uniform(2, refinement) if face in boundary else FaceSpace.uniform(1, segments)
            for face in range(len(mesh.faces))
        ),
        2,
    )
    started = perf_counter()
    with threadpool_limits(1), observe_originals() as observed:
        solution = solve_elasticity_mixed(
            mesh,
            skeleton=skeleton,
            stress_degree=2,
            enrichment=0,
            local_refinement=refinement,
            quadrature_order=assembly,
            lame_mu=lambda x: data.oscillatory_modulus(x) / (2 * (1 + poisson)),
            lame_lambda=lambda x: (
                data.oscillatory_modulus(x) * poisson / ((1 + poisson) * (1 - 2 * poisson))
            ),
            source=lambda x: data.oscillatory_fields(x)[2],
            dirichlet=lambda x: data.oscillatory_fields(x)[0],
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
        norms = {
            str(q): {
                "displacement_l2": solution.l2_error(lambda x: data.oscillatory_fields(x)[0], q),
                "stress_l2": solution.stress_l2_error(lambda x: data.oscillatory_fields(x)[1], q),
                "rotation_l2": solution.rotation_l2_error(
                    lambda x: data.oscillatory_fields(x)[3], q
                ),
                "stress_divergence_l2": solution.divergence_l2_error(
                    lambda x: -data.oscillatory_fields(x)[2], q
                ),
            }
            for q in orders
        }
        moments = {
            "fine_force_max": max(float(np.max(abs(a))) for a in solution.fine_force_residuals()),
            "weak_symmetry_max": max(
                float(np.max(abs(a))) for a in solution.weak_symmetry_residuals()
            ),
            "normal_traction_max": max(
                float(np.max(abs(a))) for a in solution.normal_traction_residuals()
            ),
        }
    lower, upper = norms.values()
    changes = {
        key: abs(lower[key] - upper[key]) / max(abs(upper[key]), np.finfo(float).tiny)
        for key in upper
    }
    accepted = bool(original["accepted"] and max(changes.values()) <= 1e-8)
    if not sources_unchanged(hashes):
        raise ValueError("A numerical owner changed during execution")
    record: dict[str, Any] = {
        "schema": "pymhm-initial-analytical-scalar-record-v1",
        "family": "l18-oscillatory",
        "case": "bdm2-trace1-enrichment0",
        "resolution": segments,
        "acquisition_uuid": str(uuid4()),
        "source_sha256": hashes,
        "macro_cells": len(mesh.cells),
        "fine_cells": sum(len(fine.cells) for fine in solution.local_meshes),
        "macro_geometry": "unit_square(4): 32 diagonal triangles",
        "macro_h": 0.25,
        "stress_degree": 2,
        "displacement_degree": 1,
        "rotation_degree": 1,
        "trace_degree": 1,
        "enrichment": 0,
        "local_refinement": refinement,
        "poisson_ratio": poisson,
        "assembly_order": assembly,
        "norm_orders": list(orders),
        "norms_by_order": norms,
        "terminal_norms": upper,
        "physical_error_quadrature_relative_changes": changes,
        "original_equations": original,
        "physical_moment_diagnostics": moments,
        "accepted": accepted,
        "elapsed_seconds": perf_counter() - started,
        "conventions": (
            "Physical row-wise H(div) Cauchy stress; lambda=-stress*n; weak P1 rotation; "
            "full nonhomogeneous displacement boundary data"
        ),
        "study_scope": (
            "Selected original geometry/data/BDM2 finite-space initial study; "
            "unresolved historical rotation column is not fitted"
        ),
        "workers": 1,
        "native_threads": 1,
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "historical_literature_reproduction": False,
        "uniform_inf_sup_proven": False,
        "independent_native_whole_case_verified": False,
    }
    write_record(output / "record.json", record)
    print(
        json.dumps(
            {
                "segments": segments,
                "accepted": accepted,
                "norms": upper,
                "original": original["full_uncondensed_relative_to_physical_rhs"],
            }
        ),
        flush=True,
    )
    if not accepted:
        raise ArithmeticError("Original/norm criteria failed; record preserved")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.segments, args.output)
