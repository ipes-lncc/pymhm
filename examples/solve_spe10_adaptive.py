"""Acquire the published L09 local spaces with an original macro-adaptive policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.solve_spe10 import load_layer, pressure_boundary
from examples.spe10_adaptive import DATA, ROOT, hashes, mesh_rectangle, natural_faces
from pymhm.adaptive_darcy import solve_adaptive_darcy


def acquire(levels: int = 5) -> list[dict[str, Any]]:
    """Preserve P2/r2/P0/RT2 and archive every actual adaptive mesh and coefficient."""
    DATA.mkdir(parents=True, exist_ok=True)
    fingerprint = hashes()
    fingerprint["examples/solve_spe10_adaptive.py"] = hashlib.sha256(
        (ROOT / "examples/solve_spe10_adaptive.py").read_bytes()
    ).hexdigest()
    mesh = mesh_rectangle(16, 16)
    started = perf_counter()
    result = solve_adaptive_darcy(
        mesh,
        iterations=levels,
        theta=0.5,
        maximum_cells=20000,
        trace_degree=0,
        reconstruction_degree=2,
        estimator_order=6,
        degree=2,
        local_refinement=2,
        permeability=load_layer(),
        dirichlet=pressure_boundary,
        neumann=natural_faces(mesh),
        quadrature_order=6,
    )
    elapsed = perf_counter() - started
    records = []
    for level, (solution, estimate, marked) in enumerate(
        zip(result.solutions, result.estimators, result.marked, strict=True)
    ):
        coarse = solution.skeleton.mesh
        archive = f"mhm-level{level}.npz"
        np.savez_compressed(
            DATA / archive,
            macro_points=coarse.points,
            macro_cells=coarse.cells,
            local_points=np.stack([fine.points for fine in solution.local_meshes]),
            local_cells=np.stack([fine.cells for fine in solution.local_meshes]),
            pressure=np.stack(solution.pressure),
            recovered_pressure=np.stack(estimate.potential.local_values),
            reconstructed_flux=np.stack(estimate.reconstructed_flux.flux),
            trace=solution.hybrid.trace,
            local_squared=estimate.local_squared,
            flux_defect=estimate.flux_defect,
            nonconformity=estimate.nonconformity,
            divergence_defect=estimate.divergence_defect,
            oscillation=estimate.oscillation,
            marked=marked,
        )
        bottom = [
            face
            for face in coarse.boundary_faces
            if np.all(coarse.points[coarse.faces[face], 1] == 0)
        ]
        inflow = -sum(
            coarse.lengths[face] * solution.hybrid.trace[solution.skeleton.dofs(face)][0]
            for face in bottom
        )
        row = {
            "level": level,
            "archive": archive,
            "macro_triangles": len(coarse.cells),
            "local_degree": 2,
            "local_refinement": 2,
            "trace_degree": 0,
            "reconstruction_degree": 2,
            "global_dofs": solution.skeleton.size + len(coarse.cells),
            "theta": 0.5,
            "assembly_order": 6,
            "estimator_order": 6,
            "marked": int(marked.sum()),
            "estimator": estimate.total,
            "macro_balance_linf": float(np.max(abs(solution.conservation_residuals()))),
            "continuous_equilibrium_l2_max": float(np.max(estimate.equilibrium_defect)),
            "inflow": float(inflow),
            "minimum_shape_quality": float(
                np.min(
                    4
                    * np.sqrt(3)
                    * coarse.areas
                    / np.sum(coarse.lengths[coarse.cell_faces] ** 2, axis=1)
                )
            ),
            "source_hashes": fingerprint,
            "source_changed_during_solve": any(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                for name, digest in fingerprint.items()
            ),
            "campaign_seconds": elapsed,
        }
        records.append(row)
        print("adaptive", row, flush=True)
    (DATA / "adaptive.json").write_text(json.dumps(records, indent=2) + "\n")
    return records


def main() -> None:
    """Run the numerical acquisition independently from the light CI suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=5)
    options = parser.parse_args()
    with threadpool_limits(1):
        acquire(options.levels)


if __name__ == "__main__":
    main()
