"""Acquire one declared Marmousi acoustic MHM configuration from Table 6.1.

Material centres are pinned independently of the pressure result. The chosen
primary-data crop is a declared input, not an identification of historical
arrays. Full complex local fields and all incident sampling conventions are
preserved separately from the sampled comparison norm.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.helmholtz_trace_family import verify_helmholtz_solution
from examples.marmousi_data import load_marmousi_crop
from examples.tutorial_helmholtz_equations import solve_acoustic
from pymhm.fem.scalar.quadrilateral import qk_basis
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]


def evaluate_fields(
    pressure: Any,
    mesh: CartesianMacroMesh,
    refinement: int,
    degree: int,
    points: Any,
    *,
    side: tuple[int, int] = (1, 1),
) -> Any:
    """Evaluate one incident macro field, retaining discontinuous vertex values.

    The signs select a positive or negative incident macro in each coordinate
    at a macro interface. Exterior points use the sole available incident cell.
    The numerical local coordinate convention is row-major equispaced Qk.
    """
    points = np.asarray(points, dtype=float)
    lower = np.asarray(mesh.bounds)[[0, 2]]
    coordinate = (points - lower) / mesh.spacing
    counts = np.array([mesh.nx, mesh.ny])
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("sampling points must be finite two-dimensional coordinates")
    if side not in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        raise ValueError("incident-side selectors must be -1 or 1")
    if np.any(coordinate < -1e-12) or np.any(coordinate > counts + 1e-12):
        raise ValueError("sampling points lie outside the macro rectangle")
    owners = np.floor(coordinate).astype(int)
    for axis in range(2):
        interfaces = np.isclose(
            coordinate[:, axis], np.rint(coordinate[:, axis]), rtol=0, atol=1e-12
        )
        owners[interfaces, axis] = np.rint(coordinate[interfaces, axis]).astype(int) - (
            side[axis] < 0
        )
    owners = np.clip(owners, 0, counts - 1)
    local = (coordinate - owners) * refinement
    cells = np.minimum(np.maximum(np.floor(local).astype(int), 0), refinement - 1)
    basis, _ = qk_basis(degree, np.clip(local - cells, 0, 1))
    row = owners[:, 0] + mesh.nx * owners[:, 1]
    width = refinement * degree + 1
    first = degree * cells[:, 0] + width * degree * cells[:, 1]
    offsets = np.array([j * width + i for j in range(degree + 1) for i in range(degree + 1)])
    coefficients = np.asarray(pressure)[row[:, None], first[:, None] + offsets]
    return np.einsum("qi,qi->q", basis, coefficients)


def source_hashes() -> dict[str, str]:
    """Identify the numerical kernels and original acquisition code actually executed."""
    names = [
        "examples/marmousi_campaign.py",
        "examples/marmousi_data.py",
        "examples/tutorial_helmholtz_equations.py",
        "examples/campaign_provenance.py",
        "examples/helmholtz_trace_family.py",
        *(path.relative_to(ROOT).as_posix() for path in sorted((ROOT / "src/pymhm").rglob("*.py"))),
    ]
    return current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )


def main() -> None:
    """Solve and archive a complete published space configuration without coefficient fitting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--H", type=int, choices=(20, 40, 80), default=20)
    parser.add_argument("--trace-degree", type=int, choices=range(5), default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/marmousi")
    args = parser.parse_args()
    material = load_marmousi_crop(args.data)
    mesh = CartesianMacroMesh(10240 // args.H, 2560 // args.H, (0, 10240, 0, 2560))
    skeleton = helmholtz_skeleton(mesh, 40 * np.pi, degree=args.trace_degree)
    absorbing = {
        int(face): 0j
        for face in mesh.boundary_faces
        if not np.all(mesh.points[mesh.faces[face], 1] == 0)
    }
    before = source_hashes()
    start = time.perf_counter()
    with threadpool_limits(1):
        solution = solve_acoustic(
            mesh,
            omega=40 * np.pi,
            density=material.density,
            bulk_modulus=material.bulk_modulus,
            point_sources=((5000, 50, 1.0),),
            absorbing=absorbing,
            skeleton=skeleton,
            degree=3,
            local_refinement=args.H // 5 * 2,
            quadrature_order=8,
            backend="process" if args.workers > 1 else "serial",
            workers=args.workers,
        )
        elapsed = time.perf_counter() - start
        balance = float(np.max(abs(solution.conservation_residuals())))
        field_diagnostics = verify_helmholtz_solution(solution)
    if before != source_hashes():
        raise RuntimeError("acquisition sources changed during the acoustic solve")
    coefficients = np.asarray(solution.pressure)
    x, y = np.meshgrid(np.linspace(0, 10240, 513), np.linspace(0, 2560, 129))
    points = np.column_stack((x.ravel(), y.ravel()))
    sides = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    samples = np.array(
        [
            evaluate_fields(coefficients, mesh, args.H // 5 * 2, 3, points, side=side)
            for side in sides
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"mhm-H{args.H}-ell{args.trace_degree}"
    archive = args.output / f"{name}.npz"
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        pressure=coefficients,
        trace=solution.trace,
        sample_points=points,
        sample_pressure=samples,
        incident_sides=np.asarray(sides),
    )
    record = {
        "method": "MHM Helmholtz 2020; continuous local Q3 and polynomial physical-flux trace",
        "doi": "10.1137/19M1255616",
        "material": material.provenance,
        "H_m": args.H,
        "macro_shape": [mesh.nx, mesh.ny],
        "macro_cells": len(mesh.cells),
        "local_refinement": args.H // 5 * 2,
        "local_degree": 3,
        "trace_degree": args.trace_degree,
        "omega": 40 * np.pi,
        "point_source": [5000, 50, 1.0],
        "point_source_allocation": [
            {"macro_cell": cell, "sources": values.tolist()}
            for cell, values in enumerate(solution.point_sources)
            if len(values)
        ],
        "point_source_convention": "Incident-angle allocation; total unit strength preserved",
        "boundary": "Top weak Dirichlet zero; other sides outgoing first-order absorption zero",
        "global_complex_dofs_total": skeleton.size // 2,
        "global_complex_dofs_free": (2 * len(mesh.cells) - mesh.ny) * (args.trace_degree + 1),
        "algebraic_residual": solution.hybrid.residual,
        "macro_balance_max": balance,
        "requested_assembly_order": 8,
        "assembly_order": solution.quadrature_order,
        **field_diagnostics,
        "elapsed_seconds": elapsed,
        "workers": args.workers,
        "archive": archive.name,
        "archive_sha256": file_digest(archive),
        "source_sha256": before,
        "source_changed_during_run": False,
        "sampling": "513 by129 nodes; four incident macro values, without averaging",
        "historical_article_arrays_identified": False,
    }
    (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
