"""Initial spatial refinement of the paper's analytical incident plane wave."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from examples.helmholtz_basis_archive import basis_payload
from examples.helmholtz_campaign import AcousticWave, norms
from examples.helmholtz_trace_family import verify_helmholtz_solution
from examples.minimal_wave_convergence import digest, quadrature_change, require_original, write
from pymhm.helmholtz import solve_helmholtz
from pymhm.helmholtz_spaces import helmholtz_skeleton
from pymhm.quadrilateral import CartesianMacroMesh, _cardinals


def acquire(output: Path) -> dict[str, Any]:
    """Keep omega=10*pi, theta=pi/13, local Q4/r2 and polynomial face P2 unchanged."""
    wave = AcousticWave(10 * np.pi)
    rows = []
    for n in (8, 12, 16):
        started = perf_counter()
        mesh = CartesianMacroMesh(n)
        skeleton = helmholtz_skeleton(mesh, wave.omega, degree=2, oscillatory=False)
        solution = solve_helmholtz(
            mesh,
            omega=wave.omega,
            skeleton=skeleton,
            degree=4,
            local_refinement=2,
            absorbing=wave.absorbing,
            quadrature_order=10,
            backend="serial",
            workers=1,
        )
        checks = verify_helmholtz_solution(solution)
        require_original(checks)
        low, high = norms(solution, wave, 12), norms(solution, wave, 16)
        error_keys = ("pressure_l2", "gradient_l2", "energy_relative_error")
        sensitivity = quadrature_change(
            {key: low[key] for key in error_keys}, {key: high[key] for key in error_keys}
        )
        if sensitivity > 1e-6:
            raise ArithmeticError("Helmholtz physical error quadrature is unresolved")
        field = output / f"n{n}-fields.npz"
        np.savez_compressed(
            field,
            pressure=np.asarray(solution.pressure),
            trace=solution.trace,
            macro_points=mesh.points,
            macro_cells=mesh.cells,
            macro_faces=mesh.faces,
            local_points=np.asarray([fine.points for fine in solution.local_meshes]),
            local_cells=np.asarray([fine.cells for fine in solution.local_meshes]),
            actual_cardinal_ascending_power_matrix=np.asarray(_cardinals(4)),
            local_degree=np.asarray(4),
            local_refinement=np.asarray(2),
            **basis_payload(skeleton),
        )
        rows.append(
            {
                "level": n,
                "macro_edge": 1 / n,
                "macro_cells": n * n,
                "norms": {key: high[key] for key in error_keys},
                "original_equations": checks,
                "quadrature_orders": [12, 16],
                "quadrature_relative_change": sensitivity,
                "archive": field.name,
                "archive_sha256": digest(field),
                "elapsed_seconds": perf_counter() - started,
            }
        )
        write(output / "progress.json", {"rows": rows})
        print(f"Helmholtz n={n}: {rows[-1]['elapsed_seconds']:.3f}s", flush=True)
    return {
        "reference": "Chaumont-Frelet and Valentin (2020), doi:10.1137/19M1255616",
        "scope": (
            "Initial three-level refinement of published analytical plane-wave data "
            "on Cartesian macro meshes"
        ),
        "configuration": {
            "omega": wave.omega,
            "angle": wave.angle,
            "bounds": [0, 1, 0, 1],
            "local_space": "continuous Q4",
            "local_refinement": 2,
            "trace_space": "face P2",
            "boundary": "grad(p).n-i*omega*p=analytical absorbing datum everywhere",
            "assembly_order": 10,
        },
        "basis_contract": (
            "Actual cached Q4 cardinal matrices, executed face basis, "
            "geometry and original coefficients retained"
        ),
        "rows": rows,
        "plot_fields": ["pressure_l2", "gradient_l2"],
        "level_label": "Macro resolution n",
        "norm_label": "Physical L2 error",
        "figure_title": "Helmholtz plane wave: initial refinement",
        "exact_solution_available": True,
        "asymptotic_convergence_verified": False,
        "limitations": [
            "Three initial levels do not establish asymptotic rates "
            "or the full angular/stability studies",
            "Cartesian local Q4 spaces are the declared analytical variant; "
            "every published mesh has not been identified",
        ],
    }
