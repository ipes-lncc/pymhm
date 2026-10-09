"""Run reproducible PDE verification cases and save numerical results as JSON.

Execute ``python -m examples.verify`` in the selected working directory.
The benchmark problems follow published analytical data, but these triangulations
and low-order local spaces do not reproduce the papers' full numerical tables.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from examples.formulations.application import brinkman as solve_brinkman
from examples.formulations.application import darcy as solve_darcy
from examples.manufactured import (
    darcy_flux,
    darcy_pressure,
    darcy_source,
    stokes_pressure,
    stokes_source,
    stokes_velocity,
)
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh


def run() -> dict[str, object]:
    """Collect convergence data with fixed local refinement and face degrees."""
    darcy, flow = [], []
    for formulation in ("primal", "mixed"):
        for n in (2, 4, 8):
            mesh = TriangleMesh.unit_square(n)
            result = solve_darcy(
                mesh,
                source=darcy_source,
                dirichlet=darcy_pressure,
                formulation=formulation,
                local_refinement=4,
                quadrature_order=6,
            )
            darcy.append(
                {
                    "formulation": formulation,
                    "macro_resolution": n,
                    "local_refinement": 4,
                    "trace_degree": 0,
                    "assembly_quadrature": 6,
                    "error_quadrature": 8,
                    "pressure_l2": result.l2_error(darcy_pressure, order=8),
                    "flux_l2": result.flux_l2_error(darcy_flux, order=8),
                    "macro_balance_max": float(np.max(np.abs(result.conservation_residuals()))),
                }
            )
    for formulation in ("taylor-hood", "usfem"):
        for n in (2, 4):
            mesh = TriangleMesh.unit_square(n)
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
            result = solve_brinkman(
                mesh,
                source=stokes_source,
                dirichlet=stokes_velocity,
                skeleton=skeleton,
                formulation=formulation,
                local_refinement=4,
            )
            flow.append(
                {
                    "formulation": formulation,
                    "macro_resolution": n,
                    "local_refinement": 4,
                    "trace_degree": 1,
                    "assembly_quadrature": 5,
                    "error_quadrature": 8,
                    "velocity_l2": result.l2_error(stokes_velocity, order=8),
                    "pressure_l2": result.pressure_l2_error(stokes_pressure, order=8),
                    "divergence_l2": result.divergence_l2(),
                    "algebraic_residual": result.hybrid.residual,
                }
            )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "darcy": darcy,
        "stokes": flow,
        "evidence": "analytical problem verification; not a matching published table",
    }


def main() -> None:
    """Parse output location and write all measured errors without rounding."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/results/verification.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.verify").main()
