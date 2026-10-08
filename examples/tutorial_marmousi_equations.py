"""A declared coarse control on the full pinned Marmousi acoustic physical input.

The geometry, pixel coefficients, frequency, source and exterior conditions
match the current primary-data case. This intentionally coarser discretization
is an API/operator control and does not establish wave accuracy or reproduce
Table 6.1. The full current Q3/H20 study remains a separate acquisition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examples.formulations.original import solve_original
from examples.marmousi_data import MarmousiMaterial, download_marmousi_data, load_marmousi_crop
from examples.tutorial_helmholtz_equations import (
    acoustic_prescribed,
    acoustic_problem,
    recover_acoustic,
)
from pymhm.core.multiscale import MultiscaleSystem, assemble
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.acoustics import HelmholtzSolution


@dataclass(frozen=True)
class MarmousiControl:
    """Declared material, executed equations, physical fields and direct coefficient defect."""

    material: MarmousiMaterial
    system: MultiscaleSystem
    solution: HelmholtzSolution
    original_difference: float


def acoustic_control(directory: str | Path, *, order: int = 4) -> MarmousiControl:
    """Solve the full current physical input with explicitly declared coarse Q1/P0 spaces.

    The macro grid is 32 by 8 over 10,240 by 2,560 metres. Each macrocell uses
    Q1 pressure on 2 by 2 fine rectangles and one P0 normal mode per macroface.
    Density and bulk-modulus integrals resolve every original 5-metre material
    pixel, regardless of this pressure-grid resolution. Omega is 40*pi rad/s;
    the unit discrete point source is at (5000,50). Top pressure is weakly zero
    and the three other sides impose homogeneous outgoing impedance.
    The direct comparison independently solves the literal original A/B/C/D
    coefficient matrix on these same spaces, not a separate refined reference.
    """
    download_marmousi_data(directory)
    material = load_marmousi_crop(directory)
    mesh = CartesianMacroMesh(32, 8, (0.0, 10240.0, 0.0, 2560.0))
    omega = 40 * np.pi
    skeleton = helmholtz_skeleton(mesh, omega, degree=0)
    absorbing = {
        int(face): 0j
        for face in mesh.boundary_faces
        if not np.all(mesh.points[mesh.faces[face], 1] == 0)
    }
    problem = acoustic_problem(
        mesh,
        omega=omega,
        density=material.density,
        bulk_modulus=material.bulk_modulus,
        point_sources=((5000.0, 50.0, 1.0),),
        dirichlet=0.0,
        absorbing=absorbing,
        skeleton=skeleton,
        degree=1,
        local_refinement=2,
        quadrature_order=order,
    )
    system = assemble(problem)
    fixed = acoustic_prescribed(system)
    solved = system.solve(fixed=fixed)
    original = solve_original(system, fixed=fixed)
    for actual, expected in zip(
        (*solved.fields, solved.trace), (*original.fields, original.trace), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
    difference = max(
        float(np.max(np.abs(actual - expected)))
        for actual, expected in zip(
            (*solved.fields, solved.trace), (*original.fields, original.trace), strict=True
        )
    )
    return MarmousiControl(material, system, recover_acoustic(problem, system, solved), difference)


def quadrature_difference(first: MarmousiControl, second: MarmousiControl) -> dict[str, Any]:
    """Compare identical physical spaces with two explicitly executed pixel rules.

    Coefficient norms measure quadrature agreement only. No physical field
    accuracy or paper reproduction follows from a small quadrature difference.
    """
    first_values = np.concatenate((*first.solution.pressure, first.solution.trace))
    second_values = np.concatenate((*second.solution.pressure, second.solution.trace))
    np.testing.assert_allclose(first_values, second_values, atol=1e-12, rtol=1e-10)
    absolute = float(np.linalg.norm(first_values - second_values))
    return {
        "coefficient_absolute_difference": absolute,
        "coefficient_relative_difference": absolute / float(np.linalg.norm(second_values)),
        "original_matrix_absolute_difference": max(
            first.original_difference, second.original_difference
        ),
        "scope": "full current physical input; coarse discretization accuracy unestablished",
    }
