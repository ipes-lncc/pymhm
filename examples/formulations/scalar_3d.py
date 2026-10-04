"""User-defined conservative scalar equations on tetrahedral or polyhedral macrocells.

Geometry supplies oriented trace pairings; the same four-block equations apply
to both macro meshes. Shared numerical kernels integrate the scalar volume form.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm._legacy.models.transport.polyhedral import (
    _boundary as _polygonal_boundary,
)
from pymhm._legacy.models.transport.polyhedral import (
    polygonal_trace_coupling as polygonal_trace_coupling,
)
from pymhm._legacy.models.transport.rad_3d import RAD3DSolution, tetra_rad_operators
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.traces.triangle_3d import (
    _boundary as _tetrahedral_boundary,
)
from pymhm.fem.traces.triangle_3d import (
    tetra_trace_coupling as tetra_trace_coupling,
)

polygonal_boundary_pairing = _polygonal_boundary
tetra_boundary_pairing = _tetrahedral_boundary


@dataclass(frozen=True)
class Scalar3DDiscretization:
    """Declared volume space and geometry-specific trace integration callbacks.

    A pairing callback returns B, including the global normal orientation.
    Boundary pairing returns weak data and fixed outward half-advection fluxes.
    Neither callback selects a PDE, local solver or global elimination method.
    """

    mesh: Any
    skeleton: Any
    trace_pairing: Callable[..., Any]
    boundary_pairing: Callable[..., Any]
    degree: int = 4
    refinement: int = 2
    order: int = 6


def scalar_equations_3d(
    cell: int, *, data: Scalar3DDiscretization, coefficients: dict[str, Any]
) -> LocalEquations:
    """Declare A u+B lambda=f and -B.T u=0 with physical constant moments.

    A integrates diffusion, skew conservative transport, c+div(beta)/2 and
    optional full-residual SUPG. Derivatives of variable coefficients are explicit
    inputs. The multiplier is (-K grad(u)+beta*u/2).n, not total physical flux.
    """
    fine = data.mesh.submesh(cell, data.refinement)
    a, load, moments, pure = tetra_rad_operators(fine, data.degree, **coefficients)
    b = data.trace_pairing(data.mesh, cell, fine, data.skeleton, data.degree)
    constant = np.ones((len(load), 1))
    return LocalEquations(
        a,
        load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        kernel=constant if pure else None,
        coarse_basis=None if pure else constant,
        moments=moments[:, None],
        metadata=(fine, moments),
    )


def scalar_problem_3d(
    data: Scalar3DDiscretization,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
) -> MultiscaleProblem[int]:
    """Declare a full weak Dirichlet example on the supplied original macrofaces.

    The face polynomial and local degree/refinement are chosen by the caller;
    no new trace unknowns are introduced on a polyhedron's triangulation edges.
    """
    order = max(data.order, data.degree + 2)
    boundary, fixed = data.boundary_pairing(data.skeleton, dirichlet, {}, order)
    coefficients = dict(
        diffusion=diffusion,
        velocity=velocity,
        reaction=reaction,
        source=source,
        velocity_divergence=velocity_divergence,
        diffusion_divergence=diffusion_divergence,
        stabilization=stabilization,
        order=order,
    )
    return MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(len(data.mesh.cells))]),
        partial(scalar_equations_3d, data=data, coefficients=coefficients),
        range(len(data.mesh.cells)),
        data.skeleton.size,
        (1,) * len(data.mesh.cells),
        fixed=fixed,
    )


def recover_scalar_3d(
    data: Scalar3DDiscretization,
    system: MultiscaleSystem,
    result: HybridSolution,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0, 0.0),
) -> RAD3DSolution:
    """Interpret scalar coefficients and physical flux -K grad(u)+beta*u without averaging."""
    return RAD3DSolution(
        tuple(record[0] for record in system.local_metadata),
        result.fields,
        data.degree,
        diffusion,
        velocity,
        result,
        data.skeleton,
        result.residual,
    )
