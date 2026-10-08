"""User-written P1/P1 Herrmann equations with physical rigid-motion moments.

This application is the constant-shear GaLS configuration of the introductory
nearly incompressible study. Basix tabulation, trace integration and element
scatter belong to shared FEM kernels; no physical solver or local factory is
called. Coefficients use displacement components interleaved at each node,
followed by pressure nodes, with exactly the declared global-centered rigid modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import LocalEquations
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.vector.pressure import triangle_elasticity_pressure_operators
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import ElasticitySolution


@dataclass(frozen=True)
class DisplacementPressureSpace:
    """Declare continuous P1/P1 locals and a two-component macroface space."""

    mesh: TriangleMesh
    skeleton: SkeletonSpace
    refinement: int = 4
    quadrature_order: int = 8
    degree: int = 1
    pressure_degree: int = 1


def displacement_pressure_equations(
    cell: int,
    *,
    space: DisplacementPressureSpace,
    source: Any,
    lame_lambda: Any,
    lame_mu: Any = 1.0,
    lame_mu_gradient: Any = (0.0, 0.0),
    shear_bounds: tuple[float, float, float] | None = None,
    formulation: str = "gals",
    stabilization_alpha: float | None = None,
) -> LocalEquations:
    """Declare Herrmann elasticity, optional residual stabilization and rigid moments.

    Pressure is p=-lambda*div(u), with zero compressibility for lambda=infinity.
    The residual div(2*mu*epsilon(u))-grad(p) includes analytical grad(mu).
    GaLS Pk/Pk subtracts its product and adds the source pairing with alpha below
    the physical inverse/coefficient bound. Taylor--Hood Pk/P(k-1) declares no
    stabilization. B is negative Cauchy traction; C=-B.T. Rigid modes and their
    moment coordinates use the explicit macro mesh center.
    """
    expected = space.degree if formulation == "gals" else space.degree - 1
    if space.pressure_degree != expected:
        raise ValueError("pressure degree must match the declared elasticity formulation")
    fine = space.mesh.submesh(cell, space.refinement)
    forms = triangle_elasticity_pressure_operators(
        fine,
        degree=space.degree,
        formulation=formulation,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=max(space.quadrature_order, space.degree + 2),
        diameter=float(max(space.mesh.lengths[space.mesh.cell_faces[cell]])),
        rigid_center=space.mesh.points.mean(axis=0),
        stabilization_alpha=stabilization_alpha,
        lame_mu_gradient=lame_mu_gradient,
        shear_bounds=shear_bounds,
    )
    nv, size = len(forms.displacement_nodes), len(forms.load)
    b = np.zeros((size, len(space.skeleton.cell_dofs(cell))))
    b[: 2 * nv] = np.kron(
        trace_coupling(space.mesh, cell, fine, space.skeleton, space.degree), np.eye(2)
    )
    selector = sparse.eye(size, format="csr")
    return LocalEquations(
        forms.matrix,
        forms.load,
        b,
        -b.T,
        space.skeleton.cell_dofs(cell),
        kernel=forms.kernel,
        moments=forms.rigid_moments,
        metadata=(
            fine,
            nv,
            forms.pressure_moments,
            forms.stabilization_alpha,
            forms.rigid_moments,
            forms.compliance_moments,
            forms.compliance_scale,
        ),
        field_data=(
            nodal_field(
                "displacement", fine, space.degree, components=2, reconstruction=selector[: 2 * nv]
            ),
            nodal_field("pressure", fine, space.pressure_degree, reconstruction=selector[2 * nv :]),
        ),
    )


def displacement_pressure_fields(
    system: MultiscaleSystem,
    solution: HybridSolution,
    space: DisplacementPressureSpace,
    lame_lambda: Any,
    lame_mu: Any = 1.0,
    formulation: str = "gals",
) -> ElasticitySolution:
    """Interpret executed P1 coefficients as displacement, Herrmann pressure and stress."""
    return ElasticitySolution(
        skeleton=space.skeleton,
        local_meshes=tuple(record[0] for record in system.local_metadata),
        values=tuple(
            field[: 2 * record[1]].reshape(-1, 2)
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        pressure=tuple(
            field[2 * record[1] :]
            for field, record in zip(solution.fields, system.local_metadata, strict=True)
        ),
        hybrid=solution,
        degree=space.degree,
        pressure_degree=space.pressure_degree,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        formulation=formulation,
        stabilization=tuple(record[3] for record in system.local_metadata),
    )
