"""Declare conservative SUPG equations from shared FEM and residual-scale kernels."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm import LocalEquations
from pymhm.fem.scalar.transport import triangle_transport_operators
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.scalar.unusual import UnusualParameters
from pymhm.postprocessing.nodal import nodal_field

from .scalar import ScalarDiscretization


def streamline_equations(
    cell: int,
    *,
    data: ScalarDiscretization,
    velocity: Any,
    velocity_divergence: Any,
    diffusion_divergence: Any,
    reaction: Any,
    source: Any,
    stabilization: str = "supg",
    unusual_parameters: UnusualParameters | None = None,
) -> LocalEquations:
    """Write conservative skew Galerkin and its complete streamline residual pairing.

    The strong trial residual is ``-div(K grad(u))+beta.grad(u)+(c+div(beta))*u``
    and the test residual is ``beta.grad(v)``. Analytical divergence data are
    explicit inputs; sampled coefficients do not certify continuum bounds.
    The positive tau comes from the existing pure residual-scale owner. The
    trace multiplier is ``(-K grad(u)+beta*u/2).n`` with the fixed normal map.
    Global weak Dirichlet moments and physical constraints are caller choices.
    """
    fine = data.local_mesh(cell)
    forms = triangle_transport_operators(
        fine,
        degree=data.degree,
        diffusion=data.diffusion,
        velocity=velocity,
        velocity_divergence=velocity_divergence,
        diffusion_divergence=diffusion_divergence,
        reaction=reaction,
        source=source,
        stabilization=stabilization,
        order=data.order,
        unusual_parameters=unusual_parameters,
    )
    b = trace_coupling(data.mesh, cell, fine, data.skeleton, data.degree)
    constant = np.ones((len(forms.load), 1))
    return LocalEquations(
        forms.matrix,
        forms.load,
        b,
        -b.T,
        data.skeleton.cell_dofs(cell),
        kernel=constant if forms.pure_diffusion else None,
        coarse_basis=None if forms.pure_diffusion else constant,
        moments=forms.moments[:, None],
        metadata=(
            fine,
            forms.moments,
            forms.pure_diffusion,
            forms.zero_reaction_divergence,
            len(forms.nodes),
            np.empty(0, dtype=np.int64),
        ),
        field_data=(nodal_field("scalar", fine, data.degree),),
    )
