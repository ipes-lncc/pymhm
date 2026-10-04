"""Declare residual jump forms and enrich scalar hybrid fields in an application.

This formulation uses the original signed fine-trace integration kernel. Its
global Equation is written after local response maps are known; the shared core
adds it without retabulating or rebuilding their executed coefficient bases.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from examples.formulations.darcy import DarcyDefinition
from pymhm import Equation, with_global_equation
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.linalg.linear import solve_linear
from pymhm.methods.petrov_galerkin import PGMHMSolution, _penalty
from pymhm.methods.robin import _ellipticity


def jump_equation(
    definition: DarcyDefinition,
    system: MultiscaleSystem,
    *,
    dirichlet: Any,
    alpha: float,
) -> tuple[Equation, tuple[Any, ...], float]:
    """Write J.T*W*J and J.T*W*(g-J_source) in executed reduced coordinates.

    The trace projection and Dirichlet functional use the same common fine
    partition. Local degree must satisfy k>=ell+2. Positive alpha is explicitly
    selected; these finite computations do not establish uniform stability.
    """
    skeleton = definition.skeleton
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if definition.degree < max(max(face.degrees) for face in skeleton.faces) + 2:
        raise ValueError("local degree must satisfy k>=ell+2")
    lower = _ellipticity(definition.permeability, None)
    penalties = tuple(
        _penalty(
            system,
            skeleton,
            face,
            definition.degree,
            definition.quadrature_order,
            alpha,
            lower,
            dirichlet,
        )
        for face in range(len(skeleton.mesh.faces))
    )
    rows, columns, entries = [], [], []
    load = np.zeros_like(system.rhs)
    for data in penalties:
        if len(data.sides) == 1:
            boundary = skeleton.faces[data.face].evaluate(data.parameter).T @ (
                data.weights * data.prescribed
            )
            load[skeleton.dofs(data.face)] -= boundary
        weighted = data.weights * data.coefficient
        block = data.jump.T @ (weighted[:, None] * data.jump)
        forcing = data.jump.T @ (weighted * (data.prescribed - data.source_jump))
        rows.extend(np.repeat(data.indices, len(data.indices)))
        columns.extend(np.tile(data.indices, len(data.indices)))
        entries.extend(block.ravel())
        np.add.at(load, data.indices, forcing)
    matrix = sparse.coo_matrix((entries, (rows, columns)), shape=system.matrix.shape).tocsc()
    return Equation(matrix, load), penalties, lower


def add_jump_form(
    definition: DarcyDefinition,
    system: MultiscaleSystem,
    *,
    dirichlet: Any,
    alpha: float = 0.1,
) -> tuple[MultiscaleSystem, tuple[Any, ...], float]:
    """Add the user-declared residual jump equation to the existing local responses."""
    equation, penalties, lower = jump_equation(definition, system, dirichlet=dirichlet, alpha=alpha)
    return with_global_equation(system, equation), penalties, lower


def recover_penalty(
    definition: DarcyDefinition,
    system: MultiscaleSystem,
    result: HybridSolution,
    penalties: tuple[Any, ...],
    lower: float,
    *,
    alpha: float = 0.1,
) -> PGMHMSolution:
    """Apply the declared residual enrichment through shared constrained local solves.

    The added normal flux is -alpha*Kmin*jump/(2H). Its volume lift has zero
    declared pressure mean. Base/enriched fields keep separate coefficients;
    only the enriched multiplier has the physical macro-conservation property.
    """
    coordinates = np.r_[result.trace, *result.coarse]
    extra = [np.zeros(len(response.problem.load)) for response in system.responses]
    for data in penalties:
        jump = data.jump @ coordinates[data.indices] + data.source_jump - data.prescribed
        normal = -data.coefficient * jump
        for cell, sign, evaluation in data.sides:
            extra[cell] += sign * (evaluation.T @ (data.weights * normal))
    enriched = []
    for response, base, load in zip(system.responses, result.fields, extra, strict=True):
        problem = response.problem.with_load(-load)
        operator, rhs = problem.condensation_system()
        correction = solve_linear(operator, rhs[:, 0])[: len(load)]
        enriched.append(base + correction)
    return PGMHMSolution(
        definition.skeleton,
        tuple(record[0] for record in system.local_metadata),
        result.fields,
        tuple(enriched),
        result,
        system,
        definition.permeability,
        definition.source,
        definition.degree,
        definition.quadrature_order,
        alpha,
        lower,
        penalties,
        tuple(extra),
    )
