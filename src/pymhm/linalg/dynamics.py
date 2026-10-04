"""Time updates of second-order coefficient equations without physical dispatch."""

from __future__ import annotations

from typing import Any

from pymhm.linalg.linear import Array, LinearFactorization


def newmark_step(
    mass: Any,
    stiffness: Any,
    step_factor: LinearFactorization,
    mass_factor: LinearFactorization,
    duration: float,
    u: Array,
    v: Array,
    old: Array,
    new: Array,
) -> tuple[Array, Array]:
    """Advance M u''+K u=f with beta=1/4 and gamma=1/2.

    ``duration`` is a positive local step. ``u``, ``v`` and the endpoint
    forces ``old``/``new`` are matching vectors or matrices of independent
    response columns, in one declared coefficient basis. The caller provides
    factors of ``mass+duration**2/4*stiffness`` and ``mass`` and retains their
    lifetime; this operation creates or closes no native resource. Real and
    complex operators follow the same algebra. No material, trace or boundary
    convention is inferred from these arrays.

    The two checked factor solves preserve the coefficient operation order
    used for both local histories and slabwise constant trace response columns.
    """
    result = step_factor.solve(
        mass @ (u + duration * v)
        - duration**2 / 4 * (stiffness @ u)
        + duration**2 / 4 * (old + new)
    )
    velocity = v + duration / 2 * mass_factor.solve(old + new - stiffness @ (u + result))
    return result, velocity
