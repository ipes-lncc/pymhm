"""Explicit Darcy coupling using public fields/materials and transport driver."""

from typing import Any, Literal

import numpy as np

from examples.formulations.transient import (
    solve_transport_trajectory as solve_transient_transport,
)
from pymhm.materials.dispersion import (
    HydrodynamicDispersion,
    RT0DarcyVelocity,
)
from pymhm.materials.dispersion import (
    PrimalDispersion as _PrimalDispersion,
)
from pymhm.materials.macro import MacroCoefficient
from pymhm.postprocessing.solutions import DarcySolution
from pymhm.postprocessing.transport import TransientTransportResult
from pymhm.postprocessing.velocity import (
    PolynomialDarcyVelocity,
    PrimalDarcyVelocity,
    polynomial_darcy_velocity,
)


def solve_darcy_trajectory(
    darcy: Any,
    times: Any,
    *,
    molecular: float = 1e-6,
    longitudinal: float = 1e-2,
    transverse: float = 1e-3,
    velocity_representation: Literal["hdiv", "primal"] = "hdiv",
    permeability_gradient: Any = None,
    **options: Any,
) -> TransientTransportResult:
    """Couple an executed Darcy field to implicit passive transport.

    Darcy is stationary and its computed flux drives both advection and the dispersion tensor, as in
    Section 5.4 of [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499). The
    transport local mesh must match Darcy's fine mesh so that coefficient jumps are integrated
    one-sided. These actual meshes are passed to transport, including nonuniform partitions. Local
    scalar degree and skeletal resolution may differ. ``hdiv`` accepts mixed RT0, RT/BDM families
    and moment reconstructions, preserving their physical vector polynomials and conservation
    constraints. ``primal`` explicitly uses the raw -K grad(p_h) in volume integrals and the Darcy
    skeletal multiplier as numerical normal velocity in the conservative Galerkin form. This pair is
    not asserted to be H(div). Its broken material derivative must be supplied as
    ``permeability_gradient``; zero denotes piecewise constant permeability. Strong-residual
    stabilization is rejected for this discrete flux pair. Other arguments, including
    capacity/porosity, are solve_transient_transport options. The transport exterior conditions must
    be supplied explicitly. An explicit ``local_meshes`` override must have identical points and
    cell connectivity to the Darcy meshes; a different partition is rejected even when the cell
    counts agree. ``local_refinement`` is only a consistency check on the cell count when explicitly
    provided, not a request to rebuild meshes.
    """
    if velocity_representation not in ("hdiv", "primal"):
        raise ValueError("velocity_representation must be hdiv or primal")
    if getattr(darcy, "skeleton", None) is None:
        raise ValueError("Darcy-coupled transport requires an executed MHM skeleton")
    if (
        isinstance(darcy, DarcySolution)
        and velocity_representation == "hdiv"
        and darcy.formulation != "mixed"
    ):
        raise ValueError("Darcy-coupled transport requires the locally conforming RT0 solution")
    if velocity_representation == "primal" and (
        not isinstance(darcy, DarcySolution) or darcy.formulation != "primal"
    ):
        raise ValueError("primal velocity representation requires a primal Darcy solution")
    if (
        velocity_representation == "primal"
        and options.get("stabilization", "galerkin") != "galerkin"
    ):
        raise ValueError("raw primal/numerical-normal coupling requires Galerkin stabilization")
    forbidden = {
        "diffusion",
        "velocity",
        "diffusion_divergence",
        "velocity_divergence",
    } & options.keys()
    if forbidden:
        raise ValueError("coupled Darcy velocity and dispersion cannot be overridden")
    counts = np.array([len(mesh.cells) for mesh in darcy.local_meshes])
    refinement = int(round(np.sqrt(counts[0])))
    if "local_refinement" in options and (
        np.any(counts != refinement**2) or options["local_refinement"] != refinement
    ):
        raise ValueError("transport local_refinement must match all Darcy local meshes")
    supplied = options.get("local_meshes")
    if supplied is not None and (
        len(supplied) != len(darcy.local_meshes)
        or any(
            not np.array_equal(first.points, second.points)
            or not np.array_equal(first.cells, second.cells)
            for first, second in zip(supplied, darcy.local_meshes, strict=True)
        )
    ):
        raise ValueError("transport local_meshes must preserve the Darcy points and connectivity")
    options["local_meshes"] = darcy.local_meshes
    velocities: tuple[RT0DarcyVelocity | PolynomialDarcyVelocity | PrimalDarcyVelocity, ...]
    if velocity_representation == "primal":
        velocities = tuple(
            PrimalDarcyVelocity(
                darcy,
                cell,
                permeability_gradient.for_cell(cell, len(darcy.local_meshes))
                if isinstance(permeability_gradient, MacroCoefficient)
                else permeability_gradient,
            )
            for cell in range(len(darcy.local_meshes))
        )
    elif isinstance(darcy, DarcySolution):
        velocities = tuple(
            RT0DarcyVelocity(mesh, flux)
            for mesh, flux in zip(darcy.local_meshes, darcy.flux, strict=True)
        )
    else:
        velocities = tuple(
            polynomial_darcy_velocity(darcy, cell) for cell in range(len(darcy.local_meshes))
        )
    dispersion_type = (
        _PrimalDispersion if velocity_representation == "primal" else HydrodynamicDispersion
    )
    dispersion = tuple(dispersion_type(v, molecular, longitudinal, transverse) for v in velocities)
    return solve_transient_transport(
        darcy.skeleton.mesh,
        times,
        velocity=MacroCoefficient(velocities),
        velocity_divergence=MacroCoefficient(tuple(v.divergence for v in velocities)),
        diffusion=MacroCoefficient(dispersion),
        diffusion_divergence=MacroCoefficient(tuple(d.divergence for d in dispersion)),
        **options,
    )
