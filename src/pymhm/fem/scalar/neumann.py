"""Diffusion Neumann maps in an explicitly declared boundary-mean complement.

The supplied local stiffness has a constant kernel. Boundary coupling represents
outward conormal moments, while volume moments are kept for physical gauges.
The returned data records the executed zero-mean basis instead of recomputing it.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm.core.subspaces import moment_complement
from pymhm.core.validation import FloatArray, IntArray
from pymhm.linalg.linear import LinearSolveError, factorize
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class NeumannMaps:
    """Local operators and Eq. (29) lifts in declared nodal/Legendre coordinates.

    ``conormal_lift`` and ``conormal_source`` use the paper's lambda convention.
    ``neumann_energy`` acts on zero-boundary-average Lambda coordinates.
    Boundary averages, rather than volume averages, define the local complement.
    """

    mesh: TriangleMesh | TetraMesh
    trace_dofs: IntArray
    stiffness: Any
    load: FloatArray
    boundary_coupling: FloatArray
    trace_pairing: FloatArray
    flux_integrals: FloatArray
    zero_mean_basis: FloatArray
    neumann_energy: FloatArray
    source_lift: FloatArray
    pressure_lift: FloatArray
    pressure_source: FloatArray
    conormal_lift: FloatArray
    conormal_source: FloatArray
    volume_moments: FloatArray
    trace_matrix: FloatArray
    trace_rhs: FloatArray


def neumann_maps(
    fine: TriangleMesh | TetraMesh,
    trace_ids: IntArray,
    stiffness: Any,
    mass: Any,
    load: FloatArray,
    coupling: FloatArray,
    pairing: FloatArray,
    constant: FloatArray,
    solver: str,
) -> NeumannMaps:
    """Construct the dimension-independent Neumann inverses and Eq. (29) lifts.

    Coupling has the unsigned outward-conormal convention. Boundary integrals
    supply its physical mean; mass supplies the volume mean used only by gauges.
    """
    integrals = coupling.sum(axis=0)
    boundary = coupling @ constant
    perimeter = float(integrals @ constant)
    zero_mean = moment_complement(integrals[:, None])
    rhs = np.column_stack((coupling @ zero_mean, load))
    rhs -= boundary[:, None] * (rhs.sum(axis=0) / perimeter)
    lifts = np.zeros_like(rhs)
    with factorize(stiffness[1:, 1:], solver=solver) as factor:
        lifts[1:] = factor.solve(rhs[1:])
    lifts -= (boundary @ lifts / perimeter)[None, :]
    neumann, eta = lifts[:, :-1], lifts[:, -1]
    energy = (coupling @ zero_mean).T @ neumann
    energy = (energy + energy.T) / 2
    moments = zero_mean.T @ pairing
    source_moments = (coupling @ zero_mean).T @ eta
    try:
        with factorize(energy, solver=solver) as factor:
            inverse = factor.solve(np.column_stack((moments, source_moments)))
    except LinearSolveError as error:
        raise ValueError(
            "Lambda and Vh violate local Neumann injectivity (Assumption A)"
        ) from error
    response, source_response = inverse[:, :-1], inverse[:, -1]
    mean_row = constant @ pairing / perimeter
    pressure_lift = neumann @ response + mean_row[None, :]
    pressure_source = eta - neumann @ source_response
    conormal_lift = zero_mean @ response
    conormal_source = -constant * load.sum() / perimeter - zero_mean @ source_response
    trace_matrix = moments.T @ response
    trace_matrix = (trace_matrix + trace_matrix.T) / 2
    trace_rhs = -pairing.T @ conormal_source
    return NeumannMaps(
        fine,
        trace_ids,
        stiffness,
        load,
        coupling,
        pairing,
        integrals,
        zero_mean,
        energy,
        eta,
        pressure_lift,
        pressure_source,
        conormal_lift,
        conormal_source,
        np.asarray(mass.sum(axis=1)).ravel(),
        trace_matrix,
        trace_rhs,
    )
