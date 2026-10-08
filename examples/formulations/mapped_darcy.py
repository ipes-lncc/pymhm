"""Mapped RT/Qk equations with explicit Piola moments and physical coordinate scaling.

On a trilinear hexahedron div(RT_k) is Qk/det(J); pressure uses the scalar Qk
pullback. The local saddle enforces every pressure-tested equilibrium moment.
Reference-face multipliers represent flux density, so boundary pressure pairs
with reference measure and physical Neumann data include the surface Jacobian.
"""

from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm.core.validation import positive_int, real_array
from pymhm.fem.hdiv.mapped_forms import (
    HexSkeleton,
    mapped_boundary_data,
    mapped_rt_operators,
    mapped_rt_trace_mapping,
)
from pymhm.meshes.hexahedron import HexMesh, QuadratureOrder, cube_quadrature
from pymhm.postprocessing.mapped import MappedRTDarcySolution
from pymhm.postprocessing.modal import modal_field
from pymhm.postprocessing.piola import hdiv_field


@dataclass(frozen=True)
class MappedDarcyDefinition:
    """Retain the declared moment spaces, quadrature and physical pressure integral."""

    problem: MultiscaleProblem[int]
    skeleton: HexSkeleton
    degree: int
    permeability: Any
    source: Any
    quadrature_order: QuadratureOrder
    pressure_integral: float | None


def local_equations(
    cell: int,
    *,
    skeleton: HexSkeleton,
    degree: int,
    refinement: int,
    permeability: Any,
    source: Any,
    order: QuadratureOrder,
) -> LocalEquations:
    """Declare mixed RT mass/divergence, normal-moment constraints and joint constant mode.

    The positive diagonal S balances physical flux and pressure units. The exact
    congruence S A S, load S f, coupling S B, kernel S^-1 z and moments S m
    preserve the original physical equations. The stored scaling reconstructs
    physical coefficients directly, including the private boundary-pressure modes.
    """
    fine, reference = skeleton.mesh.submesh(cell, refinement)
    mass, divergence, force, moment = mapped_rt_operators(fine, degree, permeability, source, order)
    nq, npres = mass.shape[0], len(force)
    count = (degree + 1) ** 2
    nb = count * len(fine.boundary_faces)
    indices = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
    normal = sparse.coo_matrix((np.ones(nb), (indices, np.arange(nb))), shape=(nq, nb)).tocsc()
    zero = sparse.csc_matrix((npres, nb))
    a = sparse.bmat(
        [[mass, -divergence.T, normal], [-divergence, None, zero], [normal.T, zero.T, None]],
        format="csc",
    )
    mapping = mapped_rt_trace_mapping(skeleton.mesh, cell, fine, reference, skeleton, degree)
    b = np.zeros((nq + npres + nb, mapping.shape[1]))
    b[-nb:] = -mapping
    pressure_constant = np.zeros((len(fine.cells), (degree + 1) ** 3))
    pressure_constant[:, 0] = 1
    boundary_constant = np.zeros((len(fine.boundary_faces), count))
    boundary_constant[:, 0] = 1
    kernel = np.r_[np.zeros(nq), pressure_constant.ravel(), boundary_constant.ravel()][:, None]
    moment = np.r_[np.zeros(nq), moment, np.zeros(nb)]
    scale = float(np.sqrt(np.max(mass.diagonal())))
    scaling = np.r_[np.full(nq, 1 / scale), np.full(npres + nb, scale)]
    transform = sparse.diags(scaling)
    b = scaling[:, None] * b
    return LocalEquations(
        transform @ a @ transform,
        scaling * np.r_[np.zeros(nq), -force, np.zeros(nb)],
        b,
        -b.T,
        skeleton.cell_dofs(cell),
        kernel=kernel / scaling[:, None],
        moments=(moment * scaling)[:, None],
        metadata=((fine, reference, nq, npres, scaling), moment * scaling),
        field_data=(
            modal_field(
                "pressure",
                fine,
                tuple(product(range(degree + 1), repeat=3)),
                convention="legendre",
                reconstruction=transform.tocsr()[nq : nq + npres],
            ),
            hdiv_field(
                "flux", fine, "mapped-RT", degree=degree, reconstruction=transform.tocsr()[:nq]
            ),
            hdiv_field(
                "flux_divergence",
                fine,
                "mapped-RT",
                degree=degree,
                reconstruction=transform.tocsr()[:nq],
                divergence=True,
            ),
        ),
    )


def define_mapped_darcy(
    mesh: HexMesh,
    *,
    degree: int = 1,
    trace_degree: int | None = None,
    subdivisions: int = 1,
    local_refinement: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: QuadratureOrder = 5,
) -> MappedDarcyDefinition:
    """Declare mapped RT_k/Qk forms, aligned normal moments and physical boundary data.

    Normal trace degree cannot exceed k and subdivisions divide refinement.
    Three independently specified volume Gauss orders are preserved; boundary
    integration uses their maximum. Material discontinuities must be resolved
    by this quadrature or the actual geometry. A pure Neumann pressure gauge
    uses integrated physical volume, not the sum of all nonconstant modal moments.
    """
    k, refinement = (
        positive_int(degree, "RT degree", 0),
        positive_int(local_refinement, "refinement"),
    )
    skeleton = HexSkeleton(mesh, k if trace_degree is None else trace_degree, subdivisions)
    if skeleton.degree > k or refinement % skeleton.subdivisions:
        raise ValueError(
            "trace degree must not exceed RT degree and subdivisions must divide refinement"
        )
    orders = quadrature_order if isinstance(quadrature_order, tuple) else (quadrature_order,) * 3
    if len(orders) != 3:
        raise ValueError("volume quadrature needs three axis orders")
    orders = tuple(max(positive_int(v, "quadrature_order"), k + 2) for v in orders)
    order = orders if isinstance(quadrature_order, tuple) else orders[0]
    natural = dict(neumann or {})
    boundary, fixed = mapped_boundary_data(skeleton, dirichlet, natural, max(orders))
    provider = partial(
        local_equations,
        skeleton=skeleton,
        degree=k,
        refinement=refinement,
        permeability=permeability,
        source=source,
        order=order,
    )
    problem = MultiscaleProblem(
        Equation(0, np.r_[-boundary, np.zeros(len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    integral = None
    if set(natural) == set(mesh.boundary_faces):
        mean = real_array(mean_pressure, "mean_pressure")
        if mean.ndim:
            raise ValueError("mean_pressure must be scalar")
        points, weights = cube_quadrature(order)
        volume = float(np.sum(mesh.geometry(points)[2] * weights))
        integral = float(mean) * volume
    return MappedDarcyDefinition(problem, skeleton, k, permeability, source, order, integral)


def recover_mapped_darcy(
    definition: MappedDarcyDefinition, system: Any, solution: Any
) -> MappedRTDarcySolution:
    """Recover original physical units and check every mixed block's backward error."""
    pressure, flux, residuals, meshes = [], [], [], []
    for response, record, field in zip(
        system.responses, system.local_metadata, solution.fields, strict=True
    ):
        fine, _, nq, npres, scaling = record[0]
        problem = response.problem
        trace = solution.trace[problem.trace_dofs]
        defect = (problem.matrix @ field + problem.coupling @ trace - problem.load) / scaling
        action = (
            abs(problem.matrix) @ abs(field)
            + abs(problem.coupling) @ abs(trace)
            + abs(problem.load)
        ) / scaling
        blocks = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        if max(blocks) > 1e-10:
            raise ValueError(
                "physical mixed block residual exceeds the declared backward-error criterion"
            )
        physical = scaling * field
        meshes.append(fine)
        flux.append(physical[:nq])
        pressure.append(physical[nq : nq + npres].reshape(len(fine.cells), -1))
        residuals.append(blocks)
    return MappedRTDarcySolution(
        definition.skeleton,
        tuple(meshes),
        tuple(pressure),
        tuple(flux),
        solution,
        definition.degree,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
        np.asarray(residuals),
    )
