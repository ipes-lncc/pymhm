"""Affine H(div) Darcy declarations for tetrahedral and prismatic local cells."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import sparse

from pymhm._legacy.models.darcy._mixed import normal_flux_blocks
from pymhm._legacy.models.darcy.hdiv_3d import (
    Mixed3DDarcySolution,
    Mixed3DSkeleton,
    _boundary,
    _trace_mapping,
    hdiv3d_operators,
)
from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.family_3d import HDiv3DFamily, face_size
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_face_offsets


@dataclass(frozen=True)
class MixedDarcy3DDefinition:
    """Declared mixed forms, diagonal physical-coordinate scaling and pressure gauge."""

    problem: MultiscaleProblem[int]
    skeleton: Mixed3DSkeleton
    family: HDiv3DFamily
    source: Any
    pressure_integral: float | None


def local_equations(
    cell: int,
    *,
    skeleton: Mixed3DSkeleton,
    family: HDiv3DFamily,
    refinement: int,
    permeability: Any,
    source: Any,
    order: int,
) -> LocalEquations:
    """Declare the scaled mixed saddle with the original Piola/moment coordinates.

    The matrix congruence S A S scales flux and pressure/private multipliers.
    Kernels transform by S inverse, and physical integral moments by S. B and
    C=-B.T use the same scaling. Normal moments are independently controlled by
    the family and trace partition; no material projection or interface fitting
    is introduced by this declaration.
    """
    fine = skeleton.mesh.submesh(cell, refinement)
    mass, divergence, force, moment = hdiv3d_operators(
        fine, family, permeability=permeability, source=source, quadrature_order=order
    )
    nq, npres = mass.shape[0], len(force)
    offsets = hdiv3d_face_offsets(fine, family)
    ids = np.concatenate([np.arange(offsets[f], offsets[f + 1]) for f in fine.boundary_faces])
    mapping = _trace_mapping(skeleton, cell, fine, family.normal_degree)
    matrix, coupling, load = normal_flux_blocks(mass, divergence, force, ids, mapping)
    constant = np.zeros((len(fine.cells), family.pressure_size))
    constant[:, 0] = 1
    boundary = np.zeros(len(ids))
    cursor = 0
    for face in fine.boundary_faces:
        boundary[cursor] = 1
        cursor += face_size(len(fine.faces[face]), family.normal_degree)
    kernel = np.r_[np.zeros(nq), constant.ravel(), boundary][:, None]
    moment = np.r_[np.zeros(nq), moment, np.zeros(len(ids))]
    scale = np.sqrt(float(np.max(mass.diagonal())))
    scaling = np.r_[np.full(nq, 1 / scale), np.full(npres + len(ids), scale)]
    transform = sparse.diags(scaling)
    b = scaling[:, None] * coupling
    weights = moment * scaling
    return LocalEquations(
        transform @ matrix @ transform,
        scaling * load,
        columns(*b.T),
        rows(*(-b.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel / scaling[:, None],
        moments=weights[:, None],
        metadata=((fine, nq, npres, scaling, weights), weights),
    )


def define_hdiv_darcy(
    mesh: AffineMixedMesh,
    *,
    pressure_degree: int = 1,
    normal_degree: int = 1,
    trace_degree: int = 1,
    subdivisions: int = 1,
    local_refinement: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: int = 5,
    boundary_quadrature_order: int | None = None,
) -> MixedDarcy3DDefinition:
    """Declare independent normal/interior H(div) spaces and weak pressure boundaries.

    Trace degree cannot exceed the family's normal degree; its subdivisions
    divide local refinement. HDiv3DFamily owns dimension-dependent admissibility
    and Piola transformations. Every pure Neumann problem fixes the physical
    pressure integral, with source/flux compatibility checked by the global solve.
    """
    family = HDiv3DFamily(mesh.kind, pressure_degree, normal_degree)
    refinement = positive_int(local_refinement, "local_refinement")
    skeleton = Mixed3DSkeleton(mesh, trace_degree, subdivisions)
    if trace_degree > normal_degree:
        raise ValueError("trace degree must not exceed local normal degree")
    if refinement % subdivisions:
        raise ValueError("trace subdivisions must divide local refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), pressure_degree + 2)
    natural = {} if neumann is None else dict(neumann)
    boundary_order = (
        order
        if boundary_quadrature_order is None
        else positive_int(boundary_quadrature_order, "boundary_quadrature_order")
    )
    boundary, fixed = _boundary(skeleton, dirichlet, natural, boundary_order)
    provider = partial(
        local_equations,
        skeleton=skeleton,
        family=family,
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
    pure = set(natural) == set(mesh.boundary_faces)
    if pure and not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    return MixedDarcy3DDefinition(
        problem,
        skeleton,
        family,
        source,
        float(mean_pressure) * float(mesh.volumes.sum()) if pure else None,
    )


def recover_hdiv_darcy(
    definition: MixedDarcy3DDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> Mixed3DDarcySolution:
    """Recover physical flux/pressure and separately check all three original block residuals."""
    meshes, pressure, flux, residuals = [], [], [], []
    for response, metadata, field in zip(
        system.responses, system.local_metadata, solution.fields, strict=True
    ):
        fine, nq, npres, scaling, _ = metadata[0]
        problem = response.problem
        trace = solution.trace[problem.trace_dofs]
        defect = (problem.matrix @ field + problem.coupling @ trace - problem.load) / scaling
        action = (
            abs(problem.matrix) @ abs(field)
            + abs(problem.coupling) @ abs(trace)
            + abs(problem.load)
        ) / scaling
        residual = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        if max(residual) > 1e-10:
            raise ValueError(
                "physical mixed block residual exceeds the original backward-error limit"
            )
        physical = scaling * field
        meshes.append(fine)
        flux.append(physical[:nq])
        pressure.append(physical[nq : nq + npres].reshape(len(fine.cells), -1))
        residuals.append(residual)
    return Mixed3DDarcySolution(
        definition.skeleton,
        definition.family,
        tuple(meshes),
        tuple(pressure),
        tuple(flux),
        solution,
        definition.source,
        np.asarray(residuals),
    )
