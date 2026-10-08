"""Enriched rectangular RT Darcy equations in the shared Basix moment coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.mixed import normal_flux_blocks
from pymhm.fem.hdiv.tensor_rt import tensor_rt_operators, tensor_rt_trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.modal import modal_field
from pymhm.postprocessing.piola import hdiv_field
from pymhm.postprocessing.solutions import TensorRTDarcySolution


@dataclass(frozen=True)
class TensorDarcyDefinition:
    """Declared enriched tensor-RT forms and the executed pressure/flux coordinate contract."""

    problem: MultiscaleProblem[int]
    skeleton: SkeletonSpace
    degree: int
    enrichment: int
    permeability: Any
    source: Any
    quadrature_order: int
    pressure_integral: float | None


def local_equations(
    cell: int,
    *,
    mesh: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
    enrichment: int,
    refinement: int,
    permeability: Any,
    source: Any,
    order: int,
) -> LocalEquations:
    """Declare the rectangular RT saddle and physical outward normal-moment constraint.

    The saddle acts on flux, modal Q_s pressure and private boundary-pressure
    coefficients, with s=k+n. Only the first modal pressure coefficient represents
    a constant. Its joint pressure/boundary-pressure kernel and dimensional
    integral moments preserve the executed Basix-based moment coordinate order.
    """
    fine = mesh.submesh(cell, refinement)
    mass, divergence, force = tensor_rt_operators(
        fine, degree, enrichment, permeability, source, order
    )
    nq, npres = mass.shape[0], divergence.shape[0]
    ids = ((degree + 1) * fine.boundary_faces[:, None] + np.arange(degree + 1)).ravel()
    mapping = tensor_rt_trace_map(mesh, cell, fine, skeleton, degree)
    matrix, coupling, load = normal_flux_blocks(mass, divergence, force, ids, mapping)
    width = (degree + enrichment + 1) ** 2
    constant = np.zeros(npres)
    constant[::width] = 1
    boundary_constant = np.tile(np.r_[1.0, np.zeros(degree)], len(fine.boundary_faces))
    kernel = np.r_[np.zeros(nq), constant, boundary_constant][:, None]
    moments = np.r_[np.zeros(nq), constant * np.repeat(fine.areas, width), np.zeros(len(ids))]
    return LocalEquations(
        matrix,
        load,
        columns(*coupling.T),
        rows(*(-coupling.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=moments[:, None],
        metadata=(fine, moments, nq, npres),
        field_data=(
            modal_field(
                "pressure",
                fine,
                tuple(
                    (a, b)
                    for b in range(degree + enrichment + 1)
                    for a in range(degree + enrichment + 1)
                ),
                convention="legendre",
                reconstruction=sparse.eye(len(load), format="csr")[nq : nq + npres],
            ),
            hdiv_field(
                "flux",
                fine,
                "tensor-RT",
                degree=degree,
                enrichment=enrichment,
                reconstruction=sparse.eye(len(load), format="csr")[:nq],
            ),
            hdiv_field(
                "flux_divergence",
                fine,
                "tensor-RT",
                degree=degree,
                enrichment=enrichment,
                reconstruction=sparse.eye(len(load), format="csr")[:nq],
                divergence=True,
            ),
        ),
    )


def define_tensor_darcy(
    mesh: CartesianMacroMesh,
    *,
    degree: int = 1,
    enrichment: int = 0,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 6,
    mean_pressure: float = 0.0,
) -> TensorDarcyDefinition:
    """Declare independent normal degree k/interior RT order k+n and global pressure data.

    Trace degrees cannot exceed k, and their breaks align with fine edges.
    The pressure uses modal Q_(k+n), without projected permeability. Boundary
    pressure contributes the mixed global sign; Neumann values are physical
    outward flux. A pure Neumann boundary imposes the physical pressure mean.
    """
    k = positive_int(degree, "RT degree", 0)
    enrichment = positive_int(enrichment, "enrichment", 0)
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order", k + enrichment + 2)
    skeleton = (
        SkeletonSpace(cast(TriangleMesh, mesh), tuple(FaceSpace.uniform(k) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("tensor RT requires its own scalar skeleton")
    for face in skeleton.faces:
        breaks = np.asarray(face.breaks) * refinement
        if max(face.degrees) > k or not np.allclose(breaks, np.rint(breaks), rtol=0, atol=1e-12):
            raise ValueError("tensor RT trace degrees exceed k or breaks do not align")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann)
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        degree=k,
        enrichment=enrichment,
        refinement=refinement,
        permeability=permeability,
        source=source,
        order=order,
    )
    problem = MultiscaleProblem(
        Equation(0, np.r_[boundary, np.zeros(len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    pure = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    return TensorDarcyDefinition(
        problem,
        skeleton,
        k,
        enrichment,
        permeability,
        source,
        order,
        mean_pressure * float(mesh.areas.sum()) if pure else None,
    )


def recover_tensor_darcy(
    definition: TensorDarcyDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> TensorRTDarcySolution:
    """Interpret executed normal/bubble moments and modal Q_s pressure without rebasing."""
    meshes, pressure, flux = [], [], []
    for metadata, field in zip(system.local_metadata, solution.fields, strict=True):
        fine, _, nq, npres = metadata
        meshes.append(fine)
        pressure.append(field[nq : nq + npres].reshape(len(fine.cells), -1))
        flux.append(field[:nq])
    return TensorRTDarcySolution(
        definition.skeleton,
        tuple(meshes),
        tuple(pressure),
        tuple(flux),
        solution,
        definition.degree,
        definition.enrichment,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
    )
