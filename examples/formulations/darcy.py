"""User-defined triangular Darcy equations and their physical field interpretation.

These application functions declare local and global forms consumed by the
generic multiscale core. The FEM kernels own integration and basis tabulation;
the existing field record owns norms and conservation diagnostics. No function
selects or invokes a package Darcy solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Protocol, cast

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.loads import point_load_vector, split_point_sources
from pymhm.fem.scalar.operators import boundary_data, face_integration, rt0_operators
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import tensor_values
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.fields import FieldDefinition
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import DarcySolution


@dataclass(frozen=True)
class DarcyDefinition:
    """Declared equations and data needed to interpret their executed coefficients.

    ``problem`` contains A/B/C/D and the additional global equation. A non-None
    ``pressure_integral`` imposes the physical pressure integral for a pure
    Neumann boundary. Flux in a primal record is the raw -K grad(p); only RT0
    carries an H(div) flux with integrated normal coefficients.
    """

    problem: MultiscaleProblem[int]
    skeleton: SkeletonSpace
    permeability: Any
    source: Any
    formulation: Literal["primal", "mixed"]
    degree: int
    quadrature_order: int
    point_parts: tuple[FloatArray, ...]
    pressure_integral: float | None


def local_equations(
    cell: int,
    *,
    mesh: TriangleMesh | PolygonMesh,
    skeleton: SkeletonSpace,
    permeability: Any,
    source: Any,
    point_parts: tuple[FloatArray, ...],
    degree: int,
    formulation: Literal["primal", "mixed"],
    local_refinement: int,
    local_meshes: tuple[TriangleMesh, ...] | None,
    quadrature_order: int,
) -> LocalEquations:
    """Declare Ap+B lambda=f and -B.T p=g in executed scalar/mixed coordinates.

    Primal coordinates are conforming Pk pressure. RT0 coordinates are normal
    flux integrals, P0 pressure, and private boundary-pressure multipliers.
    Its final local equation prescribes outward flux through the signed trace
    map; it is not a pressure-trace hybridization. Moments use the dimensionless
    pressure average, while metadata retain physical pressure integral weights.
    """
    fine = mesh.submesh(cell, local_refinement) if local_meshes is None else local_meshes[cell]
    if formulation == "primal":
        matrix, mass, load = scalar_operators(
            fine, degree, diffusion=permeability, source=source, order=quadrature_order
        )
        coupling = trace_coupling(cast(TriangleMesh, mesh), cell, fine, skeleton, degree)
        if len(point_parts[cell]):
            load += point_load_vector(fine, degree, point_parts[cell])
        kernel = np.ones((matrix.shape[0], 1))
        physical_mean = mass @ kernel
    else:
        for face in mesh.cell_faces[cell]:
            space = skeleton.faces[face]
            scaled_breaks = np.asarray(space.breaks) * local_refinement
            if any(space.degrees) or not np.allclose(
                scaled_breaks, np.round(scaled_breaks), atol=1e-12, rtol=0
            ):
                raise ValueError("RT0 needs degree-zero trace segments aligned with fine edges")
        _, flux_map = face_integration(cast(TriangleMesh, mesh), cell, fine, skeleton)
        mass, divergence, force = rt0_operators(fine, permeability, source, order=quadrature_order)
        if len(point_parts[cell]):
            force += np.array(
                [part[:, 2].sum() for part in split_point_sources(fine, point_parts[cell])]
            )
        nq, npres, nb = len(fine.faces), len(fine.cells), len(fine.boundary_faces)
        normal = sparse.coo_matrix(
            (np.ones(nb), (fine.boundary_faces, np.arange(nb))), shape=(nq, nb)
        ).tocsc()
        zero = sparse.csc_matrix((npres, nb))
        matrix = sparse.bmat(
            [[mass, -divergence.T, normal], [-divergence, None, zero], [normal.T, zero.T, None]],
            format="csc",
        )
        coupling = np.zeros((nq + npres + nb, flux_map.shape[1]))
        coupling[nq + npres :] = -flux_map
        load = np.r_[np.zeros(nq), -force, np.zeros(nb)]
        kernel = np.r_[np.zeros(nq), np.ones(npres + nb)][:, None]
        physical_mean = np.r_[np.zeros(nq), fine.areas, np.zeros(nb)][:, None]
    return LocalEquations(
        matrix,
        load,
        columns(*coupling.T),
        rows(*(-coupling.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=physical_mean / (kernel.T @ physical_mean).item(),
        metadata=(fine, physical_mean[:, 0]),
        field_data=(
            (nodal_field("pressure", fine, degree),)
            if formulation == "primal"
            else (
                nodal_field(
                    "pressure",
                    fine,
                    0,
                    discontinuous=True,
                    reconstruction=sparse.eye(len(load), format="csr")[nq : nq + npres],
                ),
                FieldDefinition(
                    "flux",
                    mesh=fine,
                    reconstruction=sparse.eye(len(load), format="csr")[:nq],
                    basis_id="RT0:integral-normal-moments",
                ),
            )
        ),
    )


def define_darcy(
    mesh: TriangleMesh | PolygonMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    point_sources: Any = (),
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 4,
    formulation: Literal["primal", "mixed"] = "primal",
    degree: int = 1,
    mean_pressure: float = 0.0,
) -> DarcyDefinition:
    """Declare triangular Darcy forms with explicit boundary signs and physical mean.

    Gauss counts retain the degree+2 assembly floor. Point sources use exact
    primal evaluations or angle-shared P0 integrated loads. Supplied local
    meshes are validated conforming partitions of their macrotriangles.
    Execution and linear-solver choices belong to the generic assemble/solve
    calls in the application and are deliberately outside the equation data.
    """
    positive_int(degree, "degree")
    positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    if formulation not in ("primal", "mixed"):
        raise ValueError("formulation must be primal or mixed")
    if formulation == "mixed" and degree != 1:
        raise ValueError("RT0 uses degree=1")
    skeleton = SkeletonSpace(cast(TriangleMesh, mesh)) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("Darcy requires its own scalar skeleton")
    if local_meshes is not None:
        if formulation != "primal" or len(local_meshes) != len(mesh.cells):
            raise ValueError("provide one local mesh per macrocell for primal Darcy only")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(mesh, cell, fine)
    point_parts = tuple(split_point_sources(mesh, point_sources))
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    rhs = -boundary if formulation == "primal" else boundary
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        permeability=permeability,
        source=source,
        point_parts=point_parts,
        degree=degree,
        formulation=formulation,
        local_refinement=local_refinement,
        local_meshes=local_meshes,
        quadrature_order=order,
    )
    problem = MultiscaleProblem(
        Equation(0, np.r_[rhs, np.zeros(len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    return DarcyDefinition(
        problem,
        skeleton,
        permeability,
        source,
        formulation,
        degree,
        order,
        point_parts,
        mean_pressure * float(np.sum(mesh.areas)) if pure_neumann else None,
    )


class PressureDefinition(Protocol):
    """Expose the physical pressure integral required by a declared pure Neumann problem."""

    @property
    def pressure_integral(self) -> float | None:
        """Return the prescribed pressure integral, or None for a fixed pressure boundary."""
        ...


def pressure_constraints(
    definition: PressureDefinition, system: MultiscaleSystem
) -> list[tuple[FloatArray, float]]:
    """Declare a physical pressure integral only when all exterior fluxes are fixed."""
    if definition.pressure_integral is None:
        return []
    weights = [record[1] for record in system.local_metadata]
    return [system.mean_constraint(weights, definition.pressure_integral)]


def recover_darcy(
    definition: DarcyDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> DarcySolution:
    """Interpret the actual local coefficients using the shared field/norm record."""
    meshes, pressures, fluxes = [], [], []
    for record, field in zip(system.local_metadata, solution.fields, strict=True):
        fine = record[0]
        meshes.append(fine)
        if definition.formulation == "primal":
            pressures.append(field)
            dofs, _, _, gradients, _ = tabulate(fine, definition.degree, np.full((1, 3), 1 / 3))
            gradient = np.einsum("ti,tia->ta", field[dofs], gradients[:, 0])
            tensors = tensor_values(definition.permeability, fine.points[fine.cells].mean(axis=1))
            fluxes.append(-np.einsum("tab,tb->ta", tensors, gradient))
        else:
            pressures.append(field[len(fine.faces) : len(fine.faces) + len(fine.cells)])
            fluxes.append(field[: len(fine.faces)])
    return DarcySolution(
        definition.skeleton,
        tuple(meshes),
        tuple(pressures),
        tuple(fluxes),
        solution,
        definition.formulation,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
        definition.degree,
        definition.point_parts if any(len(part) for part in definition.point_parts) else (),
    )
