"""User-declared MH²M equations with independent pressure and conormal spaces.

The generic assembler solves the complete coupled local saddle and global
pressure-trace equation. Shared Neumann-map algebra supplies inspection data in
the established basis; it does not select or execute a physical method driver.
Conormal coefficients are K grad(p).n, minus the physical Darcy normal flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.triangle import scalar_operators, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D, boundary_rules, broken_face_basis
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values, scalar_values_3d
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.three_field import MH2MLocal, MH2MSolution, PressureTraceSpace, _neumann_maps
from pymhm.methods.three_field_3d import MH2M3DSolution


@dataclass(frozen=True)
class ThreeFieldDefinition:
    """Declared equations, physical data and the executed pressure/conormal layout."""

    problem: MultiscaleProblem[int]
    pressure_trace: PressureTraceSpace | PressureTraceSpace3D
    flux_space: SkeletonSpace | TriangularSkeleton
    degree: int
    permeability: Any
    source: Any
    quadrature_order: int
    pressure_integral: float | None


def _equations(data: MH2MLocal) -> LocalEquations:
    """Write Ap-Beta eta=f, -Beta.T p+P rho=0 and P.T eta=g.

    Local u=(p,eta) contains a private outward conormal on every macroface.
    Gamma is a shared continuous pressure trace. The full local saddle is
    invertible when Lambda/Vh satisfy Neumann injectivity; no coarse kernel is
    invented. Physical volume integrals define only the global pressure gauge.
    """
    npres, nlambda = len(data.load), data.boundary_coupling.shape[1]
    matrix = sparse.bmat(
        [[data.stiffness, -data.boundary_coupling], [-data.boundary_coupling.T, None]],
        format="csc",
    )
    coupling = np.vstack((np.zeros((npres, len(data.trace_dofs))), data.trace_pairing))
    return LocalEquations(
        matrix,
        np.r_[data.load, np.zeros(nlambda)],
        coupling,
        coupling.T,
        data.trace_dofs,
        metadata=(data, np.r_[data.volume_moments, np.zeros(nlambda)]),
    )


def triangular_local_equations(
    cell: int,
    *,
    mesh: TriangleMesh | PolygonMesh,
    gamma: PressureTraceSpace,
    flux: SkeletonSpace,
    degree: int,
    refinement: int,
    local_meshes: tuple[TriangleMesh, ...] | None,
    material: Any,
    source: Any,
    order: int,
    inspection_solver: str,
) -> LocalEquations:
    """Integrate independent Gamma/Lambda face forms on triangular fine cells.

    The signed physical-flux tabulator is converted to an outward conormal
    pairing. Continuous pressure endpoint coordinates remain globally shared;
    unioned face partitions receive adequate product quadrature. Inspection
    maps use the existing boundary-mean complement without changing its basis.
    """
    fine = mesh.submesh(cell, refinement) if local_meshes is None else local_meshes[cell]
    stiffness, mass, load = scalar_operators(
        fine, degree, diffusion=material, source=source, order=order
    )
    beta = trace_coupling(cast(TriangleMesh, mesh), cell, fine, flux, degree)
    trace_ids = gamma.cell_dofs(cell)
    pairing = np.zeros((beta.shape[1], len(trace_ids)))
    constant = np.zeros(beta.shape[1])
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        lam, rho = flux.faces[face], gamma.faces[face]
        width = lam.size
        beta[:, offset : offset + width] *= mesh.signs[cell][side]
        constant[offset : offset + width] = lam.constant_coefficients()
        cuts = tuple(sorted(set(lam.breaks) | set(rho.breaks)))
        t, w = FaceSpace(cuts, (0,) * (len(cuts) - 1)).quadrature(
            max(lam.degrees) + max(rho.degrees) + 2
        )
        block = lam.evaluate(t).T @ (w[:, None] * rho.evaluate(t)) * mesh.lengths[face]
        pairing[
            np.ix_(
                np.arange(offset, offset + width), np.searchsorted(trace_ids, gamma.face_dofs[face])
            )
        ] = block
        offset += width
    inspection = _neumann_maps(
        fine, trace_ids, stiffness, mass, load, beta, pairing, constant, inspection_solver
    )
    return _equations(inspection)


def tetrahedral_local_equations(
    cell: int,
    *,
    mesh: TetraMesh,
    gamma: PressureTraceSpace3D,
    flux: TriangularSkeleton,
    degree: int,
    refinement: int,
    material: Any,
    source: Any,
    order: int,
    inspection_solver: str,
) -> LocalEquations:
    """Integrate Gamma/Lambda pairings in their independent triangular partitions."""
    fine = mesh.submesh(cell, refinement)
    stiffness, mass, load = tetra_operators(
        fine, degree, diffusion=material, source=source, order=order
    )
    beta = tetra_trace_coupling(mesh, cell, fine, flux, degree)
    trace_ids = gamma.cell_dofs(cell)
    pairing = np.zeros((beta.shape[1], len(trace_ids)))
    face_rows = {}
    offset = 0
    for side, face in enumerate(mesh.cell_faces[cell]):
        width = len(flux.dofs(int(face)))
        beta[:, offset : offset + width] *= mesh.signs[cell, side]
        face_rows[int(face)] = np.arange(offset, offset + width)
        offset += width
    for face, _, _, _, weights, bary in boundary_rules(
        mesh, cell, fine, degree, max(order, gamma.degree + int(flux.degrees.max()) + 2)
    ):
        lam, rho = broken_face_basis(flux, face, bary), gamma.evaluate(face, bary)
        pairing[np.ix_(face_rows[face], np.searchsorted(trace_ids, gamma.face_dofs[face]))] += (
            lam.T @ (weights[:, None] * rho)
        )
    inspection = _neumann_maps(
        fine,
        trace_ids,
        stiffness,
        mass,
        load,
        beta,
        pairing,
        np.ones(beta.shape[1]),
        inspection_solver,
    )
    return _equations(inspection)


def define_three_field(
    mesh: TriangleMesh | PolygonMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    pressure_trace: PressureTraceSpace | None = None,
    flux_space: SkeletonSpace | None = None,
    degree: int = 1,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 6,
    mean_pressure: float = 0.0,
    inspection_solver: str = "scipy",
) -> ThreeFieldDefinition:
    """Declare 2D MH²M forms, pressure interpolation and outward Neumann load.

    Local injectivity is checked by the shared Neumann-map kernel; global rank
    is checked by the generic solve. Neither check establishes mesh-uniform
    inf-sup stability. Gamma/Lambda spaces, fitted meshes and approximation
    degrees retain their independent choices and the applicable paper hypotheses.
    """
    if not isinstance(mesh, (TriangleMesh, PolygonMesh)):
        raise TypeError("three-field equations require a triangle or polygon macro mesh")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order")
    gamma = PressureTraceSpace(mesh) if pressure_trace is None else pressure_trace
    flux = SkeletonSpace(cast(TriangleMesh, mesh)) if flux_space is None else flux_space
    if gamma.mesh is not mesh or flux.mesh is not mesh or flux.components != 1:
        raise ValueError("Gamma and scalar Lambda must use the supplied mesh")
    if any(face.continuous for face in flux.faces):
        raise ValueError("Lambda requires discontinuous face polynomials")
    natural = {} if neumann is None else dict(neumann)
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("provide one local partition per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(mesh, cell, fine)
    rhs = np.zeros(gamma.size)
    fixed: dict[int, float] = {}
    for face in mesh.boundary_faces:
        ids, space = gamma.face_dofs[face], gamma.faces[face]
        if face in natural:
            t, w = space.quadrature(max(order, max(space.degrees) + 2))
            a, b = mesh.points[mesh.faces[face]]
            data = mesh.lengths[face] * w * scalar_values(natural[face], a + t[:, None] * (b - a))
            np.add.at(rhs, ids, -(space.evaluate(t).T @ data))
        else:
            fixed.update(zip(ids, scalar_values(dirichlet, gamma.nodes[ids]), strict=True))
    if fixed and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann data")
    provider = partial(
        triangular_local_equations,
        mesh=mesh,
        gamma=gamma,
        flux=flux,
        degree=degree,
        refinement=refinement,
        local_meshes=local_meshes,
        material=permeability,
        source=source,
        order=order,
        inspection_solver=inspection_solver,
    )
    problem = MultiscaleProblem(
        Equation(0, rhs),
        provider,
        range(len(mesh.cells)),
        gamma.size,
        (0,) * len(mesh.cells),
        fixed=fixed,
    )
    return ThreeFieldDefinition(
        problem,
        gamma,
        flux,
        degree,
        permeability,
        source,
        order,
        mean_pressure * float(mesh.areas.sum()) if not fixed else None,
    )


def define_three_field_3d(
    mesh: TetraMesh,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    pressure_trace: PressureTraceSpace3D | None = None,
    flux_space: TriangularSkeleton | None = None,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    inspection_solver: str = "scipy",
) -> ThreeFieldDefinition:
    """Declare tetrahedral three-field forms with independent resolved face spaces."""
    if not isinstance(mesh, TetraMesh):
        raise TypeError("three-field 3D equations require a tetrahedral macro mesh")
    degree = positive_int(degree, "degree")
    refinement = _dyadic(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    gamma = PressureTraceSpace3D(mesh) if pressure_trace is None else pressure_trace
    flux = TriangularSkeleton(mesh) if flux_space is None else flux_space
    if gamma.mesh is not mesh or flux.mesh is not mesh:
        raise ValueError("Gamma and Lambda must use the supplied mesh")
    if gamma.subdivisions > refinement or np.any(flux.subdivisions > refinement):
        raise ValueError("local refinement must resolve both interface partitions")
    natural = {} if neumann is None else dict(neumann)
    if any(face not in mesh.boundary_faces for face in natural):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be finite")
    if len(natural) != len(mesh.boundary_faces) and mean_pressure != 0:
        raise ValueError("mean_pressure applies only to pure Neumann data")
    rhs = np.zeros(gamma.size)
    fixed: dict[int, float] = {}
    for cell in range(len(mesh.cells)):
        fine = mesh.submesh(cell, refinement)
        for face, _, _, points, weights, bary in boundary_rules(
            mesh, cell, fine, degree, max(order, gamma.degree + 2)
        ):
            if face in natural:
                data = weights * scalar_values_3d(natural[face], points)
                np.add.at(rhs, gamma.face_dofs[face], -(gamma.evaluate(face, bary).T @ data))
    for boundary_face in mesh.boundary_faces:
        if boundary_face not in natural:
            ids = gamma.face_dofs[boundary_face]
            fixed.update(zip(ids, scalar_values_3d(dirichlet, gamma.nodes[ids]), strict=True))
    provider = partial(
        tetrahedral_local_equations,
        mesh=mesh,
        gamma=gamma,
        flux=flux,
        degree=degree,
        refinement=refinement,
        material=permeability,
        source=source,
        order=order,
        inspection_solver=inspection_solver,
    )
    problem = MultiscaleProblem(
        Equation(0, rhs),
        provider,
        range(len(mesh.cells)),
        gamma.size,
        (0,) * len(mesh.cells),
        fixed=fixed,
    )
    return ThreeFieldDefinition(
        problem,
        gamma,
        flux,
        degree,
        permeability,
        source,
        order,
        mean_pressure * float(mesh.volumes.sum()) if not fixed else None,
    )


def recover_three_field(
    definition: ThreeFieldDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> MH2MSolution:
    """Interpret the actual generic saddle solution as pressure and outward conormal."""
    local = tuple(item[0] for item in system.local_metadata)
    return MH2MSolution(
        cast(PressureTraceSpace, definition.pressure_trace),
        cast(SkeletonSpace, definition.flux_space),
        local,
        solution.trace,
        tuple(field[: len(data.load)] for data, field in zip(local, solution.fields, strict=True)),
        tuple(field[len(data.load) :] for data, field in zip(local, solution.fields, strict=True)),
        system.matrix,
        system.rhs,
        np.setdiff1d(
            np.arange(definition.pressure_trace.size), list(definition.problem.fixed or {})
        ),
        definition.degree,
        definition.permeability,
        solution.residual,
    )


def recover_three_field_3d(
    definition: ThreeFieldDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> MH2M3DSolution:
    """Interpret the executed tetrahedral pressure and conormal in the same declared basis."""
    local = tuple(item[0] for item in system.local_metadata)
    return MH2M3DSolution(
        cast(PressureTraceSpace3D, definition.pressure_trace),
        cast(TriangularSkeleton, definition.flux_space),
        local,
        solution.trace,
        tuple(field[: len(data.load)] for data, field in zip(local, solution.fields, strict=True)),
        tuple(field[len(data.load) :] for data, field in zip(local, solution.fields, strict=True)),
        system.matrix,
        system.rhs,
        np.setdiff1d(
            np.arange(definition.pressure_trace.size), list(definition.problem.fixed or {})
        ),
        definition.degree,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
        solution.residual,
    )
