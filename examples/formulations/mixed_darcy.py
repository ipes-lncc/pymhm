"""Application declarations for RT and BDM normal-flux Darcy formulations.

The shared element kernels supply basis coordinates, quadrature and orientation.
The application supplies the saddle, physical kernel, pressure moments and
global boundary equation. Executed field records reuse the package's physical
evaluation routines; no physical solver or local factory is invoked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.core.contracts import HybridSolution
from pymhm.core.equations import Equation, LocalEquations, columns, rows
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.hdiv.bdm import bdm2_basis, bdm2_trace_map
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.mixed import (
    normal_flux_blocks,
    triangle_mixed_operators,
    triangle_pressure_basis,
    triangle_pressure_integrals,
)
from pymhm.fem.hdiv.rt import rt_basis, rt_degree, rt_dofs
from pymhm.fem.hdiv.rt_forms import rt_operators
from pymhm.fem.hdiv.rt_trace import rt_trace_map
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.reference import legendre_values
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.linalg.linear import solve_linear
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.piola import hdiv_field
from pymhm.postprocessing.solutions import BDMDarcySolution, RTDarcySolution


@dataclass(frozen=True)
class MixedDarcyDefinition:
    """Equations and the executed RT/BDM coordinate contract.

    ``degree`` is the mathematical RT degree or the BDM normal degree.
    ``family`` distinguishes the BDM enrichment and is None for RT. Pressure
    moments are physical integrals without normalization. The skeletal unknown
    is physical normal flux in the fixed global macroface orientation.
    """

    problem: MultiscaleProblem[int]
    skeleton: SkeletonSpace
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    pressure_integral: float | None
    family: BDMFamily | None = None


def rt_local_equations(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    degree: int,
    refinement: int,
    permeability: Any,
    source: Any,
    order: int,
) -> LocalEquations:
    """Declare RTm/Pm local equations in oriented normal/interior moments.

    For u=(q,p,r), A has blocks [M,-D.T,N; -D,0,0; N.T,0,0].
    B's last block prescribes outward normal moments with minus the oriented
    trace map. The joint constant pressure/boundary-pressure kernel is explicit;
    C=-B.T declares global weak pressure continuity in the same coordinates.
    """
    fine = mesh.submesh(cell, refinement)
    bary, weights, material = material_triangle_quadrature(fine, permeability, order)
    basis, divergence = rt_basis(fine, degree, bary)
    pressure = triangle_pressure_basis(degree, bary)
    nq = (degree + 1) * len(fine.faces) + degree * (degree + 1) * len(fine.cells)
    mass, div, force = triangle_mixed_operators(
        fine,
        bary,
        weights,
        material,
        source,
        basis,
        divergence,
        pressure,
        rt_dofs(fine, degree),
        nq,
    )
    count = degree + 1
    boundary_dofs = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
    mapping = rt_trace_map(mesh, cell, fine, skeleton, degree)
    matrix, coupling, load = normal_flux_blocks(mass, div, force, boundary_dofs, mapping)
    nb, npres = len(boundary_dofs), div.shape[0]
    boundary_constant = np.zeros((len(fine.boundary_faces), count))
    boundary_constant[:, 0] = 1
    kernel = np.r_[np.zeros(nq), np.ones(npres), boundary_constant.ravel()][:, None]
    moment = triangle_pressure_integrals(fine, weights, pressure)
    physical_mean = np.r_[np.zeros(nq), moment, np.zeros(nb)]
    return LocalEquations(
        matrix,
        load,
        columns(*coupling.T),
        rows(*(-coupling.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=physical_mean[:, None],
        metadata=(fine, physical_mean, nq, npres),
        field_data=(
            nodal_field(
                "pressure",
                fine,
                degree,
                discontinuous=True,
                reconstruction=sparse.eye(len(load), format="csr")[nq : nq + npres],
            ),
            hdiv_field(
                "flux",
                fine,
                "RT",
                degree=degree,
                reconstruction=sparse.eye(len(load), format="csr")[:nq],
            ),
            hdiv_field(
                "flux_divergence",
                fine,
                "RT",
                degree=degree,
                reconstruction=sparse.eye(len(load), format="csr")[:nq],
                divergence=True,
            ),
        ),
    )


def bdm_local_equations(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    family: BDMFamily,
    refinement: int,
    permeability: Any,
    source: Any,
    order: int,
) -> LocalEquations:
    """Declare BDM(k,n)/P(k+n-1) equations in the archived family basis.

    Default BDM2 uses its established normal/interior coordinate convention.
    Enrichment preserves every zero-normal interior mode of BDM(k+n). The
    pressure and first boundary-pressure modes form the constant local kernel.
    """
    fine = mesh.submesh(cell, refinement)
    bary, weights, material = material_triangle_quadrature(fine, permeability, order)
    basis, divergence = (
        bdm2_basis(fine, bary) if family == BDMFamily() else family.basis(fine, bary)
    )
    pressure = triangle_pressure_basis(family.polynomial_degree - 1, bary)
    mass, div, force = triangle_mixed_operators(
        fine,
        bary,
        weights,
        material,
        source,
        basis,
        divergence,
        pressure,
        family.dofs(fine),
        family.size(fine),
    )
    count = family.degree + 1
    boundary_dofs = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
    mapping = (
        bdm2_trace_map(mesh, cell, fine, skeleton)
        if family == BDMFamily()
        else family.trace_map(mesh, cell, fine, skeleton)
    )
    matrix, coupling, load = normal_flux_blocks(mass, div, force, boundary_dofs, mapping)
    nq, npres, nb = mass.shape[0], div.shape[0], len(boundary_dofs)
    kernel = np.r_[
        np.zeros(nq), np.ones(npres), np.tile(np.r_[1.0, np.zeros(count - 1)], nb // count)
    ][:, None]
    moment = triangle_pressure_integrals(fine, weights, pressure)
    physical_mean = np.r_[np.zeros(nq), moment, np.zeros(nb)]
    return LocalEquations(
        matrix,
        load,
        columns(*coupling.T),
        rows(*(-coupling.T)),
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=physical_mean[:, None],
        metadata=(fine, physical_mean, nq, npres),
        field_data=(
            nodal_field(
                "pressure",
                fine,
                family.polynomial_degree - 1,
                discontinuous=True,
                reconstruction=sparse.eye(len(load), format="csr")[nq : nq + npres],
            ),
            hdiv_field(
                "flux", fine, family, reconstruction=sparse.eye(len(load), format="csr")[:nq]
            ),
            hdiv_field(
                "flux_divergence",
                fine,
                family,
                reconstruction=sparse.eye(len(load), format="csr")[:nq],
                divergence=True,
            ),
        ),
    )


def define_rt_darcy(
    mesh: TriangleMesh,
    *,
    degree: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
) -> MixedDarcyDefinition:
    """Declare RTm/Pm local forms and the additional global pressure-data load.

    RT mathematical m corresponds to Basix RT degree m+1. Trace degrees must
    not exceed m, and segment breaks must align with the fine boundary edges.
    Neumann data prescribe outward physical flux; the other exterior faces
    impose pressure weakly. Execution choices remain outside the declaration.
    """
    m = rt_degree(degree)
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), m + 3)
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(m) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("RT Darcy requires its own scalar skeleton")
    for face in skeleton.faces:
        breaks = np.asarray(face.breaks) * refinement
        if max(face.degrees) > m or not np.allclose(breaks, np.round(breaks), rtol=0, atol=1e-12):
            raise ValueError("RT trace degrees exceed m or breaks do not align with fine edges")
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann, order=order)
    provider = partial(
        rt_local_equations,
        mesh=mesh,
        skeleton=skeleton,
        degree=m,
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
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    return MixedDarcyDefinition(
        problem,
        skeleton,
        permeability,
        source,
        m,
        order,
        mean_pressure * float(mesh.areas.sum()) if pure_neumann else None,
    )


def define_bdm_darcy(
    mesh: TriangleMesh,
    *,
    degree: int = 2,
    enrichment: int = 0,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    quadrature_order: int = 4,
    mean_pressure: float = 0.0,
) -> MixedDarcyDefinition:
    """Declare BDM(k,n) forms, aligned Pk flux traces and the physical mean gauge.

    The family controls all enrichment and trace compatibility. Assembly uses
    at least k+n+2 Gauss points; boundary integration retains its established
    independent order. The global load has the mixed pressure-data sign.
    """
    family = BDMFamily(degree, enrichment)
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order", 3), family.polynomial_degree + 2)
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("BDM Darcy requires its own scalar skeleton")
    family.validate_trace(skeleton, refinement)
    boundary, fixed = boundary_data(skeleton, dirichlet, neumann)
    provider = partial(
        bdm_local_equations,
        mesh=mesh,
        skeleton=skeleton,
        family=family,
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
    pure_neumann = neumann is not None and set(neumann) == set(mesh.boundary_faces)
    return MixedDarcyDefinition(
        problem,
        skeleton,
        permeability,
        source,
        degree,
        order,
        mean_pressure * float(mesh.areas.sum()) if pure_neumann else None,
        family,
    )


def recover_rt_darcy(
    definition: MixedDarcyDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> RTDarcySolution:
    """Interpret executed RT coefficients without changing their basis or field norms."""
    if definition.family is not None:
        raise ValueError("RT recovery requires an RT declaration")
    pressure, flux = [], []
    for metadata, field in zip(system.local_metadata, solution.fields, strict=True):
        fine, _, nq, npres = metadata
        pressure.append(field[nq : nq + npres].reshape(len(fine.cells), -1))
        flux.append(field[:nq])
    return RTDarcySolution(
        tuple(item[0] for item in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        definition.degree,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
        definition.skeleton,
        solution,
        solution.residual,
    )


def recover_bdm_darcy(
    definition: MixedDarcyDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> BDMDarcySolution:
    """Interpret the executed BDM normal/interior moments and DG pressure coefficients."""
    if definition.family is None:
        raise ValueError("BDM recovery requires its declared family")
    count = definition.family.polynomial_degree * (definition.family.polynomial_degree + 1) // 2
    pressure, flux = [], []
    for metadata, field in zip(system.local_metadata, solution.fields, strict=True):
        fine, _, nq, npres = metadata
        pressure.append(field[nq : nq + npres].reshape(len(fine.cells), count))
        flux.append(field[:nq])
    return BDMDarcySolution(
        definition.skeleton,
        tuple(item[0] for item in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        solution,
        definition.permeability,
        definition.source,
        definition.quadrature_order,
        definition.family,
    )


def conforming_rt_reference(
    mesh: TriangleMesh,
    *,
    degree: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    quadrature_order: int = 5,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
) -> RTDarcySolution:
    """Solve the classical globally conforming RTm/Pm mixed problem without macro restrictions.

    Pressure is natural weak data; outward normal flux moments are essential.
    The unknowns are fine-mesh flux/pressure coefficients. A pure Neumann solve
    adds only the physical mean gauge, then checks the original free equations.
    This is a classical numerical reference, not an exact analytical solution.
    """
    # This independent reference declares its full global Equation, rather than local condensation.
    m = rt_degree(degree)
    order = max(positive_int(quadrature_order, "quadrature_order"), m + 3)
    neumann = {} if neumann is None else neumann
    if any(face not in mesh.boundary_faces for face in neumann):
        raise ValueError("Neumann data require external faces")
    M, D, f, moment = rt_operators(mesh, m, permeability, source, order)
    nq = M.shape[0]
    A = sparse.bmat([[M, -D.T], [-D, None]], format="csc")
    b = np.r_[np.zeros(nq), -f]
    x, w = leggauss(order)
    basis = legendre_values(x, m)
    fixed: dict[int, float] = {}
    lengths = mesh.lengths
    for face in mesh.boundary_faces:
        a, z = mesh.points[mesh.faces[face]]
        points = a + (x[:, None] + 1) / 2 * (z - a)
        data = scalar_values(neumann.get(int(face), dirichlet), points)
        moments = (w * data / 2) @ basis
        indices = (m + 1) * face + np.arange(m + 1)
        if face in neumann:
            fixed.update(
                (int(i), float(v)) for i, v in zip(indices, lengths[face] * moments, strict=True)
            )
        else:
            b[indices] = -(2 * np.arange(m + 1) + 1) * moments
    values = np.zeros(len(b))
    known = np.array(sorted(fixed), dtype=int)
    values[known] = [fixed[i] for i in known]
    free = np.setdiff1d(np.arange(len(b)), known)
    physical = A[free][:, free]
    rhs = b[free] - A[free][:, known] @ values[known]
    matrix = physical
    if len(neumann) == len(mesh.boundary_faces):
        weights = np.r_[np.zeros(nq), moment][free]
        matrix = sparse.bmat(
            [
                [physical, sparse.csc_matrix(weights[:, None])],
                [sparse.csc_matrix(weights[None]), None],
            ],
            format="csc",
        )
        solve_rhs = np.r_[rhs, mean_pressure * mesh.areas.sum()]
    else:
        solve_rhs = rhs
    values[free] = solve_linear(matrix, solve_rhs, solver=solver)[: len(free)]
    residual = float(
        np.linalg.norm(physical @ values[free] - rhs)
        / max(
            np.linalg.norm(rhs),
            np.linalg.norm(abs(physical) @ np.abs(values[free])),
            np.finfo(float).tiny,
        )
    )
    if residual > 1e-8:
        raise ValueError("incompatible data: pressure gauge changed the physical equations")
    return RTDarcySolution(
        (mesh,),
        (values[nq:].reshape(len(mesh.cells), -1),),
        (values[:nq],),
        m,
        permeability,
        source,
        order,
        residual=residual,
    )
