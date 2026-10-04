"""Declare scalar Robin hybrid equations and physical Neumann pressure moments.

The local Robin multiplier is (q-p*sigma).n, rather than physical Darcy outflow.
The application defines every extra boundary-pressure block through Equation;
local/global elimination uses only the generic multiscale core.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.triangle import scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.integration import integrate_dirichlet_trace
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.pressure_3d import boundary_rules, broken_face_basis
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.materials.evaluation import scalar_values_3d, tensor_values, tensor_values_3d
from pymhm.methods.robin import MHSolution, _ellipticity, _robin_operator, _volume_weights
from pymhm.methods.robin_3d import MH3DSolution, _robin_matrix


@dataclass(frozen=True)
class RobinDefinition:
    """Declared blocks and physical data for interpreting the Robin coordinates."""

    problem: MultiscaleProblem[int]
    skeleton: Any
    permeability: Any
    source: Any
    degree: int
    order: int
    parameter: float
    origin: Any
    lower: float
    boundary_pressure: dict[int, Any]
    pressure_integral: float | None


def robin_equations(
    cell: int,
    *,
    mesh: Any,
    skeleton: SkeletonSpace,
    permeability: Any,
    source: Any,
    degree: int,
    refinement: int,
    order: int,
    parameter: float,
    origin: Any,
) -> LocalEquations:
    """Write the coercive Robin operator and the negative pressure-trace balance."""
    fine = mesh.submesh(cell, refinement)
    stiffness, _, load = scalar_operators(
        fine, degree, diffusion=permeability, source=source, order=order
    )
    robin = _robin_operator(fine, degree, parameter, origin, order)
    b = trace_coupling(mesh, cell, fine, skeleton, degree)
    return LocalEquations(
        stiffness + robin, load, b, -b.T, skeleton.cell_dofs(cell), metadata=(fine, robin)
    )


def define_robin(
    mesh: Any,
    *,
    skeleton: SkeletonSpace,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    degree: int = 2,
    local_refinement: int = 4,
    quadrature_order: int = 6,
    robin_parameter: float | None = None,
    ellipticity_lower_bound: float | None = None,
    origin: Any = None,
) -> RobinDefinition:
    """Declare Robin locals and S*lambda+J*rho=t-g, J.T*lambda+R*rho=h.

    Physical Neumann data uses the additional rho coordinates; it does not fix
    Robin lambda. The two-dimensional certified parameter bound is retained.
    The physical pressure mean is requested only for a fully Neumann boundary.
    """
    order = max(quadrature_order, degree + 2)
    lower = _ellipticity(permeability, ellipticity_lower_bound)
    point = np.min(mesh.points, axis=0) if origin is None else np.asarray(origin)
    radius = float(np.max(np.linalg.norm(mesh.points - point, axis=1))) / 2
    upper = lower / (4 * radius**2)
    parameter = upper / 2 if robin_parameter is None else robin_parameter
    if not np.isfinite(parameter) or parameter <= 0 or parameter > upper:
        raise ValueError("Robin parameter must satisfy the declared coercivity bound")
    natural = {} if neumann is None else dict(neumann)
    if not set(natural) <= set(mesh.boundary_faces):
        raise ValueError("Neumann faces must be exterior")
    fine_meshes = tuple(mesh.submesh(cell, local_refinement) for cell in range(len(mesh.cells)))
    size = skeleton.size + sum(len(skeleton.dofs(face)) for face in natural)
    global_a = sparse.lil_matrix((size, size))
    global_f = np.zeros(size)
    global_f[: skeleton.size] = -integrate_dirichlet_trace(
        skeleton,
        fine_meshes,
        dirichlet,
        degree,
        order,
        faces=tuple(int(face) for face in mesh.boundary_faces if face not in natural),
    )
    pressure_maps: dict[int, Any] = {}
    offset = skeleton.size
    for face in sorted(natural):
        space = skeleton.faces[face]
        points, weights = space.quadrature(max(order, max(space.degrees) + 2))
        basis = space.evaluate(points)
        mass = basis.T @ ((weights * mesh.lengths[face])[:, None] * basis)
        midpoint = mesh.points[mesh.faces[face]].mean(axis=0)
        coefficient = parameter * ((midpoint - point) @ mesh.normals[face]) / 2
        ids = skeleton.dofs(face)
        rho = np.arange(offset, offset + len(ids))
        global_a[np.ix_(ids, rho)] = mass
        global_a[np.ix_(rho, ids)] = mass.T
        global_a[np.ix_(rho, rho)] = coefficient * mass
        global_f[rho] = integrate_dirichlet_trace(
            skeleton, fine_meshes, natural[face], degree, order, faces=(face,)
        )[ids]
        pressure_maps[face] = rho
        offset += len(ids)
    problem = MultiscaleProblem(
        Equation(global_a.tocsc(), global_f),
        partial(
            robin_equations,
            mesh=mesh,
            skeleton=skeleton,
            permeability=permeability,
            source=source,
            degree=degree,
            refinement=local_refinement,
            order=order,
            parameter=float(parameter),
            origin=point,
        ),
        range(len(mesh.cells)),
        size,
        (0,) * len(mesh.cells),
    )
    integral = (
        mean_pressure * float(np.sum(mesh.areas))
        if set(natural) == set(mesh.boundary_faces)
        else None
    )
    return RobinDefinition(
        problem,
        skeleton,
        permeability,
        source,
        degree,
        order,
        float(parameter),
        point,
        lower,
        pressure_maps,
        integral,
    )


def robin_gauge(definition: RobinDefinition, system: MultiscaleSystem) -> list[Any]:
    """Fix the physical reconstructed pressure mean for a fully Neumann problem."""
    if definition.pressure_integral is None:
        return []
    return [
        system.mean_constraint(
            [_volume_weights(record[0], definition.degree) for record in system.local_metadata],
            definition.pressure_integral,
        )
    ]


def recover_robin(
    definition: RobinDefinition, system: MultiscaleSystem, result: HybridSolution
) -> MHSolution:
    """Interpret physical fields while preserving Robin and pressure trace distinctions."""
    if result.raw_residual is not None and result.raw_residual > 1e-10:
        raise ValueError("MH original boundary equations fail after the physical pressure gauge")
    fluxes = []
    for metadata, field in zip(system.local_metadata, result.fields, strict=True):
        fine = metadata[0]
        dofs, _, _, gradients, _ = tabulate(fine, definition.degree, np.array([[1 / 3] * 3]))
        gradient = np.einsum("tia,ti->ta", gradients[:, 0], field[dofs])
        coefficient = tensor_values(definition.permeability, fine.points[fine.cells].mean(axis=1))
        fluxes.append(-np.einsum("tab,tb->ta", coefficient, gradient))
    hybrid = HybridSolution(
        result.trace[: definition.skeleton.size],
        result.coarse,
        result.fields,
        result.residual,
        result.gauge_multipliers,
        result.raw_residual,
        result.raw_residual_norm,
    )
    return MHSolution(
        definition.skeleton,
        tuple(record[0] for record in system.local_metadata),
        result.fields,
        tuple(fluxes),
        hybrid,
        system,
        definition.permeability,
        definition.source,
        definition.degree,
        definition.order,
        definition.parameter,
        definition.origin,
        definition.lower,
        {face: result.trace[dofs] for face, dofs in definition.boundary_pressure.items()},
        system.matrix,
        system.rhs,
    )


def robin_equations_3d(
    cell: int,
    *,
    mesh: Any,
    skeleton: TriangularSkeleton,
    permeability: Any,
    source: Any,
    degree: int,
    refinement: int,
    order: int,
    parameter: float,
    origin: Any,
    lower: float,
) -> LocalEquations:
    """Write tetrahedral Robin forms with sigma=parameter*(x-origin)/3.

    The three-dimensional volume and oriented triangular-face integration
    remain owned by the shared FEM kernels. Moments integrate physical pressure.
    """
    fine = mesh.submesh(cell, refinement)
    sampled = np.linalg.eigvalsh(tensor_values_3d(permeability, fine.points)).min()
    if sampled < lower * (1 - 64 * np.finfo(float).eps):
        raise ValueError("ellipticity_lower_bound exceeds a sampled material eigenvalue")
    stiffness, mass, load = tetra_operators(
        fine, degree, diffusion=permeability, source=source, order=order
    )
    robin = _robin_matrix(mesh, cell, fine, degree, order, parameter, origin)
    b = tetra_trace_coupling(mesh, cell, fine, skeleton, degree)
    return LocalEquations(
        stiffness + robin,
        load,
        b,
        -b.T,
        skeleton.cell_dofs(cell),
        metadata=(fine, robin, np.asarray(mass.sum(axis=1)).ravel()),
    )


def define_robin_3d(
    mesh: Any,
    *,
    skeleton: TriangularSkeleton,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    degree: int = 2,
    local_refinement: int = 2,
    quadrature_order: int = 5,
    robin_parameter: float | None = None,
    ellipticity_lower_bound: float | None = None,
    origin: Any = None,
) -> RobinDefinition:
    """Declare all tetrahedral boundary blocks using the dimensional coercivity bound.

    With radius=max_vertex|x-origin|/3, parameter <= K_min/(4*radius**2).
    Each natural face adds pressure coefficients rho and the explicit equations
    J.T*lambda+sigma.n*M*rho=h. No 2D parameter or face convention is reused.
    """
    order = max(quadrature_order, degree + 2)
    if ellipticity_lower_bound is None:
        if callable(permeability):
            raise ValueError("material callbacks require a certified ellipticity_lower_bound")
        lower = float(np.linalg.eigvalsh(tensor_values_3d(permeability, np.zeros((1, 3)))).min())
    else:
        lower = float(ellipticity_lower_bound)
        if not np.isfinite(lower) or lower <= 0:
            raise ValueError("ellipticity_lower_bound must be finite and positive")
    point = np.min(mesh.points, axis=0) if origin is None else np.asarray(origin)
    if point.shape != (3,) or np.iscomplexobj(point) or not np.isfinite(point).all():
        raise ValueError("origin must be a finite real three-dimensional point")
    radius = float(np.max(np.linalg.norm(mesh.points - point, axis=1))) / 3
    upper = lower / (4 * radius**2)
    parameter = upper / 2 if robin_parameter is None else robin_parameter
    if not np.isfinite(parameter) or parameter <= 0 or parameter > upper:
        raise ValueError("Robin parameter must satisfy the declared coercivity bound")
    natural = {} if neumann is None else dict(neumann)
    if not set(natural) <= set(mesh.boundary_faces):
        raise ValueError("Neumann faces must be exterior")
    fine_meshes = tuple(mesh.submesh(cell, local_refinement) for cell in range(len(mesh.cells)))
    boundary = np.zeros(skeleton.size)
    masses = {face: np.zeros((len(skeleton.dofs(face)),) * 2) for face in natural}
    loads = {face: np.zeros(len(skeleton.dofs(face))) for face in natural}
    for cell, fine in enumerate(fine_meshes):
        for face, _, _, points, weights, bary in boundary_rules(mesh, cell, fine, degree, order):
            if face not in mesh.boundary_faces:
                continue
            basis = broken_face_basis(skeleton, face, bary)
            data = scalar_values_3d(natural.get(face, dirichlet), points)
            moments = basis.T @ (weights * data)
            if face in natural:
                masses[face] += basis.T @ (weights[:, None] * basis)
                loads[face] += moments
            else:
                boundary[skeleton.dofs(face)] += moments
    size = skeleton.size + sum(len(skeleton.dofs(face)) for face in natural)
    global_a = sparse.lil_matrix((size, size))
    global_f = np.r_[-boundary, np.zeros(size - skeleton.size)]
    pressure_maps: dict[int, Any] = {}
    offset = skeleton.size
    for face in sorted(natural):
        ids = skeleton.dofs(face)
        rho = np.arange(offset, offset + len(ids))
        midpoint = mesh.points[mesh.faces[face]].mean(axis=0)
        coefficient = parameter * ((midpoint - point) @ mesh.normals[face]) / 3
        global_a[np.ix_(ids, rho)] = masses[face]
        global_a[np.ix_(rho, ids)] = masses[face].T
        global_a[np.ix_(rho, rho)] = coefficient * masses[face]
        global_f[rho] = loads[face]
        pressure_maps[face] = rho
        offset += len(ids)
    problem = MultiscaleProblem(
        Equation(global_a.tocsc(), global_f),
        partial(
            robin_equations_3d,
            mesh=mesh,
            skeleton=skeleton,
            permeability=permeability,
            source=source,
            degree=degree,
            refinement=local_refinement,
            order=order,
            parameter=float(parameter),
            origin=point,
            lower=lower,
        ),
        range(len(mesh.cells)),
        size,
        (0,) * len(mesh.cells),
    )
    integral = (
        mean_pressure * float(mesh.volumes.sum())
        if set(natural) == set(mesh.boundary_faces)
        else None
    )
    return RobinDefinition(
        problem,
        skeleton,
        permeability,
        source,
        degree,
        order,
        float(parameter),
        point,
        lower,
        pressure_maps,
        integral,
    )


def robin_gauge_3d(definition: RobinDefinition, system: MultiscaleSystem) -> list[Any]:
    """Fix the exact executed tetrahedral pressure integral for a pure Neumann boundary."""
    if definition.pressure_integral is None:
        return []
    return [
        system.mean_constraint(
            [record[2] for record in system.local_metadata], definition.pressure_integral
        )
    ]


def recover_robin_3d(
    definition: RobinDefinition, system: MultiscaleSystem, result: HybridSolution
) -> MH3DSolution:
    """Interpret the physical fields with the executed Robin normal-flux correction."""
    if result.raw_residual is not None and result.raw_residual > 1e-10:
        raise ValueError("MH original boundary equations fail after the physical pressure gauge")
    hybrid = HybridSolution(
        result.trace[: definition.skeleton.size],
        result.coarse,
        result.fields,
        result.residual,
        result.gauge_multipliers,
        result.raw_residual,
        result.raw_residual_norm,
    )
    return MH3DSolution(
        definition.skeleton,
        tuple(record[0] for record in system.local_metadata),
        result.fields,
        hybrid,
        system,
        definition.degree,
        definition.permeability,
        definition.source,
        definition.order,
        definition.parameter,
        definition.origin,
        {face: result.trace[dofs] for face, dofs in definition.boundary_pressure.items()},
        system.matrix,
        system.rhs,
    )
