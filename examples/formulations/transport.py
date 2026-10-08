"""Conservative scalar equations, explicit boundary rows and constant-mode moments."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import Equation, MultiscaleProblem
from pymhm.core.validation import positive_int
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.normal import boundary_tangent_2d
from pymhm.fem.traces.scalar import diffusive_boundary_matrix, strong_boundary_dofs
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.refinement import validate_submesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import ScalarSolution

from .residual_transport import streamline_equations
from .scalar import ScalarDiscretization


@dataclass(frozen=True)
class LocalPartition(ScalarDiscretization):
    """Declare the actual local approximation meshes independently of nominal refinement."""

    local_meshes: Any = None

    def local_mesh(self, cell: int) -> Any:
        """Return this macrocell's literal supplied or uniformly refined partition."""
        return (
            self.mesh.submesh(cell, self.refinement)
            if self.local_meshes is None
            else self.local_meshes[cell]
        )


@dataclass(frozen=True)
class TransportDefinition:
    """Declared volume/trace/boundary equations and physical constant-gauge data."""

    problem: MultiscaleProblem[int]
    data: LocalPartition
    velocity: Any
    natural: tuple[int, ...]
    strong: bool
    mean_value: float


def local_equations(
    cell: int,
    *,
    data: LocalPartition,
    velocity: Any,
    velocity_divergence: Any,
    diffusion_divergence: Any,
    reaction: Any,
    source: Any,
    stabilization: str,
    unusual_parameters: Any,
    coarse_space: str,
    strong_faces: tuple[int, ...],
    diffusive_faces: tuple[int, ...],
    dirichlet: Any,
) -> Any:
    """Declare residual A/B/C, optional physical diffusive boundary mass and essential rows.

    Strong nodal data add E.T*u=g and local reaction coordinates. Corresponding
    external multiplier columns vanish. Weak data remain the global functional.
    The selective coarse space retains only exactly eligible constant kernels;
    the default retains constants also in invertible or near-singular locals.
    """
    equations = streamline_equations(
        cell,
        data=data,
        velocity=velocity,
        velocity_divergence=velocity_divergence,
        diffusion_divergence=diffusion_divergence,
        reaction=reaction,
        source=source,
        stabilization=stabilization,
        unusual_parameters=unusual_parameters,
    )
    fine, moments, pure, zero, count, _ = equations.metadata
    a, b, load = equations.a, np.array(equations.b, copy=True), equations.L
    kernel, coarse, weights = equations.kernel, equations.coarse_basis, equations.moments
    if coarse_space == "kernel" and not pure:
        eligible = zero and boundary_tangent_2d(SkeletonSpace(fine), velocity, data.order)
        kernel = np.ones((count, 1)) if eligible else None
        coarse = None if eligible else np.empty((count, 0))
        weights = moments[:, None] if eligible else np.empty((count, 0))
    own = set(data.mesh.cell_faces[cell])
    natural = tuple(f for f in diffusive_faces if f in own)
    if natural:
        a = a + diffusive_boundary_matrix(
            data.mesh, fine, natural, data.degree, velocity, data.order
        )
    essential = tuple(f for f in strong_faces if f in own)
    ids, values = strong_boundary_dofs(data.mesh, fine, essential, data.degree, dirichlet)
    if len(ids):
        e = sparse.csc_matrix(
            (np.ones(len(ids)), (ids, np.arange(len(ids)))), shape=(count, len(ids))
        )
        a = sparse.bmat([[a, e], [e.T, None]], format="csc")
        load = np.r_[load, values]
        b = np.vstack((b, np.zeros((len(ids), b.shape[1]))))
        for face in essential:
            b[:, np.isin(equations.dofs, data.skeleton.dofs(face))] = 0
        kernel, coarse, weights = None, np.empty((len(load), 0)), np.empty((len(load), 0))
    return replace(
        equations,
        a=a,
        L=load,
        b=b,
        c=-b.T,
        kernel=kernel,
        coarse_basis=coarse,
        moments=weights,
        metadata=(fine, moments, pure, zero, count, ids),
        field_data=(
            nodal_field(
                "scalar",
                fine,
                data.degree,
                reconstruction=sparse.eye(len(load), format="csr")[:count],
            ),
        ),
    )


def define_transport(
    mesh: Any,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: Any = None,
    diffusive_flux: Any = None,
    dirichlet_enforcement: str = "weak",
    skeleton: Any = None,
    local_refinement: int = 4,
    local_meshes: Any = None,
    degree: int = 1,
    quadrature_order: int = 6,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    unusual_parameters: Any = None,
    mean_value: float = 0.0,
    coarse_space: str = "constants",
) -> TransportDefinition:
    """Declare conservative RAD with complete Galerkin/SUPG/UNUSUAL and boundary inputs.

    Natural Robin data are (-K grad(u)+beta*u/2).n; diffusive_flux is -K grad(u).n
    and includes its half-advection boundary mass explicitly. Strong nodal data
    and weak global data are distinct mathematical declarations. Variable fields
    supply their analytical divergence for conservative and residual forms.
    Selective constant kernels use sampled zero reaction/divergence and exact
    cancellation-scaled tangency checks; no sampling certifies continuum bounds.
    """
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = positive_int(quadrature_order, "quadrature_order")
    if coarse_space not in ("constants", "kernel") or dirichlet_enforcement not in (
        "weak",
        "strong",
    ):
        raise ValueError("unknown coarse space or Dirichlet enforcement")
    if (
        callable(velocity)
        and velocity_divergence is None
        and not hasattr(velocity, "advection_boundary_matrix")
    ):
        raise ValueError("variable velocity requires velocity_divergence")
    if (
        callable(diffusion)
        and not isinstance(diffusion, CartesianCellField)
        and stabilization in ("supg", "unusual")
        and diffusion_divergence is None
    ):
        raise ValueError("variable residual diffusion requires diffusion_divergence")
    db = 0.0 if velocity_divergence is None else velocity_divergence
    dk = (0.0, 0.0) if diffusion_divergence is None else diffusion_divergence
    if not callable(velocity) and np.any(scalar_values(db, mesh.points)):
        raise ValueError("constant velocity must have zero divergence")
    if not callable(diffusion) and np.any(vector_values(dk, mesh.points)):
        raise ValueError("constant diffusion must have zero divergence")
    if not np.isfinite(mean_value):
        raise ValueError("mean_value must be finite")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("transport requires a scalar trace on this mesh")
    natural, physical = dict(neumann or {}), dict(diffusive_flux or {})
    if set(natural) & set(physical):
        raise ValueError("Robin and diffusive faces must be disjoint")
    natural.update(physical)
    if local_meshes is not None:
        if len(local_meshes) != len(mesh.cells):
            raise ValueError("one local mesh is needed per macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(mesh, cell, fine)
    data = LocalPartition(mesh, skeleton, diffusion, degree, refinement, order, local_meshes)
    boundary, fixed = boundary_data(skeleton, dirichlet, natural, order=max(order, degree + 2))
    strong = (
        tuple(int(f) for f in mesh.boundary_faces if f not in natural)
        if dirichlet_enforcement == "strong"
        else ()
    )
    for face in strong:
        boundary[skeleton.dofs(face)] = 0
        fixed.update(dict.fromkeys(skeleton.dofs(face), 0.0))
    if coarse_space == "kernel" and not boundary_tangent_2d(
        skeleton, velocity, order, tuple(physical)
    ):
        raise ValueError("selective coarse space requires tangent advection on diffusive faces")
    sizes = []
    for cell in range(len(mesh.cells)):
        if any(f in strong for f in mesh.cell_faces[cell]):
            sizes.append(0)
            continue
        if coarse_space == "constants":
            sizes.append(1)
            continue
        fine = data.local_mesh(cell)
        bary, _, _ = material_triangle_quadrature(fine, diffusion, max(order, degree + 2))
        points = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells]).reshape(-1, 2)
        eligible = (
            not np.any(scalar_values(reaction, points))
            and not np.any(scalar_values(db, points))
            and boundary_tangent_2d(SkeletonSpace(fine), velocity, order)
        )
        sizes.append(int(eligible))
    provider = partial(
        local_equations,
        data=data,
        velocity=velocity,
        velocity_divergence=db,
        diffusion_divergence=dk,
        reaction=reaction,
        source=source,
        stabilization=stabilization,
        unusual_parameters=unusual_parameters,
        coarse_space=coarse_space,
        strong_faces=strong,
        diffusive_faces=tuple(physical),
        dirichlet=dirichlet,
    )
    problem = MultiscaleProblem(
        Equation(0, np.r_[-boundary, np.zeros(sum(sizes))]),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        tuple(sizes),
        fixed=fixed,
    )
    return TransportDefinition(problem, data, velocity, tuple(natural), bool(strong), mean_value)


def transport_constraints(definition: TransportDefinition, system: Any) -> list[Any]:
    """Declare the physical scalar mean only for zero reaction and tangential pure Robin data."""
    mesh = definition.data.mesh
    if (
        set(definition.natural) == set(mesh.boundary_faces)
        and all(r[3] for r in system.local_metadata)
        and boundary_tangent_2d(
            definition.data.skeleton, definition.velocity, definition.data.order
        )
    ):
        return [
            system.mean_constraint(
                [r[1] for r in system.local_metadata], definition.mean_value * sum(mesh.areas)
            )
        ]
    if definition.mean_value != 0:
        raise ValueError("mean_value requires zero reaction and tangential pure Robin data")
    return []


def recover_transport(
    definition: TransportDefinition, system: Any, solution: Any
) -> ScalarSolution:
    """Interpret executed nodal scalar values independently of auxiliary boundary reactions."""
    return ScalarSolution(
        definition.data.skeleton,
        tuple(r[0] for r in system.local_metadata),
        tuple(v[: r[4]] for v, r in zip(solution.fields, system.local_metadata, strict=True)),
        solution,
        definition.data.degree,
        definition.strong,
        definition.natural,
    )


def rad_local_assembly(
    cell: int,
    *,
    mesh: Any,
    skeleton: Any,
    degree: int,
    refinement: int,
    diffusion: Any,
    diffusion_divergence: Any,
    velocity: Any,
    velocity_divergence: Any,
    reaction: Any,
    source: Any,
    stabilization: str,
    order: int,
    strong_faces: tuple[int, ...] = (),
    dirichlet: Any = 0.0,
    diffusive_faces: tuple[int, ...] = (),
    coarse_space: str = "constants",
    unusual_parameters: Any = None,
    local_meshes: Any = None,
) -> Any:
    """Compile explicit scalar A/B/C rows for independent original-equation acquisition."""
    from pymhm import compile_local_equations
    from pymhm.core.contracts import LocalAssembly

    data = LocalPartition(mesh, skeleton, diffusion, degree, refinement, order, local_meshes)
    equation = local_equations(
        cell,
        data=data,
        velocity=velocity,
        velocity_divergence=velocity_divergence,
        diffusion_divergence=diffusion_divergence,
        reaction=reaction,
        source=source,
        stabilization=stabilization,
        unusual_parameters=unusual_parameters,
        coarse_space=coarse_space,
        strong_faces=strong_faces,
        diffusive_faces=diffusive_faces,
        dirichlet=dirichlet,
    )
    compiled = compile_local_equations(equation)
    return LocalAssembly(compiled.problem, compiled.metadata)
