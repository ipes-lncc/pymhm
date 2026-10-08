"""Displacement strain energy, declared rigid modes and weak traction equations."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.core.validation import dyadic_refinement, positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.fem.vector.elasticity import rigid_modes
from pymhm.fem.vector.elasticity_3d import rigid_modes_3d, vector_boundary_data_3d
from pymhm.fem.vector.primal import trace_detecting_enrichment, triangle_strain_operators
from pymhm.fem.vector.primal_3d import tetra_strain_operators
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.primal_elasticity import PrimalElasticitySolution
from pymhm.postprocessing.primal_elasticity_3d import Elasticity3DSolution


@dataclass(frozen=True)
class PrimalElasticityDefinition:
    """Explicit displacement problem, executed representation and physical rigid targets."""

    problem: MultiscaleProblem[int]
    skeleton: Any
    degree: int
    constitutive: Any
    lame_lambda: Any
    lame_mu: Any
    source: Any
    order: int
    dimension: int
    traction: dict[int, Any]
    rigid_targets: Any


def local_equations(
    cell: int,
    *,
    mesh: Any,
    skeleton: Any,
    degree: int,
    refinement: int,
    constitutive: Any,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    minimal_enrichment: bool,
) -> LocalEquations:
    """Write strain energy, signed B/C and literal moment-fixed rigid coordinates.

    A displacement field uses its actual executed nodal basis. Minimal enrichment
    stores the rectangular embedding in that field's reconstruction contract; it
    is applied to every operator, kernel, moment and recovered coefficient vector.
    Raw symmetric stress remains a constitutive gradient, without an H(div) claim.
    """
    fine = mesh.submesh(cell, refinement)
    dim = mesh.points.shape[1]
    k = degree + int(minimal_enrichment)
    owner = triangle_strain_operators if dim == 2 else tetra_strain_operators
    forms = owner(
        fine,
        k,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
    )
    matrix, load, nodes = forms.matrix, forms.load, forms.nodes
    scalar = (
        trace_coupling(mesh, cell, fine, skeleton, k)
        if dim == 2
        else tetra_trace_coupling(mesh, cell, fine, skeleton, k)
    )
    b = np.kron(scalar, np.eye(dim))
    measure = mesh.areas if dim == 2 else mesh.volumes
    modes = rigid_modes if dim == 2 else rigid_modes_3d
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid = modes(nodes, center).reshape(dim * len(nodes), -1)
    mass = sparse.kron(forms.mass, sparse.eye(dim))
    moments = mass @ rigid
    global_center = measure @ mesh.points[mesh.cells].mean(axis=1) / measure.sum()
    global_moments = mass @ modes(nodes, global_center).reshape(dim * len(nodes), -1)
    embedding = None
    if minimal_enrichment:
        embedding = trace_detecting_enrichment(fine, degree, scalar)
        matrix = sparse.csc_matrix(embedding.T @ matrix @ embedding)
        b, load = embedding.T @ b, embedding.T @ load
        moments, global_moments = embedding.T @ moments, embedding.T @ global_moments
        rigid = np.linalg.lstsq(embedding, rigid, rcond=None)[0]
    ids = (
        skeleton.cell_dofs(cell)
        if dim == 2
        else (dim * skeleton.cell_dofs(cell)[:, None] + np.arange(dim)).ravel()
    )
    return LocalEquations(
        matrix,
        load,
        b,
        -b.T,
        ids,
        kernel=rigid,
        moments=moments,
        metadata=(fine, global_moments, embedding),
        field_data=(
            nodal_field("displacement", fine, k, components=dim, reconstruction=embedding),
        ),
    )


def define_primal_elasticity(
    mesh: TriangleMesh | TetraMesh,
    *,
    constitutive: Any = None,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = None,
    dirichlet: Any = None,
    neumann: dict[int, Any] | None = None,
    traction: dict[int, Any] | None = None,
    skeleton: Any = None,
    degree: int = 1,
    minimal_enrichment: bool = False,
    local_refinement: int = 4,
    quadrature_order: int = 6,
    rigid_moments: Any = None,
) -> PrimalElasticityDefinition:
    """Declare the original primal spaces, strain form and physical exterior data.

    The tensor is SPD in symmetric Kelvin coordinates and lambda is finite for
    an isotropic law. Three/tetrahedral six rigid modes are retained locally.
    Pure physical traction requires their global integrated displacement targets.
    The primal discretization carries no uniform incompressible-limit estimate.
    """
    dim = mesh.points.shape[1]
    degree = positive_int(degree, "degree")
    refinement = (
        positive_int(local_refinement, "local_refinement")
        if dim == 2
        else dyadic_refinement(local_refinement, "local_refinement")
    )
    if not isinstance(minimal_enrichment, bool) or (dim == 3 and minimal_enrichment):
        raise ValueError("minimal enrichment is a declared triangular option")
    if neumann is not None and traction is not None:
        raise ValueError("provide one physical traction declaration")
    data = dict(traction if traction is not None else (neumann or {}))
    source = (0.0,) * dim if source is None else source
    dirichlet = (0.0,) * dim if dirichlet is None else dirichlet
    order = max(
        positive_int(quadrature_order, "quadrature_order"), degree + int(minimal_enrichment) + 1
    )
    if dim == 2:
        skeleton = (
            (
                SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
                if set(data) == set(mesh.boundary_faces)
                else SkeletonSpace(mesh, components=2)
            )
            if skeleton is None
            else skeleton
        )
        if skeleton.mesh is not mesh or skeleton.components != 2:
            raise ValueError("primal elasticity requires its own two-component skeleton")
        if minimal_enrichment and (
            refinement != 1
            or degree % 2
            or any(
                face.degrees != (degree - 1,) or face.breaks != (0.0, 1.0)
                for face in skeleton.faces
            )
        ):
            raise ValueError(
                "minimal enrichment requires one triangle, even k and unsplit P(k-1) traces"
            )
        boundary, physical_fixed = boundary_data(
            skeleton, dirichlet, data, order=max(order, degree + 2)
        )
        fixed = {i: -v for i, v in physical_fixed.items()}
        width = 3
    else:
        skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
        if (
            skeleton.mesh is not mesh
            or np.any(skeleton.degrees != 1)
            or np.any(skeleton.subdivisions > refinement)
        ):
            raise ValueError(
                "tetrahedral primal elasticity requires matching P1 rigid-motion traces"
            )
        boundary, fixed = vector_boundary_data_3d(skeleton, dirichlet, data, order)
        width = 6
    targets = np.zeros(width) if rigid_moments is None else np.asarray(rigid_moments)
    if targets.shape != (width,) or np.iscomplexobj(targets) or not np.isfinite(targets).all():
        raise ValueError("rigid_moments must contain finite physical integrated moments")
    provider = partial(
        local_equations,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=refinement,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
        minimal_enrichment=minimal_enrichment,
    )
    size = skeleton.size if dim == 2 else 3 * skeleton.size
    problem = MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(width * len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        size,
        (width,) * len(mesh.cells),
        fixed=fixed,
    )
    return PrimalElasticityDefinition(
        problem,
        skeleton,
        degree + int(minimal_enrichment),
        constitutive,
        lame_lambda,
        lame_mu,
        source,
        order,
        dim,
        data,
        targets,
    )


def primal_constraints(
    definition: PrimalElasticityDefinition, system: MultiscaleSystem
) -> list[Any]:
    """Prescribe physical rigid integrals only for a complete traction boundary."""
    if set(definition.traction) != set(definition.skeleton.mesh.boundary_faces):
        return []
    return [
        system.mean_constraint([entry[1][:, i] for entry in system.local_metadata], target)
        for i, target in enumerate(definition.rigid_targets)
    ]


def recover_primal_elasticity(
    definition: PrimalElasticityDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> Any:
    """Recover nodal displacement with the actual rectangular enrichment embedding."""
    meshes = tuple(entry[0] for entry in system.local_metadata)
    values = tuple(
        (field if entry[2] is None else entry[2] @ field).reshape(-1, definition.dimension)
        for field, entry in zip(solution.fields, system.local_metadata, strict=True)
    )
    if definition.dimension == 3:
        return Elasticity3DSolution(
            definition.skeleton,
            meshes,
            values,
            solution,
            definition.degree,
            definition.constitutive,
            definition.lame_lambda,
            definition.lame_mu,
            definition.source,
            definition.order,
        )
    return PrimalElasticitySolution(
        definition.skeleton,
        meshes,
        values,
        (),
        solution,
        definition.degree,
        1,
        definition.constitutive,
        definition.lame_lambda,
        definition.lame_mu,
        definition.source,
        definition.order,
    )
