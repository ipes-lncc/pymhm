"""User-defined BDM and tensor RT weak-symmetry elasticity block equations.

This tutorial reuses the existing compliance/divergence/asymmetry integrals,
rigid moments and normal maps. It declares the coupled mixed equations and
physical hydrostatic gauge without invoking a physical solver or local factory.
The focused interfaces retain the BDM2/P1/P1 and RT1/Q1/P1 spaces used in the
two introductory controls, with fully prescribed displacement boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, MultiscaleSystem, assemble
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.tensor_rt import tensor_rt_trace_map as _trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import multiindices
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.physical import boundary_normal_integral as _boundary_volume_flux
from pymhm.fem.traces.physical import require_compatible_displacement_flux
from pymhm.fem.vector.stress import displacement_rigid_moments as _displacement_moments
from pymhm.fem.vector.stress import mixed_elasticity_operators as bdm_operators
from pymhm.fem.vector.stress import rigid_values as _rigid_values
from pymhm.fem.vector.stress import stress_trace_moments as _stress_trace_moments
from pymhm.fem.vector.stress_tensor import tensor_rigid_moments as _modal_rigid
from pymhm.fem.vector.stress_tensor import tensor_stress_operators as tensor_operators
from pymhm.postprocessing.stress import MixedElasticitySolution
from pymhm.postprocessing.stress_tensor import TensorRTElasticitySolution


@dataclass(frozen=True)
class StressCell:
    """One macrocell's explicitly selected material, spaces and physical source."""

    mesh: Any
    skeleton: SkeletonSpace
    cell: int
    refinement: int
    lame_lambda: Any
    lame_mu: Any
    source: Any
    order: int
    compliance: Any = None


@dataclass(frozen=True)
class StressMetadata:
    """Executed local coefficient partition and physical hydrostatic moments."""

    mesh: Any
    skeleton: SkeletonSpace
    stress_size: int
    displacement_size: int
    rotation_size: int
    scalar_size: int
    trace_moment: Any
    bulk_moment: Any
    compliance_scale: float


def mixed_equations(
    item: StressCell,
    operators: tuple[Any, Any, Any, Any],
    normal_degree: int,
    displacement_rigid: Any,
    displacement_moments: Any,
    rotation_rigid: Any,
    mapping: Any,
    metadata: StressMetadata,
) -> LocalEquations:
    """Declare compliance, force, weak symmetry and prescribed normal stress rows.

    Unknowns are (sigma,u,q,psi), with psi on every local boundary edge.
    The matrix rows are A sigma+Div.T u+Asym.T q-S psi=0,
    Div sigma=-f, Asym sigma=0, and -S.T sigma-map lambda=0.
    Lambda is negative Cauchy traction. The three declared exact kernels and
    their displacement moments retain both translations and centered rotation.
    C=-B.T is the actual global displacement pairing in these coordinates.
    """
    mass, divergence, asymmetry, force = operators
    fine = metadata.mesh
    ns, nu, nr = metadata.stress_size, metadata.displacement_size, metadata.rotation_size
    boundary_size = 2 * (normal_degree + 1)
    nb = boundary_size * len(fine.boundary_faces)
    rows = (boundary_size * fine.boundary_faces[:, None] + np.arange(boundary_size)).ravel()
    selector = sparse.coo_matrix((np.ones(nb), (rows, np.arange(nb))), shape=(ns, nb)).tocsc()
    matrix = sparse.bmat(
        [
            [mass, divergence.T, asymmetry.T, -selector],
            [
                divergence,
                sparse.csc_matrix((nu, nu)),
                sparse.csc_matrix((nu, nr)),
                sparse.csc_matrix((nu, nb)),
            ],
            [
                asymmetry,
                sparse.csc_matrix((nr, nu)),
                sparse.csc_matrix((nr, nr)),
                sparse.csc_matrix((nr, nb)),
            ],
            [
                -selector.T,
                sparse.csc_matrix((nb, nu)),
                sparse.csc_matrix((nb, nr)),
                sparse.csc_matrix((nb, nb)),
            ],
        ],
        format="csc",
    )
    kernel = np.zeros((ns + nu + nr + nb, 3))
    kernel[ns : ns + nu] = displacement_rigid
    kernel[ns + nu : ns + nu + nr, 2] = rotation_rigid
    center = item.mesh.points[item.mesh.cells[item.cell]].mean(axis=0)
    rigid_boundary = _rigid_values(fine.points[fine.faces[fine.boundary_faces]], center)
    boundary_coefficients = np.zeros((len(rigid_boundary), normal_degree + 1, 2, 3))
    boundary_coefficients[:, 0] = rigid_boundary.mean(axis=1)
    boundary_coefficients[:, 1] = (rigid_boundary[:, 1] - rigid_boundary[:, 0]) / 2
    kernel[-nb:] = boundary_coefficients.reshape(-1, 3)
    moments = np.zeros_like(kernel)
    moments[ns : ns + nu] = displacement_moments
    coupling = np.zeros((matrix.shape[0], mapping.shape[1]))
    coupling[-nb:] = -mapping
    return LocalEquations(
        a=matrix,
        L=np.r_[np.zeros(ns), -force, np.zeros(nr + nb)],
        b=coupling,
        c=-coupling.T,
        dofs=item.skeleton.cell_dofs(item.cell),
        kernel=kernel,
        moments=moments,
        metadata=metadata,
    )


def bdm_equations(item: StressCell) -> LocalEquations:
    """Use shared BDM2/P1/P1 integrals with explicit rigid and boundary maps."""
    family = BDMFamily(2, 0)
    fine = item.mesh.submesh(item.cell, item.refinement)
    operators = bdm_operators(
        fine, item.lame_lambda, item.lame_mu, item.source, item.order, family, item.compliance
    )
    ns, nu, nr = (block.shape[0] for block in operators[:3])
    size = ns + nu + nr + 6 * len(fine.boundary_faces)
    plain, weighted, scale = _stress_trace_moments(
        fine, size, item.lame_lambda, item.lame_mu, item.order, family, item.compliance
    )
    center = item.mesh.points[item.mesh.cells[item.cell]].mean(axis=0)
    nodes = np.einsum("qi,tij->tqj", multiindices(1), fine.points[fine.cells])
    mapping = np.kron(family.trace_map(item.mesh, item.cell, fine, item.skeleton), np.eye(2))
    metadata = StressMetadata(fine, item.skeleton, ns, nu, nr, 3, plain, weighted, scale)
    return mixed_equations(
        item,
        operators,
        2,
        _rigid_values(nodes, center).reshape(-1, 3),
        _displacement_moments(fine, center, family),
        -np.ones(nr),
        mapping,
        metadata,
    )


def tensor_equations(item: StressCell) -> LocalEquations:
    """Use shared RT1/Q1/P1 integrals and physical modal rigid moments."""
    fine = item.mesh.submesh(item.cell, item.refinement)
    operators = tensor_operators(
        fine, 1, 0, item.lame_lambda, item.lame_mu, item.source, item.order, item.compliance
    )
    mass, divergence, asymmetry, force, plain, weighted, scale = operators
    ns, nu, nr = mass.shape[0], divergence.shape[0], asymmetry.shape[0]
    size = ns + nu + nr + 4 * len(fine.boundary_faces)
    center = item.mesh.points[item.mesh.cells[item.cell]].mean(axis=0)
    rigid, moments = _modal_rigid(fine, 1, center)
    rotation = np.zeros(nr)
    rotation[::3] = -1
    mapping = np.kron(_trace_map(item.mesh, item.cell, fine, item.skeleton, 1), np.eye(2))
    metadata = StressMetadata(
        fine,
        item.skeleton,
        ns,
        nu,
        nr,
        4,
        np.pad(plain, (0, size - ns)),
        np.pad(weighted, (0, size - ns)),
        scale,
    )
    return mixed_equations(
        item, (mass, divergence, asymmetry, force), 1, rigid, moments, rotation, mapping, metadata
    )


def global_equation(boundary: Any, retained_size: int) -> Equation:
    """Declare displacement trace data and zero extra load in retained coordinates."""
    return Equation(0, np.pad(boundary, (0, retained_size)))


def _assemble(
    mesh: Any,
    local_provider: Any,
    *,
    normal_degree: int,
    dirichlet: Any,
    source: Any,
    lame_lambda: Any,
    lame_mu: Any,
    refinement: int,
    order: int,
    compliance: Any,
) -> tuple[MultiscaleSystem, Any]:
    """Assemble explicit forms and impose the physical full-Dirichlet stress gauge."""
    exterior = set(mesh.boundary_faces)
    skeleton = SkeletonSpace(
        mesh,
        tuple(
            FaceSpace.uniform(normal_degree, refinement)
            if face in exterior
            else FaceSpace.uniform(1)
            for face in range(len(mesh.faces))
        ),
        components=2,
    )
    if normal_degree == 2:
        BDMFamily(2, 0).validate_trace(skeleton, refinement)
    boundary, fixed = boundary_data(skeleton, dirichlet, {}, order=max(5, order))
    items = tuple(
        StressCell(
            mesh, skeleton, cell, refinement, lame_lambda, lame_mu, source, order, compliance
        )
        for cell in range(len(mesh.cells))
    )
    problem = MultiscaleProblem(
        global_equation(boundary, 3 * len(items)),
        local_provider,
        items,
        skeleton.size,
        (3,) * len(items),
        fixed=fixed,
    )
    system = assemble(problem)
    metadata = system.local_metadata
    scale = max(item.compliance_scale for item in metadata)
    volume_flux = _boundary_volume_flux(skeleton, boundary)
    if scale == 0:
        require_compatible_displacement_flux(
            volume_flux, _boundary_volume_flux(skeleton, boundary, absolute=True)
        )
        gauge = system.mean_constraint([item.trace_moment for item in metadata], 0.0)
    else:
        gauge = system.mean_constraint(
            [item.bulk_moment / scale for item in metadata], volume_flux / scale
        )
    return system, system.solve(constraints=[gauge])


def solve_bdm(
    mesh: Any,
    *,
    dirichlet: Any,
    source: Any = (0.0, 0.0),
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    local_refinement: int = 2,
    quadrature_order: int = 4,
) -> tuple[MultiscaleSystem, MixedElasticitySolution]:
    """Solve declared BDM2/P1/P1 forms with the physical displacement stress gauge.

    ``compliance`` optionally replaces the Lamé law with a full Cartesian
    operator in (xx,xy,yx,yy) coordinates, including its declared positive
    skew extension. Existing shared material kernels validate that convention.
    """
    system, hybrid = _assemble(
        mesh,
        bdm_equations,
        normal_degree=2,
        dirichlet=dirichlet,
        source=source,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        refinement=local_refinement,
        order=quadrature_order,
        compliance=compliance,
    )
    stress, displacement, rotation = _fields(system, hybrid)
    return system, MixedElasticitySolution(
        system.local_metadata[0].skeleton,
        tuple(item.mesh for item in system.local_metadata),
        stress,
        displacement,
        rotation,
        hybrid,
        source,
        quadrature_order,
        BDMFamily(2, 0),
    )


def _fields(system: MultiscaleSystem, hybrid: Any) -> tuple[Any, Any, Any]:
    """Partition executed coefficients without changing bases, signs or evaluation."""
    stress, displacement, rotation = [], [], []
    for values, item in zip(hybrid.fields, system.local_metadata, strict=True):
        ns, nu, nr = item.stress_size, item.displacement_size, item.rotation_size
        stress.append(values[:ns].reshape(-1, 2))
        displacement.append(values[ns : ns + nu].reshape(-1, item.scalar_size, 2))
        rotation.append(values[ns + nu : ns + nu + nr].reshape(len(item.mesh.cells), -1))
    return tuple(stress), tuple(displacement), tuple(rotation)


def solve_tensor(
    mesh: Any,
    *,
    dirichlet: Any,
    source: Any = (0.0, 0.0),
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    local_refinement: int = 2,
    quadrature_order: int = 6,
) -> tuple[MultiscaleSystem, TensorRTElasticitySolution]:
    """Solve declared RT1/Q1/P1 forms with physical normal maps and stress gauge.

    The optional full Cartesian ``compliance`` follows the same (xx,xy,yx,yy)
    and explicit skew-extension convention as the BDM declaration. Ordinary
    quadrature on these fine rectangles requires material-interface alignment.
    """
    system, hybrid = _assemble(
        mesh,
        tensor_equations,
        normal_degree=1,
        dirichlet=dirichlet,
        source=source,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        refinement=local_refinement,
        order=quadrature_order,
        compliance=compliance,
    )
    stress, displacement, rotation = _fields(system, hybrid)
    return system, TensorRTElasticitySolution(
        system.local_metadata[0].skeleton,
        tuple(item.mesh for item in system.local_metadata),
        stress,
        displacement,
        rotation,
        hybrid,
        1,
        0,
        source,
        quadrature_order,
    )
