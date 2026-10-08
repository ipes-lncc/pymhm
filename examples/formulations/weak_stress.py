"""Explicit weak-symmetry elasticity equations with independent normal/interior spaces.

The application specifies compliance, divergence, asymmetry, kernels, moments,
traction signs and physical gauges. Shared FEM owners integrate the stated volume
and boundary bases; generic assembly and solves are separate user operations.
"""

from dataclasses import dataclass
from functools import partial
from itertools import product
from typing import Any

import numpy as np
from scipy import sparse

from pymhm import Equation, LocalEquations, MultiscaleProblem
from pymhm.core.contracts import HybridSolution
from pymhm.core.multiscale import MultiscaleSystem
from pymhm.core.validation import positive_int
from pymhm.fem.geometry import volume_centroid
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.hdiv.family_3d import HDiv3DFamily
from pymhm.fem.hdiv.mixed_3d import hdiv3d_trace_mapping
from pymhm.fem.hdiv.tensor_rt import tensor_rt_trace_map
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.quadrilateral import grid_resolves_material
from pymhm.fem.scalar.triangle import multiindices
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.physical import boundary_normal_integral
from pymhm.fem.traces.traction_3d import TractionSkeleton3D, traction_boundary_data
from pymhm.fem.vector.stress import (
    displacement_rigid_moments,
    mixed_elasticity_operators,
    rigid_values,
    stress_trace_moments,
    traction_mapping,
)
from pymhm.fem.vector.stress_3d import (
    boundary_selector,
    mixed_elasticity_operators_3d,
    rigid_coefficients,
)
from pymhm.fem.vector.stress_tensor import tensor_rigid_moments, tensor_stress_operators
from pymhm.materials.cartesian import CartesianCellField
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.modal import modal_field
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.piola import hdiv_field
from pymhm.postprocessing.stress import MixedElasticitySolution
from pymhm.postprocessing.stress_3d import MixedElasticity3DSolution
from pymhm.postprocessing.stress_tensor import TensorRTElasticitySolution


@dataclass(frozen=True)
class WeakStressDefinition:
    """Local/global equations, exact field coordinates and physical boundary moments."""

    problem: MultiscaleProblem[int]
    skeleton: Any
    family: Any
    dimension: int
    source: Any
    order: int
    mean_pressure: float
    volume_flux: float
    volume_scale: float
    traction: dict[int, Any]
    rigid_targets: Any


def triangular_stress_equations(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    family: BDMFamily,
    refinement: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> LocalEquations:
    """Declare row-wise BDM compliance/divergence/asymmetry and exact rigid kernels.

    Stress normal and interior degrees are independent through BDMFamily.
    Displacement/rotation are DG P(k+n-1); the multiplier is negative Cauchy
    traction. Boundary displacement auxiliaries use actual Legendre coefficients.
    Three rigid modes and their physical DG displacement moments are explicit.
    """
    coarse = mesh
    fine = mesh.submesh(cell, refinement)
    mass, divergence, asymmetry, force = mixed_elasticity_operators(
        fine, lame_lambda, lame_mu, source, order, family, compliance
    )
    ns, nu, nr = mass.shape[0], divergence.shape[0], asymmetry.shape[0]
    boundary_size = 2 * (family.degree + 1)
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
    center = coarse.points[coarse.cells[cell]].mean(axis=0)
    kernel = np.zeros((ns + nu + nr + nb, 3))
    degree = family.polynomial_degree - 1
    nodes = np.einsum("qi,tij->tqj", multiindices(degree) / degree, fine.points[fine.cells])
    kernel[ns : ns + nu] = rigid_values(nodes, center).reshape(-1, 3)
    kernel[ns + nu : ns + nu + nr, 2] = -1
    boundary = rigid_values(fine.points[fine.faces[fine.boundary_faces]], center)
    boundary_coefficients = np.zeros((len(boundary), family.degree + 1, 2, 3))
    boundary_coefficients[:, 0] = boundary.mean(axis=1)
    boundary_coefficients[:, 1] = (boundary[:, 1] - boundary[:, 0]) / 2
    kernel[ns + nu + nr :] = boundary_coefficients.reshape(-1, 3)
    constraints = np.zeros_like(kernel)
    constraints[ns : ns + nu] = displacement_rigid_moments(fine, center, family)
    mapping = traction_mapping(coarse, cell, fine, skeleton, family)
    coupling = np.zeros((matrix.shape[0], mapping.shape[1]))
    coupling[-nb:] = -mapping
    plain, weighted, scale = stress_trace_moments(
        fine, ns, lame_lambda, lame_mu, order, family, compliance
    )
    plain_full, weighted_full = np.zeros(matrix.shape[0]), np.zeros(matrix.shape[0])
    plain_full[:ns], weighted_full[:ns] = plain, weighted
    global_center = volume_centroid(mesh)
    global_moments = np.zeros_like(kernel)
    global_moments[ns : ns + nu] = displacement_rigid_moments(fine, global_center, family)
    load = np.r_[np.zeros(ns), -force, np.zeros(nr + nb)]
    return LocalEquations(
        matrix,
        load,
        coupling,
        -coupling.T,
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=constraints,
        metadata=(fine, ns, nu, nr, plain_full, weighted_full, scale, global_moments),
        field_data=(
            hdiv_field(
                "stress",
                fine,
                family,
                components=2,
                reconstruction=sparse.eye(len(load), format="csr")[:ns],
            ),
            hdiv_field(
                "stress_divergence",
                fine,
                family,
                components=2,
                reconstruction=sparse.eye(len(load), format="csr")[:ns],
                divergence=True,
            ),
            nodal_field(
                "displacement",
                fine,
                degree,
                components=2,
                discontinuous=True,
                reconstruction=sparse.eye(len(load), format="csr")[ns : ns + nu],
            ),
            nodal_field(
                "rotation",
                fine,
                degree,
                discontinuous=True,
                reconstruction=sparse.eye(len(load), format="csr")[ns + nu : ns + nu + nr],
            ),
        ),
    )


def tensor_stress_equations(
    cell: int,
    *,
    mesh: CartesianMacroMesh,
    skeleton: SkeletonSpace,
    degree: int,
    enrichment: int,
    refinement: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> LocalEquations:
    """Declare row-wise tensor RT, Q_s displacement and complete P_s rotation.

    The full stress compliance and negative weak-symmetry saddle share the public
    integration owner. Boundary auxiliary and displacement rigid coefficients
    are expressed in the actual Legendre products. s=degree+enrichment.
    """
    fine = mesh.submesh(cell, refinement)
    for coefficient in (lame_lambda, lame_mu):
        if isinstance(coefficient, CartesianCellField) and not grid_resolves_material(
            fine, coefficient
        ):
            raise ValueError("Cartesian Lame material interfaces must align with fine rectangles")
    mass, div, asym, force, trace, bulk_trace, bulk_max = tensor_stress_operators(
        fine, degree, enrichment, lame_lambda, lame_mu, source, order, compliance
    )
    ns, nu, nr = mass.shape[0], div.shape[0], asym.shape[0]
    nb = 2 * (degree + 1) * len(fine.boundary_faces)
    rows = (2 * (degree + 1) * fine.boundary_faces[:, None] + np.arange(2 * (degree + 1))).ravel()
    selector = sparse.coo_matrix((np.ones(nb), (rows, np.arange(nb))), shape=(ns, nb)).tocsc()
    matrix = sparse.bmat(
        [
            [mass, div.T, asym.T, -selector],
            [
                div,
                sparse.csc_matrix((nu, nu)),
                sparse.csc_matrix((nu, nr)),
                sparse.csc_matrix((nu, nb)),
            ],
            [
                asym,
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
    size = ns + nu + nr + nb
    center = mesh.points[mesh.cells[cell]].mean(axis=0)
    rigid, moments = tensor_rigid_moments(fine, degree + enrichment, center)
    kernel = np.zeros((size, 3))
    kernel[ns : ns + nu] = rigid
    nrot = nr // len(fine.cells)
    kernel[ns + nu : ns + nu + nr : nrot, 2] = -1
    boundary = rigid_values(fine.points[fine.faces[fine.boundary_faces]], center)
    coefficients = np.zeros((len(boundary), degree + 1, 2, 3))
    coefficients[:, 0], coefficients[:, 1] = (
        boundary.mean(axis=1),
        (boundary[:, 1] - boundary[:, 0]) / 2,
    )
    kernel[-nb:] = coefficients.reshape(-1, 3)
    constraints = np.zeros_like(kernel)
    constraints[ns : ns + nu] = moments
    mapping = np.kron(tensor_rt_trace_map(mesh, cell, fine, skeleton, degree), np.eye(2))
    coupling = np.zeros((size, mapping.shape[1]))
    coupling[-nb:] = -mapping
    plain_full, weighted_full = np.zeros(size), np.zeros(size)
    plain_full[:ns], weighted_full[:ns] = trace, bulk_trace
    center_global = mesh.areas @ mesh.points[mesh.cells].mean(axis=1) / mesh.areas.sum()
    global_moments = np.zeros_like(kernel)
    global_moments[ns : ns + nu] = tensor_rigid_moments(fine, degree + enrichment, center_global)[1]
    load = np.r_[np.zeros(ns), -force, np.zeros(nr + nb)]
    return LocalEquations(
        matrix,
        load,
        coupling,
        -coupling.T,
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=constraints,
        metadata=(fine, ns, nu, nr, plain_full, weighted_full, bulk_max, global_moments),
        field_data=(
            hdiv_field(
                "stress",
                fine,
                "tensor-RT",
                degree=degree,
                enrichment=enrichment,
                components=2,
                reconstruction=sparse.eye(size, format="csr")[:ns],
            ),
            hdiv_field(
                "stress_divergence",
                fine,
                "tensor-RT",
                degree=degree,
                enrichment=enrichment,
                components=2,
                reconstruction=sparse.eye(size, format="csr")[:ns],
                divergence=True,
            ),
            modal_field(
                "displacement",
                fine,
                tuple(
                    (a, b)
                    for b in range(degree + enrichment + 1)
                    for a in range(degree + enrichment + 1)
                ),
                convention="legendre",
                components=2,
                reconstruction=sparse.eye(size, format="csr")[ns : ns + nu],
            ),
            modal_field(
                "rotation",
                fine,
                tuple(
                    (a, b)
                    for b in range(degree + enrichment + 1)
                    for a in range(degree + enrichment + 1 - b)
                ),
                convention="legendre",
                reconstruction=sparse.eye(size, format="csr")[ns + nu : ns + nu + nr],
            ),
        ),
    )


def tetrahedral_stress_equations(
    cell: int,
    *,
    skeleton: TractionSkeleton3D,
    family: HDiv3DFamily,
    refinement: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    compliance: Any = None,
) -> LocalEquations:
    """Declare AFW tetrahedral BDM stress, DG displacement and axial rotation.

    Six rigid modes are translations then e_i cross (x-center); axial rotation
    coefficients have -I in their constant modes. Natural auxiliary coordinates
    and skeleton coefficients are original physical normal integral moments.
    """
    coarse = skeleton.mesh
    fine = coarse.submesh(cell, refinement)
    mass, div, asym, force, plain, weighted, scale = mixed_elasticity_operators_3d(
        fine,
        family,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        compliance=compliance,
        quadrature_order=order,
    )
    ns, nu = mass.shape[0], div.shape[0]
    selector = boundary_selector(fine, family, ns)
    nb = selector.shape[1]
    z = sparse.csc_matrix
    matrix = sparse.bmat(
        [
            [mass, div.T, asym.T, -selector],
            [div, z((nu, nu)), z((nu, nu)), z((nu, nb))],
            [asym, z((nu, nu)), z((nu, nu)), z((nu, nb))],
            [-selector.T, z((nb, nu)), z((nb, nu)), z((nb, nb))],
        ],
        format="csc",
    )
    center = coarse.points[coarse.cells[cell]].mean(axis=0)
    rigid, moments, boundary = rigid_coefficients(fine, family, center)
    kernel, constraints = np.zeros((matrix.shape[0], 6)), np.zeros((matrix.shape[0], 6))
    kernel[ns : ns + nu] = rigid
    rotations = kernel[ns + nu : ns + 2 * nu].reshape(len(fine.cells), family.pressure_size, 3, 6)
    rotations[:, 0, :, 3:] = -np.eye(3)
    kernel[-nb:] = boundary
    constraints[ns : ns + nu] = moments
    mapping = np.kron(
        hdiv3d_trace_mapping(skeleton.scalar, cell, fine, family.normal_degree), np.eye(3)
    )
    coupling = np.zeros((matrix.shape[0], mapping.shape[1]))
    coupling[-nb:] = -mapping
    size = matrix.shape[0]
    plain_full, weighted_full = np.zeros(size), np.zeros(size)
    plain_full[:ns], weighted_full[:ns] = plain, weighted
    center_global = coarse.volumes @ coarse.points[coarse.cells].mean(axis=1) / coarse.volumes.sum()
    global_moments = np.zeros_like(kernel)
    global_moments[ns : ns + nu] = rigid_coefficients(fine, family, center_global)[1]
    load = np.r_[np.zeros(ns), -force, np.zeros(nu + nb)]
    return LocalEquations(
        matrix,
        load,
        coupling,
        -coupling.T,
        skeleton.cell_dofs(cell),
        kernel=kernel,
        moments=constraints,
        metadata=(fine, ns, nu, nu, plain_full, weighted_full, scale, global_moments),
        field_data=(
            hdiv_field(
                "stress",
                fine,
                family,
                components=3,
                reconstruction=sparse.eye(size, format="csr")[:ns],
            ),
            hdiv_field(
                "stress_divergence",
                fine,
                family,
                components=3,
                reconstruction=sparse.eye(size, format="csr")[:ns],
                divergence=True,
            ),
            modal_field(
                "displacement",
                fine,
                tuple(
                    e
                    for e in product(range(family.pressure_degree + 1), repeat=3)
                    if sum(e) <= family.pressure_degree
                ),
                components=3,
                reconstruction=sparse.eye(size, format="csr")[ns : ns + nu],
            ),
            modal_field(
                "rotation",
                fine,
                tuple(
                    e
                    for e in product(range(family.pressure_degree + 1), repeat=3)
                    if sum(e) <= family.pressure_degree
                ),
                components=3,
                reconstruction=sparse.eye(size, format="csr")[ns + nu : ns + 2 * nu],
            ),
        ),
    )


def define_weak_stress(
    mesh: TriangleMesh | CartesianMacroMesh | AffineMixedMesh,
    *,
    stress_degree: int = 2,
    enrichment: int = 0,
    trace_degree: int = 1,
    subdivisions: int = 1,
    skeleton: Any = None,
    local_refinement: int = 2,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    source: Any = None,
    dirichlet: Any = None,
    traction: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    rigid_moments: Any = None,
    quadrature_order: int = 6,
) -> WeakStressDefinition:
    """Declare triangular BDM, rectangular RT or tetrahedral AFW weak stress forms.

    Actual normal/interior spaces stay independent. Triangular/rectangular boundary
    traces use the full normal degree on the declared fine partition, whereas
    interior traces default to P1. Tetrahedra use the stated trace degree and
    matching subdivisions. Physical traction fixes negative multiplier moments;
    other exterior faces impose displacement weakly. Hydrostatic and rigid gauges
    are separately declared by weak_stress_constraints after generic assembly.
    """
    refinement = positive_int(local_refinement, "local_refinement")
    degree = positive_int(stress_degree, "stress_degree", 1)
    enrich = positive_int(enrichment, "enrichment", 0)
    prescribed = {} if traction is None else dict(traction)
    dimension = 3 if isinstance(mesh, AffineMixedMesh) else 2
    if not np.isfinite(mean_pressure) or (prescribed and mean_pressure != 0):
        raise ValueError("finite pressure mean applies only to full displacement boundaries")
    source = (0.0,) * dimension if source is None else source
    dirichlet = (0.0,) * dimension if dirichlet is None else dirichlet
    targets = (
        np.zeros(3 if dimension == 2 else 6) if rigid_moments is None else np.asarray(rigid_moments)
    )
    if (
        np.iscomplexobj(targets)
        or targets.shape != (3 if dimension == 2 else 6,)
        or not np.isfinite(targets).all()
    ):
        raise ValueError("rigid_moments must contain the physical finite real rigid integrals")
    if dimension == 3:
        if enrich != 0 or degree < 2:
            raise ValueError(
                "tetrahedral AFW uses BDM stress degree >=2 without this enrichment option"
            )
        family = HDiv3DFamily("tetrahedron", degree - 1, degree)
        skeleton = (
            TractionSkeleton3D(mesh, trace_degree, subdivisions) if skeleton is None else skeleton
        )
        if skeleton.mesh is not mesh or trace_degree > degree or refinement % skeleton.subdivisions:
            raise ValueError("AFW trace degree/partition must fit the stress normal moments")
        order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
        boundary, fixed, flux, scale = traction_boundary_data(
            skeleton, dirichlet, prescribed, order
        )
        boundary = -boundary
        provider = partial(
            tetrahedral_stress_equations,
            skeleton=skeleton,
            family=family,
            refinement=refinement,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            compliance=compliance,
            source=source,
            order=order,
        )
    else:
        family = (
            BDMFamily(degree, enrich)
            if isinstance(mesh, (TriangleMesh, PolygonMesh))
            else (degree, enrich)
        )
        order = max(positive_int(quadrature_order, "quadrature_order"), degree + enrich + 2)
        exterior = set(mesh.boundary_faces)
        skeleton = (
            SkeletonSpace(
                mesh,
                tuple(
                    FaceSpace.uniform(degree, refinement) if f in exterior else FaceSpace.uniform(1)
                    for f in range(len(mesh.faces))
                ),
                components=2,
            )
            if skeleton is None
            else skeleton
        )
        if skeleton.mesh is not mesh or skeleton.components != 2:
            raise ValueError("weak stress requires its own two-component skeleton")
        if isinstance(mesh, (TriangleMesh, PolygonMesh)):
            if family.polynomial_degree < 2:
                raise ValueError("DG displacement must represent rigid rotations")
            family.validate_trace(skeleton, refinement)
            provider = partial(
                triangular_stress_equations,
                mesh=mesh,
                skeleton=skeleton,
                family=family,
                refinement=refinement,
                lame_lambda=lame_lambda,
                lame_mu=lame_mu,
                compliance=compliance,
                source=source,
                order=order,
            )
        else:
            for face in skeleton.faces:
                positions = np.asarray(face.breaks) * refinement
                if max(face.degrees) > degree or not np.allclose(
                    positions, np.round(positions), atol=1e-12, rtol=0
                ):
                    raise ValueError(
                        "tensor stress trace degrees/partition must fit normal moments"
                    )
            provider = partial(
                tensor_stress_equations,
                mesh=mesh,
                skeleton=skeleton,
                degree=degree,
                enrichment=enrich,
                refinement=refinement,
                lame_lambda=lame_lambda,
                lame_mu=lame_mu,
                compliance=compliance,
                source=source,
                order=order,
            )
        negative = {face: partial(_negative, datum=value) for face, value in prescribed.items()}
        moments, fixed = boundary_data(skeleton, dirichlet, negative, order=order)
        flux, scale = (
            boundary_normal_integral(skeleton, moments),
            boundary_normal_integral(skeleton, moments, absolute=True),
        )
        boundary = moments
    coarse = 3 if dimension == 2 else 6
    problem = MultiscaleProblem(
        Equation(0, np.r_[boundary, np.zeros(coarse * len(mesh.cells))]),
        provider,
        range(len(mesh.cells)),
        skeleton.size,
        (coarse,) * len(mesh.cells),
        fixed=fixed,
    )
    return WeakStressDefinition(
        problem,
        skeleton,
        family,
        dimension,
        source,
        order,
        mean_pressure,
        flux,
        scale,
        prescribed,
        targets,
    )


def _negative(points: Any, *, datum: Any) -> Any:
    """Evaluate physical outward traction with the declared negative multiplier sign."""
    return -vector_values(datum, points)


def weak_stress_constraints(
    definition: WeakStressDefinition, system: MultiscaleSystem
) -> list[Any]:
    """Declare physical hydrostatic/compressibility and pure-traction rigid integrals."""
    mesh = definition.skeleton.mesh
    measure = float((mesh.volumes if definition.dimension == 3 else mesh.areas).sum())
    gauges = []
    if not definition.traction:
        scale = max(info[6] for info in system.local_metadata)
        if scale == 0:
            if abs(definition.volume_flux) > 1e-10 * definition.volume_scale:
                raise ValueError("incompatible incompressible displacement boundary data")
            gauges.append(
                system.mean_constraint(
                    [info[4] for info in system.local_metadata],
                    -definition.dimension * definition.mean_pressure * measure,
                )
            )
        else:
            if definition.mean_pressure != 0:
                raise ValueError("pressure mean is a gauge only in the incompressible limit")
            gauges.append(
                system.mean_constraint(
                    [info[5] / scale for info in system.local_metadata],
                    definition.volume_flux / scale,
                )
            )
    if set(definition.traction) == set(mesh.boundary_faces):
        gauges.extend(
            system.mean_constraint([info[7][:, i] for info in system.local_metadata], target)
            for i, target in enumerate(definition.rigid_targets)
        )
    return gauges


def recover_weak_stress(
    definition: WeakStressDefinition, system: MultiscaleSystem, solution: HybridSolution
) -> Any:
    """Interpret actual coefficients as H(div) stress, broken displacement and rotation."""
    meshes, stresses, displacements, rotations = [], [], [], []
    for field, (fine, ns, nu, nr, *_) in zip(solution.fields, system.local_metadata, strict=True):
        meshes.append(fine)
        stresses.append(field[:ns].reshape(-1, definition.dimension))
        displacements.append(field[ns : ns + nu].reshape(len(fine.cells), -1, definition.dimension))
        rotations.append(
            field[ns + nu : ns + nu + nr].reshape(len(fine.cells), -1, 3)
            if definition.dimension == 3
            else field[ns + nu : ns + nu + nr].reshape(len(fine.cells), -1)
        )
    args = (
        definition.skeleton,
        tuple(meshes),
        tuple(stresses),
        tuple(displacements),
        tuple(rotations),
        solution,
    )
    if definition.dimension == 3:
        return MixedElasticity3DSolution(
            definition.skeleton,
            definition.family,
            tuple(meshes),
            tuple(stresses),
            tuple(displacements),
            tuple(rotations),
            solution,
            definition.source,
            definition.order,
        )
    if isinstance(definition.family, BDMFamily):
        return MixedElasticitySolution(
            *args, definition.source, definition.order, definition.family
        )
    return TensorRTElasticitySolution(
        *args, *definition.family, definition.source, definition.order
    )
