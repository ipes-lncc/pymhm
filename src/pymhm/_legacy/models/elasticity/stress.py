"""H(div)-conforming MHM elasticity with weak stress symmetry in two dimensions.

Each stress row has normal degree k and BDM(k+n) interior modes; displacement
and scalar rotation are discontinuous P(k+n-1). The default is BDM2/P1/P1.
The stress is the Cauchy stress, while the skeletal multiplier is its negative
normal traction. Rotation uses ``q=(du_x/dy-du_y/dx)/2`` and symmetry is imposed
through displacement-space moments of ``sigma_xy-sigma_yx``, not pointwise equality.
"""

from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.elasticity.mixed_pressure import _boundary_volume_flux
from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.assembly import assemble_element_blocks
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import multiindices
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.stress import bulk_compliance
from pymhm.fem.vector.stress import displacement_rigid_moments as _displacement_moments
from pymhm.fem.vector.stress import mixed_elasticity_operators as _operators
from pymhm.fem.vector.stress import rigid_values as _rigid_values
from pymhm.fem.vector.stress import stress_trace_moments as _stress_trace_moments
from pymhm.fem.vector.stress import traction_mapping as _traction_map
from pymhm.materials.evaluation import vector_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.stress import MixedElasticitySolution as MixedElasticitySolution

_DEFAULT_FAMILY = BDMFamily()


def _local_problem(
    coarse: TriangleMesh,
    cell: int,
    fine: TriangleMesh,
    skeleton: SkeletonSpace,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    family: BDMFamily = _DEFAULT_FAMILY,
    compliance: Any = None,
) -> LocalProblem:
    """Create the constrained Neumann problem and its three exact rigid modes."""
    mass, divergence, asymmetry, force = _operators(
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
    kernel[ns : ns + nu] = _rigid_values(nodes, center).reshape(-1, 3)
    kernel[ns + nu : ns + nu + nr, 2] = -1
    boundary = _rigid_values(fine.points[fine.faces[fine.boundary_faces]], center)
    boundary_coefficients = np.zeros((len(boundary), family.degree + 1, 2, 3))
    boundary_coefficients[:, 0] = boundary.mean(axis=1)
    boundary_coefficients[:, 1] = (boundary[:, 1] - boundary[:, 0]) / 2
    kernel[ns + nu + nr :] = boundary_coefficients.reshape(-1, 3)
    constraints = np.zeros_like(kernel)
    constraints[ns : ns + nu] = _displacement_moments(fine, center, family)
    mapping = _traction_map(coarse, cell, fine, skeleton, family)
    coupling = np.zeros((matrix.shape[0], mapping.shape[1]))
    coupling[-nb:] = -mapping
    return LocalProblem(
        matrix,
        coupling,
        np.r_[np.zeros(ns), -force, np.zeros(nr + nb)],
        skeleton.cell_dofs(cell),
        kernel,
        constraints,
    )


def solve_elasticity_mixed(
    mesh: TriangleMesh,
    *,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    compliance: Any = None,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    traction: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 2,
    stress_degree: int = 2,
    enrichment: int = 0,
    quadrature_order: int = 4,
    rigid_moments: Any = (0.0, 0.0, 0.0),
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MixedElasticitySolution:
    """Solve plane-strain elasticity using the BDM_k, BDM_k+ or BDM_k++ mixed family.

    The equations are ``-div(sigma)=f`` and ``sigma=2 mu eps(u)+lambda div(u) I``.
    Traction entries prescribe physical outward Cauchy traction ``sigma n``;
    other exterior faces prescribe displacement weakly. Lamé coefficients may
    be scalar fields, with finite mu>0 and lambda>=0, including infinity.
    Alternatively, ``compliance`` supplies a self-adjoint positive full-tensor
    Cartesian operator A, with coordinates (xx,xy,yx,yy), preserving symmetric
    tensors. Its explicit skew extension is required. This replaces the Lame
    material law and uses integral(tr(A sigma))=integral(g.n) for full
    displacement boundaries. See ``stress_compliance_values`` for the contract.
    With fully prescribed displacement and isotropic material, the exact identity
    integral(trace(sigma)/[2(mu+lambda)])=integral(g.n) fixes the finite-modulus
    hydrostatic stress. At infinite lambda, ``mean_pressure`` prescribes the
    global mean of -trace(sigma)/2 and the boundary volume flux must vanish.

    ``stress_degree=k`` is the normal degree; ``enrichment=n`` (0, 1 or 2)
    retains all interior bubbles of BDM(k+n). Displacement and weak rotation
    have degree k+n-1, which must be at least one to contain rigid rotations.
    The default k=2,n=0 preserves BDM2/P1/P1. The default skeleton has P1
    traction on interior macrofaces and independent Pk traction on every fine
    exterior face. Custom degrees up to k require aligned subface breaks; discrete rank checks
    remain necessary for arbitrary trace choices. Pure traction problems use
    three integrated displacement moments against (1,0), (0,1), and the rigid
    rotation about the domain centroid, supplied through ``rigid_moments``.
    """
    family = BDMFamily(stress_degree, enrichment)
    if family.polynomial_degree < 2:
        raise ValueError(
            "mixed MHM elasticity requires displacement degree >=1 for rigid rotations"
        )
    scalar_size = family.polynomial_degree * (family.polynomial_degree + 1) // 2
    refinement = positive_int(local_refinement, "local_refinement")
    positive_int(quadrature_order, "quadrature_order", family.polynomial_degree + 1)
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be a finite gauge value")
    if skeleton is None:
        boundary = set(mesh.boundary_faces)
        skeleton = SkeletonSpace(
            mesh,
            tuple(
                FaceSpace.uniform(family.degree, refinement)
                if f in boundary
                else FaceSpace.uniform(1)
                for f in range(len(mesh.faces))
            ),
            components=2,
        )
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("mixed elasticity requires a two-component skeleton on this mesh")
    family.validate_trace(skeleton, refinement)
    data = {} if traction is None else traction
    negative_traction = {
        face: (lambda points, datum=value: -vector_values(datum, points))
        for face, value in data.items()
    }
    if data and mean_pressure != 0:
        raise ValueError("mean_pressure is only a gauge for full displacement boundary data")
    boundary_load, fixed = boundary_data(
        skeleton, dirichlet, negative_traction, order=max(5, quadrature_order)
    )
    local_meshes = tuple(mesh.submesh(cell, refinement) for cell in range(len(mesh.cells)))
    problems = tuple(
        _local_problem(
            mesh,
            cell,
            fine,
            skeleton,
            lame_lambda,
            lame_mu,
            source,
            quadrature_order,
            family,
            compliance,
        )
        for cell, fine in enumerate(local_meshes)
    )
    system = HybridSystem(
        problems,
        boundary_load=-boundary_load,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = []
    if not data:
        hydrostatic_moments = [
            _stress_trace_moments(
                fine, len(problem.load), lame_lambda, lame_mu, quadrature_order, family, compliance
            )
            for fine, problem in zip(local_meshes, problems, strict=True)
        ]
        scale = max(moment[2] for moment in hydrostatic_moments)
        volume_flux = _boundary_volume_flux(skeleton, boundary_load)
        if scale == 0:
            require_compatible_displacement_flux(
                volume_flux, _boundary_volume_flux(skeleton, boundary_load, absolute=True)
            )
            gauges.append(
                system.mean_constraint(
                    [moment[0] for moment in hydrostatic_moments],
                    -2 * mean_pressure * sum(mesh.areas),
                )
            )
        else:
            if mean_pressure != 0:
                raise ValueError("mean_pressure is only a gauge in the incompressible limit")
            gauges.append(
                system.mean_constraint(
                    [moment[1] / scale for moment in hydrostatic_moments], volume_flux / scale
                )
            )
    if set(data) == set(mesh.boundary_faces):
        if np.iscomplexobj(rigid_moments):
            raise ValueError("rigid_moments must contain three finite real integrated moments")
        targets = np.asarray(rigid_moments, dtype=float)
        if targets.shape != (3,) or not np.isfinite(targets).all():
            raise ValueError("rigid_moments must contain three finite integrated moments")
        center = np.asarray(
            sum(fine.areas @ fine.points[fine.cells].mean(axis=1) for fine in local_meshes)
            / mesh.areas.sum()
        )
        moments = []
        for fine, problem in zip(local_meshes, problems, strict=True):
            ns = 2 * family.size(fine)
            local_weights = np.zeros((len(problem.load), 3))
            local_weights[ns : ns + 2 * scalar_size * len(fine.cells)] = _displacement_moments(
                fine, center, family
            )
            moments.append(local_weights)
        gauges.extend(
            system.mean_constraint([m[:, i] for m in moments], targets[i]) for i in range(3)
        )
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    stresses, displacements, rotations = [], [], []
    for fine, field in zip(local_meshes, hybrid.fields, strict=True):
        ns, nu, nr = (
            2 * family.size(fine),
            2 * scalar_size * len(fine.cells),
            scalar_size * len(fine.cells),
        )
        stresses.append(field[:ns].reshape(-1, 2))
        displacements.append(field[ns : ns + nu].reshape(-1, scalar_size, 2))
        rotations.append(field[ns + nu : ns + nu + nr].reshape(-1, scalar_size))
    return MixedElasticitySolution(
        skeleton,
        local_meshes,
        tuple(stresses),
        tuple(displacements),
        tuple(rotations),
        hybrid,
        source,
        quadrature_order,
        family,
    )


_bulk_compliance = bulk_compliance

_scatter = assemble_element_blocks
