"""H(div)-conforming MHM elasticity with weak stress symmetry in two dimensions.

Each stress row has normal degree k and BDM(k+n) interior modes; displacement
and scalar rotation are discontinuous P(k+n-1). The default is BDM2/P1/P1.
The stress is the Cauchy stress, while the skeletal multiplier is its negative
normal traction. Rotation uses ``q=(du_x/dy-du_y/dx)/2`` and symmetry is imposed
through displacement-space moments of ``sigma_xy-sigma_yx``, not pointwise equality.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.bdm_family import BDMFamily
from pymhm.elasticity import _boundary_volume_flux
from pymhm.elasticity_compatibility import require_compatible_displacement_flux
from pymhm.elasticity_compliance import compliance_products
from pymhm.elements import boundary_data, scalar_values, triangle_quadrature, vector_values
from pymhm.hybrid import HybridSolution, HybridSystem, LocalProblem
from pymhm.lagrange import multiindices, reference_basis
from pymhm.mesh import FaceSpace, FloatArray, SkeletonSpace, TriangleMesh, positive_int

_DEFAULT_FAMILY = BDMFamily()


def _scatter(block: FloatArray, rows: Any, columns: Any, shape: tuple[int, int]) -> Any:
    """Assemble rectangular cell blocks with independently supplied DOF maps."""
    return sparse.coo_matrix(
        (
            block.ravel(),
            (
                np.repeat(rows, block.shape[2], axis=1).ravel(),
                np.tile(columns, (1, block.shape[1])).ravel(),
            ),
        ),
        shape=shape,
    ).tocsc()


def _operators(
    mesh: TriangleMesh,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    family: BDMFamily = _DEFAULT_FAMILY,
    compliance: Any = None,
) -> tuple[Any, Any, Any, FloatArray]:
    """Assemble compliance, divergence, asymmetry and body-force moments."""
    bary, weights = triangle_quadrature(order)
    values, divergence = family.basis(mesh, bary)
    scalar_basis = reference_basis(family.polynomial_degree - 1, bary)[0]
    scalar_size = scalar_basis.shape[1]
    local_size = family.local_size
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    tensors = np.zeros((*values.shape[:2], 2 * local_size, 2, 2))
    tensors[:, :, 0::2, 0], tensors[:, :, 1::2, 1] = values, values
    if compliance is None:
        mu = scalar_values(lame_mu, points.reshape(-1, 2)).reshape(points.shape[:2])
        bulk = _bulk_compliance(lame_lambda, mu.ravel(), points.reshape(-1, 2)).reshape(mu.shape)
        trace = np.trace(tensors, axis1=-2, axis2=-1)
        deviator = tensors - trace[..., None, None] * np.eye(2) / 2
        # The split avoids cancellation near the incompressible limit.
        mass = np.einsum(
            "q,tq,tqiab,tqjab,t->tij", weights, 1 / (2 * mu), deviator, deviator, mesh.areas
        )
        mass += np.einsum("q,tq,tqi,tqj,t->tij", weights, bulk / 2, trace, trace, mesh.areas)
    else:
        products, _, _ = compliance_products(compliance, points, tensors)
        mass = np.einsum("q,tqij,t->tij", weights, products, mesh.areas)
    div = np.zeros((len(mesh.cells), 2 * scalar_size, 2 * local_size))
    scalar_div = np.einsum("q,qi,tqj,t->tij", weights, scalar_basis, divergence, mesh.areas)
    div[:, 0::2, 0::2], div[:, 1::2, 1::2] = scalar_div, scalar_div
    asym = np.einsum(
        "q,qi,tqj,t->tij",
        weights,
        scalar_basis,
        tensors[..., 0, 1] - tensors[..., 1, 0],
        mesh.areas,
    )
    force = np.einsum(
        "q,qi,tqa,t->tia",
        weights,
        scalar_basis,
        vector_values(source, points.reshape(-1, 2)).reshape(*points.shape[:2], 2),
        mesh.areas,
    )
    stress_dofs = (2 * family.dofs(mesh)[:, :, None] + np.arange(2)).reshape(
        -1, 2 * family.local_size
    )
    nstress = 2 * family.size(mesh)
    nc = len(mesh.cells)
    return (
        _scatter(mass, stress_dofs, stress_dofs, (nstress, nstress)),
        _scatter(
            div,
            np.arange(2 * scalar_size * nc).reshape(nc, 2 * scalar_size),
            stress_dofs,
            (2 * scalar_size * nc, nstress),
        ),
        _scatter(
            asym,
            np.arange(scalar_size * nc).reshape(nc, scalar_size),
            stress_dofs,
            (scalar_size * nc, nstress),
        ),
        force.ravel(),
    )


def _bulk_compliance(lame_lambda: Any, mu: FloatArray, points: FloatArray) -> FloatArray:
    """Evaluate 1/[2(mu+lambda)], including a zero incompressible compliance."""
    raw = lame_lambda(points) if callable(lame_lambda) else lame_lambda
    if np.iscomplexobj(raw):
        raise ValueError("Lamé lambda must be real")
    lam = np.broadcast_to(np.asarray(raw, dtype=float), (len(points),))
    if np.any(np.isnan(lam)) or np.any(lam < 0) or np.any(mu <= 0):
        raise ValueError("Lamé lambda must be nonnegative and mu strictly positive")
    scale = np.maximum(mu, lam)
    return np.asarray((0.5 / scale) / (1 + np.minimum(mu, lam) / scale))


def _stress_trace_moments(
    mesh: TriangleMesh,
    size: int,
    lame_lambda: Any,
    lame_mu: Any,
    order: int,
    family: BDMFamily = _DEFAULT_FAMILY,
    compliance: Any = None,
) -> tuple[FloatArray, FloatArray, float]:
    """Integrate trace(sigma) and its physical bulk-compliance-weighted moment."""
    bary, weights = triangle_quadrature(order)
    basis, _ = family.basis(mesh, bary)
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
    # Row zero contributes sigma_xx; row one contributes sigma_yy.
    traces = basis.reshape(len(mesh.cells), len(bary), 2 * family.local_size)
    dofs = (2 * family.dofs(mesh)[:, :, None] + np.arange(2)).reshape(-1, 2 * family.local_size)
    plain = np.einsum("q,tqi,t->ti", weights, traces, mesh.areas)
    if compliance is None:
        mu = scalar_values(lame_mu, points)
        bulk = _bulk_compliance(lame_lambda, mu, points).reshape(len(mesh.cells), -1)
        weighted = np.einsum("q,tq,tqi,t->ti", weights, bulk, traces, mesh.areas)
        scale = float(bulk.max())
    else:
        tensors = np.zeros((*basis.shape[:2], 2 * family.local_size, 2, 2))
        tensors[:, :, 0::2, 0], tensors[:, :, 1::2, 1] = basis, basis
        _, weighted_trace, scale = compliance_products(compliance, points, tensors)
        weighted = np.einsum("q,tqi,t->ti", weights, weighted_trace, mesh.areas)
    return (
        np.bincount(dofs.ravel(), weights=plain.ravel(), minlength=size),
        np.bincount(dofs.ravel(), weights=weighted.ravel(), minlength=size),
        scale,
    )


def _traction_map(
    mesh: TriangleMesh,
    cell: int,
    fine: TriangleMesh,
    skeleton: SkeletonSpace,
    family: BDMFamily = _DEFAULT_FAMILY,
) -> FloatArray:
    """Apply the shared BDM normal-moment map to each stress row."""
    return np.asarray(np.kron(family.trace_map(mesh, cell, fine, skeleton), np.eye(2)), dtype=float)


def _rigid_values(points: FloatArray, center: FloatArray) -> FloatArray:
    """Evaluate two translations and one centered rigid rotation."""
    values = np.zeros((*points.shape[:-1], 2, 3))
    values[..., 0, 0], values[..., 1, 1] = 1, 1
    values[..., 0, 2] = -(points[..., 1] - center[1])
    values[..., 1, 2] = points[..., 0] - center[0]
    return values


def _displacement_moments(
    mesh: TriangleMesh, center: FloatArray, family: BDMFamily = _DEFAULT_FAMILY
) -> FloatArray:
    """Integrate discontinuous displacement basis against the three rigid motions."""
    degree = family.polynomial_degree - 1
    bary, weights = triangle_quadrature(degree + 2)
    basis = reference_basis(degree, bary)[0]
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    return np.einsum(
        "q,qi,tqak,t->tiak", weights, basis, _rigid_values(points, center), mesh.areas
    ).reshape(-1, 3)


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


@dataclass(frozen=True)
class MixedElasticitySolution:
    """H(div) Cauchy stress, broken displacement and independent weak rotation.

    Stress coefficients have shape ``(family.size(mesh), 2)`` on each local mesh;
    displacement and rotation have shapes ``(cells,d,2)`` and ``(cells,d)``,
    where d is the scalar dimension of P(k+n-1).
    Rotation approximates half the asymmetry of the exact displacement gradient. The skeletal
    traction is ``-sigma n`` and stress symmetry holds through P(k+n-1) moments.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    stress: tuple[FloatArray, ...]
    displacement: tuple[FloatArray, ...]
    rotation: tuple[FloatArray, ...]
    hybrid: HybridSolution
    source: Any
    quadrature_order: int
    family: BDMFamily = _DEFAULT_FAMILY

    @property
    def displacement_degree(self) -> int:
        """Return the discontinuous displacement and rotation polynomial degree."""
        return self.family.polynomial_degree - 1

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate the broken displacement error against an analytical field."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.displacement, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            values = np.einsum(
                "qi,tia->tqa", reference_basis(self.displacement_degree, bary)[0], field
            )
            error = values - vector_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=2) @ weights))
        return float(np.sqrt(total))

    def stress_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate the full Cauchy stress Frobenius error, including its skew part."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            value = exact(points.reshape(-1, 2)) if callable(exact) else exact
            target = np.broadcast_to(np.asarray(value), (points.shape[0] * points.shape[1], 2, 2))
            if np.iscomplexobj(target) or not np.isfinite(target).all():
                raise ValueError("exact stress must be real and finite")
            values, _ = self.family.evaluate(mesh, field, bary)
            error = values - target.reshape(values.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=(2, 3)) @ weights))
        return float(np.sqrt(total))

    def divergence_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate stress-divergence error, distinct from integrated force moments."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = self.family.evaluate(mesh, field, bary)
            target = vector_values(exact, points.reshape(-1, 2)).reshape(divergence.shape)
            total += float(mesh.areas @ (np.sum((divergence - target) ** 2, axis=2) @ weights))
        return float(np.sqrt(total))

    def rotation_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate rotation error with the convention q=(du_x/dy-du_y/dx)/2."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, field in zip(self.local_meshes, self.rotation, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            values = field @ reference_basis(self.displacement_degree, bary)[0].T
            error = values - scalar_values(exact, points.reshape(-1, 2)).reshape(values.shape)
            total += float(mesh.areas @ (error**2 @ weights))
        return float(np.sqrt(total))

    def fine_force_residuals(self) -> tuple[FloatArray, ...]:
        """Return displacement-space moments of div(sigma)+f in every fine cell and component."""
        bary, weights = triangle_quadrature(self.quadrature_order)
        residuals = []
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            _, divergence = self.family.evaluate(mesh, field, bary)
            force = vector_values(self.source, points.reshape(-1, 2)).reshape(divergence.shape)
            residuals.append(
                np.einsum(
                    "q,qi,tqa,t->tia",
                    weights,
                    reference_basis(self.displacement_degree, bary)[0],
                    divergence + force,
                    mesh.areas,
                )
            )
        return tuple(residuals)

    def weak_symmetry_residuals(self) -> tuple[FloatArray, ...]:
        """Return rotation-space moments of sigma_xy-sigma_yx, without symmetrizing the field."""
        bary, weights = triangle_quadrature(self.family.polynomial_degree + 2)
        result = []
        for mesh, field in zip(self.local_meshes, self.stress, strict=True):
            values, _ = self.family.evaluate(mesh, field, bary)
            result.append(
                np.einsum(
                    "q,qi,tq,t->ti",
                    weights,
                    reference_basis(self.displacement_degree, bary)[0],
                    values[..., 0, 1] - values[..., 1, 0],
                    mesh.areas,
                )
            )
        return tuple(result)

    def normal_traction_residuals(self) -> tuple[FloatArray, ...]:
        """Return moments of sigma n plus the signed macro traction on fine faces."""
        residuals = []
        for cell, (mesh, stress) in enumerate(zip(self.local_meshes, self.stress, strict=True)):
            count = self.family.degree + 1
            rows = (count * mesh.boundary_faces[:, None] + np.arange(count)).ravel()
            expected = (
                _traction_map(self.skeleton.mesh, cell, mesh, self.skeleton, self.family)
                @ (self.hybrid.trace[self.skeleton.cell_dofs(cell)])
            )
            residuals.append(stress[rows] + expected.reshape(-1, 2))
        return tuple(residuals)

    def equilibrium_residuals(self) -> FloatArray:
        """Return macro force and moment defects for the negative-traction skeleton."""
        bary, weights = triangle_quadrature(self.quadrature_order)
        result = np.zeros((len(self.local_meshes), 3))
        coarse = self.skeleton.mesh
        for cell, fine in enumerate(self.local_meshes):
            center = coarse.points[coarse.cells[cell]].mean(axis=0)
            points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
            force = vector_values(self.source, points.reshape(-1, 2)).reshape(points.shape)
            result[cell] -= np.einsum(
                "t,q,tqa,tqak->k", fine.areas, weights, force, _rigid_values(points, center)
            )
            for side, face in enumerate(coarse.cell_faces[cell]):
                space = self.skeleton.faces[face]
                parameter, w = space.quadrature(max(4, self.family.degree + 2))
                start, end = coarse.points[coarse.faces[face]]
                points_face = start + parameter[:, None] * (end - start)
                traction = space.evaluate(parameter) @ self.hybrid.trace[
                    self.skeleton.dofs(int(face))
                ].reshape(-1, 2)
                result[cell] += (
                    coarse.signs[cell, side]
                    * coarse.lengths[face]
                    * np.einsum("q,qa,qak->k", w, traction, _rigid_values(points_face, center))
                )
        return result


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
