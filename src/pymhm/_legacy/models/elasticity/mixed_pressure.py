"""Mixed displacement-pressure MHM for nearly incompressible plane-strain elasticity.

The GaLS operator follows Gomes, Pereira and Valentin, arXiv:2403.16890,
Eqs. (4.5)--(4.9). Pressure is Herrmann pressure ``p=-lambda*div(u)`` and
stress is ``2*mu*sym(grad(u))-p*I``. All local rigid motions are retained.
"""

from dataclasses import dataclass
from functools import partial
from math import fsum
from typing import Any, Literal

import numpy as np
from scipy import linalg

from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.vector import VectorSolution
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import boundary_data, triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate, trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class ElasticitySolution(VectorSolution):
    """Displacement, Herrmann pressure and physical Cauchy-stress reconstruction."""

    lame_lambda: Any = 1.0
    lame_mu: Any = 1.0
    formulation: str = "gals"
    stabilization: tuple[float, ...] = ()

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate the broken displacement gradient, indexed fine cell/point/i/j."""
        dofs, _, _, derivative, _ = tabulate(self.local_meshes[cell], self.degree, bary)
        return np.einsum("tqij,tia->tqaj", derivative, self.values[cell][dofs])

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate symmetric Cauchy stress using pressure without lambda cancellation."""
        derivative = self.gradient(cell, bary)
        dofs, _, basis, _, _ = tabulate(self.local_meshes[cell], self.pressure_degree, bary)
        pressure = self.pressure[cell][dofs] @ basis.T
        mesh = self.local_meshes[cell]
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        mu = scalar_values(self.lame_mu, points.reshape(-1, 2)).reshape(pressure.shape)
        return mu[..., None, None] * (derivative + derivative.swapaxes(-1, -2)) - (
            pressure[..., None, None] * np.eye(2)
        )

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 6) -> float:
        """Integrate the full broken displacement-gradient error, not the strain norm."""
        return self._tensor_error(exact_gradient, order, stress=False)

    def stress_l2_error(self, exact_stress: Any, order: int = 6) -> float:
        """Integrate the Frobenius error of the full symmetric Cauchy stress."""
        return self._tensor_error(exact_stress, order, stress=True)

    def _tensor_error(self, exact: Any, order: int, *, stress: bool) -> float:
        """Share physical tensor quadrature between gradient and stress diagnostics."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for cell, mesh in enumerate(self.local_meshes):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            data = exact(points.reshape(-1, 2)) if callable(exact) else exact
            target = np.broadcast_to(np.asarray(data), (points.shape[0] * len(bary), 2, 2))
            if not np.isfinite(target).all() or np.iscomplexobj(target):
                raise ValueError("exact tensor must be finite and real")
            value = self.stress(cell, bary) if stress else self.gradient(cell, bary)
            error = value - target.reshape(value.shape)
            total += float(mesh.areas @ (np.sum(error**2, axis=(-1, -2)) @ weights))
        return float(np.sqrt(total))

    def compressibility_l2(self, order: int = 6) -> float:
        """Measure ``div(u)+p/lambda`` without asserting pointwise satisfaction."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for cell, mesh in enumerate(self.local_meshes):
            divergence = np.trace(self.gradient(cell, bary), axis1=-2, axis2=-1)
            dofs, _, basis, _, _ = tabulate(mesh, self.pressure_degree, bary)
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            compliance = _compressibility(self.lame_lambda, points.reshape(-1, 2)).reshape(
                divergence.shape
            )
            residual = divergence + (self.pressure[cell][dofs] @ basis.T) * compliance
            total += float(mesh.areas @ (residual**2 @ weights))
        return float(np.sqrt(total))


def _compressibility(lame_lambda: Any, points: FloatArray) -> FloatArray:
    """Evaluate inverse first Lame modulus, allowing positive infinity pointwise."""
    raw = lame_lambda(points) if callable(lame_lambda) else lame_lambda
    if np.iscomplexobj(raw):
        raise ValueError("Lame lambda must be real")
    coefficient = np.broadcast_to(np.asarray(raw, dtype=float), (len(points),))
    if np.isnan(coefficient).any() or np.any(coefficient <= 0):
        raise ValueError("Lame lambda must be positive or infinity")
    return 1 / coefficient


def _rigid(points: FloatArray, center: FloatArray) -> FloatArray:
    """Evaluate both translations and the infinitesimal counterclockwise rotation."""
    modes = np.zeros((len(points), 2, 3))
    modes[:, :, :2] = np.eye(2)
    modes[:, 0, 2] = -(points[:, 1] - center[1])
    modes[:, 1, 2] = points[:, 0] - center[0]
    return modes


def _strain_and_divergence(
    gradients: FloatArray, hessian: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tabulate engineering strain, vector divergence and divergence of symmetric strain."""
    nt, nq, ns, _ = gradients.shape
    strain = np.zeros((nt, nq, 3, 2 * ns))
    strain[:, :, 0, 0::2] = gradients[:, :, :, 0]
    strain[:, :, 1, 1::2] = gradients[:, :, :, 1]
    strain[:, :, 2, 0::2] = gradients[:, :, :, 1]
    strain[:, :, 2, 1::2] = gradients[:, :, :, 0]
    laplacian = np.trace(hessian, axis1=-2, axis2=-1)
    strong = np.empty((nt, nq, 2, 2 * ns))
    for a in range(2):
        for c in range(2):
            strong[:, :, a, c::2] = (hessian[:, :, :, a, c] + (a == c) * laplacian) / 2
    return strain, gradients.reshape(nt, nq, 2 * ns), strong


def _inverse_constant(
    strain: FloatArray, strong: FloatArray, weights: FloatArray, h: FloatArray, diameter: float
) -> float:
    """Bound Eq. (4.4) by elementwise generalized eigenvalues modulo rigid motions.

    Taking the minimum with one prevents a refinement-dependent growth of the
    stabilization. No PDE solution or reference error enters this computation.
    """
    energy = np.einsum("q,tqai,a,tqaj->tij", weights, strain, [1, 1, 0.5], strain)
    residual = np.einsum("q,tqai,tqaj->tij", weights, strong, strong)
    maximum = 1.0
    for mass, second, length in zip(energy, residual, h, strict=True):
        eigenvalues, vectors = linalg.eigh(mass)
        selected = eigenvalues > 1e-11 * eigenvalues[-1]
        if np.count_nonzero(selected) != len(eigenvalues) - 3:
            raise ValueError("inverse inequality requires exactly 3 resolved kernel modes")
        whitening = vectors[:, selected] / np.sqrt(eigenvalues[selected])
        operator = length**2 * (mass / diameter**2 + second)
        maximum = max(maximum, float(linalg.eigvalsh(whitening.T @ operator @ whitening)[-1]))
    return 1 / maximum


def _local_displacement_pressure(
    cell: int,
    *,
    mesh: TriangleMesh,
    skeleton: SkeletonSpace,
    degree: int,
    formulation: str,
    refinement: int,
    lame_lambda: Any,
    lame_mu: Any,
    source: Any,
    order: int,
    stabilization_alpha: float | None,
    lame_mu_gradient: Any,
    shear_bounds: tuple[float, float, float] | None,
) -> LocalAssembly:
    """Assemble one local Neumann operator and physical rigid-mode constraints."""
    fine = mesh.submesh(cell, refinement)
    bary, weights = triangle_quadrature(order)
    dofs, nodes, basis, gradients, hessian = tabulate(fine, degree, bary)
    pressure_degree = degree if formulation == "gals" else degree - 1
    pdofs, pnodes, pbasis, pgradients, _ = tabulate(fine, pressure_degree, bary)
    nv, npres = len(nodes), len(pnodes)
    ns, nps = len(basis.T), len(pbasis.T)
    udofs = (2 * dofs[:, :, None] + np.arange(2)).reshape(len(fine.cells), 2 * ns)
    all_dofs = np.column_stack((udofs, 2 * nv + pdofs))
    strain, divergence, strong = _strain_and_divergence(gradients, hessian)
    size = 2 * nv + npres
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    flat = physical.reshape(-1, 2)
    mu = scalar_values(lame_mu, flat).reshape(len(fine.cells), len(bary))
    if np.any(mu <= 0):
        raise ValueError("Lame mu must be positive")
    epsilon = _compressibility(lame_lambda, flat).reshape(mu.shape)
    blocks = np.zeros((len(fine.cells), 2 * ns + nps, 2 * ns + nps))
    blocks[:, : 2 * ns, : 2 * ns] = np.einsum(
        "q,tqai,a,tqaj,tq,t->tij",
        weights,
        strain,
        [2.0, 2.0, 1.0],
        strain,
        mu,
        fine.areas,
    )
    mixed = -np.einsum("q,tqi,qj,t->tij", weights, divergence, pbasis, fine.areas)
    blocks[:, : 2 * ns, 2 * ns :] = mixed
    blocks[:, 2 * ns :, : 2 * ns] = mixed.swapaxes(1, 2)
    blocks[:, 2 * ns :, 2 * ns :] = -np.einsum(
        "q,qi,qj,tq,t->tij", weights, pbasis, pbasis, epsilon, fine.areas
    )
    physical = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
    force = vector_values(source, physical.reshape(-1, 2)).reshape(len(fine.cells), -1, 2)
    element_load = np.zeros((len(fine.cells), 2 * ns + nps))
    element_load[:, : 2 * ns] = np.einsum(
        "q,qi,tqa,t->tia", weights, basis, force, fine.areas
    ).reshape(len(fine.cells), -1)
    alpha = 0.0
    if formulation == "gals":
        h = np.max(fine.lengths[fine.cell_faces], axis=1)
        diameter = max(mesh.lengths[mesh.cell_faces[cell]])
        shear_gradient = vector_values(lame_mu_gradient, flat).reshape(*mu.shape, 2)
        if shear_bounds is None:
            lower, upper, gradient_bound = float(mu.min()), float(mu.max()), 0.0
        else:
            lower, upper, gradient_bound = shear_bounds
            if (
                np.any(mu < lower * (1 - 1e-12))
                or np.any(mu > upper * (1 + 1e-12))
                or np.any(np.linalg.norm(shear_gradient, axis=-1) > gradient_bound * (1 + 1e-12))
            ):
                raise ValueError("shear_bounds do not bound the evaluated material coefficients")
        bound = (
            _inverse_constant(strain, strong, weights, h, diameter)
            * lower
            / (2 * (upper**2 + diameter**2 * gradient_bound**2))
        )
        alpha = bound / 2 if stabilization_alpha is None else stabilization_alpha
        if not np.isfinite(alpha) or not 0 < alpha < bound:
            raise ValueError(f"stabilization_alpha must lie strictly between 0 and {bound:g}")
        stress_divergence = 2 * mu[:, :, None, None] * strong
        stress_divergence[:, :, 0] += (
            2 * strain[:, :, 0] * shear_gradient[:, :, 0, None]
            + strain[:, :, 2] * shear_gradient[:, :, 1, None]
        )
        stress_divergence[:, :, 1] += (
            strain[:, :, 2] * shear_gradient[:, :, 0, None]
            + 2 * strain[:, :, 1] * shear_gradient[:, :, 1, None]
        )
        residual = np.concatenate((stress_divergence, -pgradients.swapaxes(-1, -2)), axis=3)
        blocks -= np.einsum(
            "q,tqai,tqaj,t->tij", weights, residual, residual, alpha * h**2 * fine.areas
        )
        element_load += np.einsum(
            "q,tqai,tqa,t->ti", weights, residual, force, alpha * h**2 * fine.areas
        )
    matrix = _assemble_blocks(blocks, all_dofs, size)
    load = np.bincount(all_dofs.ravel(), weights=element_load.ravel(), minlength=size)
    coupling = np.zeros((size, len(skeleton.cell_dofs(cell))))
    coupling[: 2 * nv] = np.kron(trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(2))
    kernel = np.zeros((size, 3))
    kernel[: 2 * nv] = _rigid(nodes, mesh.points.mean(axis=0)).reshape(2 * nv, 3)
    local_moments = np.einsum(
        "q,qi,tqaj,t->tiaj",
        weights,
        basis,
        _rigid(physical.reshape(-1, 2), mesh.points.mean(axis=0)).reshape(*physical.shape, 3),
        fine.areas,
    ).reshape(len(fine.cells), 2 * ns, 3)
    constraints = np.zeros_like(kernel)
    np.add.at(constraints, udofs, local_moments)
    pressure_weights = np.zeros(size)
    np.add.at(pressure_weights, 2 * nv + pdofs, fine.areas[:, None] * (weights @ pbasis))
    problem = LocalProblem(matrix, coupling, load, skeleton.cell_dofs(cell), kernel, constraints)
    compliance_weights = np.zeros(size)
    np.add.at(
        compliance_weights,
        2 * nv + pdofs,
        np.einsum("q,qi,tq,t->ti", weights, pbasis, epsilon, fine.areas),
    )
    return LocalAssembly(
        problem,
        (fine, nv, pressure_weights, alpha, constraints, compliance_weights, float(epsilon.max())),
    )


def _check_trace_refinement(skeleton: SkeletonSpace, degree: int, refinement: int) -> None:
    """Enforce the two-dimensional matching-mesh Fortin conditions of Lemma 4.5.

    These are sufficient conditions, not a characterization of every stable
    pair. They prevent invisible trace modes before factorization. Continuous
    face subspaces inherit admissibility from their discontinuous containing space.
    """
    for space in skeleton.faces:
        positions = np.asarray(space.breaks) * refinement
        if not np.allclose(positions, np.round(positions), atol=1e-12, rtol=0):
            raise ValueError("GaLS requires trace segments aligned with the local fine mesh")
        for order, intervals in zip(space.degrees, np.diff(np.round(positions)), strict=True):
            if degree >= order + 2:
                required = 1
            elif degree >= order + 1:
                required = 2
            elif degree >= order:
                required = 5 - min(order, 3)
            else:
                raise ValueError("GaLS local degree must not be lower than the trace degree")
            if intervals < required:
                raise ValueError(
                    f"GaLS trace/local compatibility requires at least {required} fine intervals "
                    "per trace segment for these polynomial degrees"
                )


def solve_displacement_pressure(
    mesh: TriangleMesh,
    *,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    degree: int = 1,
    formulation: str = "gals",
    quadrature_order: int = 6,
    stabilization_alpha: float | None = None,
    lame_mu_gradient: Any = None,
    shear_bounds: tuple[float, float, float] | None = None,
    mean_pressure: float = 0.0,
    rigid_moments: Any = (0.0, 0.0, 0.0),
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> ElasticitySolution:
    """Solve elasticity using equal-order GaLS or stable Taylor--Hood local spaces.

    ``degree`` is the displacement degree, positive for GaLS and at least two
    for Taylor--Hood. ``lame_lambda=np.inf`` imposes exact incompressibility.
    The default traction space is piecewise linear. GaLS stabilization satisfies
    the computed inverse-inequality bound from Eq. (4.9); an explicitly requested
    coefficient outside that bound is rejected. Trace/local compatibility must
    hold in addition to local stability.

    ``neumann`` maps boundary face indices to physical outward Cauchy traction.
    On a fully Dirichlet boundary the integrated compressibility identity fixes
    the finite-lambda pressure mean; only at infinity is ``mean_pressure`` a gauge.
    A pure traction problem uses three physical displacement moments about the
    mean macro vertex position, prescribed by ``rigid_moments``. Assembly and
    condensation both execute in the selected local-worker backend.

    Lame coefficients may be callables. Variable shear modulus in GaLS requires
    ``lame_mu_gradient`` and certified ``shear_bounds=(min_mu,max_mu,max_grad_mu)``.
    The full ``div(2*mu*epsilon(u))`` residual includes shear derivatives.
    Bounds must hold on the material regions resolved by the fine mesh; they
    are also checked at assembly quadrature points. For variable lambda the
    global compressibility constraint integrates ``p/lambda``, not ``p``.
    """
    _compressibility(lame_lambda, mesh.points)
    if np.any(scalar_values(lame_mu, mesh.points) <= 0):
        raise ValueError("Lame mu must be finite and positive")
    if (
        callable(lame_mu)
        and formulation == "gals"
        and (lame_mu_gradient is None or shear_bounds is None)
    ):
        raise ValueError("GaLS with variable Lame mu requires its gradient and shear_bounds")
    lame_mu_gradient = (0.0, 0.0) if lame_mu_gradient is None else lame_mu_gradient
    if not callable(lame_mu) and np.any(vector_values(lame_mu_gradient, mesh.points)):
        raise ValueError("a constant Lame mu must have zero gradient")
    if shear_bounds is not None:
        bounds = np.asarray(shear_bounds, dtype=float)
        if (
            bounds.shape != (3,)
            or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= bounds[1]
            or bounds[2] < 0
        ):
            raise ValueError(
                "shear_bounds must be (positive lower, upper, nonnegative gradient bound)"
            )
    positive_int(local_refinement, "local_refinement")
    positive_int(degree, "degree")
    positive_int(quadrature_order, "quadrature_order")
    if formulation not in ("gals", "taylor-hood") or (formulation == "taylor-hood" and degree < 2):
        raise ValueError("formulation must be gals or taylor-hood (degree >= 2)")
    if formulation != "gals" and stabilization_alpha is not None:
        raise ValueError("stabilization_alpha applies only to GaLS")
    if not np.isfinite(mean_pressure):
        raise ValueError("mean_pressure must be a finite gauge value")
    skeleton = (
        SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
        if skeleton is None
        else skeleton
    )
    if skeleton.mesh is not mesh or skeleton.components != 2:
        raise ValueError("elasticity requires a two-component skeleton on this mesh")
    if formulation == "gals":
        _check_trace_refinement(skeleton, degree, local_refinement)
    neumann = {} if neumann is None else neumann
    if neumann and mean_pressure != 0:
        raise ValueError("mean_pressure is only a gauge for a full displacement boundary")
    # Boundary faces use their outward orientation; the multiplier is -sigma*n.
    negative_traction = {
        face: partial(_negative_vector, field=value) for face, value in neumann.items()
    }
    boundary, fixed = boundary_data(
        skeleton, dirichlet, negative_traction, order=max(quadrature_order, degree + 2)
    )
    factory = partial(
        _local_displacement_pressure,
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        formulation=formulation,
        refinement=local_refinement,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=max(quadrature_order, degree + 2),
        stabilization_alpha=stabilization_alpha,
        lame_mu_gradient=lame_mu_gradient,
        shear_bounds=shear_bounds,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    constraints = []
    compliance_scale = max(data[6] for data in system.local_metadata)
    if mean_pressure and compliance_scale > 0:
        raise ValueError("mean_pressure is a finite gauge only in the incompressible limit")
    if not neumann:
        flux = _boundary_volume_flux(skeleton, boundary)
        if compliance_scale == 0:
            require_compatible_displacement_flux(
                flux, _boundary_volume_flux(skeleton, boundary, absolute=True)
            )
            pressure_integral = mean_pressure * sum(mesh.areas)
            integration = [data[2] for data in system.local_metadata]
        else:
            pressure_integral = -flux / compliance_scale
            integration = [data[5] / compliance_scale for data in system.local_metadata]
        constraints.append(system.mean_constraint(integration, pressure_integral))
    if len(neumann) == len(mesh.boundary_faces):
        moments = np.asarray(rigid_moments, dtype=float)
        if moments.shape != (3,) or not np.isfinite(moments).all():
            raise ValueError("rigid_moments must contain three finite displacement integrals")
        for i in range(3):
            constraints.append(
                system.mean_constraint(
                    [data[4][:, i] for data in system.local_metadata], moments[i]
                )
            )
    result = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    sizes = [data[1] for data in system.local_metadata]
    return ElasticitySolution(
        skeleton=skeleton,
        local_meshes=tuple(data[0] for data in system.local_metadata),
        values=tuple(
            field[: 2 * nv].reshape(-1, 2) for field, nv in zip(result.fields, sizes, strict=True)
        ),
        pressure=tuple(field[2 * nv :] for field, nv in zip(result.fields, sizes, strict=True)),
        hybrid=result,
        degree=degree,
        pressure_degree=degree if formulation == "gals" else degree - 1,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        formulation=formulation,
        stabilization=tuple(data[3] for data in system.local_metadata),
    )


def _negative_vector(points: FloatArray, *, field: Any) -> FloatArray:
    """Convert physical Cauchy traction to the signed MHM multiplier convention."""
    return -vector_values(field, points)


def _boundary_volume_flux(
    skeleton: SkeletonSpace, boundary: FloatArray, *, absolute: bool = False
) -> float:
    """Apply the hydrostatic trace to the assembled displacement boundary moments.

    This uses exactly the same quadrature as the global equations; a separate
    integration rule would make the finite-compressibility identity inconsistent.
    Compensated accumulation reduces cancellation between opposite boundaries
    on platforms with or without a wider native real type.
    ``absolute`` sums uncancelled moments for scale-invariant compatibility.
    """
    terms = []
    for face in skeleton.mesh.boundary_faces:
        start, end = skeleton.mesh.points[skeleton.mesh.faces[face]]
        tangent = end - start
        normal = np.array([tangent[1], -tangent[0]]) / skeleton.mesh.lengths[face]
        coefficients = skeleton.faces[face].constant_coefficients()[:, None] * normal
        products = boundary[skeleton.dofs(int(face))] * coefficients.ravel()
        terms.extend(abs(products) if absolute else products)
    return fsum(terms)
