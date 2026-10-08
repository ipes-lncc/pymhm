"""Mixed displacement-pressure MHM for nearly incompressible plane-strain elasticity.

The GaLS operator follows Gomes, Pereira and Valentin, arXiv:2403.16890,
Eqs. (4.5)--(4.9). Pressure is Herrmann pressure ``p=-lambda*div(u)`` and
stress is ``2*mu*sym(grad(u))-p*I``. All local rigid motions are retained.
"""

from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import trace_coupling
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.physical import boundary_normal_integral as _boundary_volume_flux
from pymhm.fem.vector.compatibility import require_strain_trace_compatibility
from pymhm.fem.vector.elasticity import rigid_modes as _compat_rigid
from pymhm.fem.vector.elasticity import strain_and_divergence as _compat_strain_and_divergence
from pymhm.fem.vector.pressure import strain_inverse_bound as _compat_inverse_constant
from pymhm.fem.vector.pressure import triangle_elasticity_pressure_operators
from pymhm.materials.compliance import compressibility_values as _compressibility
from pymhm.materials.evaluation import scalar_values, vector_values
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import ElasticitySolution as ElasticitySolution


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
    forms = triangle_elasticity_pressure_operators(
        fine,
        degree=degree,
        formulation=formulation,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        order=order,
        diameter=float(max(mesh.lengths[mesh.cell_faces[cell]])),
        rigid_center=mesh.points.mean(axis=0),
        stabilization_alpha=stabilization_alpha,
        lame_mu_gradient=lame_mu_gradient,
        shear_bounds=shear_bounds,
    )
    nv, size = len(forms.displacement_nodes), len(forms.load)
    coupling = np.zeros((size, len(skeleton.cell_dofs(cell))))
    coupling[: 2 * nv] = np.kron(trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(2))
    problem = LocalProblem(
        forms.matrix,
        coupling,
        forms.load,
        skeleton.cell_dofs(cell),
        forms.kernel,
        forms.rigid_moments,
    )
    return LocalAssembly(
        problem,
        (
            fine,
            nv,
            forms.pressure_moments,
            forms.stabilization_alpha,
            forms.rigid_moments,
            forms.compliance_moments,
            forms.compliance_scale,
        ),
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


_rigid = _compat_rigid
_strain_and_divergence = _compat_strain_and_divergence
_inverse_constant = _compat_inverse_constant

_check_trace_refinement = require_strain_trace_compatibility
