"""Three-dimensional GaLS/Taylor-Hood displacement-pressure MHM elasticity."""

from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.elasticity.pressure_forms_3d import (
    elasticity_contract_3d,
    tetra_elasticity_pressure_operators,
)
from pymhm._legacy.models.elasticity.primal_3d import _boundary_vector
from pymhm.core.contracts import LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray as FloatArray
from pymhm.core.validation import positive_int as positive_int
from pymhm.core.validation import real_array as _real
from pymhm.fem.traces.physical import boundary_normal_integral
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.meshes.tetrahedron import TetraMesh, _dyadic
from pymhm.postprocessing.solutions import GaLS3DSolution as GaLS3DSolution


def _local_elasticity_pressure(
    cell: int,
    *,
    mesh: TetraMesh,
    skeleton: TriangularSkeleton,
    refinement: int,
    degree: int,
    options: dict[str, Any],
) -> LocalAssembly:
    """Build physical mixed elasticity and the six-moment local Neumann constraints."""
    fine = mesh.submesh(cell, refinement)
    vertices = mesh.points[mesh.cells[cell]]
    diameter = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=-1).max()
    center = mesh.volumes @ mesh.points[mesh.cells].mean(axis=1) / mesh.volumes.sum()
    forms = tetra_elasticity_pressure_operators(
        fine, degree=degree, macro_diameter=float(diameter), rigid_center=center, **options
    )
    nv, size = len(forms.displacement_nodes), len(forms.load)
    coupling = np.zeros((size, 3 * len(skeleton.cell_dofs(cell))))
    coupling[: 3 * nv] = np.kron(
        tetra_trace_coupling(mesh, cell, fine, skeleton, degree), np.eye(3)
    )
    indices = (3 * skeleton.cell_dofs(cell)[:, None] + np.arange(3)).ravel()
    return LocalAssembly(
        LocalProblem(
            forms.matrix, coupling, forms.load, indices, forms.kernel, forms.rigid_moments
        ),
        (fine, forms),
    )


def solve_elasticity_gals_3d(
    mesh: TetraMesh,
    *,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0, 0.0),
    dirichlet: Any = (0.0, 0.0, 0.0),
    traction: dict[int, Any] | None = None,
    skeleton: TriangularSkeleton | None = None,
    degree: int = 1,
    local_refinement: int | None = None,
    quadrature_order: int = 6,
    formulation: str = "gals",
    stabilization_alpha: float | None = None,
    lame_mu_gradient: Any = None,
    shear_bounds: tuple[float, float, float] | None = None,
    mean_pressure: float = 0.0,
    rigid_moments: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> GaLS3DSolution:
    """Solve 3D isotropic mixed elasticity with six rigid modes and physical compressibility.

    GaLS uses Pk/Pk for k=1--4; Taylor-Hood uses Pk/P(k-1), k=2--4.
    ``lambda=inf`` means exact incompressibility. Variable shear requires its
    analytical gradient and certified lower/upper/gradient bounds for GaLS.
    The stabilization bound is computed from a physical tetrahedral inverse
    inequality, as in the d=3 formulation of Gomes et al., arXiv:2403.16890.

    Face traces are P1 on matching dyadic subdivisions, interleaved Cartesian
    negative Cauchy-traction components. Every unspecified boundary face has
    weak displacement data; ``traction`` prescribes physical outward sigma*n.
    For a full displacement boundary, finite lambda uses the integrated identity
    integral(p/lambda)=-integral(g.n), never an arbitrary pressure mean. Only
    at lambda infinity is ``mean_pressure`` a gauge. Pure traction imposes six
    integrated displacement moments about the volume centroid, ordered as
    translations followed by e_i cross (x-centroid). A mixed boundary with some
    fully prescribed displacement faces removes all six rigid modes.

    Default local refinement is four for P1 and two for higher degrees.
    Numerical rank checks reject invisible trace modes; this is not a proof of
    inf-sup stability for every tetrahedral mesh. Material jumps require fitted
    local cells. Assembly and condensation both run in the selected backend.
    """
    gradient, bounds = elasticity_contract_3d(
        lame_lambda,
        lame_mu,
        lame_mu_gradient,
        shear_bounds,
        formulation,
        degree,
        stabilization_alpha,
        mesh.points,
    )
    refinement = _dyadic(
        (4 if degree == 1 else 2) if local_refinement is None else local_refinement,
        "local_refinement",
    )
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    skeleton = TriangularSkeleton(mesh, degree=1) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or np.any(skeleton.degrees != 1):
        raise ValueError("elasticity requires P1 face traces on the supplied mesh")
    if np.any(skeleton.subdivisions > refinement):
        raise ValueError("local refinement must resolve every face subdivision")
    mean = _real(mean_pressure, "mean_pressure")
    if mean.ndim != 0:
        raise ValueError("mean_pressure must be a scalar")
    traction = {} if traction is None else traction
    if traction and mean != 0:
        raise ValueError("mean_pressure is only a gauge for a full displacement boundary")
    boundary, fixed = _boundary_vector(skeleton, dirichlet, traction, order)
    options = dict(
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        lame_mu_gradient=gradient,
        shear_bounds=bounds,
        source=source,
        formulation=formulation,
        stabilization_alpha=stabilization_alpha,
        order=order,
    )
    system = HybridSystem.from_local_factory(
        partial(
            _local_elasticity_pressure,
            mesh=mesh,
            skeleton=skeleton,
            refinement=refinement,
            degree=degree,
            options=options,
        ),
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    compliance_scale = max(data[1].compliance_scale for data in system.local_metadata)
    if mean != 0 and compliance_scale > 0:
        raise ValueError("finite lambda determines pressure through compressibility")
    gauges = []
    if not traction:
        flux = _boundary_volume_flux(skeleton, boundary)
        if compliance_scale == 0:
            require_compatible_displacement_flux(
                flux, _boundary_volume_flux(skeleton, boundary, absolute=True)
            )
            moments = [data[1].pressure_moments for data in system.local_metadata]
            target = float(mean) * mesh.volumes.sum()
        else:
            moments = [
                data[1].compliance_moments / compliance_scale for data in system.local_metadata
            ]
            target = -flux / compliance_scale
        gauges.append(system.mean_constraint(moments, target))
    if set(traction) == set(mesh.boundary_faces):
        values = _real(rigid_moments, "rigid_moments")
        if values.shape != (6,):
            raise ValueError("rigid_moments needs six displacement integrals")
        gauges.extend(
            system.mean_constraint(
                [data[1].rigid_moments[:, i] for data in system.local_metadata], values[i]
            )
            for i in range(6)
        )
    result = system.solve(solver=solver, fixed=fixed, constraints=gauges)
    sizes = [len(data[1].displacement_nodes) for data in system.local_metadata]
    return GaLS3DSolution(
        skeleton,
        tuple(data[0] for data in system.local_metadata),
        tuple(field[: 3 * n].reshape(-1, 3) for field, n in zip(result.fields, sizes, strict=True)),
        tuple(field[3 * n :] for field, n in zip(result.fields, sizes, strict=True)),
        result,
        degree,
        degree if formulation == "gals" else degree - 1,
        lame_lambda,
        lame_mu,
        formulation,
        tuple(data[1].stabilization_alpha for data in system.local_metadata),
    )


_boundary_volume_flux = boundary_normal_integral
