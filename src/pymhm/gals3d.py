"""Three-dimensional GaLS/Taylor-Hood displacement-pressure MHM elasticity."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import numpy as np

from pymhm.darcy3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.elasticity import _compressibility
from pymhm.elasticity3d import _boundary_vector
from pymhm.elasticity_compatibility import require_compatible_displacement_flux
from pymhm.gals3d_forms import elasticity_contract_3d, tetra_elasticity_pressure_operators
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mesh import FloatArray, positive_int
from pymhm.rad3d import vector_values_3d
from pymhm.tetrahedral import (
    TetraMesh,
    _dyadic,
    _real,
    scalar_values_3d,
    tetra_tabulate,
    tetrahedron_quadrature,
)


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


@dataclass(frozen=True)
class GaLS3DSolution:
    """Displacement, Herrmann pressure and raw symmetric Cauchy stress in three dimensions."""

    skeleton: TriangularSkeleton
    local_meshes: tuple[TetraMesh, ...]
    displacement: tuple[FloatArray, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    pressure_degree: int
    lame_lambda: Any
    lame_mu: Any
    formulation: str
    stabilization: tuple[float, ...]

    def _index(self, cell: int) -> int:
        """Check a macrocell index before local array access."""
        index = positive_int(cell, "cell", 0)
        if index >= len(self.local_meshes):
            raise ValueError("cell outside local meshes")
        return index

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one-sided displacement and pressure on the local tetrahedra."""
        cell = self._index(cell)
        fine = self.local_meshes[cell]
        dofs, _, basis, _ = tetra_tabulate(fine, self.degree, bary)
        pdofs, _, pbasis, _ = tetra_tabulate(fine, self.pressure_degree, bary)
        return np.einsum("qi,tia->tqa", basis, self.displacement[cell][dofs]), self.pressure[cell][
            pdofs
        ] @ pbasis.T

    def gradient(self, cell: int, bary: FloatArray) -> FloatArray:
        """Return the broken displacement gradient indexed component then derivative."""
        cell = self._index(cell)
        dofs, _, _, derivative = tetra_tabulate(self.local_meshes[cell], self.degree, bary)
        return np.einsum("tia,tqib->tqab", self.displacement[cell][dofs], derivative)

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Return 2*mu*epsilon(u)-p*I without multiplying lambda by a small divergence."""
        gradient = self.gradient(cell, bary)
        pressure = self.evaluate(cell, bary)[1]
        fine = self.local_meshes[cell]
        points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
        mu = scalar_values_3d(self.lame_mu, points.reshape(-1, 3)).reshape(pressure.shape)
        return mu[..., None, None] * (gradient + gradient.swapaxes(-1, -2)) - pressure[
            ..., None, None
        ] * np.eye(3)

    def _error(self, exact: Any, kind: str, order: int) -> float:
        """Integrate a physical scalar, vector or tensor error without interface averaging."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(-1, 3)
            if kind == "displacement":
                value = self.evaluate(cell, bary)[0]
                target = vector_values_3d(exact, points).reshape(value.shape)
                error = np.sum((value - target) ** 2, axis=-1)
            elif kind == "pressure":
                value = self.evaluate(cell, bary)[1]
                target = scalar_values_3d(exact, points).reshape(value.shape)
                error = (value - target) ** 2
            else:
                value = self.stress(cell, bary) if kind == "stress" else self.gradient(cell, bary)
                target = _real(exact(points) if callable(exact) else exact, "exact tensor")
                target = np.broadcast_to(target, (len(points), 3, 3)).reshape(value.shape)
                error = np.sum((value - target) ** 2, axis=(-1, -2))
            total += float(fine.volumes @ (error @ weights))
        return float(np.sqrt(total))

    def l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate the displacement L2 error."""
        return self._error(exact, "displacement", order)

    def pressure_l2_error(self, exact: Any, order: int = 7) -> float:
        """Integrate physical Herrmann-pressure error with the imposed mean convention."""
        return self._error(exact, "pressure", order)

    def h1_seminorm_error(self, exact_gradient: Any, order: int = 7) -> float:
        """Integrate the full broken displacement-gradient error."""
        return self._error(exact_gradient, "gradient", order)

    def stress_l2_error(self, exact_stress: Any, order: int = 7) -> float:
        """Integrate the complete symmetric Cauchy-stress Frobenius error."""
        return self._error(exact_stress, "stress", order)

    def compressibility_l2(self, order: int = 7) -> float:
        """Measure div(u)+p/lambda in physical L2, including zero inverse lambda at infinity."""
        bary, weights = tetrahedron_quadrature(order)
        total = 0.0
        for cell, fine in enumerate(self.local_meshes):
            points = np.einsum("qi,tia->tqa", bary, fine.points[fine.cells])
            compliance = _compressibility(self.lame_lambda, points.reshape(-1, 3)).reshape(
                points.shape[:2]
            )
            residual = (
                np.trace(self.gradient(cell, bary), axis1=-2, axis2=-1)
                + self.evaluate(cell, bary)[1] * compliance
            )
            total += float(fine.volumes @ (residual**2 @ weights))
        return float(np.sqrt(total))


def _boundary_volume_flux(
    skeleton: TriangularSkeleton, boundary: FloatArray, *, absolute: bool = False
) -> float:
    """Apply the hydrostatic trace to the same assembled displacement boundary moments."""
    load = boundary.reshape(-1, 3).astype(np.longdouble)
    total = np.longdouble(0)
    for face in skeleton.mesh.boundary_faces:
        terms = load[skeleton.dofs(int(face))] * skeleton.mesh.normals[face].astype(np.longdouble)
        total += np.sum(abs(terms) if absolute else terms)
    return float(total)


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
