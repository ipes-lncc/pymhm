"""Vector MHM formulations: Taylor-Hood Stokes-Brinkman and primal elasticity."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.elements import (
    scalar_values,
    triangle_quadrature,
    vector_values,
)
from pymhm.hybrid import HybridSolution
from pymhm.lagrange import tabulate
from pymhm.mesh import FloatArray, SkeletonSpace, TriangleMesh


def _lagrange(
    mesh: TriangleMesh, degree: int, bary: FloatArray
) -> tuple[Any, FloatArray, FloatArray, FloatArray]:
    """Tabulate P1 or P2 scalar nodal bases and physical gradients."""
    dofs, points, basis, gradients, _ = tabulate(mesh, degree, bary)
    return dofs, points, basis, gradients


def _p2_coupling(
    coarse: TriangleMesh, cell: int, fine: TriangleMesh, skeleton: SkeletonSpace
) -> FloatArray:
    """Integrate P2 boundary traces using the shared arbitrary-degree assembler."""
    from pymhm.lagrange import trace_coupling

    return trace_coupling(coarse, cell, fine, skeleton, 2)


def _assemble_blocks(blocks: FloatArray, dofs: Any, size: int) -> Any:
    """Scatter square element matrices to a sparse CSC operator."""
    count = dofs.shape[1]
    return sparse.coo_matrix(
        (
            blocks.ravel(),
            (np.repeat(dofs, count, axis=1).ravel(), np.tile(dofs, (1, count)).ravel()),
        ),
        shape=(size, size),
    ).tocsc()


@dataclass(frozen=True)
class VectorSolution:
    """Reconstructed velocity/displacement and optional P1 pressure fields."""

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    values: tuple[FloatArray, ...]
    pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    pressure_degree: int = 1

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate the error in velocity or displacement."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, values in zip(self.local_meshes, self.values, strict=True):
            dofs, _, basis, _ = _lagrange(mesh, self.degree, bary)
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            difference = np.einsum("qi,tia->tqa", basis, values[dofs]) - vector_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights), 2)
            total += float(mesh.areas @ (np.sum(difference**2, axis=2) @ weights))
        return float(np.sqrt(total))

    def pressure_l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate pressure error with the same global gauge as the exact field."""
        if not self.pressure:
            raise ValueError("elasticity solution has no pressure field")
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, pressure in zip(self.local_meshes, self.pressure, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            dofs, _, basis, _ = _lagrange(mesh, self.pressure_degree, bary)
            difference = pressure[dofs] @ basis.T - scalar_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights))
            total += float(mesh.areas @ (difference**2 @ weights))
        return float(np.sqrt(total))

    def divergence_l2(self) -> float:
        """Measure the pointwise divergence norm of the reconstructed vector field."""
        bary, weights = triangle_quadrature(4)
        total = 0.0
        for mesh, values in zip(self.local_meshes, self.values, strict=True):
            dofs, _, _, gradients = _lagrange(mesh, self.degree, bary)
            divergence = np.einsum("tqia,tia->tq", gradients, values[dofs])
            total += float(mesh.areas @ (divergence**2 @ weights))
        return float(np.sqrt(total))


def solve_brinkman(
    mesh: TriangleMesh,
    *,
    viscosity: float = 1.0,
    drag: Any = 0.0,
    advection: Any = (0.0, 0.0),
    advection_divergence: Any = None,
    advection_bound: float | None = None,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    traction: dict[int, Any] | None = None,
    traction_components: dict[int, dict[int, Any]] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int | tuple[int, ...] = 2,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 5,
    formulation: str = "taylor-hood",
    stabilization: Literal["tensor-2025", "minimum-2017", "pointwise-2017"] = "tensor-2025",
    gamma_min: float | None = None,
    degree: int | None = None,
    mean_pressure: float = 0.0,
    mean_velocity: Any = (0.0, 0.0),
    translation_kernel: Any = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> VectorSolution:
    """Solve Stokes--Brinkman with arbitrary-order Taylor--Hood or USFEM locals.

    The default degree is two for Taylor--Hood and one for USFEM. Pressure has
    degree k-1 or k, respectively. The USFEM residual includes all displacement
    Laplacian, drag, pressure and source terms, with an inverse-inequality-based
    stabilization. The multiplier is negative grad-grad pseudotraction, with
    half the advective normal flux for skew-form Oseen problems. See
    :func:`pymhm.flow.solve_flow` for boundary and gauge conventions, including
    Cartesian ``traction_components`` for slip walls and mixed component data.
    """
    from pymhm.flow import solve_flow

    return solve_flow(
        mesh,
        viscosity=viscosity,
        drag=drag,
        advection=advection,
        advection_divergence=advection_divergence,
        advection_bound=advection_bound,
        source=source,
        dirichlet=dirichlet,
        traction=traction,
        traction_components=traction_components,
        skeleton=skeleton,
        local_refinement=local_refinement,
        local_meshes=local_meshes,
        quadrature_order=quadrature_order,
        formulation=formulation,
        stabilization=stabilization,
        gamma_min=gamma_min,
        degree=degree,
        mean_pressure=mean_pressure,
        mean_velocity=mean_velocity,
        translation_kernel=translation_kernel,
        solver=solver,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )


def solve_elasticity(
    mesh: TriangleMesh,
    *,
    lame_lambda: Any = 1.0,
    lame_mu: Any = 1.0,
    source: Any = (0.0, 0.0),
    dirichlet: Any = (0.0, 0.0),
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    degree: int | None = None,
    formulation: str = "gals",
    constitutive: Any = None,
    minimal_enrichment: bool = False,
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
) -> VectorSolution:
    """Solve plane-strain elasticity with an explicit choice of local formulation.

    The default ``gals`` is the stabilized displacement/Herrmann-pressure MHM
    method of Gomes, Pereira and Valentin (2024), with arbitrary positive local
    degree. ``taylor-hood`` uses displacement degree two by default and pressure
    one degree lower. Both retain all three rigid motions and admit infinite
    lambda. Their reconstructed result provides Cauchy stress and pressure.

    ``primal`` uses continuous local Pk displacement and accepts a general SPD
    ``constitutive`` stiffness in Kelvin or fourth-order Cartesian coordinates;
    it is not locking-free. The dedicated
    :func:`pymhm.solve_elasticity_mixed` exposes that formulation's full options.
    Boundary ``neumann`` values are physical outward Cauchy tractions.
    """
    if formulation == "primal":
        if stabilization_alpha is not None or mean_pressure:
            raise ValueError("primal elasticity has no pressure or GaLS stabilization parameter")
        from pymhm.elasticity_primal import solve_primal_elasticity

        return solve_primal_elasticity(
            mesh,
            constitutive=constitutive,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            source=source,
            dirichlet=dirichlet,
            neumann=neumann,
            skeleton=skeleton,
            degree=1 if degree is None else degree,
            minimal_enrichment=minimal_enrichment,
            local_refinement=local_refinement,
            quadrature_order=quadrature_order,
            rigid_moments=rigid_moments,
            solver=solver,
            local_solver=local_solver,
            backend=backend,
            workers=workers,
        )
    if constitutive is not None or minimal_enrichment:
        raise ValueError("general constitutive tensors currently require formulation='primal'")
    from pymhm.elasticity import solve_displacement_pressure

    return solve_displacement_pressure(
        mesh,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=source,
        dirichlet=dirichlet,
        neumann=neumann,
        skeleton=skeleton,
        local_refinement=local_refinement,
        degree=(2 if formulation == "taylor-hood" else 1) if degree is None else degree,
        formulation=formulation,
        quadrature_order=quadrature_order,
        stabilization_alpha=stabilization_alpha,
        lame_mu_gradient=lame_mu_gradient,
        shear_bounds=shear_bounds,
        mean_pressure=mean_pressure,
        rigid_moments=rigid_moments,
        solver=solver,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
