"""Historical vector solution records and plane flow/elasticity solver dispatch."""

from typing import Any, Literal

from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.solutions import VectorSolution as VectorSolution


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
    :func:`pymhm._legacy.models.flow.solver.solve_flow` for boundary and gauge conventions,
    including
    Cartesian ``traction_components`` for slip walls and mixed component data.
    """
    from pymhm._legacy.models.flow.solver import solve_flow

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
        from pymhm._legacy.models.elasticity.primal import solve_primal_elasticity

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
    from pymhm._legacy.models.elasticity.mixed_pressure import solve_displacement_pressure

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
