"""Scalar conservative reaction-advection-diffusion and implicit heat evolution."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pymhm.core.contracts import HybridSolution, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.scalar.operators import boundary_data, triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space, scalar_operators, tabulate, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.materials.evaluation import scalar_values
from pymhm.meshes.triangle import TriangleMesh


def _transport_local_meshes(
    mesh: TriangleMesh,
    refinement: int,
    supplied: tuple[TriangleMesh, ...] | None,
) -> tuple[TriangleMesh, ...]:
    """Select actual local approximation meshes, validating supplied partitions.

    A supplied mesh takes precedence over the nominal uniform refinement. Cell
    counts alone do not identify a geometric partition or its material jumps.
    """
    if supplied is None:
        return tuple(mesh.submesh(cell, refinement) for cell in range(len(mesh.cells)))
    from pymhm.meshes.refinement import validate_submesh

    if len(supplied) != len(mesh.cells):
        raise ValueError("one local mesh is required for each macrocell")
    for cell, fine in enumerate(supplied):
        validate_submesh(mesh, cell, fine)
    return tuple(supplied)


@dataclass(frozen=True)
class ScalarSolution:
    """Broken Pk scalar field reconstructed from a hybrid system."""

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    values: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int = 1
    strong_dirichlet: bool = False
    natural_faces: tuple[int, ...] = ()

    def l2_error(self, exact: Any, order: int = 5) -> float:
        """Integrate scalar error using quadrature independent of assembly."""
        bary, weights = triangle_quadrature(order)
        total = 0.0
        for mesh, values in zip(self.local_meshes, self.values, strict=True):
            points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
            dofs, _, basis, _, _ = tabulate(mesh, self.degree, bary)
            difference = values[dofs] @ basis.T - scalar_values(
                exact, points.reshape(-1, 2)
            ).reshape(len(mesh.cells), len(weights))
            total += float(mesh.areas @ (difference**2 @ weights))
        return float(np.sqrt(total))


def solve_transport(
    mesh: TriangleMesh,
    *,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    diffusive_flux: dict[int, Any] | None = None,
    dirichlet_enforcement: str = "weak",
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    degree: int = 1,
    quadrature_order: int = 6,
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    unusual_parameters: Any = None,
    mean_value: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    coarse_space: Literal["constants", "kernel"] = "constants",
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> ScalarSolution:
    """Solve RAD with Pk Galerkin, SUPG or reaction--diffusion UNUSUAL locals.

    ``velocity`` and ``reaction`` may vary spatially. A variable velocity
    requires ``velocity_divergence``; variable diffusion in SUPG also requires
    ``diffusion_divergence``. ``neumann`` prescribes the half-advection Robin
    multiplier, not the total conservative flux. See
    :func:`pymhm._legacy.models.transport.rad.solve_rad`
    for the complete operator, stabilization, coefficient and gauge conventions.
    """
    from pymhm._legacy.models.transport.rad import solve_rad

    return solve_rad(
        mesh,
        diffusion=diffusion,
        velocity=velocity,
        reaction=reaction,
        source=source,
        dirichlet=dirichlet,
        neumann=neumann,
        diffusive_flux=diffusive_flux,
        dirichlet_enforcement=dirichlet_enforcement,
        skeleton=skeleton,
        local_refinement=local_refinement,
        local_meshes=local_meshes,
        degree=degree,
        quadrature_order=quadrature_order,
        velocity_divergence=velocity_divergence,
        diffusion_divergence=diffusion_divergence,
        stabilization=stabilization,
        unusual_parameters=unusual_parameters,
        mean_value=mean_value,
        solver=solver,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
        coarse_space=coarse_space,
        global_refinement_precision=global_refinement_precision,
    )


def solve_heat(
    mesh: TriangleMesh,
    times: Any,
    *,
    initial: Any = 0.0,
    diffusion: Any = 1.0,
    source: Any = None,
    dirichlet: Any = None,
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    degree: int = 1,
    quadrature_order: int = 6,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> tuple[ScalarSolution, ...]:
    """Advance ``u_t-div(K grad(u))=f`` with backward Euler in physical time.

    ``times`` includes the initial time and at least one later time. Initial
    data are interpolated in the local continuous Pk space selected by ``degree``.
    Time-dependent source and boundary callables
    have signatures ``field(points, time)``. The return contains one solution
    per step, excluding the initial interpolation. The diffusion tensor is time
    independent; matrices are assembled once, but factorized afresh per step.
    Retaining local constants with physical mean constraints preserves the
    steady diffusion limit when the time increment becomes large.
    ``local_meshes`` supplies one validated conforming fine partition per
    macrocell and takes precedence over uniform ``local_refinement``. The same
    meshes define initial interpolation, all volume forms, trace couplings and
    reconstructed fields.
    """
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must be a finite strictly increasing vector of length >= 2")
    positive_int(local_refinement, "local_refinement")
    positive_int(degree, "degree")
    positive_int(quadrature_order, "quadrature_order")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("heat requires a scalar skeleton on this mesh")
    meshes = _transport_local_meshes(mesh, local_refinement, local_meshes)
    operators = [
        scalar_operators(fine, degree, diffusion=diffusion, order=quadrature_order)[:2]
        for fine in meshes
    ]
    couplings = [
        trace_coupling(mesh, cell, fine, skeleton, degree) for cell, fine in enumerate(meshes)
    ]
    previous = tuple(scalar_values(initial, nodal_space(fine, degree)[1]) for fine in meshes)
    solutions = []
    for t0, time in zip(times[:-1], times[1:], strict=True):
        dt = time - t0
        boundary_field = 0.0 if dirichlet is None else lambda x, t=time: dirichlet(x, t)
        boundary, _ = boundary_data(
            skeleton, boundary_field, order=max(quadrature_order, degree + 2)
        )
        problems = []
        for cell, (fine, (stiffness, mass), coupling, old) in enumerate(
            zip(meshes, operators, couplings, previous, strict=True)
        ):
            source_field = 0.0 if source is None else lambda x, t=time: source(x, t)
            load = (
                scalar_operators(
                    fine, degree, diffusion=diffusion, source=source_field, order=quadrature_order
                )[2]
                + mass @ old / dt
            )
            constant = np.ones((len(old), 1))
            problems.append(
                LocalProblem(
                    stiffness + mass / dt,
                    coupling,
                    load,
                    skeleton.cell_dofs(cell),
                    constraints=mass @ constant,
                    coarse_basis=constant,
                )
            )
        system = HybridSystem(
            problems,
            boundary_load=boundary,
            local_solver=local_solver,
            backend=backend,
            workers=workers,
        )
        result = system.solve(solver=solver)
        previous = result.fields
        solutions.append(ScalarSolution(skeleton, meshes, previous, result, degree))
    return tuple(solutions)
