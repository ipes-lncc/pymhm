"""Backward-Euler conservative transport with reusable local and global operators."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from pymhm._legacy.models.transport.rad import _rad_local, _streamline_scale
from pymhm._legacy.models.transport.solver import ScalarSolution, _transport_local_meshes
from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import element_tabulate, nodal_space
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


@dataclass(frozen=True)
class MacroCoefficient:
    """Explicit one-sided coefficient callbacks, one for each macroelement.

    Entries follow mesh.cells. This wrapper distinguishes a tuple of local
    fields from an ordinary constant vector or tensor. A field is evaluated
    exclusively on the corresponding macroelement, including its boundary.
    """

    fields: tuple[Any, ...]

    def for_cell(self, cell: int, count: int) -> Any:
        """Select a field after checking the coefficient-to-macro association."""
        if len(self.fields) != count:
            raise ValueError("MacroCoefficient requires one field per macroelement")
        return self.fields[cell]


def _cell_field(field: Any, cell: int, count: int) -> Any:
    """Resolve a common or explicitly macro-local coefficient."""
    return field.for_cell(cell, count) if isinstance(field, MacroCoefficient) else field


def _time_field(field: Any, time: float) -> Any:
    """Bind a field(points,time), retaining the scalar constant convention."""
    return (lambda points: field(points, time)) if callable(field) else field


@dataclass(frozen=True)
class _StepReaction:
    """Add the physical capacity divided by the time increment to reaction."""

    reaction: Any
    capacity: Any
    increment: float

    def __call__(self, points: FloatArray) -> FloatArray:
        """Evaluate c + capacity/dt for the complete backward-Euler residual."""
        return (
            scalar_values(self.reaction, points)
            + scalar_values(self.capacity, points) / self.increment
        )


@dataclass(frozen=True)
class _LoadMap:
    """Cached Galerkin/Petrov test integration and consistent old-state mass."""

    dofs: IntArray
    points: FloatArray
    tests: FloatArray
    previous_mass: Any
    mass_moments: FloatArray
    count: int
    boundary_nodes: FloatArray

    def load(self, source: Any, old: FloatArray, increment: float, boundary: Any) -> FloatArray:
        """Assemble source, consistent previous state, and essential nodal values."""
        force = scalar_values(source, self.points.reshape(-1, 2)).reshape(self.points.shape[:2])
        elemental = np.einsum("tqi,tq->ti", self.tests, force)
        load = np.bincount(self.dofs.ravel(), weights=elemental.ravel(), minlength=self.count)
        load += self.previous_mass @ old / increment
        return np.r_[load, scalar_values(boundary, self.boundary_nodes)]


def _load_map(
    fine: TriangleMesh,
    degree: int,
    order: int,
    diffusion: Any,
    velocity: Any,
    divergence: Any,
    reaction: Any,
    capacity: Any,
    stabilization: str,
    boundary_dofs: IntArray,
) -> _LoadMap:
    """Build exact time-residual testing, including SUPG's old-state contribution."""
    bary, weights, material = material_triangle_quadrature(fine, diffusion, max(order, degree + 2))
    dofs, nodes, basis, gradients, _ = element_tabulate(fine, degree, bary)
    points = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    flat = points.reshape(-1, 2)
    rho = scalar_values(capacity, flat).reshape(points.shape[:2])
    if np.any(rho <= 0):
        raise ValueError("capacity must be positive at all quadrature points")
    test = basis.copy()
    if stabilization == "supg":
        beta = vector_values(velocity, flat).reshape(*points.shape[:2], 2)
        tensor = tensor_values(material, flat).reshape(*points.shape[:2], 2, 2)
        effective = (scalar_values(reaction, flat) + scalar_values(divergence, flat)).reshape(
            rho.shape
        )
        tau = _streamline_scale(fine, tensor, beta, effective)
        test += tau[:, :, None] * np.einsum("tqa,tqia->tqi", beta, gradients)
    tests = test * fine.areas[:, None, None] * weights[:, :, None]
    blocks = np.einsum("tqi,tqj,tq->tij", tests, basis, rho)
    mass = _assemble_blocks(blocks, dofs, len(nodes))
    moments = np.bincount(
        dofs.ravel(),
        weights=np.einsum("t,tq,tqi,tq->ti", fine.areas, weights, basis, rho).ravel(),
        minlength=len(nodes),
    )
    return _LoadMap(dofs, points, tests, mass, moments, len(nodes), nodes[boundary_dofs])


@dataclass(frozen=True)
class TransientTransportResult:
    """Time history, mass diagnostics and the number of distinct offline operators.

    ``times`` includes the initial time and the retained output times;
    ``solutions`` excludes the initial state. ``integration_times`` records the
    complete time grid, including steps discarded by ``output_steps``. The balance
    residuals sum the original discrete physical equations on each macrocell,
    including essential-boundary reaction forces. They measure the discrete
    weak balance, not independent error or fine-cell conservation.
    When requested, ``original_residual_norms`` and ``original_rhs_norms``
    include every executed step, in ``integration_times[1:]`` order. They check
    the full original physical rows and free trace equations, not just their
    macro sums, using the shared original-equation convention.
    """

    times: FloatArray
    solutions: tuple[ScalarSolution, ...]
    initial_values: tuple[FloatArray, ...]
    mass_moments: tuple[FloatArray, ...]
    balance_residuals: tuple[FloatArray, ...]
    operator_builds: int
    integration_times: FloatArray | None = None
    original_residual_norms: FloatArray | None = None
    original_rhs_norms: FloatArray | None = None

    def total_mass(self) -> FloatArray:
        """Integrate capacity*u at the initial time and every retained output time."""
        fields = (self.initial_values, *(solution.values for solution in self.solutions))
        return np.asarray(
            [
                sum(
                    float(moment @ value)
                    for moment, value in zip(self.mass_moments, state, strict=True)
                )
                for state in fields
            ]
        )


def solve_transient_transport(
    mesh: TriangleMesh,
    times: Any,
    *,
    initial: Any = 0.0,
    capacity: Any = 1.0,
    diffusion: Any = 1.0,
    velocity: Any = (0.0, 0.0),
    reaction: Any = 0.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    diffusive_flux: dict[int, Any] | None = None,
    dirichlet_enforcement: str = "strong",
    velocity_divergence: Any = None,
    diffusion_divergence: Any = None,
    stabilization: str = "galerkin",
    skeleton: SkeletonSpace | None = None,
    local_refinement: int = 4,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    degree: int = 1,
    quadrature_order: int = 6,
    solver: str = "scipy",
    local_solver: str = "scipy",
    output_steps: Any = None,
    on_step: Callable[[int, float, ScalarSolution, FloatArray], None] | None = None,
    check_original: bool = False,
) -> TransientTransportResult:
    """Advance rho*u_t-div(K grad(u))+div(beta*u)+c*u=f by backward Euler.

    Material, velocity, capacity and reaction are stationary. Source and boundary callbacks accept
    ``(points,time)``; initial data accept ``points``. Positive capacity is integrated consistently,
    including the previous-state part of a SUPG residual. Coefficients may be MacroCoefficient
    instances to preserve one-sided values. Velocity divergence and (for SUPG) diffusion divergence
    must be supplied for variable coefficients.

    ``neumann`` specifies the half-advection Robin flux; ``diffusive_flux`` instead specifies -K
    grad(u).n, as in equation (5.4) of
    [Harder, Paredes and Valentin (2015)](https://doi.org/10.1137/130938499). They must be disjoint.
    Local/global factorizations are reused for repeated increments; differences caused solely by
    roundoff in the time coordinates share an operator. No maximum-principle, unconditional accuracy
    or temporal adaptivity claim follows from implicit time integration. ``local_meshes`` supplies
    validated conforming fine partitions, taking precedence over uniform refinement. Spatial
    operators, old-state mass, boundary nodes and field reconstruction all use those same
    partitions. ``output_steps`` optionally selects strictly increasing one-based step indices to
    retain; every supplied time step is still computed. The result's ``times`` and balance records
    then contain only these selected outputs, while ``integration_times`` preserves the complete
    grid. By default all steps are retained. ``on_step(step, time, solution, balance)`` runs after
    each computed step, including outputs not retained in memory. Its numerical arrays are
    read-only; it may persist them without changing the subsequent evolution. Exceptions propagate
    after native factorizations are closed.

    ``check_original=True`` additionally checks every executed time step using the shared
    original-equation verifier, without correcting the solution. Its residual and physical RHS
    Euclidean norms are stored for all steps, including discarded outputs, in the order of
    ``integration_times[1:]``.
    """
    from pymhm.core.offline import OfflineHybridSystem

    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must be a finite strictly increasing vector of length >=2")
    selected = np.arange(1, len(times)) if output_steps is None else np.asarray(output_steps)
    if (
        selected.ndim != 1
        or not len(selected)
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 1)
        or np.any(selected >= len(times))
        or np.any(selected[1:] <= selected[:-1])
    ):
        raise ValueError(
            "output_steps must be strictly increasing integer indices in 1..len(times)-1"
        )
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    if type(check_original) is not bool:
        raise ValueError("check_original must be a bool")
    retained_steps = set(map(int, selected))
    for name, value in (
        ("degree", degree),
        ("local_refinement", local_refinement),
        ("quadrature_order", quadrature_order),
    ):
        positive_int(value, name)
    if stabilization not in ("galerkin", "supg") or dirichlet_enforcement not in ("weak", "strong"):
        raise ValueError("invalid stabilization or dirichlet_enforcement")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("transient transport requires a scalar skeleton on this mesh")
    robin = {} if neumann is None else dict(neumann)
    diffusive = {} if diffusive_flux is None else dict(diffusive_flux)
    if set(robin) & set(diffusive):
        raise ValueError("Robin and diffusive flux faces must be disjoint")
    natural = {**robin, **diffusive}
    strong_faces = (
        tuple(int(f) for f in mesh.boundary_faces if f not in natural)
        if dirichlet_enforcement == "strong"
        else ()
    )
    count = len(mesh.cells)
    meshes = _transport_local_meshes(mesh, local_refinement, local_meshes)
    previous = tuple(
        scalar_values(_cell_field(initial, cell, count), nodal_space(fine, degree)[1])
        for cell, fine in enumerate(meshes)
    )
    initial_values = previous
    coefficients = []
    for cell, fine in enumerate(meshes):
        k = _cell_field(diffusion, cell, count)
        beta = _cell_field(velocity, cell, count)
        div_beta = _cell_field(velocity_divergence, cell, count)
        div_k = _cell_field(diffusion_divergence, cell, count)
        c = _cell_field(reaction, cell, count)
        rho = _cell_field(capacity, cell, count)
        if callable(beta) and div_beta is None:
            raise ValueError("variable velocity requires velocity_divergence")
        if callable(k) and stabilization == "supg" and div_k is None:
            raise ValueError("SUPG with variable diffusion requires diffusion_divergence")
        div_beta = 0.0 if div_beta is None else div_beta
        div_k = (0.0, 0.0) if div_k is None else div_k
        if not callable(beta) and np.any(scalar_values(div_beta, fine.points)):
            raise ValueError("constant velocity must have zero divergence")
        if not callable(k) and np.any(vector_values(div_k, fine.points)):
            raise ValueError("constant diffusion must have zero divergence")
        coefficients.append((k, beta, div_beta, div_k, c, rho))
    solutions, balances = [], []
    original_checks = []
    operators: dict[float, Any] = {}
    mass_moments: tuple[FloatArray, ...] = ()
    roundoff = 16 * np.finfo(float).eps * max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    with ExitStack() as stack:
        for step, (time0, time) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
            increment = float(time - time0)
            key = next((dt for dt in operators if abs(dt - increment) <= roundoff), increment)
            boundary = _time_field(dirichlet, float(time))
            boundary_load, fixed = boundary_data(
                skeleton,
                boundary,
                {face: _time_field(value, float(time)) for face, value in natural.items()},
                order=max(degree + 2, quadrature_order),
            )
            for face in strong_faces:
                boundary_load[skeleton.dofs(face)] = 0
                fixed.update(dict.fromkeys(skeleton.dofs(face), 0.0))
            if key not in operators:
                problems, maps = [], []
                for cell, (fine, (k, beta, div_beta, div_k, c, rho)) in enumerate(
                    zip(meshes, coefficients, strict=True)
                ):
                    step_reaction = _StepReaction(c, rho, key)
                    assembled = _rad_local(
                        cell,
                        mesh=mesh,
                        skeleton=skeleton,
                        degree=degree,
                        refinement=local_refinement,
                        diffusion=k,
                        diffusion_divergence=div_k,
                        velocity=beta,
                        velocity_divergence=div_beta,
                        reaction=step_reaction,
                        source=0.0,
                        stabilization=stabilization,
                        order=quadrature_order,
                        strong_faces=strong_faces,
                        diffusive_faces=tuple(diffusive),
                        local_meshes=meshes,
                    )
                    maps.append(
                        _load_map(
                            fine,
                            degree,
                            quadrature_order,
                            k,
                            beta,
                            div_beta,
                            step_reaction,
                            rho,
                            stabilization,
                            assembled.metadata[5],
                        )
                    )
                    problems.append(assembled.problem)
                prepared = stack.enter_context(
                    OfflineHybridSystem(
                        problems,
                        boundary_load=boundary_load,
                        fixed=fixed,
                        local_solver=local_solver,
                        solver=solver,
                    )
                )
                operators[key] = (prepared, problems, maps)
            prepared, problems, maps = operators[key]
            loads = [
                load_map.load(
                    _time_field(_cell_field(source, cell, count), float(time)), old, key, boundary
                )
                for cell, (load_map, old) in enumerate(zip(maps, previous, strict=True))
            ]
            solved = prepared.solve(
                loads, boundary_load=boundary_load, fixed=fixed, check_original=check_original
            )
            if check_original:
                original_checks.append(prepared.original_residuals)
            previous = tuple(
                field[: load_map.count] for field, load_map in zip(solved.fields, maps, strict=True)
            )
            balance = np.array(
                [
                    np.sum(
                        (
                            problem.matrix @ field
                            + problem.coupling @ solved.trace[problem.trace_dofs]
                            - load
                        )[: load_map.count]
                    )
                    for problem, field, load, load_map in zip(
                        problems, solved.fields, loads, maps, strict=True
                    )
                ]
            )
            mass_moments = tuple(load_map.mass_moments for load_map in maps)
            solution = ScalarSolution(
                skeleton,
                meshes,
                previous,
                solved,
                degree,
                dirichlet_enforcement == "strong",
                tuple(natural),
            )
            if on_step is not None:
                for array in (
                    *previous,
                    solved.trace,
                    *solved.coarse,
                    *solved.fields,
                    solved.gauge_multipliers,
                    balance,
                ):
                    array.setflags(write=False)
                on_step(step, float(time), solution, balance)
            if step in retained_steps:
                balances.append(balance)
                solutions.append(solution)
    return TransientTransportResult(
        times[np.r_[0, selected]].copy(),
        tuple(solutions),
        initial_values,
        mass_moments,
        tuple(balances),
        len(operators),
        times.copy(),
        np.asarray(original_checks)[:, 0] if check_original else None,
        np.asarray(original_checks)[:, 1] if check_original else None,
    )
