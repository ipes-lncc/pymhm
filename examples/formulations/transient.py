"""User-written backward-Euler conservative transport on declared Pk local spaces.

Spatial Galerkin/SUPG forms are ordinary LocalEquations. The time residual
uses the same Petrov tests for its source and previous-state mass. Generic
OfflineMultiscaleSystem owns reusable factors; no physical solver is called.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.assembly import SolverConfig
from pymhm.core.equations import Equation, LocalEquations
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.online import OfflineMultiscaleSystem
from pymhm.core.refinement import refine_hybrid
from pymhm.core.validation import positive_int
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.loads import assemble_load, assemble_mass
from pymhm.fem.quadrature.material import material_triangle_quadrature
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.stabilization import streamline_scale
from pymhm.fem.scalar.triangle import element_tabulate, nodal_space
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.scalar import diffusive_boundary_matrix, strong_boundary_dofs
from pymhm.linalg.linear import factorize
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.materials.macro import MacroCoefficient
from pymhm.meshes.refinement import validate_submesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import ScalarSolution
from pymhm.postprocessing.transport import TransientTransportResult

from .residual_transport import streamline_equations
from .scalar import ScalarDiscretization

_SERIAL = ExecutionConfig()


@dataclass(frozen=True)
class LocalPartition(ScalarDiscretization):
    """Declared fine partitions, used consistently by spatial/time forms and fields."""

    local_meshes: tuple[TriangleMesh, ...] = ()

    def local_mesh(self, cell: int) -> TriangleMesh:
        """Return the exact supplied local mesh without replacing its approximation space."""
        return self.local_meshes[cell]


@dataclass(frozen=True)
class StepReaction:
    """Explicit c+rho/dt coefficient in the full backward-Euler strong residual."""

    reaction: Any
    capacity: Any
    duration: float

    def __call__(self, points: Any) -> Any:
        """Evaluate the physical reaction plus time-residual coefficient."""
        return (
            scalar_values(self.reaction, points)
            + scalar_values(self.capacity, points) / self.duration
        )


@dataclass(frozen=True)
class TimeLoad:
    """Executed Petrov test tabs, previous-state mass and strong boundary nodes."""

    dofs: Any
    points: Any
    tests: Any
    measure: Any
    previous_mass: Any
    mass_moments: Any
    nodes: Any
    boundary_dofs: Any

    def load(self, source: Any, old: Any, duration: float, dirichlet: Any) -> Any:
        """Integrate f+rho*old/dt against the executed tests and append nodal data."""
        force = scalar_values(source, self.points.reshape(-1, 2)).reshape(self.points.shape[:2])
        volume = assemble_load(self.tests, force, self.measure, self.dofs, len(self.nodes))
        volume += self.previous_mass @ old / duration
        return np.r_[volume, scalar_values(dirichlet, self.nodes[self.boundary_dofs])]


def time_load(
    fine: TriangleMesh,
    degree: int,
    order: int,
    diffusion: Any,
    velocity: Any,
    divergence: Any,
    reaction: Any,
    capacity: Any,
    stabilization: str,
    boundary_dofs: Any,
) -> TimeLoad:
    """Tabulate the literal Galerkin/SUPG source and old-state time pairing.

    For SUPG, tests are phi+tau*beta.grad(phi), with tau evaluated using the
    full step coefficient c+rho/dt+div(beta). Omitting rho*old from this Petrov
    residual would produce a different method. Capacity integrals use trial
    phi, not stabilized tests, and describe physical total mass only.
    """
    bary, weights, material = material_triangle_quadrature(fine, diffusion, max(order, degree + 2))
    dofs, nodes, basis, gradients, _ = element_tabulate(fine, degree, bary)
    points = np.einsum("tqi,tij->tqj", bary, fine.points[fine.cells])
    flat = points.reshape(-1, 2)
    rho = scalar_values(capacity, flat).reshape(points.shape[:2])
    if np.any(rho <= 0):
        raise ValueError("capacity must be positive at all quadrature points")
    tests = basis.copy()
    if stabilization == "supg":
        beta = vector_values(velocity, flat).reshape(*points.shape[:2], 2)
        tensor = tensor_values(material, flat).reshape(*points.shape[:2], 2, 2)
        effective = (scalar_values(reaction, flat) + scalar_values(divergence, flat)).reshape(
            rho.shape
        )
        tau = streamline_scale(fine, tensor, beta, effective)
        tests += tau[:, :, None] * np.einsum("tqa,tqia->tqi", beta, gradients)
    measure = fine.areas[:, None] * weights
    mass = assemble_mass(tests, basis, measure * rho, dofs, dofs, (len(nodes), len(nodes)))
    moments = assemble_load(basis, rho, measure, dofs, len(nodes))
    return TimeLoad(dofs, points, tests, measure, mass, moments, nodes, boundary_dofs)


def step_equations(
    cell: int,
    *,
    data: LocalPartition,
    velocity: Any,
    velocity_divergence: Any,
    diffusion_divergence: Any,
    reaction: Any,
    stabilization: str,
    strong_faces: tuple[int, ...],
    diffusive_faces: tuple[int, ...],
) -> LocalEquations:
    """Declare one step's spatial/time A, face B/C and exact essential constraints.

    The multiplier is (-K grad(u)+beta*u/2).n. Physical diffusive-flux faces add
    the half-advection boundary mass. Strong essential nodes add E.T*u=g with
    separate local reaction coordinates; their external multiplier columns
    vanish. Other cells retain their declared constant coarse coordinate.
    """
    equations = streamline_equations(
        cell,
        data=data,
        velocity=velocity,
        velocity_divergence=velocity_divergence,
        diffusion_divergence=diffusion_divergence,
        reaction=reaction,
        source=0.0,
        stabilization=stabilization,
    )
    fine, moments, pure, zero_reaction, count, _ = equations.metadata
    a, load, b = equations.a, equations.L, np.array(equations.b, copy=True)
    natural = tuple(face for face in diffusive_faces if face in data.mesh.cell_faces[cell])
    if natural:
        a = a + diffusive_boundary_matrix(
            data.mesh, fine, natural, data.degree, velocity, data.order
        )
    essential = tuple(face for face in strong_faces if face in data.mesh.cell_faces[cell])
    ids, _ = strong_boundary_dofs(data.mesh, fine, essential, data.degree, 0.0)
    if len(ids):
        selection = sparse.csc_matrix(
            (np.ones(len(ids)), (ids, np.arange(len(ids)))), shape=(count, len(ids))
        )
        a = sparse.bmat([[a, selection], [selection.T, None]], format="csc")
        load = np.r_[load, np.zeros(len(ids))]
        b = np.vstack((b, np.zeros((len(ids), b.shape[1]))))
        for face in essential:
            b[:, np.isin(equations.dofs, data.skeleton.dofs(face))] = 0
    return replace(
        equations,
        a=a,
        L=load,
        b=b,
        c=-b.T,
        kernel=None,
        coarse_basis=np.empty((len(load), 0)) if len(ids) else equations.coarse_basis,
        moments=np.empty((len(load), 0)) if len(ids) else equations.moments,
        metadata=(fine, moments, pure, zero_reaction, count, ids),
        field_data=(
            nodal_field(
                "concentration",
                fine,
                data.degree,
                reconstruction=sparse.eye(len(load), format="csr")[:count],
            ),
        ),
    )


def _cell(field: Any, cell: int, count: int) -> Any:
    """Select a declared one-sided macro coefficient without inferring tuple semantics."""
    return field.for_cell(cell, count) if isinstance(field, MacroCoefficient) else field


def _time(field: Any, time: float) -> Any:
    """Bind an explicitly declared (points,time) source or boundary callback."""
    return (lambda points: field(points, time)) if callable(field) else field


def solve_transport_trajectory(
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
    on_step: Callable[..., None] | None = None,
    check_original: bool = False,
    execution: ExecutionConfig = _SERIAL,
) -> TransientTransportResult:
    """Execute the declared backward-Euler equations with explicit reusable factors.

    Stationary material callbacks accept points. Time-dependent source/boundary
    callbacks accept (points,time); initial callbacks accept points. Coefficients
    may be MacroCoefficient records. Natural Robin and physical diffusive flux
    data use distinct face sets. Repeated time increments share literal A/B/C/D
    and local/global factors. This time march is an application written using
    public mathematical operators, not a dispatch to a package method solver.
    """
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("times must be finite and strictly increasing")
    selected = np.arange(1, len(times)) if output_steps is None else np.asarray(output_steps)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or np.any(selected < 1)
        or np.any(selected >= len(times))
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("output_steps must be strictly increasing step indices")
    for name, value in (
        ("degree", degree),
        ("local_refinement", local_refinement),
        ("quadrature_order", quadrature_order),
    ):
        positive_int(value, name)
    if stabilization not in ("galerkin", "supg") or dirichlet_enforcement not in ("weak", "strong"):
        raise ValueError("invalid stabilization or essential enforcement")
    skeleton = SkeletonSpace(mesh) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("declare a scalar skeleton on the given mesh")
    natural, diffusive = dict(neumann or {}), dict(diffusive_flux or {})
    if set(natural) & set(diffusive):
        raise ValueError("Robin and diffusive flux faces must be disjoint")
    natural.update(diffusive)
    strong = (
        tuple(int(f) for f in mesh.boundary_faces if f not in natural)
        if dirichlet_enforcement == "strong"
        else ()
    )
    count = len(mesh.cells)
    meshes = (
        tuple(mesh.submesh(cell, local_refinement) for cell in range(count))
        if local_meshes is None
        else tuple(local_meshes)
    )
    if len(meshes) != count:
        raise ValueError("one conforming local mesh is required per macrocell")
    for cell, fine in enumerate(meshes):
        validate_submesh(mesh, cell, fine)
    previous = tuple(
        scalar_values(_cell(initial, cell, count), nodal_space(fine, degree)[1])
        for cell, fine in enumerate(meshes)
    )
    initial_values = previous
    coefficients = []
    for cell, fine in enumerate(meshes):
        k, beta, db, dk, c, rho = tuple(
            _cell(value, cell, count)
            for value in (
                diffusion,
                velocity,
                velocity_divergence,
                diffusion_divergence,
                reaction,
                capacity,
            )
        )
        if callable(beta) and db is None:
            raise ValueError("variable velocity requires its declared divergence")
        if callable(k) and stabilization == "supg" and dk is None:
            raise ValueError("variable diffusion requires its declared divergence for SUPG")
        db, dk = 0.0 if db is None else db, (0.0, 0.0) if dk is None else dk
        if not callable(beta) and np.any(scalar_values(db, fine.points)):
            raise ValueError("constant velocity must have zero divergence")
        if not callable(k) and np.any(vector_values(dk, fine.points)):
            raise ValueError("constant diffusion must have zero divergence")
        coefficients.append((k, beta, db, dk, c, rho))
    operators: dict[float, Any] = {}
    solutions, balances, checks = [], [], []
    retained = set(map(int, selected))
    roundoff = 16 * np.finfo(float).eps * max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    with ExitStack() as resources:
        for step, (time0, time) in enumerate(zip(times[:-1], times[1:], strict=True), start=1):
            increment = float(time - time0)
            duration = next((dt for dt in operators if abs(dt - increment) <= roundoff), increment)
            boundary = _time(dirichlet, float(time))
            boundary_load, fixed = boundary_data(
                skeleton,
                boundary,
                {f: _time(value, float(time)) for f, value in natural.items()},
                order=max(degree + 2, quadrature_order),
            )
            for face in strong:
                boundary_load[skeleton.dofs(face)] = 0
                fixed.update(dict.fromkeys(skeleton.dofs(face), 0.0))
            if duration not in operators:
                equations, load_maps = [], []
                for cell, (fine, (k, beta, db, dk, c, rho)) in enumerate(
                    zip(meshes, coefficients, strict=True)
                ):
                    reaction_step = StepReaction(c, rho, duration)
                    data = LocalPartition(
                        mesh, skeleton, k, degree, local_refinement, quadrature_order, meshes
                    )
                    form = step_equations(
                        cell,
                        data=data,
                        velocity=beta,
                        velocity_divergence=db,
                        diffusion_divergence=dk,
                        reaction=reaction_step,
                        stabilization=stabilization,
                        strong_faces=strong,
                        diffusive_faces=tuple(diffusive),
                    )
                    load_maps.append(
                        time_load(
                            fine,
                            degree,
                            quadrature_order,
                            k,
                            beta,
                            db,
                            reaction_step,
                            rho,
                            stabilization,
                            form.metadata[5],
                        )
                    )
                    equations.append(form)
                problem = MultiscaleProblem(
                    Equation(
                        0,
                        -np.r_[
                            boundary_load,
                            np.zeros(sum(form.coarse_basis.shape[1] for form in equations)),
                        ],
                    ),
                    equations.__getitem__,
                    range(count),
                    skeleton.size,
                    tuple(form.coarse_basis.shape[1] for form in equations),
                    fixed=fixed,
                )
                system = assemble(
                    problem,
                    execution=execution,
                    solvers=SolverConfig(global_solver=solver, local_solver=local_solver),
                )
                cache = resources.enter_context(
                    OfflineMultiscaleSystem(system, solver=local_solver)
                )
                free = np.setdiff1d(np.arange(len(system.rhs)), list(fixed))
                factor = resources.enter_context(
                    factorize(system.matrix[free][:, free], solver=solver)
                )
                operators[duration] = (cache, tuple(load_maps), factor)
            cache, maps, factor = operators[duration]
            loads = tuple(
                mapping.load(
                    _time(_cell(source, cell, count), float(time)), old, duration, boundary
                )
                for cell, (mapping, old) in enumerate(zip(maps, previous, strict=True))
            )
            system = cache.with_loads(
                loads,
                global_load=-np.r_[
                    boundary_load, np.zeros(len(cache.template.rhs) - skeleton.size)
                ],
            )
            solved = system.solve(fixed=fixed, factorization=factor)
            if check_original:
                check = refine_hybrid(
                    system, solved, boundary_load=boundary_load, fixed=fixed, max_steps=0
                )
                checks.append((check.residual_norms[-1], check.rhs_norm))
            previous = tuple(
                values[: len(mapping.nodes)]
                for values, mapping in zip(solved.fields, maps, strict=True)
            )
            balance = np.asarray(
                [
                    np.sum(
                        (
                            record.equations.problem.matrix @ values
                            + record.equations.problem.coupling
                            @ solved.trace[record.equations.problem.trace_dofs]
                            - load
                        )[: len(mapping.nodes)]
                    )
                    for record, values, load, mapping in zip(
                        system.cells, solved.fields, loads, maps, strict=True
                    )
                ]
            )
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
                for values in (
                    *previous,
                    solved.trace,
                    *solved.coarse,
                    *solved.fields,
                    solved.gauge_multipliers,
                    balance,
                ):
                    values.setflags(write=False)
                on_step(step, float(time), solution, balance)
            if step in retained:
                solutions.append(solution)
                balances.append(balance)
    return TransientTransportResult(
        times[np.r_[0, selected]].copy(),
        tuple(solutions),
        initial_values,
        tuple(mapping.mass_moments for mapping in maps),
        tuple(balances),
        len(operators),
        times.copy(),
        np.asarray(checks)[:, 0] if check_original else None,
        np.asarray(checks)[:, 1] if check_original else None,
    )


def solve_heat_trajectory(
    mesh: TriangleMesh,
    times: Any,
    *,
    source: Any = None,
    dirichlet: Any = None,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    **options: Any,
) -> tuple[ScalarSolution, ...]:
    """Declare capacity one, zero advection/reaction and weak thermal Dirichlet data.

    All numerical operations remain the explicit backward-Euler forms above.
    This short application composition retains the tuple-of-solutions contract
    used by the heat galleries. General capacity and mixed data are declared
    through solve_transport_trajectory directly.
    """
    return solve_transport_trajectory(
        mesh,
        times,
        source=0.0 if source is None else source,
        dirichlet=0.0 if dirichlet is None else dirichlet,
        dirichlet_enforcement="weak",
        execution=ExecutionConfig(backend=backend, workers=workers),
        **options,
    ).solutions
