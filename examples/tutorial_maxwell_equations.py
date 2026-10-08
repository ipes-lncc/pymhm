"""User-defined leapfrog equations using generic curl kernels and hybrid assembly.

The tutorial uses independent DG electric/magnetic coefficients and explicit
canonical tangential moments. The integration algorithm lives in this helper,
outside the package. The package receives only LocalEquations and Equation
records; no physical model name selects a solver. This is a small stationary
patch comparison, without a temporal convergence or conforming H(curl) claim.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import partial
from types import TracebackType
from typing import Any

import numpy as np

from examples.formulations.original import solve_original
from pymhm.core.equations import Equation, LocalEquations, compile_form
from pymhm.core.multiscale import MultiscaleProblem, assemble
from pymhm.core.online import OfflineMultiscaleSystem
from pymhm.core.validation import FloatArray
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import (
    CurlOperators,
    TangentialTraceSpace,
    assemble_local,
    field_values,
    physical_basis,
    physical_points,
    quadrature,
    tangential_load,
    tangential_mass,
    tangential_moments,
    volume_load,
)
from pymhm.linalg.linear import factorize
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.electromagnetic import MaxwellSolution
from pymhm.postprocessing.nodal import nodal_field

_SERIAL = ExecutionConfig()
ElectricItem = tuple[CurlOperators, FloatArray, FloatArray, FloatArray, float]
MagneticItem = tuple[CurlOperators, FloatArray, FloatArray, float]


@dataclass(frozen=True)
class Discretization:
    """Executed DG operators, explicit impedance faces and time-step conventions."""

    skeleton: TangentialTraceSpace
    locals: tuple[CurlOperators, ...]
    impedance: Any
    absorbing: dict[int, Any]
    boundary_data: Any
    permittivity: Any
    permeability: Any
    time_step: float
    order: int
    frequency_bound: float


@dataclass(frozen=True)
class LeapfrogState(MaxwellSolution):
    """Independent coefficients with electric time leading magnetic time by dt/2."""

    constraint_moment_norm: float = 0.0
    original_electric_residual: float = 0.0


def prepare(
    mesh: Any,
    *,
    time_step: float = 0.001,
    degree: int = 2,
    local_refinement: int = 2,
    skeleton: TangentialTraceSpace | None = None,
    permittivity: Any = 1.0,
    permeability: Any = 1.0,
    absorbing: Any = 1.0,
    boundary_data: Any = 0.0,
    quadrature_order: int = 5,
) -> Discretization:
    """Assemble shared mass/curl/trace kernels and enforce their conservative CFL bound.

    The two coordinate groups are scalar/vector for 2D TM and vector/vector
    in 3D. Tangential traces have degree one. All native local factors belong
    to the generic assembly call that uses them and close before returning.
    """
    if np.iscomplexobj(time_step) or not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    if skeleton is None:
        base = (
            TriangularSkeleton(mesh, degree=1)
            if isinstance(mesh, TetraMesh)
            else SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        )
        skeleton = TangentialTraceSpace(base)
    if skeleton.mesh is not mesh:
        raise ValueError("tangential skeleton must belong to the supplied macro mesh")
    locals_ = tuple(
        assemble_local(
            mesh.submesh(cell, local_refinement),
            skeleton,
            cell,
            degree,
            permittivity,
            permeability,
            quadrature_order,
        )
        for cell in range(len(mesh.cells))
    )
    bound = max(local.frequency_bound() for local in locals_)
    if time_step * bound >= 2:
        raise ValueError("time_step violates the mass-scaled curl CFL bound")
    faces = (
        {}
        if absorbing is None
        else dict(absorbing)
        if isinstance(absorbing, dict)
        else {int(face): absorbing for face in mesh.boundary_faces}
    )
    if any(face not in mesh.boundary_faces for face in faces):
        raise ValueError("absorbing indices must identify exterior macrofaces")
    return Discretization(
        skeleton,
        locals_,
        tangential_mass(skeleton, faces, quadrature_order),
        faces,
        boundary_data,
        permittivity,
        permeability,
        float(time_step),
        quadrature_order,
        bound,
    )


def mass_equations(item: tuple[Any, FloatArray]) -> LocalEquations:
    """Declare an independent mass projection with no global trace coordinates."""
    matrix, load = item
    return LocalEquations(a=matrix, L=load, b=0, c=0, dofs=[])


def electric_equations(item: ElectricItem) -> LocalEquations:
    """Declare the midpoint electric field w=(Eold+Enew)/2 and its signed trace.

    Me w + duration/2 Q lambda = Me Eold + duration/2 (F-C.T H).
    The global local contribution is -Q.T w. The separate global Equation
    supplies +impedance lambda=-boundary, with explicit canonical face maps.
    """
    local, electric, magnetic, source, duration = item
    return LocalEquations(
        a=local.electric_mass,
        L=local.electric_mass @ electric + duration / 2 * (source - local.curl.T @ magnetic),
        b=duration / 2 * local.coupling,
        c=-local.coupling.T,
        dofs=local.trace_dofs,
        field_data=(
            nodal_field(
                "electric_midpoint",
                local.mesh,
                local.degree,
                components=1 if local.mesh.points.shape[1] == 2 else 3,
                discontinuous=True,
            ),
        ),
    )


def magnetic_equations(item: MagneticItem) -> LocalEquations:
    """Declare Mh Hnew=Mh Hold+dt C E without any global trace unknown."""
    local, electric, magnetic, duration = item
    return LocalEquations(
        a=local.magnetic_mass,
        L=local.magnetic_mass @ magnetic + duration * (local.curl @ electric),
        b=0,
        c=0,
        dofs=[],
        field_data=(
            nodal_field(
                "magnetic",
                local.mesh,
                local.degree,
                components=local.mesh.points.shape[1],
                discontinuous=True,
            ),
        ),
    )


def electric_global(data: Discretization, boundary: FloatArray) -> Equation:
    """Declare the physical impedance term and the negative boundary right side."""
    return Equation(a=data.impedance, L=-boundary)


def _independent_mass(
    items: tuple[tuple[Any, FloatArray], ...], execution: ExecutionConfig, original: bool = False
) -> tuple[FloatArray, ...]:
    """Solve independent mass equations through the same zero-trace global problem."""
    problem = MultiscaleProblem(Equation(0, 0), mass_equations, items, 0, (0,) * len(items))
    system = assemble(problem, execution=execution)
    return (solve_original(system) if original else system.solve()).fields


def _source(data: Discretization, value: Any, time: float) -> tuple[FloatArray, ...]:
    """Integrate electric forcing at its explicitly declared integer time."""
    source = partial(value, time) if callable(value) else value
    components = 1 if data.skeleton.components == 1 else 3
    return tuple(volume_load(local, source, components, 1, data.order) for local in data.locals)


def boundary_load(data: Discretization, time: float) -> FloatArray:
    """Integrate E_tan-alpha(H cross n) data in canonical tangent coordinates."""
    value = (
        partial(data.boundary_data, time) if callable(data.boundary_data) else data.boundary_data
    )
    return tangential_load(data.skeleton, value, data.absorbing, data.order)


def electric_kick(
    data: Discretization,
    electric: tuple[FloatArray, ...],
    magnetic: tuple[FloatArray, ...],
    forcing: tuple[FloatArray, ...],
    duration: float,
    boundary: FloatArray,
    *,
    execution: ExecutionConfig = _SERIAL,
    electric_provider: Callable[[ElectricItem], LocalEquations] = electric_equations,
    global_provider: Callable[[Discretization, FloatArray], Equation] = electric_global,
    runtime: EquationLeapfrog | None = None,
    original: bool = False,
) -> tuple[tuple[FloatArray, ...], FloatArray, tuple[FloatArray, ...], float, float]:
    """Solve the declared midpoint hybrid equations and recover Enew=2w-Eold.

    The local original rows and the global tangential balance are checked
    independently. Their residual scales retain uncancelled operator terms.
    These coefficient checks do not establish spatial stability or field accuracy.
    """
    items = tuple(
        (local, e, h, f, duration)
        for local, e, h, f in zip(data.locals, electric, magnetic, forcing, strict=True)
    )
    if runtime is None or original:
        problem = MultiscaleProblem(
            global_provider(data, boundary),
            electric_provider,
            items,
            data.skeleton.size,
            (0,) * len(items),
        )
        system = assemble(problem, execution=execution)
        solution = solve_original(system) if original else system.solve()
    else:
        system, solution = runtime.electric_solve(items, boundary)
    average, trace = solution.fields, solution.trace
    updated = tuple(2 * mean - old for mean, old in zip(average, electric, strict=True))
    moments = tangential_moments(data.locals, average, data.skeleton.size)
    constraint = moments - data.impedance @ trace - boundary
    original_residual = 0.0
    scale = max(
        np.linalg.norm(moments),
        np.linalg.norm(data.impedance @ trace),
        np.linalg.norm(boundary),
        np.finfo(float).tiny,
    )
    uncancelled = 0.0
    for local, old, h, f, mean, response in zip(
        data.locals, electric, magnetic, forcing, average, system.responses, strict=True
    ):
        lhs = local.electric_mass @ mean
        coupling = duration / 2 * (local.coupling @ trace[local.trace_dofs])
        load = local.electric_mass @ old + duration / 2 * (f - local.curl.T @ h)
        row_scale = max(
            np.linalg.norm(abs(local.electric_mass) @ abs(mean))
            + np.linalg.norm(abs(coupling))
            + np.linalg.norm(abs(load)),
            np.finfo(float).tiny,
        )
        original_residual = max(
            original_residual, float(np.linalg.norm(lhs + coupling - load) / row_scale)
        )
        uncancelled += np.linalg.norm(
            abs(local.coupling).T
            @ (abs(response.source) + abs(response.lifts) @ abs(trace[local.trace_dofs]))
        )
    if original_residual > 1e-10 or np.linalg.norm(constraint) > 1e-10 * max(scale, uncancelled):
        raise RuntimeError("user-defined electric equations fail their original physical balance")
    return updated, trace, average, float(np.linalg.norm(constraint)), original


def modified_energy(
    data: Discretization, electric: tuple[FloatArray, ...], magnetic: tuple[FloatArray, ...]
) -> float:
    """Evaluate the same leapfrog cross-time energy from independently supplied fields."""
    total = 0.0
    for local, e, h in zip(data.locals, electric, magnetic, strict=True):
        total += (
            e @ (local.electric_mass @ e)
            + h @ (local.magnetic_mass @ h)
            + data.time_step * h @ (local.curl @ e)
        )
    return float(total / 2)


def initialize(
    data: Discretization,
    electric: Any,
    magnetic: Any,
    *,
    source: Any = 0.0,
    execution: ExecutionConfig = _SERIAL,
    electric_provider: Callable[[ElectricItem], LocalEquations] = electric_equations,
    global_provider: Callable[[Discretization, FloatArray], Equation] = electric_global,
    runtime: EquationLeapfrog | None = None,
    original: bool = False,
) -> LeapfrogState:
    """Mass-project initial fields, enforce PEC moments and take the half electric kick."""
    dimension = data.skeleton.mesh.points.shape[1]
    components = 1 if dimension == 2 else 3
    e = _independent_mass(
        tuple(
            (
                local.electric_mass,
                volume_load(local, electric, components, data.permittivity, data.order),
            )
            for local in data.locals
        ),
        execution,
        original,
    )
    h = _independent_mass(
        tuple(
            (
                local.magnetic_mass,
                volume_load(local, magnetic, dimension, data.permeability, data.order),
            )
            for local in data.locals
        ),
        execution,
        original,
    )
    absorbing_ids = (
        np.concatenate([data.skeleton.dofs(face) for face in data.absorbing])
        if data.absorbing
        else np.empty(0, dtype=int)
    )
    if len(np.setdiff1d(np.arange(data.skeleton.size), absorbing_ids)):
        items = tuple(
            (
                local.electric_mass,
                local.electric_mass @ value,
                local.coupling,
                -local.coupling.T,
                local.trace_dofs,
            )
            for local, value in zip(data.locals, e, strict=True)
        )
        problem = MultiscaleProblem(
            Equation(0, 0),
            projection_equations,
            items,
            data.skeleton.size,
            (0,) * len(items),
            fixed={int(index): 0.0 for index in absorbing_ids},
        )
        system = assemble(problem, execution=execution)
        e = (solve_original(system) if original else system.solve()).fields
    duration = data.time_step / 2
    e, trace, _, constraint, original_residual = electric_kick(
        data,
        e,
        h,
        _source(data, source, 0.0),
        duration,
        boundary_load(data, data.time_step / 4),
        execution=execution,
        electric_provider=electric_provider,
        global_provider=global_provider,
        runtime=runtime,
        original=original,
    )
    return LeapfrogState(
        skeleton=data.skeleton,
        locals=data.locals,
        electric=e,
        magnetic=h,
        trace=trace,
        electric_time=duration,
        magnetic_time=0.0,
        time_step=data.time_step,
        frequency_bound=data.frequency_bound,
        energy=modified_energy(data, e, h),
        energy_balance_residual=0.0,
        constraint_moment_norm=constraint,
        original_electric_residual=original_residual,
    )


def projection_equations(item: tuple[Any, FloatArray, Any, Any, Any]) -> LocalEquations:
    """Declare the initial mass-orthogonal projection onto explicit zero trace moments."""
    matrix, load, coupling, balance, dofs = item
    return LocalEquations(matrix, load, coupling, balance, dofs)


def advance(
    data: Discretization,
    state: LeapfrogState,
    *,
    source: Any = 0.0,
    execution: ExecutionConfig = _SERIAL,
    electric_provider: Callable[[ElectricItem], LocalEquations] = electric_equations,
    magnetic_provider: Callable[[MagneticItem], LocalEquations] = magnetic_equations,
    global_provider: Callable[[Discretization, FloatArray], Equation] = electric_global,
    runtime: EquationLeapfrog | None = None,
    original: bool = False,
) -> LeapfrogState:
    """Advance magnetic then electric fields using user-specified LocalEquations."""
    dt = data.time_step
    magnetic_items = tuple(
        (local, e, h, dt)
        for local, e, h in zip(data.locals, state.electric, state.magnetic, strict=True)
    )
    if runtime is None:
        magnetic_problem = MultiscaleProblem(
            Equation(0, 0), magnetic_provider, magnetic_items, 0, (0,) * len(magnetic_items)
        )
        system = assemble(magnetic_problem, execution=execution)
        magnetic = (solve_original(system) if original else system.solve()).fields
    else:
        magnetic = runtime.magnetic_solve(magnetic_items)
    time = state.magnetic_time + dt
    forcing, boundary = _source(data, source, time), boundary_load(data, time)
    electric, trace, average, constraint, original_residual = electric_kick(
        data,
        state.electric,
        magnetic,
        forcing,
        dt,
        boundary,
        execution=execution,
        electric_provider=electric_provider,
        global_provider=global_provider,
        runtime=runtime,
        original=original,
    )
    energy = modified_energy(data, electric, magnetic)
    work = sum(f @ e for f, e in zip(forcing, average, strict=True))
    loss = trace @ (data.impedance @ trace + boundary)
    balance = float(energy - state.energy - dt * (work - loss))
    return LeapfrogState(
        skeleton=data.skeleton,
        locals=data.locals,
        electric=electric,
        magnetic=magnetic,
        trace=trace,
        electric_time=state.electric_time + dt,
        magnetic_time=time,
        time_step=data.time_step,
        frequency_bound=data.frequency_bound,
        energy=energy,
        energy_balance_residual=balance,
        constraint_moment_norm=constraint,
        original_electric_residual=original_residual,
    )


def l2_errors(
    data: Discretization, state: LeapfrogState, electric: Any, magnetic: Any, order: int = 5
) -> tuple[float, float]:
    """Compare physical DG fields independently at their declared staggered times."""
    totals = np.zeros(2)
    for local, e, h in zip(data.locals, state.electric, state.magnetic, strict=True):
        reference, weights, _ = quadrature(local.mesh, 1, order)
        basis, _ = physical_basis(local.mesh, local.degree, reference)
        points = physical_points(local.mesh, reference)
        dimension = points.shape[-1]
        for index, coefficients, exact, components in (
            (0, e, electric, 1 if dimension == 2 else 3),
            (1, h, magnetic, dimension),
        ):
            values = coefficients.reshape(len(local.mesh.cells), -1, components)
            field = np.einsum("tqi,tia->tqa", basis, values)
            truth = field_values(exact, points.reshape(-1, dimension), components)
            totals[index] += np.sum(
                weights * np.sum((field - truth.reshape(field.shape)) ** 2, axis=-1)
            )
    return float(np.sqrt(totals[0])), float(np.sqrt(totals[1]))


def compare_state(state: LeapfrogState, reference: Any) -> dict[str, float]:
    """Measure separate coefficient, trace and energy differences to an explicit comparator."""
    result = {}
    for name in ("electric", "magnetic"):
        result[f"{name}_coefficient_linf"] = max(
            float(np.max(abs(actual - expected)))
            for actual, expected in zip(getattr(state, name), getattr(reference, name), strict=True)
        )
    result["trace_coefficient_linf"] = float(np.max(abs(state.trace - reference.trace)))
    result["energy_difference"] = float(abs(state.energy - reference.energy))
    result["electric_time_difference"] = float(abs(state.electric_time - reference.electric_time))
    result["magnetic_time_difference"] = float(abs(state.magnetic_time - reference.magnetic_time))
    return result


class EquationLeapfrog:
    """Execute the declared leapfrog forms with explicitly owned reusable factors.

    This application helper owns its mathematical midpoint/magnetic equations;
    the package supplies only generic assembly, source updates and solves.
    Fixed spaces and materials keep all A/B/C/D blocks unchanged during this
    march. The full-step and initial half-step electric equations have separate
    offline systems. Callers changing an operator prepare a new runtime.
    """

    def __init__(
        self,
        mesh: Any,
        *,
        degree: int = 3,
        local_refinement: int = 1,
        absorbing: Any = None,
        **options: Any,
    ) -> None:
        """Prepare declared curl/mass/trace operators without invoking a physical solver."""
        self.data = prepare(
            mesh,
            degree=degree,
            local_refinement=local_refinement,
            absorbing=absorbing,
            **options,
        )
        self.skeleton, self.locals = self.data.skeleton, self.data.locals
        self.time_step, self.frequency_bound = self.data.time_step, self.data.frequency_bound
        self._resources = ExitStack()
        self._electric: dict[float, tuple[Any, Any]] = {}
        self._magnetic: Any = None
        self._closed = False
        self.state: LeapfrogState | None = None

    def electric_solve(
        self, items: tuple[ElectricItem, ...], boundary: FloatArray
    ) -> tuple[Any, Any]:
        """Update only midpoint forcing and boundary moments in fixed variational operators."""
        if self._closed:
            raise RuntimeError("equation leapfrog runtime is closed")
        duration = items[0][-1]
        if duration not in self._electric:
            problem = MultiscaleProblem(
                electric_global(self.data, boundary),
                electric_equations,
                items,
                self.skeleton.size,
                (0,) * len(items),
            )
            system = assemble(problem)
            offline = self._resources.enter_context(OfflineMultiscaleSystem(system))
            factor = self._resources.enter_context(factorize(system.matrix))
            self._electric[duration] = offline, factor
        offline, factor = self._electric[duration]
        loads = tuple(compile_form(electric_equations(item).L) for item in items)
        system = offline.with_loads(loads, global_load=-boundary)
        return system, system.solve(factorization=factor)

    def magnetic_solve(self, items: tuple[MagneticItem, ...]) -> tuple[FloatArray, ...]:
        """Reuse independent mass operators for the explicit magnetic right-hand sides."""
        if self._closed:
            raise RuntimeError("equation leapfrog runtime is closed")
        if self._magnetic is None:
            problem = MultiscaleProblem(
                Equation(0, 0), magnetic_equations, items, 0, (0,) * len(items)
            )
            self._magnetic = self._resources.enter_context(
                OfflineMultiscaleSystem(assemble(problem))
            )
        loads = tuple(compile_form(magnetic_equations(item).L) for item in items)
        return self._magnetic.solve(loads).fields

    def initialize(
        self, electric: Any = 0.0, magnetic: Any = 0.0, *, source: Any = 0.0
    ) -> LeapfrogState:
        """Mass-project the supplied physical fields and execute the initial half kick."""
        self.state = initialize(self.data, electric, magnetic, source=source, runtime=self)
        return self.state

    def advance(self, source: Any = 0.0) -> LeapfrogState:
        """Advance the explicit stored state through the user-written time equations."""
        if self.state is None:
            raise RuntimeError("initialize the physical fields before advancing")
        self.state = advance(self.data, self.state, source=source, runtime=self)
        return self.state

    def close(self) -> None:
        """Release every retained local/global factor on all exit paths."""
        self._closed = True
        self._resources.close()

    def __enter__(self) -> EquationLeapfrog:
        """Enter an open explicit-lifetime equation runtime."""
        if self._closed:
            raise RuntimeError("equation leapfrog runtime is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release factors after success or an unsuccessful field update."""
        self.close()
