"""Time-domain MHM Maxwell with central DG, mass elimination and leapfrog updates."""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from types import TracebackType
from typing import Any, cast

import numpy as np

from pymhm.core.contracts import LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, positive_int
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import (
    TangentialTraceSpace as MaxwellSkeleton,
)
from pymhm.fem.vector.curl import (
    assemble_local,
    tangential_load,
    tangential_mass,
    tangential_moments,
    tangential_rules,
    volume_load,
)
from pymhm.fem.vector.curl import field_values as field_values
from pymhm.linalg.linear import LinearSolveError, factorize
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.electromagnetic import MaxwellSolution


class MaxwellStepper:
    """Reuse DG mass factors and a positive-definite tangential trace operator.

    This is scalar TM in 2D and full vector Maxwell in 3D. The local magnetic
    operator is C=-curl and the electric operator is exactly -C.T. Separate
    time-independent SPD mass matrices carry permittivity and permeability.
    Electric fields lead magnetic fields by half a time step.

    PEC faces prescribe zero tangential electric field. ``absorbing`` maps
    exterior faces to positive scalar impedance alpha, or selects the entire
    exterior with one scalar/callback. The physical equation is
    E_tan-alpha*(H cross n)=g. Midpoint time integration dissipates energy.
    ``boundary_data(time,points,normals)`` supplies scalar TM or vector g.
    Sources receive (time,points); no damping is added on internal fine faces.
    """

    def __init__(
        self,
        mesh: TriangleMesh | CartesianMacroMesh | TetraMesh,
        *,
        time_step: float,
        degree: int = 3,
        local_refinement: int = 1,
        skeleton: MaxwellSkeleton | None = None,
        permittivity: Any = 1.0,
        permeability: Any = 1.0,
        absorbing: Any = None,
        boundary_data: Any = 0.0,
        quadrature_order: int = 6,
        solver: str = "scipy",
        local_solver: str = "scipy",
    ) -> None:
        """Assemble operators, enforce a conservative CFL bound and own all factors."""
        self._resources = ExitStack()
        self._closed = False
        if not isinstance(mesh, (TriangleMesh, CartesianMacroMesh, TetraMesh)):
            raise TypeError("Maxwell requires triangular, Cartesian or tetrahedral macro geometry")
        if np.iscomplexobj(time_step) or not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        self.time_step = float(time_step)
        self.degree = positive_int(degree, "degree")
        self.order = positive_int(quadrature_order, "quadrature_order")
        self.permittivity, self.permeability = permittivity, permeability
        self.boundary_data = boundary_data
        self.solver = solver
        self.dimension = mesh.points.shape[1]
        self.electric_components = 1 if self.dimension == 2 else 3
        self._constant_loads: dict[tuple[Any, ...], tuple[FloatArray, ...]] = {}
        self.skeleton = skeleton or MaxwellSkeleton(
            SkeletonSpace(cast(Any, mesh), tuple(FaceSpace.uniform(1) for _ in mesh.faces))
            if not isinstance(mesh, TetraMesh)
            else TriangularSkeleton(mesh, degree=1)
        )
        if self.skeleton.mesh is not mesh:
            raise ValueError("Maxwell skeleton must belong to the supplied macro mesh")
        self.absorbing = (
            {}
            if absorbing is None
            else dict(absorbing)
            if isinstance(absorbing, dict)
            else {int(face): absorbing for face in mesh.boundary_faces}
        )
        if any(face not in mesh.boundary_faces for face in self.absorbing):
            raise ValueError("absorbing indices must identify exterior macrofaces")
        try:
            self.locals = tuple(
                assemble_local(
                    mesh.submesh(cell, local_refinement),
                    self.skeleton,
                    cell,
                    self.degree,
                    permittivity,
                    permeability,
                    self.order,
                )
                for cell in range(len(mesh.cells))
            )
            self.frequency_bound = max(local.frequency_bound() for local in self.locals)
            if self.time_step * self.frequency_bound >= 2:
                raise ValueError("time_step violates the mass-scaled curl CFL bound")
            self.electric_factors, self.magnetic_factors, responses = [], [], []
            for local in self.locals:
                me, mh = local.electric_mass, local.magnetic_mass
                ef = self._resources.enter_context(factorize(me, solver=local_solver))
                hf = self._resources.enter_context(factorize(mh, solver=local_solver))
                self.electric_factors.append(ef)
                self.magnetic_factors.append(hf)
                problem = LocalProblem(
                    me, local.coupling.toarray(), np.zeros(me.shape[0]), local.trace_dofs
                )
                _, rhs = problem.condensation_system()
                responses.append(problem.response_from_solution(ef.solve(rhs)))
            self.system = HybridSystem.from_responses(tuple(responses))
            self.impedance = self._impedance_matrix()
            self.factor = self._resources.enter_context(
                factorize(self.time_step / 2 * self.system.matrix + self.impedance, solver=solver)
            )
            self._initialized = False
        except BaseException:
            self.close()
            raise

    def _boundary_rules(self) -> Any:
        """Yield the shared physical face rules and canonical tangent maps."""
        return tangential_rules(self.skeleton, self.absorbing, self.order)

    def _impedance_matrix(self) -> Any:
        """Assemble the outgoing impedance through the common trace mass owner."""
        return tangential_mass(self.skeleton, self.absorbing, self.order)

    def _boundary_load(self, time: float) -> FloatArray:
        """Integrate incident-wave data with the common tangential moment owner."""
        value = (
            partial(self.boundary_data, time)
            if callable(self.boundary_data)
            else self.boundary_data
        )
        return tangential_load(self.skeleton, value, self.absorbing, self.order)

    def _source(self, source: Any, time: float) -> tuple[FloatArray, ...]:
        """Integrate electric forcing at its integer leapfrog time."""
        if not callable(source):
            data = np.asarray(source)
            key = (data.shape, data.dtype.str, data.tobytes())
            if key not in self._constant_loads:
                self._constant_loads[key] = tuple(
                    volume_load(local, source, self.electric_components, 1, self.order)
                    for local in self.locals
                )
            return self._constant_loads[key]

        def value(points: FloatArray) -> Any:
            """Freeze only the time argument of the supplied physical forcing."""
            return source(time, points)

        return tuple(
            volume_load(local, value, self.electric_components, 1, self.order)
            for local in self.locals
        )

    def _moments(self, fields: tuple[FloatArray, ...]) -> FloatArray:
        """Accumulate signed incident fields through the shared trace moment owner."""
        return tangential_moments(self.locals, fields, self.skeleton.size)

    def initialize(
        self, electric: Any = 0.0, magnetic: Any = 0.0, *, source: Any = 0.0
    ) -> MaxwellSolution:
        """Mass-project initial fields and take the consistent half electric kick."""
        if self._closed:
            raise RuntimeError("Maxwell stepper is closed")
        e = tuple(
            factor.solve(
                volume_load(
                    local, electric, self.electric_components, self.permittivity, self.order
                )
            )
            for local, factor in zip(self.locals, self.electric_factors, strict=True)
        )
        self.magnetic = tuple(
            factor.solve(
                volume_load(local, magnetic, self.dimension, self.permeability, self.order)
            )
            for local, factor in zip(self.locals, self.magnetic_factors, strict=True)
        )
        absorbing_ids = (
            np.concatenate([self.skeleton.dofs(face) for face in self.absorbing])
            if self.absorbing
            else np.empty(0, dtype=int)
        )
        constrained = np.setdiff1d(np.arange(self.skeleton.size), absorbing_ids)
        if len(constrained):
            moments = self._moments(e)
            with factorize(
                self.system.matrix[constrained][:, constrained], solver=self.solver
            ) as projection:
                coefficients = np.zeros(self.skeleton.size)
                coefficients[constrained] = projection.solve(moments[constrained])
            e = tuple(
                field - response.lifts @ coefficients[local.trace_dofs]
                for local, response, field in zip(
                    self.locals, self.system.responses, e, strict=True
                )
            )
        self.electric = e
        self.trace = np.zeros(self.skeleton.size)
        self.step_index = 0
        self._initialized = True
        with factorize(
            self.time_step / 4 * self.system.matrix + self.impedance, solver=self.solver
        ) as initial_factor:
            self.electric, self.trace, _ = self._electric_kick(
                self.electric,
                self.magnetic,
                self._source(source, 0),
                self.time_step / 2,
                self._boundary_load(self.time_step / 4),
                initial_factor,
            )
        self.energy_balance_residual = 0.0
        return self.solution()

    def _electric_kick(
        self,
        electric: tuple[FloatArray, ...],
        magnetic: tuple[FloatArray, ...],
        source: tuple[FloatArray, ...],
        duration: float,
        boundary: FloatArray,
        factor: Any,
    ) -> tuple[tuple[FloatArray, ...], FloatArray, tuple[FloatArray, ...]]:
        """Enforce midpoint impedance/PEC moments using the positive mass Schur complement.

        Scale the residual by the uncancelled electric update terms. A stationary
        magnetic field can have a zero exact electric field even though its free
        curl update and trace lift are individually nonzero. Their magnitudes,
        before cancellation, determine the attainable floating-point accuracy.
        """
        free = tuple(
            e + duration * inverse.solve(f - local.curl.T @ h)
            for local, inverse, e, h, f in zip(
                self.locals, self.electric_factors, electric, magnetic, source, strict=True
            )
        )
        average_free = tuple((a + b) / 2 for a, b in zip(electric, free, strict=True))
        trace = factor.solve(self._moments(average_free) - boundary)
        result = tuple(
            e - duration * response.lifts @ trace[local.trace_dofs]
            for local, response, e in zip(self.locals, self.system.responses, free, strict=True)
        )
        average = tuple((a + b) / 2 for a, b in zip(electric, result, strict=True))
        moments = self._moments(average)
        defect = moments - self.impedance @ trace - boundary
        scale = max(
            np.linalg.norm(moments),
            np.linalg.norm(self.impedance @ trace),
            np.linalg.norm(boundary),
            np.finfo(float).tiny,
        )
        scale = max(
            scale,
            sum(
                np.linalg.norm(
                    abs(local.coupling).T
                    @ (
                        abs(old)
                        + abs(predicted)
                        + duration * (abs(response.lifts) @ abs(trace[local.trace_dofs]))
                    )
                    / 2
                )
                for local, response, old, predicted in zip(
                    self.locals, self.system.responses, electric, free, strict=True
                )
            ),
        )
        if np.linalg.norm(defect) > 1e-10 * scale:
            raise LinearSolveError(
                "Maxwell tangential constraint fails its physical relative residual"
            )
        return result, trace, average

    def modified_energy(self) -> float:
        """Evaluate the leapfrog cross-time energy of Equation (5.11)."""
        total = 0.0
        for local, e, h in zip(self.locals, self.electric, self.magnetic, strict=True):
            total += (
                e @ (local.electric_mass @ e)
                + h @ (local.magnetic_mass @ h)
                + self.time_step * h @ (local.curl @ e)
            )
        return float(total / 2)

    def advance(self, source: Any = 0.0) -> MaxwellSolution:
        """Advance magnetic/electric fields and record their physical energy balance."""
        if self._closed or not self._initialized:
            raise RuntimeError("initialize an open Maxwell stepper before advancing")
        old_energy = self.modified_energy()
        self.magnetic = tuple(
            h + self.time_step * inverse.solve(local.curl @ e)
            for local, inverse, e, h in zip(
                self.locals, self.magnetic_factors, self.electric, self.magnetic, strict=True
            )
        )
        time = (self.step_index + 1) * self.time_step
        forcing, boundary = self._source(source, time), self._boundary_load(time)
        self.electric, self.trace, average = self._electric_kick(
            self.electric, self.magnetic, forcing, self.time_step, boundary, self.factor
        )
        work = sum(f @ e for f, e in zip(forcing, average, strict=True))
        loss = self.trace @ (self.impedance @ self.trace + boundary)
        self.energy_balance_residual = float(
            self.modified_energy() - old_energy - self.time_step * (work - loss)
        )
        self.step_index += 1
        return self.solution()

    def solution(self) -> MaxwellSolution:
        """Copy staggered fields into an independently owned numerical record."""
        if not self._initialized:
            raise RuntimeError("initialize Maxwell fields before requesting a solution")
        return MaxwellSolution(
            self.skeleton,
            self.locals,
            tuple(e.copy() for e in self.electric),
            tuple(h.copy() for h in self.magnetic),
            self.trace.copy(),
            (self.step_index + 0.5) * self.time_step,
            self.step_index * self.time_step,
            self.time_step,
            self.frequency_bound,
            self.modified_energy(),
            self.energy_balance_residual,
        )

    def close(self) -> None:
        """Release all native factors exactly once, including partial preparation."""
        self._closed = True
        self._resources.close()

    def __enter__(self) -> MaxwellStepper:
        """Enter an open explicit-lifetime time integrator."""
        if self._closed:
            raise RuntimeError("Maxwell stepper is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release factors on normal or exceptional exit."""
        self.close()


def solve_maxwell(
    mesh: TriangleMesh | CartesianMacroMesh | TetraMesh,
    *,
    time_step: float,
    steps: int,
    electric: Any = 0.0,
    magnetic: Any = 0.0,
    source: Any = 0.0,
    on_step: Any = None,
    **options: Any,
) -> MaxwellSolution:
    """Solve a finite real TM/vector Maxwell trajectory with reusable local/global factors."""
    steps = positive_int(steps, "steps", 0)
    with MaxwellStepper(mesh, time_step=time_step, **options) as stepper:
        result = stepper.initialize(electric, magnetic, source=source)
        if on_step is not None:
            on_step(result)
        for _ in range(steps):
            result = stepper.advance(source)
            if on_step is not None:
                on_step(result)
        return result
