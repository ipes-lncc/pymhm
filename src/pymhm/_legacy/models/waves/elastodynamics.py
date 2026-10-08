"""Space-time MHM elasticity with local Newmark propagation on triangles/tetrahedra.

The slabwise constant multiplier is negative physical traction. Local responses
follow Gomes et al. (2017), equations (42)--(48), with beta=1/4 and gamma=1/2.
Independent local substeps end at a common macro time. No incompressible-limit
claim is made for the primal displacement spatial discretization.
"""

from __future__ import annotations

from contextlib import ExitStack
from functools import partial
from types import TracebackType
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm._legacy.models.elasticity.primal import _local_primal
from pymhm._legacy.models.elasticity.primal_3d import (
    _boundary_vector,
)
from pymhm._legacy.models.elasticity.primal_3d import _local as elastic_local_3d
from pymhm.core.validation import FloatArray, positive_int
from pymhm.execution.cpu import map_local
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import physical_basis, physical_points, quadrature
from pymhm.fem.vector.operators import _assemble_blocks
from pymhm.linalg.dynamics import newmark_step
from pymhm.linalg.linear import LinearSolveError, factorize
from pymhm.materials.evaluation import scalar_values_3d
from pymhm.materials.sources import (
    SeparableTriangleField,
    TriangleQuadratureField,
)
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.dynamics import (
    ElastodynamicLocal,
    ElastodynamicSolution,
    PreparedElastodynamicSource,
)
from pymhm.postprocessing.dynamics import _scale_source_load as _scale_source_load
from pymhm.postprocessing.dynamics import stress_from_gradient as _compat_stress_from_gradient

_stress_from_gradient = _compat_stress_from_gradient


def _density_values(data: Any, points: FloatArray) -> FloatArray:
    """Read scalar density, including the isotropic values from planar material cuts."""
    raw = data(points) if callable(data) else data
    array = np.asarray(raw)
    dimension = points.shape[1]
    if array.shape[-2:] == (dimension, dimension):
        if not np.all(array == array[..., :1, :1] * np.eye(dimension)):
            raise ValueError("density requires scalar or isotropic planar material data")
        raw = array[..., 0, 0]
    values = scalar_values_3d(raw, points)
    if np.any(values <= 0):
        raise ValueError("density must be positive")
    return values


def _make_local(
    cell: int,
    mesh: Any,
    skeleton: Any,
    degree: int,
    refinement: int,
    order: int,
    density: Any,
    constitutive: Any,
    lame_lambda: Any,
    lame_mu: Any,
) -> ElastodynamicLocal:
    """Reuse the static physical elasticity operator without its nullspace condensation."""
    dimension = mesh.points.shape[1]
    options = dict(
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        refinement=refinement,
        constitutive=constitutive,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        source=np.zeros(dimension),
        order=order,
    )
    if dimension == 2:
        assembly = _local_primal(cell, minimal_enrichment=False, **options)
        fine = assembly.metadata[0]
        dofs, nodes = nodal_space(fine, degree)
    else:
        assembly = elastic_local_3d(cell, **options)
        fine = assembly.metadata[0]
        dofs, nodes = tetra_nodal_space(fine, degree)
    bary, weights, density_data = quadrature(fine, density, order)
    basis, _ = physical_basis(fine, degree, bary)
    points = physical_points(fine, bary)
    rho = _density_values(density_data, points.reshape(-1, dimension)).reshape(points.shape[:2])
    blocks = np.einsum("tq,tqi,tqj->tij", weights * rho, basis, basis)
    mass = sparse.kron(
        _assemble_blocks(blocks, dofs, len(nodes)), sparse.eye(dimension), format="csc"
    )
    problem = assembly.problem
    return ElastodynamicLocal(
        fine,
        degree,
        mass,
        problem.matrix,
        problem.coupling,
        problem.trace_dofs,
        dofs,
        nodes,
        basis,
        points,
        weights,
        rho,
        constitutive,
        lame_lambda,
        lame_mu,
        order,
    )


class ElastodynamicStepper:
    """Reusable local Newmark propagators coupled through endpoint displacement moments.

    ``local_substeps`` may be a scalar or one positive count per macrocell.
    Dirichlet displacement is time independent; ``source(t, points)`` can vary
    in time. A triangular source may instead implement ``at_time(t)`` and return
    a ``TriangleQuadratureField``, preserving support-specific integration.
    Both conventions supply force per volume, not acceleration. ``traction``
    contains time-independent physical outward tractions.
    With one local substep, zero forcing and homogeneous fixed displacements,
    beta=1/4, gamma=1/2 conserves the physical discrete elastic plus kinetic energy.
    Arbitrary subcycling does not inherit that energy statement automatically.
    """

    def __init__(
        self,
        mesh: TriangleMesh | TetraMesh,
        *,
        time_step: float,
        degree: int = 3,
        local_refinement: int = 2,
        local_substeps: Any = 1,
        skeleton: Any = None,
        density: Any = 1.0,
        constitutive: Any = None,
        lame_lambda: Any = 1.0,
        lame_mu: Any = 1.0,
        dirichlet: Any = 0.0,
        traction: dict[int, Any] | None = None,
        quadrature_order: int = 8,
        solver: str = "scipy",
        local_solver: str = "scipy",
        backend: Literal["serial", "thread", "process"] = "serial",
        workers: int | None = None,
    ) -> None:
        """Assemble original spatial forms and constant-load slab responses once.

        ``backend`` and ``workers`` distribute independent spatial assemblies.
        Processes use portable spawn semantics and return ordinary arrays and
        sparse matrices; native factorization resources remain in the parent.
        The time march reuses those local factors without worker serialization.
        """
        if not isinstance(mesh, (TriangleMesh, TetraMesh)):
            raise TypeError("elastodynamics requires triangular or tetrahedral macro meshes")
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be finite and positive")
        self.mesh, self.time_step = mesh, float(time_step)
        self.dimension = mesh.points.shape[1]
        self.degree = positive_int(degree, "degree")
        self.order = positive_int(quadrature_order, "quadrature_order", degree + 2)
        refinement = positive_int(local_refinement, "local_refinement")
        counts = np.broadcast_to(np.asarray(local_substeps), (len(mesh.cells),))
        self.substeps = tuple(positive_int(value, "local_substeps") for value in counts)
        if skeleton is None:
            skeleton = (
                SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2)
                if isinstance(mesh, TriangleMesh)
                else TriangularSkeleton(mesh, degree=1)
            )
        if skeleton.mesh is not mesh:
            raise ValueError("skeleton must belong to the macro mesh")
        if self.dimension == 2 and skeleton.components != 2:
            raise ValueError("two-dimensional elastic traces require two components")
        self.skeleton = skeleton
        self.size = skeleton.size * (3 if self.dimension == 3 else 1)
        traction = {} if traction is None else traction
        if not set(traction).issubset(set(mesh.boundary_faces)):
            raise ValueError("traction keys must identify exterior faces")
        if self.dimension == 2:
            self.boundary, prescribed = boundary_data(
                skeleton, dirichlet, traction, order=self.order
            )
            prescribed = {key: -value for key, value in prescribed.items()}
        else:
            self.boundary, prescribed = _boundary_vector(skeleton, dirichlet, traction, self.order)
        self.fixed = np.zeros(self.size)
        self.fixed[list(prescribed)] = list(prescribed.values())
        self.free = np.setdiff1d(np.arange(self.size), list(prescribed))
        factory = partial(
            _make_local,
            mesh=mesh,
            skeleton=skeleton,
            degree=degree,
            refinement=refinement,
            order=self.order,
            density=density,
            constitutive=constitutive,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
        )
        self.locals = tuple(
            map_local(factory, range(len(mesh.cells)), backend=backend, workers=workers)
        )
        self._resources = ExitStack()
        self._closed, self._initialized = False, False
        self._solver = solver
        try:
            self.mass_factors = tuple(
                self._resources.enter_context(factorize(local.mass, solver=local_solver))
                for local in self.locals
            )
            self.factors = tuple(
                self._resources.enter_context(
                    factorize(
                        local.mass + (self.time_step / count) ** 2 / 4 * local.stiffness,
                        solver=local_solver,
                    )
                )
                for local, count in zip(self.locals, self.substeps, strict=True)
            )
            self.lifts, self.velocity_lifts = [], []
            for index, local in enumerate(self.locals):
                zero = np.zeros_like(local.coupling)
                u, v = zero.copy(), zero.copy()
                for _ in range(self.substeps[index]):
                    u, v = self._local_step(index, u, v, local.coupling, local.coupling)
                self.lifts.append(u)
                self.velocity_lifts.append(v)
            self.matrix = self._assemble_schur(self.lifts)
            self.factor = (
                self._resources.enter_context(
                    factorize(self.matrix[self.free][:, self.free], solver=solver)
                )
                if len(self.free)
                else None
            )
        except BaseException:
            self.close()
            raise

    def _assemble_schur(self, lifts: Any) -> Any:
        """Scatter signed physical face moments of local response columns."""
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for local, lift in zip(self.locals, lifts, strict=True):
            block = local.coupling.T @ lift
            indices = local.trace_dofs
            rows.extend(np.repeat(indices, len(indices)))
            columns.extend(np.tile(indices, len(indices)))
            values.extend(block.ravel())
        return sparse.csc_matrix((values, (rows, columns)), shape=(self.size, self.size))

    def _moments(self, fields: Any) -> FloatArray:
        """Assemble physical signed displacement moments without averaging fields."""
        result = np.zeros(self.size)
        for local, field in zip(self.locals, fields, strict=True):
            np.add.at(result, local.trace_dofs, local.coupling.T @ field)
        return result

    def prepare_source(self, source: SeparableTriangleField) -> PreparedElastodynamicSource:
        """Integrate an explicitly separable planar source once in the original local bases.

        The spatial provider/rule, force units and nodal ordering are unchanged.
        This opt-in snapshot preserves original Newmark and boundary equations;
        it is passed to ``advance`` like an ordinary source. Algebraic temporal
        scaling may differ by rounding from integrating a pre-scaled field.
        Nonseparable sources and three-dimensional sources keep their existing
        integration path. Native resources remain owned by this stepper.
        """
        if self._closed:
            raise RuntimeError("prepare a source on an open elastodynamic stepper")
        if self.dimension != 2:
            raise ValueError("prepared triangular sources require two dimensions")
        if not isinstance(source, SeparableTriangleField):
            raise TypeError("source must declare an explicit separable triangular field")
        spatial = source.spatial_field()
        if not isinstance(spatial, TriangleQuadratureField):
            raise TypeError("separable spatial source must provide triangular quadrature")
        return PreparedElastodynamicSource(
            self.locals, tuple(local.load(spatial) for local in self.locals), source.time_scale
        )

    def _local_step(
        self, index: int, u: FloatArray, v: FloatArray, old: FloatArray, new: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Advance the literal beta=1/4, gamma=1/2 local equations (42)/(43)."""
        local, duration = self.locals[index], self.time_step / self.substeps[index]
        return newmark_step(
            local.mass,
            local.stiffness,
            self.factors[index],
            self.mass_factors[index],
            duration,
            u,
            v,
            old,
            new,
        )

    def initialize(self, displacement: Any = 0.0, velocity: Any = 0.0) -> ElastodynamicSolution:
        """Project initial fields in physical mass with the prescribed trace moments."""
        if self._closed:
            raise RuntimeError("initialize an open elastodynamic stepper")
        lifts = [
            inverse.solve(local.coupling)
            for local, inverse in zip(self.locals, self.mass_factors, strict=True)
        ]
        matrix = self._assemble_schur(lifts)
        with ExitStack() as resources:
            factor = (
                resources.enter_context(
                    factorize(matrix[self.free][:, self.free], solver=self._solver)
                )
                if len(self.free)
                else None
            )
            states = []
            for datum, target in ((displacement, self.boundary), (velocity, np.zeros(self.size))):
                fields = [
                    inverse.solve(local.load(datum, density_weighted=True))
                    for local, inverse in zip(self.locals, self.mass_factors, strict=True)
                ]
                multiplier = np.zeros(self.size)
                if factor is not None:
                    multiplier[self.free] = factor.solve(
                        (self._moments(fields) - target)[self.free]
                    )
                states.append(
                    tuple(
                        field - lift @ multiplier[local.trace_dofs]
                        for field, lift, local in zip(fields, lifts, self.locals, strict=True)
                    )
                )
        self.displacement, self.velocity = states
        self.trace = self.fixed.copy()
        self.index = 0
        self._initialized = True
        return self.solution()

    def advance(self, source: Any = 0.0) -> ElastodynamicSolution:
        """Propagate each local history, solve the endpoint trace, and recover velocity."""
        if self._closed or not self._initialized:
            raise RuntimeError("initialize an open elastodynamic stepper before advancing")
        free_u, free_v = [], []
        for index, local in enumerate(self.locals):
            u, v = self.displacement[index], self.velocity[index]
            duration = self.time_step / self.substeps[index]
            time = self.index * self.time_step
            old = local.load_at_time(source, time)
            for step in range(self.substeps[index]):
                time = self.index * self.time_step + (step + 1) * duration
                new = local.load_at_time(source, time)
                u, v = self._local_step(index, u, v, old, new)
                old = new
            free_u.append(u)
            free_v.append(v)
        rhs = self._moments(free_u) - self.boundary - self.matrix @ self.fixed
        self.trace = self.fixed.copy()
        if self.factor is not None:
            self.trace[self.free] = self.factor.solve(rhs[self.free])
        self.displacement = tuple(
            u - lift @ self.trace[local.trace_dofs]
            for local, lift, u in zip(self.locals, self.lifts, free_u, strict=True)
        )
        self.velocity = tuple(
            v - lift @ self.trace[local.trace_dofs]
            for local, lift, v in zip(self.locals, self.velocity_lifts, free_v, strict=True)
        )
        self.index += 1
        return self.solution()

    def solution(self) -> ElastodynamicSolution:
        """Copy physical fields and evaluate energy and free-face displacement residual."""
        if not self._initialized:
            raise RuntimeError("initialize elastodynamic fields before requesting a solution")
        moments = self._moments(self.displacement)
        residual = float(np.linalg.norm((moments - self.boundary)[self.free]))
        scale = max(
            float(np.linalg.norm(moments)),
            float(np.linalg.norm(self.boundary)),
            sum(
                float(np.linalg.norm(abs(local.coupling).T @ abs(u)))
                for local, u in zip(self.locals, self.displacement, strict=True)
            ),
            np.finfo(float).tiny,
        )
        if residual > 1e-10 * scale:
            raise LinearSolveError(
                "elastodynamic displacement trace constraint fails its physical residual"
            )
        energy = sum(
            float(u @ (local.stiffness @ u) + v @ (local.mass @ v)) / 2
            for local, u, v in zip(self.locals, self.displacement, self.velocity, strict=True)
        )
        return ElastodynamicSolution(
            self.skeleton,
            self.locals,
            tuple(u.copy() for u in self.displacement),
            tuple(v.copy() for v in self.velocity),
            self.trace.copy(),
            self.index * self.time_step,
            energy,
            residual,
        )

    def close(self) -> None:
        """Release reusable mass, Newmark and global trace factors exactly once."""
        self._closed = True
        self._resources.close()

    def __enter__(self) -> ElastodynamicStepper:
        """Enter an open explicit-lifetime elastodynamic integrator."""
        if self._closed:
            raise RuntimeError("elastodynamic stepper is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close numerical resources on normal and exceptional context exit."""
        self.close()


def solve_elastodynamics(
    mesh: TriangleMesh | TetraMesh,
    *,
    time_step: float,
    steps: int,
    displacement: Any = 0.0,
    velocity: Any = 0.0,
    source: Any = 0.0,
    on_step: Any = None,
    **options: Any,
) -> ElastodynamicSolution:
    """Compute a real elastodynamic trajectory using reusable space-time MHM operators."""
    steps = positive_int(steps, "steps", 0)
    with ElastodynamicStepper(mesh, time_step=time_step, **options) as stepper:
        result = stepper.initialize(displacement, velocity)
        if on_step is not None:
            on_step(result)
        for _ in range(steps):
            result = stepper.advance(source)
            if on_step is not None:
                on_step(result)
        return result
