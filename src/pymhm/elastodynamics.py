"""Space-time MHM elasticity with local Newmark propagation on triangles/tetrahedra.

The slabwise constant multiplier is negative physical traction. Local responses
follow Gomes et al. (2017), equations (42)--(48), with beta=1/4 and gamma=1/2.
Independent local substeps end at a common macro time. No incompressible-limit
claim is made for the primal displacement spatial discretization.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import partial
from types import TracebackType
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.darcy3d import TriangularSkeleton
from pymhm.elasticity3d import _KELVIN3, _boundary_vector, constitutive_values_3d
from pymhm.elasticity3d import _local as elastic_local_3d
from pymhm.elasticity_primal import _KELVIN, _local_primal, constitutive_values
from pymhm.elements import boundary_data, triangle_quadrature, vector_values
from pymhm.lagrange import nodal_space, reference_values, tabulate
from pymhm.maxwell_dg import physical_basis, physical_points, quadrature
from pymhm.mesh import FaceSpace, FloatArray, IntArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.parallel import map_local
from pymhm.rad3d import vector_values_3d
from pymhm.solvers import LinearSolveError, factorize
from pymhm.tetrahedral import (
    TetraMesh,
    scalar_values_3d,
    tetra_nodal_space,
    tetra_tabulate,
    tetrahedron_quadrature,
)
from pymhm.triangle_fields import (
    SeparableTriangleField,
    TimeDependentTriangleField,
    TriangleQuadratureField,
    triangle_field_quadrature,
)
from pymhm.vector import _assemble_blocks


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


@dataclass(frozen=True)
class ElastodynamicLocal:
    """Physical mass/stiffness, signed traction moments and continuous local Pk space."""

    mesh: Any
    degree: int
    mass: Any
    stiffness: Any
    coupling: FloatArray
    trace_dofs: IntArray
    dofs: IntArray
    nodes: FloatArray
    basis: FloatArray
    points: FloatArray
    weights: FloatArray
    density_values: FloatArray
    constitutive: Any
    lame_lambda: Any
    lame_mu: Any
    quadrature_order: int

    def load(self, field: Any, *, density_weighted: bool = False) -> FloatArray:
        """Integrate physical force density or a density-weighted initial field.

        Explicit triangular quadrature providers are supported for force loads.
        They are rejected for density-weighted initial projection, which would
        require a verified joint partition of both independent fields.
        """
        dimension = self.nodes.shape[1]
        evaluate = vector_values if dimension == 2 else vector_values_3d
        if isinstance(field, TriangleQuadratureField):
            if density_weighted:
                raise ValueError(
                    "custom source quadrature cannot define a density-weighted initial projection"
                )
            if dimension != 2:
                raise ValueError("custom triangular source quadrature requires two dimensions")
            bary, normalized, data = triangle_field_quadrature(
                self.mesh, field, self.quadrature_order
            )
            if not np.any(normalized):
                return np.zeros(self.mass.shape[0])
            basis = reference_values(self.degree, bary.reshape(-1, 3)).reshape(*bary.shape[:2], -1)
            points = physical_points(self.mesh, bary)
            values = evaluate(data, points.reshape(-1, dimension)).reshape(points.shape)
            weights = normalized * self.mesh.areas[:, None]
            blocks = np.einsum("tq,tqi,tqa->tia", weights, basis, values)
        else:
            values = evaluate(field, self.points.reshape(-1, dimension)).reshape(self.points.shape)
            coefficient = self.density_values if density_weighted else np.ones_like(self.weights)
            blocks = np.einsum("tq,tqi,tqa->tia", self.weights * coefficient, self.basis, values)
        indices = dimension * self.dofs[:, :, None] + np.arange(dimension)
        return np.bincount(indices.ravel(), weights=blocks.ravel(), minlength=self.mass.shape[0])

    def load_at_time(self, source: Any, time: float) -> FloatArray:
        """Evaluate a temporal source while preserving opt-in spatial quadrature.

        Ordinary callbacks retain the existing ``source(time, points)`` path.
        A provider with ``at_time`` returns a spatial quadrature field. Both
        paths supply physical force density, with no implicit density factor.
        """
        if isinstance(source, PreparedElastodynamicSource):
            return source.load_at_time(self, time)
        if isinstance(source, SeparableTriangleField):
            return _scale_source_load(self.load(source.spatial_field()), source.time_scale, time)
        if isinstance(source, TimeDependentTriangleField):
            return self.load(source.at_time(time))
        if isinstance(source, TriangleQuadratureField):
            return self.load(source)
        return self.load(
            source(time, self.points.reshape(-1, self.nodes.shape[1]))
            if callable(source)
            else source
        )


@dataclass(frozen=True)
class PreparedElastodynamicSource:
    """Read-only original spatial force vectors in their executed local nodal bases.

    Loads use binary64 and have no implicit density weighting. Preparation is
    explicit: vectors belong to the exact local objects that integrated the
    source. They cannot be reused on another stepper or approximation space.
    The temporal factor is evaluated at every original endpoint/substep time.
    This snapshot owns no native factorization and imposes no time update.
    """

    locals: tuple[ElastodynamicLocal, ...]
    loads: tuple[FloatArray, ...]
    time_function: Callable[[float], float]
    _indices: dict[int, int] = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and copy original nodal vectors without narrowing their dtype."""
        if (
            not self.locals
            or len(self.locals) != len(self.loads)
            or len({id(local) for local in self.locals}) != len(self.locals)
            or not callable(self.time_function)
        ):
            raise ValueError("provide distinct executed locals, matching loads and a time function")
        loads = []
        for local, raw in zip(self.locals, self.loads, strict=True):
            values = np.asarray(raw)
            if (
                local.nodes.shape[1] != 2
                or values.dtype != np.dtype(float)
                or values.shape != (local.mass.shape[0],)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "prepared loads require finite binary64 original planar nodal vectors"
                )
            owned = values.copy()
            owned.setflags(write=False)
            loads.append(owned)
        object.__setattr__(self, "loads", tuple(loads))
        object.__setattr__(self, "_indices", {id(local): i for i, local in enumerate(self.locals)})

    def load_at_time(self, local: ElastodynamicLocal, time: float) -> FloatArray:
        """Scale the original vector, rejecting a different basis/mesh and nonreal time data."""
        index = self._indices.get(id(local))
        if index is None or self.locals[index] is not local:
            raise ValueError("prepared source belongs to different executed local operators")
        return _scale_source_load(self.loads[index], self.time_function, time)


def _scale_source_load(
    spatial: FloatArray, time_function: Callable[[float], float], time: float
) -> FloatArray:
    """Apply the declared separable temporal factor after the spatial integration."""
    if np.iscomplexobj(time) or np.ndim(time) != 0 or not np.isfinite(time):
        raise ValueError("prepared source time must be finite and real")
    factor = time_function(time)
    if np.iscomplexobj(factor) or np.ndim(factor) != 0 or not np.isfinite(factor):
        raise ValueError("prepared source time scale must be a finite real scalar")
    factor = float(factor)
    if not np.isfinite(factor):
        raise ValueError("prepared source time scale must be finite in binary64")
    with np.errstate(over="ignore", invalid="ignore"):
        load = spatial * factor
    if not np.isfinite(load).all():
        raise ValueError("prepared source scaled load must remain finite")
    return load


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


def _stress_from_gradient(
    local: ElastodynamicLocal, points: FloatArray, gradient: FloatArray
) -> FloatArray:
    """Apply the local Kelvin constitutive law to an already evaluated gradient.

    ``points`` has shape ``(fine cells, quadrature points, dimension)`` and
    ``gradient`` appends its derivative coordinate axis. The caller owns the
    one-sided cell evaluation; no spatial interpolation or averaging occurs.
    """
    dimension = local.nodes.shape[1]
    kelvin, evaluate = (
        (_KELVIN, constitutive_values) if dimension == 2 else (_KELVIN3, constitutive_values_3d)
    )
    stiffness = evaluate(
        local.constitutive,
        points.reshape(-1, dimension),
        lame_lambda=local.lame_lambda,
        lame_mu=local.lame_mu,
    )
    strain = np.einsum("aij,tqij->tqa", kelvin, gradient).reshape(-1, len(kelvin))
    return np.einsum("nab,nb,aij->nij", stiffness, strain, kelvin).reshape(
        *points.shape[:2], dimension, dimension
    )


@dataclass(frozen=True)
class ElastodynamicSolution:
    """Displacement and velocity at a common macro time, and slabwise negative traction."""

    skeleton: Any
    locals: tuple[ElastodynamicLocal, ...]
    displacement: tuple[FloatArray, ...]
    velocity: tuple[FloatArray, ...]
    trace: FloatArray
    time: float
    energy: float
    constraint_residual: float

    def evaluate(self, cell: int, bary: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Evaluate one-sided continuous local polynomials on every fine simplex."""
        index = positive_int(cell, "cell", 0)
        if index >= len(self.locals):
            raise ValueError("cell outside local meshes")
        local = self.locals[index]
        dimension = local.nodes.shape[1]
        result = (
            tabulate(local.mesh, local.degree, bary)
            if dimension == 2
            else tetra_tabulate(local.mesh, local.degree, bary)
        )
        dofs, _, basis = result[:3]
        return tuple(
            np.einsum("qi,tia->tqa", basis, values.reshape(-1, dimension)[dofs])
            for values in (self.displacement[index], self.velocity[index])
        )

    def l2_error(self, exact: Any, *, velocity: bool = False, order: int = 8) -> float:
        """Integrate physical displacement/velocity L2 error with independent quadrature."""
        total = 0.0
        for cell, local in enumerate(self.locals):
            dimension = local.nodes.shape[1]
            bary, weights = (
                triangle_quadrature(order) if dimension == 2 else tetrahedron_quadrature(order)
            )
            numerical = self.evaluate(cell, bary)[int(velocity)]
            points = np.einsum("qi,tij->tqj", bary, local.mesh.points[local.mesh.cells])
            evaluate = vector_values if dimension == 2 else vector_values_3d
            target = evaluate(exact, points.reshape(-1, dimension)).reshape(points.shape)
            measure = local.mesh.areas if dimension == 2 else local.mesh.volumes
            total += float(measure @ (np.sum((numerical - target) ** 2, axis=-1) @ weights))
        return float(np.sqrt(total))

    def gradient(self, cell: int, bary: FloatArray, *, velocity: bool = False) -> FloatArray:
        """Evaluate the full broken displacement or velocity gradient in physical coordinates."""
        index = positive_int(cell, "cell", 0)
        if index >= len(self.locals):
            raise ValueError("cell outside local meshes")
        local = self.locals[index]
        dimension = local.nodes.shape[1]
        forms = (
            tabulate(local.mesh, local.degree, bary)
            if dimension == 2
            else tetra_tabulate(local.mesh, local.degree, bary)
        )
        dofs, _, _, derivative = forms[:4]
        values = self.velocity[index] if velocity else self.displacement[index]
        return np.einsum("tqib,tia->tqab", derivative, values.reshape(-1, dimension)[dofs])

    def stress(self, cell: int, bary: FloatArray) -> FloatArray:
        """Evaluate raw symmetric Cauchy stress; no global H(div) conformity is implied."""
        gradient = self.gradient(cell, bary)
        local = self.locals[cell]
        points = np.einsum("qi,tij->tqj", bary, local.mesh.points[local.mesh.cells])
        return _stress_from_gradient(local, points, gradient)


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
        result = self.factors[index].solve(
            local.mass @ (u + duration * v)
            - duration**2 / 4 * (local.stiffness @ u)
            + duration**2 / 4 * (old + new)
        )
        velocity = v + duration / 2 * self.mass_factors[index].solve(
            old + new - local.stiffness @ (u + result)
        )
        return result, velocity

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
