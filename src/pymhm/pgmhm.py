"""Petrov-Galerkin MHM with residual face enrichment for two-dimensional Darcy.

Equations (27)--(34) of Fernando, Martins, Pereira and Valentin (2023) use
the conormal A grad(p).n. Here all stored multipliers use physical Darcy flux
q.n=-A grad(p).n. Only the enriched multiplier is macro conservative.
"""

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import sparse

from pymhm.darcy import DarcySolution, _DarcyLocalFactory
from pymhm.element_backends import legendre_values
from pymhm.elements import boundary_data, scalar_values, tensor_values
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly
from pymhm.lagrange import nodal_space
from pymhm.mesh import FloatArray, IntArray, SkeletonSpace, TriangleMesh, positive_int
from pymhm.mh import _ellipticity
from pymhm.polygon import PolygonMesh
from pymhm.scalar_boundary import PreparedScalarTrace, edge_basis, prepare_scalar_trace
from pymhm.scalar_trace_integration import incident_face_breaks
from pymhm.solvers import solve_linear


def _trace_matrix(
    macro: Any,
    fine: TriangleMesh,
    face: int,
    degree: int,
    parameter: FloatArray,
    prepared: PreparedScalarTrace | None = None,
) -> Any:
    """Represent one-sided local nodal traces as a sparse evaluation operator."""
    count = len(nodal_space(fine, degree)[1])
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    covered = np.zeros(len(parameter), dtype=bool)
    prepared = prepare_scalar_trace(macro, fine, face, degree) if prepared is None else prepared
    for positions, ids, candidates in prepared.selections(parameter):
        selected = candidates[~covered[candidates]]
        local = (parameter[selected] - positions[0]) / (positions[1] - positions[0])
        basis = edge_basis(degree, local)
        rows.extend(np.repeat(selected, len(ids)))
        columns.extend(np.tile(ids, len(selected)))
        values.extend(basis.ravel())
        covered[selected] = True
    if not covered.all():
        raise ValueError("fine boundary does not cover the requested macroface")
    return sparse.coo_matrix((values, (rows, columns)), shape=(len(parameter), count)).tocsr()


@dataclass(frozen=True)
class _FacePenalty:
    """Common face quadrature, signed jumps and projected Dirichlet moments."""

    face: int
    parameter: FloatArray
    weights: FloatArray
    coefficient: float
    indices: IntArray
    jump: FloatArray
    source_jump: FloatArray
    prescribed: FloatArray
    breaks: FloatArray
    projection: FloatArray
    sides: tuple[tuple[int, float, Any], ...]
    traces: tuple[PreparedScalarTrace, ...]

    def boundary_value(self, parameter: FloatArray) -> FloatArray:
        """Evaluate the piecewise Pk orthogonal boundary projection used in enrichment."""
        owners = np.clip(
            np.searchsorted(self.breaks, parameter, side="right") - 1, 0, len(self.breaks) - 2
        )
        t = 2 * (parameter - self.breaks[owners]) / np.diff(self.breaks)[owners] - 1
        return np.einsum(
            "qi,qi->q", legendre_values(t, self.projection.shape[1] - 1), self.projection[owners]
        )


@dataclass(frozen=True)
class _PGFactory:
    """Reuse the original Darcy local assembly and enforce the declared ellipticity bound."""

    darcy: _DarcyLocalFactory
    lower: float

    def __call__(self, cell: int) -> LocalAssembly:
        """Return the unchanged mean-constrained Darcy problem and its physical metadata."""
        item = self.darcy(cell)
        fine = item.metadata[0]
        minimum = np.linalg.eigvalsh(tensor_values(self.darcy.permeability, fine.points)).min()
        if minimum < self.lower * (1 - 64 * np.finfo(float).eps):
            raise ValueError("ellipticity_lower_bound exceeds a sampled material eigenvalue")
        return item


def _penalty(
    system: HybridSystem,
    skeleton: SkeletonSpace,
    face: int,
    degree: int,
    order: int,
    alpha: float,
    lower: float,
    dirichlet: Any,
) -> _FacePenalty:
    """Assemble one macroface jump on the common incident fine-edge partition."""
    mesh = skeleton.mesh
    neighbors = mesh.face_cells[face]
    neighbors = neighbors[neighbors >= 0]
    breaks = incident_face_breaks(
        skeleton,
        {int(cell): system.local_metadata[cell][0] for cell in neighbors},
        face,
        degree,
    )
    gauss, weights = leggauss(max(order, degree + 1))
    parameter = (breaks[:-1, None] + (gauss + 1) / 2 * np.diff(breaks)[:, None]).ravel()
    measure = (np.diff(breaks)[:, None] * weights / 2 * mesh.lengths[face]).ravel()
    ids = []
    for cell in neighbors:
        ids.extend(system.responses[cell].problem.trace_dofs)
        ids.append(system.kernel_offsets[cell])
    indices = np.unique(ids).astype(np.int64)
    dtype = np.result_type(*(system.responses[cell].source.dtype for cell in neighbors))
    jump = np.zeros((len(parameter), len(indices)), dtype=dtype)
    source_jump = np.zeros(len(parameter), dtype=dtype)
    sides, traces = [], []
    for cell in neighbors:
        response = system.responses[cell]
        fine = system.local_metadata[cell][0]
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        sign = float(mesh.signs[cell, side])
        prepared = prepare_scalar_trace(mesh, fine, face, degree)
        evaluation = _trace_matrix(mesh, fine, face, degree, parameter, prepared)
        local_ids = np.r_[response.problem.trace_dofs, system.kernel_offsets[cell]]
        derivative = np.column_stack((-response.lifts, response.retained_basis))
        jump[:, np.searchsorted(indices, local_ids)] += sign * (evaluation @ derivative)
        source_jump += sign * (evaluation @ response.source)
        sides.append((int(cell), sign, evaluation))
        traces.append(prepared)
    projection = np.zeros((len(breaks) - 1, degree + 1))
    if len(neighbors) == 1:
        start, end = mesh.points[mesh.faces[face]]
        points = start + parameter[:, None] * (end - start)
        values = scalar_values(dirichlet, points).reshape(-1, len(gauss))
        basis = legendre_values(gauss, degree)
        projection = (values * weights / 2) @ basis * (2 * np.arange(degree + 1) + 1)
        prescribed = (projection @ basis.T).ravel()
    else:
        prescribed = np.zeros(len(parameter))
    return _FacePenalty(
        face,
        parameter,
        measure,
        alpha * lower / (2 * mesh.lengths[face]),
        indices,
        jump,
        source_jump,
        prescribed,
        breaks,
        projection,
        tuple(sides),
        tuple(traces),
    )


@dataclass(frozen=True)
class PGMHMSolution:
    """Base and enriched Darcy fields of the residual-based Petrov-Galerkin method.

    ``pressure`` is equation (31); ``enriched_pressure`` is equation (34).
    ``hybrid.trace`` stores the unenriched physical flux multiplier. Physical
    macro conservation requires ``normal_flux(..., enriched=True)``.
    Both volume flux fields are raw gradients, not H(div) reconstructions.
    """

    skeleton: SkeletonSpace
    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[FloatArray, ...]
    enriched_pressure: tuple[FloatArray, ...]
    hybrid: HybridSolution
    system: HybridSystem
    permeability: Any
    source: Any
    degree: int
    quadrature_order: int
    stabilization_parameter: float
    ellipticity_lower_bound: float
    penalties: tuple[_FacePenalty, ...]
    enrichment_loads: tuple[FloatArray, ...]

    def _fields(self, enriched: bool) -> DarcySolution:
        """Reuse only physical volume norm integration with the selected pressure field."""
        values = self.enriched_pressure if enriched else self.pressure
        return DarcySolution(
            self.skeleton,
            self.local_meshes,
            values,
            tuple(np.empty(0) for _ in values),
            self.hybrid,
            "primal",
            self.permeability,
            self.source,
            self.quadrature_order,
            self.degree,
        )

    def l2_error(self, exact: Any, order: int = 8, *, enriched: bool = False) -> float:
        """Integrate base or enriched pressure error in the physical L2 norm."""
        return self._fields(enriched).l2_error(exact, order)

    def flux_l2_error(self, exact: Any, order: int = 8, *, enriched: bool = False) -> float:
        """Integrate -K grad(p) error with the selected base or enriched pressure."""
        return self._fields(enriched).flux_l2_error(exact, order)

    def normal_flux(
        self, cell: int, face: int, parameter: Any, *, enriched: bool = True
    ) -> FloatArray:
        """Evaluate one-sided outward flux, including the residual enrichment by default."""
        mesh = self.skeleton.mesh
        if face not in mesh.cell_faces[cell]:
            raise ValueError("face must be incident to the supplied macrocell")
        t = np.asarray(parameter, dtype=float)
        if t.ndim != 1 or not np.isfinite(t).all() or np.any((t < 0) | (t > 1)):
            raise ValueError("face parameters must be finite points in [0,1]")
        result = self.skeleton.faces[face].evaluate(t) @ self.hybrid.trace[self.skeleton.dofs(face)]
        data = next((item for item in self.penalties if item.face == face), None)
        if enriched and data is not None:
            jump = np.zeros(len(t))
            for (neighbor, sign, _), trace in zip(data.sides, data.traces, strict=True):
                jump += sign * trace.evaluate(self.pressure[neighbor], t)
            result -= data.coefficient * (jump - data.boundary_value(t))
        side = int(np.flatnonzero(mesh.cell_faces[cell] == face)[0])
        return mesh.signs[cell, side] * result

    def conservation_residuals(self, *, enriched: bool = True) -> FloatArray:
        """Return integral(q.n)-integral(f); only the enriched multiplier conserves macros."""
        residuals = []
        for response, extra in zip(self.system.responses, self.enrichment_loads, strict=True):
            p = response.problem
            residual = p.coupling @ self.hybrid.trace[p.trace_dofs] - p.load
            if enriched:
                residual = residual + extra
            residuals.append(np.sum(residual))
        return np.asarray(residuals)


def solve_pgmhm(
    mesh: TriangleMesh | PolygonMesh,
    *,
    stabilization_parameter: float,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    skeleton: SkeletonSpace | None = None,
    degree: int = 2,
    local_refinement: int = 1,
    local_meshes: tuple[TriangleMesh, ...] | None = None,
    quadrature_order: int = 6,
    ellipticity_lower_bound: float | None = None,
    mean_pressure: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    local_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> PGMHMSolution:
    """Solve equations (27)--(34), including the conservative residual enrichment.

    ``stabilization_parameter=alpha`` is explicit: the theorem requires alpha
    sufficiently small but does not provide a universal numerical threshold.
    The weight is alpha*K_min/(2*H_E), using the full macroface length, not
    the lengths of refined skeleton segments. The local degree must satisfy
    k>=ell+2. The stated uniform theorem assumes ell>=1; ell=0 is included
    among the article's numerical experiments. Rank/residual checks establish
    neither a uniform inf-sup bound nor parameter-independent conditioning.

    Dirichlet faces contribute jumps against their prescribed trace. Physical
    Neumann faces are fixed essentially and receive no jump penalty or flux
    enrichment. Compatible pure Neumann data use the physical mean-pressure
    gauge; the local enrichment has zero volume mean. The boundary projection
    in equation (32) is evaluated on the common fine-trace partition so that
    its Pk moments agree with the Dirichlet functional in equation (29).
    ``local_refinement_precision="extended"`` retains residual-correction
    digits in both the local response and its residual enrichment. It leaves
    the operator and residual tolerances unchanged and requires a platform
    with a wider long-double type, as in :func:`pymhm.solvers.solve_linear`.
    """
    if not isinstance(mesh, (TriangleMesh, PolygonMesh)):
        raise TypeError("PGMHM requires triangular or polygonal two-dimensional macrocells")
    if local_refinement_precision not in {"double", "extended"}:
        raise ValueError("local_refinement_precision must be double or extended")
    degree = positive_int(degree, "degree")
    refinement = positive_int(local_refinement, "local_refinement")
    order = max(positive_int(quadrature_order, "quadrature_order"), degree + 2)
    alpha = np.asarray(stabilization_parameter)
    if alpha.shape != () or np.iscomplexobj(alpha) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("stabilization_parameter must be a finite positive scalar")
    lower = _ellipticity(permeability, ellipticity_lower_bound)
    geometry = cast(TriangleMesh, mesh)
    skeleton = SkeletonSpace(geometry) if skeleton is None else skeleton
    if skeleton.mesh is not mesh or skeleton.components != 1:
        raise ValueError("PGMHM requires a scalar skeleton on the supplied mesh")
    if degree < max(max(space.degrees) for space in skeleton.faces) + 2:
        raise ValueError("PGMHM requires local degree k >= trace degree ell + 2")
    _, fixed = boundary_data(skeleton, 0.0, neumann, order=order)
    neumann_faces = set() if neumann is None else set(neumann)
    pure_neumann = neumann_faces == set(mesh.boundary_faces)
    if not np.isfinite(mean_pressure) or (mean_pressure != 0 and not pure_neumann):
        raise ValueError("a finite nonzero mean_pressure is only valid for pure Neumann data")
    if local_meshes is not None:
        from pymhm.refinement import validate_submesh

        if len(local_meshes) != len(mesh.cells):
            raise ValueError("one local mesh is required for each macrocell")
        for cell, fine in enumerate(local_meshes):
            validate_submesh(geometry, cell, fine)
    factory = _PGFactory(
        _DarcyLocalFactory(
            geometry,
            skeleton,
            permeability,
            source,
            tuple(np.empty((0, 3)) for _ in mesh.cells),
            refinement,
            local_meshes,
            degree,
            "primal",
            order,
        ),
        lower,
    )
    system = HybridSystem.from_local_factory(
        factory,
        range(len(mesh.cells)),
        local_solver=local_solver,
        local_refinement_precision=local_refinement_precision,
        backend=backend,
        workers=workers,
    )
    penalties = tuple(
        _penalty(system, skeleton, face, degree, order, float(alpha), lower, dirichlet)
        for face in range(len(mesh.faces))
        if face not in neumann_faces
    )
    rows: list[int] = []
    columns: list[int] = []
    entries: list[float] = []
    for data in penalties:
        if len(data.sides) == 1:
            # Both appearances of g use the same common fine-trace partition.
            # In particular, continuous piecewise polynomial data may have a
            # derivative jump within a macroface and cannot use one Gauss rule.
            boundary = skeleton.faces[data.face].evaluate(data.parameter).T @ (
                data.weights * data.prescribed
            )
            dofs = skeleton.dofs(data.face)
            system.rhs[dofs] -= boundary
            system.load_scale[dofs] += abs(boundary)
        weighted = data.weights * data.coefficient
        block = data.jump.T @ (weighted[:, None] * data.jump)
        load = data.jump.T @ (weighted * (data.prescribed - data.source_jump))
        rows.extend(np.repeat(data.indices, len(data.indices)))
        columns.extend(np.tile(data.indices, len(data.indices)))
        entries.extend(block.ravel())
        np.add.at(system.rhs, data.indices, load)
        np.add.at(system.load_scale, data.indices, abs(load))
    system.matrix = (
        system.matrix
        + sparse.coo_matrix((entries, (rows, columns)), shape=system.matrix.shape).tocsc()
    )
    constraints = None
    if pure_neumann:
        weights = [item[1] for item in system.local_metadata]
        constraints = [system.mean_constraint(weights, mean_pressure * float(np.sum(mesh.areas)))]
    hybrid = system.solve(solver=solver, fixed=fixed, constraints=constraints)
    coefficients = np.r_[hybrid.trace, np.concatenate(hybrid.coarse)]
    extra = [
        np.zeros(len(response.problem.load), dtype=base.dtype)
        for response, base in zip(system.responses, hybrid.fields, strict=True)
    ]
    for data in penalties:
        jump = data.jump @ coefficients[data.indices] + data.source_jump - data.prescribed
        flux_enrichment = -data.coefficient * jump
        for cell, sign, evaluation in data.sides:
            extra[cell] += sign * (evaluation.T @ (data.weights * flux_enrichment))
    enriched = []
    for response, base, load in zip(system.responses, hybrid.fields, extra, strict=True):
        problem = response.problem.with_load(-load)
        operator, rhs = problem.condensation_system()
        correction = solve_linear(
            operator,
            rhs[:, 0],
            solver=local_solver,
            refinement_precision=local_refinement_precision,
        )[: len(load)]
        enriched.append(base + correction)
    return PGMHMSolution(
        skeleton,
        tuple(item[0] for item in system.local_metadata),
        hybrid.fields,
        tuple(enriched),
        hybrid,
        system,
        permeability,
        source,
        degree,
        order,
        float(alpha),
        lower,
        penalties,
        tuple(extra),
    )
