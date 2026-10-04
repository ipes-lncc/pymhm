"""Three-dimensional mixed MHM on trilinear hexahedra with contravariant RT Piola maps.

Flux DOFs are integral tensor-Legendre normal moments. Skeleton unknowns are
reference-face flux densities, not physical polynomials on distorted faces.
Pressure uses the ordinary scalar pullback Q_k. On a nonaffine cell,
``div(RT_k) = Q_k / det(J)``; its physical divergence need not belong to the
scalar pressure space, while all tested pressure moments remain conservative.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.core.contracts import HybridSolution, LocalAssembly, LocalProblem
from pymhm.core.system import HybridSystem
from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.assembly import (
    assemble_element_blocks as _scatter,
)
from pymhm.fem.hdiv.mapped import (
    _modal as _modal,
)
from pymhm.fem.hdiv.mapped import (
    _reference_rt_map as _reference_rt_map,
)
from pymhm.fem.hdiv.mapped import (
    mapped_rt_basis as mapped_rt_basis,
)
from pymhm.fem.hdiv.mapped import (
    mapped_rt_dofs as mapped_rt_dofs,
)
from pymhm.materials.evaluation import scalar_values_3d, tensor_values_3d, vector_values_3d
from pymhm.meshes.hexahedron import (
    _CORNERS as _CORNERS,
)
from pymhm.meshes.hexahedron import (
    _FACE_CORNERS as _FACE_CORNERS,
)
from pymhm.meshes.hexahedron import (
    _UV as _UV,
)
from pymhm.meshes.hexahedron import (
    HexMesh as HexMesh,
)
from pymhm.meshes.hexahedron import (
    QuadratureOrder as QuadratureOrder,
)
from pymhm.meshes.hexahedron import (
    _face_coordinate_transform as _face_coordinate_transform,
)
from pymhm.meshes.hexahedron import (
    _geometry as _geometry,
)
from pymhm.meshes.hexahedron import (
    _points as _points,
)
from pymhm.meshes.hexahedron import (
    cube_quadrature as cube_quadrature,
)

_QUADRATURE_BATCH_ENTRIES = 2_000_000


def _operators(
    mesh: HexMesh, degree: int, permeability: Any, source: Any, order: QuadratureOrder
) -> tuple:
    """Assemble mixed operators with bounded quadrature batches and physical material values."""
    points, w = cube_quadrature(order)
    dofs = mapped_rt_dofs(mesh, degree)
    cells, width = dofs.shape
    pressure_width = (degree + 1) ** 3
    mass = np.zeros((cells, width, width))
    divergence = np.zeros((cells, pressure_width, width))
    load = np.zeros((cells, pressure_width))
    mean = np.zeros_like(load)
    # Bound the largest physical-basis arrays independently of the quadrature order.
    for part in _quadrature_slices(mesh, degree, len(points)):
        subset = points[part]
        physical, _, det = mesh.geometry(subset)
        basis, div, pressure = mapped_rt_basis(mesh, degree, subset)
        inverse = np.linalg.inv(tensor_values_3d(permeability, physical.reshape(-1, 3))).reshape(
            *det.shape, 3, 3
        )
        weights = det * w[part]
        weighted = np.einsum("tq,tqab,tqjb->tqja", weights, inverse, basis, optimize=True)
        mass += np.einsum("tqia,tqja->tij", basis, weighted, optimize=True)
        divergence += np.einsum("tq,qi,tqj->tij", weights, pressure, div, optimize=True)
        f = scalar_values_3d(source, physical.reshape(-1, 3)).reshape(det.shape)
        load += np.einsum("tq,qi,tq->ti", weights, pressure, f, optimize=True)
        mean += np.einsum("tq,qi->ti", weights, pressure, optimize=True)
    nq, npres = int(dofs.max()) + 1, cells * pressure_width
    pids = np.arange(npres).reshape(len(mesh.cells), -1)
    return (
        _scatter(mass, dofs, dofs, (nq, nq)),
        _scatter(divergence, pids, dofs, (npres, nq)),
        load.ravel(),
        mean.ravel(),
    )


def _quadrature_slices(mesh: HexMesh, degree: int, count: int) -> Iterator[slice]:
    """Partition one unchanged rule to bound simultaneous physical-basis storage."""
    width = 3 * (degree + 2) * (degree + 1) ** 2
    batch = max(1, _QUADRATURE_BATCH_ENTRIES // (len(mesh.cells) * width * 3))
    for start in range(0, count, batch):
        yield slice(start, start + batch)


@dataclass(frozen=True)
class HexSkeleton:
    """Tensor-Qk reference flux-density traces with aligned uniform face subdivisions."""

    mesh: HexMesh
    degree: int = 1
    subdivisions: int = 1

    def __post_init__(self) -> None:
        """Validate scalar polynomial degree and positive subdivision count."""
        positive_int(self.degree, "trace degree", 0)
        positive_int(self.subdivisions, "trace subdivisions")

    @property
    def face_size(self) -> int:
        """Return the number of reference flux-density modes on each macroface."""
        return self.subdivisions**2 * (self.degree + 1) ** 2

    @property
    def size(self) -> int:
        """Return the global number of scalar normal-flux trace modes."""
        return len(self.mesh.faces) * self.face_size

    def cell_dofs(self, cell: int) -> IntArray:
        """Return the six oriented macroface blocks in local side order."""
        return (
            self.face_size * self.mesh.cell_faces[cell, :, None] + np.arange(self.face_size)
        ).ravel()

    def evaluate(self, points: FloatArray) -> FloatArray:
        """Evaluate reference flux densities, normalized by segment moments."""
        n = self.subdivisions
        indices = np.minimum(np.floor(points * n).astype(int), n - 1)
        local = points * n - indices
        values = _modal(self.degree, local)
        factors = np.array(
            [(2 * i + 1) * (2 * j + 1) for i, j in product(range(self.degree + 1), repeat=2)]
        )
        result = np.zeros((len(points), self.face_size))
        start = (indices[:, 0] * n + indices[:, 1]) * (self.degree + 1) ** 2
        result[np.arange(len(points))[:, None], start[:, None] + np.arange(values.shape[1])] = (
            n * n * values * factors
        )
        return result


def _trace_map(
    mesh: HexMesh,
    cell: int,
    fine: HexMesh,
    reference: FloatArray,
    skeleton: HexSkeleton,
    degree: int,
) -> FloatArray:
    """Integrate fine normal moments of each oriented mapped macroface trace."""
    uv, weights = cube_quadrature(degree + skeleton.degree + 2, 2)
    count = (degree + 1) ** 2
    tests = _modal(degree, uv)
    shape = np.prod(np.where(_UV[None], uv[:, None], 1 - uv[:, None]), axis=2)
    result = np.zeros((count * len(fine.boundary_faces), 6 * skeleton.face_size))
    for row, face in enumerate(fine.boundary_faces):
        nodes = reference[fine.faces[face]]
        axis = int(np.flatnonzero(np.ptp(nodes, axis=0) < 1e-13)[0])
        end = int(round(nodes[0, axis]))
        side = 2 * axis + end
        parent_uv = (shape @ nodes)[:, np.arange(3) != axis]
        canonical = (
            np.column_stack((np.ones(len(uv)), parent_uv)) @ mesh.face_transforms[cell, side]
        )
        area = np.prod(np.ptp(nodes[:, np.arange(3) != axis], axis=0))
        result[
            row * count : (row + 1) * count,
            side * skeleton.face_size : (side + 1) * skeleton.face_size,
        ] = (
            mesh.signs[cell, side]
            * area
            * tests.T
            @ (weights[:, None] * skeleton.evaluate(canonical))
        )
    return result


def _boundary(
    skeleton: HexSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple[FloatArray, dict[int, float]]:
    """Integrate weak pressure data and project outward physical normal fluxes."""
    mesh = skeleton.mesh
    if not set(neumann).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    load, fixed = np.zeros(skeleton.size), {}
    uv, weights = cube_quadrature(order, 2)
    n = skeleton.subdivisions
    for face in mesh.boundary_faces:
        cell, side = mesh.incidence[face][0]
        axis, end = divmod(side, 2)
        transform = mesh.face_transforms[cell, side]
        for i, j in product(range(n), repeat=2):
            canonical = (uv + np.array([i, j])) / n
            local = (canonical - transform[0]) @ np.linalg.inv(transform[1:])
            points = np.empty((len(uv), 3))
            points[:, axis], points[:, np.arange(3) != axis] = end, local
            physical, jac, det = _geometry(mesh.points[mesh.cells[cell]][None], points)
            normal = det[0, :, None] * np.linalg.inv(jac[0]).transpose(0, 2, 1)[:, :, axis]
            measure = np.linalg.norm(normal, axis=1)
            trace = skeleton.evaluate(canonical)
            indices = face * skeleton.face_size + np.arange(skeleton.face_size)
            if face not in neumann:
                p = scalar_values_3d(dirichlet, physical[0])
                load[indices] -= trace.T @ (weights * p / n**2)
            else:
                flux = scalar_values_3d(neumann[face], physical[0])
                value = _modal(skeleton.degree, uv).T @ (weights * flux * measure / n**2)
                start = (i * n + j) * (skeleton.degree + 1) ** 2
                for a, v in enumerate(value):
                    fixed[int(indices[start + a])] = float(v)
    return load, fixed


@dataclass(frozen=True)
class _Factory:
    """Build one mixed Neumann local problem with a physical pressure mean constraint."""

    mesh: HexMesh
    skeleton: HexSkeleton
    degree: int
    refinement: int
    permeability: Any
    source: Any
    order: QuadratureOrder

    def __call__(self, cell: int) -> LocalAssembly:
        """Assemble a flux/pressure/boundary-pressure saddle and its joint constant kernel."""
        fine, reference = self.mesh.submesh(cell, self.refinement)
        M, D, f, moment = _operators(fine, self.degree, self.permeability, self.source, self.order)
        nq, npres = M.shape[0], len(f)
        count = (self.degree + 1) ** 2
        nb = count * len(fine.boundary_faces)
        indices = (count * fine.boundary_faces[:, None] + np.arange(count)).ravel()
        selector = sparse.coo_matrix(
            (np.ones(nb), (indices, np.arange(nb))), shape=(nq, nb)
        ).tocsc()
        zero = sparse.csc_matrix((npres, nb))
        A = sparse.bmat(
            [[M, -D.T, selector], [-D, None, zero], [selector.T, zero.T, None]], format="csc"
        )
        mapping = _trace_map(self.mesh, cell, fine, reference, self.skeleton, self.degree)
        B = np.zeros((nq + npres + nb, mapping.shape[1]))
        B[-nb:] = -mapping
        constant = np.zeros((len(fine.cells), (self.degree + 1) ** 3))
        constant[:, 0] = 1
        boundary = np.zeros((len(fine.boundary_faces), count))
        boundary[:, 0] = 1
        kernel = np.r_[np.zeros(nq), constant.ravel(), boundary.ravel()][:, None]
        weights = np.r_[np.zeros(nq), moment, np.zeros(nb)]
        # A symmetric change of physical units balances flux and pressure; it
        # leaves the Schur complement and physical pressure moment unchanged.
        scale = float(np.sqrt(np.max(M.diagonal())))
        scaling = np.r_[np.full(nq, 1 / scale), np.full(npres + nb, scale)]
        transform = sparse.diags(scaling)
        problem = LocalProblem(
            transform @ A @ transform,
            scaling[:, None] * B,
            scaling * np.r_[np.zeros(nq), -f, np.zeros(nb)],
            self.skeleton.cell_dofs(cell),
            kernel=kernel / scaling[:, None],
            constraints=(weights * scaling)[:, None],
        )
        return LocalAssembly(problem, (fine, reference, weights * scaling, nq, npres, scaling))


@dataclass(frozen=True)
class MappedRTDarcySolution:
    """Mapped H(div) flux, discontinuous scalar-pullback pressure and condensed macro traces."""

    skeleton: HexSkeleton
    local_meshes: tuple[HexMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    degree: int
    permeability: Any
    source: Any
    quadrature_order: QuadratureOrder
    physical_residuals: FloatArray

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate pressure, physical flux and exact Piola divergence in every local hexahedron."""
        mesh = self.local_meshes[cell]
        basis, div, p = mapped_rt_basis(mesh, self.degree, points)
        coefficients = self.flux[cell][mapped_rt_dofs(mesh, self.degree)]
        return (
            self.pressure[cell] @ p.T,
            np.einsum("tqia,ti->tqa", basis, coefficients),
            np.einsum("tqi,ti->tq", div, coefficients),
        )

    def errors(self, pressure: Any, flux: Any, order: QuadratureOrder = 6) -> dict[str, float]:
        """Integrate physical pressure/vector-flux L2 errors with independent quadrature."""
        points, weights = cube_quadrature(order)
        errors = np.zeros(2)
        for cell, mesh in enumerate(self.local_meshes):
            for part in _quadrature_slices(mesh, self.degree, len(points)):
                physical, _, det = mesh.geometry(points[part])
                p, q, _ = self.evaluate(cell, points[part])
                pe = scalar_values_3d(pressure, physical.reshape(-1, 3)).reshape(det.shape)
                qe = vector_values_3d(flux, physical.reshape(-1, 3)).reshape(*det.shape, 3)
                errors += [
                    np.sum(det * weights[part] * (p - pe) ** 2),
                    np.sum(det * weights[part] * np.sum((q - qe) ** 2, axis=2)),
                ]
        return dict(pressure_l2=float(np.sqrt(errors[0])), flux_l2=float(np.sqrt(errors[1])))

    def equilibrium_residuals(self) -> tuple[FloatArray, ...]:
        """Return every pressure-tested physical divergence/source defect, including cell mass."""
        points, weights = cube_quadrature(self.quadrature_order)
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            moments = np.zeros((len(mesh.cells), (self.degree + 1) ** 3))
            for part in _quadrature_slices(mesh, self.degree, len(points)):
                physical, _, det = mesh.geometry(points[part])
                divergence = self.evaluate(cell, points[part])[2]
                f = scalar_values_3d(self.source, physical.reshape(-1, 3)).reshape(det.shape)
                p = _modal(self.degree, points[part])
                moments += np.einsum("tq,qi,tq->ti", det * weights[part], p, divergence - f)
            result.append(moments)
        return tuple(result)


def solve_darcy_mapped_rt(
    mesh: HexMesh,
    *,
    degree: int = 1,
    trace_degree: int | None = None,
    subdivisions: int = 1,
    local_refinement: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: QuadratureOrder = 5,
    solver: str = "scipy",
    local_solver: str = "scipy",
    global_rtol: float = 1e-10,
    global_refinement_precision: Literal["double", "extended"] = "double",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
) -> MappedRTDarcySolution:
    """Solve a three-dimensional mixed MHM problem using mapped tensor RT_k/Q_k locals.

    Reference flux-density traces have Q_l degree l<=k, with aligned uniform
    subfaces. ``subdivisions=local_refinement`` and l=k yields the classical
    globally conforming fine-mesh mixed operator after local condensation.
    Boundary Neumann values are physical outward flux densities. Discontinuous
    coefficients must be resolved by the geometric cells or supplied quadrature.
    The volume rule may use a tuple of three positive Gauss orders; boundary
    integration then uses their maximum in each face direction. Quadrature is
    accumulated in bounded batches without modifying its points or weights.
    ``global_rtol`` controls the retained saddle's linear solve independently
    of the fixed 1e-10 physical block criterion. Explicit extended refinement
    retains correction digits in global variables and reconstructed fields;
    it requires a wider long-double type and does not guarantee that the
    physical criterion can be reached for every ill-conditioned problem.
    """
    if np.iscomplexobj(global_rtol) or not np.isfinite(global_rtol) or global_rtol <= 0:
        raise ValueError("global_rtol must be finite and positive")
    if global_refinement_precision not in {"double", "extended"}:
        raise ValueError("global_refinement_precision must be double or extended")
    k = positive_int(degree, "RT degree", 0)
    r = positive_int(local_refinement, "local refinement")
    skeleton = HexSkeleton(mesh, k if trace_degree is None else trace_degree, subdivisions)
    if skeleton.degree > k or r % skeleton.subdivisions:
        raise ValueError(
            "trace degree must not exceed RT degree and subdivisions must divide refinement"
        )
    orders = quadrature_order if isinstance(quadrature_order, tuple) else (quadrature_order,) * 3
    if len(orders) != 3:
        raise ValueError("volume quadrature needs exactly three axis orders")
    validated = tuple(max(positive_int(v, "quadrature order"), k + 2) for v in orders)
    order = validated if isinstance(quadrature_order, tuple) else validated[0]
    neumann = {} if neumann is None else neumann
    boundary, fixed = _boundary(skeleton, dirichlet, neumann, max(validated))
    system = HybridSystem.from_local_factory(
        _Factory(mesh, skeleton, k, r, permeability, source, order),
        range(len(mesh.cells)),
        boundary_load=boundary,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = None
    if set(neumann) == set(mesh.boundary_faces):
        mean = float(mean_pressure)
        if not np.isfinite(mean):
            raise ValueError("mean pressure must be finite")
        moments = [data[2] for data in system.local_metadata]
        # Only the first pressure basis integrates to volume on affine cells;
        # geometric moments of higher modes must not be included in that sum.
        volume = sum(
            np.sum(
                _geometry(mesh.points[mesh.cells[i]][None], cube_quadrature(order)[0])[2]
                * cube_quadrature(order)[1]
            )
            for i in range(len(mesh.cells))
        )
        gauges = [system.mean_constraint(moments, mean * volume)]
    hybrid = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=gauges,
        rtol=global_rtol,
        refinement_precision=global_refinement_precision,
    )
    pressure, flux, residuals = [], [], []
    for response, metadata, local_field in zip(
        system.responses, system.local_metadata, hybrid.fields, strict=True
    ):
        fine, _, _, nq, npres, scaling = metadata
        problem = response.problem
        trace = hybrid.trace[problem.trace_dofs]
        defect = (problem.matrix @ local_field + problem.coupling @ trace - problem.load) / scaling
        action = (
            abs(problem.matrix) @ abs(local_field)
            + abs(problem.coupling) @ abs(trace)
            + abs(problem.load)
        ) / scaling
        block_residuals = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        if max(block_residuals) > 1e-10:
            raise ValueError(
                "physical mixed block residual exceeds the declared backward-error criterion"
            )
        residuals.append(block_residuals)
        local_field = scaling * local_field
        flux.append(local_field[:nq])
        pressure.append(local_field[nq : nq + npres].reshape(len(fine.cells), -1))
    return MappedRTDarcySolution(
        skeleton,
        tuple(data[0] for data in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        hybrid,
        k,
        permeability,
        source,
        order,
        np.array(residuals),
    )
