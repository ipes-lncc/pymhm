"""Mixed MHM Darcy with tetrahedral BDFM and affine prismatic local spaces."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse

from pymhm.hdiv3d_family import (
    HDiv3DFamily,
    cell_quadrature,
    face_polynomials,
    face_quadrature,
    face_shape,
    face_size,
)
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_basis, hdiv3d_dofs, hdiv3d_face_offsets
from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mapped_rt import _scatter
from pymhm.mesh import FloatArray, IntArray, TriangleMesh, positive_int
from pymhm.rad3d import vector_values_3d
from pymhm.tetrahedral import scalar_values_3d, tensor_values_3d


@dataclass(frozen=True)
class _FacePartition:
    """Affine subfaces with integral-moment-dual physical normal densities."""

    nodes: FloatArray
    cells: IntArray
    degree: int
    measure: float

    @property
    def width(self) -> int:
        """Return the number of polynomial moments per subface."""
        return face_size(self.cells.shape[1], self.degree)

    @property
    def size(self) -> int:
        """Return the subdivided face dimension."""
        return len(self.cells) * self.width

    def coordinates(self, piece: int, points: FloatArray) -> tuple[FloatArray, float]:
        """Map parent coordinates into one selected affine subface."""
        vertices = self.nodes[self.cells[piece]]
        jac = (vertices[1:3] - vertices[:1]).T
        return (points - vertices[0]) @ np.linalg.inv(jac).T, abs(float(np.linalg.det(jac)))

    def locate(self, points: FloatArray) -> int:
        """Locate a contained fine face, rejecting trace partitions that cut it."""
        for piece in range(len(self.cells)):
            uv, _ = self.coordinates(piece, points)
            upper = np.max(uv.sum(axis=1)) if self.cells.shape[1] == 3 else np.max(uv)
            if np.min(uv) >= -1e-10 and upper <= 1 + 1e-10:
                return piece
        raise ValueError("skeleton subfaces must align with local normal-flux faces")

    def evaluate(self, piece: int, points: FloatArray) -> FloatArray:
        """Evaluate physical normal densities dual to subface polynomial moments."""
        uv, det = self.coordinates(piece, points)
        q, w = face_quadrature(self.cells.shape[1], max(3, self.degree + 2))
        tests = face_polynomials(q, self.cells.shape[1], self.degree)
        mass = tests.T @ (w[:, None] * tests)
        return (
            face_polynomials(uv, self.cells.shape[1], self.degree)
            @ np.linalg.inv(mass)
            / (self.measure * det)
        )


def _partition(corners: int, subdivisions: int, degree: int, measure: float) -> _FacePartition:
    """Create uniform subfaces without replacing original macroface identities."""
    n = subdivisions
    if corners == 3:
        triangle = TriangleMesh(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
        ).submesh(0, n)
        return _FacePartition(triangle.points, triangle.cells, degree, measure)
    nodes = np.array([(i / n, j / n) for i in range(n + 1) for j in range(n + 1)])
    cells = np.array(
        [
            [i * (n + 1) + j, (i + 1) * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j + 1]
            for i in range(n)
            for j in range(n)
        ]
    )
    return _FacePartition(nodes, cells, degree, measure)


class Mixed3DSkeleton:
    """Pk triangular and Qk rectangular physical flux traces on original macrofaces."""

    def __init__(self, mesh: AffineMixedMesh, degree: int = 1, subdivisions: int = 1) -> None:
        """Create canonical moment blocks for every face and aligned subdivision."""
        self.degree = positive_int(degree, "trace degree", 0)
        self.subdivisions = positive_int(subdivisions, "subdivisions")
        self.mesh = mesh
        self.partitions = tuple(
            _partition(len(face), subdivisions, degree, mesh.measures[i])
            for i, face in enumerate(mesh.faces)
        )
        self.offsets = np.r_[0, np.cumsum([part.size for part in self.partitions])]

    @property
    def size(self) -> int:
        """Return the global number of scalar normal-flux moments."""
        return int(self.offsets[-1])

    def cell_dofs(self, cell: int) -> IntArray:
        """Return the concatenated face blocks of one macrocell."""
        return np.concatenate(
            [np.arange(self.offsets[f], self.offsets[f + 1]) for f in self.mesh.cell_faces[cell]]
        )


def hdiv3d_operators(
    mesh: AffineMixedMesh,
    family: HDiv3DFamily,
    *,
    permeability: Any = 1.0,
    source: Any = 0.0,
    quadrature_order: int = 5,
) -> tuple:
    """Assemble conforming mass, divergence, source and physical pressure moments."""
    points, weights = cell_quadrature(mesh.kind, max(quadrature_order, family.pressure_degree + 2))
    dofs = hdiv3d_dofs(mesh, family)
    basis, div, pressure = hdiv3d_basis(mesh, family, points)
    physical = mesh.geometry(points)
    inverse = np.linalg.inv(tensor_values_3d(permeability, physical.reshape(-1, 3))).reshape(
        *physical.shape[:2], 3, 3
    )
    w = mesh.determinants[:, None] * weights
    mass = np.einsum("tq,tqia,tqab,tqjb->tij", w, basis, inverse, basis, optimize=True)
    divergence = np.einsum("tq,qi,tqj->tij", w, pressure, div, optimize=True)
    f = scalar_values_3d(source, physical.reshape(-1, 3)).reshape(w.shape)
    load = np.einsum("tq,qi,tq->ti", w, pressure, f)
    moment = np.einsum("tq,qi->ti", w, pressure)
    nq, npres = int(dofs.max()) + 1, len(mesh.cells) * family.pressure_size
    pids = np.arange(npres).reshape(len(mesh.cells), -1)
    return (
        _scatter(mass, dofs, dofs, (nq, nq)),
        _scatter(divergence, pids, dofs, (npres, nq)),
        load.ravel(),
        moment.ravel(),
    )


def _trace_mapping(
    skeleton: Mixed3DSkeleton, cell: int, fine: AffineMixedMesh, normal_degree: int = 1
) -> FloatArray:
    """Integrate canonical fine-face moments of each oriented macro normal density."""
    mesh = skeleton.mesh
    coarse_faces = mesh.cell_faces[cell]
    offsets = np.r_[0, np.cumsum([skeleton.partitions[f].size for f in coarse_faces])]
    mapping = np.zeros(
        (
            sum(face_size(len(fine.faces[f]), normal_degree) for f in fine.boundary_faces),
            offsets[-1],
        )
    )
    row = 0
    for face in fine.boundary_faces:
        nodes = fine.points[fine.faces[face]]
        count = face_size(len(nodes), normal_degree)
        uv, w = face_quadrature(len(nodes), max(4, normal_degree + 2))
        physical = face_shape(uv, len(nodes)) @ nodes
        found = False
        for side, parent in enumerate(coarse_faces):
            origin = mesh.points[mesh.faces[parent][0]]
            if np.max(abs((nodes - origin) @ mesh.normals[parent])) > 1e-10 * np.linalg.norm(
                mesh.jacobian[cell]
            ):
                continue
            canonical = mesh.face_coordinates(parent, physical)
            partition = skeleton.partitions[parent]
            piece = partition.locate(mesh.face_coordinates(parent, nodes))
            normal = partition.evaluate(piece, canonical)
            block = face_polynomials(uv, len(nodes), normal_degree).T @ (
                w[:, None] * normal * fine.measures[face] * mesh.signs[cell, side]
            )
            start = offsets[side] + piece * partition.width
            mapping[row : row + count, start : start + partition.width] = block
            found = True
            break
        if not found:
            raise ValueError("local boundary face has no containing macroface")
        row += count
    return mapping


def _boundary(
    skeleton: Mixed3DSkeleton, dirichlet: Any, neumann: dict[int, Any], order: int
) -> tuple:
    """Integrate weak pressure or outward physical normal-flux moments at the boundary."""
    mesh = skeleton.mesh
    if not set(neumann).issubset(set(mesh.boundary_faces)):
        raise ValueError("Neumann keys must identify exterior macrofaces")
    load, fixed = np.zeros(skeleton.size), {}
    for face in mesh.boundary_faces:
        partition = skeleton.partitions[face]
        uv, w = face_quadrature(len(mesh.faces[face]), order)
        tests = face_polynomials(uv, len(mesh.faces[face]), skeleton.degree)
        for piece, cell in enumerate(partition.cells):
            nodes = partition.nodes[cell]
            canonical = face_shape(uv, len(cell)) @ nodes
            physical = face_shape(canonical, len(cell)) @ mesh.points[mesh.faces[face]]
            _, det = partition.coordinates(piece, canonical)
            indices = skeleton.offsets[face] + piece * partition.width + np.arange(partition.width)
            if face in neumann:
                value = tests.T @ (
                    w * det * partition.measure * scalar_values_3d(neumann[face], physical)
                )
                fixed.update({int(i): float(v) for i, v in zip(indices, value, strict=True)})
            else:
                load[indices] -= partition.evaluate(piece, canonical).T @ (
                    w * det * partition.measure * scalar_values_3d(dirichlet, physical)
                )
    return load, fixed


@dataclass(frozen=True)
class _LocalFactory:
    """Assemble one mixed Neumann problem and its physical constant pressure mode."""

    skeleton: Mixed3DSkeleton
    family: HDiv3DFamily
    refinement: int
    permeability: Any
    source: Any
    order: int

    def __call__(self, cell: int) -> LocalAssembly:
        """Return the physical saddle, trace map and dimensional pressure constraint."""
        fine = self.skeleton.mesh.submesh(cell, self.refinement)
        M, D, f, moment = hdiv3d_operators(
            fine,
            self.family,
            permeability=self.permeability,
            source=self.source,
            quadrature_order=self.order,
        )
        nq, npres = M.shape[0], len(f)
        offsets = hdiv3d_face_offsets(fine, self.family)
        ids = np.concatenate(
            [np.arange(offsets[face], offsets[face + 1]) for face in fine.boundary_faces]
        )
        nb = len(ids)
        selector = sparse.coo_matrix((np.ones(nb), (ids, np.arange(nb))), shape=(nq, nb)).tocsc()
        zero = sparse.csc_matrix((npres, nb))
        A = sparse.bmat(
            [[M, -D.T, selector], [-D, None, zero], [selector.T, zero.T, None]], format="csc"
        )
        mapping = _trace_mapping(self.skeleton, cell, fine, self.family.normal_degree)
        B = np.zeros((nq + npres + nb, mapping.shape[1]))
        B[-nb:] = -mapping
        constant = np.zeros((len(fine.cells), self.family.pressure_size))
        constant[:, 0] = 1
        boundary = np.zeros(nb)
        cursor = 0
        for face in fine.boundary_faces:
            boundary[cursor] = 1
            cursor += face_size(len(fine.faces[face]), self.family.normal_degree)
        kernel = np.r_[np.zeros(nq), constant.ravel(), boundary][:, None]
        moment = np.r_[np.zeros(nq), moment, np.zeros(nb)]
        scale = np.sqrt(float(np.max(M.diagonal())))
        scaling = np.r_[np.full(nq, 1 / scale), np.full(npres + nb, scale)]
        transform = sparse.diags(scaling)
        problem = LocalProblem(
            transform @ A @ transform,
            scaling[:, None] * B,
            scaling * np.r_[np.zeros(nq), -f, np.zeros(nb)],
            self.skeleton.cell_dofs(cell),
            kernel=kernel / scaling[:, None],
            constraints=(moment * scaling)[:, None],
        )
        return LocalAssembly(problem, (fine, nq, npres, scaling, moment * scaling))


@dataclass(frozen=True)
class Mixed3DDarcySolution:
    """Affine H(div) flux and discontinuous pressure with their MHM skeleton."""

    skeleton: Mixed3DSkeleton
    family: HDiv3DFamily
    local_meshes: tuple[AffineMixedMesh, ...]
    pressure: tuple[FloatArray, ...]
    flux: tuple[FloatArray, ...]
    hybrid: HybridSolution
    source: Any
    physical_residuals: FloatArray

    def evaluate(self, cell: int, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate pressure, physical flux and divergence in every local cell."""
        mesh = self.local_meshes[cell]
        basis, div, p = hdiv3d_basis(mesh, self.family, points)
        local = self.flux[cell][hdiv3d_dofs(mesh, self.family)]
        return (
            self.pressure[cell] @ p.T,
            np.einsum("tqia,ti->tqa", basis, local),
            np.einsum("tqi,ti->tq", div, local),
        )

    def errors(self, pressure: Any, flux: Any, order: int = 6) -> dict[str, float]:
        """Integrate physical pressure/vector-flux L2 errors with positive quadrature."""
        points, w = cell_quadrature(self.family.kind, order)
        errors = np.zeros(2)
        for cell, mesh in enumerate(self.local_meshes):
            physical = mesh.geometry(points)
            p, q, _ = self.evaluate(cell, points)
            pe = scalar_values_3d(pressure, physical.reshape(-1, 3)).reshape(p.shape)
            qe = vector_values_3d(flux, physical.reshape(-1, 3)).reshape(q.shape)
            errors += [
                np.sum(mesh.determinants[:, None] * w * (p - pe) ** 2),
                np.sum(mesh.determinants[:, None] * w * np.sum((q - qe) ** 2, axis=2)),
            ]
        return dict(pressure_l2=float(np.sqrt(errors[0])), flux_l2=float(np.sqrt(errors[1])))

    def equilibrium_residuals(self, order: int = 6) -> tuple[FloatArray, ...]:
        """Return every pressure-tested fine-cell divergence/source defect."""
        points, w = cell_quadrature(self.family.kind, order)
        p = self.family.tabulate(points)[2]
        result = []
        for cell, mesh in enumerate(self.local_meshes):
            div = self.evaluate(cell, points)[2]
            f = scalar_values_3d(self.source, mesh.geometry(points).reshape(-1, 3)).reshape(
                div.shape
            )
            result.append(np.einsum("t,q,qi,tq->ti", mesh.determinants, w, p, div - f))
        return tuple(result)


def solve_darcy_hdiv3d(
    mesh: AffineMixedMesh,
    *,
    pressure_degree: int = 1,
    normal_degree: int = 1,
    trace_degree: int = 1,
    subdivisions: int = 1,
    local_refinement: int = 2,
    permeability: Any = 1.0,
    source: Any = 0.0,
    dirichlet: Any = 0.0,
    neumann: dict[int, Any] | None = None,
    mean_pressure: float = 0.0,
    quadrature_order: int = 5,
    boundary_quadrature_order: int | None = None,
    solver: str = "scipy",
    local_solver: str = "scipy",
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int | None = None,
    global_rtol: float = 1e-10,
    global_refinement_precision: Literal["double", "extended"] = "double",
) -> Mixed3DDarcySolution:
    """Solve Darcy with independent normal order and interior mixed enrichment.

    Pressure degree 1 selects tetrahedral BDFM with 18 modes or prism flux with
    27 modes. Tetrahedral pressure degree 2 retains every zero-normal P3 bubble
    (32 modes) at the default normal degree one. General admissible orders are
    those of HDiv3DFamily. Pressure uses ordinary composition;
    flux uses contravariant Piola on affine cells. With trace_degree=normal_degree and
    subdivisions=local_refinement the trace is the complete classical fine-grid
    conforming space. Neumann values are physical outward normal flux; pressure
    is otherwise imposed weakly. Materials must be resolved by the local mesh
    or the supplied quadrature; coefficient-interface fitting is not implicit.
    """
    family = HDiv3DFamily(mesh.kind, pressure_degree, normal_degree)
    r = positive_int(local_refinement, "local refinement")
    skeleton = Mixed3DSkeleton(mesh, trace_degree, subdivisions)
    if trace_degree > normal_degree:
        raise ValueError("trace degree must not exceed local normal degree")
    if r % subdivisions:
        raise ValueError("trace subdivisions must divide local refinement")
    order = max(positive_int(quadrature_order, "quadrature order"), pressure_degree + 2)
    neumann = {} if neumann is None else neumann
    boundary_order = (
        order
        if boundary_quadrature_order is None
        else positive_int(boundary_quadrature_order, "boundary quadrature order")
    )
    load, fixed = _boundary(skeleton, dirichlet, neumann, boundary_order)
    system = HybridSystem.from_local_factory(
        _LocalFactory(skeleton, family, r, permeability, source, order),
        range(len(mesh.cells)),
        boundary_load=load,
        local_solver=local_solver,
        backend=backend,
        workers=workers,
    )
    gauges = None
    if set(neumann) == set(mesh.boundary_faces):
        if not np.isfinite(mean_pressure):
            raise ValueError("mean pressure must be finite")
        gauges = [
            system.mean_constraint(
                [data[4] for data in system.local_metadata],
                float(mean_pressure) * sum(mesh.volumes),
            )
        ]
    hybrid = system.solve(
        solver=solver,
        fixed=fixed,
        constraints=gauges,
        rtol=global_rtol,
        refinement_precision=global_refinement_precision,
    )
    pressure, flux, residuals = [], [], []
    for response, data, field in zip(
        system.responses, system.local_metadata, hybrid.fields, strict=True
    ):
        fine, nq, npres, scale, _ = data
        problem = response.problem
        trace = hybrid.trace[problem.trace_dofs]
        defect = (problem.matrix @ field + problem.coupling @ trace - problem.load) / scale
        action = (
            abs(problem.matrix) @ abs(field)
            + abs(problem.coupling) @ abs(trace)
            + abs(problem.load)
        ) / scale
        residual = [
            float(
                np.linalg.norm(defect[part])
                / max(np.linalg.norm(action[part]), np.finfo(float).tiny)
            )
            for part in (slice(0, nq), slice(nq, nq + npres), slice(nq + npres, None))
        ]
        if max(residual) > 1e-10:
            raise ValueError(
                "physical mixed block residual exceeds the declared backward-error criterion"
            )
        field = scale * field
        flux.append(field[:nq])
        pressure.append(field[nq : nq + npres].reshape(len(fine.cells), -1))
        residuals.append(residual)
    return Mixed3DDarcySolution(
        skeleton,
        family,
        tuple(data[0] for data in system.local_metadata),
        tuple(pressure),
        tuple(flux),
        hybrid,
        source,
        np.array(residuals),
    )
