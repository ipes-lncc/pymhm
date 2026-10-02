"""Three-dimensional mixed MHM on trilinear hexahedra with contravariant RT Piola maps.

Flux DOFs are integral tensor-Legendre normal moments. Skeleton unknowns are
reference-face flux densities, not physical polynomials on distorted faces.
Pressure uses the ordinary scalar pullback Q_k. On a nonaffine cell,
``div(RT_k) = Q_k / det(J)``; its physical divergence need not belong to the
scalar pressure space, while all tested pressure moments remain conservative.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Any, Literal

import numpy as np
from numpy.polynomial.legendre import Legendre, leggauss, legvander
from scipy import sparse

from pymhm.hybrid import HybridSolution, HybridSystem, LocalAssembly, LocalProblem
from pymhm.mesh import FloatArray, IntArray, positive_int
from pymhm.rad3d import vector_values_3d
from pymhm.tetrahedral import scalar_values_3d, tensor_values_3d

_CORNERS = np.array(list(product((0, 1), repeat=3)))
_FACE_CORNERS = np.array([np.flatnonzero(_CORNERS[:, a] == b) for a in range(3) for b in (0, 1)])
_UV = np.array(list(product((0, 1), repeat=2)))
_QUADRATURE_BATCH_ENTRIES = 2_000_000
QuadratureOrder = int | tuple[int, ...]


def cube_quadrature(order: QuadratureOrder, dimension: int = 3) -> tuple[FloatArray, FloatArray]:
    """Return tensor Gauss points/weights, with optional separate orders per axis."""
    orders = order if isinstance(order, tuple) else (order,) * dimension
    if len(orders) != dimension:
        raise ValueError("quadrature orders must match the integration dimension")
    rules = [leggauss(positive_int(n, "quadrature order")) for n in orders]
    indices = np.array(list(product(*(range(len(x)) for x, _ in rules))))
    points = np.column_stack([(x[indices[:, a]] + 1) / 2 for a, (x, _) in enumerate(rules)])
    weights = np.prod(
        np.column_stack([w[indices[:, a]] / 2 for a, (_, w) in enumerate(rules)]), axis=1
    )
    return points, weights


def _points(values: Any) -> FloatArray:
    """Validate finite unit-cube reference coordinates with a trailing triple."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError("reference coordinates must be finite real triples")
    if np.any(raw < -1e-13) or np.any(raw > 1 + 1e-13):
        raise ValueError("reference coordinates must lie in the unit cube")
    return np.asarray(raw, dtype=float)


def _geometry(
    vertices: FloatArray, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate a trilinear geometric map, Jacobian and positive determinant."""
    factors = np.where(_CORNERS[None], points[:, None], 1 - points[:, None])
    shape = factors.prod(axis=2)
    gradients = np.stack(
        [
            (2 * _CORNERS[:, a] - 1) * np.prod(factors[:, :, np.arange(3) != a], axis=2)
            for a in range(3)
        ],
        axis=2,
    )
    physical = np.einsum("qi,tia->tqa", shape, vertices)
    jacobian = np.einsum("qib,tia->tqab", gradients, vertices)
    determinant = np.linalg.det(jacobian)
    if np.any(determinant <= 0):
        raise ValueError("hexahedral map requires a positive Jacobian at all sampled points")
    return physical, jacobian, determinant


@lru_cache(maxsize=24)
def _face_coordinate_transform(permutation: tuple[int, ...]) -> FloatArray:
    """Cache the immutable integer affine map of a tensor face corner ordering.

    The same least-squares compatibility check rejects the sixteen non-tensor
    permutations. Eight square symmetries remain; their rounded maps give the
    exact existing trace coordinates and determinant orientation.
    """
    canonical = _UV[np.asarray(permutation)]
    coordinates = np.column_stack((np.ones(4), _UV))
    transform = np.linalg.lstsq(coordinates, canonical, rcond=None)[0]
    if not np.allclose(coordinates @ transform, canonical, atol=1e-13):
        raise ValueError("shared facets require compatible tensor corner orderings")
    rounded = np.rint(transform)
    rounded.setflags(write=False)
    return rounded


@dataclass(frozen=True)
class HexMesh:
    """Conforming trilinear hexahedra, with lexicographic (x,y,z) reference corners.

    Geometric positivity is checked at corners and a 3×3×3 Gauss rule and again
    at every requested evaluation point. This is a sampled validity check, not
    a global proof of injectivity for arbitrary warped input cells.
    """

    points: FloatArray
    cells: IntArray
    faces: IntArray = field(init=False)
    cell_faces: IntArray = field(init=False)
    signs: FloatArray = field(init=False)
    face_transforms: FloatArray = field(init=False)
    incidence: tuple[tuple[tuple[int, int], ...], ...] = field(init=False)
    boundary_faces: IntArray = field(init=False)

    def __post_init__(self) -> None:
        """Build oriented shared quadrilateral facets and check basic mesh validity."""
        points, cells = np.asarray(self.points), np.asarray(self.cells)
        if (
            np.iscomplexobj(points)
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
            or cells.ndim != 2
            or cells.shape[1] != 8
            or not np.issubdtype(cells.dtype, np.integer)
            or len(cells) == 0
            or np.any(cells < 0)
            or np.any(cells >= len(points))
        ):
            raise ValueError(
                "hexahedral mesh requires finite points and integer eight-vertex cells"
            )
        if any(len(set(cell)) != 8 for cell in cells):
            raise ValueError("hexahedral cells require eight distinct vertices")
        object.__setattr__(self, "points", np.array(points, dtype=float, copy=True))
        object.__setattr__(self, "cells", np.array(cells, dtype=np.int64, copy=True))
        facets: list[IntArray] = []
        lookup: dict[tuple, int] = {}
        incidence: list[list[tuple[int, int]]] = []
        cell_faces = np.empty((len(cells), 6), dtype=np.int64)
        signs = np.ones((len(cells), 6))
        transforms = np.empty((len(cells), 6, 3, 2))
        for cell, nodes in enumerate(cells):
            for side, corners in enumerate(_FACE_CORNERS):
                local = nodes[corners]
                key = tuple(sorted(local))
                if key not in lookup:
                    lookup[key] = len(facets)
                    facets.append(local)
                    incidence.append([])
                face = lookup[key]
                if len(incidence[face]) == 2:
                    raise ValueError("nonmanifold hexahedral facet")
                signs[cell, side] = 1 if not incidence[face] else -1
                incidence[face].append((cell, side))
                cell_faces[cell, side] = face
                permutation = tuple(int(np.flatnonzero(facets[face] == node)[0]) for node in local)
                transform = _face_coordinate_transform(permutation)
                transforms[cell, side] = np.rint(transform)
                if len(incidence[face]) == 2:
                    _, first_side = incidence[face][0]
                    first_axis, first_end = divmod(first_side, 2)
                    first_orientation = (2 * first_end - 1) * (-1) ** first_axis
                    orientation = (
                        (2 * (side % 2) - 1)
                        * (-1) ** (side // 2)
                        * round(np.linalg.det(transform[1:]))
                    )
                    if orientation == first_orientation:
                        raise ValueError(
                            "neighboring cell facets must have opposite outward orientations"
                        )
        object.__setattr__(self, "faces", np.array(facets, dtype=np.int64))
        object.__setattr__(self, "cell_faces", cell_faces)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "face_transforms", transforms)
        object.__setattr__(self, "incidence", tuple(tuple(v) for v in incidence))
        object.__setattr__(
            self,
            "boundary_faces",
            np.array([i for i, v in enumerate(incidence) if len(v) == 1], dtype=np.int64),
        )

        self.geometry(np.vstack((_CORNERS, cube_quadrature(3)[0])))

    def geometry(self, points: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Evaluate physical coordinates, Jacobians and determinants in every cell."""
        return _geometry(self.points[self.cells], _points(points))

    def submesh(self, cell: int, refinement: int) -> tuple["HexMesh", FloatArray]:
        """Subdivide one reference cube and retain the parent coordinates of all fine vertices."""
        r = positive_int(refinement, "local refinement")
        reference = np.array(list(product(np.linspace(0, 1, r + 1), repeat=3)))
        physical = _geometry(self.points[self.cells[cell]][None], reference)[0][0]
        cells = np.array(
            [
                [(i + a) * (r + 1) ** 2 + (j + b) * (r + 1) + k + c for a, b, c in _CORNERS]
                for i, j, k in product(range(r), repeat=3)
            ]
        )
        return HexMesh(physical, cells), reference

    def refined(self, subdivisions: int | tuple[int, int, int]) -> "HexMesh":
        """Refine every macro map with topological vertex merging and unchanged geometry.

        Vertices are identified by their exact rational trilinear weights on
        original mesh nodes, so neighboring faces share points without geometric
        rounding, nearest-neighbor searches or an implicit tolerance.
        Separate axis counts are accepted when their face resolutions agree
        across every oriented shared face; nonconforming subdivisions are rejected.
        """
        values = subdivisions if isinstance(subdivisions, tuple) else (subdivisions,) * 3
        if len(values) != 3:
            raise ValueError("hexahedral refinement requires three axis counts")
        counts = np.array([positive_int(v, "hexahedral subdivisions") for v in values])
        for attached in self.incidence:
            resolutions = []
            for cell, side in attached:
                tangent_counts = counts[np.arange(3) != side // 2]
                resolutions.append(abs(self.face_transforms[cell, side, 1:]).T @ tangent_counts)
            if len(resolutions) == 2 and not np.array_equal(resolutions[0], resolutions[1]):
                raise ValueError("anisotropic refinement must match across oriented shared faces")
        lattice = np.array(list(product(*(range(r + 1) for r in counts))))
        numerators = np.prod(
            np.where(_CORNERS[None], lattice[:, None], counts - lattice[:, None]), axis=2
        )
        vertices: list[FloatArray] = []
        cells: list[IntArray] = []
        lookup: dict[tuple, int] = {}
        pattern = np.array(
            [
                [
                    (i + a) * (counts[1] + 1) * (counts[2] + 1) + (j + b) * (counts[2] + 1) + k + c
                    for a, b, c in _CORNERS
                ]
                for i, j, k in product(*(range(r) for r in counts))
            ]
        )
        for nodes in self.cells:
            ids = []
            for weights in numerators:
                key = tuple(
                    sorted(
                        (int(node), int(weight))
                        for node, weight in zip(nodes, weights, strict=True)
                        if weight
                    )
                )
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append(weights @ self.points[nodes] / np.prod(counts))
                ids.append(lookup[key])
            cells.extend(np.array(ids)[pattern])
        return HexMesh(np.array(vertices), np.array(cells))

    @classmethod
    def unit_cube(cls, subdivisions: int = 1) -> "HexMesh":
        """Create a conforming uniform unit-cube grid."""
        base = cls(_CORNERS.astype(float), np.arange(8)[None])
        return base.submesh(0, subdivisions)[0]

    @classmethod
    def annular_prism(cls, radii: FloatArray, height: float, sectors: int = 8) -> "HexMesh":
        """Extrude a polygonal annulus, with explicitly supplied graded radial coordinates.

        This geometry has planar polygonal walls and trilinear nonaffine cells;
        it does not replace them by exact cylindrical surfaces. Height is centered
        at z=0, and each radius contains ``sectors`` equally spaced vertices.
        """
        raw = np.asarray(radii)
        n = positive_int(sectors, "azimuthal sectors", 3)
        if (
            np.iscomplexobj(raw)
            or raw.ndim != 1
            or len(raw) < 2
            or not np.isfinite(raw).all()
            or np.any(raw <= 0)
            or np.any(np.diff(raw) <= 0)
            or not np.isfinite(height)
            or height <= 0
        ):
            raise ValueError("annular radii must increase positively and height must be positive")
        angles = 2 * np.pi * np.arange(n) / n
        points = np.array(
            [
                [r * np.cos(theta), r * np.sin(theta), z]
                for r in raw
                for theta in angles
                for z in (-height / 2, height / 2)
            ]
        )
        cells = np.array(
            [
                [(i + a) * n * 2 + ((j + b) % n) * 2 + c for a, b, c in _CORNERS]
                for i in range(len(raw) - 1)
                for j in range(n)
            ]
        )
        return cls(points, cells)


def _modal(degree: int, points: FloatArray) -> FloatArray:
    """Evaluate tensor Legendre polynomials, with the last coordinate varying fastest."""
    tables = [legvander(2 * points[:, a] - 1, degree) for a in range(points.shape[1])]
    return np.column_stack(
        [
            np.prod([tables[a][:, index[a]] for a in range(len(tables))], axis=0)
            for index in product(range(degree + 1), repeat=len(tables))
        ]
    )


def mapped_rt_dofs(mesh: HexMesh, degree: int) -> IntArray:
    """Map shared integral normal moments and independent cell-interior bubbles."""
    k = positive_int(degree, "RT degree", 0)
    count, interior = (k + 1) ** 2, 3 * k * (k + 1) ** 2
    face = (count * mesh.cell_faces[:, :, None] + np.arange(count)).reshape(len(mesh.cells), -1)
    inner = (
        count * len(mesh.faces)
        + interior * np.arange(len(mesh.cells))[:, None]
        + np.arange(interior)
    )
    return np.column_stack((face, inner))


def mapped_rt_basis(
    mesh: HexMesh, degree: int, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Tabulate oriented physical RT_k vectors, their exact Piola divergence and scalar Q_k."""
    k = positive_int(degree, "RT degree", 0)
    points = _points(points)
    _, jacobian, determinant = mesh.geometry(points)
    count = (k + 1) ** 2
    width = 3 * (k + 2) * count
    reference = np.zeros((len(mesh.cells), len(points), width, 3))
    divergence = np.zeros(reference.shape[:-1])
    normalizers = np.array([(2 * i + 1) * (2 * j + 1) for i, j in product(range(k + 1), repeat=2)])
    for side in range(6):
        axis, end = divmod(side, 2)
        uv = np.column_stack((np.ones(len(points)), points[:, np.arange(3) != axis]))
        canonical = np.einsum("qi,tia->tqa", uv, mesh.face_transforms[:, side])
        face = _modal(k, canonical.reshape(-1, 2)).reshape(len(mesh.cells), len(points), count)
        face *= normalizers * mesh.signs[:, side, None, None]
        indices = slice(side * count, (side + 1) * count)
        reference[:, :, indices, axis] = (points[:, axis] - (1 - end))[None, :, None] * face
        divergence[:, :, indices] = face
    offset = 6 * count
    for axis in range(3):
        for index in product(*(range(k) if a == axis else range(k + 1) for a in range(3))):
            t = points[:, axis]
            polynomial = np.ones(len(t))
            for a in range(3):
                polynomial *= Legendre.basis(index[a])(2 * points[:, a] - 1)
            transverse = np.ones(len(t))
            for a in range(3):
                if a != axis:
                    transverse *= Legendre.basis(index[a])(2 * points[:, a] - 1)
            derivative = 2 * Legendre.basis(index[axis]).deriv()(2 * t - 1) * transverse
            reference[:, :, offset, axis] = t * (1 - t) * polynomial
            divergence[:, :, offset] = (1 - 2 * t) * polynomial + t * (1 - t) * derivative
            offset += 1
    values = np.einsum("tqab,tqib->tqia", jacobian, reference) / determinant[:, :, None, None]
    return values, divergence / determinant[:, :, None], _modal(k, points)


def _scatter(blocks: FloatArray, rows: IntArray, columns: IntArray, shape: tuple[int, int]) -> Any:
    """Assemble dense element blocks into a sparse global operator."""
    return sparse.coo_matrix(
        (
            blocks.ravel(),
            (
                np.repeat(rows, columns.shape[1], axis=1).ravel(),
                np.tile(columns, (1, rows.shape[1])).ravel(),
            ),
        ),
        shape=shape,
    ).tocsc()


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
