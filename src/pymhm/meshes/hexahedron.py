"""Trilinear hexahedral geometry, oriented tensor faces and exact subdivisions.

Reference corners are lexicographic in (x,y,z). Geometric validity uses the
same sampled positive-Jacobian convention as the physical map evaluations.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

from pymhm.core.validation import FloatArray, IntArray, positive_int
from pymhm.fem.reference import tensor_lagrange_tabulation

_CORNERS = np.array(list(product((0, 1), repeat=3)))


_FACE_CORNERS = np.array([np.flatnonzero(_CORNERS[:, a] == b) for a in range(3) for b in (0, 1)])


_UV = np.array(list(product((0, 1), repeat=2)))


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


def hexahedral_mapping(
    vertices: FloatArray, points: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate batched trilinear maps from the unit cube to physical hexahedra.

    ``vertices`` has shape ``(cells, 8, 3)`` with corners in lexicographic
    ``(x, y, z)`` order; ``points`` has shape ``(quadrature_points, 3)``.
    Coordinates outside ``[0, 1]^3`` evaluate the same trilinear extension,
    which permits inverse-map iterations. Return physical points
    ``(cells, points, 3)``, Jacobians
    ``(cells, points, 3, 3)`` and determinants ``(cells, points)``. Jacobian
    rows are physical coordinates and columns are reference coordinates.
    Determinants must be positive at every supplied point; this sampled check
    does not certify invertibility throughout a curved cell.
    """
    vertices = np.asarray(vertices)
    if (
        vertices.ndim != 3
        or vertices.shape[1:] != (8, 3)
        or np.iscomplexobj(vertices)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("vertices must be finite real arrays with shape (cells, 8, 3)")
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or np.iscomplexobj(points)
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("reference coordinates must be finite real triples")
    shape, gradients = tensor_lagrange_tabulation("hexahedron", 1, points, nodes=_CORNERS, nderiv=1)
    physical = np.einsum("qi,tia->tqa", shape, vertices)
    jacobian = np.einsum("qib,tia->tqab", gradients, vertices)
    determinant = np.linalg.det(jacobian)
    if np.any(determinant <= 0):
        raise ValueError("hexahedral map requires a positive Jacobian at all sampled points")
    return physical, jacobian, determinant


_geometry = hexahedral_mapping


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
        """Create a uniform grid with exact rational geometric corner weights.

        Mesh vertices use topological interpolation rather than floating-point
        finite-element tabulation, preserving the prescribed exterior planes.
        """
        base = cls(_CORNERS.astype(float), np.arange(8)[None])
        return base.refined(subdivisions)

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
