"""Validated one-sided ownership for large physical point samples on triangular meshes."""

from fractions import Fraction
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pymhm.core.validation import FloatArray, IntArray, positive_int, real_array
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing._sampling_kernels import candidate_barycentric, containment_filter


def _orientation_filter(
    first: FloatArray, second: FloatArray, third: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return signed twice-area and a conservative binary64 roundoff bound.

    Inputs broadcast on their leading axes and have final coordinate axis 2.
    Each determinant term uses two rounded subtractions and one product; the
    final subtraction gives a first-order error of at most four unit-roundoff
    factors times the sum of absolute products. The bound uses 8 machine eps
    (16 unit roundoffs), including higher-order terms. Nonfinite or underflowed
    products use an infinite bound so their sign is resolved with exact binary
    input coordinates instead of assuming a relative-error model is valid.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        a, b = first - third, second - third
        products = np.stack((a[..., 0] * b[..., 1], a[..., 1] * b[..., 0]), axis=-1)
        determinant = products[..., 0] - products[..., 1]
        bound = 8 * np.finfo(float).eps * np.sum(abs(products), axis=-1)
        nonzero = np.stack(
            ((a[..., 0] != 0) & (b[..., 1] != 0), (a[..., 1] != 0) & (b[..., 0] != 0)),
            axis=-1,
        )
        underflow = np.any(nonzero & (abs(products) < np.finfo(float).tiny), axis=-1)
        unreliable = underflow | ~np.isfinite(determinant) | ~np.isfinite(bound)
    return determinant, np.where(unreliable, np.inf, bound)


_ExactPoint = tuple[Fraction, Fraction]


def _exact_orientation(first: _ExactPoint, second: _ExactPoint, third: _ExactPoint) -> Fraction:
    """Return the exact signed determinant of the represented binary coordinates."""
    return (first[0] - third[0]) * (second[1] - third[1]) - (first[1] - third[1]) * (
        second[0] - third[0]
    )


class TrianglePointLocator:
    """Locate points through centroid candidates and actual barycentric containment.

    Candidates are a search acceleration, never a nearest-cell interpolation.
    A vectorized search tests ``candidates`` nearest centroids; unresolved points
    use all cells, including skew or strongly graded meshes. Among valid
    candidates the first cKDTree result is chosen. Shared interfaces therefore
    select one incident side deterministically for the same mesh and query;
    ``coordinates(..., cells=...)`` selects an explicit independent side.
    Containment uses physical orientation predicates with the declared
    nonnegative, dimensionless barycentric tolerance. Roundoff-ambiguous signs
    are resolved exactly for the represented binary coordinates; zero
    tolerance therefore retains vertices and edges while excluding points
    strictly outside. Coordinates are neither moved nor averaged. Returned
    barycentric coordinates retain their original binary64 affine tabulation
    and can have roundoff-size negative entries at a geometrically valid edge.
    A conservative mesh bounding box excludes strictly exterior points, and
    bounded batches keep exhaustive fallback storage independent of query size.
    """

    def __init__(
        self, mesh: TriangleMesh, *, candidates: int = 4, tolerance: float = 1e-10
    ) -> None:
        """Cache the actual affine maps; validate the geometric query convention."""
        if not isinstance(mesh, TriangleMesh):
            raise TypeError("triangle point ownership requires a TriangleMesh")
        count = positive_int(candidates, "candidates")
        if np.iscomplexobj(tolerance) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("barycentric tolerance must be finite and nonnegative")
        self.mesh = mesh
        self.candidates = min(count, len(mesh.cells))
        self.tolerance = float(tolerance)
        vertices = mesh.points[mesh.cells]
        self._vertices = vertices
        self._area, self._area_error = _orientation_filter(
            vertices[:, 1], vertices[:, 2], vertices[:, 0]
        )
        self.origins = vertices[:, 0].copy()
        self.inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        self.tree = cKDTree(vertices.mean(axis=1))
        lower, upper = mesh.points.min(axis=0), mesh.points.max(axis=0)
        # Three barycentric coordinates summing to one and >= -tolerance can
        # extend a coordinate by at most 2*tolerance times its vertex span.
        with np.errstate(over="ignore"):
            margin = 2 * self.tolerance * (upper - lower)
            margin += 8 * np.finfo(float).eps * np.maximum(1, np.maximum(abs(lower), abs(upper)))
            self._lower, self._upper = lower - margin, upper + margin
        self.origins.setflags(write=False)
        self.inverse.setflags(write=False)

    def _points(self, points: Any) -> FloatArray:
        """Validate finite physical planar points without interpreting their field values."""
        value = real_array(points, "evaluation points")
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError("evaluation points must have shape (n,2)")
        return value

    def _barycentric(self, points: FloatArray, candidates: IntArray) -> FloatArray:
        """Apply only the supplied cell maps, retaining the query and candidate axes."""
        return candidate_barycentric(points, candidates, self.origins, self.inverse)

    def _contains(self, points: FloatArray, candidates: IntArray) -> np.ndarray:
        """Test closed triangles with filtered physical determinants and exact fallback."""
        inside, pending_mask = containment_filter(
            points, candidates, self._vertices, self._area, self._area_error, self.tolerance
        )
        pending = np.argwhere(pending_mask)
        exact_cells: dict[int, tuple[tuple[_ExactPoint, ...], Fraction]] = {}
        tolerance = Fraction.from_float(self.tolerance)
        for query, slot in pending:
            cell = int(candidates[query, slot])
            if cell not in exact_cells:
                exact_vertices = tuple(
                    (Fraction.from_float(float(x)), Fraction.from_float(float(y)))
                    for x, y in self._vertices[cell]
                )
                exact_area = _exact_orientation(*exact_vertices)
                exact_cells[cell] = exact_vertices, exact_area
            exact_vertices, exact_area = exact_cells[cell]
            point = (
                Fraction.from_float(float(points[query, 0])),
                Fraction.from_float(float(points[query, 1])),
            )
            sign = 1 if exact_area > 0 else -1
            limit = tolerance * abs(exact_area)
            inside[query, slot] = all(
                sign
                * _exact_orientation(
                    exact_vertices[(i + 1) % 3], exact_vertices[(i + 2) % 3], point
                )
                + limit
                >= 0
                for i in range(3)
            )
        return inside

    def locate(self, points: Any, *, allow_outside: bool = False) -> IntArray:
        """Return one cell per point, optionally marking exterior points with ``-1``.

        By default every point must lie in the mesh within the declared
        barycentric tolerance. ``allow_outside=True`` preserves the query order
        and returns ``-1`` only for points outside every triangle; this supports
        masks when sampling independent subdomains. Exterior indices must be
        filtered before evaluating a cell field.
        """
        if not isinstance(allow_outside, (bool, np.bool_)):
            raise TypeError("allow_outside must be a boolean")
        value = self._points(points)
        exterior = np.any((value < self._lower) | (value > self._upper), axis=1)
        if np.any(exterior) and not allow_outside:
            raise ValueError("evaluation point lies outside the declared triangle mesh")
        result = np.full(len(value), -1, dtype=np.int64)
        active = np.flatnonzero(~exterior)
        if not len(active):
            return result
        candidates = np.asarray(
            self.tree.query(value[active], k=self.candidates)[1], dtype=np.int64
        ).reshape(len(active), self.candidates)
        available = candidates < len(self.mesh.cells)
        candidates = np.where(available, candidates, 0)
        valid = self._contains(value[active], candidates) & available
        result[active] = candidates[np.arange(len(active)), np.argmax(valid, axis=1)]
        unresolved = active[~np.any(valid, axis=1)]
        all_cells = np.arange(len(self.mesh.cells), dtype=np.int64)
        batch_size = max(1, min(256, 65536 // len(all_cells)))
        for begin in range(0, len(unresolved), batch_size):
            queries = unresolved[begin : begin + batch_size]
            complete = np.broadcast_to(all_cells, (len(queries), len(all_cells)))
            inside = self._contains(value[queries], complete)
            found = np.any(inside, axis=1)
            if not np.all(found) and not allow_outside:
                raise ValueError("evaluation point lies outside the declared triangle mesh")
            result[queries] = np.where(found, np.argmax(inside, axis=1), -1)
        return result

    def coordinates(self, points: Any, *, cells: Any = None) -> tuple[IntArray, FloatArray]:
        """Return owners and barycentric triples, optionally preserving declared incident sides."""
        value = self._points(points)
        owners = self.locate(value) if cells is None else np.asarray(cells)
        if (
            owners.shape != (len(value),)
            or owners.dtype.kind not in "iu"
            or np.any(owners < 0)
            or np.any(owners >= len(self.mesh.cells))
        ):
            raise ValueError("provide one valid integer incident cell per evaluation point")
        owners = owners.astype(np.int64, copy=False)
        bary = self._barycentric(value, owners[:, None])[:, 0]
        if not np.all(self._contains(value, owners[:, None])):
            raise ValueError("evaluation point lies outside its declared incident triangle")
        return owners, bary
