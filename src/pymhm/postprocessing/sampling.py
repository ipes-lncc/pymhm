"""Validated one-sided ownership for large physical point samples on triangular meshes."""

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from pymhm.core.validation import FloatArray, IntArray, positive_int, real_array
from pymhm.meshes.triangle import TriangleMesh


class TrianglePointLocator:
    """Locate points through centroid candidates and actual barycentric containment.

    Candidates are a search acceleration, never a nearest-cell interpolation.
    A vectorized search tests ``candidates`` nearest centroids; unresolved points
    use all cells, including skew or strongly graded meshes. Among valid
    candidates the first cKDTree result is chosen. Shared interfaces therefore
    select one incident side deterministically for the same mesh and query;
    ``coordinates(..., cells=...)`` selects an explicit independent side.
    Containment uses the declared nonnegative, dimensionless barycentric
    tolerance. Coordinates are neither moved nor averaged.
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
        self.origins = vertices[:, 0].copy()
        self.inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        self.tree = cKDTree(vertices.mean(axis=1))
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
        local = np.einsum(
            "qkba,qka->qkb", self.inverse[candidates], points[:, None] - self.origins[candidates]
        )
        return np.concatenate(((1 - local.sum(axis=2))[..., None], local), axis=2)

    def locate(self, points: Any) -> IntArray:
        """Return one validated cell per point, rejecting points outside every triangle."""
        value = self._points(points)
        candidates = np.asarray(
            self.tree.query(value, k=self.candidates)[1], dtype=np.int64
        ).reshape(len(value), self.candidates)
        bary = self._barycentric(value, candidates)
        valid = np.min(bary, axis=2) >= -self.tolerance
        result = candidates[np.arange(len(value)), np.argmax(valid, axis=1)]
        for query in np.flatnonzero(~np.any(valid, axis=1)):
            all_cells = np.arange(len(self.mesh.cells), dtype=np.int64)[None]
            barycentric = self._barycentric(value[query : query + 1], all_cells)[0]
            inside = np.flatnonzero(barycentric.min(axis=1) >= -self.tolerance)
            if not len(inside):
                raise ValueError("evaluation point lies outside the declared triangle mesh")
            result[query] = inside[0]
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
        if np.any(bary < -self.tolerance):
            raise ValueError("evaluation point lies outside its declared incident triangle")
        return owners, bary
