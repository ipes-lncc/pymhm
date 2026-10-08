"""Replay crisscross P1 fields and integrate on their exact common partition."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from examples.mh2m_heterogeneous_norms import overlay_triangles
from pymhm.meshes.geometry import clip_polygon

PATTERN = np.array(
    [
        [[0, 0], [1, 0], [0.5, 0.5]],
        [[1, 0], [1, 1], [0.5, 0.5]],
        [[1, 1], [0, 1], [0.5, 0.5]],
        [[0, 1], [0, 0], [0.5, 0.5]],
    ],
    dtype=float,
)
INVERSE = np.linalg.inv(np.concatenate((np.ones((4, 3, 1)), PATTERN), axis=2))


def _quadrants(xy: np.ndarray) -> np.ndarray:
    """Return south/east/north/west triangles without moving physical points."""
    lower = xy[:, 1] <= xy[:, 0]
    left = xy.sum(axis=1) <= 1
    return np.where(lower, np.where(left, 0, 1), np.where(left, 3, 2))


@dataclass(frozen=True)
class CrossedP1:
    """Independent nodal triples on the four triangles of every fine square."""

    values: np.ndarray

    @classmethod
    def from_arrays(cls, vertices: np.ndarray, pressure: np.ndarray) -> CrossedP1:
        """Validate complete geometry before assigning canonical triangle coordinates."""
        count = len(vertices)
        n = round(np.sqrt(count / 4))
        if count != 4 * n**2 or pressure.shape != (count, 3):
            raise ValueError("a complete crisscross P1 partition is required")
        index = np.floor(n * vertices.mean(axis=1)).astype(int)
        scaled = n * vertices - index[:, None]
        half = _quadrants(scaled.mean(axis=1))
        distance = np.linalg.norm(scaled[:, :, None] - PATTERN[half, None], axis=-1)
        permutation = distance.argmin(axis=2)
        keys = 4 * (index[:, 0] * n + index[:, 1]) + half
        if (
            np.max(distance.min(axis=2)) > 1e-10
            or np.any(index < 0)
            or np.any(index >= n)
            or len(np.unique(keys)) != count
        ):
            raise ValueError("fine triangles must form a unique crisscross grid")
        values = np.empty((n, n, 4, 3), dtype=pressure.dtype)
        for corner in range(3):
            values[index[:, 0], index[:, 1], half, permutation[:, corner]] = pressure[:, corner]
        return cls(values)

    @property
    def resolution(self) -> int:
        """Return the number of fine Cartesian squares per coordinate."""
        return self.values.shape[0]

    def evaluate(self, points: np.ndarray, *, y_side: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pressure and physical gradient, preserving horizontal incident sides."""
        if y_side not in (-1, 1) or np.any(points < 0) or np.any(points > 1):
            raise ValueError("unit-square points and incident y_side -1 or 1 are required")
        n = self.resolution
        scaled = points * n
        index = np.floor(scaled).astype(int)
        if y_side == -1:
            index[scaled[:, 1] == np.rint(scaled[:, 1]), 1] -= 1
        index = np.clip(index, 0, n - 1)
        xy = scaled - index
        half = _quadrants(xy)
        coefficients = self.values[index[:, 0], index[:, 1], half]
        polynomial = np.einsum("qij,qj->qi", INVERSE[half], coefficients)
        pressure = polynomial[:, 0] + np.sum(polynomial[:, 1:] * xy, axis=1)
        return pressure, n * polynomial[:, 1:]


def common_triangles(reference_resolution: int, crossed_resolution: int) -> Iterator[np.ndarray]:
    """Split the shared SW–NE overlay additionally along every crisscross NW–SE edge."""
    n = crossed_resolution
    for vertices in overlay_triangles(reference_resolution, n):
        index = np.floor(vertices.mean(axis=1) * n).astype(int)
        offsets = (index.sum(axis=1) + 1) / n
        distance = vertices.sum(axis=2) - offsets[:, None]
        cut = (distance.min(axis=1) < 0) & (distance.max(axis=1) > 0)
        # Cells with no crossing retain their original coordinates and integration.
        output = [vertices[~cut]]
        fragments = []
        for triangle, offset in zip(vertices[cut], offsets[cut], strict=True):
            transformed = np.column_stack((triangle.sum(axis=1), triangle[:, 1]))
            for side in (False, True):
                polygon = clip_polygon(transformed, 0, offset, side)
                physical = np.column_stack((polygon[:, 0] - polygon[:, 1], polygon[:, 1]))
                for corner in range(1, len(physical) - 1):
                    tri = physical[[0, corner, corner + 1]]
                    if abs(np.linalg.det(tri[1:] - tri[:1])) > 0:
                        fragments.append(tri)
        if fragments:
            output.append(np.asarray(fragments))
        yield np.concatenate(output)
