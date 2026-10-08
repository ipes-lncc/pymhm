"""Physical common-overlay norms for broken P1 fields on uniform SW--NE grids."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from math import fsum
from multiprocessing import get_context
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.meshes.geometry import clip_polygon


@dataclass(frozen=True)
class StructuredP1:
    """Store independent nodal triples on every unit-square fine triangle.

    Arrays follow square indices (x,y), then lower/upper triangle, then
    vertices (SW,SE,NE)/(SW,NE,NW). No interface averaging is performed.
    """

    values: np.ndarray

    @classmethod
    def from_arrays(cls, vertices: np.ndarray, pressure: np.ndarray) -> StructuredP1:
        """Validate and reorder a complete physical uniform triangular partition."""
        count = len(vertices)
        n = round(np.sqrt(count / 2))
        if count != 2 * n**2 or pressure.shape != (count, 3):
            raise ValueError("a complete uniform P1 triangular partition is required")
        centers = vertices.mean(axis=1)
        indices = np.floor(n * centers).astype(int)
        scaled = n * vertices - indices[:, None]
        half = (scaled.mean(axis=1)[:, 1] > scaled.mean(axis=1)[:, 0]).astype(int)
        canonical = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]])
        distances = np.linalg.norm(scaled[:, :, None] - canonical[half, None], axis=-1)
        permutation = distances.argmin(axis=2)
        keys = 2 * (indices[:, 0] * n + indices[:, 1]) + half
        if (
            np.max(distances.min(axis=2)) > 1e-10
            or np.any(indices < 0)
            or np.any(indices >= n)
            or len(np.unique(keys)) != count
        ):
            raise ValueError("fine cells must be unique SW--NE unit-square triangles")
        values = np.empty((n, n, 2, 3), dtype=pressure.dtype)
        for corner in range(3):
            values[indices[:, 0], indices[:, 1], half, permutation[:, corner]] = pressure[:, corner]
        return cls(values)

    @property
    def resolution(self) -> int:
        """Return the number of fine squares in either coordinate direction."""
        return self.values.shape[0]

    def evaluate(self, points: np.ndarray, *, y_side: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate physical pressure and gradient without smoothing across interfaces.

        On an exactly horizontal mesh interface, ``y_side`` selects the
        incident row geometrically; no physical coordinate is perturbed.
        """
        if y_side not in (-1, 1) or np.any(points < 0) or np.any(points > 1):
            raise ValueError("unit-square points and an incident y_side of -1 or 1 are required")
        n = self.resolution
        scaled = points * n
        indices = np.floor(scaled).astype(int)
        if y_side == -1:
            on_face = scaled[:, 1] == np.rint(scaled[:, 1])
            indices[on_face, 1] -= 1
        indices = np.clip(indices, 0, n - 1)
        xy = scaled - indices
        half = (xy[:, 1] > xy[:, 0]).astype(int)
        coefficients = self.values[indices[:, 0], indices[:, 1], half]
        low = half == 0
        bary = np.where(
            low[:, None],
            np.column_stack((1 - xy[:, 0], xy[:, 0] - xy[:, 1], xy[:, 1])),
            np.column_stack((1 - xy[:, 1], xy[:, 0], xy[:, 1] - xy[:, 0])),
        )
        gradient = n * np.where(
            low[:, None],
            np.column_stack(
                (coefficients[:, 1] - coefficients[:, 0], coefficients[:, 2] - coefficients[:, 1])
            ),
            np.column_stack(
                (coefficients[:, 1] - coefficients[:, 2], coefficients[:, 2] - coefficients[:, 0])
            ),
        )
        return np.einsum("qi,qi->q", bary, coefficients), gradient


def _overlay_batches(first: int, second: int, batch: int = 2048) -> Iterator[np.ndarray]:
    """Yield triangles resolving both uniform grids, including their diagonal edges."""
    if first == second and not first & (first - 1):
        # Exact dyadic coordinates retain the generic clipping path's upper/lower
        # triangle order and batch reductions. No field or quadrature is changed.
        pattern = (
            np.array([[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
            / first
        )
        squares_per_batch = (batch + 1) // 2
        for start in range(0, first * first, squares_per_batch):
            square = np.arange(start, min(start + squares_per_batch, first * first))
            origin = np.column_stack((square // first, square % first)) / first
            yield (origin[:, None, None, :] + pattern[None]).reshape(-1, 3, 2)
        return
    axes = np.unique(np.r_[np.linspace(0, 1, first + 1), np.linspace(0, 1, second + 1)])
    triangles: list[np.ndarray] = []
    for i in range(len(axes) - 1):
        for j in range(len(axes) - 1):
            origin = np.array([axes[i], axes[j]])
            width = axes[[i + 1, j + 1]] - origin
            center = origin + width / 2
            offsets = np.unique(
                [
                    (np.floor(center[0] * n) - np.floor(center[1] * n)) / n - origin[0] + origin[1]
                    for n in (first, second)
                ]
            )
            polygon = np.array([[0.0, 0.0], [width[0], 0.0], width, [0.0, width[1]]])
            transformed = np.column_stack((polygon[:, 0] - polygon[:, 1], polygon[:, 1]))
            parts = [transformed]
            for offset in offsets:
                divided: list[np.ndarray] = []
                for part in parts:
                    if part[:, 0].min() < offset < part[:, 0].max():
                        divided.extend(
                            clip_polygon(part, 0, offset, sign) for sign in (False, True)
                        )
                    else:
                        divided.append(part)
                parts = divided
            for part in parts:
                physical = origin + np.column_stack((part[:, 0] + part[:, 1], part[:, 1]))
                for corner in range(1, len(physical) - 1):
                    tri = physical[[0, corner, corner + 1]]
                    if abs(np.linalg.det(tri[1:] - tri[:1])) > 0:
                        triangles.append(tri)
            if len(triangles) >= batch:
                yield np.asarray(triangles)
                triangles = []
    if triangles:
        yield np.asarray(triangles)


@lru_cache(maxsize=2)
def _cached_overlay(first: int, second: int) -> tuple[np.ndarray, ...]:
    """Keep two immutable geometry partitions for independent quadrature checks."""
    arrays = tuple(_overlay_batches(first, second))
    for array in arrays:
        array.flags.writeable = False
    return arrays


def overlay_triangles(first: int, second: int) -> Iterator[np.ndarray]:
    """Reuse common physical geometry without caching or modifying any sampled field."""
    if not first & (first - 1) and not second & (second - 1):
        first = second = max(first, second)
    yield from _cached_overlay(*sorted((first, second)))


def _batch_integrals(
    vertices: np.ndarray,
    reference: StructuredP1,
    other: StructuredP1,
    coefficient: Any,
    bary: np.ndarray,
    weights: np.ndarray,
) -> list[float]:
    """Evaluate the original nine ordered physical integrals in one geometry batch."""
    points = np.einsum("qi,tia->tqa", bary, vertices).reshape(-1, 2)
    ref, grad = reference.evaluate(points)
    val, other_grad = other.evaluate(points)
    kval = np.broadcast_to(
        coefficient(points) if callable(coefficient) else coefficient, (len(points),)
    )
    delta = np.sum((grad - other_grad) ** 2, axis=1)
    gradient = np.sum(grad**2, axis=1)
    physical_weights = (
        abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1]))[:, None] * weights / 2
    ).ravel()
    return [
        float(np.sum(physical_weights * integrand))
        for integrand in (
            (ref - val) ** 2,
            kval**2 * delta,
            ref**2,
            kval**2 * gradient,
            delta,
            gradient,
            kval * delta,
            kval * gradient,
            np.ones(len(points)),
        )
    ]


_worker_data: tuple | None = None
_worker_threads: Any = None


def _initialize_worker(
    reference: StructuredP1,
    other: StructuredP1,
    coefficient: Any,
    bary: np.ndarray,
    weights: np.ndarray,
) -> None:
    """Install immutable field data once in each spawned norm-integration worker."""
    global _worker_data, _worker_threads
    _worker_data = reference, other, coefficient, bary, weights
    _worker_threads = threadpool_limits(1)


def _worker_integrals(vertices: np.ndarray) -> list[float]:
    """Return one batch without changing its quadrature or reduction order."""
    if _worker_data is None:
        raise RuntimeError("norm worker was not initialized")
    return _batch_integrals(vertices, *_worker_data)


def difference(
    reference: StructuredP1,
    other: StructuredP1,
    coefficient: Any,
    order: int = 6,
    *,
    workers: int = 1,
    geometry: Iterator[np.ndarray] | None = None,
) -> dict[str, float | int | None]:
    """Integrate physical differences with deterministic serial or spawned reductions.

    Each denominator uses the declared reference field in its physical norm.
    A zero denominator has no relative comparison. Parallel integration retains
    every quadrature point, batch boundary and parent-side summation order.
    Coefficients must be picklable when more than one worker is requested.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("norm workers must be a positive integer")
    bary, weights = triangle_quadrature(order)
    reductions: list[list[float]] = [[] for _ in range(9)]
    if geometry is None:
        geometry = overlay_triangles(reference.resolution, other.resolution)
    arguments = reference, other, coefficient, bary, weights

    def accumulate(batches: Iterator[list[float]]) -> None:
        """Preserve the serial sequence independently of task completion order."""
        for values in batches:
            for accumulation, value in zip(reductions, values, strict=True):
                accumulation.append(value)

    if workers == 1:
        accumulate(_batch_integrals(vertices, *arguments) for vertices in geometry)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize_worker,
            initargs=arguments,
        ) as executor:
            accumulate(executor.map(_worker_integrals, geometry))
    integrals = np.array([fsum(values) for values in reductions])
    if abs(integrals[-1] - 1) > 2e-12:
        raise ValueError("common overlay does not partition the physical unit square")
    norms = np.sqrt(integrals[:-1])
    result: dict[str, float | int | None] = {
        "quadrature_order": order,
        "overlay_area": integrals[-1],
    }
    for name, numerator, denominator in (
        ("pressure", 0, 2),
        ("flux", 1, 3),
        ("gradient", 4, 5),
        ("energy", 6, 7),
    ):
        result[f"{name}_difference"] = float(norms[numerator])
        result[f"reference_{name}_norm"] = float(norms[denominator])
        result[f"{name}_relative_difference"] = (
            float(norms[numerator] / norms[denominator]) if norms[denominator] else None
        )
    return result
