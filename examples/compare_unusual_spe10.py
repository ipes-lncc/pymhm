"""Integrate UNUSUAL and conforming CG2 fields on their exact geometric overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.solve_unusual_spe10 import ROOT, UnusualSPE10Field
from examples.solve_unusual_spe10_reference import CG2Field
from examples.spe10_adaptive_norms import _clip_affine_polygon
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.io.provenance import current_source_manifest

_FIELDS: tuple[UnusualSPE10Field, CG2Field] | None = None
_LIMIT: Any = None


def overlay_quadrature(
    vertices: np.ndarray, reference: CG2Field, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fine triangle by all intersected reference rectangles and their diagonals.

    Polygon clipping uses bounded barycentric coordinates. The reference grid
    retains every permeability pixel, so each resulting triangle also lies in
    one constant-material region. Duffy order three integrates squared P1/P2
    scalar differences and their diffusion--reaction energy exactly.
    """
    origin, edges = vertices[0], vertices[1:] - vertices[0]
    axes = (reference.x_axis, reference.y_axis)
    ranges = [
        range(
            max(0, int(np.searchsorted(axis, vertices[:, dim].min(), side="right")) - 1),
            min(len(axis) - 1, int(np.searchsorted(axis, vertices[:, dim].max(), side="left"))),
        )
        for dim, axis in enumerate(axes)
    ]
    parts = []
    for i in ranges[0]:
        for j in ranges[1]:
            polygon = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            for dim, index, positive in (
                (0, i, True),
                (0, i + 1, False),
                (1, j, True),
                (1, j + 1, False),
            ):
                polygon = _clip_affine_polygon(
                    polygon, edges[:, dim], axes[dim][index] - origin[dim], positive
                )
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            lengths = np.array([axes[0][i + 1] - axes[0][i], axes[1][j + 1] - axes[1][j]])
            corner = np.array([axes[0][i], axes[1][j]])
            normal = edges[:, 0] / lengths[0] - edges[:, 1] / lengths[1]
            offset = -(origin[0] - corner[0]) / lengths[0] + (origin[1] - corner[1]) / lengths[1]
            for positive in (True, False):
                piece = _clip_affine_polygon(polygon, normal, offset, positive)
                for k in range(1, len(piece) - 1):
                    triangle = piece[[0, k, k + 1]]
                    a, b = triangle[1:] - triangle[0]
                    area = abs(a[0] * b[1] - a[1] * b[0]) / 2
                    if area > 0:
                        parts.append((triangle, area))
    fractions = np.array([item[1] for item in parts])
    if abs(fractions.sum() - 0.5) > 1e-12:
        raise ValueError("geometric overlay does not partition the complete fine triangle")
    triangles = np.array([item[0] for item in parts])
    bary, weights = triangle_quadrature(order)
    coordinates = np.einsum("qi,tia->tqa", bary, triangles)
    points = origin + coordinates @ edges
    determinant = abs(float(np.linalg.det(edges)))
    return points.reshape(-1, 2), (determinant * fractions[:, None] * weights).ravel()


def integrate_cells(
    mhm: UnusualSPE10Field, reference: CG2Field, macro: int, order: int
) -> np.ndarray:
    """Integrate physical squared differences and reference denominators for one macro."""
    result = np.zeros(6, dtype=np.longdouble)
    for contribution in cell_contributions(
        mhm, reference, macro, order, 0, len(mhm.vertices[macro])
    ):
        result += contribution
    return result


def cell_contributions(
    mhm: UnusualSPE10Field,
    reference: CG2Field,
    macro: int,
    order: int,
    first: int,
    stop: int,
) -> Iterator[np.ndarray]:
    """Yield original fine-cell integrals, without regrouping their floating-point sums."""
    for cell in range(first, stop):
        vertices = mhm.vertices[macro][cell]
        points, weights = overlay_quadrature(vertices, reference, order)
        p, gradient, flux = mhm.evaluate_local(macro, points, np.full(len(points), cell))
        truth, exact_gradient, exact_flux = reference.evaluate(points)
        scalar_difference = p - truth
        gradient_difference = gradient - exact_gradient
        coefficient = reference.material(points)
        values = np.column_stack(
            (
                scalar_difference**2,
                np.sum((flux - exact_flux) ** 2, axis=1),
                coefficient * np.sum(gradient_difference**2, axis=1) + scalar_difference**2,
                truth**2,
                np.sum(exact_flux**2, axis=1),
                coefficient * np.sum(exact_gradient**2, axis=1) + truth**2,
            )
        )
        yield np.sum(weights[:, None] * values, axis=0, dtype=np.longdouble)


def cell_tasks(counts: Iterable[int], order: int, batch_size: int) -> Iterator[tuple[int, ...]]:
    """Partition each macro's original cell order into bounded, independent work blocks."""
    if batch_size < 1:
        raise ValueError("cell batch size must be positive")
    for macro, count in enumerate(counts):
        for first in range(0, count, batch_size):
            yield macro, order, first, min(first + batch_size, count)


def accumulate_blocks(
    blocks: Iterable[tuple[int, int, np.ndarray]], counts: np.ndarray
) -> np.ndarray:
    """Recover the exact original cell-by-cell, then macro-by-macro reduction order."""
    result = np.zeros((len(counts), 6), dtype=np.longdouble)
    next_cell = np.zeros(len(counts), dtype=np.int64)
    for macro, first, values in blocks:
        if (
            not 0 <= macro < len(counts)
            or first != next_cell[macro]
            or values.ndim != 2
            or values.shape[1] != 6
            or not 0 < len(values) <= counts[macro] - first
        ):
            raise ValueError("cell blocks must cover each macro once in original cell order")
        for contribution in values:
            result[macro] += contribution
        next_cell[macro] += len(values)
    if not np.array_equal(next_cell, counts):
        raise ValueError("cell blocks do not cover every original fine cell")
    return result


def initialize(archive: str, reference: str) -> None:
    """Load each physical field once per spawn worker and bound native threading."""
    global _FIELDS, _LIMIT
    _LIMIT = threadpool_limits(1)
    _FIELDS = UnusualSPE10Field(Path(archive)), CG2Field.load(Path(reference))


def integrate(task: tuple[int, int]) -> np.ndarray:
    """Evaluate one ordered macrocell contribution without changing the field arrays."""
    if _FIELDS is None:
        raise RuntimeError("initialize fields before computing physical norms")
    return integrate_cells(*_FIELDS, *task)


def integrate_block(task: tuple[int, ...]) -> tuple[int, int, np.ndarray]:
    """Return individual cell contributions so worker boundaries never change a sum."""
    if _FIELDS is None:
        raise RuntimeError("initialize fields before computing physical norms")
    macro, order, first, stop = task
    return (
        macro,
        first,
        np.asarray(
            list(cell_contributions(*_FIELDS, macro, order, first, stop)), dtype=np.longdouble
        ),
    )


def acquire(
    archive: Path,
    reference: Path,
    output: Path,
    workers: int = 8,
    cell_batch_size: int = 256,
) -> dict[str, Any]:
    """Record quadrature checks and attribution separately from either PDE acquisition."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if cell_batch_size < 0:
        raise ValueError("cell batch size must be nonnegative; zero selects whole macros")
    sources = [
        "examples/compare_unusual_spe10.py",
        "examples/solve_unusual_spe10.py",
        "examples/solve_unusual_spe10_reference.py",
        "examples/spe10_adaptive_norms.py",
        "src/pymhm/fem/scalar/operators.py",
        "src/pymhm/meshes/triangle.py",
    ]
    fingerprint = current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    )
    with np.load(archive) as arrays:
        count = len(arrays["macro_cells"])
        cell_counts = np.diff(arrays["cell_offsets"])
    result: dict[str, Any] = {
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "reference": reference.name,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "integration": "exact intersections of both finite-element meshes and material pixels",
        "denominator": "corresponding physical norm of the stated numerical CG2 reference",
        "energy": "integral of K|gradient difference|^2 + |pressure difference|^2",
        "cell_batch_size": cell_batch_size,
        "reduction": "original fine-cell additions within each macro, then original macro order",
        "source_hashes": fingerprint,
        "rows": [],
    }
    for order in (3, 4):
        started = perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(archive), str(reference)),
        ) as executor:
            if cell_batch_size:
                values = accumulate_blocks(
                    executor.map(integrate_block, cell_tasks(cell_counts, order, cell_batch_size)),
                    cell_counts,
                )
            else:
                values = np.asarray(
                    list(executor.map(integrate, [(cell, order) for cell in range(count)]))
                )
        norms = np.sqrt(np.sum(values, axis=0, dtype=np.longdouble))
        row = {
            key: float(value)
            for name, difference, norm in zip(
                ("pressure", "flux", "energy"), norms[:3], norms[3:], strict=True
            )
            for key, value in (
                (f"{name}_absolute", difference),
                (f"{name}_reference", norm),
                (f"{name}_relative", difference / norm),
            )
        }
        result["rows"].append({"order": order, "seconds": perf_counter() - started, **row})
        result["source_changed_during_run"] = any(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
            for name, digest in fingerprint.items()
        )
        if result["source_changed_during_run"]:
            raise RuntimeError("physical norm source changed during acquisition")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(result["rows"][-1], flush=True)
    return result


def main() -> None:
    """Compare solved fields without rerunning or altering either discretization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cell-batch-size", type=int, default=256)
    args = parser.parse_args()
    acquire(args.archive, args.reference, args.output, args.workers, args.cell_batch_size)


if __name__ == "__main__":
    main()
