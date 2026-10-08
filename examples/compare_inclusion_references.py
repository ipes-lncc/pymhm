"""Compare nonnested conforming inclusion references on exact periodic intersections."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.compare_unusual_spe10 import overlay_quadrature
from examples.pgmhm_inclusion_data import axis
from examples.solve_pgmhm_inclusions_reference import InclusionField, load_field
from examples.solve_unusual_spe10_reference import CG2Field
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
_DATA: tuple[InclusionField, InclusionField, np.ndarray, np.ndarray, np.ndarray] | None = None
_LIMIT: Any = None


def fractions(field: CG2Field, base: np.ndarray) -> np.ndarray:
    """Require the same normalized subpartition in every material coordinate interval."""
    if not np.array_equal(field.x_axis, field.y_axis):
        raise ValueError("periodic overlay requires identical x and y coordinate axes")
    count, remainder = divmod(len(field.x_axis) - 1, len(base) - 1)
    if remainder or count < 1 or not np.array_equal(field.x_axis[::count], base):
        raise ValueError("field axes must preserve all parent material intervals")
    indices = np.arange(len(base) - 1)[:, None] * count + np.arange(count + 1)
    normalized = (field.x_axis[indices] - base[:-1, None]) / np.diff(base)[:, None]
    if not np.allclose(normalized, normalized[0], rtol=0, atol=128 * np.finfo(float).eps):
        raise ValueError("normalized subdivisions must repeat in every material interval")
    return normalized[0]


def intersection_pattern(
    fine: CG2Field, coarse: CG2Field, base: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect both SW–NE triangulations once on the normalized parent square."""
    fa, ca = fractions(fine, base), fractions(coarse, base)
    old = CG2Field(
        np.zeros((2 * len(ca) - 1, 2 * len(ca) - 1)),
        np.ones((1, 1)),
        (0, 1, 0, 1),
        ca,
        ca,
    )
    points, weights = [], []
    for i in range(len(fa) - 1):
        for j in range(len(fa) - 1):
            vertices = np.array(
                [[fa[i], fa[j]], [fa[i + 1], fa[j]], [fa[i + 1], fa[j + 1]], [fa[i], fa[j + 1]]]
            )
            for indices in ((0, 1, 2), (0, 2, 3)):
                q, w = overlay_quadrature(vertices[list(indices)], old, order)
                points.append(q)
                weights.append(w)
    x, w = np.concatenate(points), np.concatenate(weights)
    # The physical differences contain degree-four polynomials on every
    # intersection. Check all parent-square moments through degree four too.
    for i in range(5):
        for j in range(5 - i):
            if not np.isclose(
                np.sum(w * x[:, 0] ** i * x[:, 1] ** j), 1 / (i + 1) / (j + 1), rtol=0, atol=2e-13
            ):
                raise ValueError("intersection quadrature does not preserve parent moments")
    return x, w


def initialize(reference: str, previous: str, order: int) -> None:
    """Load immutable coefficient fields and the geometric pattern once per worker."""
    global _DATA, _LIMIT
    _LIMIT = threadpool_limits(1)
    fine, coarse = load_field(Path(reference)), load_field(Path(previous))
    if not np.array_equal(fine.permeability, coarse.permeability):
        raise ValueError("reference comparisons require identical physical materials")
    base = axis(1)
    points, weights = intersection_pattern(fine, coarse, base, order)
    _DATA = fine, coarse, base, points, weights


def integrate_group(ids: np.ndarray) -> np.ndarray:
    """Integrate pressure, physical flux and diffusion energy over disjoint parent cells."""
    if _DATA is None:
        raise RuntimeError("initialize reference fields before integration")
    fine, coarse, base, q, weight = _DATA
    n = len(base) - 1
    i, j = ids % n, ids // n
    lower = np.column_stack((base[i], base[j]))
    lengths = np.column_stack((np.diff(base)[i], np.diff(base)[j]))
    points = (lower[:, None] + lengths[:, None] * q).reshape(-1, 2)
    weights = (np.prod(lengths, axis=1)[:, None] * weight).ravel()
    p, grad, flux = fine.evaluate(points)
    old_p, old_grad, old_flux = coarse.evaluate(points)
    k = fine.material(points)
    values = np.column_stack(
        (
            (p - old_p) ** 2,
            np.sum((flux - old_flux) ** 2, axis=1),
            k * np.sum((grad - old_grad) ** 2, axis=1),
            p**2,
            np.sum(flux**2, axis=1),
            k * np.sum(grad**2, axis=1),
        )
    )
    return np.sum(weights[:, None] * values, axis=0, dtype=np.longdouble)


def acquire(reference: Path, previous: Path, output: Path, workers: int = 4) -> dict:
    """Archive two exact-overlay norm rules, preserving actual input and source digests."""
    paths = [
        Path(__file__),
        ROOT / "examples/compare_unusual_spe10.py",
        ROOT / "examples/solve_pgmhm_inclusions_reference.py",
        ROOT / "examples/solve_unusual_spe10_reference.py",
        ROOT / "examples/pgmhm_inclusion_data.py",
        ROOT / "examples/spe10_adaptive_norms.py",
        ROOT / "src/pymhm/fem/scalar/operators.py",
    ]
    hashes = current_source_manifest(
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    )
    record = dict(
        reference=reference.name,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        previous=previous.name,
        previous_sha256=hashlib.sha256(previous.read_bytes()).hexdigest(),
        denominator="corresponding physical norm of the reference field",
        integration="exact nonnested-triangle intersections, repeated per material rectangle",
        source_hashes=hashes,
        rows=[],
    )
    count = (len(axis(1)) - 1) ** 2
    groups = [np.arange(start, min(start + 4, count)) for start in range(0, count, 4)]
    for order in (3, 4):
        started = perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(reference), str(previous), order),
        ) as pool:
            sums = np.zeros(6, dtype=np.longdouble)
            for index, part in enumerate(pool.map(integrate_group, groups)):
                sums += part
                if (index + 1) % 256 == 0:
                    print(dict(order=order, groups=index + 1, total=len(groups)), flush=True)
        norms = np.sqrt(sums)
        row = {"order": order, "seconds": perf_counter() - started}
        for name, error, scale in zip(
            ("pressure", "flux", "energy"), norms[:3], norms[3:], strict=True
        ):
            row.update(
                {
                    name + "_absolute": float(error),
                    name + "_reference": float(scale),
                    name + "_relative": float(error / scale),
                }
            )
        record["rows"].append(row)
        record["source_changed_during_run"] = hashes != current_source_manifest(
            {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            }
        )
        if record["source_changed_during_run"]:
            raise RuntimeError("reference comparison sources changed during integration")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(row, flush=True)
    return record


def main() -> None:
    """Compare archived conforming fields without invoking a PDE solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    acquire(args.reference, args.previous, args.output, args.workers)


if __name__ == "__main__":
    main()
