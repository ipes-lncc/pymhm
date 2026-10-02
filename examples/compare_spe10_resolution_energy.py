"""Check nested MHM resolution increments against their exact Galerkin energy identity."""

from __future__ import annotations

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

from examples.spe10_adaptive_norms import BrokenP2
from pymhm.cut_cells import material_triangle_quadrature
from pymhm.reconstruction_moments import _PrimalFlux

_FIELDS: tuple[BrokenP2, BrokenP2, int] | None = None
_LIMIT: Any = None


def initialize(first: str, second: str, order: int) -> None:
    """Load unchanged physical coefficients once in each independent worker."""
    global _FIELDS, _LIMIT
    _LIMIT = threadpool_limits(1)
    _FIELDS = (BrokenP2(Path(first)), BrokenP2(Path(second)), order)
    left, right, _ = _FIELDS
    if not np.array_equal(left.macro.points, right.macro.points) or not np.array_equal(
        left.macro.cells, right.macro.cells
    ):
        raise ValueError("the energy control requires the identical macro partition")


def raw_evaluator(field: BrokenP2, cell: int) -> _PrimalFlux:
    """Use the shared physical-gradient evaluator with archived nodal coefficients."""
    vertices = field.meshes[cell].points[field.meshes[cell].cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    return _PrimalFlux(
        2,
        field.material,
        field.pressure[cell],
        field.dofs[cell],
        vertices,
        inverse,
        field.geometry[cell],
    )


def integrate(cells: np.ndarray) -> np.ndarray:
    """Integrate both energies and their difference over material-cut fine cells."""
    if _FIELDS is None:
        raise RuntimeError("physical fields must be initialized")
    first, second, order = _FIELDS
    totals = np.zeros(3, dtype=np.longdouble)
    for cell in cells:
        fine = second.meshes[cell]
        bary, weights, tensors = material_triangle_quadrature(fine, second.material, order)
        points = np.einsum("tqi,tia->tqa", bary, fine.points[fine.cells])
        old, new = raw_evaluator(first, cell), raw_evaluator(second, cell)
        centers = fine.points[fine.cells].mean(axis=1)
        local = np.einsum("tab,qtb->qta", old.inverse, centers[:, None] - old.vertices[None, :, 0])
        scores = np.minimum(local.min(axis=2), 1 - local.sum(axis=2))
        owners = np.argmax(scores, axis=1)
        if np.min(scores[np.arange(len(owners)), owners]) < -1e-10:
            raise ValueError("the second local partition must refine the first")
        vertices = fine.points[fine.cells]
        parent_coordinates = np.einsum(
            "tab,tqb->tqa", old.inverse[owners], vertices - old.vertices[owners, :1]
        )
        if min(parent_coordinates.min(), (1 - parent_coordinates.sum(axis=2)).min()) < -1e-10:
            raise ValueError("a fine triangle crosses its proposed coarse parent")
        coordinates = points.reshape(-1, 2)
        old_flux = old(coordinates, np.repeat(owners, len(bary[0])))
        new_flux = new(coordinates, np.repeat(np.arange(len(fine.cells)), len(bary[0])))
        inverse = np.linalg.inv(tensors)
        factors = (fine.areas[:, None] * weights).ravel().astype(np.longdouble)
        for index, vector in enumerate((old_flux, new_flux, new_flux - old_flux)):
            values = np.einsum("qa,qab,qb->q", vector, inverse, vector)
            totals[index] += np.sum(factors * values, dtype=np.longdouble)
    return totals


def main() -> None:
    """Measure a local-enrichment or trace-enrichment identity without another solve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--kind", choices=("local", "trace"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    with np.load(options.second) as data:
        count = len(data["macro_cells"])
    groups = [np.arange(i, min(i + 32, count)) for i in range(0, count, 32)]
    started = perf_counter()
    with ProcessPoolExecutor(
        max_workers=options.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(str(options.first), str(options.second), options.order),
    ) as executor:
        totals = np.sum(list(executor.map(integrate, groups)), axis=0, dtype=np.longdouble)
    expected = (totals[0] - totals[1]) * (1 if options.kind == "local" else -1)
    result = {
        "first": options.first.name,
        "second": options.second.name,
        "first_sha256": hashlib.sha256(options.first.read_bytes()).hexdigest(),
        "second_sha256": hashlib.sha256(options.second.read_bytes()).hexdigest(),
        "kind": options.kind,
        "first_energy_squared": float(totals[0]),
        "second_energy_squared": float(totals[1]),
        "increment_energy_squared": float(totals[2]),
        "signed_energy_difference": float(expected),
        "identity_relative_defect": float(abs(expected - totals[2]) / totals[2]),
        "quadrature_order": options.order,
        "integration": "nested local cells, exact material intersections",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seconds": perf_counter() - started,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
