"""Physical intersection norms for MHM/PGMHM square-annulus inclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import restore_precision
from examples.compare_unusual_spe10 import overlay_quadrature
from examples.pgmhm_inclusion_data import coefficient
from examples.solve_pgmhm_inclusions_reference import InclusionField, load_field
from pymhm.fem.scalar.operators import p1_geometry
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]
_FIELDS: tuple[InclusionMHMField, InclusionField] | None = None
_LIMIT = None


class InclusionMHMField:
    """Three original P2 coefficient fields on the same independent macro-local spaces."""

    def __init__(self, archive: Path) -> None:
        """Restore ragged geometry and all portable correction components once."""
        with np.load(archive) as arrays:
            point, cell, value = (
                arrays[name] for name in ("point_offsets", "cell_offsets", "coefficient_offsets")
            )
            local_points, local_cells = arrays["local_points"], arrays["local_cells"]
            self.meshes = tuple(
                TriangleMesh(
                    local_points[point[i] : point[i + 1]],
                    local_cells[cell[i] : cell[i + 1]],
                )
                for i in range(len(point) - 1)
            )
            fields = np.stack(
                [
                    restore_precision(
                        arrays[name], arrays[name + "_correction"], arrays[name + "_tail"]
                    )
                    for name in ("mhm", "pgmhm", "enriched")
                ]
            )
            self.coefficients = tuple(
                fields[:, value[i] : value[i + 1]] for i in range(len(self.meshes))
            )
        self.dofs = tuple(nodal_space(mesh, 2)[0] for mesh in self.meshes)
        self.gradients = tuple(p1_geometry(mesh)[0] for mesh in self.meshes)
        self.vertices = tuple(mesh.points[mesh.cells] for mesh in self.meshes)
        self.inverse = tuple(
            np.linalg.inv((v[:, 1:] - v[:, :1]).swapaxes(1, 2)) for v in self.vertices
        )

    def evaluate(self, macro: int, cell: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return all three pressure values and physical gradients in a stated fine cell."""
        coordinate = (points - self.vertices[macro][cell, 0]) @ self.inverse[macro][cell].T
        bary = np.column_stack((1 - coordinate.sum(axis=1), coordinate))
        if bary.min() < -1e-10:
            raise ValueError("integration points lie outside their stated local triangle")
        basis, derivative, _ = reference_basis(2, bary)
        values = self.coefficients[macro][:, self.dofs[macro][cell]]
        gradient_basis = np.einsum("qib,ba->qia", derivative, self.gradients[macro][cell])
        return values @ basis.T, np.einsum("fi,qia->fqa", values, gradient_basis)


def initialize(archive: str, reference: str) -> None:
    """Load immutable acquisition data once per spawn worker with bounded native threads."""
    global _FIELDS, _LIMIT
    _LIMIT = threadpool_limits(1)
    _FIELDS = InclusionMHMField(Path(archive)), load_field(Path(reference))


def integrate(task: tuple[int, int]) -> np.ndarray:
    """Integrate all three fields together on the exact fine/reference overlay."""
    if _FIELDS is None:
        raise RuntimeError("initialize fields before integration")
    mhm, reference = _FIELDS
    macro, order = task
    sums = np.zeros(12, dtype=np.longdouble)
    for cell, vertices in enumerate(mhm.vertices[macro]):
        points, weights = overlay_quadrature(vertices, reference, order)
        pressure, gradient = mhm.evaluate(macro, cell, points)
        exact, exact_gradient, exact_flux = reference.evaluate(points)
        material = coefficient(points)
        difference = gradient - exact_gradient
        scalar = pressure - exact
        values = np.concatenate(
            (
                np.stack(
                    (
                        scalar**2,
                        material[None, :] ** 2 * np.sum(difference**2, axis=2),
                        material[None, :] * np.sum(difference**2, axis=2),
                    ),
                    axis=2,
                )
                .reshape(-1, len(points), 3)
                .transpose(1, 0, 2)
                .reshape(len(points), 9),
                np.column_stack(
                    (
                        exact**2,
                        np.sum(exact_flux**2, axis=1),
                        material * np.sum(exact_gradient**2, axis=1),
                    )
                ),
            ),
            axis=1,
        )
        sums += np.sum(weights[:, None] * values, axis=0, dtype=np.longdouble)
    return sums


def acquire(archive: Path, reference: Path, workers: int, output: Path | None = None) -> dict:
    """Check two polynomial-exact quadratures and persist data/source attribution."""
    names = (
        "examples/compare_pgmhm_inclusions.py",
        "examples/pgmhm_inclusion_data.py",
        "examples/solve_pgmhm_inclusions_reference.py",
        "examples/compare_unusual_spe10.py",
        "examples/spe10_adaptive_norms.py",
        "src/pymhm/fem/scalar/triangle.py",
    )
    hashes = current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )
    with np.load(archive) as data:
        count = len(data["point_offsets"]) - 1
    row = dict(
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        reference=reference.name,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        integration="exact geometric intersections; Duffy orders3/4; raw -K grad p",
        denominator="corresponding physical norm of the stated CG2 reference",
        source_hashes=hashes,
        rows=[],
    )
    output = output or archive.with_name(archive.stem + "-comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for order in (3, 4):
        started = perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize,
            initargs=(str(archive), str(reference)),
        ) as pool:
            squared = np.sum(
                list(pool.map(integrate, [(i, order) for i in range(count)])),
                axis=0,
                dtype=np.longdouble,
            )
        norms = np.sqrt(squared)
        record = dict(order=order, seconds=perf_counter() - started)
        for index, name in enumerate(("mhm", "pgmhm", "enriched")):
            record[name] = {
                key: float(value)
                for metric, error, denominator in zip(
                    ("pressure", "flux", "energy"),
                    norms[3 * index : 3 * index + 3],
                    norms[-3:],
                    strict=True,
                )
                for key, value in (
                    (metric + "_absolute", error),
                    (metric + "_reference", denominator),
                    (metric + "_relative", error / denominator),
                )
            }
        row["rows"].append(record)
        row["source_changed_during_run"] = hashes != current_source_manifest(
            {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
        )
        if row["source_changed_during_run"]:
            raise RuntimeError("norm sources changed during acquisition")
        output.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    return row


def main() -> None:
    """Compare complete archived fields; no PDE is solved by this command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    acquire(args.archive, args.reference, args.workers, args.output)


if __name__ == "__main__":
    main()
