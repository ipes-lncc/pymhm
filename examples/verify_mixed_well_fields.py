"""Replay archived affine mixed well fields and separate fine-space and trace errors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from solve_mapped_well import WellData
from threadpoolctl import threadpool_limits

from pymhm.hdiv3d_family import HDiv3DFamily, cell_quadrature, face_quadrature, face_shape
from pymhm.hdiv3d_mesh import AffineMixedMesh, hdiv3d_dofs, hdiv3d_transform

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/mixed-well-geometries"


def digest(path: Path) -> str:
    """Return a file's SHA256 without relying on filesystem timestamps."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def powers(kind: str, degree: int) -> np.ndarray:
    """Enumerate the documented component-monomial rows of an archived reference basis."""
    return np.array(
        [
            exponent
            for exponent in product(range(degree + 1), repeat=3)
            if (sum(exponent) <= degree if kind == "tetrahedron" else sum(exponent[:2]) <= degree)
        ]
    )


@dataclass(frozen=True)
class ArchivedField:
    """Physical cell geometry and polynomial coordinates recovered from an executed basis."""

    record: dict[str, Any]
    vertices: np.ndarray
    jacobian: np.ndarray
    inverse: np.ndarray
    determinants: np.ndarray
    flux: np.ndarray
    pressure: np.ndarray
    q_powers: np.ndarray
    p_powers: np.ndarray
    top_cells: np.ndarray
    top_vertices: np.ndarray

    def evaluate(self, cells: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate stored pressure and contravariant-Piola flux on cell-specific points."""
        monomials = np.prod(reference[:, :, None, :] ** self.q_powers, axis=3)
        pressure_basis = np.prod(reference[:, :, None, :] ** self.p_powers, axis=3)
        reference_flux = np.einsum("tqe,tae->tqa", monomials, self.flux[cells])
        flux = np.einsum("tab,tqb->tqa", self.jacobian[cells], reference_flux)
        flux /= self.determinants[cells, None, None]
        return np.einsum("tqe,te->tq", pressure_basis, self.pressure[cells]), flux


def read_field(name: str, directory: Path = DATA) -> ArchivedField:
    """Validate the archive and recover its physical field using its executed reference matrix."""
    record = json.loads((directory / f"{name}.json").read_text())
    path = directory / record["archive"]
    if record.get("archive_schema") != 2 or digest(path) != record["sha256"]:
        raise ValueError("the archived field requires schema 2 and its recorded SHA256")
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    matrix = arrays["flux_basis_coefficients"]
    if hashlib.sha256(matrix.tobytes()).hexdigest() != record["flux_basis_sha256"]:
        raise ValueError("the executed flux basis digest does not match the record")
    family = HDiv3DFamily(record["kind"], record["pressure_degree"])
    q_powers = powers(family.kind, family.pressure_degree + 1)
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in ("vertices", "jacobian", "inverse", "determinants", "flux", "pressure")
    }
    top_cells, top_vertices = [], []
    offset = 0
    for index, points in enumerate(arrays["local_points"]):
        mesh = AffineMixedMesh(points, arrays["local_cells"][index], family.kind)
        transform = hdiv3d_transform(mesh, family, coefficients=matrix)
        canonical = np.einsum(
            "tij,tj->ti", transform, arrays["flux"][index][hdiv3d_dofs(mesh, family)]
        )
        collected["flux"].append((canonical @ matrix.T).reshape(-1, 3, len(q_powers)))
        collected["pressure"].append(arrays["pressure"][index])
        collected["vertices"].append(mesh.points[mesh.cells])
        for key in ("jacobian", "inverse", "determinants"):
            collected[key].append(getattr(mesh, key))
        for face in mesh.boundary_faces:
            physical = mesh.points[mesh.faces[face]]
            if np.allclose(physical[:, 2], WellData().height / 2, rtol=0, atol=1e-12):
                top_cells.append(offset + mesh.incidence[face][0][0])
                top_vertices.append(physical)
        offset += len(mesh.cells)
    return ArchivedField(
        record=record,
        **{key: np.concatenate(value) for key, value in collected.items()},
        q_powers=q_powers,
        p_powers=powers(family.kind, family.pressure_degree),
        top_cells=np.array(top_cells, dtype=int),
        top_vertices=np.array(top_vertices),
    )


def matching_cells(candidate: ArchivedField, reference: ArchivedField) -> tuple[np.ndarray, float]:
    """Match identical physical tetrahedra/prisms, rejecting a merely equal fine-cell count."""
    distances, matched = cKDTree(reference.vertices.mean(axis=1)).query(
        candidate.vertices.mean(axis=1)
    )
    scale = max(float(np.max(abs(reference.vertices))), 1.0)
    tolerance = 2048 * np.finfo(float).eps * scale
    if (
        np.max(distances) > tolerance
        or len(np.unique(matched)) != len(matched)
        or len(matched) != len(reference.vertices)
    ):
        raise ValueError("the physical fine-cell partitions are different")
    vertex_distances = np.linalg.norm(
        candidate.vertices[:, :, None, :] - reference.vertices[matched, None, :, :], axis=3
    )
    if np.max(vertex_distances.min(axis=2)) > tolerance:
        raise ValueError("matched centroids do not identify the same fine cells")
    return matched, float(np.max(distances))


def integrated_pair(
    candidate: ArchivedField, reference: ArchivedField, order: int
) -> dict[str, Any]:
    """Integrate physical errors, their cross term and the Galerkin Pythagorean identity."""
    matched, geometry_error = matching_cells(candidate, reference)
    points, weights = cell_quadrature(candidate.record["kind"], order)
    total = np.zeros(12, dtype=np.longdouble)
    data = WellData()
    for start in range(0, len(matched), 64):
        cells = np.arange(start, min(start + 64, len(matched)))
        ref_cells = matched[cells]
        locations = candidate.vertices[cells, 0, None] + np.einsum(
            "tab,qb->tqa", candidate.jacobian[cells], points
        )
        ref_points = np.einsum(
            "tab,tqb->tqa",
            reference.inverse[ref_cells],
            locations - reference.vertices[ref_cells, 0, None],
        )
        p, q = candidate.evaluate(cells, np.broadcast_to(points, (len(cells), *points.shape)))
        pr, qr = reference.evaluate(ref_cells, ref_points)
        pe = data.pressure(locations.reshape(-1, 3)).reshape(p.shape)
        qe = data.flux(locations.reshape(-1, 3)).reshape(q.shape)
        physical_weights = candidate.determinants[cells, None] * weights
        integrands = [
            (p - pr) ** 2,
            np.sum((q - qr) ** 2, axis=2),
            pr**2,
            np.sum(qr**2, axis=2),
            (p - pe) ** 2,
            np.sum((q - qe) ** 2, axis=2),
            (pr - pe) ** 2,
            np.sum((qr - qe) ** 2, axis=2),
            (pe - data.outer_pressure) ** 2,
            np.sum(qe**2, axis=2),
            pe**2,
            np.sum((qe - qr) * (qr - q), axis=2),
        ]
        for index, value in enumerate(integrands):
            total[index] += np.sum(physical_weights * value, dtype=np.longdouble)
    return {
        "order": order,
        "fine_cells": len(matched),
        "geometry_discrepancy_m": geometry_error,
        "pressure_difference_over_reference": float(np.sqrt(total[0] / total[2])),
        "flux_difference_over_reference": float(np.sqrt(total[1] / total[3])),
        "mhm_pressure_error_relative": float(np.sqrt(total[4] / total[10])),
        "mhm_flux_error_relative": float(np.sqrt(total[5] / total[9])),
        "classical_pressure_error_relative": float(np.sqrt(total[6] / total[10])),
        "classical_flux_error_relative": float(np.sqrt(total[7] / total[9])),
        "mhm_drawdown_error_relative": float(np.sqrt(total[4] / total[8])),
        "classical_drawdown_error_relative": float(np.sqrt(total[6] / total[8])),
        "flux_difference_over_exact": float(np.sqrt(total[1] / total[9])),
        "flux_cross_term_over_exact_squared": float(total[11] / total[9]),
        "pythagoras_relative_defect": float((total[5] - total[7] - total[1]) / total[5]),
    }


def top_surface_error(field: ArchivedField, order: int = 10) -> dict[str, float]:
    """Integrate tangential and full vector-flux errors on the physical top cap."""
    uv, weights = face_quadrature(len(field.top_vertices[0]), order)
    shape = face_shape(uv, len(field.top_vertices[0]))
    total = np.zeros(3, dtype=np.longdouble)
    for start in range(0, len(field.top_cells), 128):
        vertices = field.top_vertices[start : start + 128]
        cells = field.top_cells[start : start + 128]
        locations = np.einsum("qv,tva->tqa", shape, vertices)
        reference = np.einsum(
            "tab,tqb->tqa",
            field.inverse[cells],
            locations - field.vertices[cells, 0, None],
        )
        q = field.evaluate(cells, reference)[1]
        qe = WellData().flux(locations.reshape(-1, 3)).reshape(q.shape)
        measure = np.linalg.norm(
            np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]), axis=1
        )
        for index, value in enumerate(
            (np.sum((q - qe) ** 2, axis=2), np.sum(qe**2, axis=2), q[..., 2] ** 2)
        ):
            total[index] += np.sum(measure[:, None] * weights * value, dtype=np.longdouble)
    return {
        "quadrature_order": order,
        "flux_error_l2": float(np.sqrt(total[0])),
        "exact_flux_l2": float(np.sqrt(total[1])),
        "flux_error_relative": float(np.sqrt(total[0] / total[1])),
        "vertical_flux_l2": float(np.sqrt(total[2])),
    }


def main() -> None:
    """Verify the current immutable archives without solving or changing any physical field."""
    watched = [
        Path(__file__),
        ROOT / "examples/solve_mapped_well.py",
        ROOT / "src/pymhm/hdiv3d_family.py",
        ROOT / "src/pymhm/hdiv3d_mesh.py",
    ]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in watched}
    rows, replay = [], []
    for kind, degree in (("prism", 1), ("tetrahedron", 1), ("tetrahedron", 2)):
        prefix = f"{kind}-p{degree}-fine4"
        with threadpool_limits(1):
            reference = read_field(prefix + "-macro4")
            candidates = [
                read_field(prefix + f"-macro{m}") for m in ((1, 2) if kind == "prism" else (1,))
            ]
            for candidate in candidates:
                row = {
                    "candidate": candidate.record["archive"],
                    "reference": reference.record["archive"],
                    "candidate_sha256": candidate.record["sha256"],
                    "reference_sha256": reference.record["sha256"],
                    "norms": [integrated_pair(candidate, reference, order) for order in (7, 10)],
                    "candidate_top_surface": top_surface_error(candidate),
                    "reference_top_surface": top_surface_error(reference),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            points = np.array([[0.17, 0.21, 0.13], [0.23, 0.11, 0.51]])
            ids = np.arange(len(reference.vertices))
            expected = reference.evaluate(ids, np.broadcast_to(points, (len(ids), *points.shape)))
        with threadpool_limits(4):
            alternative = read_field(prefix + "-macro4")
            actual = alternative.evaluate(ids, np.broadcast_to(points, (len(ids), *points.shape)))
        differences = {
            label: float(np.max(abs(a - b)))
            for label, a, b in zip(
                ("pressure_max_absolute", "flux_max_absolute"), actual, expected, strict=True
            )
        }
        if differences["pressure_max_absolute"] != 0 or differences["flux_max_absolute"] > 1e-13:
            raise ArithmeticError("archived field replay depends on native thread count")
        replay.append(
            {
                "archive": reference.record["archive"],
                "sha256": reference.record["sha256"],
                "native_threads": [1, 4],
                "sample_count": len(ids) * len(points),
                **differences,
            }
        )
    if sources != {str(path.relative_to(ROOT)): digest(path) for path in watched}:
        raise RuntimeError("field verification source changed during execution")
    metadata = {
        "source_hashes": sources,
        "source_changed_during_run": False,
        "field_interpolation": (
            "Executed monomial basis, exact affine Piola, no averaging or smoothing."
        ),
    }
    (DATA / "fine-space-separation.json").write_text(
        json.dumps({**metadata, "rows": rows}, indent=2) + "\n"
    )
    (DATA / "replay-verification.json").write_text(
        json.dumps({**metadata, "rows": replay}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
