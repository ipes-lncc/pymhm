"""Physical differences of uniform-local Pk fields on the same macro mesh.

The common local lattice uses the least common multiple of both resolutions.
Containment of every common triangle is checked before integrating the two
independent polynomial fields. Only the smooth uniform-material study uses
this helper; fitted material meshes are not silently treated as uniform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from examples.archive_precision import restore_precision
from pymhm.elements import p1_geometry, triangle_quadrature
from pymhm.lagrange import nodal_space, reference_basis
from pymhm.mesh import TriangleMesh


def template(refinement: int) -> TriangleMesh:
    """Return the canonical positive reference-triangle lattice."""
    return TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, refinement)


def containing_cells(mesh: TriangleMesh, vertices: np.ndarray) -> np.ndarray:
    """Find and prove containment, with a full-search fallback after spatial candidates."""
    old = mesh.points[mesh.cells]
    inverses = np.linalg.inv((old[:, 1:] - old[:, :1]).swapaxes(1, 2))
    centers = vertices.mean(axis=1)
    _, candidates = cKDTree(old.mean(axis=1)).query(centers, k=min(8, len(old)))
    candidates = np.asarray(candidates).reshape(len(centers), -1)
    owners = np.empty(len(centers), dtype=np.int64)
    tolerance = (
        128
        * np.finfo(float).eps
        * max(1.0, float(np.max(np.linalg.norm(inverses, ord=np.inf, axis=(-2, -1)))))
    )
    for index, point in enumerate(centers):
        possible = candidates[index]
        coordinates = np.einsum("tij,tj->ti", inverses[possible], point - old[possible, 0])
        bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
        best = bary.min(axis=1).argmax()
        if bary[best].min() < -tolerance:
            possible = np.arange(len(old))
            coordinates = np.einsum("tij,tj->ti", inverses, point - old[:, 0])
            bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
            best = bary.min(axis=1).argmax()
        owners[index] = possible[best]
    coordinates = np.einsum("tij,tqj->tqi", inverses[owners], vertices - old[owners, :1])
    bary = np.concatenate((1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2)
    if bary.min() < -tolerance:
        raise ValueError("common triangles are not contained in the declared local partition")
    return owners


def read_uniform(path: Path) -> tuple[TriangleMesh, TriangleMesh, int, np.ndarray]:
    """Restore exact field components and verify the archived uniform local topology.

    Wider coefficients require a wider native NumPy type for numerical replay.
    Their portable float64 components remain readable on other platforms, but
    this comparison never silently discards a nonzero correction or tail.
    """
    with np.load(path) as data:
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        degree = int(data["degree"])
        count = len(data["cells_0"])
        refinement = math.isqrt(count)
        if refinement**2 != count:
            raise ValueError("the archive is not a uniform local triangular subdivision")
        fine = template(refinement)
        fields = []
        for cell, nodes in enumerate(macro.cells):
            vertices = macro.points[nodes]
            expected = vertices[0] + fine.points @ (vertices[1:] - vertices[:1])
            actual = data[f"points_{cell}"]
            scale = max(1.0, float(np.max(np.abs(vertices))))
            if (
                actual.shape != expected.shape
                or not np.array_equal(data[f"cells_{cell}"], fine.cells)
                or np.max(np.abs(actual - expected)) > 64 * np.finfo(float).eps * scale
            ):
                raise ValueError("archived geometry differs from the uniform local lattice")
            name = f"pressure_{cell}"
            values = data[name]
            if f"{name}_correction" in data:
                low, tail = data[f"{name}_correction"], data[f"{name}_tail"]
                if np.finfo(np.longdouble).eps >= np.finfo(float).eps and (
                    np.any(low != 0) or np.any(tail != 0)
                ):
                    raise ValueError("replaying wider field coefficients requires wider longdouble")
                values = restore_precision(values, low, tail)
            fields.append(values)
    return macro, fine, degree, np.stack(fields)


def difference(first: Path, second: Path, *, order: int = 13) -> dict[str, float]:
    """Integrate pressure and broken-gradient differences without sampling transfer."""
    macro, mesh_a, degree_a, fields_a = read_uniform(first)
    other, mesh_b, degree_b, fields_b = read_uniform(second)
    if not np.array_equal(macro.cells, other.cells) or not np.array_equal(
        macro.points, other.points
    ):
        raise ValueError("the local-resolution comparison requires identical macro geometry")
    refinement = math.lcm(math.isqrt(len(mesh_a.cells)), math.isqrt(len(mesh_b.cells)))
    common = template(refinement)
    vertices = common.points[common.cells]
    owners_a, owners_b = containing_cells(mesh_a, vertices), containing_cells(mesh_b, vertices)
    quadrature, weights = triangle_quadrature(order)
    inverse_macro = np.linalg.inv(
        (macro.points[macro.cells[:, 1:]] - macro.points[macro.cells[:, :1]]).swapaxes(1, 2)
    )
    metadata = []
    for mesh, degree, coefficients, owners in (
        (mesh_a, degree_a, fields_a, owners_a),
        (mesh_b, degree_b, fields_b, owners_b),
    ):
        nodes = mesh.points[mesh.cells]
        inverse = np.linalg.inv((nodes[:, 1:] - nodes[:, :1]).swapaxes(1, 2))
        dofs, _ = nodal_space(mesh, degree)
        metadata.append((nodes, inverse, dofs, p1_geometry(mesh)[0], coefficients, owners, degree))
    totals = np.zeros(2, dtype=np.longdouble)
    for start in range(0, len(vertices), 32):
        stop = min(start + 32, len(vertices))
        points = np.einsum("qi,tia->tqa", quadrature, vertices[start:stop])
        values = []
        for nodes, inverse, dofs, geometry, coefficients, owners, degree in metadata:
            ids = owners[start:stop]
            coordinates = np.einsum("tij,tqj->tqi", inverse[ids], points - nodes[ids, :1])
            bary = np.concatenate((1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2)
            basis, derivative, _ = reference_basis(degree, bary.reshape(-1, 3))
            basis = basis.reshape(*bary.shape[:2], -1)
            derivative = derivative.reshape(*bary.shape[:2], -1, 3)
            local = coefficients[:, dofs[ids]]
            pressure = np.einsum("tqi,mti->mtq", basis, local)
            reference_gradient = np.einsum("tqib,tba,mti->mtqa", derivative, geometry[ids], local)
            gradient = np.einsum("mtqb,mba->mtqa", reference_gradient, inverse_macro)
            values.append((pressure, gradient))
        dp = values[0][0] - values[1][0]
        dg = values[0][1] - values[1][1]
        measure = 2 * macro.areas[:, None, None] * common.areas[None, start:stop, None] * weights
        totals[0] += np.sum(measure * dp**2, dtype=np.longdouble)
        totals[1] += np.sum(measure * np.sum(dg**2, axis=-1), dtype=np.longdouble)
    return {
        "pressure_l2": float(np.sqrt(totals[0])),
        "broken_gradient_l2": float(np.sqrt(totals[1])),
        "common_triangles": len(macro.cells) * len(common.cells),
    }


def main() -> None:
    """Compare complete archived smooth-study rows and preserve actual field digests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--order", type=int, default=13)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = (
        "examples/unfitted_local_resolution.py",
        "examples/archive_precision.py",
        "src/pymhm/lagrange.py",
        "src/pymhm/elements.py",
        "src/pymhm/mesh.py",
    )
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    fields = [
        dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in (args.first, args.second)
    ]
    with threadpool_limits(1):
        norms = difference(args.first, args.second, order=args.order)
    if any(
        hashlib.sha256((root / name).read_bytes()).hexdigest() != value
        for name, value in before.items()
    ) or any(
        hashlib.sha256(path.read_bytes()).hexdigest() != field["sha256"]
        for path, field in zip((args.first, args.second), fields, strict=True)
    ):
        raise RuntimeError("integration source or acquired field changed")
    record = dict(
        method="Common uniform-triangle overlay; physical Pk values and gradients",
        fields=fields,
        quadrature_order=args.order,
        norms=norms,
        source_sha256=before,
        source_changed_during_run=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
