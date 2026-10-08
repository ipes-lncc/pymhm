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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from examples.archive_precision import restore_precision
from pymhm.fem.scalar.operators import p1_geometry, triangle_quadrature
from pymhm.fem.scalar.triangle import multiindices, nodal_space, reference_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file
from pymhm.meshes.triangle import TriangleMesh


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


def _phase_data(path: Path) -> dict[str, np.ndarray]:
    """Verify the accepted phase history and its actual archived local bases."""
    from examples.unfitted_phases import UnfittedAcquisition

    matches = []
    for receipt in path.parent.glob("*-phases.json"):
        record = json.loads(receipt.read_text())
        for name, row in record.get("fields", {}).items():
            if row.get("archive") == path.name:
                matches.append((receipt, record, name))
    if len(matches) != 1:
        raise ValueError("phase field requires its unique executed acquisition receipt")
    receipt, record, name = matches[0]
    if record.get("schema") != "pymhm-unfitted-phases-v2" or not record.get("complete"):
        raise ValueError("unsupported or incomplete unfitted phase field schema")
    config = record["configuration"]
    run = UnfittedAcquisition(
        path.parent,
        refinement=config["local_refinement"],
        names=config["requested_names"],
        degree=config["local_degree"],
        assembly_order=config["requested_assembly_order"],
        norm_orders=config["norm_orders"],
        local_solver=config["local_solver"],
        refinement_precision=config["local_refinement_precision"],
        original_refinement_steps=config["original_refinement_steps"],
    )
    if run.path != receipt:
        raise ValueError("phase field receipt differs from its declared configuration")
    try:
        arrays = run._field(name)
    except KeyError as error:
        raise ValueError("phase field is missing its declared archived basis or history") from error
    if int(arrays["degree"]) != config["local_degree"]:
        raise ValueError("phase field degree differs from its executed cardinal basis")
    for cell in range(len(run.mesh.cells)):
        stored = run._cell(cell)
        for key in ("points", "cells", "nodal_dofs", "nodal_points", "multiindices", "constraints"):
            if f"{key}_{cell}" not in arrays or not np.array_equal(
                arrays[f"{key}_{cell}"], stored[key]
            ):
                raise ValueError(
                    "phase archived cardinal basis or local map differs from execution"
                )
    run._current_sources()
    return arrays


@dataclass
class _UniformField:
    """Keep each executed physical mesh and cardinal map with its coefficients."""

    macro: TriangleMesh
    fine: TriangleMesh
    degree: int
    fields: np.ndarray
    local_meshes: tuple[TriangleMesh, ...]
    dofs: tuple[np.ndarray, ...]


def _read_uniform(path: Path) -> _UniformField:
    """Restore exact field components and verify the archived uniform local topology.

    Wider coefficients require a wider native NumPy type for numerical replay.
    Their portable float64 components remain readable on other platforms, but
    this comparison never silently discards a nonzero correction or tail.
    """
    with np.load(path, allow_pickle=False) as archive:
        phased = any(
            name in archive
            for name in (
                "acquisition_id",
                "refinement_steps",
                "retained_basis_0",
                "multiindices_0",
                "initial_pressure_0",
                "initial_trace",
            )
        )
        if (
            not phased
            and "field_archive_version" not in archive
            and any(
                name in archive for name in ("basis_multiindices", "nodal_dofs_0", "macro_faces")
            )
        ):
            raise ValueError("archived cardinal basis requires a supported field schema")
        data = _phase_data(path) if phased else archive
        if "coefficient_precision_bits" in data and np.finfo(np.longdouble).nmant + 1 < int(
            data["coefficient_precision_bits"]
        ):
            raise ValueError("replaying field coefficients requires their declared precision")
        macro = TriangleMesh(data["macro_points"], data["macro_cells"])
        degree = int(data["degree"])
        declared_basis = phased or "field_archive_version" in data
        if declared_basis:
            if not phased and int(data["field_archive_version"]) != 2:
                raise ValueError("unsupported unfitted field archive version")
            expected_maps = dict(
                macro_faces=macro.faces,
                macro_face_cells=macro.face_cells,
                macro_normals=macro.normals,
                macro_signs=macro.signs,
                macro_cell_faces=macro.cell_faces,
            )
            if not phased:
                expected_maps["basis_multiindices"] = multiindices(degree)
            for name, expected_map in expected_maps.items():
                if name not in data or not np.array_equal(data[name], expected_map):
                    raise ValueError("archived cardinal basis or macro orientation differs")
        count = len(data["cells_0"])
        refinement = math.isqrt(count)
        if refinement**2 != count:
            raise ValueError("the archive is not a uniform local triangular subdivision")
        fine = template(refinement)
        _, canonical_nodes = nodal_space(fine, degree)
        fields, local_meshes, archived_dofs = [], [], []
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
            local = TriangleMesh(actual, data[f"cells_{cell}"])
            dofs, coordinates = nodal_space(local, degree)
            if declared_basis:
                expected_maps = {
                    f"nodal_dofs_{cell}": dofs,
                    f"nodal_points_{cell}": coordinates,
                }
                if phased:
                    expected_maps[f"multiindices_{cell}"] = multiindices(degree)
                for name, expected_map in expected_maps.items():
                    if name not in data or not np.array_equal(data[name], expected_map):
                        raise ValueError("archived cardinal coordinates or nodal DOF map differs")
                dofs = data[f"nodal_dofs_{cell}"]
            # The declared equidistant nodes and multiindices uniquely fix the
            # cardinal polynomials. Evaluation retains the archived physical
            # geometry and checked DOF map rather than inferring coefficient order.
            local_meshes.append(local)
            archived_dofs.append(dofs)
            name = f"pressure_{cell}"
            values = data[name]
            if values.shape != (len(canonical_nodes),):
                raise ValueError("archived field coefficients differ from their nodal basis")
            if f"{name}_correction" in data:
                low, tail = data[f"{name}_correction"], data[f"{name}_tail"]
                if np.finfo(np.longdouble).eps >= np.finfo(float).eps and (
                    np.any(low != 0) or np.any(tail != 0)
                ):
                    raise ValueError("replaying wider field coefficients requires wider longdouble")
                values = restore_precision(values, low, tail)
            if np.iscomplexobj(values) or not np.isfinite(values).all():
                raise ValueError("archived field coefficients must be finite real nodal values")
            fields.append(values)
    return _UniformField(
        macro, fine, degree, np.stack(fields), tuple(local_meshes), tuple(archived_dofs)
    )


def read_uniform(path: Path) -> tuple[TriangleMesh, TriangleMesh, int, np.ndarray]:
    """Verify monolithic v2 or accepted phase v2 bases before restoring coefficients.

    Legacy geometry-only archives use the declared canonical cardinal ordering.
    A phase archive requires its complete source-verified acquisition receipt,
    executed retained basis and ordered field history; it cannot enter that
    legacy path. Wider portable coefficients require their declared host precision.
    """
    field = _read_uniform(path)
    return field.macro, field.fine, field.degree, field.fields


def difference(first: Path, second: Path, *, order: int = 13) -> dict[str, float]:
    """Integrate pressure and broken-gradient differences without sampling transfer."""
    field_a, field_b = _read_uniform(first), _read_uniform(second)
    macro, other = field_a.macro, field_b.macro
    mesh_a, mesh_b = field_a.fine, field_b.fine
    if not np.array_equal(macro.cells, other.cells) or not np.array_equal(
        macro.points, other.points
    ):
        raise ValueError("the local-resolution comparison requires identical macro geometry")
    refinement = math.lcm(math.isqrt(len(mesh_a.cells)), math.isqrt(len(mesh_b.cells)))
    common = template(refinement)
    vertices = common.points[common.cells]
    owners_a, owners_b = containing_cells(mesh_a, vertices), containing_cells(mesh_b, vertices)
    quadrature, weights = triangle_quadrature(order)
    metadata = []
    for field, owners in ((field_a, owners_a), (field_b, owners_b)):
        nodes = np.stack([mesh.points[mesh.cells] for mesh in field.local_meshes])
        inverse = np.linalg.inv((nodes[:, :, 1:] - nodes[:, :, :1]).swapaxes(2, 3))
        geometry = np.stack([p1_geometry(mesh)[0] for mesh in field.local_meshes])
        metadata.append(
            (nodes, inverse, np.stack(field.dofs), geometry, field.fields, owners, field.degree)
        )
    totals = np.zeros(2, dtype=np.longdouble)
    for start in range(0, len(vertices), 8):
        stop = min(start + 8, len(vertices))
        reference_points = np.einsum("qi,tia->tqa", quadrature, vertices[start:stop])
        points = np.einsum(
            "tqi,mia->mtqa",
            np.concatenate(
                (1 - reference_points.sum(axis=2, keepdims=True), reference_points), axis=2
            ),
            macro.points[macro.cells],
        )
        values = []
        for nodes, inverse, dofs, geometry, coefficients, owners, degree in metadata:
            ids = owners[start:stop]
            coordinates = np.einsum("mtij,mtqj->mtqi", inverse[:, ids], points - nodes[:, ids, :1])
            bary = np.concatenate((1 - coordinates.sum(axis=3, keepdims=True), coordinates), axis=3)
            basis, derivative, _ = reference_basis(degree, bary.reshape(-1, 3))
            basis = basis.reshape(*bary.shape[:3], -1)
            derivative = derivative.reshape(*bary.shape[:3], -1, 3)
            local = coefficients[np.arange(len(macro.cells))[:, None, None], dofs[:, ids]]
            pressure = np.einsum("mtqi,mti->mtq", basis, local)
            gradient = np.einsum("mtqib,mtba,mti->mtqa", derivative, geometry[:, ids], local)
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
    root = case_workspace()
    sources = (
        "examples/unfitted_local_resolution.py",
        "examples/archive_precision.py",
        "src/pymhm/fem/scalar/triangle.py",
        "src/pymhm/fem/scalar/operators.py",
        "src/pymhm/meshes/triangle.py",
    )
    before = current_source_manifest(
        {
            name: hashlib.sha256((source_file(name, root=root)).read_bytes()).hexdigest()
            for name in sources
        },
        packages=("pymhm", "examples"),
    )
    fields = [
        dict(name=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in (args.first, args.second)
    ]
    with threadpool_limits(1):
        norms = difference(args.first, args.second, order=args.order)
    if any(
        hashlib.sha256((source_file(name, root=root)).read_bytes()).hexdigest() != value
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
    from importlib import import_module

    import_module("examples.unfitted_local_resolution").main()
