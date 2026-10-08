"""Common-overlay physical norms for the SPE10 adaptive and classical RT2 fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from examples.archive_precision import restore_precision
from examples.solve_spe10 import load_layer
from examples.spe10_adaptive import DATA, DOMAIN, StructuredRT
from pymhm.fem.hdiv.rt import rt_evaluate_points
from pymhm.fem.quadrature.material import cartesian_trace_values
from pymhm.fem.scalar.operators import p1_geometry, triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space, reference_basis
from pymhm.io.workspace import local_resource, read_resource_bytes, read_resource_text
from pymhm.meshes.geometry import clip_polygon
from pymhm.meshes.triangle import TriangleMesh


class BrokenP2:
    """Preserve all macro-local P2 and reconstructed RT2 coefficients from an archive."""

    def __init__(self, path: Path) -> None:
        """Restore the exact geometry and independent one-sided local fields."""
        with np.load(local_resource(path)) as arrays:
            self.macro = TriangleMesh(arrays["macro_points"], arrays["macro_cells"])
            if "point_offsets" in arrays:
                count = len(arrays["point_offsets"]) - 1

                def partition(name: str, offsets: str) -> tuple[np.ndarray, ...]:
                    """Restore variable-sized local arrays without changing their dtype."""
                    stops = arrays[offsets]
                    combined = arrays[name]
                    if f"{name}_correction" in arrays:
                        combined = restore_precision(
                            combined, arrays[f"{name}_correction"], arrays[f"{name}_tail"]
                        )
                    return tuple(combined[stops[i] : stops[i + 1]] for i in range(count))

                points = partition("local_points", "point_offsets")
                cells = partition("local_cells", "cell_offsets")
                self.pressure = partition("pressure", "pressure_offsets")
                self.flux = partition("reconstructed_flux", "flux_offsets")
            else:
                points, cells = arrays["local_points"], arrays["local_cells"]
                self.pressure = tuple(arrays["pressure"])
                self.flux = tuple(arrays["reconstructed_flux"])
        self.meshes = tuple(TriangleMesh(p, c) for p, c in zip(points, cells, strict=True))
        self.material = load_layer()
        self.tree = cKDTree(self.macro.points[self.macro.cells].mean(axis=1))
        vertices = self.macro.points[self.macro.cells]
        self.origins = vertices[:, 0]
        self.inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        self.dofs = [nodal_space(mesh, 2)[0] for mesh in self.meshes]
        self.geometry = [p1_geometry(mesh)[0] for mesh in self.meshes]

    def locate(self, points: np.ndarray) -> np.ndarray:
        """Verify barycentric membership and exhaustively search any unresolved point."""
        _, candidates = self.tree.query(points, k=min(16, len(self.meshes)))
        candidates = np.atleast_2d(candidates).reshape(len(points), -1)
        coordinate = np.einsum(
            "nlab,nlb->nla", self.inverse[candidates], points[:, None] - self.origins[candidates]
        )
        score = np.minimum(coordinate.min(axis=2), 1 - coordinate.sum(axis=2))
        chosen = np.argmax(score, axis=1)
        owners = candidates[np.arange(len(points)), chosen]
        for point in np.flatnonzero(score[np.arange(len(points)), chosen] < -1e-11):
            coordinate = np.einsum("tab,tb->ta", self.inverse, points[point] - self.origins)
            score_all = np.minimum(coordinate.min(axis=1), 1 - coordinate.sum(axis=1))
            owners[point] = np.argmax(score_all)
            if score_all[owners[point]] < -1e-11:
                raise ValueError("point outside archived macro partition")
        return owners

    def evaluate_local(
        self,
        cell: int,
        points: np.ndarray,
        owners: np.ndarray | None = None,
        material_centers: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate pressure, raw flux and RT2 flux on explicit incident fine triangles."""
        mesh = self.meshes[cell]
        vertices = mesh.points[mesh.cells]
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        if owners is None:
            coordinate = np.einsum("tab,ntb->nta", inverse, points[:, None] - vertices[None, :, 0])
            score = np.minimum(coordinate.min(axis=2), 1 - coordinate.sum(axis=2))
            owners = np.argmax(score, axis=1)
            if np.any(score[np.arange(len(points)), owners] < -1e-11):
                raise ValueError("point outside declared macrocell")
        coordinate = np.einsum("nab,nb->na", inverse[owners], points - vertices[owners, 0])
        bary = np.column_stack((1 - coordinate.sum(axis=1), coordinate))
        basis, derivative, _ = reference_basis(2, bary)
        coefficients = self.pressure[cell][self.dofs[cell][owners]]
        pressure = np.einsum("qi,qi->q", basis, coefficients)
        gradient = np.einsum(
            "qin,qna,qi->qa", derivative, self.geometry[cell][owners], coefficients
        )
        tensors = (
            self.material(points)
            if material_centers is None
            else cartesian_trace_values(self.material, points, material_centers)
        )
        raw = -np.einsum("qab,qb->qa", tensors, gradient)
        reconstructed, _ = rt_evaluate_points(mesh, self.flux[cell], 2, points, owners)
        return pressure, raw, reconstructed


def overlay_quadrature(
    vertices: np.ndarray, reference: StructuredRT, order: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate on intersections with reference triangles and physical material pixels.

    Each quadrature triangle is inside one original fine triangle, one RT2
    reference triangle and one material pixel. Thus squared P2/RT2 differences
    have degree at most six, and positive Duffy order four is sufficient.
    """
    reference_spacing = DOMAIN / [reference.nx, reference.ny]
    # Keep all clipping and area arithmetic close to the fine triangle.
    # Repeated reconstruction of world coordinates would lose small cut areas
    # when a short edge lies at a large reservoir coordinate.
    anchor = vertices[0].copy()
    vertices = vertices - anchor
    axes = [
        np.unique(
            np.r_[np.linspace(0, DOMAIN[axis], count + 1), np.arange(pixel_count + 1) * pixel_width]
        )
        - anchor[axis]
        for axis, count, pixel_count, pixel_width in (
            (0, reference.nx, 60, 20.0),
            (1, reference.ny, 220, 10.0),
        )
    ]
    indices = [
        range(
            max(0, np.searchsorted(grid, vertices[:, axis].min(), side="right") - 1),
            min(len(grid) - 1, np.searchsorted(grid, vertices[:, axis].max(), side="left")),
        )
        for axis, grid in enumerate(axes)
    ]
    triangles = []
    for i in indices[0]:
        xpart = clip_polygon(vertices, 0, axes[0][i], True)
        if not len(xpart):
            continue
        xpart = clip_polygon(xpart, 0, axes[0][i + 1], False)
        for j in indices[1]:
            polygon = clip_polygon(xpart, 1, axes[1][j], True)
            if not len(polygon):
                continue
            polygon = clip_polygon(polygon, 1, axes[1][j + 1], False)
            if len(polygon) < 3:
                continue
            midpoint = np.array(
                [(axes[0][i] + axes[0][i + 1]) / 2, (axes[1][j] + axes[1][j + 1]) / 2]
            )
            origin = np.floor((midpoint + anchor) / reference_spacing) * reference_spacing - anchor
            local = (polygon - origin) / reference_spacing
            transformed = np.column_stack((local[:, 0] - local[:, 1], local[:, 1]))
            for positive in (True, False):
                part = clip_polygon(transformed, 0, 0.0, positive)
                if len(part) < 3:
                    continue
                physical = np.column_stack((part[:, 0] + part[:, 1], part[:, 1]))
                physical = origin + physical * reference_spacing
                for k in range(1, len(physical) - 1):
                    triangle = physical[[0, k, k + 1]]
                    a, b = triangle[1:] - triangle[0]
                    if abs(a[0] * b[1] - a[1] * b[0]) > 0:
                        triangles.append(triangle)
    array = np.array(triangles)
    a, b = (array[:, 1:] - array[:, :1]).transpose(1, 0, 2)
    areas = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    bary, weights = triangle_quadrature(order)
    actual_area = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 2
    if abs(areas.sum() - actual_area) > 2e-12 * actual_area:
        return _barycentric_overlay(vertices + anchor, reference, order)
    return anchor + np.einsum("qi,tia->tqa", bary, array).reshape(-1, 2), (
        areas[:, None] * weights
    ).ravel()


def _clip_affine_polygon(
    polygon: np.ndarray, normal: np.ndarray, offset: float, positive: bool
) -> np.ndarray:
    """Clip a canonical polygon by an affine physical half-plane without remapping vertices."""
    values = polygon @ normal - offset
    result = []
    previous, value = polygon[-1], values[-1]
    inside = value >= 0 if positive else value <= 0
    for current, current_value in zip(polygon, values, strict=True):
        current_inside = current_value >= 0 if positive else current_value <= 0
        if inside != current_inside:
            fraction = value / (value - current_value)
            result.append(previous + fraction * (current - previous))
        if current_inside:
            result.append(current)
        previous, value, inside = current, current_value, current_inside
    return np.asarray(result).reshape(-1, 2)


def _barycentric_overlay(
    vertices: np.ndarray, reference: StructuredRT, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the partition gate for thin cells using their unit reference triangle.

    The intersection planes are physical, but polygon vertices remain in bounded
    reference coordinates throughout all cuts. This avoids subtracting nearly
    parallel physical edges when a thin triangle spans several reference cells.
    Its area fractions must still sum to one within the original 2e-12 criterion.
    """
    spacing = DOMAIN / [reference.nx, reference.ny]
    axes = [
        np.unique(np.r_[np.linspace(0, DOMAIN[axis], count + 1), np.arange(pixels + 1) * width])
        for axis, count, pixels, width in (
            (0, reference.nx, 60, 20.0),
            (1, reference.ny, 220, 10.0),
        )
    ]
    indices = [
        range(
            max(0, np.searchsorted(grid, vertices[:, axis].min(), side="right") - 1),
            min(len(grid) - 1, np.searchsorted(grid, vertices[:, axis].max(), side="left")),
        )
        for axis, grid in enumerate(axes)
    ]
    origin, edges = vertices[0], vertices[1:] - vertices[0]
    parts: list[np.ndarray] = []
    for i in indices[0]:
        for j in indices[1]:
            polygon = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            for axis, index, lower in (
                (0, i, True),
                (0, i + 1, False),
                (1, j, True),
                (1, j + 1, False),
            ):
                polygon = _clip_affine_polygon(
                    polygon, edges[:, axis], axes[axis][index] - origin[axis], lower
                )
                if len(polygon) < 3:
                    break
            if len(polygon) < 3:
                continue
            midpoint = np.array(
                [(axes[0][i] + axes[0][i + 1]) / 2, (axes[1][j] + axes[1][j + 1]) / 2]
            )
            reference_origin = np.floor(midpoint / spacing) * spacing
            offset = (origin - reference_origin) / spacing
            normal = edges[:, 0] / spacing[0] - edges[:, 1] / spacing[1]
            for positive in (True, False):
                part = _clip_affine_polygon(polygon, normal, offset[1] - offset[0], positive)
                parts.extend(part[[0, k, k + 1]] for k in range(1, len(part) - 1))
    array = np.asarray(parts)
    first, second = (array[:, 1:] - array[:, :1]).transpose(1, 0, 2)
    areas = abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / 2
    if abs(areas.sum() - 0.5) > 1e-12:
        raise ValueError("overlay quadrature does not partition the fine triangle")
    physical_area = abs(np.linalg.det(edges.T)) / 2
    bary, weights = triangle_quadrature(order)
    canonical = np.einsum("qi,tia->tqa", bary, array)
    return (origin + canonical @ edges).reshape(-1, 2), (
        2 * physical_area * areas[:, None] * weights
    ).ravel()


def integrated_squared_norms(
    mhm: BrokenP2,
    reference: StructuredRT,
    order: int = 4,
    *,
    cells: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate thirteen squared norms on selected complete macroelements.

    A partition into disjoint ``cells`` arrays permits process parallelism
    without changing any geometric intersection, quadrature point, or physical
    field evaluation. The default integrates the complete domain.
    """
    totals = np.zeros(13, dtype=np.longdouble)
    for cell in range(len(mhm.meshes)) if cells is None else cells:
        mesh = mhm.meshes[cell]
        point_parts, weight_parts, owner_parts = [], [], []
        for fine_cell, vertices in enumerate(mesh.points[mesh.cells]):
            points, weights = overlay_quadrature(vertices, reference, order)
            point_parts.append(points)
            weight_parts.append(weights)
            owner_parts.append(np.full(len(points), fine_cell))
        all_points = np.concatenate(point_parts)
        all_weights = np.concatenate(weight_parts)
        all_owners = np.concatenate(owner_parts)
        for start in range(0, len(all_points), 16384):
            part = slice(start, start + 16384)
            points, weights = all_points[part], all_weights[part]
            p, raw, reconstructed = mhm.evaluate_local(cell, points, all_owners[part])
            ref_p, ref_q, _ = reference.evaluate(points)
            inverse = np.linalg.inv(mhm.material(points))
            raw_difference, reconstructed_difference = raw - ref_q, reconstructed - ref_q
            totals += [
                weights @ (p - ref_p) ** 2,
                weights @ np.sum((raw - ref_q) ** 2, axis=1),
                weights @ np.sum((reconstructed - ref_q) ** 2, axis=1),
                weights @ ref_p**2,
                weights @ np.sum(ref_q**2, axis=1),
                weights @ p**2,
                weights @ np.sum(raw**2, axis=1),
                weights @ np.sum(reconstructed**2, axis=1),
                weights @ np.einsum("qa,qab,qb->q", raw_difference, inverse, raw_difference),
                weights
                @ np.einsum(
                    "qa,qab,qb->q", reconstructed_difference, inverse, reconstructed_difference
                ),
                weights @ np.einsum("qa,qab,qb->q", ref_q, inverse, ref_q),
                weights @ np.einsum("qa,qab,qb->q", raw, inverse, raw),
                weights @ np.einsum("qa,qab,qb->q", reconstructed, inverse, reconstructed),
            ]
    return totals


def norm_record(totals: np.ndarray, order: int) -> dict[str, Any]:
    """Normalize common-overlay squared norms by the corresponding reference norm."""
    norms = np.sqrt(totals)
    names = (
        "pressure_difference_l2",
        "raw_flux_difference_l2",
        "reconstructed_flux_difference_l2",
        "reference_pressure_l2",
        "reference_flux_l2",
        "pressure_l2",
        "raw_flux_l2",
        "reconstructed_flux_l2",
        "raw_flux_energy_difference",
        "reconstructed_flux_energy_difference",
        "reference_flux_energy_norm",
        "raw_flux_energy_norm",
        "reconstructed_flux_energy_norm",
    )
    result = dict(zip(names, map(float, norms), strict=True))
    result.update(
        pressure_relative_difference=float(norms[0] / norms[3]),
        raw_flux_relative_difference=float(norms[1] / norms[4]),
        reconstructed_flux_relative_difference=float(norms[2] / norms[4]),
        raw_flux_energy_relative_difference=float(norms[8] / norms[10]),
        reconstructed_flux_energy_relative_difference=float(norms[9] / norms[10]),
        quadrature_order=order,
    )
    return result


def compare(mhm: BrokenP2, reference: StructuredRT, order: int = 4) -> dict[str, Any]:
    """Compute integrated pressure/raw-flux/reconstructed-flux discrepancies, not sample norms."""
    return norm_record(integrated_squared_norms(mhm, reference, order), order)


def main() -> None:
    """Integrate archived coefficient arrays without resolving either finite element system."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA / "longest-edge")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--levels", type=int, nargs="+")
    options = parser.parse_args()
    levels = (
        [row["level"] for row in json.loads(read_resource_text(options.data / "adaptive.json"))]
        if options.levels is None
        else options.levels
    )
    reference = StructuredRT.load(options.reference)
    reference_digest = hashlib.sha256(read_resource_bytes(options.reference)).hexdigest()
    source_digest = hashlib.sha256(read_resource_bytes(Path(__file__))).hexdigest()
    rows = []
    destination = options.output or options.data / f"comparison-order{options.order}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(1):
        for level in levels:
            archive = options.data / f"mhm-level{level}.npz"
            row = {
                "level": level,
                "reference": options.reference.name,
                "reference_sha256": reference_digest,
                "archive_sha256": hashlib.sha256(read_resource_bytes(archive)).hexdigest(),
                "norm_source_sha256": source_digest,
                **compare(BrokenP2(archive), reference, options.order),
            }
            rows.append(row)
            print(row, flush=True)
            destination.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.spe10_adaptive_norms").main()
