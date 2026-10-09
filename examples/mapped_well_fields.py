"""Evaluate and integrate archived native RT1 fields on the declared annular hierarchy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymhm.fem.hdiv.mapped import mapped_rt_basis, mapped_rt_dofs
from pymhm.io.workspace import local_resource, read_resource_bytes, read_resource_text
from pymhm.meshes.hexahedron import HexMesh, cube_quadrature, hexahedral_mapping


@dataclass(frozen=True)
class MappedWellField:
    """Physical RT1/Q1 coefficients ordered by base cell and fine tensor-grid indices."""

    vertices: np.ndarray
    pressure: np.ndarray
    flux: np.ndarray
    factor: int
    vertical_factor: int | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the actual reference-grid counts in the three physical directions."""
        return (
            self.factor,
            self.factor,
            self.factor if self.vertical_factor is None else self.vertical_factor,
        )

    @classmethod
    def load(cls, record_path: Path) -> tuple[MappedWellField, dict]:
        """Verify a field digest and convert oriented coefficients to reference-cell moments."""
        report = json.loads(read_resource_text(record_path))
        path = record_path.parent / report["archive"]
        if hashlib.sha256(read_resource_bytes(path)).hexdigest() != report["sha256"]:
            raise ValueError("mapped field archive digest mismatch")
        with np.load(local_resource(path)) as archive:
            if report.get("archive_layout") == "canonical-reference-cell":
                field = cls(
                    archive["vertices"],
                    archive["pressure"],
                    archive["flux"],
                    report["fine_factor"],
                    report["vertical_factor"],
                )
            else:
                field = cls.from_arrays(
                    dict(archive), report["fine_factor"], report["macro_factor"]
                )
        return field, report

    @classmethod
    def from_arrays(cls, arrays: dict, fine_factor: int, macro_factor: int) -> MappedWellField:
        """Use the exact integer refinement hierarchy, without coordinate matching or averaging."""
        f, m = fine_factor, macro_factor
        if f < 1 or m < 1 or f % m:
            raise ValueError("fine factor must be a positive multiple of the macro factor")
        r = f // m
        macro_count = len(arrays["macro_cells"])
        if macro_count % m**3:
            raise ValueError("macro grid does not match its declared tensor refinement")
        local, _ = HexMesh.unit_cube().submesh(0, r)
        if not np.all(arrays["local_cells"] == local.cells):
            raise ValueError("local cells must follow the declared tensor-grid ordering")
        dofs = mapped_rt_dofs(local, 1)
        signs = np.column_stack((np.repeat(local.signs, 4, axis=1), np.ones((r**3, 12))))
        canonical = arrays["flux"][:, dofs] * signs[None]
        physical = arrays["local_points"][
            np.arange(macro_count)[:, None, None], arrays["local_cells"]
        ]
        local_indices = np.indices((r, r, r)).reshape(3, -1).T
        macro_indices = np.array(np.unravel_index(np.arange(macro_count) % m**3, (m, m, m))).T
        indices = macro_indices[:, None] * r + local_indices[None]
        identifiers = ((indices[..., 0] * f + indices[..., 1]) * f + indices[..., 2]) + (
            np.arange(macro_count) // m**3
        )[:, None] * f**3
        count = macro_count * r**3
        vertices = np.empty((count, 8, 3))
        pressure = np.empty((count, 8))
        flux = np.empty((count, 36))
        vertices[identifiers.ravel()] = physical.reshape(-1, 8, 3)
        pressure[identifiers.ravel()] = arrays["pressure"].reshape(-1, 8)
        flux[identifiers.ravel()] = canonical.reshape(-1, 36)
        return cls(vertices, pressure, flux, f)

    def values(
        self,
        cells: np.ndarray,
        reference: np.ndarray,
        *,
        tables: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pressure and physical flux at shared reference coordinates in selected cells."""
        if tables is None:
            unit_basis, _, modal = mapped_rt_basis(HexMesh.unit_cube(), 1, reference)
        else:
            unit_basis, modal = tables
        _, jacobian, determinant = hexahedral_mapping(self.vertices[cells], reference)
        reference_flux = np.einsum("ti,qia->tqa", self.flux[cells], unit_basis[0], optimize=True)
        physical_flux = (
            np.einsum("tqab,tqb->tqa", jacobian, reference_flux) / determinant[..., None]
        )
        return self.pressure[cells] @ modal.T, physical_flux


def difference(
    fine: MappedWellField,
    coarse: MappedWellField,
    order: int | tuple[int, int, int] = 8,
    pressure_offset: float = 0.0,
) -> dict:
    """Integrate physical L2 differences on a common grid, using the first field's norms.

    Relative values are ``None`` when their reference norm is zero; the absolute
    norm remains defined. ``pressure_offset`` affects only the increment denominator.
    """
    fine_shape, coarse_shape = np.array(fine.shape), np.array(coarse.shape)
    common = np.maximum(fine_shape, coarse_shape)
    if np.any(common % fine_shape) or np.any(common % coarse_shape):
        raise ValueError("physical difference requires nested grids in each coordinate direction")
    base_count = len(fine.vertices) // np.prod(fine_shape)
    if base_count != len(coarse.vertices) // np.prod(coarse_shape):
        raise ValueError("field hierarchies must share their base mesh")
    fine_ratio, coarse_ratio = common // fine_shape, common // coarse_shape
    points, weights = cube_quadrature(order)
    identifiers = np.arange(base_count * np.prod(common))
    indices = np.array(np.unravel_index(identifiers % np.prod(common), common)).T
    base_ids = identifiers // np.prod(common)
    fine_ids = np.ravel_multi_index((indices // fine_ratio).T, fine_shape) + base_ids * np.prod(
        fine_shape
    )
    coarse_ids = np.ravel_multi_index(
        (indices // coarse_ratio).T, coarse_shape
    ) + base_ids * np.prod(coarse_shape)
    offsets, group = np.unique(
        np.column_stack((indices % fine_ratio, indices % coarse_ratio)), axis=0, return_inverse=True
    )
    totals = np.zeros(5)
    for group_id, offset in enumerate(offsets):
        selected = identifiers[group == group_id]
        fine_points = (points + offset[:3]) / fine_ratio
        coarse_points = (points + offset[3:]) / coarse_ratio
        fine_basis, _, fine_modal = mapped_rt_basis(HexMesh.unit_cube(), 1, fine_points)
        coarse_basis, _, coarse_modal = mapped_rt_basis(HexMesh.unit_cube(), 1, coarse_points)
        for start in range(0, len(selected), 64):
            ids = selected[start : start + 64]
            fp, fq = fine.values(fine_ids[ids], fine_points, tables=(fine_basis, fine_modal))
            cp, cq = coarse.values(
                coarse_ids[ids], coarse_points, tables=(coarse_basis, coarse_modal)
            )
            x, _, det = hexahedral_mapping(fine.vertices[fine_ids[ids]], fine_points)
            cx = hexahedral_mapping(coarse.vertices[coarse_ids[ids]], coarse_points)[0]
            if not np.allclose(x, cx, rtol=0.0, atol=2e-12):
                raise ValueError("field hierarchies have inconsistent physical geometry")
            physical_weights = det * weights / np.prod(fine_ratio)
            totals += [
                np.sum(physical_weights * (fp - cp) ** 2),
                np.sum(physical_weights * np.sum((fq - cq) ** 2, axis=2)),
                np.sum(physical_weights * fp * fp),
                np.sum(physical_weights * np.sum(fq * fq, axis=2)),
                np.sum(physical_weights * (fp - pressure_offset) ** 2),
            ]
    absolute = np.sqrt(totals[:2])
    denominators = np.sqrt(totals[2:])
    return dict(
        pressure_l2=float(absolute[0]),
        flux_l2=float(absolute[1]),
        pressure_reference_l2=float(denominators[0]),
        flux_reference_l2=float(denominators[1]),
        pressure_relative=float(absolute[0] / denominators[0]) if denominators[0] > 0 else None,
        flux_relative=float(absolute[1] / denominators[1]) if denominators[1] > 0 else None,
        pressure_reference_increment_l2=float(denominators[2]),
        pressure_increment_relative=(
            float(absolute[0] / denominators[2]) if denominators[2] > 0 else None
        ),
        quadrature_order=order,
        integration_axis_counts=common.tolist(),
    )
