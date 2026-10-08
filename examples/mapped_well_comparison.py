"""Joint physical norms with reused maps for archived mapped RT1 well fields.

Each candidate retains the quadrature, common partition, batch order and geometry
sentinel of the pairwise comparison. Reuse changes only repeated evaluations;
no geometry, field, quadrature point or norm weight is approximated or omitted.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.fem.hdiv.mapped import mapped_rt_basis
from pymhm.meshes.hexahedron import HexMesh, cube_quadrature, hexahedral_mapping

if __package__:
    from .mapped_well_fields import MappedWellField
else:
    from examples.mapped_well_fields import MappedWellField

Geometry = tuple[np.ndarray, np.ndarray, np.ndarray]
Tables = tuple[np.ndarray, np.ndarray]


def _values(
    field: MappedWellField,
    cells: np.ndarray,
    tables: Tables,
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the original RT1 Piola evaluation using a previously evaluated map."""
    unit_basis, modal = tables
    _, jacobian, determinant = geometry
    reference_flux = np.einsum("ti,qia->tqa", field.flux[cells], unit_basis[0], optimize=True)
    physical_flux = np.einsum("tqab,tqb->tqa", jacobian, reference_flux) / determinant[..., None]
    return field.pressure[cells] @ modal.T, physical_flux


def _result(totals: np.ndarray, order: int | tuple[int, int, int], common: np.ndarray) -> dict:
    """Return the same physical norms and zero-denominator convention as one pair."""
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


@dataclass(frozen=True)
class _GroupIntegrator:
    """Read-only integration plan copied once per process or shared by threads."""

    fine: MappedWellField
    candidates: Sequence[MappedWellField]
    fine_ids: np.ndarray
    coarse_ids: np.ndarray
    grouped_ids: np.ndarray
    group_offsets: np.ndarray
    offsets: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    fine_ratio: np.ndarray
    coarse_ratio: np.ndarray
    geometry_owner: list[int]
    pressure_offset: float

    def __call__(self, group_id: int) -> np.ndarray:
        """Evaluate each batch independently without reducing across batches or groups."""
        fine, candidates = self.fine, self.candidates
        fine_ids, coarse_ids = self.fine_ids, self.coarse_ids
        grouped_ids, group_offsets = self.grouped_ids, self.group_offsets
        points, weights = self.points, self.weights
        fine_ratio, coarse_ratio = self.fine_ratio, self.coarse_ratio
        geometry_owner, pressure_offset = self.geometry_owner, self.pressure_offset
        offset = self.offsets[group_id]
        contributions = []
        selected = grouped_ids[group_offsets[group_id] : group_offsets[group_id + 1]]
        fine_points = (points + offset[:3]) / fine_ratio
        coarse_points = (points + offset[3:]) / coarse_ratio
        fine_basis, _, fine_modal = mapped_rt_basis(HexMesh.unit_cube(), 1, fine_points)
        coarse_basis, _, coarse_modal = mapped_rt_basis(HexMesh.unit_cube(), 1, coarse_points)
        for start in range(0, len(selected), 64):
            ids = selected[start : start + 64]
            fine_geometry = hexahedral_mapping(fine.vertices[fine_ids[ids]], fine_points)
            fp, fq = _values(fine, fine_ids[ids], (fine_basis, fine_modal), fine_geometry)
            x, _, det = fine_geometry
            physical_weights = det * weights / np.prod(fine_ratio)
            reference_terms = [
                np.sum(physical_weights * fp * fp),
                np.sum(physical_weights * np.sum(fq * fq, axis=2)),
                np.sum(physical_weights * (fp - pressure_offset) ** 2),
            ]
            geometries: dict[int, Geometry] = {}
            batch = np.empty((len(candidates), 5))
            for index, field in enumerate(candidates):
                owner = geometry_owner[index]
                if owner not in geometries:
                    geometries[owner] = hexahedral_mapping(
                        field.vertices[coarse_ids[ids]], coarse_points
                    )
                coarse_geometry = geometries[owner]
                cp, cq = _values(
                    field, coarse_ids[ids], (coarse_basis, coarse_modal), coarse_geometry
                )
                if not np.allclose(x, coarse_geometry[0], rtol=0.0, atol=2e-12):
                    raise ValueError("field hierarchies have inconsistent physical geometry")
                batch[index] = [
                    np.sum(physical_weights * (fp - cp) ** 2),
                    np.sum(physical_weights * np.sum((fq - cq) ** 2, axis=2)),
                    *reference_terms,
                ]
            contributions.append(batch)
        return np.asarray(contributions)


_PROCESS_INTEGRATOR: _GroupIntegrator | None = None


def _initialize_integrator(integrator: _GroupIntegrator) -> None:
    """Install one private field/geometry plan per portable spawned worker."""
    global _PROCESS_INTEGRATOR
    _PROCESS_INTEGRATOR = integrator


def _process_group(group_id: int) -> np.ndarray:
    """Evaluate one offset group with bounded worker-local native threading."""
    if _PROCESS_INTEGRATOR is None:
        raise RuntimeError("integration worker was not initialized")
    with threadpool_limits(1):
        return _PROCESS_INTEGRATOR(group_id)


def _same_shape_differences(
    fine: MappedWellField,
    candidates: Sequence[MappedWellField],
    order: int | tuple[int, int, int],
    pressure_offset: float,
    progress: Callable[[int, int], None] | None,
    workers: int,
    backend: Literal["thread", "process"],
) -> list[dict]:
    """Stream one common partition, preserving each candidate's original reduction order."""
    fine_shape, coarse_shape = np.array(fine.shape), np.array(candidates[0].shape)
    common = np.maximum(fine_shape, coarse_shape)
    if np.any(common % fine_shape) or np.any(common % coarse_shape):
        raise ValueError("physical difference requires nested grids in each coordinate direction")
    base_count = len(fine.vertices) // np.prod(fine_shape)
    if any(base_count != len(field.vertices) // np.prod(coarse_shape) for field in candidates):
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
    grouped_ids = np.argsort(group, kind="stable")
    group_offsets = np.r_[0, np.cumsum(np.bincount(group, minlength=len(offsets)))]
    geometry_owner = []
    for index, field in enumerate(candidates):
        geometry_owner.append(
            next(
                (j for j in range(index) if np.array_equal(field.vertices, candidates[j].vertices)),
                index,
            )
        )

    integrate_group = _GroupIntegrator(
        fine,
        candidates,
        fine_ids,
        coarse_ids,
        grouped_ids,
        group_offsets,
        offsets,
        points,
        weights,
        fine_ratio,
        coarse_ratio,
        geometry_owner,
        pressure_offset,
    )
    totals = np.zeros((len(candidates), 5))
    # Bounded groups prevent excessive queued work after a geometry sentinel fails.
    if workers == 1:
        context = nullcontext(None)
    elif backend == "process":
        context = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_integrator,
            initargs=(integrate_group,),
        )
    else:
        context = ThreadPoolExecutor(max_workers=workers)
    with context as pool:
        for first in range(0, len(offsets), workers):
            group_ids = range(first, min(first + workers, len(offsets)))
            rows = (
                map(integrate_group, group_ids)
                if pool is None
                else pool.map(
                    _process_group if backend == "process" else integrate_group, group_ids
                )
            )
            for group_id, contributions in zip(group_ids, rows, strict=True):
                for batch in contributions:
                    totals += batch
                if progress is not None:
                    progress(group_id + 1, len(offsets))
    return [_result(row, order, common) for row in totals]


def differences(
    fine: MappedWellField,
    candidates: Sequence[MappedWellField],
    order: int | tuple[int, int, int] = 8,
    pressure_offset: float = 0.0,
    *,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
    backend: Literal["thread", "process"] = "thread",
) -> list[dict]:
    """Compare candidates in input order, sharing only exactly identical geometry.

    Candidates with different grid shapes are evaluated in separate groups so
    each pair retains its original common partition and quadrature. Within a
    group the reference evaluation is reused; candidate geometry is reused only
    after an exact array-equality check. Every pair still executes the original
    physical-coordinate sentinel, including its absolute tolerance of 2e-12.
    Relative norms use the first field; zero denominators return None. Progress
    receives completed and total offset groups, independently for each shape.
    Threads or portable spawned processes evaluate distinct groups; individual
    batch contributions are accumulated in the original serial order. Process
    workers each own one copied plan and limit native pools to one thread. In
    thread mode, bound BLAS pools in the caller to avoid nested parallelism.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if backend not in {"thread", "process"}:
        raise ValueError("backend must be thread or process")
    if not candidates:
        raise ValueError("at least one candidate field is required")
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index, field in enumerate(candidates):
        grouped.setdefault(field.shape, []).append(index)
    result: list[dict] = [{} for _ in candidates]
    for indices in grouped.values():
        rows = _same_shape_differences(
            fine,
            [candidates[i] for i in indices],
            order,
            pressure_offset,
            progress,
            workers,
            backend,
        )
        for index, row in zip(indices, rows, strict=True):
            result[index] = row
    return result
