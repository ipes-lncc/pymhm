"""Sample broken scalar or vector Pk fields without gluing macro interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymhm import TriangleMesh
from pymhm.fem.scalar.triangle import nodal_space, reference_basis, tabulate


def sample_field(
    meshes: Any, fields: Any, degree: int, refinement: int = 3
) -> dict[str, np.ndarray]:
    """Evaluate values and physical gradients on separate fine-element display grids.

    Every point retains its incident field index, local cell index and physical
    cell center. These declarations let a comparison choose the same side of
    an interface on a different mesh, without merging coincident coordinates.
    """
    template = TriangleMesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]])
    ).submesh(0, refinement)
    bary = np.column_stack((1 - template.points.sum(axis=1), template.points))
    points, cells, values, derivatives = [], [], [], []
    centers, cell_owners, field_owners = [], [], []
    offset = 0
    for field_index, (mesh, field) in enumerate(zip(meshes, fields, strict=True)):
        dofs, _, basis, gradient, _ = tabulate(mesh, degree, bary)
        coordinates = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        points.append(coordinates.reshape(-1, 2))
        centers.append(np.repeat(mesh.points[mesh.cells].mean(axis=1), len(bary), axis=0))
        cell_owners.append(np.repeat(np.arange(len(mesh.cells), dtype=np.int64), len(bary)))
        field_owners.append(np.full(len(mesh.cells) * len(bary), field_index, dtype=np.int64))
        cells.extend(template.cells + offset + len(bary) * i for i in range(len(mesh.cells)))
        sampled = np.einsum("qi,ti...->tq...", basis, field[dofs])
        differentiated = np.einsum("tqia,ti...->tq...a", gradient, field[dofs])
        values.append(sampled.reshape((-1, *sampled.shape[2:])))
        derivatives.append(differentiated.reshape((-1, *differentiated.shape[2:])))
        offset += len(mesh.cells) * len(bary)
    return {
        "points": np.concatenate(points),
        "cells": np.concatenate(cells),
        "values": np.concatenate(values),
        "gradient": np.concatenate(derivatives),
        "incident_centers": np.concatenate(centers),
        "incident_cells": np.concatenate(cell_owners),
        "field_indices": np.concatenate(field_owners),
    }


def local_values(
    mesh: TriangleMesh, field: np.ndarray, degree: int, points: np.ndarray
) -> np.ndarray:
    """Evaluate a field from one specified macrocell at physical points."""
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    coordinates = np.einsum("tij,tqj->tqi", inverse, points[None] - vertices[:, None, 0])
    bary = np.concatenate((1 - coordinates.sum(axis=2, keepdims=True), coordinates), axis=2)
    selected = np.argmax(bary.min(axis=2), axis=0)
    point_bary = bary[selected, np.arange(len(points))]
    if np.min(point_bary) < -2e-10:
        raise ValueError("sample lies outside its prescribed macrocell")
    dofs, _ = nodal_space(mesh, degree)
    return np.einsum(
        "qi,qi...->q...", reference_basis(degree, point_bary)[0], field[dofs[selected]]
    )


def sample_profile(
    solution: Any, fields: Any, degree: int, *, height: float = 0.37
) -> dict[str, np.ndarray]:
    """Return independent one-sided segments on a horizontal unit-square profile."""
    return sample_segment_profile(solution, fields, degree, (0.0, height), (1.0, height))


def sample_segment_profile(
    solution: Any, fields: Any, degree: int, start: Any, end: Any
) -> dict[str, np.ndarray]:
    """Evaluate a broken nodal field separately on each crossed macrocell.

    Parameters increase from zero at ``start`` to one at ``end``. Coincident
    endpoint coordinates retain both one-sided values; plot each returned row
    separately. The segment must lie in the domain and must not follow a
    macroface, where selecting a side would require an additional convention.
    """
    from examples.plot_mesh import macro_profile_breaks

    mesh = solution.skeleton.mesh
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    breaks = macro_profile_breaks(mesh, start, end)
    vertices = mesh.points[mesh.cells]
    inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
    parameters, physical, numerical = [], [], []
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        midpoint = start + (left + right) / 2 * (end - start)
        coordinates = np.einsum("tij,tj->ti", inverse, midpoint - vertices[:, 0])
        bary = np.column_stack((1 - coordinates.sum(axis=1), coordinates))
        contained = np.flatnonzero(bary.min(axis=1) >= -64 * np.finfo(float).eps)
        if not len(contained):
            raise ValueError("profile lies outside the macro mesh")
        if len(contained) != 1:
            raise ValueError("profile coincides with a macroface; select a one-sided line")
        cell = int(contained[0])
        parameter = np.linspace(left, right, 121)
        points = start + parameter[:, None] * (end - start)
        parameters.append(parameter)
        physical.append(points)
        numerical.append(local_values(solution.local_meshes[cell], fields[cell], degree, points))
    return {
        "profile_parameter": np.stack(parameters),
        "profile_points": np.stack(physical),
        "profile_values": np.stack(numerical),
        "profile_breaks": breaks,
    }


def sample_darcy_pressure_profile(solution: Any, start: Any, end: Any) -> dict[str, np.ndarray]:
    """Return exact P1 or P0 pressure segments with independent fine-cell endpoints.

    Each segment belongs to one fine triangle. Plot the segments separately:
    joining their endpoint values would interpolate across a pressure jump.
    A line coincident with a fine interface has no unique one-sided value and
    is rejected. Macroface intersections are retained separately for marking.
    """
    from examples.plot_mesh import macro_profile_breaks

    first, last = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    macro_breaks = macro_profile_breaks(solution.skeleton.mesh, first, last)
    parameters, locations, values = [], [], []
    for mesh, pressure in zip(solution.local_meshes, solution.pressure, strict=True):
        vertices = mesh.points[mesh.cells]
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        breaks = macro_profile_breaks(mesh, first, last)
        for left, right in zip(breaks[:-1], breaks[1:], strict=True):
            midpoint = first + (left + right) / 2 * (last - first)
            xi = np.einsum("tij,tj->ti", inverse, midpoint - vertices[:, 0])
            bary = np.column_stack((1 - xi.sum(axis=1), xi))
            contained = np.flatnonzero(bary.min(axis=1) >= -64 * np.finfo(float).eps)
            if not len(contained):
                continue
            if len(contained) != 1:
                raise ValueError("profile coincides with a fine interface; select a one-sided line")
            cell = int(contained[0])
            parameter = np.array([left, right])
            points = first + parameter[:, None] * (last - first)
            if solution.formulation == "mixed":
                sampled = np.full(2, pressure[cell])
            else:
                xi = (points - vertices[cell, 0]) @ inverse[cell].T
                sampled = np.column_stack((1 - xi.sum(axis=1), xi)) @ pressure[mesh.cells[cell]]
            parameters.append(parameter)
            locations.append(points)
            values.append(sampled)
    if not parameters:
        raise ValueError("profile does not intersect the finite-element mesh")
    ordering = np.argsort(np.array(parameters)[:, 0])
    parameter = np.array(parameters)[ordering]
    if not np.isclose(np.diff(parameter, axis=1).sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("profile must cross a complete nonoverlapping mesh partition")
    return {
        "profile_parameter": parameter,
        "profile_points": np.array(locations)[ordering],
        "profile_values": np.array(values)[ordering],
        "profile_breaks": macro_breaks,
    }
