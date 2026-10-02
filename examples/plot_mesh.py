"""Shared rendering of macrocell boundaries and their intersections with profiles."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from pymhm import TriangleMesh

if TYPE_CHECKING:
    from matplotlib.collections import LineCollection


def draw_macro_mesh(axis: Any, mesh: TriangleMesh, *, label: bool = False) -> LineCollection:
    """Overlay the actual macrofaces with a light halo and dark vector contours."""
    import matplotlib.patheffects as effects
    from matplotlib.collections import LineCollection

    width = max(0.3, min(0.9, 6 / np.sqrt(len(mesh.faces))))
    artist = LineCollection(
        mesh.points[mesh.faces],
        colors="#23323a",
        linewidths=width,
        zorder=3,
        label="Macro mesh" if label else "_nolegend_",
    )
    artist.set_path_effects(
        [effects.Stroke(linewidth=width + 0.6, foreground="white", alpha=0.7), effects.Normal()]
    )
    axis.add_collection(artist)
    return artist


def macro_profile_breaks(mesh: TriangleMesh, start: Any, end: Any) -> NDArray[np.float64]:
    """Return sorted profile fractions at intersections with actual macrofaces.

    Fractions parameterize ``start + t * (end - start)``. For a profile lying
    along a macroface, its endpoints are included. Duplicate vertex crossings
    are grouped within roundoff tolerance, preserving computed coordinates.
    Fractions are not rounded: doing so would move one-sided field samples
    away from their actual macrointerface on highly refined local meshes.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    if start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all():
        raise ValueError("profile endpoints must be finite two-dimensional coordinates")
    direction = end - start
    length_squared = float(direction @ direction)
    if length_squared == 0:
        raise ValueError("profile endpoints must be distinct")
    edges = mesh.points[mesh.faces]
    displacement = edges[:, 0] - start
    tangent = edges[:, 1] - edges[:, 0]
    denominator = direction[0] * tangent[:, 1] - direction[1] * tangent[:, 0]
    tolerance = 64 * np.finfo(float).eps * np.sqrt(length_squared)
    nonparallel = np.abs(denominator) > tolerance * np.linalg.norm(tangent, axis=1)
    fractions = (
        displacement[nonparallel, 0] * tangent[nonparallel, 1]
        - displacement[nonparallel, 1] * tangent[nonparallel, 0]
    ) / denominator[nonparallel]
    face_fractions = (
        displacement[nonparallel, 0] * direction[1] - displacement[nonparallel, 1] * direction[0]
    ) / denominator[nonparallel]
    valid = (face_fractions >= -1e-12) & (face_fractions <= 1 + 1e-12)
    parallel_offset = displacement[:, 0] * direction[1] - displacement[:, 1] * direction[0]
    coincident = ~nonparallel & (np.abs(parallel_offset) <= tolerance)
    endpoints = ((edges[coincident] - start) @ direction / length_squared).ravel()
    values = np.r_[0.0, fractions[valid], endpoints, 1.0]
    values = values[(values >= -1e-12) & (values <= 1 + 1e-12)]
    values = np.sort(np.clip(values, 0, 1))
    result = values[np.r_[True, np.diff(values) > 64 * np.finfo(float).eps]]
    result[0], result[-1] = 0.0, 1.0
    return result


def mark_macro_interfaces(axis: Any, positions: Any, *, label: bool = False) -> None:
    """Mark profile crossings with dotted lines behind the unmodified field curves."""
    for index, position in enumerate(positions):
        axis.axvline(
            position,
            color="#66737a",
            linewidth=0.65,
            linestyle=":",
            alpha=0.7,
            zorder=0,
            label="Macro interface" if label and index == 0 else "_nolegend_",
        )
