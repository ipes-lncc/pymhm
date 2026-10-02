"""One-sided vertical profiles of canonical CG2 fields on Cartesian triangles."""

from __future__ import annotations

import numpy as np

from examples.solve_unusual_spe10_reference import CG2Field


def vertical_traces(field: CG2Field, x: float) -> dict[str, np.ndarray]:
    """Retain separate endpoint values for every incident triangle along x=constant.

    Each row represents one open profile segment and its two limiting values.
    At a vertical grid edge, both incident columns are included. Endpoint
    coordinates remain identical on adjacent segments; no averaging, displaced
    point, or artificial line joins the independent flux limits.
    """
    xmin, xmax, _, _ = field.bounds
    if not np.isfinite(x) or x < xmin or x > xmax:
        raise ValueError("profile coordinate must belong to the closed horizontal interval")
    axes = field.x_axis
    right = min(int(np.searchsorted(axes, x, side="right") - 1), field.shape[0] - 1)
    columns = [right]
    if right > 0 and axes[right] == x:
        columns.insert(0, right - 1)
    segments, owners, sides = [], [], []
    for column in columns:
        fraction = (x - axes[column]) / (axes[column + 1] - axes[column])
        for row, (bottom, top) in enumerate(zip(field.y_axis[:-1], field.y_axis[1:], strict=True)):
            diagonal = bottom + (top - bottom) * fraction
            for lo, hi, lower in ((bottom, diagonal, True), (diagonal, top, False)):
                if hi > lo:
                    segments.append(((x, lo), (x, hi)))
                    owners.append((column, row))
                    sides.append(lower)
    points = np.asarray(segments)
    rectangles = np.asarray(owners, dtype=int)
    lower = np.asarray(sides, dtype=bool)
    pressure, _, flux = field.evaluate_cells(
        points.reshape(-1, 2), rectangles.repeat(2, axis=0), lower.repeat(2)
    )
    return {
        "points": points,
        "pressure": pressure.reshape(-1, 2),
        "flux": flux.reshape(-1, 2, 2),
        "rectangle_owners": rectangles,
        "lower_triangle": lower,
    }
