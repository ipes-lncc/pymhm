"""One-sided piecewise constant fields on Cartesian material grids."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pymhm.meshes.roundoff import cartesian_coordinates

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CartesianCellField:
    """Piecewise constant scalar/tensor values on a Cartesian cell grid.

    The first ``len(spacing)`` array axes are spatial (x, y, optionally z).
    Remaining axes are value components. Internal interfaces use the cell on
    their positive side; the external upper boundary uses the last cell. This
    pointwise convention resolves grid-line representations within their propagated
    physical-coordinate roundoff; it does not average material jumps. Quadrature must
    resolve those jumps when using this field as a PDE coefficient.
    """

    values: Array
    spacing: tuple[float, ...]
    origin: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and freeze independent real field and geometry arrays."""
        spacing = np.asarray(self.spacing, dtype=float)
        if spacing.shape not in ((2,), (3,)) or not np.all(np.isfinite(spacing) & (spacing > 0)):
            raise ValueError("spacing must contain two or three finite positive lengths")
        dimension = len(spacing)
        origin = (
            np.zeros(dimension) if self.origin is None else np.asarray(self.origin, dtype=float)
        )
        if origin.shape != spacing.shape or not np.isfinite(origin).all():
            raise ValueError("origin must contain one finite coordinate per spatial axis")
        values = np.asarray(self.values)
        if (
            values.ndim < dimension
            or 0 in values.shape
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
        ):
            raise ValueError("values must be a nonempty finite real array with spatial axes")
        copied = np.array(values, dtype=float, copy=True)
        copied.flags.writeable = False
        object.__setattr__(self, "values", copied)
        object.__setattr__(self, "spacing", tuple(spacing))
        object.__setattr__(self, "origin", tuple(origin))

    def __call__(self, points: Any) -> Array:
        """Evaluate points of shape ``(n, dimension)`` without extrapolation."""
        points = np.asarray(points, dtype=float)
        dimension = len(self.spacing)
        if points.ndim != 2 or points.shape[1] != dimension or not np.isfinite(points).all():
            raise ValueError("points must be a finite array of shape (n, dimension)")
        coordinates, error = cartesian_coordinates(
            points, np.asarray(self.origin), np.asarray(self.spacing)
        )
        shape = np.asarray(self.values.shape[:dimension])
        tolerance = np.maximum(32 * np.finfo(float).eps * np.maximum(shape, 1), 2 * error)
        if np.any(coordinates < -tolerance) or np.any(coordinates > shape + tolerance):
            raise ValueError("points lie outside the Cartesian field")
        levels = np.rint(coordinates)
        coordinates = np.where(abs(coordinates - levels) <= 2 * error, levels, coordinates)
        indices = np.minimum(np.maximum(coordinates, 0).astype(np.int64), shape - 1)
        return self.values[tuple(indices.T)]
