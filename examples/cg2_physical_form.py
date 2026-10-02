"""Physical CG2 diffusion action on material-fitted Cartesian triangulations.

The diagonal runs from southwest to northeast. Each rectangle has a positive
scalar, constant permeability. Quadratic gradients are integrated by exact
degree-two triangle moments. Coefficients may be an unevaluated sum of two
binary64 arrays; their nodal differences are formed separately. This operator
does not assemble or modify a sparse matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PATTERNS = (
    (
        np.array([[0, 0], [2, 0], [2, 2], [1, 0], [2, 1], [1, 1]]),
        np.array([[-1, 0], [1, -1], [0, 1]], dtype=np.longdouble),
    ),
    (
        np.array([[0, 0], [0, 2], [2, 2], [0, 1], [1, 2], [1, 1]]),
        np.array([[0, -1], [-1, 1], [1, 0]], dtype=np.longdouble),
    ),
)


@dataclass(frozen=True)
class CG2DiffusionForm:
    """Exact polynomial weak form evaluated with extended local accumulation.

    Axes and permeability are copied and read-only. The latter has shape
    ``(len(y_axis)-1, len(x_axis)-1)``. The source is constant. No assumption
    that an assembled CSR matrix has zero row sums enters this action. Platforms
    without additional long-double significand bits are rejected explicitly.
    """

    x_axis: np.ndarray
    y_axis: np.ndarray
    permeability: np.ndarray
    source: float = 1.0

    def __post_init__(self) -> None:
        """Validate physical intervals, positive coefficients and precision support."""
        if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
            raise RuntimeError("physical-form refinement requires extended longdouble precision")
        for name in ("x_axis", "y_axis", "permeability"):
            values = np.array(getattr(self, name), dtype=np.longdouble, copy=True)
            if not np.all(np.isfinite(values)):
                raise ValueError("physical geometry and permeability must be finite")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for axis in (self.x_axis, self.y_axis):
            if axis.ndim != 1 or len(axis) < 2 or np.any(np.diff(axis) <= 0):
                raise ValueError("physical axes must be strictly increasing vectors")
        if self.permeability.shape != (len(self.y_axis) - 1, len(self.x_axis) - 1):
            raise ValueError("permeability must contain one value per rectangle")
        if np.any(self.permeability <= 0) or not np.isfinite(self.source):
            raise ValueError("permeability must be positive and source finite")

    @property
    def shape(self) -> tuple[int, int]:
        """Return the canonical quadratic nodal array shape (y, x)."""
        return 2 * len(self.y_axis) - 1, 2 * len(self.x_axis) - 1

    def load(self) -> np.ndarray:
        """Integrate the constant source using exact CG2 nodal moments."""
        area_source = (
            self.source * np.diff(self.y_axis)[:, None] * np.diff(self.x_axis)[None, :] / 6
        )
        result = np.zeros(self.shape, dtype=np.longdouble)
        result[1::2, 1::2] = 2 * area_source
        result[::2, 1::2][:-1] += area_source
        result[::2, 1::2][1:] += area_source
        result[1::2, ::2][:, :-1] += area_source
        result[1::2, ::2][:, 1:] += area_source
        return result

    def action(self, high: np.ndarray, low: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the physical stiffness action, preserving constant components.

        Local gradients use only ``p_i-p_0``. Differentiating that expression
        gives five independent nodal forces and their negative sum for node
        zero. This is the unchanged finite-element weak form, not a diagonal
        adjustment of a previously assembled matrix. Both coefficient arrays
        participate before cancellation and scatter accumulation.
        """
        high = np.asarray(high)
        low = np.zeros_like(high) if low is None else np.asarray(low)
        if high.shape != self.shape or low.shape != self.shape:
            raise ValueError("coefficient components must have the canonical CG2 shape")
        if not np.all(np.isfinite(high)) or not np.all(np.isfinite(low)):
            raise ValueError("coefficient components must be finite")
        nx, ny = len(self.x_axis) - 1, len(self.y_axis) - 1
        dx, dy = np.diff(self.x_axis), np.diff(self.y_axis)
        result = np.zeros(self.shape, dtype=np.longdouble)
        bary = np.full((3, 3), np.longdouble(1) / 6)
        np.fill_diagonal(bary, np.longdouble(2) / 3)
        for first in range(0, nx * ny, 4096):
            ids = np.arange(first, min(first + 4096, nx * ny))
            i, j = ids % nx, ids // nx
            lengths = np.column_stack((dx[i], dy[j]))
            area_k = dx[i] * dy[j] * self.permeability[j, i] / 2
            for offsets, derivative in _PATTERNS:
                ix, iy = 2 * i[:, None] + offsets[:, 0], 2 * j[:, None] + offsets[:, 1]
                principal = high[iy, ix].astype(np.longdouble)
                correction = low[iy, ix].astype(np.longdouble)
                difference = (principal[:, 1:] - principal[:, :1]) + (
                    correction[:, 1:] - correction[:, :1]
                )
                gradients = np.concatenate(
                    (
                        (4 * bary - 1)[..., None] * derivative,
                        np.stack(
                            [
                                4
                                * (
                                    bary[:, a, None] * derivative[b]
                                    + bary[:, b, None] * derivative[a]
                                )
                                for a, b in ((0, 1), (1, 2), (2, 0))
                            ],
                            axis=1,
                        ),
                    ),
                    axis=1,
                )
                physical_gradient = np.einsum("qia,ti->tqa", gradients[:, 1:], difference)
                physical_gradient /= lengths[:, None]
                tail = (
                    area_k[:, None]
                    * np.einsum(
                        "qia,tqa->ti", gradients[:, 1:], physical_gradient / lengths[:, None]
                    )
                    / 3
                )
                local = np.column_stack((-tail.sum(axis=1), tail))
                np.add.at(result, (iy, ix), local)
        return result
