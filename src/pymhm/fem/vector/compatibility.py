"""Dimension-specific sufficient compatibility conditions for declared vector traces."""

import numpy as np

from pymhm.fem.traces.interval import SkeletonSpace


def require_strain_trace_compatibility(
    skeleton: SkeletonSpace, degree: int, refinement: int
) -> None:
    """Enforce the two-dimensional matching-mesh Fortin conditions of Lemma 4.5.

    These are sufficient conditions, not a characterization of every stable
    pair. They prevent invisible trace modes before factorization. Continuous
    face subspaces inherit admissibility from their discontinuous containing space.
    """
    for space in skeleton.faces:
        positions = np.asarray(space.breaks) * refinement
        if not np.allclose(positions, np.round(positions), atol=1e-12, rtol=0):
            raise ValueError("GaLS requires trace segments aligned with the local fine mesh")
        for order, intervals in zip(space.degrees, np.diff(np.round(positions)), strict=True):
            if degree >= order + 2:
                required = 1
            elif degree >= order + 1:
                required = 2
            elif degree >= order:
                required = 5 - min(order, 3)
            else:
                raise ValueError("GaLS local degree must not be lower than the trace degree")
            if intervals < required:
                raise ValueError(
                    f"GaLS trace/local compatibility requires at least {required} fine intervals "
                    "per trace segment for these polynomial degrees"
                )
