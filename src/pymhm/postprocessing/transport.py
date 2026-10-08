"""Physical scalar transport trajectories and stated discrete mass diagnostics."""

from dataclasses import dataclass

import numpy as np

from pymhm.core.validation import FloatArray
from pymhm.postprocessing.solutions import ScalarSolution


@dataclass(frozen=True)
class TransientTransportResult:
    """Time history, mass diagnostics and the number of distinct offline operators.

    ``times`` includes the initial time and the retained output times;
    ``solutions`` excludes the initial state. ``integration_times`` records the
    complete time grid, including steps discarded by ``output_steps``. The balance
    residuals sum the original discrete physical equations on each macrocell,
    including essential-boundary reaction forces. They measure the discrete
    weak balance, not independent error or fine-cell conservation.
    When requested, ``original_residual_norms`` and ``original_rhs_norms``
    include every executed step, in ``integration_times[1:]`` order. They check
    the full original physical rows and free trace equations, not just their
    macro sums, using the shared original-equation convention.
    """

    times: FloatArray
    solutions: tuple[ScalarSolution, ...]
    initial_values: tuple[FloatArray, ...]
    mass_moments: tuple[FloatArray, ...]
    balance_residuals: tuple[FloatArray, ...]
    operator_builds: int
    integration_times: FloatArray | None = None
    original_residual_norms: FloatArray | None = None
    original_rhs_norms: FloatArray | None = None

    def total_mass(self) -> FloatArray:
        """Integrate capacity*u at the initial time and every retained output time."""
        fields = (self.initial_values, *(solution.values for solution in self.solutions))
        return np.asarray(
            [
                sum(
                    float(moment @ value)
                    for moment, value in zip(self.mass_moments, state, strict=True)
                )
                for state in fields
            ]
        )
