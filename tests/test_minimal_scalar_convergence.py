"""Original-field acceptance is independent of a reported reduced residual."""

from dataclasses import replace

import numpy as np
import pytest

from examples.minimal_scalar_convergence import hybrid_original, residual_record
from pymhm.hybrid import HybridSystem, LocalProblem


def test_reduced_solution_does_not_certify_corrupted_reconstruction() -> None:
    """Reject a changed physical field while its original reduced vector is intact."""
    problem = LocalProblem([[1.0]], [[1.0]], [1.0], [0])
    system = HybridSystem([problem], boundary_load=[2.0])
    solution = system.solve()
    record = hybrid_original(system, solution, {}, np.array([2.0]))
    assert record["full_uncondensed_relative_to_physical_rhs"]["relative"] < 1e-10
    corrupted = replace(solution, fields=(solution.fields[0] + 1e-7,))
    assert corrupted.residual == solution.residual
    with pytest.raises(ArithmeticError, match="Original rows"):
        hybrid_original(system, corrupted, {}, np.array([2.0]))


def test_zero_original_forcing_has_no_absolute_acceptance_floor() -> None:
    """A nonzero defect with zero physical scale cannot be accepted."""
    assert residual_record([0.0], [0.0])["relative"] == 0.0
    with pytest.raises(ArithmeticError, match="Original rows"):
        residual_record([1e-30], [0.0])


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_original_rows_are_rejected(value: float) -> None:
    """Nonfinite defects never enter a public initial convergence record."""
    with pytest.raises(ArithmeticError, match="Original rows"):
        residual_record([value], [1.0])
