"""The spatial dimension controls the sufficient L09 local polynomial degree."""

import pytest

from pymhm.fem.conditions import minimum_estimator_degree, validate_estimator_spaces


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("trace_degree", [0, 1, 2, 4])
def test_estimator_dimension_and_all_reconstruction_boundaries(dimension, trace_degree):
    """Test the exact admissible boundary and reject its immediately adjacent polynomial orders."""
    k = trace_degree + dimension
    assert minimum_estimator_degree(trace_degree, dimension) == k
    for m in range(trace_degree, k + 1):
        validate_estimator_spaces(k, trace_degree, m, dimension)
    for local, reconstructed in ((k - 1, trace_degree), (k, k + 1)):
        with pytest.raises(ValueError, match="spaces require"):
            validate_estimator_spaces(local, trace_degree, reconstructed, dimension)
    if trace_degree:
        with pytest.raises(ValueError, match="spaces require"):
            validate_estimator_spaces(k, trace_degree, trace_degree - 1, dimension)


def test_supported_dimension_is_explicit():
    """A lower-dimensional or unspecified geometry cannot silently inherit the simplex theorem."""
    for dimension in (1, 4, 2.5):
        with pytest.raises(ValueError, match="dimension"):
            minimum_estimator_degree(0, dimension)
