"""Declared moments fix the numerical complement and its coefficient convention."""

import numpy as np
import pytest

from pymhm.core.subspaces import moment_complement


def test_single_moment_coordinates_and_empty_spaces():
    moment = np.array([2.0, 5.0, -1.0, 0.0])
    basis = moment_complement(moment)
    np.testing.assert_array_equal(basis[[0, 2, 3]], np.eye(3))
    np.testing.assert_array_equal(basis[1], [-0.4, 0.2, 0.0])
    np.testing.assert_allclose(moment @ basis, 0.0, atol=1e-12, rtol=1e-10)
    np.testing.assert_array_equal(moment_complement(np.empty((3, 0))), np.eye(3))
    np.testing.assert_array_equal(moment_complement(np.empty((3, 0)), pivots=[]), np.eye(3))
    assert moment_complement(np.empty((0, 0))).shape == (0, 0)
    assert moment_complement(np.eye(2)).shape == (2, 0)
    assert moment_complement([1.0]).shape == (1, 0)


@pytest.mark.parametrize("pivots", [None, [1, 0], [0, 3]])
def test_independent_moments_preserve_nullspace_and_declared_free_coordinates(pivots):
    moments = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, -1.0], [0.0, 2.0]])
    basis = moment_complement(moments, pivots=pivots)
    np.testing.assert_allclose(moments.T @ basis, 0.0, atol=1e-12, rtol=1e-10)
    assert np.linalg.matrix_rank(basis) == 2
    if pivots is not None:
        free = np.setdiff1d(np.arange(4), pivots)
        np.testing.assert_array_equal(basis[free], np.eye(2))
        rotated = moments @ np.array([[1.0, 2.0], [-1.0, 3.0]])
        np.testing.assert_allclose(
            moment_complement(rotated, pivots=pivots), basis, atol=1e-12, rtol=1e-10
        )


@pytest.mark.parametrize(
    ("moments", "pivots", "message"),
    [
        (np.ones((1, 1, 1)), None, "vector or a matrix"),
        ([[1.0, 1.0]], None, "more independent"),
        ([0.0, 0.0], None, "linearly independent"),
        ([1.0, 2.0], [True], "valid integer"),
        ([1.0, 2.0], [], "valid integer"),
        ([1.0, 2.0], [-1], "valid integer"),
        ([1.0, 2.0], [2], "valid integer"),
        (np.eye(2), [0, 0], "valid integer"),
        ([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]], [0, 1], "determine all moments"),
    ],
)
def test_invalid_moment_or_pivot_declarations(moments, pivots, message):
    with pytest.raises(ValueError, match=message):
        moment_complement(moments, pivots=pivots)
