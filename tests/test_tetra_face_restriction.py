"""Exact nodal face support without altering the executed volume basis."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from simplex_native_bounds import reference_roundoff_bounds

from pymhm.fem.scalar.tetrahedron import tetra_basis, tetra_face_basis
from pymhm.fem.scalar.tetrahedron_topology import tetra_indices


@pytest.mark.parametrize("degree", range(1, 7))
@pytest.mark.parametrize("opposite", range(4))
def test_face_support_excludes_interior_values_and_retains_executed_basis(
    degree: int, opposite: int
) -> None:
    """A trace is independent of arbitrarily large coefficients outside its support."""
    bary = np.zeros((9, 4))
    bary[:, np.arange(4) != opposite] = np.random.default_rng(85).dirichlet(np.ones(3), 9)
    indices = tetra_indices(degree)
    support = indices[:, opposite] == 0
    volume = tetra_basis(degree, bary)[0]
    restricted = tetra_face_basis(degree, bary, opposite_vertex=opposite)
    assert_array_equal(restricted[:, support], volume[:, support])
    assert_array_equal(restricted[:, ~support], 0)
    assert_array_equal(tetra_basis(degree, bary)[0], volume)

    # An independently evaluated affine field belongs to every positive degree.
    nodes = indices / degree
    coefficients = 1 + nodes[:, 1:] @ np.array([2.0, -1.0, 3.0])
    expected = 1 + bary[:, 1:] @ np.array([2.0, -1.0, 3.0])
    bound = reference_roundoff_bounds("tetrahedron", degree, bary, nodes)[0]
    action = restricted @ coefficients
    assert np.all(abs(action - expected) <= bound[:, support] @ abs(coefficients[support]))
    coefficients[~support] = 1e100
    assert_array_equal(restricted @ coefficients, action)


@pytest.mark.parametrize("opposite", [-1, 4, True, 1.5])
def test_face_support_rejects_invalid_opposite_vertex(opposite: int) -> None:
    with pytest.raises(ValueError, match="opposite_vertex"):
        tetra_face_basis(1, np.array([[0.0, 0.2, 0.3, 0.5]]), opposite_vertex=opposite)


def test_face_support_rejects_volume_points_and_retains_empty_tables() -> None:
    with pytest.raises(ValueError, match="vanish"):
        tetra_face_basis(2, np.array([[0.1, 0.2, 0.3, 0.4]]), opposite_vertex=0)
    assert tetra_face_basis(2, np.empty((0, 4)), opposite_vertex=0).shape == (0, 10)
