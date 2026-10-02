"""Independent Basix verification of tensor-product interpolation and derivatives."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.quadrilateral import qk_basis, quadrilateral_quadrature


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
def test_tensor_product_basis_matches_native_basix(degree):
    basix = pytest.importorskip("basix")
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        degree,
        basix.LagrangeVariant.equispaced,
    )
    ours = np.array(
        [(x, y) for y in np.linspace(0, 1, degree + 1) for x in np.linspace(0, 1, degree + 1)]
    )
    indices = np.argmin(np.linalg.norm(ours[:, None] - element.points[None], axis=2), axis=1)
    assert len(np.unique(indices)) == len(ours)
    assert_allclose(element.points[indices], ours, atol=1e-15)
    points, _ = quadrilateral_quadrature(6)
    table = element.tabulate(1, points)
    values, derivatives = qk_basis(degree, points)
    assert_allclose(values, table[0, :, :, 0][:, indices], atol=7e-14)
    assert_allclose(derivatives[..., 0], table[1, :, :, 0][:, indices], atol=3e-13)
    assert_allclose(derivatives[..., 1], table[2, :, :, 0][:, indices], atol=3e-13)
