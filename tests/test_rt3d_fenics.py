"""Independent Basix checks of tetrahedral RT values and differential operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.fem.hdiv.family_3d import cell_quadrature
from pymhm.fem.hdiv.rt_3d import RTTetraFamily

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", range(4))
def test_rt_space_and_divergence_against_basix(degree):
    """Independent native RT tabulation matches values and divergence on separate rules."""
    basix = pytest.importorskip("basix")
    with threadpool_limits(1):
        native = basix.create_element(
            basix.ElementFamily.RT,
            basix.CellType.tetrahedron,
            degree + 1,
            basix.LagrangeVariant.legendre,
        )
        family = RTTetraFamily(degree)
        points, _ = cell_quadrature("tetrahedron", degree + 3)
        external = native.tabulate(0, points)[0]
        own = family.tabulate(points)[0]
        change = np.linalg.lstsq(
            external.transpose(0, 2, 1).reshape(-1, native.dim),
            own.transpose(0, 2, 1).reshape(-1, family.local_size),
            rcond=None,
        )[0]
        check, _ = cell_quadrature("tetrahedron", degree + 4)
        external = native.tabulate(1, check)
        values = np.einsum("qia,ij->qja", external[0], change)
        div = (
            sum(
                external[basix.index(*(int(i == axis) for i in range(3)))][:, :, axis]
                for axis in range(3)
            )
            @ change
        )
        actual, actualdiv, _ = family.tabulate(check)
        assert_allclose(values, actual, atol=1e-10, rtol=2e-12)
        assert_allclose(div, actualdiv, atol=2e-9, rtol=2e-12)
