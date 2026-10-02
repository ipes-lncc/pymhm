"""Native symmetric factorization and spawn controls for scalar trace reuse."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from examples.unfitted_trace_family import ScalarTraceFamily
from pymhm.darcy import solve_darcy
from pymhm.mesh import TriangleMesh
from pymhm.reservoir import CartesianCellField


def _source(points):
    """Return a nonconstant source independently of the boundary polynomial."""
    return 1 + points[:, 0] - 0.3 * points[:, 1]


def _boundary(points):
    """Exercise nonhomogeneous Dirichlet loading as well as the volume load."""
    return 0.2 + points[:, 0] ** 2 - 0.1 * points[:, 1]


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_pardiso_spawn_preserves_equations_and_matches_direct_scipy(heterogeneous):
    """Compare native serial/spawn lifts and lower trace fields to direct equations."""
    pytest.importorskip("pypardiso")
    mesh = TriangleMesh.unit_square()
    material = (
        CartesianCellField(np.array([[2.0, 1.0], [3.0, 1.5]]), (0.5, 0.5)) if heterogeneous else 1.0
    )
    options = dict(
        trace_degree=2,
        segments=4,
        local_degree=3,
        local_refinement=8,
        permeability=material,
        source=_source,
        dirichlet=_boundary,
        quadrature_order=6,
        local_solver="pypardiso-symmetric",
    )
    with threadpool_limits(1):
        serial = ScalarTraceFamily.prepare(mesh, **options)
        parallel = ScalarTraceFamily.prepare(mesh, **options, workers=2)
        assert_array_equal(serial.matrix.toarray(), parallel.matrix.toarray())
        assert_array_equal(serial.load, parallel.load)
        for first, second in zip(serial.cells, parallel.cells, strict=True):
            assert_array_equal(first.source, second.source)
            assert_array_equal(first.lifts, second.lifts)
        result, diagnostics = parallel.solve(1, 2)
        direct = solve_darcy(
            mesh,
            skeleton=result.skeleton,
            degree=3,
            local_refinement=8,
            permeability=material,
            source=_source,
            dirichlet=_boundary,
            quadrature_order=6,
            local_solver="scipy",
        )
    assert_allclose(result.hybrid.trace, direct.hybrid.trace, rtol=2e-11, atol=2e-12)
    assert_allclose(result.hybrid.coarse, direct.hybrid.coarse, rtol=2e-11, atol=2e-12)
    for actual, expected in zip(result.pressure, direct.pressure, strict=True):
        assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
    assert_allclose(result.conservation_residuals(), 0, atol=2e-12)
    assert diagnostics["original_trace_residual"] < 1e-10
    assert diagnostics["local_componentwise_backward_error_max"] < 1e-10
