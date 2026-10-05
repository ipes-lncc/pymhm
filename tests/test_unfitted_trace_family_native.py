"""Native symmetric factorization and spawn controls for scalar trace reuse."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from examples.unfitted_trace_family import ScalarTraceFamily
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


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
        assert serial.degree == parallel.degree == 3
        assert serial.quadrature_order == parallel.quadrature_order == 6
        for name in ("points", "cells", "faces", "normals"):
            assert_array_equal(
                getattr(serial.skeleton.mesh, name), getattr(parallel.skeleton.mesh, name)
            )
        # Independently initialized native libraries can produce slightly
        # different floating-point Basix coefficients. Verify exact geometry
        # and coordinate maps, numerical operators, and original equations
        # rather than requiring bitwise equality of independently solved lifts.
        assert_allclose(serial.matrix.toarray(), parallel.matrix.toarray(), rtol=2e-11, atol=2e-12)
        assert_allclose(serial.load, parallel.load, rtol=2e-11, atol=2e-12)
        for first, second in zip(serial.cells, parallel.cells, strict=True):
            for name in ("points", "cells", "faces", "normals"):
                assert_array_equal(getattr(first.mesh, name), getattr(second.mesh, name))
            assert_array_equal(first.trace_dofs, second.trace_dofs)
            for name in ("matrix", "coupling"):
                assert_allclose(
                    getattr(first, name).toarray(),
                    getattr(second, name).toarray(),
                    rtol=2e-11,
                    atol=2e-12,
                )
            for name in ("load", "kernel", "source", "lifts"):
                assert_allclose(getattr(first, name), getattr(second, name), rtol=2e-11, atol=2e-12)
        solutions = [family.solve(1, 2) for family in (serial, parallel)]
        direct = solve_darcy(
            mesh,
            skeleton=solutions[0][0].skeleton,
            degree=3,
            local_refinement=8,
            permeability=material,
            source=_source,
            dirichlet=_boundary,
            quadrature_order=6,
            local_solver="scipy",
        )
    for result, diagnostics in solutions:
        assert_allclose(result.hybrid.trace, direct.hybrid.trace, rtol=2e-11, atol=2e-12)
        assert_allclose(result.hybrid.coarse, direct.hybrid.coarse, rtol=2e-11, atol=2e-12)
        for name in ("pressure", "flux"):
            for actual, expected in zip(getattr(result, name), getattr(direct, name), strict=True):
                assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)
        assert_allclose(result.conservation_residuals(), 0, atol=2e-12)
        assert diagnostics["original_trace_residual"] < 1e-10
        assert diagnostics["original_local_residual_max"] < 1e-10
        assert diagnostics["local_componentwise_backward_error_max"] < 1e-10
