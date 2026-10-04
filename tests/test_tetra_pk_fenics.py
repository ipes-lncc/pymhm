"""Independent native Basix/UFL verification of tetrahedral P5/P6 and quadratic traces."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.tetrahedron_topology import tetra_indices, tetra_polynomials
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, tetra_trace_coupling
from pymhm.meshes.tetrahedron import TetraMesh

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("degree", [5, 6])
def test_native_basix_cardinality_gradients_hessians(degree):
    """Compare independent orthogonal-polynomial tabulation after matching equispaced nodes."""
    basix = pytest.importorskip("basix")
    with threadpool_limits(1):
        element = basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.tetrahedron,
            degree,
            basix.LagrangeVariant.equispaced,
        )
        nodes = tetra_indices(degree)[:, 1:] / degree
        distances, order = cKDTree(element.points).query(nodes)
        assert distances.max() < 2e-16
        bary = np.vstack(
            (
                np.random.default_rng(2100 + degree).dirichlet(np.ones(4), size=41),
                np.eye(4),
                [[0, 0.2, 0.3, 0.5]],
            )
        )
        values, derivatives, second = tetra_polynomials(degree, bary)
        actual_gradient = derivatives[..., 1:] - derivatives[..., :1]
        actual_hessian = (
            second[..., 1:, 1:] - second[..., 1:, :1] - second[..., :1, 1:] + second[..., :1, :1]
        )
        native = element.tabulate(2, bary[:, 1:])[:, :, order, 0]
        assert_allclose(values, native[0], atol=7e-14, rtol=3e-12)
        for a in range(3):
            multiindex = np.eye(3, dtype=int)[a]
            assert_allclose(
                actual_gradient[..., a],
                native[basix.index(*multiindex)],
                atol=8e-13,
                rtol=5e-12,
            )
            for b in range(3):
                multiindex = np.eye(3, dtype=int)[a] + np.eye(3, dtype=int)[b]
                assert_allclose(
                    actual_hessian[..., a, b],
                    native[basix.index(*multiindex)],
                    atol=2e-11,
                    rtol=8e-12,
                )


@pytest.mark.parametrize("degree", [5, 6])
def test_native_ufl_volume_and_physical_quadratic_trace(degree):
    """Compare stiffness, source and every P2 face moment on a sheared physical tetrahedron."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix")
    pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    from test_darcy3d_saddle_fenics import _local_ufl, _source

    macro = TetraMesh(
        [[0.1, 0.2, -0.1], [1.2, 0.3, 0.0], [0.3, 1.1, 0.2], [-0.1, 0.4, 1.0]],
        [[0, 1, 2, 3]],
    )
    skeleton = TriangularSkeleton(macro, degree=2)
    with threadpool_limits(1):
        native_A, native_B, native_load, _ = _local_ufl(
            macro, 0, macro, skeleton, degree, 2, (dolfinx, ufl, basix, mpi)
        )
        matrix, _, load = tetra_operators(macro, degree, source=_source, order=degree + 2)
        coupling = tetra_trace_coupling(macro, 0, macro, skeleton, degree)
    assert_allclose(matrix.toarray(), native_A.toarray(), atol=7e-12, rtol=2e-11)
    assert_allclose(load, native_load, atol=2e-14, rtol=3e-12)
    assert_allclose(coupling, native_B, atol=2e-14, rtol=3e-12)
