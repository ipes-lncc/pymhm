"""High-order tetrahedral topology, stable derivatives and admissible MHM spaces."""

from itertools import combinations, product
from math import comb, factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import cKDTree
from simplex_native_bounds import gamma, reference_roundoff_bounds
from threadpoolctl import threadpool_limits

from pymhm import TetraMesh, TriangularSkeleton
from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.estimators.darcy_3d import estimate_darcy_error_3d
from pymhm.fem.inequalities import laplacian_inverse_bound
from pymhm.fem.reference import simplex_lagrange_basis, tabulate_reference
from pymhm.fem.scalar.tetrahedron import (
    tetra_element_tabulate,
    tetra_nodal_space,
    tetra_operators,
    tetrahedron_quadrature,
)
from pymhm.fem.scalar.tetrahedron_topology import (
    _compositions,
    tetra_indices,
    tetra_polynomials,
    tetra_values_gradients,
)


@pytest.fixture(autouse=True)
def native_threads():
    """Bound native threading for small polynomial and local saddle controls."""
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize("degree", range(1, 5))
def test_existing_topological_order_is_exact(degree):
    """Preserve the published vertex/edge/face/interior ordering through P4."""
    expected = []
    for count in range(1, 5):
        for axes in combinations(range(4), count):
            for weights in product(range(1, degree + 1), repeat=count):
                if sum(weights) == degree:
                    entry = np.zeros(4, dtype=int)
                    entry[list(axes)] = weights
                    expected.append(entry)
    assert np.array_equal(tetra_indices(degree), expected)
    assert list(_compositions(0, 1)) == []


@pytest.mark.parametrize("degree", [5, 6, 8, 12])
def test_unrestricted_topology_cardinality_and_derivative_partition(degree):
    """Cardinality and constant reproduction hold at nodes, edges and interior points."""
    indices = tetra_indices(degree)
    assert indices.shape == (comb(degree + 3, 3), 4)
    assert np.all(indices.sum(axis=1) == degree)
    assert len(np.unique(indices, axis=0)) == len(indices)
    assert not indices.flags.writeable
    import basix

    values, node_first, node_second = tetra_polynomials(degree, indices / degree)
    native = simplex_lagrange_basis("tetrahedron", degree, nodes=indices / degree)
    raw = tabulate_reference(native.element, indices[:, 1:] / degree, nderiv=2)[
        :, :, native.permutation, 0
    ]
    assert np.array_equal(values, np.eye(len(indices)))
    node_bound, _, _ = reference_roundoff_bounds(
        "tetrahedron", degree, indices / degree, indices / degree
    )
    assert np.all(abs(raw[0] - values) <= node_bound)
    assert np.array_equal(node_first[..., 0], np.zeros_like(values))
    assert np.array_equal(node_second[..., 0, :], np.zeros((*values.shape, 4)))
    assert np.array_equal(node_second[..., :, 0], np.zeros((*values.shape, 4)))
    for axis in range(3):
        direction = np.eye(3, dtype=int)[axis]
        assert np.array_equal(node_first[..., axis + 1], raw[basix.index(*direction)])
        for other in range(3):
            assert np.array_equal(
                node_second[..., axis + 1, other + 1],
                raw[basix.index(*(direction + np.eye(3, dtype=int)[other]))],
            )
    probe = np.array([[0.17, 0.21, 0.29, 0.33]])
    assert np.array_equal(
        tetra_polynomials(degree, probe)[0],
        tabulate_reference(native.element, probe[:, 1:])[:, :, native.permutation, 0][0],
    )
    bary = np.vstack((np.eye(4), [[0.5, 0.5, 0, 0]], [[0.1, 0.2, 0.3, 0.4]]))
    values, first, second = tetra_polynomials(degree, bary)
    direct = tetra_values_gradients(degree, bary)
    assert np.array_equal(values, direct[0])
    assert np.array_equal(first, direct[1])
    gradient = first[..., 1:] - first[..., :1]
    hessian = second[..., 1:, 1:] - second[..., 1:, :1] - second[..., :1, 1:] + second[..., :1, :1]
    bound0, bound1, bound2 = reference_roundoff_bounds(
        "tetrahedron", degree, bary, indices / degree
    )
    accumulation = gamma(len(indices))
    assert np.all(
        abs(values.sum(axis=1) - 1) <= bound0.sum(axis=1) + accumulation * abs(values).sum(axis=1)
    )
    assert np.all(
        abs(gradient.sum(axis=1)) <= bound1.sum(axis=1) + accumulation * abs(gradient).sum(axis=1)
    )
    assert np.all(
        abs(hessian.sum(axis=1)) <= bound2.sum(axis=1) + accumulation * abs(hessian).sum(axis=1)
    )
    assert np.array_equal(second, second.swapaxes(-1, -2))
    empty = tetra_polynomials(degree, np.empty((0, 4)))
    assert empty[0].shape == (0, len(indices))


@pytest.mark.parametrize("degree", [5, 6])
def test_physical_polynomial_derivatives_and_quadrature(degree):
    """Verify full physical Hessians on a translated, sheared tetrahedron and exact mass degree."""
    mesh = TetraMesh(
        [[0.1, 0.2, -0.1], [1.2, 0.3, 0.0], [0.3, 1.1, 0.2], [-0.1, 0.4, 1.0]],
        [[0, 1, 2, 3]],
    )
    bary, weights = tetrahedron_quadrature(degree + 2)
    dofs, nodes, values, gradient, hessian = tetra_element_tabulate(mesh, degree, bary)
    direction = np.array([0.2, -0.3, 0.4])
    nodal = (0.7 + nodes @ direction) ** degree
    physical = bary @ mesh.points[mesh.cells[0]]
    scalar = 0.7 + physical @ direction
    assert_allclose(nodal[dofs[0]] @ values.T, scalar**degree, atol=4e-14, rtol=4e-13)
    actual_gradient = np.einsum("i,qia->qa", nodal[dofs[0]], gradient[0])
    expected_gradient = degree * scalar[:, None] ** (degree - 1) * direction
    assert_allclose(actual_gradient, expected_gradient, atol=1e-12, rtol=3e-12)
    actual_hessian = np.einsum("i,qiab->qab", nodal[dofs[0]], hessian[0])
    expected_hessian = (
        degree
        * (degree - 1)
        * scalar[:, None, None] ** (degree - 2)
        * np.outer(direction, direction)
    )
    assert_allclose(actual_hessian, expected_hessian, atol=2e-11, rtol=2e-11)
    expected_integral = 6 * factorial(degree) ** 2 / factorial(2 * degree + 3)
    assert_allclose(weights @ (bary[:, 0] * bary[:, 1]) ** degree, expected_integral, rtol=2e-14)
    A, M, load = tetra_operators(mesh, degree, source=2.0, order=1)
    assert np.linalg.eigvalsh(M.toarray())[0] > 0
    eigenvalues = np.linalg.eigvalsh(A.toarray())
    assert np.count_nonzero(eigenvalues > 1e-11 * eigenvalues[-1]) == len(nodes) - 1
    assert_allclose(A @ np.ones(len(nodes)), 0, atol=2e-13)
    assert_allclose(2 * M @ np.ones(len(nodes)), load, atol=2e-15)
    assert_allclose(M.sum(), mesh.volumes.sum(), atol=2e-15)
    diameter = np.max(np.linalg.norm(mesh.points[:, None] - mesh.points[None, :], axis=-1))
    inverse_bound = laplacian_inverse_bound(gradient, hessian, weights, np.array([diameter]))
    coefficients = np.random.default_rng(degree).normal(size=(11, len(nodes)))
    gradients = np.einsum("si,qia->sqa", coefficients, gradient[0])
    laplacians = coefficients @ np.trace(hessian[0], axis1=-2, axis2=-1).T
    left = inverse_bound[0] * diameter**2 * ((laplacians**2) @ weights)
    right = np.einsum("q,sqa,sqa->s", weights, gradients, gradients)
    assert np.all(left <= right * (1 + 2e-13))


def test_shared_face_nodes_are_permutation_invariant():
    """Node ownership depends on exact integer weights, including face/interior nodes beyond P4."""
    mesh = TetraMesh.unit_cube()
    permuted = TetraMesh(mesh.points, mesh.cells[:, [2, 3, 0, 1]])
    dofs, nodes = tetra_nodal_space(mesh, 6)
    other_dofs, other_nodes = tetra_nodal_space(permuted, 6)
    distances, ordering = cKDTree(other_nodes).query(nodes)
    assert distances.max() < 3e-16
    assert len(nodes) == 7**3
    for first, second in zip(dofs, other_dofs, strict=True):
        assert set(ordering[first]) == set(second)


@pytest.mark.parametrize("degree", [5, 6])
def test_p5_p2_admissibility_boundary_with_nonhomogeneous_patch(degree):
    """The complete 3D estimator accepts k=ell+3 and its next local degree on an exact patch."""
    mesh = TetraMesh.unit_cube()
    tensor = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 0.8]])

    def exact(x):
        """Quadratic potential with a nonzero boundary trace."""
        return 1 + x[:, 0] + x[:, 1] ** 2 + 0.5 * x[:, 2] ** 2

    def flux(x):
        """Physical anisotropic flux, derived independently of finite element derivatives."""
        return -np.column_stack((np.ones(len(x)), 2 * x[:, 1], x[:, 2])) @ tensor.T

    solution = solve_darcy_3d(
        mesh,
        degree=degree,
        local_refinement=1,
        skeleton=TriangularSkeleton(mesh, degree=2),
        permeability=tensor,
        source=-2 * tensor[1, 1] - tensor[2, 2],
        dirichlet=exact,
        quadrature_order=degree + 2,
    )
    estimate = estimate_darcy_error_3d(
        solution, degree=2, dirichlet=exact, quadrature_order=degree + 2
    )
    assert solution.l2_error(exact, degree + 2) < 2e-12
    assert estimate.reconstruction.flux_l2_error(flux, degree + 2) < 2e-11
    assert estimate.total < 5e-11
    assert max(estimate.equilibrium_defect) < 2e-12


@pytest.mark.parametrize("degree", [0, -1, True, 2.5])
def test_degree_contract(degree):
    """General degree support does not accept nonpositive, Boolean or nonintegral degrees."""
    for operation in (tetra_polynomials, tetra_values_gradients):
        with pytest.raises(ValueError, match="degree"):
            operation(degree, np.array([[0.25] * 4]))


@pytest.mark.parametrize("bary", [np.ones((3, 3)), [[1j, 0, 0, 0]], [[np.inf, 0, 0, 0]]])
def test_point_contract(bary):
    """Higher-order tabulation keeps the real, finite, four-coordinate input contract."""
    for operation in (tetra_polynomials, tetra_values_gradients):
        with pytest.raises(ValueError):
            operation(5, bary)
