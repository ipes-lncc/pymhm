"""Geometric material integration, polynomial moments and flow invariance."""

from math import factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_brinkman
from pymhm.cut_cells import cartesian_triangle_quadrature
from pymhm.elements import triangle_quadrature
from pymhm.flow import _flow_local
from pymhm.reservoir import CartesianCellField


def triangle():
    """Return the unit right triangle with a positively oriented topology."""
    return TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])


@pytest.mark.parametrize("order", [3, 5, 8])
def test_cut_quadrature_integrates_exact_monomial_moments(order):
    """Integrate x^a y^b using an independent simplex factorial identity."""
    mesh = triangle()
    field = CartesianCellField(np.arange(35).reshape(5, 7), (0.2, 1 / 7))
    bary, weights, cells = cartesian_triangle_quadrature(mesh, field, order, return_cells=True)
    points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    assert np.all(weights >= 0)
    assert_allclose(weights.sum(axis=1), 1, atol=3e-16)
    assert_allclose(bary.sum(axis=-1), 1, atol=3e-16)
    assert np.min(bary) > -2e-15
    assert_array_equal(field(points.reshape(-1, 2)), field.values[tuple(cells.reshape(-1, 2).T)])
    for a in range(2 * order - 1):
        for b in range(2 * order - 1 - a):
            actual = mesh.areas @ (weights * points[..., 0] ** a * points[..., 1] ** b).sum(axis=1)
            expected = factorial(a) * factorial(b) / factorial(a + b + 2)
            assert_allclose(actual, expected, rtol=2e-13, atol=1e-17)


def pixel_triangle_area(i, j, nx, ny):
    """Integrate a clipped vertical segment analytically, without polygon clipping."""
    left, right, bottom, top = i / nx, (i + 1) / nx, j / ny, (j + 1) / ny
    breaks = sorted({left, right, *(v for v in (1 - top, 1 - bottom) if left < v < right)})
    total = 0.0
    for a, b in zip(breaks[:-1], breaks[1:], strict=True):
        middle = (a + b) / 2
        if 1 - middle >= top:
            total += (top - bottom) * (b - a)
        elif 1 - middle > bottom:
            total += (1 - bottom) * (b - a) - (b * b - a * a) / 2
    return total


def test_each_positive_area_pixel_is_present_and_touching_pixels_are_absent():
    """Recover every material area independently, including exact face/vertex contacts."""
    mesh = triangle()
    field = CartesianCellField(np.arange(6).reshape(3, 2), (1 / 3, 0.5))
    _, weights, indices = cartesian_triangle_quadrature(mesh, field, 3, return_cells=True)
    areas = np.zeros((3, 2))
    np.add.at(areas, tuple(indices[0].T), weights[0] * mesh.areas[0])
    expected = np.array([[pixel_triangle_area(i, j, 3, 2) for j in range(2)] for i in range(3)])
    assert_allclose(areas, expected, atol=1e-16)
    present = set(map(tuple, indices[0, weights[0] > 0]))
    assert present == set(map(tuple, np.argwhere(expected > 0)))
    corner = CartesianCellField(np.ones((2, 2)), (0.5, 0.5))
    _, w, cell = cartesian_triangle_quadrature(mesh, corner, return_cells=True)
    assert not np.any(np.all(cell[0, w[0] > 0] == [1, 1], axis=1))


def test_physical_affine_geometry_and_zero_padding_preserve_moments():
    """Resolve unequal intersection counts without adding fictitious material or area."""
    reference = TriangleMesh.unit_square(2)
    mesh = TriangleMesh(reference.points * [2.3, 1.7] + [-3.0, 5.0], reference.cells)
    field = CartesianCellField(np.ones((3, 7, 2, 2)), (2.3 / 3, 1.7 / 7), (-3.0, 5.0))
    bary, weights, pixels = cartesian_triangle_quadrature(mesh, field, 5, return_cells=True)
    assert np.any(weights == 0)
    for cell in range(len(mesh.cells)):
        pad = weights[cell] == 0
        assert_allclose(bary[cell, pad], np.tile(bary[cell, 0], (sum(pad), 1)), atol=0)
        assert_array_equal(pixels[cell, pad], np.tile(pixels[cell, 0], (sum(pad), 1)))
    points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    normal_bary, normal_weights = triangle_quadrature(5)
    normal_points = np.einsum("qi,tij->tqj", normal_bary, mesh.points[mesh.cells])
    for a, b in ((0, 0), (1, 2), (3, 3), (0, 7)):
        actual = mesh.areas @ np.sum(weights * points[..., 0] ** a * points[..., 1] ** b, axis=1)
        expected = mesh.areas @ (
            (normal_points[..., 0] ** a * normal_points[..., 1] ** b) @ normal_weights
        )
        assert_allclose(actual, expected, rtol=2e-14)
    plain_bary, plain_weights = cartesian_triangle_quadrature(mesh, field, 5)
    assert_array_equal(plain_bary, bary)
    assert_array_equal(plain_weights, weights)


@pytest.mark.parametrize("field", [1.0, CartesianCellField(np.ones((2, 2, 2)), (1, 1, 1))])
def test_invalid_material_geometry_is_rejected(field):
    """Reject non-Cartesian and spatially three-dimensional material fields."""
    with pytest.raises(ValueError, match="planar"):
        cartesian_triangle_quadrature(triangle(), field)


def test_invalid_order_outside_domain_and_zero_intersection_are_rejected():
    """Distinguish invalid quadrature, unavailable material and collapsed roundoff geometry."""
    field = CartesianCellField(np.ones((1, 1)), (1, 1))
    with pytest.raises(ValueError, match="quadrature order"):
        cartesian_triangle_quadrature(triangle(), field, 0)
    with pytest.raises(ValueError, match="outside"):
        cartesian_triangle_quadrature(triangle(), CartesianCellField(np.ones((1, 1)), (0.1, 0.1)))
    thin = TriangleMesh([[-1e-15, 0], [0, 0], [-1e-15, 1]], [[0, 1, 2]])
    with pytest.raises(ValueError, match="positive-area"):
        cartesian_triangle_quadrature(thin, field)


@pytest.mark.parametrize(
    "method,degree",
    [("usfem", 1), ("usfem", 2), ("usfem", 3), ("taylor-hood", 2), ("taylor-hood", 3)],
)
def test_cut_local_forms_are_invariant_under_exact_polynomial_quadrature(method, degree):
    """Check the complete matrix, force and physical moments independently of order."""
    mesh = triangle()
    tensor = np.zeros((3, 4, 2, 2))
    tensor[..., 0, 0] = np.arange(12).reshape(3, 4) + 1
    tensor[..., 1, 1] = 0.2
    tensor[..., 0, 1] = tensor[..., 1, 0] = 0.1
    field = CartesianCellField(tensor, (1 / 3, 0.25))
    skeleton = SkeletonSpace(mesh, components=2)
    data = dict(
        mesh=mesh,
        skeleton=skeleton,
        degree=degree,
        formulation=method,
        refinement=1,
        viscosity=0.3,
        drag=field,
        beta=np.zeros(2),
        source=(1.0, 2.0),
    )
    first, second = (_flow_local(0, order=q, **data) for q in (5, 8))
    assert_allclose(first.problem.matrix.toarray(), second.problem.matrix.toarray(), atol=3e-13)
    assert_allclose(first.problem.load, second.problem.load, atol=3e-14)
    for index in range(2, 7):
        assert_allclose(first.metadata[index], second.metadata[index], atol=3e-14)


@pytest.mark.parametrize(
    "method,degree,rule",
    [
        ("usfem", 3, "tensor-2025"),
        ("usfem", 3, "pointwise-2017"),
        ("taylor-hood", 2, "tensor-2025"),
    ],
)
def test_piecewise_material_flow_fields_are_quadrature_invariant(method, degree, rule):
    """Compare nontrivial velocity and pressure with exact integration of all jumps."""
    mesh = TriangleMesh.unit_square()
    field = CartesianCellField(
        10.0 ** np.random.default_rng(13).uniform(-1, 2, (3, 5)), (1 / 3, 0.2)
    )
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace(degrees=(1,)) for _ in mesh.faces), 2)
    solutions = [
        solve_brinkman(
            mesh,
            viscosity=0.3,
            drag=field,
            source=(1.0, 2.0),
            skeleton=skeleton,
            degree=degree,
            formulation=method,
            stabilization=rule,
            local_refinement=2,
            quadrature_order=order,
        )
        for order in (5, 8)
    ]
    for attribute in ("values", "pressure"):
        assert_allclose(
            np.asarray(getattr(solutions[0], attribute)),
            np.asarray(getattr(solutions[1], attribute)),
            rtol=2e-10,
            atol=3e-13,
        )


@pytest.mark.parametrize("minimum", [None, 1.0, "pointwise"])
def test_p1_usfem_matrix_matches_independent_exact_pixel_moments(minimum):
    """Derive all stabilized blocks from closed-form affine triangle moments."""
    mesh = triangle()
    field = CartesianCellField(np.array([[1.0, 7.0], [3.0, 1000.0]]), (0.5, 0.5))
    subtriangles = [
        ([[0, 0], [0.5, 0], [0.5, 0.5]], 1.0),
        ([[0, 0], [0.5, 0.5], [0, 0.5]], 1.0),
        ([[0.5, 0], [1, 0], [0.5, 0.5]], 3.0),
        ([[0, 0.5], [0.5, 0.5], [0, 1]], 7.0),
    ]
    weighted_mass, squared_mass = np.zeros((3, 3)), np.zeros((3, 3))
    weighted_moments = np.zeros(3)
    viscosity, area, diameter_squared = 0.3, 0.5, 2.0
    tau_integral = 0.0
    for vertices, gamma in subtriangles:
        bound = gamma if minimum == "pointwise" else 7.0 if minimum is None else minimum
        tau = diameter_squared / (max(bound * diameter_squared, 12 * viscosity) + 12 * viscosity)
        vertices = np.asarray(vertices)
        values = np.column_stack((1 - vertices.sum(axis=1), vertices))
        affine_mass = 0.125 / 12 * (np.ones((3, 3)) + np.eye(3))
        mass = values.T @ affine_mass @ values
        weighted_mass += gamma * mass
        squared_mass += tau * gamma**2 * mass
        weighted_moments += tau * gamma * 0.125 / 3 * values.sum(axis=0)
        tau_integral += tau * 0.125
    gradient = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    moments = np.full(3, area / 3)
    matrix = np.zeros((9, 9))
    matrix[:6, :6] = np.kron(
        viscosity * area * (gradient @ gradient.T) + weighted_mass - squared_mass,
        np.eye(2),
    )
    cross = -gradient.ravel()[:, None] * moments[None]
    cross -= np.einsum("i,ja->iaj", weighted_moments, gradient).reshape(6, 3)
    matrix[:6, 6:] = cross
    matrix[6:, :6] = cross.T
    matrix[6:, 6:] = -tau_integral * gradient @ gradient.T
    force = np.array([1.0, 2.0])
    load = np.r_[
        ((moments - weighted_moments)[:, None] * force).ravel(),
        -tau_integral * gradient @ force,
    ]
    result = _flow_local(
        0,
        mesh=mesh,
        skeleton=SkeletonSpace(mesh, components=2),
        degree=1,
        formulation="usfem",
        refinement=1,
        viscosity=viscosity,
        drag=field,
        beta=np.zeros(2),
        source=force,
        order=5,
        gamma_min=None if minimum == "pointwise" else minimum,
        pointwise=minimum == "pointwise",
    )
    node_order = np.argmin(
        np.linalg.norm(result.metadata[0].points[:, None] - mesh.points[None], axis=2), axis=1
    )
    dof_order = np.r_[(2 * node_order[:, None] + np.arange(2)).ravel(), 6 + node_order]
    assert_allclose(
        result.problem.matrix.toarray(),
        matrix[np.ix_(dof_order, dof_order)],
        rtol=2e-14,
        atol=3e-16,
    )
    assert_allclose(result.problem.load, load[dof_order], rtol=2e-14, atol=3e-16)
