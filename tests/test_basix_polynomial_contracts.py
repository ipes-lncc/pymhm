"""Native polynomial conventions, derivatives and explicit interpolation maps."""

import numpy as np
import pytest
from numpy.polynomial.legendre import legder, legval, legvander
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.fem.reference import (
    ReferenceElementSpec,
    barycentric_simplex_tabulation,
    create_reference_element,
    interpolate_reference,
    legendre_tabulation,
    legendre_values,
    monomial_tabulation,
    orthogonal_polynomial_tabulation,
    reference_interpolation_points,
    simplex_lagrange_basis,
    simplex_lagrange_tabulation,
    tabulate_reference,
    tensor_lagrange_basis,
    tensor_lagrange_tabulation,
)


@pytest.mark.parametrize("cell", ["interval", "triangle", "tetrahedron"])
def test_literal_simplex_interpolation_preserves_exact_zero_and_native_derivatives(cell):
    element = create_reference_element(ReferenceElementSpec("P", cell, 2))
    points = reference_interpolation_points(element)
    nodes = np.column_stack((1 - points.sum(axis=1), points))
    declared = simplex_lagrange_basis(cell, 2, nodes=nodes)
    values, first, _ = simplex_lagrange_tabulation(cell, 2, nodes, nodes=nodes)
    native = tabulate_reference(element, points, 1)[:, :, declared.permutation, 0]
    assert_array_equal(values, np.eye(len(nodes)))
    assert_array_equal(first, native[1:].transpose(1, 2, 0))
    nearby = nodes.copy()
    nearby[:, 1] = np.nextafter(nearby[:, 1], np.inf)
    nearby[:, 0] = 1 - nearby[:, 1:].sum(axis=1)
    actual = simplex_lagrange_tabulation(cell, 2, nearby, nodes=nodes, nderiv=0)[0]
    expected = tabulate_reference(element, nearby[:, 1:])[0, :, declared.permutation, 0].T
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("cell", ["interval", "quadrilateral", "hexahedron"])
def test_literal_tensor_interpolation_does_not_snap_nearby_coordinates(cell):
    element = create_reference_element(ReferenceElementSpec("P", cell, 2))
    nodes = reference_interpolation_points(element)
    declared = tensor_lagrange_basis(cell, 2, nodes=nodes)
    values, gradients = tensor_lagrange_tabulation(cell, 2, nodes, nodes=nodes)
    native = tabulate_reference(element, nodes, 1)[:, :, declared.permutation, 0]
    assert_array_equal(values, np.eye(len(nodes)))
    assert_array_equal(gradients, native[1:].transpose(1, 2, 0))
    nearby = nodes.copy()
    nearby[:, 0] = np.nextafter(nearby[:, 0], np.inf)
    actual = tensor_lagrange_tabulation(cell, 2, nearby, nodes=nodes, nderiv=0)[0]
    expected = tabulate_reference(element, nearby)[0, :, declared.permutation, 0].T
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(), (9,), (2, 3), (0,)])
@pytest.mark.parametrize("degree", [0, 1, 5, 9])
def test_conventional_legendre_values_derivatives_and_shape(shape, degree):
    count = int(np.prod(shape))
    points = np.linspace(-1.07, 1.13, count).reshape(shape)
    actual = legendre_tabulation(points, degree, 2)
    assert actual.shape == (3, *shape, degree + 1)
    assert_allclose(actual[0], legvander(points, degree).reshape(*shape, degree + 1), atol=8e-14)
    for derivative in (1, 2):
        for mode in range(degree + 1):
            expected = legval(points, legder(np.eye(degree + 1)[mode], derivative))
            assert_allclose(actual[derivative, ..., mode], expected, atol=3e-12)
    assert_array_equal(legendre_values(points, degree), actual[0])


@pytest.mark.parametrize("points", [[1j], [np.nan], ["bad"]])
def test_legendre_rejects_nonreal_or_nonfinite_coordinates(points):
    with pytest.raises(ValueError, match="finite real"):
        legendre_values(points, 2)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4])
@pytest.mark.parametrize("nderiv", [0, 1])
def test_explicit_monomial_order_and_cartesian_derivatives(dimension, nderiv):
    points = np.random.default_rng(71).uniform(-0.1, 1.1, (13, dimension))
    powers = tuple(tuple((i + axis) % 4 for axis in range(dimension)) for i in range(5))
    actual = monomial_tabulation(points, powers, nderiv=nderiv)
    expected = np.column_stack([np.prod(points**e, axis=1) for e in powers])
    assert_allclose(actual[0], expected, atol=2e-14)
    assert actual.shape == (1 + nderiv * dimension, len(points), len(powers))
    if nderiv:
        for axis in range(dimension):
            derivative = np.zeros_like(expected)
            for column, exponent in enumerate(powers):
                if exponent[axis]:
                    lower = list(exponent)
                    lower[axis] -= 1
                    derivative[:, column] = exponent[axis] * np.prod(points**lower, axis=1)
            assert_allclose(actual[axis + 1], derivative, atol=4e-14)
    assert monomial_tabulation(points, (), nderiv=nderiv).shape[-1] == 0


@pytest.mark.parametrize(
    "points,powers,nderiv,match",
    [
        ([1], ((0,),), 0, "shape"),
        (np.zeros((2, 0)), (), 0, "shape"),
        ([[0, np.inf]], ((0, 0),), 0, "finite"),
        ([[0, 1]], ((0,),), 0, "dimension"),
        ([[0]], ((-1,),), 0, "power"),
        ([[0]], ((True,),), 0, "power"),
        ([[0]], ((1,),), 2, "nderiv"),
    ],
)
def test_monomial_contract_rejects_invalid_data(points, powers, nderiv, match):
    with pytest.raises(ValueError, match=match):
        monomial_tabulation(points, powers, nderiv=nderiv)


def test_orthogonal_reference_measure_and_invalid_cell():
    assert_allclose(orthogonal_polynomial_tabulation("triangle", 0, [[0.2, 0.3]])[0], np.sqrt(2))
    assert_allclose(
        orthogonal_polynomial_tabulation("tetrahedron", 0, [[0.1, 0.2, 0.3]])[0], np.sqrt(6)
    )
    with pytest.raises(ValueError, match="cell"):
        orthogonal_polynomial_tabulation("unknown", 0, [[0.2]])


@pytest.mark.parametrize("nderiv", [0, 1, 2])
def test_interval_nodal_order_and_canonical_barycentric_extension(nderiv):
    parameters = np.array([0.0, 1.0, 0.5])
    nodes = np.column_stack((1 - parameters, parameters))
    value, gradient, hessian = barycentric_simplex_tabulation(
        "interval", 2, nodes, nodes=nodes, nderiv=nderiv
    )
    assert_allclose(value, np.eye(3), atol=2e-15)
    assert_array_equal(gradient[..., 0], 0)
    assert_array_equal(hessian[..., 0, :], 0)
    if nderiv:
        assert_allclose(gradient[..., 1].sum(axis=1), 0, atol=2e-15)
    if nderiv == 2:
        assert_allclose(hessian[..., 1, 1], np.tile([4, 4, -8], (3, 1)), atol=3e-15)


@pytest.mark.parametrize(
    "spec",
    [
        ReferenceElementSpec("RT", "triangle", 2, lagrange_variant="legendre"),
        ReferenceElementSpec("BDM", "tetrahedron", 2, lagrange_variant="legendre"),
        ReferenceElementSpec(
            "P", "triangle", 2, lagrange_variant="equispaced", dof_ordering=(1, 2, 3, 4, 5, 0)
        ),
    ],
)
def test_native_interpolation_duality_in_actual_executed_order(spec):
    element = create_reference_element(spec)
    points = reference_interpolation_points(element)
    assert not points.flags.writeable
    values = tabulate_reference(element, points)[0]
    assert_allclose(interpolate_reference(element, values), np.eye(element.dimension), atol=3e-14)
    for bad in (
        values.astype(complex),
        values[:, :, :0],
        values[..., 0],
        np.full_like(values, np.nan),
    ):
        with pytest.raises(ValueError, match="values"):
            interpolate_reference(element, bad)


def test_interpolation_requires_element_and_value_functionals():
    with pytest.raises(TypeError, match="element"):
        reference_interpolation_points(None)
    with pytest.raises(TypeError, match="element"):
        interpolate_reference(None, np.zeros((1, 1, 1)))
    hermite = create_reference_element(ReferenceElementSpec("Hermite", "interval", 3))
    with pytest.raises(ValueError, match="derivative"):
        interpolate_reference(hermite, np.zeros((2, 1, 1)))


@pytest.mark.parametrize(
    "cell,dimension", [("interval", 1), ("quadrilateral", 2), ("hexahedron", 3)]
)
@pytest.mark.parametrize("nderiv", [0, 1])
def test_tensor_nodal_order_and_polynomial_reproduction(cell, dimension, nderiv):
    import itertools

    nodes = np.array(list(itertools.product([0.0, 0.5, 1.0], repeat=dimension)))[::-1]
    basis = tensor_lagrange_basis(cell, 2, nodes=nodes)
    values, gradients = tensor_lagrange_tabulation(cell, 2, nodes, nodes=nodes, nderiv=nderiv)
    assert_allclose(values, np.eye(len(nodes)), atol=8e-15)
    assert not basis.basis_matrix.flags.writeable
    if nderiv:
        for axis in range(dimension):
            assert_allclose(gradients[..., axis] @ nodes[:, axis], 1, atol=5e-14)
    else:
        assert gradients.shape[-1] == 0


@pytest.mark.parametrize(
    "cell,nodes,nderiv,match",
    [
        ("triangle", [[0, 0]], 0, "cell"),
        ("quadrilateral", [[0, 0]], 0, "complete"),
        ("interval", [[0], [0], [1]], 0, "uniquely"),
        ("interval", [[0], [0.5], [1]], 2, "nderiv"),
    ],
)
def test_tensor_nodal_contract_rejects_changed_lattice(cell, nodes, nderiv, match):
    with pytest.raises(ValueError, match=match):
        tensor_lagrange_tabulation(cell, 2, [[0.1]], nodes=nodes, nderiv=nderiv)
