"""Independent quadrature, cancellation and physical scalar energy contracts."""

from decimal import Decimal, localcontext
from math import fsum

import numpy as np
import pytest

from pymhm.fem.scalar import _integration
from pymhm.fem.scalar.operators import _boundary_moments, _scalar_diffusion_blocks, p1_operators
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.triangle import nodal_space, scalar_operators
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("value", [0.0, -(2.0**180), 2.0**-180])
def test_ordinary_product_range_includes_its_declared_boundary(value):
    """The conservative range includes zeros and its exact two-sided endpoints."""
    for function in (
        _integration.ordinary_product_range,
        _integration.ordinary_product_range.py_func,
    ):
        assert function(np.array([1.0, value]))


@pytest.mark.parametrize("value", [np.nextafter(2.0**180, np.inf), np.nextafter(2.0**-180, 0.0)])
def test_ordinary_product_range_delegates_immediately_excluded_exponents(value):
    """A represented neighbor outside either endpoint requests wider accumulation."""
    for function in (
        _integration.ordinary_product_range,
        _integration.ordinary_product_range.py_func,
    ):
        assert not function(np.array([value, 1.0]))


@pytest.mark.parametrize(
    "dimension,spacing,coefficient",
    [(2, 1e-155, 1.0), (2, 1e50, 1e-320), (2, 1e-100, 1e308), (3, 1e-75, 1e308)],
)
def test_extreme_valid_simplices_retain_finite_physical_stiffness(dimension, spacing, coefficient):
    """Valid 2D/3D geometry and subnormal materials retain their finite operator."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("this native-range comparison requires a wider real accumulator")
    vertices = np.vstack((np.zeros(dimension), spacing * np.eye(dimension)))
    connectivity = np.arange(dimension + 1)[None]
    mesh = (
        TriangleMesh(vertices, connectivity)
        if dimension == 2
        else TetraMesh(vertices, connectivity)
    )
    measure = float(mesh.areas[0]) if dimension == 2 else float(mesh.volumes[0])
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        actual = (
            p1_operators(mesh, diffusion=coefficient)[0]
            if dimension == 2
            else tetra_operators(mesh, degree=1, diffusion=coefficient, order=4)[0]
        ).toarray()
    with localcontext() as context:
        context.prec = 100
        scale = (
            Decimal.from_float(measure)
            * Decimal.from_float(float(1 / spacing)) ** 2
            * Decimal.from_float(coefficient)
        )
        ideal = np.eye(dimension + 1, dtype=int)
        ideal[0, 0] = dimension
        ideal[0, 1:] = ideal[1:, 0] = -1
        expected = np.array([[float(scale * int(entry)) for entry in row] for row in ideal])
    assert np.isfinite(actual).all()
    # Basix represents nominally zero P1 Cartesian derivative entries with
    # roundoff. Bound that executed-basis error in physical matrix units.
    np.testing.assert_allclose(
        actual, expected, rtol=3e-15, atol=4 * np.finfo(float).eps * np.max(abs(expected))
    )
    gradients = np.vstack((-np.ones(dimension), np.eye(dimension)))[None] / spacing
    represented = _scalar_diffusion_blocks(
        np.ones(1), gradients, (coefficient * np.eye(dimension))[None], measure
    )
    np.testing.assert_allclose(represented, expected, rtol=3e-15, atol=0)


def test_portable_exceptional_diffusion_retains_compensated_quadrature(monkeypatch):
    """Conservative range delegation keeps finite binary64 products and q cancellation."""
    monkeypatch.setattr(np, "longdouble", np.float64)
    weights = np.array([1e16, 1.0, -1e16])
    actual = _scalar_diffusion_blocks(
        weights, np.full((1, 1, 1), 1e100), np.ones((1, 1, 1)), 1e-200
    )
    np.testing.assert_array_equal(actual, [[1.0]])


def test_portable_exceptional_diffusion_rejects_unrepresentable_intermediates(monkeypatch):
    """A platform without wider products reports its range limit before publishing NaNs."""
    monkeypatch.setattr(np, "longdouble", np.float64)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="native real range"),
    ):
        _scalar_diffusion_blocks(np.ones(1), np.full((1, 1, 1), 1e155), np.ones((1, 1, 1)), 1e-310)


def test_exceptional_boundary_moments_retain_range_until_the_physical_measure():
    """A finite face load may require a wider moment before multiplication by its length."""
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("this native-range comparison requires a wider real accumulator")
    moments = _boundary_moments(np.full((1, 1), 1e200), np.ones(1), np.full((1, 1), 1e200))
    assert moments.dtype == np.dtype(np.longdouble)
    assert np.isfinite(moments).all()
    load = np.asarray(moments * np.longdouble(1e-200), dtype=float)
    np.testing.assert_allclose(load, [[1e200]], rtol=3e-15, atol=0)


def test_portable_exceptional_boundary_moments_preserve_the_fsum_contract(monkeypatch):
    """The portable rare path retains finite tiny terms through compensated summation."""
    monkeypatch.setattr(np, "longdouble", np.float64)
    actual = _boundary_moments(
        np.ones((3, 1)), np.ones(3), np.array([[1e-100], [1e-200], [-1e-100]])
    )
    np.testing.assert_array_equal(actual, [[1e-200]])


def test_portable_exceptional_boundary_moments_reject_unrepresentable_products(monkeypatch):
    """A platform lacking wider moment products reports unsupported range explicitly."""
    monkeypatch.setattr(np, "longdouble", np.float64)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="native real range"),
    ):
        _boundary_moments(np.full((1, 1), 1e200), np.ones(1), np.full((1, 1), 1e200))


@pytest.mark.parametrize(
    "total,correction,value,expected",
    [(1e16, 0.0, 1.0, 1e16), (1.0, 0.0, 1e16, 1e16), (1e16, 1.0, -1e16, 1.0)],
)
def test_compensated_add_retains_represented_terms(total, correction, value, expected):
    """Both magnitude branches retain a small term without an entry cutoff."""
    for function in (_integration._compensated_add, _integration._compensated_add.py_func):
        actual, carry = function(total, correction, value)
        assert actual + carry == expected


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("contrast", [1.0, 1e8])
def test_diffusion_matches_independently_accumulated_quadrature(dimension, contrast):
    """Anisotropic real operators preserve independent q/a/b contraction values."""
    rng = np.random.default_rng(1709 + dimension)
    gradients = rng.normal(size=(2, 5, 4, dimension))
    factors = rng.normal(size=(2, 5, dimension, dimension))
    factors[..., 0] *= np.sqrt(contrast)
    tensors = factors @ factors.swapaxes(-1, -2) + np.eye(dimension)
    weights = rng.uniform(0.1, 1.0, size=(2, 5))
    measures = np.array([0.25, 0.125])
    expected = np.array(
        [
            [
                [
                    fsum(
                        float(
                            gradients[c, q, i, a]
                            * tensors[c, q, a, b]
                            * gradients[c, q, j, b]
                            * weights[c, q]
                            * measures[c]
                        )
                        for q in range(5)
                        for a in range(dimension)
                        for b in range(dimension)
                    )
                    for j in range(4)
                ]
                for i in range(4)
            ]
            for c in range(2)
        ]
    )
    for function in (_integration.diffusion_blocks, _integration.diffusion_blocks.py_func):
        actual = function(weights, gradients, tensors, measures)
        scale = np.max(abs(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=12 * np.finfo(float).eps * scale)
        np.testing.assert_allclose(
            actual, actual.swapaxes(-1, -2), rtol=0, atol=8 * np.finfo(float).eps * scale
        )
    np.testing.assert_array_equal(
        _scalar_diffusion_blocks(weights, gradients, tensors, measures),
        _integration.diffusion_blocks(weights, gradients, tensors, measures),
    )


@pytest.mark.parametrize("cancellation_axis", ["quadrature", "tensor"])
def test_diffusion_retains_small_terms_inside_every_reduction(cancellation_axis):
    """A true unit term survives both q and Cartesian contraction cancellation."""
    if cancellation_axis == "quadrature":
        weights = np.array([[1e16, 1.0, -1e16]])
        gradients = np.ones((1, 3, 1, 1))
        tensors = np.ones((1, 3, 1, 1))
    else:
        weights = np.ones((1, 1))
        gradients = np.ones((1, 1, 1, 3))
        tensors = np.diag([1e16, 1.0, -1e16])[None, None]
    for function in (_integration.diffusion_blocks, _integration.diffusion_blocks.py_func):
        np.testing.assert_array_equal(function(weights, gradients, tensors, np.ones(1)), [[[1.0]]])


def test_scalar_leaf_broadcasts_cell_and_quadrature_axes_without_narrowing_the_domain():
    """Singleton q, several cell axes and reversed views retain their integration maps."""
    weights = np.array([0.25, 0.75])[::-1]
    gradients = np.array([[[1.0, -1.0], [-1.0, 1.0]]])
    tensors = np.tile(np.array([[3.0, 0.5], [0.5, 2.0]]), (2, 1, 2, 1, 1))
    measures = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    actual = _scalar_diffusion_blocks(weights, gradients, tensors, measures)
    expected = measures[..., None, None] * np.array([[4.0, -4.0], [-4.0, 4.0]])
    np.testing.assert_allclose(actual, expected, rtol=3e-15, atol=0)
    np.testing.assert_allclose(
        _scalar_diffusion_blocks(weights, gradients, tensors[0, 0], 1e-25),
        1e-25 * np.array([[4.0, -4.0], [-4.0, 4.0]]),
        rtol=3e-15,
        atol=0,
    )


def test_empty_cell_batches_have_the_declared_matrix_shape():
    """An empty broadcasted batch yields empty blocks without inventing a cell."""
    actual = _scalar_diffusion_blocks(
        np.empty((0, 2)), np.empty((0, 2, 3, 2)), np.empty((0, 2, 2, 2)), np.empty(0)
    )
    assert actual.shape == (0, 3, 3)


@pytest.mark.parametrize("scale", [1.0, 1e-25])
def test_boundary_moments_retain_cancelled_and_tiny_physical_data(scale):
    """Each component is accumulated independently without a moment truncation."""
    basis = np.column_stack((np.ones(3), np.array([1.0, 2.0, 3.0])))
    weights = np.ones(3)
    values = scale * np.column_stack((np.array([1e16, 1.0, -1e16]), np.array([2.0, 3.0, 4.0])))
    expected = np.array(
        [[fsum(basis[:, i] * weights * values[:, j]) for j in range(2)] for i in range(2)]
    )
    for function in (_integration.boundary_moments, _integration.boundary_moments.py_func):
        np.testing.assert_array_equal(function(basis, weights, values), expected)
    np.testing.assert_array_equal(_boundary_moments(basis, weights, values), expected)


@pytest.mark.parametrize("degree", [2, 3])
def test_assembled_anisotropic_energy_matches_exact_polynomial_integral(degree):
    """The scalar operator integrates an independently expanded quadratic energy."""
    mesh = TriangleMesh.unit_square(2)
    tensor = np.array([[3.0, 0.4], [0.4, 2.0]])
    matrix = scalar_operators(mesh, degree, diffusion=tensor, order=6)[0]
    nodes = nodal_space(mesh, degree)[1]
    x, y = nodes.T
    pressure = x * x + 3 * x * y + y * y
    # grad(u)=(2x+3y,3x+2y). Integrate x²,y² ->1/3 and xy ->1/4.
    expected = (12 + 4.8 + 18) / 3 + (36 + 10.4 + 24) / 4 + (27 + 4.8 + 8) / 3
    np.testing.assert_allclose(pressure @ matrix @ pressure, expected, rtol=3e-14, atol=0)
    np.testing.assert_allclose(matrix @ np.ones(len(nodes)), 0, rtol=0, atol=4e-13)
