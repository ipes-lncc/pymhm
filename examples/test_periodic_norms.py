"""Independent positive-quadrature verification of the tensor-moment comparison."""

import numpy as np
import pytest

from examples.periodic_norms import difference, field_norms, quadrature_difference
from pymhm._legacy.models.darcy.conforming import ConformingQuadrilateralSolution
from pymhm.fem.scalar.quadrilateral import qk_space, quadrilateral_quadrature
from pymhm.meshes.cartesian import CartesianMacroMesh


def field(n: int, degree: int, seed: int, bounds=(0.0, 1.0, 0.0, 1.0)):
    """Use independent continuous nodal coefficients to exercise every tensor basis mode."""
    mesh = CartesianMacroMesh(n, bounds=bounds)
    return ConformingQuadrilateralSolution(
        mesh, degree, np.random.default_rng(seed).normal(size=(n * degree + 1) ** 2), 1.0, 0.0
    )


@pytest.mark.parametrize("degrees", [(1, 1), (3, 1), (1, 5), (5, 3), (3, 5)])
@pytest.mark.parametrize("reverse", [False, True])
def test_tensor_moments_match_independent_pointwise_quadrature(degrees, reverse):
    """Compare full H1/L2 norms in both directions of the nested reference hierarchy."""
    first, second = field(2, degrees[0], 17), field(8, degrees[1], 19)
    if reverse:
        first, second = second, first
    actual = difference(first, (second,))
    expected = quadrature_difference(first, (second,))
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-12, atol=1e-13)


def test_broken_macrocells_are_integrated_without_interface_averaging():
    """Preserve independent values in every subrectangle of a global reference."""
    reference = field(4, 3, 23)
    local = tuple(
        field(4, 1, 29 + i, bounds)
        for i, bounds in enumerate(
            ((0, 0.5, 0, 0.5), (0.5, 1, 0, 0.5), (0, 0.5, 0.5, 1), (0.5, 1, 0.5, 1))
        )
    )
    actual = difference(reference, local)
    expected = quadrature_difference(reference, local)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12)


def test_near_identical_fields_use_positive_difference_quadrature():
    """Avoid an artificial energy-subtraction floor in exactly represented fields."""
    first, second = field(2, 2, 1), field(8, 3, 2)
    for item in (first, second):
        nodes = qk_space(item.mesh, item.degree)[1]
        item.pressure[:] = 1 + nodes[:, 0] + nodes[:, 0] * nodes[:, 1] + nodes[:, 1] ** 2
    result = difference(first, (second,))
    assert result["l2"] < 1e-12
    assert result["h1_seminorm"] < 1e-11
    # Integral of grad(p)^2 = (1+y)^2 + (x+2y)^2 = 5 on the unit square.
    np.testing.assert_allclose(field_norms(first)[1], 5, rtol=2e-14)


def test_non_nested_partitions_are_rejected():
    """Do not silently compare the wrong coarse/fine integration overlay."""
    with pytest.raises(ValueError, match="align"):
        difference(field(3, 1, 1), (field(8, 1, 2),))


def test_gradient_products_preserve_nearly_constant_high_order_fields():
    """Do not lose small physical gradients to constant-mode energy cancellation."""
    first, second = field(4, 5, 71), field(8, 5, 73)
    for item in (first, second):
        item.pressure[:] = 0.001 + 1e-9 * item.pressure
    points, weights = quadrilateral_quadrature(6)
    for item in (first, second):
        physical = item.mesh.points[item.mesh.cells[:, 0], None] + points[None] * item.mesh.spacing
        gradient = item.evaluate(physical.reshape(-1, 2))[1]
        integral = np.prod(item.mesh.spacing) * np.sum(
            np.sum(gradient**2, axis=1).reshape(-1, len(weights)) * weights
        )
        np.testing.assert_allclose(field_norms(item)[1], integral, rtol=2e-7, atol=1e-25)
    actual = difference(first, (second,))
    expected = quadrature_difference(first, (second,))
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-7, atol=1e-16)


def test_matching_basis_subtracts_coefficients_before_integrating():
    """Use the exact common basis for very small differences without subtracting energies."""
    first, second = field(3, 5, 97), field(3, 5, 97)
    second.pressure[:] += 1e-9
    result = difference(first, (second,))
    np.testing.assert_allclose(result["l2"], 1e-9, rtol=1e-7)
    assert result["h1_seminorm"] < 2e-14


def test_field_domains_must_align_and_reference_must_be_nonzero():
    """Do not round a shifted rectangle onto another integration partition."""
    with pytest.raises(ValueError, match="cell-aligned"):
        difference(field(8, 1, 1), (field(4, 1, 2, (0.1, 0.6, 0.1, 0.6)),))
    reference = field(4, 2, 1)
    reference.pressure[:] = 0
    with pytest.raises(ValueError, match="nonzero reference"):
        difference(reference, (field(4, 2, 2),))
