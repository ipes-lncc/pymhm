"""Independent elementwise verification of exact separated Cartesian assembly."""

from typing import Any

import numpy as np
import pytest

from pymhm._legacy.models.darcy.conforming import solve_conforming_quadrilateral
from pymhm._legacy.models.darcy.separable import (
    SeparableField,
    separable_diffusion_operators,
    solve_separable_diffusion,
)
from pymhm.fem.scalar.quadrilateral import qk_space, quadrilateral_operators
from pymhm.meshes.cartesian import CartesianMacroMesh


def material() -> SeparableField:
    """Return a positive nonconstant coefficient with two independent tensor terms."""
    return SeparableField(((2.0, 1.0), (lambda x: 1.5 * x**2, lambda y: 1 + y)))


def forcing() -> SeparableField:
    """Return a signed, nonpolynomial source represented without fitting."""
    return SeparableField(((np.sin, np.cos), (lambda x: -0.25 * x, lambda y: y)))


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("variable", [False, True])
def test_separated_operators_equal_elementwise_gauss_assembly(degree: int, variable: bool) -> None:
    """Match every matrix/load entry on translated anisotropic cells and several degrees."""
    mesh = CartesianMacroMesh(2, 3, bounds=(-0.25, 0.75, 0.1, 1.4))
    coefficient = material() if variable else SeparableField(((2.5, 1.0),))
    load = forcing()
    actual = separable_diffusion_operators(
        mesh, degree, permeability=coefficient, source=load, order=7
    )
    expected = quadrilateral_operators(mesh, degree, permeability=coefficient, source=load, order=7)
    for first, second in zip(actual, expected, strict=True):
        if hasattr(first, "toarray"):
            first, second = first.toarray(), second.toarray()
        np.testing.assert_allclose(first, second, rtol=8e-12, atol=5e-13)


@pytest.mark.parametrize("degree", [2, 3, 5])
def test_full_inhomogeneous_dirichlet_and_source_patch(degree: int) -> None:
    """Check the complete physical equation against an exact field and general assembly."""
    mesh = CartesianMacroMesh(2, 3, bounds=(-0.25, 0.75, 0.1, 1.4))
    coefficient = SeparableField(((2, 1), (lambda x: x, 1), (0.5, lambda y: y)))
    load = SeparableField(((-5, 1), (lambda x: -2.5 * x, 1), (-3, lambda y: y)))

    def exact(points: np.ndarray) -> np.ndarray:
        """Use a quadratic pressure with nonzero values on every boundary side."""
        x, y = points.T
        return 1 + x + x * y + y**2

    options = dict(
        degree=degree, permeability=coefficient, source=load, dirichlet=exact, quadrature_order=7
    )
    actual = solve_separable_diffusion(mesh, **options)
    expected = solve_conforming_quadrilateral(mesh, **options)
    nodes = qk_space(mesh, degree)[1]
    np.testing.assert_allclose(actual.pressure, exact(nodes), rtol=2e-11, atol=5e-12)
    np.testing.assert_allclose(actual.pressure, expected.pressure, rtol=2e-11, atol=5e-12)
    points = np.array([[0.17, 0.32], [0.72, 1.3]])
    _, gradient = actual.evaluate(points)
    np.testing.assert_allclose(
        gradient, np.column_stack((1 + points[:, 1], points[:, 0] + 2 * points[:, 1])), atol=5e-11
    )
    np.testing.assert_allclose(
        actual.physical_flux(points), -coefficient(points)[:, None] * gradient, atol=1e-14
    )
    assert actual.residual < 1e-12


def test_signed_terms_are_checked_as_the_sum_at_gauss_points() -> None:
    """Allow cancellation between separated terms without assuming each term is positive."""
    mesh = CartesianMacroMesh(3)
    coefficient = SeparableField(((1, 1), (lambda x: 2 * x, 1), (lambda x: -2 * x, 1)))
    actual = separable_diffusion_operators(mesh, 2, permeability=coefficient)
    expected = separable_diffusion_operators(mesh, 2)
    np.testing.assert_allclose(actual[0].toarray(), expected[0].toarray(), atol=2e-14)
    assert not actual[2].any()
    for bad in (SeparableField(((0, 1),)), SeparableField(((0.1, 1), (lambda x: -2 * x, 1)))):
        with pytest.raises(ValueError, match="positive"):
            separable_diffusion_operators(mesh, 2, permeability=bad)


def test_all_boundary_nodes_and_zero_forcing() -> None:
    """A single Q1 cell has no free nodes; its complete boundary interpolant is returned."""
    result = solve_separable_diffusion(CartesianMacroMesh(1), degree=1, dirichlet=3.0)
    np.testing.assert_array_equal(result.pressure, np.full(4, 3.0))
    assert result.residual == 0
    np.testing.assert_array_equal(SeparableField(())(np.zeros((2, 2))), np.zeros(2))


@pytest.mark.parametrize(
    "terms", [1, ((1,),), ((1, 2, 3),), (([1, 2], 1),), ((1j, 1),), ((np.inf, 1),)]
)
def test_invalid_term_contract(terms: Any) -> None:
    """Reject malformed factor pairs and invalid constants before numerical assembly."""
    with pytest.raises(ValueError, match="separable"):
        SeparableField(terms)


@pytest.mark.parametrize(
    "value", [lambda x: np.full((len(x), 1), 1.0), lambda x: x * 1j, lambda x: x * np.nan]
)
def test_invalid_callback_contract(value: Any) -> None:
    """Callback outputs cannot change shape, become complex or contain nonfinite values."""
    coefficient = SeparableField(((value, 1),))
    with pytest.raises(ValueError, match="factors"):
        coefficient(np.zeros((3, 2)))


@pytest.mark.parametrize("points", [[1, 2], np.zeros((2, 3)), [[np.inf, 1]], [[1j, 0]]])
def test_invalid_evaluation_points(points: Any) -> None:
    """Reject invalid geometry rather than reinterpret dimensions."""
    with pytest.raises(ValueError, match="XY pairs"):
        material()(points)


def test_operator_and_arithmetic_contracts() -> None:
    """Reject invalid mesh/space/source inputs and floating-point overflow explicitly."""
    mesh = CartesianMacroMesh(1)
    with pytest.raises(TypeError, match="Cartesian"):
        separable_diffusion_operators(None, 1)
    for invalid in (None, SeparableField(())):
        with pytest.raises(ValueError, match="nonempty"):
            separable_diffusion_operators(mesh, 1, permeability=invalid)
    with pytest.raises(TypeError, match="source"):
        separable_diffusion_operators(mesh, 1, source=1.0)
    for options in (dict(degree=0), dict(degree=1, order=0)):
        with pytest.raises(ValueError, match="integer"):
            separable_diffusion_operators(mesh, **options)
    huge = SeparableField(((1e308, 1e308),))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="remain finite"):
            huge(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="remain finite"):
            separable_diffusion_operators(mesh, 1, permeability=huge)
        with pytest.raises(ValueError, match="remain finite"):
            separable_diffusion_operators(mesh, 1, source=huge)
        discontinuous_huge = SeparableField(
            (
                (lambda x: np.where(x > 0.5, 1e308, 0), lambda y: np.where(y > 0.5, 1e308, 0)),
                (-1, 1),
            )
        )
        with pytest.raises(ValueError, match="positive"):
            separable_diffusion_operators(mesh, 1, permeability=discontinuous_huge)
