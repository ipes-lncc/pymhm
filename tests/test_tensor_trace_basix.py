"""Ordering, polynomial completeness and oriented trace conventions under Basix."""

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval
from numpy.testing import assert_allclose

from pymhm.mesh import FaceSpace
from pymhm.quadrilateral import _cardinals, qk_basis, quadrilateral_quadrature


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 6])
def test_basix_interval_export_replays_tensor_coefficients_and_derivatives(degree):
    """Exported polynomial coordinates and native tables describe the same Qk field."""
    coordinates = np.linspace(0, 1, degree + 1)
    reference, _ = quadrilateral_quadrature(degree + 2)
    values, gradients = qk_basis(degree, reference)
    x = np.column_stack([polyval(reference[:, 0], c) for c in _cardinals(degree)])
    y = np.column_stack([polyval(reference[:, 1], c) for c in _cardinals(degree)])
    exported = np.einsum("qi,qj->qij", y, x).reshape(values.shape)
    assert_allclose(exported, values, rtol=0, atol=6e-12)
    # An independently specified complete Qk monomial, using x-fast nodal DOFs.
    coefficients = np.outer(coordinates**degree, coordinates**degree).ravel()
    assert_allclose(
        values @ coefficients,
        reference[:, 0] ** degree * reference[:, 1] ** degree,
        rtol=0,
        atol=3e-14,
    )
    assert_allclose(
        np.einsum("qia,i->qa", gradients, coefficients),
        degree
        * np.column_stack(
            (
                reference[:, 0] ** (degree - 1) * reference[:, 1] ** degree,
                reference[:, 0] ** degree * reference[:, 1] ** (degree - 1),
            )
        ),
        rtol=0,
        atol=3e-13,
    )


def test_continuous_hp_faces_preserve_breakpoint_first_node_order_and_support():
    """P1/P3/P4 reproduce one physical cubic where each segment permits it."""
    face = FaceSpace((0.0, 0.17, 0.63, 1.0), (1, 3, 4), continuous=True)
    internal = np.r_[
        0.17 + (0.63 - 0.17) * np.arange(1, 3) / 3,
        0.63 + (1 - 0.63) * np.arange(1, 4) / 4,
    ]
    nodes = np.r_[face.breaks, internal]
    assert_allclose(face.evaluate(nodes), np.eye(face.size), rtol=0, atol=3e-15)
    parameters = np.r_[0.17, np.linspace(0.2, 0.6, 13), 0.63, np.linspace(0.7, 1, 13)]
    assert_allclose(face.evaluate(parameters) @ nodes**3, parameters**3, atol=2e-15)
    values = face.evaluate([0.1, 0.4, 0.8])
    assert_allclose(values @ face.constant_coefficients(), 1, rtol=0, atol=2e-15)
    # No segment acquires the independent interior DOFs of another segment.
    np.testing.assert_array_equal(values[0, 4:], 0)
    np.testing.assert_array_equal(values[1, 6:], 0)
    np.testing.assert_array_equal(values[2, 4:6], 0)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 6])
def test_legendre_face_orientation_and_unnormalized_physical_mass(degree):
    """Face reversal changes odd-mode signs and preserves degree-j mass 1/(2j+1)."""
    face = FaceSpace.uniform(degree)
    parameters, weights = face.quadrature(degree + 1)
    values = face.evaluate(parameters)
    assert_allclose(
        face.evaluate(1 - parameters),
        values * (-1.0) ** np.arange(degree + 1),
        rtol=0,
        atol=3e-15,
    )
    assert_allclose(
        values.T @ (weights[:, None] * values),
        np.diag(1 / (2 * np.arange(degree + 1) + 1)),
        rtol=0,
        atol=5e-16,
    )
    assert face.evaluate([]).shape == (0, degree + 1)
