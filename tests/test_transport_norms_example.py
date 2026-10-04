"""P1 transport norms preserve quadrature, geometry and independent physical integrals."""

import numpy as np
import pytest

from examples.transport_campaign import layer
from examples.transport_coefficient_controls import norm_contribution, norm_contributions
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("epsilon", [0.1, 1.0])
def test_values_and_gradients_reproduce_full_tabulation_bitwise(epsilon):
    """Removing unused Hessians preserves both norms on translated, skew triangles."""
    base = TriangleMesh.unit_square(3)
    mesh = TriangleMesh(
        base.points @ np.array([[0.8, 0.2], [-0.1, 0.7]]) + [0.03, 0.08], base.cells
    )
    coefficients = np.sin(mesh.points[:, 0] * 2.3) + mesh.points[:, 1] ** 2
    for order in (8, 12):
        bary, weights = triangle_quadrature(order)
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        dofs, _, basis, gradient, _ = tabulate(mesh, 1, bary)
        difference = coefficients[dofs] @ basis.T - layer(points.reshape(-1, 2), epsilon).reshape(
            len(mesh.cells), -1
        )
        g = np.einsum("ti,tqia->tqa", coefficients[dofs], gradient)
        g[:, :, 0] -= 1 - np.exp((points[:, :, 0] - 1) / epsilon) / (
            epsilon * -np.expm1(-1 / epsilon)
        )
        expected = np.array(
            [mesh.areas @ (difference**2 @ weights), mesh.areas @ (np.sum(g**2, axis=2) @ weights)]
        )
        actual = norm_contribution((mesh, coefficients, epsilon, order))
        assert actual.tobytes() == expected.tobytes()


def test_nonzero_affine_field_has_independent_tensor_product_norms():
    """Absolute norms agree with independent rectangular Gauss integration, including area."""
    mesh = TriangleMesh.unit_square(3)
    coefficients = 2 + mesh.points[:, 0] - 3 * mesh.points[:, 1]
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x, y = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2, indexing="ij")
    measure = weights[:, None] * weights[None, :] / 4
    epsilon = 0.1
    exact = x - (np.exp((x - 1) / epsilon) - np.exp(-1 / epsilon)) / (1 - np.exp(-1 / epsilon))
    derivative = 1 - np.exp((x - 1) / epsilon) / (epsilon * (1 - np.exp(-1 / epsilon)))
    expected = np.array(
        [
            np.sum(measure * (2 + x - 3 * y - exact) ** 2),
            np.sum(measure * ((1 - derivative) ** 2 + 9)),
        ]
    )
    np.testing.assert_allclose(
        norm_contribution((mesh, coefficients, epsilon, 12)), expected, rtol=2e-14, atol=0
    )


def test_unused_hessian_transformation_is_never_requested(monkeypatch):
    """The norm owner needs only values and first physical derivatives."""
    import examples.transport_coefficient_controls as owner

    calls = []
    original = np.einsum

    def record(subscripts, *operands, **kwargs):
        """Count tensor transformations while retaining their original arithmetic."""
        calls.append(subscripts)
        return original(subscripts, *operands, **kwargs)

    monkeypatch.setattr(owner.np, "einsum", record)
    mesh = TriangleMesh.unit_square(1)
    norm_contribution((mesh, np.zeros(len(mesh.points)), 1.0, 8))
    assert calls.count("qin,tna->tqia") == 1
    assert "qinm,tna,tmb->tqiab" not in calls


@pytest.mark.parametrize("epsilon", [0.1, 1.0])
def test_grouped_fields_preserve_each_ordered_integral_and_inputs(epsilon):
    """Distinct affine and nonaffine fields retain their independent contributions."""
    mesh = TriangleMesh.unit_square(3)
    x, y = mesh.points.T
    coefficients = np.stack((2 + x - 3 * y, np.sin(2.3 * x) + y**2, np.zeros_like(x)))
    before = coefficients.tobytes()
    for order in (8, 12):
        expected = np.array(
            [norm_contribution((mesh, field, epsilon, order)) for field in coefficients]
        )
        actual = norm_contributions((mesh, coefficients, epsilon, order))
        assert actual.tobytes() == expected.tobytes()
    assert coefficients.tobytes() == before


def test_group_reuses_geometry_quadrature_basis_and_exact_field(monkeypatch):
    """A call with four fields prepares each invariant once, without a persistent cache."""
    import examples.transport_coefficient_controls as owner

    counts = {}
    for name in ("triangle_quadrature", "nodal_space", "reference_basis", "p1_geometry", "layer"):
        original = getattr(owner, name)

        def count(*args, _name=name, _original=original, **kwargs):
            """Count preparation calls without changing any array or arithmetic."""
            counts[_name] = counts.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, count)
    mesh = TriangleMesh.unit_square(2)
    coefficients = np.tile(mesh.points[:, 0], (4, 1))
    norm_contributions((mesh, coefficients, 1.0, 8))
    assert counts == dict.fromkeys(counts, 1)
    assert len(counts) == 5
    norm_contributions((mesh, coefficients, 1.0, 8))
    assert counts == dict.fromkeys(counts, 2)


@pytest.mark.parametrize("coefficients", [np.zeros(4), np.zeros((2, 3))])
def test_group_rejects_incompatible_field_coordinates(coefficients):
    """A batch must preserve the same complete P1 nodal coordinate system."""
    with pytest.raises(ValueError, match="one row per field"):
        norm_contributions((TriangleMesh.unit_square(1), coefficients, 1.0, 8))


def test_empty_field_group_has_two_norm_columns():
    """An empty batch has a well-defined result without inventing a field."""
    mesh = TriangleMesh.unit_square(1)
    assert norm_contributions((mesh, np.empty((0, len(mesh.points))), 1.0, 8)).shape == (0, 2)
