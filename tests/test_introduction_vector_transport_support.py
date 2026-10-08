"""Introduction helpers preserve physical norms, executed coordinates and boundary gauges."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.introduction.transport import scalar_error_norms, scalar_reference
from examples.introduction.vector import (
    BrokenVectorEvaluator,
    TriangleVectorEvaluator,
    brinkman_reference,
    cauchy_stress,
    elasticity_errors,
    flow_error_norms,
    triangle_grid_quadrature,
)
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.meshes.triangle import TriangleMesh


def test_vector_evaluation_and_plane_strain_norms() -> None:
    """Executed nodal coordinates reproduce affine vectors and the declared Cauchy stress."""
    mesh = TriangleMesh.unit_square(2)
    _, nodes = nodal_space(mesh, 2)
    matrix = np.array([[2.0, 3.0], [-1.0, 4.0]])
    evaluator = TriangleVectorEvaluator(mesh, 2, nodes @ matrix.T)
    points, weights = triangle_grid_quadrature(3, 4)
    value, gradient = evaluator(points)
    np.testing.assert_allclose(value, points @ matrix.T, atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(gradient, np.tile(matrix, (len(points), 1, 1)), atol=1e-12)

    def modulus(x):
        return np.ones(len(x))

    stress = cauchy_stress(points, gradient, modulus=modulus)
    np.testing.assert_allclose(stress, np.tile([[10, 2], [2, 14]], (len(points), 1, 1)), atol=1e-12)

    def zero(x):
        return (np.zeros((len(x), 2)), np.zeros((len(x), 2, 2)))

    errors = elasticity_errors(zero, evaluator, points, weights, modulus=modulus)
    np.testing.assert_allclose(errors["stress_L2"], np.sqrt(304), atol=1e-12, rtol=1e-10)
    assert errors["stress_relative"] == pytest.approx(1)
    broken = BrokenVectorEvaluator(mesh, tuple(evaluator for _ in mesh.cells))
    np.testing.assert_allclose(broken(points)[0], value, atol=1e-12, rtol=1e-10)


class PolynomialLayer:
    """Manufactured P2/P1 fields with analytic source and mean-zero pressure."""

    epsilon = 0.2
    viscosity = 0.3
    drag = 2.0

    @staticmethod
    def value(points: np.ndarray) -> np.ndarray:
        """Return a scalar quadratic with zero vertical exterior trace."""
        return points[:, 0] * (1 - points[:, 0])

    @staticmethod
    def scalar_gradient(points: np.ndarray) -> np.ndarray:
        """Differentiate the quadratic independently in physical coordinates."""
        return np.column_stack((1 - 2 * points[:, 0], np.zeros(len(points))))

    @staticmethod
    def velocity(points: np.ndarray) -> np.ndarray:
        """Return a divergence-free quadratic velocity with nonhomogeneous exterior data."""
        reversed_points = points[:, ::-1]
        return reversed_points * (1 - reversed_points)

    @staticmethod
    def pressure(points: np.ndarray) -> np.ndarray:
        """Return physical zero-volume-mean pressure."""
        return points[:, 0] - points[:, 1]

    @staticmethod
    def gradient(points: np.ndarray) -> np.ndarray:
        """Return the independently differentiated velocity Jacobian."""
        value = np.zeros((len(points), 2, 2))
        value[:, 0, 1] = 1 - 2 * points[:, 1]
        value[:, 1, 0] = 1 - 2 * points[:, 0]
        return value


def test_broken_physical_scalar_and_flow_norms() -> None:
    """Exact coefficient fields have zero raw-gradient errors, macro mass and pressure mean."""
    mesh = TriangleMesh.unit_square(2)
    _, nodes = nodal_space(mesh, 2)
    truth = PolynomialLayer()
    scalar = scalar_error_norms(
        (mesh,),
        (truth.value(nodes),),
        2,
        truth.value,
        truth.scalar_gradient,
        truth.epsilon,
        order=5,
    )
    assert scalar["scalar_l2"] < 1e-12 and scalar["flux_l2"] < 1e-12
    _, pressure_nodes = nodal_space(mesh, 1)
    flow = flow_error_norms(
        (mesh,), (truth.velocity(nodes),), (truth.pressure(pressure_nodes),), 2, 1, truth, order=5
    )
    for name in (
        "velocity_l2",
        "pressure_l2",
        "velocity_h1_seminorm",
        "divergence_l2",
        "pseudostress_l2",
        "macro_mass_defect",
        "pressure_integral",
    ):
        assert abs(flow[name]) < 1e-12


def test_supplied_conforming_scalar_form_and_natural_horizontal_data(tmp_path: Path) -> None:
    """Generic global assembly preserves zero vertical data and exact quadratic source."""
    pytest.importorskip("dolfinx")
    import ufl

    truth = PolynomialLayer()

    def forms(space: Any, epsilon: float) -> tuple[Any, Any, Any, Any]:
        """Declare the manufactured reaction-diffusion form independently."""
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        domain = space.ufl_domain()
        x = ufl.SpatialCoordinate(domain)
        exact = x[0] * (1 - x[0])
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
        return (
            (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) + u * v) * dx,
            (2 * epsilon + exact) * v * dx,
            exact,
            dx,
        )

    (mesh, coefficients), metrics, archive = scalar_reference(
        truth.epsilon, 2, 2, forms, truth, reports=tmp_path, root=tmp_path
    )
    assert metrics["scalar_l2"] < 1e-12 and metrics["gradient_l2"] < 1e-12
    _, nodes = nodal_space(mesh, 2)
    np.testing.assert_allclose(coefficients, truth.value(nodes), atol=1e-12, rtol=1e-10)
    assert (tmp_path / archive["path"]).is_file()


def test_supplied_mixed_form_preserves_nonzero_boundary_and_physical_gauge(tmp_path: Path) -> None:
    """Generic constrained assembly exactly recovers a P2/P1 manufactured solution."""
    pytest.importorskip("dolfinx")
    import ufl

    truth = PolynomialLayer()

    def forms(space: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Declare the vector-Laplacian mixed energy and independent analytic source."""
        u, p = ufl.TrialFunctions(space)
        v, q = ufl.TestFunctions(space)
        domain = space.ufl_domain()
        x = ufl.SpatialCoordinate(domain)
        exact_u = ufl.as_vector((x[1] * (1 - x[1]), x[0] * (1 - x[0])))
        exact_p = x[0] - x[1]
        forcing = (
            2 * truth.viscosity * ufl.as_vector((1, 1))
            + truth.drag * exact_u
            + ufl.as_vector((1, -1))
        )
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 6})
        a = (
            truth.viscosity * ufl.inner(ufl.grad(u), ufl.grad(v))
            + truth.drag * ufl.inner(u, v)
            - p * ufl.div(v)
            - q * ufl.div(u)
        ) * dx
        return a, ufl.inner(forcing, v) * dx, q * dx, exact_u, exact_p, dx

    (mesh, velocity, pressure), metrics, archive = brinkman_reference(
        2, forms, truth, reports=tmp_path, root=tmp_path
    )
    assert metrics["velocity_l2"] < 1e-12 and metrics["pressure_l2"] < 1e-12
    assert abs(metrics["pressure_integral"]) < 1e-12
    _, velocity_nodes = nodal_space(mesh, 2)
    _, pressure_nodes = nodal_space(mesh, 1)
    np.testing.assert_allclose(velocity, truth.velocity(velocity_nodes), atol=1e-12, rtol=1e-10)
    np.testing.assert_allclose(pressure, truth.pressure(pressure_nodes), atol=1e-12, rtol=1e-10)
    assert (tmp_path / archive["path"]).is_file()
