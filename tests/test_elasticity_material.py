"""Heterogeneous constitutive-law and stabilization-bound verification."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_elasticity


@pytest.mark.parametrize("formulation,degree", [("gals", 1), ("gals", 2), ("taylor-hood", 2)])
def test_variable_lame_coefficients_exact_affine_equilibrium(formulation, degree):
    mesh = TriangleMesh.unit_square(2)

    def mu(x):
        return 1 + 0.4 * x[:, 0] + 0.2 * x[:, 1]

    def lam(x):
        return 2 + 0.3 * x[:, 0] - 0.1 * x[:, 1]

    gradient = np.array([[0.5, 0.3], [-0.1, 0.2]])

    def displacement(x):
        return x @ gradient.T

    def pressure(x):
        return -np.trace(gradient) * lam(x)

    def stress(x):
        return mu(x)[:, None, None] * (gradient + gradient.T) + lam(x)[:, None, None] * np.trace(
            gradient
        ) * np.eye(2)

    source = -(gradient + gradient.T) @ [0.4, 0.2] - np.trace(gradient) * np.array([0.3, -0.1])
    solution = solve_elasticity(
        mesh,
        lame_lambda=lam,
        lame_mu=mu,
        lame_mu_gradient=(0.4, 0.2),
        shear_bounds=(1.0, 1.6, np.sqrt(0.2)),
        formulation=formulation,
        degree=degree,
        source=source,
        dirichlet=displacement,
    )
    assert solution.l2_error(displacement) < 2e-12
    assert solution.pressure_l2_error(pressure) < 2e-11
    assert solution.stress_l2_error(stress) < 2e-11
    assert solution.compressibility_l2() < 2e-11
    assert_allclose(solution.hybrid.gauge_multipliers, 0, atol=1e-12)


def test_variable_shear_derivatives_in_quadratic_incompressible_patch():
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces), 2)

    def mu(x):
        return 1 + x[:, 0]

    def displacement(x):
        return np.column_stack((x[:, 0] ** 2, -2 * x[:, 0] * x[:, 1]))

    def source(x):
        return np.column_stack((-2 - 6 * x[:, 0], 2 * x[:, 1]))

    solution = solve_elasticity(
        mesh,
        lame_lambda=lambda x: np.full(len(x), np.inf),
        lame_mu=mu,
        lame_mu_gradient=(1.0, 0.0),
        shear_bounds=(1.0, 2.0, 1.0),
        degree=3,
        local_refinement=2,
        source=source,
        dirichlet=displacement,
        skeleton=skeleton,
    )
    assert solution.l2_error(displacement) < 2e-12
    assert solution.pressure_l2_error(0) < 2e-11
    assert solution.compressibility_l2() < 2e-11


@pytest.mark.parametrize(
    "options,match",
    [
        ({"lame_lambda": 1j}, "real"),
        ({"lame_mu": lambda x: np.ones(len(x))}, "requires"),
        ({"lame_mu_gradient": (1.0, 0.0)}, "zero gradient"),
        ({"shear_bounds": (1.0,)}, "shear_bounds"),
        ({"shear_bounds": (-1.0, 2.0, 0.0)}, "shear_bounds"),
        ({"shear_bounds": (1.0, 0.5, 0.0)}, "shear_bounds"),
        ({"shear_bounds": (1.0, 2.0, -1.0)}, "shear_bounds"),
        ({"shear_bounds": (2.0, 3.0, 0.0)}, "do not bound"),
        ({"shear_bounds": (0.1, 0.5, 0.0)}, "do not bound"),
        (
            {
                "lame_mu": lambda x: 1 + x[:, 0],
                "lame_mu_gradient": (1.0, 0.0),
                "shear_bounds": (1.0, 2.0, 0.5),
            },
            "do not bound",
        ),
        (
            {
                "lame_mu": lambda x: np.where((x[:, 0] == 0) | (x[:, 0] == 1), 1.0, -1.0),
                "lame_mu_gradient": (0.0, 0.0),
                "shear_bounds": (0.1, 1.0, 0.0),
            },
            "positive",
        ),
    ],
)
def test_material_input_rejections(options, match):
    with pytest.raises(ValueError, match=match):
        solve_elasticity(TriangleMesh.unit_square(), **options)
