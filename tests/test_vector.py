"""Velocity, pressure-gauge and rigid-motion verification for vector MHM."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_brinkman, solve_elasticity
from pymhm.vector import _lagrange


def rotation(x):
    """Divergence-free rigid rotation, shared by flow and elasticity patches."""
    return np.column_stack((x[:, 1], -x[:, 0]))


@pytest.mark.parametrize("drag", [0, 1.5, 100])
def test_brinkman_affine_patch_and_pressure_gauge(drag):
    mesh = TriangleMesh.unit_square(2)
    result = solve_brinkman(
        mesh, drag=drag, source=lambda x: drag * rotation(x), dirichlet=rotation, mean_pressure=2
    )
    assert result.l2_error(rotation) < 1e-10
    assert result.pressure_l2_error(2) < 1e-9
    assert result.divergence_l2() < 1e-10
    assert_allclose(result.hybrid.gauge_multipliers, 0, atol=1e-10)


def test_stokes_quadratic_patch_with_linear_traction():
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)

    def velocity(x):
        return np.column_stack((x[:, 1] * (1 - x[:, 1]), np.zeros(len(x))))

    result = solve_brinkman(mesh, source=[2, 0], dirichlet=velocity, skeleton=space)
    assert result.l2_error(velocity) < 2e-13
    assert result.pressure_l2_error(0) < 2e-12
    assert result.divergence_l2() < 2e-12


def test_stokes_pressure_gradient_patch():
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces), 2)
    result = solve_brinkman(mesh, source=[1, 2], skeleton=space, local_refinement=4)
    assert result.l2_error([0, 0]) < 1e-12
    assert result.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1] - 1.5) < 1e-11


def test_incompatible_incompressible_boundary_data():
    with pytest.raises(ValueError, match="incompatible"):
        solve_brinkman(TriangleMesh.unit_square(), dirichlet=lambda x: x)


@pytest.mark.parametrize(
    "field",
    [rotation, lambda x: np.column_stack((2 * x[:, 0] + x[:, 1], -x[:, 0] + 0.5 * x[:, 1]))],
)
def test_elasticity_rigid_and_affine_patch(field):
    result = solve_elasticity(TriangleMesh.unit_square(2), dirichlet=field, formulation="primal")
    assert result.l2_error(field) < 2e-13
    with pytest.raises(ValueError, match="no pressure"):
        result.pressure_l2_error(0)


def test_lagrange_partition_unity_gradient_and_nodal_values():
    mesh = TriangleMesh.unit_square()
    bary = np.array(
        [[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
    )
    for degree in (1, 2):
        _, _, basis, gradient = _lagrange(mesh, degree, bary)
        assert_allclose(basis.sum(axis=1), 1)
        assert_allclose(gradient.sum(axis=2), 0, atol=1e-14)
    assert_allclose(basis, np.eye(6))


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (solve_brinkman, {"viscosity": 0}),
        (solve_brinkman, {"viscosity": np.nan}),
        (solve_brinkman, {"drag": -1}),
        (solve_brinkman, {"drag": np.inf}),
        (solve_elasticity, {"lame_mu": 0}),
        (solve_elasticity, {"lame_lambda": -1}),
        (solve_elasticity, {"lame_lambda": np.nan}),
    ],
)
def test_vector_parameter_validation(function, kwargs):
    with pytest.raises(ValueError):
        function(TriangleMesh.unit_square(), **kwargs)


@pytest.mark.parametrize("function", [solve_brinkman, solve_elasticity])
def test_vector_skeleton_validation(function):
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="skeleton"):
        function(mesh, skeleton=SkeletonSpace(mesh))
    with pytest.raises(ValueError, match="skeleton"):
        function(mesh, skeleton=SkeletonSpace(TriangleMesh.unit_square(), components=2))


@pytest.mark.parametrize("drag", [0.0, 1.0, 100.0])
def test_usfem_consistent_affine_velocity_and_pressure(drag):
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_brinkman(
        mesh,
        drag=drag,
        source=lambda x: drag * rotation(x) + np.array([1.0, 2.0]),
        dirichlet=rotation,
        skeleton=space,
        formulation="usfem",
        local_refinement=4,
    )
    assert result.l2_error(rotation) < 1e-11
    assert result.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1] - 1.5) < 1e-10
    assert result.divergence_l2() < 1e-10


def test_oseen_constant_transport_patch():
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_brinkman(
        mesh,
        advection=(2.0, 3.0),
        drag=1,
        source=lambda x: rotation(x) + np.array([3.0, -2.0]),
        dirichlet=rotation,
        skeleton=space,
    )
    assert result.l2_error(rotation) < 1e-11
    assert result.pressure_l2_error(0) < 1e-10


def test_invalid_flow_formulation():
    with pytest.raises(ValueError, match="formulation"):
        solve_brinkman(TriangleMesh.unit_square(), formulation="invalid")
    with pytest.raises(ValueError, match="Taylor-Hood"):
        solve_brinkman(TriangleMesh.unit_square(), advection=(1, 0), formulation="usfem")


@pytest.mark.parametrize("options", [{"lame_mu": 0.0}, {"lame_lambda": -1.0}, {"degree": 0}])
def test_primal_elasticity_rejects_unsupported_material_and_degree(options):
    """The explicit displacement baseline must retain its physical guards."""
    with pytest.raises(ValueError):
        solve_elasticity(TriangleMesh.unit_square(), formulation="primal", **options)


def test_primal_elasticity_requires_a_vector_skeleton():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="skeleton"):
        solve_elasticity(mesh, formulation="primal", skeleton=SkeletonSpace(mesh))
