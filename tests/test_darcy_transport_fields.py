"""Higher Darcy velocities and primal-volume/numerical-normal transport invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.darcy_mixed import _evaluate, solve_darcy_bdm
from pymhm.darcy_rt import solve_darcy_rt, solve_darcy_rt_conforming
from pymhm.darcy_transport import HydrodynamicDispersion, solve_darcy_transport
from pymhm.darcy_velocity import (
    PolynomialDarcyVelocity,
    PrimalDarcyVelocity,
    polynomial_darcy_velocity,
)
from pymhm.elements import triangle_quadrature
from pymhm.lagrange import multiindices, nodal_space
from pymhm.reconstruction_moments import reconstruct_flux_moments
from pymhm.reservoir import CartesianCellField
from pymhm.rt import rt_evaluate
from pymhm.scalar_boundary import diffusive_boundary_matrix
from pymhm.scalar_transient import MacroCoefficient, solve_transient_transport


def _pressure(x):
    """A harmonic nonaffine pressure with an exact divergence-free quadratic velocity."""
    return 2 + x[:, 0] ** 3 - 3 * x[:, 0] * x[:, 1] ** 2


def _flux(x):
    """Return minus the independently differentiated harmonic cubic pressure."""
    return np.column_stack((-3 * x[:, 0] ** 2 + 3 * x[:, 1] ** 2, 6 * x[:, 0] * x[:, 1]))


def _gradient(x):
    """Return the complete nonradial physical derivative of the quadratic flux."""
    return np.stack(
        (np.column_stack((-6 * x[:, 0], 6 * x[:, 1])), np.column_stack((6 * x[:, 1], 6 * x[:, 0]))),
        axis=1,
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_rt_polynomial_conversion_preserves_executed_vectors_divergence_and_orientation(degree):
    """Equivalent P(m+1) coordinates differentiate the executed Piola field exactly."""
    mesh = TriangleMesh.unit_square()
    result = solve_darcy_rt(mesh, degree=degree, local_refinement=2, dirichlet=_pressure)
    bary = triangle_quadrature(4)[0]
    for cell, fine in enumerate(result.local_meshes):
        velocity = polynomial_darcy_velocity(result, cell)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells]).reshape(-1, 2)
        owners = np.repeat(np.arange(len(fine.cells)), len(bary))
        q, div = rt_evaluate(fine, result.flux[cell], degree, bary)
        assert_allclose(velocity.evaluate(points, owners), q.reshape(-1, 2), rtol=3e-11, atol=3e-11)
        assert_allclose(velocity.divergence(points, owners), div.ravel(), atol=3e-10)
        assert_allclose(velocity(points), velocity.evaluate(points, owners), atol=3e-11)


@pytest.mark.parametrize("degree,enrichment", [(1, 0), (2, 0), (2, 1)])
def test_bdm_conversion_retains_both_historical_and_enriched_coordinates(degree, enrichment):
    """Both BDM evaluators retain their canonical moment conventions in transport."""
    result = solve_darcy_bdm(
        TriangleMesh.unit_square(), degree=degree, enrichment=enrichment, dirichlet=_pressure
    )
    bary = triangle_quadrature(4)[0]
    for cell, fine in enumerate(result.local_meshes):
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells]).reshape(-1, 2)
        owners = np.repeat(np.arange(len(fine.cells)), len(bary))
        velocity = polynomial_darcy_velocity(result, cell)
        q, div = _evaluate(result.family, fine, result.flux[cell], bary)
        assert_allclose(velocity.evaluate(points, owners), q.reshape(-1, 2), atol=3e-10)
        assert_allclose(velocity.divergence(points, owners), div.ravel(), atol=3e-9)


def test_exact_nonradial_dispersion_derivative_and_zero_convention():
    """General grad(q) terms are checked against a separate tensor finite difference."""
    mesh = TriangleMesh.unit_square()
    nodes = np.einsum("qi,tij->tqj", multiindices(2) / 2, mesh.points[mesh.cells])
    velocity = PolynomialDarcyVelocity(mesh, _flux(nodes.reshape(-1, 2)).reshape(2, 6, 2), 2)
    points = np.array([[0.2, 0.4], [0.4, 0.2], [0.0, 0.0]])
    assert_allclose(velocity.gradient(points), _gradient(points), atol=3e-14)
    material = HydrodynamicDispersion(velocity, 0.1, 0.2, 0.03)
    actual = material.divergence(points)
    expected = np.zeros((2, 2))
    step = 1e-5
    for j in range(2):
        delta = np.eye(2)[j] * step
        expected += (
            material(points[:2] + delta)[:, :, j] - material(points[:2] - delta)[:, :, j]
        ) / (2 * step)
    assert_allclose(actual[:2], expected, rtol=3e-9, atol=3e-10)
    assert_array_equal(actual[2], np.zeros(2))
    assert_array_equal(material(points)[2], 0.1 * np.eye(2))


def test_published_primal_degrees_preserve_constant_concentration_and_weak_darcy_balance():
    """P3/r8 Darcy Λ2² drives P3/r8 transport Λ2⁸ without changing the raw field."""
    mesh = TriangleMesh.unit_square()
    pressure_space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces))
    transport_space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 8) for _ in mesh.faces))
    darcy = solve_darcy(
        mesh, degree=3, local_refinement=8, skeleton=pressure_space, dirichlet=_pressure
    )
    result = solve_darcy_transport(
        darcy,
        [0, 0.01, 0.02],
        velocity_representation="primal",
        permeability_gradient=0.0,
        skeleton=transport_space,
        degree=3,
        initial=1.0,
        dirichlet=1.0,
        diffusive_flux=dict.fromkeys(map(int, mesh.boundary_faces), 0.0),
        dirichlet_enforcement="weak",
        molecular=0.1,
        longitudinal=0.2,
        transverse=0.03,
    )
    for solution in result.solutions:
        for values in solution.values:
            assert_allclose(values, 1.0, atol=2e-12)
    assert_allclose(result.balance_residuals, 0.0, atol=3e-11)
    for cell, fine in enumerate(darcy.local_meshes):
        velocity = PrimalDarcyVelocity(darcy, cell, 0.0)
        nodes = nodal_space(fine, 3)[1]
        interior = fine.points[fine.cells].mean(axis=1)
        assert_allclose(velocity(interior), _flux(interior), atol=3e-12)
        assert_allclose(velocity.gradient(interior), _gradient(interior), atol=8e-11)
        assert_allclose(velocity.divergence(interior), 0.0, atol=1e-10)
        boundary = velocity.advection_boundary_matrix(3, 6)
        faces = tuple(map(int, mesh.cell_faces[cell]))
        assert_array_equal(
            diffusive_boundary_matrix(mesh, fine, faces, 3, velocity, 6).toarray(),
            boundary.toarray(),
        )
        assert len(nodes) == len(boundary.toarray())
        with pytest.raises(ValueError, match="face"):
            velocity.normal_trace(len(mesh.faces) + 1, interior[:1])
        face = int(mesh.cell_faces[cell, 0])
        with pytest.raises(ValueError, match="macroface"):
            velocity.normal_trace(face, interior[:1])


def test_material_derivatives_are_explicit_and_anisotropic():
    """Differentiation includes all tensor derivatives rather than an RT0 surrogate."""
    mesh = TriangleMesh.unit_square()

    def material(x):
        """Return an SPD tensor with different derivatives in each coordinate."""
        return np.stack(
            (
                np.column_stack((2 + x[:, 0], 0.2 * np.ones(len(x)))),
                np.column_stack((0.2 * np.ones(len(x)), 3 + x[:, 1])),
            ),
            axis=1,
        )

    grad = np.zeros((2, 2, 2))
    grad[0, 0, 0] = grad[1, 1, 1] = 1.0
    darcy = solve_darcy(
        mesh,
        degree=2,
        permeability=material,
        source=lambda x: -4 - x[:, 0] - x[:, 1],
        dirichlet=lambda x: 1 + x[:, 0] + x[:, 1],
    )
    velocity = PrimalDarcyVelocity(darcy, 0, lambda x: np.broadcast_to(grad, (len(x), 2, 2, 2)))
    points = darcy.local_meshes[0].points[darcy.local_meshes[0].cells].mean(axis=1)
    # Compare exact polynomial/material differentiation to the raw field itself.
    step = 1e-6
    expected = np.stack(
        [
            (velocity(points + step * np.eye(2)[j]) - velocity(points - step * np.eye(2)[j]))
            / (2 * step)
            for j in range(2)
        ],
        axis=-1,
    )
    assert_allclose(velocity.gradient(points), expected, atol=1e-8)
    for bad in (1.0, np.ones(2), np.ones((2, 2, 2)) * 1j, np.full((2, 2, 2), np.nan)):
        with pytest.raises(ValueError, match="material gradient"):
            PrimalDarcyVelocity(darcy, 0, bad).gradient(points)


def test_hdiv_coupling_and_reconstructed_velocity_use_the_actual_polynomial():
    """A nonzero exact initial/boundary patch exercises RT2 and moment reconstruction."""
    mesh = TriangleMesh.unit_square()
    darcy = solve_darcy_rt(mesh, degree=2, local_refinement=2, dirichlet=_pressure)
    actual = solve_darcy_transport(
        darcy,
        [0, 0.01],
        initial=1.0,
        dirichlet=1.0,
        degree=2,
        molecular=0.1,
        skeleton=darcy.skeleton,
    )
    assert_allclose(actual.solutions[0].values, 1.0, atol=3e-12)
    primal = solve_darcy(
        mesh, degree=3, local_refinement=2, skeleton=darcy.skeleton, dirichlet=_pressure
    )
    raw = tuple(PrimalDarcyVelocity(primal, c, 0.0).evaluate for c in range(2))
    reconstruction = reconstruct_flux_moments(
        primal.skeleton, primal.hybrid.trace, primal.local_meshes, raw, degree=2
    )
    for cell in range(2):
        points = primal.local_meshes[cell].points[primal.local_meshes[cell].cells].mean(axis=1)
        assert_allclose(
            polynomial_darcy_velocity(reconstruction, cell)(points), _flux(points), atol=3e-12
        )


def test_velocity_geometry_and_coupling_rejections():
    """Reject ambiguous geometry, nonreal data and unsupported discrete flux contracts."""
    mesh = TriangleMesh.unit_square()
    nodes = np.einsum("qi,tij->tqj", multiindices(1), mesh.points[mesh.cells])
    field = PolynomialDarcyVelocity(mesh, np.ones((2, 3, 2)), 1)
    for points, cells in (
        ([[2.0, 2.0]], None),
        ([[0.1, 0.2]], [-1]),
        ([[0.1, 0.2]], [1.0]),
        ([[0.1, 0.2]], [1, 0]),
        ([[0.1, 0.2]], [0]),
    ):
        with pytest.raises(ValueError):
            field.evaluate(points, cells)
    with pytest.raises(ValueError, match="real"):
        field.locate(np.array([[0.1, 0.2]]) * 1j)
    for points in (np.ones(3), [[np.nan, 0.0]], np.array([[0.1, 0.2]]) * 1j):
        with pytest.raises(ValueError):
            field(points)
    for values in (np.zeros(2), np.full((2, 3, 2), np.nan), np.ones((2, 3, 2)) * 1j):
        with pytest.raises(ValueError):
            PolynomialDarcyVelocity(mesh, values, 1)
    assert nodes.shape == (2, 3, 2)
    with pytest.raises(ValueError, match="triangular RT"):
        polynomial_darcy_velocity(None, 0)
    with pytest.raises(ValueError, match="MHM skeleton"):
        solve_darcy_transport(solve_darcy_rt_conforming(mesh), [0, 1])
    primal = solve_darcy(mesh)
    for kwargs, match in (
        ({"velocity_representation": "unknown"}, "representation"),
        ({"velocity_representation": "primal"}, "material gradient"),
        (
            {
                "velocity_representation": "primal",
                "permeability_gradient": 0.0,
                "stabilization": "supg",
            },
            "Galerkin",
        ),
        ({}, "RT0"),
    ):
        with pytest.raises(ValueError, match=match):
            solve_darcy_transport(primal, [0, 0.1], **kwargs)
    with pytest.raises(ValueError, match="primal Darcy"):
        solve_darcy_transport(solve_darcy_rt(mesh), [0, 0.1], velocity_representation="primal")
    with pytest.raises(ValueError, match="overridden"):
        solve_darcy_transport(solve_darcy_rt(mesh), [0, 0.1], velocity=0.0)


def test_primal_nonhomogeneous_material_and_time_data_are_exact():
    """The independently derived source x+tK preserves u=1+tx with K=1+y."""
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 2) for _ in mesh.faces))
    darcy = solve_darcy(
        mesh,
        degree=3,
        local_refinement=4,
        skeleton=space,
        permeability=lambda x: 1 + x[:, 1],
        dirichlet=lambda x: 3 - x[:, 0],
    )
    gradient = np.zeros((2, 2, 2))
    gradient[:, :, 1] = np.eye(2)
    result = solve_darcy_transport(
        darcy,
        [0, 0.03, 0.08],
        velocity_representation="primal",
        permeability_gradient=MacroCoefficient((gradient, gradient)),
        initial=1.0,
        degree=3,
        skeleton=space,
        molecular=0.1,
        dirichlet=lambda x, t: 1 + t * x[:, 0],
        source=lambda x, t: x[:, 0] + t * (1 + x[:, 1]),
    )
    for time, solution in zip(result.times[1:], result.solutions, strict=True):
        for fine, values in zip(solution.local_meshes, solution.values, strict=True):
            assert_allclose(values, 1 + time * nodal_space(fine, 3)[1][:, 0], atol=3e-12)
    assert_allclose(result.balance_residuals, 0.0, atol=3e-12)


def test_cartesian_cut_dispersion_and_raw_supg_rejection():
    """Material intersections are retained in both dispersion and old-state mass."""
    mesh = TriangleMesh.unit_square()
    material = CartesianCellField(np.array([[1.0, 2.0], [3.0, 4.0]]), (0.5, 0.5))
    darcy = solve_darcy(
        mesh, degree=2, local_refinement=2, permeability=material, dirichlet=lambda x: x[:, 0]
    )
    result = solve_darcy_transport(
        darcy,
        [0, 0.01],
        velocity_representation="primal",
        permeability_gradient=0.0,
        initial=0.0,
        dirichlet=0.0,
        degree=2,
    )
    assert_allclose(result.solutions[0].values, 0.0, atol=1e-14)
    velocities = tuple(PrimalDarcyVelocity(darcy, c, 0.0) for c in range(2))
    with pytest.raises(ValueError, match="Galerkin"):
        solve_transient_transport(
            mesh,
            [0, 0.01],
            velocity=MacroCoefficient(velocities),
            velocity_divergence=MacroCoefficient(tuple(v.divergence for v in velocities)),
            stabilization="supg",
            local_meshes=darcy.local_meshes,
        )
