"""Analytical MHM lifts preserve RT0 traces and source/mean conventions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import TetraMesh, TriangleMesh, solve_darcy, solve_darcy_3d
from pymhm.analytic import analytic_darcy_local, solve_darcy_analytic
from pymhm.elements import triangle_quadrature
from pymhm.tetrahedral import tetrahedron_quadrature


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("natural", [False, True])
def test_analytical_quadratic_patch_and_physical_mean(dimension, natural):
    """The metric quadratic has constant source and exactly RT0 normal flux in 2D/3D."""
    mesh = TriangleMesh.unit_square(2) if dimension == 2 else TetraMesh.unit_cube()
    tensor = np.eye(dimension) * 2 + 0.2
    inverse = np.linalg.inv(tensor)
    shift = np.arange(1.0, dimension + 1)

    def exact(points):
        """Quadratic potential lying in the analytical local span."""
        return 1 + points @ shift + 0.5 * np.einsum("qi,ij,qj->q", points, inverse, points)

    def flux(points):
        """Physical flux whose normal is constant on every affine face."""
        return -shift @ tensor - points

    mean = (
        1
        + shift.sum() / 2
        + 0.5 * (np.trace(inverse) / 3 + (inverse.sum() - np.trace(inverse)) / 4)
    )
    boundary = {int(f): lambda x, n=mesh.normals[f]: flux(x) @ n for f in mesh.boundary_faces}
    result = solve_darcy_analytic(
        mesh,
        permeability=tensor,
        source=-dimension,
        dirichlet=exact,
        neumann=boundary if natural else None,
        mean_pressure=mean,
    )
    assert_allclose(result.errors(exact, flux, 5), 0, atol=3e-12)
    assert result.hybrid.residual < 3e-14
    bary, weights = triangle_quadrature(5) if dimension == 2 else tetrahedron_quadrature(5)
    for cell in range(len(mesh.cells)):
        problem, space = analytic_darcy_local(mesh, cell, tensor)
        values = space.evaluate(bary @ mesh.points[mesh.cells[cell]])[0]
        assert_allclose(weights @ values, np.r_[1.0, np.zeros(dimension + 1)], atol=3e-15)
        vector = np.arange(1.0, dimension + 1)
        coefficients = space.integrate_rt0(2.5, vector, 3.1)
        _, gradients = space.evaluate(bary @ mesh.points[mesh.cells[cell]])
        recovered = -np.einsum("ab,qib,i->qa", tensor, gradients, coefficients)
        expected = vector + 3.1 / dimension * (bary @ mesh.points[mesh.cells[cell]] - space.center)
        assert_allclose(recovered, expected, atol=2e-14)
        assert_allclose(weights @ values @ coefficients, 2.5, atol=2e-14)
        # Mean-free exact Neumann lifts have one prescribed face flux at a time.
        lifts = problem.condense().lifts
        for side, face in enumerate(mesh.cell_faces[cell]):
            points = mesh.points[mesh.faces[face]].mean(axis=0, keepdims=True)
            gradients = space.evaluate(points)[1]
            physical = np.einsum("ab,qib,ij->qja", tensor, gradients, lifts)
            assert_allclose(
                physical[0] @ mesh.normals[face], np.eye(dimension + 1)[side], atol=1e-13
            )


@pytest.mark.parametrize("dimension", [2, 3])
def test_variable_source_matches_full_p2_mhm(dimension):
    """Exact harmonic moments and a separate source solve equal the complete P2 solve."""
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()

    def source(points):
        """Nonconstant forcing exercises the complete source moments."""
        return 1 + points[:, 0] + points[:, 1] ** 2

    result = solve_darcy_analytic(mesh, source=source, dirichlet=1.0, source_refinement=2)
    routine = solve_darcy if dimension == 2 else solve_darcy_3d
    classical = routine(
        mesh, source=source, dirichlet=1.0, degree=2, local_refinement=2, quadrature_order=8
    )
    assert_allclose(result.hybrid.trace, classical.hybrid.trace, atol=2e-12)
    assert_allclose(result.hybrid.coarse, classical.hybrid.coarse, atol=2e-12)
    bary, _ = triangle_quadrature(3) if dimension == 2 else tetrahedron_quadrature(3)
    from pymhm.lagrange import tabulate
    from pymhm.tetrahedral import tetra_tabulate

    for cell, fine in enumerate(result.source_meshes):
        if dimension == 2:
            dofs, _, basis, _, _ = tabulate(fine, 2, bary)
        else:
            dofs, _, basis, _ = tetra_tabulate(fine, 2, bary)
        coefficients = classical.pressure[cell]
        assert_allclose(
            result.evaluate(cell, bary)[0],
            np.einsum("qi,ti->tq", basis, coefficients[dofs]),
            atol=3e-12,
        )


def test_analytic_input_contracts():
    """Reject unsupported geometry, variable materials and malformed physical inputs."""
    mesh = TriangleMesh.unit_square()
    for cell in (2, -1):
        with pytest.raises(ValueError):
            analytic_darcy_local(mesh, cell)
    with pytest.raises(ValueError, match="macrocell-constant"):
        analytic_darcy_local(mesh, 0, lambda x: np.ones(len(x)))
    with pytest.raises(ValueError, match="mean_pressure"):
        solve_darcy_analytic(mesh, mean_pressure=np.inf)
    _, space = analytic_darcy_local(mesh, 0)
    with pytest.raises(ValueError, match="RT0 integration"):
        space.integrate_rt0(0.0, [np.inf, 1.0], 1.0)
    for points in (np.ones((2, 3)), np.array([[1j, 0]]), np.array([[np.inf, 0]])):
        with pytest.raises(ValueError, match="physical coordinates"):
            space.evaluate(points)
    result = solve_darcy_analytic(mesh)
    with pytest.raises(ValueError, match="outside"):
        result.pressure_update(2, np.ones((1, 2)))


def test_neumann_source_lifts_can_cancel_in_global_assembly():
    """A symmetric nonzero source can have zero resolved trace and coarse load."""
    mesh = TriangleMesh.unit_square(2)
    result = solve_darcy_analytic(
        mesh,
        source=lambda x: 8 * np.pi**2 * np.prod(np.cos(2 * np.pi * x), axis=1),
        neumann={int(face): 0.0 for face in mesh.boundary_faces},
        quadrature_order=10,
    )
    assert result.hybrid.residual < 2e-13
    assert_allclose(result.hybrid.trace, 0, atol=2e-12)
    assert max(np.linalg.norm(p) for p in result.source_pressure) > 1.0
