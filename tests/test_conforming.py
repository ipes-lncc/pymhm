"""Classical Qk baseline polynomial, physical-boundary and evaluation checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.darcy.conforming import solve_conforming_quadrilateral
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_affine_anisotropic_patch(degree):
    """The classical baseline reproduces affine pressure on a translated rectangle."""
    mesh = CartesianMacroMesh(2, 3, (2.0, 4.0, -1.0, 2.0))
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])
    result = solve_conforming_quadrilateral(
        mesh, degree=degree, permeability=tensor, dirichlet=lambda x: 1 + x[:, 0] - 2 * x[:, 1]
    )
    points = np.array([[2.0, -1.0], [2.2, 0.1], [3.8, 1.9], [4.0, 2.0]])
    p, grad = result.evaluate(points)
    assert_allclose(p, 1 + points[:, 0] - 2 * points[:, 1], atol=3e-13)
    assert_allclose(grad, np.tile([1.0, -2.0], (4, 1)), atol=3e-13)
    assert_allclose(
        result.physical_flux(points), np.tile(-tensor @ [1.0, -2.0], (4, 1)), atol=3e-13
    )
    assert result.residual < 2e-15


def test_mixed_and_pure_neumann_mean():
    """Both natural-boundary conventions recover the same physical quadratic patch."""
    mesh = CartesianMacroMesh(2)

    def exact(x):
        """Quadratic pressure with mean 2/3 on the unit square."""
        return x[:, 0] ** 2 + x[:, 1] ** 2

    all_neumann = {int(f): lambda x, n=mesh.normals[f]: -2 * x @ n for f in mesh.boundary_faces}
    points = np.array([[0.15, 0.27], [0.99, 0.13], [0.37, 0.95]])
    for natural in (all_neumann, {min(all_neumann): all_neumann[min(all_neumann)]}):
        result = solve_conforming_quadrilateral(
            mesh, degree=2, source=-4.0, dirichlet=exact, neumann=natural, mean_pressure=2 / 3
        )
        assert_allclose(result.evaluate(points)[0], exact(points), atol=3e-13)
        assert result.residual < 3e-15


def test_empty_free_set_and_invalid_inputs():
    """Handle fully prescribed Q1 cells and reject incompatible physical input."""
    mesh = CartesianMacroMesh()
    result = solve_conforming_quadrilateral(mesh, degree=1, dirichlet=1.0)
    assert_allclose(result.pressure, 1.0)
    assert result.residual == 0
    for points in (np.array([[1j, 0]]), np.ones((2, 3)), np.array([[np.nan, 0]])):
        with pytest.raises(ValueError, match="finite real"):
            result.evaluate(points)
    with pytest.raises(ValueError, match="outside"):
        result.evaluate(np.array([[1.1, 0.2]]))
    with pytest.raises(ValueError, match="boundary faces"):
        solve_conforming_quadrilateral(mesh, neumann={99: 0.0})
    neumann = {int(f): 0.0 for f in mesh.boundary_faces}
    with pytest.raises(ValueError, match="incompatible"):
        solve_conforming_quadrilateral(mesh, source=1.0, neumann=neumann)
    with pytest.raises(ValueError, match="finite"):
        solve_conforming_quadrilateral(mesh, neumann=neumann, mean_pressure=np.inf)
