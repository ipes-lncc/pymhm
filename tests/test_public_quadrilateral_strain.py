"""Pixel-resolved strain forms preserve physical energy, force and strong data."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.formulations.conforming_elasticity import (
    conforming_elasticity,
    displacement_difference,
)
from pymhm.fem.geometry import volume_centroid
from pymhm.fem.vector.quadrilateral import quadrilateral_strain_operators
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.hexahedron import HexMesh


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_strain_rigid_kernel_exact_affine_patch_and_reference_increment(degree):
    mesh = CartesianMacroMesh(2, 1, (0.0, 1.0, 0.0, 0.45))

    def exact(points):
        return np.column_stack((1 + points[:, 0], -points[:, 1]))

    forms = quadrilateral_strain_operators(mesh, degree, lame_lambda=3, lame_mu=2)
    displacement = exact(forms.nodes).ravel()
    # epsilon=diag(1,-1), so epsilon:C:epsilon=8 independently of lambda.
    assert_allclose(displacement @ forms.matrix @ displacement, 8 * 0.45, atol=1e-12)
    rotation = np.column_stack((-forms.nodes[:, 1], forms.nodes[:, 0])).ravel()
    assert_allclose(forms.matrix @ rotation, 0, atol=1e-12)
    assert_allclose(forms.mass.sum(), 0.45, atol=1e-12)
    coarse, residual = conforming_elasticity(mesh, degree=degree, dirichlet=exact)
    fine, _ = conforming_elasticity(
        CartesianMacroMesh(4, 2, mesh.bounds), degree=degree, dirichlet=exact
    )
    assert residual < 1e-10
    points = np.array([[0.2, 0.1], [0.7, 0.3]])
    assert_allclose(coarse.evaluate(points), exact(points), atol=1e-12, rtol=1e-10)
    increment = displacement_difference(fine, coarse)
    assert increment["displacement_l2"] < 1e-12
    assert increment["gradient_l2"] < 1e-12
    with pytest.raises(ValueError, match="nested"):
        displacement_difference(coarse, fine)


def test_exact_pixel_integrals_without_fitting_the_approximation_cells():
    mesh = CartesianMacroMesh(1)
    mu = CartesianCellField(np.array([[1.0, 3.0], [2.0, 4.0], [5.0, 6.0]]), (1 / 3, 1 / 2))
    body = CartesianCellField(np.stack((mu.values, -2 * mu.values), axis=-1), mu.spacing)
    forms = quadrilateral_strain_operators(
        mesh, 2, lame_lambda=3, lame_mu=mu, source=body, integration_field=mu, order=3
    )
    u = np.column_stack((forms.nodes[:, 0], -forms.nodes[:, 1])).ravel()
    assert_allclose(u @ forms.matrix @ u, 4 * mu.values.mean(), atol=1e-12)
    assert_allclose(
        forms.load.reshape(-1, 2).sum(axis=0), [mu.values.mean(), -2 * mu.values.mean()], atol=1e-12
    )
    assert_allclose(forms.mass.sum(), 1, atol=1e-12)
    free = tuple(int(f) for f in mesh.boundary_faces)
    with pytest.raises(ValueError, match="strongly fixed"):
        conforming_elasticity(mesh, free_faces=free)
    assert_allclose(volume_centroid(mesh), [0.5, 0.5])
    with pytest.raises(TypeError, match="quadrature"):
        volume_centroid(HexMesh.unit_cube())
