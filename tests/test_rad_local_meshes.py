"""Explicit material-fitted RAD spaces preserve strong residual and trace contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.cut_cells import fit_material_faces, fit_material_mesh
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.rad import solve_rad
from pymhm.reservoir import CartesianCellField
from pymhm.transport import solve_transport


def test_material_fitted_unusual_patch_and_transport_forwarding():
    """A tangential affine solution is exact on an explicitly resolved material jump."""
    mesh = TriangleMesh.unit_square()
    material = CartesianCellField(np.array([[0.1], [3.0]]), (0.5, 1.0))
    fine = tuple(fit_material_mesh(mesh.submesh(cell, 1), material) for cell in range(2))
    skeleton = fit_material_faces(SkeletonSpace(mesh), material)

    def exact(points):
        """Pressure varies tangentially to the interface and has nonzero boundary values."""
        return 1 - points[:, 1]

    def source(points):
        """The conservative diffusion term vanishes and reaction is two."""
        return 2 * exact(points)

    options = dict(
        diffusion=material,
        reaction=2.0,
        source=source,
        dirichlet=exact,
        skeleton=skeleton,
        local_meshes=fine,
        stabilization="unusual",
        degree=1,
    )
    result = solve_rad(mesh, **options)
    wrapper = solve_transport(mesh, **options)
    assert result.l2_error(exact) < 5e-14
    for supplied, actual, value, forwarded in zip(
        fine, result.local_meshes, result.values, wrapper.values, strict=True
    ):
        assert actual is supplied
        assert_allclose(value, exact(actual.points), atol=5e-14)
        assert_allclose(value, forwarded, atol=0, rtol=0)


def test_supplied_rad_meshes_must_partition_their_macrocells():
    """Incomplete or geometrically incompatible local collections fail before assembly."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="one local mesh"):
        solve_rad(mesh, local_meshes=(mesh.submesh(0, 2),))
    with pytest.raises(ValueError, match="cover its macro triangle"):
        solve_rad(mesh, local_meshes=(mesh.submesh(1, 2), mesh.submesh(0, 2)))
