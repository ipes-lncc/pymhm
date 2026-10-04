"""Physical incompressibility compatibility is invariant under displacement rescaling."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.elasticity.boundary import require_compatible_displacement_flux
from pymhm._legacy.models.elasticity.mixed_pressure import solve_displacement_pressure
from pymhm._legacy.models.elasticity.mixed_pressure_3d import solve_elasticity_gals_3d
from pymhm._legacy.models.elasticity.stress import solve_elasticity_mixed
from pymhm._legacy.models.elasticity.stress_3d import solve_elasticity_mixed_3d
from pymhm._legacy.models.elasticity.stress_tensor import solve_elasticity_tensor_rt
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.meshes.mixed import AffineMixedMesh
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("method", ["gals2", "stress2", "tensor2", "gals3", "stress3"])
def test_incompatible_boundary_volume_at_every_amplitude(scale, method):
    """A nonzero uniform dilation cannot be hidden by an absolute unit-sized threshold."""

    def dilation(points):
        """Prescribe positive constant divergence at the requested physical amplitude."""
        return scale * points

    common = dict(lame_lambda=np.inf, dirichlet=dilation)
    with pytest.raises(ValueError, match="incompatible incompressible"):
        if method == "gals2":
            solve_displacement_pressure(TriangleMesh.unit_square(), **common)
        elif method == "stress2":
            solve_elasticity_mixed(TriangleMesh.unit_square(), local_refinement=1, **common)
        elif method == "tensor2":
            solve_elasticity_tensor_rt(CartesianMacroMesh(1), local_refinement=1, **common)
        elif method == "gals3":
            solve_elasticity_gals_3d(TetraMesh.unit_cube(), degree=2, local_refinement=2, **common)
        else:
            solve_elasticity_mixed_3d(AffineMixedMesh.unit_cube(), local_refinement=1, **common)


@pytest.mark.parametrize("scale", [1e-20, 1.0, 1e20])
def test_moment_compatibility_preserves_cancellation_at_every_amplitude(scale):
    """Compatible roundoff stays admissible while resolved net flux is rejected relatively."""
    require_compatible_displacement_flux(0.0, 0.0)
    require_compatible_displacement_flux(3 * np.finfo(float).eps * scale, scale)
    with pytest.raises(ValueError, match="incompatible"):
        require_compatible_displacement_flux(1e-8 * scale, scale)


def test_small_compatible_shear_is_not_removed_by_pressure_gauge():
    """The mixed 3D solver preserves a genuinely nonzero displacement below unit scale."""
    scale = 1e-12

    def displacement(points):
        """Affine shear with exactly zero volume change."""
        return scale * np.column_stack((points[:, 1], np.zeros((len(points), 2))))

    solution = solve_elasticity_mixed_3d(
        AffineMixedMesh.unit_cube(),
        lame_lambda=np.inf,
        dirichlet=displacement,
        local_refinement=1,
    )
    points = np.array([[0.2, 0.3, 0.1]])
    for cell, fine in enumerate(solution.local_meshes):
        actual = solution.evaluate(cell, points)[0]
        expected = displacement(fine.geometry(points).reshape(-1, 3)).reshape(actual.shape)
        assert_allclose(actual / scale, expected / scale, atol=1e-12, rtol=0)
