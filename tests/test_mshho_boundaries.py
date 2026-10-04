"""Physical Neumann loads and pressure gauges for cell/face-moment MsHHO."""

import numpy as np
import pytest

from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.hho import solve_mshho


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("cell_degree", [-1, 0, 1])
def test_affine_mixed_boundary_and_volume_mean(pure, cell_degree):
    """Natural data use outward Darcy flux, and the all-Neumann gauge is a volume mean."""
    mesh = TriangleMesh.unit_square()
    faces = mesh.boundary_faces if pure else mesh.boundary_faces[:1]
    flux = {int(f): np.array([-1, -2]) @ mesh.normals[f] for f in faces}

    def exact(x):
        """Return a pressure with a nonzero volume mean."""
        return 1 + x[:, 0] + 2 * x[:, 1]

    result = solve_mshho(
        mesh,
        neumann=flux,
        dirichlet=exact,
        mean_pressure=2.5 if pure else 0,
        cell_degree=cell_degree,
        source_variant="reconstructed",
        local_refinement=2,
    )
    assert result.l2_error(exact) < 2e-12
    assert result.flux_l2_error([-1, -2]) < 2e-11


def test_neumann_compatibility_and_boundary_validation():
    """Reject inconsistent source/flux totals and invalid or meaningless gauge data."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="incompatible"):
        solve_mshho(mesh, source=1, neumann={int(f): 0 for f in mesh.boundary_faces})
    with pytest.raises(ValueError, match="exterior"):
        solve_mshho(mesh, neumann={len(mesh.faces): 0})
    for value in (np.nan, 1):
        with pytest.raises(ValueError, match="mean_pressure"):
            solve_mshho(mesh, mean_pressure=value)
