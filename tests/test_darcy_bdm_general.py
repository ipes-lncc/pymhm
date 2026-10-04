"""Complete BDM and interior-enriched Darcy spaces with physical boundary and moment checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import BDMFamily, FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.mixed_bdm import solve_darcy_bdm


def pressure(x):
    """Nonaffine physical pressure with mean 5/4 on the unit square."""
    return x[:, 0] ** 2 + x[:, 0] * x[:, 1] + 2 * x[:, 1] ** 2


def flux(x):
    """Exact Darcy flux for the fixed anisotropic tensor."""
    return -np.column_stack((2 * x[:, 0] + x[:, 1], x[:, 0] + 4 * x[:, 1])) @ np.array(
        [[2.0, 0.2], [0.2, 1.0]]
    )


@pytest.mark.parametrize("degree,enrichment", [(1, 0), (2, 1), (3, 0), (4, 0), (1, 3)])
@pytest.mark.parametrize("natural", [False, True])
def test_quadratic_pressure_projection_and_exact_physical_flux(degree, enrichment, natural):
    """Local bubbles enrich pressure independently of normal degree, with correct Neumann gauge."""
    mesh = TriangleMesh.unit_square()
    boundary = (
        {int(f): lambda x, f=f: flux(x) @ mesh.normals[f] for f in mesh.boundary_faces}
        if natural
        else None
    )
    with threadpool_limits(1):
        result = solve_darcy_bdm(
            mesh,
            degree=degree,
            enrichment=enrichment,
            permeability=[[2.0, 0.2], [0.2, 1.0]],
            dirichlet=pressure,
            source=-8.4,
            neumann=boundary,
            mean_pressure=1.25 if natural else 0.0,
            local_refinement=1,
        )
        assert result.family == BDMFamily(degree, enrichment)
        assert result.flux_l2_error(flux, 7) < 3e-10
        assert result.divergence_l2_error(-8.4, 7) < 3e-10
        if degree + enrichment >= 3:
            assert result.l2_error(pressure, 7) < 3e-11
        else:
            assert result.l2_error(pressure, 7) > 1e-3
        assert_allclose(result.conservation_residuals(), 0, atol=1e-11)
        for residual in (
            *result.fine_equilibrium_residuals(),
            *result.fine_conservation_residuals(),
            *result.normal_flux_residuals(),
        ):
            assert_allclose(residual, 0, atol=2e-10)


def test_cubic_flux_needs_cubic_normal_trace():
    """BDM3/DG2 reproduces a cubic flux when the independently chosen trace contains it."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(3) for _ in mesh.faces))
    with threadpool_limits(1):
        result = solve_darcy_bdm(
            mesh,
            degree=3,
            skeleton=skeleton,
            local_refinement=1,
            source=lambda x: -12 * (x[:, 0] ** 2 + x[:, 1] ** 2),
            dirichlet=lambda x: (x**4).sum(axis=1),
        )
        assert result.flux_l2_error(lambda x: -4 * x**3, 7) < 3e-11
        assert result.l2_error(lambda x: (x**4).sum(axis=1), 7) > 1e-3
    incompatible = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    with pytest.raises(ValueError, match="trace degrees"):
        solve_darcy_bdm(mesh, degree=1, skeleton=incompatible)
