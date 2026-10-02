"""Canonical RT moments use physical incident-side material values on interfaces."""

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.darcy import solve_darcy
from pymhm.mesh import TriangleMesh
from pymhm.planar_fitting import fit_planar_material, fit_planar_skeleton
from pymhm.planar_material import PlanarMaterial, PlanarRegion
from pymhm.reconstruction_moments import reconstruct_darcy_moments


def test_fitted_planar_interface_preserves_exact_physical_flux() -> None:
    """Closed halfspace evaluation cannot replace the two incident traces in RT moments."""
    mesh = TriangleMesh.unit_square()
    normal = np.array([1.0, 0.4])
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])
    material = PlanarMaterial(tensor, (PlanarRegion([normal], [0.63], 25 * tensor),))
    skeleton = fit_planar_skeleton(mesh, material)
    locals_ = tuple(
        fit_planar_material(mesh.submesh(c, 1), material).mesh for c in range(len(mesh.cells))
    )

    def pressure(points):
        """Continuous piecewise-linear pressure with a discontinuous physical gradient."""
        coordinate = points @ normal - 0.63
        return coordinate / np.where(coordinate <= 0, 25.0, 1.0)

    with threadpool_limits(1):
        solution = solve_darcy(
            mesh,
            degree=2,
            skeleton=skeleton,
            local_meshes=locals_,
            permeability=material,
            dirichlet=pressure,
        )
        assert solution.l2_error(pressure) < 1e-12
        recovered = reconstruct_darcy_moments(solution, degree=1)
        assert recovered.flux_l2_error(-tensor @ normal) < 1e-11
