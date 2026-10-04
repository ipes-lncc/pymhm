"""Physical mixed invariants and BDM assembly ownership across worker backends."""

import multiprocessing

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm._legacy.models.darcy._mixed import normal_flux_blocks
from pymhm._legacy.models.darcy.mixed_bdm import solve_darcy_bdm
from pymhm.meshes.triangle import TriangleMesh


def pressure(points):
    """Return an anisotropic Darcy polynomial with physical mean 5/4."""
    return points[:, 0] ** 2 + points[:, 0] * points[:, 1] + 2 * points[:, 1] ** 2


def flux(points):
    """Differentiate the prescribed pressure and apply the fixed SPD material."""
    return -np.column_stack((2 * points[:, 0] + points[:, 1], points[:, 0] + 4 * points[:, 1])) @ (
        np.array([[2.0, 0.2], [0.2, 1.0]])
    )


def test_mixed_constant_mode_and_original_normal_flux_equations():
    """The shared saddle retains joint pressure freedom and physical flux signs."""
    mass = sparse.diags([2.0, 3.0, 5.0], format="csc")
    divergence = sparse.csc_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]])
    source = np.array([0.6, 0.9])
    mapping = np.array([[1.0, -2.0], [-3.0, 4.0]])
    matrix, coupling, load = normal_flux_blocks(mass, divergence, source, np.array([0, 2]), mapping)
    constant_pressure = np.r_[np.zeros(3), np.ones(2), np.ones(2)]
    assert_allclose(matrix @ constant_pressure, 0, rtol=0, atol=0)
    values = np.array([0.25, -0.4, 1.2, 0.3, -0.2, 0.7, -0.5])
    trace = np.array([0.2, -0.1])
    defect = matrix @ values + coupling @ trace - load
    assert_allclose(defect[3:5], source - divergence @ values[:3], rtol=0, atol=0)
    assert_allclose(defect[5:], values[[0, 2]] - mapping @ trace, rtol=0, atol=0)
    shifted = values + 7.5 * constant_pressure
    assert_allclose(matrix @ shifted, matrix @ values, rtol=0, atol=2e-15)


@pytest.mark.parametrize("degree,enrichment", [(2, 0), (1, 2)])
@pytest.mark.parametrize("natural", [False, True])
@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_bdm_parallel_assembly_preserves_physical_fields_and_gauge(
    degree, enrichment, natural, backend
):
    """Workers preserve original spaces, fine balance, macrofaces and physical mean."""
    mesh = TriangleMesh.unit_square(2)
    boundary = (
        {
            int(face): lambda points, face=face: flux(points) @ mesh.normals[face]
            for face in mesh.boundary_faces
        }
        if natural
        else None
    )
    parameters = dict(
        permeability=[[2.0, 0.2], [0.2, 1.0]],
        source=-8.4,
        dirichlet=pressure,
        neumann=boundary,
        mean_pressure=1.25,
        degree=degree,
        enrichment=enrichment,
        local_refinement=1,
    )
    original = solve_darcy_bdm(mesh, **parameters)
    children = {process.pid for process in multiprocessing.active_children()}
    actual = solve_darcy_bdm(mesh, **parameters, parallel_assembly=True, backend=backend, workers=2)
    assert {process.pid for process in multiprocessing.active_children()} == children
    assert actual.family == original.family
    assert_allclose(actual.hybrid.trace, original.hybrid.trace, rtol=2e-13, atol=1e-14)
    for field, expected in zip(
        actual.pressure + actual.flux, original.pressure + original.flux, strict=True
    ):
        assert_allclose(field, expected, rtol=2e-13, atol=1e-14)
    assert actual.flux_l2_error(flux, 7) < 3e-10
    if degree + enrichment >= 3:
        assert actual.l2_error(pressure, 7) < 3e-11
    assert_allclose(actual.conservation_residuals(), 0, atol=1e-11)
    for residual in (*actual.fine_equilibrium_residuals(), *actual.normal_flux_residuals()):
        assert_allclose(residual, 0, atol=2e-10)


def test_bdm_default_process_condensation_still_accepts_source_closure():
    """Default assembly remains parent-owned even when condensation uses spawn."""
    mesh = TriangleMesh.unit_square()

    def source(points):
        """Represent a deliberately parent-owned, non-picklable local callback."""
        return -8.4 * np.ones(len(points))

    original = solve_darcy_bdm(mesh, source=source, dirichlet=pressure, local_refinement=1)
    actual = solve_darcy_bdm(
        mesh, source=source, dirichlet=pressure, local_refinement=1, backend="process", workers=2
    )
    assert_allclose(actual.hybrid.trace, original.hybrid.trace, rtol=2e-13, atol=1e-14)
    for field, expected in zip(
        actual.pressure + actual.flux, original.pressure + original.flux, strict=True
    ):
        assert_allclose(field, expected, rtol=2e-13, atol=1e-14)
