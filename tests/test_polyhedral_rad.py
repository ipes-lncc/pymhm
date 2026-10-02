"""Physical RAD patches, gauges and mixed data on native convex polyhedral macrocells."""

import numpy as np
import pytest

from pymhm.polyhedral import PolyhedralMesh
from pymhm.polyhedral_rad import PolygonalSkeleton3D, solve_polyhedral_rad
from pymhm.solvers import LinearSolveError


def exact(points: np.ndarray) -> np.ndarray:
    """Evaluate an affine nonhomogeneous scalar field."""
    return 1 + points[:, 0] + 2 * points[:, 1] - points[:, 2]


def forcing(points: np.ndarray) -> np.ndarray:
    """Apply conservative constant-velocity RAD to the affine field."""
    return 0.7 + 0.7 * exact(points)


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
@pytest.mark.parametrize("coarse_space", ["constants", "kernel"])
def test_affine_rad_and_complete_physical_flux(stabilization: str, coarse_space: str) -> None:
    """Match the complete source, weak boundary, mean and physical-flux conventions."""
    mesh = PolyhedralMesh.cubes()
    beta = np.array([0.2, 0.3, 0.1])
    result = solve_polyhedral_rad(
        mesh,
        degree=3,
        diffusion=0.7,
        velocity=beta,
        reaction=0.7,
        source=forcing,
        dirichlet=exact,
        stabilization=stabilization,
        coarse_space=coarse_space,
    )
    assert result.l2_error(exact) < 2e-12
    assert result.h1_seminorm_error((1, 2, -1)) < 2e-11
    assert (
        result.flux_l2_error(lambda p: -0.7 * np.array([1, 2, -1]) + exact(p)[:, None] * beta)
        < 2e-11
    )
    assert result.degree == 3
    assert result.evaluate(0, np.array([[0.25] * 4]))[0].shape[0] == 12


def test_variable_coefficients_full_supg_residual() -> None:
    """Verify coefficient divergence in diffusion and conservative transport independently."""
    result = solve_polyhedral_rad(
        PolyhedralMesh.cubes(),
        degree=3,
        diffusion=lambda p: 1 + p[:, 0],
        diffusion_divergence=(1, 0, 0),
        velocity=lambda p: np.column_stack((p[:, 0], np.zeros((len(p), 2)))),
        velocity_divergence=1,
        reaction=0.3,
        source=lambda p: -1 + p[:, 0] + 1.3 * exact(p),
        dirichlet=exact,
        stabilization="supg",
    )
    assert result.l2_error(exact) < 3e-12


def test_physical_neumann_projection_and_mean_gauge() -> None:
    """Use prescribed outward flux on every original polygonal face and a physical mean."""
    mesh = PolyhedralMesh.cubes()
    prescribed = {int(f): np.array([-1, -2, 1]) @ mesh.normals[f] for f in mesh.boundary_faces}
    result = solve_polyhedral_rad(mesh, degree=3, neumann=prescribed, mean_value=2)
    assert result.l2_error(exact) < 2e-12
    # A single prescribed side and weak Dirichlet on the others have no pressure gauge.
    face = int(mesh.boundary_faces[0])
    result = solve_polyhedral_rad(mesh, degree=3, neumann={face: prescribed[face]}, dirichlet=exact)
    assert result.l2_error(exact) < 2e-12


def tangent(points: np.ndarray) -> np.ndarray:
    """Return a nonzero divergence-free velocity tangent to the entire cube exterior."""
    x, y, _ = points.T
    return np.column_stack(
        (x * (1 - x) * (1 - 2 * y), -y * (1 - y) * (1 - 2 * x), np.zeros(len(points)))
    )


def test_tangential_transport_kernel_and_small_reaction_limit() -> None:
    """Keep structural null modes and preserve small positive reaction without a rank cutoff."""
    mesh = PolyhedralMesh.cubes()
    result = solve_polyhedral_rad(
        mesh,
        degree=3,
        velocity=tangent,
        velocity_divergence=0,
        neumann=dict.fromkeys(map(int, mesh.boundary_faces), 0.0),
        mean_value=2,
        coarse_space="kernel",
    )
    assert result.l2_error(2) < 3e-12
    result = solve_polyhedral_rad(mesh, degree=3, reaction=1e-12, source=2e-12, dirichlet=2)
    assert result.l2_error(2) < 2e-12


def test_invalid_gauges_and_discrete_instability_are_explicit() -> None:
    """Do not silently remove physical incompatibility or invisible skeleton modes."""
    mesh = PolyhedralMesh.cubes()
    natural = dict.fromkeys(map(int, mesh.boundary_faces), 0.0)
    with pytest.raises(ValueError, match="mean_value"):
        solve_polyhedral_rad(mesh, degree=3, reaction=1, neumann=natural, mean_value=1)
    with pytest.raises(ValueError, match="incompatible|compatibility"):
        solve_polyhedral_rad(mesh, degree=3, source=1, neumann=natural)
    with pytest.raises(LinearSolveError, match="rank"):
        solve_polyhedral_rad(mesh, degree=2, dirichlet=exact)


def test_zero_degree_traces_and_parallel_factory() -> None:
    """Use one P0 mode per polygonal face and compare the actual spawn-worker field."""
    mesh = PolyhedralMesh.cubes()
    options = dict(
        degree=2, local_refinement=2, skeleton=PolygonalSkeleton3D(mesh, 0), dirichlet=exact
    )
    serial = solve_polyhedral_rad(mesh, **options)
    parallel = solve_polyhedral_rad(mesh, **options, backend="process", workers=2)
    np.testing.assert_allclose(serial.values[0], parallel.values[0], atol=2e-13)
    assert serial.l2_error(exact) < 2e-12


def test_invalid_solver_arguments() -> None:
    """Validate geometry, gauge values, spaces and natural data without changing equations."""
    mesh = PolyhedralMesh.cubes()
    with pytest.raises(TypeError, match="PolyhedralMesh"):
        solve_polyhedral_rad(None)
    with pytest.raises(ValueError, match="coarse_space"):
        solve_polyhedral_rad(mesh, coarse_space="x")
    with pytest.raises(ValueError, match="finite"):
        solve_polyhedral_rad(mesh, mean_value=np.nan)
    with pytest.raises(ValueError, match="belong"):
        solve_polyhedral_rad(mesh, skeleton=PolygonalSkeleton3D(PolyhedralMesh.cubes()))
    with pytest.raises(ValueError, match="exterior"):
        solve_polyhedral_rad(mesh, neumann={99: 0.0})


def test_selective_non_tangential_advection_has_no_local_constant_kernel() -> None:
    """Drop an invertible local constant mode only under the explicit selective policy."""
    result = solve_polyhedral_rad(
        PolyhedralMesh.cubes(),
        degree=3,
        velocity=(1, 0, 0),
        source=1,
        dirichlet=exact,
        coarse_space="kernel",
    )
    assert result.l2_error(exact) < 3e-12
    assert not result.hybrid.coarse[0].size
