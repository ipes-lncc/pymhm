"""Energy scaling, mixed-boundary exactness and weighted-estimator contracts."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.estimator import estimate_darcy_error
from pymhm.reservoir import CartesianCellField
from pymhm.weighted_estimator import estimate_weighted_darcy_error, recover_dirichlet_potential


def sine(x):
    """Homogeneous-boundary nonpolynomial exact pressure."""
    return np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def gradient(x):
    """Analytical sine gradient."""
    a, b = np.pi * x.T
    return np.pi * np.column_stack((np.cos(a) * np.sin(b), np.sin(a) * np.cos(b)))


def scalar_problem(coefficient=1.0):
    """Small degree-two MHM with independently integrated forcing."""
    return solve_darcy(
        TriangleMesh.unit_square(),
        degree=2,
        local_refinement=2,
        permeability=coefficient,
        source=lambda x: coefficient * 2 * np.pi**2 * sine(x),
        quadrature_order=9,
    )


@pytest.mark.parametrize("coefficient", [1e-4, 1.0, 1e4])
def test_energy_scaling_and_identity_reduction(coefficient):
    """Changing units of diffusion scales every indicator by sqrt(A), not by A."""
    unit = scalar_problem()
    baseline = estimate_darcy_error(unit, homogeneous_dirichlet=True, quadrature_order=9)
    solution = scalar_problem(coefficient)
    result = estimate_weighted_darcy_error(solution, quadrature_order=9)
    for name in ("flux_defect", "nonconformity", "divergence_defect", "oscillation"):
        assert_allclose(
            getattr(result, name),
            np.sqrt(coefficient) * getattr(baseline, name),
            rtol=3e-9,
            atol=1e-11 * np.sqrt(coefficient),
        )
    assert result.total >= result.energy_error(gradient)
    assert_allclose(
        result.energy_error(gradient),
        np.sqrt(coefficient) * baseline.energy_error(gradient),
        rtol=2e-10,
    )


def test_anisotropic_mixed_nonzero_boundary_patch():
    """Represented physical Neumann data and nonzero Dirichlet lifting yield zero error."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    tensor = np.diag([2.0, 0.5])

    def pressure(x):
        """Cubic pressure."""
        return 1 + x[:, 0] + x[:, 1] + x[:, 0] ** 2 * x[:, 1]

    def grad(x):
        """Analytical cubic pressure gradient."""
        return np.column_stack((1 + 2 * x[:, 0] * x[:, 1], 1 + x[:, 0] ** 2))

    natural = {}
    for face in mesh.boundary_faces:
        midpoint = mesh.points[mesh.faces[face]].mean(axis=0)
        normal = mesh.normals[face]
        if np.isclose(midpoint.max(), 1):
            natural[int(face)] = lambda x, n=normal: -(grad(x) @ tensor.T) @ n
    solution = solve_darcy(
        mesh,
        skeleton=skeleton,
        degree=4,
        local_refinement=2,
        permeability=tensor,
        source=lambda x: -4 * x[:, 1],
        dirichlet=pressure,
        neumann=natural,
        quadrature_order=8,
    )
    estimator = estimate_weighted_darcy_error(
        solution, degree=2, dirichlet=pressure, neumann=natural, quadrature_order=8
    )
    assert estimator.total < 5e-11
    assert estimator.energy_error(grad) < 5e-11
    assert estimator.potential.l2_error(pressure) < 2e-12


def test_boundary_and_ellipticity_contracts():
    """Reject missing bounds, wrong certificates and omitted boundary mismatch terms."""
    solution = scalar_problem()
    with pytest.raises(ValueError, match="Neumann keys"):
        recover_dirichlet_potential(solution, neumann={999: 0.0})
    with pytest.raises(ValueError, match="Dirichlet data"):
        recover_dirichlet_potential(solution, dirichlet=lambda x: np.exp(x[:, 0] + x[:, 1]))
    face = int(solution.skeleton.mesh.boundary_faces[0])
    with pytest.raises(ValueError, match="Neumann data"):
        estimate_weighted_darcy_error(solution, neumann={face: 100.0})
    with pytest.raises(ValueError, match="certified"):
        estimate_weighted_darcy_error(replace(solution, permeability=lambda x: np.ones(len(x))))
    for bound in (-1, np.nan, [1, 2, 3], 1j):
        with pytest.raises(ValueError, match="bound"):
            estimate_weighted_darcy_error(solution, ellipticity_lower_bound=bound)
    with pytest.raises(ValueError, match="exceeds"):
        estimate_weighted_darcy_error(solution, ellipticity_lower_bound=2.0)
    callback = replace(solution, permeability=lambda x: np.ones(len(x)))
    assert estimate_weighted_darcy_error(callback, ellipticity_lower_bound=1.0).total > 0
    with pytest.raises(ValueError, match="k>=ell"):
        estimate_weighted_darcy_error(replace(solution, degree=1))
    with pytest.raises(ValueError, match="point wells"):
        estimate_weighted_darcy_error(
            replace(solution, point_sources=(np.array([[0.1, 0.2, 1.0]]),))
        )
    with pytest.raises(ValueError, match="continuous-test"):
        estimate_weighted_darcy_error(replace(solution, source=0.0))


@pytest.mark.parametrize("tensor", [False, True])
def test_aligned_material_jumps_preserve_one_sided_flux_and_bound(tensor):
    """A tangential flux jump is integrated without replacing either material limit."""
    mesh = TriangleMesh.unit_square()
    values = np.array([[1.0], [7.0]])
    if tensor:
        values = values[..., None, None] * np.diag([2.0, 1.0])
    material = CartesianCellField(values, (0.5, 1.0))
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 2) for _ in mesh.faces))
    solution = solve_darcy(
        mesh,
        skeleton=skeleton,
        degree=2,
        local_refinement=2,
        permeability=material,
        dirichlet=lambda x: x[:, 1],
        quadrature_order=6,
    )
    result = estimate_weighted_darcy_error(solution, degree=1, dirichlet=lambda x: x[:, 1])
    assert result.total < 2e-11
    assert result.energy_error((0.0, 1.0)) < 2e-11
    assert_allclose(result.ellipticity_lower_bounds, 1.0)


def test_degree_zero_projection_and_zero_source():
    """Constant continuous projection remains a single macro moment for m=0."""
    mesh = TriangleMesh.unit_square()
    solution = solve_darcy(mesh, degree=2, dirichlet=lambda x: x[:, 0])
    result = estimate_weighted_darcy_error(solution, degree=0, dirichlet=lambda x: x[:, 0])
    assert result.total < 2e-12


def test_polygon_cannot_reuse_the_triangular_reliability_constant():
    """A valid polygon Darcy solution does not satisfy the estimator geometry contract."""
    from pymhm.polygon import PolygonMesh, solve_darcy_polygons

    mesh = PolygonMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), (np.arange(4),))
    solution = solve_darcy_polygons(mesh, degree=2, dirichlet=1.0)
    with pytest.raises(ValueError, match="triangular macrocells"):
        estimate_weighted_darcy_error(solution, dirichlet=1.0)
