"""Mixed stress families: exact polynomial fields, rigid moments and incompressibility."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import TriangleMesh
from pymhm.elasticity_mixed import solve_elasticity_mixed


def affine_displacement(points: np.ndarray) -> np.ndarray:
    """An affine displacement with nonzero strain and independent rigid rotation."""
    return np.column_stack((points[:, 0] + 2 * points[:, 1], 3 * points[:, 0] - points[:, 1]))


@pytest.mark.parametrize("degree,enrichment", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 0)])
def test_affine_elasticity_mixed_families(degree: int, enrichment: int) -> None:
    """Every family contains all rigid modes and reproduces the same affine Cauchy field."""
    solution = solve_elasticity_mixed(
        TriangleMesh.unit_square(),
        stress_degree=degree,
        enrichment=enrichment,
        local_refinement=1,
        quadrature_order=degree + enrichment + 2,
        dirichlet=affine_displacement,
    )
    assert solution.l2_error(affine_displacement) < 2e-10
    assert solution.stress_l2_error([[2, 5], [5, -2]]) < 2e-10
    assert solution.rotation_l2_error(-0.5) < 2e-10
    assert solution.divergence_l2_error((0, 0)) < 5e-9
    for residuals in (
        solution.fine_force_residuals(),
        solution.weak_symmetry_residuals(),
        solution.normal_traction_residuals(),
    ):
        assert max(np.max(np.abs(value)) for value in residuals) < 2e-10
    assert_allclose(solution.equilibrium_residuals(), 0, atol=2e-10)


def quadratic_displacement(points: np.ndarray) -> np.ndarray:
    """A nonaffine divergence-free field representable by P2 displacement."""
    x, y = points.T
    return np.column_stack((x * x, -2 * x * y))


def quadratic_stress(points: np.ndarray) -> np.ndarray:
    """Cauchy stress 2 eps(u), with zero hydrostatic mean and trace."""
    x, y = points.T
    return np.array([[4 * x, -2 * y], [-2 * y, -4 * x]]).transpose(2, 0, 1)


@pytest.mark.parametrize("degree,enrichment", [(1, 2), (2, 1), (3, 0)])
@pytest.mark.parametrize("lame_lambda", [2.0, 1e8, np.inf])
def test_quadratic_patch_to_incompressible_limit(
    degree: int, enrichment: int, lame_lambda: float
) -> None:
    """Quadratic u and affine stress stay exact for large and infinite bulk modulus."""
    solution = solve_elasticity_mixed(
        TriangleMesh.unit_square(),
        stress_degree=degree,
        enrichment=enrichment,
        local_refinement=1,
        quadrature_order=5,
        lame_lambda=lame_lambda,
        dirichlet=quadratic_displacement,
        source=(-2.0, 0.0),
    )
    assert solution.l2_error(quadratic_displacement) < 2e-10
    assert solution.stress_l2_error(quadratic_stress) < 2e-10
    assert solution.rotation_l2_error(lambda x: x[:, 1]) < 2e-10
    assert solution.divergence_l2_error((2.0, 0.0)) < 2e-10


def test_enriched_family_pure_traction_rigid_gauge() -> None:
    """Physical force/moment balance and the three displacement gauges remain unchanged."""
    mesh = TriangleMesh.unit_square()
    stress = np.array([[2.0, 5.0], [5.0, -2.0]])
    traction = {int(face): stress @ mesh.normals[face] for face in mesh.boundary_faces}
    # u=(x+2y,3x-y); moments against (1,0),(0,1),(-(y-.5),x-.5).
    solution = solve_elasticity_mixed(
        mesh,
        stress_degree=1,
        enrichment=1,
        traction=traction,
        local_refinement=2,
        rigid_moments=(1.5, 1.0, 1 / 12),
    )
    assert solution.l2_error(affine_displacement) < 2e-10
    assert_allclose(solution.equilibrium_residuals(), 0, atol=2e-10)


def test_bdm1_unenriched_cannot_hold_local_rigid_rotation() -> None:
    """Global AFW stability is insufficient for the local MHM rigid-motion contract."""
    with pytest.raises(ValueError, match="rigid rotations"):
        solve_elasticity_mixed(TriangleMesh.unit_square(), stress_degree=1)


def test_high_degree_requires_sufficient_assembly_quadrature() -> None:
    """Do not silently underintegrate an enriched polynomial compliance matrix."""
    with pytest.raises(ValueError, match="quadrature_order"):
        solve_elasticity_mixed(TriangleMesh.unit_square(), stress_degree=3, enrichment=2)
