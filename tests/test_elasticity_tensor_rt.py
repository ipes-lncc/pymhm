"""Quadrilateral mixed-stress spaces, rigid gauges and the incompressible limit."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace
from pymhm._legacy.models.elasticity.stress_tensor import (
    _rotation_basis,
    solve_elasticity_tensor_rt,
)
from pymhm.meshes.cartesian import CartesianMacroMesh


def affine(points: np.ndarray) -> np.ndarray:
    """Evaluate an affine field with strain, shear and rigid rotation components."""
    x, y = points.T
    return np.column_stack((x + 2 * y, -x + 3 * y))


@pytest.mark.parametrize("degree,enrichment", [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_rectangular_family_affine_patch(degree: int, enrichment: int) -> None:
    """All supported tested families reproduce affine stress and physical balance moments."""
    result = solve_elasticity_tensor_rt(
        CartesianMacroMesh(2),
        degree=degree,
        enrichment=enrichment,
        local_refinement=1,
        dirichlet=affine,
    )
    errors = result.errors(affine, [[6, 1], [1, 10]], (0, 0), 1.5)
    assert max(errors.values()) < 2e-10
    assert np.max(np.abs(result.equilibrium_residuals())) < 2e-11
    assert max(np.max(np.abs(v)) for v in result.fine_force_residuals()) < 2e-11
    assert max(np.max(np.abs(v)) for v in result.weak_symmetry_residuals()) < 2e-11
    assert max(np.max(np.abs(v)) for v in result.normal_traction_residuals()) < 2e-11
    points = np.array([[0.2, 0.3], [0.6, 0.8]])
    for shared, cellwise in zip(
        result.evaluate(0, points), result.evaluate(0, points[None]), strict=True
    ):
        assert_allclose(shared, cellwise, atol=2e-13)


@pytest.mark.parametrize("lam", [1.0, 1e8, np.inf])
@pytest.mark.parametrize("degree,enrichment", [(1, 1), (2, 0)])
def test_quadratic_patch_and_incompressible_pressure_gauge(
    lam: float, degree: int, enrichment: int
) -> None:
    """Quadratic solenoidal displacement and affine stress survive finite and infinite lambda."""

    def displacement(points: np.ndarray) -> np.ndarray:
        """Return an exactly solenoidal quadratic field."""
        x, y = points.T
        return np.column_stack((x * x, -2 * x * y))

    def stress(points: np.ndarray) -> np.ndarray:
        """Return its stress, including a prescribed hydrostatic mean at infinity."""
        x, y = points.T
        sigma = np.zeros((len(x), 2, 2))
        sigma[:, 0, 0], sigma[:, 1, 1] = 4 * x, -4 * x
        sigma[:, 0, 1] = sigma[:, 1, 0] = -2 * y
        return sigma - (2.3 if np.isinf(lam) else 0) * np.eye(2)

    result = solve_elasticity_tensor_rt(
        CartesianMacroMesh(2),
        degree=degree,
        enrichment=enrichment,
        lame_lambda=lam,
        local_refinement=1,
        source=(-2.0, 0.0),
        dirichlet=displacement,
        mean_pressure=2.3 if np.isinf(lam) else 0.0,
    )
    assert max(result.errors(displacement, stress, (2.0, 0.0), lambda x: x[:, 1]).values()) < 5e-9


def test_pure_traction_rigid_moments() -> None:
    """All-Neumann data fixes three displacement moments without an independent rotation gauge."""
    mesh = CartesianMacroMesh(2)
    sigma = np.array([[6, 1], [1, 10]])
    result = solve_elasticity_tensor_rt(
        mesh,
        traction={int(f): sigma @ mesh.normals[f] for f in mesh.boundary_faces},
        rigid_moments=(1.5, 1.0, -0.25),
        local_refinement=1,
    )
    assert max(result.errors(affine, sigma, (0, 0), 1.5).values()) < 2e-10


def test_variable_moduli_and_partial_traction() -> None:
    """Material derivatives enter equilibrium and physical traction signs remain consistent."""
    mesh = CartesianMacroMesh(2)
    sigma = np.array([[6, 1], [1, 10]])
    face = int(mesh.boundary_faces[0])
    result = solve_elasticity_tensor_rt(
        mesh,
        lame_lambda=lambda x: 1 + x[:, 0],
        lame_mu=lambda x: 1 + x[:, 0],
        source=(-6.0, -1.0),
        dirichlet=affine,
        traction={face: lambda x: (1 + x[:, 0, None]) * (sigma @ mesh.normals[face])},
        local_refinement=2,
        quadrature_order=12,
    )
    errors = result.errors(
        affine, lambda x: (1 + x[:, 0, None, None]) * sigma, (6, 1), 1.5, order=12
    )
    assert max(errors.values()) < 3e-9


def test_rotation_space_is_total_degree_not_tensor_product() -> None:
    """P2 has six independent modes, excluding the higher cross terms present in Q2."""
    points = np.random.default_rng(8).uniform(size=(20, 2))
    basis = _rotation_basis(2, points)
    assert basis.shape == (20, 6)
    assert np.linalg.matrix_rank(basis) == 6


@pytest.mark.parametrize(
    "options,match",
    [
        ({"degree": 0}, "stress degree"),
        ({"quadrature_order": 2}, "quadrature_order"),
        ({"mean_pressure": np.nan}, "mean_pressure"),
        ({"mean_pressure": 1.0}, "incompressible limit"),
        ({"lame_lambda": np.inf, "dirichlet": affine}, "incompatible"),
        ({"lame_mu": 0.0}, "strictly positive"),
    ],
)
def test_invalid_configuration(options: dict, match: str) -> None:
    """Reject unsupported spaces, coefficient assumptions and physical gauge data."""
    with pytest.raises(ValueError, match=match):
        solve_elasticity_tensor_rt(CartesianMacroMesh(), **options)


@pytest.mark.parametrize("moments", [(0, 0), [0j, 0, 0], [0, np.nan, 0]])
def test_invalid_traction_gauge(moments) -> None:
    """All rigid constraints are finite real integrated moments."""
    mesh = CartesianMacroMesh()
    with pytest.raises(ValueError, match="rigid_moments"):
        solve_elasticity_tensor_rt(
            mesh, traction={int(f): (0, 0) for f in mesh.boundary_faces}, rigid_moments=moments
        )


@pytest.mark.parametrize("degree,segments", [(2, 1), (1, 3)])
def test_trace_compatibility(degree: int, segments: int) -> None:
    """A trace must fit the local normal polynomial space and fine-edge partition."""
    mesh = CartesianMacroMesh()
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(degree, segments) for _ in mesh.faces), 2
    )
    with pytest.raises(ValueError, match="RT trace"):
        solve_elasticity_tensor_rt(mesh, skeleton=skeleton, degree=1, local_refinement=2)


def test_skeleton_and_traction_pressure_contracts() -> None:
    """Boundary traction determines hydrostatic stress and requires a vector skeleton."""
    mesh = CartesianMacroMesh()
    with pytest.raises(ValueError, match="two-component"):
        solve_elasticity_tensor_rt(mesh, skeleton=SkeletonSpace(mesh))
    with pytest.raises(ValueError, match="full displacement"):
        solve_elasticity_tensor_rt(
            mesh, traction={int(mesh.boundary_faces[0]): (0, 0)}, mean_pressure=1.0
        )
    result = solve_elasticity_tensor_rt(mesh)
    for bad in (np.full((2, 2), np.nan), np.eye(2) * 1j):
        with pytest.raises(ValueError, match="exact stress"):
            result.errors((0, 0), bad, (0, 0), 0)


def test_cartesian_material_alignment_contract() -> None:
    """Declared discontinuous material uses aligned cells instead of aliased Gaussian sampling."""
    from pymhm.materials.cartesian import CartesianCellField

    coefficient = CartesianCellField(np.array([[1.0], [2.0]]), spacing=(0.5, 1.0))
    with pytest.raises(ValueError, match="must align"):
        solve_elasticity_tensor_rt(CartesianMacroMesh(), lame_mu=coefficient, local_refinement=1)
    result = solve_elasticity_tensor_rt(
        CartesianMacroMesh(), lame_mu=coefficient, local_refinement=2
    )
    assert result.hybrid.residual < 1e-12
