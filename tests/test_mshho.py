"""Verify moment unisolvence and independent primal/dual hybrid equivalence."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.mshho import solve_mshho


@pytest.mark.parametrize("face_degree,cell_degree", [(0, 0), (1, 0), (1, 1)])
@pytest.mark.parametrize("variant", ["projected", "reconstructed"])
def test_mshho_polynomial_source_equals_mhm(face_degree, cell_degree, variant):
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(face_degree) for _ in mesh.faces))
    source = (lambda x: 1 + x[:, 0] - 2 * x[:, 1]) if cell_degree else 2.0

    def boundary(x):
        return x[:, 0] - x[:, 1] + 0.4

    tensor = np.array([[3.0, 0.4], [0.4, 1.0]])
    options = dict(
        skeleton=skeleton,
        degree=2,
        local_refinement=3,
        source=source,
        dirichlet=boundary,
        permeability=tensor,
        quadrature_order=6,
    )
    primal = solve_mshho(mesh, cell_degree=cell_degree, source_variant=variant, **options)
    dual = solve_darcy(mesh, **options)
    for actual, expected, local in zip(primal.pressure, dual.pressure, primal.local, strict=True):
        assert_allclose(actual, expected, atol=3e-12)
        assert_allclose(
            local.moments.T @ local.reconstruction, np.eye(local.moments.shape[1]), atol=5e-13
        )
    free = skeleton.dofs(int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0]))
    assert np.linalg.eigvalsh(primal.matrix[free][:, free].toarray()).min() > 0
    assert primal.residual < 1e-12


@pytest.mark.parametrize("cell_degree", [-1, 0, 1])
def test_mshho_affine_anisotropic_patch_and_norms(cell_degree):
    mesh = TriangleMesh.unit_square()

    def exact(x):
        return 2 + x[:, 0] - 3 * x[:, 1]

    tensor = np.array([[2.0, 0.5], [0.5, 3.0]])
    result = solve_mshho(
        mesh,
        cell_degree=cell_degree,
        source_variant="reconstructed",
        dirichlet=exact,
        permeability=tensor,
        local_refinement=3,
    )
    assert result.l2_error(exact) < 1e-12
    assert result.flux_l2_error(-(tensor @ [1.0, -3.0])) < 2e-11


def test_face_only_variant_is_not_the_mhm_source_lifting():
    mesh = TriangleMesh.unit_square()
    result = solve_mshho(mesh, cell_degree=-1, source_variant="reconstructed", source=1)
    dual = solve_darcy(mesh, degree=2, source=1)
    assert (
        max(np.linalg.norm(a - b) for a, b in zip(result.pressure, dual.pressure, strict=True))
        > 1e-3
    )
    assert all(len(value) == 0 for value in result.cell_moments)


def test_projected_and_reconstructed_nonpolynomial_loads_differ():
    mesh = TriangleMesh.unit_square()
    options = dict(source=lambda x: np.exp(x[:, 0] + 2 * x[:, 1]), local_refinement=3)
    projected = solve_mshho(mesh, **options)
    reconstructed = solve_mshho(mesh, source_variant="reconstructed", **options)
    assert np.linalg.norm(projected.face_moments - reconstructed.face_moments) > 1e-4


@pytest.mark.parametrize(
    "options,match",
    [
        ({"cell_degree": -2}, "cell_degree"),
        ({"cell_degree": -1}, "m=-1"),
        ({"source_variant": "other"}, "source_variant"),
        ({"local_refinement": 1, "degree": 1}, "independent"),
    ],
)
def test_invalid_mshho_spaces(options, match):
    with pytest.raises(ValueError, match=match):
        solve_mshho(TriangleMesh.unit_square(), **options)


def test_scalar_skeleton_identity_is_required():
    mesh = TriangleMesh.unit_square()
    other = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="scalar skeleton"):
        solve_mshho(
            mesh, skeleton=SkeletonSpace(other, tuple(FaceSpace.uniform(0) for _ in other.faces))
        )
    with pytest.raises(ValueError, match="scalar skeleton"):
        solve_mshho(
            mesh, skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces), 2)
        )


def test_one_triangle_has_only_prescribed_face_moments():
    mesh = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    result = solve_mshho(mesh, dirichlet=2)
    assert result.l2_error(2) < 1e-12
    assert result.residual == 0
