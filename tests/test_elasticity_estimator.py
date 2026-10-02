"""Exact moments of the L17 face indicator and its explicit one-level limitation."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_elasticity
from pymhm.elasticity_estimator import estimate_primal_elasticity_error


def test_affine_tensor_patch_has_zero_face_indicator() -> None:
    """One-sided polynomial evaluation preserves the exact displacement trace."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces), 2)

    def exact(x: np.ndarray) -> np.ndarray:
        """Evaluate an affine displacement with nonzero normal and shear strain."""
        return np.column_stack((x[:, 0] + 2 * x[:, 1], -x[:, 0] + 3 * x[:, 1]))

    result = solve_elasticity(
        mesh, formulation="primal", degree=3, local_refinement=4, skeleton=skeleton, dirichlet=exact
    )
    indicator = estimate_primal_elasticity_error(
        result, dirichlet=exact, c_min=np.sqrt(2), full_dirichlet=True
    )
    assert indicator.eta < 2e-11
    assert all(len(values) == 2 for values in indicator.face_squared)


def test_original_macro_length_half_jump_and_interior_multiplicity() -> None:
    """For unit displacement on one macro, eta²=2 c_min²+2 c_min²/4."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_elasticity(mesh, formulation="primal", local_refinement=4, skeleton=skeleton)
    fields = tuple(
        np.column_stack((np.ones(len(v)) if cell == 0 else np.zeros(len(v)), np.zeros(len(v))))
        for cell, v in enumerate(result.values)
    )
    changed = replace(result, values=fields)
    indicator = estimate_primal_elasticity_error(changed, c_min=1.0, full_dirichlet=True)
    assert_allclose(indicator.eta**2, 2.5, atol=2e-14)


def test_zero_jump_does_not_certify_interior_discretization_error() -> None:
    """A bubble displacement can hide from the one-level face-only indicator."""
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_elasticity(
        mesh, formulation="primal", degree=3, local_refinement=1, skeleton=skeleton
    )
    from pymhm.lagrange import nodal_space

    _, points = nodal_space(result.local_meshes[0], 3)
    bubble = points[:, 0] * points[:, 1] * (1 - points.sum(axis=1))
    changed = replace(result, values=(np.column_stack((bubble, np.zeros(len(bubble)))),))
    indicator = estimate_primal_elasticity_error(changed, c_min=1.0, full_dirichlet=True)
    assert indicator.eta < 1e-15
    assert changed.l2_error((0.0, 0.0)) > 1e-3


@pytest.mark.parametrize(
    "options,match",
    [
        ({"full_dirichlet": False}, "Dirichlet"),
        ({"c_min": 0}, "c_min"),
        ({"c_min": np.inf}, "c_min"),
        ({"c_min": 3}, "ellipticity"),
        ({"quadrature_order": 1}, "quadrature_order"),
    ],
)
def test_indicator_contract(options: dict, match: str) -> None:
    """Require explicit physical boundary and material assumptions."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    result = solve_elasticity(
        mesh, formulation="primal", degree=3, local_refinement=1, skeleton=skeleton
    )
    arguments = dict(c_min=1.0, full_dirichlet=True)
    arguments.update(options)
    with pytest.raises(ValueError, match=match):
        estimate_primal_elasticity_error(result, **arguments)


def test_indicator_requires_rigid_motion_traces() -> None:
    """An injective P0 trace does not meet the separate L17 global Korn assumption."""
    result = solve_elasticity(TriangleMesh.unit_square(), formulation="primal", local_refinement=1)
    with pytest.raises(ValueError, match="rigid-motion traces"):
        estimate_primal_elasticity_error(result, c_min=1.0, full_dirichlet=True)
