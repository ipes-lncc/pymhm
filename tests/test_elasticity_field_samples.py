"""Owned polynomial values, discontinuous line profiles and orthogonal displacement errors."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import CartesianMacroMesh
from pymhm._legacy.models.elasticity.stress_tensor import solve_elasticity_tensor_rt
from pymhm._legacy.models.geometry import solve_elasticity_mixed_polygons


@pytest.fixture
def helpers(monkeypatch):
    """Load original analytical display helpers without changing the package dependencies."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return (
        importlib.import_module("examples.core_extension_data"),
        importlib.import_module("examples.elasticity_field_samples"),
        importlib.import_module("examples.solve_core_extensions").polygon_grid,
    )


@pytest.mark.parametrize("rectangular", [False, True])
def test_polynomial_display_and_profiles_keep_their_physical_owner(rectangular, helpers):
    """Affine displacement and full anisotropic stress remain exact at all sampled coordinates."""
    data, api, polygon_grid = helpers
    mesh = CartesianMacroMesh(2) if rectangular else polygon_grid(1)
    solver = solve_elasticity_tensor_rt if rectangular else solve_elasticity_mixed_polygons

    def displacement(x):
        """Affine displacement with nonzero cross derivatives and dilation."""
        return np.column_stack((x[:, 0] + 2 * x[:, 1], 3 * x[:, 0] - x[:, 1]))

    expected_stress = np.linalg.solve(data.compliance(), np.array([1.0, 2.5, 2.5, -1.0]))
    with threadpool_limits(1):
        solution = solver(
            mesh,
            compliance=data.compliance(),
            dirichlet=displacement,
            local_refinement=1,
            quadrature_order=7,
        )
        samples = api.sample_elasticity_fields(solution, 3)
        assert_allclose(samples["actual"][:, :2], displacement(samples["points"]), atol=5e-12)
        assert_allclose(
            samples["actual"][:, 2:],
            np.broadcast_to(expected_stress, samples["actual"][:, 2:].shape),
            atol=2e-11,
        )
        for start, end in (([0.43, 0], [0.43, 1]), ([0, 0.37], [1, 0.37])):
            profile = api.sample_elasticity_profile(solution, start, end)
            assert_allclose(
                profile["actual"][..., :2],
                displacement(profile["points"].reshape(-1, 2)).reshape(
                    *profile["points"].shape[:2], 2
                ),
                atol=5e-12,
            )
            assert_allclose(profile["parameter"][1:, 0], profile["parameter"][:-1, -1], atol=3e-15)


def test_projection_identity_and_unjoined_broken_profiles(helpers):
    """Projection errors are orthogonal and real jumps survive profile sampling."""
    data, api, polygon_grid = helpers
    with threadpool_limits(1):
        solution = solve_elasticity_mixed_polygons(
            polygon_grid(2),
            compliance=data.compliance(),
            source=data.force,
            dirichlet=data.displacement,
            local_refinement=1,
            quadrature_order=8,
        )
        report, _ = api.projection_decomposition(solution, data.displacement, 8)
        assert np.max(abs(np.array(report["pythagorean_defect"]))) < 1e-18
        assert np.max(abs(np.array(report["orthogonality"]))) < 1e-18
        assert_allclose(
            np.linalg.norm(report["total_l2"]), solution.l2_error(data.displacement, 9), rtol=2e-13
        )
        profile = api.sample_elasticity_profile(solution, [0.43, 0], [0.43, 1])
        jump = profile["actual"][1:, 0, :2] - profile["actual"][:-1, -1, :2]
        assert abs(jump).max() > 1e-4
        fine = solution.local_meshes[0]
        edge = fine.faces[np.flatnonzero(fine.face_cells[:, 1] >= 0)[0]]
        with pytest.raises(ValueError, match="coincides"):
            api.sample_elasticity_profile(solution, *fine.points[edge])
        with pytest.raises(ValueError, match="complete"):
            api.sample_elasticity_profile(solution, [-0.1, 0.37], [1.1, 0.37])
