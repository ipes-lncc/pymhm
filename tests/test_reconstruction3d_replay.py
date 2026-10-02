"""Physical replay and one-sided section/profile ownership for documented tetrahedral fields."""

import importlib
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import TetraMesh, TriangularSkeleton, reconstruct_darcy_moments_3d, solve_darcy_3d


def test_quadratic_physical_replay_on_distinct_tetrahedral_orientations(monkeypatch):
    """Stored coefficients recover exact physical polynomials, including profile endpoints."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    replay = importlib.import_module("examples.reconstruction3d_replay")
    mesh = TetraMesh.unit_cube()

    def exact(x):
        """A quadratic pressure with a genuinely three-component affine physical flux."""
        return x[:, 0] ** 2 + 2 * x[:, 1] ** 2 + 3 * x[:, 2] ** 2 + x[:, 0] * x[:, 2]

    def expected(x):
        """Independent physical derivatives, with the Darcy minus sign."""
        return np.column_stack(
            (exact(x), -2 * x[:, 0] - x[:, 2], -4 * x[:, 1], -6 * x[:, 2] - x[:, 0])
        )

    with threadpool_limits(1):
        solution = solve_darcy_3d(
            mesh,
            skeleton=TriangularSkeleton(mesh, degree=1),
            degree=2,
            local_refinement=2,
            dirichlet=exact,
            source=-12,
        )
        reconstruction = reconstruct_darcy_moments_3d(solution, degree=1)
        archive = dict(
            macro_points=mesh.points,
            macro_cells=mesh.cells,
            local_degree=2,
            reconstruction_degree=1,
            rt_basis=reconstruction.family.coefficients.copy(),
        )
        for cell, fine in enumerate(solution.local_meshes):
            archive.update(
                {
                    f"points_{cell}": fine.points,
                    f"cells_{cell}": fine.cells,
                    f"pressure_{cell}": solution.pressure[cell].copy(),
                    f"rt_flux_{cell}": reconstruction.flux[cell].copy(),
                }
            )
            owners = np.arange(len(fine.cells))
            bary = np.tile([0.17, 0.23, 0.29, 0.31], (len(owners), 1))
            physical = np.einsum("qi,qij->qj", bary, fine.points[fine.cells])
            assert_allclose(
                replay.evaluate(archive, cell, owners, bary), expected(physical), atol=3e-12
            )
        first, last = np.array([0.0, 0.413, 0.37]), np.array([1.0, 0.413, 0.37])
        sampled = replay.profile(archive, first, last)
        assert_allclose(
            sampled["actual"].reshape(-1, 4), expected(sampled["points"].reshape(-1, 3)), atol=3e-12
        )
        assert_allclose(np.sum(np.diff(sampled["parameter"][:, [0, -1]])), 1, atol=2e-15)
        assert len(sampled["macro_breaks"]) > 2
        assert not len(
            replay.line_intervals(mesh, np.array([2.0, 2.0, 2.0]), np.array([3.0, 2.0, 2.0]))
        )
