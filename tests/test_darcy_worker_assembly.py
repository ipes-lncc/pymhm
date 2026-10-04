"""Local assembly in native workers preserves the original Darcy discrete equations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.meshes.triangle import TriangleMesh


def quadratic_pressure(points):
    """Provide picklable non-affine boundary data for both local formulations."""
    return points[:, 0] ** 2 + points[:, 1] ** 2


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_assembled_worker_fields_and_boundary_loads(formulation, backend):
    """All physical fields coincide when only the owner of local assembly changes."""
    mesh = TriangleMesh.unit_square(2)
    parameters = dict(
        source=-4.0,
        dirichlet=quadratic_pressure,
        local_refinement=2,
        degree=2 if formulation == "primal" else 1,
        formulation=formulation,
    )
    original = solve_darcy(mesh, **parameters)
    parallel = solve_darcy(mesh, **parameters, backend=backend, workers=2, parallel_assembly=True)
    assert_allclose(parallel.hybrid.trace, original.hybrid.trace, rtol=2e-13, atol=1e-14)
    for actual, expected in zip(
        parallel.pressure + parallel.flux, original.pressure + original.flux, strict=True
    ):
        assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)
    assert np.max(abs(parallel.conservation_residuals())) < 2e-13


def test_worker_mean_gauge_and_supplied_geometry():
    """Worker metadata retain physical mean weights and custom fine partitions."""
    mesh = TriangleMesh.unit_square(2)
    local = tuple(mesh.submesh(cell, 2) for cell in range(len(mesh.cells)))
    solution = solve_darcy(
        mesh,
        neumann={int(face): 0.0 for face in mesh.boundary_faces},
        mean_pressure=3,
        local_meshes=local,
        parallel_assembly=True,
        backend="process",
        workers=2,
    )
    for values in solution.pressure:
        assert_allclose(values, 3, rtol=0, atol=2e-13)
    for actual, expected in zip(solution.local_meshes, local, strict=True):
        assert_allclose(actual.points, expected.points)
        assert np.array_equal(actual.cells, expected.cells)
