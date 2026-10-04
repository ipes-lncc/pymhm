"""Nested local energy differences independently checked at fixed skeleton data."""

from dataclasses import replace

import numpy as np
import pytest

from pymhm import TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.estimators.darcy_local import estimate_darcy_local_refinement


def test_affine_pressure_has_zero_local_refinement_defect():
    """Every enriched local Neumann solve has the same exact affine gradient."""
    mesh = TriangleMesh.unit_square()
    solution = solve_darcy(mesh, degree=2, dirichlet=lambda x: 1 + x @ [2, 3])
    estimate = estimate_darcy_local_refinement(solution, quadrature_order=5)
    assert estimate.total < 2e-13
    assert len(estimate.local_meshes) == len(mesh.cells)
    assert all(
        len(new.cells) == 4 * len(old.cells)
        for old, new in zip(solution.local_meshes, estimate.local_meshes, strict=True)
    )


def test_nonpolynomial_local_energy_difference_reduces_with_refinement():
    """Measure convergence of local lifts for a smooth nonpolynomial source."""
    mesh = TriangleMesh.unit_square()

    def source(points):
        """Use nonpolynomial data without a pure-Neumann compatibility cancellation."""
        return np.exp(points.sum(axis=1))

    differences = []
    for refinement in (2, 4):
        solution = solve_darcy(
            mesh,
            degree=2,
            source=source,
            local_refinement=refinement,
            quadrature_order=8,
        )
        estimate = estimate_darcy_local_refinement(solution)
        assert estimate.total**2 == pytest.approx(estimate.local_squared.sum())
        differences.append(estimate.total)
    assert 0 < differences[1] < 0.5 * differences[0]


def test_local_refinement_contract_rejects_nonenergy_problems():
    """Point wells and mixed P0 pressures do not satisfy the local H1 comparison contract."""
    mesh = TriangleMesh.unit_square()
    solution = solve_darcy(mesh, degree=2)
    with pytest.raises(ValueError, match="primal Darcy"):
        estimate_darcy_local_refinement(replace(solution, formulation="mixed"))
    with pytest.raises(ValueError, match="L2 source"):
        estimate_darcy_local_refinement(replace(solution, point_sources=(np.ones((1, 3)),) * 2))
    with pytest.raises(ValueError, match="quadrature_order"):
        estimate_darcy_local_refinement(solution, quadrature_order=2)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_local_energy_workers_preserve_order_and_fields(backend):
    """Spawned or threaded factors reconstruct the same physical local pressures."""
    solution = solve_darcy(TriangleMesh.unit_square(), degree=2, source=1.0)
    serial = estimate_darcy_local_refinement(solution)
    parallel = estimate_darcy_local_refinement(solution, backend=backend, workers=2)
    np.testing.assert_allclose(serial.local_squared, parallel.local_squared, rtol=1e-13)
    for first, second in zip(serial.pressure, parallel.pressure, strict=True):
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-14)
