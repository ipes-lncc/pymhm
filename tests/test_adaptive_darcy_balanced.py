"""Light PDE checks for explicit local/macro refinement balance and budgets."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.adaptivity.darcy import solve_adaptive_darcy
from pymhm.adaptivity.darcy_balanced import solve_balanced_adaptive_darcy
from pymhm.estimators.darcy_local import estimate_darcy_local_refinement
from pymhm.fem.quadrature.material import fit_material_mesh
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


def oscillation(points):
    """A smooth source unresolved by a coarse continuous quadratic projection."""
    return np.sin(7 * points[:, 0] + 5 * points[:, 1])


def test_uniform_local_refinement_resolves_source_projection():
    """Retain the macro geometry while decreasing the measured unresolved local component."""
    mesh = TriangleMesh.unit_square()
    result = solve_balanced_adaptive_darcy(
        mesh,
        source=oscillation,
        iterations=2,
        local_refinement=2,
        estimator_order=5,
    )
    assert result.decisions == ("local", "stop")
    assert result.local_refinements == (2, 4)
    assert result.stop_reason == "iterations"
    assert all(solution.skeleton.mesh is mesh for solution in result.result.solutions)
    assert not result.result.refinements
    first, last = result.result.estimators
    assert np.linalg.norm(last.oscillation) < np.linalg.norm(first.oscillation)
    ordinary = solve_adaptive_darcy(
        mesh,
        source=oscillation,
        iterations=1,
        local_refinement=2,
        reconstruction_degree=2,
        estimator_order=5,
    )
    for a, b in zip(
        result.result.solutions[0].pressure, ordinary.solutions[0].pressure, strict=True
    ):
        assert_allclose(a, b, rtol=0, atol=0)


def test_macro_refinement_transfers_mixed_boundary_and_material_bounds():
    """Neumann side ancestry and certified local bounds survive a macro step."""
    mesh = TriangleMesh.unit_square()
    natural = {int(face): 0.0 for face in mesh.boundary_faces if mesh.normals[face, 0] > 0.5}
    result = solve_balanced_adaptive_darcy(
        mesh,
        source=1.0,
        iterations=2,
        local_refinement=2,
        estimator_order=5,
        local_error_ratio=1e3,
        neumann=natural,
        ellipticity_lower_bound=np.ones(2),
    )
    assert result.decisions == ("macro", "stop")
    assert len(result.result.solutions[-1].skeleton.mesh.cells) > 2
    assert_allclose(result.result.estimators[-1].ellipticity_lower_bounds, 1.0)
    assert max(abs(result.result.solutions[-1].conservation_residuals())) < 1e-12


def test_fitted_factory_and_explicit_limits():
    """Use interface-fitted partitions and report budget exhaustion without changing the policy."""
    mesh = TriangleMesh.unit_square()
    field = CartesianCellField(np.array([[1.0], [2.0]]), (0.5, 1.0))
    calls = []

    def fitted(coarse, cell, refinement):
        """Fit material interfaces while retaining the requested coarse cell."""
        calls.append(refinement)
        return fit_material_mesh(coarse.submesh(cell, refinement), field)

    zero = solve_balanced_adaptive_darcy(
        mesh, iterations=2, permeability=field, local_mesh_factory=fitted, estimator_order=5
    )
    assert zero.stop_reason == "tolerance"
    assert calls == [1, 1]
    local = solve_balanced_adaptive_darcy(
        mesh,
        source=oscillation,
        iterations=2,
        local_refinement=2,
        maximum_local_refinement=2,
        estimator_order=5,
    )
    assert local.stop_reason == "local_refinement_limit"
    assert len(local.result.solutions) == 1
    macro = solve_balanced_adaptive_darcy(
        mesh,
        source=1,
        iterations=2,
        local_refinement=2,
        maximum_cells=2,
        local_error_ratio=1e3,
        estimator_order=5,
    )
    assert macro.stop_reason == "maximum_cells"
    assert len(macro.result.solutions) == 1
    # The no-Neumann/no-bound macro path is a separate physical boundary case.
    free = solve_balanced_adaptive_darcy(
        mesh, source=1, iterations=2, local_refinement=2, estimator_order=5
    )
    assert free.decisions[0] == "macro"


@pytest.mark.parametrize(
    "parameters",
    [
        {"maximum_local_refinement": 1, "local_refinement": 2},
        {"local_error_ratio": 0},
        {"local_error_ratio": 1j},
        {"local_error_ratio": np.inf},
        {"iterations": 0},
    ],
)
def test_invalid_refinement_contracts(parameters):
    """Reject invalid budgets and nonphysical comparison scales before assembly."""
    with pytest.raises(ValueError):
        solve_balanced_adaptive_darcy(TriangleMesh.unit_square(), **parameters)


def test_nested_energy_indicator_controls_the_declared_policy():
    """Use an actual nested local solve and preserve its reported energy difference."""
    result = solve_balanced_adaptive_darcy(
        TriangleMesh.unit_square(),
        source=1,
        iterations=2,
        local_refinement=2,
        estimator_order=5,
        local_error_indicator=lambda solution: estimate_darcy_local_refinement(solution).total,
    )
    assert result.decisions == ("macro", "stop")
    assert len(result.local_indicators) == 2
    assert all(value >= 0 for value in result.local_indicators)
    with pytest.raises(ValueError, match="local_error_indicator"):
        solve_balanced_adaptive_darcy(
            TriangleMesh.unit_square(),
            iterations=1,
            estimator_order=5,
            local_error_indicator=lambda solution: -1.0,
        )


def test_checkpoint_callback_receives_only_validated_completed_states():
    """A callback archives the first state even if later work is interrupted."""
    saved = []

    def checkpoint(state):
        """Retain accumulated states and deliberately stop after the second solve."""
        saved.append(state)
        if len(state.local_refinements) == 2:
            raise RuntimeError("checkpoint stop")

    with pytest.raises(RuntimeError, match="checkpoint stop"):
        solve_balanced_adaptive_darcy(
            TriangleMesh.unit_square(),
            source=1,
            iterations=3,
            local_refinement=2,
            estimator_order=5,
            on_state=checkpoint,
        )
    assert [len(state.result.solutions) for state in saved] == [1, 2]
    assert saved[0].decisions == ("macro",)
    assert len(saved[0].result.estimators) == 1
