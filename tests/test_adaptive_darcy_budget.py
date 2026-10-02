"""Budget marking preserves indicator bulk, geometry and refinement ancestry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.adaptive_darcy import mark_dorfler
from pymhm.adaptive_darcy_budget import refine_darcy_budget
from pymhm.longest_edge import refine_longest_edge
from pymhm.mesh import TriangleMesh
from pymhm.refinement import TriangleRefinement


def test_budget_enlarges_bulk_without_reference_field():
    """A concentrated indicator still admits a prescribed finer complexity."""
    mesh = TriangleMesh.unit_square(4)
    values = np.arange(1, len(mesh.cells) + 1, dtype=float) ** 8
    bulk = mark_dorfler(values, 0.5)
    result = refine_darcy_budget(mesh, values, target_cells=60)
    assert np.all(result.marked[bulk])
    assert result.marked.sum() > bulk.sum()
    assert result.captured_fraction >= 0.5
    assert len(result.refinement.mesh.cells) >= 60
    assert result.target_cells == 60
    assert_allclose(result.refinement.mesh.areas.sum(), mesh.areas.sum(), rtol=1e-14)
    for parent in range(len(mesh.cells)):
        assert_allclose(
            result.refinement.mesh.areas[result.refinement.parent_cells == parent].sum(),
            mesh.areas[parent],
            rtol=1e-14,
        )
    assert np.all(np.bincount(result.refinement.parent_cells)[result.marked] >= 2)


def test_existing_bulk_closure_and_deterministic_ties():
    """A sufficiently large bulk set is reused, and tied indicators retain input order."""
    mesh = TriangleMesh.unit_square(2)
    values = np.ones(len(mesh.cells))
    bulk = mark_dorfler(values, 1.0)
    expected = refine_longest_edge(mesh, bulk)
    result = refine_darcy_budget(mesh, values, target_cells=9, theta=1.0)
    assert np.array_equal(result.marked, bulk)
    assert np.array_equal(result.refinement.mesh.cells, expected.mesh.cells)
    assert result.captured_fraction == 1.0


@pytest.mark.parametrize("target", [0, True, 1, 2])
def test_invalid_budgets(target):
    """A new refinement target is an integer strictly beyond the current mesh."""
    with pytest.raises(ValueError, match="target_cells"):
        refine_darcy_budget(TriangleMesh.unit_square(), [1.0, 2.0], target_cells=target)


def test_invalid_indicators_and_unreachable_target():
    """Zero, mismatched and unreachable requests fail explicitly."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="one squared"):
        refine_darcy_budget(mesh, [1.0], target_cells=3)
    with pytest.raises(ValueError, match="positive indicators"):
        refine_darcy_budget(mesh, [0.0, 0.0], target_cells=3)
    with pytest.raises(ValueError, match="one-level"):
        refine_darcy_budget(mesh, [1.0, 2.0], target_cells=100)

    def unchanged(current, marked):
        """A valid identity ancestry cannot satisfy a strict refinement budget."""
        return TriangleRefinement(
            current, np.arange(len(current.cells)), np.arange(len(current.faces))
        )

    with pytest.raises(ValueError, match="one-level"):
        refine_darcy_budget(mesh, [1.0, 2.0], target_cells=3, refiner=unchanged)
