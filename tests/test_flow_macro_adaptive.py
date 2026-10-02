"""L14 element marking and unchanged local subdivision under conforming macro refinement."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.flow_macro_adaptive import adapt_flow_macros, mark_flow_cells
from pymhm.mesh import TriangleMesh


def test_macro_adaptation_reduces_physical_pressure_error() -> None:
    """Three actual solves refine macros while retaining one local cell per macro."""
    result = adapt_flow_macros(
        TriangleMesh.unit_square(), iterations=2, source=lambda x: 2 * x, theta=0.75
    )
    errors = [
        solution.pressure_l2_error(lambda x: np.sum(x * x, axis=1) - 2 / 3)
        for solution in result.solutions
    ]
    assert len(result.solutions) == 3
    assert result.stop_reason == "iterations"
    assert errors[-1] < errors[1] < errors[0]
    assert all(
        len(fine.cells) == 1 for solution in result.solutions for fine in solution.local_meshes
    )
    assert all(estimate.second_level_scale == 1 for estimate in result.estimators)
    assert len(result.refinements) == 2
    for prior, after, refinement in zip(
        result.solutions[:-1], result.solutions[1:], result.refinements, strict=True
    ):
        assert after.skeleton.mesh is refinement.mesh
        assert len(after.skeleton.mesh.cells) > len(prior.skeleton.mesh.cells)
    assert max(solution.hybrid.residual for solution in result.solutions) < 1e-10


def test_zero_indicator_and_cell_cap_have_explicit_stop_reasons() -> None:
    """Zero data stop immediately; memory limits never change the requested discretization."""
    mesh = TriangleMesh.unit_square()
    zero = adapt_flow_macros(mesh, iterations=3, tolerance=1e-12)
    assert zero.stop_reason == "tolerance"
    assert len(zero.solutions) == 1
    assert not np.any(zero.marked[0])
    capped = adapt_flow_macros(mesh, maximum_cells=2, source=lambda x: 2 * x)
    assert capped.stop_reason == "cell_limit"
    assert len(capped.solutions) == 1
    assert not capped.refinements


@pytest.mark.parametrize("macro_refiner", ["red-green", "longest-edge"])
def test_macro_children_retain_nonuniform_parent_local_resolution(macro_refiner: str) -> None:
    """Macro marking does not turn unequal local resolutions into uniform local refinement."""
    result = adapt_flow_macros(
        TriangleMesh.unit_square(),
        local_refinement=(1, 2),
        iterations=1,
        source=lambda x: 2 * x,
        macro_refiner=macro_refiner,
    )
    parent_counts = np.array([1, 4])
    assert len(result.refinements) == 1
    assert_allclose(
        [len(mesh.cells) for mesh in result.solutions[-1].local_meshes],
        parent_counts[result.refinements[0].parent_cells],
    )


def test_marking_uses_sum_of_face_norms_and_unscaled_local_norm() -> None:
    """The element rule differs from both root-of-squares and face marking."""
    estimate = adapt_flow_macros(TriangleMesh.unit_square(), iterations=0).estimators[0]
    mesh = estimate.solution.skeleton.mesh
    face_values = tuple(np.array([1.0, 3.0]) for _ in mesh.faces)
    estimate = replace(estimate, face_squared=face_values, local_squared=np.array([16.0, 0.0]))
    # Every face contributes sqrt(1+3)=2: eta_K=[10,6].
    assert_allclose(mark_flow_cells(estimate, theta=0.7), [True, False])
    assert_allclose(mark_flow_cells(estimate, theta=0.6), [True, True])
    with pytest.raises(ValueError, match="theta"):
        mark_flow_cells(estimate, theta=0)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"iterations": -1}, "iterations"),
        ({"maximum_cells": 0}, "maximum_cells"),
        ({"maximum_cells": 1}, "initial macro"),
        ({"theta": np.nan}, "theta"),
        ({"tolerance": -1}, "tolerance"),
        ({"traction": {0: (0, 0)}}, "Dirichlet"),
        ({"traction_components": {0: (0, 0)}}, "Dirichlet"),
        ({"advection": (1.0, 0.0)}, "advection"),
        ({"advection": lambda x: x}, "advection"),
        ({"local_meshes": ()}, "own skeleton"),
        ({"skeleton": None}, "own skeleton"),
        ({"local_refinement": 0}, "local_refinement"),
        ({"macro_refiner": "unknown"}, "macro_refiner"),
    ],
)
def test_invalid_macro_adaptation_contracts(options: dict, match: str) -> None:
    """Reject incompatible PDE and approximation inputs before assembling a local problem."""
    with pytest.raises(ValueError, match=match):
        adapt_flow_macros(TriangleMesh.unit_square(), **options)
