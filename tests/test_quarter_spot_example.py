"""Exact geometry and source integrals for the square-obstacle example."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.mark.parametrize("refinement", [4, 8, 16])
def test_square_material_and_well_interfaces_are_fitted(monkeypatch, refinement):
    """Preserve 25% material area, fixed unit well rates and uncut fine cells."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    problem = importlib.import_module("quarter_spot_problem")
    macro = problem.macro_mesh()
    assert len(macro.cells) == 200
    fine = [macro.submesh(cell, refinement) for cell in range(len(macro.cells))]
    areas = np.concatenate([mesh.areas for mesh in fine])
    vertices = np.concatenate([mesh.points[mesh.cells] for mesh in fine])
    centers = vertices.mean(axis=1)
    lower, upper = problem.OBSTACLE_LOWER, problem.OBSTACLE_UPPER
    assert_allclose(lower + upper, 1.0, rtol=0, atol=2e-16)
    assert_allclose((upper - lower) ** 2, 0.25, rtol=0, atol=2e-16)
    assert_allclose(areas.sum(), 1.0, rtol=0, atol=5e-16)
    assert_allclose(areas @ (problem.coefficient(centers) == 1e-4), 0.25, atol=3e-16)
    load = problem.source(centers)
    assert_allclose(areas @ np.maximum(load, 0), 1.0, rtol=0, atol=5e-16)
    assert_allclose(areas @ np.minimum(load, 0), -1.0, rtol=0, atol=5e-16)
    for direction in (0, 1):
        for interface in (0.1, lower, upper, 0.9):
            crosses = (vertices[..., direction].min(axis=1) < interface - 1e-14) & (
                vertices[..., direction].max(axis=1) > interface + 1e-14
            )
            assert not crosses.any()
    assert_allclose(problem.coefficient(centers), problem.coefficient(1 - centers), atol=0)
    assert_allclose(problem.source(centers), -problem.source(1 - centers), atol=0)


def test_material_interface_cuts_macrocells(monkeypatch):
    """Require genuine submacro heterogeneity, with no obstacle edge on the macrogrid."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    problem = importlib.import_module("quarter_spot_problem")
    macro = problem.macro_mesh()
    for axis in (0, 1):
        coordinates = np.unique(macro.points[:, axis])
        for bound in (problem.OBSTACLE_LOWER, problem.OBSTACLE_UPPER):
            assert not np.isclose(coordinates, bound, atol=1e-14, rtol=0).any()
    cut_cells = 0
    for index in range(len(macro.cells)):
        fine = macro.submesh(index, 4)
        material = problem.coefficient(fine.points[fine.cells].mean(axis=1))
        cut_cells += int(material.min() != material.max())
    # Twenty macro squares meet the perimeter; two corner squares cut one triangle.
    assert cut_cells == 38
