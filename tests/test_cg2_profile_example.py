"""Incident CG2 profile traces preserve physical points, material sides and low components."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


def modules(monkeypatch):
    """Load original field and profile owners without importing an optional FEM runtime."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return tuple(
        importlib.import_module(f"examples.{name}")
        for name in (
            "solve_unusual_spe10_reference",
            "cg2_profiles",
            "solve_pgmhm_inclusions_reference",
        )
    )


def test_exact_diagonal_and_material_traces(monkeypatch):
    """A continuous scalar retains different gradients on the same physical diagonal point."""
    owner, profiles, _ = modules(monkeypatch)
    x, y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    field = owner.CG2Field(abs(x - y), np.ones((1, 1)), (0, 1, 0, 1))
    traces = profiles.vertical_traces(field, 0.3)
    assert_array_equal(traces["points"][0, 1], traces["points"][1, 0])
    assert_allclose(traces["pressure"][:, [1, 0]].diagonal(), 0, atol=1e-16)
    assert_allclose(traces["flux"][0], [[-1, 1], [-1, 1]], atol=2e-16)
    assert_allclose(traces["flux"][1], [[1, -1], [1, -1]], atol=2e-16)
    x, _ = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 3))
    field = owner.CG2Field(x, np.array([[1.0], [4.0]]), (0, 1, 0, 1))
    traces = profiles.vertical_traces(field, 0.5)
    assert_array_equal(traces["points"][0], traces["points"][1])
    assert_allclose(traces["flux"][0], [[-1, 0], [-1, 0]], atol=0)
    assert_allclose(traces["flux"][1], [[-4, 0], [-4, 0]], atol=0)
    for coordinate in (0.0, 1.0):
        assert profiles.vertical_traces(field, coordinate)["points"].shape == (1, 2, 2)
    for coordinate in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="profile"):
            profiles.vertical_traces(field, coordinate)


def test_explicit_owner_validation_and_bulk_equivalence(monkeypatch):
    """Owners cannot select nonincident triangles; interior arithmetic agrees bitwise."""
    owner, _, _ = modules(monkeypatch)
    field = owner.CG2Field(np.arange(9.0).reshape(3, 3), np.ones((1, 1)), (0, 1, 0, 1))
    points = np.array([[0.7, 0.1], [0.1, 0.7]])
    rectangles = np.zeros((2, 2), dtype=int)
    lower = np.array([True, False])
    for first, second in zip(
        field.evaluate(points), field.evaluate_cells(points, rectangles, lower), strict=True
    ):
        assert_array_equal(first, second)
    cases = (
        (points[:, 0], rectangles, lower),
        (np.zeros((2, 3)), rectangles, lower),
        (np.full((2, 2), np.nan), rectangles, lower),
        (points, rectangles[:, 0], lower),
        (points, rectangles.astype(float), lower),
        (points, rectangles, lower[:1]),
        (points, rectangles, lower.astype(int)),
        (points, -np.ones_like(rectangles), lower),
        (points, np.ones_like(rectangles), lower),
        (points, rectangles, ~lower),
        (np.array([[-0.1, 0.1], [0.1, 0.7]]), rectangles, lower),
        (np.array([[1.1, 0.1], [0.1, 0.7]]), rectangles, lower),
        (np.array([[0.7, -0.1], [0.1, 0.7]]), rectangles, lower),
        (np.array([[0.7, 1.1], [0.1, 0.7]]), rectangles, lower),
    )
    for values, owners, sides in cases:
        with pytest.raises(ValueError):
            field.evaluate_cells(values, owners, sides)


def test_component_traces_keep_small_gradients(monkeypatch):
    """A large principal datum never absorbs the separately stored correction gradient."""
    _, profiles, owner = modules(monkeypatch)
    x, y = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    field = owner.InclusionField(
        np.full((3, 3), 1e100), np.ones((1, 1)), (0, 1, 0, 1), correction=1e-20 * (x - y)
    )
    traces = profiles.vertical_traces(field, 0.3)
    assert_allclose(
        traces["flux"], np.broadcast_to([-1e-20, 1e-20], traces["flux"].shape), rtol=4e-16, atol=0
    )
