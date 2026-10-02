"""Nonnested classical comparisons preserve exact polynomial norms and physical areas."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


def modules(monkeypatch):
    """Load the original application without adding optional imports to the core."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    comparison = importlib.import_module("examples.compare_inclusion_references")
    return comparison, importlib.import_module("examples.solve_pgmhm_inclusions_reference")


def field(reference, coordinates, active):
    """Interpolate an exact global quadratic on explicit, possibly graded P2 coordinates."""
    nodes = np.sort(np.r_[coordinates, (coordinates[:-1] + coordinates[1:]) / 2])
    x, y = np.meshgrid(nodes, nodes)
    return reference.InclusionField(
        (x**2 + 2 * y**2) * active,
        np.ones((1, 1)),
        (0, 1, 0, 1),
        coordinates,
        coordinates,
    )


@pytest.mark.parametrize("order", [3, 4])
def test_exact_nonnested_quadratic_norms_and_replay(monkeypatch, order):
    """Nonzero exact moments and zero difference detect weight or diagonal mistakes."""
    comparison, reference = modules(monkeypatch)
    fine = field(reference, np.array([0.0, 0.1, 0.5, 0.9, 1.0]), 1)
    coarse = field(reference, np.linspace(0, 1, 4), 0)
    base = np.array([0.0, 1.0])
    points, weights = comparison.intersection_pattern(fine, coarse, base, order)
    monkeypatch.setattr(comparison, "_DATA", (fine, coarse, base, points, weights))
    expected = np.array([13 / 9, 20 / 3, 20 / 3] * 2)
    assert_allclose(comparison.integrate_group(np.array([0])), expected, rtol=8e-15, atol=2e-14)
    matching = field(reference, np.linspace(0, 1, 4), 1)
    monkeypatch.setattr(comparison, "_DATA", (fine, matching, base, points, weights))
    assert_allclose(comparison.integrate_group(np.array([0]))[:3], 0, rtol=0, atol=2e-27)


def test_parent_partition_and_initialization_are_explicit(monkeypatch):
    """A repeated geometric pattern cannot silently replace a different physical grid."""
    comparison, reference = modules(monkeypatch)
    irregular = field(reference, np.array([0.0, 0.2, 0.5, 0.6, 1.0]), 1)
    with pytest.raises(ValueError, match="repeat"):
        comparison.fractions(irregular, np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="parent material"):
        comparison.fractions(irregular, np.array([0.0, 0.4, 1.0]))
    monkeypatch.setattr(comparison, "_DATA", None)
    with pytest.raises(RuntimeError, match="initialize"):
        comparison.integrate_group(np.array([0]))
