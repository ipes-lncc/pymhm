"""Keep broken acoustic vertex traces independent in the published sampling norm."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.loads import split_point_sources
from pymhm.quadrilateral import CartesianMacroMesh, qk_space


def test_signed_incident_samples_preserve_complex_polynomials(monkeypatch):
    """Every quadrant recovers its own Q3 polynomial at a shared macro vertex."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    driver = importlib.import_module("examples.marmousi_campaign")
    mesh = CartesianMacroMesh(2, 2, (0, 4, 0, 2))
    pressure = []
    for cell in range(4):
        _, nodes = qk_space(mesh.submesh(cell, 2), 3)
        pressure.append(1 + nodes[:, 0] ** 3 + 2j * nodes[:, 1] ** 2 + 100 * cell)
    pressure = np.asarray(pressure)
    for side, owner in [((-1, -1), 0), ((1, -1), 1), ((-1, 1), 2), ((1, 1), 3)]:
        value = driver.evaluate_fields(pressure, mesh, 2, 3, [[2, 1]], side=side)
        assert_allclose(value, 9 + 2j + 100 * owner, atol=2e-13)
    points = np.array([[0.12, 0.72], [3.8, 1.32], [4, 2]])
    expected = 1 + points[:, 0] ** 3 + 2j * points[:, 1] ** 2 + [0, 300, 300]
    assert_allclose(driver.evaluate_fields(pressure, mesh, 2, 3, points), expected, atol=2e-13)
    with pytest.raises(ValueError, match="outside"):
        driver.evaluate_fields(pressure, mesh, 2, 3, [[-0.1, 0.2]])
    with pytest.raises(ValueError, match="selectors"):
        driver.evaluate_fields(pressure, mesh, 2, 3, [[0.1, 0.2]], side=(0, 1))
    assert len(driver.source_hashes()) >= 15


@pytest.mark.parametrize("width,incident_count", [(20, 2), (40, 2), (80, 1)])
def test_marmousi_interface_source_preserves_total_load(width, incident_count):
    """The published source geometry has two half loads or one interior unit load."""
    mesh = CartesianMacroMesh(320 // width, 160 // width, (4800, 5120, 0, 160))
    allocation = split_point_sources(mesh, [(5000, 50, 1.0)])
    active = [values for values in allocation if len(values)]
    assert len(active) == incident_count
    rows = np.concatenate(active)
    assert_allclose(rows[:, :2], np.tile([5000, 50], (incident_count, 1)))
    assert_allclose(rows[:, 2], np.full(incident_count, 1 / incident_count))
    assert rows[:, 2].sum() == 1.0
