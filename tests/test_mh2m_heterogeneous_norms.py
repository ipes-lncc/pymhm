"""Check common-overlay physical norms on independent nonnested P1 partitions."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.mesh import TriangleMesh


@pytest.fixture
def norms(monkeypatch):
    """Load the original example postprocessor without requiring an installed examples package."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.mh2m_heterogeneous_norms")


def make_field(norms, n, function):
    """Construct a reordered physical P1 field to exercise the archive-coordinate contract."""
    mesh = TriangleMesh.unit_square(n)
    vertices = mesh.points[mesh.cells][::-1, ::-1]
    values = function(vertices.reshape(-1, 2)).reshape(-1, 3)
    return norms.StructuredP1.from_arrays(vertices, values)


@pytest.mark.parametrize("reference_resolution", [3, 4])
def test_nonnested_physical_flux_and_energy(norms, reference_resolution):
    """Check nonzero exact norms on both nonnested and nested dyadic partitions."""
    reference = make_field(norms, reference_resolution, lambda x: 1 + x @ [2, 3])
    other = make_field(norms, 2, lambda x: 1 + x @ [1, -1])
    result = norms.difference(reference, other, 2.0, 5)
    assert_allclose(result["pressure_difference"], np.sqrt(23 / 3), rtol=2e-14)
    assert_allclose(result["reference_pressure_norm"], np.sqrt(40 / 3), rtol=2e-14)
    assert_allclose(result["flux_difference"], 2 * np.sqrt(17), rtol=2e-14)
    assert_allclose(result["reference_flux_norm"], 2 * np.sqrt(13), rtol=2e-14)
    assert_allclose(result["energy_difference"], np.sqrt(34), rtol=2e-14)
    assert_allclose(result["gradient_difference"], np.sqrt(17), rtol=2e-14)
    assert_allclose(result["overlay_area"], 1, atol=2e-15)


def test_nonnested_broken_pressure_has_no_smoothed_interfaces(norms):
    """Integrate distinct piecewise-constant triangle values directly on common cuts."""
    reference = norms.StructuredP1(np.zeros((3, 3, 2, 3)))
    other_values = np.empty((2, 2, 2, 3))
    other_values[:, :, 0] = 2
    other_values[:, :, 1] = -1
    result = norms.difference(reference, norms.StructuredP1(other_values), 1.0)
    assert_allclose(result["pressure_difference"], np.sqrt(2.5), rtol=2e-14)
    assert result["pressure_relative_difference"] is None
    assert result["flux_difference"] == 0
    assert result["flux_relative_difference"] is None


def test_explicit_incident_profile_and_validation(norms):
    """Keep the upper and lower traces at the same unperturbed horizontal interface."""
    values = np.zeros((2, 2, 2, 3))
    values[:, 0] = 1
    values[:, 1] = 3
    field = norms.StructuredP1(values)
    points = np.array([[0.2, 0.5], [0.7, 0.5]])
    assert_allclose(field.evaluate(points, y_side=-1)[0], 1)
    assert_allclose(field.evaluate(points, y_side=1)[0], 3)
    with pytest.raises(ValueError, match="incident"):
        field.evaluate(points, y_side=0)
    with pytest.raises(ValueError, match="complete"):
        norms.StructuredP1.from_arrays(np.zeros((3, 3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="unique"):
        norms.StructuredP1.from_arrays(np.zeros((2, 3, 2)), np.zeros((2, 3)))


def test_varying_material_is_integrated_physically(norms):
    """Check q=-(1+x)grad(p) against a zero-gradient field without a midpoint material."""
    reference = make_field(norms, 2, lambda x: x[:, 0])
    other = make_field(norms, 3, lambda x: np.zeros(len(x)))
    result = norms.difference(reference, other, lambda x: 1 + x[:, 0], 5)
    assert_allclose(result["flux_difference"], np.sqrt(7 / 3), rtol=2e-14)
    assert_allclose(result["energy_difference"], np.sqrt(1.5), rtol=2e-14)


@pytest.mark.parametrize("order", [6, 8])
def test_spawned_norms_preserve_every_reduction(norms, order, monkeypatch):
    """Require bitwise identity of nonzero nested/nonnested integrals across processes."""
    reference = make_field(norms, 5, lambda x: 1 + x @ [2, 3])
    other = make_field(norms, 4, lambda x: 1 + x @ [1, -1])
    vertices = np.concatenate(tuple(norms.overlay_triangles(5, 4)))
    batches = np.array_split(vertices, 5)
    monkeypatch.setattr(norms, "overlay_triangles", lambda first, second: iter(batches))
    serial = norms.difference(reference, other, 2.0, order)
    assert norms.difference(reference, other, 2.0, order, workers=2) == serial


def test_invalid_norm_workers(norms):
    """Reject ambiguous or nonpositive worker counts before launching processes."""
    field = norms.StructuredP1(np.zeros((1, 1, 2, 3)))
    for count in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            norms.difference(field, field, 1.0, workers=count)
