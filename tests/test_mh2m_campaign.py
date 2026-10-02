"""Check analytical data and common-cell physical norms without a large campaign."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.mesh import TriangleMesh


@pytest.fixture
def campaign(monkeypatch):
    """Load the public campaign through its package-style invocation path."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.mh2m_campaign")


def test_manufactured_source_and_gradient_by_complex_step(campaign):
    points = np.array([[0.23, 0.57], [0.88, 0.63]])
    gradient = np.column_stack(
        [np.imag(campaign.exact(points + 1e-25j * axis)) / 1e-25 for axis in np.eye(2)]
    )
    assert_allclose(gradient, campaign.exact_gradient(points), atol=1e-14)
    laplacian = sum(
        np.imag(campaign.exact_gradient(points + 1e-25j * axis)[:, i]) / 1e-25
        for i, axis in enumerate(np.eye(2))
    )
    assert_allclose(-laplacian, campaign.source(points), atol=1e-14)
    assert np.all(campaign.oscillatory(points) > 0)


def test_common_partition_norms_against_exact_polynomial_integrals(monkeypatch, campaign):
    monkeypatch.setattr(campaign, "oscillatory", lambda x: np.full(len(x), 2.5))
    fine, coarse = TriangleMesh.unit_square(8), TriangleMesh.unit_square(2)
    reference = {
        "vertices": fine.points[fine.cells],
        "pressure": (fine.points[:, 0] + 2 * fine.points[:, 1])[fine.cells],
    }
    other = {
        "vertices": coarse.points[coarse.cells],
        "pressure": (0.5 * coarse.points[:, 0] + 0.5 * coarse.points[:, 1])[coarse.cells],
    }
    norms = campaign.difference(reference, other, 4)
    assert_allclose(
        norms["pressure_difference_l2"],
        np.sqrt(0.5**2 / 3 + 1.5**2 / 3 + 0.5 * 1.5 / 2),
        atol=2e-14,
    )
    assert_allclose(norms["flux_difference_l2"], 2.5 * np.sqrt(2.5), atol=2e-14)
    assert_allclose(norms["reference_pressure_l2"], np.sqrt(8 / 3), atol=2e-14)
    assert_allclose(norms["reference_flux_l2"], 2.5 * np.sqrt(5), atol=2e-14)
    assert_allclose(campaign.difference(reference, reference)["flux_difference_l2"], 0, atol=1e-14)
    displaced = {"vertices": reference["vertices"] + 10, "pressure": reference["pressure"]}
    with pytest.raises(ValueError, match="outside"):
        campaign.difference(reference, displaced)


def test_conforming_reference_small_residual_and_zero_boundary(campaign):
    result = campaign.conforming_reference(4)
    assert result["residual"] < 1e-12
    mask = np.any((result["vertices"] == 0) | (result["vertices"] == 1), axis=2)
    assert_allclose(result["pressure"][mask], 0)
