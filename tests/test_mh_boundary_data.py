"""Verify the anisotropic campaign data independently of either discretization."""

import importlib
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose


def test_boundary_campaign_derivatives_and_physical_norms(monkeypatch):
    """Check complex-step derivatives and independent tensor-Gauss integrals."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    data = importlib.import_module("examples.mh_boundary_campaign")
    points = np.array([[0.13, 0.27], [0.41, 0.88], [0.91, 0.39]])
    gradient = np.column_stack(
        [np.imag(data.pressure(points + 1e-25j * axis)) / 1e-25 for axis in np.eye(2)]
    )
    assert_allclose(data.flux(points), -gradient @ data.MATERIAL, atol=2e-14)
    divergence = sum(
        np.imag(data.flux(points + 1e-25j * axis)[:, i]) / 1e-25 for i, axis in enumerate(np.eye(2))
    )
    assert_allclose(data.source(points), divergence, rtol=2e-14)
    t, w = leggauss(18)
    x, y = np.meshgrid((t + 1) / 2, (t + 1) / 2, indexing="ij")
    samples = np.column_stack((x.ravel(), y.ravel()))
    weights = np.outer(w, w).ravel() / 4
    assert_allclose(weights @ data.pressure(samples), 2.5 + 4 / np.pi**2, rtol=2e-14)
    assert_allclose(
        weights @ data.pressure(samples) ** 2, 20 / 3 + 1 / 4 + 20 / np.pi**2, rtol=2e-14
    )
    assert_allclose(
        weights @ np.sum(data.flux(samples) ** 2, axis=1), 33.8 + 3.33 * np.pi**2, rtol=2e-14
    )
