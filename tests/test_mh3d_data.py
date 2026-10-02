"""Independent differential and integral checks of the original three-dimensional data."""

import importlib
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


def test_manufactured_differential_identities_and_exact_norms(monkeypatch):
    """Differentiate with complex steps and integrate exact separable norms independently."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    data = importlib.import_module("examples.mh3d_campaign")
    flux, pressure, source = data.flux, data.pressure, data.source
    points = np.random.default_rng(71).uniform(0.05, 0.95, (13, 3))
    gradient, divergence = [], np.zeros(len(points))
    for axis in range(3):
        shifted = points.astype(complex)
        shifted[:, axis] += 1e-30j
        gradient.append(np.imag(pressure(shifted)) / 1e-30)
        divergence += np.imag(flux(shifted)[:, axis]) / 1e-30
    assert_allclose(flux(points), -np.array(gradient).T, atol=2e-15)
    assert_allclose(source(points), divergence, atol=2e-14)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    points = np.array(np.meshgrid(*([0.5 * (nodes + 1)] * 3), indexing="ij")).reshape(3, -1).T
    weights = np.einsum("i,j,k->ijk", weights, weights, weights).ravel() / 8
    assert_allclose(weights @ pressure(points) ** 2, 103 / 6 + 1 / 8 + 64 / np.pi**3, rtol=1e-14)
    assert_allclose(weights @ np.sum(flux(points) ** 2, axis=1), 14 + 3 * np.pi**2 / 8, rtol=1e-14)
