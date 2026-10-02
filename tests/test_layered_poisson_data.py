"""Independent checks of the analytical reference used in the unfitted campaign."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose


@pytest.fixture
def series_class(monkeypatch):
    """Load the analytical example independently of the installed runtime package."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("layered_poisson").LayeredPoissonSeries


def test_series_satisfies_boundary_transmission_and_differentiated_pde(series_class):
    data = series_class(10, 25)
    x = np.linspace(0, 1, 17)
    boundary = np.concatenate(
        [
            np.column_stack((x, x * 0)),
            np.column_stack((x, x * 0 + 1)),
            np.column_stack((x * 0, x)),
            np.column_stack((x * 0 + 1, x)),
        ]
    )
    assert_allclose(data.evaluate(boundary)[0], 0, atol=1e-16)
    first = np.column_stack((x[1:-1], x[1:-1] * 0 + 0.5 - 1e-9))
    second = first + (0, 2e-9)
    p1, g1 = data.evaluate(first)
    p2, g2 = data.evaluate(second)
    assert_allclose(p1, p2, atol=4e-10)
    assert_allclose(10 * g1[:, 1], g2[:, 1], atol=3e-9)
    points = np.array([[0.13, 0.19], [0.42, 0.82]])
    step = 1e-6
    gradient, laplace = [], np.zeros(2)
    for axis in range(2):
        delta = np.eye(2)[axis] * step
        plus, derivative_plus = data.evaluate(points + delta)
        minus, derivative_minus = data.evaluate(points - delta)
        gradient.append((plus - minus) / (2 * step))
        laplace += (derivative_plus[:, axis] - derivative_minus[:, axis]) / (2 * step)
    assert_allclose(np.array(gradient).T, data.evaluate(points)[1], atol=2e-11)
    wave = np.pi * np.arange(1, 50, 2)
    forcing = np.sum(4 / wave * np.sin(points[:, 0, None] * wave), axis=1)
    assert_allclose(-np.where(points[:, 1] < 0.5, 10, 1) * laplace, forcing, atol=3e-9)


def test_analytical_energy_integral_matches_independent_tensor_quadrature(series_class):
    data = series_class(10, 12)
    gauss, weights = leggauss(80)
    x = (gauss + 1) / 2
    y = np.r_[(gauss + 1) / 4, 0.5 + (gauss + 1) / 4]
    points = np.array([(a, b) for a in x for b in y])
    measure = (weights[:, None] / 2 * np.tile(weights / 4, 2)).ravel()
    assert_allclose(measure @ data.evaluate(points)[0], data.energy_squared(), atol=3e-16)
    values = [series_class(10, n).energy_squared() for n in (127, 255, 511)]
    assert values[0] < values[1] < values[2]
    assert values[2] - values[1] < 3e-10
