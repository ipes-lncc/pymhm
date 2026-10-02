"""Independent differential identities for the native-extension gallery data."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def data(monkeypatch):
    """Import the independently implemented public analytical data."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("native_extension_data")


def test_variable_rad_source_equals_divergence_of_full_physical_flux(data):
    """Check all coefficient-derivative terms through complex-step flux divergence."""
    points = np.random.default_rng(153).uniform(0.01, 0.99, (41, 2))
    step = 1e-30
    derivatives = []
    for direction in np.eye(2):
        shifted = points + 1j * step * direction
        flux = -np.einsum("nij,nj->ni", data.rad_diffusion(shifted), data.sine_gradient(shifted))
        flux += data.rad_velocity(shifted) * data.sine(shifted)[:, None]
        derivatives.append(flux.imag / step)
    divergence = derivatives[0][:, 0] + derivatives[1][:, 1]
    assert_allclose(
        data.rad_source(points),
        divergence + data.rad_reaction(points) * data.sine(points),
        atol=3e-14,
    )
    tensor_derivatives = [data.rad_diffusion(points + 1j * step * d).imag / step for d in np.eye(2)]
    assert_allclose(
        tensor_derivatives[0][:, 0] + tensor_derivatives[1][:, 1],
        np.tile([0.115, 0.065], (len(points), 1)),
        atol=1e-16,
    )
    assert np.linalg.eigvalsh(data.rad_diffusion(points)).min() > 0
    assert np.linalg.eigvalsh(data.resistance(points)).min() > 0


@pytest.mark.parametrize("epsilon", [0.005, 0.02])
def test_layer_equation_boundary_values_and_true_maximum(data, epsilon):
    """Verify the layer PDE, vertical boundary data and analytical range."""
    layer = data.BoundaryLayer(epsilon)
    points = np.column_stack((np.linspace(0, 1, 83), np.full(83, 0.37)))
    step = 1e-30
    dx = layer.value(points + [1j * step, 0]).imag / step
    dxx = layer.gradient(points + [1j * step, 0]).imag[:, 0] / step
    assert_allclose(layer.gradient(points)[:, 0], dx, atol=1e-13)
    assert_allclose(-epsilon * dxx + dx, 1.0, atol=3e-14)
    assert_allclose(layer.value(np.array([[0.0, 0.1], [1.0, 0.9]])), 0, atol=0)
    assert 0 < layer.maximum < 1
    assert np.max(layer.value(points)) <= layer.maximum


@pytest.mark.parametrize("epsilon", [0, -0.1, np.nan, np.inf])
def test_invalid_layer_diffusivity_is_rejected(data, epsilon):
    """Reject nonphysical or nonfinite layer scales."""
    with pytest.raises(ValueError):
        data.BoundaryLayer(epsilon)


def test_sine_gradient_and_time_linear_heat_force(data):
    """Check spatial and temporal derivatives independently of the PDE assembler."""
    points = np.random.default_rng(45).uniform(size=(13, 2))
    step = 1e-30
    gradient = np.column_stack([data.sine(points + 1j * step * d).imag / step for d in np.eye(2)])
    assert_allclose(data.sine_gradient(points), gradient, atol=1e-15)
    laplacian = sum(
        data.sine_gradient(points + 1j * step * d).imag[:, i] / step
        for i, d in enumerate(np.eye(2))
    )
    time = 0.13
    time_derivative = data.heat_exact(points, time + 1j * step).imag / step
    assert_allclose(
        data.heat_source(points, time), time_derivative - (1 + time) * laplacian, atol=2e-14
    )


def test_layer_boundary_quadrature_resolves_thin_coarse_face(data):
    """Resolve the nonhomogeneous exponential trace on the outflow-adjacent coarse edge."""
    epsilon = 0.005
    layer = data.BoundaryLayer(epsilon)
    abscissa, weight = np.polynomial.legendre.leggauss(20)
    left, right = 0.75, 1.0
    x = left + (right - left) * (abscissa + 1) / 2
    points = np.column_stack((x, np.zeros_like(x)))
    integral = (right - left) / 2 * (weight @ layer.value(points))
    exponential = np.exp(-1 / epsilon)
    reference = (right**2 - left**2) / 2 - (
        epsilon * (np.exp((right - 1) / epsilon) - np.exp((left - 1) / epsilon))
        - exponential * (right - left)
    ) / (1 - exponential)
    assert_allclose(integral, reference, rtol=0, atol=3e-14)
