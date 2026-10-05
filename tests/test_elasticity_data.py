"""Independent differential and integral identities for exact elasticity data."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose


@pytest.fixture(params=["ElasticityData", "TrigonometricElasticityData"])
def data_class(monkeypatch, request):
    """Import the public example without making it a runtime package dependency."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return getattr(importlib.import_module("elasticity_data"), request.param)


@pytest.fixture
def points():
    """Use reproducible interior points away from symmetry-induced zero fields."""
    return np.random.default_rng(2410).uniform(0.03, 0.97, (23, 2))


# Cover each modulus regime with pressure present. Zero-amplitude independence
# and amplitude linearity have their own tests below, so they need not multiply
# every differential-identity case.
@pytest.mark.parametrize(
    "lame_lambda,amplitude",
    [
        (0.0, 1.0),
        (1e-16, 1.0),
        (2.3, 1.0),
        (1e4, 1.0),
        (1e8, 1.0),
        (np.inf, 1.0),
        (2.3, -0.7),
        (1e4, 0.0),
    ],
)
def test_exact_fields_satisfy_momentum_and_constitutive_equations(
    data_class, points, lame_lambda, amplitude
):
    data = data_class(lame_lambda, lame_mu=1.7, pressure_amplitude=amplitude)
    step = 1e-30
    gradient = np.stack(
        [data.displacement(points + 1j * step * direction).imag / step for direction in np.eye(2)],
        axis=-1,
    )
    assert_allclose(data.gradient(points), gradient, rtol=2e-14, atol=2e-14)
    assert_allclose(data.divergence(points), np.trace(gradient, axis1=1, axis2=2), atol=2e-14)
    stress_divergence = sum(
        data.stress(points + 1j * step * direction).imag[:, :, axis] / step
        for axis, direction in enumerate(np.eye(2))
    )
    assert_allclose(data.source(points), -stress_divergence, rtol=3e-14, atol=3e-12)
    assert_allclose(data.stress(points), data.stress(points).swapaxes(1, 2))
    if np.isinf(lame_lambda):
        assert_allclose(data.divergence(points), 0.0, atol=0.0)
    else:
        assert_allclose(data.pressure(points), -lame_lambda * data.divergence(points), atol=1e-14)


@pytest.mark.parametrize("lame_lambda", [1.0, np.inf])
def test_homogeneous_boundary_and_zero_mean_pressure(data_class, lame_lambda):
    data = data_class(lame_lambda)
    line = np.linspace(0.0, 1.0, 19)
    boundary = np.concatenate(
        [
            np.column_stack((line, 0 * line)),
            np.column_stack((line, 0 * line + 1)),
            np.column_stack((0 * line, line)),
            np.column_stack((0 * line + 1, line)),
        ]
    )
    assert_allclose(data.displacement(boundary), 0.0, atol=8 * np.finfo(float).eps)
    nodes, weights = leggauss(6)
    x, y = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2)
    points = np.column_stack((x.ravel(), y.ravel()))
    weights = np.outer(weights, weights).ravel() / 4
    assert abs(weights @ data.pressure(points)) < 5e-14
    assert abs(weights @ data.divergence(points)) < 5e-14


def test_force_stays_bounded_and_pressure_remains_nonzero_at_infinite_lambda(data_class, points):
    incompressible = data_class(np.inf, lame_mu=2.0)
    zero_lambda = data_class(0.0, lame_mu=2.0)
    upper_bound = np.maximum(
        np.abs(incompressible.source(points)), np.abs(zero_lambda.source(points))
    )
    for ratio in (0.0, 1e-16, 1.0, 1e2, 1e4, 1e6, 1e8, 1e16, np.inf):
        data = data_class(2.0 * ratio, lame_mu=2.0)
        assert np.isfinite(data.source(points)).all()
        assert np.all(np.abs(data.source(points)) <= upper_bound + 2e-13)
    assert np.linalg.norm(incompressible.pressure(points)) > 1.0
    almost = data_class(2e8, lame_mu=2.0)
    assert_allclose(almost.displacement(points), incompressible.displacement(points), atol=2e-8)
    assert_allclose(almost.pressure(points), incompressible.pressure(points), rtol=1e-8)
    small = data_class(1e-16, lame_mu=2.0)
    assert np.linalg.norm(small.pressure(points)) > 0.0


def test_solenoidal_family_is_independent_of_lambda(data_class, points):
    reference = data_class(0.0, pressure_amplitude=0.0)
    for lame_lambda in (1.0, 1e4, 1e8, np.inf):
        data = data_class(lame_lambda, pressure_amplitude=0.0)
        assert_allclose(data.displacement(points), reference.displacement(points), atol=0.0)
        assert_allclose(data.source(points), reference.source(points), atol=0.0)
        assert_allclose(data.pressure(points), 0.0, atol=0.0)
        assert_allclose(data.divergence(points), 0.0, atol=0.0)


def test_displacement_amplitude_changes_force_and_pressure_consistently(data_class, points):
    zero, half, full = [data_class(4999, pressure_amplitude=a) for a in (0.0, 0.5, 1.0)]
    for field in ("displacement", "gradient", "pressure", "source", "stress"):
        assert_allclose(
            getattr(half, field)(points),
            0.5 * (getattr(zero, field)(points) + getattr(full, field)(points)),
            rtol=1e-14,
            atol=1e-13,
        )


# The amplitude-zero field is modulus independent; retain one exact integral
# for it and all modulus regimes for the pressure-carrying field.
@pytest.mark.parametrize(
    "lame_lambda,amplitude",
    [(0.0, 1.0), (1.0, 1.0), (1e8, 1.0), (np.inf, 1.0), (1.0, 0.0)],
)
def test_polynomial_displacement_norm_and_elastic_energy_have_exact_integrals(
    monkeypatch, lame_lambda, amplitude
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    data = importlib.import_module("elasticity_data").ElasticityData(
        lame_lambda, lame_mu=2.0, pressure_amplitude=amplitude
    )
    nodes, weights = leggauss(8)
    x, y = np.meshgrid((nodes + 1) / 2, (nodes + 1) / 2)
    points = np.column_stack((x.ravel(), y.ravel()))
    weights = np.outer(weights, weights).ravel() / 4
    beta = 0.0 if np.isinf(lame_lambda) else 2.0 / (lame_lambda + 2.0)
    # The one-dimensional bubble moments are 1/630, 2/105 and 4/5 for
    # a², (a')² and (a'')² respectively. Curl(phi) is orthogonal to grad(phi).
    exact_norm_squared = (32768 / 33075) * (1 + (amplitude * beta) ** 2)
    exact_energy = 2.0 * (65536 / 1225) * (1 + amplitude**2 * beta * (1 + beta))
    displacement = data.displacement(points)
    gradient = data.gradient(points)
    strain = (gradient + gradient.swapaxes(1, 2)) / 2
    assert_allclose(weights @ np.sum(displacement**2, axis=1), exact_norm_squared, rtol=5e-15)
    assert_allclose(
        weights @ np.einsum("nij,nij->n", data.stress(points), strain), exact_energy, rtol=5e-15
    )


@pytest.mark.parametrize(
    "data_class,kwargs",
    [
        pytest.param("ElasticityData", kwargs, id=name)
        for name, kwargs in (
            ("negative-lambda", {"lame_lambda": -1.0}),
            ("negative-infinite-lambda", {"lame_lambda": -np.inf}),
            ("nonfinite-lambda", {"lame_lambda": np.nan}),
            ("zero-mu", {"lame_mu": 0.0}),
            ("negative-mu", {"lame_mu": -1.0}),
            ("nonfinite-mu", {"lame_mu": np.nan}),
            ("infinite-mu", {"lame_mu": np.inf}),
            ("nonfinite-amplitude", {"pressure_amplitude": np.nan}),
            ("infinite-amplitude", {"pressure_amplitude": np.inf}),
        )
    ]
    + [
        pytest.param(
            "TrigonometricElasticityData", {"lame_lambda": -1.0}, id="inherited-validation"
        )
    ],
    indirect=["data_class"],
)
def test_invalid_material_or_amplitude_rejected(data_class, kwargs):
    """Cover every material failure in the shared constructor and its inherited use."""
    with pytest.raises(ValueError):
        data_class(**kwargs)
