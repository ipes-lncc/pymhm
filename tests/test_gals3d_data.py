"""Independent differential checks of the bounded-forcing elasticity family."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

SPEC = importlib.util.spec_from_file_location(
    "gals3d_data", Path(__file__).resolve().parents[1] / "examples/gals3d_data.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
GaLS3DData = MODULE.GaLS3DData


@pytest.mark.parametrize("lam", [1.0, 1e8, np.inf])
@pytest.mark.parametrize("variable", [False, True])
def test_derivatives_and_equilibrium(lam, variable):
    """Central differences of exact stress verify the independently derived source."""
    data = GaLS3DData(lam, variable)
    points = np.random.default_rng(817).uniform(0.1, 0.9, (24, 3))
    step = 1e-5
    grad = np.zeros((len(points), 3, 3))
    force = np.zeros((len(points), 3))
    for axis in range(3):
        increment = step * np.eye(3)[axis]
        grad[:, :, axis] = (
            data.displacement(points + increment) - data.displacement(points - increment)
        ) / (2 * step)
        force -= (
            data.stress(points + increment)[:, :, axis]
            - data.stress(points - increment)[:, :, axis]
        ) / (2 * step)
    assert_allclose(grad, data.gradient(points), atol=2e-9, rtol=2e-7)
    assert_allclose(force, data.source(points), atol=3e-8, rtol=2e-7)
    assert_allclose(
        np.trace(data.gradient(points), axis1=1, axis2=2),
        -data.inverse_lambda * data.pressure(points),
        atol=4 * np.finfo(float).eps * np.max(np.abs(data.gradient(points))),
    )
    low, high, bound = data.shear_bounds
    assert np.min(data.shear(points)) >= low and np.max(data.shear(points)) <= high
    assert np.linalg.norm(data.shear_gradient(points), axis=1).max() <= bound
