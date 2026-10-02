"""Independent differentiation checks for the original 3D manufactured flow data."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


def _load_data():
    """Load the standalone example without adding examples to the package runtime."""
    path = Path(__file__).resolve().parents[1] / "examples/flow3d_data.py"
    specification = importlib.util.spec_from_file_location("flow3d_data", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.Flow3DData


@pytest.mark.parametrize("kind", ["stokes", "brinkman", "oseen"])
def test_manufactured_source_by_independent_finite_differences(kind):
    """Differentiate all vector components and pressure without reusing analytic derivatives."""
    data = _load_data()(kind)
    points = np.random.default_rng(428).uniform(0.1, 0.9, (13, 3))
    h = 1e-4
    gradient = np.empty((len(points), 3, 3))
    pressure_gradient = np.empty((len(points), 3))
    laplacian = np.zeros_like(points)
    for axis in range(3):
        offset = np.eye(3)[axis] * h
        plus, minus = data.velocity(points + offset), data.velocity(points - offset)
        gradient[:, :, axis] = (plus - minus) / (2 * h)
        pressure_gradient[:, axis] = (
            data.pressure(points + offset) - data.pressure(points - offset)
        ) / (2 * h)
        laplacian += (plus - 2 * data.velocity(points) + minus) / h**2
    assert_allclose(gradient, data.gradient(points), atol=6e-8, rtol=0)
    assert_allclose(pressure_gradient, data.pressure_gradient(points), atol=2e-12, rtol=0)
    assert_allclose(np.trace(data.gradient(points), axis1=-2, axis2=-1), 0.0, atol=0)
    independent = (
        -data.viscosity * laplacian
        + np.einsum("nij,nj->ni", gradient, data.advection(points))
        + np.einsum("nij,nj->ni", data.resistance(points), data.velocity(points))
        + pressure_gradient
    )
    assert_allclose(independent, data.source(points), atol=5e-7, rtol=0)
    options = data.options()
    assert options["viscosity"] == data.viscosity
    if kind == "oseen":
        assert np.max(np.linalg.norm(data.advection(points), axis=1)) < options["advection_bound"]


def test_manufactured_pressure_has_zero_cube_mean_and_named_cases():
    """The global pressure target follows the exact physical integral."""
    from pymhm.tetrahedral import TetraMesh, tetrahedron_quadrature

    data = _load_data()()
    mesh = TetraMesh.unit_cube()
    bary, weights = tetrahedron_quadrature(3)
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    assert (
        abs(
            mesh.volumes
            @ (data.pressure(points.reshape(-1, 3)).reshape(len(mesh.cells), -1) @ weights)
        )
        < 1e-16
    )
    with pytest.raises(ValueError, match="kind must"):
        _load_data()("unknown")
