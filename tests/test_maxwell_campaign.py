"""Independent derivatives and physical norm checks for Maxwell analytical data."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm._legacy.models.waves.maxwell import MaxwellStepper
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


def differentiated(field, points):
    """Differentiate an analytic field by complex-step in every Cartesian direction."""
    return np.stack(
        [np.imag(field(points + 1e-25j * axis)) / 1e-25 for axis in np.eye(points.shape[1])],
        axis=-1,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_cavity_maxwell_equations_pec_and_staggered_norms(dimension, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    data = importlib.import_module("examples.maxwell_data")
    norms_module = importlib.import_module("examples.maxwell_norms")
    mode = data.CavityMode(dimension)
    points = np.random.default_rng(735).uniform(0.1, 0.9, (7, dimension))
    de = differentiated(mode.electric_shape, points)
    dh = differentiated(mode.magnetic_shape, points)
    if dimension == 2:
        ce = np.column_stack((de[:, 0, 1], -de[:, 0, 0]))
        ch = (dh[:, 1, 0] - dh[:, 0, 1])[:, None]
    else:
        ce = np.column_stack(
            (de[:, 2, 1] - de[:, 1, 2], de[:, 0, 2] - de[:, 2, 0], de[:, 1, 0] - de[:, 0, 1])
        )
        ch = np.column_stack(
            (dh[:, 2, 1] - dh[:, 1, 2], dh[:, 0, 2] - dh[:, 2, 0], dh[:, 1, 0] - dh[:, 0, 1])
        )
        assert_allclose(np.trace(de, axis1=1, axis2=2), 0, atol=1e-14)
    assert_allclose(np.trace(dh, axis1=1, axis2=2), 0, atol=1e-14)
    assert_allclose(ce, -mode.omega * mode.magnetic_shape(points), atol=1e-14)
    assert_allclose(ch, -mode.omega * mode.electric_shape(points), atol=2e-14)
    for axis in range(dimension):
        for value in (0, 1):
            boundary = points.copy()
            boundary[:, axis] = value
            e = mode.electric_shape(boundary)
            tangential = e if dimension == 2 else e[:, np.arange(3) != axis]
            assert_allclose(tangential, 0, atol=1e-15)
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    with MaxwellStepper(mesh, time_step=0.001, quadrature_order=6) as stepper:
        stepper.initialize(mode.electric_shape)
        solution = stepper.advance()
        norms = norms_module.MaxwellNorms(solution, mode, order=6).measure(solution)
        independent = solution.l2_errors(
            lambda p: mode.electric(solution.electric_time, p),
            lambda p: mode.magnetic(solution.magnetic_time, p),
            order=6,
        )
        assert_allclose([norms["electric_l2"], norms["magnetic_l2"]], independent, atol=2e-15)
        assert norms["combined_hcurl"] >= norms["combined_l2"] > 0


def test_time_refinement_against_exact_constrained_semidiscrete_evolution(monkeypatch):
    """An exact matrix exponential separates temporal error from spatial consistency."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    campaign = importlib.import_module("examples.maxwell_campaign")
    first, second = campaign.temporal_row(0.004), campaign.temporal_row(0.002)
    assert 3.9 < first["combined_error_l2"] / second["combined_error_l2"] < 4.1
    assert (
        max(first["modified_energy_relative_drift"], second["modified_energy_relative_drift"])
        < 2e-14
    )


def test_nanoguide_common_partition_norms_and_staggered_times(tmp_path, monkeypatch):
    """Non-nested Q2 archives recover analytical norms and reject mismatched times."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    replay = importlib.import_module("examples.maxwell_nanoguide_results")

    def archive(n, offset, name, electric_time=0.7):
        """Sample one global quadratic at the actual discontinuous cell nodes."""
        nodes = np.array([(x, y) for y in (0, 0.5, 1) for x in (0, 0.5, 1)])
        cells = np.array([[(x, y) for x in range(n)] for y in range(n)])
        xy = (cells[:, :, None, :] + nodes) * (10 / n)
        x, y = xy[..., 0], xy[..., 1]
        fields = np.stack((1 + x**2 / 100, 2 + y**2 / 100, 3 + x * y / 100), axis=-1)
        fields += np.asarray(offset)
        path = tmp_path / name
        np.savez(
            path,
            degree=2,
            bounds=[0, 10, 0, 10],
            electric=fields[..., 0],
            magnetic=fields[..., 1:],
            electric_time=electric_time,
            magnetic_time=0.65,
        )
        return path

    first = archive(2, [0.1, -0.2, 0.3], "first.npz")
    second = archive(3, [0, 0, 0], "second.npz")
    result = replay.compare(first, second)
    assert result["integration_grid"] == 6
    assert_allclose(result["absolute_l2"], [1, 2, 3], rtol=2e-14)
    assert_allclose(result["reference_l2"], np.sqrt(100 * np.array([28 / 15, 83 / 15, 191 / 18])))
    wrong_time = archive(3, [0, 0, 0], "wrong-time.npz", electric_time=0.8)
    with pytest.raises(ValueError, match="different electric_time"):
        replay.compare(first, wrong_time)


def test_nanoguide_sampling_preserves_discontinuous_traces(monkeypatch):
    """Each side of a fine-cell interface retains its independent polynomial."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    replay = importlib.import_module("examples.maxwell_nanoguide_results")
    values = np.ones((2, 2, 9, 3))
    values[:, 1] = np.array([3, -2, 4])
    points = np.array([[np.nextafter(5.0, 0), 2], [5.0, 2], [10.0, 2]])
    assert_allclose(replay.sample(values, points), [[1, 1, 1], [3, -2, 4], [3, -2, 4]])
