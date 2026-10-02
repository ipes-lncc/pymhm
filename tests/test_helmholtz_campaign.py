"""Independent differential checks of the analytical wave and PML experiment data."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


def test_archived_q4_profile_preserves_interface_values(monkeypatch):
    """Quartic replay preserves the two distinct limits at a fine interface."""
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    plotting = importlib.import_module("examples.plot_helmholtz")
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    first = np.column_stack((x.ravel() / 2, y.ravel()))
    second = first + [0.5, 0]
    data = {
        "points": np.concatenate((first, second)),
        "values": np.concatenate(
            (first[:, 0] ** 4 + first[:, 1], second[:, 0] ** 4 + second[:, 1] + 1)
        ),
    }
    profiles = plotting.q4_profile(data, 0.473)
    assert len(profiles) == 2
    for offset, (coordinate, value) in enumerate(profiles):
        assert_allclose(value, coordinate**4 + 0.473 + offset, atol=2e-14)
    assert_allclose(profiles[1][1][0] - profiles[0][1][-1], 1, atol=2e-14)
    with pytest.raises(ValueError, match="explicit side"):
        plotting.q4_profile(data, 0)


def test_stability_threshold_keeps_rejected_resolutions(monkeypatch):
    """A resonant grid inside a seemingly successful suffix invalidates that suffix."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    diagnostic = importlib.import_module("examples.helmholtz_threshold")
    rows = [{"n": n, "ratio": ratio} for n, ratio in ((22, 3.2), (23, 2.6), (32, 1.4), (64, 1.1))]
    assert diagnostic.sampled_threshold(rows)["sampled_suffix_threshold"] == 1 / 23
    checked = diagnostic.sampled_threshold(rows, [{"n": 30}])
    assert checked["sampled_suffix_threshold"] == 1 / 32
    assert checked["transition_bracket"] == [1 / 32, 1 / 30]
    assert checked["accepted_suffix_resolutions"] == [32, 64]
    assert diagnostic.sampled_threshold(rows, [{"n": 64}])["sampled_suffix_threshold"] is None
    assert diagnostic.sampled_threshold([])["sampled_suffix_threshold"] is None


def test_stability_acquisition_checkpoints_failed_attempts(tmp_path, monkeypatch):
    """A rejected solve is persisted and remains a barrier on a subsequent resume."""
    from pymhm.solvers import LinearSolveError

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    campaign = importlib.import_module("examples.helmholtz_stability")

    def rejected(*args, **kwargs):
        """Represent a solver rejection without accepting a fabricated field."""
        raise LinearSolveError("local physical residual criterion failed")

    monkeypatch.setattr(campaign, "solve_helmholtz", rejected)
    with pytest.raises(LinearSolveError, match="physical residual"):
        campaign.run(tmp_path, 0, 10, [1], 1)
    path = tmp_path / "ell0-frequency10.json"
    record = json.loads(path.read_text())
    assert record["sampled_threshold"]["rejected_resolutions"] == [1]
    assert record["largest_suffix_H_star"] is None
    assert record["rows"] == []
    before = path.read_bytes()
    campaign.run(tmp_path, 0, 10, [1], 1)
    assert path.read_bytes() == before
    record["acquisition_batches"] = [{"source_sha256": record.pop("source_sha256")}]
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="fresh --output"):
        campaign.run(tmp_path, 0, 10, [1], 1)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["plane", "hankel", "pml"])
def test_analytical_wave_and_transformed_balance(kind, monkeypatch):
    """Finite differences verify gradients and the original/transformed PDE separately."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    campaign = importlib.import_module("examples.helmholtz_campaign")
    wave = campaign.AcousticWave(10.0, kind=kind)
    points = np.array([[0.21, 0.27], [0.73, 0.64], [0.94, 0.17]])
    h = 1e-5
    laplacian = np.zeros(len(points), dtype=complex)
    for axis in range(2):
        shift = np.eye(2)[axis] * h
        derivative = (wave.pressure(points + shift) - wave.pressure(points - shift)) / (2 * h)
        assert_allclose(derivative, wave.gradient(points)[:, axis], atol=3e-8)
        plus, minus = wave.gradient(points + shift), wave.gradient(points - shift)
        if kind == "pml":
            plus *= wave.stretch(points + shift)[:, ::-1] / wave.stretch(points + shift)
            minus *= wave.stretch(points - shift)[:, ::-1] / wave.stretch(points - shift)
        laplacian += (plus[:, axis] - minus[:, axis]) / (2 * h)
    weight = np.prod(wave.stretch(points), axis=1) if kind == "pml" else 1
    assert_allclose(laplacian + wave.omega**2 * weight * wave.pressure(points), 0, atol=2e-6)
    normals = np.tile([0.6, 0.8], (len(points), 1))
    assert_allclose(
        wave.absorbing(points, normals),
        np.sum(wave.gradient(points) * normals, axis=1) - 1j * wave.omega * wave.pressure(points),
    )
