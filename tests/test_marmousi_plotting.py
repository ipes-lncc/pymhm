"""Check that acoustic plotting samples retain actual broken macro values."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.fixture
def driver(monkeypatch):
    """Load the sampling helpers without importing the optional plotting backend."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.plot_marmousi_mhm")


def test_pixel_centre_values_keep_complex_macro_jumps(driver) -> None:
    """Four different constant incident fields remain distinct in the raster samples."""
    mesh = CartesianMacroMesh(2, 2, (0, 40, 0, 40))
    constants = np.arange(1, 5) * (1 + 2j)
    candidate = driver.BrokenQField(np.repeat(constants[:, None], 4, axis=1), mesh, 1, 1)
    reference = driver.PixelCGField(np.full((3, 3), 3 + 5j), 1, mesh.bounds)
    values, truth = driver.centre_fields(candidate, reference, batch_size=3)
    # Constant reproduction evaluates a native polynomial partition of unity;
    # its last rounding bit is independent of retaining the macro jumps.
    roundoff = 4 * np.finfo(float).eps
    np.testing.assert_allclose(values, constants.reshape(2, 2).T, rtol=roundoff, atol=0)
    np.testing.assert_allclose(truth, np.full((2, 2), 3 + 5j), rtol=roundoff, atol=0)


@pytest.mark.parametrize("perturbed", ["candidate", "reference"])
def test_pixel_centre_difference_preserves_executed_extended_mantissa(driver, perturbed) -> None:
    """Below-double perturbations survive each field's own executed sampling basis.

    Q1 and P1 tabulators have different partition-of-unity roundoff. Measuring
    the change through the same tabulator isolates preservation of coefficient
    mantissa bits from that independent representation error.
    """
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("the host has no wider floating-point mantissa")
    mesh = CartesianMacroMesh(2, 2, (0, 40, 0, 40))
    increment = np.ldexp(np.longdouble(1), -60)
    candidate = driver.BrokenQField(np.ones((4, 4), dtype=np.clongdouble), mesh, 1, 1)
    reference = driver.PixelCGField(np.ones((3, 3), dtype=np.clongdouble), 1, mesh.bounds)
    baseline = driver.centre_fields(candidate, reference, batch_size=1)
    coefficients = candidate.pressure if perturbed == "candidate" else reference.nodes
    coefficients += increment
    sampled = driver.centre_fields(candidate, reference, batch_size=1)
    selected = 0 if perturbed == "candidate" else 1
    assert all(value.dtype == np.dtype(np.clongdouble) for value in sampled)
    np.testing.assert_array_equal(
        sampled[selected] - baseline[selected], np.full((2, 2), increment)
    )
    np.testing.assert_array_equal(sampled[1 - selected], baseline[1 - selected])
    coefficients[...] = coefficients.astype(np.complex128)
    narrowed = driver.centre_fields(candidate, reference, batch_size=1)
    np.testing.assert_array_equal(narrowed[selected], baseline[selected])


def test_profiles_retain_two_exact_incident_values_at_each_macro_face(driver) -> None:
    """The shared coordinate carries two independent values rather than a mean."""
    mesh = CartesianMacroMesh(2, 2, (0, 40, 0, 40))
    constants = np.arange(1, 5) * (1 + 2j)
    candidate = driver.BrokenQField(np.repeat(constants[:, None], 4, axis=1), mesh, 1, 1)
    x, values = driver.profile_segments(candidate, 5, samples_per_macro=5)
    assert x[0, -1] == x[1, 0] == 20
    np.testing.assert_array_equal(values[0], np.full(5, constants[0]))
    np.testing.assert_array_equal(values[1], np.full(5, constants[1]))


def test_horizontal_profile_interface_requires_an_explicit_side_convention(driver) -> None:
    """A profile with two horizontal incidents is rejected instead of silently merged."""
    mesh = CartesianMacroMesh(2, 2, (0, 40, 0, 40))
    candidate = driver.BrokenQField(np.ones((4, 4), dtype=complex), mesh, 1, 1)
    with pytest.raises(ValueError, match="between horizontal"):
        driver.profile_segments(candidate, 20)


@pytest.mark.visualization
def test_recorded_marmousi_norm_plots_preserve_square_cutout_without_fields(tmp_path, monkeypatch):
    """Recorded polynomial increments remain distinct from field reconstruction."""
    import json

    plt = pytest.importorskip("matplotlib.pyplot")
    owner = importlib.import_module("examples.plot_marmousi")
    source = Path(__file__).resolve().parents[1] / "examples/results/marmousi"
    captured = []
    original_close = plt.close

    def capture(figure):
        """Inspect each plotted physical norm before releasing the figure."""
        captured.append(figure)
        original_close(figure)

    def forbidden(*args, **kwargs):
        """Retained measurements must not require missing coefficient arrays."""
        raise AssertionError("unexpected field read")

    monkeypatch.setattr(plt, "close", capture)
    monkeypatch.setattr(owner, "load_reference", forbidden)
    result = owner.plot_recorded_results(source, tmp_path)
    records = json.loads((source / "classical-convergence.json").read_text())["references"]
    expected = [
        100 * row["increment"]["pressure_difference"] / row["increment"]["reference_pressure_norm"]
        for row in records[1:]
    ]
    np.testing.assert_allclose(
        captured[0].axes[0].lines[0].get_ydata(), expected, rtol=1e-12, atol=0
    )
    assert result["gradient_cutout_m"] == [4975.0, 5025.0, 25.0, 75.0]
    assert len(result["figure_sha256"]) == 4
    assert all((tmp_path / name).is_file() for name in result["figure_sha256"])


@pytest.mark.visualization
@pytest.mark.parametrize("defect", ["cutout", "acquisition"])
def test_recorded_marmousi_plot_rejects_changed_measure_or_acquisition(tmp_path, defect):
    """A changed source exclusion or unrelated reference cannot relabel recorded norms."""
    import json
    import shutil

    pytest.importorskip("matplotlib.pyplot")
    owner = importlib.import_module("examples.plot_marmousi")
    source = Path(__file__).resolve().parents[1] / "examples/results/marmousi"
    copied = tmp_path / "records"
    shutil.copytree(source, copied)
    path = copied / "classical-convergence.json"
    payload = json.loads(path.read_text())
    if defect == "cutout":
        payload["references"][1]["increment"]["gradient_cutout"] = [5000.0, 50.0, 50.0]
    else:
        payload["references"][3]["reference_record_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="square cutout|different acquisition"):
        owner.plot_recorded_results(copied, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()
