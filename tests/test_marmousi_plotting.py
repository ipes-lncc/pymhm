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
