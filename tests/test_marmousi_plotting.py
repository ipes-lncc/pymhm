"""Check that acoustic plotting samples retain actual broken macro values."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from pymhm.quadrilateral import CartesianMacroMesh


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
    np.testing.assert_array_equal(values, constants.reshape(2, 2).T)
    np.testing.assert_array_equal(truth, np.full((2, 2), 3 + 5j))


def test_pixel_centre_difference_preserves_executed_extended_mantissa(driver) -> None:
    """A physical sample difference below double epsilon survives field-buffer assembly."""
    if np.finfo(np.longdouble).nmant <= np.finfo(float).nmant:
        pytest.skip("the host has no wider floating-point mantissa")
    mesh = CartesianMacroMesh(2, 2, (0, 40, 0, 40))
    increment = np.ldexp(np.longdouble(1), -60)
    candidate = driver.BrokenQField(
        np.full((4, 4), np.clongdouble(1) + increment, dtype=np.clongdouble), mesh, 1, 1
    )
    reference = driver.PixelCGField(np.ones((3, 3), dtype=np.clongdouble), 1, mesh.bounds)
    values, truth = driver.centre_fields(candidate, reference, batch_size=1)
    assert values.dtype == truth.dtype == np.dtype(np.clongdouble)
    np.testing.assert_array_equal(values - truth, np.full((2, 2), increment))


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
