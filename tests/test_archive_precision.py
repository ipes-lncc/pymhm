"""Cross-platform serialization of native extended-accumulation field coefficients."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def api(monkeypatch):
    """Load the original example encoder without changing the installed package path."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.archive_precision")


def test_precision_pair_preserves_native_values_without_platform_specific_npz(tmp_path, api):
    """Sub-double-ulp remainders survive an archive containing only portable float64."""
    values = np.array([0.3, -2.0, 1e-100, 1e100], dtype=np.longdouble)
    values = np.nextafter(values, np.full_like(values, np.inf))
    high, low, tail = api.split_precision(values)
    path = tmp_path / "coefficients.npz"
    np.savez(path, high=high, low=low, tail=tail)
    with np.load(path) as arrays:
        assert all(arrays[key].dtype == np.dtype(np.float64) for key in arrays)
        np.testing.assert_array_equal(
            api.restore_precision(arrays["high"], arrays["low"], arrays["tail"]), values
        )
    if np.finfo(np.longdouble).eps < np.finfo(float).eps:
        assert np.any(low != 0)


def test_precision_split_when_native_extended_equals_double(monkeypatch, api):
    """The high component never aliases the working remainder on double-only platforms."""
    monkeypatch.setattr(np, "longdouble", np.float64)
    original = np.array([0.3, -2.0, 1e100])
    parts = api.split_precision(original)
    np.testing.assert_array_equal(parts[0], original)
    np.testing.assert_array_equal(api.restore_precision(*parts), original)
    assert not np.any(parts[1])
    assert not np.any(parts[2])


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [1j]])
def test_precision_archive_rejects_nonfinite_or_complex_input(values, api):
    """An archive never silently discards an imaginary part or nonfinite coefficient."""
    with pytest.raises(ValueError, match="finite real"):
        api.split_precision(np.asarray(values))


@pytest.mark.parametrize("high,low", [([1], [1, 2]), ([1j], [0]), ([0], [np.nan])])
def test_precision_restore_requires_matching_real_finite_components(high, low, api):
    """The pair has a single unambiguous shape and numeric domain."""
    with pytest.raises(ValueError, match="equal shape"):
        api.restore_precision(np.asarray(high), np.asarray(low))
