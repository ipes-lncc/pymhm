"""Lightweight checks for the independently assembled Taylor--Hood example.

Run explicitly with ``python -m pytest examples/test_spe10_taylor_hood.py``.
The distributed wheel has no dependency on this example or on DOLFINx.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from examples import solve_spe10_taylor_hood as driver


def polynomial_field(nx: int, ny: int) -> driver.TaylorHoodField:
    """Sample an exactly represented quadratic velocity and affine pressure."""
    x, y = np.meshgrid(np.linspace(0, 1, 2 * nx + 1), np.linspace(0, 1, 2 * ny + 1))
    velocity = np.stack((1 + x + x * y + y**2, 2 - x**2 + 2 * y), axis=-1)
    x, y = np.meshgrid(np.linspace(0, 1, nx + 1), np.linspace(0, 1, ny + 1))
    return driver.TaylorHoodField(velocity, 3 + 4 * x - 5 * y)


def test_exact_polynomial_evaluation_and_nested_norm() -> None:
    """Check both triangle orientations and the physical-area quadrature."""
    field = polynomial_field(2, 3)
    points = np.vstack(
        (np.random.default_rng(17).random((100, 2)), [[0, 0], [1, 1], [0, 1], [1, 0]])
    )
    x, y = points.T
    velocity, pressure = field.evaluate(points * [1200, 2200])
    np.testing.assert_allclose(
        velocity, np.column_stack((1 + x + x * y + y**2, 2 - x**2 + 2 * y)), rtol=0, atol=3e-15
    )
    np.testing.assert_allclose(pressure, 3 + 4 * x - 5 * y, rtol=0, atol=3e-15)
    differences = driver.difference(polynomial_field(4, 6), field)
    assert differences["velocity_l2"] < 3e-12
    assert differences["pressure_l2"] < 3e-12
    constant = driver.TaylorHoodField(np.broadcast_to([3.0, 4.0], (13, 9, 2)), np.full((7, 5), 2.0))
    zero = driver.TaylorHoodField(np.zeros_like(field.velocity), np.zeros_like(field.pressure))
    differences = driver.difference(constant, zero)
    np.testing.assert_allclose(
        [differences["velocity_l2"], differences["pressure_l2"]],
        np.array([5.0, 2.0]) * np.sqrt(1200 * 2200),
        rtol=32 * np.finfo(float).eps,
        atol=0,
    )
    np.testing.assert_allclose(
        [differences["velocity_relative"], differences["pressure_relative"]], [1.0, 1.0]
    )


def test_non_nested_diagonal_grid_rejected() -> None:
    """Reject anisotropic refinements that cross a coarse triangle diagonal."""
    with pytest.raises(ValueError, match="nested"):
        driver.difference(polynomial_field(4, 9), polynomial_field(2, 3))


def test_physical_gradients_and_shifted_bounds() -> None:
    """Polynomial Jacobians must transform with the physical rectangle, in both triangles."""
    data = polynomial_field(2, 3)
    field = driver.TaylorHoodField(data.velocity, data.pressure, (-2.0, 4.0, 3.0, 5.0))
    xy = np.random.default_rng(29).random((61, 2))
    x, y = xy.T
    points = xy * [6.0, 2.0] + [-2.0, 3.0]
    gradient, pressure_gradient = field.gradient(points)
    expected = np.stack(
        (
            np.column_stack(((1 + y) / 6, (x + 2 * y) / 2)),
            np.column_stack((-2 * x / 6, np.ones(len(x)))),
        ),
        axis=1,
    )
    np.testing.assert_allclose(gradient, expected, atol=3e-14)
    np.testing.assert_allclose(
        pressure_gradient, np.broadcast_to([4 / 6, -5 / 2], (len(x), 2)), atol=2e-14
    )
    with pytest.raises(ValueError, match="physical bounds"):
        driver.difference(field, data)


def test_outside_reference_domain_rejected() -> None:
    """Avoid extrapolating the reference beyond its physical domain."""
    with pytest.raises(ValueError, match="outside"):
        polynomial_field(2, 3).evaluate(np.array([[-1.0, 1.0]]))


def test_zero_reference_and_physical_archive(tmp_path: Path) -> None:
    """A zero baseline cannot hide a nonzero field or gradient discrepancy."""
    data = polynomial_field(2, 3)
    field = driver.TaylorHoodField(data.velocity, data.pressure, (-2.0, 4.0, 3.0, 5.0))
    zero = driver.TaylorHoodField(
        np.zeros_like(data.velocity), np.zeros_like(data.pressure), field.bounds
    )
    assert all(value == 0 for value in driver.difference(zero, zero, gradients=True).values())
    errors = driver.difference(zero, field, gradients=True)
    for name in ("velocity_relative", "pressure_relative", "velocity_h1_relative"):
        assert np.isinf(errors[name])
    archive = tmp_path / "field.npz"
    np.savez(archive, velocity=field.velocity, pressure=field.pressure, bounds=field.bounds)
    restored = driver.load_field(archive)
    assert restored.bounds == field.bounds
    np.testing.assert_array_equal(restored.velocity, field.velocity)


def test_aligned_corner_cutouts_integrate_the_exact_physical_area() -> None:
    """Corner exclusion is geometric, with exact retained area and polynomial gradients."""
    fine = driver.TaylorHoodField(
        np.broadcast_to([3.0, 4.0], (9, 9, 2)), np.full((5, 5), 2.0), (0.0, 6.0, 0.0, 2.0)
    )
    coarse = driver.TaylorHoodField(np.zeros((5, 5, 2)), np.zeros((3, 3)), fine.bounds)
    differences = driver.difference(fine, coarse, corner_cutout=0.25)
    np.testing.assert_allclose(
        [differences["velocity_l2"], differences["pressure_l2"]],
        np.array([5.0, 2.0]) * np.sqrt(12 * (1 - 2 * 0.25**2)),
        atol=1e-13,
    )
    for cut, message in ((0.1, "align"), (0.5, "lie in"), (-0.1, "lie in")):
        with pytest.raises(ValueError, match=message):
            driver.difference(fine, coarse, corner_cutout=cut)


@pytest.fixture
def archived_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Create a real coefficient archive and its independently hashed record."""
    monkeypatch.setattr(driver, "OUTPUT", tmp_path)
    monkeypatch.setattr(driver, "COEFFICIENTS", tmp_path / "coefficients")
    driver.COEFFICIENTS.mkdir()
    field = polynomial_field(2, 3)
    coefficients = driver.COEFFICIENTS / "taylor-hood-2x3.npz"
    samples = tmp_path / "taylor-hood-2x3.npz"
    np.savez_compressed(coefficients, velocity=field.velocity, pressure=field.pressure)
    np.savez_compressed(samples, velocity=np.array([1.0]))
    record = {
        "mesh_shape": [2, 3],
        "archive": samples.name,
        "sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
        "coefficient_sha256": hashlib.sha256(coefficients.read_bytes()).hexdigest(),
    }
    metadata = samples.with_suffix(".json")
    metadata.write_text(json.dumps(record))
    return coefficients, metadata


def test_verified_archive_roundtrip(archived_reference: tuple[Path, Path]) -> None:
    """Use the stored coefficients only after their provenance is verified."""
    field, record = driver.load_archived_reference((2, 3))
    assert record["mesh_shape"] == [2, 3]
    np.testing.assert_array_equal(field.velocity, polynomial_field(2, 3).velocity)


@pytest.mark.parametrize(
    "invalid", ["coefficient_hash", "sample_hash", "metadata_shape", "array_shape", "nonfinite"]
)
def test_invalid_archive_rejected(archived_reference: tuple[Path, Path], invalid: str) -> None:
    """Reject changed files, inconsistent dimensions and nonfinite coefficients."""
    coefficients, metadata = archived_reference
    record = json.loads(metadata.read_text())
    if invalid == "coefficient_hash":
        coefficients.write_bytes(coefficients.read_bytes() + b"changed")
    elif invalid == "sample_hash":
        record["sha256"] = "invalid"
    elif invalid == "metadata_shape":
        record["mesh_shape"] = [3, 2]
    else:
        field = polynomial_field(2, 3)
        values = field.velocity.copy()
        if invalid == "array_shape":
            values = values[:-1]
        else:
            values[0, 0, 0] = np.nan
        np.savez_compressed(coefficients, velocity=values, pressure=field.pressure)
        record["coefficient_sha256"] = hashlib.sha256(coefficients.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="archive|shape|finite|checksum"):
        driver.load_archived_reference((2, 3))


def test_patch_cannot_publish_mhm_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject incompatible input before assembling a finite-element problem."""
    monkeypatch.setattr(
        "sys.argv", ["solve_spe10_taylor_hood.py", "--patch", "--mhm", "unused.npz"]
    )
    with pytest.raises(SystemExit, match="2"):
        driver.main()
