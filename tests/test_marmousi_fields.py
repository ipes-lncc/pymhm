"""Verify complex native-field replay and physical reference norms analytically."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def driver(monkeypatch):
    """Import the original example without requiring any native FEM dependency."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.marmousi_fields")


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_polynomial_replay_and_complex_physical_norms(driver, degree):
    """A complex affine increment has independent exact physical norm integrals."""
    x, y = np.meshgrid(
        np.linspace(0, 2, 2 * degree + 1), np.linspace(0, 3, 3 * degree + 1), indexing="ij"
    )
    exact = 1 + x + 2j * y
    fine = driver.PixelCGField(exact, degree, (0, 2, 0, 3))
    coarse = driver.PixelCGField(np.ones((3, 4), dtype=complex), 1, fine.bounds)
    points = np.array([[0, 0], [2, 3], [0.37, 0.81], [1.5, 2.5]])
    assert_allclose(fine.sample(points), 1 + points[:, 0] + 2j * points[:, 1], atol=2e-13)
    norms = driver.reference_difference(
        coarse, fine, np.full((2, 3), 2.0), np.full((2, 3), 3.0), omega=2, batch_size=2
    )
    # Integral |x+2iy|²=8+72=80; |1+x+2iy|²=26+72=98.
    assert_allclose(norms["pressure_difference"], np.sqrt(80), rtol=2e-13)
    assert_allclose(norms["reference_pressure_norm"], np.sqrt(98), rtol=2e-13)
    assert_allclose(norms["gradient_difference"], np.sqrt(30), rtol=2e-13)
    assert_allclose(norms["weighted_gradient_difference"], np.sqrt(15), rtol=2e-13)
    assert_allclose(norms["flux_difference"], np.sqrt(30) / 2, rtol=2e-13)
    assert_allclose(norms["graph_difference"], np.sqrt(15 + 4 * 80 / 3), rtol=2e-13)
    assert_allclose(norms["domain_area"], 6)


def test_nonlinear_pk_replay(driver):
    """P4 cardinal interpolation preserves a complex non-affine total-degree polynomial."""
    x, y = np.meshgrid(np.linspace(0, 4, 17), np.linspace(0, 2, 9), indexing="ij")
    field = driver.PixelCGField(x**4 + x * y**3 + 1j * (x**2 - y**2), 4, (0, 4, 0, 2))
    points = np.random.default_rng(73).uniform(size=(29, 2)) * [4, 2]
    x, y = points.T
    assert_allclose(field.sample(points), x**4 + x * y**3 + 1j * (x**2 - y**2), rtol=2e-13)


def test_fixed_cutout_preserves_global_pressure_norm(driver):
    """Exclude exact pixel regions only from derivative/graph integrals, not pressure."""
    x, y = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    fine = driver.PixelCGField(1 + x + 2j * y, 1, (0, 2, 0, 3))
    coarse = driver.PixelCGField(np.ones((3, 4), dtype=complex), 1, fine.bounds)
    material = np.ones((2, 3))
    norms = driver.reference_difference(
        coarse,
        fine,
        2 * material,
        3 * material,
        omega=2,
        gradient_cutout=(0, 1, 0, 1),
    )
    assert_allclose(norms["pressure_difference"], np.sqrt(80), rtol=2e-13)
    assert_allclose(norms["gradient_difference"], 5, rtol=2e-13)
    assert_allclose(norms["flux_difference"], 2.5, rtol=2e-13)
    assert_allclose(norms["graph_difference"], np.sqrt(12.5 + 4 * (80 - 5 / 3) / 3), rtol=2e-13)
    assert norms["gradient_domain_area"] == 5
    with pytest.raises(ValueError, match="pixel-aligned"):
        driver.reference_difference(
            coarse,
            fine,
            material,
            material,
            omega=2,
            gradient_cutout=(0, 0.5, 0, 1),
        )


def test_radial_mask_has_physical_boundary_and_shared_numerator_denominator(driver):
    """Use independently specified points and both complex norm denominators."""
    points = np.array([[[5000.0, 50.0], [5050.0, 50.0], [5000.0, 100.0], [5025.0, 50.0]]])
    keep = driver.radial_exclusion_mask(points, (5000.0, 50.0, 50.0))
    assert np.array_equal(keep, [[False, True, True, False]])
    pressure = np.array([[2 + 1j, 3 + 2j, -1 + 0.5j, 1 - 1j]])
    reference = np.full((1, 4), 1 + 2j)
    gradient = np.broadcast_to([3 + 1j, 2 - 2j], (1, 4, 2))
    exact_gradient = np.broadcast_to([1 + 2j, -1j], (1, 4, 2))
    actual = driver.acoustic_norm_contributions(
        pressure, reference, gradient, exact_gradient, np.array([2.0]), np.array([3.0]), 4.0, keep
    )
    assert_allclose(actual[0], abs(pressure - reference) ** 2)
    assert_allclose(actual[1], abs(reference) ** 2)
    assert_allclose(actual[2], [[0.0, 10.0, 10.0, 0.0]])
    assert_allclose(actual[3], [[0.0, 6.0, 6.0, 0.0]])
    assert_allclose(actual[6], 16 / 3 * abs(pressure - reference) ** 2 * keep)
    assert_allclose(actual[7], 16 / 3 * abs(reference) ** 2 * keep)
    with pytest.raises(ValueError, match="positive radius"):
        driver.radial_exclusion_mask(points, (5000.0, 50.0, 0.0))
    with pytest.raises(ValueError, match="mask"):
        driver.acoustic_norm_contributions(
            pressure,
            reference,
            gradient,
            exact_gradient,
            np.array([2.0]),
            np.array([3.0]),
            4.0,
            keep.astype(float),
        )


def test_circular_integration_matches_independent_disk_area_control(driver):
    """The affine gradient integral equals its constant integrand times disk-complement area."""
    n = 24
    axis = np.linspace(-1, 1, n + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    fine = driver.PixelCGField(x + 2j * y, 1, (-1, 1, -1, 1))
    zero = driver.PixelCGField(np.zeros_like(x, dtype=complex), 1, fine.bounds)
    material = np.ones((n, n))
    exact = 4 - np.pi * 0.4**2
    for order in (8, 12):
        actual = driver.reference_difference(
            zero,
            fine,
            material,
            material,
            omega=2.0,
            order=order,
            gradient_exclusion=(0.0, 0.0, 0.4),
        )
        assert abs(actual["gradient_domain_area"] - exact) < 0.004
        assert_allclose(
            actual["gradient_difference"] ** 2, 5 * actual["gradient_domain_area"], rtol=2e-13
        )
        assert actual["gradient_relative_difference"] == pytest.approx(1, abs=1e-14)
        assert_allclose(actual["pressure_difference"] ** 2, 20 / 3, rtol=2e-13)
    with pytest.raises(ValueError, match="one derivative exclusion"):
        driver.reference_difference(
            zero,
            fine,
            material,
            material,
            omega=2.0,
            gradient_exclusion=(0.0, 0.0, 0.4),
            gradient_cutout=(-1, 0, -1, 0),
        )


def test_archive_ownership_digests_and_geometry(driver, tmp_path):
    """MPI-owned shards reconstruct once and reject missing, repeated or altered nodes."""
    x, y = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    entries = []
    for rank, selection in enumerate((slice(0, 4), slice(4, 9))):
        path = tmp_path / f"rank{rank}.npz"
        points = coordinates[selection]
        np.savez(path, coordinates=points, pressure=points[:, 0] + 2j * points[:, 1])
        entries.append(
            {"archive": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    record = tmp_path / "record.json"
    metadata = {"degree": 2, "geometry": [1, 1], "bounds": [0, 2, 0, 2], "archives": entries}
    record.write_text(json.dumps(metadata))
    assert_allclose(driver.load_reference(record).nodes, x + 2j * y)
    metadata["archives"] = entries[:1]
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="cover"):
        driver.load_reference(record)
    metadata["archives"] = [entries[0], entries[0]]
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="exactly once"):
        driver.load_reference(record)
    metadata["archives"] = [{"archive": "rank0.npz", "sha256": "0" * 64}]
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest"):
        driver.load_reference(record)


def test_reference_replay_retains_executed_wider_complex_coefficients(driver, tmp_path):
    if np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("wider persisted mantissa requires a wider host dtype")
    coordinates = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    pressure = np.full(4, np.clongdouble(1 + 2j))
    pressure.real += np.longdouble(2) ** -60
    archive = tmp_path / "native.npz"
    np.savez(archive, coordinates=coordinates, pressure=pressure)
    record = tmp_path / "native.json"
    record.write_text(
        json.dumps(
            {
                "degree": 1,
                "geometry": [1, 1],
                "bounds": [0, 1, 0, 1],
                "archives": [
                    {
                        "archive": archive.name,
                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    field = driver.load_reference(record)
    assert field.nodes.dtype == pressure.dtype
    assert (field.nodes.real > np.longdouble(1)).all()
    assert np.array_equal(field.nodes.ravel(), pressure)


def test_invalid_contracts_and_zero_denominator(driver):
    """Reject incompatible geometry and preserve absolute errors against a zero field."""
    with pytest.raises(ValueError, match="complete"):
        driver.PixelCGField(np.zeros((4, 4)), 2, (0, 1, 0, 1))
    fine = driver.PixelCGField(np.zeros((3, 3)), 2, (0, 1, 0, 1))
    coarse = driver.PixelCGField(np.ones((2, 2)), 1, fine.bounds)
    result = driver.reference_difference(coarse, fine, np.ones((1, 1)), np.ones((1, 1)), omega=1)
    assert result["pressure_relative_difference"] is None
    assert_allclose(result["pressure_difference"], 1)
    with pytest.raises(ValueError, match="outside"):
        fine.sample(np.array([[-0.1, 0.5]]))
    with pytest.raises(ValueError, match="finite two"):
        fine.sample(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="half"):
        fine.coefficients(np.array([[0, 0]]), 2)
    with pytest.raises(ValueError, match="positive"):
        driver.reference_difference(coarse, fine, np.zeros((1, 1)), np.ones((1, 1)), omega=1)
    with pytest.raises(ValueError, match="same"):
        driver.reference_difference(
            coarse,
            driver.PixelCGField(np.zeros((3, 3)), 1, fine.bounds),
            np.ones((1, 1)),
            np.ones((1, 1)),
            omega=1,
        )
