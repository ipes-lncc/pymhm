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
