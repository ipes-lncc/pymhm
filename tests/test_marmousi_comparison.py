"""Verify nonzero broken tensor/polynomial acoustic comparison integrals."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose

from pymhm.fem.scalar.quadrilateral import qk_space
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.fixture
def driver(monkeypatch):
    """Load the original comparison without optional native FEM imports."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.marmousi_comparison")


def fields(driver):
    """Construct independent non-affine Q3 and P4 data with macro pressure jumps."""
    mesh = CartesianMacroMesh(2, 3, (0, 2, 0, 3))
    pressure = []
    for cell in range(6):
        _, points = qk_space(mesh.submesh(cell, 2), 3)
        x, y = points.T
        pressure.append(x**3 * y**3 + 1j * x**2 * y + 0.1j * cell)
    candidate = driver.BrokenQField(np.asarray(pressure), mesh, 2, 3)
    x, y = np.meshgrid(np.linspace(0, 2, 9), np.linspace(0, 3, 13), indexing="ij")
    reference = driver.PixelCGField(1 + x**2 + 2j * y**2, 4, mesh.bounds)
    return candidate, reference


def test_broken_tensor_vs_simplex_physical_integrals(driver):
    """Tensor Gauss integration of explicit polynomials independently checks common triangles."""
    candidate, reference = fields(driver)
    density = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    bulk = density + 1
    actual = driver.mhm_difference(
        candidate,
        reference,
        density,
        bulk,
        omega=2,
        order=8,
        batch_size=2,
        gradient_cutout=(0, 1, 0, 1),
    )
    t, w = leggauss(8)
    t, w = (t + 1) / 2, w / 2
    expected = dict.fromkeys(("pressure", "gradient", "weighted_gradient", "flux", "graph"), 0.0)
    denominators = dict(expected)
    for ix in range(2):
        for iy in range(3):
            x, y = np.meshgrid(ix + t, iy + t, indexing="ij")
            measure = w[:, None] * w[None, :]
            p = x**3 * y**3 + 1j * x**2 * y + 0.1j * (ix + 2 * iy)
            exact = 1 + x**2 + 2j * y**2
            gradient = np.stack((3 * x**2 * y**3 + 2j * x * y, 3 * x**3 * y**2 + 1j * x**2))
            exact_gradient = np.stack((2 * x + 0j, 4j * y))
            p_squared = [abs(p - exact) ** 2, abs(exact) ** 2]
            g_squared = [
                np.sum(abs(gradient - exact_gradient) ** 2, axis=0),
                np.sum(abs(exact_gradient) ** 2, axis=0),
            ]
            rho, kappa = density[ix, iy], bulk[ix, iy]
            for totals, ps, gs in zip((expected, denominators), p_squared, g_squared, strict=True):
                totals["pressure"] += float(np.sum(measure * ps))
                if (ix, iy) != (0, 0):
                    totals["gradient"] += float(np.sum(measure * gs))
                    totals["weighted_gradient"] += float(np.sum(measure * gs / rho))
                    totals["flux"] += float(np.sum(measure * gs / rho**2))
                    totals["graph"] += float(np.sum(measure * (gs / rho + 4 * ps / kappa)))
    for name in expected:
        assert_allclose(actual[f"{name}_difference"], np.sqrt(expected[name]), rtol=3e-14)
        assert_allclose(actual[f"reference_{name}_norm"], np.sqrt(denominators[name]), rtol=3e-14)
    assert actual["common_triangles"] == 48
    assert actual["domain_area"] == 6
    assert actual["gradient_domain_area"] == 5


def test_threaded_order_and_quadrature(driver):
    """Parallel batch order is bitwise stable and two sufficient quadratures agree."""
    candidate, reference = fields(driver)
    material = np.ones((2, 3))
    serial = driver.mhm_difference(candidate, reference, material, material, omega=1, batch_size=1)
    threaded = driver.mhm_difference(
        candidate, reference, material, material, omega=1, batch_size=1, workers=2
    )
    assert serial == threaded
    higher = driver.mhm_difference(
        candidate, reference, material, material, omega=1, batch_size=1, order=10
    )
    for key in serial:
        if key.endswith("difference") or key.endswith("norm"):
            assert_allclose(serial[key], higher[key], rtol=2e-14)


def test_radial_domain_matches_independent_affine_disk_integral(driver):
    """Keep full pressure integration separate from one common derivative disk."""
    mesh = CartesianMacroMesh(2, 2, (-1, 1, -1, 1))
    values = []
    for cell in range(4):
        _, points = qk_space(mesh.submesh(cell, 8), 1)
        values.append(points[:, 0] + 2j * points[:, 1])
    candidate = driver.BrokenQField(np.asarray(values), mesh, 8, 1)
    axis = np.linspace(-1, 1, 9)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    reference = driver.PixelCGField(2 * (x + 2j * y), 1, mesh.bounds)
    material = np.ones((8, 8))
    for order in (8, 12):
        actual = driver.mhm_difference(
            candidate,
            reference,
            material,
            material,
            omega=2.0,
            order=order,
            gradient_exclusion=(0.0, 0.0, 0.4),
        )
        assert abs(actual["gradient_domain_area"] - (4 - np.pi * 0.4**2)) < 0.004
        assert_allclose(
            actual["gradient_difference"] ** 2, 5 * actual["gradient_domain_area"], rtol=3e-13
        )
        assert_allclose(
            actual["reference_gradient_norm"] ** 2, 20 * actual["gradient_domain_area"], rtol=3e-13
        )
        assert actual["gradient_relative_difference"] == pytest.approx(0.5, abs=1e-14)
        assert_allclose(actual["pressure_difference"] ** 2, 20 / 3, rtol=3e-13)
    with pytest.raises(ValueError, match="one derivative exclusion"):
        driver.mhm_difference(
            candidate,
            reference,
            material,
            material,
            omega=2.0,
            gradient_exclusion=(0.0, 0.0, 0.4),
            gradient_cutout=(-1, 0, -1, 0),
        )


def test_archive_digest_and_geometry(driver, tmp_path, monkeypatch):
    """Load the executed local coefficient vectors only with their exact macro geometry."""
    candidate, _ = fields(driver)
    archive = tmp_path / "mhm.npz"
    record = tmp_path / "mhm.json"
    np.savez(
        archive,
        pressure=candidate.pressure,
        macro_points=candidate.mesh.points,
        macro_cells=candidate.mesh.cells,
    )
    data = dict(
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        macro_shape=[2, 3],
        bounds=candidate.mesh.bounds,
        local_refinement=2,
        local_degree=3,
    )
    record.write_text(json.dumps(data))
    original_read_bytes = Path.read_bytes

    def bounded_read_bytes(path):
        """Prevent an archive-sized extra buffer while preserving ordinary source reads."""
        if path.suffix == ".npz":
            pytest.fail("field archive digests must stream rather than allocate the full file")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", bounded_read_bytes)
    assert np.array_equal(driver.BrokenQField.load(record).pressure, candidate.pressure)
    data["macro_shape"] = [3, 2]
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="geometry"):
        driver.BrokenQField.load(record)
    data["archive_sha256"] = "0" * 64
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest"):
        driver.BrokenQField.load(record)


def test_incompatible_partitions_and_data(driver):
    """Reject unrepresented interfaces and keep zero-denominator absolute differences."""
    candidate, reference = fields(driver)
    material = np.ones((2, 3))
    with pytest.raises(ValueError, match="complete"):
        driver.BrokenQField(candidate.pressure[:, :-1], candidate.mesh, 2, 3)
    with pytest.raises(ValueError, match="subdivide"):
        driver.mhm_difference(
            candidate,
            driver.PixelCGField(np.ones((4, 4)), 1, reference.bounds),
            material,
            material,
            omega=1,
        )
    with pytest.raises(ValueError, match="positive"):
        driver.mhm_difference(candidate, reference, -material, material, omega=1)
    with pytest.raises(ValueError, match="pixel-aligned"):
        driver.mhm_difference(
            candidate, reference, material, material, omega=1, gradient_cutout=(0, 0.1, 0, 1)
        )
    zero = driver.PixelCGField(np.zeros_like(reference.nodes), 4, reference.bounds)
    values = driver.mhm_difference(candidate, zero, material, material, omega=1)
    assert values["pressure_difference"] > 0
    assert values["pressure_relative_difference"] is None
