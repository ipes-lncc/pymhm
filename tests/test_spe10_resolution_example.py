"""Physical measures and subdomain additivity of the SPE10 resolution comparison."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.meshes.triangle import TriangleMesh


class ConstantReference:
    """A constant classical field on a declared rectangular triangulation."""

    nx, ny = 2, 2

    def evaluate(self, points):
        """Return exact scalar and vector fields with nonzero norm denominators."""
        return np.ones(len(points)), np.tile([3.0, 4.0], (len(points), 1)), np.zeros(len(points))


class ConstantCandidate:
    """Two distinct raw/reconstructed vectors on a physical triangular partition."""

    def __init__(self):
        """Create two macrotriangles with one local triangle each."""
        base = TriangleMesh.unit_square()
        macro = TriangleMesh(base.points * [2, 3], base.cells)
        self.meshes = tuple(macro.submesh(cell, 1) for cell in range(2))

    def evaluate_local(self, cell, points, owners):
        """Preserve independent raw and reconstructed constant vector values."""
        return (
            np.full(len(points), 2.0),
            np.tile([4.0, 2.0], (len(points), 1)),
            np.tile([1.0, 5.0], (len(points), 1)),
        )

    def material(self, points):
        """Return a constant anisotropic SPD permeability."""
        return np.broadcast_to(np.diag([2.0, 3.0]), (len(points), 2, 2))


@pytest.mark.parametrize("order", [4, 5])
def test_common_overlay_measures_denominators_and_partition_additivity(monkeypatch, order):
    """Check all thirteen physical integrals against exact nonzero constants."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("examples.spe10_adaptive_norms")
    monkeypatch.setattr(module, "DOMAIN", np.array([2.0, 3.0]))
    candidate, reference = ConstantCandidate(), ConstantReference()
    totals = module.integrated_squared_norms(candidate, reference, order)
    expected = 6 * np.array([1, 5, 5, 1, 25, 4, 20, 26, 11 / 6, 7 / 3, 59 / 6, 28 / 3, 53 / 6])
    assert_allclose(totals, expected, rtol=3e-15)
    parts = [
        module.integrated_squared_norms(candidate, reference, order, cells=np.array([cell]))
        for cell in (0, 1)
    ]
    assert_allclose(np.sum(parts, axis=0), totals, rtol=3e-16)
    result = module.compare(candidate, reference, order)
    assert_allclose(result["pressure_relative_difference"], 1.0, rtol=3e-15)
    assert_allclose(result["raw_flux_relative_difference"], 1 / np.sqrt(5), rtol=3e-15)
    assert_allclose(result["raw_flux_energy_relative_difference"], np.sqrt(11 / 59))
    assert_allclose(result["reconstructed_flux_energy_relative_difference"], np.sqrt(14 / 59))


def test_resolution_controls_reject_unresolved_trace_breaks_before_reading_data(monkeypatch):
    """A segmented RT reconstruction requires actual aligned local boundary edges."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("examples.solve_spe10_resolution")
    with pytest.raises(ValueError, match="divide"):
        module.acquire(Path("absent.npz"), Path("unused"), 4, 3)


def test_small_overlay_at_large_coordinate_preserves_area_and_quadratic_moments(monkeypatch):
    """Clipping uses translated geometry so small areas survive world-coordinate reconstruction."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("examples.spe10_adaptive_norms")
    vertices = np.array(
        [
            [99.577629307175, 1732.24027504],
            [99.5123825571125, 1732.26252790625],
            [99.59738331315, 1732.1572826749998],
        ]
    )
    points, weights = module.overlay_quadrature(vertices, SimpleNamespace(nx=240, ny=880), 4)
    sides = vertices[1:] - vertices[0]
    area = abs(np.linalg.det(sides.T)) / 2
    # Cutting the diagonal involves subtraction within a reference cell;
    # this tolerance remains twenty times tighter than the partition gate.
    assert_allclose(weights.sum(), area, rtol=1e-13)
    assert_allclose(weights @ points, area * vertices.mean(axis=0), rtol=1e-13)
    exact_second = area / 12 * (np.sum(vertices**2, axis=0) + np.sum(vertices, axis=0) ** 2)
    assert_allclose(weights @ (points**2), exact_second, rtol=1e-13)


def test_thin_material_fitted_triangle_has_canonical_partition_and_exact_moments(monkeypatch):
    """Barycentric clipping preserves a thin physical triangle spanning reference cells."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("examples.spe10_adaptive_norms")
    vertices = np.array(
        [
            [726.9033436863547, 649.3910523457499],
            [724.5814035664968, 650.0],
            [724.5813249245, 649.9999664775],
        ]
    )
    points, weights = module.overlay_quadrature(vertices, SimpleNamespace(nx=240, ny=880), 4)
    area = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / 2
    assert_allclose(weights.sum(), area, rtol=3e-15)
    assert_allclose(weights @ points, area * vertices.mean(axis=0), rtol=3e-15)
    second = area / 12 * (np.sum(vertices**2, axis=0) + np.sum(vertices, axis=0) ** 2)
    assert_allclose(weights @ (points**2), second, rtol=3e-15)


@pytest.mark.visualization
def test_fitted_display_uses_original_owners_without_recutting(monkeypatch):
    """Preserve analytical samples bitwise while bypassing redundant fitting and point searches."""
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("examples.plot_spe10_adaptive")
    from pymhm.fem.quadrature.material import fit_material_mesh
    from pymhm.materials.cartesian import CartesianCellField

    material = CartesianCellField(np.array([[1.0, 7.0]]), (1.0, 0.5))
    mesh = fit_material_mesh(TriangleMesh.unit_square(2), material)
    captured = []

    def evaluate(cell, points, owners=None):
        """Return unambiguous physical polynomials while recording explicit ownership."""
        captured.append(owners)
        return points[:, 0] + 2 * points[:, 1], points, 3 * points

    field = SimpleNamespace(meshes=(mesh,), material=material, evaluate_local=evaluate)
    original = module.display_fields(field)
    assert captured[-1] is None

    def reject_recut(*args):
        """Reject unnecessary geometric reconstruction of an already fitted archive."""
        raise AssertionError("fitted geometry must be reused")

    monkeypatch.setattr(module, "fit_material_mesh", reject_recut)
    fitted = module.display_fields(field, material_fitted=True)
    assert np.array_equal(captured[-1], np.arange(len(mesh.cells)))
    first = original["points"][original["cells"]].mean(axis=1)
    second = fitted["points"][fitted["cells"]].mean(axis=1)
    from scipy.spatial import cKDTree

    distance, indices = cKDTree(first).query(second)
    assert np.array_equal(distance, np.zeros(len(second)))
    assert np.array_equal(fitted["values"], original["values"][indices])
