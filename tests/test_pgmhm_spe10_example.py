"""Physical integration and portable field contracts of the PGMHM reservoir study."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.fem.scalar.triangle import nodal_space
from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


def example(monkeypatch):
    """Import the original example without making examples a runtime dependency."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.solve_pgmhm_spe10")


def test_reference_norms_use_physical_area_and_current_denominator(monkeypatch):
    """Nonzero analytical differences detect missing areas and incorrect denominators."""
    module = example(monkeypatch)
    monkeypatch.setattr(
        module, "load_material", lambda component="kz": CartesianCellField([[4.0]], (2.0, 3.0))
    )
    mesh = TriangleMesh(TriangleMesh.unit_square().points * [2, 3], [[0, 1, 3], [0, 3, 2]])

    def current_values(points):
        """Constant pressure two and flux (3,4) have explicit physical norms."""
        return (
            np.full(len(points), 2.0),
            np.tile([3.0, 4.0], (len(points), 1)),
            np.zeros(len(points)),
        )

    def previous_values(points):
        """The zero previous field gives unit relative differences."""
        return np.zeros(len(points)), np.zeros((len(points), 2)), np.zeros(len(points))

    current = SimpleNamespace(nx=120, ny=440, mesh=mesh, evaluate=current_values)
    previous = SimpleNamespace(nx=60, ny=220, mesh=mesh, evaluate=previous_values)
    result = module.reference_norms(current, previous, 4)
    assert result["pressure_l2"] == pytest.approx(2 * np.sqrt(6))
    assert result["flux_l2"] == pytest.approx(5 * np.sqrt(6))
    assert result["flux_energy"] == pytest.approx(2.5 * np.sqrt(6))
    for name in (
        "pressure_relative_difference",
        "flux_relative_difference",
        "flux_energy_relative_difference",
    ):
        assert result[name] == pytest.approx(1)
    assert module.reference_norms(current, None, 4)["pressure_relative_difference"] == 0
    with pytest.raises(ValueError, match="dyadic"):
        module.reference_norms(current, current, 4)
    current.nx = 61
    with pytest.raises(ValueError, match="align"):
        module.reference_norms(current, None, 4)


def test_archived_pgmhm_field_replays_quadratic_pressure_and_physical_flux(monkeypatch, tmp_path):
    """The reader respects both enrichment arrays and the permeability factor in -K grad p."""
    module = example(monkeypatch)
    material = CartesianCellField([[2.0]], (1.0, 1.0))
    monkeypatch.setattr(module, "load_material", lambda component="kz": material)
    mesh = TriangleMesh.unit_square()
    _, nodes = nodal_space(mesh, 2)
    values = nodes[:, 0] ** 2 + 3 * nodes[:, 1] ** 2
    path = tmp_path / "fields.npz"
    np.savez(
        path,
        local_points=mesh.points[None],
        local_cells=mesh.cells[None],
        **module.precision_fields("pressure", values[None]),
        **module.precision_fields("enriched_pressure", 2 * values[None]),
    )
    field = module.PGMHMField(path)
    for cell in range(2):
        points = np.mean(mesh.points[mesh.cells[cell]], axis=0)[None]
        pressure, flux = field.evaluate(0, cell, points)
        expected_p = points[:, 0] ** 2 + 3 * points[:, 1] ** 2
        expected_q = -2 * points * [2, 6]
        assert_allclose(pressure, np.stack([expected_p, 2 * expected_p]), atol=1e-14)
        assert_allclose(flux, np.stack([expected_q, 2 * expected_q]), atol=1e-14)


def test_macro_partition_follows_figure_diagonal_and_boundary_labels(monkeypatch):
    """The profile crossing and prescribed side flux refer to the actual two macros."""
    module = example(monkeypatch)
    mesh = module.macro_mesh()
    internal = np.flatnonzero(mesh.face_cells[:, 1] >= 0)
    assert_allclose(mesh.points[mesh.faces[internal[0]]], [[1200, 0], [0, 2200]])
    assert len(module.natural_faces(mesh)) == 2
    assert_allclose(module.pressure_boundary(np.array([[600, 0], [600, 2200]])), [1, 0])


def test_material_fitted_control_archives_ragged_polynomials(monkeypatch, tmp_path):
    """An explicit fitted acquisition retains the physical affine pressure and raw flux."""
    module = example(monkeypatch)
    material = CartesianCellField(np.ones((1, 1)), (1200.0, 2200.0))
    monkeypatch.setattr(module, "load_material", lambda component="kx": material)
    module.acquire_mhm(tmp_path, 1, 5, 0.1, refinement=1, fitted=True, trace_fitted=True)
    path = tmp_path / "pgmhm-fitted-tracefit-r1-s1-q5.npz"
    field = module.PGMHMField(path)
    for macro, mesh in enumerate(field.meshes):
        for cell, vertices in enumerate(mesh.points[mesh.cells]):
            points = vertices.mean(axis=0)[None]
            pressure, flux = field.evaluate(macro, cell, points)
            assert_allclose(pressure, np.full((2, 1), 1 - points[0, 1] / 2200), atol=2e-12)
            assert_allclose(flux, np.tile([0, 1 / 2200], (2, 1, 1)), atol=2e-14)
