"""Independent material-weighted norm integration on unfitted reference cells."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymhm.materials.cartesian import CartesianCellField
from pymhm.meshes.triangle import TriangleMesh


def test_reference_boundary_selection_constructs_normals_once(monkeypatch):
    """Selecting physical side fluxes costs one geometry pass and preserves every face."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    spe10_adaptive = importlib.import_module("examples.spe10_adaptive")
    mesh = TriangleMesh.unit_square(7, 11)
    expected = {int(face): 0.0 for face in mesh.boundary_faces if abs(mesh.normals[face, 0]) > 0.5}
    original = TriangleMesh.normals.fget
    calls = []

    def counted(instance):
        """Count complete geometry evaluations without changing their numerical values."""
        calls.append(instance)
        return original(instance)

    monkeypatch.setattr(TriangleMesh, "normals", property(counted))
    assert spe10_adaptive.natural_faces(mesh) == expected
    assert len(calls) == 1


def test_reference_energy_splits_material_before_integration(monkeypatch):
    """A non-dyadic jump in K has an exact weighted constant-flux norm."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    spe10_adaptive = importlib.import_module("examples.spe10_adaptive")
    field = CartesianCellField(
        np.array([np.eye(2), 4 * np.eye(2), 4 * np.eye(2)])[:, None], (1 / 3, 1.0)
    )
    monkeypatch.setattr(spe10_adaptive, "load_layer", lambda: field)
    mesh = TriangleMesh.unit_square()

    def evaluate(points):
        """Use constant physical fields so geometry/material integration is isolated."""
        return np.ones(len(points)), np.tile([3.0, 4.0], (len(points), 1)), np.zeros(len(points))

    current = SimpleNamespace(nx=2, ny=2, mesh=mesh, areas=mesh.areas, evaluate=evaluate)
    previous = SimpleNamespace(nx=1, ny=1, mesh=mesh, areas=mesh.areas, evaluate=evaluate)
    result = spe10_adaptive.reference_integrals(current, previous, order=3)
    assert result["pressure_l2"] == pytest.approx(1.0)
    assert result["flux_l2"] == pytest.approx(5.0)
    assert result["flux_energy_norm"] == pytest.approx(np.sqrt(12.5))
    assert result["previous_flux_energy_difference"] == pytest.approx(0.0)
