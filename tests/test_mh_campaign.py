"""Independent analytical data, physical norm and polygonal geometry contracts."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.lagrange import nodal_space
from pymhm.mesh import TriangleMesh


@pytest.fixture
def campaign(monkeypatch):
    """Load public analytical examples without relying on pytest's launch directory."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.mh_campaign")


def test_sinusoidal_source_and_flux_independently_differentiated(campaign):
    points = np.array([[0.13, 0.24], [0.56, 0.73], [0.78, 0.39]])
    divergence = np.zeros(len(points))
    for axis in range(2):
        shifted = points.astype(complex)
        shifted[:, axis] += 1e-25j
        derivative = np.imag(campaign.exact(shifted)) / 1e-25
        assert_allclose(-derivative, campaign.exact_flux(points)[:, axis], rtol=3e-14)
        divergence += np.imag(campaign.exact_flux(shifted)[:, axis]) / 1e-25
    assert_allclose(divergence, campaign.source(points), rtol=3e-14)


def test_complementary_nonconvex_partition_and_true_diameters(campaign):
    mesh = campaign.l_mesh(2)
    assert len(mesh.cells) == 8
    assert_allclose(mesh.areas.sum(), 1, atol=1e-15)
    assert np.count_nonzero(mesh.face_cells[:, 1] < 0) == 20
    for cell in mesh.cells:
        points = mesh.points[cell]
        diameter = np.linalg.norm(points[:, None] - points[None], axis=2).max()
        assert_allclose(diameter, np.sqrt(13) / 6, atol=1e-15)


def test_common_partition_energy_norm_nonzero_and_mismatch(campaign):
    mesh = TriangleMesh.unit_square(2)
    _, points = nodal_space(mesh, 2)
    first = SimpleNamespace(
        local_meshes=(mesh,), pressure=(3 * points[:, 0] + 4 * points[:, 1],), degree=2
    )
    second = SimpleNamespace(local_meshes=(mesh,), pressure=(np.zeros(len(points)),))
    assert_allclose(campaign.energy_difference(first, second), 5, rtol=3e-14)
    second.local_meshes = (TriangleMesh.unit_square(1),)
    with pytest.raises(ValueError, match="identical fine"):
        campaign.energy_difference(first, second)
