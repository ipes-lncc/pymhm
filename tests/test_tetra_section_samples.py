"""Section coverage and unmerged polynomial coordinates for publication figures."""

import importlib
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from pymhm import TetraMesh


def test_section_area_and_exact_cell_coordinates(monkeypatch):
    """A nonaligned cube section covers unit area and every sample stays in its owning cell."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    section_grid = importlib.import_module("examples.tetra_section_samples").section_grid
    mesh = TetraMesh.unit_cube(2)
    section = section_grid(mesh, height=0.37, refinement=3)
    vertices = section["points"][section["cells"], :2]
    a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    area = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    assert_allclose(area.sum(), 1, atol=2e-15)
    assert_allclose(section["points"][:, 2], 0.37, atol=1e-15)
    mapped = np.einsum(
        "qi,qia->qa", section["barycentric"], mesh.points[mesh.cells[section["parents"]]]
    )
    assert_allclose(mapped, section["points"], atol=1e-15)
    assert section["barycentric"].min() > -1e-14
    assert len(np.unique(section["points"], axis=0)) < len(section["points"])
    outside = section_grid(mesh, height=1.1)
    assert outside["points"].shape == (0, 3)
    assert outside["cells"].shape == (0, 3)
