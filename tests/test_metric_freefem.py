"""Native optional FreeFEM/BAMG mesh-generation integration."""

import os
import shutil

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.mesh import TriangleMesh
from pymhm.metric_adapt import remesh_freefem, residual_mesh_size

PROGRAM = (
    os.environ.get("PYMHM_FREEFEM") or shutil.which("FreeFem++-nw") or shutil.which("FreeFem++")
)


@pytest.mark.meshing
@pytest.mark.skipif(PROGRAM is None, reason="requires a native FreeFEM executable")
def test_native_residual_metric_and_polygon_boundary():
    """BAMG generates a nonnested mesh while preserving the represented rectangle."""
    mesh = TriangleMesh.unit_square(4)
    indicators = np.arange(1, len(mesh.cells) + 1, dtype=float) ** 2
    metric = residual_mesh_size(mesh, indicators)
    refined = remesh_freefem(mesh, metric.requested, executable=PROGRAM)
    assert len(refined.cells) != len(mesh.cells)
    assert_allclose(refined.areas.sum(), 1, rtol=0, atol=2e-14)
    points = refined.points[np.unique(refined.faces[refined.boundary_faces])]
    distance = np.minimum.reduce(
        [abs(points[:, 0]), abs(points[:, 1]), abs(points[:, 0] - 1), abs(points[:, 1] - 1)]
    )
    assert np.max(distance) < 1e-13
    assert np.all(refined.areas > 0)
