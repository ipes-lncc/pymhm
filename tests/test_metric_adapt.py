"""Mesh-size moments and isolated optional remesher contracts."""

import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.adaptivity.metric import _read_mesh, remesh_freefem, residual_mesh_size
from pymhm.meshes.triangle import TriangleMesh


def test_lumped_metric_moments_and_scaling():
    """Shared vertices average area-weighted indicators and repeated incident edges."""
    mesh = TriangleMesh.unit_square()
    metric = residual_mesh_size(mesh, [1.0, 9.0])
    assert_allclose(metric.projected_indicator, [2, 1, 3, 2])
    assert_allclose(metric.current, [(1 + np.sqrt(2)) / 2, 1, 1, (1 + np.sqrt(2)) / 2])
    assert_allclose(metric.factors, [1, 0.5, 1.5, 1])
    assert metric.threshold == 2
    assert_allclose(metric.requested * metric.factors, metric.current)
    scaled = residual_mesh_size(mesh, [1e200, 9e200])
    assert_allclose(scaled.requested, metric.requested)
    zero = residual_mesh_size(mesh, [0.0, 0.0])
    assert zero.threshold == 0
    assert_allclose(zero.requested, zero.current)
    clipped = residual_mesh_size(TriangleMesh.unit_square(4), [1e10] + [0] * 31)
    assert clipped.factors.min() == 1 / 3
    assert clipped.factors.max() == 3


@pytest.mark.parametrize("values", [[1], [-1, 2], [np.inf, 2], [1j, 2]])
def test_invalid_metric_indicator(values):
    """Indicators must be real, nonnegative, finite and cellwise."""
    with pytest.raises(ValueError, match="squared indicator"):
        residual_mesh_size(TriangleMesh.unit_square(), values)


def test_invalid_metric_parameters_and_unused_points():
    """The metric is undefined for isolated vertices and nonpositive scaling."""
    mesh = TriangleMesh.unit_square()
    for coefficient in (0, np.inf):
        with pytest.raises(ValueError, match="coefficient"):
            residual_mesh_size(mesh, [1, 1], coefficient=coefficient)
    unused = TriangleMesh(np.vstack((mesh.points, [3, 3])), mesh.cells)
    with pytest.raises(ValueError, match="unused"):
        residual_mesh_size(unused, [1, 1])


@pytest.mark.parametrize("values", [[1], [1, -1, 1, 1], [1j] * 4, [np.inf] * 4])
def test_invalid_remesher_sizes(values):
    """The external engine never receives invalid physical sizes."""
    with pytest.raises(ValueError, match="sizes"):
        remesh_freefem(TriangleMesh.unit_square(), values)


def test_remesher_validation_and_missing_executable(monkeypatch):
    """Optional execution is explicit, and time/complexity parameters are validated."""
    import pymhm.adaptivity.metric as module

    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="max_vertices"):
        remesh_freefem(mesh, np.ones(4), max_vertices=0)
    with pytest.raises(ValueError, match="timeout"):
        remesh_freefem(mesh, np.ones(4), timeout=0)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(ImportError, match="FreeFEM"):
        remesh_freefem(mesh, np.ones(4))


def test_optional_remesher_protocol(monkeypatch):
    """The bridge transfers physical nodal sizes and leaves the PDE assembly local."""
    import pymhm.adaptivity.metric as module

    mesh = TriangleMesh.unit_square(2)
    sizes = np.linspace(0.1, 0.4, len(mesh.points))
    monkeypatch.setattr(module.shutil, "which", lambda name: "/native/freefem")

    def run(arguments, **options):
        """An identity engine validates serialization and P1 node-to-DOF mapping."""
        assert arguments == ["/native/freefem", "-nw", "remesh.edp"]
        folder = options["cwd"]
        assert_allclose(np.loadtxt(folder / "sizes.txt"), sizes)
        script = (folder / "remesh.edp").read_text()
        assert "Vh(k,j)" in script and "nbvx=2000" in script
        (folder / "output.msh").write_bytes((folder / "input.msh").read_bytes())
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", run)
    result = remesh_freefem(mesh, sizes, max_vertices=2000)
    assert_allclose(result.points, mesh.points)
    assert np.array_equal(result.cells, mesh.cells)


def test_remesher_failures_and_area_guard(monkeypatch, tmp_path):
    """A failed engine or a changed domain is rejected independently of its exit status."""
    import pymhm.adaptivity.metric as module

    monkeypatch.setattr(module.shutil, "which", lambda name: "/native/freefem")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 2, "bad", ""),
    )
    with pytest.raises(RuntimeError, match="bad"):
        remesh_freefem(TriangleMesh.unit_square(), np.ones(4))

    def changed(arguments, **options):
        """A malformed engine preserves connectivity but changes the physical domain."""
        mesh = TriangleMesh.unit_square()
        module._write_mesh(options["cwd"] / "output.msh", TriangleMesh(mesh.points * 2, mesh.cells))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", changed)
    with pytest.raises(ValueError, match="domain area"):
        remesh_freefem(TriangleMesh.unit_square(), np.ones(4))
    invalid = Path(tmp_path) / "invalid.msh"
    invalid.write_text("3 1 3\n")
    with pytest.raises(ValueError, match="entity counts"):
        _read_mesh(invalid)
