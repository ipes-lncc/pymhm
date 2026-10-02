"""Geometry and unaveraged sampling invariants for the SPE10 flow example."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace
from pymhm.lagrange import nodal_space


@pytest.fixture
def driver(monkeypatch):
    """Load the public PyMHM calculation driver without starting its CLI."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("solve_spe10_brinkman")


def test_literature_crisscross_geometry_and_boundary_counts(driver):
    """Check the declared 264 macrotriangles and 16,520 raw skeleton coefficients."""
    mesh = driver.crisscross_mesh()
    assert len(mesh.cells) == 264
    assert_allclose(mesh.areas, 10000, atol=0)
    assert_allclose(mesh.points.min(axis=0), [0, 0], atol=0)
    assert_allclose(mesh.points.max(axis=0), [1200, 2200], atol=0)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 10) for _ in mesh.faces), 2)
    assert skeleton.size == 16520
    top = [f for f in mesh.boundary_faces if mesh.normals[f, 1] > 0.5]
    sides = [f for f in mesh.boundary_faces if abs(mesh.normals[f, 0]) > 0.5]
    assert len(top) == 6
    assert len(sides) == 22
    assert 2 * sum(skeleton.faces[f].size for f in top) == 240
    assert sum(skeleton.faces[f].size for f in sides) == 440
    values = driver.inlet_velocity(np.array([[20, 0], [0, 100], [30, 2200], [1200, 50]]))
    assert_array_equal(values, [[0, 1], [0, 0], [0, 0], [0, 0]])


def test_sampled_profile_keeps_distinct_values_at_shared_macrointerfaces(driver):
    """Keep a discontinuous macro pressure exactly, including both profile limits."""
    mesh = driver.crisscross_mesh(1, 1)
    fine = tuple(mesh.submesh(cell, 2) for cell in range(len(mesh.cells)))
    values = tuple(np.tile([0.0, 1.0], (len(nodal_space(local, 3)[1]), 1)) for local in fine)
    pressure = tuple(np.full(len(field), cell) for cell, field in enumerate(values))
    solution = SimpleNamespace(
        skeleton=SkeletonSpace(mesh, components=2),
        local_meshes=fine,
        values=values,
        pressure=pressure,
        degree=3,
        pressure_degree=3,
        hybrid=SimpleNamespace(trace=np.zeros(len(mesh.faces) * 2), coarse=np.zeros((4, 2))),
    )
    arrays = driver.archive_fields(solution)
    assert_allclose(arrays["velocity"], np.broadcast_to([0, 1], (60, 220, 2)), atol=2e-15)
    assert_allclose(arrays["pressure"], arrays["owner"], atol=2e-15)
    for index, owner in enumerate(arrays["profile_owner"]):
        assert_allclose(arrays["profile_pressure"][index], owner, atol=2e-15)
    assert_allclose(arrays["profile_points"][:-1, -1], arrays["profile_points"][1:, 0], atol=0)
    assert np.any(arrays["profile_pressure"][:-1, -1] != arrays["profile_pressure"][1:, 0])
    with pytest.raises(ValueError, match="outside"):
        driver.owners(mesh, np.array([[-0.1, 100]]))


def test_profile_intersections_preserve_one_sided_support_at_high_local_refinement(driver):
    """Do not round a macroface crossing outside its highly refined supporting cell."""
    mesh = driver.crisscross_mesh()
    start, end = np.array([199.0, 0.0]), np.array([199.0, 2200.0])
    breaks = driver.macro_profile_breaks(mesh, start, end)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        owner = int(driver.owners(mesh, (start + (left + right) / 2 * (end - start))[None])[0])
        fine = mesh.submesh(owner, 30)
        nodes = nodal_space(fine, 3)[1]
        points = start + np.array([left, right])[:, None] * (end - start)
        field = 1 + nodes[:, 0] + 2 * nodes[:, 1]
        values = driver.local_values(fine, field, 3, points)
        assert_allclose(values, 1 + points[:, 0] + 2 * points[:, 1], rtol=3e-14)
