"""Verify that scientific plots preserve polynomial values and derivatives."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import TriangleMesh
from pymhm.fem.scalar.triangle import nodal_space


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("components", [1, 2])
def test_sampling_preserves_pk_values_and_gradients(monkeypatch, degree, components):
    """Evaluate polynomials at independent display points, including vector gradients."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    sampling = importlib.import_module("field_sampling")
    mesh = TriangleMesh.unit_square(2)
    _, nodes = nodal_space(mesh, degree)
    field = 1 + nodes[:, 0] ** degree + 2 * nodes[:, 1] ** degree
    if components == 2:
        field = np.column_stack((field, 3 * field))
    sampled = sampling.sample_field([mesh], [field], degree)
    points = sampled["points"]
    expected = 1 + points[:, 0] ** degree + 2 * points[:, 1] ** degree
    gradient = degree * points ** (degree - 1) * [1, 2]
    if components == 2:
        expected = np.column_stack((expected, 3 * expected))
        gradient = np.stack((gradient, 3 * gradient), axis=1)
    assert_allclose(sampled["values"], expected, atol=3e-13)
    assert_allclose(sampled["gradient"], gradient, atol=2e-12)
    assert_allclose(sampling.local_values(mesh, field, degree, points), expected, atol=3e-13)
    assert len(sampled["points"]) > len(np.unique(sampled["points"], axis=0))
    with pytest.raises(ValueError, match="outside"):
        sampling.local_values(mesh, field, degree, np.array([[1.1, 0.5]]))


def test_sampling_preserves_both_sides_of_a_macro_jump(monkeypatch):
    """Keep different one-sided values at identical physical interface points."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    sampling = importlib.import_module("field_sampling")
    macro = TriangleMesh.unit_square(1)
    meshes = [macro.submesh(cell, 2) for cell in range(2)]
    fields = [np.full(len(nodal_space(mesh, 2)[1]), cell) for cell, mesh in enumerate(meshes)]
    sampled = sampling.sample_field(meshes, fields, 2)
    interface = sampled["points"]
    diagonal = np.isclose(interface[:, 0], interface[:, 1], atol=1e-14, rtol=0)
    for point in np.unique(interface[diagonal], axis=0):
        at_point = np.all(np.isclose(interface, point, atol=1e-14, rtol=0), axis=1)
        assert np.min(sampled["values"][at_point]) == 0
        assert_allclose(np.max(sampled["values"][at_point]), 1, atol=1e-15)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    ("start", "end"),
    [((0.0, 0.37), (1.0, 0.37)), ((0.43, 0.0), (0.43, 1.0))],
)
def test_profiles_retain_independent_vector_limits(monkeypatch, degree, start, end):
    """Preserve both sides of a polynomial vector jump on either profile direction."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    sampling = importlib.import_module("field_sampling")
    macro = TriangleMesh.unit_square()
    meshes = tuple(macro.submesh(cell, 2) for cell in range(2))
    fields = []
    for cell, mesh in enumerate(meshes):
        _, points = nodal_space(mesh, degree)
        scalar = points[:, 0] ** degree + 2 * points[:, 1] ** degree + cell
        fields.append(np.column_stack((scalar, -3 * scalar)))
    solution = SimpleNamespace(skeleton=SimpleNamespace(mesh=macro), local_meshes=meshes)
    sampled = sampling.sample_segment_profile(solution, fields, degree, start, end)
    assert sampled["profile_values"].shape == (2, 121, 2)
    assert_allclose(sampled["profile_points"][0, -1], sampled["profile_points"][1, 0])
    jump = sampled["profile_values"][1, 0] - sampled["profile_values"][0, -1]
    assert_allclose(abs(jump), [1, 3], atol=3e-14, rtol=0)
    for points, values in zip(sampled["profile_points"], sampled["profile_values"], strict=True):
        expected = points[:, 0] ** degree + 2 * points[:, 1] ** degree
        offset = round(float(values[0, 0] - expected[0]))
        assert offset in (0, 1)
        assert_allclose(values, np.column_stack((expected + offset, -3 * (expected + offset))))
    backward = sampling.sample_segment_profile(solution, fields, degree, end, start)
    assert_allclose(backward["profile_values"], sampled["profile_values"][::-1, ::-1], atol=3e-14)
    if start[0] == 0:
        horizontal = sampling.sample_profile(solution, fields, degree, height=start[1])
        for key in sampled:
            np.testing.assert_array_equal(horizontal[key], sampled[key])


def test_profile_rejects_ambiguous_or_exterior_segments(monkeypatch):
    """Do not invent a side for a macro-aligned line or extrapolate outside the domain."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    sampling = importlib.import_module("field_sampling")
    macro = TriangleMesh.unit_square()
    meshes = tuple(macro.submesh(cell, 2) for cell in range(2))
    solution = SimpleNamespace(skeleton=SimpleNamespace(mesh=macro), local_meshes=meshes)
    fields = [np.ones(len(mesh.points)) for mesh in meshes]
    with pytest.raises(ValueError, match="coincides with a macroface"):
        sampling.sample_segment_profile(solution, fields, 1, (0, 0), (1, 1))
    with pytest.raises(ValueError, match="outside"):
        sampling.sample_segment_profile(solution, fields, 1, (-0.1, 0.37), (1.1, 0.37))
