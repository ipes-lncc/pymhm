"""Broken P0/P1 profile sampling preserves cell limits and rejects ambiguous cuts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import SkeletonSpace, TriangleMesh


@pytest.fixture
def sample(monkeypatch):
    """Import the data-only sampler without requiring Matplotlib or a renderer."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from field_sampling import sample_darcy_pressure_profile

    return sample_darcy_pressure_profile


def field(formulation):
    """Supply an affine nodal field or distinct exact constants on two triangles."""
    mesh = TriangleMesh.unit_square()
    fine = tuple(mesh.submesh(cell, 1) for cell in range(len(mesh.cells)))
    pressure = tuple(
        1 + part.points[:, 0] + 2 * part.points[:, 1]
        if formulation == "primal"
        else np.where(
            part.points[part.cells, 0].mean(axis=1) < part.points[part.cells, 1].mean(axis=1),
            2.0,
            7.0,
        )
        for part in fine
    )
    return SimpleNamespace(
        skeleton=SkeletonSpace(mesh), local_meshes=fine, pressure=pressure, formulation=formulation
    )


def test_constant_profile_keeps_both_values_at_the_same_interface_point(sample):
    """A jump has two limits at x=0.37 and no interpolated line between them."""
    result = sample(field("mixed"), [0, 0.37], [1, 0.37])
    assert_allclose(result["profile_parameter"], [[0, 0.37], [0.37, 1]], atol=2e-15)
    assert_allclose(result["profile_values"], [[2, 2], [7, 7]], atol=0)
    assert_allclose(result["profile_points"][0, -1], result["profile_points"][1, 0], atol=0)
    assert_allclose(result["profile_breaks"], [0, 0.37, 1], atol=2e-15)


def test_affine_profile_values_follow_the_local_polynomial_at_both_endpoints(sample):
    """P1 evaluation is exact on a sloped interior cut, including shared endpoints."""
    result = sample(field("primal"), [0, 0.2], [1, 0.6])
    points = result["profile_points"]
    assert_allclose(result["profile_values"], 1 + points[..., 0] + 2 * points[..., 1], atol=2e-15)


@pytest.mark.parametrize("start,end", [([0, 0], [1, 1]), ([-0.2, 0.37], [1, 0.37])])
def test_ambiguous_or_incomplete_profiles_are_rejected(sample, start, end):
    """A coincident two-sided line or a partially external line has no unique full profile."""
    with pytest.raises(ValueError, match="nonoverlapping|interface"):
        sample(field("mixed"), start, end)
