"""Selective generalized MHM keeps only genuine local constant null modes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import TetraMesh, solve_rad_3d
from pymhm.polygon import PolygonMesh, solve_transport_polygons


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("reaction", [0.0, 1.0])
@pytest.mark.parametrize("rotation", [False, True])
def test_selective_null_modes_preserve_full_discrete_fields(dimension, reaction, rotation):
    """Divergence-free tangential advection has a kernel even when its magnitude is large."""
    if dimension == 2:
        mesh = PolygonMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), (np.arange(4),))
        solve = solve_transport_polygons

        def velocity(points):
            """Curl of a square bubble, with exactly zero boundary-normal component."""
            x, y = points.T
            return 75 * np.column_stack((x * (1 - x) * (1 - 2 * y), -(1 - 2 * x) * y * (1 - y)))

    else:
        mesh = TetraMesh(np.vstack((np.zeros(3), np.eye(3))), np.arange(4)[None])
        solve = solve_rad_3d

        def velocity(points):
            """Curl of phi*e_z, phi=x*y*z*(1-x-y-z), tangent to every tetra face."""
            x, y, z = points.T
            return 75 * np.column_stack(
                (x * z * (1 - x - 2 * y - z), -y * z * (1 - 2 * x - y - z), np.zeros(len(x)))
            )

    options = dict(
        degree=2,
        local_refinement=2,
        source=1.0,
        reaction=reaction,
        velocity=velocity if rotation else (0.0,) * dimension,
        velocity_divergence=0.0,
        quadrature_order=6,
    )
    full = solve(mesh, **options)
    selected = solve(mesh, coarse_space="kernel", **options)
    assert sum(len(c) for c in selected.hybrid.coarse) == (0 if reaction else 1)
    assert_allclose(selected.hybrid.trace, full.hybrid.trace, atol=1e-12)
    for a, b in zip(selected.values, full.values, strict=True):
        assert_allclose(a, b, atol=1e-12)
    with pytest.raises(ValueError, match="coarse_space"):
        solve(mesh, coarse_space="numerical_threshold", **options)


def test_selective_mixed_primal_partition_matches_retained_form():
    """Different macrocells can be primal or mixed within the same global system."""
    points = np.array([[0, 0], [0.5, 0], [1, 0], [0, 1], [0.5, 1], [1, 1]])
    mesh = PolygonMesh(points, (np.array([0, 1, 4, 3]), np.array([1, 2, 5, 4])))
    options = dict(
        degree=2, local_refinement=2, source=1.0, reaction=lambda x: (x[:, 0] > 0.5).astype(float)
    )
    full = solve_transport_polygons(mesh, **options)
    selected = solve_transport_polygons(mesh, coarse_space="kernel", **options)
    assert [len(c) for c in selected.hybrid.coarse] == [1, 0]
    for a, b in zip(selected.values, full.values, strict=True):
        assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 3])
def test_selective_transport_with_nonzero_normal_advection(dimension):
    """Normal advection lifts the constant null mode without changing the solution."""
    if dimension == 2:
        mesh = PolygonMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), (np.arange(4),))
        solve = solve_transport_polygons
    else:
        mesh = TetraMesh(np.vstack((np.zeros(3), np.eye(3))), np.arange(4)[None])
        solve = solve_rad_3d
    options = dict(
        degree=2, local_refinement=2, source=1.0, velocity=(1.0,) + (0.0,) * (dimension - 1)
    )
    full = solve(mesh, **options)
    selected = solve(mesh, coarse_space="kernel", **options)
    assert sum(len(c) for c in selected.hybrid.coarse) == 0
    for a, b in zip(selected.values, full.values, strict=True):
        assert_allclose(a, b, atol=2e-12)


def test_selective_diffusive_faces_require_tangency():
    """A physical advective outflow changes the local nullspace contract."""
    mesh = PolygonMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), (np.arange(4),))
    vertical = [int(f) for f in mesh.boundary_faces if abs(mesh.normals[f, 0]) == 1]
    with pytest.raises(ValueError, match="tangent advection"):
        solve_transport_polygons(
            mesh,
            degree=2,
            velocity=(1.0, 0.0),
            coarse_space="kernel",
            diffusive_flux={vertical[0]: 0.0},
        )


def test_true_tiny_normal_advection_is_not_classified_as_zero():
    """Tangency checks are relative to the physical field, with no absolute floor."""
    from pymhm import SkeletonSpace, TriangularSkeleton
    from pymhm.rad import _boundary_tangent
    from pymhm.rad3d import _boundary_tangent_3d

    square = PolygonMesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), (np.arange(4),))
    tetra = TetraMesh(np.vstack((np.zeros(3), np.eye(3))), np.arange(4)[None])
    assert not _boundary_tangent(SkeletonSpace(square), (1e-30, 0.0), 6)
    assert not _boundary_tangent_3d(TriangularSkeleton(tetra), (1e-30, 0.0, 0.0), 6)
