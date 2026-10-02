"""Analytical cut-simplex moments and exact incident-side material integration."""

from math import factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import (
    PlanarMaterial,
    PlanarRegion,
    TetraMesh,
    TriangleMesh,
    material_triangle_quadrature,
    planar_edge_quadrature,
    planar_simplex_quadrature,
)


@pytest.mark.parametrize("dimension", [2, 3])
def test_exact_tensor_weighted_moments(dimension):
    """Oblique material integration reproduces analytical zeroth through quadratic moments."""
    mesh = TriangleMesh.unit_square(2) if dimension == 2 else TetraMesh.unit_cube()
    material = PlanarMaterial(1, (PlanarRegion([np.ones(dimension)], [0.83], 4),))
    bary, weights, tensors = planar_simplex_quadrature(mesh, material, 4)
    points = np.einsum("tqi,tia->tqa", bary, mesh.points[mesh.cells])
    measure = mesh.areas if dimension == 2 else mesh.volumes
    physical = weights * measure[:, None] * tensors[:, 0, 0].reshape(weights.shape)
    assert_allclose(weights.sum(axis=1), 1, atol=4e-15)
    assert np.all(weights >= 0)
    assert np.min(bary) > -1e-14
    assert_allclose(physical.sum(), 1 + 3 * 0.83**dimension / factorial(dimension), atol=8e-15)
    for a in range(dimension):
        assert_allclose(
            np.sum(physical * points[..., a]),
            0.5 + 3 * 0.83 ** (dimension + 1) / factorial(dimension + 1),
            atol=4e-15,
        )
        for b in range(dimension):
            exact = (1 / 3 if a == b else 1 / 4) + (
                3 * (2 if a == b else 1) * 0.83 ** (dimension + 2) / factorial(dimension + 2)
            )
            assert_allclose(np.sum(physical * points[..., a] * points[..., b]), exact, atol=4e-15)
    if dimension == 2:
        for expected, actual in zip(
            (bary, weights, tensors), material_triangle_quadrature(mesh, material, 4), strict=True
        ):
            assert_allclose(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_edge_crossing_and_exact_one_sided_traces(dimension):
    """An edge crossing and an edge lying on the plane retain physical material values."""
    normal = np.zeros(dimension)
    normal[0] = 1
    material = PlanarMaterial(1, (PlanarRegion([normal], [0.37], 4),))
    a, b, interior = np.zeros(dimension), np.ones(dimension), np.full(dimension, 0.2)
    points, weights, tensors = planar_edge_quadrature(a, b, material, 3, interior_point=interior)
    assert_allclose(weights.sum(), 1, atol=2e-15)
    assert_allclose(np.dot(weights, tensors[:, 0, 0]), 0.37 * 4 + 0.63, atol=2e-15)
    assert_allclose(np.dot(weights * tensors[:, 0, 0], points[:, 0]), 0.5 + 1.5 * 0.37**2)
    a[0] = b[0] = 0.37
    for side, value in ((0.2, 4), (0.5, 1)):
        interior[0] = side
        _, _, tensors = planar_edge_quadrature(a, b, material, interior_point=interior)
        assert_allclose(tensors, np.broadcast_to(value * np.eye(dimension), tensors.shape))
    with pytest.raises(ValueError, match="strict side"):
        planar_edge_quadrature(a, b, material, interior_point=a)


def test_edge_geometry_contract():
    """Malformed and degenerate edge rules cannot silently change dimensional conventions."""
    material = PlanarMaterial(1, (PlanarRegion([[1, 1]], [0.5], 2),))
    for a, b, c in (([0j, 0], [1, 1], [0, 1]), ([0, 0], [1, 1], [0]), ([0, 0], [0, 0], [1, 1])):
        with pytest.raises(ValueError):
            planar_edge_quadrature(a, b, material, interior_point=c)
    with pytest.raises(ValueError):
        planar_simplex_quadrature(TriangleMesh.unit_square(), material, 0)
