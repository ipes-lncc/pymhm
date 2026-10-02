"""Physical-coordinate roundoff contracts for Cartesian material intersections."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial import cKDTree

from pymhm._geometry_roundoff import cartesian_coordinates
from pymhm.cut_cells import (
    _material_edge_breaks,
    _merge_grid_vertices,
    cartesian_edge_quadrature,
    cartesian_trace_values,
    cartesian_triangle_quadrature,
    fit_material_mesh,
    material_triangle_quadrature,
)
from pymhm.mesh import TriangleMesh
from pymhm.polygon import PolygonMesh
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    _cartesian_rectangle_quadrature,
    _grid_resolves_material,
    quadrilateral_operators,
)
from pymhm.refinement import validate_submesh
from pymhm.reservoir import CartesianCellField

ORIGINS = [(0, 0), (500, 2100), (5000, 21000), (-500, -2100)]


def l_shape(origin):
    """Return the same nonconvex L with a translated physical representation."""
    points = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    return PolygonMesh(points + origin, (np.arange(6),))


@pytest.mark.parametrize("origin", ORIGINS)
def test_fitted_l_preserves_pixels_boundary_and_translation(origin):
    """Compare every material area with independent rectangle intersections."""
    macro = l_shape(origin)
    field = CartesianCellField(np.arange(49).reshape(7, 7), (0.3, 0.3), origin)
    fine = macro.submesh(0, 3)
    fitted = fit_material_mesh(fine, field)
    validate_submesh(macro, 0, fitted)
    assert (len(fitted.points), len(fitted.cells), len(fitted.boundary_faces)) == (235, 426, 42)
    coords, errors = cartesian_coordinates(fine.points, np.asarray(origin), np.array([0.3, 0.3]))
    del coords
    envelope = 2 * errors.max() * 0.3 + 64 * np.finfo(float).eps * 7 * 0.3
    assert abs(fitted.areas.sum() - 3) <= 16 * envelope
    areas = np.bincount(
        field(fitted.points[fitted.cells].mean(axis=1)).astype(int),
        weights=fitted.areas,
        minlength=49,
    ).reshape(7, 7)
    bounds = np.arange(8) * 0.3
    dx2 = np.maximum(0, np.minimum(bounds[1:], 2) - bounds[:-1])
    dx1 = np.maximum(0, np.minimum(bounds[1:], 1) - bounds[:-1])
    dy12 = np.maximum(0, np.minimum(bounds[1:], 2) - np.maximum(bounds[:-1], 1))
    expected = dx2[:, None] * dx1[None, :] + dx1[:, None] * dy12[None, :]
    assert_allclose(areas, expected, atol=16 * envelope, rtol=0)
    reference = fit_material_mesh(
        l_shape((0, 0)).submesh(0, 3), CartesianCellField(np.ones((7, 7)), (0.3, 0.3))
    )
    distance, mapping = cKDTree(reference.points).query(fitted.points - origin)
    assert distance.max() <= 4 * envelope
    assert len(np.unique(mapping)) == len(mapping)
    assert {tuple(sorted(row)) for row in mapping[fitted.cells]} == {
        tuple(sorted(row)) for row in reference.cells
    }


@pytest.mark.parametrize("origin", ORIGINS)
def test_domain_and_strict_one_sided_traces(origin):
    """Keep opposite values distinct at represented interfaces and reject extrapolation."""
    field = CartesianCellField(np.arange(49).reshape(7, 7), (0.3, 0.3), origin)
    origin = np.asarray(origin)
    assert field([origin + 2.1])[0] == 48
    with pytest.raises(ValueError, match="outside"):
        field([origin + [2.1 + 1e-6, 0]])
    points = origin + np.column_stack((0.3 * np.arange(1, 7), np.full(6, 0.15)))
    left = cartesian_trace_values(field, points, points - [0.05, 0])
    right = cartesian_trace_values(field, points, points + [0.05, 0])
    assert_array_equal(left, 7 * np.arange(6))
    assert_array_equal(right, 7 * np.arange(1, 7))
    assert_array_equal(field(points), right)
    with pytest.raises(ValueError, match="strict side"):
        cartesian_trace_values(field, points, points + [0, 0.02])


@pytest.mark.parametrize("origin", ORIGINS)
def test_edge_partition_keeps_existing_breaks_and_integrates_material(origin):
    """Do not introduce tiny duplicate intervals at matching pixel intersections."""
    field = CartesianCellField(np.arange(49).reshape(7, 7), (0.3, 0.3), origin)
    origin = np.asarray(origin)
    existing = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
    start, end = origin + [0, 0.15], origin + [2, 0.15]
    assert_array_equal(_material_edge_breaks(start, end, field, existing), existing)
    points, weights, values = cartesian_edge_quadrature(
        start, end, field, 3, interior_point=origin + [0.5, 0.2]
    )
    assert_allclose(weights.sum(), 1, atol=3e-15, rtol=0)
    expected = (0.3 * 7 * np.arange(6).sum() + 0.2 * 42) / 2
    assert_allclose(weights @ values, expected, atol=5e-11, rtol=0)
    assert np.all(np.diff(np.sort(np.unique(points[:, 0]))) > 0.001)


def test_coincidence_chains_cannot_extend_the_roundoff_envelope():
    """A connected graph does not authorize merging distinguishable endpoints."""
    points = np.array([[0.0, 0.0], [0.09, 0.0], [0.18, 0.0]])
    with pytest.raises(ValueError, match="envelope"):
        _merge_grid_vertices(points, 0.1)
    first, groups = _merge_grid_vertices(points[:2], 0.1)
    assert_array_equal(first, [0])
    assert_array_equal(groups, [0, 0])


@pytest.mark.parametrize("dimension", [2, 3])
def test_unresolved_pixel_geometry_is_rejected(dimension):
    """Pixels below physical coordinate resolution must not be silently merged."""
    field = CartesianCellField(np.ones((2,) * dimension), (0.1,) * dimension, (1e16,) * dimension)
    with pytest.raises(ValueError, match="distinguishable"):
        field(np.full((1, dimension), 1e16))


@pytest.mark.parametrize("origin", ORIGINS)
def test_cut_triangle_quadrature_resolves_independent_material_areas(origin):
    """A right triangle has analytically known area in each vertical material strip."""
    origin = np.asarray(origin)
    mesh = TriangleMesh(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]) + origin, [[0, 1, 2]])
    field = CartesianCellField(np.arange(7)[:, None] + np.ones((7, 7)), (0.3, 0.3), tuple(origin))
    bary, weights, pixels = cartesian_triangle_quadrature(mesh, field, 3, return_cells=True)
    values = field.values[tuple(pixels[0].T)]
    edges = np.r_[np.arange(7) * 0.3, 2.0]
    areas = 2 * np.diff(edges) - np.diff(edges**2) / 2
    expected = (np.arange(7) + 1) @ areas
    assert_allclose(2 * weights[0] @ values, expected, atol=3e-11, rtol=0)
    assert_allclose(bary.sum(axis=2), 1, atol=5e-16, rtol=0)
    assert np.all(weights >= 0)
    fitted = fit_material_mesh(mesh, field)
    fb, fw, fv = material_triangle_quadrature(fitted, field, 3)
    fv = fv.reshape(len(fitted.cells), -1)
    assert_allclose(np.einsum("t,tq,tq->", fitted.areas, fw, fv), expected, atol=3e-11, rtol=0)
    assert fb.shape[1] == 9


@pytest.mark.parametrize("origin", ORIGINS)
def test_rectangle_integration_and_alignment_are_translation_covariant(origin):
    """Verify aligned-grid detection and constant-gradient material energy independently."""
    origin = np.asarray(origin)
    field = CartesianCellField(np.arange(1, 50).reshape(7, 7), (0.3, 0.3), tuple(origin))
    mesh = CartesianMacroMesh(7, 7, (origin[0], origin[0] + 2.1, origin[1], origin[1] + 2.1))
    assert _grid_resolves_material(mesh, field)
    cut_mesh = CartesianMacroMesh(2, 2, (origin[0], origin[0] + 2.0, origin[1], origin[1] + 2.0))
    assert not _grid_resolves_material(cut_mesh, field)
    origins = cut_mesh.points[cut_mesh.cells[:, 0]]
    reference, weights = _cartesian_rectangle_quadrature(origins, cut_mesh.spacing, field, 3)
    assert_allclose(weights.sum(axis=1), 1, atol=2e-12, rtol=0)
    points = origins[:, None, :] + reference * cut_mesh.spacing
    values = field(points.reshape(-1, 2)).reshape(weights.shape)
    lengths = np.r_[np.full(6, 0.3), 0.2]
    expected = lengths @ field.values @ lengths
    assert_allclose(np.sum(weights * values), expected, atol=2e-9, rtol=0)
    matrix, _, _ = quadrilateral_operators(cut_mesh, 1, permeability=field)
    nodal = cut_mesh.points[:, 0] - origin[0]
    assert_allclose(nodal @ matrix @ nodal, expected, atol=2e-9, rtol=0)


@pytest.mark.parametrize("origin", ORIGINS)
def test_translated_fitted_nonconvex_affine_diffusion_patch(origin):
    """Recover a nonhomogeneous affine pressure with a discontinuous SPD material."""
    from pymhm.polygon import solve_darcy_polygons

    macro = l_shape(origin)
    tensor = np.zeros((7, 7, 2, 2))
    tensor[..., 0, 0] = 1 + np.arange(7)[:, None]
    tensor[..., 1, 1] = 1
    field = CartesianCellField(tensor, (0.3, 0.3), origin)
    fine = fit_material_mesh(macro.submesh(0, 3), field)

    def exact(points):
        """Select a gradient with constant physical flux across all material jumps."""
        return 1 + points[:, 1] - origin[1]

    result = solve_darcy_polygons(
        macro, degree=1, permeability=field, dirichlet=exact, local_meshes=(fine,)
    )
    assert result.l2_error(exact) < 3e-10
    assert result.flux_l2_error(lambda x: np.tile([0.0, -1.0], (len(x), 1))) < 3e-10


def test_large_physical_unit_does_not_impose_a_fictitious_unit_grid():
    """Scaling a well-resolved translated mesh preserves material-grid alignment."""
    origin = np.array([1e18, -1e18])
    step = 1e10
    field = CartesianCellField(np.ones((7, 7)), (step, step), tuple(origin))
    mesh = CartesianMacroMesh(
        14, 14, (origin[0], origin[0] + 7 * step, origin[1], origin[1] + 7 * step)
    )
    assert _grid_resolves_material(mesh, field)
    assert_allclose(field(mesh.points), 1, rtol=0, atol=0)
