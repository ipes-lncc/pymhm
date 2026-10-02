"""Polygon topology and physical polynomial patches on nonconvex macrocells."""

import numpy as np
import pytest

from pymhm.mesh import FaceSpace, SkeletonSpace
from pymhm.polygon import (
    PolygonMesh,
    solve_brinkman_polygons,
    solve_darcy_polygons,
    solve_mshho_polygons,
    solve_transport_polygons,
)


def partition():
    """Partition the unit square into an L and a square, preserving collinear nodes."""
    return PolygonMesh(
        np.array(((0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1), (1, 1))),
        (np.array((0, 1, 2, 3, 4, 5)), np.array((3, 2, 6, 4))),
    )


def pressure(x):
    """Affine pressure with nonzero boundary values."""
    return 1 + x[:, 0] + 2 * x[:, 1]


def test_nonconvex_geometry_and_refinement():
    """Ear refinement preserves area, conformity, orientation and macro boundaries."""
    mesh = partition()
    np.testing.assert_allclose(mesh.areas, (0.75, 0.25))
    assert not mesh.points.flags.writeable
    assert len(mesh.cell_faces[0]) == 6
    assert mesh.signs[0, 0] == 1
    np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)
    for cell in range(2):
        faces = mesh.cell_faces[cell]
        outward = mesh.signs[cell][:, None] * mesh.normals[faces]
        np.testing.assert_allclose(mesh.lengths[faces] @ outward, 0, atol=2e-15)
        centers = mesh.points[mesh.faces[faces]].mean(axis=1)
        np.testing.assert_allclose(
            np.sum(mesh.lengths[faces] * np.einsum("fi,fi->f", centers, outward)),
            2 * mesh.areas[cell],
        )
        for r in (1, 2, 3):
            fine = mesh.submesh(cell, r)
            assert len(fine.cells) == (len(mesh.cells[cell]) - 2) * r**2
            np.testing.assert_allclose(fine.areas.sum(), mesh.areas[cell])
            np.testing.assert_allclose(
                fine.lengths[fine.boundary_faces].sum(), mesh.lengths[mesh.cell_faces[cell]].sum()
            )
    reversed_mesh = PolygonMesh(mesh.points, tuple(c[::-1] for c in mesh.cells))
    np.testing.assert_allclose(reversed_mesh.areas, mesh.areas)


@pytest.mark.parametrize(
    "shape",
    [
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)],
        [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)],
    ],
)
def test_boundary_separated_triangulation_and_affine_patch(shape):
    """Every macroface has its own adjacent triangle, even on a non-star-shaped cell."""
    mesh = PolygonMesh(np.array(shape), (np.arange(len(shape)),), local_triangulation="boundary")
    fine = mesh.submesh(0, 1)
    adjacent = fine.face_cells[fine.boundary_faces, 0]
    assert len(np.unique(adjacent)) == len(adjacent)
    np.testing.assert_allclose(fine.areas.sum(), mesh.areas[0], atol=3e-15)
    for r in (2, 3):
        refined = mesh.submesh(0, r)
        np.testing.assert_allclose(refined.areas.sum(), mesh.areas[0], atol=5e-15)
        assert len(refined.cells) == len(fine.cells) * r * r
    result = solve_darcy_polygons(mesh, dirichlet=pressure, degree=2, local_refinement=1)
    assert result.l2_error(pressure) < 3e-12
    assert result.flux_l2_error((-1.0, -2.0)) < 3e-12
    with pytest.raises(ValueError, match="local_triangulation"):
        PolygonMesh(mesh.points, mesh.cells, local_triangulation="undocumented")


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_polygon_darcy_affine_patch(formulation):
    """Both shared Darcy formulations reproduce the physical affine patch."""
    mesh = partition()
    result = solve_darcy_polygons(
        mesh, formulation=formulation, dirichlet=pressure, local_refinement=2
    )
    if formulation == "primal":
        assert result.l2_error(pressure) < 2e-13
    else:
        # RT0 pressure is the cellwise mean, while its constant flux is exact.
        for fine, values in zip(result.local_meshes, result.pressure, strict=True):
            np.testing.assert_allclose(
                values, pressure(fine.points[fine.cells].mean(axis=1)), atol=2e-13
            )
    assert result.flux_l2_error((-1.0, -2.0)) < 2e-13


@pytest.mark.parametrize("enforcement", ["weak", "strong"])
def test_polygon_rad_affine_patch(enforcement):
    """The half-advection trace of an affine field needs degree one."""
    mesh = partition()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))

    def source(x):
        """Source for -Delta u + (1,0).grad u + u."""
        return 1 + pressure(x)

    result = solve_transport_polygons(
        mesh,
        skeleton=skeleton,
        degree=2,
        local_refinement=2,
        velocity=(1.0, 0.0),
        reaction=1.0,
        source=source,
        dirichlet=pressure,
        dirichlet_enforcement=enforcement,
    )
    assert result.l2_error(pressure) < 2e-13


def test_polygon_brinkman_affine_patch():
    """Incompressible affine velocity and constant pressure survive nonconvex condensation."""
    mesh = partition()

    def velocity(x):
        """Divergence-free affine field."""
        return np.column_stack((x[:, 0], -x[:, 1]))

    result = solve_brinkman_polygons(
        mesh,
        skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2),
        dirichlet=velocity,
        local_refinement=2,
        degree=2,
        formulation="taylor-hood",
    )
    assert result.l2_error(velocity) < 3e-13


@pytest.mark.parametrize("cell_degree", [0, 1])
def test_polygon_mshho_field_equivalence(cell_degree):
    """MHM and MsHHO agree for polynomial forcing on a nonconvex partition."""
    mesh = partition()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    options = dict(skeleton=skeleton, degree=3, source=2.0, dirichlet=pressure, local_refinement=2)
    mhm = solve_darcy_polygons(mesh, **options)
    hho = solve_mshho_polygons(mesh, cell_degree=cell_degree, **options)
    for a, b in zip(mhm.pressure, hho.pressure, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-12)


@pytest.mark.parametrize(
    "points,cells,match",
    [
        ([[0, 0], [1, 0], [0, 1j]], [[0, 1, 2]], "real"),
        ([[0, 0, 0]], [[0, 1, 2]], "XY"),
        ([[0, 0], [0, 0], [0, 1]], [[0, 1, 2]], "unique"),
        ([[0, 0], [1, 0], [0, 1]], [], "at least"),
        ([[0, 0], [1, 0], [0, 1]], [[0, 1]], "three"),
        ([[0, 0], [1, 0], [0, 1]], [[0.0, 1.0, 2.0]], "integer"),
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 3]], "distinct"),
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 1]], "distinct"),
        ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]], "area"),
        ([[0, 0], [1, 1], [0, 1], [2, 0]], [[0, 1, 2, 3]], "self-intersects"),
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2], [0, 1, 2]], "nonmanifold"),
        ([[0, 0], [1, 0], [0, 1], [0.5, 0]], [[0, 1, 2]], "hanging"),
    ],
)
def test_polygon_invalid_topology(points, cells, match):
    """Reject geometries which cannot define a conforming simple-polygon partition."""
    with pytest.raises(ValueError, match=match):
        PolygonMesh(np.asarray(points), tuple(np.asarray(c) for c in cells))


def test_polygon_unsupported_point_well_and_cell():
    """Reject triangular-only point allocation and invalid macro indices explicitly."""
    mesh = partition()
    with pytest.raises(ValueError, match="point wells"):
        solve_darcy_polygons(mesh, point_sources=[((0.1, 0.1), 1)])
    with pytest.raises(ValueError, match="cell index"):
        mesh.submesh(2, 1)
