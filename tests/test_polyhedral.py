"""Geometric identities and original-face invariance of convex polyhedral local meshes."""

import numpy as np
import pytest

from pymhm._legacy.models.transport.polyhedral import (
    PolygonalSkeleton3D,
    polygonal_trace_coupling,
    solve_polyhedral_rad,
)
from pymhm.fem.scalar.tetrahedron import tetra_nodal_space
from pymhm.meshes.polygonal import PolygonMesh
from pymhm.meshes.polyhedral import PolyhedralMesh


def two_cells() -> PolyhedralMesh:
    """Construct two cubes sharing one original quadrilateral face."""
    points = np.array([[0, 0], [0.5, 0], [1, 0], [0, 1], [0.5, 1], [1, 1]])
    polygon = PolygonMesh(points, (np.array([0, 1, 4, 3]), np.array([1, 2, 5, 4])))
    return PolyhedralMesh.extrude(polygon)


def affine(points: np.ndarray) -> np.ndarray:
    """Use a nonhomogeneous affine physical field on all exterior faces."""
    return 1 + points[:, 0] + 2 * points[:, 1] - points[:, 2]


@pytest.mark.parametrize("refinement", [1, 2, 4])
def test_volume_surface_and_divergence_theorem(refinement: int) -> None:
    """Retain area/volume and exact affine flux moments through tetrahedral refinement."""
    mesh = two_cells()
    assert len(mesh.cells) == 2 and len(mesh.faces) == 11
    np.testing.assert_allclose(mesh.volumes, [0.5, 0.5], atol=1e-15)
    assert np.sum(mesh.areas[mesh.boundary_faces]) == pytest.approx(6)
    skeleton = PolygonalSkeleton3D(mesh)
    assert skeleton.size == 33
    for cell in range(2):
        fine = mesh.submesh(cell, refinement)
        assert fine.volumes.sum() == pytest.approx(mesh.volumes[cell])
        external = fine.boundary_faces
        assert fine.areas[external].sum() == pytest.approx(4)
        coupling = polygonal_trace_coupling(mesh, cell, fine, skeleton, 2)
        _, nodes = tetra_nodal_space(fine, 2)
        vector = np.array([0.5, -1, 2.0])
        coefficients = np.concatenate(
            [
                skeleton.constant_coefficients[skeleton.dofs(int(face))]
                * (vector @ mesh.normals[face])
                for face in mesh.cell_faces[cell]
            ]
        )
        np.testing.assert_allclose(np.ones(len(nodes)) @ coupling @ coefficients, 0, atol=3e-14)
        np.testing.assert_allclose(
            affine(nodes) @ coupling @ coefficients,
            mesh.volumes[cell] * np.array([1, 2, -1]) @ vector,
            atol=3e-14,
        )
    for face in np.flatnonzero(mesh.face_cells[:, 1] >= 0):
        first, second = mesh.face_cells[face]
        assert mesh.signs[first][np.flatnonzero(mesh.cell_faces[first] == face)[0]] == 1
        assert mesh.signs[second][np.flatnonzero(mesh.cell_faces[second] == face)[0]] == -1


def test_face_reordering_reversal_preserves_physical_field() -> None:
    """Reverse cyclic vertices and reorder face lists, preserving a shared quadrilateral's field."""
    original = two_cells()
    permutation = np.arange(len(original.faces))[::-1]
    inverse = np.argsort(permutation)
    modified = PolyhedralMesh(
        original.points,
        [np.roll(original.faces[i][::-1], 1) for i in permutation],
        [inverse[faces][::-1] for faces in original.cells],
    )
    results = [
        solve_polyhedral_rad(mesh, degree=4, dirichlet=affine) for mesh in (original, modified)
    ]
    for result in results:
        assert result.l2_error(affine) < 3e-13
        assert result.h1_seminorm_error((1, 2, -1)) < 3e-12
        assert result.flux_l2_error((-1, -2, 1)) < 3e-12
    for cell in range(2):
        orders = []
        for result in results:
            _, nodes = tetra_nodal_space(result.local_meshes[cell], result.degree)
            orders.append(np.lexsort(nodes.T))
        np.testing.assert_allclose(
            results[0].values[cell][orders[0]], results[1].values[cell][orders[1]], atol=3e-12
        )


def test_general_convex_octahedron_and_prisms() -> None:
    """Accept nonhexahedral cells and polygonal faces with more than four vertices."""
    vertices = np.r_[np.eye(3), -np.eye(3)]
    faces = [[a, b, c] for a in (0, 3) for b in (1, 4) for c in (2, 5)]
    octahedron = PolyhedralMesh(vertices, faces, [list(range(8))])
    assert octahedron.volumes[0] == pytest.approx(4 / 3)
    angles = np.arange(6) * np.pi / 3
    base = PolygonMesh(np.column_stack((np.cos(angles), np.sin(angles))), (np.arange(6),))
    prism = PolyhedralMesh.extrude(base, 2, interval=(-1, 1))
    assert prism.volumes.sum() == pytest.approx(3 * np.sqrt(3))
    assert max(map(len, prism.faces)) == 6
    result = solve_polyhedral_rad(prism, degree=2, dirichlet=affine)
    assert result.l2_error(affine) < 3e-12


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate_face",
        "duplicate_cell",
        "bad_index",
        "fractional",
        "open",
        "nonplanar",
        "concave",
        "unused_face",
        "zero",
    ],
)
def test_geometry_contracts(kind: str) -> None:
    """Reject invalid topology and geometry before tetrahedralization or assembly."""
    mesh = PolyhedralMesh.cubes()
    points = mesh.points.copy()
    faces = [f.tolist() for f in mesh.faces]
    cells = [list(range(6))]
    if kind == "duplicate_face":
        faces.append(faces[0])
    elif kind == "duplicate_cell":
        cells.append(cells[0])
    elif kind == "bad_index":
        faces[0][0] = 99
    elif kind == "fractional":
        faces[0][0] = 0.5
    elif kind == "open":
        cells[0] = cells[0][:-1]
    elif kind == "nonplanar":
        points[0] += [0.1, 0.1, 0.1]
    elif kind == "concave":
        faces[0][1], faces[0][2] = faces[0][2], faces[0][1]
    elif kind == "unused_face":
        faces.append([0, 1, 2])
    elif kind == "zero":
        points[:] = 0
    with pytest.raises(ValueError):
        PolyhedralMesh(points, faces, cells)


def test_bad_inputs_and_skeleton_contracts() -> None:
    """Validate dimensions, extrusion, face degrees, and explicit index ranges."""
    mesh = two_cells()
    for points in (np.zeros((3, 3)), np.ones((4, 2)), np.ones((4, 3)) * 1j):
        with pytest.raises(ValueError):
            PolyhedralMesh(points, mesh.faces, mesh.cells)
    with pytest.raises(ValueError, match="empty"):
        PolyhedralMesh(mesh.points, [], [])
    with pytest.raises(TypeError, match="PolygonMesh"):
        PolyhedralMesh.extrude(None)
    polygon = PolygonMesh(np.array([[0, 0], [1, 0], [0, 1]]), (np.array([0, 1, 2]),))
    with pytest.raises(ValueError, match="interval"):
        PolyhedralMesh.extrude(polygon, interval=(1, 0))
    with pytest.raises(ValueError, match="outside"):
        mesh.submesh(2)
    with pytest.raises(ValueError, match="power"):
        mesh.submesh(0, 3)
    with pytest.raises(TypeError, match="PolyhedralMesh"):
        PolygonalSkeleton3D(None)
    for degree in ([0, 1], 2):
        with pytest.raises(ValueError, match="degree"):
            PolygonalSkeleton3D(mesh, degree)
    skeleton = PolygonalSkeleton3D(mesh, 0)
    assert skeleton.size == len(mesh.faces)
    np.testing.assert_array_equal(skeleton.basis(0, np.zeros((2, 3))), np.ones((2, 1)))
    for action in (lambda: skeleton.dofs(99), lambda: skeleton.cell_dofs(99)):
        with pytest.raises(ValueError, match="outside"):
            action()
    with pytest.raises(ValueError, match="XYZ"):
        skeleton.basis(0, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="belong"):
        polygonal_trace_coupling(mesh, 0, mesh.submesh(0), PolygonalSkeleton3D(two_cells()), 2)


def test_multiple_cubes_and_star_shaped_nonconvex_geometry() -> None:
    """Check shared generated faces and accept nonconvex cells with a geometric kernel."""
    mesh = PolyhedralMesh.cubes(2)
    assert len(mesh.cells) == 8 and mesh.volumes.sum() == pytest.approx(1.0)
    polygon = PolygonMesh(
        np.array([[0.0, 0.0], [1, 0], [0.4, 0.4], [1, 1], [0, 1]]), (np.arange(5),)
    )
    prism = PolyhedralMesh.extrude(polygon)
    assert prism.volumes[0] == pytest.approx(0.7)
    assert not prism.convex_cells[0]
    vertices = np.r_[np.eye(3), -np.eye(3)]
    vertices[0] = [-0.1, 0, 0]
    faces = [[a, b, c] for a in (0, 3) for b in (1, 4) for c in (2, 5)]
    dented = PolyhedralMesh(vertices, faces, [list(range(8))])
    assert dented.volumes[0] == pytest.approx(0.6)
    assert dented.kernel_radii[0] > 0


def test_geometric_validation_uses_the_individual_face_and_cell_scale() -> None:
    """Accept resolvable tiny cells without measuring their area against the entire domain."""
    original = PolyhedralMesh.cubes(2)
    points = original.points.copy()
    points[points == 0.5] = 1e-8
    mesh = PolyhedralMesh(points, original.faces, original.cells)
    assert mesh.volumes[0] == pytest.approx(1e-24, rel=2e-14, abs=0)
    assert mesh.volumes.sum() == pytest.approx(1)
    fine = mesh.submesh(0, 2)
    assert fine.volumes.sum() == pytest.approx(mesh.volumes[0], rel=2e-14, abs=0)
    coupling = polygonal_trace_coupling(mesh, 0, fine, PolygonalSkeleton3D(mesh, 0), 2)
    np.testing.assert_allclose(
        coupling.sum(axis=0), mesh.areas[mesh.cell_faces[0]], rtol=3e-14, atol=0
    )


def test_distinct_coplanar_macrofaces_keep_independent_moments() -> None:
    """Integrate each half of a coplanar boundary independently instead of duplicating its plane."""
    original = two_cells()
    faces = [original.faces[face] for face in original.boundary_faces]
    mesh = PolyhedralMesh(original.points, faces, [np.arange(len(faces))])
    assert len(mesh.faces) == 10
    skeleton = PolygonalSkeleton3D(mesh)
    for refinement in (1, 2):
        fine = mesh.submesh(0, refinement)
        _, nodes = tetra_nodal_space(fine, 3)
        coupling = polygonal_trace_coupling(mesh, 0, fine, skeleton, 3)
        for face in range(len(mesh.faces)):
            constant = coupling[:, skeleton.dofs(face)[0]]
            np.testing.assert_allclose(constant.sum(), mesh.areas[face], atol=3e-14)
            np.testing.assert_allclose(
                nodes.T @ constant,
                mesh.areas[face] * mesh.face_origins[face],
                atol=3e-14,
            )
    solution = solve_polyhedral_rad(mesh, degree=4, dirichlet=affine)
    assert solution.l2_error(affine) < 3e-12
    assert solution.h1_seminorm_error((1, 2, -1)) < 3e-11
