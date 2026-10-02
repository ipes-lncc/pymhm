"""Nonconvex polyhedral geometry, exact physical patches and original-face invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm import PolygonMesh, PolyhedralMesh
from pymhm.polyhedral_geometry import (
    kernel_center,
    oriented_cell_faces,
    triangle_in_face,
    validate_disjoint_cones,
)
from pymhm.polyhedral_rad import PolygonalSkeleton3D, polygonal_trace_coupling, solve_polyhedral_rad
from pymhm.tetrahedral import tetra_nodal_space


@pytest.fixture(autouse=True)
def native_threads():
    """Bound native threading for small geometric and local mixed systems."""
    with threadpool_limits(1):
        yield


def l_prism(layers=2):
    """Extrude an L-shaped area0.64 without filling its square reentrant notch."""
    polygon = PolygonMesh(
        np.array([[0.0, 0.0], [1, 0], [1, 0.4], [0.4, 0.4], [0.4, 1], [0, 1]]),
        (np.arange(6),),
    )
    return PolyhedralMesh.extrude(polygon, layers)


def exact(points):
    """Quadratic potential with nonzero flux and nonhomogeneous boundary values."""
    x, y, z = points.T
    return x * x + y * y + 0.5 * z * z + x + y


def gradient(points):
    """Independently differentiated affine gradient of the manufactured potential."""
    return np.column_stack((2 * points[:, 0] + 1, 2 * points[:, 1] + 1, points[:, 2]))


@pytest.mark.parametrize("refinement", [1, 2])
def test_reentrant_volume_surface_and_original_face_moments(refinement):
    """Tetrahedra cover the L prism, retain its concave shared face and satisfy Gauss identities."""
    mesh = l_prism()
    assert len(mesh.faces) == 15
    assert not mesh.convex_cells.any()
    assert_allclose(mesh.volumes, [0.32, 0.32], atol=2e-15)
    assert_allclose(mesh.kernel_radii, 0.2, atol=2e-15)
    assert_allclose(mesh.areas[mesh.boundary_faces].sum(), 5.28, atol=2e-15)
    skeleton = PolygonalSkeleton3D(mesh)
    assert skeleton.size == 45
    for cell in range(2):
        fine = mesh.submesh(cell, refinement)
        assert_allclose(fine.volumes.sum(), 0.32, atol=2e-15)
        centers = fine.points[fine.cells].mean(axis=1)
        assert np.all((centers[:, 0] < 0.4) | (centers[:, 1] < 0.4))
        matrix = polygonal_trace_coupling(mesh, cell, fine, skeleton, 2)
        _, nodes = tetra_nodal_space(fine, 2)
        offset = 0
        for side, face in enumerate(mesh.cell_faces[cell]):
            constant = matrix[:, offset]
            sign = mesh.signs[cell][side]
            assert_allclose(constant.sum(), sign * mesh.areas[face], atol=4e-14)
            assert_allclose(
                nodes.T @ constant, sign * mesh.areas[face] * mesh.face_origins[face], atol=4e-14
            )
            offset += 3
        normal = mesh.normals[mesh.cell_faces[cell]]
        area = mesh.areas[mesh.cell_faces[cell]]
        outward = normal * mesh.signs[cell][:, None]
        assert_allclose(area @ outward, 0, atol=2e-15)
        first_moment = np.einsum(
            "f,fi,fj->ij", area, mesh.face_origins[mesh.cell_faces[cell]], outward
        )
        assert_allclose(first_moment, 0.32 * np.eye(3), atol=2e-15)


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_complete_anisotropic_quadratic_patch(boundary):
    """Nonconvex patches reproduce quadratic pressure and affine flux for all boundary types."""
    mesh = l_prism()
    tensor = np.array([[2.0, 0.2, -0.1], [0.2, 1.4, 0.1], [-0.1, 0.1, 0.7]])
    natural = {}
    if boundary != "dirichlet":
        selected = mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:3]
        for face in selected:
            normal = mesh.normals[face].copy()
            natural[int(face)] = lambda points, normal=normal: -gradient(points) @ tensor @ normal
    options = {}
    if boundary == "neumann":
        integral_xy = 1 / 3 - 0.6 * (1 - 0.4**3) / 3
        integral_x = 0.5 - 0.6 * (1 - 0.4**2) / 2
        options["mean_value"] = (2 * integral_xy + 2 * integral_x + 0.64 / 6) / 0.64
    solution = solve_polyhedral_rad(
        mesh,
        degree=3,
        diffusion=tensor,
        source=-2 * tensor[0, 0] - 2 * tensor[1, 1] - tensor[2, 2],
        dirichlet=exact,
        neumann=natural,
        quadrature_order=6,
        **options,
    )
    assert solution.l2_error(exact, 7) < 2e-12
    assert solution.h1_seminorm_error(gradient, 7) < 3e-11
    assert solution.flux_l2_error(lambda points: -gradient(points) @ tensor, 7) < 5e-11
    assert solution.hybrid.residual < 1e-11


def test_reordering_shared_concave_face_and_sheared_cells():
    """Full physical affine fields survive cyclic reversal, cell/face ordering and affine shear."""
    original = l_prism()
    transform = np.array([[1.2, 0.15, 0.1], [0.1, 0.8, -0.05], [0.03, 0.2, 1.1]])
    shift = np.array([0.3, -0.5, 0.2])
    points = original.points @ transform.T + shift
    permutation = np.arange(len(original.faces))[::-1]
    inverse = np.argsort(permutation)
    meshes = (
        PolyhedralMesh(points, original.faces, original.cells),
        PolyhedralMesh(
            points,
            [np.roll(original.faces[i][::-1], 2) for i in permutation],
            [inverse[ids][::-1] for ids in original.cells[::-1]],
        ),
    )
    for mesh in meshes:
        assert_allclose(mesh.volumes.sum(), 0.64 * np.linalg.det(transform), atol=3e-15)
        solution = solve_polyhedral_rad(mesh, degree=2, dirichlet=lambda x: 1 + x @ [1, 2, -1])
        assert solution.l2_error(lambda x: 1 + x @ [1, 2, -1]) < 2e-12
        assert solution.h1_seminorm_error([1, 2, -1]) < 3e-11


def test_nonaffine_field_is_invariant_under_face_reordering():
    """Degenerate largest-ball centers and local polynomial fields do not depend on face order."""
    original = l_prism()
    permutation = np.random.default_rng(3).permutation(len(original.faces))
    inverse = np.argsort(permutation)
    reordered = PolyhedralMesh(
        original.points,
        [np.roll(original.faces[i][::-1], 2) for i in permutation],
        [inverse[ids][::-1] for ids in original.cells],
    )
    assert_allclose(original.centers, reordered.centers, atol=2e-15)
    results = [
        solve_polyhedral_rad(
            mesh,
            degree=3,
            source=lambda x: 1 + x[:, 0] - 2 * x[:, 1] + x[:, 2] ** 2,
            dirichlet=lambda x: 0.25 + x[:, 0] - x[:, 1],
        )
        for mesh in (original, reordered)
    ]
    for cell in range(2):
        first = tetra_nodal_space(results[0].local_meshes[cell], 3)[1]
        second = tetra_nodal_space(results[1].local_meshes[cell], 3)[1]
        distances, order = cKDTree(second).query(first)
        assert distances.max() < 2e-15
        assert_allclose(
            results[0].values[cell], results[1].values[cell][order], atol=2e-12, rtol=2e-12
        )


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_reactive_advective_patch_on_nonconvex_cells(stabilization):
    """The complete conservative RAD operator and Robin multiplier reproduce an affine field."""
    mesh = l_prism()
    beta = np.array([0.2, 0.1, 0.3])
    direction = np.array([1.0, 2.0, -1.0])

    def potential(points):
        """Evaluate exactly represented nonhomogeneous Dirichlet data."""
        return 1 + points @ direction

    solution = solve_polyhedral_rad(
        mesh,
        degree=3,
        diffusion=0.2,
        velocity=beta,
        reaction=0.4,
        source=lambda points: beta @ direction + 0.4 * potential(points),
        dirichlet=potential,
        stabilization=stabilization,
        quadrature_order=6,
    )
    assert solution.l2_error(potential) < 2e-12
    assert solution.h1_seminorm_error(direction) < 3e-11
    assert (
        solution.flux_l2_error(lambda points: -0.2 * direction + potential(points)[:, None] * beta)
        < 2e-11
    )


def test_empty_kernel_and_nonmanifold_surfaces_are_rejected():
    """A U-prism has an empty kernel; disconnected cavity shells are not star-shaped cells."""
    polygon = PolygonMesh(
        np.array([[0.0, 0.0], [1, 0], [1, 1], [0.7, 1], [0.7, 0.3], [0.3, 0.3], [0.3, 1], [0, 1]]),
        (np.arange(8),),
    )
    with pytest.raises(ValueError, match="kernel"):
        PolyhedralMesh.extrude(polygon)
    cube = PolyhedralMesh.cubes()
    points = np.vstack((cube.points, 0.2 + 0.6 * cube.points))
    faces = [*cube.faces, *(face + 8 for face in cube.faces)]
    with pytest.raises(ValueError, match="connected"):
        PolyhedralMesh(points, faces, [np.arange(12)])
    with pytest.raises(ValueError, match="closed"):
        oriented_cell_faces(cube.points, list(cube.faces), [np.arange(5)])
    with pytest.raises(ValueError, match="volume"):
        oriented_cell_faces(np.zeros((8, 3)), list(cube.faces), [np.arange(6)])


def test_cone_overlap_and_concave_face_membership():
    """Positive volume alone is insufficient: overlapping cones and a notch-spanning face fail."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]])
    with pytest.raises(ValueError, match="overlapping"):
        validate_disjoint_cones(points, np.array([[0, 1, 2, 3], [4, 1, 2, 3]]))
    mesh = l_prism(1)
    face = int(np.argmax([len(ids) for ids in mesh.faces]))
    triangles = mesh.points[mesh.face_triangles[face]]
    assert triangle_in_face(triangles[0], triangles, mesh.normals[face])
    spanning = mesh.points[mesh.faces[face]][[0, 2, 4]]
    assert not triangle_in_face(spanning, triangles, mesh.normals[face])
    displaced = triangles[0] + 0.01 * mesh.normals[face]
    assert not triangle_in_face(displaced, triangles, mesh.normals[face])
    with pytest.raises(ValueError, match="kernel ball"):
        kernel_center(np.eye(3), np.zeros((2, 3)), np.array([[1.0, 0, 0], [-1, 0, 0]]))


def test_collinear_face_vertices_and_reflex_initial_corner():
    """Keep collinear boundary vertices and correctly orient a reflex initial corner."""
    points = np.array([[0.0, 0.0], [0.5, 0], [1, 0], [1, 1], [0, 1]])
    mesh = PolyhedralMesh.extrude(PolygonMesh(points, (np.arange(5),)))
    assert mesh.volumes[0] == pytest.approx(1)
    assert max(map(len, mesh.faces)) == 5
    original = l_prism(1)
    reordered = PolyhedralMesh(
        original.points, [np.roll(face, -2) for face in original.faces], original.cells
    )
    assert reordered.volumes[0] == pytest.approx(0.64)
    fine = reordered.submesh(0)
    assert fine.volumes.sum() == pytest.approx(0.64)


def test_collapsed_face_overlapping_cells_and_nonorientable_topology():
    """Reject collapsed faces, overlapping neighbors and a projective-plane boundary."""
    cube = PolyhedralMesh.cubes()
    points = cube.points.copy()
    face = cube.faces[0]
    points[face] = np.column_stack((-2 + 0.1 * np.arange(4), np.full((4, 2), 0.25)))
    with pytest.raises(ValueError, match="nondegenerate"):
        PolyhedralMesh(points, cube.faces, cube.cells)
    top = np.flatnonzero(cube.points[:, 2] == 1)
    points = np.vstack((cube.points, cube.points[top] * [1, 1, 2]))
    mapping = np.arange(8)
    mapping[top] = np.arange(8, 12)
    faces = list(cube.faces)
    second = []
    for face in cube.faces:
        mapped = mapping[face]
        if np.array_equal(mapped, face):
            second.append(next(i for i, f in enumerate(faces) if np.array_equal(f, face)))
        else:
            second.append(len(faces))
            faces.append(mapped)
    with pytest.raises(ValueError, match="opposing"):
        PolyhedralMesh(points, faces, [np.arange(6), np.asarray(second)])
    projective = (
        np.asarray(
            [
                [1, 2, 3],
                [1, 2, 4],
                [1, 3, 5],
                [1, 4, 6],
                [1, 5, 6],
                [2, 3, 6],
                [2, 4, 5],
                [2, 5, 6],
                [3, 4, 5],
                [3, 4, 6],
            ]
        )
        - 1
    )
    with pytest.raises(ValueError, match="orientable"):
        oriented_cell_faces(
            np.random.default_rng(30).normal(size=(6, 3)), list(projective), [np.arange(10)]
        )
    flat = cube.points.copy()
    flat[:, 2] = 0
    with pytest.raises(ValueError, match="signed volume"):
        oriented_cell_faces(flat, list(cube.faces), [np.arange(6)])


@pytest.mark.meshing
def test_native_meshio_nonconvex_face_and_tag_roundtrip(tmp_path):
    """VTU polyhedron exchange preserves concave original faces, volumes and material tags."""
    pytest.importorskip("meshio")
    from pymhm.mesh_exchange import VolumeMeshData, read_volume_mesh, write_volume_mesh

    mesh = l_prism()
    tags = np.arange(len(mesh.cells)) + 10
    data = VolumeMeshData(mesh, cell_tags=tags, face_tags=np.arange(len(mesh.faces)))
    path = tmp_path / "reentrant.vtu"
    write_volume_mesh(path, data)
    actual = read_volume_mesh(path)
    assert len(actual.mesh.faces) == len(mesh.faces)
    assert max(map(len, actual.mesh.faces)) == 6
    assert_allclose(actual.mesh.volumes, mesh.volumes, rtol=0, atol=3e-16)
    assert_allclose(actual.mesh.kernel_radii, mesh.kernel_radii, rtol=0, atol=3e-16)
    assert not np.any(actual.mesh.convex_cells)
    np.testing.assert_array_equal(actual.cell_tags, tags)
    for cell in range(len(mesh.cells)):
        assert_allclose(actual.mesh.submesh(cell).volumes.sum(), mesh.volumes[cell], atol=3e-16)
