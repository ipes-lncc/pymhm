"""Physical tensor regions, exact one-sided traces and conforming cut geometry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.materials.planar import PlanarMaterial, PlanarRegion
from pymhm.meshes.fitting import fit_planar_material, fit_planar_skeleton, planar_face_partitions
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("dimension", [2, 3])
def test_oblique_halfspace_volume_and_first_moments(dimension):
    """The simplex x1+...+xd<=a has analytical volume and barycenter within the cube."""
    base = TriangleMesh.unit_square(2) if dimension == 2 else TetraMesh.unit_cube()
    material = PlanarMaterial(1, (PlanarRegion([np.ones(dimension)], [0.83], 1e-4),))
    result = fit_planar_material(base, material)
    volumes = result.mesh.areas if dimension == 2 else result.mesh.volumes
    from math import factorial

    assert_allclose(volumes.sum(), 1, atol=5e-15)
    selected = result.regions == 1
    assert_allclose(volumes[selected].sum(), 0.83**dimension / factorial(dimension), atol=5e-15)
    centers = result.mesh.points[result.mesh.cells].mean(axis=1)
    moment = np.sum(volumes[selected, None] * centers[selected], axis=0) / volumes[selected].sum()
    assert_allclose(moment, 0.83 / (dimension + 1), atol=5e-15)
    assert_allclose(result.tensors, material(centers), atol=0)
    for normal, offset in zip(*material.planes, strict=True):
        distances = result.mesh.points[result.mesh.cells] @ normal - offset
        assert not np.any((distances.min(axis=1) < -1e-13) & (distances.max(axis=1) > 1e-13))
    original = base.areas if dimension == 2 else base.volumes
    assert_allclose(np.bincount(result.parents, weights=volumes), original, atol=4e-15)


@pytest.mark.parametrize("dimension", [2, 3])
def test_centered_box_area_volume_and_translated_geometry(dimension):
    """A centered inclusion cuts macrocell interiors and conserves its exact physical measure."""
    base = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    shift = np.arange(dimension) * 12.5
    mesh = type(base)(base.points + shift, base.cells)
    material = PlanarMaterial(2, (PlanarRegion.box(0.27 + shift, 0.73 + shift, 1e-4),))
    result = fit_planar_material(mesh, material)
    measure = result.mesh.areas if dimension == 2 else result.mesh.volumes
    assert_allclose(measure[result.regions == 1].sum(), 0.46**dimension, atol=2e-14)
    assert_allclose(measure.sum(), 1, atol=2e-14)
    assert len(result.mesh.cells) > len(mesh.cells)


def test_regions_overlap_and_unambiguous_one_sided_values():
    """Later inclusions win; a material plane uses its incident-side tensor without perturbation."""
    tensor = np.array([[3.0, 0.5], [0.5, 2.0]])
    material = PlanarMaterial(
        1,
        (
            PlanarRegion.box([0.2, 0.2], [0.8, 0.8], 0.01),
            PlanarRegion([[1.0, 1.0]], [1.0], tensor),
        ),
    )
    assert_allclose(material.region_ids([[0.1, 0.1], [0.75, 0.75], [0.9, 0.9]]), [2, 1, 0])
    point = [[0.4, 0.6]]
    assert_allclose(material.trace_values(point, [0.3, 0.5]), tensor[None])
    assert_allclose(material.trace_values(point, [0.5, 0.7]), (0.01 * np.eye(2))[None])
    with pytest.raises(ValueError, match="strict side"):
        material.trace_values(point, point)
    # An unrelated plane outside the region cannot make the trace ambiguous.
    assert_allclose(material.trace_values([[0.2, 0.9]], [0.2, 1.0]), np.eye(2)[None])


def test_planar_skeleton_exact_intersections():
    """All oblique crossings are inserted as original macroface coordinates."""
    mesh = TriangleMesh.unit_square()
    material = PlanarMaterial(1, (PlanarRegion([[1.0, 1.0]], [0.83], 0.01),))
    skeleton = fit_planar_skeleton(mesh, material, degree=2)
    for face, ids in enumerate(mesh.faces):
        a, b = mesh.points[ids]
        space = skeleton.faces[face]
        midpoint = (np.array(space.breaks[:-1]) + space.breaks[1:]) / 2
        assert len(midpoint) in (1, 2)
        for first, second in zip(space.breaks[:-1], space.breaks[1:], strict=True):
            distance = (a + np.array([first, second])[:, None] * (b - a)).sum(axis=1) - 0.83
            assert not (distance.min() < -1e-14 and distance.max() > 1e-14)


def test_planar_data_validation():
    """Reject indefinite tensors, ambiguous shapes, dimensions and malformed trace points."""
    for normals, offsets in (
        ([[0.0, 0.0]], [0]),
        ([[1j, 0]], [1]),
        ([[1, 0]], [1j]),
        ([[1, 0]], []),
        ([[1, 0]], [np.nan]),
    ):
        with pytest.raises(ValueError):
            PlanarRegion(normals, offsets, 1)
    for tensor in (1j, [1, 2], [[1, 1], [0, 1]], -1, 0, np.nan):
        with pytest.raises(ValueError):
            PlanarRegion([[1, 0]], [0.5], tensor)
    for a, b in (([0, 0], [0, 1]), ([0j, 0], [1, 1]), ([0], [1])):
        with pytest.raises(ValueError):
            PlanarRegion.box(a, b, 1)
    region = PlanarRegion([[1, 0]], [0.5], 1)
    with pytest.raises(ValueError):
        PlanarMaterial(1, ())
    with pytest.raises(ValueError):
        PlanarMaterial(1, (region, PlanarRegion([[1, 0, 0]], [0.5], 1)))
    material = PlanarMaterial(1, (region,))
    for points in ([[1, np.nan]], [[1j, 0]], [[1, 0, 0]], [1, 0]):
        with pytest.raises(ValueError):
            material(points)
    with pytest.raises(ValueError):
        material.trace_values([[0, 0]], [[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        fit_planar_material(TetraMesh.unit_cube(), material)
    with pytest.raises(ValueError):
        fit_planar_skeleton(
            TriangleMesh.unit_square(), PlanarMaterial(1, (PlanarRegion([[1, 0, 0]], [0.5], 1),))
        )


def test_three_dimensional_material_face_partitions():
    """Every fitted boundary triangle belongs to exactly one material-fitted subface."""
    mesh = TetraMesh.unit_cube()
    rotation = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotation /= np.linalg.norm(rotation, axis=1)[:, None]
    normals = np.vstack((rotation, -rotation))
    center = np.array([0.5, 0.5, 0.5])
    material = PlanarMaterial(1, (PlanarRegion(normals, normals @ center + 0.19, 0.001),))
    partitions = planar_face_partitions(mesh, material)
    for partition in partitions:
        determinant = abs(np.linalg.det(partition))
        assert np.all(determinant > 0)
        assert_allclose(determinant.sum(), 1, atol=3e-14)
        assert_allclose(partition.sum(axis=2), 1, atol=2e-15)
        assert partition.min() > -2e-15
    for cell in range(len(mesh.cells)):
        fine = fit_planar_material(mesh.submesh(cell, 1), material).mesh
        for face in fine.boundary_faces:
            nodes = fine.points[fine.faces[face]]
            for parent in mesh.cell_faces[cell]:
                vertices = mesh.points[mesh.faces[parent]]
                if np.max(abs((nodes - vertices[0]) @ mesh.normals[parent])) > 1e-12:
                    continue
                xi = (nodes - vertices[0]) @ np.linalg.pinv((vertices[1:] - vertices[0]).T).T
                bary = np.column_stack((1 - xi.sum(1), xi))
                contained = [
                    np.min(bary @ np.linalg.inv(part)) >= -1e-12 for part in partitions[parent]
                ]
                assert sum(contained) == 1
    fitted = fit_planar_material(mesh, material)
    assert_allclose(fitted.mesh.volumes[fitted.regions == 1].sum(), 0.38**3, atol=1e-14)
    planar = PlanarMaterial(1, (PlanarRegion([[1.0, 0.0]], [0.5], 1),))
    with pytest.raises(ValueError):
        planar_face_partitions(mesh, planar)
