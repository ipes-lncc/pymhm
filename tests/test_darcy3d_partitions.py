"""Nonuniform physical face partitions and conforming explicit tetrahedral locals."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.core.validation import FloatArray
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary, tetra_trace_coupling
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.methods.hho_3d import _face_rule
from pymhm.recovery.moments_3d import reconstruct_darcy_moments_3d


def pressure(points: FloatArray) -> FloatArray:
    """Affine pressure with every Cartesian component nonzero."""
    return 1 + points @ np.array([1.0, 2.0, -0.5])


def fan_partition() -> FloatArray:
    """Triangulate a reference face with unequal area fractions."""
    vertices = np.eye(3)
    center = np.array([0.2, 0.3, 0.5])
    return np.array([[vertices[i], vertices[(i + 1) % 3], center] for i in range(3)])


def star_meshes(mesh: TetraMesh, partitions: tuple[FloatArray, ...]) -> tuple[TetraMesh, ...]:
    """Cone each boundary subtriangle to its macrocell centroid without duplicating nodes."""
    meshes = []
    for cell, indices in enumerate(mesh.cells):
        center = mesh.points[indices].mean(axis=0)
        tetrahedra = np.concatenate(
            [
                np.concatenate(
                    (
                        np.broadcast_to(center, (len(partitions[f]), 1, 3)),
                        partitions[f] @ mesh.points[mesh.faces[f]],
                    ),
                    axis=1,
                )
                for f in mesh.cell_faces[cell]
            ]
        )
        points, inverse = np.unique(tetrahedra.reshape(-1, 3), axis=0, return_inverse=True)
        meshes.append(TetraMesh(points, inverse.reshape(-1, 4)))
    return tuple(meshes)


@pytest.fixture(autouse=True)
def bounded_threads():
    """Keep CI dense local algebra single-threaded."""
    with threadpool_limits(1):
        yield


@pytest.fixture
def setup():
    """Two tetrahedra sharing a genuinely nonuniform triangular face partition."""
    cube = TetraMesh.unit_cube()
    mesh = TetraMesh(cube.points, cube.cells[:2])
    parts = tuple(fan_partition() for _ in mesh.faces)
    skeleton = TriangularSkeleton(mesh, degree=1, face_partitions=parts)
    return mesh, skeleton, star_meshes(mesh, parts)


def test_partition_weights_boundary_moments_and_mshho_rule(setup) -> None:
    """Physical surface integration weights, Pk means and weak boundary loads agree."""
    mesh, skeleton, _ = setup
    load, _ = _boundary(skeleton, 1.0, {}, 5)
    for face in range(len(mesh.faces)):
        assert_allclose(skeleton.face_weights(face), [0.5, 0.2, 0.3], atol=1e-15)
        assert_array_equal(skeleton.face_partition(face), fan_partition())
        points, weights, basis = _face_rule(mesh, skeleton, face, 5)
        assert_allclose(weights.sum(), mesh.areas[face], rtol=2e-15)
        assert_allclose(weights @ points, mesh.areas[face] * mesh.points[mesh.faces[face]].mean(0))
        if face in mesh.boundary_faces:
            assert_allclose(weights @ basis, load[skeleton.dofs(face)], atol=1e-15)
    constant = TriangularSkeleton(mesh, face_partitions=tuple(fan_partition() for _ in mesh.faces))
    loads, fixed = _boundary(constant, 1, {int(mesh.boundary_faces[0]): 2}, 4)
    for face in mesh.boundary_faces[1:]:
        assert_allclose(
            loads[constant.dofs(int(face))], mesh.areas[face] * constant.face_weights(int(face))
        )
    assert_allclose(list(fixed.values()), 2)


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_anisotropic_affine_patch_on_explicit_meshes(setup, boundary) -> None:
    """Nonuniform face moments retain affine fields, signed flux and physical mean."""
    mesh, skeleton, locals_ = setup
    permeability = np.array([[100.0, 2, 1], [2, 3, 0.2], [1, 0.2, 1]])
    flux = -permeability @ np.array([1.0, 2.0, -0.5])
    faces = mesh.boundary_faces if boundary == "neumann" else mesh.boundary_faces[:1]
    neumann = {} if boundary == "dirichlet" else {int(f): flux @ mesh.normals[f] for f in faces}
    exact_mean = mesh.volumes @ pressure(mesh.points[mesh.cells].mean(axis=1)) / mesh.volumes.sum()
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        local_meshes=locals_,
        degree=4,
        local_refinement=1,
        permeability=permeability,
        dirichlet=pressure,
        neumann=neumann,
        mean_pressure=exact_mean,
    )
    assert solution.l2_error(pressure) < 2e-10
    assert solution.flux_l2_error(flux) < 2e-9
    assert max(abs(solution.conservation_residuals())) < 2e-10
    reconstructed = reconstruct_darcy_moments_3d(solution, degree=1)
    assert reconstructed.flux_l2_error(flux) < 2e-9


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_explicit_meshes_preserve_worker_factory(setup, backend) -> None:
    """Explicit geometries travel through ordered thread/spawn workers unchanged."""
    mesh, skeleton, locals_ = setup
    result = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        local_meshes=locals_,
        degree=4,
        local_refinement=1,
        dirichlet=pressure,
        backend=backend,
        workers=2,
    )
    assert result.l2_error(pressure) < 2e-11


def test_uniform_partition_compatibility_and_rejections(setup) -> None:
    """Uniform API geometry is preserved, with malformed or unaligned data rejected."""
    mesh, skeleton, locals_ = setup
    uniform = TriangularSkeleton(mesh, 2, degree=2)
    explicit = TriangularSkeleton(
        mesh,
        2,
        degree=2,
        face_partitions=tuple(uniform.face_partition(f) for f in range(len(mesh.faces))),
    )
    assert_array_equal(uniform.offsets, explicit.offsets)
    fine = mesh.submesh(0, 2)
    assert_array_equal(
        tetra_trace_coupling(mesh, 0, fine, uniform, 3),
        tetra_trace_coupling(mesh, 0, fine, explicit, 3),
    )
    assert_array_equal(
        _boundary(uniform, pressure, {}, 5)[0], _boundary(explicit, pressure, {}, 5)[0]
    )
    with pytest.raises(ValueError, match="one partition"):
        TriangularSkeleton(mesh, face_partitions=())
    with pytest.raises(ValueError, match="one mesh"):
        solve_darcy_3d(mesh, local_meshes=locals_[:1])
    with pytest.raises(ValueError, match="contained|align"):
        solve_darcy_3d(mesh, skeleton=skeleton, degree=4, local_refinement=1)
    with pytest.raises(ValueError, match="outside"):
        skeleton.face_weights(len(mesh.faces))
    resolved = TriangularSkeleton(
        mesh, 8, degree=1, face_partitions=tuple(fan_partition() for _ in mesh.faces)
    )
    result = solve_darcy_3d(
        mesh,
        skeleton=resolved,
        local_meshes=locals_,
        local_refinement=1,
        degree=4,
        dirichlet=pressure,
    )
    assert result.l2_error(pressure) < 2e-11


def test_quadratic_pressure_uses_nonuniform_flux_means(setup) -> None:
    """Area-weighted subface means are necessary when normal flux varies on a face."""
    mesh, skeleton, locals_ = setup
    permeability = np.array([[3.0, 0.5, 0.2], [0.5, 2.0, 0.1], [0.2, 0.1, 1.0]])

    def quadratic(points):
        """Return a quadratic potential reproduced by the declared local space."""
        return np.sum(points**2, axis=1)

    def flux(points):
        """Evaluate its anisotropic physical flux."""
        return -2 * points @ permeability

    result = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        local_meshes=locals_,
        degree=4,
        local_refinement=1,
        permeability=permeability,
        source=-2 * np.trace(permeability),
        dirichlet=quadratic,
    )
    assert result.l2_error(quadratic) < 2e-11
    assert result.flux_l2_error(flux) < 2e-10
    assert max(abs(result.conservation_residuals())) < 2e-11
    wrong = sum(
        mesh.signs[0, side] * mesh.areas[f] * result.hybrid.trace[skeleton.dofs(int(f))].mean()
        for side, f in enumerate(mesh.cell_faces[0])
    )
    assert abs(wrong + 2 * np.trace(permeability) * mesh.volumes[0]) > 1e-3


def test_vector_equilibrium_uses_nonuniform_area_fractions(setup) -> None:
    """Exact affine symmetric stress obeys force and torque on the partitioned boundary."""
    from pymhm._legacy.models.elasticity.primal_3d import Elasticity3DSolution
    from pymhm.core.contracts import HybridSolution

    mesh, skeleton, locals_ = setup
    tensor = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 3.0]])
    gradient = np.array([1.0, 2.0, -0.5])
    trace = np.zeros((skeleton.size, 3))
    for face in range(len(mesh.faces)):
        points = skeleton.face_partition(face) @ mesh.points[mesh.faces[face]]
        trace[skeleton.dofs(face)] = (
            -((points @ gradient)[..., None] * (tensor @ mesh.normals[face]))
        ).reshape(-1, 3)
    hybrid = HybridSolution(trace.ravel(), (), (), 0.0, np.array([]))
    solution = Elasticity3DSolution(
        skeleton, locals_, (), hybrid, 1, None, 1.0, 1.0, -tensor @ gradient, 5
    )
    assert_allclose(solution.equilibrium_residuals(), 0.0, atol=2e-14)


def test_nonuniform_independent_face_degrees(setup) -> None:
    """Pk can differ between original faces without changing subface area conventions."""
    mesh, _, _ = setup
    degrees = np.arange(len(mesh.faces)) % 3
    skeleton = TriangularSkeleton(
        mesh, degree=degrees, face_partitions=tuple(fan_partition() for _ in mesh.faces)
    )
    assert skeleton.size == sum(3 * (degree + 1) * (degree + 2) // 2 for degree in degrees)
    for face in range(len(mesh.faces)):
        basis = skeleton.basis(face, np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]))
        assert_allclose(basis.sum(axis=1), 1.0, atol=1e-15)
    with pytest.raises(ValueError, match="degree must be"):
        TriangularSkeleton(mesh, degree=[0])


def test_reversed_subtriangle_vertices_preserve_physical_solution(setup) -> None:
    """Subface basis coordinates follow their declared vertices, independently of orientation."""
    mesh, original, locals_ = setup
    partitions = []
    for face in range(len(mesh.faces)):
        part = original.face_partition(face)[::-1].copy()
        part[1] = part[1, ::-1]
        partitions.append(part)
    skeleton = TriangularSkeleton(mesh, degree=1, face_partitions=tuple(partitions))
    result = solve_darcy_3d(
        mesh, skeleton=skeleton, local_meshes=locals_, degree=4, dirichlet=pressure
    )
    assert result.l2_error(pressure) < 2e-11
    assert result.flux_l2_error([-1.0, -2.0, 0.5]) < 2e-10
