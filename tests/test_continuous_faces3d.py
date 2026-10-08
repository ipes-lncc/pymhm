"""C0 Bernstein traces on each macroface and their physical primal Darcy equations."""

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton, _boundary
from pymhm.linalg.linear import LinearSolveError
from pymhm.meshes.tetrahedron import TetraMesh


@pytest.fixture(autouse=True)
def single_thread() -> Iterator[None]:
    """Keep the small original-equation checks independent of BLAS scheduling."""
    with threadpool_limits(1):
        yield


def _tetrahedron() -> TetraMesh:
    """Return the reference tetrahedron with its canonical outward macro normals."""
    return TetraMesh(np.vstack((np.zeros(3), np.eye(3))), np.array([[0, 1, 2, 3]]))


def _fan_partition() -> np.ndarray:
    """Split a macroface around an off-center point, reversing one triangle."""
    vertices = np.eye(3)
    center = np.array([0.2, 0.3, 0.5])
    return np.array(
        [
            [vertices[0], vertices[1], center],
            [center, vertices[2], vertices[1]],
            [vertices[2], vertices[0], center],
        ]
    )


@pytest.mark.parametrize("degree", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_shared_edge_values_with_reversed_subtriangle_vertices(
    degree: int, nonuniform: bool
) -> None:
    """Independent one-sided polynomial evaluations agree across every internal edge."""
    mesh = _tetrahedron()
    original = TriangularSkeleton(mesh, subdivisions=2, degree=degree)
    partition = _fan_partition() if nonuniform else original.face_partition(0).copy()
    if not nonuniform:
        partition[1] = partition[1, ::-1]
    skeleton = TriangularSkeleton(
        mesh,
        degree=degree,
        continuous=True,
        face_partitions=tuple(partition.copy() for _ in mesh.faces),
    )
    coefficients = np.sin(0.37 + np.arange(skeleton.size))
    parameters = np.array([0.0, 0.13, 0.5, 0.89, 1.0])
    count = 0
    for left, right in combinations(range(len(partition)), 2):
        shared = [
            point for point in partition[left] if np.any(np.all(partition[right] == point, 1))
        ]
        if len(shared) != 2:
            continue
        points = (1 - parameters[:, None]) * shared[0] + parameters[:, None] * shared[1]
        left_values = (
            skeleton.basis(0, points @ np.linalg.inv(partition[left]))
            @ coefficients[skeleton.subtriangle_dofs(0, left)]
        )
        right_values = (
            skeleton.basis(0, points @ np.linalg.inv(partition[right]))
            @ coefficients[skeleton.subtriangle_dofs(0, right)]
        )
        assert_allclose(left_values, right_values, atol=2e-13, rtol=2e-13)
        assert_allclose(
            skeleton.evaluate(0, points) @ coefficients[skeleton.dofs(0)],
            left_values,
            atol=2e-13,
            rtol=2e-13,
        )
        count += 1
    assert count == 3
    assert_allclose(skeleton.evaluate(0, np.vstack((np.eye(3), [0.2, 0.3, 0.5]))).sum(1), 1)


@pytest.mark.parametrize("degree", [1, 2, 4, 7])
def test_c0_dimension_and_no_identification_between_macrofaces(degree: int) -> None:
    """Share vertex and edge modes within a macroface while retaining distinct face traces."""
    mesh = _tetrahedron()
    skeleton = TriangularSkeleton(mesh, 2, degree=degree, continuous=True)
    expected = (2 * degree + 1) * (2 * degree + 2) // 2
    assert skeleton.size == len(mesh.faces) * expected
    for face in range(len(mesh.faces)):
        ids = skeleton.dofs(face)
        assert len(ids) == expected
        assert_allclose(
            np.unique(np.concatenate([skeleton.subtriangle_dofs(face, i) for i in range(4)])),
            ids,
        )
    for left, right in combinations(range(len(mesh.faces)), 2):
        assert len(np.intersect1d(skeleton.dofs(left), skeleton.dofs(right))) == 0


@pytest.mark.parametrize("degree", [5, 7])
def test_high_degree_macro_polynomial_reproduction_on_nonuniform_partition(degree: int) -> None:
    """The multinomial expansion gives independent Bernstein coefficients of a Pk field."""
    mesh = _tetrahedron()
    partition = _fan_partition()
    skeleton = TriangularSkeleton(
        mesh,
        degree=degree,
        continuous=True,
        face_partitions=tuple(partition.copy() for _ in mesh.faces),
    )
    powers = np.array(
        [(i, j, degree - i - j) for i in range(degree, -1, -1) for j in range(degree - i, -1, -1)]
    )
    coefficients = np.full(skeleton.size, np.nan)
    for segment, triangle in enumerate(partition):
        # (a dot lambda)^k has Bernstein coefficients prod(a**alpha).
        local = np.prod(triangle[:, 0] ** powers, axis=1)
        local += 0.7 * np.prod(triangle[:, 1] ** powers, axis=1)
        local -= 0.2 * np.prod(triangle[:, 2] ** powers, axis=1)
        ids = skeleton.subtriangle_dofs(0, segment)
        assigned = np.isfinite(coefficients[ids])
        assert_allclose(coefficients[ids][assigned], local[assigned], atol=2e-15)
        coefficients[ids] = local
    points, _ = triangle_quadrature(8)
    expected = points[:, 0] ** degree + 0.7 * points[:, 1] ** degree - 0.2 * points[:, 2] ** degree
    assert_allclose(
        skeleton.evaluate(0, points) @ coefficients[skeleton.dofs(0)],
        expected,
        atol=2e-14,
        rtol=2e-13,
    )


@pytest.mark.parametrize("continuous", [False, True])
def test_degree_five_boundary_projection_raises_a_low_requested_quadrature(
    continuous: bool,
) -> None:
    """Boundary moments resolve P5 data even when requested order is below the Gram degree."""
    mesh = _tetrahedron()
    skeleton = TriangularSkeleton(mesh, 2, degree=5, continuous=continuous)

    def datum(points: np.ndarray) -> np.ndarray:
        """Give a nonconstant physical P5 polynomial on every tetrahedral macroface."""
        x, y, z = points.T
        return (0.3 + x - 2 * y + 0.7 * z) ** 5 - 0.2 * (0.5 - x + y + z) ** 3

    load, fixed = _boundary(skeleton, 0.0, {int(face): datum for face in mesh.boundary_faces}, 2)
    assert_allclose(load, 0.0, atol=0.0)
    assert len(fixed) == skeleton.size
    bary, _ = triangle_quadrature(8)
    bary = np.vstack((bary, np.eye(3), [[0.5, 0.5, 0], [0, 0.5, 0.5]]))
    for face in mesh.boundary_faces:
        face = int(face)
        coefficients = np.array([fixed[int(index)] for index in skeleton.dofs(face)])
        assert_allclose(
            skeleton.evaluate(face, bary) @ coefficients,
            datum(bary @ mesh.points[mesh.faces[face]]),
            atol=2e-11,
            rtol=2e-12,
        )
    requested_low, low_fixed = _boundary(skeleton, datum, {}, 2)
    requested_high, high_fixed = _boundary(skeleton, datum, {}, 10)
    assert low_fixed == high_fixed == {}
    assert_allclose(requested_low, requested_high, atol=2e-14, rtol=2e-12)


def test_mixed_c0_and_discontinuous_degrees_preserve_the_declared_face_spaces() -> None:
    """Different macrofaces can independently select continuity and polynomial degree."""
    mesh = _tetrahedron()
    degrees = [1, 2, 3, 0]
    continuous = [True, False, True, False]
    skeleton = TriangularSkeleton(mesh, 2, degree=degrees, continuous=continuous)
    assert [len(skeleton.dofs(face)) for face in range(4)] == [6, 24, 28, 4]
    points = np.array([[0.15, 0.2, 0.65], [0.7, 0.2, 0.1], [0.2, 0.65, 0.15]])
    assert_allclose(skeleton.evaluate(0, points).sum(1), 1)
    assert_allclose(skeleton.evaluate(1, points).sum(1), 1)
    for segment in range(4):
        assert len(skeleton.subtriangle_dofs(1, segment)) == 6
        for other in range(segment):
            assert (
                len(
                    np.intersect1d(
                        skeleton.subtriangle_dofs(1, segment), skeleton.subtriangle_dofs(1, other)
                    )
                )
                == 0
            )
    default = TriangularSkeleton(mesh, 2, degree=2)
    explicit = TriangularSkeleton(mesh, 2, degree=2, continuous=False)
    assert_allclose(default.offsets, explicit.offsets)
    assert_allclose(default.evaluate(0, points), explicit.evaluate(0, points))


@pytest.mark.parametrize(
    "options",
    [
        {"degree": 0, "continuous": True},
        {"degree": [1, 0, 1, 1], "continuous": [True, True, False, False]},
        {"degree": 1, "continuous": [True]},
        {"degree": 1, "continuous": [1, 0, 1, 0]},
        {"degree": 1, "continuous": "yes"},
        {"degree": 1, "continuous": 1},
        {"degree": 1, "continuous": [[True], [False], [True], [False]]},
    ],
)
def test_invalid_continuity_declarations_are_rejected(options: dict) -> None:
    """A C0 face requires a positive degree and an explicit boolean declaration."""
    with pytest.raises(ValueError):
        TriangularSkeleton(_tetrahedron(), **options)


def test_face_evaluation_and_local_map_validate_geometry_and_indices() -> None:
    """Reject undefined points and subtriangle indices instead of inventing a trace."""
    skeleton = TriangularSkeleton(_tetrahedron(), 2, degree=2, continuous=True)
    for points in (
        np.array([[np.nan, 0, 1]]),
        np.array([[np.inf, 0, 1]]),
        np.array([[-0.1, 0.2, 0.9]]),
        np.zeros((1, 3)),
        np.array([[1j, 0, 1]]),
        np.ones(3) / 3,
        np.ones((1, 2)) / 2,
    ):
        with pytest.raises(ValueError):
            skeleton.evaluate(0, points)
    for face, segment in ((-1, 0), (4, 0), (0, -1), (0, 4)):
        with pytest.raises(ValueError):
            skeleton.subtriangle_dofs(face, segment)


@pytest.mark.parametrize("subdivisions,excluded_refinements", [(1, (1, 2)), (2, (2,))])
def test_linear_local_space_admits_resolved_c0_traces_and_rejects_the_previous_mesh(
    subdivisions: int, excluded_refinements: tuple[int, ...]
) -> None:
    """Exercise the admissible and immediately excluded algebraic MHM trace choices."""
    mesh = TetraMesh.unit_cube()
    skeleton = TriangularSkeleton(mesh, subdivisions, degree=1, continuous=True)
    for refinement in excluded_refinements:
        with pytest.raises(LinearSolveError, match="singular|rank"):
            solve_darcy_3d(
                mesh, skeleton=skeleton, degree=1, local_refinement=refinement, source=1.0
            )
    solution = solve_darcy_3d(mesh, skeleton=skeleton, degree=1, local_refinement=4, source=1.0)
    assert solution.hybrid.residual < 1e-10
    assert_allclose(solution.conservation_residuals(), 0, atol=2e-11)


@pytest.mark.parametrize("boundary", ["dirichlet", "mixed", "neumann"])
def test_anisotropic_quadratic_pressure_physical_flux_and_full_mean(boundary: str) -> None:
    """C0 normal traces reproduce complete fields and physical data on all boundary types."""
    mesh = TetraMesh.unit_cube()
    tensor = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]])

    def pressure(points: np.ndarray) -> np.ndarray:
        """Return a quadratic with unit-cube volume average two."""
        return 1 + np.sum(points**2, axis=1)

    def flux(points: np.ndarray) -> np.ndarray:
        """Differentiate the pressure before applying the full permeability tensor."""
        return -2 * points @ tensor

    faces = [
        int(face)
        for face in mesh.boundary_faces
        if boundary == "neumann"
        or (boundary == "mixed" and np.all(mesh.points[mesh.faces[face], 0] == 1))
    ]
    natural = {face: lambda x, normal=mesh.normals[face]: flux(x) @ normal for face in faces}
    skeleton = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        degree=2,
        local_refinement=2,
        permeability=tensor,
        source=-2 * np.trace(tensor),
        dirichlet=pressure,
        neumann=natural,
        mean_pressure=2 if boundary == "neumann" else 0,
        quadrature_order=6,
    )
    assert solution.l2_error(pressure, 6) < 2e-11
    assert solution.flux_l2_error(flux, 6) < 2e-11
    assert_allclose(solution.conservation_residuals(), 0, atol=2e-11)
    points, _ = triangle_quadrature(4)
    for face in range(len(mesh.faces)):
        physical = points @ mesh.points[mesh.faces[face]]
        represented = skeleton.evaluate(face, points) @ solution.hybrid.trace[skeleton.dofs(face)]
        assert_allclose(represented, flux(physical) @ mesh.normals[face], atol=2e-11)
    if boundary == "neumann":
        bary, weights = tetrahedron_quadrature(6)
        integral = sum(
            fine.volumes @ (solution.evaluate(cell, bary)[0] @ weights)
            for cell, fine in enumerate(solution.local_meshes)
        )
        assert_allclose(integral / mesh.volumes.sum(), 2, atol=2e-12)


@pytest.mark.parametrize("mixed", [False, True])
def test_nonuniform_c0_anisotropic_patch_on_an_aligned_oblique_star(mixed: bool) -> None:
    """Unequal reversed subfaces retain physical fields and boundary data on an explicit volume."""
    vertices = np.array([[0.0, 0, 0], [1.2, 0.1, 0], [0.2, 1.0, 0.1], [0.1, 0.2, 1.4]])
    mesh = TetraMesh(vertices, np.array([[0, 1, 2, 3]]))
    partitions = tuple(_fan_partition() for _ in mesh.faces)
    skeleton = TriangularSkeleton(mesh, degree=1, continuous=True, face_partitions=partitions)
    center = np.array([0.2, 0.3, 0.1, 0.4]) @ vertices
    outer_triangles = np.concatenate(
        [partition @ mesh.points[mesh.faces[face]] for face, partition in enumerate(partitions)]
    )
    tetrahedra = np.concatenate(
        (np.broadcast_to(center, (len(outer_triangles), 1, 3)), outer_triangles), axis=1
    )
    points, inverse = np.unique(tetrahedra.reshape(-1, 3), axis=0, return_inverse=True)
    fine = TetraMesh(points, inverse.reshape(-1, 4))
    tensor = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]])

    def pressure(points: np.ndarray) -> np.ndarray:
        """Prescribe the complete quadratic physical pressure on the oblique tetrahedron."""
        return 1 + np.sum(points**2, axis=1)

    def flux(points: np.ndarray) -> np.ndarray:
        """Apply anisotropic permeability to the independent analytical gradient."""
        return -2 * points @ tensor

    natural_face = int(mesh.boundary_faces[0])
    natural = {natural_face: lambda x: flux(x) @ mesh.normals[natural_face]} if mixed else {}
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        local_meshes=(fine,),
        degree=3,
        permeability=tensor,
        source=-2 * np.trace(tensor),
        dirichlet=pressure,
        neumann=natural,
        quadrature_order=6,
    )
    assert solution.l2_error(pressure, 6) < 2e-11
    assert solution.flux_l2_error(flux, 6) < 2e-11
    assert_allclose(solution.conservation_residuals(), 0.0, atol=2e-11)
    bary, _ = triangle_quadrature(5)
    for face in range(len(mesh.faces)):
        points = bary @ mesh.points[mesh.faces[face]]
        assert_allclose(
            skeleton.evaluate(face, bary) @ solution.hybrid.trace[skeleton.dofs(face)],
            flux(points) @ mesh.normals[face],
            atol=2e-11,
            rtol=2e-12,
        )


def test_nonzero_homogeneous_dirichlet_tetrahedral_bubble() -> None:
    """A quartic pressure vanishes on all four faces with independently derived source."""
    mesh = _tetrahedron()

    def pressure(points: np.ndarray) -> np.ndarray:
        """Evaluate lambda0*lambda1*lambda2*lambda3 on the reference tetrahedron."""
        return (1 - points.sum(axis=1)) * np.prod(points, axis=1)

    def source(points: np.ndarray) -> np.ndarray:
        """Use the three independent second derivatives of the quartic pressure."""
        x, y, z = points.T
        return 2 * (y * z + x * z + x * y)

    def flux(points: np.ndarray) -> np.ndarray:
        """Return the negative analytical gradient, independently of FE differentiation."""
        x, y, z = points.T
        w = 1 - x - y - z
        return -np.column_stack((y * z * (w - x), x * z * (w - y), x * y * (w - z)))

    skeleton = TriangularSkeleton(mesh, 2, degree=3, continuous=True)
    with pytest.raises(LinearSolveError, match="singular|rank"):
        solve_darcy_3d(
            mesh,
            skeleton=skeleton,
            source=source,
            degree=4,
            local_refinement=2,
            quadrature_order=8,
        )
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        source=source,
        degree=4,
        local_refinement=4,
        quadrature_order=8,
    )
    assert solution.l2_error(pressure, 8) < 2e-12
    assert solution.flux_l2_error(flux, 8) < 2e-11
    assert_allclose(solution.conservation_residuals(), 0, atol=2e-12)


def test_degree_five_trace_reproduces_a_sixth_degree_homogeneous_pressure() -> None:
    """Resolve physical P5 normal data with a sufficiently rich local P8 space."""
    mesh = _tetrahedron()

    def pressure(points: np.ndarray) -> np.ndarray:
        """Return lambda0*x^3*y*z, which vanishes on the entire tetrahedral boundary."""
        x, y, z = points.T
        return (1 - x - y - z) * x**3 * y * z

    def source(points: np.ndarray) -> np.ndarray:
        """Calculate minus the sum of three analytical second derivatives."""
        x, y, z = points.T
        w = 1 - x - y - z
        return -6 * x * y * z * (w - x) + 2 * x**3 * (y + z)

    def flux(points: np.ndarray) -> np.ndarray:
        """Evaluate the physical gradient of the sixth-degree pressure independently."""
        x, y, z = points.T
        w = 1 - x - y - z
        return -np.column_stack(
            (x**2 * y * z * (3 * w - x), x**3 * z * (w - y), x**3 * y * (w - z))
        )

    skeleton = TriangularSkeleton(mesh, degree=5, continuous=True)
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        source=source,
        degree=8,
        local_refinement=1,
        quadrature_order=10,
    )
    assert solution.l2_error(pressure, 10) < 2e-12
    assert solution.flux_l2_error(flux, 10) < 2e-11
    assert_allclose(solution.conservation_residuals(), 0, atol=2e-12)
    bary, _ = triangle_quadrature(8)
    for face in range(len(mesh.faces)):
        points = bary @ mesh.points[mesh.faces[face]]
        assert_allclose(
            skeleton.evaluate(face, bary) @ solution.hybrid.trace[skeleton.dofs(face)],
            flux(points) @ mesh.normals[face],
            atol=2e-11,
        )
