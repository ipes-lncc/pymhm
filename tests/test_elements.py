"""Independent polynomial, conservation, and local element identities."""

from math import factorial

import numpy as np
import pytest

from pymhm.fem.scalar.operators import (
    boundary_data,
    face_integration,
    p1_geometry,
    p1_operators,
    rt0_evaluate,
    rt0_operators,
    triangle_quadrature,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.materials.evaluation import scalar_values, tensor_values, vector_values
from pymhm.meshes.triangle import TriangleMesh


@pytest.mark.parametrize("scale", [1.0, 1e-25])
@pytest.mark.parametrize("partition", [(0.0, 1.0), (0.0, 0.17, 0.63, 1.0)])
def test_boundary_volume_preserves_small_physical_dilatation(scale, partition):
    """The same trace moments retain true volume changes without a zero cutoff."""
    from pymhm._legacy.models.elasticity.mixed_pressure import _boundary_volume_flux

    mesh = TriangleMesh.unit_square()
    face = FaceSpace(partition, (2,) * (len(partition) - 1))
    skeleton = SkeletonSpace(mesh, tuple(face for _ in mesh.faces), components=2)
    load, fixed = boundary_data(skeleton, lambda x: scale * x, order=5)
    assert fixed == {}
    np.testing.assert_allclose(_boundary_volume_flux(skeleton, load), 2 * scale, rtol=3e-15, atol=0)


def test_boundary_solenoidal_quadratic_has_no_artificial_volume_change():
    """Opposite constant moments cancel before finite bulk compliance amplifies them."""
    from pymhm._legacy.models.elasticity.mixed_pressure import _boundary_volume_flux

    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces), components=2)
    load, _ = boundary_data(
        skeleton,
        lambda x: np.column_stack((x[:, 0] ** 2, -2 * x[:, 0] * x[:, 1])),
        order=5,
    )
    assert _boundary_volume_flux(skeleton, load) == 0.0


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_triangle_quadrature_exact_barycentric_polynomial_moments(order):
    barycentric, weights = triangle_quadrature(order)
    assert barycentric.shape == (order**2, 3)
    assert np.all(weights > 0)
    assert np.all(barycentric > 0)
    np.testing.assert_allclose(barycentric.sum(axis=1), 1)
    np.testing.assert_allclose(weights.sum(), 1)
    for a in range(2 * order - 1):
        for b in range(2 * order - 1 - a):
            for c in range(2 * order - 1 - a - b):
                exact = 2 * factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 2)
                numerical = weights @ np.prod(barycentric ** [a, b, c], axis=1)
                np.testing.assert_allclose(numerical, exact, rtol=2e-13, atol=1e-16)


def test_triangle_quadrature_rejects_nonpositive_order():
    with pytest.raises(ValueError):
        triangle_quadrature(0)


def test_coefficient_evaluation_broadcasting_and_callable_fields():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_array_equal(scalar_values(3.0, points), [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(scalar_values(lambda x: x[:, 0], points), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(vector_values([1.0, 2.0], points), [[1.0, 2.0]] * 3)
    np.testing.assert_array_equal(vector_values(lambda x: 2 * x, points), 2 * points)
    np.testing.assert_array_equal(tensor_values(2.0, points), np.tile(2 * np.eye(2), (3, 1, 1)))
    values = tensor_values(lambda x: x[:, 0] + 1, points)
    np.testing.assert_array_equal(values, np.array([np.eye(2), 2 * np.eye(2), np.eye(2)]))
    tensor = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(tensor_values(tensor, points), np.tile(tensor, (3, 1, 1)))
    np.testing.assert_array_equal(tensor_values(lambda x: values, points), values)
    copied = scalar_values(np.array([1.0, 2.0, 3.0]), points)
    assert copied.flags.writeable


@pytest.mark.parametrize(
    "evaluator,field",
    [
        (scalar_values, np.nan),
        (scalar_values, [1.0, 2.0]),
        (vector_values, [1.0, np.inf]),
        (vector_values, [1.0, 2.0, 3.0]),
        (tensor_values, np.nan),
        (tensor_values, 0.0),
        (tensor_values, -1.0),
        (tensor_values, [[1.0, 1.0], [0.0, 1.0]]),
        (tensor_values, [[1.0, 2.0], [2.0, 1.0]]),
        (tensor_values, [[1.0, 0.0], [0.0, np.inf]]),
    ],
)
def test_invalid_coefficients_fail_explicitly(evaluator, field):
    with pytest.raises(ValueError):
        evaluator(field, np.zeros((3, 2)))


@pytest.mark.parametrize("as_callable", [False, True])
@pytest.mark.parametrize(
    "evaluator,field",
    [
        (scalar_values, np.full(3, 1 + 2j)),
        (vector_values, np.array([1 + 2j, 3 - 4j])),
        (tensor_values, np.eye(2, dtype=complex) * (1 + 2j)),
    ],
)
def test_complex_coefficients_are_rejected_without_discarding_imaginary_part(
    evaluator, field, as_callable
):
    coefficient = (lambda points: field) if as_callable else field
    with pytest.raises(ValueError, match="must be real"):
        evaluator(coefficient, np.zeros((3, 2)))


def test_p1_reference_triangle_stiffness_mass_and_constant_load():
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    gradients, areas = p1_geometry(mesh)
    exact_gradients = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(gradients[0], exact_gradients)
    np.testing.assert_allclose(areas, [0.5])
    tensor = np.array([[2.0, 0.3], [0.3, 3.0]])
    matrix, mass, source = p1_operators(mesh, diffusion=tensor, source=6.0)
    np.testing.assert_allclose(matrix.toarray(), 0.5 * exact_gradients @ tensor @ exact_gradients.T)
    np.testing.assert_allclose(mass.toarray(), (np.ones((3, 3)) + np.eye(3)) / 24)
    np.testing.assert_allclose(source, np.ones(3))
    roundoff = 5 * np.finfo(float).eps * np.linalg.norm(matrix.toarray(), ord=np.inf)
    np.testing.assert_allclose(matrix @ np.ones(3), 0, atol=roundoff)
    np.testing.assert_allclose(np.ones(3) @ mass @ np.ones(3), 0.5)
    assert np.linalg.eigvalsh(mass.toarray()).min() > 0


def test_p1_reaction_advection_and_affine_patch_energy():
    mesh = TriangleMesh.unit_square(3, 2)
    diffusion, mass, load = p1_operators(mesh, source=lambda x: 1 + x[:, 0])
    combined, _, _ = p1_operators(mesh, reaction=2.0, advection=(2.0, -1.0))
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    ones = np.ones(len(mesh.points))
    np.testing.assert_allclose(x @ diffusion @ x, 1.0, atol=1e-14)
    np.testing.assert_allclose(y @ diffusion @ y, 1.0, atol=1e-14)
    np.testing.assert_allclose(x @ diffusion @ y, 0.0, atol=1e-14)
    np.testing.assert_allclose(load.sum(), 1.5)
    transport = combined - diffusion - 2 * mass
    np.testing.assert_allclose(transport @ x, 2 * mass @ ones, atol=1e-15)
    np.testing.assert_allclose(transport @ y, -mass @ ones, atol=1e-15)
    # Extracting transport subtracts assembled diffusion and reaction terms.
    # Its constant-kernel error therefore scales with those operands, even
    # though the exact convection operator annihilates constants.
    scale = abs(combined) + abs(diffusion) + 2 * abs(mass)
    roundoff = 8 * np.finfo(float).eps * np.asarray(scale.sum(axis=1)).ravel()
    np.testing.assert_array_less(abs(transport @ ones), roundoff)
    with pytest.raises(ValueError, match="reaction"):
        p1_operators(mesh, reaction=-1.0)


@pytest.mark.parametrize("cell", [0, 1])
def test_trace_integration_maps_constant_vector_flux_with_correct_orientation(cell):
    mesh = TriangleMesh.unit_square()
    fine = mesh.submesh(cell, 3)
    skeleton = SkeletonSpace(mesh)
    p1, rt = face_integration(mesh, cell, fine, skeleton)
    vector = np.array([2.0, -3.0])
    coefficients = mesh.normals[mesh.cell_faces[cell]] @ vector
    boundary = fine.boundary_faces
    exact_rt = (fine.normals[boundary] @ vector) * fine.lengths[boundary]
    exact_p1 = np.zeros(len(fine.points))
    np.add.at(exact_p1, fine.faces[boundary].ravel(), np.repeat(exact_rt / 2, 2))
    np.testing.assert_allclose(rt @ coefficients, exact_rt, atol=2e-15)
    np.testing.assert_allclose(p1 @ coefficients, exact_p1, atol=2e-15)
    np.testing.assert_allclose((p1 @ coefficients).sum(), 0.0, atol=2e-15)
    expected = mesh.signs[cell] * mesh.lengths[mesh.cell_faces[cell]]
    np.testing.assert_allclose(rt.sum(axis=0), expected)
    np.testing.assert_allclose(p1.sum(axis=0), expected)


def test_trace_integration_splits_nonmatching_hp_partitions():
    mesh = TriangleMesh.unit_square()
    spaces = tuple(FaceSpace((0.0, 0.2, 0.75, 1.0), (2, 1, 3)) for _ in mesh.faces)
    skeleton = SkeletonSpace(mesh, spaces)
    for cell in range(2):
        fine = mesh.submesh(cell, 4)
        p1, rt = face_integration(mesh, cell, fine, skeleton)
        expected = []
        for side, face in enumerate(mesh.cell_faces[cell]):
            space = spaces[face]
            for length, degree in zip(np.diff(space.breaks), space.degrees, strict=True):
                expected.extend(
                    [mesh.signs[cell, side] * mesh.lengths[face] * length] + [0.0] * degree
                )
        np.testing.assert_allclose(p1.sum(axis=0), expected, atol=3e-15)
        np.testing.assert_allclose(rt.sum(axis=0), expected, atol=3e-15)


@pytest.mark.parametrize("components,value", [(1, 2.0), (2, (2.0, -3.0))])
def test_boundary_moments_and_neumann_l2_projection(components, value):
    mesh = TriangleMesh.unit_square()
    spaces = tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces)
    skeleton = SkeletonSpace(mesh, spaces, components)
    neumann_face = int(mesh.boundary_faces[0])
    load, fixed = boundary_data(skeleton, value, {neumann_face: value})
    vector = np.atleast_1d(value)
    for face in mesh.boundary_faces:
        moments = np.array([0.5, 0.0, 0.5, 0.0])[:, None] * vector
        dofs = skeleton.dofs(int(face))
        if face == neumann_face:
            expected = np.array([1.0, 0.0, 1.0, 0.0])[:, None] * vector
            np.testing.assert_allclose(
                [fixed[int(dof)] for dof in dofs], expected.ravel(), atol=1e-14
            )
            np.testing.assert_array_equal(load[dofs], 0.0)
        else:
            np.testing.assert_allclose(load[dofs], moments.ravel() * mesh.lengths[face], atol=1e-15)
    interior = np.flatnonzero(mesh.face_cells[:, 1] >= 0)
    np.testing.assert_array_equal(load[skeleton.dofs(int(interior[0]))], 0.0)
    with pytest.raises(ValueError, match="boundary faces"):
        boundary_data(skeleton, value, {int(interior[0]): value})
    _, free = boundary_data(skeleton, value)
    assert free == {}


def test_neumann_projection_exactly_reproduces_quadratic_face_data():
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    face = int(mesh.boundary_faces[0])
    start, end = mesh.points[mesh.faces[face]]
    tangent = end - start

    def quadratic(points):
        parameter = (points - start) @ tangent / (tangent @ tangent)
        return 1 + parameter + parameter**2

    _, fixed = boundary_data(skeleton, 0.0, {face: quadratic})
    coefficients = np.array([fixed[int(dof)] for dof in skeleton.dofs(face)])
    parameter = np.linspace(0, 1, 31)
    reconstructed = skeleton.faces[face].evaluate(parameter) @ coefficients
    np.testing.assert_allclose(reconstructed, 1 + parameter + parameter**2, atol=2e-15)


def test_rt0_reference_triangle_mass_divergence_and_source():
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    mass, divergence, load = rt0_operators(mesh, source=6.0)
    expected = np.array([[1 / 3, 0.0, -1 / 6], [0.0, 1 / 6, 0.0], [-1 / 6, 0.0, 1 / 3]])
    np.testing.assert_allclose(mass.toarray(), expected, atol=1e-16)
    np.testing.assert_array_equal(divergence.toarray(), [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(load, [3.0])
    assert np.linalg.eigvalsh(mass.toarray()).min() > 0


@pytest.mark.parametrize(
    "constant,slope", [([0.0, 0.0], 1.0), ([2.0, -3.0], 0.0), ([1.0, 2.0], -2.0)]
)
def test_rt0_reproduction_and_cellwise_divergence_theorem(constant, slope):
    mesh = TriangleMesh.unit_square(3, 2)
    constant = np.array(constant)
    midpoints = mesh.points[mesh.faces].mean(axis=1)
    face_vectors = constant + slope * midpoints
    flux = np.einsum("ij,ij->i", face_vectors, mesh.normals) * mesh.lengths
    _, divergence, load = rt0_operators(mesh, source=2 * slope)
    np.testing.assert_allclose(divergence @ flux, load, atol=1e-15)
    boundary = mesh.boundary_faces
    np.testing.assert_allclose((divergence @ flux).sum(), flux[boundary].sum(), atol=1e-15)
    barycentric, _ = triangle_quadrature(3)
    points = np.einsum("qi,tij->tqj", barycentric, mesh.points[mesh.cells])
    np.testing.assert_allclose(
        rt0_evaluate(mesh, flux, barycentric), constant + slope * points, atol=3e-15
    )


def test_rt0_anisotropic_mass_matches_exact_continuous_energy():
    mesh = TriangleMesh.unit_square(4)
    tensor = np.array([[2.0, 0.4], [0.4, 3.0]])
    inverse = np.linalg.inv(tensor)
    midpoints = mesh.points[mesh.faces].mean(axis=1)
    flux = np.einsum("ij,ij->i", midpoints, mesh.normals) * mesh.lengths
    mass, _, _ = rt0_operators(mesh, tensor)
    expected_energy = np.trace(inverse) / 3 + inverse[0, 1] / 2
    np.testing.assert_allclose(flux @ mass @ flux, expected_energy, rtol=1e-14)
