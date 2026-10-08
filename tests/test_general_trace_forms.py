"""PDE-independent trace integration retains shared modes and physical frames."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import (
    ComponentTraceSpace,
    FaceSpace,
    PressureTraceSpace,
    PressureTraceSpace3D,
    SkeletonSpace,
    TangentialTraceSpace,
    TetraMesh,
    TriangleMesh,
    TriangularSkeleton,
    bind_interface,
    project_trace,
    trace_bilinear_form,
    trace_boundary_data,
    trace_linear_form,
    trace_quadrature,
)


@pytest.mark.parametrize(
    "kind", ["edge", "pressure2", "triangle", "pressure3", "vector", "tangent"]
)
def test_trace_projection_preserves_declared_polynomial_fields_and_shared_nodes(kind):
    macro2, macro3 = TriangleMesh.unit_square(), TetraMesh.unit_cube()
    spaces = {
        "edge": SkeletonSpace(
            macro2, tuple(FaceSpace.uniform(3, 2, continuous=True) for _ in macro2.faces)
        ),
        "pressure2": PressureTraceSpace.uniform(macro2, 3, 2),
        "triangle": TriangularSkeleton(macro3, 2, degree=3, continuous=True),
        "pressure3": PressureTraceSpace3D(macro3, 2, 2),
        "vector": ComponentTraceSpace(TriangularSkeleton(macro3, 2, degree=2, continuous=True), 3),
        "tangent": TangentialTraceSpace(TriangularSkeleton(macro3, 2, degree=2, continuous=True)),
    }
    space = spaces[kind]

    def value(points):
        scalar = 1 + points.sum(axis=1)
        return scalar[:, None] * np.array([1, -2, 3]) if kind in {"vector", "tangent"} else scalar

    coefficients = project_trace(space, value)
    full = np.array([coefficients[index] for index in range(space.size)])
    mass = trace_bilinear_form(space)
    assert_allclose(mass @ full, trace_linear_form(space, value), atol=1e-12, rtol=1e-10)
    for face in range(len(space.mesh.faces)):
        dofs, points, weights, basis = trace_quadrature(space, face)
        expected = value(points)
        if kind == "tangent":
            frame = space.frames[face]
            expected = expected @ frame @ frame.T
        if expected.ndim == 1:
            expected = expected[:, None]
        assert_allclose(np.einsum("qia,i->qa", basis, full[dofs]), expected, atol=1e-12, rtol=1e-10)
        assert np.all(weights > 0)
    assert project_trace(space, 0, faces=[]) == {}
    binding = bind_interface(space).binding(0)
    assert_array_equal(binding.dofs, space.cell_dofs(0))


def test_component_layout_and_boundary_pairings_accept_general_tensor_coefficients():
    scalar = SkeletonSpace(TriangleMesh.unit_square())
    space = ComponentTraceSpace(scalar, 2)
    assert space.size == 2 * scalar.size
    assert space.mesh is scalar.mesh
    assert_array_equal(space.dofs(0), (2 * scalar.dofs(0)[:, None] + [0, 1]).ravel())
    coefficient = np.array([[2.0, 0.25], [-0.5, 3]])
    expected = np.kron(trace_bilinear_form(scalar).toarray(), coefficient)
    assert_allclose(trace_bilinear_form(space, coefficient).toarray(), expected, atol=1e-12)
    assert_allclose(
        trace_bilinear_form(
            space, lambda x: np.broadcast_to(coefficient, (len(x), 2, 2))
        ).toarray(),
        expected,
        atol=1e-12,
    )
    assert_allclose(
        trace_bilinear_form(scalar, lambda x: np.full(len(x), 2)).toarray(),
        2 * trace_bilinear_form(scalar).toarray(),
        atol=1e-12,
    )
    fixed = project_trace(space, [1.0, -2], faces=[0])
    assert set(fixed) == set(space.dofs(0))
    assert_allclose(list(fixed.values()), [1, -2], atol=1e-12)
    assert_array_equal(
        ComponentTraceSpace(PressureTraceSpace.uniform(scalar.mesh), 2).dofs(0),
        np.array([0, 1, 2, 3]),
    )


def test_general_trace_forms_validate_capabilities_shapes_and_integration_subsets():
    space = SkeletonSpace(TriangleMesh.unit_square())
    for value, expected in [(None, "mesh-associated"), (ComponentTraceSpace(space, 2), "scalar")]:
        with pytest.raises((TypeError, ValueError), match=expected):
            ComponentTraceSpace(value, 2)
    with pytest.raises(ValueError, match="components"):
        ComponentTraceSpace(space, 0)
    with pytest.raises(ValueError, match="outside"):
        trace_quadrature(space, len(space.mesh.faces))
    with pytest.raises(ValueError, match="distinct"):
        trace_linear_form(space, 1, faces=[0, 0])
    with pytest.raises(ValueError, match="value components"):
        trace_linear_form(space, [1, 2])
    with pytest.raises(ValueError, match="value components"):
        trace_bilinear_form(space, np.ones((2, 3)))
    custom = SimpleNamespace(mesh=space.mesh)
    with pytest.raises(TypeError, match="trace_quadrature"):
        trace_quadrature(custom, 0)
    custom.trace_quadrature = lambda face, order: (face, order)
    assert trace_quadrature(custom, 0, order=3) == (0, 3)


@pytest.mark.parametrize("geometry", ["tetrahedron", "polygon"])
def test_vector_boundary_data_matches_literal_weak_moments_and_fixed_trace(geometry):
    from pymhm import GlobalContext, MeshHierarchy
    from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
    from pymhm.meshes.polyhedral import PolyhedralMesh

    if geometry == "tetrahedron":
        mesh = TetraMesh.unit_cube()
        base = TriangularSkeleton(mesh, degree=2, continuous=True)
    else:
        mesh = PolyhedralMesh.cubes()
        base = PolygonalSkeleton3D(mesh)
    space = ComponentTraceSpace(base, 3)
    context = GlobalContext(
        MeshHierarchy(mesh, lambda cell: mesh.submesh(cell)),
        bind_interface(space),
        (0,) * len(mesh.cells),
    )
    face = int(mesh.boundary_faces[0])
    value = [1.0, -2, 3]
    load, fixed = context.boundary_data(value, {face: value})
    weak_faces = [int(item) for item in mesh.boundary_faces if item != face]
    assert_allclose(load, trace_linear_form(space, value, faces=weak_faces), atol=1e-12, rtol=1e-10)
    assert fixed.keys() == set(space.dofs(face))
    ids, points, _, basis = trace_quadrature(space, face)
    actual = np.einsum("qia,i->qa", basis, [fixed[int(item)] for item in ids])
    assert_allclose(actual, np.broadcast_to(value, points.shape), atol=1e-12, rtol=1e-10)
    all_load, empty = trace_boundary_data(space, value)
    assert not empty
    assert_allclose(
        all_load, trace_linear_form(space, value, faces=mesh.boundary_faces), atol=1e-12, rtol=1e-10
    )
    with pytest.raises(ValueError, match="exterior"):
        trace_boundary_data(space, value, {len(mesh.faces): value})


def test_boundary_projection_accumulates_shared_coordinates_across_distinct_data():
    mesh = TriangleMesh.unit_square()
    space = PressureTraceSpace.uniform(mesh, 2)
    faces = [int(face) for face in mesh.boundary_faces[:2]]
    _, fixed = trace_boundary_data(space, 0, dict(zip(faces, [1.0, 2.0], strict=True)))
    indices = np.array(sorted(fixed))
    mass = trace_bilinear_form(space, faces=faces)
    load = trace_linear_form(space, 1, faces=faces[:1]) + trace_linear_form(
        space, 2, faces=faces[1:]
    )
    assert_allclose(
        mass[indices][:, indices] @ list(fixed.values()), load[indices], atol=1e-12, rtol=1e-10
    )
