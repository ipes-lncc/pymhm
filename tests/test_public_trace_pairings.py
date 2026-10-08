"""General trace pairing integrates compatible partitions and literal vector frames."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.core.spaces import ComponentTraceSpace
from pymhm.fem.quadrature.partitions import simplex_partition_refines
from pymhm.fem.traces.forms import trace_quadrature
from pymhm.fem.traces.pairing import evaluate_trace_basis, interface_pairing
from pymhm.fem.traces.polygon_3d import PolygonalSkeleton3D
from pymhm.fem.traces.pressure_3d import PressureTraceSpace3D
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import TangentialTraceSpace
from pymhm.meshes.polyhedral import PolyhedralMesh
from pymhm.meshes.tetrahedron import TetraMesh


def partitions():
    """Use two intersecting medians and their explicit common centroid fan."""
    a, b, c = np.eye(3)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    center = np.ones(3) / 3
    perimeter = [a, ab, b, bc, c, ca]
    fan = np.array([[center, perimeter[i], perimeter[(i + 1) % 6]] for i in range(6)])
    return np.array([[a, b, bc], [a, bc, c]]), np.array([[b, c, ca], [b, ca, a]]), fan


def test_general_simplex_containment_and_immediately_excluded_inputs():
    left, right, fan = partitions()
    assert simplex_partition_refines(fan, left)
    assert simplex_partition_refines(fan, right)
    assert not simplex_partition_refines(left, right)
    assert simplex_partition_refines(np.eye(4)[None], np.eye(4)[None])
    with pytest.raises(ValueError, match="nondegenerate"):
        simplex_partition_refines(left, np.zeros_like(left))
    for fine, coarse, tolerance in (
        (left[0], right, 1e-12),
        (left[:0], right, 1e-12),
        (left, right, -1.0),
    ):
        with pytest.raises(ValueError, match="compatible"):
            simplex_partition_refines(fine, coarse, tolerance=tolerance)


def test_3d_nonnested_rejected_nested_nonuniform_and_explicit_common_rule():
    mesh = TetraMesh.unit_cube()
    left, right, fan = partitions()
    a = TriangularSkeleton(mesh, degree=1, face_partitions=(left,) * len(mesh.faces))
    b = TriangularSkeleton(mesh, degree=1, face_partitions=(right,) * len(mesh.faces))
    fine = TriangularSkeleton(mesh, degree=1, face_partitions=(fan,) * len(mesh.faces))
    with pytest.raises(ValueError, match="nested"):
        interface_pairing(a, b, 0)
    assert_allclose(interface_pairing(a, fine, 0).T, interface_pairing(fine, a, 0), atol=1e-12)
    common = SimpleNamespace(
        mesh=mesh,
        base=b,
        size=b.size,
        dofs=b.dofs,
        cell_dofs=b.cell_dofs,
        common_trace_quadrature=lambda other, face, order: trace_quadrature(
            fine, face, order=order
        )[1:3],
    )
    matrix = interface_pairing(a, common, 0)
    assert_allclose(
        np.ones(matrix.shape[0]) @ matrix @ np.ones(matrix.shape[1]),
        mesh.areas[mesh.cell_faces[0]].sum(),
        atol=1e-12,
    )
    assert_allclose(matrix.T, interface_pairing(common, a, 0), atol=1e-12)


def test_independent_3d_pressure_partitions_and_cartesian_tangent_frames():
    mesh = TetraMesh.unit_cube()
    conormal = TriangularSkeleton(mesh, degree=1, subdivisions=2)
    pressure = PressureTraceSpace3D(mesh, degree=2, subdivisions=1)
    scalar = interface_pairing(conormal, pressure, 0)
    assert_allclose(scalar.T, interface_pairing(pressure, conormal, 0), atol=1e-12)
    vector = interface_pairing(
        ComponentTraceSpace(conormal, 3), ComponentTraceSpace(pressure, 3), 0
    )
    assert_allclose(vector, np.kron(scalar, np.eye(3)), atol=1e-12)
    tangent = TangentialTraceSpace(conormal)
    assert_allclose(
        interface_pairing(tangent, tangent, 0),
        np.kron(interface_pairing(conormal, conormal, 0), np.eye(2)),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="dimensions"):
        interface_pairing(conormal, ComponentTraceSpace(pressure, 3), 0)
    face = int(mesh.cell_faces[0, 0])
    points = mesh.points[mesh.faces[face]]
    with pytest.raises(ValueError, match="lie on"):
        evaluate_trace_basis(conormal, face, points + mesh.normals[face])
    with pytest.raises(ValueError, match="dimension"):
        evaluate_trace_basis(conormal, face, points[:, :2])
    with pytest.raises(ValueError, match="outside"):
        evaluate_trace_basis(conormal, len(mesh.faces), points)
    custom = SimpleNamespace(
        trace_evaluation=lambda face, points: (np.array([7]), np.ones((len(points), 1, 1)))
    )
    assert evaluate_trace_basis(custom, face, points)[0][0] == 7
    with pytest.raises(TypeError, match="trace_evaluation"):
        evaluate_trace_basis(object(), face, points)
    with pytest.raises(TypeError, match="mesh-associated"):
        interface_pairing(object(), conormal, 0)
    with pytest.raises(ValueError, match="same macro"):
        interface_pairing(conormal, TriangularSkeleton(TetraMesh.unit_cube()), 0)


def test_polygonal_scalar_and_component_pairing_uses_original_macrofaces():
    mesh = PolyhedralMesh.cubes()
    scalar = PolygonalSkeleton3D(mesh, 1)
    matrix = interface_pairing(scalar, scalar, 0)
    assert_allclose(matrix, matrix.T, atol=1e-12)
    constant = scalar.constant_coefficients[scalar.cell_dofs(0)]
    assert_allclose(constant @ matrix @ constant, mesh.areas.sum(), atol=1e-12)
    assert_allclose(
        interface_pairing(ComponentTraceSpace(scalar, 2), ComponentTraceSpace(scalar, 2), 0),
        np.kron(matrix, np.eye(2)),
        atol=1e-12,
    )
