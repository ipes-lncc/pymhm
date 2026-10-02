"""Point functional exactness, angular sharing and pure-Neumann well balance."""

import numpy as np
import pytest

from pymhm import TriangleMesh, solve_darcy
from pymhm.lagrange import nodal_space
from pymhm.loads import point_load_vector, split_point_sources
from pymhm.polygon import PolygonMesh
from pymhm.quadrilateral import CartesianMacroMesh, qk_space


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_point_functional_evaluates_representable_polynomial(degree):
    mesh = TriangleMesh.unit_square(3)
    points = np.array([[0, 0, -1], [1, 1, 2], [0.37, 0.21, 3], [0.5, 0.5, -4]])
    _, nodes = nodal_space(mesh, degree)
    values = nodes[:, 0] ** degree + 2 * nodes[:, 1] ** degree + 1
    expected = points[:, 2] @ (points[:, 0] ** degree + 2 * points[:, 1] ** degree + 1)
    load = point_load_vector(mesh, degree, points)
    assert load @ values == pytest.approx(expected, abs=1e-13)
    assert load.sum() == pytest.approx(points[:, 2].sum(), abs=1e-14)
    np.testing.assert_array_equal(point_load_vector(mesh, degree, []), 0)


def test_geometric_sharing_preserves_strength_at_corners_edges_and_interiors():
    mesh = TriangleMesh.unit_square(2)
    for point in [[0, 0, -1], [1, 1, 2], [0.5, 0.5, 7], [0.5, 0.2, 3], [0.3, 0.1, 4]]:
        parts = split_point_sources(mesh, [point])
        assert sum(part[:, 2].sum() for part in parts) == pytest.approx(point[2])
    corners = split_point_sources(mesh, [[0, 0, 1]])
    np.testing.assert_allclose([part[:, 2].sum() for part in corners], [0.5, 0.5, 0, 0, 0, 0, 0, 0])
    interior = split_point_sources(mesh, [[0.5, 0.5, 1]])
    weights = sorted(part[0, 2] for part in interior if len(part))
    np.testing.assert_allclose(weights, [0.125, 0.125, 0.125, 0.125, 0.25, 0.25])
    assert all(part.shape == (0, 3) for part in split_point_sources(mesh, []))


def test_vertex_sharing_uses_actual_angles_in_nonuniform_mesh():
    mesh = TriangleMesh(
        np.array([[0, 0], [2, 0], [2, 1], [0, 1]]), np.array([[0, 1, 2], [0, 2, 3]])
    )
    parts = split_point_sources(mesh, [[0, 0, 1]])
    assert parts[0][0, 2] == pytest.approx(np.arctan(0.5) / (np.pi / 2))
    assert sum(part[:, 2].sum() for part in parts) == pytest.approx(1)


@pytest.mark.parametrize(
    "values", [[1, 2, 3], [[1, 2]], [[1j, 0, 1]], [[0, 0, np.inf]], [["x", 0, 1]]]
)
def test_point_sources_reject_invalid_values(values):
    mesh = TriangleMesh.unit_square()
    for function in (split_point_sources, lambda m, s: point_load_vector(m, 2, s)):
        with pytest.raises(ValueError, match="finite real"):
            function(mesh, values)


def test_point_sources_reject_extrapolation():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="outside"):
        split_point_sources(mesh, [[-0.01, 0.4, 1]])
    with pytest.raises(ValueError, match="outside"):
        point_load_vector(mesh, 1, [[1.01, 0.4, 1]])


@pytest.mark.parametrize("formulation,degree", [("primal", 1), ("primal", 2), ("mixed", 1)])
def test_quarter_wells_mean_gauge_conservation_sign_and_symmetry(formulation, degree):
    mesh = TriangleMesh.unit_square(3)
    solution = solve_darcy(
        mesh,
        point_sources=[[0, 0, -1], [1, 1, 1]],
        neumann={int(face): 0 for face in mesh.boundary_faces},
        formulation=formulation,
        degree=degree,
        local_refinement=2,
    )
    assert solution.pressure[0].mean() < 0 < solution.pressure[-1].mean()
    np.testing.assert_allclose(solution.conservation_residuals(), 0, atol=3e-14, rtol=0)
    assert sum(part[:, 2].sum() for part in solution.point_sources) == pytest.approx(0)
    # Centrosymmetry exchanges source and sink. Both discretizations preserve
    # the corresponding zero mean without an arbitrary pinned pressure node.
    mean = 0.0
    for fine, p in zip(solution.local_meshes, solution.pressure, strict=True):
        if formulation == "mixed":
            mean += fine.areas @ p
        else:
            from pymhm.lagrange import scalar_operators

            _, mass, _ = scalar_operators(fine, degree)
            mean += np.ones(len(p)) @ (mass @ p)
    assert mean == pytest.approx(0, abs=1e-14)
    if formulation == "mixed":
        for residual in solution.fine_conservation_residuals():
            np.testing.assert_allclose(residual, 0, atol=3e-14, rtol=0)


def test_unbalanced_wells_are_incompatible_with_impermeable_boundary():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy(
            mesh,
            point_sources=[[1, 1, 1]],
            neumann={int(face): 0 for face in mesh.boundary_faces},
        )


@pytest.mark.parametrize("degree", [1, 2, 3, 5])
def test_cartesian_point_functional_preserves_tensor_polynomials(degree):
    """Qk point loads preserve tensor monomials at corners, interfaces and interiors."""
    mesh = CartesianMacroMesh(3, 2, (-2, 4, 1, 5))
    points = np.array([[-2, 1, 1], [4, 5, -2], [0, 3, 3], [0.37, 2.19, -4]])
    _, nodes = qk_space(mesh, degree)

    def polynomial(x):
        """Evaluate a represented tensor polynomial in physical coordinates."""
        return (1 + x[:, 0] / 4) ** degree * (x[:, 1] / 5) ** degree

    load = point_load_vector(mesh, degree, points)
    assert load @ polynomial(nodes) == pytest.approx(points[:, 2] @ polynomial(points), abs=2e-12)
    assert load.sum() == pytest.approx(points[:, 2].sum(), abs=2e-13)
    np.testing.assert_array_equal(point_load_vector(mesh, degree, []), 0)
    for part in split_point_sources(mesh, points):
        assert part.shape[1] == 3
    assert sum(p[:, 2].sum() for p in split_point_sources(mesh, points)) == pytest.approx(-2)
    interface = split_point_sources(mesh, [[0, 3, 1]])
    np.testing.assert_allclose(sorted(p[0, 2] for p in interface if len(p)), [0.25] * 4)
    for function in (split_point_sources, lambda m, s: point_load_vector(m, degree, s)):
        with pytest.raises(ValueError, match="outside"):
            function(mesh, [[4.0001, 3, 1]])


def test_nonconvex_polygon_point_sharing_uses_interior_angle():
    """A source at a reentrant macro vertex splits as 3/4 and 1/4, not equally."""
    mesh = PolygonMesh(
        np.array([[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 1], [0, 1], [1, 1]]),
        (np.array([0, 1, 2, 3, 4, 5]), np.array([3, 2, 6, 4])),
    )
    parts = split_point_sources(mesh, [[0.5, 0.5, 1]])
    np.testing.assert_allclose([p[:, 2].sum() for p in parts], [0.75, 0.25], atol=2e-16)
    for cell, part in enumerate(parts):
        fine = mesh.submesh(cell, 2)
        load = point_load_vector(fine, 3, part)
        _, nodes = nodal_space(fine, 3)
        assert load @ (nodes[:, 0] ** 3 + nodes[:, 1]) == pytest.approx(
            part[:, 2].sum() * 0.625, abs=2e-14
        )
    assert all(p.shape == (0, 3) for p in split_point_sources(mesh, []))
