"""Material-intersection moments and scalar formulations on unfitted triangles."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.bdm import bdm2_basis
from pymhm.cut_cells import (
    cartesian_edge_quadrature,
    cartesian_trace_values,
    fit_material_faces,
    material_triangle_quadrature,
)
from pymhm.darcy_mixed import _operators as bdm_operators
from pymhm.elements import p1_geometry, p1_operators, rt0_operators, triangle_quadrature
from pymhm.lagrange import element_tabulate, scalar_operators, tabulate
from pymhm.rad import _rad_local
from pymhm.reservoir import CartesianCellField
from pymhm.rt import rt_basis
from pymhm.scalar_transient import solve_transient_transport


def triangle():
    """Return a single triangle crossing x=0.37 inside a larger material domain."""
    return TriangleMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]])


def material():
    """Return two unit-height pixels with a tenfold contrast at x=0.37."""
    return CartesianCellField(np.array([[1.0], [10.0], [10.0]]), (0.37, 1.0))


def test_p1_energy_matches_analytical_clipped_area():
    mesh = triangle()
    gradient, _ = p1_geometry(mesh)
    left = 0.37 - 0.37**2 / 2
    integral = left + 10 * (0.5 - left)
    expected = integral * gradient[0] @ gradient[0].T
    first, _, _ = p1_operators(mesh, material(), order=3)
    second, _, _ = scalar_operators(mesh, 1, diffusion=material(), order=5)
    assert_allclose(first.toarray(), expected, atol=2e-14)
    assert_allclose(second.toarray(), expected, atol=2e-14)


def test_rt0_energy_matches_independent_integrated_polynomial():
    mesh = triangle()
    matrix, _, _ = rt0_operators(mesh, material(), order=3)
    midpoint = mesh.points[mesh.faces].mean(axis=1)
    coefficients = np.sum(midpoint * mesh.normals, axis=1) * mesh.lengths
    x = Polynomial([0, 1])
    integral = (x**2 * (1 - x) + (1 - x) ** 3 / 3).integ()
    left = integral(0.37) - integral(0)
    right = integral(1) - integral(0.37)
    assert_allclose(coefficients @ matrix @ coefficients, left + right / 10, atol=2e-15)


def test_unfitted_primal_patch_and_norms_resolve_the_material_flux_jump():
    mesh = TriangleMesh.unit_square()
    field = CartesianCellField(np.array([[1.0, 10.0, 10.0]]), (1.0, 0.37))
    skeleton = fit_material_faces(SkeletonSpace(mesh), field)
    solution = solve_darcy(
        mesh,
        permeability=field,
        dirichlet=lambda x: x[:, 0],
        skeleton=skeleton,
        local_refinement=4,
        degree=1,
        quadrature_order=4,
    )
    assert solution.l2_error(lambda x: x[:, 0], 6) < 3e-13
    assert (
        solution.flux_l2_error(lambda x: np.column_stack((-field(x), np.zeros(len(x)))), 6) < 3e-12
    )
    assert_allclose(solution.conservation_residuals(), 0, atol=3e-13)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_primal_and_mixed_matrices_are_invariant_under_exact_cut_quadrature(degree):
    mesh = TriangleMesh.unit_square()
    coefficient = CartesianCellField(np.array([[2.0, 5.0], [1.0, 11.0], [3.0, 4.0]]), (1 / 3, 0.5))
    matrices = [
        scalar_operators(mesh, degree, diffusion=coefficient, source=1, order=order)
        for order in (5, 8)
    ]
    mixed = [bdm_operators(mesh, coefficient, 1, order) for order in (5, 8)]
    for pair in (matrices, mixed):
        for first, second in zip(*pair, strict=True):
            if hasattr(first, "toarray"):
                first, second = first.toarray(), second.toarray()
            assert_allclose(first, second, atol=4e-13)


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_complete_rad_material_operators_resolve_cut_interfaces(stabilization):
    mesh = triangle()
    options = dict(
        mesh=mesh,
        skeleton=SkeletonSpace(mesh),
        degree=3,
        refinement=1,
        diffusion=material(),
        diffusion_divergence=(0, 0),
        velocity=(1, 0.3),
        velocity_divergence=0,
        reaction=2,
        source=lambda x: 1 + x[:, 0],
        stabilization=stabilization,
    )
    first, second = (_rad_local(0, order=order, **options) for order in (5, 8))
    assert_allclose(first.problem.matrix.toarray(), second.problem.matrix.toarray(), atol=3e-13)
    assert_allclose(first.problem.load, second.problem.load, atol=3e-14)
    assert_allclose(first.problem.constraints, second.problem.constraints, atol=3e-14)


def test_elementwise_nodal_and_hdiv_tabulation_agree_with_direct_reference_points():
    mesh = TriangleMesh.unit_square()
    common, _ = triangle_quadrature(3)
    bary = np.stack((common, common[:, ::-1]))
    _, _, values, gradient, hessian = element_tabulate(mesh, 3, bary)
    for cell in range(len(mesh.cells)):
        _, _, direct, first, second = tabulate(mesh, 3, bary[cell])
        assert_allclose(values[cell], direct, atol=0)
        assert_allclose(gradient[cell], first[cell], atol=0)
        assert_allclose(hessian[cell], second[cell], atol=0)
        for evaluator in (bdm2_basis, lambda grid, points: rt_basis(grid, 2, points)):
            actual = evaluator(mesh, bary)
            expected = evaluator(mesh, bary[cell])
            for a, b in zip(actual, expected, strict=True):
                scale = max(1.0, float(np.max(np.abs(b[cell]))))
                assert_allclose(a[cell], b[cell], rtol=2e-14, atol=64 * np.finfo(float).eps * scale)
    broadcast, weights, resolved = material_triangle_quadrature(mesh, 2.0, 3)
    assert resolved == 2.0
    assert_allclose(weights.sum(axis=1), 1)
    _, _, broadcast_values, _, _ = element_tabulate(mesh, 3, broadcast)
    assert_allclose(broadcast_values[0], values[0])
    for evaluator in (bdm2_basis, lambda grid, points: rt_basis(grid, 2, points)):
        with pytest.raises(ValueError, match="barycentric"):
            evaluator(mesh, bary[:1])


@pytest.mark.parametrize("continuous", [False, True])
def test_automatic_face_fitting_preserves_spaces_and_existing_breakpoints(continuous):
    mesh = TriangleMesh.unit_square()
    old = FaceSpace((0, 0.25, 1), (1, 2), continuous)
    skeleton = SkeletonSpace(mesh, (old,) * len(mesh.faces), components=2)
    field = CartesianCellField(np.ones((2, 2)), (0.5, 0.5))
    fitted = fit_material_faces(skeleton, field)
    assert fitted.components == 2
    assert fitted.mesh is mesh
    for space in fitted.faces:
        assert space.breaks == (0, 0.25, 0.5, 1)
        assert space.degrees == (1, 2, 2)
        assert space.continuous == continuous
    twice = fit_material_faces(fitted, field)
    assert twice.faces == fitted.faces
    with pytest.raises(ValueError, match="planar"):
        fit_material_faces(skeleton, 1.0)


def test_material_fast_path_and_geometry_contracts():
    mesh = TriangleMesh.unit_square(2)
    field = CartesianCellField(np.array([[1, 2], [3, 4]]), (0.5, 0.5))
    bary, weights, values = material_triangle_quadrature(mesh, field, 3)
    points = np.einsum("tqi,tij->tqj", bary, mesh.points[mesh.cells])
    assert bary.strides[0] == 0
    assert_allclose(values, field(points.reshape(-1, 2)))
    assert_allclose(weights.sum(axis=1), 1)
    with pytest.raises(ValueError, match="planar"):
        material_triangle_quadrature(mesh, CartesianCellField(np.ones((1, 1, 1)), (1, 1, 1)))
    skeleton = SkeletonSpace(mesh, (FaceSpace((0, 1e-16, 1), (0, 0)),) * len(mesh.faces))
    assert all(1e-16 in face.breaks for face in fit_material_faces(skeleton, field).faces)


@pytest.mark.parametrize("reverse", [False, True])
def test_one_sided_material_edge_rule_without_coordinate_perturbations(reverse):
    field = CartesianCellField(np.array([[1.0, 3.0], [10.0, 30.0]]), (0.5, 0.5))
    start, end = ([0.5, 0.0], [0.5, 1.0])
    if reverse:
        start, end = end, start
    for side, factor in ((0.25, 1), (0.75, 10)):
        points, weights, values = cartesian_edge_quadrature(
            start, end, field, 3, interior_point=[side, 0.25]
        )
        assert_allclose(points[:, 0], 0.5, atol=0)
        assert_allclose(weights.sum(), 1, atol=3e-16)
        assert_allclose(values, factor * np.where(points[:, 1] < 0.5, 1, 3))
        assert_allclose(weights @ (values * points[:, 1] ** 2), factor * (1 / 24 + 7 / 8))
    points, weights, values = cartesian_edge_quadrature(
        [0, 0], [1, 1], field, 4, interior_point=[0.6, 0.2]
    )
    assert_allclose(weights @ values, 15.5)
    assert_allclose(points[:, 0], points[:, 1], atol=0)


def test_material_junction_traces_are_selected_independently_per_incident_cell():
    field = CartesianCellField(np.array([[1.0, 3.0], [10.0, 30.0]]), (0.5, 0.5))
    points = np.tile([0.5, 0.5], (4, 1))
    interior = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    assert_allclose(cartesian_trace_values(field, points, interior), [1, 3, 10, 30])
    with pytest.raises(ValueError, match="planar"):
        cartesian_trace_values(1, points, interior)
    for points, interior in (([[np.nan, 0]], [0.3, 0.2]), ([[0, 0]], [0.3j, 0.2]), ([[0, 0]], [0])):
        with pytest.raises(ValueError, match="finite real"):
            cartesian_trace_values(field, points, interior)


@pytest.mark.parametrize(
    "start,end,inside,field,match",
    [
        ([0, 0], [1, 0], [0.5, 0.5], 1.0, "planar"),
        ([0, 0], [0, 0], [0.5, 0.5], None, "positive length"),
        ([0, 0], [1, 0], [np.nan, 0.5], None, "finite real"),
        ([0, 0], [1, 0], [0.5j, 0.5], None, "finite real"),
        ([0, 0], [0, 1], [0, 0.5], None, "strict side"),
        ([0, 0], [0, 1], [-1e-16, 0.5], None, "outside"),
    ],
)
def test_invalid_material_edge_data_are_rejected(start, end, inside, field, match):
    if field is None:
        field = CartesianCellField(np.ones((2, 2)), (0.5, 0.5))
    with pytest.raises(ValueError, match=match):
        cartesian_edge_quadrature(start, end, field, interior_point=inside)


@pytest.mark.parametrize("stabilization", ["galerkin", "supg"])
def test_transient_loads_use_the_same_cut_rule_as_the_operator(stabilization):
    mesh = triangle()
    result = solve_transient_transport(
        mesh,
        [0, 0.1, 0.2],
        diffusion=material(),
        diffusion_divergence=(0, 0),
        velocity=(0.3, 0.5),
        initial=lambda x: x[:, 1],
        source=lambda x, t: x[:, 1] + 0.5 * (1 + t),
        dirichlet=lambda x, t: (1 + t) * x[:, 1],
        degree=3,
        local_refinement=2,
        stabilization=stabilization,
    )
    for time, solution in zip(result.times[1:], result.solutions, strict=True):
        assert solution.l2_error(lambda x, t=time: (1 + t) * x[:, 1], order=6) < 3e-13


def test_material_fitted_local_mesh_represents_interface_gradient_jump():
    """Fitting the approximation mesh resolves a transmission kink exactly."""
    from pymhm import solve_darcy
    from pymhm.cut_cells import fit_material_mesh
    from pymhm.refinement import validate_submesh

    coarse = TriangleMesh.unit_square()
    material = CartesianCellField(np.array([[1.0], [7.0]]), (0.5, 1.0))
    fine = tuple(fit_material_mesh(coarse.submesh(cell, 3), material) for cell in range(2))
    for cell, mesh in enumerate(fine):
        validate_submesh(coarse, cell, mesh)
        assert_allclose(mesh.areas.sum(), 0.5)
        bary, _, _ = material_triangle_quadrature(mesh, material, 4)
        assert bary.strides[0] == 0  # Every fitted triangle lies in one pixel.

    def pressure(x):
        """Continuous pressure with continuous physical normal flux."""
        return np.minimum(x[:, 0], 0.5) + np.maximum(x[:, 0] - 0.5, 0) / 7

    result = solve_darcy(
        coarse,
        local_meshes=fine,
        skeleton=fit_material_faces(SkeletonSpace(coarse), material),
        permeability=material,
        dirichlet=pressure,
        degree=1,
        quadrature_order=6,
    )
    assert result.l2_error(pressure) < 2e-13
    assert result.flux_l2_error((-1.0, 0.0)) < 2e-12
    with pytest.raises(ValueError, match="primal Darcy only"):
        solve_darcy(coarse, local_meshes=fine, formulation="mixed")
    with pytest.raises(ValueError, match="primal Darcy only"):
        solve_darcy(coarse, local_meshes=fine[:1])


def test_fitted_cartesian_junctions_and_submesh_validation():
    """Grid corners retain conformity and invalid local-domain replacements fail."""
    from pymhm.cut_cells import fit_material_mesh
    from pymhm.refinement import validate_submesh

    mesh = TriangleMesh.unit_square()
    field = CartesianCellField(np.arange(6).reshape(3, 2) + 1.0, (1 / 3, 0.5))
    for cell in range(2):
        fitted = fit_material_mesh(mesh.submesh(cell, 2), field)
        validate_submesh(mesh, cell, fitted)
    rounding_mesh = TriangleMesh(
        np.array([[0.5, 0.0], [0.5, 2 / 3], [0.25, 1 / 3]]), np.array([[0, 1, 2]])
    )
    layered = CartesianCellField(np.array([[10.0, 1.0]]), (1.0, 0.5))
    fitted = fit_material_mesh(rounding_mesh.submesh(0, 8), layered)
    validate_submesh(rounding_mesh, 0, fitted)
    with pytest.raises(ValueError, match="planar"):
        fit_material_mesh(mesh, 1.0)
    with pytest.raises(ValueError, match="TriangleMesh"):
        validate_submesh(mesh, 0, None)
    with pytest.raises(ValueError, match="cover"):
        validate_submesh(mesh, 0, mesh)
    # Duplicate coverage is balanced by a missing region but exposes interior edges.
    duplicate_points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.0], [1.0, 0.0], [1.0, 0.5]]
    )
    invalid = TriangleMesh(duplicate_points, np.array([[0, 1, 2], [3, 4, 5]]))
    with pytest.raises(ValueError, match="unmatched interior"):
        validate_submesh(mesh, 0, invalid)


def test_material_mesh_area_guard_and_reaction_validation(monkeypatch):
    """Reject a clipping operator that loses measure and invalid elliptic reaction."""
    import pymhm.cut_cells as cutting

    mesh = TriangleMesh.unit_square()
    field = CartesianCellField(np.ones((1, 1)), (1.0, 1.0))
    clip = cutting._clip_polygon

    def losing_area(polygon, axis, level, lower):
        """Perturb a polygon inward to simulate a loss of geometric measure."""
        polygon = clip(polygon, axis, level, lower)
        if axis == 1 and not lower and polygon[:, 0].mean() > 0.5:
            return np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
        return polygon.mean(axis=0) + 0.99 * (polygon - polygon.mean(axis=0))

    monkeypatch.setattr(cutting, "_clip_polygon", losing_area)
    with pytest.raises(ValueError, match="preserve domain area"):
        cutting.fit_material_mesh(mesh, field)
    with pytest.raises(ValueError, match="reaction"):
        scalar_operators(mesh, 2, reaction=-1.0)


def test_fitted_area_accounts_for_global_coordinate_roundoff():
    """Small translated cells retain their area under material intersection and merging."""
    from pymhm.cut_cells import fit_material_mesh

    points = np.array([[1096.875, 550.0], [1101.5625, 550.0], [1115.625, 584.375]])
    material = CartesianCellField(np.ones((60, 220)), (20.0, 10.0))
    mesh = TriangleMesh(points, np.array([[0, 1, 2]]))
    fitted = fit_material_mesh(mesh, material)
    assert abs(fitted.areas.sum() - mesh.areas.sum()) < 1e-11
    # Translation changes neither the geometry nor its material partition.
    translation = np.array([1000.0, 500.0])
    shifted = TriangleMesh(points - translation, mesh.cells)
    field = CartesianCellField(np.ones((60, 220)), (20.0, 10.0), origin=-translation)
    other = fit_material_mesh(shifted, field)
    assert_allclose(fitted.points - translation, other.points, rtol=0, atol=3e-13)
    assert_array_equal(fitted.cells, other.cells)
