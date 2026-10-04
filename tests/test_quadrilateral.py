"""Independent Cartesian-Qk, hybrid conservation and tensor-product patch checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.cartesian import solve_darcy_quadrilateral
from pymhm.fem.scalar.quadrilateral import (
    qk_basis,
    qk_space,
    quadrilateral_operators,
    quadrilateral_quadrature,
    quadrilateral_trace_coupling,
)
from pymhm.linalg.linear import LinearSolveError
from pymhm.meshes.cartesian import CartesianMacroMesh


def affine(x):
    return 0.7 + x[:, 0] - 2 * x[:, 1]


def square_product(x):
    return x[:, 0] ** 2 * x[:, 1] ** 2


def square_product_source(x):
    return -2 * (x[:, 0] ** 2 + x[:, 1] ** 2)


def square_product_flux(x):
    return -2 * np.column_stack((x[:, 0] * x[:, 1] ** 2, x[:, 0] ** 2 * x[:, 1]))


def cosine(x):
    return np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])


def cosine_source(x):
    return 2 * np.pi**2 * cosine(x)


def cosine_flux(x):
    return np.pi * np.column_stack(
        (
            np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
            np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
        )
    )


def test_geometry_and_independent_subdivisions():
    mesh = CartesianMacroMesh(3, 2, (-1, 2, -2, 2))
    assert mesh.cells.shape == (6, 4)
    assert len(mesh.faces) == 17
    assert len(mesh.boundary_faces) == 10
    assert_allclose(mesh.areas, 2)
    assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1)
    for cell in range(len(mesh.cells)):
        faces = mesh.cell_faces[cell]
        assert_allclose(
            np.sum(
                mesh.signs[cell, :, None] * mesh.normals[faces] * mesh.lengths[faces, None], axis=0
            ),
            0,
        )
    fine = mesh.submesh(4, (2, 3))
    assert fine.bounds == (0, 1, 0, 2)
    assert len(fine.cells) == 6
    assert_allclose(fine.areas, 1 / 3)
    assert not mesh.points.flags.writeable
    assert CartesianMacroMesh(2).ny == 2
    assert len(mesh.submesh(0, 2).cells) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(nx=0),
        dict(ny=1.5),
        dict(bounds=(0, 1, 2)),
        dict(bounds=(0, np.inf, 0, 1)),
        dict(bounds=(0, 0, 0, 1)),
        dict(bounds=(0, 1, 1, 0)),
        dict(bounds=(0, 1j, 0, 1)),
        dict(nx=20, bounds=(1e16, 1e16 + 4, 0, 1)),
    ],
)
def test_geometry_validation(kwargs):
    with pytest.raises(ValueError):
        CartesianMacroMesh(**kwargs)


@pytest.mark.parametrize("value", [0, (1,), (1, 0), (1, 2.3), True])
def test_refinement_validation(value):
    with pytest.raises(ValueError):
        CartesianMacroMesh().submesh(0, value)


def test_cell_validation():
    with pytest.raises(ValueError, match="outside"):
        CartesianMacroMesh().submesh(1, 2)
    with pytest.raises(ValueError, match="integer"):
        CartesianMacroMesh().submesh(-1, 2)


@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
def test_cardinality_partition_and_polynomial_derivatives(degree):
    nodes = np.array(
        [(x, y) for y in np.linspace(0, 1, degree + 1) for x in np.linspace(0, 1, degree + 1)]
    )
    values, gradients = qk_basis(degree, nodes)
    assert_allclose(values, np.eye(len(nodes)), atol=6e-14)
    reference, weights = quadrilateral_quadrature(6)
    values, gradients = qk_basis(degree, reference)
    assert_allclose(values.sum(axis=1), 1, atol=6e-14)
    assert_allclose(gradients.sum(axis=1), 0, atol=3e-13)
    nodal = nodes[:, 0] ** degree * nodes[:, 1] ** degree
    assert_allclose(
        values @ nodal, reference[:, 0] ** degree * reference[:, 1] ** degree, atol=3e-14
    )
    assert_allclose(
        np.einsum("qia,i->qa", gradients, nodal),
        degree
        * np.column_stack(
            (
                reference[:, 0] ** (degree - 1) * reference[:, 1] ** degree,
                reference[:, 0] ** degree * reference[:, 1] ** (degree - 1),
            )
        ),
        atol=3e-13,
    )
    assert_allclose(weights @ (values @ nodal), 1 / (degree + 1) ** 2, atol=1e-14)
    assert qk_basis(degree, np.empty((0, 2)))[0].shape == (0, (degree + 1) ** 2)


@pytest.mark.parametrize(
    "points", [[[0j, 0]], [[0, np.nan]], [[1.1, 0]], [[-0.1, 0]], [0, 1], [[0, 1, 2]]]
)
def test_basis_validation(points):
    with pytest.raises(ValueError, match="reference points"):
        qk_basis(2, points)


def test_q1_tensor_product_matrices_independently():
    mesh = CartesianMacroMesh(1, 1, (-2, 0, -1, 2))
    diffusion = np.diag([2.0, 3.0])
    matrix, mass, load = quadrilateral_operators(mesh, permeability=diffusion, source=4)
    m1 = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6
    a1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    expected = 2 * 3 / 2 * np.kron(m1, a1) + 3 * 2 / 3 * np.kron(a1, m1)
    assert sparse.issparse(matrix)
    assert_allclose(matrix.toarray(), expected, atol=1e-14)
    assert_allclose(mass.toarray(), 6 * np.kron(m1, m1), atol=1e-14)
    assert_allclose(load, 6, atol=1e-14)
    assert_allclose(matrix @ np.ones(4), 0, atol=1e-14)
    assert np.linalg.eigvalsh(matrix.toarray())[1] > 0


def test_qk_dof_sharing_and_batch_assembly():
    mesh = CartesianMacroMesh(17, 16, (-2, 3, 0, 1))
    dofs, nodes = qk_space(mesh, 2)
    assert len(nodes) == 35 * 33
    assert set(dofs[0, [2, 5, 8]]) == set(dofs[1, [0, 3, 6]])
    matrix, mass, load = quadrilateral_operators(mesh, 1, source=2)
    assert_allclose(load.sum(), 10, atol=1e-12)
    assert_allclose(mass.sum(), 5, atol=1e-12)
    assert_allclose(matrix @ np.ones(matrix.shape[0]), 0, atol=1e-12)


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("tensor", [2.0, [[3.0, 0.5], [0.5, 1.0]]])
def test_anisotropic_affine_hybrid_patch(degree, tensor):
    mesh = CartesianMacroMesh(2, 1, (-1, 1, 0, 2))
    result = solve_darcy_quadrilateral(
        mesh, permeability=tensor, dirichlet=affine, degree=degree, local_refinement=(3, 4)
    )
    k = np.eye(2) * tensor if np.ndim(tensor) == 0 else np.asarray(tensor)
    assert result.l2_error(affine) < 3e-13
    assert result.flux_l2_error(-k @ [1.0, -2.0]) < 2e-12
    assert_allclose(result.conservation_residuals(order=7), 0, atol=1e-12)
    for cell, fine in enumerate(result.local_meshes):
        centers = fine.points[fine.cells].mean(axis=1)
        p, q = result.evaluate(cell, centers)
        assert_allclose(p, affine(centers), atol=1e-13)
        assert_allclose(q, result.flux[cell], atol=1e-14)
        assert result.evaluate(cell, np.empty((0, 2)))[0].shape == (0,)


def test_q2_nonaffine_tensor_product_exact_pde_and_trace_moments():
    mesh = CartesianMacroMesh(2, 2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2, 1) for _ in mesh.faces))
    result = solve_darcy_quadrilateral(
        mesh,
        source=square_product_source,
        dirichlet=square_product,
        degree=2,
        skeleton=space,
        local_refinement=(2, 3),
    )
    assert result.l2_error(square_product) < 5e-14
    assert result.flux_l2_error(square_product_flux) < 1e-12
    assert_allclose(result.conservation_residuals(), 0, atol=1e-12)
    for face in range(len(mesh.faces)):
        t, weights = space.faces[face].quadrature(5)
        start, end = mesh.points[mesh.faces[face]]
        physical = start + t[:, None] * (end - start)
        value = space.faces[face].evaluate(t) @ result.hybrid.trace[space.dofs(face)]
        assert_allclose(value, square_product_flux(physical) @ mesh.normals[face], atol=1e-12)


def test_nonmatching_trace_breaks_and_reversed_faces():
    mesh = CartesianMacroMesh(2, 1)
    faces = tuple(
        FaceSpace((0.0, 0.27, 1.0), (1, 2)) if f % 2 else FaceSpace.uniform(1, 2, continuous=True)
        for f in range(len(mesh.faces))
    )
    space = SkeletonSpace(mesh, faces)
    result = solve_darcy_quadrilateral(
        mesh, dirichlet=affine, skeleton=space, degree=2, local_refinement=(4, 5)
    )
    assert result.l2_error(affine) < 1e-13
    assert result.flux_l2_error([-1, 2]) < 2e-12
    for cell in range(2):
        fine = result.local_meshes[cell]
        b = quadrilateral_trace_coupling(mesh, cell, fine, space, 2)
        exact = []
        for side, face in enumerate(mesh.cell_faces[cell]):
            t, w = faces[face].quadrature(6)
            exact.extend(
                mesh.signs[cell, side] * mesh.lengths[face] * (w @ faces[face].evaluate(t))
            )
        assert_allclose(b.sum(axis=0), exact, atol=5e-15)


def test_neumann_compatibility_and_mean_gauge():
    mesh = CartesianMacroMesh(2, 2)
    neumann = {
        int(face): float(np.array([-1.0, 2.0]) @ mesh.normals[face]) for face in mesh.boundary_faces
    }
    result = solve_darcy_quadrilateral(mesh, neumann=neumann, mean_pressure=0.2)
    assert result.l2_error(affine) < 1e-13
    assert result.flux_l2_error([-1.0, 2.0]) < 1e-12
    mixed = solve_darcy_quadrilateral(mesh, neumann={0: neumann[0]}, dirichlet=affine)
    assert mixed.l2_error(affine) < 1e-13
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy_quadrilateral(mesh, neumann=neumann, source=1)


def test_high_contrast_jump_inside_macro_resolved_by_rectangular_fine_grid():
    mesh = CartesianMacroMesh()
    neumann = {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 1]) > 0.5}

    def coefficient(x):
        return np.where(x[:, 0] < 0.5, 1.0, 1000.0)

    def exact(x):
        return np.where(x[:, 0] < 0.5, 1 - x[:, 0], 0.5 - (x[:, 0] - 0.5) / 1000)

    result = solve_darcy_quadrilateral(
        mesh, permeability=coefficient, neumann=neumann, dirichlet=exact, local_refinement=(4, 3)
    )
    assert result.l2_error(exact) < 2e-13
    assert result.flux_l2_error([1.0, 0.0]) < 2e-10
    assert_allclose(result.conservation_residuals(), 0, atol=1e-12)


def test_cosine_pressure_and_flux_converge():
    pressure, flux = [], []
    for n in (4, 8, 16):
        result = solve_darcy_quadrilateral(
            CartesianMacroMesh(n), source=cosine_source, dirichlet=cosine, local_refinement=3
        )
        pressure.append(result.l2_error(cosine, 7))
        flux.append(result.flux_l2_error(cosine_flux, 7))
    assert min(np.log2(np.array(pressure[:-1]) / pressure[1:])) > 1.8
    assert min(np.log2(np.array(flux[:-1]) / flux[1:])) > 0.9


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_worker_factory_path_and_source_reconstruction(backend):
    mesh = CartesianMacroMesh(2, 1)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
    result = solve_darcy_quadrilateral(
        mesh,
        source=square_product_source,
        dirichlet=square_product,
        degree=2,
        skeleton=space,
        local_refinement=3,
        backend=backend,
        workers=2,
    )
    assert result.l2_error(square_product) < 1e-13
    assert result.flux_l2_error(square_product_flux) < 1e-12


def test_l07_skeleton_and_reduced_global_unknown_count():
    mesh = CartesianMacroMesh(6, 11, (0, 1200, 0, 2200))
    space = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, 32, continuous=True) for _ in mesh.faces)
    )
    vertical = [f for f in mesh.boundary_faces if abs(mesh.normals[f, 0]) > 0.5]
    fixed = sum(space.faces[f].size for f in vertical)
    assert len(mesh.faces) == 149
    assert space.size - fixed + len(mesh.cells) == 4257


@pytest.mark.parametrize(
    "kwargs", [dict(degree=0), dict(local_refinement=(1,)), dict(quadrature_order=0)]
)
def test_solve_validation(kwargs):
    with pytest.raises(ValueError):
        solve_darcy_quadrilateral(CartesianMacroMesh(), **kwargs)


def test_skeleton_and_geometry_mismatch():
    mesh = CartesianMacroMesh()
    with pytest.raises(TypeError, match="CartesianMacroMesh"):
        solve_darcy_quadrilateral(TriangleMesh.unit_square())
    for space in (SkeletonSpace(mesh, components=2), SkeletonSpace(CartesianMacroMesh())):
        with pytest.raises(ValueError, match="scalar skeleton"):
            solve_darcy_quadrilateral(mesh, skeleton=space)


def test_sampling_validation_and_macro_side_selection():
    result = solve_darcy_quadrilateral(CartesianMacroMesh(), dirichlet=affine)
    with pytest.raises(ValueError, match="outside mesh"):
        result.evaluate(1, [[0, 0]])
    for points in ([[1j, 0]], [0, 0], [[np.nan, 0]], [[-0.1, 0]], [[1.1, 0]]):
        with pytest.raises(ValueError):
            result.evaluate(0, points)
    assert_allclose(result.evaluate(0, [[0, 0], [1, 1]])[0], [0.7, -0.3], atol=1e-13)


def test_invisible_skeleton_modes_are_rejected():
    mesh = CartesianMacroMesh()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    with pytest.raises(LinearSolveError, match="rank|singular|pivot"):
        solve_darcy_quadrilateral(mesh, skeleton=space, local_refinement=1, dirichlet=affine)


def test_sampling_material_and_gradient_use_same_macro_side():
    mesh = CartesianMacroMesh(2, 1)

    def permeability(x):
        return np.where(x[:, 0] < 0.5, 1.0, 1000.0)

    def pressure(x):
        return np.where(x[:, 0] <= 0.5, 1 - x[:, 0], 0.5 - (x[:, 0] - 0.5) / 1000)

    neumann = {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 1]) > 0.5}
    result = solve_darcy_quadrilateral(
        mesh, permeability=permeability, dirichlet=pressure, neumann=neumann, local_refinement=2
    )
    for cell in (0, 1):
        p, q = result.evaluate(cell, [[0.5, 0.3], [0.5, 0.5], [0.5, 1.0]])
        assert_allclose(p, 0.5, atol=2e-14)
        assert_allclose(q, np.broadcast_to([1.0, 0.0], q.shape), atol=2e-10)


def test_sampling_positive_side_survives_coordinate_roundoff():
    from dataclasses import replace

    result = solve_darcy_quadrilateral(CartesianMacroMesh(), local_refinement=5)
    fine = result.local_meshes[0]
    _, nodes = qk_space(fine, 1)
    sampled = replace(result, pressure=(nodes[:, 0] ** 2,))
    # 0.6 / 0.2 rounds below 3: floor alone selects the wrong fine cell.
    assert_allclose(sampled.evaluate(0, [[0.6, 0.3]])[1], [[-1.4, 0]], atol=2e-15)


def test_exact_cartesian_cut_quadrature_has_unit_moments_and_padding():
    from pymhm.fem.scalar.quadrilateral import _cartesian_rectangle_quadrature
    from pymhm.materials.cartesian import CartesianCellField

    field = CartesianCellField(np.ones((5, 6)), (0.2, 0.15))
    points, weights = _cartesian_rectangle_quadrature(
        np.array([[0.02, 0.02], [0.17, 0.12]]), np.array([0.1, 0.1]), field, 3
    )
    assert np.any(weights[0] == 0)
    assert np.all(weights[1] > 0)
    assert_allclose(weights.sum(axis=1), 1, atol=3e-15)
    for a in range(4):
        for b in range(4):
            assert_allclose(
                np.sum(weights * points[..., 0] ** a * points[..., 1] ** b, axis=1),
                1 / ((a + 1) * (b + 1)),
                atol=3e-15,
            )


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_cartesian_material_cut_operator_is_quadrature_invariant(degree):
    from pymhm.materials.cartesian import CartesianCellField

    tensors = np.zeros((5, 4, 2, 2))
    for i in range(5):
        for j in range(4):
            tensors[i, j] = np.array([[2.0, 0.3], [0.3, 1.0]]) * (1 + 7 * i + 11 * j)
    field = CartesianCellField(tensors, (0.2, 0.25))
    mesh = CartesianMacroMesh(3, 2)
    a5, m5, f5 = quadrilateral_operators(
        mesh, degree, permeability=field, source=square_product_source, order=5
    )
    a8, m8, f8 = quadrilateral_operators(
        mesh, degree, permeability=field, source=square_product_source, order=8
    )
    assert_allclose(a5.toarray(), a8.toarray(), atol=3e-11, rtol=5e-13)
    assert_allclose(m5.toarray(), m8.toarray(), atol=2e-15)
    assert_allclose(f5, f8, atol=3e-14)
    assert_allclose(a5 @ np.ones(a5.shape[0]), 0, atol=5e-11)


def test_exact_material_energy_on_a_rectangle_spanning_many_pixels():
    from pymhm.materials.cartesian import CartesianCellField

    values = np.arange(1.0, 13.0).reshape(3, 4)
    field = CartesianCellField(values, (2.0, 1.0))
    mesh = CartesianMacroMesh(1, 1, (0.0, 6.0, 0.0, 4.0))
    stiffness, _, _ = quadrilateral_operators(mesh, 3, permeability=field, order=4)
    _, points = qk_space(mesh, 3)
    # p=x+2y has energy 5 * sum(K_pixel * pixel_area), independent of any FE solver.
    coefficients = points[:, 0] + 2 * points[:, 1]
    assert_allclose(coefficients @ stiffness @ coefficients, 10 * values.sum(), rtol=3e-14)


def test_material_aligned_fast_path_matches_plain_callback():
    from pymhm.materials.cartesian import CartesianCellField

    field = CartesianCellField(np.array([[1.0, 2.0], [3.0, 1000.0]]), (0.5, 0.5))
    mesh = CartesianMacroMesh(4, 6)
    automatic = quadrilateral_operators(mesh, 2, permeability=field)
    callback = quadrilateral_operators(mesh, 2, permeability=lambda x: field(x))
    assert_allclose(automatic[0].toarray(), callback[0].toarray(), atol=0, rtol=0)


def test_cartesian_cut_material_validation():
    from pymhm.fem.scalar.quadrilateral import _cartesian_rectangle_quadrature
    from pymhm.materials.cartesian import CartesianCellField

    field3d = CartesianCellField(np.ones((2, 2, 2)), (0.5, 0.5, 0.5))
    with pytest.raises(ValueError, match="two-dimensional"):
        quadrilateral_operators(CartesianMacroMesh(), permeability=field3d)
    with pytest.raises(ValueError, match="two-dimensional"):
        _cartesian_rectangle_quadrature(np.array([[0.0, 0.0]]), np.ones(2), field3d, 4)
    field = CartesianCellField(np.ones((3, 3)), (0.2, 0.2))
    with pytest.raises(ValueError, match="outside"):
        quadrilateral_operators(CartesianMacroMesh(), permeability=field)


def test_nonaffine_cross_diffusion_matches_analytic_hessian():
    mesh = CartesianMacroMesh(2, 1, (0, 2, 0, 1))
    tensor = np.array([[2.0, 0.3], [0.3, 1.0]])
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))

    def forcing(x):
        return -4 * x[:, 1] ** 2 - 2 * x[:, 0] ** 2 - 2.4 * x[:, 0] * x[:, 1]

    def physical_flux(x):
        return square_product_flux(x) @ tensor.T

    result = solve_darcy_quadrilateral(
        mesh,
        permeability=tensor,
        source=forcing,
        dirichlet=square_product,
        degree=2,
        skeleton=space,
        local_refinement=(2, 3),
    )
    assert result.l2_error(square_product) < 3e-13
    assert result.flux_l2_error(physical_flux) < 3e-12
