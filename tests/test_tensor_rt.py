"""Check RT tensor spaces, enrichment and physical mixed equilibrium."""

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss, legvander
from numpy.testing import assert_allclose

from pymhm import CartesianMacroMesh, FaceSpace, SkeletonSpace
from pymhm._legacy.models.darcy.tensor import solve_darcy_tensor_rt
from pymhm.fem.hdiv.tensor_rt import tensor_rt_basis, tensor_rt_dofs
from pymhm.fem.scalar.quadrilateral import quadrilateral_quadrature
from pymhm.materials.cartesian import CartesianCellField


def test_unfitted_inverse_mass_uses_exact_material_intersections():
    """A displaced pixel grid gives an analytically integrable RT0 inverse mass."""
    from pymhm.fem.hdiv.tensor_rt import _operators

    mesh = CartesianMacroMesh(1, 1)
    material = CartesianCellField(np.array([[1.0], [10.0], [10.0]]), (0.4, 1.0))
    mass, divergence, load = _operators(mesh, 0, 0, material, 0.0, 2)
    # On the unit cell, bottom-face RT0 basis is (0,y-1).
    assert_allclose(mass[0, 0], (0.4 + 0.6 / 10) / 3, rtol=1e-12, atol=1e-12)
    assert_allclose(divergence.toarray(), np.ones((1, 4)), rtol=1e-12, atol=1e-12)
    assert_allclose(load, 0, rtol=0, atol=1e-12)
    higher = _operators(mesh, 0, 0, material, 0.0, 7)[0]
    assert_allclose(mass.toarray(), higher.toarray(), rtol=1e-12, atol=1e-12)


def test_cellwise_basis_and_uncached_solves():
    """Cellwise tabulation and operator cache reuse preserve the same physical fields."""
    mesh = CartesianMacroMesh(2, 1)
    points = np.array([[[0.1, 0.2], [0.3, 0.4]], [[0.8, 0.7], [0.6, 0.5]]])
    evaluated = tensor_rt_basis(mesh, 1, 1, points)
    for cell in range(2):
        shared = tensor_rt_basis(mesh, 1, 1, points[cell])
        # Different native batch sizes need a nonzero floor at polynomial zeros.
        for actual, expected in (
            (evaluated[0][cell], shared[0][cell]),
            (evaluated[1][cell], shared[1][cell]),
            (evaluated[2][cell], shared[2]),
        ):
            scale = max(1.0, float(np.max(np.abs(expected))))
            assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * scale)
    material = CartesianCellField(np.array([[1.0], [10.0], [10.0]]), (0.4, 1.0))
    cached = solve_darcy_tensor_rt(mesh, permeability=material, dirichlet=lambda x: x[:, 1])
    uncached = solve_darcy_tensor_rt(
        mesh, permeability=material, dirichlet=lambda x: x[:, 1], reuse_operators=False
    )
    assert_allclose(cached.hybrid.trace, uncached.hybrid.trace, rtol=1e-10, atol=1e-10)
    assert all(np.max(abs(v)) < 1e-10 for v in cached.equilibrium_residuals())

    def pressure(x):
        """Exact affine pressure with tangential material jumps."""
        return x[:, 1]

    def flux(x):
        """Exact divergence-free physical flux with the same material."""
        return np.column_stack((np.zeros(len(x)), -material(x)))

    low = cached.errors(pressure, flux, 0.0, order=4)
    high = cached.errors(pressure, flux, 0.0, order=7)
    assert_allclose(list(low.values()), list(high.values()), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("k,n", [(0, 0), (1, 0), (1, 2), (2, 1), (3, 2)])
def test_oriented_face_moments_and_divergence_range(k, n):
    mesh = CartesianMacroMesh(1, 1, (0.2, 1.9, -0.3, 0.4))
    x, w = leggauss(k + 3)
    t = (1 + x) / 2
    for edge in range(4):
        points = (
            np.column_stack((t, np.zeros_like(t))),
            np.column_stack((np.ones_like(t), t)),
            np.column_stack((1 - t, np.ones_like(t))),
            np.column_stack((np.zeros_like(t), 1 - t)),
        )[edge]
        basis, _, _ = tensor_rt_basis(mesh, k, n, points)
        normal = basis[0] @ mesh.normals[mesh.cell_faces[0, edge]]
        moments = mesh.lengths[mesh.cell_faces[0, edge]] * (
            legvander(x, k).T @ (w[:, None] / 2 * normal)
        )
        assert_allclose(
            moments,
            np.eye(basis.shape[2])[edge * (k + 1) : (edge + 1) * (k + 1)],
            rtol=1e-12,
            atol=1e-12,
        )
    points, w = quadrilateral_quadrature(k + n + 2)
    _, div, p = tensor_rt_basis(mesh, k, n, points)
    divergence = p.T @ (w[:, None] * div[0])
    assert np.linalg.matrix_rank(divergence) == (k + n + 1) ** 2


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unenriched_space_agrees_with_independent_basix(k):
    basix = pytest.importorskip("basix")
    element = basix.create_element(
        basix.ElementFamily.RT,
        basix.CellType.quadrilateral,
        k + 1,
        lagrange_variant=basix.LagrangeVariant.legendre,
    )
    points, _ = quadrilateral_quadrature(k + 3)
    table = element.tabulate(1, points)
    basis, div, _ = tensor_rt_basis(CartesianMacroMesh(), k, 0, points)
    sample = basis[0].transpose(0, 2, 1).reshape(-1, basis.shape[2])
    reference = table[0].transpose(0, 2, 1).reshape(-1, element.dim)
    change = np.linalg.lstsq(reference, sample, rcond=None)[0]
    assert_allclose(reference @ change, sample, rtol=1e-12, atol=4e-12)
    assert_allclose(
        (table[1, :, :, 0] + table[2, :, :, 1]) @ change, div[0], rtol=1e-12, atol=2e-11
    )


@pytest.mark.parametrize("k,n", [(1, 0), (1, 1), (2, 0), (2, 2), (3, 1)])
def test_affine_tensor_patch(k, n):
    mesh = CartesianMacroMesh(2, 1)
    tensor = np.array([[2.0, 0.4], [0.4, 1.0]])
    result = solve_darcy_tensor_rt(
        mesh,
        degree=k,
        enrichment=n,
        local_refinement=1,
        permeability=tensor,
        dirichlet=lambda x: 1 + x[:, 0] - 2 * x[:, 1],
        quadrature_order=k + n + 3,
    )
    errors = result.errors(lambda x: 1 + x[:, 0] - 2 * x[:, 1], -(tensor @ [1, -2]), 0)
    assert max(errors.values()) < 2e-11
    assert max(np.max(abs(v)) for v in result.equilibrium_residuals()) < 2e-12
    assert max(np.max(abs(v)) for v in result.normal_flux_residuals()) < 2e-12


def test_interior_enrichment_reproduces_zero_normal_quartic_bubble():
    def exact(x):
        return np.prod((x * (1 - x)) ** 2, axis=1)

    def flux(x):
        b = (x * (1 - x)) ** 2
        d = 2 * x * (1 - x) * (1 - 2 * x)
        return -np.column_stack((d[:, 0] * b[:, 1], b[:, 0] * d[:, 1]))

    def source(x):
        b = (x * (1 - x)) ** 2
        d = 2 - 12 * x + 12 * x * x
        return -d[:, 0] * b[:, 1] - b[:, 0] * d[:, 1]

    result = solve_darcy_tensor_rt(
        CartesianMacroMesh(), degree=1, enrichment=3, local_refinement=1, source=source
    )
    assert max(result.errors(exact, flux, source).values()) < 1e-10


def test_neumann_gauge_orientation_and_split_traces():
    mesh = CartesianMacroMesh(2)
    data = {int(f): float(mesh.normals[f] @ [-1.0, -2.0]) for f in mesh.boundary_faces}
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces))
    result = solve_darcy_tensor_rt(mesh, skeleton=skeleton, neumann=data, mean_pressure=3.5)
    assert max(result.errors(lambda x: 2 + x[:, 0] + 2 * x[:, 1], [-1, -2], 0).values()) < 2e-11
    assert_allclose(result.hybrid.gauge_multipliers, 0, atol=1e-12)
    data[next(iter(data))] += 1
    with pytest.raises(ValueError, match="incompatible"):
        solve_darcy_tensor_rt(mesh, neumann=data)


def test_trace_degree_and_alignment_rejections():
    mesh = CartesianMacroMesh()
    for face in (FaceSpace.uniform(2), FaceSpace((0.0, 0.50004, 1.0), (0, 0))):
        skeleton = SkeletonSpace(mesh, tuple(face for _ in mesh.faces))
        with pytest.raises(ValueError, match="align"):
            solve_darcy_tensor_rt(mesh, skeleton=skeleton)
    with pytest.raises(ValueError, match="scalar skeleton"):
        solve_darcy_tensor_rt(
            mesh, skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces), 2)
        )
    for points in ([1, 2], [[1j, 0]], [[np.inf, 0]]):
        with pytest.raises(ValueError, match="real pairs"):
            tensor_rt_basis(mesh, 1, 0, points)
    assert tensor_rt_dofs(mesh, 0, 0).shape == (1, 4)
