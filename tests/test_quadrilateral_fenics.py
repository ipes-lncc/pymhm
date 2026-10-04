"""Independent Basix verification of tensor-product interpolation and derivatives."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.mesh import FaceSpace, SkeletonSpace
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    qk_basis,
    qk_space,
    quadrilateral_operators,
    quadrilateral_quadrature,
    quadrilateral_trace_coupling,
)


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2, 3, 4, 5, 6])
def test_tensor_product_basis_matches_native_basix(degree):
    basix = pytest.importorskip("basix")
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        degree,
        basix.LagrangeVariant.equispaced,
    )
    ours = np.array(
        [(x, y) for y in np.linspace(0, 1, degree + 1) for x in np.linspace(0, 1, degree + 1)]
    )
    indices = np.argmin(np.linalg.norm(ours[:, None] - element.points[None], axis=2), axis=1)
    assert len(np.unique(indices)) == len(ours)
    assert_allclose(element.points[indices], ours, atol=1e-15)
    points, _ = quadrilateral_quadrature(6)
    table = element.tabulate(1, points)
    values, derivatives = qk_basis(degree, points)
    assert_allclose(values, table[0, :, :, 0][:, indices], atol=7e-14)
    assert_allclose(derivatives[..., 0], table[1, :, :, 0][:, indices], atol=3e-13)
    assert_allclose(derivatives[..., 1], table[2, :, :, 0][:, indices], atol=3e-13)


@pytest.mark.fem
def test_native_q2_original_diffusion_and_signed_p1_boundary_equations():
    """Independent full Q2/P1 assembly recovers the anisotropic quadratic field."""
    basix = pytest.importorskip("basix")
    mesh = CartesianMacroMesh(1, bounds=(-1.0, 2.0, 0.2, 1.4))
    fine = mesh.submesh(0, 2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    dofs, nodes = qk_space(fine, 2)
    element = basix.create_element(
        basix.ElementFamily.P,
        basix.CellType.quadrilateral,
        2,
        basix.LagrangeVariant.equispaced,
    )
    reference_nodes = np.array([(x, y) for y in [0.0, 0.5, 1.0] for x in [0.0, 0.5, 1.0]])
    indices = np.argmin(
        np.linalg.norm(reference_nodes[:, None] - element.points[None], axis=2), axis=1
    )
    assert len(np.unique(indices)) == 9
    points, weights = quadrilateral_quadrature(8)
    table = element.tabulate(1, points)[:, :, indices, 0]
    values = table[0]
    gradient = np.moveaxis(table[1:3], 0, -1) / fine.spacing
    tensor = np.array([[2.0, 0.3], [0.3, 1.2]])
    area = float(fine.areas[0])
    local_A = area * np.einsum("q,qia,ab,qjb->ij", weights, gradient, tensor, gradient)
    local_M = area * np.einsum("q,qi,qj->ij", weights, values, values)
    # p=x²+xy+2y²; f=-div(K grad p) from its independent Hessian.
    source = -(2 * tensor[0, 0] + 2 * tensor[0, 1] + 4 * tensor[1, 1])
    local_f = area * source * np.einsum("q,qi->i", weights, values)
    native_A, native_M = np.zeros((25, 25)), np.zeros((25, 25))
    native_f = np.zeros(25)
    for ids in dofs:
        native_A[np.ix_(ids, ids)] += local_A
        native_M[np.ix_(ids, ids)] += local_M
        native_f[ids] += local_f
    A, M, f = quadrilateral_operators(fine, 2, permeability=tensor, source=source, order=8)
    assert_allclose(A.toarray(), native_A, rtol=3e-14, atol=2e-14)
    assert_allclose(M.toarray(), native_M, rtol=3e-14, atol=2e-15)
    assert_allclose(f, native_f, rtol=3e-14, atol=2e-15)
    # Every native boundary is parameterized by the actual PyMHM face endpoints.
    s, w = np.polynomial.legendre.leggauss(8)
    t = np.r_[(s + 1) / 4, (s + 1) / 4 + 0.5]
    w = np.tile(w / 4, 2)
    trace = np.column_stack((np.ones_like(t), 2 * t - 1))
    native_B = np.zeros((25, 8))
    boundary_load = np.empty(8)
    exact_trace = np.empty(8)
    for side, face in enumerate(mesh.cell_faces[0]):
        endpoints = mesh.points[mesh.faces[face]]
        physical = endpoints[0] + t[:, None] * (endpoints[1] - endpoints[0])
        coordinates = (physical - mesh.points[0]) / fine.spacing
        grid = np.minimum(coordinates.astype(int), 1)
        unit = coordinates - grid
        local_phi = element.tabulate(0, unit)[0, :, indices, 0].T
        cells = grid[:, 1] * 2 + grid[:, 0]
        phi = np.zeros((len(t), 25))
        phi[np.arange(len(t))[:, None], dofs[cells]] = local_phi
        signed_weights = w * mesh.lengths[face] * mesh.signs[0, side]
        native_B[:, 2 * side : 2 * side + 2] = phi.T @ (signed_weights[:, None] * trace)
        x, y = physical.T
        pressure = x * x + x * y + 2 * y * y
        boundary_load[2 * side : 2 * side + 2] = trace.T @ (signed_weights * pressure)
        x, y = endpoints.T
        flux = -np.column_stack((2 * x + y, x + 4 * y)) @ tensor
        normal_flux = flux @ mesh.normals[face]
        exact_trace[2 * side : 2 * side + 2] = [
            normal_flux.mean(),
            (normal_flux[1] - normal_flux[0]) / 2,
        ]
    B = quadrilateral_trace_coupling(mesh, 0, fine, skeleton, 2)
    assert_allclose(B, native_B, rtol=3e-14, atol=2e-15)
    assert np.linalg.matrix_rank(native_B) == 8
    assert np.linalg.matrix_rank(native_A) == 24
    x, y = nodes.T
    exact = np.r_[x * x + x * y + 2 * y * y, exact_trace]
    rhs = np.r_[native_f, boundary_load]
    solved = []
    for diffusion, coupling in ((native_A, native_B), (A.toarray(), B)):
        original = np.block([[diffusion, coupling], [coupling.T, np.zeros((8, 8))]])
        assert np.linalg.matrix_rank(original) == 33
        field = np.linalg.solve(original, rhs)
        assert_allclose(field, exact, rtol=0, atol=3e-13)
        assert_allclose(original @ field, rhs, rtol=0, atol=3e-13)
        solved.append(field)
    assert_allclose(
        values @ solved[0][:25][dofs[0]], values @ solved[1][:25][dofs[0]], rtol=0, atol=2e-13
    )
