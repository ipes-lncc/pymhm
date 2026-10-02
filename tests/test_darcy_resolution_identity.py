"""Galerkin energy identities distinguish local enrichment from skeletal restriction."""

import numpy as np
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.cut_cells import material_triangle_quadrature
from pymhm.darcy import solve_darcy
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.reconstruction_moments import _primal_flux
from pymhm.reservoir import CartesianCellField


def physical_energy_difference(first, second):
    """Integrate raw-flux energy differences on the finer common local partition."""
    total = 0.0
    for cell, fine in enumerate(second.local_meshes):
        bary, weights, tensors = material_triangle_quadrature(fine, second.permeability, 4)
        points = np.einsum("tqi,tia->tqa", bary, fine.points[fine.cells])
        coordinates = points.reshape(-1, 2)
        first_mesh = first.local_meshes[cell]
        vertices = first_mesh.points[first_mesh.cells]
        inverse = np.linalg.inv((vertices[:, 1:] - vertices[:, :1]).swapaxes(1, 2))
        local = np.einsum("tab,qtb->qta", inverse, coordinates[:, None] - vertices[None, :, 0])
        score = np.minimum(local.min(axis=2), 1 - local.sum(axis=2))
        owners = np.argmax(score, axis=1)
        assert np.min(np.max(score, axis=1)) > -1e-12
        fine_owners = np.repeat(np.arange(len(fine.cells)), len(bary[0]))
        old = _primal_flux(first, cell)(coordinates, owners)
        new = _primal_flux(second, cell)(coordinates, fine_owners)
        difference = new - old
        factors = (fine.areas[:, None] * weights).ravel()
        total += factors @ np.einsum("qa,qab,qb->q", difference, np.linalg.inv(tensors), difference)
    return total


def solve_control(mesh, material, refinement, segments):
    """Preserve the physical BVP while selecting two nested approximation parameters."""
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
    neumann = {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 0]) > 0.5}
    solution = solve_darcy(
        mesh,
        degree=2,
        permeability=material,
        dirichlet=lambda x: 1 - x[:, 1],
        neumann=neumann,
        skeleton=skeleton,
        local_refinement=refinement,
        quadrature_order=4,
    )
    bottom = [f for f in mesh.boundary_faces if np.all(mesh.points[mesh.faces[f], 1] == 0)]
    inflow = -sum(
        mesh.lengths[f] * np.mean(solution.hybrid.trace[skeleton.dofs(f)]) for f in bottom
    )
    assert_allclose(solution.conservation_residuals(), 0, atol=4e-13)
    return solution, inflow


def test_nested_local_and_trace_energy_identities_have_opposite_monotonicity():
    """Local enrichment lowers the energy minimum; extra trace constraints raise it."""
    mesh = TriangleMesh.unit_square()
    material = CartesianCellField(
        np.array([[1.0, 100.0], [3.0, 2.0]])[..., None, None] * np.eye(2), (0.5, 0.5)
    )
    with threadpool_limits(1):
        coarse, q_coarse = solve_control(mesh, material, 2, 1)
        fine, q_fine = solve_control(mesh, material, 4, 1)
        trace, q_trace = solve_control(mesh, material, 4, 2)
        local_increment = physical_energy_difference(coarse, fine)
        trace_increment = physical_energy_difference(fine, trace)
    assert local_increment > 1e-5
    assert trace_increment > 1e-5
    assert_allclose(q_coarse - q_fine, local_increment, rtol=2e-11, atol=2e-13)
    assert_allclose(q_trace - q_fine, trace_increment, rtol=2e-11, atol=2e-13)
