"""Independent variational assembly checks across finite element backends."""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg

from pymhm import TriangleMesh
from pymhm._legacy.models.vector import solve_brinkman
from pymhm.backends.fenics import from_ufl
from pymhm.core.contracts import LocalAssembly
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import nodal_space


def _independent_inverse_constants(mesh: TriangleMesh, degree: int) -> np.ndarray:
    """Solve the inverse inequality in analytic monomials, independently of native Pk bases."""
    import basix

    points, weights = basix.make_quadrature(basix.CellType.triangle, 2 * degree + 2)
    powers = [(i, total - i) for total in range(1, degree + 1) for i in range(total + 1)]
    gradient = np.zeros((len(points), len(powers), 2))
    hessian = np.zeros((len(points), len(powers), 2, 2))
    x, y = points.T
    for column, (i, j) in enumerate(powers):
        if i:
            gradient[:, column, 0] = i * x ** (i - 1) * y**j
        if j:
            gradient[:, column, 1] = j * x**i * y ** (j - 1)
        if i > 1:
            hessian[:, column, 0, 0] = i * (i - 1) * x ** (i - 2) * y**j
        if j > 1:
            hessian[:, column, 1, 1] = j * (j - 1) * x**i * y ** (j - 2)
        if i and j:
            hessian[:, column, 0, 1] = hessian[:, column, 1, 0] = (
                i * j * x ** (i - 1) * y ** (j - 1)
            )
    constants = []
    for vertices in mesh.points[mesh.cells]:
        inverse = np.linalg.inv((vertices[1:] - vertices[0]).T)
        physical_gradient = np.einsum("qia,ab->qib", gradient, inverse)
        physical_hessian = np.einsum("qiab,ac,bd->qicd", hessian, inverse, inverse)
        laplacian = np.trace(physical_hessian, axis1=-2, axis2=-1)
        energy = np.einsum("q,qia,qja->ij", weights, physical_gradient, physical_gradient)
        residual = np.einsum("q,qi,qj->ij", weights, laplacian, laplacian)
        maximum = linalg.eigvalsh(residual, energy)[-1]
        diameter = np.max(np.linalg.norm(vertices[:, None] - vertices[None], axis=2))
        constants.append(min(1 / 3, 1 / (diameter**2 * maximum)) if maximum > 0 else 1 / 3)
    return np.array(constants)


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("drag", [0.0, 3.0, 1000.0, "tensor", "variable-tensor"])
def test_usfem_matrix_and_force_match_independent_ufl(
    monkeypatch: pytest.MonkeyPatch, degree: int, drag: float | str
) -> None:
    """Compare P1--P3 Hessians, tensor drag cross-terms and stabilized loads with UFL."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    import pymhm._legacy.models.flow.solver

    native_problems = []
    native_constants = []
    assemble = pymhm._legacy.models.flow.solver._flow_local
    inverse_bound = pymhm._legacy.models.flow.solver._laplacian_inverse_bound

    def capture(cell: int, **kwargs: Any) -> LocalAssembly:
        """Capture the exact native local matrix before global condensation."""
        local = assemble(cell, **kwargs)
        native_problems.append(local.problem)
        return local

    def capture_bound(*args: Any) -> np.ndarray:
        """Record the native inverse bound solely for an independent equality assertion."""
        value = inverse_bound(*args)
        native_constants.append(value)
        return value

    tensor = np.array([[20.0, 3.0], [3.0, 1.0]])

    def resistance(points: np.ndarray) -> np.ndarray:
        """Evaluate the prescribed scalar or SPD tensor independently of native helpers."""
        if isinstance(drag, str):
            factor = 1 + points[:, 0] + 2 * points[:, 1] if drag == "variable-tensor" else 1.0
            return np.broadcast_to(
                np.asarray(factor)[..., None, None] * tensor, (len(points), 2, 2)
            )
        return np.broadcast_to(drag * np.eye(2), (len(points), 2, 2))

    monkeypatch.setattr(pymhm._legacy.models.flow.solver, "_flow_local", capture)
    monkeypatch.setattr(pymhm._legacy.models.flow.solver, "_laplacian_inverse_bound", capture_bound)
    viscosity = 2.0
    square = TriangleMesh.unit_square()
    coarse = TriangleMesh(square.points @ np.array([[1.1, 0.25], [-0.15, 0.85]]), square.cells)
    solution = solve_brinkman(
        coarse,
        formulation="usfem",
        degree=degree,
        viscosity=viscosity,
        drag=resistance,
        local_refinement=2,
        quadrature_order=5,
        source=lambda points: np.column_stack(
            (
                1 + points[:, 0] + 2 * points[:, 1] + points[:, 0] * points[:, 1],
                -2 + 3 * points[:, 0] - points[:, 1] + points[:, 0] ** 2,
            )
        ),
    )
    for native, fine, native_bound in zip(
        native_problems, solution.local_meshes, native_constants, strict=True
    ):
        independent_bound = _independent_inverse_constants(fine, degree)
        assert_allclose(native_bound, independent_bound, atol=2e-14, rtol=2e-12)
        geometry = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
        domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, fine.cells, fine.points, geometry)
        # The native nodal Pk basis is equispaced. Basix's default GLL points
        # differ at degree three, requiring a basis change instead of a permutation.
        variant = basix.LagrangeVariant.equispaced
        velocity = basix.ufl.element(
            "Lagrange", "triangle", degree, shape=(2,), lagrange_variant=variant
        )
        pressure = basix.ufl.element("Lagrange", "triangle", degree, lagrange_variant=variant)
        space = dolfinx.fem.functionspace(domain, basix.ufl.mixed_element([velocity, pressure]))
        # Only the sampling points of the native cellwise resistance bound
        # are shared. Its polynomial residuals and all integrals below are UFL.
        bary, _ = triangle_quadrature(max(5, degree + 2))
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        gamma_max = np.linalg.eigvalsh(resistance(points.reshape(-1, 2)))[:, -1]
        gamma_max = gamma_max.reshape(len(fine.cells), -1).max(axis=1)
        diameter_squared = np.max(fine.lengths[fine.cell_faces], axis=1) ** 2
        viscous_scale = 4 * viscosity / independent_bound
        values = diameter_squared / (
            np.maximum(gamma_max * diameter_squared, viscous_scale) + viscous_scale
        )
        cell_space = dolfinx.fem.functionspace(domain, ("DG", 0))
        tau = dolfinx.fem.Function(cell_space)
        for cell, original in enumerate(domain.topology.original_cell_index):
            tau.x.array[cell_space.dofmap.cell_dofs(cell)] = values[original]
        x = ufl.SpatialCoordinate(domain)
        force = ufl.as_vector((1 + x[0] + 2 * x[1] + x[0] * x[1], -2 + 3 * x[0] - x[1] + x[0] ** 2))
        gamma = drag * ufl.Identity(2) if isinstance(drag, float) else ufl.as_matrix(tensor)
        if drag == "variable-tensor":
            gamma *= 1 + x[0] + 2 * x[1]
        # Write the signed strong residual directly: this must not reuse the
        # package's UFL helper, Hessians or native block-construction formulas.
        u, p = ufl.TrialFunctions(space)
        v, q = ufl.TestFunctions(space)
        trial_residual = -viscosity * ufl.div(ufl.grad(u)) + gamma * u + ufl.grad(p)
        test_residual = -viscosity * ufl.div(ufl.grad(v)) + gamma * v + ufl.grad(q)
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree + 4})
        a = (
            viscosity * ufl.inner(ufl.grad(u), ufl.grad(v))
            + ufl.inner(gamma * u, v)
            - p * ufl.div(v)
            - q * ufl.div(u)
            - tau * ufl.inner(trial_residual, test_residual)
        ) * dx
        load = (ufl.inner(force, v) - tau * ufl.inner(force, test_residual)) * dx
        assembled = from_ufl(a, load, [], np.empty(0, dtype=int))
        permutation = np.empty(assembled.matrix.shape[0], dtype=np.int64)
        _, native_nodes = nodal_space(fine, degree)
        for component in (0, 1):
            subspace, mapping = space.sub(component).collapse()
            points = subspace.tabulate_dof_coordinates()[:, :2]
            distances = np.linalg.norm(points[:, None] - native_nodes[None], axis=2)
            nodes = np.argmin(distances, axis=1)
            assert np.max(distances[np.arange(len(points)), nodes]) < 2e-14
            assert len(np.unique(nodes)) == len(native_nodes)
            if component == 0:
                permutation[mapping] = (2 * nodes[:, None] + np.arange(2)).ravel()
            else:
                permutation[mapping] = 2 * len(native_nodes) + nodes
        assert len(np.unique(permutation)) == len(permutation)
        assert_allclose(
            assembled.matrix.toarray(),
            native.matrix.toarray()[permutation][:, permutation],
            atol=3e-13,
            rtol=2e-13,
        )
        assert_allclose(assembled.load, native.load[permutation], atol=3e-14, rtol=2e-13)
