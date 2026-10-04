"""Independent native UFL matrices on genuinely material-fitted tetrahedral geometry."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from pymhm.fem.scalar.tetrahedron import tetra_nodal_space, tetra_operators
from pymhm.materials.planar import PlanarMaterial, PlanarRegion
from pymhm.meshes.fitting import fit_planar_material
from pymhm.meshes.tetrahedron import TetraMesh


@pytest.mark.fem
@pytest.mark.parametrize("degree", [2, 4])
def test_native_fitted_tensor_transmission_matrix(degree) -> None:
    """Match every volume entry using DG0 material ownership and independent UFL gradients."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix")
    pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    tensor = np.array([[2.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.5]])
    normal = np.array([1.0, 0.4, 0.2])
    material = PlanarMaterial(tensor, (PlanarRegion([normal], [0.63], 25 * tensor),))
    cube = TetraMesh.unit_cube()
    fine = fit_planar_material(TetraMesh(cube.points, cube.cells[:1]), material).mesh
    with threadpool_limits(1):
        domain = dolfinx.mesh.create_mesh(
            mpi.COMM_SELF,
            fine.cells,
            fine.points,
            ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
        )
        space = dolfinx.fem.functionspace(
            domain,
            basix.ufl.element(
                "Lagrange", "tetrahedron", degree, lagrange_variant=basix.LagrangeVariant.equispaced
            ),
        )
        coefficient_space = dolfinx.fem.functionspace(
            domain, basix.ufl.element("DG", "tetrahedron", 0, shape=(3, 3))
        )
        coefficient = dolfinx.fem.Function(coefficient_space)
        coefficient.interpolate(lambda x: material(x.T).reshape(-1, 9).T)
        u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
        measure = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * degree})
        operators = []
        for form in [ufl.inner(coefficient * ufl.grad(u), ufl.grad(v)) * measure, u * v * measure]:
            assembled = dolfinx.fem.assemble_matrix(dolfinx.fem.form(form))
            assembled.scatter_reverse()
            operators.append(assembled.to_scipy().toarray())
        source = float(-0.6 * normal @ tensor @ normal)
        load = dolfinx.fem.assemble_vector(dolfinx.fem.form(source * v * measure)).array
        stiffness, mass, rhs = tetra_operators(
            fine, degree, diffusion=material, source=source, order=degree + 2
        )
        distance, permutation = cKDTree(space.tabulate_dof_coordinates()).query(
            tetra_nodal_space(fine, degree)[1]
        )
        assert max(distance) < 2e-14
        for actual, independent in [
            (stiffness.toarray(), operators[0]),
            (mass.toarray(), operators[1]),
        ]:
            expected = independent[np.ix_(permutation, permutation)]
            assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 3e-13
        assert_allclose(rhs, load[permutation], atol=3e-16, rtol=3e-13)
