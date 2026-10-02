"""Independent DOLFINx/UFL tests of trilinear Piola mixed Darcy operators."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.fenics import from_ufl
from pymhm.mapped_rt import HexMesh, _operators, cube_quadrature, mapped_rt_basis


@pytest.mark.fem
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_mapped_piola_operator_against_native_ufl(degree):
    """Compare full RT/Q mixed matrices and loads on a genuinely nonaffine hexahedron."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    base = HexMesh.unit_cube()
    points = base.points.copy()
    points[:, 0] *= 1 + 0.2 * points[:, 1] + 0.1 * points[:, 2]
    mesh = HexMesh(points, base.cells)
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        mesh.cells[:, [0, 4, 2, 6, 1, 5, 3, 7]],
        mesh.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))),
    )
    vector = basix.ufl.element("RT", "hexahedron", degree + 1)
    scalar = basix.ufl.element(
        "DG", "hexahedron", degree, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(domain, basix.ufl.mixed_element([vector, scalar]))
    q, p = ufl.TrialFunctions(space)
    v, w = ufl.TestFunctions(space)
    x = ufl.SpatialCoordinate(domain)
    coefficient = (1 + x[0] + x[2]) * ufl.as_matrix(
        ((2.0, 0.1, 0.2), (0.1, 3.0, 0.3), (0.2, 0.3, 4.0))
    )
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 22})
    a = (ufl.inner(ufl.inv(coefficient) * q, v) - p * ufl.div(v) - w * ufl.div(q)) * dx
    L = -(1 + x[0] * x[2]) * w * dx
    independent = from_ufl(a, L, [], np.empty(0, dtype=int))
    reference, _ = cube_quadrature(degree + 3)
    physical = mesh.geometry(reference)[0][0]
    vectors, _, pressure = mapped_rt_basis(mesh, degree, reference)
    targets = (vectors[0].transpose(0, 2, 1).reshape(-1, vectors.shape[2]), pressure)
    transform = np.zeros((independent.matrix.shape[0], vectors.shape[2] + pressure.shape[1]))
    offset = 0
    for component, target in enumerate(targets):
        collapsed, mapping = space.sub(component).collapse()
        function = dolfinx.fem.Function(collapsed)
        basis = []
        for i in range(len(function.x.array)):
            function.x.array[:] = 0
            function.x.array[i] = 1
            basis.append(function.eval(physical, np.zeros(len(physical), dtype=np.int32)).ravel())
        basis = np.array(basis).T
        change = np.linalg.lstsq(basis, target, rcond=None)[0]
        assert_allclose(basis @ change, target, atol=8e-11, rtol=8e-11)
        transform[np.ix_(mapping, np.arange(offset, offset + target.shape[1]))] = change
        offset += target.shape[1]
    K = np.array([[2.0, 0.1, 0.2], [0.1, 3.0, 0.3], [0.2, 0.3, 4.0]])
    mass, div, f, _ = _operators(
        mesh,
        degree,
        lambda p: (1 + p[:, 0] + p[:, 2])[:, None, None] * K,
        lambda p: 1 + p[:, 0] * p[:, 2],
        12,
    )
    expected = sparse.bmat([[mass, -div.T], [-div, None]]).toarray()
    assert_allclose(transform.T @ independent.matrix @ transform, expected, atol=2e-10, rtol=2e-10)
    assert_allclose(
        transform.T @ independent.load, np.r_[np.zeros(mass.shape[0]), -f], atol=2e-11, rtol=2e-11
    )
