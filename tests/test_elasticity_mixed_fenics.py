"""Independent DOLFINx/Basix AFW assembly versus condensed native mixed elasticity."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.bdm import bdm2_evaluate
from pymhm.elasticity_mixed import solve_elasticity_mixed
from pymhm.elements import triangle_quadrature
from pymhm.fenics import from_ufl
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.solvers import solve_linear

pytestmark = pytest.mark.fem


@pytest.mark.parametrize("lame_lambda,mean_pressure", [(2.0, 0.0), (np.inf, 0.0), (np.inf, 2.3)])
def test_condensed_bdm2_matches_independent_conforming_dolfinx(lame_lambda, mean_pressure):
    """Compare non-exact AFW fields, including incompressible stress with a global gauge."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    coarse = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(
        coarse, tuple(FaceSpace.uniform(2, 2) for _ in coarse.faces), components=2
    )

    def exact(points):
        """Cubic displacement; the mixed finite element solution is not exact."""
        x, y = points.T
        if np.isinf(lame_lambda):
            return np.column_stack((x**3, -3 * x * x * y))
        return np.column_stack((x**3 + x * y, -2 * x * x * y + y * y))

    def source(points):
        """Differentiate compressible stress or incompressible stress 2 eps(u)-p I."""
        x, y = points.T
        if np.isinf(lame_lambda):
            return np.column_stack((-6 * x + 1, 6 * y + 1))
        return np.column_stack((-12 * x, -11 + 4 * y))

    native = solve_elasticity_mixed(
        coarse,
        skeleton=skeleton,
        dirichlet=exact,
        source=source,
        lame_lambda=lame_lambda,
        mean_pressure=mean_pressure,
    )
    all_points = np.vstack([mesh.points for mesh in native.local_meshes])
    points, inverse = np.unique(all_points, axis=0, return_inverse=True)
    cells = []
    offset = 0
    for mesh in native.local_meshes:
        cells.extend(inverse[offset + mesh.cells])
        offset += len(mesh.points)
    cells = np.array(cells)
    coordinate_element = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, points, coordinate_element)
    bdm = basix.ufl.element("BDM", "triangle", 2)
    displacement = basix.ufl.element("DG", "triangle", 1, shape=(2,))
    rotation = basix.ufl.element("DG", "triangle", 1)
    space = dolfinx.fem.functionspace(
        domain, basix.ufl.mixed_element([bdm, bdm, displacement, rotation])
    )
    row0, row1, u, q = ufl.TrialFunctions(space)
    test0, test1, v, r = ufl.TestFunctions(space)
    sigma, tau = (
        ufl.as_matrix(((row0[0], row0[1]), (row1[0], row1[1]))),
        ufl.as_matrix(((test0[0], test0[1]), (test1[0], test1[1]))),
    )
    compliance = (sigma - ufl.tr(sigma) * ufl.Identity(2) / 2) / 2
    if np.isfinite(lame_lambda):
        compliance += ufl.tr(sigma) * ufl.Identity(2) / (4 * (1 + lame_lambda))
    asym_sigma, asym_tau = sigma[0, 1] - sigma[1, 0], tau[0, 1] - tau[1, 0]
    a = (
        ufl.inner(compliance, tau)
        + ufl.dot(u, ufl.div(tau))
        + ufl.dot(ufl.div(sigma), v)
        + q * asym_tau
        + r * asym_sigma
    ) * ufl.dx
    x = ufl.SpatialCoordinate(domain)
    prescribed = ufl.as_vector((x[0] ** 3 + x[0] * x[1], -2 * x[0] ** 2 * x[1] + x[1] ** 2))
    force = ufl.as_vector((-12 * x[0], -11 + 4 * x[1]))
    if np.isinf(lame_lambda):
        prescribed = ufl.as_vector((x[0] ** 3, -3 * x[0] ** 2 * x[1]))
        force = ufl.as_vector((-6 * x[0] + 1, 6 * x[1] + 1))
    load = -ufl.dot(force, v) * ufl.dx + ufl.dot(prescribed, tau * ufl.FacetNormal(domain)) * ufl.ds
    problem = from_ufl(a, load, [], np.empty(0, dtype=int))
    field = dolfinx.fem.Function(space)
    if np.isinf(lame_lambda):
        trace_moment = dolfinx.fem.assemble_vector(dolfinx.fem.form(ufl.tr(tau) * ufl.dx)).array
        column = sparse.csc_matrix(trace_moment[:, None])
        augmented = sparse.bmat([[problem.matrix, column], [column.T, None]], format="csc")
        area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=domain)))
        coefficients = solve_linear(augmented, np.r_[problem.load, -2 * mean_pressure * area])
        assert abs(coefficients[-1]) < 2e-12
        field.x.array[:] = coefficients[:-1]
        assert_allclose(trace_moment @ field.x.array, -2 * mean_pressure * area, atol=2e-12)
    else:
        field.x.array[:] = problem.condense().source
    components = [component.collapse() for component in field.split()]
    original_to_local = np.argsort(domain.topology.original_cell_index)
    bary, _ = triangle_quadrature(4)
    offset = 0
    for mesh, stress, displacement, rotation in zip(
        native.local_meshes, native.stress, native.displacement, native.rotation, strict=True
    ):
        coordinates = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
        evaluation_points = np.column_stack((coordinates, np.zeros(len(coordinates))))
        indices = np.repeat(original_to_local[offset : offset + len(mesh.cells)], len(bary))
        reference = [part.eval(evaluation_points, indices.astype(np.int32)) for part in components]
        values, _ = bdm2_evaluate(mesh, stress, bary)
        assert_allclose(values[:, :, 0].reshape(-1, 2), reference[0], atol=4e-11, rtol=0)
        assert_allclose(values[:, :, 1].reshape(-1, 2), reference[1], atol=4e-11, rtol=0)
        assert_allclose(
            np.einsum("qi,tia->tqa", bary, displacement).reshape(-1, 2),
            reference[2],
            atol=3e-12,
            rtol=0,
        )
        assert_allclose((rotation @ bary.T).ravel(), reference[3].ravel(), atol=3e-12, rtol=0)
        offset += len(mesh.cells)
