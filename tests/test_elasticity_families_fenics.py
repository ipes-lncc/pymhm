"""Independent UFL compliance/divergence/weak-symmetry operators for mixed stress families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm import TriangleMesh
from pymhm._legacy.models.elasticity.stress import _operators
from pymhm.backends.fenics import from_ufl
from pymhm.fem.hdiv.bdm_family import BDMFamily
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import reference_basis


@pytest.mark.fem
@pytest.mark.parametrize("degree,enrichment", [(1, 1), (1, 2), (2, 1), (3, 0)])
def test_mixed_family_operator_against_independent_ufl(degree: int, enrichment: int) -> None:
    """Change basis by physical field evaluation, then compare complete mixed operators."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    family = BDMFamily(degree, enrichment)
    order = family.polynomial_degree
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        mesh.cells,
        x=mesh.points,
        e=ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))),
    )
    vector = basix.ufl.element("BDM", "triangle", order)
    scalar = basix.ufl.element(
        "DG", "triangle", order - 1, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(
        domain, basix.ufl.mixed_element([vector, vector, scalar, scalar, scalar])
    )
    sx, sy, ux, uy, rotation = ufl.TrialFunctions(space)
    tx, ty, vx, vy, test_rotation = ufl.TestFunctions(space)
    stress, test_stress = ufl.as_tensor((sx, sy)), ufl.as_tensor((tx, ty))
    displacement, test_displacement = ufl.as_vector((ux, uy)), ufl.as_vector((vx, vy))
    x = ufl.SpatialCoordinate(domain)
    mu, lam = 1.0 + x[0], 2.0 + x[1]
    compliance = (stress - lam / (2 * mu + 2 * lam) * ufl.tr(stress) * ufl.Identity(2)) / (2 * mu)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 2 * (order + 12)})
    a = (
        ufl.inner(compliance, test_stress)
        + ufl.inner(displacement, ufl.div(test_stress))
        + ufl.inner(ufl.div(stress), test_displacement)
        + rotation * (test_stress[0, 1] - test_stress[1, 0])
        + test_rotation * (stress[0, 1] - stress[1, 0])
    ) * dx
    force = ufl.as_vector((1 + x[0], 2 - x[1]))
    independent = from_ufl(a, -ufl.inner(force, test_displacement) * dx, [], np.empty(0, dtype=int))
    # Rational compliance is integrated at high order in both independent rules.
    bary, _ = triangle_quadrature(order + 3)
    points = np.column_stack((bary[:, 1:], np.zeros(len(bary))))
    target_vector = family.basis(mesh, bary)[0][0].transpose(0, 2, 1).reshape(-1, family.local_size)
    target_scalar = reference_basis(order - 1, bary)[0]
    ns, nd = 2 * family.local_size, target_scalar.shape[1]
    transform = np.zeros((independent.matrix.shape[0], ns + 3 * nd))
    for component in range(5):
        collapsed, mapping = space.sub(component).collapse()
        function = dolfinx.fem.Function(collapsed)
        values = []
        for column in range(len(function.x.array)):
            function.x.array[:] = 0
            function.x.array[column] = 1
            values.append(function.eval(points, np.zeros(len(points), dtype=np.int32)).ravel())
        native_values = np.asarray(values).T
        target = target_vector if component < 2 else target_scalar
        change = np.linalg.lstsq(native_values, target, rcond=None)[0]
        assert_allclose(native_values @ change, target, atol=3e-11, rtol=3e-12)
        columns = (
            2 * np.arange(family.local_size) + component
            if component < 2
            else ns
            + (2 * np.arange(nd) + component - 2 if component < 4 else 2 * nd + np.arange(nd))
        )
        transform[np.ix_(mapping, columns)] = change
    mass, div, asym, body = _operators(
        mesh,
        lambda x: 2 + x[:, 1],
        lambda x: 1 + x[:, 0],
        lambda x: np.column_stack((1 + x[:, 0], 2 - x[:, 1])),
        order + 10,
        family,
    )
    expected = sparse.bmat([[mass, div.T, asym.T], [div, None, None], [asym, None, None]]).toarray()
    assert_allclose(transform.T @ independent.matrix @ transform, expected, rtol=2e-10, atol=2e-9)
    assert_allclose(
        transform.T @ independent.load,
        np.r_[np.zeros(ns), -body, np.zeros(nd)],
        atol=2e-12,
        rtol=2e-12,
    )
