"""Independent DOLFINx/UFL compliance and weak symmetry on rectangular RT families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from pymhm.elasticity_tensor_rt import _operators, _rotation_basis
from pymhm.fenics import from_ufl
from pymhm.quadrilateral import CartesianMacroMesh, quadrilateral_quadrature
from pymhm.tensor_rt import tensor_rt_basis


@pytest.mark.fem
@pytest.mark.parametrize("degree,enrichment", [(1, 0), (1, 1), (2, 0)])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_tensor_mixed_operator_against_independent_ufl(
    degree: int, enrichment: int, anisotropic: bool
) -> None:
    """Restrict Basix RT_s/Q_s/Q_s operators to the published normal and rotation spaces."""
    dolfinx = pytest.importorskip("dolfinx")
    import basix.ufl
    import ufl
    from mpi4py import MPI

    mesh = CartesianMacroMesh(bounds=(0.2, 1.3, -0.1, 0.8))
    s = degree + enrichment
    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        mesh.cells[:, [0, 1, 3, 2]],
        mesh.points,
        ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(2,))),
    )
    rt = basix.ufl.element("RT", "quadrilateral", s + 1)
    scalar = basix.ufl.element(
        "DG", "quadrilateral", s, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(
        domain, basix.ufl.mixed_element([rt, rt, scalar, scalar, scalar])
    )
    sx, sy, ux, uy, rot = ufl.TrialFunctions(space)
    tx, ty, vx, vy, w = ufl.TestFunctions(space)
    sigma, tau = ufl.as_tensor((sx, sy)), ufl.as_tensor((tx, ty))
    x = ufl.SpatialCoordinate(domain)
    mu, lam = 1 + x[0], 2 + x[1]
    compliance = (sigma - lam / (2 * mu + 2 * lam) * ufl.tr(sigma) * ufl.Identity(2)) / (2 * mu)
    full_matrix = np.array(
        [
            [0.3, 0.01, 0.01, -0.1],
            [0.01, 0.5, 0.1, -0.03],
            [0.01, 0.1, 0.5, -0.03],
            [-0.1, -0.03, -0.03, 0.4],
        ]
    )
    if anisotropic:
        compliance = ufl.as_tensor(
            [
                [
                    sum(
                        full_matrix[2 * i + j, 2 * k + component] * sigma[k, component]
                        for k in range(2)
                        for component in range(2)
                    )
                    / (1 + x[0])
                    for j in range(2)
                ]
                for i in range(2)
            ]
        )
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 32})
    a = (
        ufl.inner(compliance, tau)
        + ufl.inner(ufl.as_vector((ux, uy)), ufl.div(tau))
        + ufl.inner(ufl.div(sigma), ufl.as_vector((vx, vy)))
        + rot * (tau[0, 1] - tau[1, 0])
        + w * (sigma[0, 1] - sigma[1, 0])
    ) * dx
    load = -ufl.inner(ufl.as_vector((1 + x[0], 2 - x[1])), ufl.as_vector((vx, vy))) * dx
    independent = from_ufl(a, load, [], np.empty(0, dtype=int))
    reference, _ = quadrilateral_quadrature(s + 3)
    physical = mesh.points[mesh.cells[0, 0]] + reference * mesh.spacing
    points = np.column_stack((physical, np.zeros(len(physical))))
    vectors, _, displacement = tensor_rt_basis(mesh, degree, enrichment, reference)
    vector_target = vectors[0].transpose(0, 2, 1).reshape(-1, vectors.shape[2])
    rotation = _rotation_basis(s, reference)
    ns, nd, nr = 2 * vectors.shape[2], displacement.shape[1], rotation.shape[1]
    transform = np.zeros((independent.matrix.shape[0], ns + 2 * nd + nr))
    for component in range(5):
        collapsed, mapping = space.sub(component).collapse()
        function = dolfinx.fem.Function(collapsed)
        columns = []
        for column in range(len(function.x.array)):
            function.x.array[:] = 0
            function.x.array[column] = 1
            columns.append(function.eval(points, np.zeros(len(points), dtype=np.int32)).ravel())
        basis = np.asarray(columns).T
        target = vector_target if component < 2 else displacement if component < 4 else rotation
        change = np.linalg.lstsq(basis, target, rcond=None)[0]
        assert_allclose(basis @ change, target, atol=3e-12, rtol=3e-12)
        ids = (
            2 * np.arange(ns // 2) + component
            if component < 2
            else ns + 2 * np.arange(nd) + component - 2
            if component < 4
            else ns + 2 * nd + np.arange(nr)
        )
        transform[np.ix_(mapping, ids)] = change
    mass, div, asym, force, *_ = _operators(
        mesh,
        degree,
        enrichment,
        lambda p: 2 + p[:, 1],
        lambda p: 1 + p[:, 0],
        lambda p: np.column_stack((1 + p[:, 0], 2 - p[:, 1])),
        18,
        compliance=(lambda p: full_matrix / (1 + p[:, 0, None, None])) if anisotropic else None,
    )
    expected = sparse.bmat([[mass, div.T, asym.T], [div, None, None], [asym, None, None]]).toarray()
    assert_allclose(transform.T @ independent.matrix @ transform, expected, atol=2e-11, rtol=2e-11)
    assert_allclose(
        transform.T @ independent.load,
        np.r_[np.zeros(ns), -force, np.zeros(nr)],
        atol=3e-12,
        rtol=3e-12,
    )
