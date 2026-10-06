"""Independent native UFL stress/divergence/skew operators and conforming AFW3D solutions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.stress_3d import solve_elasticity_mixed_3d
from pymhm._legacy.models.elasticity.stress_forms_3d import mixed_elasticity_operators_3d
from pymhm.backends.fenics import from_ufl
from pymhm.fem.hdiv.family_3d import HDiv3DFamily, cell_quadrature
from pymhm.meshes.mixed import AffineMixedMesh, hdiv3d_basis


def native_space(mesh, degree):
    """Build native Basix BDM rows and independent discontinuous displacement/rotation fields."""
    import basix.ufl
    import dolfinx
    import ufl
    from mpi4py import MPI

    domain = dolfinx.mesh.create_mesh(
        MPI.COMM_SELF,
        mesh.cells,
        x=mesh.points,
        e=ufl.Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))),
    )
    bdm = basix.ufl.element("BDM", "tetrahedron", degree)
    scalar = basix.ufl.element(
        "DG", "tetrahedron", degree - 1, lagrange_variant=basix.LagrangeVariant.equispaced
    )
    space = dolfinx.fem.functionspace(domain, basix.ufl.mixed_element([bdm] * 3 + [scalar] * 6))
    return domain, space


def forms(space, mu, lam, force, dx):
    """Write the full tensor weak-symmetry variational equations independently in UFL."""
    import ufl

    trial, test = ufl.TrialFunctions(space), ufl.TestFunctions(space)
    sigma, tau = ufl.as_tensor(trial[:3]), ufl.as_tensor(test[:3])
    u, v = ufl.as_vector(trial[3:6]), ufl.as_vector(test[3:6])
    r, w = ufl.as_vector(trial[6:]), ufl.as_vector(test[6:])

    def asym(a):
        """Select the three independent skew components without a dimensional shortcut."""
        return ufl.as_vector((a[1, 2] - a[2, 1], a[2, 0] - a[0, 2], a[0, 1] - a[1, 0]))

    compliance = (sigma - lam / (2 * mu + 3 * lam) * ufl.tr(sigma) * ufl.Identity(3)) / (2 * mu)
    a = (
        ufl.inner(compliance, tau)
        + ufl.inner(ufl.div(tau), u)
        + ufl.inner(ufl.div(sigma), v)
        + ufl.inner(asym(tau), r)
        + ufl.inner(asym(sigma), w)
    ) * dx
    return a, -ufl.inner(force, v) * dx


@pytest.mark.fem
@pytest.mark.parametrize("degree,variable", [(2, False), (2, True), (3, True)])
def test_operator_matches_independent_ufl(degree, variable):
    """Complete BDM stress, DG divergence and three rotation blocks match native UFL on a shear."""
    dolfinx = pytest.importorskip("dolfinx")
    import ufl

    points = np.array([[0.2, -0.1, 0.3], [1.1, 0.1, 0.4], [0.1, 1.2, 0.2], [0.3, 0.2, 1.4]])
    mesh = AffineMixedMesh(points, np.array([[0, 1, 2, 3]]))
    family = HDiv3DFamily("tetrahedron", degree - 1, degree)
    domain, space = native_space(mesh, degree)
    x = ufl.SpatialCoordinate(domain)
    mu = 1 + x[0] if variable else 1
    lam = 2 + x[1] if variable else 2
    xi, weights = cell_quadrature("tetrahedron", degree + 4)
    dx = ufl.Measure(
        "dx",
        domain=domain,
        metadata={
            "quadrature_rule": "custom",
            "quadrature_points": xi,
            "quadrature_weights": weights,
        },
    )
    a, b = forms(space, mu, lam, ufl.as_vector((1 + x[0], 2 - x[1], x[2])), dx)
    independent = from_ufl(a, b, [], np.empty(0, dtype=int))
    basis, _, scalar = hdiv3d_basis(mesh, family, xi)
    physical = mesh.geometry(xi)[0]
    target_vector = basis[0].transpose(0, 2, 1).reshape(-1, family.local_size)
    ns, nu = 3 * family.local_size, 3 * family.pressure_size
    transform = np.zeros((independent.matrix.shape[0], ns + 2 * nu))
    for component in range(9):
        collapsed, mapping = space.sub(component).collapse()
        function = dolfinx.fem.Function(collapsed)
        columns = []
        for column in range(len(function.x.array)):
            function.x.array[:] = 0
            function.x.array[column] = 1
            columns.append(function.eval(physical, np.zeros(len(xi), dtype=np.int32)).ravel())
        native = np.asarray(columns).T
        target = target_vector if component < 3 else scalar
        change = np.linalg.lstsq(native, target, rcond=None)[0]
        assert_allclose(native @ change, target, atol=2e-11, rtol=2e-11)
        indices = (
            3 * np.arange(family.local_size) + component
            if component < 3
            else ns + 3 * np.arange(family.pressure_size) + component - 3
            if component < 6
            else ns + nu + 3 * np.arange(family.pressure_size) + component - 6
        )
        transform[np.ix_(mapping, indices)] = change
    with threadpool_limits(1):
        mass, div, asym, force, *_ = mixed_elasticity_operators_3d(
            mesh,
            family,
            lame_mu=(lambda p: 1 + p[:, 0]) if variable else 1,
            lame_lambda=(lambda p: 2 + p[:, 1]) if variable else 2,
            source=lambda p: np.column_stack((1 + p[:, 0], 2 - p[:, 1], p[:, 2])),
            quadrature_order=degree + 4,
        )
    expected = sparse.bmat([[mass, div.T, asym.T], [div, None, None], [asym, None, None]]).toarray()
    assert_allclose(transform.T @ independent.matrix @ transform, expected, atol=2e-10, rtol=2e-10)
    assert_allclose(
        transform.T @ independent.load,
        np.r_[np.zeros(ns), -force, np.zeros(nu)],
        atol=2e-12,
        rtol=2e-12,
    )


@pytest.mark.fem
@pytest.mark.parametrize("degree", [2, 3])
def test_full_classical_saddle_matches_mhm_full_trace(degree):
    """Full traces reproduce native conforming AFW fields for a nonaffine load."""
    dolfinx = pytest.importorskip("dolfinx")
    import ufl
    from scipy.sparse.linalg import spsolve

    mesh = AffineMixedMesh.unit_cube()
    domain, space = native_space(mesh, degree)
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    scalar = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) * ufl.sin(pi * x[2])
    force = ufl.as_vector(
        (
            6 * pi * pi * scalar,
            -3 * pi * pi * ufl.cos(pi * x[0]) * ufl.cos(pi * x[1]) * ufl.sin(pi * x[2]),
            -3 * pi * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1]) * ufl.cos(pi * x[2]),
        )
    )
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 28})
    a, b = forms(space, 1, 2, force, dx)
    independent = from_ufl(a, b, [], np.empty(0, dtype=int))
    native = dolfinx.fem.Function(space)
    with threadpool_limits(1):
        native.x.array[:] = spsolve(independent.matrix, independent.load)
        native_fields = [native.sub(i).collapse() for i in range(9)]
        assert np.linalg.norm(independent.matrix @ native.x.array - independent.load) < 2e-12

        def physical_force(points):
            """Analytically differentiated three-dimensional linear-elasticity source."""
            sx, sy, sz = np.sin(pi * points).T
            cx, cy, cz = np.cos(pi * points).T
            return (
                pi * pi * np.column_stack((6 * sx * sy * sz, -3 * cx * cy * sz, -3 * cx * sy * cz))
            )

        result = solve_elasticity_mixed_3d(
            mesh,
            stress_degree=degree,
            trace_degree=degree,
            local_refinement=1,
            lame_lambda=2,
            source=physical_force,
            quadrature_order=12,
        )
        xi, _ = cell_quadrature("tetrahedron", degree + 2)
        for cell, fine in enumerate(result.local_meshes):
            points = fine.geometry(xi)[0]
            tree = dolfinx.geometry.bb_tree(domain, 3)
            candidates = dolfinx.geometry.compute_collisions_points(tree, points)
            collisions = dolfinx.geometry.compute_colliding_cells(domain, candidates, points)
            owners = np.array([collisions.links(i)[0] for i in range(len(points))], dtype=np.int32)
            expected = np.column_stack([field.eval(points, owners) for field in native_fields])
            displacement, stress, rotation, _ = result.evaluate(cell, xi)
            assert_allclose(displacement[0], expected[:, 9:12], atol=4e-10, rtol=2e-9)
            assert_allclose(stress[0], expected[:, :9].reshape(-1, 3, 3), atol=4e-10, rtol=2e-9)
            assert_allclose(rotation[0], expected[:, 12:15], atol=4e-10, rtol=2e-9)
