"""RT0/RT1/RT2 MHM, classical mixed references and conservation contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh, solve_darcy
from pymhm.darcy_rt import solve_darcy_rt, solve_darcy_rt_conforming
from pymhm.elements import triangle_quadrature
from pymhm.reservoir import CartesianCellField
from pymhm.rt import rt_evaluate


def affine(x):
    return 1 + x[:, 0] + 2 * x[:, 1]


@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("classical", [False, True])
def test_rt_polynomial_flux_and_full_pressure_moments(degree, classical):
    m = degree
    mesh = TriangleMesh.unit_square(2)

    def p(x):
        return np.sum(x ** (m + 1), axis=1)

    def q(x):
        return -(m + 1) * x**m

    f = 0.0 if m == 0 else lambda x: -m * (m + 1) * np.sum(x ** (m - 1), axis=1)
    solve = solve_darcy_rt_conforming if classical else solve_darcy_rt
    result = solve(mesh, degree=m, source=f, dirichlet=p)
    assert result.flux_l2_error(q) < 3e-12
    assert result.divergence_l2_error(f) < 2e-10
    assert result.l2_error(p) > 0
    assert max(np.max(np.abs(v)) for v in result.fine_equilibrium_residuals()) < 3e-12
    assert np.max(abs(result.conservation_residuals())) < 3e-12
    if classical:
        assert result.normal_flux_residuals() == ()
    else:
        assert max(np.max(np.abs(v)) for v in result.normal_flux_residuals()) < 1e-13


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_rt_constant_pressure_and_anisotropic_affine_patch(degree):
    mesh = TriangleMesh.unit_square(2)
    K = np.array([[2.0, 0.3], [0.3, 1.0]])
    a = solve_darcy_rt(mesh, degree=degree, permeability=K, dirichlet=affine)
    assert a.flux_l2_error(-K @ np.array([1.0, 2.0])) < 3e-12
    if degree:
        assert a.l2_error(affine) < 1e-12
    if degree == 0:
        old = solve_darcy(
            mesh, formulation="mixed", local_refinement=2, permeability=K, dirichlet=affine
        )
        assert_allclose(a.flux, old.flux, atol=1e-12)
        for actual, expected in zip(a.pressure, old.pressure, strict=True):
            assert_allclose(actual[:, 0], expected, atol=1e-12)
    constant = solve_darcy_rt(mesh, degree=degree, dirichlet=3.0)
    assert constant.l2_error(3.0) < 1e-12


@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("classical", [False, True])
def test_rt_neumann_physical_mean_and_incompatible_source(degree, classical):
    mesh = TriangleMesh.unit_square(1)
    q = -np.array([1.0, 2.0])
    neumann = {int(face): float(q @ mesh.normals[face]) for face in mesh.boundary_faces}
    solve = solve_darcy_rt_conforming if classical else solve_darcy_rt
    actual = solve(mesh, degree=degree, neumann=neumann, mean_pressure=2.5)
    assert actual.flux_l2_error(q) < 3e-12
    if degree:
        assert actual.l2_error(affine) < 1e-12
    partial = {next(iter(neumann)): next(iter(neumann.values()))}
    mixed = solve(mesh, degree=degree, neumann=partial, dirichlet=affine)
    assert mixed.flux_l2_error(q) < 3e-12
    with pytest.raises(ValueError, match="incompatible"):
        solve(
            mesh,
            degree=degree,
            neumann=dict.fromkeys(map(int, mesh.boundary_faces), 0.0),
            source=1.0,
        )


@pytest.mark.parametrize("continuous", [False, True])
def test_rt_enriched_trace_partitions_and_cartesian_material(continuous):
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(1, 2, continuous=continuous) for _ in mesh.faces)
    )
    field = CartesianCellField(np.array([[2.0], [1.0]]), (0.5, 1.0))

    def p(x):
        return np.where(x[:, 0] <= 0.5, x[:, 0] / 2, x[:, 0] - 0.25)

    solution = solve_darcy_rt(
        mesh, degree=2, skeleton=skeleton, local_refinement=2, permeability=field, dirichlet=p
    )
    assert solution.flux_l2_error([-1.0, 0.0]) < 2e-12
    assert solution.l2_error(p) < 1e-12
    assert max(np.max(abs(x)) for x in solution.normal_flux_residuals()) < 1e-13


def test_rt_input_contracts():
    mesh = TriangleMesh.unit_square(1)
    for skeleton in (SkeletonSpace(mesh, components=2), SkeletonSpace(TriangleMesh.unit_square(1))):
        with pytest.raises(ValueError, match="scalar skeleton"):
            solve_darcy_rt(mesh, skeleton=skeleton)
    for space in (FaceSpace.uniform(2), FaceSpace.uniform(0, 3)):
        with pytest.raises(ValueError, match="exceed m|not aligned"):
            solve_darcy_rt(
                mesh, degree=1, skeleton=SkeletonSpace(mesh, tuple(space for _ in mesh.faces))
            )
    interior = int(np.flatnonzero(mesh.face_cells[:, 1] >= 0)[0])
    with pytest.raises(ValueError, match="external"):
        solve_darcy_rt_conforming(mesh, neumann={interior: 0})


def test_classical_rt_neumann_geometry_is_evaluated_once(monkeypatch):
    """Normal-flux boundary moments reuse face lengths without changing the fields."""
    mesh = TriangleMesh.unit_square(3, 5)
    q = -np.array([1.0, 2.0])
    normals = mesh.normals
    neumann = {int(face): float(q @ normals[face]) for face in mesh.boundary_faces}
    baseline = solve_darcy_rt_conforming(mesh, degree=2, neumann=neumann, mean_pressure=2.5)
    original = TriangleMesh.lengths.fget
    calls = []

    def counted(instance):
        """Record each full face-length construction with its original floating arithmetic."""
        calls.append(instance)
        return original(instance)

    monkeypatch.setattr(TriangleMesh, "lengths", property(counted))
    actual = solve_darcy_rt_conforming(mesh, degree=2, neumann=neumann, mean_pressure=2.5)
    assert len(calls) == 1
    assert np.array_equal(actual.pressure, baseline.pressure)
    assert np.array_equal(actual.flux, baseline.flux)
    assert actual.flux_l2_error(q) < 3e-12
    assert actual.l2_error(affine) < 2e-12


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_rt_parallel_factory_fields(backend):
    mesh = TriangleMesh.unit_square(1)
    baseline = solve_darcy_rt(mesh, degree=1, dirichlet=affine)
    actual = solve_darcy_rt(mesh, degree=1, dirichlet=affine, backend=backend, workers=2)
    assert_allclose(actual.flux, baseline.flux, atol=1e-13)
    assert_allclose(actual.pressure, baseline.pressure, atol=1e-13)


@pytest.mark.fem
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_native_dolfinx_classical_rt_fields(degree):
    """Compare physical fields against independent mixed UFL assembly on identical triangles."""
    dolfinx = pytest.importorskip("dolfinx")
    ufl = pytest.importorskip("ufl")
    basix = pytest.importorskip("basix.ufl")
    mpi = pytest.importorskip("mpi4py.MPI")
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree

    mesh = TriangleMesh.unit_square(2)
    geometry = ufl.Mesh(basix.element("Lagrange", "triangle", 1, shape=(2,)))
    domain = dolfinx.mesh.create_mesh(mpi.COMM_SELF, mesh.cells, mesh.points, geometry)
    element = basix.mixed_element(
        [basix.element("RT", "triangle", degree + 1), basix.element("DG", "triangle", degree)]
    )
    space = dolfinx.fem.functionspace(domain, element)
    q, p = ufl.TrialFunctions(space)
    v, w = ufl.TestFunctions(space)
    x = ufl.SpatialCoordinate(domain)
    normal = ufl.FacetNormal(domain)
    pressure = x[0] ** (degree + 1) + x[1] ** (degree + 1)
    source = (
        0 if degree == 0 else -degree * (degree + 1) * (x[0] ** (degree - 1) + x[1] ** (degree - 1))
    )
    a = (ufl.inner(q, v) - p * ufl.div(v) - w * ufl.div(q)) * ufl.dx
    L = -source * w * ufl.dx(domain=domain) - pressure * ufl.dot(v, normal) * ufl.ds
    A = dolfinx.fem.assemble_matrix(dolfinx.fem.form(a))
    A.scatter_reverse()
    rhs = dolfinx.fem.assemble_vector(dolfinx.fem.form(L)).array
    function = dolfinx.fem.Function(space)
    function.x.array[:] = spsolve(A.to_scipy().tocsc(), rhs)
    numerical = solve_darcy_rt_conforming(
        mesh,
        degree=degree,
        source=0.0
        if degree == 0
        else lambda x: -degree * (degree + 1) * np.sum(x ** (degree - 1), axis=1),
        dirichlet=lambda x: np.sum(x ** (degree + 1), axis=1),
    )
    bary, _ = triangle_quadrature(degree + 3)
    native_centers = dolfinx.mesh.compute_midpoints(
        domain, 2, np.arange(len(mesh.cells), dtype=np.int32)
    )[:, :2]
    distance, cells = cKDTree(native_centers).query(mesh.points[mesh.cells].mean(axis=1))
    assert distance.max() < 1e-12
    points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
    padded = np.column_stack((points.reshape(-1, 2), np.zeros(points.size // 2)))
    indices = np.repeat(cells, len(bary)).astype(np.int32)
    q_native = function.sub(0).eval(padded, indices).reshape(*points.shape)
    p_native = function.sub(1).eval(padded, indices).reshape(points.shape[:2])
    from pymhm.darcy_rt import pressure_basis

    q_actual = rt_evaluate(mesh, numerical.flux[0], degree, bary)[0]
    p_actual = numerical.pressure[0] @ pressure_basis(degree, bary).T
    assert_allclose(q_actual, q_native, atol=4e-12, rtol=2e-12)
    assert_allclose(p_actual, p_native, atol=2e-12, rtol=2e-12)
