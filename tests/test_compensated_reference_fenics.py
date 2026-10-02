"""Native high-contrast CG2 patch checks the original equations and physical field."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

requires_extended = pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps,
    reason="physical-form accumulation requires a wider long-double type",
)


@pytest.mark.fem
def test_native_compensated_high_contrast_tangential_patch(monkeypatch):
    """A continuous affine field crosses a 1e5 material jump without interface forcing."""
    fem = pytest.importorskip("dolfinx.fem")
    native = pytest.importorskip("dolfinx.fem.petsc")
    mesh = pytest.importorskip("dolfinx.mesh")
    ufl = pytest.importorskip("ufl")
    MPI = pytest.importorskip("mpi4py.MPI")
    PETSc = pytest.importorskip("petsc4py.PETSc")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    arithmetic = importlib.import_module("examples.compensated_reference")
    domain = mesh.create_unit_square(MPI.COMM_SELF, 4, 4)
    space, dg = fem.functionspace(domain, ("Lagrange", 2)), fem.functionspace(domain, ("DG", 0))
    material = fem.Function(dg)
    material.x.array[:] = np.where(dg.tabulate_dof_coordinates()[:, 1] < 0.5, 1.0, 1e5)
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    matrix = native.assemble_matrix(
        fem.form(material * ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx)
    )
    matrix.assemble()
    rhs, value, boundary = matrix.createVecRight(), matrix.createVecRight(), matrix.createVecRight()
    rhs.set(0)
    boundary.set(0)
    nodes = space.tabulate_dof_coordinates()
    exterior = np.any((nodes[:, :2] < 1e-13) | (nodes[:, :2] > 1 - 1e-13), axis=1)
    fixed = np.flatnonzero(exterior).astype(PETSc.IntType)
    boundary.array[fixed] = 1 + nodes[fixed, 0]
    matrix.zeroRowsColumns(fixed, diag=1, x=boundary, b=rhs)
    b = rhs.array.copy()
    pointers, columns, data = matrix.getValuesCSR()
    csr = sparse.csr_matrix((data, columns, pointers), shape=matrix.getSize())
    ksp = PETSc.KSP().create(MPI.COMM_SELF)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("cholesky")
    ksp.getPC().setFactorSolverType("mumps")

    def solve(forcing):
        """Use the same native factorization for each original residual correction."""
        rhs.array[:] = forcing
        ksp.solve(rhs, value)
        return value.array.copy()

    try:
        high, low, history = arithmetic.refine_components(csr, b, solve)
        assert history[-1] < 1e-10
        assert_allclose(high.astype(np.longdouble) + low, 1 + nodes[:, 0], atol=2e-13, rtol=0)
        assert np.linalg.norm(
            arithmetic.component_residual(csr, b, high, low)
        ) <= 1e-10 * np.linalg.norm(b)
    finally:
        for native_object in (ksp, boundary, value, rhs, matrix):
            native_object.destroy()


@pytest.mark.fem
@requires_extended
def test_native_graded_inclusion_reference_quadratic_patch(monkeypatch):
    """Strong nonzero Dirichlet data and source reproduce a quadratic on graded CG2 cells."""
    pytest.importorskip("dolfinx")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    reference = importlib.import_module("examples.solve_pgmhm_inclusions_reference")
    field, report = reference.solve(1, patch=True, graded=True)
    points = np.array([[0.004, 0.03], [0.06, 0.92], [0.6, 0.4], [0.97, 0.985]])
    pressure, gradient, flux = field.evaluate(points)
    expected_gradient = np.column_stack((1 - 2 * points[:, 0], np.zeros(len(points))))
    assert_allclose(pressure, points[:, 0] * (1 - points[:, 0]), atol=2e-13, rtol=0)
    assert_allclose(gradient, expected_gradient, atol=2e-12, rtol=0)
    assert_allclose(flux, -2 * expected_gradient, atol=4e-12, rtol=0)
    assert report["residual"] < 1e-10
    assert report["mesh_family"] == "interface-graded"


@pytest.mark.fem
@requires_extended
def test_native_physical_moments_match_ufl_variable_material(monkeypatch):
    """Independent UFL verifies all CG2 operator rows on nonuniform fitted rectangles."""
    fem = pytest.importorskip("dolfinx.fem")
    native = pytest.importorskip("dolfinx.fem.petsc")
    mesh = pytest.importorskip("dolfinx.mesh")
    ufl = pytest.importorskip("ufl")
    MPI = pytest.importorskip("mpi4py.MPI")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    owner = importlib.import_module("examples.cg2_physical_form")
    reference = importlib.import_module("examples.solve_unusual_spe10_reference")
    x, y = np.array([0.0, 0.015625, 0.25, 1.0]), np.array([0.0, 0.25, 1.0])
    domain = mesh.create_unit_square(MPI.COMM_SELF, 3, 2, diagonal=mesh.DiagonalType.right)
    domain.geometry.x[:, 0] = x[np.rint(3 * domain.geometry.x[:, 0]).astype(int)]
    domain.geometry.x[:, 1] = y[np.rint(2 * domain.geometry.x[:, 1]).astype(int)]
    space, dg = fem.functionspace(domain, ("Lagrange", 2)), fem.functionspace(domain, ("DG", 0))
    material = fem.Function(dg)
    centers = dg.tabulate_dof_coordinates()
    material.x.array[:] = np.where(centers[:, 1] < 0.25, 1.0, 1e5)
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    measure = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
    matrix = native.assemble_matrix(
        fem.form(material * ufl.inner(ufl.grad(trial), ufl.grad(test)) * measure)
    )
    matrix.assemble()
    forcing = native.assemble_vector(fem.form(7 * test * measure))
    nodes = space.tabulate_dof_coordinates()[:, :2]
    gx, gy = reference.subdivide_axis(x, 2), reference.subdivide_axis(y, 2)
    ix = np.argmin(abs(nodes[:, 0, None] - gx), axis=1)
    iy = np.argmin(abs(nodes[:, 1, None] - gy), axis=1)
    form = owner.CG2DiffusionForm(x, y, np.array([[1.0] * 3, [1e5] * 3]), source=7)
    values = np.random.default_rng(7).normal(size=form.shape)
    vector, action = matrix.createVecRight(), matrix.createVecLeft()
    vector.array[:] = values[iy, ix]
    try:
        matrix.mult(vector, action)
        expected = form.action(values)[iy, ix]
        assert np.linalg.norm(expected - action.array) < 8e-15 * np.linalg.norm(expected)
        assert np.linalg.norm(form.load()[iy, ix] - forcing.array) < (
            16 * np.finfo(float).eps * np.linalg.norm(forcing.array)
        )
        # A continuous field tangential to the jump has no interface source.
        xx, _ = np.meshgrid(gx, gy)
        tangential = form.action(xx)
        assert_allclose(tangential[1:-1, 1:-1], 0, atol=4e-13, rtol=0)
        assert np.count_nonzero(form.action(np.ones(form.shape))) == 0
    finally:
        for obj in (vector, action, forcing, matrix):
            obj.destroy()
