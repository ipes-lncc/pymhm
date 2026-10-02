"""Physical curl, tangential orientation, staggered energy and Maxwell solver contracts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve

from pymhm.darcy3d import TriangularSkeleton
from pymhm.maxwell import MaxwellStepper, field_values, solve_maxwell
from pymhm.maxwell_dg import (
    MaxwellSkeleton,
    assemble_local,
    derivatives,
    physical_basis,
    physical_points,
    quadrature,
)
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.planar_material import PlanarMaterial, PlanarRegion
from pymhm.quadrilateral import CartesianMacroMesh, qk_basis
from pymhm.reservoir import CartesianCellField
from pymhm.solvers import LinearSolveError
from pymhm.tetrahedral import TetraMesh


@pytest.mark.parametrize("amplitude", [1e-20, 1.0, 1e20])
@pytest.mark.parametrize("kind", ["triangle", "rectangle", "tetrahedron"])
def test_stationary_pec_magnetic_field_uses_uncancelled_constraint_scale(
    kind, amplitude, monkeypatch
):
    """Physical kick terms set the residual scale when the exact electric field is zero."""
    dimension = 3 if kind == "tetrahedron" else 2
    mesh = (
        TetraMesh.unit_cube()
        if dimension == 3
        else CartesianMacroMesh(2)
        if kind == "rectangle"
        else TriangleMesh.unit_square()
    )
    magnetic = amplitude * np.arange(1, dimension + 1) / 5
    with MaxwellStepper(mesh, time_step=0.001, degree=2, local_refinement=2) as stepper:
        initial = stepper.initialize(magnetic=magnetic)
        for _ in range(3):
            result = stepper.advance()
        assert max(result.l2_errors(0, magnetic)) < 1e-12 * amplitude
        assert abs(result.energy - initial.energy) < 1e-12 * amplitude**2
        original = stepper.factor.solve

        def incorrect_trace(rhs):
            """Introduce a relative defect that remains far above the unchanged tolerance."""
            return original(rhs) * (1 + 1e-6)

        monkeypatch.setattr(stepper.factor, "solve", incorrect_trace)
        with pytest.raises(LinearSolveError, match="tangential"):
            stepper.advance()


@pytest.mark.parametrize("kind", ["triangle", "rectangle", "tetrahedron"])
def test_constant_fields_and_absorbing_orientation(kind):
    """Nonzero E/H patches check outgoing H cross n, source signs and tangent frames."""
    dimension = 3 if kind == "tetrahedron" else 2
    mesh = (
        TetraMesh.unit_cube()
        if dimension == 3
        else CartesianMacroMesh(2, 1, (-1, 2, 0, 2))
        if kind == "rectangle"
        else TriangleMesh.unit_square(1)
    )
    e = np.array([2.0]) if dimension == 2 else np.array([1.2, -0.3, 0.7])
    h = np.array([3.0, 4.0]) if dimension == 2 else np.array([0.4, 0.8, -0.2])

    def boundary(time, points, normals):
        """Physical E_tan-alpha*(H cross n), with alpha=1 and constant exact fields."""
        cross = (
            h[0] * normals[:, 1] - h[1] * normals[:, 0] if dimension == 2 else np.cross(h, normals)
        )
        return e[0] - cross if dimension == 2 else e - cross

    with MaxwellStepper(mesh, time_step=0.001, absorbing=1, boundary_data=boundary) as stepper:
        initial = stepper.initialize(e, h)
        for _ in range(4):
            result = stepper.advance()
            assert abs(result.energy_balance_residual) < 2e-14
        errors = result.l2_errors(e, h)
        assert max(errors) < 2e-13
        assert_allclose(result.energy, initial.energy, atol=2e-14)
        assert result.electric_time == pytest.approx(0.0045)
        assert result.magnetic_time == pytest.approx(0.004)
        for cell in range(len(mesh.cells)):
            reference = (
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
                if kind == "rectangle"
                else np.eye(dimension + 1)
            )
            _, actual_e, actual_h = result.sample(cell, reference)
            assert_allclose(actual_e, np.broadcast_to(e, actual_e.shape), atol=3e-13)
            assert_allclose(actual_h, np.broadcast_to(h, actual_h.shape), atol=3e-13)


@pytest.mark.parametrize("dimension", [2, 3])
def test_nonzero_pec_energy_and_cfl(dimension):
    """Leapfrog preserves its actual cross-time energy; unstable time steps are rejected."""
    mesh = TriangleMesh.unit_square(1) if dimension == 2 else TetraMesh.unit_cube()

    def electric(points):
        """Smooth PEC-compatible out-of-plane electric field in either dimension."""
        u = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
        return u if dimension == 2 else np.column_stack((u * 0, u * 0, u))

    with MaxwellStepper(mesh, time_step=0.001) as stepper:
        initial = stepper.initialize(electric)
        for _ in range(12):
            result = stepper.advance()
            assert_allclose(result.energy, initial.energy, atol=3e-15)
            assert abs(result.energy_balance_residual) < 3e-15
            assert np.linalg.norm(stepper._moments(result.electric)) < 2e-14
        assert result.energy > 0.1
        assert result.l2_errors(0, 0)[1] > 0.01
        bound = stepper.frequency_bound
    with pytest.raises(ValueError, match="CFL"):
        MaxwellStepper(mesh, time_step=2.01 / bound)


@pytest.mark.parametrize("rectangle", [False, True])
def test_midpoint_impedance_uncondensed_saddle_and_dissipation(rectangle):
    """One time step agrees with a separately assembled full electric/trace saddle."""
    mesh = CartesianMacroMesh(2, 1) if rectangle else TriangleMesh.unit_square(1)
    with MaxwellStepper(mesh, time_step=0.003, absorbing=2.0) as stepper:
        stepper.initialize(lambda p: np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
        eold = np.concatenate(stepper.electric)
        dt = stepper.time_step
        local_h = tuple(
            h + dt * inverse.solve(local.curl @ e)
            for local, inverse, e, h in zip(
                stepper.locals,
                stepper.magnetic_factors,
                stepper.electric,
                stepper.magnetic,
                strict=True,
            )
        )
        mass = sparse.block_diag([local.electric_mass for local in stepper.locals], format="csc")
        coupling = sparse.lil_matrix((len(eold), stepper.skeleton.size))
        offset = 0
        rhs = []
        for local, e, h in zip(stepper.locals, stepper.electric, local_h, strict=True):
            ids = offset + np.arange(len(e))
            coupling[np.ix_(ids, local.trace_dofs)] = local.coupling
            rhs.append(local.electric_mass @ e - dt * local.curl.T @ h)
            offset += len(e)
        coupling = coupling.tocsc()
        matrix = sparse.bmat(
            [[mass, dt * coupling], [coupling.T, -2 * stepper.impedance]], format="csc"
        )
        full = spsolve(matrix, np.r_[np.concatenate(rhs), -coupling.T @ eold])
        energy = stepper.modified_energy()
        result = stepper.advance()
        assert_allclose(np.concatenate(result.electric), full[: len(eold)], atol=4e-14)
        assert_allclose(result.trace, full[len(eold) :], atol=2e-13)
        assert result.energy < energy
        assert abs(result.energy_balance_residual) < 1e-15


@pytest.mark.parametrize("dimension", [2, 3])
def test_central_dg_derivative_and_material_mass(dimension):
    """Affine derivatives are exact on discontinuous submeshes and plane-cut masses are SPD."""
    macro = TriangleMesh.unit_square(1) if dimension == 2 else TetraMesh.unit_cube()
    fine = macro.submesh(0, 2)
    base = SkeletonSpace(macro) if dimension == 2 else TriangularSkeleton(macro)
    skeleton = MaxwellSkeleton(base)
    material = PlanarMaterial(1, (PlanarRegion([np.ones(dimension)], [0.51], 4),))
    local = assemble_local(fine, skeleton, 0, 1, material, material, 5)
    vertex_fields = fine.points[fine.cells]
    # Degree-one cardinal node order agrees with the simplex vertex order.
    derivative = derivatives(fine, 1, 5)
    scalar_mass = local.magnetic_mass[::dimension, ::dimension]
    _, weights, _ = quadrature(fine, 1, 5)
    bary, _, _ = quadrature(fine, 1, 5)
    basis, _ = physical_basis(fine, 1, bary)
    one_load = np.einsum("tq,tqi->ti", weights, basis).ravel()
    for axis, matrix in enumerate(derivative):
        assert_allclose(matrix @ vertex_fields[..., axis].ravel(), one_load, atol=2e-15)
        assert_allclose(matrix @ np.ones(len(one_load)), 0, atol=2e-15)
    assert np.linalg.eigvalsh(scalar_mass.toarray()).min() > 0
    assert local.frequency_bound() > 0


def test_contracts_resources_callback_and_forcing(monkeypatch):
    """Reject malformed states and release reusable native resources on every exit."""
    mesh = TriangleMesh.unit_square(1)
    for value in (1j, np.nan, [1, 2, 3]):
        with pytest.raises(ValueError):
            field_values(value, np.ones((2, 2)), 2)
    with pytest.raises(TypeError):
        MaxwellSkeleton(None)
    with pytest.raises(ValueError):
        MaxwellSkeleton(SkeletonSpace(mesh, components=2))
    for options in (
        {"time_step": 0},
        {"time_step": 1j},
        {"absorbing": {999: 1}},
        {"skeleton": MaxwellSkeleton(SkeletonSpace(TriangleMesh.unit_square(1)))},
    ):
        with pytest.raises(ValueError):
            MaxwellStepper(mesh, **({"time_step": 0.001} | options))
    with pytest.raises(TypeError):
        MaxwellStepper(None, time_step=0.1)
    with MaxwellStepper(mesh, time_step=0.001) as stepper:
        with pytest.raises(RuntimeError):
            stepper.advance()
        with pytest.raises(RuntimeError):
            stepper.solution()
        result = stepper.initialize()
        assert result.energy == 0
        result = stepper.advance(lambda time, points: time + points[:, 0])
        assert result.energy > 0
        assert abs(result.energy_balance_residual) < 1e-17
        for bary in ([[0, 0]], [[1j, 0, 0]], [[-1, 1, 1]], [[0.1, 0.1, 0.1]]):
            with pytest.raises(ValueError):
                result.sample(0, bary)
        with pytest.raises(ValueError):
            result.sample(99, [[1, 0, 0]])
        monkeypatch.setattr(stepper.factor, "solve", lambda rhs: np.ones_like(rhs) * 100)
        with pytest.raises(LinearSolveError, match="tangential"):
            stepper.advance()
    stepper.close()
    with pytest.raises(RuntimeError):
        stepper.__enter__()
    with pytest.raises(RuntimeError):
        stepper.initialize()
    seen = []
    solve_maxwell(mesh, time_step=0.001, steps=1, on_step=seen.append)
    assert len(seen) == 2


def test_tangential_partition_and_incompatible_trace():
    """The trace remains tangential and a too-rich local mass coupling is rejected."""
    mesh = TetraMesh.unit_cube()
    skeleton = MaxwellSkeleton(TriangularSkeleton(mesh, subdivisions=2, degree=0))
    for normal, frame in zip(mesh.normals, skeleton.frames, strict=True):
        assert_allclose(frame.T @ frame, np.eye(2), atol=4e-16)
        assert_allclose(normal @ frame, 0, atol=4e-16)
    with pytest.raises(ValueError, match="align"):
        assemble_local(mesh.submesh(0, 1), skeleton, 0, 2, 1, 1, 4)
    vertices = np.eye(3)
    center = np.ones(3) / 3
    partition = np.array([[vertices[j], vertices[(j + 1) % 3], center] for j in range(3)])
    centered = MaxwellSkeleton(
        TriangularSkeleton(mesh, face_partitions=tuple(partition for _ in mesh.faces))
    )
    with pytest.raises(ValueError, match="align"):
        assemble_local(mesh.submesh(0, 1), centered, 0, 2, 1, 1, 4)
    with pytest.raises(LinearSolveError):
        MaxwellStepper(TriangleMesh.unit_square(1), time_step=0.001, degree=1)
    triangle = TriangleMesh.unit_square(1)
    segmented = MaxwellSkeleton(
        SkeletonSpace(triangle, tuple(FaceSpace.uniform(0, 2) for _ in triangle.faces))
    )
    with MaxwellStepper(
        triangle, time_step=0.001, skeleton=segmented, local_refinement=2
    ) as stepper:
        assert stepper.initialize().energy == 0


def test_single_absorbing_macro_and_optional_callback():
    """One absorbing macro has no initial PEC/interface constraints to project."""
    mesh = TriangleMesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]])
    result = solve_maxwell(mesh, time_step=0.001, steps=1, absorbing=lambda p: np.ones(len(p)))
    assert result.energy == 0
    with pytest.raises(ValueError, match="positive"):
        MaxwellStepper(mesh, time_step=0.001, absorbing=-1)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_cartesian_dg_polynomials_and_interface_mass(degree):
    """Tensor derivatives, cut masses and independently segmented faces retain moments."""
    macro = CartesianMacroMesh(1)
    fine = macro.submesh(0, 3)
    skeleton = MaxwellSkeleton(
        SkeletonSpace(macro, tuple(FaceSpace.uniform(1, 3) for _ in macro.faces))
    )
    field = CartesianCellField(np.array([[1.0], [4.0]]), (0.5, 1.0))
    planar = PlanarMaterial(1, (PlanarRegion([[1, 1]], [0.51], 4),))
    reference = np.array(
        [(x, y) for y in np.linspace(0, 1, degree + 1) for x in np.linspace(0, 1, degree + 1)]
    )
    nodes = physical_points(fine, reference)
    bary, weights, _ = quadrature(fine, 1, 7)
    basis, _ = physical_basis(fine, degree, bary)
    points = physical_points(fine, bary)
    p = (nodes[..., 0] ** degree * nodes[..., 1] ** degree).ravel()
    for axis, matrix in enumerate(derivatives(fine, degree, 7)):
        exact = degree * points[..., axis] ** (degree - 1) * points[..., 1 - axis] ** degree
        load = np.einsum("tq,tqi,tq->ti", weights, basis, exact).ravel()
        assert_allclose(matrix @ p, load, atol=6e-15)
    for material, integral in ((field, 2.5), (planar, 1 + 3 * 0.51**2 / 2)):
        local = assemble_local(fine, skeleton, 0, degree, material, 1, 7)
        assert np.sum(local.electric_blocks) == pytest.approx(integral, abs=3e-14)
    identity, _ = qk_basis(degree, reference)
    assert_allclose(identity, np.eye(len(reference)), atol=2e-14)
    with MaxwellStepper(macro, time_step=0.001, degree=2, local_refinement=2) as stepper:
        result = stepper.initialize()
        with pytest.raises(ValueError, match="coordinates"):
            result.sample(0, [[1.1, 0]])


def test_single_q2_with_four_linear_face_traces_has_one_redundant_constraint():
    """The single-square Q2/P1 trace pair has rank seven, not eight."""
    macro = CartesianMacroMesh(1)
    skeleton = MaxwellSkeleton(
        SkeletonSpace(macro, tuple(FaceSpace.uniform(1) for _ in macro.faces))
    )
    local = assemble_local(macro, skeleton, 0, 2, 1, 1, 5)
    assert np.linalg.matrix_rank(local.coupling.toarray(), tol=1e-13) == 7
    with pytest.raises(LinearSolveError, match="rank"):
        MaxwellStepper(macro, time_step=0.001, degree=2)


@pytest.mark.parametrize("kind", ["triangle", "rectangle", "tetrahedron"])
def test_weighted_gram_cfl_bound_exceeds_full_spectrum(kind):
    """The positive-vector bound is compared with every singular value, not Ritz data."""
    macro = (
        TetraMesh.unit_cube()
        if kind == "tetrahedron"
        else CartesianMacroMesh(2)
        if kind == "rectangle"
        else TriangleMesh.unit_square(1)
    )
    base = TriangularSkeleton(macro) if kind == "tetrahedron" else SkeletonSpace(macro)
    local = assemble_local(macro.submesh(0, 2), MaxwellSkeleton(base), 0, 2, 1.7, 2.3, 5)
    le = np.linalg.cholesky(local.electric_mass.toarray())
    lh = np.linalg.cholesky(local.magnetic_mass.toarray())
    scaled = solve_triangular(lh, local.curl.toarray(), lower=True)
    scaled = solve_triangular(le, scaled.T, lower=True).T
    exact = np.linalg.svd(scaled, compute_uv=False)[0]
    assert local.frequency_bound() >= exact
    assert local.frequency_bound() <= np.sqrt(
        np.linalg.norm(scaled, 1) * np.linalg.norm(scaled, np.inf)
    ) * (1 + 1e-10)
