"""Shared triangular coefficients preserve physical consumers and basis replay."""

import pickle
from math import factorial, prod

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm._legacy.models.darcy.primal_3d import solve_darcy_3d
from pymhm._legacy.models.elasticity.primal_3d import Elasticity3DSolution
from pymhm._legacy.models.flow.solver_3d import _translation_trace_compatible
from pymhm._legacy.models.waves.maxwell import MaxwellStepper
from pymhm.core.contracts import HybridSolution
from pymhm.core.spaces import bind_interface
from pymhm.estimators.darcy import estimate_darcy_error
from pymhm.estimators.darcy_3d import estimate_darcy_error_3d
from pymhm.estimators.darcy_energy import estimate_darcy_indicator
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.fem.vector.curl import (
    TangentialTraceSpace,
    tangential_load,
    tangential_mass,
    trace_coupling,
)
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.methods.hho_3d import _face_rule, solve_mshho_3d
from pymhm.methods.three_field_3d import solve_mh2m_3d
from pymhm.recovery.moments_3d import reconstruct_darcy_moments_3d


@pytest.fixture(autouse=True)
def bounded_threads():
    """Bound native resources for small local reconstructions."""
    with threadpool_limits(1):
        yield


def two_tetrahedra():
    """Return two incident cells with a shared oblique triangular macroface."""
    cube = TetraMesh.unit_cube()
    return TetraMesh(cube.points, cube.cells[:2])


@pytest.mark.parametrize("mixed", [False, True])
def test_shared_flux_coefficients_preserve_reconstruction_and_replay(mixed):
    """A nonzero-source tensor patch checks physical moments and persisted coordinates."""
    mesh = two_tetrahedra()
    flags = np.arange(len(mesh.faces)) % 2 == 0 if mixed else True
    skeleton = TriangularSkeleton(mesh, 2, degree=1, continuous=flags)
    tensor = np.array([[3.0, 0.2, 0.1], [0.2, 2.0, 0.3], [0.1, 0.3, 1.0]])

    def pressure(points):
        """Prescribe a quadratic pressure with nonhomogeneous exterior data."""
        return 1 + np.sum(points**2, axis=1)

    def flux(points):
        """Differentiate the pressure independently and apply the Darcy sign."""
        return -2 * points @ tensor

    face = int(mesh.boundary_faces[0])
    natural = {face: lambda points: flux(points) @ mesh.normals[face]}
    solution = solve_darcy_3d(
        mesh,
        skeleton=skeleton,
        permeability=tensor,
        degree=3,
        local_refinement=2,
        source=-2 * np.trace(tensor),
        dirichlet=pressure,
        neumann=natural,
    )
    assert solution.l2_error(pressure) < 2e-11
    assert_allclose(solution.conservation_residuals(), 0, atol=2e-12, rtol=1e-10)
    recovered = reconstruct_darcy_moments_3d(solution, degree=1)
    assert recovered.flux_l2_error(flux) < 3e-10
    assert max(np.max(abs(row)) for row in recovered.continuous_moment_residuals()) < 2e-12
    archive = pickle.dumps((skeleton, solution.hybrid.trace, recovered.family.coefficients))
    points = np.array([[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]])
    original = skeleton.evaluate(face, points) @ solution.hybrid.trace[skeleton.dofs(face)]
    interface = bind_interface(skeleton, convention="normal")
    expected_digest = interface.binding(0).basis_digest
    for threads in (1, 2):
        with threadpool_limits(threads):
            replay, coefficients, rt_basis = pickle.loads(archive)
            assert (
                bind_interface(replay, convention="normal").binding(0).basis_digest
                == expected_digest
            )
            assert_array_equal(
                replay.evaluate(face, points) @ coefficients[replay.dofs(face)], original
            )
            actual = recovered.family.tabulate(points, coefficients=rt_basis)
            expected = recovered.family.tabulate(points)
            for current, stored in zip(actual, expected, strict=True):
                assert_allclose(current, stored, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("subdivisions", [1, 2])
def test_estimator_theorem_admits_single_face_and_excludes_subdivided_c0(dimension, subdivisions):
    """Independent subface tests, rather than numerical rank, determine this hypothesis."""
    if dimension == 2:
        mesh = TriangleMesh.unit_square()
        skeleton = SkeletonSpace(
            mesh,
            tuple(FaceSpace.uniform(1, subdivisions, continuous=True) for _ in mesh.faces),
        )
        solution = solve_darcy(mesh, skeleton=skeleton, degree=3, local_refinement=2)

        def estimate():
            """Evaluate the published identity-diffusion estimator."""
            return estimate_darcy_error(solution, homogeneous_dirichlet=True)

        def weighted():
            """Exercise the second public owner of the same theorem hypotheses."""
            return estimate_darcy_indicator(solution)

        estimates = (estimate, weighted)
    else:
        mesh = two_tetrahedra()
        skeleton = TriangularSkeleton(mesh, subdivisions, degree=1, continuous=True)
        solution = solve_darcy_3d(mesh, skeleton=skeleton, degree=4, local_refinement=2)

        def estimate():
            """Evaluate the tetrahedral counterpart under its dimension-specific degree."""
            return estimate_darcy_error_3d(solution)

        estimates = (estimate,)
    for evaluate in estimates:
        if subdivisions == 1:
            assert_allclose(evaluate().total, 0, atol=1e-12, rtol=1e-10)
        else:
            with pytest.raises(ValueError, match="independent polynomials on each subface"):
                evaluate()


def test_moment_face_rules_and_scalar_reconstruction_use_the_shared_basis():
    """MsHHO moments reproduce an affine field using C0 triangular face test spaces."""
    mesh = two_tetrahedra()
    skeleton = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    points, weights, basis = _face_rule(mesh, skeleton, 0, 4)
    assert basis.shape == (len(points), len(skeleton.dofs(0)))
    assert_allclose(basis.sum(axis=1), 1, atol=1e-12, rtol=1e-10)
    assert_allclose(weights.sum(), mesh.areas[0], atol=1e-12, rtol=1e-10)
    result = solve_mshho_3d(
        mesh,
        skeleton=skeleton,
        degree=3,
        local_refinement=2,
        dirichlet=lambda x: 1 + x @ np.array([1.0, 2.0, -1.0]),
    )
    assert result.l2_error(lambda x: 1 + x @ np.array([1.0, 2.0, -1.0])) < 2e-11
    assert result.flux_l2_error([-1.0, -2.0, 1.0]) < 2e-10


def test_elastic_force_and_torque_use_shared_face_coefficients():
    """Independent Gauss identities check affine traction and all six rigid moments."""
    mesh = two_tetrahedra()
    skeleton = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    tensor = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 3.0]])
    gradient = np.array([1.0, 2.0, -0.5])
    trace = np.zeros((skeleton.size, 3))
    for face in range(len(mesh.faces)):
        for segment, partition in enumerate(skeleton.face_partition(face)):
            points = partition @ mesh.points[mesh.faces[face]]
            trace[skeleton.subtriangle_dofs(face, segment)] = -(points @ gradient)[:, None] * (
                tensor @ mesh.normals[face]
            )
    hybrid = HybridSolution(trace.ravel(), (), (), 0, np.array([]))
    solution = Elasticity3DSolution(
        skeleton,
        tuple(mesh.submesh(cell, 2) for cell in range(len(mesh.cells))),
        (),
        hybrid,
        1,
        None,
        1.0,
        1.0,
        -tensor @ gradient,
        5,
    )
    assert_allclose(solution.equilibrium_residuals(), 0, atol=1e-12, rtol=1e-10)


def test_translation_gauge_checks_continuity_of_the_full_face_projection():
    """A piecewise constant normal advection trace belongs to DG but not subdivided C0."""
    mesh = two_tetrahedra()

    def broken(points):
        """Make a jump aligned with the dyadic triangular face partition."""
        return np.column_stack((np.zeros((len(points), 2)), (points[:, 0] > 0.5).astype(float)))

    dg = TriangularSkeleton(mesh, 2, degree=1)
    c0 = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    assert _translation_trace_compatible(dg, broken, 5)
    assert not _translation_trace_compatible(c0, broken, 5)
    assert _translation_trace_compatible(c0, [1.0, 2.0, 3.0], 5)


def test_tangential_mass_projection_and_maxwell_evolution_with_shared_coefficients():
    """Face mass, incident orientation and time evolution retain a constant physical field."""
    mesh = two_tetrahedra()
    base = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    skeleton = TangentialTraceSpace(base)
    electric, magnetic = np.array([1.2, -0.3, 0.7]), np.array([0.4, 0.8, -0.2])
    mass = tangential_mass(skeleton, dict.fromkeys(range(len(mesh.faces)), 1.0), 5)
    load = tangential_load(skeleton, electric, range(len(mesh.faces)), 5)
    expected = np.concatenate(
        [
            np.tile(electric @ skeleton.frames[face], len(base.dofs(face)))
            for face in range(len(mesh.faces))
        ]
    )
    assert_allclose(mass @ expected, load, atol=1e-12, rtol=1e-10)
    assert np.linalg.eigvalsh(mass.toarray()).min() > 0

    def boundary(time, points, normals):
        """Prescribe the outgoing physical condition for constant electric/magnetic fields."""
        return electric - np.cross(magnetic, normals)

    with MaxwellStepper(
        mesh,
        skeleton=skeleton,
        degree=3,
        local_refinement=2,
        time_step=0.001,
        absorbing=1.0,
        boundary_data=boundary,
    ) as stepper:
        initial = stepper.initialize(electric, magnetic)
        result = stepper.advance()
        assert max(result.l2_errors(electric, magnetic)) < 2e-11
        assert_allclose(result.energy, initial.energy, atol=2e-12, rtol=1e-10)
        assert abs(result.energy_balance_residual) < 2e-12


@pytest.mark.parametrize("tangential", [False, True])
def test_bound_basis_detects_changed_subtriangle_coordinates(tangential):
    """A persisted binding rejects a changed coefficient map even when dimensions agree."""
    mesh = two_tetrahedra()
    base = TriangularSkeleton(mesh, 2, degree=1, continuous=True)
    space = TangentialTraceSpace(base) if tangential else base
    interface = bind_interface(space)
    archived = pickle.loads(pickle.dumps(interface))
    assert interface.binding(0).basis_digest == archived.binding(0).basis_digest
    original = base.subtriangle_dofs
    base.subtriangle_dofs = lambda face, segment: original(face, segment)[::-1]
    with pytest.raises(ValueError, match="basis identity is stale"):
        interface.binding(0)


def test_mh2m_maintains_its_discontinuous_conormal_contract():
    """Pressure-trace continuity does not authorize a different Lambda formulation."""
    mesh = two_tetrahedra()
    with pytest.raises(ValueError, match="Lambda uses discontinuous face polynomials"):
        solve_mh2m_3d(mesh, flux_space=TriangularSkeleton(mesh, degree=1, continuous=True))


@pytest.mark.parametrize("continuous", [False, True])
def test_high_degree_face_masses_match_independent_bernstein_integrals(continuous):
    """Exact factorial moments distinguish a full-rank P8 mass from underintegration."""
    mesh = two_tetrahedra()
    degree, face = 8, 0
    base = TriangularSkeleton(mesh, 2, degree=degree, continuous=continuous)
    exponents = [
        (i, j, degree - i - j) for i in range(degree, -1, -1) for j in range(degree - i, -1, -1)
    ]
    reference = np.array(
        [
            [
                2
                * factorial(degree) ** 2
                * prod(factorial(a + b) for a, b in zip(alpha, beta, strict=True))
                / (factorial(2 * degree + 2) * prod(factorial(a) for a in (*alpha, *beta)))
                for beta in exponents
            ]
            for alpha in exponents
        ]
    )
    expected = np.zeros((len(base.dofs(face)),) * 2)
    for segment, fraction in enumerate(base.face_weights(face)):
        ids = np.searchsorted(base.dofs(face), base.subtriangle_dofs(face, segment))
        expected[np.ix_(ids, ids)] += mesh.areas[face] * fraction * reference
    _, weights, basis = _face_rule(mesh, base, face, 1)
    # Compare native tabulation and BLAS accumulation at portable binary64 tolerances.
    assert_allclose(basis.T @ (weights[:, None] * basis), expected, atol=1e-12, rtol=1e-10)
    skeleton = TangentialTraceSpace(base)
    mass = tangential_mass(skeleton, {face: 1.0}, 1)
    ids = skeleton.dofs(face)
    actual = mass[ids][:, ids].toarray()
    assert_allclose(actual, np.kron(expected, np.eye(2)), atol=1e-12, rtol=1e-10)
    assert np.linalg.eigvalsh(actual).min() > 0
    fine = mesh.submesh(0, 2)
    minimal = trace_coupling(skeleton, 0, fine, 2, 1).toarray()
    overintegrated = trace_coupling(skeleton, 0, fine, 2, 11).toarray()
    assert_allclose(minimal, overintegrated, atol=1e-12, rtol=1e-10)


def test_high_degree_planar_tangential_mass_and_coupling_are_integrated():
    """Exact Legendre orthogonality checks the corresponding two-dimensional owner."""
    mesh = TriangleMesh.unit_square()
    base = SkeletonSpace(mesh, tuple(FaceSpace.uniform(8) for _ in mesh.faces))
    skeleton = TangentialTraceSpace(base)
    face = int(mesh.boundary_faces[0])
    ids = skeleton.dofs(face)
    actual = tangential_mass(skeleton, {face: 1.0}, 1)[ids][:, ids].toarray()
    expected = np.diag(mesh.lengths[face] / (2 * np.arange(9) + 1))
    assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
    fine = mesh.submesh(0, 2)
    assert_allclose(
        trace_coupling(skeleton, 0, fine, 2, 1).toarray(),
        trace_coupling(skeleton, 0, fine, 2, 11).toarray(),
        atol=1e-12,
        rtol=1e-10,
    )


@pytest.mark.parametrize("continuous", [False, True])
def test_high_degree_translation_projection_avoids_normal_equation_conditioning(continuous):
    """A constant physical advection trace is representable in the resolved P8 space."""
    mesh = two_tetrahedra()
    skeleton = TriangularSkeleton(mesh, degree=8, continuous=continuous)
    assert _translation_trace_compatible(skeleton, [1.0, 2.0, 3.0], 1)


@pytest.mark.parametrize("continuous", [False, True])
def test_unresolved_translation_projection_is_rejected(continuous, monkeypatch):
    """A failed basis-rank check cannot authorize a physical translation gauge."""
    mesh = two_tetrahedra()
    skeleton = TriangularSkeleton(mesh, 2, degree=1, continuous=continuous)
    original = np.linalg.lstsq

    def unresolved(*args, **kwargs):
        """Retain the computed projection but report a numerically unresolved basis."""
        coefficients, residuals, _, singular = original(*args, **kwargs)
        return coefficients, residuals, 0, singular

    monkeypatch.setattr(np.linalg, "lstsq", unresolved)
    assert not _translation_trace_compatible(skeleton, [1.0, 2.0, 3.0], 5)
