"""Physical inertia, endpoint coupling and local Newmark invariants."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, sparse

from pymhm.elastodynamics import ElastodynamicStepper, solve_elastodynamics
from pymhm.mesh import SkeletonSpace, TriangleMesh
from pymhm.planar_material import PlanarMaterial, PlanarRegion
from pymhm.solvers import LinearSolveError
from pymhm.tetrahedral import TetraMesh


@pytest.mark.parametrize("backend", ["thread", "process"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_parallel_spatial_assembly_preserves_newmark_trajectory(backend, dimension):
    """Worker arrays retain orientation/inertia and create native resources in the owner."""
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    options = dict(time_step=0.04, degree=2, local_refinement=2, quadrature_order=4)
    with (
        ElastodynamicStepper(mesh, **options) as serial,
        ElastodynamicStepper(mesh, backend=backend, workers=2, **options) as parallel,
    ):
        serial.initialize()
        parallel.initialize()
        for _ in range(3):
            a = serial.advance(np.arange(1, dimension + 1, dtype=float))
            b = parallel.advance(np.arange(1, dimension + 1, dtype=float))
            assert_allclose(a.trace, b.trace, rtol=1e-12, atol=1e-12)
            for ua, ub, va, vb in zip(
                a.displacement, b.displacement, a.velocity, b.velocity, strict=True
            ):
                assert_allclose(ua, ub, rtol=1e-12, atol=1e-12)
                assert_allclose(va, vb, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("substeps", [1, 3])
def test_free_translation_under_time_dependent_uniform_acceleration(dimension, substeps):
    """An exact rigid translation checks inertial units, forces and every spatial component."""
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    vector = np.arange(1, dimension + 1)
    rho, dt = 2.4, 0.02
    with ElastodynamicStepper(
        mesh,
        time_step=dt,
        degree=2,
        density=rho,
        local_substeps=substeps,
        traction={int(f): np.zeros(dimension) for f in mesh.boundary_faces},
    ) as stepper:
        stepper.initialize(np.ones(dimension), vector)
        for index in range(4):
            solution = stepper.advance(lambda t, p: rho * vector)
            t = (index + 1) * dt
            for u, v in zip(solution.displacement, solution.velocity, strict=True):
                assert_allclose(
                    u.reshape(-1, dimension),
                    np.broadcast_to(1 + (t + 0.5 * t * t) * vector, u.reshape(-1, dimension).shape),
                    atol=2e-11,
                )
                assert_allclose(
                    v.reshape(-1, dimension),
                    np.broadcast_to((1 + t) * vector, v.reshape(-1, dimension).shape),
                    atol=3e-10,
                )
            assert solution.l2_error(1 + (t + 0.5 * t * t) * vector) < 2e-11
            assert solution.l2_error((1 + t) * vector, velocity=True) < 3e-10
            bary = np.full((1, dimension + 1), 1 / (dimension + 1))
            assert_allclose(solution.gradient(0, bary), 0, atol=1e-9)
            assert_allclose(solution.gradient(0, bary, velocity=True), 0, atol=1e-8)
            assert_allclose(solution.stress(0, bary), 0, atol=1e-8)


@pytest.mark.parametrize("substeps", [1, 4])
def test_uncondensed_space_time_saddle(substeps):
    """Independent full block Newmark propagation matches condensed endpoint coupling."""
    mesh = TriangleMesh.unit_square()
    with ElastodynamicStepper(mesh, time_step=0.013, degree=2, local_substeps=substeps) as stepper:
        stepper.initialize()
        mass = sparse.block_diag([local.mass for local in stepper.locals]).toarray()
        stiffness = sparse.block_diag([local.stiffness for local in stepper.locals]).toarray()
        offsets = np.r_[0, np.cumsum([local.mass.shape[0] for local in stepper.locals])]
        b = np.zeros((len(mass), stepper.size))
        for i, local in enumerate(stepper.locals):
            b[np.ix_(np.arange(offsets[i], offsets[i + 1]), local.trace_dofs)] = local.coupling
        u, v = np.zeros(len(mass)), np.zeros(len(mass))
        delta = stepper.time_step / substeps
        effective = mass + delta**2 / 4 * stiffness
        for _iteration in range(3):
            free_u, free_v = u.copy(), v.copy()
            lift, lift_v = np.zeros_like(b), np.zeros_like(b)
            for _j in range(substeps):
                force = np.concatenate([local.load((1.0, 2.0)) for local in stepper.locals])
                new_u = linalg.solve(
                    effective,
                    mass @ (free_u + delta * free_v)
                    - delta**2 / 4 * stiffness @ free_u
                    + delta**2 / 2 * force,
                )
                free_v += delta * linalg.solve(mass, force - stiffness @ ((free_u + new_u) / 2))
                free_u = new_u
                new_lift = linalg.solve(
                    effective,
                    mass @ (lift + delta * lift_v)
                    - delta**2 / 4 * stiffness @ lift
                    + delta**2 / 2 * b,
                )
                lift_v += delta * linalg.solve(mass, b - stiffness @ ((lift + new_lift) / 2))
                lift = new_lift
            multiplier = linalg.solve(b.T @ lift, b.T @ free_u)
            u, v = free_u - lift @ multiplier, free_v - lift_v @ multiplier
            solution = stepper.advance((1.0, 2.0))
            assert_allclose(np.concatenate(solution.displacement), u, atol=3e-15)
            assert_allclose(np.concatenate(solution.velocity), v, atol=2e-13)


def test_physical_energy_and_semidiscrete_second_order():
    """Unforced beta=1/4 trajectories conserve energy and converge to exp(t A)."""
    mesh = TriangleMesh.unit_square()
    errors = []
    for dt in (0.02, 0.01, 0.005):
        with ElastodynamicStepper(mesh, time_step=dt, degree=2) as stepper:
            initial = stepper.initialize(
                velocity=lambda p: np.column_stack(
                    (p[:, 0] * (1 - p[:, 0]), p[:, 1] * (1 - p[:, 1]))
                )
            )
            mass = sparse.block_diag([local.mass for local in stepper.locals]).toarray()
            stiffness = sparse.block_diag([local.stiffness for local in stepper.locals]).toarray()
            b = np.zeros((len(mass), stepper.size))
            offset = 0
            for local in stepper.locals:
                b[np.ix_(np.arange(offset, offset + local.mass.shape[0]), local.trace_dofs)] = (
                    local.coupling
                )
                offset += local.mass.shape[0]
            null = linalg.null_space(b.T)
            reduced_m, reduced_k = null.T @ mass @ null, null.T @ stiffness @ null
            count = len(reduced_m)
            generator = np.block(
                [
                    [np.zeros_like(reduced_m), np.eye(count)],
                    [-linalg.solve(reduced_m, reduced_k), np.zeros_like(reduced_m)],
                ]
            )
            exact = (
                linalg.expm(0.2 * generator)
                @ np.r_[
                    null.T @ np.concatenate(initial.displacement),
                    null.T @ np.concatenate(initial.velocity),
                ]
            )
            for _ in range(round(0.2 / dt)):
                result = stepper.advance()
                assert_allclose(result.energy, initial.energy, rtol=2e-13)
            difference = np.r_[
                null @ exact[:count] - np.concatenate(result.displacement),
                null @ exact[count:] - np.concatenate(result.velocity),
            ]
            errors.append(np.linalg.norm(difference))
    assert 3.1 < errors[0] / errors[1] < 4.4
    assert 3.6 < errors[1] / errors[2] < 4.2


def test_mixed_boundary_static_affine_equilibrium_and_subcycles():
    """Nonzero essential displacement and physical traction preserve affine equilibrium."""
    mesh = TriangleMesh.unit_square()
    gradient = np.array([[0.2, 0.3], [-0.1, -0.2]])
    stress = gradient + gradient.T
    boundary = [int(f) for f in mesh.boundary_faces if mesh.normals[f, 0] > 0.5]
    traction = {f: stress @ mesh.normals[f] for f in boundary}

    def exact(points):
        """Time-independent affine displacement with a nonzero rigid component."""
        return points @ gradient.T + np.array([1.0, -0.5])

    history = []
    result = solve_elastodynamics(
        mesh,
        time_step=0.01,
        steps=5,
        degree=2,
        displacement=exact,
        dirichlet=exact,
        traction=traction,
        local_substeps=[1, 3],
        on_step=history.append,
    )
    assert len(history) == 6
    assert result.l2_error(exact) < 3e-14
    assert result.l2_error((0, 0), velocity=True) < 5e-12
    bary = np.array([[0.2, 0.3, 0.5]])
    assert_allclose(result.stress(0, bary), np.broadcast_to(stress, (4, 1, 2, 2)), atol=3e-12)
    with pytest.raises(ValueError, match="cell"):
        result.gradient(3, bary)


def test_lifecycle_invalid_data_and_single_free_body():
    """Invalid geometry/material/state fail explicitly; a free single body needs no gauge."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(TypeError, match="triangular"):
        ElastodynamicStepper(None, time_step=0.01)
    for dt in (0, -1, np.inf):
        with pytest.raises(ValueError, match="time_step"):
            ElastodynamicStepper(mesh, time_step=dt)
    with pytest.raises(ValueError, match="density"):
        ElastodynamicStepper(mesh, time_step=0.01, density=0)
    with pytest.raises(ValueError, match="belong"):
        ElastodynamicStepper(
            mesh, time_step=0.01, skeleton=SkeletonSpace(TriangleMesh.unit_square())
        )
    with pytest.raises(ValueError, match="components"):
        ElastodynamicStepper(mesh, time_step=0.01, skeleton=SkeletonSpace(mesh))
    with pytest.raises(ValueError, match="exterior"):
        ElastodynamicStepper(mesh, time_step=0.01, traction={-1: (0, 0)})
    with pytest.raises(ValueError, match="local_substeps"):
        ElastodynamicStepper(mesh, time_step=0.01, local_substeps=0)
    one = TriangleMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]))
    with ElastodynamicStepper(
        one, time_step=0.01, degree=1, traction={int(f): (0, 0) for f in one.boundary_faces}
    ) as stepper:
        with pytest.raises(RuntimeError, match="initialize"):
            stepper.solution()
        with pytest.raises(RuntimeError, match="initialize"):
            stepper.advance()
        result = stepper.initialize((1, 2), (3, 4))
        stepper.advance()
        with pytest.raises(ValueError, match="cell"):
            result.evaluate(2, np.array([[1 / 3] * 3]))
    stepper.close()
    with pytest.raises(RuntimeError, match="closed"):
        stepper.__enter__()
    with pytest.raises(RuntimeError, match="open"):
        stepper.initialize()
    with pytest.raises(ValueError, match="solver"):
        ElastodynamicStepper(mesh, time_step=0.01, local_solver="does-not-exist")


def test_constraint_check_and_callback_free_wrapper():
    """The physical residual rejects a violated displacement trace without hiding it."""
    mesh = TriangleMesh.unit_square()
    with ElastodynamicStepper(mesh, time_step=0.01, degree=2) as stepper:
        stepper.initialize()
        stepper.displacement[0][:] = 1
        with pytest.raises(LinearSolveError, match="trace constraint"):
            stepper.solution()
    result = solve_elastodynamics(mesh, time_step=0.01, steps=1, degree=2)
    assert result.energy == 0


@pytest.mark.parametrize("dimension", [2, 3])
def test_cut_density_preserves_total_inertia(dimension):
    """A density interface cuts fine elements; exact material integration preserves total mass."""
    mesh = TriangleMesh.unit_square() if dimension == 2 else TetraMesh.unit_cube()
    normal = np.eye(dimension)[0:1]
    density = PlanarMaterial(1.0, (PlanarRegion(normal, [0.37], 3.0),))
    with ElastodynamicStepper(
        mesh,
        time_step=0.01,
        degree=2,
        local_refinement=1,
        density=density,
        traction={int(f): np.zeros(dimension) for f in mesh.boundary_faces},
    ) as stepper:
        constant = np.eye(dimension)[0]
        mass = 0.0
        for local in stepper.locals:
            field = np.tile(constant, len(local.nodes))
            mass += field @ (local.mass @ field)
        assert_allclose(mass, 1 + 2 * 0.37, atol=4e-14)
        assert_allclose(stepper.initialize(velocity=constant).energy, 0.5 * mass, atol=2e-14)
    with pytest.raises(ValueError, match="isotropic"):
        ElastodynamicStepper(mesh, time_step=0.01, density=np.diag(np.arange(1, dimension + 1)))
