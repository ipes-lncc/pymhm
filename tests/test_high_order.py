"""Independent polynomial and continuity checks for enriched MHM discretizations."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm._legacy.models.vector import solve_brinkman
from pymhm.fem.scalar.operators import p1_operators, triangle_quadrature
from pymhm.fem.scalar.triangle import scalar_operators, tabulate
from pymhm.recovery.equilibrated import equilibrate_flux


@pytest.mark.parametrize("degrees", [(1, 1), (2, 3), (4, 2)])
def test_continuous_piecewise_face_is_conforming_and_reproduces_polynomials(degrees):
    face = FaceSpace((0.0, 0.37, 1.0), degrees, continuous=True)
    nodes = list(face.breaks)
    for i, degree in enumerate(degrees):
        nodes.extend(np.linspace(face.breaks[i], face.breaks[i + 1], degree + 1)[1:-1])
    nodes = np.asarray(nodes)
    assert_allclose(face.evaluate(nodes), np.eye(len(nodes)), atol=1e-14, rtol=0)
    coordinates = np.linspace(0, 1, 137)
    assert_allclose(
        face.evaluate(coordinates) @ nodes ** min(degrees), coordinates ** min(degrees), atol=1e-14
    )
    assert_allclose(face.evaluate(coordinates) @ face.constant_coefficients(), 1, atol=1e-14)
    values = face.evaluate([0.37 - 1e-10, 0.37 + 1e-10])
    assert_allclose(values[0], values[1], atol=2e-8, rtol=0)
    assert face.size == 1 + sum(degrees)


@pytest.mark.parametrize("continuous,degree", [(True, 0), (1, 1), (None, 1)])
def test_continuous_face_requires_a_positive_degree_and_boolean_flag(continuous, degree):
    with pytest.raises(ValueError, match="continuous"):
        FaceSpace.uniform(degree, continuous=continuous)


def test_discontinuous_face_constant_representation_and_continuous_embedding():
    continuous = FaceSpace((0, 0.4, 1), (2, 1), True)
    discontinuous = FaceSpace(continuous.breaks, continuous.degrees)
    points, weights = continuous.quadrature(5)
    d, c = discontinuous.evaluate(points), continuous.evaluate(points)
    transform = np.linalg.solve(d.T @ (weights[:, None] * d), d.T @ (weights[:, None] * c))
    assert_allclose(d @ transform, c, atol=1e-14)
    assert_allclose(d @ discontinuous.constant_coefficients(), 1, atol=1e-15)
    assert_allclose(continuous.evaluate([0.4])[:, 1], 1)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("continuous", [False, True])
def test_primal_darcy_reproduces_arbitrary_degree_with_tensor_and_nonzero_boundary(
    degree, continuous
):
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(
        mesh,
        tuple(FaceSpace.uniform(max(1, degree - 1), 2, continuous=continuous) for _ in mesh.faces),
    )
    coefficient = np.array([[3.0, 0.3], [0.3, 2.0]])

    def exact(x):
        return x[:, 0] ** degree + 2 * x[:, 1] ** degree

    def source(x):
        return (
            -degree
            * (degree - 1)
            * (3 * x[:, 0] ** max(0, degree - 2) + 4 * x[:, 1] ** max(0, degree - 2))
        )

    def flux(x):
        return (
            -degree
            * np.column_stack((x[:, 0] ** (degree - 1), 2 * x[:, 1] ** (degree - 1)))
            @ coefficient
        )

    result = solve_darcy(
        mesh,
        permeability=coefficient,
        source=source,
        dirichlet=exact,
        degree=degree,
        skeleton=space,
        local_refinement=6,
    )
    assert result.l2_error(exact) < 2e-12
    assert result.flux_l2_error(flux) < 2e-11
    assert_allclose(result.conservation_residuals(), 0, atol=2e-11)


def test_high_order_darcy_neumann_gauge_and_equilibration():
    mesh = TriangleMesh.unit_square()

    def exact(x):
        return x[:, 0] + 2 * x[:, 1]

    neumann = {
        int(face): -mesh.normals[face] @ np.array([1.0, 2.0]) for face in mesh.boundary_faces
    }
    solution = solve_darcy(mesh, degree=3, neumann=neumann, mean_pressure=1.5)
    assert solution.l2_error(exact) < 1e-12
    recovered = equilibrate_flux(solution)
    assert recovered.l2_error([-1.0, -2.0]) < 1e-11
    for residual in recovered.conservation_residuals():
        assert_allclose(residual, 0, atol=1e-12)
    with pytest.raises(ValueError, match="RT0"):
        solve_darcy(mesh, formulation="mixed", degree=2)


@pytest.mark.parametrize("skew", [False, True])
def test_general_scalar_operator_matches_independent_p1_kernel(skew):
    mesh = TriangleMesh.unit_square(2)
    tensor = [[2.0, 0.1], [0.1, 3.0]]
    actual, mass, load = scalar_operators(
        mesh,
        1,
        diffusion=tensor,
        source=2.0,
        reaction=3.0,
        advection=(2.0, -1.0),
        skew_advection=skew,
        order=4,
    )
    base, old_mass, old_load = p1_operators(mesh, tensor, 2.0, 3.0, order=4)
    advective = p1_operators(mesh, tensor, 2.0, 3.0, (2.0, -1.0), order=4)[0] - base
    expected = base + ((advective - advective.T) / 2 if skew else advective)
    assert_allclose(actual.toarray(), expected.toarray(), atol=1e-14)
    assert_allclose(mass.toarray(), old_mass.toarray(), atol=1e-15)
    assert_allclose(load, old_load, atol=1e-15)


def test_scalar_operators_reject_negative_reaction_and_high_order_mass_integrates_volume():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="reaction"):
        scalar_operators(mesh, 2, reaction=-1.0)
    matrix, mass, load = scalar_operators(mesh, 3, source=1.0)
    ones = np.ones(len(load))
    assert_allclose(ones @ mass @ ones, 1, atol=1e-14)
    assert_allclose(matrix @ ones, 0, atol=1e-14)
    assert_allclose(load, mass @ ones, atol=1e-14)


@pytest.mark.parametrize("method,degree", [("usfem", 2), ("usfem", 3), ("taylor-hood", 3)])
@pytest.mark.parametrize("drag", [0.0, 2.5])
def test_high_order_flow_exact_polynomial_and_consistent_source(method, degree, drag):
    mesh = TriangleMesh.unit_square(2)
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces), 2)

    def velocity(x):
        return np.column_stack((x[:, 0] ** 2, -2 * x[:, 0] * x[:, 1]))

    def pressure(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2 - 2 / 3

    def source(x):
        return drag * velocity(x) + 2 * x - [2.0, 0.0]

    solution = solve_brinkman(
        mesh,
        degree=degree,
        formulation=method,
        drag=drag,
        source=source,
        dirichlet=velocity,
        skeleton=space,
        local_refinement=4,
    )
    assert solution.l2_error(velocity) < 3e-12
    assert solution.pressure_l2_error(pressure) < 2e-11
    assert solution.divergence_l2() < 2e-11


def test_flow_inverse_constant_obeys_polynomial_inequality():
    from pymhm.fem.inequalities import laplacian_inverse_bound

    mesh = TriangleMesh.unit_square()
    bary, weights = triangle_quadrature(6)
    for degree in [1, 2, 3]:
        _, _, _, gradient, hessian = tabulate(mesh, degree, bary)
        diameters = np.max(mesh.lengths[mesh.cell_faces], axis=1)
        m = laplacian_inverse_bound(gradient, hessian, weights, diameters)
        assert np.all((m > 0) & (m <= 1 / 3))
        coefficients = np.random.default_rng(903).normal(size=(31, gradient.shape[2]))
        grad = np.einsum("tqia,ri->rtqa", gradient, coefficients)
        lap = np.einsum("tqi,ri->rtq", np.trace(hessian, axis1=-2, axis2=-1), coefficients)
        left = m[None] * diameters[None] ** 2 * (lap**2 @ weights)
        right = np.sum(grad**2, axis=-1) @ weights
        assert np.all(left <= right + 1e-12)


def test_flow_mixed_and_pure_traction_boundary_conditions():
    mesh = TriangleMesh.unit_square()

    def velocity(x):
        return np.column_stack((x[:, 1], -x[:, 0]))

    gradient = np.array([[0.0, 1.0], [-1.0, 0.0]])
    traction = {int(face): gradient @ mesh.normals[face] for face in mesh.boundary_faces}
    for faces in [{int(mesh.boundary_faces[0]): traction[int(mesh.boundary_faces[0])]}, traction]:
        solution = solve_brinkman(
            mesh, dirichlet=velocity, traction=faces, mean_velocity=[0.5, -0.5]
        )
        assert solution.l2_error(velocity) < 1e-12
        assert solution.pressure_l2_error(0.0) < 1e-11
    with pytest.raises(ValueError, match="gauge"):
        solve_brinkman(mesh, traction=traction, mean_pressure=1.0)
    with pytest.raises(ValueError, match="finite"):
        solve_brinkman(mesh, mean_pressure=np.nan)
    with pytest.raises(ValueError, match="degree"):
        solve_brinkman(mesh, degree=1)


@pytest.mark.parametrize("formulation,degree", [("taylor-hood", 2), ("usfem", 1), ("usfem", 2)])
@pytest.mark.parametrize("variable", [False, True])
def test_tensor_brinkman_resistance_and_force_consistency(formulation, degree, variable):
    mesh = TriangleMesh.unit_square(2)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), 2)
    tensor = np.array([[20.0, 3.0], [3.0, 1.0]])
    coefficient = (
        (lambda x: (1 + x[:, 0] + 2 * x[:, 1])[:, None, None] * tensor) if variable else tensor
    )

    def velocity(x):
        return np.column_stack((x[:, 1], -x[:, 0]))

    def force(x):
        values = coefficient(x) if variable else np.broadcast_to(tensor, (len(x), 2, 2))
        return np.einsum("qij,qj->qi", values, velocity(x)) + [1.0, 2.0]

    result = solve_brinkman(
        mesh,
        drag=coefficient,
        formulation=formulation,
        degree=degree,
        local_refinement=4,
        source=force,
        dirichlet=velocity,
        skeleton=skeleton,
    )
    assert result.l2_error(velocity) < 2e-12
    assert result.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1] - 1.5) < 2e-11


@pytest.mark.parametrize(
    "drag", [-1.0, np.inf, 1j, [[1.0, 2.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]]
)
def test_brinkman_resistance_rejects_invalid_material(drag):
    with pytest.raises(ValueError, match="drag"):
        solve_brinkman(TriangleMesh.unit_square(), drag=drag)


def test_p2_trace_wrapper_matches_shared_assembly():
    from pymhm.fem.scalar.triangle import trace_coupling
    from pymhm.fem.vector.operators import _p2_coupling

    mesh = TriangleMesh.unit_square()
    fine = mesh.submesh(0, 3)
    space = SkeletonSpace(mesh)
    assert_allclose(_p2_coupling(mesh, 0, fine, space), trace_coupling(mesh, 0, fine, space, 2))


@pytest.mark.parametrize("rotated", [False, True])
def test_semidefinite_brinkman_retains_only_unresisted_translation(rotated):
    """A physical mean fixes the surviving null mode, including a rotated basis."""
    mesh = TriangleMesh.unit_square()
    material = np.array([[1.0, 1.0], [1.0, 1.0]]) if rotated else np.diag([1.0, 0.0])
    direction = np.array([1.0, -1.0]) if rotated else np.array([0.0, 1.0])
    solution = solve_brinkman(
        mesh,
        drag=material,
        traction={int(face): (0.0, 0.0) for face in mesh.boundary_faces},
        mean_velocity=direction,
        translation_kernel=direction[:, None] if rotated else None,
    )
    assert solution.l2_error(direction) < 2e-12
    assert solution.pressure_l2_error(0.0) < 2e-11


@pytest.mark.parametrize("kernel", [[[0.0], [1.0]], [[1.0], [0.0]]])
def test_small_positive_resistance_is_not_a_translation_gauge(kernel):
    """A weakly resisted direction remains physical regardless of its scale."""
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="resisted"):
        solve_brinkman(
            mesh,
            drag=np.diag([1.0, 1e-16]),
            traction={int(face): (0.0, 0.0) for face in mesh.boundary_faces},
            translation_kernel=kernel,
        )


@pytest.mark.parametrize("kernel", [[1.0, 0.0], [[1.0]], [[1, 1], [0, 0]], [[np.nan], [1]]])
def test_invalid_translation_kernel_is_rejected(kernel):
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="basis"):
        solve_brinkman(
            mesh,
            traction={int(face): (0.0, 0.0) for face in mesh.boundary_faces},
            translation_kernel=kernel,
        )


def test_translation_gauge_cannot_change_resisted_or_boundary_fixed_components():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="pure-traction"):
        solve_brinkman(mesh, translation_kernel=np.eye(2))
    with pytest.raises(ValueError, match="unresisted"):
        solve_brinkman(
            mesh,
            drag=np.diag([1.0, 0.0]),
            traction={int(face): (0.0, 0.0) for face in mesh.boundary_faces},
            mean_velocity=(1.0, 0.0),
        )
