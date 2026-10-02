"""Physical patches, gauges and contracts for native three-dimensional mixed flow."""

from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm.darcy3d import TriangularSkeleton
from pymhm.flow3d import _boundary_flow, solve_flow_3d
from pymhm.flow3d_forms import _minimum_resistance_3d, resistance_values_3d, tetra_flow_operators
from pymhm.reservoir import CartesianCellField
from pymhm.tetrahedral import TetraMesh, tetrahedron_quadrature

GRADIENT = np.array([[1.0, 0.3, -0.2], [0.2, -2.0, 0.4], [-0.1, 0.5, 1.0]])
TENSOR = np.array([[2.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 4.0]])


@pytest.fixture(autouse=True)
def bounded_threads():
    """Limit native threading for the small matrices used in CI."""
    with threadpool_limits(1):
        yield


def velocity(points):
    """Return an affine incompressible field involving all gradient entries."""
    return points @ GRADIENT.T + [0.3, 0.2, 0.1]


def pressure(points):
    """Return a zero-mean affine pressure on the cube."""
    return points[:, 0] + points[:, 1] - 1


def forcing(points, gamma=TENSOR, beta=(0.0, 0.0, 0.0)):
    """Compute force directly from the strong affine momentum equation."""
    resistance = gamma(points) if callable(gamma) else gamma
    tensor = np.broadcast_to(resistance, (len(points), 3, 3))
    transport = beta(points) if callable(beta) else np.broadcast_to(beta, (len(points), 3))
    return (
        np.einsum("nij,nj->ni", tensor, velocity(points)) + transport @ GRADIENT.T + [1.0, 1.0, 0.0]
    )


@pytest.mark.parametrize(
    "formulation,degree,refinement",
    [
        ("taylor-hood", 2, 2),
        ("taylor-hood", 3, 2),
        ("taylor-hood", 4, 1),
        ("usfem", 1, 4),
        ("usfem", 2, 2),
        ("usfem", 3, 2),
        ("usfem", 4, 1),
        ("oseen", 1, 4),
        ("oseen", 2, 2),
        ("oseen", 3, 2),
        ("oseen", 4, 1),
    ],
)
def test_affine_incompressible_patch(formulation, degree, refinement):
    """Reproduce nonhomogeneous velocity/pressure with every supported local degree."""
    beta = (1.0, 0.5, 0.2) if formulation == "oseen" else (0.0, 0.0, 0.0)
    gamma = 2 * np.eye(3) if formulation == "oseen" else TENSOR
    solution = solve_flow_3d(
        TetraMesh.unit_cube(),
        degree=degree,
        local_refinement=refinement,
        formulation=formulation,
        drag=2.0 if formulation == "oseen" else gamma,
        advection=beta,
        source=partial(forcing, gamma=gamma, beta=beta),
        dirichlet=velocity,
        quadrature_order=degree + 2,
    )
    assert solution.l2_error(velocity, 4) < 3e-11
    assert solution.pressure_l2_error(pressure, 4) < 3e-10
    assert solution.h1_seminorm_error(GRADIENT, 4) < 3e-10
    assert solution.divergence_l2(4) < 3e-10
    bary = np.array([[0.1, 0.2, 0.3, 0.4]])
    u, p = solution.evaluate(0, bary)
    expected = -GRADIENT + p[..., None, None] * np.eye(3) + np.einsum("tqa,b->tqab", u, beta) / 2
    assert_allclose(solution.pseudostress(0, bary), expected, atol=3e-10)
    with pytest.raises(ValueError, match="outside"):
        solution.evaluate(len(solution.local_meshes), bary)


@pytest.mark.parametrize("stabilization", ["tensor-2025", "minimum-2017", "pointwise-2017"])
def test_variable_tensor_resistance_and_small_positive_limit(stabilization):
    """Use a genuine tensor field and the exact source, including all drag components."""

    def material(x):
        """Evaluate a varying SPD tensor."""
        return (1 + x[:, 0] + x[:, 2])[:, None, None] * TENSOR

    solution = solve_flow_3d(
        TetraMesh.unit_cube(),
        formulation="usfem",
        degree=2,
        drag=material,
        source=partial(forcing, gamma=material),
        dirichlet=velocity,
        stabilization=stabilization,
        gamma_min=1.0 if stabilization == "minimum-2017" else None,
    )
    assert solution.l2_error(velocity, 4) < 1e-11
    assert solution.pressure_l2_error(pressure, 4) < 1e-10


@pytest.mark.parametrize("formulation", ["taylor-hood", "usfem"])
@pytest.mark.parametrize("gamma", [0.0, 1e-16, 1e-8])
def test_retained_translations_preserve_near_stokes_limit(formulation, gamma):
    """Preserve tiny positive resistance without classifying it as a nullspace."""
    solution = solve_flow_3d(
        TetraMesh.unit_cube(),
        formulation=formulation,
        degree=2,
        drag=gamma,
        source=partial(forcing, gamma=gamma * np.eye(3)),
        dirichlet=velocity,
    )
    assert solution.l2_error(velocity, 4) < 3e-11
    assert solution.pressure_l2_error(pressure, 4) < 3e-10


def test_variable_advection_strong_form_and_pressure_gauge():
    """The skew correction gives beta.grad(u) for nonzero divergence."""

    def beta(x):
        """Return affine convection with divergence three."""
        return x + [1.0, 0.5, 0.2]

    for formulation in ["taylor-hood", "oseen"]:
        solution = solve_flow_3d(
            TetraMesh.unit_cube(),
            degree=2,
            formulation=formulation,
            drag=3.0,
            advection=beta,
            advection_divergence=3.0,
            advection_bound=4.0,
            source=partial(forcing, gamma=3 * np.eye(3), beta=beta),
            dirichlet=velocity,
        )
        assert solution.l2_error(velocity, 4) < 3e-11
        assert solution.pressure_l2_error(pressure, 4) < 3e-10


def test_full_traction_chooses_only_three_translations():
    """A grad-grad Stokes operator has translations, not the six elasticity rigid modes."""
    mesh = TetraMesh.unit_cube()
    traction = {
        int(face): lambda x, n=mesh.normals[face]: GRADIENT @ n - pressure(x)[:, None] * n
        for face in mesh.boundary_faces
    }
    solution = solve_flow_3d(
        mesh,
        traction=traction,
        source=(1.0, 1.0, 0.0),
        mean_velocity=velocity(np.array([[0.5, 0.5, 0.5]]))[0],
    )
    assert solution.l2_error(velocity, 4) < 3e-11
    assert solution.pressure_l2_error(pressure, 4) < 3e-10


@pytest.mark.parametrize("formulation", ["taylor-hood", "usfem", "oseen"])
def test_component_slip_and_top_pressure_datum(formulation):
    """Vertical slip walls leave tangential velocity free; top traction determines pressure."""
    mesh = TetraMesh.unit_cube()
    centers = mesh.points[mesh.faces].mean(axis=1)
    top = {int(f): (0.0, 0.0, 0.0) for f in mesh.boundary_faces if np.isclose(centers[f, 2], 1)}
    sides = {int(f): {2: 0.0} for f in mesh.boundary_faces if abs(mesh.normals[f, 2]) < 0.5}

    def boundary(x):
        """Prescribe bottom inflow and zero normal flow on vertical walls."""
        return np.column_stack((np.zeros((len(x), 2)), np.isclose(x[:, 2], 0).astype(float)))

    result = solve_flow_3d(
        mesh,
        formulation=formulation,
        degree=2,
        drag=2.0,
        dirichlet=boundary,
        traction=top,
        traction_components=sides,
    )
    assert result.l2_error((0.0, 0.0, 1.0), 4) < 3e-11
    assert result.pressure_l2_error(lambda x: 2 * (1 - x[:, 2]), 4) < 3e-10


def test_tangential_traction_keeps_pressure_gauge():
    """Fixing only tangential traction cannot remove the pressure constant mode."""
    mesh = TetraMesh.unit_cube()
    tangential = {
        int(f): {i: 0.0 for i in range(3) if mesh.normals[f, i] == 0} for f in mesh.boundary_faces
    }
    result = solve_flow_3d(mesh, traction_components=tangential, mean_pressure=2.5, drag=1.0)
    assert result.l2_error((0.0, 0.0, 0.0), 4) < 3e-12
    assert result.pressure_l2_error(2.5, 4) < 3e-11


def test_semidefinite_material_and_declared_rotated_kernel():
    """Gauge only physically unresisted free translations, including rotated modes."""
    mesh = TetraMesh.unit_cube()
    traction = {int(f): (0.0, 0.0, 0.0) for f in mesh.boundary_faces}
    for tensor, mean, basis in [
        (np.diag([1.0, 2.0, 0.0]), [0.0, 0.0, 1.0], None),
        (
            np.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 2.0]]),
            [1.0, 1.0, 0.0],
            [[1.0], [1.0], [0.0]],
        ),
    ]:
        result = solve_flow_3d(
            mesh, drag=tensor, traction=traction, mean_velocity=mean, translation_kernel=basis
        )
        assert result.l2_error(mean, 4) < 4e-12
        assert result.pressure_l2_error(0.0, 4) < 5e-11


def test_segmented_faces_and_parallel_factory():
    """Complete local assembly is picklable and preserves oriented nonmatching macro fields."""
    cube = TetraMesh.unit_cube()
    mesh = TetraMesh(cube.points, cube.cells[:2])
    skeleton = TriangularSkeleton(mesh, subdivisions=2, degree=0)
    options = dict(
        skeleton=skeleton, degree=2, dirichlet=(1.0, 2.0, 3.0), drag=1.0, source=(1.0, 2.0, 3.0)
    )
    serial = solve_flow_3d(mesh, **options)
    for backend in ["thread", "process"]:
        result = solve_flow_3d(mesh, backend=backend, workers=2, **options)
        for expected, actual in zip(serial.hybrid.fields, result.hybrid.fields, strict=True):
            assert_allclose(actual, expected, atol=1e-12)
    assert serial.l2_error((1.0, 2.0, 3.0), 3) < 1e-12


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"viscosity": 0}, "positive"),
        ({"viscosity": 1j}, "finite"),
        ({"formulation": "bad"}, "formulation"),
        ({"stabilization": "bad"}, "stabilization"),
        ({"formulation": "taylor-hood", "stabilization": "minimum-2017"}, "USFEM"),
        ({"gamma_min": 0}, "only used"),
        ({"degree": 1}, "velocity degree"),
        ({"degree": 5}, "velocity degree"),
        ({"advection": lambda x: x}, "advection_divergence"),
        (
            {"formulation": "oseen", "advection": lambda x: x, "advection_divergence": 3},
            "advection_bound",
        ),
        ({"advection_divergence": 1}, "zero divergence"),
        ({"advection_divergence": lambda x: 0.0}, "zero divergence"),
        ({"advection_bound": -1}, "finite"),
        ({"advection": (1.0, 0.0, 0.0), "advection_bound": 0}, "sampled"),
        ({"formulation": "usfem", "advection": (1.0, 0.0, 0.0)}, "advection requires"),
        ({"formulation": "oseen", "drag": np.eye(3)}, "constant scalar"),
        ({"formulation": "oseen", "drag": lambda x: 1.0}, "constant scalar"),
        ({"formulation": "oseen", "drag": -1}, "finite"),
        ({"local_refinement": 3}, "power of two"),
        ({"mean_pressure": [0.0]}, "scalar"),
        ({"mean_pressure": np.nan}, "finite"),
        ({"mean_velocity": (1.0, 0.0, 0.0)}, "only available"),
        ({"translation_kernel": np.eye(3)}, "free traction"),
        ({"traction": {100: 0.0}}, "exterior"),
        ({"traction_components": {100: {0: 0.0}}}, "exterior"),
    ],
)
def test_invalid_solver_contracts(kwargs, message):
    """Reject unsupported formulations, physical bounds and inconsistent boundary/gauge inputs."""
    with pytest.raises(ValueError, match=message):
        solve_flow_3d(TetraMesh.unit_cube(), **kwargs)


@pytest.mark.parametrize("datum", [True, -1, np.nan, np.inf, 1j, [1.0], "bad"])
def test_invalid_minimum_bounds(datum):
    """Minimum-eigenvalue inputs must be finite real scalars."""
    with pytest.raises(ValueError, match="finite nonnegative"):
        _minimum_resistance_3d(1.0, datum)


@pytest.mark.parametrize(
    "datum,message",
    [
        (np.eye(2), "3-by-3"),
        (np.eye(3) * 1j, "real"),
        (np.diag([1.0, -1.0, 1.0]), "nonnegative"),
        (np.triu(np.ones((3, 3))), "symmetric"),
    ],
)
def test_invalid_material(datum, message):
    """No imaginary, nonsymmetric or indefinite resistance is silently coerced."""
    with pytest.raises(ValueError, match=message):
        resistance_values_3d(datum, np.zeros((2, 3)))


def test_minimum_resistance_contract_and_cartesian_values():
    """Use every Cartesian material value for the declared global minimum."""
    assert _minimum_resistance_3d(TENSOR, None) == np.linalg.eigvalsh(TENSOR)[0]
    assert _minimum_resistance_3d(TENSOR, 0.0) == 0.0
    scalar = CartesianCellField(np.arange(1.0, 9.0).reshape(2, 2, 2), (0.5, 0.5, 0.5))
    assert _minimum_resistance_3d(scalar, None) == 1.0
    tensors = CartesianCellField(np.broadcast_to(TENSOR, (2, 2, 2, 3, 3)), (0.5, 0.5, 0.5))
    assert _minimum_resistance_3d(tensors, None) == np.linalg.eigvalsh(TENSOR)[0]
    with pytest.raises(ValueError, match="three spatial"):
        _minimum_resistance_3d(CartesianCellField(np.ones((2, 2)), (0.5, 0.5)), None)
    with pytest.raises(ValueError, match="exceeds"):
        _minimum_resistance_3d(TENSOR, 10.0)
    with pytest.raises(ValueError, match="explicit"):
        _minimum_resistance_3d(lambda x: 1.0, None)
    with pytest.raises(ValueError, match="sampled"):
        tetra_flow_operators(
            TetraMesh.unit_cube(),
            formulation="usfem",
            drag=lambda x: 1.0,
            stabilization="minimum-2017",
            gamma_min=2.0,
        )


def test_boundary_and_kernel_input_contracts():
    """Check partial-traction structure and physical compatibility of proposed gauges."""
    mesh = TetraMesh.unit_cube()
    skeleton = TriangularSkeleton(mesh)
    face = int(mesh.boundary_faces[0])
    for components in [{face: {}}, {face: {3: 0.0}}, {face: {True: 0.0}}, {face: 0.0}]:
        with pytest.raises(ValueError, match="nonempty"):
            _boundary_flow(skeleton, (0, 0, 0), {}, components, 3)
    with pytest.raises(ValueError, match="overlap"):
        _boundary_flow(skeleton, (0, 0, 0), {face: (0, 0, 0)}, {face: {0: 0.0}}, 3)
    with pytest.raises(ValueError, match="only a gauge"):
        solve_flow_3d(mesh, traction={face: (0, 0, 0)}, mean_pressure=1.0)
    with pytest.raises(ValueError, match="belong"):
        solve_flow_3d(mesh, skeleton=TriangularSkeleton(TetraMesh.unit_cube()))
    with pytest.raises(ValueError, match="resolve"):
        solve_flow_3d(mesh, skeleton=TriangularSkeleton(mesh, 4), local_refinement=2)
    allfaces = {int(f): (0, 0, 0) for f in mesh.boundary_faces}
    for basis in [[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], np.eye(2), np.zeros((3, 0))]:
        with pytest.raises(ValueError, match="3-by-r"):
            solve_flow_3d(mesh, traction=allfaces, translation_kernel=basis)
    with pytest.raises(ValueError, match="resisted"):
        solve_flow_3d(mesh, traction=allfaces, drag=1.0, translation_kernel=np.eye(3))
    with pytest.raises(ValueError, match="unresisted"):
        solve_flow_3d(mesh, traction=allfaces, drag=1.0, mean_velocity=(1, 0, 0))


def test_local_pressure_constant_tests_enforce_macro_divergence():
    """Stabilization cannot alter constant-pressure divergence moments."""
    mesh = TetraMesh.unit_cube()
    solution = solve_flow_3d(mesh, formulation="usfem", degree=2, source=(1, 2, 3))
    bary, weights = tetrahedron_quadrature(4)
    for cell, fine in enumerate(solution.local_meshes):
        div = np.trace(solution.gradient(cell, bary), axis1=-2, axis2=-1)
        assert abs(fine.volumes @ (div @ weights)) < 2e-13


def test_tangential_advection_retains_translation_gauge():
    """Representable Robin traces preserve the three unresisted physical translations."""
    mesh = TetraMesh.unit_cube()

    def beta(points):
        """Transport has zero normal trace on every macroface and exterior face."""
        x, y, z = points.T
        return np.column_stack((x * (1 - x) * (x - y) * (x - z), np.zeros((len(x), 2))))

    def divergence(points):
        """Differentiate the quartic convection exactly."""
        x, y, z = points.T
        return (1 - 2 * x) * (x - y) * (x - z) + x * (1 - x) * (2 * x - y - z)

    traction = {
        int(face): lambda x, n=mesh.normals[face]: GRADIENT @ n - pressure(x)[:, None] * n
        for face in mesh.boundary_faces
    }
    result = solve_flow_3d(
        mesh,
        advection=beta,
        advection_divergence=divergence,
        quadrature_order=7,
        traction=traction,
        source=partial(forcing, gamma=np.zeros((3, 3)), beta=beta),
        mean_velocity=velocity(np.array([[0.5, 0.5, 0.5]]))[0],
    )
    assert result.l2_error(velocity, 4) < 2e-11
    assert result.pressure_l2_error(pressure, 4) < 2e-10


def test_declared_translation_must_preserve_every_dirichlet_component():
    """Partial traction permits only constants that leave prescribed components unchanged."""
    mesh = TetraMesh.unit_cube()
    components = {int(face): {2: 0.0} for face in mesh.boundary_faces}
    with pytest.raises(ValueError, match="Dirichlet component"):
        solve_flow_3d(
            mesh, traction_components=components, translation_kernel=[[1.0], [0.0], [0.0]]
        )


def test_boundary_tangency_preserves_planar_coordinates_without_erasing_small_flux():
    """Keep exact face coordinates while rejecting genuinely nonzero tiny normal transport."""
    from pymhm.rad3d import _boundary_tangent_3d

    skeleton = TriangularSkeleton(TetraMesh.unit_cube())

    def tangent(points):
        """Vanish in the normal component on the two x-normal faces."""
        return np.column_stack((points[:, 0] * (1 - points[:, 0]), np.zeros((len(points), 2))))

    assert _boundary_tangent_3d(skeleton, tangent, 8)
    assert not _boundary_tangent_3d(skeleton, (1e-30, 0.0, 0.0), 8)


def test_nonrepresentable_robin_translation_is_rejected_before_imposing_a_gauge():
    """Exterior tangency does not imply that internal quadratic Robin traces lie in P1."""
    mesh = TetraMesh.unit_cube()

    def beta(x):
        """Generate a quadratic internal normal trace with zero exterior normal component."""
        return np.column_stack((x[:, 0] * (1 - x[:, 0]), np.zeros((len(x), 2))))

    with pytest.raises(ValueError, match="representable normal advection"):
        solve_flow_3d(
            mesh,
            advection=beta,
            advection_divergence=lambda x: 1 - 2 * x[:, 0],
            traction={int(face): (0.0, 0.0, 0.0) for face in mesh.boundary_faces},
            mean_velocity=(1.0, 2.0, 3.0),
        )


@pytest.mark.parametrize("formulation,degree", [("taylor-hood", 2), ("usfem", 1), ("usfem", 2)])
def test_invisible_trace_modes_are_not_regularized(formulation, degree):
    """A deficient local/trace pairing fails instead of being repaired by a diagonal shift."""
    from pymhm.solvers import LinearSolveError

    with pytest.raises(LinearSolveError, match="numerical rank"):
        solve_flow_3d(
            TetraMesh.unit_cube(),
            formulation=formulation,
            degree=degree,
            local_refinement=1,
            source=(1.0, 2.0, 3.0),
        )


def test_quadratic_normal_advection_trace_retains_translations():
    """Bernstein P2 represents the internal quadratic Robin trace of constant velocity."""
    mesh = TetraMesh.unit_cube()
    skeleton = TriangularSkeleton(mesh, degree=2)

    def beta(x):
        """Tangential exterior convection with quadratic internal normal components."""
        return np.column_stack((x[:, 0] * (1 - x[:, 0]), np.zeros((len(x), 2))))

    def divergence(x):
        """Analytic divergence of the convection field."""
        return 1 - 2 * x[:, 0]

    def pressure(x):
        """Linear pressure whose physical traction fixes its mean."""
        return x.sum(axis=1) - 1.5

    traction = {}
    for face in mesh.boundary_faces:
        normal = mesh.normals[face]
        traction[int(face)] = lambda x, normal=normal: -pressure(x)[:, None] * normal
    result = solve_flow_3d(
        mesh,
        skeleton=skeleton,
        degree=3,
        formulation="taylor-hood",
        local_refinement=2,
        advection=beta,
        advection_divergence=divergence,
        source=(1.0, 1.0, 1.0),
        traction=traction,
        mean_velocity=(1.0, 2.0, 3.0),
    )
    assert result.l2_error((1.0, 2.0, 3.0), 5) < 3e-11
    assert result.pressure_l2_error(pressure, 5) < 3e-10
