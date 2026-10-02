"""Mixed elasticity consistency, rigid modes and incompressible-limit regressions."""

import importlib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.elasticity import solve_displacement_pressure


@pytest.fixture(autouse=True)
def single_native_thread():
    """Keep small dense local factorizations deterministic and inexpensive."""
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def analytical_data(monkeypatch):
    """Reuse exact independently differentiated manufactured data."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("elasticity_data").ElasticityData


def affine_displacement(points):
    """Return a general affine field with nonzero strain and trace."""
    return points @ np.array([[0.5, -0.1], [0.2, 0.25]]) + [0.3, -0.4]


def solenoidal_affine(points):
    """Return an exactly trace-free affine displacement with nonzero strain."""
    return points @ np.array([[0.5, -0.1], [0.2, -0.5]]) + [0.3, -0.4]


def quadratic_solenoidal(points):
    """Return a quadratic field with a representable linear Cauchy traction."""
    return np.column_stack((points[:, 1] ** 2, -(points[:, 0] ** 2)))


@pytest.mark.parametrize(
    "formulation,degree",
    [("gals", 1), ("gals", 2), ("gals", 3), ("taylor-hood", 2), ("taylor-hood", 3)],
)
def test_general_affine_patch_and_physical_pressure_mean(formulation, degree):
    lam, mu = 3.0, 1.7
    gradient = np.array([[0.5, 0.2], [-0.1, 0.25]])
    pressure = -lam * np.trace(gradient)
    stress = mu * (gradient + gradient.T) - pressure * np.eye(2)
    result = solve_displacement_pressure(
        TriangleMesh.unit_square(),
        formulation=formulation,
        degree=degree,
        lame_lambda=lam,
        lame_mu=mu,
        dirichlet=affine_displacement,
        local_refinement=4 if degree == 1 else 2,
    )
    assert result.l2_error(affine_displacement) < 2e-13
    assert result.pressure_l2_error(pressure) < 3e-12
    assert result.h1_seminorm_error(gradient) < 2e-12
    assert result.stress_l2_error(stress) < 5e-12
    assert result.compressibility_l2() < 2e-12
    assert_allclose(result.hybrid.gauge_multipliers, 0.0, atol=2e-12)
    assert all(len(coarse) == 3 for coarse in result.hybrid.coarse)
    assert result.pressure_degree == (degree if formulation == "gals" else degree - 1)
    if formulation == "gals":
        assert min(result.stabilization) > 0
    else:
        assert_allclose(result.stabilization, 0.0)


@pytest.mark.parametrize(
    "formulation,degree", [("gals", 1), ("gals", 2), ("gals", 3), ("taylor-hood", 2)]
)
def test_incompressible_pressure_gradient_patch_and_gauge(formulation, degree):
    result = solve_displacement_pressure(
        TriangleMesh.unit_square(),
        formulation=formulation,
        degree=degree,
        lame_lambda=np.inf,
        source=(1.0, 2.0),
        mean_pressure=2.3,
        local_refinement=4 if degree == 1 else 2,
    )
    assert result.l2_error((0.0, 0.0)) < 2e-13
    assert result.pressure_l2_error(lambda x: x[:, 0] + 2 * x[:, 1] + 0.8) < 3e-12
    assert result.compressibility_l2() < 2e-12
    assert_allclose(result.hybrid.gauge_multipliers, 0.0, atol=2e-12)


@pytest.mark.parametrize(
    "formulation,degree", [("gals", 2), ("gals", 3), ("taylor-hood", 2), ("taylor-hood", 3)]
)
def test_quadratic_solution_exercises_strong_symmetric_gradient_and_hessian(formulation, degree):
    result = solve_displacement_pressure(
        TriangleMesh.unit_square(),
        formulation=formulation,
        degree=degree,
        lame_lambda=np.inf,
        source=(-2.0, 2.0),
        dirichlet=quadratic_solenoidal,
        local_refinement=2,
    )
    assert result.l2_error(quadratic_solenoidal) < 2e-13
    assert result.pressure_l2_error(0.0) < 3e-12
    assert result.compressibility_l2() < 2e-12


@pytest.mark.parametrize("formulation", ["gals", "taylor-hood"])
def test_cubic_solenoidal_patch_with_quadratic_skeleton(formulation):
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces), 2)

    def exact(x):
        """Evaluate the cubic solenoidal displacement."""
        return np.column_stack((x[:, 1] ** 3, -(x[:, 0] ** 3)))

    def force(x):
        """Evaluate the exact linear momentum load."""
        return np.column_stack((-6 * x[:, 1], 6 * x[:, 0]))

    result = solve_displacement_pressure(
        mesh,
        formulation=formulation,
        degree=3,
        lame_lambda=np.inf,
        source=force,
        dirichlet=exact,
        skeleton=space,
        local_refinement=2,
    )
    assert result.l2_error(exact) < 5e-13
    assert result.pressure_l2_error(0.0) < 5e-12


@pytest.mark.parametrize("pure_traction", [False, True])
@pytest.mark.parametrize("lame_lambda", [2.0, np.inf])
def test_physical_neumann_traction_and_rigid_displacement_moments(pure_traction, lame_lambda):
    mesh = TriangleMesh.unit_square()
    gradient = np.array([[0.5, 0.2], [-0.1, -0.5]])
    pressure = 2.3 if np.isinf(lame_lambda) else 0.0
    stress = gradient + gradient.T - pressure * np.eye(2)
    boundary_faces = mesh.boundary_faces if pure_traction else mesh.boundary_faces[:1]
    neumann = {}
    for face in boundary_faces:
        start, end = mesh.points[mesh.faces[face]]
        tangent = end - start
        normal = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
        neumann[int(face)] = stress @ normal
    mean = solenoidal_affine(np.array([[0.5, 0.5]]))[0]
    rotation_moment = (gradient[1, 0] - gradient[0, 1]) / 12
    result = solve_displacement_pressure(
        mesh,
        lame_lambda=lame_lambda,
        dirichlet=solenoidal_affine,
        neumann=neumann,
        rigid_moments=(*mean, rotation_moment),
        local_refinement=4,
    )
    assert result.l2_error(solenoidal_affine) < 3e-13
    assert result.pressure_l2_error(pressure) < 4e-12
    assert result.stress_l2_error(stress) < 5e-12
    assert_allclose(result.hybrid.gauge_multipliers, 0.0, atol=3e-12)


def test_incompatible_volume_flux_and_unbalanced_traction_are_rejected():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="incompatible"):
        solve_displacement_pressure(mesh, lame_lambda=np.inf, dirichlet=lambda x: x)
    with pytest.raises(ValueError, match="incompatible"):
        solve_displacement_pressure(
            mesh,
            source=(1.0, 0.0),
            neumann={int(f): (0.0, 0.0) for f in mesh.boundary_faces},
        )


@pytest.mark.parametrize("formulation,degree", [("gals", 1), ("gals", 2), ("taylor-hood", 2)])
def test_nonaffine_fields_converge_to_incompressible_limit_and_under_refinement(
    analytical_data, formulation, degree
):
    mesh = TriangleMesh.unit_square(2)
    results, errors = [], []
    for lame_lambda in (1.0, 1e4, 1e8, np.inf):
        exact = analytical_data(lame_lambda)
        result = solve_displacement_pressure(
            mesh,
            formulation=formulation,
            degree=degree,
            lame_lambda=lame_lambda,
            source=exact.source,
            local_refinement=4,
            quadrature_order=8,
        )
        results.append(result)
        errors.append(result.l2_error(exact.displacement, 8))
        assert result.hybrid.residual < 1e-12
        assert_allclose(result.hybrid.gauge_multipliers, 0.0, atol=2e-11)
    assert max(errors) < 2 * errors[0]
    limit = results[-1]
    distances = []
    for result in results[1:3]:
        distances.append(
            sum(np.linalg.norm(a - b) for a, b in zip(result.values, limit.values, strict=True))
        )
    # The data and the operator perturbation are both O(1/lambda).
    assert distances[1] < 2e-4 * distances[0]
    for field, reference in zip(results[-2].pressure, limit.pressure, strict=True):
        assert_allclose(field, reference, rtol=2e-7, atol=2e-7)
    exact = analytical_data(np.inf)
    fine = solve_displacement_pressure(
        TriangleMesh.unit_square(4),
        formulation=formulation,
        degree=degree,
        lame_lambda=np.inf,
        source=exact.source,
        local_refinement=4,
        quadrature_order=8,
    )
    assert fine.l2_error(exact.displacement, 8) < 0.4 * errors[-1]
    assert fine.pressure_l2_error(exact.pressure, 8) < 0.7 * limit.pressure_l2_error(
        exact.pressure, 8
    )
    assert fine.h1_seminorm_error(exact.gradient, 8) < 0.6 * limit.h1_seminorm_error(
        exact.gradient, 8
    )
    assert fine.stress_l2_error(exact.stress, 8) < 0.6 * limit.stress_l2_error(exact.stress, 8)
    assert fine.compressibility_l2(8) < 0.6 * limit.compressibility_l2(8)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_parallel_local_assembly_preserves_mixed_elasticity_fields(backend, analytical_data):
    data = analytical_data(np.inf)
    mesh = TriangleMesh.unit_square()
    options = {"lame_lambda": np.inf, "source": data.source, "local_refinement": 4}
    reference = solve_displacement_pressure(mesh, **options)
    result = solve_displacement_pressure(
        mesh,
        backend=backend,
        workers=2,
        **options,
    )
    assert_allclose(result.values, reference.values, rtol=1e-12, atol=1e-12)
    assert_allclose(result.pressure, reference.pressure, rtol=1e-12, atol=1e-12)
    assert_allclose(result.hybrid.trace, reference.hybrid.trace, rtol=1e-12, atol=2e-11)


def test_explicit_admissible_stabilization_and_strict_upper_bound():
    mesh = TriangleMesh.unit_square()
    default = solve_displacement_pressure(mesh)
    alpha = min(default.stabilization)
    result = solve_displacement_pressure(
        mesh,
        dirichlet=affine_displacement,
        stabilization_alpha=alpha / 2,
    )
    assert result.l2_error(affine_displacement) < 2e-13
    assert_allclose(result.stabilization, alpha / 2)
    with pytest.raises(ValueError, match="strictly between"):
        solve_displacement_pressure(mesh, stabilization_alpha=2 * alpha)


@pytest.mark.parametrize(
    "degree,trace_degree,refinement,segments,message",
    [
        (1, 1, 2, 1, "at least 4"),
        (2, 1, 1, 1, "at least 2"),
        (2, 2, 2, 1, "at least 3"),
        (3, 3, 1, 1, "at least 2"),
        (1, 2, 8, 1, "not be lower"),
        (1, 1, 4, 2, "at least 4"),
    ],
)
def test_underresolved_gals_trace_local_pairs_fail_before_solving(
    degree, trace_degree, refinement, segments, message
):
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(
        mesh, tuple(FaceSpace.uniform(trace_degree, segments) for _ in mesh.faces), 2
    )
    with pytest.raises(ValueError, match=message):
        solve_displacement_pressure(
            mesh, skeleton=space, degree=degree, local_refinement=refinement
        )


def test_unaligned_gals_trace_segments_are_rejected():
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(
        mesh, tuple(FaceSpace((0.0, 0.3000001, 1.0), (1, 1)) for _ in mesh.faces), 2
    )
    with pytest.raises(ValueError, match="aligned"):
        solve_displacement_pressure(mesh, skeleton=space, degree=2, local_refinement=10)


def test_high_order_local_space_permits_one_fine_interval_for_linear_trace():
    result = solve_displacement_pressure(
        TriangleMesh.unit_square(), degree=3, local_refinement=1, dirichlet=affine_displacement
    )
    assert result.l2_error(affine_displacement) < 2e-13
    assert result.pressure_l2_error(-0.75) < 2e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lame_lambda": 0.0},
        {"lame_lambda": -1.0},
        {"lame_lambda": np.nan},
        {"lame_mu": 0.0},
        {"lame_mu": np.inf},
        {"degree": 0},
        {"degree": 1.5},
        {"local_refinement": 0},
        {"quadrature_order": 0},
        {"formulation": "unknown"},
        {"formulation": "taylor-hood", "degree": 1},
        {"formulation": "taylor-hood", "degree": 2, "stabilization_alpha": 0.01},
        {"mean_pressure": 1.0},
        {"lame_lambda": np.inf, "mean_pressure": np.nan},
        {"stabilization_alpha": 0.0},
        {"stabilization_alpha": -0.1},
        {"stabilization_alpha": np.nan},
        {"stabilization_alpha": np.inf},
        {"stabilization_alpha": 1e6},
    ],
)
def test_invalid_material_spaces_gauges_and_stabilization_are_rejected(kwargs):
    options = {"local_refinement": 4} | kwargs
    with pytest.raises(ValueError):
        solve_displacement_pressure(TriangleMesh.unit_square(), **options)


def test_invalid_skeleton_rigid_moments_and_tensor_data_are_rejected():
    mesh = TriangleMesh.unit_square()
    for space in (SkeletonSpace(mesh), SkeletonSpace(TriangleMesh.unit_square(), components=2)):
        with pytest.raises(ValueError, match="skeleton"):
            solve_displacement_pressure(mesh, skeleton=space)
    traction = {int(f): (0.0, 0.0) for f in mesh.boundary_faces}
    with pytest.raises(ValueError, match="full displacement boundary"):
        solve_displacement_pressure(mesh, lame_lambda=np.inf, neumann=traction, mean_pressure=1.0)
    for moments in ([1, 2], [0, 0, np.nan]):
        with pytest.raises(ValueError, match="rigid_moments"):
            solve_displacement_pressure(mesh, neumann=traction, rigid_moments=moments)
    result = solve_displacement_pressure(mesh, local_refinement=4)
    for field in (np.full((2, 2), np.nan), np.eye(2) * 1j):
        with pytest.raises(ValueError, match="finite and real"):
            result.h1_seminorm_error(field)
        with pytest.raises(ValueError, match="finite and real"):
            result.stress_l2_error(lambda x, value=field: value)
