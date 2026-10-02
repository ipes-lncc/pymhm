"""Mixed elasticity patches, physical balance, convergence and incompressibility."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pymhm.bdm import bdm2_evaluate
from pymhm.elasticity_mixed import _local_problem, solve_elasticity_mixed
from pymhm.elements import triangle_quadrature
from pymhm.mesh import FaceSpace, SkeletonSpace, TriangleMesh


def affine(points):
    """An affine displacement with extension, shear and rigid rotation."""
    x, y = points.T
    return np.column_stack((1 + 2 * x - y, -2 + x + 3 * y))


def quadratic(points):
    """A non-affine solenoidal displacement whose stress is exactly linear."""
    x, y = points.T
    return np.column_stack((x * x, -2 * x * y))


def quadratic_stress(points):
    """Cauchy stress for the quadratic patch with mu=1, independently differentiated."""
    x, y = points.T
    result = np.zeros((len(points), 2, 2))
    result[:, 0, 0], result[:, 1, 1] = 4 * x, -4 * x
    result[:, 0, 1] = result[:, 1, 0] = -2 * y
    return result


def smooth(points):
    """A trigonometric divergence-free displacement."""
    x, y = np.pi * points.T
    return np.column_stack((np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)))


def smooth_stress(points):
    """Exact diagonal Cauchy stress of the smooth solenoidal displacement."""
    result = np.zeros((len(points), 2, 2))
    result[:, 0, 0] = 2 * np.pi * np.cos(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])
    result[:, 1, 1] = -result[:, 0, 0]
    return result


def test_three_rigid_modes_are_complete_and_exact():
    """The local constrained Neumann operator has exactly the three rigid modes."""
    mesh = TriangleMesh.unit_square()
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2)
    problem = _local_problem(mesh, 0, mesh.submesh(0, 1), skeleton, 2.0, 1.0, (0.0, 0.0), 4)
    matrix = problem.matrix.toarray()
    assert_allclose(matrix, matrix.T, atol=1e-14)
    assert_allclose(matrix @ problem.kernel, 0, atol=2e-14)
    assert np.linalg.matrix_rank(matrix) == len(matrix) - 3
    assert np.linalg.eigvalsh(problem.constraints.T @ problem.kernel).min() > 0


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_affine_patch_and_signed_traction(backend):
    result = solve_elasticity_mixed(
        TriangleMesh.unit_square(), dirichlet=affine, backend=backend, workers=2
    )
    assert result.l2_error(affine) < 2e-12
    assert result.stress_l2_error([[9, 0], [0, 11]]) < 3e-12
    assert result.rotation_l2_error(-1) < 2e-12
    assert_allclose(result.equilibrium_residuals(), 0, atol=3e-12)
    for method in (
        result.fine_force_residuals,
        result.weak_symmetry_residuals,
        result.normal_traction_residuals,
    ):
        for residual in method():
            assert_allclose(residual, 0, atol=3e-12)


@pytest.mark.parametrize("lame_lambda", [1.0, 1e3, 1e6, np.inf])
def test_nonaffine_patch_is_exactly_projected_and_does_not_lock(lame_lambda):
    """BDM2 contains the linear stress and DG1 gives the L2 displacement projection."""
    result = solve_elasticity_mixed(
        TriangleMesh.unit_square(2), dirichlet=quadratic, source=(-2, 0), lame_lambda=lame_lambda
    )
    assert result.stress_l2_error(quadratic_stress) < 1e-8
    assert result.rotation_l2_error(lambda x: x[:, 1]) < 2e-12
    assert result.divergence_l2_error([2, 0]) < 3e-12
    bary, weights = triangle_quadrature(5)
    reference_mass = np.einsum("q,qi,qj->ij", weights, bary, bary)
    for mesh, displacement in zip(result.local_meshes, result.displacement, strict=True):
        points = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells])
        moments = np.einsum(
            "q,qi,tqa->tia", weights, bary, quadratic(points.reshape(-1, 2)).reshape(points.shape)
        )
        projection = np.linalg.solve(reference_mass, moments.transpose(1, 0, 2).reshape(3, -1))
        projection = projection.reshape(3, len(mesh.cells), 2).transpose(1, 0, 2)
        assert_allclose(displacement, projection, atol=2e-12, rtol=0)
    assert_allclose(result.equilibrium_residuals(), 0, atol=3e-12)


def test_smooth_convergence_remains_uniform_near_incompressibility():
    """Refinement reduces errors at both ordinary and nearly incompressible moduli."""
    errors = []
    for lam in (1.0, 1e6, np.inf):
        sequence = []
        for n in (1, 2, 4):
            result = solve_elasticity_mixed(
                TriangleMesh.unit_square(n),
                dirichlet=smooth,
                source=lambda x: 2 * np.pi**2 * smooth(x),
                lame_lambda=lam,
                quadrature_order=6,
            )
            sequence.append((result.l2_error(smooth, 8), result.stress_l2_error(smooth_stress, 8)))
        errors.append(sequence)
    errors = np.asarray(errors)
    assert np.all(errors[:, 2] < errors[:, 1] / 2.5)
    assert np.all(errors[:, 1] < errors[:, 0] / 1.8)
    assert np.all(errors[1:] < 1.1 * errors[0])


def test_pure_traction_gauges_and_partial_traction_patch():
    """Physical Cauchy tractions use the opposite sign to the skeletal multiplier."""
    mesh = TriangleMesh.unit_square()
    traction = {int(f): np.array([[9, 0], [0, 11]]) @ mesh.normals[f] for f in mesh.boundary_faces}
    # u's moments against translations and centered rigid rotation on the unit square.
    moments = (1.5, 0.0, 1 / 6)
    pure = solve_elasticity_mixed(mesh, traction=traction, rigid_moments=moments)
    assert pure.l2_error(affine) < 3e-12
    assert pure.stress_l2_error([[9, 0], [0, 11]]) < 3e-12
    partial = {next(iter(traction)): lambda x: np.broadcast_to([0, -11], (len(x), 2))}
    mixed = solve_elasticity_mixed(mesh, traction=partial, dirichlet=affine)
    assert mixed.l2_error(affine) < 3e-12
    with pytest.raises(ValueError, match="incompatible"):
        solve_elasticity_mixed(mesh, traction=traction, source=(1.0, 0.0))
    for targets in ([0, 0], [0, 0, np.inf], [0, 0, 1j]):
        with pytest.raises(ValueError, match="rigid_moments"):
            solve_elasticity_mixed(mesh, traction=traction, rigid_moments=targets)


def test_heterogeneous_aligned_lame_coefficients():
    """A layered uniaxial stress stays continuous while strain changes by modulus."""
    mesh = TriangleMesh.unit_square(2)

    def mu(x):
        return np.where(x[:, 0] < 0.5, 1.0, 100.0)

    def displacement(x):
        return np.column_stack(
            (np.where(x[:, 0] <= 0.5, x[:, 0] / 2, 0.25 + (x[:, 0] - 0.5) / 200), np.zeros(len(x)))
        )

    result = solve_elasticity_mixed(mesh, lame_lambda=0.0, lame_mu=mu, dirichlet=displacement)
    assert result.l2_error(displacement) < 2e-12
    assert result.stress_l2_error([[1, 0], [0, 0]]) < 3e-11


def test_custom_segmented_traces_and_weak_symmetry_is_not_pointwise():
    mesh = TriangleMesh.unit_square()
    space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1, 2) for _ in mesh.faces), components=2)
    result = solve_elasticity_mixed(
        mesh,
        skeleton=space,
        dirichlet=smooth,
        source=lambda x: 2 * np.pi**2 * smooth(x),
        quadrature_order=6,
    )
    bary, _ = triangle_quadrature(5)
    skew = []
    for fine, field in zip(result.local_meshes, result.stress, strict=True):
        values, _ = bdm2_evaluate(fine, field, bary)
        skew.append(np.max(np.abs(values[..., 0, 1] - values[..., 1, 0])))
    assert max(skew) > 1e-3
    for residual in result.weak_symmetry_residuals():
        assert_allclose(residual, 0, atol=2e-12)
    for target in ([[np.nan, 0], [0, 1]], [[1j, 0], [0, 1]]):
        with pytest.raises(ValueError, match="stress"):
            result.stress_l2_error(target)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"local_refinement": 0},
        {"quadrature_order": 2},
        {"lame_lambda": -1},
        {"lame_mu": 0},
        {"lame_lambda": np.nan},
        {"lame_lambda": 1j},
        {"lame_mu": np.nan},
        {"source": (np.nan, 0)},
    ],
)
def test_invalid_coefficients_and_discretization(kwargs):
    with pytest.raises(ValueError):
        solve_elasticity_mixed(TriangleMesh.unit_square(), **kwargs)


def test_invalid_skeletons_and_misaligned_segments():
    mesh = TriangleMesh.unit_square()
    for skeleton in (SkeletonSpace(mesh), SkeletonSpace(TriangleMesh.unit_square(), components=2)):
        with pytest.raises(ValueError, match="skeleton"):
            solve_elasticity_mixed(mesh, skeleton=skeleton)
    for face in (FaceSpace.uniform(3), FaceSpace((0.0, 0.500004, 1.0), (1, 1))):
        skeleton = SkeletonSpace(mesh, tuple(face for _ in mesh.faces), components=2)
        with pytest.raises(ValueError, match="aligned"):
            solve_elasticity_mixed(mesh, skeleton=skeleton)


def test_both_spatial_lame_fields_preserve_affine_patch():
    """Variable Lamé fields need the derivative of stress in the source term."""

    def displacement(x):
        return np.column_stack((x[:, 0], np.zeros(len(x))))

    def stress(x):
        value = np.zeros((len(x), 2, 2))
        value[:, 0, 0] = 5 + x[:, 0] + 2 * x[:, 1]
        value[:, 1, 1] = 1 + x[:, 0]
        return value

    result = solve_elasticity_mixed(
        TriangleMesh.unit_square(2),
        lame_lambda=lambda x: 1 + x[:, 0],
        lame_mu=lambda x: 2 + x[:, 1],
        dirichlet=displacement,
        source=(-1, 0),
    )
    assert result.l2_error(displacement) < 2e-12
    assert result.stress_l2_error(stress) < 3e-12
    for defect in result.fine_force_residuals():
        assert_allclose(defect, 0, atol=2e-12)


@pytest.mark.parametrize("boundary", ["displacement", "mixed", "traction"])
def test_incompressible_hydrostatic_stress_and_physical_gauge(boundary):
    """Pressure-driven equilibrium keeps a global, not a per-macro, pressure gauge."""
    mesh = TriangleMesh.unit_square(2)

    def stress(x):
        return -(x[:, 0] + 2 * x[:, 1] + 1.5)[:, None, None] * np.eye(2)

    traction = {
        int(face): lambda x, face=face: stress(x) @ mesh.normals[face]
        for face in mesh.boundary_faces
    }
    options = {"mean_pressure": 3.0}
    if boundary == "mixed":
        options = {"traction": {next(iter(traction)): traction[next(iter(traction))]}}
    elif boundary == "traction":
        options = {"traction": traction}
    result = solve_elasticity_mixed(mesh, lame_lambda=np.inf, source=(1.0, 2.0), **options)
    assert result.l2_error((0.0, 0.0)) < 2e-12
    assert result.stress_l2_error(stress) < 3e-11
    assert result.rotation_l2_error(0.0) < 2e-12


def test_mixed_elasticity_physical_pressure_constraint_validation():
    mesh = TriangleMesh.unit_square()
    with pytest.raises(ValueError, match="finite"):
        solve_elasticity_mixed(mesh, mean_pressure=np.nan)
    with pytest.raises(ValueError, match="gauge"):
        solve_elasticity_mixed(mesh, mean_pressure=1.0)
    with pytest.raises(ValueError, match="gauge"):
        solve_elasticity_mixed(
            mesh, traction={int(mesh.boundary_faces[0]): (0, 0)}, mean_pressure=1
        )
    with pytest.raises(ValueError, match="incompatible"):
        solve_elasticity_mixed(mesh, lame_lambda=np.inf, dirichlet=lambda x: x)


def test_mixed_stress_homogeneous_boundary_converges_to_exact_incompressible_limit(monkeypatch):
    """A fixed-load bubble isolates locking from cancellation of nonzero boundary fluxes."""
    import importlib
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    exact = importlib.import_module("elasticity_data").ElasticityData(pressure_amplitude=0.0)
    mesh = TriangleMesh.unit_square(2)
    results = [
        solve_elasticity_mixed(mesh, lame_lambda=lam, source=exact.source, quadrature_order=8)
        for lam in [1.0, 1e6, 1e12, np.inf]
    ]
    errors = [result.l2_error(exact.displacement, 8) for result in results]
    assert max(errors) < 1.1 * errors[0]
    for actual, limit in zip(results[-2].stress, results[-1].stress, strict=True):
        assert_allclose(actual, limit, rtol=0, atol=2e-10)
    for actual, limit in zip(results[-2].displacement, results[-1].displacement, strict=True):
        assert_allclose(actual, limit, rtol=0, atol=2e-11)


def test_bulk_compliance_avoids_overflow_in_the_sum_of_large_moduli():
    from pymhm.elasticity_mixed import _bulk_compliance

    with np.errstate(over="raise", invalid="raise"):
        value = _bulk_compliance(1e308, np.array([1e308]), np.zeros((1, 2)))
    assert_allclose(value / 1e-308, 0.25, rtol=3e-15, atol=0)
