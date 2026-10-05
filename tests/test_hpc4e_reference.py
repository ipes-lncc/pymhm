"""Polynomial, archive and integration controls for the native HPC4E example."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


def _example(name: str) -> ModuleType:
    """Load an original example and its declared peers without changing sys.path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_example("hpc4e_data")
driver = _example("solve_hpc4e_reference")


def zero_field(nx: int, ny: int, degree: int = 1) -> driver.ReferenceField:
    """Return the identically zero canonical mixed field on a declared grid."""
    count, k = nx * ny, degree
    return driver.ReferenceField(
        np.zeros((count, 2 * (k + 1) * (k + 2), 2)),
        np.zeros((count, (k + 1) ** 2, 2)),
        np.zeros((count, (k + 1) * (k + 2) // 2)),
        nx,
        ny,
        degree,
    )


def test_archive_and_exact_constant_norm(tmp_path: Path) -> None:
    """Integrate independent constant displacement/rotation and roundtrip coefficients."""
    fine, coarse = zero_field(4, 6), zero_field(2, 3)
    fine.displacement[:, 0] = [3, 4]
    fine.rotation[:, 0] = 2
    result = driver.difference(fine, coarse)
    assert_allclose(result["displacement_l2"], 5 * np.sqrt(0.45), atol=2e-14)
    assert_allclose(result["rotation_l2"], 2 * np.sqrt(0.45), atol=2e-14)
    assert result["stress_relative"] == 0
    assert result["displacement_relative"] == 1
    assert not np.isinf(driver.difference(coarse, zero_field(1, 1))["stress_relative"])
    path = tmp_path / "field.npz"
    fine.save(path)
    assert_allclose(driver.load_field(path).displacement, fine.displacement)
    assert all(value == 0 for value in driver.difference(coarse, coarse).values())
    assert np.isinf(driver.difference(zero_field(4, 6), fine)["rotation_relative"])


def test_reference_contracts() -> None:
    """Reject invalid arrays, out-of-domain samples and inexact/non-nested norm grids."""
    field = zero_field(2, 3)
    for points in ([[[-1, 0]]], [[np.nan, 0]], [[-0.2, 0.1]]):
        with pytest.raises(ValueError):
            field.evaluate(np.asarray(points))
    with pytest.raises(ValueError, match="nested"):
        driver.difference(zero_field(3, 4), field)
    with pytest.raises(ValueError, match="quadrature"):
        driver.difference(field, field, order=1)
    with pytest.raises(ValueError, match="degree"):
        driver.ReferenceField(field.stress, field.displacement, field.rotation, 2, 3, 0)
    with pytest.raises(ValueError, match="shapes"):
        driver.ReferenceField(field.stress * np.nan, field.displacement, field.rotation, 2, 3, 1)
    with pytest.raises(ValueError, match="bounds"):
        driver.ReferenceField(
            field.stress, field.displacement, field.rotation, 2, 3, 1, (1, 0, 0, 1)
        )


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2])
def test_native_mixed_polynomial_patch(degree: int) -> None:
    """Verify RT stress, weak rotation and exact displacement L2 projection independently."""
    pytest.importorskip("dolfinx")
    field, record = driver.solve(4, 2, degree, None, threads=1)
    points = np.random.default_rng(293).random((239, 2)) * [1, 0.45]
    x, y = points.T
    stress, displacement, rotation = field.evaluate(points)
    expected_stress = np.stack((np.column_stack((8 * y, x)), np.column_stack((x, 10 * y))), axis=1)
    hy = 0.45 / field.ny
    yc = (np.floor(y / hy) + 0.5) * hy
    expected_y = y**2 if degree == 2 else 2 * yc * y - yc**2 + hy**2 / 12
    assert_allclose(stress, expected_stress, atol=2e-12)
    assert_allclose(displacement, np.column_stack((x * y, expected_y)), atol=2e-13)
    assert_allclose(rotation, x / 2, atol=2e-13)
    assert record["residual"] < 1e-12
    assert record["equilibrium_relative"] < 1e-12
    assert record["weak_symmetry_moment_max"] < 1e-13
    assert record["conversion_relative"] < 1e-12
    if degree == 2:
        coarse, _ = driver.solve(4, 2, 1, None, threads=1)
        norms = driver.difference(field, coarse)
        assert norms["stress_l2"] < 2e-13
        assert norms["rotation_l2"] < 2e-13
        assert_allclose(norms["displacement_l2"], hy**2 * np.sqrt(0.45 / 180), atol=2e-14)


@pytest.mark.fem
@pytest.mark.parametrize("degree", [1, 2])
def test_native_layered_gravity_equilibrium(degree: int) -> None:
    """Check two material layers, vertical orientation and exact clamped gravity solution."""
    pytest.importorskip("dolfinx")
    data = driver.HPC4EData(
        np.tile([1e8, 2e8], (2, 1)), np.zeros((2, 2)), np.tile([1000.0, 2000.0], (2, 1))
    )
    field, record = driver.solve(4, 4, degree, data, threads=1)
    points = np.random.default_rng(673).random((191, 2)) * [1, 0.45]
    y = points[:, 1]
    lower, h, gravity = y < 0.225, 0.225, 9.81e-4
    sigma = np.where(lower, -gravity * 3000 * h + 1000 * gravity * y, -gravity * 2000 * (0.45 - y))
    base = -gravity * 3000 * h * h + 500 * gravity * h**2
    exact_u = np.where(
        lower,
        -gravity * 3000 * h * y + 500 * gravity * y**2,
        base + gravity * 1000 * ((y * y - h * h) / 2 - 0.45 * (y - h)),
    )
    if degree == 1:
        hy = 0.45 / 4
        yc = (np.floor(y / hy) + 0.5) * hy
        exact_u -= 500 * gravity * ((y - yc) ** 2 - hy**2 / 12)
    stress, displacement, rotation = field.evaluate(points)
    expected = np.zeros_like(stress)
    expected[:, 1, 1] = sigma
    assert_allclose(stress, expected, atol=2e-12)
    assert_allclose(displacement, np.column_stack((np.zeros_like(y), exact_u)), atol=2e-13)
    assert_allclose(rotation, 0, atol=2e-13)
    assert record["residual"] < 1e-12
    assert record["energy_work_relative"] < 1e-12
    assert record["equilibrium_relative"] < 1e-12


def test_compliance_energy_has_physical_scaling_and_no_implicit_orthogonality() -> None:
    """Integrate a constant nonsymmetric stress exactly without declaring it admissible."""
    complementary_energy = _example("hpc4e_comparison").complementary_energy

    reference = zero_field(2, 3)
    other = zero_field(2, 3)
    one = driver.CartesianMacroMesh(bounds=(0, 0.5, 0, 0.15))
    points, _ = driver.quadrilateral_quadrature(3)
    basis = driver.tensor_rt_basis(one, 1, 0, points)[0][0]
    sigma = np.array([[2.0, 1.0], [-1.0, 3.0]])
    matrix = basis.transpose(0, 2, 1).reshape(-1, basis.shape[1])
    samples = np.broadcast_to(sigma.T, (len(points), 2, 2)).reshape(-1, 2)
    reference.stress[:] = np.linalg.lstsq(matrix, samples, rcond=None)[0]
    other.stress[:] = 2 * reference.stress
    data = driver.HPC4EData(np.full((1, 1), 1e8), np.full((1, 1), 0.25), np.ones((1, 1)))
    result = complementary_energy(reference, other, data)
    energy = (np.sum(sigma**2) - 0.25 * np.trace(sigma) ** 2) / 0.8 * 0.45
    assert_allclose(
        [
            result["reference_energy"],
            result["comparison_energy"],
            result["cross_energy"],
            result["stress_compliance_difference_squared"],
        ],
        energy * np.array([1, 4, 2, 1]),
        atol=3e-14,
    )
    assert_allclose(result["identity_defect"], -2 * energy, atol=3e-14)
    assert abs(result["polarization_defect"]) < 1e-14
    same = complementary_energy(reference, reference, data, order=6)
    assert same["identity_relative_energy"] < 1e-14
    with pytest.raises(ValueError, match="same bounds and local spaces"):
        complementary_energy(reference, zero_field(2, 3, 2), data)


def test_isolated_algebra_archive_and_original_residual(tmp_path: Path) -> None:
    """Solve a scaled saddle archive and verify serialization and the original equations."""
    from scipy import sparse

    algebra = _example("solve_hpc4e_algebra")
    matrix = sparse.csr_matrix([[0.0, 1e5, 0], [1e5, 1e-3, 2], [0, 2, -3]])
    expected = np.array([1.0, 2.0, 3.0])
    rhs = matrix @ expected
    matrix_path, rhs_path = tmp_path / "matrix.npz", tmp_path / "rhs.npy"
    solution_path = tmp_path / "solution.npy"
    sparse.save_npz(matrix_path, matrix)
    np.save(rhs_path, rhs)
    record = algebra.solve_archive(
        matrix_path, rhs_path, solution_path, solver="scipy", equilibration="symmetric", threads=1
    )
    assert_allclose(np.load(solution_path), expected, rtol=2e-12)
    assert record["original_rhs_relative_residual"] < 1e-12
    assert record["matrix_sha256"] == algebra._digest(matrix_path)
    assert all("filepath" not in pool for pool in record["native_threadpools"])
    assert algebra.relative_residual(matrix, np.zeros(3), np.zeros(3)) == 0
    with pytest.raises(ValueError, match="threads"):
        algebra.solve_archive(matrix_path, rhs_path, solution_path, threads=0)
    np.save(rhs_path, rhs[:, None])
    with pytest.raises(ValueError, match="right-hand-side"):
        algebra.solve_archive(matrix_path, rhs_path, solution_path, solver="scipy")


@pytest.fixture
def hpc4e_native_lifecycle(monkeypatch):
    """Observe genuine PETSc handles while retaining caller and unrelated options."""
    pytest.importorskip("dolfinx")
    PETSc = pytest.importorskip("petsc4py.PETSc")

    handles = []
    own = driver._own_petsc

    def tracked(stack, handle):
        """Keep a Python reference so destruction cannot be attributed to garbage collection."""
        handles.append(handle)
        return own(stack, handle)

    monkeypatch.setattr(driver, "_own_petsc", tracked)
    options = PETSc.Options()
    seed = "hpc4e_reference_mat_mumps_icntl_14"
    unrelated = "hpc4e_cleanup_unrelated"
    original = options.getAll()
    options[seed] = 777
    options[unrelated] = "preserve-me"
    expected = options.getAll()
    try:
        yield handles, options, expected
    finally:
        for key in (seed, unrelated):
            if key in original:
                options[key] = original[key]
            else:
                del options[key]


def _assert_petsc_released(handles, options, expected):
    """Require every owned native object released and exactly the original option values."""
    assert len(handles) >= 11
    assert all(handle.handle == 0 for handle in handles)
    actual = options.getAll()
    keys = {key for key in set(actual) | set(expected) if key.startswith("hpc4e_")}
    assert {key: actual[key] for key in keys if key in actual} == {
        key: expected[key] for key in keys if key in expected
    }


@pytest.mark.fem
def test_native_ldlt_equilibration_cleanup_success(hpc4e_native_lifecycle) -> None:
    """The scaled LDLt route preserves the exact quadratic field and cleans successful solves."""
    handles, options, expected = hpc4e_native_lifecycle
    field, record = driver.solve(
        4, 2, 2, None, threads=1, factorization="ldlt", equilibration="symmetric"
    )
    points = np.array([[0.17, 0.12], [0.63, 0.31], [0.93, 0.43]])
    x, y = points.T
    stress, displacement, rotation = field.evaluate(points)
    sigma = np.stack((np.column_stack((8 * y, x)), np.column_stack((x, 10 * y))), axis=1)
    assert_allclose(stress, sigma, atol=2e-12)
    assert_allclose(displacement, np.column_stack((x * y, y * y)), atol=2e-13)
    assert_allclose(rotation, x / 2, atol=2e-13)
    assert record["residual"] < 1e-12
    assert record["algebra_refinement_rtol"] == 1e-11
    _assert_petsc_released(handles, options, expected)


@pytest.mark.fem
def test_native_export_failure_releases_factors_and_options(
    hpc4e_native_lifecycle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure after factorization and physical checks still destroys all resources."""
    handles, options, expected = hpc4e_native_lifecycle

    def failure(*args):
        """Inject a conversion failure after real matrices, vectors and factors are allocated."""
        raise RuntimeError("injected physical export failure")

    monkeypatch.setattr(driver, "_export", failure)
    with pytest.raises(RuntimeError, match="injected physical export failure"):
        driver.solve(4, 2, 1, None, threads=1, factorization="ldlt", equilibration="symmetric")
    _assert_petsc_released(handles, options, expected)


@pytest.mark.serial
@pytest.mark.fem
@pytest.mark.mpi
@pytest.mark.parametrize("ranks", [2, 4])
def test_native_distributed_reference_matches_serial(ranks: int) -> None:
    """Exercise actual MPI ownership, boundary elimination, Ruiz scaling and field export."""
    import os
    import shutil
    import subprocess

    pytest.importorskip("dolfinx")
    MPI = pytest.importorskip("mpi4py.MPI")
    if MPI.COMM_WORLD.size > 1:
        pytest.skip("the subprocess integration must start outside an existing MPI job")
    launcher = Path(sys.executable).parent / "mpiexec"
    command = str(launcher) if launcher.exists() else shutil.which("mpiexec")
    if command is None:
        pytest.skip("a native MPI launcher is required")
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1")
    result = subprocess.run(
        [
            command,
            "-n",
            str(ranks),
            sys.executable,
            str(Path(__file__).with_name("hpc4e_mpi_cases.py")),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS on {ranks} ranks" in result.stdout


def test_sparsity_count_identifies_the_serial_index_boundary() -> None:
    """Check exact small patterns and the fine-grid32-bit CSR overflow before assembly."""
    assert driver.sparsity_estimate(4, 3, 1) == {
        "native_dofs": 352,
        "structural_nonzeros": 14428,
    }
    assert driver.sparsity_estimate(16, 8, 2) == {
        "native_dofs": 7824,
        "structural_nonzeros": 655200,
    }
    count = driver.sparsity_estimate(1024, 512, 2)
    assert count == {"native_dofs": 31466496, "structural_nonzeros": 2680215552}
    assert count["native_dofs"] < np.iinfo(np.int32).max < count["structural_nonzeros"]
    with pytest.raises(ValueError, match="positive"):
        driver.sparsity_estimate(0, 4, 2)
