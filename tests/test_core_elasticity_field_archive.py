"""Actual anisotropic operator/basis contracts and independent physical invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from threadpoolctl import threadpool_limits

from examples import core_extension_data as exact
from examples.campaign_provenance import file_digest
from examples.core_elasticity_field_archive import (
    _csr,
    _put_csr,
    basis_checks,
    field_arrays,
    observe_system,
    original_checks,
    read_field,
    replay,
    restore,
    write_field,
)
from examples.local_response_cache import array_identity
from examples.verify_core_elasticity import CASES, physical_errors, run, solve_case


@pytest.fixture(scope="module", params=CASES)
def acquired(request):
    """Capture each geometry and its current nonhomogeneous physical problem."""
    with threadpool_limits(1), observe_system() as observed:
        solution = solve_case(request.param, 1)
        arrays = field_arrays(solution, observed)
    return request.param, solution, arrays


def test_original_physical_blocks_and_anisotropic_mean(acquired):
    name, solution, arrays = acquired
    result = original_checks(arrays)
    assert result["accepted"]
    assert result["full_uncondensed_relative_to_physical_rhs"] < 1e-10
    assert result["physical_mean_relative_defect"] < 1e-12
    assert int(arrays["gauge_count"]) == 1
    assert int(arrays["local_count"]) == (1 if name == "rectangle-rt1" else 2)
    # Integral div(u)= integral(2*x*y^2+x^3) over the unit square is 7/12.
    # This independent physical identity uses anisotropic A, not an isotropic bulk proxy.
    volume = np.longdouble(0)
    for cell in range(int(arrays["local_count"])):
        _, sigma, _, _ = replay(arrays, cell, 10)
        strain = np.einsum("ij,tqj->tqi", exact.compliance(), sigma.reshape(*sigma.shape[:2], 4))
        volume += np.sum(
            arrays[f"areas_{cell}"][:, None]
            * arrays["q10_weights"]
            * (strain[..., 0] + strain[..., 3]),
            dtype=np.longdouble,
        )
    assert_allclose(volume, 7 / 12, rtol=2e-12, atol=2e-12)
    assert max(np.max(abs(v)) for v in solution.weak_symmetry_residuals()) < 1e-11
    assert max(np.max(abs(v)) for v in solution.fine_force_residuals()) < 1e-11
    assert max(np.max(abs(v)) for v in solution.normal_traction_residuals()) < 1e-10


def test_replay_own_bases_matches_producer_fields_and_norms(acquired):
    name, solution, arrays = acquired
    for order in (9, 10):
        reference = arrays[f"q{order}_reference_points"]
        for cell in range(int(arrays["local_count"])):
            u, sigma, div, rot = replay(arrays, cell, order)
            if name == "rectangle-rt1":
                expected = solution.evaluate(cell, reference)
            else:
                from pymhm.lagrange import reference_basis

                basis = reference_basis(1, reference)[0]
                expected_sigma, expected_div = solution.family.evaluate(
                    solution.local_meshes[cell], solution.stress[cell], reference
                )
                expected = (
                    np.einsum("qi,tia->tqa", basis, solution.displacement[cell]),
                    expected_sigma,
                    expected_div,
                    solution.rotation[cell] @ basis.T,
                )
            for actual, target in zip((u, sigma, div, rot), expected, strict=True):
                assert_allclose(actual, target, rtol=2e-12, atol=2e-12)
        actual_norms = physical_errors(arrays, order)
        if name == "rectangle-rt1":
            expected_norms = solution.errors(
                exact.displacement, exact.stress, lambda x: -exact.force(x), exact.rotation, order
            )
        else:
            expected_norms = {
                "displacement_l2": solution.l2_error(exact.displacement, order),
                "stress_l2": solution.stress_l2_error(exact.stress, order),
                "divergence_l2": solution.divergence_l2_error(lambda x: -exact.force(x), order),
                "rotation_l2": solution.rotation_l2_error(exact.rotation, order),
            }
        for key in actual_norms:
            assert_allclose(actual_norms[key], expected_norms[key], rtol=2e-12, atol=2e-12)
    assert max(basis_checks(arrays).values()) < 1e-10


def test_reader_and_replay_require_no_new_basis_or_solve(acquired, tmp_path, monkeypatch):
    import pymhm.bdm_family as bdm
    import pymhm.tensor_rt as rt
    from pymhm.hybrid import HybridSystem

    _, _, arrays = acquired
    path = tmp_path / "actual.npz"
    write_field(path, arrays, acquisition_uuid="executed-test", source_sha256={}, configuration={})

    def forbidden(*args, **kwargs):
        raise AssertionError("A new numerical basis/solve must not enter archive replay")

    monkeypatch.setattr(bdm, "_full_dual", forbidden)
    monkeypatch.setattr(bdm.BDMFamily, "basis", forbidden)
    monkeypatch.setattr(rt, "tensor_rt_basis", forbidden)
    monkeypatch.setattr(HybridSystem, "solve", forbidden)
    restored, metadata = read_field(path)
    assert not metadata["native_whole_field_agreement_verified"]
    for cell in range(int(arrays["local_count"])):
        for actual, expected in zip(
            replay(restored, cell, 9), replay(arrays, cell, 9), strict=True
        ):
            assert np.array_equal(actual, expected)


def _rewrite(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez(stream, **arrays)
    sidecar = path.with_suffix(".json")
    record = json.loads(sidecar.read_text())
    record["archive_sha256"] = file_digest(path)
    record["arrays_sha256"] = {key: array_identity(value) for key, value in arrays.items()}
    sidecar.write_text(json.dumps(record))


@pytest.mark.parametrize(
    "mutation",
    [
        "load",
        "retained_basis",
        "boundary_sign",
        "physical_field",
        "face_basis",
        "rotation_basis",
        "norm_stress_basis",
        "norm_divergence_basis",
        "piola",
        "nonfinite",
        "uuid",
        "precision",
        "sparse_index",
    ],
)
def test_rehashed_semantic_corruption_rejected(acquired, tmp_path, mutation):
    _, _, arrays = acquired
    path = tmp_path / "field.npz"
    write_field(
        path, arrays, acquisition_uuid="original-acquisition", source_sha256={}, configuration={}
    )
    with np.load(path, allow_pickle=False) as archive:
        corrupted = {key: archive[key].copy() for key in archive.files}
    if mutation == "load":
        corrupted["original_f_0"][0] += 1
    elif mutation == "retained_basis":
        corrupted["retained_basis_e_0"][0, 0] += 1
    elif mutation == "boundary_sign":
        corrupted["applied_boundary_load"] *= -1
    elif mutation == "physical_field":
        corrupted["stress_0"][0, 0] += 1
    elif mutation == "face_basis":
        corrupted["face_stress_basis_0_0"][0, :, 0] += corrupted[
            "physical_outward_measure_normals_0"
        ][0, 0]
    elif mutation == "rotation_basis":
        corrupted["q9_rotation_basis_0"][:, 0] += 1
    elif mutation == "norm_stress_basis":
        corrupted["q9_stress_basis_0"][0, :, 8, 0] += 1
    elif mutation == "norm_divergence_basis":
        corrupted["q9_divergence_basis_0"][0, :, 8] += 1
    elif mutation == "piola":
        largest = np.unravel_index(
            np.argmax(abs(corrupted["jacobian_0"])), corrupted["jacobian_0"].shape
        )
        corrupted["jacobian_0"][largest] += 1
    elif mutation == "nonfinite":
        corrupted["q9_stress_basis_0"].flat[0] = np.nan
    elif mutation == "uuid":
        corrupted["acquisition_uuid"][0] += 1
    elif mutation == "precision":
        corrupted["coefficient_nmant"] += 1
    else:
        corrupted["original_a_0_indices"][0] = 10**6
    _rewrite(path, corrupted)
    with pytest.raises(ValueError):
        read_field(path)


@pytest.mark.parametrize("name", CASES)
def test_observation_preserves_operating_bytes_and_homogeneous_solution(name):
    """Instrumentation changes neither A/B/f/Z/C/E nor the actual solved coefficients."""
    with threadpool_limits(1):
        plain = solve_case(name, 1)
        with observe_system() as observed:
            instrumented = solve_case(name, 1)
            arrays = field_arrays(instrumented, observed)
        assert np.array_equal(plain.hybrid.trace, instrumented.hybrid.trace)
        for actual, expected in zip(plain.hybrid.fields, instrumented.hybrid.fields, strict=True):
            assert np.array_equal(actual, expected)
        with observe_system() as observed:
            zero = solve_case(name, 1, homogeneous=True)
            homogeneous = field_arrays(zero, observed)
        assert not np.any(zero.hybrid.trace)
        assert all(not np.any(field) for field in zero.hybrid.fields)
        assert original_checks(homogeneous)["full_uncondensed_relative_to_physical_rhs"] == 0
        assert original_checks(arrays)["accepted"]


@pytest.mark.parametrize("name", CASES)
def test_fresh_blas_thread_replay_bytes(name):
    captures = []
    for threads in (1, 2):
        with threadpool_limits(threads), observe_system() as observed:
            solution = solve_case(name, 1)
            captures.append(field_arrays(solution, observed))
    first, second = captures
    assert first.keys() == second.keys()
    assert {key: array_identity(value) for key, value in first.items()} == {
        key: array_identity(value) for key, value in second.items()
    }
    for cell in range(int(first["local_count"])):
        for actual, expected in zip(replay(first, cell, 10), replay(second, cell, 10), strict=True):
            assert np.array_equal(actual, expected)


def test_independently_differentiated_force_and_rotation():
    points = np.array([[0.23, 0.37], [0.71, 0.61]])
    h = 2**-18
    directions = np.eye(2) * h
    ds = [(exact.stress(points + d) - exact.stress(points - d)) / (2 * h) for d in directions]
    assert_allclose(
        -ds[0][..., :, 0] - ds[1][..., :, 1], exact.force(points), rtol=3e-10, atol=3e-10
    )
    du = [
        (exact.displacement(points + d) - exact.displacement(points - d)) / (2 * h)
        for d in directions
    ]
    assert_allclose((du[1][:, 0] - du[0][:, 1]) / 2, exact.rotation(points), rtol=3e-10, atol=3e-10)


def test_coherent_retained_rotation_preserves_fields(acquired):
    """Coherent equivalent local retained coordinates leave physical replay invariant."""
    from examples.archive_precision import precision_fields

    _, _, arrays = acquired
    changed = {key: value.copy() for key, value in arrays.items()}
    # Orthogonal signed permutation has exactly represented entries and fixes no field.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transforms = [
        sparse.eye(int(arrays["trace_size"])),
        *[sparse.csr_matrix(rotation) for _ in range(int(arrays["local_count"]))],
    ]
    transform = sparse.block_diag(transforms, format="csr")
    _put_csr(changed, "global_matrix", transform.T @ _csr(arrays, "global_matrix") @ transform)
    changed.update(precision_fields("global_rhs", transform.T @ restore(arrays, "global_rhs")))
    changed.update(precision_fields("gauge_rows", restore(arrays, "gauge_rows") @ transform))
    for cell in range(int(arrays["local_count"])):
        for name in ("kernel_z", "coarse_basis_z", "retained_basis_e", "test_basis"):
            changed.update(
                precision_fields(f"{name}_{cell}", restore(arrays, f"{name}_{cell}") @ rotation)
            )
        changed.update(
            precision_fields(f"coarse_{cell}", rotation.T @ restore(arrays, f"coarse_{cell}"))
        )
    assert original_checks(changed)["accepted"]
    for cell in range(int(arrays["local_count"])):
        for actual, expected in zip(replay(changed, cell, 9), replay(arrays, cell, 9), strict=True):
            assert np.array_equal(actual, expected)


def test_context_restores_shared_owners_on_failure():
    from pymhm.hybrid import HybridSystem

    original = HybridSystem.solve
    with pytest.raises(RuntimeError), observe_system():
        raise RuntimeError("deliberate producer failure")
    assert HybridSystem.solve is original


def test_finite_kernel_and_gauge_ensure_small_case_uniqueness(acquired):
    """Check finite ranks without claiming a mesh-uniform inf-sup estimate."""
    _, _, arrays = acquired
    with threadpool_limits(1):
        for cell in range(int(arrays["local_count"])):
            matrix = _csr(arrays, f"original_a_{cell}").toarray()
            assert len(matrix) - np.linalg.matrix_rank(matrix) == 3
            assert (
                np.linalg.matrix_rank(restore(arrays, f"original_b_{cell}").astype(float))
                == restore(arrays, f"original_b_{cell}").shape[1]
            )
        matrix = _csr(arrays, "global_matrix").toarray()
        # Finite positive compliance fixes hydrostatic stress. This volume row
        # preserves a physical identity; it does not repair an incompressible kernel.
        assert np.linalg.matrix_rank(matrix) == len(matrix)
        row = restore(arrays, "gauge_rows").astype(float)
        augmented = np.block([[matrix, row.T], [row, np.zeros((1, 1))]])
        assert np.linalg.matrix_rank(augmented) == len(augmented)


def test_negative_cauchy_traction_and_three_physical_rigid_means():
    """A constant Cauchy stress on pure traction boundaries fixes physical sign and gauge."""
    from pymhm.elasticity_mixed import solve_elasticity_mixed
    from pymhm.mesh import TriangleMesh

    mesh = TriangleMesh.unit_square()
    traction = {int(face): mesh.normals[face] for face in mesh.boundary_faces}
    with threadpool_limits(1), observe_system() as observed:
        solution = solve_elasticity_mixed(
            mesh,
            compliance=exact.compliance(),
            traction=traction,
            quadrature_order=8,
            local_refinement=1,
        )
        arrays = field_arrays(solution, observed)
    assert int(arrays["gauge_count"]) == 3
    assert original_checks(arrays)["accepted"]
    strain = (exact.compliance() @ np.array([1.0, 0.0, 0.0, 1.0])).reshape(2, 2)
    for cell in range(int(arrays["local_count"])):
        displacement, stress, divergence, rotation = replay(arrays, cell, 10)
        expected_u = (arrays[f"q10_physical_points_{cell}"] - 0.5) @ strain.T
        assert_allclose(displacement, expected_u, rtol=2e-11, atol=2e-11)
        assert_allclose(stress, np.broadcast_to(np.eye(2), stress.shape), rtol=2e-11, atol=2e-11)
        assert_allclose(divergence, 0, atol=2e-11)
        assert_allclose(rotation, 0, atol=2e-11)
    for face in mesh.boundary_faces:
        start, end = arrays["trace_offsets"][face : face + 2]
        face_trace = restore(arrays, "global_trace")[start:end].reshape(-1, 2)
        assert_allclose(face_trace[0], -mesh.normals[face], atol=2e-12)
        assert_allclose(face_trace[1:], 0, atol=2e-12)


def test_current_acquisition_all_three_families(tmp_path):
    with threadpool_limits(1):
        result = run([1], CASES, tmp_path / "fresh")
    assert result["complete"] and not result["source_changed"]
    assert len(result["rows"]) == 3
    assert len(result["fields"]) == 3
    assert not result["literal_literature_reproduction"]
    assert not result["native_whole_field_agreement_verified"]
    for row in result["rows"]:
        assert max(row["norm_quadrature_relative_changes"].values()) < 1e-9
        read_field(tmp_path / "fresh" / row["archive"])


@pytest.mark.parametrize(
    "levels,names",
    [([3], CASES), ([1, 1], CASES), ([1], [CASES[0], CASES[0]]), ([1], ["unsupported"])],
)
def test_configuration_validation_precedes_acquisition(tmp_path, levels, names):
    with pytest.raises(ValueError):
        run(levels, names, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists()
