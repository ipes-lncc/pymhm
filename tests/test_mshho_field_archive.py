"""Physical polynomial, moment, orientation and executed-basis replay invariants."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.local_response_cache import array_identity
from examples.mshho_field_archive import (
    ROOT,
    attach_mhm,
    field_arrays,
    read_field,
    replay,
    restore,
    validate_arrays,
    write_field,
)
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.io.provenance import current_source_manifest, file_digest
from pymhm.methods.hho import MsHHOSolution, solve_mshho


@pytest.fixture(scope="module")
def solution() -> MsHHOSolution:
    """An affine physical patch with nonzero boundary data and anisotropic diffusion."""
    mesh = TriangleMesh.unit_square(1)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
    return solve_mshho(
        mesh,
        skeleton=skeleton,
        cell_degree=1,
        degree=3,
        local_refinement=2,
        permeability=np.diag([2.0, 3.0]),
        dirichlet=lambda x: x[:, 0] + 2 * x[:, 1],
    )


@pytest.fixture(scope="module")
def captured_arrays(solution: MsHHOSolution) -> dict[str, np.ndarray]:
    """Capture the executed P3 tables once independently of reader corruption tests."""
    arrays = {name: value.copy() for name, value in field_arrays(solution, 4).items()}
    for value in arrays.values():
        value.setflags(write=False)
    return arrays


@pytest.fixture
def executed_arrays(captured_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keep each reader's writable arrays separate from the immutable acquisition."""
    return {name: value.copy() for name, value in captured_arrays.items()}


def sources() -> dict[str, str]:
    """Use an actual checked-in lock and numerical owner in the fixture contract."""
    return current_source_manifest(
        {name: file_digest(ROOT / name) for name in ("pixi.lock", "src/pymhm/methods/hho.py")}
    )


def test_affine_pressure_gradient_physical_flux_and_thread_replay(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray]
) -> None:
    """Actual archived tables preserve one-sided physical fields across BLAS counts."""
    arrays = executed_arrays
    validate_arrays(arrays)
    with threadpool_limits(1):
        first = [replay(arrays, cell) for cell in range(len(solution.local))]
    with threadpool_limits(2):
        second = [replay(arrays, cell) for cell in range(len(solution.local))]
    for cell, (pressure, gradient, flux) in enumerate(first):
        points = arrays[f"physical_quadrature_points_{cell}"]
        assert np.allclose(pressure, points[..., 0] + 2 * points[..., 1], rtol=0, atol=3e-13)
        assert np.allclose(gradient, [1, 2], rtol=0, atol=3e-12)
        assert np.allclose(flux, [-2, -6], rtol=0, atol=8e-12)
        for a, b in zip(first[cell], second[cell], strict=True):
            assert np.array_equal(a, b)


def test_coherent_sign_change_of_executed_moment_basis(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray]
) -> None:
    """Stored matrices and their coordinates transform together; physical fields agree."""
    arrays = executed_arrays
    altered = {name: value.copy() for name, value in arrays.items()}
    altered.update(precision_fields("face_moments", -restore(arrays, "face_moments")))
    for cell in range(len(solution.local)):
        altered[f"executed_moments_{cell}"] *= -1
        for name in ("executed_reconstruction", "executed_load"):
            altered.update(precision_fields(f"{name}_{cell}", -restore(arrays, f"{name}_{cell}")))
        altered.update(
            precision_fields(f"cell_moments_{cell}", -restore(arrays, f"cell_moments_{cell}"))
        )
    validate_arrays(altered)
    for cell in range(len(solution.local)):
        for a, b in zip(replay(arrays, cell), replay(altered, cell), strict=True):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("order", ["C", "F"])
def test_archive_memory_layout_preserves_literal_field_replay(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray], order: str
) -> None:
    """C/F storage of identical archived numbers preserves extended contractions."""
    arrays = executed_arrays
    altered = {name: np.array(value, order=order, copy=True) for name, value in arrays.items()}
    validate_arrays(altered)
    with threadpool_limits(1):
        expected = [replay(arrays, cell) for cell in range(len(solution.local))]
    with threadpool_limits(2):
        actual = [replay(altered, cell) for cell in range(len(solution.local))]
    for first, second in zip(expected, actual, strict=True):
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)


def test_coherent_nodal_permutation(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray]
) -> None:
    """Actual nodes, matrices, coefficients and dof maps retain the same polynomial."""
    arrays = executed_arrays
    altered = {name: value.copy() for name, value in arrays.items()}
    for cell in range(len(solution.local)):
        permutation = np.arange(len(arrays[f"nodes_{cell}"]))[::-1]
        inverse = np.argsort(permutation)
        for name in ("nodes", "executed_moments"):
            altered[f"{name}_{cell}"] = arrays[f"{name}_{cell}"][permutation]
        altered.update(
            precision_fields(
                f"executed_reconstruction_{cell}",
                restore(arrays, f"executed_reconstruction_{cell}")[permutation],
            )
        )
        altered[f"nodal_dofs_{cell}"] = inverse[arrays[f"nodal_dofs_{cell}"]]
        altered.update(
            precision_fields(f"pressure_{cell}", restore(arrays, f"pressure_{cell}")[permutation])
        )
    validate_arrays(altered)
    for cell in range(len(solution.local)):
        for a, b in zip(replay(arrays, cell), replay(altered, cell), strict=True):
            assert np.array_equal(a, b)


def test_atomic_field_contract_and_changed_basis_with_rehashed_archive(
    solution: MsHHOSolution, tmp_path: Path
) -> None:
    """A recomputed checksum does not validate a matrix with incompatible coordinates."""
    path = tmp_path / "field.npz"
    record = write_field(
        path,
        solution,
        {"physical_patch": "affine"},
        acquisition_uuid="fixture",
        source_sha256=sources(),
        order=4,
    )
    arrays, loaded = read_field(path)
    assert loaded == record
    arrays["executed_reconstruction_0"][0, 0] += 1
    np.savez_compressed(path, **arrays)
    record["archive_sha256"] = file_digest(path)
    record["array_sha256"] = {name: array_identity(value) for name, value in arrays.items()}
    path.with_suffix(".json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="reconstruction matrix"):
        read_field(path)


@pytest.mark.parametrize("order", [0, 3, True])
def test_excluded_quadrature(solution: MsHHOSolution, order: int) -> None:
    """Reject nonintegral or insufficient squared-polynomial rules."""
    with pytest.raises(ValueError):
        field_arrays(solution, order)


def test_excluded_coefficients_coordinates_and_missing_sources(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Invalid basis membership, geometry and actual numerical-source identities fail."""
    with pytest.raises(ValueError, match="source variant"):
        field_arrays(replace(solution, source_variant="unknown"), 4)
    broken = list(solution.pressure)
    broken[0] = broken[0] + 1
    with pytest.raises(ValueError, match="energy reconstruction"):
        field_arrays(replace(solution, pressure=tuple(broken)), 4)
    arrays = executed_arrays
    arrays["nodes_0"][0] += 0.1
    with pytest.raises(ValueError, match="physical cardinal"):
        validate_arrays(arrays)
    with pytest.raises(ValueError, match="actual acquisition"):
        write_field(tmp_path / "empty.npz", solution, {}, acquisition_uuid="", source_sha256={})
    with pytest.raises(ValueError, match="numerical source"):
        write_field(
            tmp_path / "changed.npz",
            solution,
            {},
            acquisition_uuid="fixture",
            source_sha256={"pixi.lock": "0" * 64},
        )


def test_excluded_digest_and_duplicate_output(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray], tmp_path: Path
) -> None:
    """Completed field bytes and output identities cannot be silently replaced."""
    path = tmp_path / "field.npz"
    write_field(path, solution, {}, acquisition_uuid="fixture", source_sha256=sources(), order=4)
    with pytest.raises(ValueError, match="fresh"):
        write_field(
            path, solution, {}, acquisition_uuid="fixture", source_sha256=sources(), order=4
        )
    with path.open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="archive digest"):
        read_field(path)
    with pytest.raises(ValueError, match="outside"):
        replay(executed_arrays, len(solution.local))


def test_executed_cardinal_tables_against_native_basix(
    executed_arrays: dict[str, np.ndarray],
) -> None:
    """Independently ordered native P3 values and derivatives match the persisted basis."""
    basix = pytest.importorskip("basix")
    arrays = executed_arrays
    element = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 3, basix.LagrangeVariant.equispaced
    )
    nodes = arrays["cardinal_multiindices"][:, 1:] / 3
    permutation = [
        int(np.argmin(np.linalg.norm(element.points - point, axis=1))) for point in nodes
    ]
    assert len(set(permutation)) == len(nodes)
    table = element.tabulate(1, arrays["quadrature_barycentric"][:, 1:])[..., 0][:, :, permutation]
    assert np.allclose(table[0], arrays["executed_cardinal_values"], rtol=0, atol=5e-14)
    derivative = arrays["executed_cardinal_derivatives"]
    assert np.allclose(table[1], derivative[..., 1] - derivative[..., 0], rtol=0, atol=7e-14)
    assert np.allclose(table[2], derivative[..., 2] - derivative[..., 0], rtol=0, atol=7e-14)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("degree", np.asarray(3.5)),
        ("local_count", np.asarray(True)),
        ("quadrature_order", np.asarray(3)),
        ("cell_count_0", np.asarray(3.0)),
    ],
)
def test_reject_changed_discrete_degree_and_quadrature_headers(
    executed_arrays: dict[str, np.ndarray], name: str, value: np.ndarray
) -> None:
    """A coefficient vector cannot silently change space or its integration rule."""
    arrays = executed_arrays
    arrays[name] = value
    with pytest.raises(ValueError):
        validate_arrays(arrays)


@pytest.mark.parametrize(
    ("name", "change"),
    [
        ("physical_barycentric_gradients_0", 0),
        ("areas_0", -1),
        ("physical_quadrature_points_0", 0),
        ("permeability_tensors_0", np.nan),
    ],
)
def test_reject_incompatible_physical_evaluation_maps(
    executed_arrays: dict[str, np.ndarray], name: str, change: float
) -> None:
    """Invalid physical maps cannot pass through a small algebraic moment residual."""
    arrays = executed_arrays
    arrays[name].fill(change)
    with pytest.raises(ValueError):
        validate_arrays(arrays)


@pytest.mark.parametrize("name", ["executed_cardinal_values", "executed_cardinal_derivatives"])
def test_reject_rehashed_cardinal_tables(executed_arrays: dict[str, np.ndarray], name: str) -> None:
    """A checksum update cannot change the declared P3 physical interpolation map."""
    arrays = executed_arrays
    arrays[name].flat[0] += 0.1
    with pytest.raises(ValueError, match="declared Pk"):
        validate_arrays(arrays)


def test_both_original_saddles_for_anisotropic_affine_boundary(
    solution: MsHHOSolution, executed_arrays: dict[str, np.ndarray]
) -> None:
    """Both methods retain pressure and physical outward traces on the complete patch."""
    arrays = executed_arrays
    dual = solve_darcy(
        solution.skeleton.mesh,
        skeleton=solution.skeleton,
        degree=3,
        local_refinement=2,
        permeability=np.diag([2.0, 3.0]),
        dirichlet=lambda x: x[:, 0] + 2 * x[:, 1],
    )
    diagnostics = attach_mhm(arrays, dual, lambda x: x[:, 0] + 2 * x[:, 1])
    for method in ("mhm", "mshho"):
        assert diagnostics[f"{method}_original_relative_residual"] < 1e-10
    assert np.allclose(restore(arrays, "mhm_trace"), restore(arrays, "mshho_trace"), atol=2e-11)
    for cell in range(2):
        assert np.allclose(
            restore(arrays, f"mhm_pressure_{cell}"), solution.pressure[cell], atol=2e-12
        )
    seen = np.zeros(solution.skeleton.size, dtype=bool)
    for cell, local in enumerate(solution.local):
        ids = solution.skeleton.cell_dofs(cell)
        coordinates = np.r_[
            restore(arrays, f"cell_moments_{cell}"), restore(arrays, "face_moments")[ids]
        ]
        derivative = local.energy @ coordinates - local.load
        signs = np.repeat(solution.skeleton.mesh.signs[cell], 2)
        expected = -signs * derivative[local.cell_count :]
        first = ~seen[ids]
        assert np.array_equal(restore(arrays, "mshho_trace")[ids[first]], expected[first])
        seen[ids] = True
