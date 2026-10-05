"""Original point-functional equations and persisted numerical field bases."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from examples.quarter_point_archive import (
    matrix_digest,
    point_field_arrays,
    replay_point_fields,
    write_point_archive,
    write_point_record,
)
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.darcy.primal import solve_darcy
from pymhm.linalg.linear import LinearSolveError


def point_solution(formulation: str = "primal", *, refinement: int = 2, segments: int = 1):
    """Use the exact unit Dirac functional and an independently stated fitted layer."""
    mesh = TriangleMesh.unit_square(2)
    return solve_darcy(
        mesh,
        point_sources=np.array([[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]]),
        permeability=lambda x: np.where(x[:, 1] < 0.5, 1000.0, 1.0),
        neumann={int(face): 0 for face in mesh.boundary_faces},
        formulation=formulation,
        degree=2 if formulation == "primal" else 1,
        local_refinement=refinement,
        skeleton=SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces)),
    )


@pytest.fixture(scope="module")
def point_solutions():
    """Acquire the two distinct point-source formulations once with physical gauges."""
    return {formulation: point_solution(formulation) for formulation in ("primal", "mixed")}


@pytest.fixture(scope="module")
def point_fields(point_solutions):
    """Capture each formulation's executed replay tables before any corruption test."""
    result = {}
    for formulation, solution in point_solutions.items():
        arrays, record = point_field_arrays(solution)
        captured = {name: value.copy() for name, value in arrays.items()}
        for value in captured.values():
            value.setflags(write=False)
        result[formulation] = captured, record
    return result


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_point_basis_replay_threads_and_original_saddle(
    tmp_path, point_solutions, point_fields, formulation
):
    """Actual archived matrices reproduce fields without recomputed orientations."""
    solution = point_solutions[formulation]
    arrays, record = point_fields[formulation]
    assert record["original_saddle_relative_load_residual"] <= 1e-10
    assert abs(record["physical_pressure_integral"]) < 1e-10
    assert_allclose(arrays["point_sources"][:, 2].sum(), 0, atol=0)
    np.testing.assert_array_equal(arrays["macro_signs"], solution.skeleton.mesh.signs)
    path = tmp_path / "field.npz"
    digest = write_point_archive(path, arrays)
    assert len(digest) == 64 and not path.with_suffix(".npz.pending").exists()
    with np.load(path) as data:
        archived = dict(data)
    replays = []
    for threads in (1, 2):
        with threadpool_limits(threads):
            replay = replay_point_fields(archived, record)
        assert_allclose(replay[0], arrays["pressure_quadrature"], rtol=2e-14, atol=2e-14)
        assert_allclose(replay[1], arrays["flux_quadrature"], rtol=2e-14, atol=2e-14)
        replays.append(replay)
    for first, second in zip(*replays, strict=True):
        assert_allclose(first, second, rtol=2e-14, atol=2e-14)
    for key in ("macro_signs", "pressure_cell_dofs"):
        damaged = {**archived, key: archived[key].copy()}
        damaged[key].flat[0] += 1
        with pytest.raises(ValueError, match="basis digest"):
            replay_point_fields(damaged, record)


def test_rt0_equivalent_rotated_executed_basis_replays_same_field(point_fields):
    """Coefficient rotation is paired with the archived execution matrix."""
    arrays, record = point_fields["mixed"]
    rng = np.random.default_rng(17)
    width = arrays["flux_basis_values"].shape[2]
    rotation = np.linalg.qr(rng.standard_normal((width, width)))[0]
    rotated = arrays.copy()
    rotated["flux_basis_values"] = np.einsum("tqia,ij->tqja", arrays["flux_basis_values"], rotation)
    rotated["flux_coefficients"] = (
        arrays["flux_coefficients"].reshape(-1, width) @ rotation
    ).ravel()
    recorded = {
        **record,
        "basis_sha256": {
            **record["basis_sha256"],
            "flux_basis_values": matrix_digest(rotated["flux_basis_values"]),
        },
    }
    pressure, flux = replay_point_fields(rotated, recorded)
    assert_allclose(pressure, arrays["pressure_quadrature"], rtol=0, atol=0)
    assert_allclose(flux, arrays["flux_quadrature"], rtol=2e-14, atol=2e-14)


def test_original_point_saddle_rejects_wrong_field_and_gauge(point_solutions):
    """A reduced residual cannot certify inconsistent physical reconstruction."""
    solution = point_solutions["primal"]
    fields = list(solution.hybrid.fields)
    fields[0] = fields[0] + 1
    inconsistent = replace(
        solution, pressure=tuple(fields), hybrid=replace(solution.hybrid, fields=tuple(fields))
    )
    with pytest.raises(LinearSolveError, match="original point-well saddle"):
        point_field_arrays(inconsistent)
    shifted = tuple(field + 1 for field in solution.hybrid.fields)
    wrong_gauge = replace(
        solution, pressure=shifted, hybrid=replace(solution.hybrid, fields=shifted)
    )
    with pytest.raises(LinearSolveError, match="pressure integral"):
        point_field_arrays(wrong_gauge)
    trace = solution.hybrid.trace.copy()
    trace[solution.skeleton.dofs(int(solution.skeleton.mesh.boundary_faces[0]))] = 1
    wrong_boundary = replace(solution, hybrid=replace(solution.hybrid, trace=trace))
    with pytest.raises(ValueError, match="exterior physical flux"):
        point_field_arrays(wrong_boundary)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_point_archive_rejects_coefficients_outside_checked_field(point_solutions, formulation):
    """The equations must certify exactly the coefficients written to the archive."""
    solution = point_solutions[formulation]
    pressure = (solution.pressure[0] + 1, *solution.pressure[1:])
    with pytest.raises(ValueError, match="pressure coefficients"):
        point_field_arrays(replace(solution, pressure=pressure))
    if formulation == "mixed":
        flux = (solution.flux[0] + 1, *solution.flux[1:])
        with pytest.raises(ValueError, match="flux coefficients"):
            point_field_arrays(replace(solution, flux=flux))


def test_point_archive_requires_declared_local_and_trace_spaces(point_solutions):
    """An archive cannot silently relabel other degrees, partitions or sources."""
    solution = point_solutions["primal"]
    with pytest.raises(ValueError, match="P2"):
        point_field_arrays(replace(solution, degree=1))
    with pytest.raises(ValueError, match="two local subdivisions"):
        point_field_arrays(point_solution(refinement=1))
    with pytest.raises(ValueError, match="constant trace"):
        point_field_arrays(point_solution(segments=2))
    with pytest.raises(ValueError, match="point-source allocations"):
        point_field_arrays(replace(solution, point_sources=()))
    with pytest.raises(ValueError, match="quadrature"):
        point_field_arrays(solution, order=2)
    arrays, record = point_field_arrays(solution, order=3)
    assert record["norm_order"] == 3
    pressure, flux = replay_point_fields(arrays, record)
    assert_allclose(pressure, arrays["pressure_quadrature"], rtol=2e-14, atol=2e-14)
    assert_allclose(flux, arrays["flux_quadrature"], rtol=2e-14, atol=2e-14)


def test_point_metadata_is_atomic_and_rejects_nonfinite_numbers(tmp_path):
    """Invalid metadata cannot replace the accepted field record."""
    import json

    path = tmp_path / "field.json"
    write_point_record(path, {"archive": "field.npz", "residual": 1e-14})
    original = path.read_bytes()
    assert json.loads(original)["archive"] == "field.npz"
    assert not path.with_suffix(".json.pending").exists()
    with pytest.raises(ValueError):
        write_point_record(path, {"residual": float("nan")})
    assert path.read_bytes() == original
    assert not path.with_suffix(".json.pending").exists()
