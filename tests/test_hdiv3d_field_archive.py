"""Executed mixed-field contracts, independent physical replay and semantic corruption."""

from __future__ import annotations

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from threadpoolctl import threadpool_limits

from examples import core_extension_data as exact
from examples.hdiv3d_field_archive import (
    field_arrays,
    observe_system,
    original_checks,
    read_field,
    reference_tables,
    replay,
    write_field,
)
from examples.local_response_cache import array_identity
from pymhm._legacy.models.darcy.hdiv_3d import solve_darcy_hdiv3d
from pymhm.fem.hdiv.family_3d import cell_quadrature
from pymhm.io.provenance import file_digest
from pymhm.meshes.mixed import AffineMixedMesh


@pytest.fixture(
    scope="module",
    params=[("tetrahedron", 1, 1), ("tetrahedron", 2, 2), ("tetrahedron", 3, 1), ("prism", 2, 2)],
)
def acquired(request):
    """Capture every selected family under its actual production boundary convention."""
    kind, pressure, normal = request.param
    with threadpool_limits(1), observe_system() as observed:
        solution = solve_darcy_hdiv3d(
            AffineMixedMesh.unit_cube(1, kind),
            pressure_degree=pressure,
            normal_degree=normal,
            trace_degree=normal,
            local_refinement=1,
            source=exact.source3d,
            quadrature_order=12,
        )
        arrays = field_arrays(solution, observed, assembly_order=12, norm_orders=(6, 8))
    return solution, arrays


def test_literal_physical_replay_and_original_equations(acquired):
    """Literal tables/maps reproduce independently evaluated pressure, H(div) flux and div."""
    solution, arrays = acquired
    checks = original_checks(arrays)
    assert checks["original_full_saddle_relative_to_physical_rhs"] < 1e-10
    assert not checks["native_whole_field_agreement_verified"]
    assert (
        max(abs(row["physical_pressure_integral_minus_retained_pairing"]) for row in checks["rows"])
        < 1e-12
    )
    for order in (6, 8):
        points = cell_quadrature(solution.family.kind, order)[0]
        for cell in range(len(solution.local_meshes)):
            actual = replay(arrays, cell, order)
            expected = solution.evaluate(cell, points)
            for value, target in zip(actual, expected, strict=True):
                assert_allclose(value, target, rtol=2e-12, atol=2e-12)


def test_archive_round_trip_without_new_nullspace(acquired, tmp_path, monkeypatch):
    """Reading/replay consumes actual executed C/T rather than regenerating flux coordinates."""
    import pymhm.fem.hdiv.family_3d as legacy
    import pymhm.fem.hdiv.moments_3d as general

    _, arrays = acquired
    path = tmp_path / "field.npz"
    write_field(
        path, arrays, acquisition_uuid="actual-small-production", source_sha256={}, configuration={}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A new numerical nullspace is forbidden in replay")

    monkeypatch.setattr(legacy, "null_space", forbidden)
    monkeypatch.setattr(general, "coefficients", forbidden)
    restored, _ = read_field(path)
    for threads in (1, 2):
        with threadpool_limits(threads):
            for a, b in zip(replay(restored, 0, 8), replay(arrays, 0, 8), strict=True):
                assert np.array_equal(a, b)
            assert_allclose(
                reference_tables(restored, arrays["q8_points"])[0],
                arrays["q8_reference_flux"],
                rtol=1e-12,
                atol=1e-12,
            )


def test_rehashed_basis_table_corruption_is_rejected(acquired, tmp_path):
    """A coherent digest rewrite cannot turn a different polynomial basis into the producer's."""
    _, original = acquired
    arrays = {name: value.copy() for name, value in original.items()}
    path = tmp_path / "field.npz"
    write_field(
        path, arrays, acquisition_uuid="semantic-table-control", source_sha256={}, configuration={}
    )
    with np.load(path) as source:
        modified = {key: source[key] for key in source.files}
    modified["q8_reference_flux"][0, 0, 0] += 0.01
    np.savez_compressed(path, **modified)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["archive_sha256"] = file_digest(path)
    metadata["arrays_sha256"] = {key: array_identity(value) for key, value in modified.items()}
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="basis table"):
        read_field(path)


def test_rehashed_physical_coordinate_change_is_rejected(acquired, tmp_path):
    """Physical pressure/flux must retain the actual producer's scaled field relation."""
    _, original = acquired
    arrays = {name: value.copy() for name, value in original.items()}
    arrays["pressure_0"][0, 0] += 1
    with pytest.raises(ValueError, match="Physical fields"):
        write_field(
            tmp_path / "different.npz",
            arrays,
            acquisition_uuid="tamper",
            source_sha256={},
            configuration={},
        )


def test_incoherent_normal_orientation_is_rejected(acquired, tmp_path):
    """Normal orientation participates in the field contract independently of A/B/f."""
    _, original = acquired
    arrays = {name: value.copy() for name, value in original.items()}
    arrays["moment_transform_0"][0, 0, 0] *= -1
    with pytest.raises(ValueError, match="normal duality"):
        write_field(
            tmp_path / "orientation.npz",
            arrays,
            acquisition_uuid="orientation",
            source_sha256={},
            configuration={},
        )


def test_nonhomogeneous_pressure_and_physical_moment_pairing(acquired):
    """An affine pressure patch keeps its prescribed boundary and dimensional integral."""
    original, _ = acquired
    family = original.family

    def pressure(points):
        return points.sum(axis=1)

    with threadpool_limits(1), observe_system() as observed:
        solution = solve_darcy_hdiv3d(
            AffineMixedMesh.unit_cube(1, family.kind),
            pressure_degree=family.pressure_degree,
            normal_degree=family.normal_degree,
            trace_degree=family.normal_degree,
            local_refinement=1,
            dirichlet=pressure,
            quadrature_order=8,
        )
        arrays = field_arrays(solution, observed, assembly_order=8, norm_orders=(6, 8))
    checks = original_checks(arrays)
    assert checks["all_local_physical_rows_relative_to_source"] is None
    assert checks["original_full_saddle_relative_to_physical_rhs"] < 1e-10
    assert sum(row["physical_pressure_integral"] for row in checks["rows"]) == pytest.approx(
        1.5, abs=1e-11
    )
    for cell in range(len(solution.local_meshes)):
        scalar, flux, divergence = replay(arrays, cell, 8)
        reference = arrays["q8_points"]
        fine = solution.local_meshes[cell]
        expected = pressure(fine.geometry(reference).reshape(-1, 3)).reshape(scalar.shape)
        assert_allclose(scalar, expected, rtol=0, atol=1e-10)
        assert_allclose(flux, -np.ones(flux.shape), rtol=0, atol=1e-10)
        assert_allclose(divergence, 0, rtol=0, atol=1e-10)
