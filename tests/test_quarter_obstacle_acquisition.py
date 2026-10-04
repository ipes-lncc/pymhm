"""Scientific contracts of the complete finite-well square-obstacle acquisition."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm import FaceSpace, SkeletonSpace
from pymhm.fem.scalar.triangle import trace_coupling


@pytest.fixture
def acquisition(monkeypatch):
    """Import the normal example namespace used by portable spawned workers."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.solve_quarter_obstacle")


def test_primal_trace_condition_has_the_claimed_discrete_boundary(acquisition):
    """An interior fine-edge node detects each segment; the excluded cyclic pair has a kernel."""
    macro = acquisition.macro_mesh()
    for refinement, segments, expected_rank in ((2, 1, 3), (4, 2, 6), (2, 2, 5)):
        skeleton = SkeletonSpace(macro, tuple(FaceSpace.uniform(0, segments) for _ in macro.faces))
        coupling = trace_coupling(macro, 0, macro.submesh(0, refinement), skeleton, 1)
        assert np.linalg.matrix_rank(coupling) == expected_rank
        if (refinement, segments) == (2, 2):
            with pytest.raises(ValueError, match="two fine edges"):
                acquisition.ObstacleConfiguration(refinement, segments)
        else:
            acquisition.ObstacleConfiguration(refinement, segments)
    acquisition.ObstacleConfiguration(2, 2, "mixed")


@pytest.mark.parametrize("refinement,segments", [(3, 1), (4, 3)])
def test_unresolved_material_or_unaligned_trace_is_rejected(acquisition, refinement, segments):
    with pytest.raises(ValueError, match="even refinement and aligned"):
        acquisition.ObstacleConfiguration(refinement, segments)


@pytest.mark.parametrize("formulation", ["primal", "mixed"])
def test_original_acquisition_preserves_full_case_and_executed_basis(
    acquisition, tmp_path, formulation
):
    """Check all 200 macros, source strengths, physical gauge and archived basis contract."""
    config = acquisition.ObstacleConfiguration(2, 1, formulation)
    row = acquisition.acquire(config, tmp_path)
    checks = row["physical_checks"]
    assert row["macro_cells"] == 200
    assert row["fine_cells"] == 800
    assert checks["original_free_equations_relative_load_residual"] < 1e-10
    assert checks["macro_conservation_linf"] < 1e-10
    assert checks["weak_pressure_continuity_linf"] < 1e-10
    assert abs(checks["physical_pressure_integral"]) < 1e-12
    assert checks["exterior_trace_linf"] == 0
    assert_allclose(
        [
            checks["obstacle_measured_area"],
            checks["extraction_integral"],
            checks["injection_integral"],
        ],
        [0.25, -1, 1],
        rtol=0,
        atol=1e-15,
    )
    archive = tmp_path / row["archive"]
    assert json.loads(archive.with_suffix(".json").read_text()) == row
    assert acquisition._fingerprint(archive) == row["archive_sha256"]
    loaded, manifest = acquisition.load_archive(archive)
    assert manifest == row
    assert loaded["macro_cells"].shape == (200, 3)
    with np.load(archive) as data:
        kernel, constraints = data["executed_kernel"], data["executed_constraints"]
        assert acquisition._array_digest(kernel) == row["basis_sha256"]
        assert acquisition._array_digest(constraints) == row["constraints_sha256"]
        assert_allclose(np.einsum("tni,tnj->tij", constraints, kernel), 1, atol=5e-16)
        assert data["local_fields"].shape == kernel.shape[:-1]
        assert data["flux_quadrature"].shape == (800, 3, 2)
        assert_array_equal(data["macro_signs"], acquisition.macro_mesh().signs)
        assert_array_equal(
            data["permeability"],
            acquisition.coefficient(data["points"][data["cells"]].mean(axis=1)),
        )
        assert_array_equal(
            data["source_density"], acquisition.source(data["points"][data["cells"]].mean(axis=1))
        )
        if formulation == "primal":
            assert_array_equal(data["pressure"], data["local_fields"].ravel())
            assert checks["fine_cell_conservation_linf"] is None
            assert "not H(div)" in row["flux_convention"]
        else:
            assert checks["fine_cell_conservation_linf"] < 1e-10
            assert "H(div) RT0" in row["flux_convention"]


def test_replay_rejects_a_rehashed_changed_basis_or_orientation(acquisition, tmp_path):
    """A valid file digest cannot substitute for the executed matrix/face contract."""
    row = acquisition.acquire(acquisition.ObstacleConfiguration(2, 1), tmp_path)
    path = tmp_path / row["archive"]
    original, _ = acquisition.load_archive(path)
    for name in ("executed_kernel", "executed_constraints", "macro_signs", "macro_normals"):
        changed = {key: value.copy() for key, value in original.items()}
        changed[name].flat[np.argmax(np.abs(changed[name]))] *= -1
        acquisition._atomic_archive(path, changed)
        record = {**row, "archive_sha256": acquisition._fingerprint(path)}
        acquisition._atomic_record(path.with_suffix(".json"), record)
        with pytest.raises(ValueError, match="basis or trace orientation"):
            acquisition.load_archive(path)
    acquisition._atomic_archive(path, original)
    record = {**row, "archive_sha256": acquisition._fingerprint(path)}
    record["source_sha256"] = {**row["source_sha256"], "examples/quarter_spot_problem.py": "stale"}
    acquisition._atomic_record(path.with_suffix(".json"), record)
    with pytest.raises(ValueError, match="physical case provenance"):
        acquisition.load_archive(path)


def test_spawn_and_native_thread_counts_preserve_physical_fields(acquisition, tmp_path):
    """A spawned complete-case acquisition keeps the declared basis and field spaces."""
    config = acquisition.ObstacleConfiguration(2, 1)
    rows = [
        acquisition.acquire(config, tmp_path / "serial", native_threads=1),
        acquisition.acquire(config, tmp_path / "threads", native_threads=2),
        acquisition.acquire(config, tmp_path / "spawn", backend="process", workers=2),
    ]
    arrays = [
        acquisition.load_archive(tmp_path / directory / row["archive"])[0]
        for directory, row in zip(("serial", "threads", "spawn"), rows, strict=True)
    ]
    for row, data in zip(rows[1:], arrays[1:], strict=True):
        assert row["basis_sha256"] == rows[0]["basis_sha256"]
        assert row["constraints_sha256"] == rows[0]["constraints_sha256"]
        for name in ("pressure", "flux_quadrature", "trace", "coarse"):
            # These are fresh solves of the contrast-1e4 system, not bitwise replay.
            # Check the same relative numerical accuracy as the original equations;
            # the archived supplied basis itself is checked exactly above.
            difference = np.linalg.norm(data[name] - arrays[0][name])
            scale = np.linalg.norm(arrays[0][name])
            assert difference / scale < 1e-10
        assert row["physical_checks"]["original_free_equations_relative_load_residual"] < 1e-10
