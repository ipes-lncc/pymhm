"""Streamed responses preserve the complete acoustic pollution diagnostics."""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples import helmholtz_stability_phases as phases
from examples.helmholtz_campaign import AcousticWave, norms
from examples.helmholtz_stability import projected_solution
from pymhm._legacy.models.waves.helmholtz import solve_helmholtz
from pymhm.fem.traces.helmholtz import helmholtz_skeleton
from pymhm.linalg.linear import LinearSolveError
from pymhm.meshes.cartesian import CartesianMacroMesh


@pytest.mark.parametrize("ell", [0, 1])
@pytest.mark.parametrize("exact_cache", [False, True])
def test_phased_full_case_matches_in_memory_hankel_norms_and_projection(ell, tmp_path, exact_cache):
    """All nine macrocells and their one-sided gradients match the original solve."""
    wave = AcousticWave(2 * np.pi * 2.3, 0, "hankel")
    mesh = CartesianMacroMesh(3)
    original = solve_helmholtz(
        mesh,
        omega=wave.omega,
        degree=ell + 3,
        local_refinement=2,
        quadrature_order=12,
        skeleton=helmholtz_skeleton(mesh, wave.omega, degree=ell),
        absorbing=wave.absorbing,
    )
    actual = norms(original, wave, 16)
    interpolated = norms(projected_solution(original, wave, 24), wave, 16)
    row = phases.solve_configuration(
        3,
        ell,
        2.3,
        tmp_path,
        sources=phases.hashes(),
        batch_size=2,
        exact_response_cache=exact_cache,
    )
    assert_allclose(
        [
            row["mhm_gradient_l2"],
            row["interpolated_gradient_l2"],
            row["mhm_gradient_relative"],
            row["interpolated_gradient_relative"],
        ],
        [
            actual["gradient_l2"],
            interpolated["gradient_l2"],
            actual["gradient_relative_error"],
            interpolated["gradient_relative_error"],
        ],
        rtol=2e-12,
        atol=2e-13,
    )
    assert row["reconstructed_macro_count"] == 9
    assert row["algebraic_residual"] < 1e-12
    assert row["local_equation_residual_max"] < 1e-12
    assert row["macro_balance_max"] < 1e-12
    assert row["local_response_storage_bytes"] > 0
    manifest = json.loads((tmp_path / "responses.json").read_text())
    assert [part["stop"] - part["start"] for part in manifest["batches"]] == [2, 2, 2, 2, 1]


def test_phased_checkpoint_preserves_physical_failure_and_resonance_barriers(tmp_path, monkeypatch):
    """Rejected continuous or numerical liftings cannot become accepted suffix points."""

    def rejected(*args, **kwargs):
        raise LinearSolveError("original local equations fail")

    monkeypatch.setattr(phases, "solve_configuration", rejected)
    phases.run(tmp_path, 1, 15, [30])
    path = tmp_path / "ell1-frequency15.json"
    record = json.loads(path.read_text())
    assert record["rows"] == []
    assert record["sampled_threshold"]["rejected_resolutions"] == [30]
    with pytest.raises(LinearSolveError, match="original local"):
        phases.run(tmp_path, 1, 15, [30, 32])
    record = json.loads(path.read_text())
    assert record["sampled_threshold"]["rejected_resolutions"] == [30, 32]
    assert record["largest_suffix_H_star"] is None
    with pytest.raises(ValueError, match="checkpoint identity"):
        phases.run(tmp_path, 1, 15, [30, 32], norm_order=20)


def test_phased_resume_verifies_accepted_norm_and_response_manifest(tmp_path):
    """An accepted row is bound to its finite metrics and executed response bytes."""
    phases.run(tmp_path, 0, 2.3, [1], batch_size=1)
    path = tmp_path / "ell0-frequency2.3.json"
    before = path.read_bytes()
    phases.run(tmp_path, 0, 2.3, [1], batch_size=1)
    assert path.read_bytes() == before
    record = json.loads(path.read_text())
    manifest = tmp_path / "responses/ell0-frequency2.3-n1/responses.json"
    original_manifest = manifest.read_bytes()
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="response manifest digest"):
        phases.run(tmp_path, 0, 2.3, [1], batch_size=1)
    manifest.write_bytes(original_manifest)
    first_batch = manifest.parent / json.loads(original_manifest)["batches"][0]["archive"]
    first_batch.write_bytes(first_batch.read_bytes() + b" ")
    with pytest.raises(ValueError, match="response batch digest"):
        phases.run(tmp_path, 0, 2.3, [1], batch_size=1)
    record["rows"][0]["mhm_gradient_l2"] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="finite"):
        phases.run(tmp_path, 0, 2.3, [1], batch_size=1)
