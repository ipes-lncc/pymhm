"""Numerical archive identity and ancestry when reusing the initial SPE state."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pymhm import TriangleMesh
from pymhm.longest_edge import refine_longest_edge

MATERIAL = "examples/results/spe10/layer-36.npz"


def material_sources(api: Any, monkeypatch: pytest.MonkeyPatch, material: Path) -> dict[str, str]:
    """Fingerprint real numerical sources with one generated, mutable material fixture."""
    original_path, original_hashes = api.source_path, api.hashes

    def paths(name: str) -> Path:
        """Override only the fixture material; all numerical owners remain real files."""
        return material if name == MATERIAL else original_path(name)

    def hashes() -> dict[str, str]:
        """Update the material digest while preserving all actual source identities."""
        return {**original_hashes(), MATERIAL: hashlib.sha256(material.read_bytes()).hexdigest()}

    monkeypatch.setattr(api, "source_path", paths)
    monkeypatch.setattr(api, "hashes", hashes)
    return api.acquisition_hashes()


def test_initial_checkpoint_preserves_field_bytes_and_changes_only_next_geometry(
    monkeypatch, tmp_path
):
    """A verified initial state is transferred without resolving or rounding its coefficients."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    api = importlib.import_module("examples.solve_spe10_balanced")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    mesh = TriangleMesh.unit_square()
    material = tmp_path / "material.npz"
    material.write_bytes(b"fixed-material-identity")
    sources = material_sources(api, monkeypatch, material)
    archive = source / "mhm-level0.npz"
    np.savez(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        marked=np.array([True, False]),
        pressure=np.array([0.3, 0.5]),
    )
    row = {
        "level": 0,
        "local_degree": 2,
        "local_refinement": 1,
        "trace_degree": 0,
        "reconstruction_degree": 2,
        "material_fitted": True,
        "decision": "macro",
        "theta": 0.5,
        "local_error_ratio": 0.25,
        "assembly_order": 6,
        "estimator_order": 6,
        "source_changed_during_solve": False,
        "local_indicator": 0.1,
        "flux_defect": 1.0,
        "nonconformity": 0.0,
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_hashes": sources,
        "estimator_convention": "energy",
        "local_refinement_precision": "extended",
        "estimator": 1.0,
        "divergence_defect": 0.0,
        "oscillation": 0.0,
        "macro_balance_linf": 0.0,
    }
    record = source / "adaptive.json"
    record.write_text(json.dumps([row]))
    rows, next_mesh = api.reuse_initial_checkpoint(source, output, mesh, refine_longest_edge)
    assert (output / archive.name).read_bytes() == archive.read_bytes()
    assert rows[0]["source_hashes"] == row["source_hashes"]
    assert rows[0]["reused_initial_field"]
    assert len(next_mesh.cells) == 4
    assert rows[0]["minimum_macro_angle_degrees"] == pytest.approx(45.0)
    with pytest.raises(ValueError, match="geometry"):
        api.reuse_initial_checkpoint(
            source, output, TriangleMesh.unit_square(2), refine_longest_edge
        )
    material.write_bytes(b"changed")
    with pytest.raises(ValueError, match="acquisition sources differ"):
        api.reuse_initial_checkpoint(source, output, mesh, refine_longest_edge)
    material.write_bytes(b"fixed-material-identity")
    row["archive_sha256"] = "invalid"
    record.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="digest"):
        api.reuse_initial_checkpoint(source, output, mesh, refine_longest_edge)
    row["trace_degree"] = 1
    record.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="checkpoint identity"):
        api.reuse_initial_checkpoint(source, output, mesh, refine_longest_edge)


@pytest.fixture
def completed_state(monkeypatch, tmp_path):
    """Build a checksum-verified scalar checkpoint without solving a reservoir problem."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    api = importlib.import_module("examples.solve_spe10_balanced")
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    material = tmp_path / "material.npz"
    material.write_bytes(b"unchanged-material")
    sources = material_sources(api, monkeypatch, material)
    mesh = TriangleMesh.unit_square()
    archive = source / "mhm-level0.npz"
    np.savez(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        marked=np.array([True, False]),
        pressure=np.array([0.25, 0.75]),
    )
    row = {
        "level": 0,
        "local_degree": 2,
        "local_refinement": 1,
        "trace_degree": 0,
        "reconstruction_degree": 2,
        "material_fitted": True,
        "theta": 0.5,
        "local_error_ratio": 0.25,
        "assembly_order": 6,
        "estimator_order": 6,
        "source_changed_during_solve": False,
        "macro_refinement": "longest-edge",
        "decision": "macro",
        "stop_reason": "iterations",
        "flux_defect": 1.0,
        "nonconformity": 0.0,
        "local_indicator": 0.1,
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_hashes": sources,
        "estimator_convention": "energy",
        "local_refinement_precision": "extended",
        "estimator": 1.0,
        "divergence_defect": 0.0,
        "oscillation": 0.0,
        "macro_balance_linf": 0.0,
    }
    return api, source, output, row


@pytest.mark.parametrize(
    ("decision", "local_indicator", "cells", "refinement"),
    [("macro", 0.1, 4, 1), ("local", 1.0, 2, 2), ("stop", 0.1, 4, 1), ("stop", 1.0, 2, 2)],
)
def test_resume_preserves_fields_and_applies_the_recorded_next_step(
    completed_state, decision, local_indicator, cells, refinement
):
    """Continue either mesh scale while retaining all existing fields and their provenance."""
    api, source, output, row = completed_state
    row.update(decision=decision, local_indicator=local_indicator)
    (source / "adaptive.json").write_text(json.dumps([row]))
    records, mesh, level = api.resume_checkpoint(
        source, output, refine_longest_edge, "longest-edge"
    )
    assert records == [row]
    assert len(mesh.cells) == cells
    assert level == refinement
    assert (output / row["archive"]).read_bytes() == (source / row["archive"]).read_bytes()
    assert json.loads((output / "adaptive.json").read_text()) == [row]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"local_degree": 3}, "checkpoint identity"),
        ({"archive_sha256": "changed"}, "digest"),
        ({"macro_refinement": "red-green"}, "refinement policy"),
        ({"decision": "stop", "stop_reason": "tolerance"}, "stopping decision"),
        ({"decision": "local", "local_refinement": 16}, "stopping decision"),
        ({"source_hashes": {MATERIAL: "changed"}}, "acquisition sources differ"),
    ],
)
def test_resume_rejects_incompatible_state(completed_state, changes, message):
    """A continuation cannot silently alter approximation, material, checksums or stopping rules."""
    api, source, output, row = completed_state
    if "source_hashes" in changes:
        changes = {"source_hashes": {**row["source_hashes"], **changes["source_hashes"]}}
    row.update(changes)
    (source / "adaptive.json").write_text(json.dumps([row]))
    with pytest.raises(ValueError, match=message):
        api.resume_checkpoint(source, output, refine_longest_edge, "longest-edge")
