"""Small checkpoint regressions independent of FEM and acquired numerical fields."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def checkpoint(monkeypatch):
    """Load the public helper from a normal checkout or an isolated candidate."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("examples.campaign_checkpoint")


def test_identity_sources_and_archive_bytes(checkpoint, tmp_path):
    """Only exact configuration and acquisition bytes may authorize a completed skip."""
    p = tmp_path / "field.npz"
    p.write_bytes(b"small acquired fixture")
    row = {"degree": 2, "error": 0.1, **checkpoint.archive_identity(p)}
    checkpoint.require_sources({"operator": "hash"}, {"operator": "hash"})
    checkpoint.verify_checkpoint(row, {"degree": 2}, directory=tmp_path, metrics=("error",))
    for identity in ({}, {"unknown": 1}, {"degree": 3}, {"degree": 2.0}):
        with pytest.raises(ValueError):
            checkpoint.verify_checkpoint(row, identity, directory=tmp_path)
    for before, after in (({}, {}), ({"a": "x"}, {}), ({}, {"a": "x"}), ({"a": "x"}, {"a": "y"})):
        with pytest.raises(ValueError):
            checkpoint.require_sources(before, after)
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        checkpoint.verify_checkpoint(row, {"degree": 2}, directory=tmp_path)


@pytest.mark.parametrize(
    "metric", [None, True, "1", -1, float("nan"), float("inf"), 10**1000, object()]
)
def test_incomplete_nonfinite_or_negative_metrics_rejected(checkpoint, tmp_path, metric):
    """A completed checkpoint cannot consist of absent or invalid numerical metrics."""
    with pytest.raises(ValueError):
        checkpoint.verify_checkpoint(
            {"n": 1, "error": metric}, {"n": 1}, directory=tmp_path, metrics=("error",)
        )


def test_nonfinite_float_conversion_is_rejected(checkpoint, tmp_path):
    """A real scalar with nonfinite float conversion cannot pass a metric check."""

    class Nonfinite(int):
        """Exercise conversion separately from JSON's integer representation."""

        def __float__(self):
            """Return an invalid numerical value."""
            return float("nan")

    with pytest.raises(ValueError, match="finite"):
        checkpoint.verify_checkpoint(
            {"n": 1, "error": Nonfinite(1)}, {"n": 1}, directory=tmp_path, metrics=("error",)
        )


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "/absolute.npz",
        "../outside.npz",
        "dir/../../outside.npz",
        r"C:\field.npz",
        r"dir\field.npz",
    ],
)
def test_archive_names_cannot_escape(checkpoint, tmp_path, name):
    """Cross-platform absolute and traversing names are rejected before file access."""
    with pytest.raises(ValueError, match="safe relative"):
        checkpoint.archive_path(tmp_path, name)


def test_legacy_observation_never_invents_acquisition_digest(checkpoint, tmp_path):
    """Historical files without a stored digest are observations, not authorized resumes."""
    p = tmp_path / "field.npz"
    p.write_bytes(b"original")
    row = {"n": 1, "archive": p.name}
    with pytest.raises(ValueError, match="no acquisition digest"):
        checkpoint.verify_checkpoint(row, {"n": 1}, directory=tmp_path)
    with pytest.raises(ValueError, match="missing"):
        checkpoint.verify_checkpoint({"n": 1}, {"n": 1}, directory=tmp_path, archive_required=True)
    acquired = {**row, **checkpoint.archive_identity(p)}
    checks = checkpoint.retrospective_files(tmp_path, [{"n": 1}, row, acquired])
    assert checks[0]["recorded_acquisition_sha256"] is None
    assert checks[0]["acquisition_bytes_verified"] is False
    assert checks[1]["acquisition_bytes_verified"] is True
    assert checks[0]["observed_sha256"] == checks[1]["observed_sha256"]


def test_retrospective_manifest_preserves_bytes(checkpoint, tmp_path, monkeypatch):
    """A separate validation points to an immutable acquisition snapshot without solving."""
    module = importlib.import_module("examples.validate_campaign_checkpoint")
    p = tmp_path / "field.npz"
    p.write_bytes(b"field")
    record = {
        "source_sha256": {"old.py": "old"},
        "rows": [
            {"n": 1, **checkpoint.archive_identity(p)},
            {"n": 2, "fields": p.name},
        ],
    }
    manifest = tmp_path / "campaign.json"
    raw = json.dumps(record).encode()
    manifest.write_bytes(raw)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(
        "sys.argv", ["validate", "--manifest", str(manifest), "--output", str(output)]
    )
    module.main()
    module.main()
    checked = json.loads(output.read_text())
    assert manifest.read_bytes() == raw
    assert (tmp_path / checked["acquired_manifest_snapshot"]).read_bytes() == raw
    assert checked["archive_checks"][1]["acquisition_bytes_verified"] is False
    monkeypatch.setattr(
        "sys.argv", ["validate", "--manifest", str(manifest), "--output", str(manifest)]
    )
    with pytest.raises(ValueError, match="differ"):
        module.main()


def test_cli_refuses_changed_manifest_or_snapshot(checkpoint, tmp_path, monkeypatch):
    """A separately stored validation must describe the exact bytes actually inspected."""
    import runpy

    module = importlib.import_module("examples.validate_campaign_checkpoint")
    manifest, output = tmp_path / "campaign.json", tmp_path / "validation.json"
    manifest.write_text('{"rows": []}')
    monkeypatch.setattr(
        "sys.argv", ["validate", "--manifest", str(manifest), "--output", str(output)]
    )
    runpy.run_path(module.__file__, run_name="__main__")
    checked = json.loads(output.read_text())
    snapshot = tmp_path / checked["acquired_manifest_snapshot"]
    snapshot.write_text("wrong bytes")
    with pytest.raises(ValueError, match="snapshot"):
        module.main()
    original = module.inspect_manifest

    def mutate(path):
        """Emulate concurrent acquisition manifest replacement after the inspected read."""
        result = original(path)
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(module, "inspect_manifest", mutate)
    with pytest.raises(ValueError, match="changed during"):
        module.main()
