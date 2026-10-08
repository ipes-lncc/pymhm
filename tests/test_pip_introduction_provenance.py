"""Introduction receipts identify downloaded owners and actual notebook files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from examples.introduction.provenance import (
    execution_source_manifest,
    notebook_provenance,
    support_manifest,
    workspace_revision,
)


def test_downloaded_reference_is_separate_from_unknown_executed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An downloaded reference cannot serve as evidence for an unidentified user notebook."""
    reference = tmp_path / "notebooks/introduction/case.ipynb"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"local reference")
    monkeypatch.delenv("PYMHM_NOTEBOOK_SOURCE", raising=False)
    monkeypatch.chdir(tmp_path)
    evidence = notebook_provenance("notebooks/introduction/case.ipynb", workspace=tmp_path)
    assert evidence["notebook_source"] is None
    assert evidence["notebook_sha256"] is None
    assert evidence["pixi_lock_sha256"] is None
    assert evidence["reference_source"] == "notebooks/introduction/case.ipynb"
    assert evidence["reference_source_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert execution_source_manifest("introduction/case.ipynb", workspace=tmp_path) == {
        "reference_notebook": evidence["reference_source_sha256"]
    }
    assert workspace_revision(tmp_path) is None


def test_declared_and_local_sources_hash_actual_bytes_with_optional_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner declarations and interactive local files retain their distinct literal identities."""
    reference = tmp_path / "notebooks/introduction/case.ipynb"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"reference")
    source = tmp_path / "case.ipynb"
    source.write_bytes(b"modified notebook")
    lock = tmp_path / "pixi.lock"
    lock.write_bytes(b"real optional lock")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYMHM_NOTEBOOK_SOURCE", raising=False)
    local = notebook_provenance("introduction/case.ipynb", workspace=tmp_path)
    assert local["notebook_source"] == str(source)
    assert local["notebook_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert local["notebook_sha256"] != local["reference_source_sha256"]
    assert local["pixi_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    monkeypatch.setenv("PYMHM_NOTEBOOK_SOURCE", str(source))
    assert notebook_provenance("introduction/case.ipynb", workspace=tmp_path) == local
    manifest = execution_source_manifest("introduction/case.ipynb", workspace=tmp_path)
    assert manifest == {
        "reference_notebook": local["reference_source_sha256"],
        "source_notebook": local["notebook_sha256"],
        "pixi.lock": local["pixi_lock_sha256"],
    }
    monkeypatch.setenv("PYMHM_NOTEBOOK_SOURCE", str(tmp_path / "missing.ipynb"))
    with pytest.raises(FileNotFoundError):
        notebook_provenance("introduction/case.ipynb", workspace=tmp_path)


def test_support_manifest_hashes_downloaded_package_paths() -> None:
    """Actual downloaded helper bytes are addressed independently of the writable workspace."""
    from examples.introduction import provenance

    source = Path(provenance.__file__)
    assert support_manifest((source, source)) == {
        "examples/introduction/provenance.py": hashlib.sha256(source.read_bytes()).hexdigest()
    }
