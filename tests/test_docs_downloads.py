"""Validate website downloads retain identities without bundling third-party inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.docs_downloads import DOWNLOAD_BASE, export_downloads


@pytest.mark.parametrize("problem", ["valid", "digest", "url", "traversal"])
def test_download_export_uses_only_identified_available_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Changed bytes/URLs fail; unavailable original fields stay explicit and uncopied."""
    monkeypatch.setattr("scripts.docs_downloads.build_notebook_companions", lambda root, site: {})
    root, site = tmp_path / "source", tmp_path / "site"
    name = "examples/results/case/record.json"
    payload = b'{"norm": 0.5}'
    digest = hashlib.sha256(payload).hexdigest()
    source = root / name
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    notebook = "notebooks/darcy/example.ipynb"
    notebook_source = root / notebook
    notebook_source.parent.mkdir(parents=True)
    notebook_source.write_bytes(b'{"cells": []}')
    notebook_digest = hashlib.sha256(notebook_source.read_bytes()).hexdigest()
    remote = {
        name: {
            "sha256": digest,
            "size_bytes": len(payload),
            "url": DOWNLOAD_BASE + digest + "/record.json",
            "kind": "record",
            "provenance": "Own physical norm; no external execution claim.",
        }
    }
    unavailable = {"build/original-external.npz": {"reason": "Original source unavailable"}}
    if problem == "digest":
        source.write_bytes(b"changed")
    elif problem == "url":
        remote[name]["url"] = "https://unidentified.example/record.json"
    elif problem == "traversal":
        remote["../private.json"] = remote.pop(name)
    index = root / "examples/resource_manifest.json"
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "bundled": {notebook: notebook_digest},
                "remote": remote,
                "unavailable": unavailable,
            }
        )
    )
    if problem != "valid":
        with pytest.raises(ValueError, match="identity|URL|Invalid"):
            export_downloads(root, site)
        return
    result = export_downloads(root, site)
    assert result == {
        "resources": 2,
        "companions": 0,
        "unique_files": 2,
        "size_bytes": len(payload) + notebook_source.stat().st_size,
        "unavailable": 1,
    }
    assert (site / "downloads" / digest / "record.json").read_bytes() == payload
    catalogue = json.loads((site / "downloads/catalogue.json").read_text())
    assert catalogue["unavailable"] == unavailable
    assert catalogue["downloads"][name]["provenance"] == remote[name]["provenance"]
    assert not list(site.rglob("*.npz"))
