"""Actual source owners are fingerprinted without relabeling historical records."""

from pathlib import Path

import pymhm
from pymhm.io.provenance import current_source_manifest, file_digest


def test_recursive_source_manifest_preserves_archived_identity(tmp_path, monkeypatch):
    """Fingerprint current owners without relabeling independent archived records."""
    package = tmp_path / "pymhm"
    owner = package / "core/contracts.py"
    owner.parent.mkdir(parents=True)
    owner.write_text('"""A package owner."""\n')
    init = package / "__init__.py"
    init.write_text('"""Package root."""\n')
    monkeypatch.setattr(pymhm, "__file__", str(init))
    archived = {"src/pymhm/hybrid.py": "historical", "input.json": "unchanged"}
    observed = current_source_manifest({"input.json": "unchanged"})
    assert observed == {
        "input.json": "unchanged",
        "src/pymhm/__init__.py": file_digest(init),
        "src/pymhm/core/contracts.py": file_digest(owner),
    }
    assert archived["src/pymhm/hybrid.py"] == "historical"


def test_live_manifest_contains_all_physical_source_owners():
    """Fresh acquisition guards include both interfaces and delegated numerical owners."""
    root = Path(pymhm.__file__).parent
    expected = {
        f"src/pymhm/{path.relative_to(root).as_posix()}": file_digest(path)
        for path in root.rglob("*.py")
    }
    observed = current_source_manifest({})
    assert observed == expected
    assert "src/pymhm/core/contracts.py" in observed
    assert "src/pymhm/core/system.py" in observed
    assert "src/pymhm/fem/reference.py" in observed
