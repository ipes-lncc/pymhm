"""Scientific support records identify exact source bytes and their repository paths."""

from pathlib import Path

import pytest

from examples.introduction.provenance import source_digests


def test_source_digests_track_literal_bytes_and_explicit_relative_paths(tmp_path: Path) -> None:
    """Equal bytes share an identity; changing a file changes its recorded digest."""
    support = tmp_path / "examples" / "support.py"
    support.parent.mkdir()
    support.write_bytes(b"abc")
    duplicate = tmp_path / "same.py"
    duplicate.write_bytes(b"abc")
    original = source_digests(tmp_path, (Path("same.py"), support, Path("examples/support.py")))
    assert list(original) == ["examples/support.py", "same.py"]
    assert set(original.values()) == {
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    }
    assert source_digests(tmp_path, iter((support, duplicate))) == original
    support.write_bytes(b"abd")
    changed = source_digests(tmp_path, (support, duplicate))
    assert changed["examples/support.py"] != original["examples/support.py"]
    assert changed["same.py"] == original["same.py"]
    assert source_digests(tmp_path, ()) == {}


def test_source_digests_require_files_inside_the_declared_root(tmp_path: Path) -> None:
    """Escaping paths are rejected and absent inputs are not silently omitted."""
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"abc")
    for path in (outside, Path("../outside.py")):
        with pytest.raises(ValueError):
            source_digests(root, (path,))
    with pytest.raises(FileNotFoundError):
        source_digests(root, (Path("missing.py"),))
