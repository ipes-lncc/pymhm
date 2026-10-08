"""Practical round trips for external resources, safe archives and literal provenance."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
import threading
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pymhm.io import provenance, resources, workspace


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(workspace, "_CATALOGUES", {})
    monkeypatch.setattr(workspace, "_RESOURCE_LABELS", {})
    monkeypatch.setattr(workspace, "_SOURCE_ROOTS", {})
    monkeypatch.setenv("PYMHM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PYMHM_RESOURCE_ORIGIN", raising=False)


@pytest.fixture
def web_payload():
    payloads = {"/good": b"scientific immutable bytes", "/empty": b"", "/other": b"different"}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path not in payloads:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payloads[self.path])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", payloads, requests
    finally:
        server.shutdown()
        worker.join()
        server.server_close()


def test_download_roundtrip_cache_limits_and_failure_preservation(
    tmp_path, web_payload, monkeypatch
):
    """A failed transfer never replaces user data; a verified cache needs no network."""
    base, payloads, requests = web_payload
    payload = payloads["/good"]
    path = tmp_path / "data" / "field.bin"
    assert resources.download_resource(base + "/good", path, sha256=digest(payload)) == path
    assert resources.download_resource(base + "/missing", path, sha256=digest(payload)) == path
    assert requests == ["/good"]
    with pytest.raises(ValueError, match="existing resource"):
        resources.download_resource(base + "/other", path, sha256=digest(b"different"))
    assert path.read_bytes() == payload
    target = tmp_path / "missing"
    for url, checksum, limit, error in (
        ("/good", digest(payload), 1, ValueError),
        ("/other", digest(payload), 1024, ValueError),
        ("/missing", digest(payload), 1024, OSError),
    ):
        with pytest.raises(error):
            resources.download_resource(base + url, target, sha256=checksum, maximum_bytes=limit)
    assert not target.exists()
    assert not list(tmp_path.rglob(".pymhm-resource-*"))
    resources.download_resource(base + "/empty", target, sha256=digest(b""))
    assert target.read_bytes() == b""
    monkeypatch.setenv("PYMHM_RESOURCE_ORIGIN", base)
    mirrored = tmp_path / "mirrored"
    resources.download_resource("https://invalid.example/good", mirrored, sha256=digest(payload))
    assert mirrored.read_bytes() == payload


def test_invalid_download_controls_and_concurrent_writers(tmp_path, monkeypatch):
    """Invalid controls and concurrent data cannot produce an accepted wrong coefficient file."""
    for options in (
        {"sha256": "A" * 64},
        {"sha256": "g" * 64},
        {"sha256": None},
        {"maximum_bytes": 0},
        {"maximum_bytes": True},
        {"timeout": 0},
        {"timeout": float("nan")},
    ):
        with pytest.raises(ValueError):
            resources.download_resource(
                "unused", tmp_path / "out", **({"sha256": "0" * 64} | options)
            )
    for competing in (b"literal", b"unexpected"):
        target = tmp_path / "race.bin"

        @contextmanager
        def response(*args, target=target, competing=competing, **kwargs):
            yield io.BytesIO(b"literal")
            target.write_bytes(competing)

        monkeypatch.setattr(resources, "urlopen", response)
        if competing == b"literal":
            assert (
                resources.download_resource("unused", target, sha256=digest(b"literal")) == target
            )
        else:
            with pytest.raises(ValueError, match="concurrently created"):
                resources.download_resource("unused", target, sha256=digest(b"literal"))
        assert target.read_bytes() == competing
        target.unlink()


def test_external_catalogue_lazy_inputs_globs_and_local_edits(tmp_path, web_payload, monkeypatch):
    """Only requested fields and their notices are acquired; local edits remain local."""
    base, payloads, requests = web_payload
    root = tmp_path / "work"
    root.mkdir()
    names = ("fields/a.bin", "fields/deeper/b.bin", "notes/license.txt", "notes/config.json")
    records = {
        name: {
            "sha256": digest(payloads["/good"]),
            "url": base + "/good",
            "size_bytes": len(payloads["/good"]),
        }
        for name in names
    }
    records["fields/a.bin"]["related_resources"] = ["notes/license.txt"]
    registry = {
        "resources": records,
        "unavailable": {"fields/original.bin": {"reason": "not published"}},
    }
    (root / workspace.RESOURCE_MANIFEST).write_text(json.dumps(registry))
    monkeypatch.setenv("PYMHM_WORKSPACE", str(root))
    assert workspace.case_workspace() == root
    assert workspace.case_workspace(root) == root
    assert requests == []
    cached = workspace.resource_file("fields/a.bin")
    assert workspace.source_label(cached, root) == "fields/a.bin"
    assert (
        workspace.resource_file("fields/a.bin", cache=tmp_path / "another").read_bytes()
        == payloads["/good"]
    )
    assert workspace.ensure_resource("fields/a.bin", root) == root / "fields/a.bin"
    assert (root / "notes/license.txt").is_file()
    assert workspace.resource_glob(root / "fields", "*.bin") == [root / "fields/a.bin"]
    assert workspace.resource_glob(root / "fields", "*.bin", recursive=True, root=root) == [
        root / "fields/a.bin",
        root / "fields/deeper/b.bin",
    ]
    assert workspace.resource_glob(root, "**/b.b?n", root=root) == [root / "fields/deeper/b.bin"]
    assert workspace.resource_glob(root / "fields", "[ab].bin", root=root) == [
        root / "fields/a.bin"
    ]
    assert workspace.resource_glob(tmp_path / "elsewhere", "*.bin") == []
    assert workspace.read_resource_bytes(root / "fields/a.bin") == payloads["/good"]
    assert workspace.source_file("notes/config.json", root=root).read_bytes() == payloads["/good"]
    assert (
        workspace.read_resource_text(root / "notes/config.json", encoding="utf-8")
        == payloads["/good"].decode()
    )
    (root / "fields/a.bin").write_bytes(b"user field")
    assert workspace.local_resource(root / "fields/a.bin") == root / "fields/a.bin"
    assert workspace.read_resource_bytes(root / "fields/a.bin") == b"user field"
    with pytest.raises(ValueError, match="differs"):
        workspace.resource_file("fields/a.bin")
    with pytest.raises(FileNotFoundError, match="not published"):
        workspace.local_resource(root / "fields/original.bin", root=root)
    with pytest.raises(FileNotFoundError, match="Missing"):
        workspace.resource_file("missing")
    assert workspace.local_resource(root / "unknown") == root / "unknown"
    assert workspace.local_resource(tmp_path / "outside", root=root) == tmp_path / "outside"
    external = tmp_path / "outside"
    external.mkdir()
    (root / "escape").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        workspace.materialize_resources(["escape/field.bin"], root)


def test_catalogues_reject_ambiguous_or_incomplete_input_identities(tmp_path):
    """Malformed external declarations fail before any dataset acquisition."""
    record = {"sha256": "a" * 64}
    invalid = [
        {"resources": []},
        {"unavailable": []},
        {"resources": {"file": []}},
        {"resources": {"a\\b": record}},
        {"resources": {"../file": record}},
        {"resources": {"/file": record}},
        {"resources": {"C:/file": record}},
        {"resources": {"": record}},
        {"resources": {"file": {"sha256": 4}}},
        {"resources": {"file": {"sha256": "g" * 64}}},
        {"resources": {"file": record | {"url": ""}}},
        {"resources": {"file": record | {"url": "http://source"}}},
        {"resources": {"file": record | {"related_resources": ["../escape"]}}},
        {"unavailable": {"file": {}}},
        {"resources": {"file": record}, "unavailable": {"file": {"reason": "ambiguous"}}},
    ]
    for manifest in invalid:
        with pytest.raises(ValueError):
            workspace.register_resources(tmp_path, manifest)
    workspace.register_resources(tmp_path, {"resources": {"absent": record}})
    with pytest.raises(FileNotFoundError):
        workspace.resource_file("absent", directory=tmp_path)
    config = tmp_path / "local.json"
    config.write_text("{}")
    assert workspace.resource_file(config.name, directory=tmp_path) == config


def zip_payload(entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as zipped:
        for name, content in entries:
            zipped.writestr(name, content)
    return data.getvalue()


def test_archive_roundtrip_is_explicit_and_preserves_user_sources(
    tmp_path, web_payload, monkeypatch
):
    """Archive extraction preserves user sources and does not execute downloaded code."""
    base, payloads, _ = web_payload
    catalogue = {
        "resources": {
            "field.bin": {
                "sha256": digest(payloads["/good"]),
                "url": base + "/good",
                "size_bytes": len(payloads["/good"]),
            }
        }
    }
    payload = zip_payload(
        [
            ("empty/", b""),
            ("helpers/provider.py", b"raise RuntimeError('do not execute downloads')\n"),
            ("nested/data.txt", b"literal"),
            (workspace.RESOURCE_MANIFEST, json.dumps(catalogue)),
        ]
    )
    payloads["/archive"] = payload
    root = workspace.workspace_from_archive(
        base + "/archive", sha256=digest(payload), directory=tmp_path / "work"
    )
    assert not (root / "field.bin").exists()
    assert workspace.read_resource_bytes(root / "field.bin") == payloads["/good"]
    source = root / "helpers/provider.py"
    source.write_text("# user's source")
    assert (
        workspace.workspace_from_archive(base + "/archive", sha256=digest(payload), directory=root)
        == root
    )
    assert source.read_text() == "# user's source"
    assert not list(root.glob(".pymhm-extract-*"))
    plain = zip_payload([("values.txt", b"a reusable input")])
    payloads["/plain"] = plain
    plain_root = workspace.workspace_from_archive(
        base + "/plain", sha256=digest(plain), directory=tmp_path / "independent"
    )
    assert (plain_root / "values.txt").read_bytes() == b"a reusable input"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYMHM_WORKSPACE", raising=False)
    assert workspace.case_workspace() == tmp_path


def test_unsafe_archives_are_rejected_before_writing(tmp_path, web_payload):
    """Unsafe names and oversized or unsupported ZIPs cannot overwrite files."""
    base, payloads, _ = web_payload
    link = zipfile.ZipInfo("link")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.warns(UserWarning, match="Duplicate"):
        duplicate = zip_payload([("file", b"a"), ("file", b"b")])
    encrypted = bytearray(zip_payload([("file", b"a")]))
    encrypted[6] |= 1
    central = encrypted.index(b"PK\x01\x02")
    encrypted[central + 8] |= 1
    malicious = [
        zip_payload([("../escape", b"a")]),
        zip_payload([(link, b"outside")]),
        duplicate,
        bytes(encrypted),
    ]
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    malicious.append(zip_payload([("escape/file", b"a")]))
    for index, payload in enumerate(malicious):
        payloads[f"/bad{index}"] = payload
        with pytest.raises(ValueError):
            workspace.workspace_from_archive(
                base + f"/bad{index}", sha256=digest(payload), directory=root
            )
    payload = zip_payload([("file", b"large")])
    payloads["/big"] = payload
    with pytest.raises(ValueError, match="extracted byte"):
        workspace.workspace_from_archive(
            base + "/big", sha256=digest(payload), directory=root, maximum_extracted_bytes=1
        )
    with pytest.raises(ValueError, match="maximum extracted"):
        workspace.workspace_from_archive(
            "unused", sha256="0" * 64, directory=root, maximum_extracted_bytes=0
        )
    assert not (root / "file").exists()
    assert not list(outside.iterdir())


def test_package_resources_and_actual_source_contracts(tmp_path, monkeypatch):
    """External packages and zipped resources retain actual source and data identities."""
    zipped = tmp_path / "portable.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.writestr("portable_fixture/__init__.py", "")
        archive.writestr("portable_fixture/input.bin", b"literal")
    monkeypatch.syspath_prepend(str(zipped))
    try:
        cached = workspace.resource_file("input.bin", package="portable_fixture")
        assert cached.read_bytes() == b"literal"
        assert workspace.resource_file("input.bin", package="portable_fixture") == cached
        cached.write_bytes(b"corrupt")
        assert (
            workspace.resource_file("input.bin", package="portable_fixture").read_bytes()
            == b"literal"
        )
        with pytest.raises(FileNotFoundError):
            workspace.resource_file("missing", package="portable_fixture")
        copied = workspace.materialize_resources(
            ["input.bin"], tmp_path / "copies", package="portable_fixture"
        )
        assert copied[0].read_bytes() == b"literal"
    finally:
        sys.modules.pop("portable_fixture", None)
    assert workspace.resource_file("__init__.py", package="pymhm").is_file()
    source = workspace.source_file("src/pymhm/io/resources.py", root=tmp_path)
    assert workspace.source_file(source, root=tmp_path) == source
    assert workspace.source_label(source, tmp_path) == "src/pymhm/io/resources.py"
    assert workspace.source_identity(tmp_path, ["src/pymhm/io/resources.py"]) == {
        "src/pymhm/io/resources.py": provenance.file_digest(source)
    }
    literal = tmp_path / "local.py"
    literal.write_text("pass")
    assert workspace.source_file(literal.name, root=tmp_path) == literal
    assert workspace.source_label(literal, tmp_path) == "local.py"
    outside = tmp_path.parent / "external.py"
    assert workspace.source_label(outside, tmp_path) == outside.as_posix()
    assert (
        workspace.source_file("unknown_package/file.py", root=tmp_path)
        == tmp_path / "unknown_package/file.py"
    )
    assert workspace.source_file("optional.lock", root=tmp_path) == tmp_path / "optional.lock"
    assert provenance.optional_file_digest(tmp_path / "optional.lock") is None
    assert provenance.optional_file_digest(literal) == provenance.file_digest(literal)
    package = SimpleNamespace(__file__=None)
    monkeypatch.setitem(sys.modules, "namespace_fixture", package)
    with pytest.raises(ValueError, match="identifiable"):
        workspace.source_file("src/namespace_fixture/file.py", root=tmp_path)
    with pytest.raises(ValueError, match="identifiable"):
        provenance.current_source_manifest({}, packages=("namespace_fixture",))
    record = provenance.current_source_manifest({"input": "literal"}, packages=("pymhm",))
    assert record["input"] == "literal"
    assert record["src/pymhm/io/resources.py"] == provenance.file_digest(source)


def test_optional_git_records_describe_only_the_selected_workspace(tmp_path, monkeypatch):
    """Absent Git metadata stays absent in reproducibility records."""
    assert provenance.workspace_revision(tmp_path) is None
    assert provenance.workspace_git_dirty(tmp_path) is None
    (tmp_path / ".git").write_text("gitdir: external")
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout=" abc123\n"))
    monkeypatch.setattr(provenance.subprocess, "run", run)
    assert provenance.workspace_revision(tmp_path) == "abc123"
    assert run.call_args.kwargs["cwd"] == tmp_path
    assert provenance.workspace_git_dirty(tmp_path) is True
    run.return_value = SimpleNamespace(returncode=0, stdout="")
    assert provenance.workspace_git_dirty(tmp_path) is False
    run.return_value = SimpleNamespace(returncode=1, stdout="discard")
    assert provenance.workspace_revision(tmp_path) is None
    run.side_effect = FileNotFoundError("git absent")
    assert provenance.workspace_revision(tmp_path) is None
