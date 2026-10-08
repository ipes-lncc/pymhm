"""Generic writable workspaces, external catalogues and literal source identities."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from fnmatch import fnmatchcase
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from pymhm.core.validation import positive_int
from pymhm.io.provenance import file_digest
from pymhm.io.resources import download_resource

RESOURCE_MANIFEST = ".pymhm-resources.json"
_CATALOGUES: dict[Path, dict[str, Any]] = {}
_RESOURCE_LABELS: dict[Path, str] = {}
_SOURCE_ROOTS: dict[Path, str] = {}


def _package_directory(package: str) -> Path:
    """Identify an importable regular package's source directory."""
    module = importlib.import_module(package)
    if module.__file__ is None:
        raise ValueError("source lookup requires identifiable package files")
    return Path(module.__file__).resolve().parent


def _relative_name(value: str | Path) -> str:
    """Require a portable relative name without traversal or drive letters."""
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or ":" in text:
        raise ValueError("resource names must be relative paths without parent traversal")
    return path.as_posix()


def register_resources(
    directory: str | Path, manifest: Mapping[str, Any] | str | Path | None = None
) -> None:
    """Register external resources for one explicit directory without acquiring them.

    A catalogue maps resources to records with sha256 and, for downloads, url
    and size_bytes. Records can declare related_resources such as attribution
    notices. An unavailable mapping provides reasons for absent original data.
    Pass a mapping or JSON path; the default is .pymhm-resources.json in the
    directory. The library contains no application catalogue or dataset.
    """
    root = Path(directory).expanduser().resolve()
    if isinstance(manifest, Mapping):
        payload = dict(manifest)
    else:
        path = root / RESOURCE_MANIFEST if manifest is None else Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("resources", "unavailable"):
        entries = payload.setdefault(key, {})
        if not isinstance(entries, dict):
            raise ValueError(f"resource catalogue {key} must be a mapping")
        for name, record in entries.items():
            if _relative_name(name) != name or not isinstance(record, dict):
                raise ValueError("resource records require normalized names and mappings")
            if key == "resources":
                digest = record.get("sha256")
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("resource records require a lowercase SHA256 digest")
                if "url" in record:
                    if not isinstance(record["url"], str) or not record["url"]:
                        raise ValueError("resource URL must be a nonempty string")
                    positive_int(record.get("size_bytes"), "resource size bytes")
                for related in record.get("related_resources", []):
                    _relative_name(related)
            elif not isinstance(record.get("reason"), str):
                raise ValueError("unavailable resources require an explicit reason")
    if set(payload["resources"]) & set(payload["unavailable"]):
        raise ValueError("available and unavailable names must be disjoint")
    _CATALOGUES[root] = payload


def case_workspace(directory: str | Path | None = None) -> Path:
    """Return a writable directory without copying anything from the installation.

    An explicit directory takes precedence over PYMHM_WORKSPACE and the current
    directory. A local .pymhm-resources.json is registered once when present.
    No resources are downloaded, and the process working directory is unchanged.
    Spawn workers can inherit the same environment and working directory.
    """
    root = (
        Path(directory if directory is not None else os.environ.get("PYMHM_WORKSPACE", Path.cwd()))
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    if root not in _CATALOGUES and (root / RESOURCE_MANIFEST).is_file():
        register_resources(root)
    return root


def _catalogue(directory: Path) -> dict[str, Any]:
    """Read an explicitly registered directory's resource records."""
    return _CATALOGUES.get(directory, {"resources": {}, "unavailable": {}})


def _cache_directory(cache: str | Path | None) -> Path:
    """Select an explicit or conventional writable resource cache."""
    return (
        Path(cache).expanduser()
        if cache is not None
        else Path(os.environ.get("PYMHM_CACHE_DIR", Path.home() / ".cache" / "pymhm"))
    )


def _remember_resource(path: Path, name: str) -> Path:
    """Associate cached literal bytes with their external resource label."""
    _RESOURCE_LABELS[path.resolve()] = name
    return path


def resource_file(
    name: str | Path,
    *,
    directory: str | Path | None = None,
    package: str | None = None,
    cache: str | Path | None = None,
) -> Path:
    """Resolve a workspace, explicitly catalogued or package-owned immutable input.

    Without package, names are relative to the workspace. Missing catalogued
    inputs are acquired with their declared SHA256 and byte limit. An explicit
    package resolves that package's own relative resource, including zipped
    installations. Existing declared immutable files are verified. For mutable
    local inputs use local_resource; no PyMHM case assets are distributed.
    """
    relative = _relative_name(name)
    if package is not None:
        candidate = resources.files(package).joinpath(relative)
        if not candidate.is_file():
            raise FileNotFoundError(f"Package {package!r} does not supply resource {relative!r}")
        if isinstance(candidate, Path):
            return candidate
        payload = candidate.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        target = _cache_directory(cache) / "resources" / digest / PurePosixPath(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or file_digest(target) != digest:
            target.write_bytes(payload)
        return target
    root = case_workspace(directory)
    candidate = root / relative
    catalogue = _catalogue(root)
    record = catalogue["resources"].get(relative)
    if candidate.is_file():
        if record is not None and file_digest(candidate) != record["sha256"]:
            raise ValueError(f"Declared resource differs from its catalogue: {relative}")
        return _remember_resource(candidate, relative)
    if record is not None and "url" in record:
        target = _cache_directory(cache) / "resources" / record["sha256"] / Path(relative).name
        return _remember_resource(
            download_resource(
                record["url"], target, sha256=record["sha256"], maximum_bytes=record["size_bytes"]
            ),
            relative,
        )
    if relative in catalogue["unavailable"]:
        raise FileNotFoundError(
            f"Resource {relative}: {catalogue['unavailable'][relative]['reason']}"
        )
    raise FileNotFoundError(f"Missing local or externally catalogued resource: {relative}")


def materialize_resources(
    names: Iterable[str | Path], directory: str | Path, *, package: str | None = None
) -> list[Path]:
    """Copy requested resources and attribution files, preserving existing user files.

    Destination symlinks cannot direct writes outside the directory. Only
    requested names and their explicitly declared related resources are acquired.
    """
    root = case_workspace(directory)
    pending = list(names)
    if package is None:
        for name in list(pending):
            pending.extend(
                _catalogue(root)["resources"].get(str(name), {}).get("related_resources", [])
            )
    copied = []
    for name in dict.fromkeys(pending):
        relative = _relative_name(name)
        destination = root / relative
        if not destination.resolve().is_relative_to(root):
            raise ValueError("resource destination escapes the working directory")
        if destination.exists():
            continue
        source = resource_file(relative, directory=root, package=package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(destination)
    return copied


def ensure_resource(name: str | Path, directory: str | Path) -> Path:
    """Acquire one declared input and its attribution files only when absent."""
    relative = _relative_name(name)
    materialize_resources([relative], directory)
    return Path(directory).expanduser().resolve() / relative


def local_resource(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Acquire absent declared inputs while preserving existing local files."""
    value = Path(path)
    if value.exists():
        return value
    absolute = value.resolve()
    roots = (
        [case_workspace(root)]
        if root is not None
        else sorted(_CATALOGUES, key=lambda p: len(p.parts), reverse=True)
    )
    for directory in roots:
        if absolute.is_relative_to(directory):
            relative = absolute.relative_to(directory).as_posix()
            catalogue = _catalogue(directory)
            if relative in catalogue["resources"] or relative in catalogue["unavailable"]:
                return ensure_resource(relative, directory)
    return value


def _matches_glob(parts: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    """Match components, with ** admitting zero or more directories."""
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return _matches_glob(parts, pattern[1:]) or bool(
            parts and _matches_glob(parts[1:], pattern)
        )
    return bool(
        parts and fnmatchcase(parts[0], pattern[0]) and _matches_glob(parts[1:], pattern[1:])
    )


def resource_glob(
    directory: str | Path,
    pattern: str,
    *,
    recursive: bool = False,
    root: str | Path | None = None,
) -> list[Path]:
    """List sorted unique local and available external inputs matching a glob.

    Wildcards match components; ** matches zero or more directories.
    recursive=True searches descendants as Path.rglob. Unavailable originals
    are never substituted. Acquisition checkpoint search must use filesystem
    globbing so published observations cannot silently become resumed results.
    """
    folder = Path(directory).expanduser().resolve()
    components = PurePosixPath(_relative_name(pattern)).parts
    if recursive:
        components = ("**", *components)
    found = set(folder.rglob(pattern) if recursive else folder.glob(pattern))
    roots = (
        [case_workspace(root)]
        if root is not None
        else sorted(_CATALOGUES, key=lambda p: len(p.parts), reverse=True)
    )
    for workspace in roots:
        if folder.is_relative_to(workspace):
            prefix = folder.relative_to(workspace).parts
            for name in _catalogue(workspace)["resources"]:
                parts = PurePosixPath(name).parts
                if parts[: len(prefix)] == prefix and _matches_glob(
                    parts[len(prefix) :], components
                ):
                    found.add(ensure_resource(name, workspace))
            break
    return sorted(found)


def read_resource_text(
    path: str | Path, encoding: str | None = None, errors: str | None = None
) -> str:
    """Read local or declared external text with ordinary Path decoding rules."""
    return local_resource(path).read_text(encoding=encoding, errors=errors)


def read_resource_bytes(path: str | Path) -> bytes:
    """Read literal bytes of a local or externally catalogued workspace input."""
    return local_resource(path).read_bytes()


def workspace_from_archive(
    url: str,
    *,
    sha256: str,
    directory: str | Path | None = None,
    maximum_bytes: int = 64 * 1024**2,
    maximum_extracted_bytes: int = 256 * 1024**2,
    timeout: float = 60.0,
) -> Path:
    """Acquire a verified ZIP and extract regular files into a writable workspace.

    Download and expanded byte limits are independent. Absolute paths,
    traversal, duplicate names, symlinks, encryption and destination escapes
    are rejected before extraction. Existing local files are preserved.
    Extraction does not import or execute any code. An external catalogue
    named .pymhm-resources.json is registered after extraction when present.
    """
    limit = positive_int(maximum_extracted_bytes, "maximum extracted bytes")
    archive = download_resource(
        url,
        _cache_directory(None) / "archives" / sha256 / "resources.zip",
        sha256=sha256,
        maximum_bytes=maximum_bytes,
        timeout=timeout,
    )
    root = case_workspace(directory)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        names: set[str] = set()
        total = 0
        for member in members:
            name = _relative_name(member.filename)
            target = root / name
            if name in names or not target.resolve().is_relative_to(root):
                raise ValueError("archive has duplicate names or escaping destinations")
            names.add(name)
            mode = member.external_attr >> 16
            if member.flag_bits & 1 or (mode & 0o170000) not in (0, 0o100000, 0o040000):
                raise ValueError("archive supports only unencrypted regular files and directories")
            total += member.file_size
            if total > limit:
                raise ValueError("archive exceeds the maximum extracted byte limit")
        with tempfile.TemporaryDirectory(prefix=".pymhm-extract-", dir=root) as temporary:
            stage = Path(temporary)
            for member in members:
                target = root / _relative_name(member.filename)
                if member.is_dir() or target.exists():
                    continue
                pending = stage / _relative_name(member.filename)
                pending.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as stream, pending.open("wb") as output:
                    shutil.copyfileobj(stream, output)
            for pending in sorted(stage.rglob("*")):
                if pending.is_file():
                    target = root / pending.relative_to(stage)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pending.replace(target)
    if (root / RESOURCE_MANIFEST).is_file():
        register_resources(root)
    return root


def source_file(name: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve a real workspace source or an importable package's logical owner.

    src/<package>/... identifies that installed package. Other relative names
    first identify a workspace file, then an importable package with that name.
    Absolute paths retain their literal meaning. No missing development
    metadata or source identity is manufactured.
    """
    path = Path(name)
    if path.is_absolute():
        return path
    relative = _relative_name(name)
    base = Path(root) if root is not None else case_workspace()
    candidate = base / relative
    if not relative.startswith("src/") and candidate.exists():
        return candidate
    parts = PurePosixPath(relative).parts
    package, suffix = (
        (parts[1], parts[2:]) if parts[0] == "src" and len(parts) > 2 else (parts[0], parts[1:])
    )
    if suffix and path.suffix == ".py":
        try:
            directory = _package_directory(package)
        except ModuleNotFoundError:
            return candidate
        prefix = f"src/{package}" if parts[0] == "src" else package
        _SOURCE_ROOTS[directory] = prefix
        return directory.joinpath(*suffix)
    if relative in _catalogue(base.resolve())["resources"]:
        return resource_file(relative, directory=base)
    return candidate


def source_label(path: str | Path, root: str | Path) -> str:
    """Label literal files by their resource, source-owner or workspace path."""
    value = Path(path).resolve()
    if value in _RESOURCE_LABELS:
        return _RESOURCE_LABELS[value]
    _SOURCE_ROOTS.setdefault(_package_directory("pymhm"), "src/pymhm")
    for base, prefix in _SOURCE_ROOTS.items():
        if value.is_relative_to(base):
            return (PurePosixPath(prefix) / value.relative_to(base).as_posix()).as_posix()
    base = Path(root).resolve()
    return value.relative_to(base).as_posix() if value.is_relative_to(base) else value.as_posix()


def source_identity(root: str | Path, files: Iterable[str | Path]) -> dict[str, str]:
    """Fingerprint required source bytes with stable logical labels and SHA256."""
    identities = {}
    for file in files:
        path = source_file(file, root=root)
        identities[source_label(path, root)] = file_digest(path)
    return dict(sorted(identities.items()))
