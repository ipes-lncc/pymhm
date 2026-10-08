"""Portable source identities for reproducible scientific acquisition records."""

from __future__ import annotations

import hashlib
import importlib
import os
from collections.abc import Iterable
from pathlib import Path

from pymhm.io.provenance import file_digest, optional_file_digest
from pymhm.io.provenance import workspace_revision as workspace_revision
from pymhm.io.workspace import source_identity


def source_digests(root: Path, files: Iterable[Path]) -> dict[str, str]:
    """Hash literal file bytes under ``root`` using SHA256.

    Relative input paths are interpreted relative to ``root``; absolute paths
    are accepted inside that same root. Keys are sorted, root-relative POSIX
    paths after resolving symlinks, and values are hexadecimal SHA256 digests.
    Repeated paths produce one entry. Paths outside the root raise ``ValueError``;
    missing or unreadable files retain their ordinary filesystem exceptions.
    No optional numerical backend is imported or required.
    """
    base = root.resolve()
    identities = {}
    for path in files:
        resolved = (path if path.is_absolute() else base / path).resolve()
        relative = resolved.relative_to(base).as_posix()
        identities[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return dict(sorted(identities.items()))


def support_manifest(files: Iterable[Path]) -> dict[str, str]:
    """Hash downloaded example owners with stable package-relative labels.

    Stable ``examples/...`` labels resolve to actual ``examples`` owners.
    They identify local companion bytes; a source checkout is unnecessary.
    """
    package = importlib.import_module("examples")
    if package.__file__ is None:
        raise ValueError("companion sources require an identifiable examples package")
    return source_identity(Path(package.__file__).resolve().parent.parent, files)


def notebook_provenance(notebook: str, *, workspace: Path) -> dict[str, str | None]:
    """Identify an actual notebook source separately from an available local reference.

    ``PYMHM_NOTEBOOK_SOURCE`` declares the executed file when a runner provides
    it. Otherwise a file with the notebook's basename in the current directory
    is identifiable. If neither exists, its executed-source digest is unknown;
    an available local reference remains separately labeled. A present workspace
    ``pixi.lock`` is optional environment evidence, never an invented requirement.
    """
    selector = notebook.removeprefix("notebooks/")
    reference = workspace / "notebooks" / selector
    declared = os.environ.get("PYMHM_NOTEBOOK_SOURCE")
    source = Path(declared).expanduser().resolve() if declared else Path.cwd() / Path(selector).name
    if declared and not source.is_file():
        raise FileNotFoundError(source)
    identified = source if source.is_file() else None
    lock = workspace / "pixi.lock"
    return {
        "notebook": "notebooks/" + selector,
        "notebook_source": str(identified) if identified is not None else None,
        "notebook_sha256": file_digest(identified) if identified is not None else None,
        "reference_source": "notebooks/" + selector if reference.is_file() else None,
        "reference_source_sha256": file_digest(reference) if reference.is_file() else None,
        "pixi_lock_sha256": optional_file_digest(lock),
    }


def execution_source_manifest(notebook: str, *, workspace: Path) -> dict[str, str]:
    """Return available literal source digests without claiming unknown execution bytes."""
    evidence = notebook_provenance(notebook, workspace=workspace)
    result = {}
    if evidence["reference_source_sha256"] is not None:
        result["reference_notebook"] = evidence["reference_source_sha256"]
    if evidence["notebook_sha256"] is not None:
        result["source_notebook"] = evidence["notebook_sha256"]
    if evidence["pixi_lock_sha256"] is not None:
        result["pixi.lock"] = evidence["pixi_lock_sha256"]
    return result
