"""Portable source identities for reproducible scientific acquisition records."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


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
