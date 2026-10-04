"""Source fingerprints for fresh acquisitions, independent of archived result manifests."""

import hashlib
from collections.abc import Mapping
from pathlib import Path


def current_source_manifest(entries: Mapping[str, str]) -> dict[str, str]:
    """Include every imported-package owner when producing a fresh source manifest.

    New labels describe actual current paths and bytes, including reference basis,
    local elimination and execution owners. Existing acquisition manifests must be
    validated separately and must never be relabeled through this producer.
    """
    import pymhm

    package = Path(pymhm.__file__).resolve().parent
    result = dict(entries)
    result.update(
        {
            f"src/pymhm/{path.relative_to(package).as_posix()}": file_digest(path)
            for path in sorted(package.rglob("*.py"))
        }
    )
    return result


def file_digest(path: Path) -> str:
    """Hash a source or archive in bounded memory, including large field files."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
