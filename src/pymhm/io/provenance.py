"""Source fingerprints for fresh acquisitions, independent of archived result manifests."""

import hashlib
import importlib
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path


def current_source_manifest(
    entries: Mapping[str, str], *, packages: Iterable[str] = ("pymhm",)
) -> dict[str, str]:
    """Include every imported-package owner when producing a fresh source manifest.

    New labels describe actual current paths and bytes, including reference basis,
    local elimination and execution owners. Existing acquisition manifests must be
    validated separately and must never be relabeled through this producer.
    """
    result = dict(entries)
    for name in packages:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError("source manifests require a package with identifiable source files")
        package = Path(module.__file__).resolve().parent
        prefix = "src/pymhm" if name == "pymhm" else name.replace(".", "/")
        result.update(
            {
                f"{prefix}/{path.relative_to(package).as_posix()}": file_digest(path)
                for path in sorted(package.rglob("*.py"))
            }
        )
    return result


def file_digest(path: Path) -> str:
    """Hash a source or archive in bounded memory, including large field files."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def optional_file_digest(path: str | Path) -> str | None:
    """Return SHA256 for an existing file, or ``None`` for unavailable metadata.

    Use this for optional installation/development metadata such as a Pixi
    lockfile. Required scientific inputs must continue to use ``file_digest``
    and its explicit filesystem errors. No placeholder digest is manufactured.
    """
    source = Path(path)
    return file_digest(source) if source.is_file() else None


def _git_output(directory: str | Path, arguments: list[str]) -> str | None:
    """Query only the explicitly selected repository, without requiring Git."""
    root = Path(directory).resolve()
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def workspace_revision(directory: str | Path) -> str | None:
    """Return the selected repository's actual HEAD, or ``None`` outside a repository.

    A pip installation does not carry a Git checkout. This optional observation
    describes the workspace, not the identity of the installed numerical sources;
    source manifests retain the latter independently through their literal bytes.
    """
    return _git_output(directory, ["rev-parse", "HEAD"])


def workspace_git_dirty(directory: str | Path) -> bool | None:
    """Report tracked/untracked workspace changes, or ``None`` without repository metadata."""
    output = _git_output(directory, ["status", "--porcelain"])
    return None if output is None else bool(output)
