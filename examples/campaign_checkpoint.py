"""Identity and byte checks for original numerical-example checkpoints.

These checks establish the identity and completeness of persisted records,
not physical accuracy. Historical acquisition sources are never overwritten.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from examples.campaign_provenance import require_equal, verify_archive
from pymhm.io.provenance import file_digest


def require_sources(recorded: Mapping[str, str], current: Mapping[str, str]) -> None:
    """Require the full numerical source contract before adding to an acquisition.

    A driver may own forcing, spaces, norms or quadrature. It therefore receives
    the same guard as an imported operator. Archived results remain readable
    after a source change; a new acquisition uses a fresh output directory.
    """
    if not recorded or not current:
        raise ValueError("acquisition sources must be nonempty; select a fresh --output directory")
    try:
        require_equal(dict(recorded), dict(current), label="acquisition sources")
    except ValueError as error:
        raise ValueError(
            "acquisition sources differ; preserve the archived results and select "
            "a fresh --output directory for new calculations"
        ) from error


def verify_checkpoint(
    row: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    directory: Path,
    metrics: Sequence[str] = (),
    archive_key: str = "archive",
    digest_key: str = "archive_sha256",
    archive_required: bool = False,
) -> None:
    """Verify declared mathematics, finite metrics and any acquired field bytes.

    An old field without an acquisition digest cannot be certified by computing
    a digest now. Such a record requires separate retrospective validation and
    must not be silently reused by a new acquisition.
    """
    if not expected or not expected.keys() <= row.keys():
        raise ValueError("checkpoint lacks its mathematical identity")
    require_equal({key: row[key] for key in expected}, dict(expected), label="checkpoint identity")
    try:
        json.dumps(dict(row), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint contains nonfinite or invalid JSON values") from error
    for name in metrics:
        value = row.get(name)
        if not isinstance(value, Real) or isinstance(value, bool):
            raise ValueError(f"checkpoint lacks a real numerical metric: {name}")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"checkpoint metric is not finite: {name}") from error
        if not isfinite(number) or number < 0:
            raise ValueError(f"checkpoint metric must be finite and nonnegative: {name}")
    if archive_required and archive_key not in row:
        raise ValueError("checkpoint is missing its acquired field archive")
    if archive_key in row:
        if digest_key not in row:
            raise ValueError(
                "checkpoint has no acquisition digest for its field; preserve the record "
                "and use a fresh --output directory for new calculations"
            )
        verify_archive(archive_path(directory, row[archive_key]), row[digest_key])


def archive_identity(path: Path) -> dict[str, str]:
    """Record the bytes of a field immediately after its acquisition/export."""
    return {"archive": path.name, "archive_sha256": file_digest(path)}


def retrospective_files(
    directory: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    archive_key: str = "archive",
    digest_key: str = "archive_sha256",
) -> list[dict[str, Any]]:
    """Identify historical archives without inventing an earlier acquisition hash.

    A recorded digest is checked. When absent, the current bytes are explicitly
    observations only and cannot authorize resuming the old acquisition.
    """
    result = []
    for row in rows:
        if archive_key not in row:
            continue
        path = archive_path(directory, row[archive_key])
        observed = file_digest(path)
        recorded = row.get(digest_key)
        if recorded is not None:
            verify_archive(path, recorded)
        result.append(
            dict(
                archive=row[archive_key],
                recorded_acquisition_sha256=recorded,
                observed_sha256=observed,
                acquisition_bytes_verified=recorded is not None,
            )
        )
    return result


def archive_path(directory: Path, name: str) -> Path:
    """Resolve a portable relative archive name, excluding traversal and absolute paths."""
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or Path(name).is_absolute()
        or ".." in Path(name).parts
    ):
        raise ValueError("archive name must be a safe relative path")
    return directory / name
