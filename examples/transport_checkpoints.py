"""Persist verified transport fields before potentially long norm integration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from pymhm.io.provenance import file_digest


def write_progress(path: Path, record: dict[str, Any]) -> None:
    """Atomically publish progress without marking an unfinished case accepted."""
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_field(
    archive: Path, arrays: Mapping[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Save canonical nodal fields and the already verified physical-equation metadata.

    The companion progress record is not a completed acquisition. It keeps
    the executed sources and field digest available if later diagnostics are
    interrupted; the final case record is written only by its owning driver.
    """
    json.dumps(metadata, allow_nan=False)
    if any(
        np.asarray(value).dtype.kind not in "biufc" or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("transport coefficient archives require finite numerical arrays")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    record = dict(
        metadata,
        archive=archive.name,
        archive_sha256=file_digest(archive),
        status="physical-field-verified; diagnostics incomplete",
        quadrature={},
    )
    write_progress(archive.with_suffix(".progress.json"), record)
    return record


def checkpoint_norm(
    archive: Path, record: dict[str, Any], order: int, norms: dict[str, float]
) -> None:
    """Persist a completed integration order separately from final case acceptance."""
    record["quadrature"][str(order)] = norms
    write_progress(archive.with_suffix(".progress.json"), record)
