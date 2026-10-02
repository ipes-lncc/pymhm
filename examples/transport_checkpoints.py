"""Persist verified transport fields before potentially long norm integration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def write_progress(path: Path, record: dict[str, Any]) -> None:
    """Atomically publish progress without marking an unfinished case accepted."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def checkpoint_field(
    archive: Path, arrays: Mapping[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Save canonical nodal fields and the already verified physical-equation metadata.

    The companion progress record is not a completed acquisition. It keeps
    the executed sources and field digest available if later diagnostics are
    interrupted; the final case record is written only by its owning driver.
    """
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(archive)
    record = dict(
        metadata,
        archive=archive.name,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
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
