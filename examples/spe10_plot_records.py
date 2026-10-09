"""Immutable field identities and complete comparison families for SPE10 figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pymhm.io.workspace import local_resource, read_resource_text

if TYPE_CHECKING:
    from examples.solve_unusual_spe10 import UnusualSPE10Field


def digest(path: Path) -> str:
    """Identify field archives with bounded memory during figure replay."""
    with local_resource(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_comparison(path: Path, data: Path) -> dict:
    """Require both integration rules and the actual MHM and reference field identities."""
    row = json.loads(read_resource_text(path))
    if (
        len(row.get("norms", [])) != 2
        or {item.get("order") for item in row["norms"]} != {4, 5}
        or row.get("source_changed_during_run") is not False
        or digest(path.parent / row["mhm"]) != row["mhm_sha256"]
        or digest(data / row["reference"]) != row["reference_sha256"]
    ):
        raise ValueError(f"incomplete or inconsistent physical comparison in {path.name}")
    return row


def comparison_reference(data: Path, comparisons: list[dict]) -> Path:
    """Select the common reference of completed norms, rather than a fixed refinement label."""
    identities = {(row["reference"], row["reference_sha256"]) for row in comparisons}
    if len(identities) != 1:
        raise ValueError("all displayed norms must use the same physical reference")
    name, expected = identities.pop()
    path = data / name
    if digest(path) != expected:
        raise ValueError("the selected physical reference archive changed")
    return path


CASES = (
    ("fitted-r8-s8", "r8 / s8"),
    ("fitted-r16-s8", "r16 / s8"),
    ("fitted-r32-s8", "r32 / s8"),
    ("fitted-r32-s16", "r32 / s16"),
    ("fitted-r32-s32", "r32 / s32"),
    ("layer0.5-r8-s8", "c=.5 / r8 / s8"),
    ("layer0.5-r32-s32", "c=.5 / r32 / s32"),
    ("layer0.5-tracefit-r32-s32", "c=.5 / r32 / s32 + pixel trace"),
    ("layer0.25-tracefit-r32-s32", "c=.25 / r32 / s32 + pixel trace"),
)


PAIRED_CASES = (
    ("layer0.25-tracefit-r64-s32", "c=.25 / r64 / s32 + pixel trace"),
    ("layer0.25-tracefit-r64-s64", "c=.25 / r64 / s64 + pixel trace"),
)


def completed_cases(data: Path) -> tuple[tuple[str, str], ...]:
    """Include the paired r64 experiment only when both fields and comparisons are present."""
    paths = [
        data / f"mhm-unusual-{stem}-q5{suffix}"
        for stem, _ in PAIRED_CASES
        for suffix in (".json", ".npz", "-comparison.json")
    ]
    if any(local_resource(path).exists() for path in paths):
        if not all(local_resource(path).is_file() for path in paths):
            raise ValueError("both paired r64 fields and their norm records must be complete")
        return (*CASES, *PAIRED_CASES)
    return CASES


def validate_pair(rows: list[dict], fields: list[UnusualSPE10Field]) -> None:
    """Require the s32/s64 fields to use the same prepared local geometry and responses."""
    coarse, fine = rows[-2:]
    reuse = coarse.get("response_reuse", {})
    if (
        any(
            row.get("nominal_local_refinement") != 64
            or row.get("reaction_layer_resolution") != 0.25
            or row.get("local_degree") != 1
            or row.get("trace_degree") != 0
            or row.get("material_fitted_trace") is not True
            for row in (coarse, fine)
        )
        or (coarse.get("trace_segments"), fine.get("trace_segments")) != (32, 64)
        or reuse.get("prepared_archive_sha256") != fine["archive_sha256"]
        or reuse.get("local_responses_are_shared") is not True
    ):
        raise ValueError("the paired traces must use the same declared local problem")
    a, b = fields[-2:]
    if (
        not np.array_equal(a.macro.points, b.macro.points)
        or not np.array_equal(a.macro.cells, b.macro.cells)
        or len(a.meshes) != len(b.meshes)
        or any(
            not np.array_equal(first.points, second.points)
            or not np.array_equal(first.cells, second.cells)
            for first, second in zip(a.meshes, b.meshes, strict=True)
        )
    ):
        raise ValueError("the paired traces changed the physical local geometry")
