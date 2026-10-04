"""Validate campaign resumes without rewriting acquired numerical provenance.

Call these checks before a checkpoint or a completed-result skip. Mathematical
configuration and archive identities are immutable; worker counts, stage and
case selection belong to execution control and must be kept outside that
configuration. Validation of an old result is recorded separately from its
executed-source manifest and does not imply that it was recomputed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pymhm.io.provenance import current_source_manifest as current_source_manifest
from pymhm.io.provenance import file_digest as file_digest


def positive_integers(
    values: Sequence[int], *, label: str, increasing: bool = False
) -> tuple[int, ...]:
    """Require nonempty, distinct positive Python integers in a declared order."""
    result = tuple(values)
    if (
        not result
        or any(type(value) is not int or value <= 0 for value in result)
        or len(set(result)) != len(result)
        or (increasing and result != tuple(sorted(result)))
    ):
        raise ValueError(f"{label} must contain distinct positive integers in the required order")
    return result


def require_equal(recorded: Any, requested: Any, *, label: str) -> None:
    """Compare finite JSON contracts exactly, including list order and scalar types."""
    before = json.dumps(recorded, sort_keys=True, allow_nan=False)
    after = json.dumps(requested, sort_keys=True, allow_nan=False)
    if before != after:
        raise ValueError(f"{label} changed; existing numerical results cannot be reused")


def verify_archive(path: Path, expected_digest: str) -> str:
    """Require actual field bytes to match their recorded SHA256 before any skip."""
    actual = file_digest(path)
    if actual != expected_digest:
        raise ValueError(f"archive digest changed: {path.name}")
    return actual


def index_records(
    records: Sequence[Mapping[str, Any]], *, key: str
) -> dict[str, Mapping[str, Any]]:
    """Index completed rows without silently collapsing duplicate or absent identities."""
    result: dict[str, Mapping[str, Any]] = {}
    for row in records:
        identity = row.get(key)
        if not isinstance(identity, str) or not identity or identity in result:
            raise ValueError(f"duplicate or invalid result identity: {key}")
        result[identity] = row
    return result


def verify_result(
    recorded: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    archives: Mapping[Path, str],
    norm_orders: Sequence[int],
    norm_validator: Callable[[Mapping[str, Any], int], None] | None = None,
) -> None:
    """Validate a completed norm row against its inputs before retaining that row.

    ``identity`` names all immutable fields relevant to this comparison, including
    the reference. ``archives`` supplies the actual paths and expected digests;
    it must include at least one input field. Numerical norm values are preserved,
    while their quadrature keys and any reported orders must match the requested rules.
    Each norm map must be nonempty and JSON-finite; ``norm_validator`` supplies
    the caller's required fields and denominator conventions.
    """
    orders = positive_integers(norm_orders, label="norm orders")
    if not identity or not archives:
        raise ValueError("a completed result needs an identity and input archives")
    if any(key not in recorded for key in identity):
        raise ValueError("completed result is missing an identity field")
    require_equal({key: recorded[key] for key in identity}, dict(identity), label="result identity")
    norms = recorded.get("norms")
    if not isinstance(norms, dict) or set(norms) != {f"quadrature_{order}" for order in orders}:
        raise ValueError("completed result has different or incomplete norm quadratures")
    for order in orders:
        values = norms[f"quadrature_{order}"]
        if not isinstance(values, Mapping) or not values:
            raise ValueError("completed norm data must be a nonempty mapping")
        json.dumps(dict(values), allow_nan=False)
        if "quadrature_order" in values:
            require_equal(values["quadrature_order"], order, label="reported norm quadrature")
        if norm_validator is not None:
            norm_validator(values, order)
    for path, expected in archives.items():
        verify_archive(path, expected)


def source_validation(
    executed: Mapping[str, str],
    current: Mapping[str, str],
    *,
    numerical_sources: Collection[str],
    reviewed: Mapping[str, Mapping[str, str | None]],
) -> dict[str, Any]:
    """Keep acquired hashes and explicitly identify reviewed validation-only changes.

    Each changed or newly observed file needs a review with exact ``before`` and
    ``after`` digests and a nonempty ``reason``. Changes to already recorded
    numerical owners, removed guards and unused reviews are rejected. A newly
    observed source is not retroactive evidence of its bytes during acquisition.
    """
    if set(executed) - set(current):
        raise ValueError("an acquired source guard was removed")
    changed = {name for name in current if executed.get(name) != current[name]}
    if any(name in numerical_sources and name in executed for name in changed):
        raise ValueError("a numerical source changed; a validation-only resume is insufficient")
    if set(reviewed) != changed:
        raise ValueError("source changes need an exact, explicit validation review")
    for name in changed:
        review = reviewed[name]
        reason = review.get("reason")
        if (
            review.get("before") != executed.get(name)
            or review.get("after") != current[name]
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(f"source review does not identify the observed bytes: {name}")
    return {
        "executed_source_sha256": dict(executed),
        "validation_source_sha256": dict(current),
        "reviewed_validation_changes": {name: dict(reviewed[name]) for name in sorted(changed)},
        "newly_observed_sources": sorted(set(current) - set(executed)),
        "scope": "Validation only; newly observed sources are not acquisition provenance.",
    }


def validation_record(
    manifest: Path,
    *,
    configuration: Mapping[str, Any],
    sources: Mapping[str, Any],
    checked_results: Sequence[str],
) -> dict[str, Any]:
    """Describe a separate validation of exact acquired bytes without modifying them.

    The caller writes this return value to a separate file only after all checks.
    No timestamp or execution settings are inserted, so identical validation
    produces identical records. This record does not replace the manifest hash.
    """
    return {
        "schema": 1,
        "scope": "Resume-contract validation; no numerical result was recomputed.",
        "acquired_manifest": manifest.name,
        "acquired_manifest_sha256": file_digest(manifest),
        "mathematical_configuration": dict(configuration),
        "source_validation": dict(sources),
        "checked_results": list(checked_results),
    }
