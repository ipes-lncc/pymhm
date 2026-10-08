"""Inspect historical checkpoint files without resuming or relabeling their acquisition."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from examples.campaign_checkpoint import retrospective_files
from pymhm.io.provenance import file_digest


def inspect_manifest(path: Path) -> dict[str, Any]:
    """Check finite records and archive digests, retaining the exact acquisition manifest.

    This check does not establish physical accuracy or compatibility with current
    sources. A field lacking an acquisition digest is reported as observed only.
    It cannot authorize continuation of that historical acquisition.
    """
    raw = path.read_bytes()
    record = json.loads(raw)
    json.dumps(record, allow_nan=False)
    rows: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        """Locate archive-bearing records without interpreting formulation-specific norms."""
        if isinstance(value, dict):
            if "archive" in value:
                rows.append(value)
            elif "fields" in value:
                rows.append(
                    {
                        **value,
                        "archive": value["fields"],
                        "archive_sha256": value.get("fields_sha256"),
                    }
                )
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(record)
    return {
        "schema": 1,
        "scope": "Retrospective byte/finite-JSON validation only; no solve, norm replay, "
        "source equivalence or authorization to resume.",
        "acquired_manifest": path.name,
        "acquired_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_checks": retrospective_files(path.parent, rows),
        "validation_source_sha256": {
            item.name: file_digest(item)
            for item in (
                Path(__file__),
                Path(__file__).with_name("campaign_checkpoint.py"),
                Path(__file__).with_name("campaign_provenance.py"),
            )
        },
    }


def main() -> None:
    """Write separate evidence and an immutable copy of the inspected original manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        raise ValueError("validation output must differ from its acquisition manifest")
    result = inspect_manifest(args.manifest)
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != result["acquired_manifest_sha256"]:
        raise ValueError("acquisition manifest changed during inspection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.parent / f"acquired-{result['acquired_manifest_sha256']}.json"
    if snapshot.exists() and snapshot.read_bytes() != raw:
        raise ValueError("existing acquisition snapshot has different bytes")
    snapshot.write_bytes(raw)
    result["acquired_manifest_snapshot"] = snapshot.name
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
