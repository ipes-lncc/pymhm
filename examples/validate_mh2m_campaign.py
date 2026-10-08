"""Validate archived MH2M campaigns without rerunning or relabelling acquisitions.

Use ``python -m examples.validate_mh2m_campaign`` after obtaining the archived
fields. Configuration, archive bytes and norm quadratures are checked against
the acquired records. The resulting sidecars retain a content-addressed copy of
each input manifest. They do not assert equality of different source revisions
or recompute the physical norms.

For new CG3 analyses use the comparator's ``--output NEW.json``. For new MHM or
MH2M acquisitions use the campaign's ``--output NEW_DIRECTORY``. These commands
do not need access to comparison tools or review files outside the repository.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from examples.campaign_provenance import verify_archive
from examples.compare_mh2m_cg3 import (
    DATA,
    ROOT,
    SOURCES,
    recorded_configuration,
    reference_metadata,
    save_validation,
    validate_existing,
)
from examples.compare_mh2m_cg3_controls import validate_existing as validate_controls
from examples.mh2m_crisscross_campaign import validate_campaign_resume as validate_crisscross
from examples.mh2m_heterogeneous import data_conventions
from examples.mh2m_heterogeneous import validate_campaign_resume as validate_structured
from pymhm.io.provenance import file_digest

KINDS = {
    "structured": "comparison.json",
    "crisscross": "crisscross/comparison.json",
    "cg3": "cg3/comparison.json",
    "cg3-controls": "cg3/structured-comparison.json",
}


def independent_reference_evidence(data: Path) -> dict[str, Any] | None:
    """Link an existing native verification without claiming to execute its solver."""
    path = data / "cg1-native-verification.json"
    if not path.exists():
        return None
    payload = path.read_bytes()
    record = json.loads(payload)
    if record.get("source_changed_during_comparison") is not False:
        raise ValueError("independent reference comparison lacks a successful source guard")
    for row in record["fields"]:
        verify_archive(data / row["archive"], row["archive_sha256"])
    digest = hashlib.sha256(payload).hexdigest()
    if file_digest(path) != digest:
        raise ValueError("independent verification record changed during validation")
    return dict(
        record=path.name,
        record_sha256=digest,
        declared_method=record["method"],
        declared_scope=record["scope"],
        linkage_scope=(
            "Existing independent numerical verification. This validation command checks "
            "only its report/archive identities; it does not execute the native solver."
        ),
    )


def validate_campaign(data: Path, kind: str) -> dict[str, Any]:
    """Check one acquired manifest and write a separately attributed validation record."""
    target = data / KINDS[kind]
    payload = target.read_bytes()
    record = json.loads(payload)
    source_names = set(SOURCES) | {
        "examples/validate_mh2m_campaign.py",
        "examples/mh2m_crisscross_campaign.py",
        "examples/compare_mh2m_cg3_controls.py",
    }
    sources = {name: file_digest(ROOT / name) for name in sorted(source_names)}
    configuration: dict[str, Any]
    checked: list[str]
    if kind == "cg3":
        configuration = recorded_configuration(record)
        acquisitions = [
            reference_metadata(n, configuration["assembly_order"], data / "cg3")
            for n in configuration["sizes"]
        ]
        control = reference_metadata(
            configuration["sizes"][-1], configuration["quadrature_control"], data / "cg3"
        )
        cases = json.loads((data / "crisscross/comparison.json").read_text())["cases"]
        checked = validate_existing(
            record, acquisitions, control, cases, data / "cg3", configuration
        )
    elif kind == "cg3-controls":
        metadata = reference_metadata(512, 16, data / "cg3")
        cases = json.loads((data / "comparison.json").read_text())["cases"]
        checked = validate_controls(record, metadata, cases, data)
        configuration = dict(reference_resolution=512, assembly_order=16, orders=[8, 10])
    elif kind == "crisscross":
        reference_record = json.loads((data / "comparison.json").read_text())
        references = [
            {
                **row,
                "archive": Path(os.path.relpath(data / row["archive"], target.parent)).as_posix(),
            }
            for row in reference_record["references"]
        ]
        validate_crisscross(record, target.parent, references)
        configuration = dict(conventions=record["conventions"], references=references)
        checked = [row["name"] for row in record["cases"]]
    else:
        configuration = validate_structured(record, data)
        checked = [row["archive"] for row in (*record["references"], *record["cases"])]
    source_check = {
        "executed_source_sha256": record["source_sha256"],
        "acquisition_source_phases": record.get("acquisition_sources", []),
        "recorded_source_changed_during_run": record.get("source_changed_during_run"),
        "validation_source_sha256": sources,
        "current_data_attribution": data_conventions(),
        "attribution_scope": (
            "Current scientific attribution; acquired metadata are preserved verbatim."
        ),
        "scope": (
            "Configuration/archive validation only, not a numerical replay or proof of "
            "equivalence between source revisions. Historical source hashes are retained."
        ),
    }
    evidence = independent_reference_evidence(data)
    if evidence is not None:
        source_check["independent_reference_verification"] = evidence
    if sources != {name: file_digest(ROOT / name) for name in sorted(source_names)}:
        raise RuntimeError("validation sources changed during the checks")
    acquired_digest = hashlib.sha256(payload).hexdigest()
    save_validation(target, configuration, source_check, checked, acquired_digest)
    return dict(kind=kind, acquired_manifest_sha256=acquired_digest, checked_results=len(checked))


def main() -> None:
    """Validate selected archived campaigns without executing any FEM solve or norm integral."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--kind", nargs="+", choices=list(KINDS), default=list(KINDS))
    args = parser.parse_args()
    print(json.dumps([validate_campaign(args.data, kind) for kind in args.kind], indent=2))


if __name__ == "__main__":
    main()
