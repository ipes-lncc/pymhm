"""Use the independent P3 reference for the separate SW–NE MH²M resolution controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from examples.campaign_provenance import (
    index_records,
    require_equal,
    source_validation,
)
from examples.compare_mh2m_cg3 import (
    DATA,
    ROOT,
    SOURCES,
    digest,
    reference_metadata,
    save_validation,
)
from examples.mh2m_campaign_contracts import verify_difference_result as verify_result
from examples.mh2m_cg_reference import CubicTriangularField, coefficient
from examples.mh2m_heterogeneous import load_field
from examples.mh2m_heterogeneous_norms import difference


def validate_existing(
    record: dict[str, Any],
    metadata: dict[str, Any],
    cases: list[dict[str, Any]],
    data: Path,
) -> list[str]:
    """Validate completed structured controls and both archived fields before a skip."""
    require_equal(record["reference_acquisition"], metadata, label="reference acquisition")
    current_cases = index_records(cases, key="archive")
    checked = []
    for name, row in index_records(record["cases"], key="archive").items():
        if name not in current_cases:
            raise ValueError("completed control is absent from the acquisition manifest")
        case = current_cases[name]
        verify_result(
            row,
            {key: case[key] for key in ("archive", "archive_sha256", "method", "family")},
            archives={
                data / name: case["archive_sha256"],
                data / "cg3" / metadata["archive"]: metadata["archive_sha256"],
            },
            norm_orders=[8, 10],
        )
        checked.append(name)
    return checked


def main() -> None:
    """Reevaluate unchanged archives using shared physical norms and checked provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--output", type=Path, help="new comparison JSON using the same input archives"
    )
    parser.add_argument(
        "--resume-review",
        type=Path,
        help="explicit before/after SHA review of validation-only source changes",
    )
    args = parser.parse_args()
    directory = DATA / "cg3"
    target = args.output if args.output is not None else directory / "structured-comparison.json"
    sources = (*SOURCES, "examples/compare_mh2m_cg3_controls.py")
    hashes = {name: digest(ROOT / name) for name in sources}
    metadata = reference_metadata(512, 16, directory)
    acquired_payload = target.read_bytes() if target.exists() else None
    record: dict[str, Any] = (
        json.loads(acquired_payload)
        if acquired_payload is not None
        else {
            "method": "SW-NE resolution controls versus independent UFL/DOLFINx P3",
            "reference_acquisition": metadata,
            "norms": "physical L2 pressure, L2 flux, L2 gradient and diffusion energy",
            "source_sha256": hashes,
            "cases": [],
        }
    )
    require_equal(record["reference_acquisition"], metadata, label="reference acquisition")
    reviewed = json.loads(args.resume_review.read_text()) if args.resume_review else {}
    try:
        source_check = source_validation(
            record["source_sha256"],
            hashes,
            numerical_sources=set(sources)
            - {
                "examples/compare_mh2m_cg3.py",
                "examples/compare_mh2m_cg3_controls.py",
                "examples/campaign_provenance.py",
            },
            reviewed=reviewed,
        )
    except ValueError as error:
        raise ValueError(
            f"{error}. Validate archived results with python -m examples.validate_mh2m_campaign "
            "or acquire a new analysis with --output NEW.json."
        ) from error
    cases = json.loads((DATA / "comparison.json").read_text())["cases"]
    checked = validate_existing(record, metadata, cases, DATA)
    if acquired_payload is not None:
        save_validation(
            target,
            dict(reference_resolution=512, assembly_order=16, orders=[8, 10]),
            source_check,
            checked,
            hashlib.sha256(acquired_payload).hexdigest(),
        )
    reference = CubicTriangularField.load(directory / metadata["archive"])
    with threadpool_limits(1):
        for case in cases:
            if any(row["archive"] == case["archive"] for row in record["cases"]):
                continue
            archive = DATA / case["archive"]
            if digest(archive) != case["archive_sha256"]:
                raise ValueError("control field differs from its acquisition record")
            other = load_field(archive)
            row = {
                "archive": case["archive"],
                "archive_sha256": case["archive_sha256"],
                "method": case["method"],
                "family": case["family"],
                "postprocessing_source_sha256": hashes,
                "norms": {
                    f"quadrature_{order}": difference(
                        reference, other, coefficient, order, workers=args.workers
                    )
                    for order in (8, 10)
                },
            }
            if hashes != {name: digest(ROOT / name) for name in sources}:
                raise RuntimeError("comparison sources changed while integrating")
            record["cases"].append(row)
            validate_existing(record, metadata, cases, DATA)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n")
            temporary.replace(target)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
