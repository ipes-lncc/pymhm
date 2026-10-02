"""Compare archived MH²M fields with an independently refined classical P3 solution.

Every integral resolves both physical fine meshes. This reference assessment is
separate from comparisons with the article's continuous P1 reference on n=128.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.campaign_provenance import (
    index_records,
    positive_integers,
    require_equal,
    source_validation,
    validation_record,
)
from examples.mh2m_campaign_contracts import verify_difference_result as verify_result
from examples.mh2m_cg_reference import CubicTriangularField, coefficient
from examples.mh2m_crisscross_norms import CrossedP1, common_triangles
from examples.mh2m_heterogeneous import load_field
from examples.mh2m_heterogeneous_norms import difference

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/mh2m-heterogeneous"
SOURCES = (
    "examples/compare_mh2m_cg3.py",
    "examples/campaign_provenance.py",
    "examples/mh2m_campaign_contracts.py",
    "examples/mh2m_heterogeneous.py",
    "examples/mh2m_cg_reference.py",
    "examples/mh2m_crisscross_norms.py",
    "examples/mh2m_heterogeneous_norms.py",
    "src/pymhm/cut_cells.py",
    "src/pymhm/elements.py",
)


def digest(path: Path) -> str:
    """Identify the bytes actually used for a source or physical field."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_metadata(n: int, order: int, directory: Path) -> dict[str, Any]:
    """Verify an independent acquisition before using its persisted field."""
    metadata = json.loads((directory / f"reference-cg3-n{n}-q{order}.json").read_text())
    if (
        metadata["source_changed_during_run"]
        or metadata["archive_sha256"] != digest(directory / metadata["archive"])
        or metadata["resolution"] != n
        or metadata["quadrature_degree"] != order
        or metadata["degree"] != 3
    ):
        raise ValueError("cubic reference provenance does not match its physical field")
    return metadata


def mathematical_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Exclude scheduling choices while freezing all mathematical CLI parameters."""
    sizes = positive_integers(args.sizes, label="resolutions", increasing=True)
    orders = positive_integers(args.orders, label="norm quadratures")
    positive_integers([args.assembly_order], label="assembly quadrature")
    positive_integers([args.quadrature_control], label="control quadrature")
    return dict(
        sizes=list(sizes),
        assembly_order=args.assembly_order,
        quadrature_control=args.quadrature_control,
        orders=list(orders),
    )


def recorded_configuration(record: dict[str, Any]) -> dict[str, Any]:
    """Read an explicit contract or infer only values present in complete legacy records."""
    if "mathematical_configuration" in record:
        return record["mathematical_configuration"]
    acquisitions = record["reference_acquisitions"]
    control = record.get("assembly_quadrature_control")
    if not acquisitions or control is None:
        raise ValueError("legacy campaign lacks a complete recorded mathematical configuration")
    return dict(
        sizes=[row["resolution"] for row in acquisitions],
        assembly_order=acquisitions[0]["quadrature_degree"],
        quadrature_control=control["acquisition"]["quadrature_degree"],
        orders=[int(key.removeprefix("quadrature_")) for key in control["norms"]],
    )


def validate_existing(
    record: dict[str, Any],
    acquisitions: list[dict[str, Any]],
    control: dict[str, Any],
    cases: list[dict[str, Any]],
    directory: Path,
    configuration: dict[str, Any],
) -> list[str]:
    """Validate every completed row and archive before selecting stages or case names."""
    require_equal(recorded_configuration(record), configuration, label="mathematical configuration")
    require_equal(record["reference_acquisitions"], acquisitions, label="reference acquisitions")
    orders = configuration["orders"]
    finest = acquisitions[-1]
    finest_path = directory / finest["archive"]
    checked = []
    expected_increments = {
        f"cg3-n{coarse['resolution']}-to-n{fine['resolution']}": (coarse, fine)
        for coarse, fine in zip(acquisitions[:-1], acquisitions[1:], strict=True)
    }
    for name, row in index_records(record["increments"], key="name").items():
        if name not in expected_increments:
            raise ValueError("completed increment is not in the requested resolution sequence")
        coarse, fine = expected_increments[name]
        verify_result(
            row,
            dict(
                name=name,
                reference_archive=fine["archive"],
                reference_sha256=fine["archive_sha256"],
                other_archive=coarse["archive"],
                other_sha256=coarse["archive_sha256"],
            ),
            archives={
                directory / fine["archive"]: fine["archive_sha256"],
                directory / coarse["archive"]: coarse["archive_sha256"],
            },
            norm_orders=orders,
        )
        checked.append(name)
    if "assembly_quadrature_control" in record:
        verify_result(
            record["assembly_quadrature_control"],
            {"acquisition": control},
            archives={
                directory / control["archive"]: control["archive_sha256"],
                finest_path: finest["archive_sha256"],
            },
            norm_orders=orders,
        )
        checked.append("assembly_quadrature_control")
    if "cg1_reference_comparison" in record:
        row = record["cg1_reference_comparison"]
        archive = directory.parent / "reference-n1024.npz"
        verify_result(
            row,
            dict(
                archive=archive.name,
                archive_sha256=digest(archive),
                reference_archive=finest["archive"],
            ),
            archives={archive: row["archive_sha256"], finest_path: finest["archive_sha256"]},
            norm_orders=orders,
        )
        checked.append("cg1_reference_comparison")
    current_cases = index_records(cases, key="name")
    for name, row in index_records(record["cases"], key="name").items():
        if name not in current_cases:
            raise ValueError("completed case is absent from the acquisition manifest")
        case = current_cases[name]
        verify_result(
            row,
            dict(
                name=name,
                archive=case["archive"],
                archive_sha256=case["archive_sha256"],
                reference_archive=finest["archive"],
                reference_sha256=finest["archive_sha256"],
            ),
            archives={
                directory.parent / "crisscross" / case["archive"]: case["archive_sha256"],
                finest_path: finest["archive_sha256"],
            },
            norm_orders=orders,
        )
        checked.append(name)
    return checked


def save_validation(
    target: Path,
    configuration: dict[str, Any],
    source_check: dict[str, Any],
    checked: list[str],
    expected_manifest_sha256: str,
) -> None:
    """Write a separate validation record linked to the exact acquired manifest bytes."""
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("acquired manifest changed during validation")
    snapshot = target.with_name(
        target.stem + "-validated-input-" + expected_manifest_sha256 + ".json"
    )
    if snapshot.exists():
        if snapshot.read_bytes() != payload:
            raise ValueError("validation snapshot already exists with different bytes")
    else:
        snapshot.write_bytes(payload)
    report = validation_record(
        snapshot, configuration=configuration, sources=source_check, checked_results=checked
    )
    path = target.with_name(target.stem + "-resume-validation.json")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Checkpoint reference increments and unchanged-case comparisons with two norm rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--output", type=Path, help="new comparison JSON using the same input archives"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--assembly-order", type=int, default=16)
    parser.add_argument("--quadrature-control", type=int, default=12)
    parser.add_argument("--orders", nargs=2, type=int, default=[8, 10])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage", choices=["reference", "cases", "all"], default="all")
    parser.add_argument("--names", nargs="*")
    parser.add_argument(
        "--resume-review",
        type=Path,
        help="explicit before/after SHA review of validation-only source changes",
    )
    args = parser.parse_args()
    configuration = mathematical_configuration(args)
    directory = args.data / "cg3"
    target = args.output if args.output is not None else directory / "comparison.json"
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    acquired_payload = target.read_bytes() if target.exists() else None
    record: dict[str, Any] = (
        json.loads(acquired_payload)
        if acquired_payload is not None
        else {
            "method": "Classical continuous P3 assembled independently with UFL/DOLFINx",
            "material": "gamma=1.8, epsilon=1/14; unmodified scalar coefficient",
            "source": "-2*x*(x-1)-2*y*(y-1)",
            "boundary": "homogeneous Dirichlet",
            "norms": "physical L2 pressure, L2 flux, L2 gradient and diffusion energy",
            "source_sha256": hashes,
            "mathematical_configuration": configuration,
            "reference_acquisitions": [],
            "increments": [],
            "cases": [],
        }
    )
    reviewed = json.loads(args.resume_review.read_text()) if args.resume_review else {}
    try:
        source_check = source_validation(
            record["source_sha256"],
            hashes,
            numerical_sources=set(SOURCES)
            - {"examples/compare_mh2m_cg3.py", "examples/campaign_provenance.py"},
            reviewed=reviewed,
        )
    except ValueError as error:
        raise ValueError(
            f"{error}. Validate archived results with python -m examples.validate_mh2m_campaign "
            "or acquire a new analysis with --output NEW.json."
        ) from error
    if acquired_payload is not None:
        require_equal(
            recorded_configuration(record), configuration, label="mathematical configuration"
        )

    def checkpoint() -> None:
        """Atomically persist only completed norms from unchanged source files."""
        if hashes != {name: digest(ROOT / name) for name in SOURCES}:
            raise RuntimeError("comparison sources changed while integrating")
        validate_existing(record, acquisitions, control, case_rows, directory, configuration)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(target)

    def compare(reference: Any, other: Any, *, crossed: bool = False) -> dict[str, Any]:
        """Reuse the shared physical norm owner and its deterministic reduction order."""
        result = {}
        for order in args.orders:
            geometry = common_triangles(reference.resolution, other.resolution) if crossed else None
            result[f"quadrature_{order}"] = difference(
                reference, other, coefficient, order, workers=args.workers, geometry=geometry
            )
        return result

    with threadpool_limits(1):
        acquisitions = [reference_metadata(n, args.assembly_order, directory) for n in args.sizes]
        control = reference_metadata(args.sizes[-1], args.quadrature_control, directory)
        campaign = args.data / "crisscross"
        case_rows = (
            json.loads((campaign / "comparison.json").read_text())["cases"]
            if record["cases"] or args.stage in ("cases", "all")
            else []
        )
        if acquired_payload is not None:
            checked = validate_existing(
                record, acquisitions, control, case_rows, directory, configuration
            )
            save_validation(
                target,
                configuration,
                source_check,
                checked,
                hashlib.sha256(acquired_payload).hexdigest(),
            )
        else:
            record["reference_acquisitions"] = acquisitions
            checkpoint()
        finest_metadata = acquisitions[-1]
        finest = CubicTriangularField.load(directory / finest_metadata["archive"])
        if args.stage in ("reference", "all"):
            for coarse, fine in zip(acquisitions[:-1], acquisitions[1:], strict=True):
                name = f"cg3-n{coarse['resolution']}-to-n{fine['resolution']}"
                if any(row["name"] == name for row in record["increments"]):
                    continue
                row = dict(
                    name=name,
                    reference_archive=fine["archive"],
                    reference_sha256=fine["archive_sha256"],
                    other_archive=coarse["archive"],
                    other_sha256=coarse["archive_sha256"],
                    postprocessing_source_sha256=hashes,
                    norms=compare(
                        CubicTriangularField.load(directory / fine["archive"]),
                        CubicTriangularField.load(directory / coarse["archive"]),
                    ),
                )
                record["increments"].append(row)
                checkpoint()
                print(json.dumps(row), flush=True)
            if "assembly_quadrature_control" not in record:
                record["assembly_quadrature_control"] = dict(
                    acquisition=control,
                    postprocessing_source_sha256=hashes,
                    norms=compare(
                        finest, CubicTriangularField.load(directory / control["archive"])
                    ),
                )
                checkpoint()
            if "cg1_reference_comparison" not in record:
                archive = args.data / "reference-n1024.npz"
                record["cg1_reference_comparison"] = dict(
                    archive=archive.name,
                    archive_sha256=digest(archive),
                    reference_archive=finest_metadata["archive"],
                    postprocessing_source_sha256=hashes,
                    norms=compare(finest, load_field(archive)),
                )
                checkpoint()
                print(json.dumps(record["cg1_reference_comparison"]), flush=True)
        if args.stage in ("cases", "all"):
            for case in case_rows:
                if (args.names and case["name"] not in args.names) or any(
                    row["name"] == case["name"] for row in record["cases"]
                ):
                    continue
                archive = campaign / case["archive"]
                if digest(archive) != case["archive_sha256"]:
                    raise ValueError("MH2M field differs from its acquisition record")
                with np.load(archive, allow_pickle=False) as arrays:
                    other = CrossedP1.from_arrays(arrays["vertices"], arrays["pressure"])
                row = dict(
                    name=case["name"],
                    archive=case["archive"],
                    archive_sha256=case["archive_sha256"],
                    reference_archive=finest_metadata["archive"],
                    reference_sha256=finest_metadata["archive_sha256"],
                    postprocessing_source_sha256=hashes,
                    norms=compare(finest, other, crossed=True),
                )
                record["cases"].append(row)
                checkpoint()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
