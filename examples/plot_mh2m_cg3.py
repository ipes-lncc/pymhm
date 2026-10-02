"""Render both MH²M mesh families against the same independently refined P3 field.

Historical pressure profiles retain the article's P1 n=128 reference. Physical
volume-error curves and spatial differences use the explicitly identified P3
reference; the original P1 comparison records are preserved separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from examples import plot_mh2m_crisscross as crossed
from examples import plot_mh2m_heterogeneous as structured
from examples.mh2m_cg_reference import CubicTriangularField

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Identify the actual archived field or scientific record used for rendering."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_reference(directory: Path, record: dict) -> tuple[CubicTriangularField, dict]:
    """Require a complete independent refinement series and its assembly control."""
    acquisitions = record["reference_acquisitions"]
    if len(acquisitions) != 5 or len(record["increments"]) != 4:
        raise ValueError("five completed classical levels and four increments are required")
    control = record["assembly_quadrature_control"]["acquisition"]
    for row in (*acquisitions, control):
        if (
            row["source_changed_during_run"]
            or digest(directory / row["archive"]) != row["archive_sha256"]
        ):
            raise ValueError("reference archive differs from its accepted acquisition")
    metadata = acquisitions[-1]
    for previous, current, row in zip(
        acquisitions[:-1], acquisitions[1:], record["increments"], strict=True
    ):
        if (
            row["reference_archive"] != current["archive"]
            or row["reference_sha256"] != current["archive_sha256"]
            or row["other_archive"] != previous["archive"]
            or row["other_sha256"] != previous["archive_sha256"]
        ):
            raise ValueError("classical increment does not identify its two acquired fields")
        if "quadrature_8" not in row["norms"] or "quadrature_10" not in row["norms"]:
            raise ValueError("each classical increment requires both norm quadratures")
    return CubicTriangularField.load(directory / metadata["archive"]), metadata


def promoted_record(
    acquired: dict,
    comparisons: list[dict],
    source: Path,
    expected: int,
    metadata: dict,
    *,
    record_reference: dict | None = None,
) -> dict:
    """Join norms by verified archive identity, without modifying acquisition records."""
    if record_reference is not None and record_reference != metadata:
        raise ValueError("record-level reference differs from the accepted acquisition")
    if len(acquired["cases"]) != expected or len(comparisons) != expected:
        raise ValueError("every acquired case requires a completed P3 comparison")
    by_archive = {row["archive"]: row for row in comparisons}
    if len(by_archive) != expected:
        raise ValueError("comparison archives must be unique")
    cases = []
    for case in acquired["cases"]:
        row = by_archive[case["archive"]]
        reference_sha = (
            row.get("reference_sha256", record_reference["archive_sha256"])
            if record_reference is not None
            else row.get("reference_sha256")
        )
        reference_archive = (
            row.get("reference_archive", record_reference["archive"])
            if record_reference is not None
            else row.get("reference_archive")
        )
        if (
            digest(source / case["archive"]) != case["archive_sha256"]
            or row["archive_sha256"] != case["archive_sha256"]
            or reference_sha != metadata["archive_sha256"]
            or reference_archive != metadata["archive"]
        ):
            raise ValueError("comparison and acquisition do not identify the same fields")
        cases.append(
            {
                **case,
                "norms": row["norms"]["quadrature_8"],
                "norms_quadrature_check": row["norms"]["quadrature_10"],
                "reference_resolution": metadata["resolution"],
                "reference_degree": 3,
                "reference_archive_sha256": metadata["archive_sha256"],
            }
        )
    return {
        **acquired,
        "cases": cases,
        "norm_reference_suffix": f"\nReference: classical P3 n={metadata['resolution']}",
    }


def main() -> None:
    """Render final P3-based comparisons only after every physical norm is available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "examples/results/mh2m-heterogeneous")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/figures/mh2m-heterogeneous")
    args = parser.parse_args()
    cg = args.source / "cg3"
    reference_record = json.loads((cg / "comparison.json").read_text())
    control_record = json.loads((cg / "structured-comparison.json").read_text())
    reference, metadata = checked_reference(cg, reference_record)
    if control_record["reference_acquisition"] != metadata:
        raise ValueError("both mesh families must use the same P3 reference acquisition")
    diagonal = promoted_record(
        json.loads((args.source / "comparison.json").read_text()),
        control_record["cases"],
        args.source,
        18,
        metadata,
        record_reference=control_record["reference_acquisition"],
    )
    cross_source, cross_output = args.source / "crisscross", args.output / "crisscross"
    crisscross = promoted_record(
        json.loads((cross_source / "comparison.json").read_text()),
        reference_record["cases"],
        cross_source,
        24,
        metadata,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cross_output.mkdir(parents=True, exist_ok=True)
    label = f"Classical P3 n={reference.resolution}"
    field_options = {"reference_field": reference, "reference_label": label}
    for record, source, output, loader in (
        (diagonal, args.source, args.output, structured.load_field),
        (crisscross, cross_source, cross_output, crossed.load_crossed),
    ):
        spatial = {
            **record,
            "cases": [
                {**row, "family": "figure-5-discretization"}
                for row in record["cases"]
                if row.get("name") == "figure-5" or row.get("family") == "figure-5-discretization"
            ],
        }
        for transform in (False, True):
            structured.fields(
                spatial,
                source,
                output,
                field_loader=loader,
                flux_asinh=transform,
                **field_options,
            )
    structured.profiles(diagonal, args.source, args.output, **field_options)
    structured.enrichment(diagonal, args.output)
    acquisitions = {row["archive"]: row for row in reference_record["reference_acquisitions"]}
    refinement = {
        **diagonal,
        "reference_refinement_label": "Independent classical P3 refinement",
        "references": [
            {
                **acquisitions[row["reference_archive"]],
                "increment": row["norms"]["quadrature_8"],
                "increment_quadrature_check": row["norms"]["quadrature_10"],
            }
            for row in reference_record["increments"]
        ],
    }
    structured.refinement(refinement, args.output)
    crossed.profiles(crisscross, cross_source, cross_output, **field_options)
    crossed.figure6_profiles(crisscross, cross_source, cross_output, **field_options)
    crossed.published_profiles(crisscross, cross_source, cross_output)
    crossed.norms(crisscross, cross_output)
    target = args.output / "cg3"
    target.mkdir(parents=True, exist_ok=True)
    for path in cg.glob("*.json"):
        shutil.copy2(path, target / path.name)
    render_record = {
        "reference_archive": metadata["archive"],
        "reference_archive_sha256": metadata["archive_sha256"],
        "physical_norm_reference": label,
        "historical_profile_reference": "Classical P1 n=128",
        "inputs": {
            str(path.relative_to(args.source)): digest(path)
            for path in (
                cg / "comparison.json",
                cg / "structured-comparison.json",
                args.source / "comparison.json",
                cross_source / "comparison.json",
            )
        },
        "case_counts": {"crisscross": 24, "SW-NE": 18},
    }
    (target / "rendering.json").write_text(json.dumps(render_record, indent=2) + "\n")


if __name__ == "__main__":
    main()
