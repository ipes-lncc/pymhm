"""Compare archived periodic Darcy fields by exact common-partition polynomial norms."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from .periodic_norms import difference
    from .periodic_reference import validate_reference_archive
    from .verify_periodic import (
        ARTIFACTS,
        ROOT,
        fingerprint,
        material,
        source_hashes,
        validate_case_provenance,
    )
else:
    from examples.periodic_norms import difference
    from examples.periodic_reference import validate_reference_archive
    from examples.verify_periodic import (
        ARTIFACTS,
        ROOT,
        fingerprint,
        material,
        source_hashes,
        validate_case_provenance,
    )

from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution

REFERENCE_RECORDS: Path | None = None


def load_reference(specification: str) -> tuple[ConformingQuadrilateralSolution, Path]:
    """Read an archived reference, preferring its explicit acquisition manifest.

    The syntax is ``degree:subdivisions[:order[:assembly]]``. For an LOR
    acquisition, its JSON record identifies the archive and digest independently
    of thread-count or acquisition-name suffixes. Earlier elementwise/separated
    records retain their deterministic archive names.
    Legacy records without case sources leave their physical material unverified.
    New manifests with executed basis matrices verify those matrices, their
    evaluation owners and the effective coefficient precision before evaluation.
    A present archive sidecar must declare the requested quadrature and assembly;
    an archive without a manifest leaves those acquisition details unverified.
    """
    field, path, _ = _load_reference(specification)
    return field, path


def _load_reference(specification: str) -> tuple[ConformingQuadrilateralSolution, Path, bool]:
    """Retain the archive's case verification state for fresh comparison records."""
    parts = specification.split(":")
    if len(parts) not in (2, 3, 4) or (len(parts) == 4 and parts[3] not in {"separable", "lor"}):
        raise ValueError("reference syntax is degree:subdivisions[:assembly_order[:separable|lor]]")
    entries = list(map(int, parts[:3]))
    if min(entries) < 1:
        raise ValueError("reference degree, subdivisions and quadrature must be positive")
    degree, n = entries[:2]
    order = entries[2] if len(entries) == 3 else max(4, degree + 1)
    assembly = parts[3] if len(parts) == 4 else "elementwise"
    suffix = f"-order{entries[2]}" if len(entries) == 3 else ""
    if len(parts) == 4:
        suffix += f"-{parts[-1]}"
    path = ARTIFACTS / f"reference-q{degree}-{n}{suffix}.npz"
    record = None
    if len(parts) == 4 and parts[-1] == "lor":
        folder = ROOT / "examples/results" if REFERENCE_RECORDS is None else REFERENCE_RECORDS
        manifest = folder / f"periodic-reference-q{degree}-{n}-order{entries[2]}.json"
        if manifest.exists():
            record = json.loads(manifest.read_text())
            if (
                (record["degree"], record["n"], record["quadrature_order"])
                != (degree, n, entries[2])
                or record["solver"] != "low-order-refined-pyamg-cg"
                or Path(record["archive"]).name != record["archive"]
            ):
                raise ValueError("periodic reference manifest does not match the requested field")
            path = ARTIFACTS / record["archive"]
            if fingerprint(path) != record["archive_sha256"]:
                raise ValueError("periodic reference archive digest mismatch")
    if record is None and path.with_suffix(".json").exists():
        record = json.loads(path.with_suffix(".json").read_text())
        if (
            record["degree"] != degree
            or record["n"] != n
            or (record.get("quadrature_order"), record.get("assembly")) != (order, assembly)
            or record["archive"] != path.name
            or record["archive_sha256"] != fingerprint(path)
        ):
            raise ValueError("periodic reference manifest mismatch")
    verified = validate_case_provenance(record, allow_legacy=True) if record is not None else False
    if record is not None and (
        "basis_sha256" in record
        or "coefficient_storage" in record
        or record.get("schema") == "pymhm-conforming-periodic-reference-v1"
    ):
        validate_reference_archive(path, record)
    recorded_residual = None if record is None else record.get("relative_equation_residual")
    if record is not None and recorded_residual is None:
        recorded_residual = record.get("residual")
    with np.load(path) as data:
        pressure, residual = data["pressure"], float(data["residual"])
        if (
            pressure.shape != ((n * degree + 1) ** 2,)
            or not np.isfinite(pressure).all()
            or not np.isfinite(residual)
            or residual < 0
            or (record is not None and residual != recorded_residual)
        ):
            raise ValueError("periodic reference arrays violate their acquisition contract")
        result = ConformingQuadrilateralSolution(
            CartesianMacroMesh(n), degree, pressure, material, residual
        )
    return result, path, verified


def load_fields(macro: int, refinement: int, segments: int) -> tuple[tuple, Path]:
    """Keep the original independent nodal field in each physical macrocell."""
    if __package__:
        from .periodic_phases import load_fields as load_archived_fields
    else:
        from examples.periodic_phases import load_fields as load_archived_fields
    path = ARTIFACTS / f"mhm-{macro}-r{refinement}-s{segments}.npz"
    fields = load_archived_fields(path, macro=macro, refinement=refinement, segments=segments)
    return fields, path


def save(path: Path, record: dict) -> None:
    """Publish a complete JSON checkpoint by atomic replacement."""
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Separate reference, local-discretization and face-discretization sensitivities."""
    global ARTIFACTS, REFERENCE_RECORDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--compare-references", nargs="*", default=[])
    parser.add_argument("--refinements", nargs="*", type=int, default=[])
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--macro", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--primary", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--reference-records", type=Path, default=ROOT / "examples/results")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/periodic-comparison.json"
    )
    args = parser.parse_args()
    ARTIFACTS, REFERENCE_RECORDS = args.artifacts, args.reference_records
    sources = source_hashes()
    sources[Path(__file__).relative_to(ROOT).as_posix()] = fingerprint(Path(__file__))
    for name, digest in sources.items():
        destination = ARTIFACTS / "acquisition-sources" / f"{digest}.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    reference, reference_path, reference_verified = _load_reference(args.reference)
    reference_hash = fingerprint(reference_path)
    record = json.loads(args.output.read_text()) if args.output.exists() else {"comparisons": []}
    record["norm_convention"] = (
        "Full broken H1: sqrt(integral(dp^2+|grad dp|^2) / "
        "integral(p_ref^2+|grad p_ref|^2)); common nested Cartesian partitions, "
        "exact tensor polynomial products, no interface averaging."
    )
    record["references_are_numerical"] = True
    record["reference_material_case_provenance_verified"] = reference_verified

    def compare(label: str, fields: tuple, path: Path, **metadata: int) -> None:
        """Append a fresh archive pair once, retaining each denominator explicitly."""
        archive_hash = fingerprint(path)
        existing = [
            row
            for row in record["comparisons"]
            if row["reference_sha256"] == reference_hash and row["field_sha256"] == archive_hash
        ]
        if existing and not args.refresh:
            return
        norms = difference(reference, fields)
        record["comparisons"] = [
            row
            for row in record["comparisons"]
            if not (
                row["reference_sha256"] == reference_hash and row["field_sha256"] == archive_hash
            )
        ]
        row = dict(
            field=label,
            reference=args.reference,
            reference_archive=reference_path.name,
            reference_sha256=reference_hash,
            field_archive=path.name,
            field_sha256=archive_hash,
            comparison_source_hashes=sources,
            **metadata,
            **norms,
        )
        record["comparisons"].append(row)
        save(args.output, record)
        print(label, norms, flush=True)

    with threadpool_limits(1):
        for specification in args.compare_references:
            field, path, verified = _load_reference(specification)
            compare(
                f"conforming:{specification}",
                (field,),
                path,
                material_case_provenance_verified=verified,
            )
            del field
        for refinement in args.refinements:
            for segments in args.segments:
                fields, path = load_fields(args.macro, refinement, segments)
                compare(
                    f"mhm:r{refinement}:s{segments}",
                    fields,
                    path,
                    macro=args.macro,
                    refinement=refinement,
                    segments=segments,
                    material_case_provenance_verified=path.with_suffix(".json").exists(),
                )
                del fields
    if args.primary:
        record["primary_reference"] = args.reference
        save(args.output, record)


if __name__ == "__main__":
    main()
