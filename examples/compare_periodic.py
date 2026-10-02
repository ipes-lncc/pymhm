"""Compare archived periodic Darcy fields by exact common-partition polynomial norms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from .periodic_norms import difference
    from .verify_periodic import ARTIFACTS, ROOT, fingerprint, material, source_hashes
else:
    from periodic_norms import difference
    from verify_periodic import ARTIFACTS, ROOT, fingerprint, material, source_hashes

from pymhm.conforming import ConformingQuadrilateralSolution
from pymhm.quadrilateral import CartesianMacroMesh


def load_reference(specification: str) -> tuple[ConformingQuadrilateralSolution, Path]:
    """Read an archived reference, preferring its explicit acquisition manifest.

    The syntax is ``degree:subdivisions[:order[:assembly]]``. For an LOR
    acquisition, its JSON record identifies the archive and digest independently
    of thread-count or acquisition-name suffixes. Earlier elementwise/separated
    records retain their deterministic archive names.
    """
    parts = specification.split(":")
    if len(parts) not in (2, 3, 4) or (len(parts) == 4 and parts[3] not in {"separable", "lor"}):
        raise ValueError("reference syntax is degree:subdivisions[:assembly_order[:separable|lor]]")
    entries = list(map(int, parts[:3]))
    if min(entries) < 1:
        raise ValueError("reference degree, subdivisions and quadrature must be positive")
    degree, n = entries[:2]
    suffix = f"-order{entries[2]}" if len(entries) == 3 else ""
    if len(parts) == 4:
        suffix += f"-{parts[-1]}"
    path = ARTIFACTS / f"reference-q{degree}-{n}{suffix}.npz"
    record = None
    if len(parts) == 4 and parts[-1] == "lor":
        manifest = (
            ROOT / "examples/results" / f"periodic-reference-q{degree}-{n}-order{entries[2]}.json"
        )
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
    with np.load(path) as data:
        pressure, residual = data["pressure"], float(data["residual"])
        if (
            pressure.shape != ((n * degree + 1) ** 2,)
            or not np.isfinite(pressure).all()
            or not np.isfinite(residual)
            or residual < 0
            or (record is not None and residual != record["relative_equation_residual"])
        ):
            raise ValueError("periodic reference arrays violate their acquisition contract")
        result = ConformingQuadrilateralSolution(
            CartesianMacroMesh(n), degree, pressure, material, residual
        )
    return result, path


def load_fields(macro: int, refinement: int, segments: int) -> tuple[tuple, Path]:
    """Keep the original independent nodal field in each physical macrocell."""
    mesh = CartesianMacroMesh(macro)
    path = ARTIFACTS / f"mhm-{macro}-r{refinement}-s{segments}.npz"
    with np.load(path) as data:
        fields = tuple(
            ConformingQuadrilateralSolution(
                mesh.submesh(cell, refinement), 1, p, material, float(data["residual"])
            )
            for cell, p in enumerate(data["fields"])
        )
    return fields, path


def save(path: Path, record: dict) -> None:
    """Publish a complete JSON checkpoint by atomic replacement."""
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    """Separate reference, local-discretization and face-discretization sensitivities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--compare-references", nargs="*", default=[])
    parser.add_argument("--refinements", nargs="*", type=int, default=[])
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--macro", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--primary", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/periodic-comparison.json"
    )
    args = parser.parse_args()
    sources = source_hashes(snapshot=True)
    sources[str(Path(__file__).relative_to(ROOT))] = fingerprint(Path(__file__))
    reference, reference_path = load_reference(args.reference)
    reference_hash = fingerprint(reference_path)
    record = json.loads(args.output.read_text()) if args.output.exists() else {"comparisons": []}
    record["norm_convention"] = (
        "Full broken H1: sqrt(integral(dp^2+|grad dp|^2) / "
        "integral(p_ref^2+|grad p_ref|^2)); common nested Cartesian partitions, "
        "exact tensor polynomial products, no interface averaging."
    )
    record["references_are_numerical"] = True

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
            field, path = load_reference(specification)
            compare(f"conforming:{specification}", (field,), path)
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
                )
                del fields
    if args.primary:
        record["primary_reference"] = args.reference
        save(args.output, record)


if __name__ == "__main__":
    main()
