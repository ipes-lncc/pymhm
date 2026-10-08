"""Periodic Darcy resonance and face enrichment from Paredes et al. (2017)."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm.io.provenance import current_source_manifest

if __package__:
    from .periodic_norms import difference
else:
    from examples.periodic_norms import difference

from examples.formulations.conforming import (
    conforming_quadrilateral as solve_conforming_quadrilateral,
)
from examples.formulations.conforming import conforming_separated as solve_separable_diffusion
from pymhm.linalg.separable import solve_separable_krylov
from pymhm.materials.separable import SeparableField
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "build/results/periodic"
RECORD = ROOT / "examples/results/periodic.json"
EPSILON = np.pi / 150


def case_conventions() -> dict[str, float | str]:
    """Declare the selected dimensionless unit-square Darcy data and physical BC."""
    return dict(
        epsilon=EPSILON,
        coefficient="1+100*cos(pi*x/epsilon)^2*sin(pi*y/epsilon)^2",
        source="sin(x)*sin(y)",
        boundary="homogeneous Dirichlet pressure on every exterior face",
    )


def validate_case_provenance(record: dict, *, allow_legacy: bool = False) -> bool:
    """Reject a manifest that would reinterpret archived fields with different data.

    Provided case conventions must match the selected physical problem, and a
    provided source manifest must identify this executed case owner. Historical
    records without source hashes are accepted only with ``allow_legacy=True``;
    the returned False explicitly leaves their material provenance unverified.
    Hash equality verifies source identity, rather than a numerical error bound.
    """
    for declared in (record.get("configuration"), record.get("case_conventions")):
        if declared is not None and any(
            declared.get(key) != value for key, value in case_conventions().items()
        ):
            raise ValueError("periodic material/case conventions differ from the archived problem")
    hashes = record.get("source_hashes", record.get("source_sha256"))
    if hashes is None:
        if allow_legacy:
            return False
        raise ValueError("periodic material/case source provenance is unavailable")
    if hashes.get("examples/verify_periodic.py") != fingerprint(Path(__file__)):
        raise ValueError("periodic material/case source provenance mismatch")
    return True


def material(points: np.ndarray) -> np.ndarray:
    """Evaluate the published 101:1 periodic scalar coefficient."""
    x, y = np.pi * points.T / EPSILON
    return 1 + 100 * np.cos(x) ** 2 * np.sin(y) ** 2


def source(points: np.ndarray) -> np.ndarray:
    """Use sin(x)sin(y), without inserting a spurious pi frequency."""
    return np.sin(points[:, 0]) * np.sin(points[:, 1])


def factor_x(coordinates: np.ndarray) -> np.ndarray:
    """Represent the exact oscillatory x factor, preserving the original argument order."""
    return 100 * np.cos(np.pi * coordinates / EPSILON) ** 2


def factor_y(coordinates: np.ndarray) -> np.ndarray:
    """Represent the exact oscillatory y factor without coefficient approximation."""
    return np.sin(np.pi * coordinates / EPSILON) ** 2


def fingerprint(path: Path) -> str:
    """Hash an artifact or source in bounded memory for numerical provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(*, snapshot: bool = False) -> dict[str, str]:
    """Record the implementation used by a new acquisition, without inferring old hashes."""
    sources = [Path(__file__), ROOT / "examples/periodic_norms.py"]
    sources += sorted((ROOT / "src/pymhm").rglob("*.py"))
    result = {}
    for path in sources:
        digest = fingerprint(path)
        result[path.relative_to(ROOT).as_posix()] = digest
        if snapshot:
            destination = ARTIFACTS / "acquisition-sources" / f"{digest}.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    return current_source_manifest(result)


def save(record: dict) -> None:
    """Write current numerical records atomically after each completed solve."""
    record["reference"].sort(key=lambda row: row["n"])
    path = RECORD.with_suffix(".json.part")
    path.write_text(json.dumps(record, indent=2) + "\n")
    path.replace(RECORD)


def main() -> None:
    """Acquire reference refinement separately from the MHM face study."""
    global ARTIFACTS, RECORD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-levels", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--local-refinement", type=int, default=128)
    parser.add_argument("--segments", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--macro", type=int, default=8)
    parser.add_argument("--reference-solver", default="pyamg")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument(
        "--reference-assembly", choices=("elementwise", "separable", "lor"), default="elementwise"
    )
    parser.add_argument("--reference-degree", type=int, default=1)
    parser.add_argument("--reference-quadrature", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--refinement-precision", choices=("double", "extended"), default="double")
    parser.add_argument("--original-refinement-steps", type=int, default=2)
    parser.add_argument(
        "--stage", choices=("all", "condense", "solve", "reconstruct", "norms"), default="all"
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.native_threads < 1:
        parser.error("native-threads must be positive")
    if args.reference_assembly == "lor" and args.reference_solver != "pyamg":
        parser.error("the lor reference uses its declared PyAMG preconditioned CG solver")
    if args.workers != 1 and not args.reference_only:
        parser.error("cellwise periodic phases currently require workers=1")
    if args.reference_only and args.stage != "all":
        parser.error("reference-only acquisition uses stage=all")
    ARTIFACTS = args.artifacts
    if __package__:
        from .periodic_phases import PeriodicAcquisition, load_fields
    else:
        from examples.periodic_phases import PeriodicAcquisition, load_fields
    acquisition = (
        None
        if args.reference_only
        else PeriodicAcquisition(
            ARTIFACTS,
            macro=args.macro,
            refinement=args.local_refinement,
            segments=args.segments,
            native_threads=args.native_threads,
            refinement_precision=args.refinement_precision,
            original_refinement_steps=args.original_refinement_steps,
        )
    )
    if args.stage in {"condense", "solve", "reconstruct"}:
        getattr(acquisition, args.stage)()
        return
    acquisition_sources = source_hashes(snapshot=True)
    reference_order = args.reference_quadrature or max(4, args.reference_degree + 1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.reference_degree != 1:
        RECORD = RECORD.with_name(f"periodic-q{args.reference_degree}.json")
    suffix = f"-order{reference_order}" if args.reference_quadrature else ""
    if args.reference_assembly != "elementwise":
        suffix += f"-{args.reference_assembly}"
    if suffix:
        RECORD = RECORD.with_name(f"{RECORD.stem}{suffix}.json")
    if args.output is not None:
        RECORD = args.output
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    record = (
        json.loads(RECORD.read_text())
        if RECORD.exists()
        else dict(
            citation="10.1090/mcom/3108, section 4, Figures 4 and 6",
            epsilon=EPSILON,
            coefficient="1+100*cos(pi*x/epsilon)^2*sin(pi*y/epsilon)^2",
            source="sin(x)*sin(y)",
            reference=[],
            mhm=[],
        )
    )
    previous = None
    with threadpool_limits(args.native_threads):
        for n in args.reference_levels:
            path = ARTIFACTS / f"reference-q{args.reference_degree}-{n}{suffix}.npz"
            if path.exists():
                with np.load(path) as data:
                    pressure, residual = data["pressure"], float(data["residual"])
                    if (
                        pressure.shape != ((n * args.reference_degree + 1) ** 2,)
                        or not np.isfinite(pressure).all()
                        or not np.isfinite(residual)
                        or residual < 0
                    ):
                        raise ValueError(
                            "periodic reference arrays violate their acquisition contract"
                        )
                    reference = ConformingQuadrilateralSolution(
                        CartesianMacroMesh(n),
                        args.reference_degree,
                        pressure,
                        material,
                        residual,
                    )
                if path.with_suffix(".json").exists():
                    row = json.loads(path.with_suffix(".json").read_text())
                    if row["archive_sha256"] != fingerprint(path) or (
                        row["degree"],
                        row["n"],
                        row["quadrature_order"],
                        row["assembly"],
                    ) != (args.reference_degree, n, reference_order, args.reference_assembly):
                        raise ValueError("periodic reference manifest mismatch")
                    validate_case_provenance(row, allow_legacy=True)
                    if (
                        "basis_sha256" in row
                        or "coefficient_storage" in row
                        or row.get("schema") == "pymhm-conforming-periodic-reference-v1"
                    ):
                        if __package__:
                            from .periodic_reference import validate_reference_archive
                        else:
                            from examples.periodic_reference import validate_reference_archive
                        validate_reference_archive(path, row)
                else:
                    row = next(
                        (
                            dict(old)
                            for old in record["reference"]
                            if old["n"] == n and old.get("archive_sha256") == fingerprint(path)
                        ),
                        dict(
                            n=n,
                            degree=args.reference_degree,
                            residual=residual,
                            quadrature_order=reference_order,
                            assembly=args.reference_assembly,
                            acquisition_provenance="unavailable for legacy nodal archive",
                        ),
                    )
            else:
                if args.stage == "norms":
                    raise ValueError("the norms phase requires an acquired reference archive")
                start = perf_counter()
                assemble = {
                    "elementwise": solve_conforming_quadrilateral,
                    "separable": solve_separable_diffusion,
                    "lor": solve_separable_krylov,
                }[args.reference_assembly]
                coefficient = (
                    SeparableField(((1.0, 1.0), (factor_x, factor_y)))
                    if args.reference_assembly != "elementwise"
                    else material
                )
                forcing = (
                    SeparableField(((np.sin, np.sin),))
                    if args.reference_assembly != "elementwise"
                    else source
                )
                reference = assemble(
                    CartesianMacroMesh(n),
                    degree=args.reference_degree,
                    permeability=coefficient,
                    source=forcing,
                    quadrature_order=reference_order,
                    refinement_precision=args.refinement_precision,
                    **(
                        {"solver": args.reference_solver}
                        if args.reference_assembly != "lor"
                        else {}
                    ),
                )
                np.savez_compressed(path, pressure=reference.pressure, residual=reference.residual)
                row = dict(
                    n=n,
                    degree=args.reference_degree,
                    cells=n * n,
                    dofs=len(reference.pressure),
                    residual=reference.residual,
                    acquisition_seconds=perf_counter() - start,
                    refinement_precision=args.refinement_precision,
                    quadrature_order=reference_order,
                    solver="low-order-refined-pyamg-cg"
                    if args.reference_assembly == "lor"
                    else args.reference_solver,
                    iterations=getattr(reference, "iterations", None),
                    relative_equation_residual=getattr(
                        reference, "relative_equation_residual", None
                    ),
                    preconditioner_levels=getattr(reference, "preconditioner_levels", None),
                    assembly=args.reference_assembly,
                    native_threads=args.native_threads,
                    source_hashes=acquisition_sources,
                    python=sys.version.split()[0],
                    platform=platform.platform(),
                    numpy=np.__version__,
                    case_conventions=case_conventions(),
                )
                print("reference", {k: v for k, v in row.items() if "hash" not in k}, flush=True)
            row["archive"] = path.name
            row["archive_sha256"] = fingerprint(path)
            row["material_case_provenance_verified"] = validate_case_provenance(
                row, allow_legacy=True
            )
            reference_material_verified = row["material_case_provenance_verified"]
            record["reference"] = [old for old in record["reference"] if old["n"] != n]
            record["reference"].append(row)
            manifest = path.with_suffix(".json")
            temporary = manifest.with_suffix(".json.part")
            temporary.write_text(json.dumps(row, indent=2) + "\n")
            temporary.replace(manifest)
            if previous is not None:
                row["difference_to_previous"] = difference(reference, (previous,))
                row["comparison_source_hashes"] = acquisition_sources
                save(record)
                print("reference comparison", n, row["difference_to_previous"], flush=True)
            save(record)
            previous = reference
        if args.reference_only:
            return
        if args.stage == "all":
            acquisition.condense()
            acquisition.solve()
            acquisition.reconstruct()
        for segments in args.segments:
            field_path = ARTIFACTS / f"mhm-{args.macro}-r{args.local_refinement}-s{segments}.npz"
            local = load_fields(
                field_path, macro=args.macro, refinement=args.local_refinement, segments=segments
            )
            metadata = json.loads(field_path.with_suffix(".json").read_text())
            row = dict(
                macro=args.macro,
                refinement=args.local_refinement,
                segments=segments,
                reference_n=reference.mesh.nx,
                reference_degree=reference.degree,
                trace_dofs=args.macro * (args.macro + 1) * 2 * segments,
                residual=local[0].residual,
                archive=field_path.name,
                archive_sha256=fingerprint(field_path),
                acquisition_id=metadata["acquisition_id"],
                acquisition_source_hashes=metadata["source_hashes"],
                comparison_source_hashes=acquisition_sources,
                material_case_provenance_verified=True,
                reference_material_case_provenance_verified=reference_material_verified,
                reference_archive=path.name,
                reference_archive_sha256=fingerprint(path),
                **difference(reference, local),
            )
            record["mhm"] = [
                old
                for old in record["mhm"]
                if not (
                    old["macro"] == args.macro
                    and old["refinement"] == args.local_refinement
                    and old["segments"] == segments
                    and old["reference_n"] == reference.mesh.nx
                    and old.get("reference_degree", 1) == reference.degree
                )
            ]
            record["mhm"].append(row)
            save(record)
            print("MHM", {k: v for k, v in row.items() if "hash" not in k}, flush=True)


if __name__ == "__main__":
    main()
