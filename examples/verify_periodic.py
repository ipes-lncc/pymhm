"""Periodic Darcy resonance and face enrichment from Paredes et al. (2017)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

if __package__:
    from .periodic_norms import difference
else:
    from periodic_norms import difference

from pymhm import FaceSpace, HybridSystem, SkeletonSpace
from pymhm.conforming import ConformingQuadrilateralSolution, solve_conforming_quadrilateral
from pymhm.quadrilateral import (
    CartesianMacroMesh,
    _assemble_quad,
    _QuadTask,
)
from pymhm.separable import SeparableField, solve_separable_diffusion
from pymhm.separable_krylov import solve_separable_krylov
from pymhm.subspaces import restrict_response

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "build/results/periodic"
RECORD = ROOT / "examples/results/periodic.json"
EPSILON = np.pi / 150


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
    sources += sorted((ROOT / "src/pymhm").glob("*.py"))
    result = {}
    for path in sources:
        digest = fingerprint(path)
        result[str(path.relative_to(ROOT))] = digest
        if snapshot:
            destination = ARTIFACTS / "acquisition-sources" / f"{digest}.py"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    return result


def save(record: dict) -> None:
    """Write current numerical records atomically after each completed solve."""
    record["reference"].sort(key=lambda row: row["n"])
    path = RECORD.with_suffix(".json.part")
    path.write_text(json.dumps(record, indent=2) + "\n")
    path.replace(RECORD)


def main() -> None:
    """Acquire reference refinement separately from the MHM face study."""
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
    args = parser.parse_args()
    if args.native_threads < 1:
        parser.error("native-threads must be positive")
    if args.reference_assembly == "lor" and args.reference_solver != "pyamg":
        parser.error("the lor reference uses its declared PyAMG preconditioned CG solver")
    acquisition_sources = source_hashes(snapshot=True)
    reference_order = args.reference_quadrature or max(4, args.reference_degree + 1)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    global RECORD
    if args.reference_degree != 1:
        RECORD = RECORD.with_name(f"periodic-q{args.reference_degree}.json")
    suffix = f"-order{reference_order}" if args.reference_quadrature else ""
    if args.reference_assembly != "elementwise":
        suffix += f"-{args.reference_assembly}"
    if suffix:
        RECORD = RECORD.with_name(f"{RECORD.stem}{suffix}.json")
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
                    reference = ConformingQuadrilateralSolution(
                        CartesianMacroMesh(n),
                        args.reference_degree,
                        data["pressure"],
                        material,
                        float(data["residual"]),
                    )
            else:
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
                    refinement_precision="extended",
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
                    refinement_precision="extended",
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
                )
                record["reference"].append(row)
                save(record)
                print("reference", {k: v for k, v in row.items() if "hash" not in k}, flush=True)
            row = next(r for r in record["reference"] if r["n"] == n)
            row["archive"] = path.name
            row["archive_sha256"] = fingerprint(path)
            if previous is not None:
                row = next(r for r in record["reference"] if r["n"] == n)
                row["difference_to_previous"] = difference(reference, (previous,))
                row["comparison_source_hashes"] = acquisition_sources
                save(record)
                print("reference comparison", n, row["difference_to_previous"], flush=True)
            save(record)
            previous = reference
        if args.reference_only:
            return
        macro = CartesianMacroMesh(args.macro)
        max_segments = max(args.segments)
        if any(max_segments % s for s in args.segments):
            raise ValueError("each face partition must divide the largest prepared partition")
        prepared_space = SkeletonSpace(
            macro, tuple(FaceSpace.uniform(0, max_segments) for _ in macro.faces)
        )
        prepared = None
        for segments in args.segments:
            old = next(
                (
                    r
                    for r in record["mhm"]
                    if r["macro"] == args.macro
                    and r["refinement"] == args.local_refinement
                    and r["segments"] == segments
                    and r["reference_n"] == reference.mesh.nx
                    and r.get("reference_degree", 1) == reference.degree
                ),
                None,
            )
            if old is not None:
                continue
            skeleton = SkeletonSpace(
                macro, tuple(FaceSpace.uniform(0, segments) for _ in macro.faces)
            )
            field_path = ARTIFACTS / f"mhm-{args.macro}-r{args.local_refinement}-s{segments}.npz"
            acquired = not field_path.exists()
            if field_path.exists():
                with np.load(field_path) as data:
                    fields, residual = data["fields"], float(data["residual"])
            else:
                if prepared is None:
                    prepared = HybridSystem.from_local_factory(
                        _assemble_quad,
                        [
                            _QuadTask(
                                macro,
                                cell,
                                (args.local_refinement,) * 2,
                                prepared_space,
                                1,
                                material,
                                source,
                                4,
                            )
                            for cell in range(len(macro.cells))
                        ],
                        backend="process" if args.workers > 1 else "serial",
                        workers=args.workers,
                    )
                    print("Prepared maximum trace responses", flush=True)
                if segments == max_segments:
                    system = prepared
                else:
                    face_map = np.eye(segments)[
                        np.arange(max_segments) // (max_segments // segments)
                    ]
                    injection = np.kron(np.eye(4), face_map)
                    system = HybridSystem.from_responses(
                        [
                            restrict_response(response, injection, skeleton.cell_dofs(cell))
                            for cell, response in enumerate(prepared.responses)
                        ]
                    )
                result = system.solve()
                fields, residual = result.fields, result.residual
                np.savez_compressed(field_path, fields=np.array(fields), residual=residual)
                del result, system
            local = tuple(
                ConformingQuadrilateralSolution(
                    macro.submesh(cell, args.local_refinement), 1, p, material, residual
                )
                for cell, p in enumerate(fields)
            )
            errors = difference(reference, local)
            row = dict(
                macro=args.macro,
                refinement=args.local_refinement,
                segments=segments,
                reference_n=reference.mesh.nx,
                reference_degree=reference.degree,
                trace_dofs=skeleton.size,
                residual=residual,
                archive=field_path.name,
                archive_sha256=fingerprint(field_path),
                comparison_source_hashes=acquisition_sources,
                **errors,
            )
            if acquired:
                row["acquisition_source_hashes"] = acquisition_sources
                row["acquisition_source_changed"] = acquisition_sources != source_hashes()
            record["mhm"].append(row)
            save(record)
            print("MHM", {k: v for k, v in row.items() if "hash" not in k}, flush=True)


if __name__ == "__main__":
    main()
