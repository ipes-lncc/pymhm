"""Integrate RT1 well differences for quadrature, macro and fine refinement."""

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

from threadpoolctl import threadpool_limits

from examples.mapped_well_comparison import differences
from examples.mapped_well_fields import MappedWellField
from pymhm.execution.cpu import map_local
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "examples/results/mapped-well-oscillatory"


def comparisons(
    reference: str, candidates: list[str], kind: str, *, workers: int = 1
) -> list[dict]:
    """Integrate archive pairs jointly while preserving every physical quadrature point."""
    fine, ref_report = MappedWellField.load(DIRECTORY / (reference + ".json"))
    loaded = [MappedWellField.load(DIRECTORY / (name + ".json")) for name in candidates]
    with threadpool_limits(1):
        measured = {
            str(order): differences(
                fine,
                [field for field, _ in loaded],
                (order, order, 3),
                pressure_offset=25e6,
                workers=workers,
            )
            for order in (6, 8)
        }
    return [
        dict(
            kind=kind,
            reference=reference,
            candidate=candidate,
            reference_sha256=ref_report["sha256"],
            candidate_sha256=loaded[index][1]["sha256"],
            norms={order: values[index] for order, values in measured.items()},
            denominator="physical norm of the explicitly named reference field",
            geometry="exact nested tensor hierarchy; integration respects every fine hexahedron",
        )
        for index, candidate in enumerate(candidates)
    ]


def comparison(reference: str, candidate: str, kind: str) -> dict:
    """Integrate one digested archive pair with the shared physical norm implementation."""
    return comparisons(reference, [candidate], kind)[0]


def _comparison_job(job: tuple[str, str, str]) -> dict:
    """Evaluate one independent physical norm pair with bounded BLAS threading."""
    with threadpool_limits(1):
        return comparison(*job)


def main() -> None:
    """Produce separate, auditable physical norm tables from the published native field archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    source_paths = [
        Path(__file__),
        ROOT / "examples/mapped_well_fields.py",
        ROOT / "examples/mapped_well_comparison.py",
        ROOT / "src/pymhm/_legacy/models/darcy/mapped.py",
    ]
    source_hashes = current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        }
    )
    classical = {}
    for path in DIRECTORY.glob("classical-*.json"):
        record = json.loads(path.read_text())
        level = record["fine_factor"]
        priority = (record["quadrature"][0], "invariant" in path.stem)
        if level >= 8 and (level not in classical or priority > classical[level][0]):
            classical[level] = (priority, path.stem)
    levels = [classical[key][1] for key in sorted(classical)]
    if len(levels) < 3:
        raise ValueError("at least three completed fine references are required")
    ref = levels[-1]
    fine_eight = classical[8][1]
    rows = comparisons(
        ref,
        [f"fine8-macro{m}-s1-q40z10" for m in (1, 2, 4)],
        "MHM versus refined reference",
        workers=args.workers,
    )
    rows += comparisons(
        fine_eight,
        [f"fine8-macro{m}-s1-q40z10" for m in (1, 2, 4)],
        "macro trace restriction",
        workers=args.workers,
    )
    jobs = [
        (ref, fine_eight, "classical fine-resolution gap"),
        ("classical-xy8-z1-q40z10", "fine8-macro2-s4-q40z10", "vertical invariance"),
        ("fine8-macro1-s1-q28z10", "fine8-macro1-s1-q20z10", "material quadrature"),
        ("fine8-macro1-s1-q40z10", "fine8-macro1-s1-q28z10", "material quadrature"),
    ]
    local_control = "fine16-macro4-s1-q40z10"
    local_reference = "classical-invariant-xy16-z1-q40"
    if (DIRECTORY / f"{local_control}.json").exists():
        if not (DIRECTORY / f"{local_reference}.json").exists():
            raise ValueError("the F16 local control requires its matching quadrature-40 reference")
        jobs += [
            (ref, local_control, "local-control refined reference"),
            (local_reference, local_control, "local-control trace restriction"),
            (local_control, "fine8-macro4-s1-q40z10", "MHM local refinement"),
            (local_control, "fine16-macro4-s1-q28z10", "material quadrature"),
            (ref, local_reference, "classical fine-resolution gap"),
            (
                local_reference,
                "classical-xy16-z1-q20z10",
                "classical material quadrature",
            ),
        ]
    if (DIRECTORY / "classical-xy32-z1-q20z10.json").exists():
        jobs.append(
            (
                "classical-xy32-z1-q20z10",
                "classical-xy32-z1-q10z10",
                "classical material quadrature",
            )
        )
    if (DIRECTORY / "classical-invariant-xy64-z1-q14.json").exists():
        jobs.append(
            (
                "classical-invariant-xy64-z1-q14",
                "classical-invariant-xy64-z1-q10",
                "classical material quadrature",
            )
        )
    for factor, order in ((2, 20), (8, 40), (32, 20)):
        invariant = f"classical-invariant-xy{factor}-z1-q{order}"
        full = f"classical-xy{factor}-z1-q{order}z10"
        if (DIRECTORY / (invariant + ".json")).exists():
            jobs.append((invariant, full, "exact vertical subspace"))
    jobs += [
        (fine, coarse, "classical spatial refinement")
        for coarse, fine in zip(levels[:-1], levels[1:], strict=True)
    ]
    rows += map_local(
        _comparison_job,
        jobs,
        backend="serial" if args.workers == 1 else "process",
        workers=args.workers,
    )
    report = dict(
        method="Native Piola RT1/Q1 pressure and vector-flux physical L2 differences",
        pressure_offset=25e6,
        reference=ref,
        units="m, Pa, s",
        rows=rows,
        analysis_source_sha256=source_hashes[Path(__file__).relative_to(ROOT).as_posix()],
        field_reader_sha256=source_hashes["examples/mapped_well_fields.py"],
        norm_workers=args.workers,
        source_hashes=source_hashes,
    )
    if source_hashes != current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        }
    ):
        raise RuntimeError("Physical-norm sources changed during acquisition")
    (DIRECTORY / "comparisons.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
