"""Resolve the independent pressure-trace error while holding local MH2M spaces fixed."""

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
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import three_field_diffusion as solve_mh2m
from examples.mh2m_campaign import (
    OUTPUT,
    archive_p1,
    diagnostics,
    difference,
    oscillatory,
    source,
    source_hashes,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.meshes.triangle import TriangleMesh


def main() -> None:
    """Acquire five Gamma refinements against the existing finest classical reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    record = json.loads((args.output / "comparison.json").read_text())
    n = record["references"][-1]["resolution"]
    with np.load(args.output / f"reference-n{n}.npz") as archive:
        reference = {key: archive[key] for key in archive.files}
    hashes = source_hashes()
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    mesh = TriangleMesh.unit_square(4)
    rows = []
    with threadpool_limits(limits=1):
        for segments in (1, 2, 4, 8, 16):
            start = perf_counter()
            result = solve_mh2m(
                mesh,
                permeability=oscillatory,
                source=source,
                pressure_trace=PressureTraceSpace.uniform(mesh, 1, segments),
                flux_space=SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, 16) for _ in mesh.faces)),
                degree=1,
                local_refinement=32,
                quadrature_order=8,
            )
            data = archive_p1(result, args.output / f"enriched-gamma{segments}.npz")
            row = {
                "pressure_segments": segments,
                "flux_segments": 16,
                "local_refinement": 32,
                **diagnostics(result),
                **difference(reference, data),
                "elapsed_seconds": perf_counter() - start,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if (
        source_hashes() != hashes
        or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != driver_hash
    ):
        raise RuntimeError("acquisition sources changed during the numerical campaign")
    output = {
        "pressure_enrichment": rows,
        "reference_resolution": n,
        "reference_last_refinement": record["references"][-1],
        "oscillatory_convention": record["oscillatory_convention"],
        "source_hashes": {**hashes, "examples/mh2m_trace_campaign.py": driver_hash},
        "source_changed_during_run": False,
    }
    (args.output / "pressure-enrichment.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
