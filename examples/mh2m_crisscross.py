"""Check the explicit fine geometry in Barros's MH2M dissertation, Figure 4."""

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
from examples.mh2m_campaign import diagnostics, source, source_hashes
from examples.mh2m_heterogeneous import OscillatoryCoefficient
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.crisscross import crisscross_submesh
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]


def hashes() -> dict[str, str]:
    """Record geometry, data and executed formulation without private paths."""
    result = source_hashes()
    for name in (
        "examples/mh2m_crisscross.py",
        "examples/mh2m_heterogeneous.py",
        "src/pymhm/meshes/crisscross.py",
        "src/pymhm/meshes/refinement.py",
        "src/pymhm/meshes/roundoff.py",
    ):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return current_source_manifest(result)


def main() -> None:
    """Acquire the independent-Gamma case with Lambda resolution equal to the grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=8)
    parser.add_argument("--gamma-segments", type=int, default=1)
    parser.add_argument("--epsilon-denominator", type=int, default=14)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/mh2m-heterogeneous")
    args = parser.parse_args()
    if args.epsilon_denominator <= 0:
        raise ValueError("epsilon denominator must be positive")
    before, start = hashes(), perf_counter()
    original = TriangleMesh.unit_square(args.resolution)
    mesh = TriangleMesh(
        np.column_stack((1 - original.points[:, 0], original.points[:, 1])), original.cells
    )
    local = tuple(
        crisscross_submesh(mesh, cell, args.refinement) for cell in range(len(mesh.cells))
    )
    with threadpool_limits(1):
        result = solve_mh2m(
            mesh,
            permeability=OscillatoryCoefficient(epsilon=1 / args.epsilon_denominator),
            source=source,
            pressure_trace=PressureTraceSpace.uniform(mesh, 1, args.gamma_segments),
            flux_space=SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(0, args.refinement) for _ in mesh.faces)
            ),
            local_meshes=local,
            quadrature_order=10,
        )
        spectrum = [np.linalg.eigvalsh(data.neumann_energy) for data in result.local]
        report = {
            "geometry_source": "Barros (2022), Figure 4, printed page 54 / PDF page 56",
            "primary_url": "https://www.lncc.br/~alm/students/frankdissert.pdf",
            "coefficient_source": "Article v3 Equation 61; thesis Equations 4.1 and 4.4",
            "gamma": 1.8,
            "epsilon": 1 / args.epsilon_denominator,
            "source": "-2*(x*(x-1)+y*(y-1))",
            "macro_diagonal": "northwest-southeast",
            "macro_resolution": args.resolution,
            "macro_count": len(mesh.cells),
            "macro_max_diameter": np.sqrt(2) / args.resolution,
            "macro_cartesian_spacing": 1 / args.resolution,
            "local_refinement": args.refinement,
            "local_triangles_per_macro": 2 * args.refinement**2,
            "fine_max_diameter": 1 / (args.resolution * args.refinement),
            "boundary_edge_counts_per_macro": [
                args.refinement,
                args.refinement,
                2 * args.refinement,
            ],
            "pressure_degree": 1,
            "Gamma_degree": 1,
            "Gamma_segments": args.gamma_segments,
            "Lambda_degree": 0,
            "Lambda_segments": args.refinement,
            "minimum_neumann_energy_eigenvalue": min(float(x[0]) for x in spectrum),
            "minimum_neumann_energy_eigenvalue_ratio": min(float(x[0] / x[-1]) for x in spectrum),
            "pressure_range": [
                min(float(p.min()) for p in result.pressure),
                max(float(p.max()) for p in result.pressure),
            ],
            "pressure_l2": result.l2_error(0, order=10),
            "raw_flux_l2": result.flux_l2_error((0, 0), order=10),
            "assembly_order": 10,
            "diagnostics": diagnostics(result),
            "elapsed_seconds": perf_counter() - start,
            "source_sha256": before,
            "source_changed_during_run": before != hashes(),
            "scope": (
                "Explicit thesis crisscross geometry and declared epsilon; admissibility "
                "control, not a quantitative reproduction of the article's figures."
            ),
        }
    if report["source_changed_during_run"]:
        raise RuntimeError("MH2M crisscross acquisition sources changed")
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"crisscross-n{args.resolution}-r{args.refinement}-g{args.gamma_segments}"
    archive = args.output / f"{name}.npz"
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        vertices=np.concatenate([fine.points[fine.cells] for fine in local]),
        pressure=np.concatenate(
            [p[fine.cells] for p, fine in zip(result.pressure, local, strict=True)]
        ),
        gamma_trace=result.trace,
        conormal=np.array(result.conormal),
    )
    report["archive"], report["archive_sha256"] = (
        archive.name,
        hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    (args.output / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
