"""Acquire the recovered crisscross geometry with the article's oscillatory data.

The geometry follows Barros's 2022 dissertation, Figure 4. Its Cartesian
spacing and maximum simplex diameters are recorded separately. Figure-6
integer dimensions and its lower-row subdivision label are not conflated.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from examples.campaign_provenance import index_records, require_equal, verify_archive
from examples.formulations.application import darcy as solve_darcy
from examples.formulations.application import three_field_diffusion as solve_mh2m
from examples.mh2m_campaign import diagnostics, source
from examples.mh2m_campaign_contracts import verify_difference_result as verify_result
from examples.mh2m_crisscross_norms import CrossedP1, common_triangles
from examples.mh2m_heterogeneous import OscillatoryCoefficient, load_field, source_hashes
from examples.mh2m_heterogeneous_norms import difference
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.crisscross import crisscross_submesh
from pymhm.meshes.triangle import TriangleMesh

ROOT = Path(__file__).resolve().parents[1]


def configurations() -> list[tuple[str, str, int, int, int, int]]:
    """Declare figure identifiers, method, n, r and Gamma/Lambda subdivisions."""
    rows = [("figure-5", "MH2M", 31, 4, 2, 2)]
    rows += [(f"figure-7-n{n}", "MHM", n, 8, 1, 1) for n in (8, 16, 30, 50, 60)]
    rows += [(f"figure-8-n{n}", "MH2M", n, 8, 1, 8) for n in (16, 34, 65, 80, 100)]
    rows += [
        (f"figure-6-upper-{method}-n{n}", method, n, 4, 1, 1)
        for method, n in (("MHM", 8), ("MHM", 30), ("MH2M", 16), ("MH2M", 65))
    ]
    rows += [
        (f"figure-6-lower-dofs-{method}-n{n}-lambda{s}", method, n, 8, 4, s)
        for method, n, s in (
            ("MHM", 5, 4),
            ("MHM", 9, 4),
            ("MH2M", 5, 4),
            ("MH2M", 10, 4),
            ("MH2M", 5, 8),
            ("MH2M", 10, 8),
        )
    ]
    rows += [(f"fixed-Gamma-lambda{s}", "MH2M", 16, 8, 1, s) for s in (1, 2, 4)]
    return rows


def hashes() -> dict[str, str]:
    """Identify executed geometry, numerical maps and common norm owners."""
    result = source_hashes()
    for name in (
        "examples/mh2m_crisscross_campaign.py",
        "examples/mh2m_crisscross_norms.py",
        "src/pymhm/meshes/crisscross.py",
        "src/pymhm/meshes/refinement.py",
        "src/pymhm/meshes/roundoff.py",
    ):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return current_source_manifest(result)


def acquire(name: str, method: str, n: int, r: int, gamma: int, flux: int, output: Path) -> dict:
    """Solve one declared finite-dimensional method and persist all incident fields."""
    started = perf_counter()
    original = TriangleMesh.unit_square(n)
    mesh = TriangleMesh(
        np.column_stack((1 - original.points[:, 0], original.points[:, 1])), original.cells
    )
    fine = tuple(crisscross_submesh(mesh, cell, r) for cell in range(len(mesh.cells)))
    flux_space = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, flux) for _ in mesh.faces))
    options = dict(
        permeability=OscillatoryCoefficient(),
        source=source,
        local_meshes=fine,
        degree=1,
        quadrature_order=10,
    )
    if method == "MH2M":
        result = solve_mh2m(
            mesh,
            pressure_trace=PressureTraceSpace.uniform(mesh, 1, gamma),
            flux_space=flux_space,
            **options,
        )
        diagnostic = diagnostics(result)
        eigenvalues = [np.linalg.eigvalsh(local.neumann_energy) for local in result.local]
        diagnostic["minimum_neumann_energy_eigenvalue"] = min(float(v[0]) for v in eigenvalues)
        total, free = result.trace_space.size, len(result.free_dofs)
    else:
        result = solve_darcy(mesh, skeleton=flux_space, **options)
        diagnostic = {
            "algebraic_residual": result.hybrid.residual,
            "macro_balance_max": float(np.max(abs(result.conservation_residuals()))),
        }
        total = free = len(result.hybrid.trace) + len(result.hybrid.coarse)
    archive = output / f"{name}.npz"
    np.savez_compressed(
        archive,
        layout="crisscross",
        vertices=np.concatenate([m.points[m.cells] for m in fine]),
        pressure=np.concatenate([p[m.cells] for p, m in zip(result.pressure, fine, strict=True)]),
        macro_points=mesh.points,
        macro_cells=mesh.cells,
    )
    return {
        "name": name,
        "method": method,
        "macro_resolution": n,
        "macro_cells": len(mesh.cells),
        "macro_diagonal": "northwest-southeast",
        "macro_cartesian_spacing": 1 / n,
        "macro_max_diameter": np.sqrt(2) / n,
        "fine_max_diameter": 1 / (n * r),
        "local_refinement": r,
        "local_triangles_per_macro": 2 * r * r,
        "local_degree": 1,
        "Gamma_degree": 1 if method == "MH2M" else None,
        "Gamma_segments": gamma if method == "MH2M" else None,
        "Lambda_degree": 0,
        "Lambda_segments": flux,
        "global_dofs_total": total,
        "global_dofs_free": free,
        "assembly_order": 10,
        "diagnostics": diagnostic,
        "pressure_range": [
            min(float(p.min()) for p in result.pressure),
            max(float(p.max()) for p in result.pressure),
        ],
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "solve_seconds": perf_counter() - started,
    }


def validate_campaign_resume(
    record: dict[str, Any],
    output: Path,
    references: list[dict[str, Any]],
) -> None:
    """Validate references and all persisted cases before acquisition or norm skips."""
    require_equal(record["references"], references, label="reference acquisitions")
    if not references:
        raise ValueError("the crisscross campaign requires an archived reference")
    for reference in references:
        verify_archive(output / reference["archive"], reference["archive_sha256"])
    finest = references[-1]
    configured = {row[0]: row for row in configurations()}
    for name, row in index_records(record["cases"], key="name").items():
        if name not in configured:
            raise ValueError("completed case is absent from the declared configurations")
        _, method, n, refinement, gamma, flux = configured[name]
        identity = dict(
            method=method,
            macro_resolution=n,
            local_refinement=refinement,
            local_degree=1,
            Gamma_degree=1 if method == "MH2M" else None,
            Gamma_segments=gamma if method == "MH2M" else None,
            Lambda_degree=0,
            Lambda_segments=flux,
            assembly_order=10,
        )
        require_equal({key: row[key] for key in identity}, identity, label="case discretization")
        verify_archive(output / row["archive"], row["archive_sha256"])
        if "norms" in row or "norms_quadrature_check" in row:
            view = {
                **row,
                "norms": {
                    f"quadrature_{q}": row[key]
                    for q, key in ((6, "norms"), (8, "norms_quadrature_check"))
                    if key in row
                },
            }
            verify_result(
                view,
                {**identity, "reference_resolution": finest["resolution"]},
                archives={
                    output / row["archive"]: row["archive_sha256"],
                    output / finest["archive"]: finest["archive_sha256"],
                },
                norm_orders=[6, 8],
            )


def main() -> None:
    """Checkpoint each case before integrating against the unchanged classical reference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "examples/results/mh2m-heterogeneous/crisscross"
    )
    parser.add_argument(
        "--reference-data", type=Path, default=ROOT / "examples/results/mh2m-heterogeneous"
    )
    parser.add_argument("--stage", choices=("acquire", "norms", "all"), default="all")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--norm-workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path, before = args.output / "comparison.json", hashes()
    reference_path = args.reference_data / "comparison.json"
    prior = json.loads(reference_path.read_text())
    references = [
        {
            **row,
            "archive": Path(
                os.path.relpath(reference_path.parent / row["archive"], args.output)
            ).as_posix(),
        }
        for row in prior["references"]
    ]
    record: dict[str, Any] = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "paper": "de Barros, Madureira, Valentin, arXiv:2404.16978v3, Section 8.2",
            "geometry_source": "Barros (2022), Figure 4, printed p54 / PDF p56",
            "primary_url": "https://www.lncc.br/~alm/students/frankdissert.pdf",
            "conventions": {
                "gamma": 1.8,
                "epsilon": "1/14",
                "source": "-2*x*(x-1)-2*y*(y-1)",
                "data_source": "Thesis Equations4.1/4.4; articlev3 epsilon",
                "boundary": "homogeneous Dirichlet",
                "fine_geometry": "Cartesian crisscross",
                "diameters": "Hmax=sqrt(2)/n; hmax=1/(n*r); Cartesian macro spacing=1/n",
                "figure6_lower": (
                    "Four segments reproduce the printed integer dimensions; "
                    "its i=4 caption instead gives eight."
                ),
                "figure6_upper_star": (
                    "Not acquired: r4 has 16 boundary P1 nodes, fewer than 24 Lambda "
                    "coefficients; Assumption A fails."
                ),
                "figure7_8_local_refinement": 8,
            },
            "references": references,
            "reference_series_record_sha256": hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest(),
            "cases": [],
            "source_sha256": before,
            "source_changed_during_run": False,
        }
    )
    validate_campaign_resume(record, args.output, references)
    if record["source_sha256"] != before:
        raise ValueError(
            "campaign sources changed; use --output NEW_DIRECTORY for acquisition or "
            "python -m examples.validate_mh2m_campaign to validate the archived records"
        )

    def checkpoint() -> None:
        """Write complete records atomically and require unchanged executed sources."""
        if before != hashes():
            raise RuntimeError("crisscross campaign sources changed")
        validate_campaign_resume(record, args.output, references)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)

    with threadpool_limits(1):
        if args.stage in ("acquire", "all"):
            for configuration in configurations():
                if args.names and configuration[0] not in args.names:
                    continue
                if any(row["name"] == configuration[0] for row in record["cases"]):
                    continue
                row = acquire(*configuration, args.output)
                record["cases"].append(row)
                checkpoint()
                print(json.dumps(row), flush=True)
                gc.collect()
        if args.stage in ("norms", "all"):
            reference_row = references[-1]
            archive = args.output / reference_row["archive"]
            if hashlib.sha256(archive.read_bytes()).hexdigest() != reference_row["archive_sha256"]:
                raise ValueError("classical reference archive digest changed")
            reference = load_field(archive)
            for row in record["cases"]:
                if (
                    args.names and row["name"] not in args.names
                ) or "norms_quadrature_check" in row:
                    continue
                path_field = args.output / row["archive"]
                if hashlib.sha256(path_field.read_bytes()).hexdigest() != row["archive_sha256"]:
                    raise ValueError("crisscross archive digest changed")
                with np.load(path_field) as arrays:
                    other = CrossedP1.from_arrays(arrays["vertices"], arrays["pressure"])
                for order, key in ((6, "norms"), (8, "norms_quadrature_check")):
                    row[key] = difference(
                        reference,
                        other,
                        OscillatoryCoefficient(),
                        order,
                        workers=args.norm_workers,
                        geometry=common_triangles(reference.resolution, other.resolution),
                    )
                row["reference_resolution"] = reference.resolution
                checkpoint()
                print(
                    json.dumps(
                        {"completed_norms": row["name"], "norms": row["norms_quadrature_check"]}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
