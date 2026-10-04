"""Acquire MHM and PGMHM on the same resolved square-annulus material and local spaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.archive_precision import precision_fields
from examples.pgmhm_inclusion_data import (
    CONTRAST,
    COUNT,
    INNER_RADIUS,
    OUTER_RADIUS,
    coefficient,
    local_meshes,
)
from pymhm._legacy.models.geometry import solve_darcy_polygons
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.methods.petrov_galerkin import solve_pgmhm

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/pgmhm-inclusions"


def acquire(factor: int, *, segments: int = 2, workers: int = 8) -> dict:
    """Compare both forms without changing the material, local mesh or skeletal moments."""
    names = [
        "examples/solve_pgmhm_inclusions.py",
        "examples/pgmhm_inclusion_data.py",
        "examples/mh_campaign.py",
        "examples/archive_precision.py",
    ] + [
        f"src/pymhm/{name}.py"
        for name in (
            "methods/petrov_galerkin",
            "_legacy/models/darcy/primal",
            "meshes/polygonal",
            "meshes/refinement",
            "fem/scalar/triangle",
            "core/contracts",
            "linalg/linear",
            "execution/cpu",
            "fem/traces/scalar",
        )
    ]
    hashes = current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )
    mesh, local = local_meshes(factor)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0, segments) for _ in mesh.faces))
    options = dict(
        permeability=coefficient,
        source=1.0,
        dirichlet=0.0,
        skeleton=skeleton,
        degree=2,
        local_meshes=local,
        quadrature_order=5,
        local_refinement_precision="extended",
        backend="process" if workers > 1 else "serial",
        workers=workers,
    )
    started = perf_counter()
    mhm = solve_darcy_polygons(mesh, **options, parallel_assembly=True)
    mhm_seconds = perf_counter() - started
    started = perf_counter()
    pg = solve_pgmhm(mesh, **options, stabilization_parameter=0.1, ellipticity_lower_bound=1.0)
    pg_seconds = perf_counter() - started
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"mhm-pgmhm-factor{factor}-s{segments}.npz"
    np.savez_compressed(
        path,
        macro_points=mesh.points,
        macro_faces=mesh.faces,
        macro_offsets=np.r_[0, np.cumsum([len(cell) for cell in mesh.cells])],
        macro_cells=np.concatenate(mesh.cells),
        point_offsets=np.r_[0, np.cumsum([len(fine.points) for fine in local])],
        cell_offsets=np.r_[0, np.cumsum([len(fine.cells) for fine in local])],
        coefficient_offsets=np.r_[0, np.cumsum([len(value) for value in pg.pressure])],
        local_points=np.concatenate([fine.points for fine in local]),
        local_cells=np.concatenate([fine.cells for fine in local]),
        **precision_fields("mhm", np.concatenate(mhm.pressure)),
        **precision_fields("pgmhm", np.concatenate(pg.pressure)),
        **precision_fields("enriched", np.concatenate(pg.enriched_pressure)),
        mhm_trace=mhm.hybrid.trace,
        pgmhm_trace=pg.hybrid.trace,
    )
    row = dict(
        method="MHM and PGMHM P0/P2 on identical fitted local meshes",
        article="10.1007/s40314-023-02304-y, Section 6.2",
        geometry_source=(
            "square-annulus geometry scaled from DOI10.1007/s00211-020-01103-5 Figure7; "
            "ratios checked against Figure9 raster"
        ),
        inclusion_count_per_axis=COUNT,
        inner_radius=INNER_RADIUS,
        outer_radius=OUTER_RADIUS,
        contrast=CONTRAST,
        high_permeability_area=4 * COUNT**2 * (OUTER_RADIUS**2 - INNER_RADIUS**2),
        macro_cells=len(mesh.cells),
        macro_faces=len(mesh.faces),
        global_dofs=skeleton.size + len(mesh.cells),
        factor=factor,
        segments=segments,
        degree=2,
        alpha=0.1,
        quadrature_order=5,
        fine_cells=sum(len(fine.cells) for fine in local),
        residual_mhm=float(mhm.hybrid.residual),
        residual_pgmhm=float(pg.hybrid.residual),
        macro_balance_mhm=float(np.max(abs(mhm.conservation_residuals()))),
        macro_balance_pgmhm=float(np.max(abs(pg.conservation_residuals()))),
        mhm_seconds=mhm_seconds,
        pgmhm_seconds=pg_seconds,
        archive=path.name,
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_hashes=hashes,
    )
    row["source_changed_during_run"] = hashes != current_source_manifest(
        {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    )
    if row["source_changed_during_run"]:
        raise RuntimeError("numerical source changed during acquisition")
    path.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    """Run material-fitted local-resolution controls with a declared unchanged trace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(1):
        for factor in args.factors:
            acquire(factor, segments=args.segments, workers=args.workers)


if __name__ == "__main__":
    main()
