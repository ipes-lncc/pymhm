"""Five-level native polyhedral study of the smooth three-dimensional L12 RAD problem."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from polygon_meshes import polygon_partition
from solve_rad3d import exact, gradient, physical_flux, source
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.transport.polyhedral import PolygonalSkeleton3D, solve_polyhedral_rad
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.polyhedral import PolyhedralMesh

ROOT = Path(__file__).resolve().parents[1]


def partition(n: int, family: str) -> tuple:
    """Extrude explicit convex planar partitions; macroface polygons remain intact."""
    names = {"cube": "square", "triangular-prism": "triangle", "hexagonal-prism": "hexagon"}
    base = polygon_partition(n, names[family])
    return PolyhedralMesh.extrude(base, n), base


def fingerprint(path: Path) -> str:
    """Hash exact acquisition inputs without exposing private execution locations."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Measure original P4/P1 polyhedral discretizations against the published exact PDE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["cube", "triangular-prism", "hexagonal-prism"],
        default=["cube", "triangular-prism", "hexagonal-prism"],
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    destination = ROOT / "examples/results/polyhedral-rad.json"
    folder = ROOT / "build/results/polyhedral-rad"
    folder.mkdir(exist_ok=True, parents=True)
    paths = [Path(__file__), ROOT / "examples/polygon_meshes.py", ROOT / "examples/solve_rad3d.py"]
    paths += [
        ROOT / f"src/pymhm/{name}"
        for name in (
            "meshes/polyhedral.py",
            "_legacy/models/transport/polyhedral.py",
            "fem/scalar/tetrahedron.py",
            "fem/scalar/tetrahedron_topology.py",
            "_legacy/models/transport/rad_3d.py",
            "core/contracts.py",
            "linalg/linear.py",
            "execution/cpu.py",
        )
    ]
    hashes = current_source_manifest(
        {str(path.relative_to(ROOT)): fingerprint(path) for path in paths}
    )
    snapshots = folder / "acquisition-sources"
    snapshots.mkdir(exist_ok=True)
    for path in paths:
        (snapshots / f"{fingerprint(path)}.py").write_bytes(path.read_bytes())
    record = (
        json.loads(destination.read_text())
        if destination.exists()
        else dict(
            reference="10.1016/j.cma.2024.117089, section 5.2.1, three-dimensional smooth problem",
            scope=(
                "Published PDE and P4 local / P1 face degrees; "
                "explicitly constructed convex polyhedral meshes."
            ),
            coefficient=dict(diffusion=0.1, velocity=[1.0, 0.0, 0.0], reaction=0.0),
            exact="sin(6*pi*x)*sin(4*pi*y)*sin(2*pi*z)",
            source="5.6*pi**2*u+partial_x(u)",
            local_degree=4,
            trace_degree=1,
            local_refinement=1,
            trace=(
                "One affine polynomial per original polygonal face; "
                "three DOFs independent of triangulation."
            ),
            assembly_quadrature=8,
            error_quadrature=10,
            convergence=[],
        )
    )
    with threadpool_limits(1):
        for family in args.families:
            for n in args.levels:
                if any(row["family"] == family and row["n"] == n for row in record["convergence"]):
                    continue
                start = perf_counter()
                mesh, base = partition(n, family)
                solution = solve_polyhedral_rad(
                    mesh,
                    degree=4,
                    skeleton=PolygonalSkeleton3D(mesh, 1),
                    diffusion=0.1,
                    velocity=(1.0, 0.0, 0.0),
                    source=source,
                    quadrature_order=8,
                    backend="process" if args.workers > 1 else "serial",
                    workers=args.workers,
                )
                l2 = solution.l2_error(exact, order=10)
                h1 = solution.h1_seminorm_error(gradient, order=10)
                flux = solution.flux_l2_error(physical_flux, order=10)
                row = dict(
                    family=family,
                    n=n,
                    macros=len(mesh.cells),
                    original_faces=len(mesh.faces),
                    trace_dofs=solution.skeleton.size,
                    global_dofs=solution.skeleton.size + len(mesh.cells),
                    local_dofs=sum(len(v) for v in solution.values),
                    fine_tetrahedra=sum(len(f.cells) for f in solution.local_meshes),
                    l2_error=l2,
                    h1_seminorm_error=h1,
                    V_error=float(np.sqrt(h1 * h1 + l2 * l2 / 3)),
                    physical_flux_l2_error=flux,
                    backward_residual=solution.hybrid.residual,
                    wall_seconds=perf_counter() - start,
                    source_hashes=hashes,
                    timestamp_utc=datetime.now(UTC).isoformat(),
                )
                if n == max(args.levels):
                    higher = solution.l2_error(exact, order=12)
                    row["error_quadrature_l2_relative_change"] = abs(higher - l2) / higher
                    arrays = {
                        "macro_points": mesh.points,
                        "base_points": base.points,
                        "base_faces": base.faces,
                    }
                    for cell, fine in enumerate(solution.local_meshes):
                        arrays[f"points_{cell}"] = fine.points
                        arrays[f"cells_{cell}"] = fine.cells
                        arrays[f"coefficients_{cell}"] = solution.values[cell]
                    archive = folder / f"{family}-n{n}.npz"
                    np.savez_compressed(archive, **arrays)
                    row["field_archive"] = archive.name
                    row["field_sha256"] = fingerprint(archive)
                row["source_changed"] = any(
                    fingerprint(path) != hashes[str(path.relative_to(ROOT))] for path in paths
                )
                record["convergence"].append(row)
                temporary = destination.with_suffix(".json.part")
                temporary.write_text(json.dumps(record, indent=2) + "\n")
                temporary.replace(destination)
                print(
                    {key: value for key, value in row.items() if key != "source_hashes"}, flush=True
                )


if __name__ == "__main__":
    main()
