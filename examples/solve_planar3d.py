"""Verify fitted tetrahedral local meshes and nonuniform skeletal face partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import tetrahedral_darcy as solve_darcy_3d
from examples.planar3d_data import Planar3DData
from pymhm.fem.scalar.tetrahedron import tetrahedron_quadrature
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.meshes.fitting import fit_planar_material, planar_face_partitions
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.recovery.moments_3d import reconstruct_darcy_moments_3d

ROOT = case_workspace()


def snapshot() -> dict[str, str]:
    """Capture executed numerical owners and analytical data without local machine paths."""
    paths = [Path(__file__), source_file("examples/planar3d_data.py", root=ROOT)]
    paths.extend(
        source_file(f"src/pymhm/{name}", root=ROOT)
        for name in (
            "_legacy/models/darcy/primal_3d.py",
            "meshes/validation.py",
            "fem/scalar/tetrahedron.py",
            "fem/scalar/tetrahedron_topology.py",
            "materials/planar.py",
            "meshes/fitting.py",
            "core/contracts.py",
            "linalg/linear.py",
            "recovery/moments_3d.py",
            "fem/hdiv/rt_3d.py",
            "fem/hdiv/family_3d.py",
            "meshes/mixed.py",
        )
    )
    return current_source_manifest(source_identity(ROOT, paths), packages=("pymhm", "examples"))


def main() -> None:
    """Run nine transmission patches and archive owned local coefficients and physical norms."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/planar3d")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = snapshot()
    data = Planar3DData()
    report = dict(
        timestamp_utc=datetime.now(UTC).isoformat(),
        source_sha256=hashes,
        domain="unit cube; Freudenthal macro tetrahedra",
        interface="x+0.4*y+0.2*z=0.63",
        contrast=data.contrast,
        tensor=data.tensor.tolist(),
        source=data.source,
        local_degree=4,
        trace_degree=1,
        assembly_quadrature_order=6,
        error_quadrature_orders=[6, 7],
        evidence=(
            "original exact transmission patches; "
            "not a convergence-rate or historical reproduction claim"
        ),
        rows=[],
    )
    with threadpool_limits(1):
        for n in args.levels:
            mesh = TetraMesh.unit_cube(n)
            parts = planar_face_partitions(mesh, data.material)
            skeleton = TriangularSkeleton(mesh, degree=1, face_partitions=parts)
            locals_ = tuple(
                fit_planar_material(mesh.submesh(cell, 1), data.material).mesh
                for cell in range(len(mesh.cells))
            )
            bary, weights = tetrahedron_quadrature(6)
            mean = (
                sum(
                    float(
                        fine.volumes
                        @ (
                            data.pressure(
                                np.einsum("qi,tia->tqa", bary, fine.points[fine.cells]).reshape(
                                    -1, 3
                                )
                            ).reshape(len(fine.cells), -1)
                            @ weights
                        )
                    )
                    for fine in locals_
                )
                / mesh.volumes.sum()
            )
            for boundary in ("dirichlet", "mixed", "neumann"):
                faces = mesh.boundary_faces
                if boundary == "dirichlet":
                    faces = faces[:0]
                elif boundary == "mixed":
                    faces = np.array(
                        [f for f in faces if np.all(mesh.points[mesh.faces[f], 0] == 1)]
                    )
                neumann = {int(f): partial(data.normal_flux, normal=mesh.normals[f]) for f in faces}
                solution = solve_darcy_3d(
                    mesh,
                    skeleton=skeleton,
                    local_meshes=locals_,
                    degree=4,
                    permeability=data.material,
                    source=data.source,
                    dirichlet=data.pressure,
                    neumann=neumann,
                    mean_pressure=mean,
                    quadrature_order=6,
                )
                norms = [
                    [
                        solution.l2_error(data.pressure, order),
                        solution.flux_l2_error(data.flux, order),
                    ]
                    for order in (6, 7)
                ]
                reconstruction = reconstruct_darcy_moments_3d(solution, degree=1)
                reconstructed = reconstruction.flux_l2_error(data.flux, 7)
                balances = max(abs(solution.conservation_residuals()))
                if max(np.ravel(norms)) > 2e-8 or reconstructed > 2e-8 or balances > 2e-10:
                    raise ValueError(
                        f"exact transmission patch exceeds physical gates: "
                        f"{norms=}, {reconstructed=}, {balances=}"
                    )
                contents = dict(
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                    degree=np.array(4),
                    hybrid_trace=solution.hybrid.trace,
                    face_partition_offsets=np.r_[0, np.cumsum([len(p) for p in parts])],
                    face_partition_barycentric=np.concatenate(parts),
                )
                for cell, fine in enumerate(locals_):
                    contents[f"local_points_{cell}"] = fine.points
                    contents[f"local_cells_{cell}"] = fine.cells
                    contents[f"pressure_{cell}"] = solution.pressure[cell]
                archive = args.output / f"fields-n{n}-{boundary}.npz"
                np.savez_compressed(archive, **contents)
                row = dict(
                    macro_subdivisions=n,
                    macro_cells=len(mesh.cells),
                    boundary=boundary,
                    fine_cells=sum(len(f.cells) for f in locals_),
                    skeleton_coefficients=skeleton.size,
                    subtriangle_area_fraction_min=min(
                        skeleton.face_weights(f).min() for f in range(len(mesh.faces))
                    ),
                    subtriangle_area_fraction_max=max(
                        skeleton.face_weights(f).max() for f in range(len(mesh.faces))
                    ),
                    pressure_l2=norms[-1][0],
                    raw_flux_l2=norms[-1][1],
                    reconstructed_flux_l2=reconstructed,
                    absolute_error_quadrature_difference=float(np.max(abs(np.diff(norms, axis=0)))),
                    macro_balance=float(balances),
                    backward_residual=solution.hybrid.residual,
                    fields=archive.name,
                    fields_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                )
                report["rows"].append(row)
                if snapshot() != hashes:
                    raise ValueError("acquisition source changed during execution")
                report["source_changed_during_run"] = False
                (args.output / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_planar3d").main()
