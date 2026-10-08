"""Acquire vertically invariant classical 3D RT1 references for the oscillatory well."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import mapped_darcy as solve_darcy_mapped_rt
from examples.solve_mapped_oscillatory_well import OUTPUT, ROOT, OscillatoryWellData
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.hexahedron import HexMesh, cube_quadrature


def main() -> None:
    """Refine radial/azimuthal coordinates while retaining a valid full 3D mixed operator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-factor", type=int, default=16)
    parser.add_argument("--quadrature-xy", type=int, default=20)
    parser.add_argument("--quadrature-z", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--balance-order", type=int, default=3)
    parser.add_argument("--global-rtol", type=float, default=1e-10)
    parser.add_argument(
        "--global-refinement-precision", choices=["double", "extended"], default="double"
    )
    args = parser.parse_args()
    if args.balance_order < 2:
        parser.error("balance-order must be at least two for RT1/Q1 divergence moments")
    data = OscillatoryWellData()
    radii = np.geomspace(data.inner_radius, data.outer_radius, 5)
    mesh = HexMesh.annular_prism(radii, data.height, 8).refined(
        (args.fine_factor, args.fine_factor, 1)
    )
    zero = {
        int(f): 0.0 for f in mesh.boundary_faces if np.ptp(mesh.points[mesh.faces[f], 2]) < 1e-12
    }
    order = (args.quadrature_xy, args.quadrature_xy, args.quadrature_z)
    sources = [
        Path(__file__),
        ROOT / "examples/solve_mapped_oscillatory_well.py",
        ROOT / "examples/solve_mapped_well.py",
        ROOT / "src/pymhm/_legacy/models/darcy/mapped.py",
        ROOT / "src/pymhm/linalg/linear.py",
        ROOT / "src/pymhm/core/contracts.py",
    ]
    hashes = current_source_manifest(
        {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        }
    )
    snapshot = ROOT / "build/source-snapshots/mapped-well-oscillatory"
    snapshot.mkdir(parents=True, exist_ok=True)
    for p in sources:
        (snapshot / f"{hashes[p.relative_to(ROOT).as_posix()]}-{p.name}").write_bytes(
            p.read_bytes()
        )
    with threadpool_limits(1):
        solution = solve_darcy_mapped_rt(
            mesh,
            degree=1,
            local_refinement=1,
            permeability=data.tensor,
            dirichlet=data.pressure,
            neumann=zero,
            quadrature_order=order,
            backend="process",
            workers=args.workers,
            global_rtol=args.global_rtol,
            global_refinement_precision=args.global_refinement_precision,
        )
        # det(J) cancels from the Piola divergence moment. With f=0,
        # Q1 divergence times Q1 pressure is integrated exactly by order>=2.
        diagnostic = replace(solution, quadrature_order=args.balance_order)
        balance = max(float(abs(v).max()) for v in diagnostic.equilibrium_residuals())
        points = cube_quadrature(3)[0]
        vertical_flux = max(
            float(abs(solution.evaluate(c, points)[1][..., 2]).max())
            for c in range(len(mesh.cells))
        )
    rates = dict(inner=0.0, outer=0.0, top_bottom=0.0)
    for face in mesh.boundary_faces:
        label = (
            "top_bottom"
            if int(face) in zero
            else "inner"
            if np.max(np.linalg.norm(mesh.points[mesh.faces[face], :2], axis=1)) < 1
            else "outer"
        )
        rates[label] += float(solution.hybrid.trace[4 * face])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"classical-xy{args.fine_factor}-z1-q{order[0]}z{order[2]}"
    if args.global_rtol != 1e-10 or args.global_refinement_precision != "double":
        name += f"-rtol{args.global_rtol:g}-{args.global_refinement_precision}"
    archive = OUTPUT / (name + ".npz")
    np.savez_compressed(
        archive,
        vertices=mesh.points[mesh.cells],
        pressure=np.stack(solution.pressure)[:, 0],
        flux=np.stack(solution.flux),
        macro_points=mesh.points,
        macro_cells=mesh.cells,
    )
    report = dict(
        global_rtol=args.global_rtol,
        global_refinement_precision=args.global_refinement_precision,
        case="Classical mapped RT1/Q1 for the vertically invariant oscillatory well",
        archive_layout="canonical-reference-cell",
        fine_factor=args.fine_factor,
        vertical_factor=1,
        macro_cells=len(mesh.cells),
        fine_cells=len(mesh.cells),
        trace_dofs=solution.skeleton.size,
        pressure_coarse_dofs=len(mesh.cells),
        degree=1,
        quadrature=order,
        symmetry=(
            "Kxy and pressure boundary data are independent of z; Kz depends only on z, "
            "with impermeable horizontal caps. The unique solution has dz(p)=0 and qz=0."
        ),
        vertical_flux_sample_linf=vertical_flux,
        pressure_vertical_modal_linf=float(abs(np.stack(solution.pressure)[..., 1::2]).max()),
        physical_parameters=data.__dict__,
        radii=radii.tolist(),
        units="m, Pa, s; physical flux m/s",
        boundary_rates=rates,
        divergence_moment_linf=balance,
        divergence_moment_quadrature=args.balance_order,
        algebraic_residual=solution.hybrid.residual,
        physical_block_backward_residual_max=solution.physical_residuals.max(axis=0).tolist(),
        archive=archive.name,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_hashes=hashes,
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.relative_to(ROOT).as_posix()]
        for p in sources
    ):
        raise RuntimeError("Acquisition source changed; inspect snapshots before publication")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
