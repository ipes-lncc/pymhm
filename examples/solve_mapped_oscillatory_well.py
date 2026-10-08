"""Acquire the L05 oscillatory well operator with explicit physical units and quadrature."""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.formulations.application import mapped_darcy as solve_darcy_mapped_rt
from examples.solve_mapped_well import WellData
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.hexahedron import HexMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/mapped-well-oscillatory"


@dataclass(frozen=True)
class OscillatoryWellData(WellData):
    """Physical producer data with the tensor scale implied by the published color bars.

    The printed outer factor is 10^-3, whereas Figure 16 implies 10^-11 m².
    This declared scale is independent of the positive rate convention used
    to prescribe the inner/outer pressure. No historical-input identity is assumed.
    """

    tensor_scale: float = 1e-11

    def tensor(self, points: np.ndarray) -> np.ndarray:
        """Return the physical mobility K/eta in Cartesian coordinates."""
        x, y, z = points.T
        delta = np.pi / 25
        common = 2 + 1.8 * np.sin(delta * x * y)
        diagonal = np.column_stack(
            (
                0.1 * common / (2 + 1.8 * np.sin(delta * y)),
                0.001 * common / (2 + 1.8 * np.sin(delta * x)),
                0.001 * (2 + 1.8 * np.sin(delta * z * z)) / (2 + 1.8 * np.sin(delta * z)),
            )
        )
        return diagonal[:, :, None] * np.eye(3) * (self.tensor_scale / self.viscosity)


def main() -> None:
    """Solve an MHM or complete-fine-trace classical mixed configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-factor", type=int, default=8)
    parser.add_argument("--macro-factor", type=int, default=1)
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--quadrature-xy", type=int, default=20)
    parser.add_argument("--quadrature-z", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--balance-order", type=int, default=3)
    parser.add_argument("--global-rtol", type=float, default=1e-10)
    parser.add_argument(
        "--global-refinement-precision", choices=["double", "extended"], default="double"
    )
    args = parser.parse_args()
    if args.fine_factor < 1 or args.macro_factor < 1 or args.fine_factor % args.macro_factor:
        parser.error("positive macro factor must divide the fine factor")
    if args.balance_order < 2:
        parser.error("balance-order must be at least two for RT1/Q1 divergence moments")
    data = OscillatoryWellData()
    radii = np.geomspace(data.inner_radius, data.outer_radius, 5)
    mesh = HexMesh.annular_prism(radii, data.height, 8).refined(args.macro_factor)
    refinement = args.fine_factor // args.macro_factor
    subdivisions = refinement if args.classical else 1
    order = (args.quadrature_xy, args.quadrature_xy, args.quadrature_z)
    zero = {
        int(f): 0.0 for f in mesh.boundary_faces if np.ptp(mesh.points[mesh.faces[f], 2]) < 1e-12
    }
    sources = [
        Path(__file__),
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
            local_refinement=refinement,
            subdivisions=subdivisions,
            permeability=data.tensor,
            dirichlet=data.pressure,
            neumann=zero,
            quadrature_order=order,
            backend="process" if args.workers > 1 else "serial",
            workers=args.workers,
            global_rtol=args.global_rtol,
            global_refinement_precision=args.global_refinement_precision,
        )
        diagnostic = replace(solution, quadrature_order=args.balance_order)
        balance = max(float(abs(v).max()) for v in diagnostic.equilibrium_residuals())
    rates = dict(inner=0.0, outer=0.0, top_bottom=0.0)
    for face in mesh.boundary_faces:
        label = (
            "top_bottom"
            if int(face) in zero
            else "inner"
            if np.max(np.linalg.norm(mesh.points[mesh.faces[face], :2], axis=1)) < 1
            else "outer"
        )
        start = face * solution.skeleton.face_size
        rates[label] += float(
            solution.hybrid.trace[start : start + solution.skeleton.face_size : 4].sum()
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"fine{args.fine_factor}-macro{args.macro_factor}-s{subdivisions}-q{order[0]}z{order[2]}"
    if args.global_rtol != 1e-10 or args.global_refinement_precision != "double":
        name += f"-rtol{args.global_rtol:g}-{args.global_refinement_precision}"
    archive = OUTPUT / (name + ".npz")
    np.savez_compressed(
        archive,
        macro_points=mesh.points,
        macro_cells=mesh.cells,
        local_points=np.stack([m.points for m in solution.local_meshes]),
        local_cells=np.stack([m.cells for m in solution.local_meshes]),
        pressure=np.stack(solution.pressure),
        flux=np.stack(solution.flux),
        trace=solution.hybrid.trace,
    )
    report = dict(
        global_rtol=args.global_rtol,
        global_refinement_precision=args.global_refinement_precision,
        case="L05 Problem 5 tensor on an explicitly graded polygonal producing-well reservoir",
        physical_parameters=data.__dict__,
        radii=radii.tolist(),
        units="m, Pa, s; permeability m² and physical flux m/s",
        tensor_scale_convention=(
            "10^-11 m² outer scale matches Figure16 ranges; printed 10^-3 differs"
        ),
        rate_convention=(
            "Q>0 prescribes production pressure data; actual heterogeneous rate is computed, "
            "not prescribed"
        ),
        fine_factor=args.fine_factor,
        macro_factor=args.macro_factor,
        local_refinement=refinement,
        subdivisions=subdivisions,
        classical_complete_trace=args.classical,
        macro_cells=len(mesh.cells),
        fine_cells=sum(len(m.cells) for m in solution.local_meshes),
        degree=1,
        trace_dofs=solution.skeleton.size,
        pressure_coarse_dofs=len(mesh.cells),
        quadrature=order,
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
        raise RuntimeError(
            "Acquisition source changed; inspect stored source snapshots before publication"
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
