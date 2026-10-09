"""Acquire manufactured three-dimensional flow with independent error quadrature."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.flow3d_data import Flow3DData
from examples.formulations.application import flow as solve_flow_3d
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.solutions import Flow3DSolution

ROOT = case_workspace()
CASES = (
    ("stokes-th2", "stokes", "taylor-hood", 2, 2),
    ("brinkman-usfem1", "brinkman", "usfem", 1, 4),
    ("brinkman-usfem2", "brinkman", "usfem", 2, 2),
    ("oseen-p2", "oseen", "oseen", 2, 2),
)


def norms(solution: Flow3DSolution, data: Flow3DData, order: int) -> dict[str, float]:
    """Integrate separate velocity, pressure, gradient and divergence diagnostics."""
    return dict(
        velocity_l2=solution.l2_error(data.velocity, order),
        pressure_l2=solution.pressure_l2_error(data.pressure, order),
        velocity_h1_seminorm=solution.h1_seminorm_error(data.gradient, order),
        divergence_l2=solution.divergence_l2(order),
        backward_residual=solution.hybrid.residual,
    )


def snapshot() -> dict[str, str]:
    """Hash the exact shared runtime and original analytic acquisition sources."""
    files = [Path(__file__), source_file("examples/flow3d_data.py", root=ROOT)]
    files.extend(
        source_file(f"src/pymhm/{name}", root=ROOT)
        for name in (
            "_legacy/models/flow/solver_3d.py",
            "_legacy/models/flow/forms_3d.py",
            "_legacy/models/flow/solver.py",
            "_legacy/models/darcy/primal_3d.py",
            "_legacy/models/transport/rad_3d.py",
            "fem/scalar/tetrahedron.py",
            "fem/scalar/tetrahedron_topology.py",
            "core/contracts.py",
            "linalg/linear.py",
            "execution/cpu.py",
        )
    )
    return current_source_manifest(source_identity(ROOT, files), packages=("pymhm", "examples"))


def main() -> None:
    """Run five macro refinements for each flow family and persist one-sided nodal fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/flow3d")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--cases", nargs="+", choices=[case[0] for case in CASES])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = snapshot()
    report = dict(
        timestamp_utc=datetime.now(UTC).isoformat(),
        source_sha256=hashes,
        python=platform.python_version(),
        numpy=np.__version__,
        native_threads=1,
        operator="-nu Delta(u) + beta.grad(u) + Gamma*u + grad(p) = f; div(u)=0",
        domain="unit cube; six Freudenthal tetrahedra per Cartesian macro cube",
        boundary="exact velocity weakly on the entire exterior; exact zero mean pressure",
        evidence=(
            "original manufactured 3D verification of dimension-independent forms; "
            "not a historical figure reproduction"
        ),
        norm_denominator="absolute physical L2 and broken gradient norms",
        rows=[],
    )
    with threadpool_limits(1):
        for name, kind, formulation, degree, refinement in CASES:
            if args.cases and name not in args.cases:
                continue
            data = Flow3DData(kind)
            for n in args.levels:
                mesh = TetraMesh.unit_cube(n)
                skeleton = TriangularSkeleton(mesh, degree=1)
                started = perf_counter()
                solution = solve_flow_3d(
                    mesh,
                    skeleton=skeleton,
                    formulation=formulation,
                    degree=degree,
                    local_refinement=refinement,
                    quadrature_order=6,
                    backend="process" if args.workers > 1 else "serial",
                    workers=args.workers,
                    **data.options(),
                )
                result = norms(solution, data, 7)
                check = norms(solution, data, 8)
                discrepancy = max(
                    abs(result[key] - check[key]) / max(check[key], 1e-30)
                    for key in (
                        "velocity_l2",
                        "pressure_l2",
                        "velocity_h1_seminorm",
                        "divergence_l2",
                    )
                )
                if discrepancy > 1e-8:
                    raise ValueError(
                        f"error quadrature insufficient: {name}, n={n}, delta={discrepancy}"
                    )
                archive = args.output / f"{name}-n{n}.npz"
                np.savez_compressed(
                    archive,
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                    local_points=np.asarray([fine.points for fine in solution.local_meshes]),
                    local_cells=np.asarray([fine.cells for fine in solution.local_meshes]),
                    velocity=np.asarray(solution.velocity),
                    pressure=np.asarray(solution.pressure),
                    degree=solution.degree,
                    pressure_degree=solution.pressure_degree,
                    hybrid_trace=solution.hybrid.trace,
                )
                row = dict(
                    case=name,
                    physical_case=kind,
                    formulation=formulation,
                    degree=degree,
                    pressure_degree=solution.pressure_degree,
                    local_refinement=refinement,
                    face_degree=1,
                    face_subdivisions=1,
                    macro_subdivisions=n,
                    macro_tetrahedra=len(mesh.cells),
                    fine_tetrahedra=sum(len(f.cells) for f in solution.local_meshes),
                    assembly_quadrature_order=6,
                    error_quadrature_orders=[7, 8],
                    max_error_quadrature_relative_difference=discrepancy,
                    stabilization="tensor-2025"
                    if formulation == "usfem"
                    else "oseen-2021"
                    if formulation == "oseen"
                    else "none",
                    fields=archive.name,
                    fields_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    elapsed_seconds=perf_counter() - started,
                    **check,
                )
                report["rows"].append(row)
                if snapshot() != hashes:
                    raise ValueError("acquisition source changed during execution")
                report["source_changed_during_run"] = False
                (args.output / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_flow3d").main()
