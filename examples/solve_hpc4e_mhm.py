"""Acquire the published 16×8, RT1/Q1/P1 HPC4e elastic-gravity comparison."""

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

from examples.formulations.application import weak_stress_elasticity as solve_elasticity_tensor_rt
from examples.hpc4e_data import BOUNDS, DATA_DIRECTORY, LENGTH_SCALE, STRESS_SCALE, load_data
from pymhm import FaceSpace, SkeletonSpace
from pymhm.fem.hdiv.tensor_rt import tensor_rt_dofs
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]


def archive(solution, path: Path) -> str:
    """Store element-local polynomial coefficients with positive local RT orientation."""
    macro = solution.skeleton.mesh
    fine = solution.local_meshes[0]
    nx, ny = macro.nx * fine.nx, macro.ny * fine.ny
    k, enrichment = solution.degree, solution.enrichment
    width = tensor_rt_dofs(fine, k, enrichment).shape[1]
    stress = np.empty((nx * ny, width, 2))
    displacement = np.empty((nx * ny, (k + enrichment + 1) ** 2, 2))
    rotation = np.empty((nx * ny, (k + enrichment + 1) * (k + enrichment + 2) // 2))
    for cell, local in enumerate(solution.local_meshes):
        centers = local.points[local.cells].mean(axis=1)
        indices = np.floor(centers / fine.spacing).astype(np.int64)
        ids = indices[:, 0] + nx * indices[:, 1]
        values = solution.stress[cell][tensor_rt_dofs(local, k, enrichment)].copy()
        orientation = (local.signs[:, :, None] ** np.arange(1, k + 2)).reshape(len(local.cells), -1)
        values[:, : 4 * (k + 1)] *= orientation[:, :, None]
        stress[ids] = values
        displacement[ids] = solution.displacement[cell]
        rotation[ids] = solution.rotation[cell]
    np.savez_compressed(
        path,
        stress=stress,
        displacement=displacement,
        rotation=rotation,
        nx=nx,
        ny=ny,
        degree=k,
        enrichment=enrichment,
        bounds=BOUNDS,
        macro_points=macro.points,
        macro_cells=macro.cells,
        macro_faces=macro.faces,
        trace=solution.hybrid.trace,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Solve identical material, fine space and boundary data for four skeletal partitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--local-solver", default="scipy")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "examples/results/hpc4e")
    parser.add_argument("--field-dir", type=Path, default=ROOT / "build/results/hpc4e")
    args = parser.parse_args()
    data = load_data(args.data, download=args.download)
    lam, mu, body = data.fields()
    output, fields = args.record_dir.resolve(), args.field_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fields.mkdir(parents=True, exist_ok=True)
    snapshots = fields / "acquisition-sources"
    snapshots.mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for path in (
        Path(__file__),
        ROOT / "examples/hpc4e_data.py",
        ROOT / "src/pymhm/_legacy/models/elasticity/stress_tensor.py",
        ROOT / "src/pymhm/fem/hdiv/tensor_rt.py",
        ROOT / "src/pymhm/core/contracts.py",
        ROOT / "src/pymhm/linalg/linear.py",
    ):
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        source_hashes[path.relative_to(ROOT).as_posix()] = digest
        (snapshots / f"{digest}.py").write_bytes(content)
    mesh = CartesianMacroMesh(16, 8, BOUNDS)
    mids = mesh.points[mesh.faces].mean(axis=1)
    free_sides = {
        int(face): (0.0, 0.0)
        for face in mesh.boundary_faces
        if not np.isclose(mids[face, 1], 0, rtol=0, atol=1e-14)
    }
    with threadpool_limits(1):
        for segments in args.segments:
            if segments not in (1, 2, 4, 8):
                raise ValueError("published skeletal partitions are 1, 2, 4 and 8")
            started = perf_counter()
            print(
                f"Solving HPC4e: RT1, 128 macrorectangles, 131072 fine cells, s={segments}",
                flush=True,
            )
            skeleton = SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(1, segments) for _ in mesh.faces), components=2
            )
            result = solve_elasticity_tensor_rt(
                mesh,
                degree=1,
                enrichment=0,
                lame_lambda=lam,
                lame_mu=mu,
                source=body,
                dirichlet=(0.0, 0.0),
                traction=free_sides,
                skeleton=skeleton,
                local_refinement=32,
                quadrature_order=4,
                backend="process" if args.workers > 1 else "serial",
                workers=args.workers,
                local_solver=args.local_solver,
            )
            solve_seconds = perf_counter() - started
            target = fields / f"mhm-s{segments}.npz"
            digest = archive(result, target)
            diagnostics = dict(
                algebraic_residual=result.hybrid.residual,
                force_moment_linf=float(np.abs(result.equilibrium_residuals()).max()),
                fine_force_linf=max(float(np.abs(v).max()) for v in result.fine_force_residuals()),
                weak_symmetry_linf=max(
                    float(np.abs(v).max()) for v in result.weak_symmetry_residuals()
                ),
                normal_traction_linf=max(
                    float(np.abs(v).max()) for v in result.normal_traction_residuals()
                ),
            )
            report = dict(
                case="Devloo et al. (2021), Section 6.2, Figures 10 and 11",
                macro_shape=[16, 8],
                fine_shape=[512, 256],
                degree=1,
                enrichment=0,
                displacement_space="Q1 squared",
                rotation_space="total-degree P1",
                skeleton_degree=1,
                skeleton_segments=segments,
                skeleton_dofs=skeleton.size,
                coarse_rigid_dofs=3 * len(mesh.cells),
                unknown_global_dofs=skeleton.size - 4 * segments * len(free_sides) + 384,
                assembly_quadrature=4,
                length_scale_m=LENGTH_SCALE,
                stress_scale_pa=STRESS_SCALE,
                solve_seconds=solve_seconds,
                workers=args.workers,
                local_solver=args.local_solver,
                diagnostics=diagnostics,
                field_archive=target.relative_to(ROOT).as_posix(),
                sha256=digest,
                source_hashes=source_hashes,
            )
            (output / f"mhm-s{segments}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            del result


if __name__ == "__main__":
    main()
