"""Acquire mixed-stress-family convergence and incompressible-limit evidence.

These are original polynomial manufactured problems with bounded force, rather
than a claim of reproducing a particular historical table. Numerical campaigns
remain separate from the small algebraic and native-assembly tests in CI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymhm.postprocessing.stress import MixedElasticitySolution

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples.elasticity_data import ElasticityData
from examples.formulations.application import weak_stress_elasticity as solve_elasticity_mixed
from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.fem.scalar.triangle import reference_basis
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/elasticity-families"


def acquire(degree: int, enrichment: int, resolution: int, lam: float) -> tuple:
    """Solve one declared family and integrate physical errors and balance defects."""
    data = ElasticityData(lam)
    mesh = TriangleMesh.unit_square(resolution)
    trace_degree = min(degree, degree + enrichment - 1)
    boundary = set(mesh.boundary_faces)
    skeleton = SkeletonSpace(
        mesh,
        tuple(
            FaceSpace.uniform(degree if face in boundary else trace_degree)
            for face in range(len(mesh.faces))
        ),
        2,
    )
    solution = solve_elasticity_mixed(
        mesh,
        stress_degree=degree,
        enrichment=enrichment,
        skeleton=skeleton,
        local_refinement=1,
        quadrature_order=9,
        lame_lambda=lam,
        source=data.source,
        dirichlet=data.displacement,
    )

    def rotation(points: np.ndarray) -> np.ndarray:
        """Evaluate the analytical weak-rotation convention from exact derivatives."""
        gradient = data.gradient(points)
        return (gradient[:, 0, 1] - gradient[:, 1, 0]) / 2

    row = dict(
        resolution=resolution,
        macro_cells=len(mesh.cells),
        trace_degree=trace_degree,
        trace_dofs=skeleton.size,
        stress_degree=degree,
        enrichment=enrichment,
        lame_lambda="infinity" if np.isinf(lam) else lam,
        displacement_l2=solution.l2_error(data.displacement, 10),
        stress_l2=solution.stress_l2_error(data.stress, 10),
        rotation_l2=solution.rotation_l2_error(rotation, 10),
        divergence_l2=solution.divergence_l2_error(lambda x: -data.source(x), 10),
        force_moment_linf=float(np.abs(solution.equilibrium_residuals()).max()),
        fine_force_linf=max(float(np.abs(v).max()) for v in solution.fine_force_residuals()),
        weak_symmetry_linf=max(float(np.abs(v).max()) for v in solution.weak_symmetry_residuals()),
        normal_traction_linf=max(
            float(np.abs(v).max()) for v in solution.normal_traction_residuals()
        ),
        algebraic_residual=solution.hybrid.residual,
    )
    print(json.dumps(row), flush=True)
    return solution, data, row


def archive(solution: MixedElasticitySolution, data: ElasticityData, filename: Path) -> str:
    """Save broken display samples, keeping every fine triangle's one-sided values."""
    reference = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]]).submesh(0, 3)
    bary = np.column_stack((1 - reference.points.sum(axis=1), reference.points))
    scalar = reference_basis(solution.displacement_degree, bary)[0]
    points, cells, values, stresses = [], [], [], []
    offset = 0
    for mesh, u, sigma in zip(
        solution.local_meshes, solution.displacement, solution.stress, strict=True
    ):
        physical = np.einsum("qi,tij->tqj", bary, mesh.points[mesh.cells]).reshape(-1, 2)
        points.append(physical)
        values.append(np.einsum("qi,tia->tqa", scalar, u).reshape(-1, 2))
        stresses.append(solution.family.evaluate(mesh, sigma, bary)[0].reshape(-1, 2, 2))
        cells.extend(reference.cells + offset + cell * len(bary) for cell in range(len(mesh.cells)))
        offset += len(physical)
    physical = np.concatenate(points)
    np.savez_compressed(
        filename,
        points=physical,
        cells=np.concatenate(cells),
        displacement=np.concatenate(values),
        stress=np.concatenate(stresses),
        exact_displacement=data.displacement(physical),
        exact_stress=data.stress(physical),
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
    )
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def main() -> None:
    """Acquire a five-level family study and an independent fixed-space bulk-modulus sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--enrichment", type=int, default=1)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"bdm{args.degree}-enrichment{args.enrichment}"
    with threadpool_limits(1):
        rows = []
        for resolution in args.levels:
            solution, data, row = acquire(args.degree, args.enrichment, resolution, 1.0)
            rows.append(row)
        checksum = archive(solution, data, OUTPUT / (name + ".npz"))
        sweep = [
            acquire(args.degree, args.enrichment, 2, lam)[2] for lam in (1.0, 1e2, 1e4, 1e8, np.inf)
        ]
    sources = (
        Path(__file__),
        source_file("examples/elasticity_data.py", root=ROOT),
        source_file("src/pymhm/fem/hdiv/bdm_family.py", root=ROOT),
        source_file("src/pymhm/_legacy/models/elasticity/stress.py", root=ROOT),
    )
    report = dict(
        case="original bounded-force polynomial elasticity",
        rows=rows,
        sweep=sweep,
        assembly_quadrature=9,
        error_quadrature=10,
        local_refinement=1,
        archive=name + ".npz",
        sha256=checksum,
        source_hashes=current_source_manifest(
            source_identity(ROOT, sources), packages=("pymhm", "examples")
        ),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_elasticity_families").main()
