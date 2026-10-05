"""Acquire original rectangular RT mixed-elasticity convergence and locking studies."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from elasticity_data import ElasticityData
from threadpoolctl import threadpool_limits

from pymhm._legacy.models.elasticity.stress_tensor import solve_elasticity_tensor_rt
from pymhm.io.provenance import current_source_manifest
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/elasticity-tensor-rt"


def acquire(degree: int, enrichment: int, resolution: int, lam: float) -> tuple:
    """Solve the declared rectangular family and integrate errors and equilibrium moments."""
    data = ElasticityData(lam)
    mesh = CartesianMacroMesh(resolution)
    result = solve_elasticity_tensor_rt(
        mesh,
        degree=degree,
        enrichment=enrichment,
        local_refinement=1,
        quadrature_order=9,
        lame_lambda=lam,
        source=data.source,
        dirichlet=data.displacement,
    )

    def rotation(points: np.ndarray) -> np.ndarray:
        """Use the analytical skew-gradient convention of the independent rotation."""
        gradient = data.gradient(points)
        return (gradient[:, 0, 1] - gradient[:, 1, 0]) / 2

    row = dict(
        resolution=resolution,
        macro_cells=len(mesh.cells),
        local_refinement=1,
        degree=degree,
        enrichment=enrichment,
        trace_dofs=result.skeleton.size,
        interior_trace_degree=1,
        exterior_trace_degree=degree,
        lame_lambda="infinity" if np.isinf(lam) else lam,
        **result.errors(
            data.displacement, data.stress, lambda x: -data.source(x), rotation, order=10
        ),
        algebraic_residual=result.hybrid.residual,
        force_moment_linf=float(np.abs(result.equilibrium_residuals()).max()),
        fine_force_linf=max(float(np.abs(v).max()) for v in result.fine_force_residuals()),
        weak_symmetry_linf=max(float(np.abs(v).max()) for v in result.weak_symmetry_residuals()),
        normal_traction_linf=max(
            float(np.abs(v).max()) for v in result.normal_traction_residuals()
        ),
    )
    print(json.dumps(row), flush=True)
    return result, data, row


def archive(solution, data: ElasticityData, path: Path) -> str:
    """Archive independent one-sided values inside every fine rectangular cell."""
    reference = CartesianMacroMesh(5)
    points, values, stresses, cells = [], [], [], []
    triangles = np.concatenate((reference.cells[:, [0, 1, 2]], reference.cells[:, [0, 2, 3]]))
    offset = 0
    for cell, mesh in enumerate(solution.local_meshes):
        physical = mesh.points[mesh.cells[:, 0], None] + reference.points * mesh.spacing
        displacement, stress, _, _ = solution.evaluate(cell, reference.points)
        points.append(physical.reshape(-1, 2))
        values.append(displacement.reshape(-1, 2))
        stresses.append(stress.reshape(-1, 2, 2))
        cells.extend(triangles + offset + i * len(reference.points) for i in range(len(mesh.cells)))
        offset += physical.shape[0] * physical.shape[1]
    coordinates = np.concatenate(points)
    np.savez_compressed(
        path,
        points=coordinates,
        cells=np.concatenate(cells),
        displacement=np.concatenate(values),
        stress=np.concatenate(stresses),
        exact_displacement=data.displacement(coordinates),
        exact_stress=data.stress(coordinates),
        macro_points=solution.skeleton.mesh.points,
        macro_cells=solution.skeleton.mesh.cells,
        macro_faces=solution.skeleton.mesh.faces,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run five spatial levels and five finite/infinite moduli with fixed declared spaces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--enrichment", type=int, default=0)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = parser.parse_args()
    rows = []
    with threadpool_limits(1):
        for n in args.resolutions:
            result, data, row = acquire(args.degree, args.enrichment, n, 1.0)
            rows.append(row)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        name = f"rt{args.degree}-enrichment{args.enrichment}"
        path = OUTPUT / (name + ".npz")
        digest = archive(result, data, path)
        sweep = [
            acquire(args.degree, args.enrichment, 2, lam)[2] for lam in (1.0, 1e2, 1e4, 1e8, np.inf)
        ]
    sources = [
        Path(__file__),
        ROOT / "examples/elasticity_data.py",
        ROOT / "src/pymhm/_legacy/models/elasticity/stress_tensor.py",
        ROOT / "src/pymhm/fem/hdiv/tensor_rt.py",
    ]
    report = dict(
        case="Original bounded-force polynomial elasticity",
        geometry="Cartesian rectangles",
        displacement_space="Q_(k+n)^2",
        rotation_space="P_(k+n) total degree",
        rows=rows,
        sweep=sweep,
        assembly_quadrature=9,
        error_quadrature=10,
        archive=path.name,
        sha256=digest,
        source_hashes=current_source_manifest(
            {
                p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            }
        ),
    )
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
