"""Separate high-order resolution control for the analytical reaction layer."""

import hashlib
import json
from pathlib import Path

import numpy as np
from field_sampling import sample_field, sample_profile
from solve_unusual import configuration, errors
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm.rad import solve_rad

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "examples/results/unusual"


def main() -> None:
    """Check P3/P2 spaces over three additional meshes at unchanged epsilon=0.001."""
    rows = []
    names = (
        "src/pymhm/rad.py",
        "src/pymhm/unusual.py",
        "examples/solve_unusual.py",
        "examples/verify_unusual_resolution.py",
    )
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    with threadpool_limits(1):
        for n in (8, 16, 32):
            mesh = TriangleMesh.unit_square(n)
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(2) for _ in mesh.faces))
            options, exact, gradient = configuration("layer", 1e-3)
            natural = {int(f): 0.0 for f in mesh.boundary_faces if abs(mesh.normals[f, 1]) > 0.99}
            solution = solve_rad(
                mesh,
                skeleton=skeleton,
                degree=3,
                local_refinement=2,
                stabilization="unusual",
                dirichlet_enforcement="strong",
                neumann=natural,
                quadrature_order=10,
                **options,
            )
            row = dict(
                n=n,
                case="layer-high-order",
                method="unusual",
                epsilon=1e-3,
                local_degree=3,
                trace_degree=2,
                local_refinement=2,
                macro_cells=len(mesh.cells),
                trace_dofs=skeleton.size,
                residual=solution.hybrid.residual,
                error_quadrature=12,
                **errors(solution, 1e-3, exact, gradient, 12),
            )
            row["quadrature_check"] = dict(order=20, **errors(solution, 1e-3, exact, gradient, 20))
            if n == 16:
                data = sample_field(solution.local_meshes, solution.values, 3, refinement=4)
                data.update(exact=exact(data["points"]), exact_gradient=gradient(data["points"]))
                data.update(
                    flux=-1e-3 * data["gradient"], exact_flux=-1e-3 * data["exact_gradient"]
                )
                data.update(sample_profile(solution, solution.values, 3))
                data["profile_exact"] = exact(data["profile_points"].reshape(-1, 2)).reshape(
                    data["profile_parameter"].shape
                )
                data.update(macro_points=mesh.points, macro_cells=mesh.cells)
                filename = "layer-high-order-n16.npz"
                np.savez_compressed(TARGET / filename, **data)
                row.update(
                    archive=filename,
                    archive_sha256=hashlib.sha256((TARGET / filename).read_bytes()).hexdigest(),
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
            result = dict(
                method="MHM-UNUSUAL; additional resolution study, not a historical mesh claim",
                rows=rows,
                source_hashes=hashes,
                source_changed_during_run=any(
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
                    for name, digest in hashes.items()
                ),
            )
            (TARGET / "resolution-control.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
