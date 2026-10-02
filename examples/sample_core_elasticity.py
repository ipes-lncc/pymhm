"""Acquire complete one-sided displacement/stress display fields and projection error checks."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from examples import core_extension_data as data
from examples.elasticity_field_samples import (
    projection_decomposition,
    sample_elasticity_fields,
    sample_elasticity_profile,
)
from examples.solve_core_extensions import polygon_grid
from pymhm import CartesianMacroMesh, TriangleMesh
from pymhm.elasticity_mixed import solve_elasticity_mixed
from pymhm.elasticity_tensor_rt import solve_elasticity_tensor_rt
from pymhm.polygon import solve_elasticity_mixed_polygons

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/core-extensions"


def projection_study() -> None:
    """Measure the local projection and discrete error independently on four polygon meshes."""
    owners = [
        Path(__file__),
        ROOT / "examples/elasticity_field_samples.py",
        ROOT / "examples/core_extension_data.py",
        ROOT / "src/pymhm/elasticity_mixed.py",
        ROOT / "src/pymhm/bdm_family.py",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    rows = []
    for n in (2, 4, 8, 16):
        solution = solve_elasticity_mixed_polygons(
            polygon_grid(n),
            compliance=data.compliance(),
            source=data.force,
            dirichlet=data.displacement,
            local_refinement=1,
            quadrature_order=8,
        )
        decomposition, _ = projection_decomposition(solution, data.displacement, 9)
        rows.append({"resolution": n, **decomposition})
        print(json.dumps(rows[-1]), flush=True)
    if hashes != {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners
    }:
        raise RuntimeError("projection study sources changed during acquisition")
    (OUTPUT / "elasticity-projection.json").write_text(
        json.dumps({"source_sha256": hashes, "rows": rows}, indent=2) + "\n"
    )


def run() -> None:
    """Recompute each archived finest state and verify its physical norms before field export."""
    record = json.loads((OUTPUT / "elasticity.json").read_text())
    owner = ROOT / "examples/elasticity_field_samples.py"
    report = dict(
        sampling_refinement=6,
        source_sha256=hashlib.sha256(owner.read_bytes()).hexdigest(),
        cases={},
    )
    for name in record["fields"]:
        row = [r for r in record["rows"] if r["case"] == name][-1]
        n = row["resolution"]
        if name.startswith("triangle"):
            mesh, solve = TriangleMesh.unit_square(n), solve_elasticity_mixed
        elif name.startswith("rectangle"):
            mesh, solve = CartesianMacroMesh(n), solve_elasticity_tensor_rt
        else:
            mesh, solve = polygon_grid(n), solve_elasticity_mixed_polygons
        solution = solve(
            mesh,
            compliance=data.compliance(),
            source=data.force,
            dirichlet=data.displacement,
            local_refinement=1,
            quadrature_order=8,
        )
        error = (
            solution.errors(
                data.displacement, data.stress, lambda x: -data.force(x), data.rotation, 9
            )["displacement_l2"]
            if name.startswith("rectangle")
            else solution.l2_error(data.displacement, 9)
        )
        if not np.isclose(error, row["displacement_l2"], rtol=2e-11, atol=1e-15):
            raise ArithmeticError("the re-evaluated displacement differs from the acquired state")
        fields = sample_elasticity_fields(solution, 6)
        fields["exact"] = np.column_stack(
            (data.displacement(fields["points"]), data.stress(fields["points"]).reshape(-1, 4))
        )
        for label, start, end in (
            ("vertical", [0.43, 0], [0.43, 1]),
            ("horizontal", [0, 0.37], [1, 0.37]),
        ):
            profile = sample_elasticity_profile(solution, start, end)
            points = profile["points"]
            profile["exact"] = np.column_stack(
                (
                    data.displacement(points.reshape(-1, 2)),
                    data.stress(points.reshape(-1, 2)).reshape(-1, 4),
                )
            ).reshape(*points.shape[:2], 6)
            fields.update({f"profile_{label}_{key}": value for key, value in profile.items()})
        path = OUTPUT / f"{name}-polynomial-fields.npz"
        np.savez_compressed(path, **fields)
        info = dict(
            archive=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            displacement_l2=error,
        )
        if not name.startswith("rectangle"):
            projection, _ = projection_decomposition(solution, data.displacement, 9)
            info["projection"] = projection
        report["cases"][name] = info
        print(json.dumps({name: info}), flush=True)
        (OUTPUT / "elasticity-field-sampling.json").write_text(json.dumps(report, indent=2) + "\n")
    if hashlib.sha256(owner.read_bytes()).hexdigest() != report["source_sha256"]:
        raise RuntimeError("field sampling owner changed during acquisition")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-study", action="store_true")
    arguments = parser.parse_args()
    with threadpool_limits(1):
        projection_study() if arguments.projection_study else run()
