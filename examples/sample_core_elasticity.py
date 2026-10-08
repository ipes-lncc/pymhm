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
from examples.formulations.application import weak_stress_elasticity as solve_elasticity_mixed
from examples.formulations.application import (
    weak_stress_elasticity as solve_elasticity_mixed_polygons,
)
from examples.formulations.application import (
    weak_stress_elasticity as solve_elasticity_tensor_rt,
)
from examples.solve_core_extensions import polygon_grid
from pymhm import CartesianMacroMesh, TriangleMesh
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, read_resource_text, source_file, source_identity

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/core-extensions"


def projection_study() -> None:
    """Measure the local projection and discrete error independently on four polygon meshes."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    owners = [
        Path(__file__),
        source_file("examples/elasticity_field_samples.py", root=ROOT),
        source_file("examples/core_extension_data.py", root=ROOT),
        source_file("src/pymhm/_legacy/models/elasticity/stress.py", root=ROOT),
        source_file("src/pymhm/fem/hdiv/bdm_family.py", root=ROOT),
    ]
    hashes = current_source_manifest(source_identity(ROOT, owners), packages=("pymhm", "examples"))
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
    if hashes != current_source_manifest(
        source_identity(ROOT, owners), packages=("pymhm", "examples")
    ):
        raise RuntimeError("projection study sources changed during acquisition")
    (OUTPUT / "elasticity-projection.json").write_text(
        json.dumps({"source_sha256": hashes, "rows": rows}, indent=2) + "\n"
    )


def run() -> None:
    """Recompute each archived finest state and verify its physical norms before field export."""
    record = json.loads(read_resource_text(OUTPUT / "elasticity.json"))
    owner = source_file("examples/elasticity_field_samples.py", root=ROOT)
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


def main() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-study", action="store_true")
    arguments = parser.parse_args()
    with threadpool_limits(1):
        projection_study() if arguments.projection_study else run()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.sample_core_elasticity").main()
