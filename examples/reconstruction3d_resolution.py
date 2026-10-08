"""Separate local and skeletal resolution for the localized three-dimensional Darcy case."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.polynomial.legendre import leggauss
from threadpoolctl import threadpool_limits

from examples.formulations.application import tetrahedral_darcy as solve_darcy_3d
from examples.reconstruction3d_data import fields
from pymhm import TetraMesh, TriangularSkeleton, reconstruct_darcy_moments_3d
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/reconstruction3d"


def reference_norms() -> np.ndarray:
    """Integrate the analytical volume norms with a separately converged tensor Gauss rule."""
    norms = []
    for order in (32, 40):
        nodes, weights = leggauss(order)
        coordinates = np.stack(np.meshgrid(*([(nodes + 1) / 2] * 3), indexing="ij"), axis=-1)
        products = np.prod(np.stack(np.meshgrid(*([weights / 2] * 3), indexing="ij")), axis=0)
        p, q, _ = fields(coordinates.reshape(-1, 3), True)
        norms.append(np.sqrt(products.ravel() @ np.column_stack((p**2, (q**2).sum(axis=1)))))
    if np.max(abs(norms[0] - norms[1])) > 2e-13:
        raise ArithmeticError("analytical norm integration is unresolved")
    return norms[-1]


def run() -> None:
    """Hold the physical problem and geometry fixed while changing approximation spaces."""
    path = ROOT / "examples/data/reconstruction3d-macro.json"
    geometry = json.loads(path.read_text())
    macro = TetraMesh(np.asarray(geometry["macro_points"]), np.asarray(geometry["macro_cells"]))
    norms = reference_norms()
    owners = [
        ROOT / f"src/pymhm/{name}.py"
        for name in (
            "_legacy/models/darcy/primal_3d",
            "recovery/moments_3d",
            "fem/hdiv/rt_3d",
            "meshes/mixed",
            "fem/hdiv/family_3d",
            "fem/scalar/tetrahedron",
            "fem/scalar/tetrahedron_topology",
            "core/contracts",
            "linalg/linear",
        )
    ] + [Path(__file__), ROOT / "examples/reconstruction3d_data.py"]
    hashes = current_source_manifest(
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    )
    record = dict(
        reference="analytical",
        fixed_macro_input=path.relative_to(ROOT).as_posix(),
        fixed_macro_input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashes,
        exact_norms=norms.tolist(),
        rows=[],
    )
    for degree, trace_degree, segments, rt_degree in (
        (2, 0, 1, 1),
        (4, 0, 1, 1),
        (4, 2, 1, 2),
        (4, 2, 2, 2),
    ):
        started = perf_counter()
        skeleton = TriangularSkeleton(macro, segments, degree=trace_degree)
        solution = solve_darcy_3d(
            macro,
            skeleton=skeleton,
            degree=degree,
            local_refinement=2,
            source=lambda x: fields(x, True)[2],
            quadrature_order=12,
        )
        print(f"Solved P{degree}, trace P{trace_degree}, subdivisions {segments}", flush=True)
        reconstruction = reconstruct_darcy_moments_3d(
            solution, degree=rt_degree, quadrature_order=12
        )
        errors = []
        for order in (10, 12):
            errors.append(
                np.array(
                    [
                        solution.l2_error(lambda x: fields(x, True)[0], order),
                        solution.flux_l2_error(lambda x: fields(x, True)[1], order),
                        reconstruction.flux_l2_error(lambda x: fields(x, True)[1], order),
                    ]
                )
            )
        change = float(np.max(abs(errors[1] - errors[0])))
        if change > 2e-9:
            raise ArithmeticError("physical error integration is unresolved")
        name = f"resolution-p{degree}-t{trace_degree}-s{segments}"
        arrays = dict(
            macro_points=macro.points,
            macro_cells=macro.cells,
            local_degree=degree,
            reconstruction_degree=rt_degree,
            rt_basis=reconstruction.family.coefficients,
            trace=solution.hybrid.trace,
        )
        for cell, fine in enumerate(solution.local_meshes):
            arrays.update(
                {
                    f"points_{cell}": fine.points,
                    f"cells_{cell}": fine.cells,
                    f"pressure_{cell}": solution.pressure[cell],
                    f"rt_flux_{cell}": reconstruction.flux[cell],
                }
            )
        archive_path = OUTPUT / f"{name}.npz"
        np.savez_compressed(archive_path, **arrays)
        row = dict(
            name=name,
            macro_cells=len(macro.cells),
            trace_dofs=skeleton.size,
            local_degree=degree,
            trace_degree=trace_degree,
            trace_subdivisions=segments,
            local_refinement=2,
            reconstruction_degree=rt_degree,
            pressure_l2=errors[1][0],
            raw_flux_l2=errors[1][1],
            rt_flux_l2=errors[1][2],
            pressure_relative=errors[1][0] / norms[0],
            raw_flux_relative=errors[1][1] / norms[1],
            rt_flux_relative=errors[1][2] / norms[1],
            error_quadrature_change=change,
            macro_balance=float(np.max(abs(solution.conservation_residuals()))),
            elapsed_seconds=perf_counter() - started,
            archive=archive_path.name,
            archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        )
        record["rows"].append(row)
        (OUTPUT / "resolution.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if hashes != current_source_manifest(
        {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    ):
        raise RuntimeError("campaign sources changed during acquisition")
    record["source_changed_during_run"] = False
    (OUTPUT / "resolution.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
