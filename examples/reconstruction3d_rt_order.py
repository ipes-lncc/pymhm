"""Compare RT2 and RT3 reconstruction of one unchanged archived P4 MHM field."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.reconstruction3d_data import fields
from pymhm import TetraMesh, TriangularSkeleton, reconstruct_darcy_moments_3d
from pymhm.core.contracts import HybridSolution
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.postprocessing.solutions import Darcy3DSolution

ROOT = case_workspace()
OUTPUT = ROOT / "examples/results/reconstruction3d"


def run() -> None:
    """Reconstruct new moments without assembling or resolving the primal MHM system.

    Only the archived primal fields and multiplier are required. Retained
    amplitudes and the original algebraic residual are not reconstructed;
    the restored container therefore has an empty coarse tuple and NaN
    residual, neither of which enters canonical RT moment evaluation.
    """
    record = json.loads((OUTPUT / "resolution.json").read_text())
    row = record["rows"][-1]
    if (row["local_degree"], row["trace_degree"], row["trace_subdivisions"]) != (4, 2, 2):
        raise ValueError("the control requires the archived P4/P2, two-subdivision solution")
    archive_path = OUTPUT / row["archive"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("the original field archive differs from its recorded digest")
    owners = [
        source_file(f"src/pymhm/{name}.py", root=ROOT)
        for name in (
            "_legacy/models/darcy/primal_3d",
            "recovery/moments_3d",
            "fem/hdiv/rt_3d",
            "meshes/mixed",
            "fem/hdiv/family_3d",
            "fem/scalar/tetrahedron",
            "fem/scalar/tetrahedron_topology",
        )
    ] + [Path(__file__), source_file("examples/reconstruction3d_data.py", root=ROOT)]
    hashes = current_source_manifest(source_identity(ROOT, owners), packages=("pymhm", "examples"))
    with np.load(archive_path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    macro = TetraMesh(arrays["macro_points"], arrays["macro_cells"])
    local_meshes = tuple(
        TetraMesh(arrays[f"points_{cell}"], arrays[f"cells_{cell}"])
        for cell in range(len(macro.cells))
    )
    pressure = tuple(arrays[f"pressure_{cell}"] for cell in range(len(macro.cells)))
    hybrid = HybridSolution(arrays["trace"], (), pressure, float("nan"), np.empty(0))
    solution = Darcy3DSolution(
        TriangularSkeleton(macro, 2, degree=2),
        local_meshes,
        pressure,
        hybrid,
        4,
        1.0,
        lambda x: fields(x, True)[2],
        12,
    )
    started = perf_counter()
    reconstruction = reconstruct_darcy_moments_3d(solution, degree=3, quadrature_order=12)
    print("RT3 canonical moments reconstructed from the archived P4 field", flush=True)
    errors = [
        reconstruction.flux_l2_error(lambda x: fields(x, True)[1], order) for order in (10, 12)
    ]
    if abs(errors[1] - errors[0]) > 2e-9:
        raise ArithmeticError("RT3 physical error integration is unresolved")
    print(f"RT3 absolute flux errors, quadrature 10/12: {errors}", flush=True)
    continuous = max(float(np.max(abs(r))) for r in reconstruction.continuous_moment_residuals())
    normal = max(float(np.max(abs(r))) for r in reconstruction.normal_flux_residuals())
    fine_balance = max(float(np.max(abs(r))) for r in reconstruction.fine_conservation_residuals())
    macro_balance = float(np.max(abs(solution.conservation_residuals())))
    arrays["reconstruction_degree"] = np.asarray(3)
    arrays["rt_basis"] = reconstruction.family.coefficients
    for cell, flux in enumerate(reconstruction.flux):
        arrays[f"rt_flux_{cell}"] = flux
    destination = OUTPUT / "resolution-p4-t2-s2-rt3.npz"
    np.savez_compressed(destination, **arrays)
    result = dict(
        operation="canonical RT3 reconstruction of an unchanged archived P4 MHM solution",
        new_primal_solve=False,
        parent_archive=row["archive"],
        parent_archive_sha256=row["archive_sha256"],
        name="resolution-p4-t2-s2-rt3",
        macro_cells=len(macro.cells),
        fine_cells=sum(len(mesh.cells) for mesh in local_meshes),
        local_degree=4,
        trace_degree=2,
        trace_subdivisions=2,
        local_refinement=2,
        reconstruction_degree=3,
        error_orders=[10, 12],
        rt_flux_l2_by_order=errors,
        rt_flux_l2=errors[-1],
        rt_flux_relative=errors[-1] / record["exact_norms"][1],
        rt2_flux_l2=row["rt_flux_l2"],
        rt2_flux_relative=row["rt_flux_relative"],
        raw_flux_l2=row["raw_flux_l2"],
        raw_flux_relative=row["raw_flux_relative"],
        pressure_l2=row["pressure_l2"],
        pressure_relative=row["pressure_relative"],
        exact_norms=record["exact_norms"],
        error_quadrature_change=abs(errors[1] - errors[0]),
        continuous_moment_residual_max=continuous,
        boundary_normal_moment_residual_max=normal,
        fine_cell_balance_residual_max=fine_balance,
        macro_balance_residual_max=macro_balance,
        archive=destination.name,
        archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        basis_sha256=hashlib.sha256(np.ascontiguousarray(arrays["rt_basis"]).tobytes()).hexdigest(),
        basis_digest_format="SHA256 of contiguous float64 C-order array bytes",
        source_sha256=hashes,
        elapsed_seconds=perf_counter() - started,
        interpretation=(
            "RT3 contains the piecewise P3 raw gradient, but canonical reconstruction changes "
            "normal moments through interior averages and the prescribed skeletal density. "
            "Its error need not decrease monotonically with the reconstruction degree. "
            "This P4/P2 local/trace pair does not meet the L09 three-dimensional "
            "error-estimate condition k >= ell + 3; no such estimate is asserted. "
            "Continuous-test equilibrium differs from separate fine-cell source balance."
        ),
    )
    current = current_source_manifest(source_identity(ROOT, owners), packages=("pymhm", "examples"))
    if current != hashes:
        raise RuntimeError("sources changed during the reconstruction control")
    result["source_changed_during_run"] = False
    (OUTPUT / "reconstruction-order.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    """Parse the declared CLI controls and run the original case with its thread limits."""
    with threadpool_limits(1):
        run()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.reconstruction3d_rt_order").main()
