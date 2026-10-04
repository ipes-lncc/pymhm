"""Measure RT3 reconstruction and its admissible estimate on an unchanged archived P5 field."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from examples.reconstruction3d_data import fields
from pymhm import TetraMesh, TriangularSkeleton
from pymhm._legacy.models.darcy.primal_3d import Darcy3DSolution
from pymhm.core.contracts import HybridSolution
from pymhm.estimators.darcy_3d import estimate_darcy_error_3d
from pymhm.io.provenance import current_source_manifest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/tetra-pk"


def run() -> None:
    """Reuse the primal coefficients and trace; only canonical moments and their estimate change."""
    previous = json.loads((OUTPUT / "fixed.json").read_text())
    row = previous["rows"][-1]
    path = OUTPUT / row["archive"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["archive_sha256"]:
        raise ValueError("the parent archive differs from its recorded digest")
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if (int(arrays["local_degree"]), row["trace_degree"], row["trace_subdivisions"]) != (5, 2, 2):
        raise ValueError("the reconstruction control requires the P5/P2, four-subface archive")
    owners = [
        ROOT / f"src/pymhm/{name}.py"
        for name in (
            "_legacy/models/darcy/primal_3d",
            "fem/scalar/tetrahedron",
            "fem/scalar/tetrahedron_topology",
            "recovery/moments_3d",
            "estimators/darcy_3d",
            "fem/conditions",
            "fem/hdiv/rt_3d",
            "fem/hdiv/family_3d",
            "meshes/mixed",
            "linalg/linear",
        )
    ] + [Path(__file__), ROOT / "examples/reconstruction3d_data.py"]
    hashes = current_source_manifest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    )
    snapshot = ROOT / "build/results/tetra-pk/acquisition-sources"
    snapshot.mkdir(parents=True, exist_ok=True)
    for owner in owners:
        destination = snapshot / f"{hashes[str(owner.relative_to(ROOT))]}-{owner.name}"
        destination.write_bytes(owner.read_bytes())
    mesh = TetraMesh(arrays["macro_points"], arrays["macro_cells"])
    fine = tuple(
        TetraMesh(arrays[f"points_{i}"], arrays[f"cells_{i}"]) for i in range(len(mesh.cells))
    )
    pressure = tuple(arrays[f"pressure_{i}"] for i in range(len(mesh.cells)))
    hybrid = HybridSolution(
        arrays["trace"], (), pressure, row["global_relative_residual"], np.empty(0)
    )
    solution = Darcy3DSolution(
        TriangularSkeleton(mesh, 2, degree=2),
        fine,
        pressure,
        hybrid,
        5,
        1.0,
        lambda x: fields(x, True)[2],
        12,
    )
    started = perf_counter()
    estimate = estimate_darcy_error_3d(solution, degree=3, quadrature_order=12)
    recovered = estimate.reconstruction
    print("Canonical RT3 reconstruction and estimate complete", flush=True)
    errors = [recovered.flux_l2_error(lambda x: fields(x, True)[1], order) for order in (12, 14)]
    if abs(errors[1] - errors[0]) > 2e-9:
        raise ArithmeticError("RT3 flux-error quadrature is unresolved")
    continuous = max(float(np.max(abs(r))) for r in recovered.continuous_moment_residuals())
    normal = max(float(np.max(abs(r))) for r in recovered.normal_flux_residuals())
    if max(continuous, normal) > 1e-9:
        raise ArithmeticError("canonical RT3 moments are unresolved")
    arrays["reconstruction_degree"] = np.asarray(3)
    arrays["rt_basis"] = recovered.family.coefficients
    arrays["local_squared"] = estimate.local_squared
    for cell, flux in enumerate(recovered.flux):
        arrays[f"rt_flux_{cell}"] = flux
    destination = OUTPUT / "fixed-s2-rt3.npz"
    np.savez_compressed(destination, **arrays)
    result = dict(row)
    result.update(
        name="fixed-s2-rt3",
        reconstruction_degree=3,
        estimator_condition="k >= ell + d: 5 = 2 + 3; ell <= m <= k: 2 <= 3 <= 5",
        new_primal_solve=False,
        parent_archive=path.name,
        parent_archive_sha256=row["archive_sha256"],
        source_sha256=hashes,
        rt_flux_l2=errors[-1],
        rt_flux_relative=errors[-1] / previous["exact_norms"][1],
        rt_flux_l2_by_order=errors,
        error_quadrature_change=abs(errors[1] - errors[0]),
        indicator=estimate.total,
        effectivity=estimate.total / row["energy_error"],
        continuous_equilibrium=continuous,
        normal_moments=normal,
        fine_balance=max(float(np.max(abs(r))) for r in recovered.fine_conservation_residuals()),
        terms={
            key: float(np.linalg.norm(getattr(estimate, key)))
            for key in (
                "flux_defect",
                "nonconformity",
                "divergence_defect",
                "oscillation",
            )
        },
        archive=destination.name,
        archive_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        basis_sha256=hashlib.sha256(np.ascontiguousarray(arrays["rt_basis"]).tobytes()).hexdigest(),
        elapsed_seconds=perf_counter() - started,
    )
    result.pop("errors_by_order")
    if hashes != current_source_manifest(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in owners}
    ):
        raise RuntimeError("reconstruction sources changed during acquisition")
    result["source_changed_during_run"] = False
    (OUTPUT / "reconstruction-order.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        run()
