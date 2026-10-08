"""Acquire three-dimensional nearly incompressible GaLS and Taylor-Hood elasticity."""

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

from examples.campaign_checkpoint import require_sources, verify_checkpoint
from examples.formulations.application import herrmann_elasticity as solve_elasticity_gals_3d
from examples.formulations.application import primal_elasticity as solve_elasticity_3d
from pymhm.io.provenance import current_source_manifest
from pymhm.io.workspace import case_workspace, source_file, source_identity
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.postprocessing.solutions import GaLS3DSolution

if __package__:
    from .gals3d_data import GaLS3DData
else:
    from examples.gals3d_data import GaLS3DData

ROOT = case_workspace()
CASES = (("gals-p1", "gals", 1, 4), ("gals-p2", "gals", 2, 2), ("th-p2", "taylor-hood", 2, 2))


def snapshot() -> dict[str, str]:
    """Capture the exact operators and original analytic-data source bytes."""
    paths = [Path(__file__), source_file("examples/gals3d_data.py", root=ROOT)]
    paths.extend(
        source_file(f"src/pymhm/{name}", root=ROOT)
        for name in (
            "_legacy/models/elasticity/mixed_pressure_3d.py",
            "_legacy/models/elasticity/pressure_forms_3d.py",
            "_legacy/models/elasticity/mixed_pressure.py",
            "_legacy/models/elasticity/primal_3d.py",
            "_legacy/models/darcy/primal_3d.py",
            "fem/scalar/tetrahedron.py",
            "fem/scalar/tetrahedron_topology.py",
            "core/contracts.py",
            "linalg/linear.py",
            "execution/cpu.py",
        )
    )
    return current_source_manifest(source_identity(ROOT, paths), packages=("pymhm", "examples"))


def norms(solution: GaLS3DSolution, data: GaLS3DData, order: int) -> dict[str, float]:
    """Integrate all physical errors with independent positive volume quadrature."""
    return dict(
        displacement_l2=solution.l2_error(data.displacement, order),
        pressure_l2=solution.pressure_l2_error(data.pressure, order),
        gradient_l2=solution.h1_seminorm_error(data.gradient, order),
        stress_l2=solution.stress_l2_error(data.stress, order),
        compressibility_l2=solution.compressibility_l2(order),
    )


def resume_report(path: Path, hashes: dict[str, str], *, primal_precision: str = "double") -> dict:
    """Validate accepted checkpoints and retain the exact source signature of each row."""
    report = json.loads(path.read_text())
    require_sources(report["source_sha256"], hashes)
    cases = {
        name: (formulation, degree, refinement) for name, formulation, degree, refinement in CASES
    }
    for row in report["rows"]:
        formulation, degree, refinement = cases[row["case"]]
        verify_checkpoint(
            row,
            dict(
                formulation=formulation,
                degree=degree,
                local_refinement=refinement,
                face_degree=1,
                face_subdivisions=1,
                assembly_quadrature_order=7,
                error_quadrature_orders=[8, 9],
                shear="1+x/4+z/8",
            ),
            directory=path.parent,
            archive_key="fields",
            digest_key="fields_sha256",
            archive_required=True,
            metrics=(
                "displacement_l2",
                "pressure_l2",
                "gradient_l2",
                "stress_l2",
                "compressibility_l2",
                "backward_residual",
            ),
        )
    for row in report["primal_control"]:
        verify_checkpoint(
            row,
            dict(
                degree=2,
                local_refinement=2,
                macro_subdivisions=2,
                local_refinement_precision=primal_precision,
                assembly_quadrature_order=7,
                error_quadrature_order=8,
            ),
            directory=path.parent,
            metrics=("backward_residual", "displacement_l2", "stress_l2"),
        )
    sources = report.setdefault("acquisition_sources", [report["source_sha256"]])
    for row in report["rows"] + report["primal_control"]:
        row.setdefault("acquisition_source_index", 0)
    if hashes not in sources:
        sources.append(hashes)
    report["source_sha256"] = hashes
    return report


def main() -> None:
    """Run five mesh levels, a seven-value lambda sweep and a finite-lambda primal control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results/gals3d")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--primal-only", action="store_true")
    parser.add_argument(
        "--primal-refinement-precision", choices=("double", "extended"), default="double"
    )
    parser.add_argument("--resume", action="store_true", help="Validate and reuse completed fields")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = snapshot()
    report = dict(
        timestamp_utc=datetime.now(UTC).isoformat(),
        source_sha256=hashes,
        python=platform.python_version(),
        numpy=np.__version__,
        native_threads=1,
        operator="-div(2*mu*sym(grad(u))-p*I)=f; div(u)+p/lambda=0",
        domain="unit cube; Freudenthal tetrahedral macrocells",
        boundary=(
            "exact displacement; integrated finite compressibility "
            "or zero pressure mean at infinity"
        ),
        evidence=(
            "original analytic 3D verification of L19 dimensional formulation; "
            "not a historical figure reproduction"
        ),
        rows=[],
        primal_control=[],
        acquisition_sources=[hashes],
    )
    if args.resume:
        report = resume_report(
            args.output / "campaign.json", hashes, primal_precision=args.primal_refinement_precision
        )
    acquisition = report["acquisition_sources"].index(hashes)

    def save() -> None:
        """Checkpoint only completed cases with unchanged acquisition sources."""
        if snapshot() != hashes:
            raise ValueError("acquisition source changed during execution")
        report["source_changed_during_run"] = False
        (args.output / "campaign.json").write_text(json.dumps(report, indent=2) + "\n")

    with threadpool_limits(1):
        for name, formulation, degree, refinement in () if args.primal_only else CASES:
            studies = [("refinement", n, 1e8) for n in args.levels]
            if not args.skip_sweeps:
                studies.extend(
                    ("lambda", 2, lam) for lam in [1.0, 1e2, 1e4, 1e6, 1e8, 1e12, np.inf]
                )
            for study, n, lam in studies:
                lamtext = "infinity" if np.isinf(lam) else f"{lam:g}"
                if any(
                    row["case"] == name
                    and row["study"] == study
                    and row["macro_subdivisions"] == n
                    and row["lame_lambda"] == lamtext
                    for row in report["rows"]
                ):
                    continue
                data = GaLS3DData(lam)
                mesh = TetraMesh.unit_cube(n)
                started = perf_counter()
                solution = solve_elasticity_gals_3d(
                    mesh,
                    formulation=formulation,
                    degree=degree,
                    local_refinement=refinement,
                    lame_lambda=lam,
                    lame_mu=data.shear,
                    lame_mu_gradient=data.shear_gradient,
                    shear_bounds=data.shear_bounds,
                    source=data.source,
                    dirichlet=data.displacement,
                    quadrature_order=7,
                    backend="process" if args.workers > 1 else "serial",
                    workers=args.workers,
                )
                first, checked = norms(solution, data, 8), norms(solution, data, 9)
                discrepancy = max(
                    abs(first[key] - checked[key]) / max(checked[key], 1e-30) for key in first
                )
                if discrepancy > 1e-8:
                    raise ValueError(
                        f"error quadrature not converged: {name}/{n}/{lam}: {discrepancy}"
                    )
                archive = args.output / f"{name}-{study}-n{n}-lambda{lamtext}.npz"
                np.savez_compressed(
                    archive,
                    macro_points=mesh.points,
                    macro_cells=mesh.cells,
                    local_points=np.asarray([fine.points for fine in solution.local_meshes]),
                    local_cells=np.asarray([fine.cells for fine in solution.local_meshes]),
                    displacement=np.asarray(solution.displacement),
                    pressure=np.asarray(solution.pressure),
                    degree=degree,
                    pressure_degree=solution.pressure_degree,
                    lame_lambda=lam,
                    hybrid_trace=solution.hybrid.trace,
                )
                row = dict(
                    case=name,
                    formulation=formulation,
                    study=study,
                    macro_subdivisions=n,
                    degree=degree,
                    pressure_degree=solution.pressure_degree,
                    local_refinement=refinement,
                    face_degree=1,
                    face_subdivisions=1,
                    lame_lambda=lamtext,
                    shear="1+x/4+z/8",
                    macro_tetrahedra=len(mesh.cells),
                    fine_tetrahedra=sum(len(f.cells) for f in solution.local_meshes),
                    fields=archive.name,
                    fields_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    assembly_quadrature_order=7,
                    error_quadrature_orders=[8, 9],
                    acquisition_source_index=acquisition,
                    max_error_quadrature_relative_difference=discrepancy,
                    stabilization_min=min(solution.stabilization),
                    stabilization_max=max(solution.stabilization),
                    backward_residual=solution.hybrid.residual,
                    elapsed_seconds=perf_counter() - started,
                    **checked,
                )
                report["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
        if not args.skip_sweeps:
            for lam in [1.0, 1e2, 1e4, 1e6]:
                if any(row["lame_lambda"] == lam for row in report["primal_control"]):
                    continue
                data = GaLS3DData(lam)
                solution = solve_elasticity_3d(
                    TetraMesh.unit_cube(2),
                    degree=2,
                    local_refinement=2,
                    lame_lambda=lam,
                    lame_mu=data.shear,
                    local_refinement_precision=args.primal_refinement_precision,
                    source=data.source,
                    dirichlet=data.displacement,
                    quadrature_order=7,
                    backend="process" if args.workers > 1 else "serial",
                    workers=args.workers,
                )
                metrics = solution.errors(data.displacement, data.stress, order=8)
                record = dict(
                    lame_lambda=lam,
                    local_refinement_precision=args.primal_refinement_precision,
                    assembly_quadrature_order=7,
                    error_quadrature_order=8,
                    degree=2,
                    local_refinement=2,
                    macro_subdivisions=2,
                    backward_residual=solution.hybrid.residual,
                    acquisition_source_index=acquisition,
                    **metrics,
                )
                report["primal_control"].append(record)
                save()
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.solve_gals3d").main()
