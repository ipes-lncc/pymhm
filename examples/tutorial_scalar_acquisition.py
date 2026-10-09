"""Acquire current scalar tutorial refinements with independent physical errors.

The notebooks declare their mathematical equations explicitly. This acquisition
helper repeats those existing application declarations on larger meshes; local
assembly, condensation, orientation, reconstruction and integration remain in
their shared owners. Norm-only records do not persist coefficient vectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from examples.formulations.application import (
    moment_diffusion,
    petrov_galerkin_diffusion,
    three_field_diffusion,
)
from examples.formulations.mixed_darcy import define_rt_darcy, recover_rt_darcy
from examples.mh2m_campaign import exact as three_field_exact
from examples.mh2m_campaign import exact_gradient as three_field_gradient
from examples.mh2m_campaign import source as three_field_source
from examples.minimal_scalar_convergence import _mh2m_original, hybrid_original, residual_record
from examples.native_extension_data import sine as moment_exact
from examples.native_extension_data import sine_gradient
from examples.solve_darcy_rt import flux, potential, source
from pymhm import ExecutionConfig, FaceSpace, SkeletonSpace, TriangleMesh, assemble
from pymhm.fem.traces.pressure_2d import PressureTraceSpace
from pymhm.io.provenance import current_source_manifest, file_digest, workspace_revision
from pymhm.linalg.linear import accurate_residual


def source_manifest(root: Path) -> dict[str, str]:
    """Fingerprint imported application owners and the complete executed package.

    Only imported example modules enter this closure. Concurrent edits to an
    unrelated tutorial therefore do not invalidate the numerical acquisition.
    Package ownership is conservatively recorded in full.
    """
    entries = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(root / "examples"):
            entries[path.relative_to(root).as_posix()] = file_digest(path)
    for name in ("pixi.lock", "pixi.toml", "pyproject.toml"):
        entries[name] = file_digest(root / name)
    return current_source_manifest(entries)


def moment_source(points: np.ndarray) -> np.ndarray:
    """Apply minus the Laplacian to sin(pi*x) sin(pi*y), independently of assembly."""
    return 2 * np.pi**2 * moment_exact(points)


def moment_flux(points: np.ndarray) -> np.ndarray:
    """Return the physical Darcy flux minus the independently differentiated sine gradient."""
    return -sine_gradient(points)


def physical_errors(solution: Any, method: str, order: int) -> dict[str, float]:
    """Integrate full pressure and physical vector flux using shared field owners."""
    exact, exact_flux = (moment_exact, moment_flux) if method == "mshho" else (potential, flux)
    if method == "mh2m":
        return {
            "pressure_l2": solution.l2_error(three_field_exact, order=order),
            "gradient_l2": solution.gradient_l2_error(three_field_gradient, order=order),
        }
    options = {"enriched": True} if method == "pgmhm" else {}
    return {
        "pressure_l2": solution.l2_error(exact, order=order, **options),
        "flux_l2": solution.flux_l2_error(exact_flux, order=order, **options),
    }


def _mshho_original(solution: Any) -> dict[str, Any]:
    """Check original cell/face moment equations and executed reconstruction moments.

    Norms refer to algebraic moment rows, not physical L2 field norms. The
    source projection and the prescribed exterior face moments retain exactly
    the coordinates of the executed method. Accurate matrix actions delegate
    to the common residual owner.
    """
    weak = np.zeros(solution.skeleton.size, dtype=np.longdouble)
    scale = np.zeros_like(weak)
    local_squared = rhs_squared = np.longdouble(0)
    local_max = reconstruction_max = 0.0
    for cell, (data, interior, field) in enumerate(
        zip(solution.local, solution.cell_moments, solution.pressure, strict=True)
    ):
        ids = solution.skeleton.cell_dofs(cell)
        coordinates = np.r_[interior, solution.face_moments[ids]]
        defect = accurate_residual(sparse.csr_matrix(data.energy), data.load, coordinates)
        action = abs(data.energy) @ abs(coordinates)
        local = residual_record(defect[: data.cell_count], action[: data.cell_count])
        reconstruction = accurate_residual(sparse.csr_matrix(data.moments.T), coordinates, field)
        moment = residual_record(reconstruction, abs(coordinates))
        local_max = max(local_max, local["relative"])
        reconstruction_max = max(reconstruction_max, moment["relative"])
        np.add.at(weak, ids, defect[data.cell_count :])
        np.add.at(scale, ids, action[data.cell_count :])
        local_squared += np.sum(defect[: data.cell_count] ** 2)
        rhs_squared += np.sum(data.load**2)
    fixed = np.concatenate(
        [solution.skeleton.dofs(int(face)) for face in solution.skeleton.mesh.boundary_faces]
    )
    free = np.setdiff1d(np.arange(solution.skeleton.size), fixed)
    face = residual_record(weak[free], scale[free])
    total = residual_record(
        np.array([np.sqrt(local_squared + np.sum(weak[free] ** 2))]),
        np.array([np.sqrt(rhs_squared)]),
    )
    return {
        "cell_equilibrium_relative_max": local_max,
        "face_equilibrium": face,
        "reconstruction_moment_relative_max": reconstruction_max,
        "all_moment_equations_relative_to_projected_source": total,
    }


def _pgmhm_original(solution: Any) -> dict[str, Any]:
    """Check full enriched local nodal equations with the physical boundary load.

    The base local problem tests zero-mean functions. Its enriched physical
    field instead satisfies A*p+B*lambda+extra=f in all nodal rows. The global
    residual retains the declared residual-jump equation of the base method.
    """
    total_squared = rhs_squared = np.longdouble(0)
    maximum = 0.0
    for response, field, extra in zip(
        solution.system.responses,
        solution.enriched_pressure,
        solution.enrichment_loads,
        strict=True,
    ):
        problem = response.problem
        traction = (
            problem.coupling.astype(np.longdouble) @ solution.hybrid.trace[problem.trace_dofs]
        )
        rhs = problem.load.astype(np.longdouble) - traction - extra
        defect = accurate_residual(problem.matrix.tocsr(), rhs, field)
        local = residual_record(defect, abs(problem.load) + abs(traction) + abs(extra))
        maximum = max(maximum, local["relative"])
        total_squared += np.sum(defect**2)
        rhs_squared += np.sum(problem.load.astype(np.longdouble) ** 2)
    total = residual_record(np.array([np.sqrt(total_squared)]), np.array([np.sqrt(rhs_squared)]))
    return {
        "enriched_local_nodal_equation_relative_max": maximum,
        "enriched_local_equations_relative_to_physical_rhs": total,
        "global_jump_equation_raw_relative_residual": solution.hybrid.raw_residual,
    }


def acquire_level(method: str, n: int, workers: int) -> dict[str, Any]:
    """Repeat one declared smooth problem, retaining its original spaces and gauges.

    RT1/P1 uses two fine subdivisions and P1 normal-flux traces. MsHHO uses
    matching P0 cell/face moments, P3/r2 local energy reconstruction and the
    projected source. PGMHM uses P3 single-triangle locals, P1 traces and alpha
    0.1. The compatible MH2M family has P2 Gamma, two P1 Lambda segments per
    macroface and P2/r4 local pressure, hence two fine edges per Lambda segment.
    All cases have identity permeability and homogeneous exterior pressure.
    """
    mesh = TriangleMesh.unit_square(n)
    start = perf_counter()
    parallel = workers > 1 and n >= 16 and method != "mshho"
    execution = {"backend": "process" if parallel else "serial", "workers": workers}
    with threadpool_limits(1):
        if method == "mixed-mhm":
            definition = define_rt_darcy(
                mesh,
                degree=1,
                source=source,
                local_refinement=2,
                quadrature_order=8,
            )
            system = assemble(definition.problem, execution=ExecutionConfig(**execution))
            coefficients = system.solve()
            solution = recover_rt_darcy(definition, system, coefficients)
        elif method == "mshho":
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(0) for _ in mesh.faces))
            solution = moment_diffusion(
                mesh,
                skeleton=skeleton,
                cell_degree=0,
                degree=3,
                local_refinement=2,
                source=moment_source,
                quadrature_order=8,
                reconstruction_precision="extended",
                local_refinement_precision="extended",
            )
        elif method == "pgmhm":
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
            solution = petrov_galerkin_diffusion(
                mesh,
                skeleton=skeleton,
                degree=3,
                local_refinement=1,
                source=source,
                stabilization_parameter=0.1,
                quadrature_order=8,
                **execution,
            )
        elif method == "mh2m":
            gamma = PressureTraceSpace.uniform(mesh, degree=2)
            conormal = SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(1, subdivisions=2) for _ in mesh.faces)
            )
            solution = three_field_diffusion(
                mesh,
                pressure_trace=gamma,
                flux_space=conormal,
                degree=2,
                local_refinement=4,
                source=three_field_source,
                quadrature_order=8,
                **execution,
            )
        else:
            raise ValueError(f"Unsupported scalar tutorial method: {method}")
        solved = perf_counter()
        norms = {str(order): physical_errors(solution, method, order) for order in (10, 12)}
        discrepancy = max(abs(norms["10"][key] / norms["12"][key] - 1) for key in norms["10"])
        if discrepancy > 1e-9:
            raise ArithmeticError(f"Independent error quadratures disagree: {discrepancy}")
        residual = solution.residual if method != "pgmhm" else solution.hybrid.residual
        if residual > 1e-10:
            raise ArithmeticError(f"Condensed algebraic residual exceeds 1e-10: {residual}")
        diagnostics = {"global_compatibility_relative_residual": residual}
        if method == "mixed-mhm":
            fine = max(float(abs(part).max()) for part in solution.fine_equilibrium_residuals())
            normal = max(float(abs(part).max()) for part in solution.normal_flux_residuals())
            if max(fine, normal) > 1e-9:
                raise ArithmeticError("Mixed fine equilibrium or normal continuity failed")
            diagnostics.update(fine_equilibrium_moment_max=fine, normal_flux_moment_max=normal)
            diagnostics["original_equations"] = hybrid_original(
                system,
                coefficients,
                dict(definition.problem.fixed),
                system.global_load[: system.trace_size],
            )
        elif method == "mshho":
            diagnostics["original_equations"] = _mshho_original(solution)
        elif method == "pgmhm":
            balance = float(max(abs(solution.conservation_residuals())))
            if balance > 1e-9:
                raise ArithmeticError("Enriched PGMHM macro conservation failed")
            diagnostics["enriched_macro_balance_max"] = balance
            diagnostics["original_equations"] = _pgmhm_original(solution)
        elif method == "mh2m":
            diagnostics["original_equations"] = _mh2m_original(solution)
    return {
        "n": n,
        "macro_diameter": float(np.sqrt(2) / n),
        "macro_cells": len(mesh.cells),
        **norms["12"],
        "norm_quadratures": norms,
        "norm_quadrature_relative_difference": discrepancy,
        "diagnostics": diagnostics,
        "backend": "process" if parallel else "serial",
        "workers": workers if parallel else 1,
        "assembly_and_solve_seconds": solved - start,
        "total_seconds": perf_counter() - start,
    }


def acquire(method: str, levels: list[int], output: Path, *, workers: int = 16) -> dict[str, Any]:
    """Write an attributed fresh sequence without modifying historical records."""
    if len(levels) < 4 or levels != sorted(set(levels)) or min(levels) < 1:
        raise ValueError(
            "At least four strictly increasing positive macro resolutions are required"
        )
    root = Path(__file__).resolve().parents[1]
    sources = source_manifest(root)
    record: dict[str, Any] = {
        "schema": "pymhm-scalar-tutorial-refinement-v1",
        "method": method,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "workspace_revision": workspace_revision(root),
        "source_sha256": sources,
        "source_changed_during_run": False,
        "python": sys.version,
        "packages": {name: version(name) for name in ("numpy", "scipy", "fenics-basix")},
        "native_threads": 1,
        "problem": (
            "Unit square, identity permeability, homogeneous exterior pressure; "
            "quartic pressure x(x-1)y(y-1) for MH2M, sine pressure otherwise"
        ),
        "source_frequency": 1 if method == "mshho" else (None if method == "mh2m" else 2),
        "assembly_quadrature": 8,
        "error_quadratures": [10, 12],
        "scope": (
            "Current finite local Galerkin realization; smooth macro refinement, "
            "not a paper mesh reproduction"
        ),
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for n in levels:
        row = acquire_level(method, n, workers)
        record["rows"].append(row)
        if any(file_digest(root / name) != digest for name, digest in sources.items()):
            raise RuntimeError("An executed numerical source changed during acquisition")
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"method": method, **row}), flush=True)
    record["record_source_closure_sha256"] = hashlib.sha256(
        json.dumps(sources, sort_keys=True).encode()
    ).hexdigest()
    output.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    """Run one reproducible acquisition with explicit method, levels and CPU budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("mixed-mhm", "mshho", "pgmhm", "mh2m"))
    parser.add_argument("--levels", required=True, nargs="+", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    acquire(options.method, options.levels, options.output, workers=options.workers)


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.tutorial_scalar_acquisition").main()
