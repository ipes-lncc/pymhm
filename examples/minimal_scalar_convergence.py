"""Acquire initial scalar convergence records without persisted coefficient vectors.

These short studies report separate physical errors and original discrete rows.
They use the existing numerical owners, explicit partitions and analytical data.
Three levels are initial observations, not uniform stability certificates or
matched historical meshes. Coefficient replay is outside this norm-only format.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_limits

from pymhm import FaceSpace, SkeletonSpace, TriangleMesh
from pymhm._legacy.models.transport.solver import solve_transport
from pymhm.core.system import HybridSystem
from pymhm.io.provenance import current_source_manifest
from pymhm.linalg.linear import _accurate_residual

ROOT = Path(__file__).resolve().parents[1]
CASES = ("mh", "mh2m", "tensor-rt", "polygons", "rad-layer", "transport-layer")


def digest(path: Path) -> str:
    """Hash executed source bytes without loading a field archive into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes() -> dict[str, str]:
    """Identify the locked core and actually imported example numerical owners."""
    files = set((ROOT / "src/pymhm").rglob("*.py"))
    files.update(ROOT / name for name in ("pixi.lock", "pixi.toml", "pyproject.toml"))
    files.add(Path(__file__))
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(ROOT / "examples") and path.suffix == ".py":
                files.add(path)
    return current_source_manifest(
        {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)}
    )


def residual_record(defect: Any, scale: Any) -> dict[str, float]:
    """Measure original coefficient rows using an explicitly stated action scale.

    This Euclidean row norm is an algebraic diagnostic, not a physical L2 norm.
    Zero-scale rows require an exactly zero defect; no absolute floor is added.
    """
    values = np.asarray(defect, dtype=np.longdouble)
    norm = float(np.sqrt(np.sum(abs(values) ** 2)))
    denominator = float(np.sqrt(np.sum(abs(np.asarray(scale, dtype=np.longdouble)) ** 2)))
    relative = norm / denominator if denominator else (0.0 if norm == 0 else float("inf"))
    if not np.isfinite(relative) or relative > 1e-10:
        raise ArithmeticError(f"Original rows fail the unchanged 1e-10 gate: {relative}")
    return {"euclidean_norm": norm, "action_scale": denominator, "relative": relative}


def hybrid_original(
    system: HybridSystem, solution: Any, fixed: dict[int, float], boundary: Any
) -> dict:
    """Check original volume/trace rows and separately report free condensed rows."""
    local = []
    trace_action = np.zeros(system.trace_size, dtype=np.longdouble)
    trace_scale = np.zeros_like(trace_action)
    physical_rhs_squared = np.longdouble(0)
    for response, field in zip(system.responses, solution.fields, strict=True):
        problem = response.problem
        traction = problem.coupling.astype(np.longdouble) @ solution.trace[problem.trace_dofs]
        load = np.asarray(problem.load, dtype=np.longdouble)
        defect = _accurate_residual(problem.matrix.tocsr(), load - traction, field)
        local.append(residual_record(defect, abs(load) + abs(traction)))
        action = problem.test_coupling.astype(np.longdouble).T @ field
        np.add.at(trace_action, problem.trace_dofs, action)
        np.add.at(trace_scale, problem.trace_dofs, abs(action))
        physical_rhs_squared += np.sum(load**2)
    applied_boundary = (
        np.zeros(system.trace_size, dtype=np.longdouble)
        if boundary is None
        else np.asarray(boundary, dtype=np.longdouble)
    )
    free_trace = np.setdiff1d(np.arange(system.trace_size), list(fixed))
    trace_defect = (trace_action - applied_boundary)[free_trace]
    trace_rows = residual_record(trace_defect, (trace_scale + abs(applied_boundary))[free_trace])
    physical_rhs_squared += np.sum(applied_boundary[free_trace] ** 2)
    full_norm = np.sqrt(sum(row["euclidean_norm"] ** 2 for row in local) + np.sum(trace_defect**2))
    full = residual_record(np.array([full_norm]), np.array([np.sqrt(physical_rhs_squared)]))
    coefficients = np.concatenate((solution.trace, *solution.coarse))
    free = np.setdiff1d(np.arange(system.matrix.shape[0]), list(fixed))
    defect = _accurate_residual(system.matrix.tocsr(), system.rhs, coefficients)[free]
    scale = np.asarray(system.load_scale)[free]
    return {
        "local_original_rows": local,
        "local_original_max": max(row["relative"] for row in local),
        "original_trace_rows": trace_rows,
        "full_uncondensed_relative_to_physical_rhs": full,
        "free_condensed_rows": residual_record(defect, scale),
        "scope": "Original A*u+B*lambda=f and C.T*u=g; free reduced rows separately",
    }


def _mh2m_original(result: Any) -> dict:
    local, moments = [], []
    weak = np.zeros(len(result.trace), dtype=np.longdouble)
    weak_scale = np.zeros_like(weak)
    original_rhs_squared = np.longdouble(0)
    for data, values, conormal in zip(result.local, result.pressure, result.conormal, strict=True):
        action = data.boundary_coupling.astype(np.longdouble) @ conormal
        load = np.asarray(data.load, dtype=np.longdouble)
        local.append(
            residual_record(
                _accurate_residual(data.stiffness.tocsr(), load + action, values),
                abs(load) + abs(action),
            )
        )
        left = data.boundary_coupling.astype(np.longdouble).T @ values
        right = data.trace_pairing.astype(np.longdouble) @ result.trace[data.trace_dofs]
        moments.append(residual_record(left - right, load))
        gamma = data.trace_pairing.astype(np.longdouble).T @ conormal
        np.add.at(weak, data.trace_dofs, gamma)
        np.add.at(
            weak_scale,
            data.trace_dofs,
            abs(data.trace_pairing.astype(np.longdouble)).T @ abs(conormal),
        )
        original_rhs_squared += np.sum(load**2)
    gamma_rows = residual_record(weak[result.free_dofs], weak_scale[result.free_dofs])
    full_norm = np.sqrt(
        sum(row["euclidean_norm"] ** 2 for row in (*local, *moments))
        + np.sum(weak[result.free_dofs] ** 2)
    )
    full = residual_record(np.array([full_norm]), np.array([np.sqrt(original_rhs_squared)]))
    defect = _accurate_residual(result.matrix.tocsr(), result.rhs, result.trace)[result.free_dofs]
    return {
        "local_original_rows": local,
        "pressure_trace_moments": moments,
        "original_gamma_rows": gamma_rows,
        "full_uncondensed_relative_to_physical_rhs": full,
        "local_original_max": max(row["relative"] for row in local),
        "free_condensed_rows": residual_record(defect, np.asarray(result.rhs)[result.free_dofs]),
        "macro_balance_max_absolute": float(np.max(abs(result.conservation_residuals()))),
        "scope": (
            "A*p-B*conormal=f; Lambda-tested pressure traces; original free Gamma equations. "
            "Homogeneous moment defects are normalized by the physical cell load; "
            "the full original gate remains relative to the physical load at 1e-10."
        ),
    }


def _acquire(case: str, variant: str, n: int) -> tuple[Any, dict, dict]:
    result: Any
    mesh: Any
    if case == "mh":
        from pymhm.methods.robin import solve_mh

        data = importlib.import_module("examples.mh_campaign")
        mesh = TriangleMesh.unit_square(n) if variant == "triangles" else data.l_mesh(n)
        skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
        result = solve_mh(
            mesh,
            source=data.source,
            skeleton=skeleton,
            degree=3,
            local_refinement=2,
            robin_parameter=0.25,
            quadrature_order=12,
            local_refinement_precision="extended",
            refinement_precision="extended",
        )
        norms = {
            str(q): {
                "pressure_l2": result.l2_error(data.exact, q),
                "flux_l2": result.flux_l2_error(data.exact_flux, q),
            }
            for q in (14, 16)
        }
        convention = {
            "problem": "L20 Section 4.1 sine(3*pi*x)*sine(7*pi*y), K=I, homogeneous Dirichlet",
            "spaces": "local P3 / segment P1, r2; Robin parameter .25",
            "multiplier": "Robin multiplier; physical flux is -grad(p)",
            "assembly_order": 12,
        }
    elif case == "mh2m":
        from pymhm.methods.three_field import PressureTraceSpace, solve_mh2m

        data = importlib.import_module("examples.mh2m_campaign")
        mesh = TriangleMesh.unit_square(n)
        k = int(variant)
        result = solve_mh2m(
            mesh,
            source=data.source,
            pressure_trace=PressureTraceSpace.uniform(mesh, k + 1),
            flux_space=SkeletonSpace(mesh, tuple(FaceSpace.uniform(k) for _ in mesh.faces)),
            degree=k + 1,
            local_refinement=2 if k == 1 else 1,
            quadrature_order=8,
        )
        norms = {
            str(q): {
                "pressure_l2": result.l2_error(data.exact, q),
                "gradient_l2": result.gradient_l2_error(data.exact_gradient, q),
            }
            for q in (10, 12)
        }
        convention = {
            "problem": (
                "L21 Section 8.1 quartic pressure x*(x-1)*y*(y-1), K=I, homogeneous Dirichlet"
            ),
            "spaces": f"Gamma P{k + 1}, Lambda P{k}, local P{k + 1}",
            "local_refinement": 2 if k == 1 else 1,
            "multiplier": "conormal=grad(p).n; physical normal flux has opposite sign",
            "assembly_order": 8,
        }
    elif case == "tensor-rt":
        from pymhm._legacy.models.darcy.tensor import solve_darcy_tensor_rt
        from pymhm.meshes.cartesian import CartesianMacroMesh

        data = importlib.import_module("examples.verify_tensor_rt")
        k, enrichment = map(int, variant.split("-"))
        result = solve_darcy_tensor_rt(
            CartesianMacroMesh(n),
            degree=k,
            enrichment=enrichment,
            source=data.source,
            dirichlet=data.pressure,
            local_refinement=2,
            quadrature_order=10,
        )
        norms = {
            str(q): {
                key: float(value)
                for key, value in result.errors(
                    data.pressure, data.flux, data.source, order=q
                ).items()
            }
            for q in (12, 14)
        }
        convention = {
            "problem": "L03 analytical cosine pressure, K=I, exact weak Dirichlet",
            "spaces": (
                f"tensor RT face degree {k}, interior degree {k + enrichment}; DG Q{k + enrichment}"
            ),
            "local_refinement": 2,
            "assembly_order": 10,
            "multiplier": "physical globally oriented Darcy normal flux",
        }
    else:
        from pymhm._legacy.models.geometry import solve_transport_polygons

        polygon_partition = importlib.import_module("examples.polygon_meshes").polygon_partition

        if case == "polygons":
            data = importlib.import_module("examples.verify_polygons")
            mesh = polygon_partition(n, variant)
            skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces))
            result = solve_transport_polygons(
                mesh,
                skeleton=skeleton,
                degree=3,
                local_refinement=1,
                diffusion=0.1,
                velocity=(1.0, 0.0),
                source=data.source,
                dirichlet_enforcement="strong",
                quadrature_order=20,
            )
            norms = {str(q): {"scalar_l2": result.l2_error(data.exact, q)} for q in (22, 24)}
            convention = {
                "problem": (
                    "L12 Section 5.2.1 sin(6*pi*x)*sin(14*pi*y), diffusion .1, "
                    "velocity (1,0), strong homogeneous Dirichlet"
                ),
                "spaces": "local P3 / segment P1, r1; explicit boundary centroid fans",
                "assembly_order": 20,
            }
        elif case == "rad-layer":
            data = importlib.import_module("examples.verify_rad_layer")
            result = data.solve_case(n, 2)
            norms = {
                str(q): data.errors(result.local_meshes, result.values, 3, order=q)
                for q in (24, 28)
            }
            convention = {
                "problem": (
                    "L12 analytical boundary layer, diffusion .01, velocity (1,0), source1; "
                    "vertical strong Dirichlet and horizontal zero diffusive flux"
                ),
                "spaces": "hexagonal explicit mesh; local P3 / segment P1, r2; selective kernel",
                "assembly_order": 8,
            }
        else:
            data = importlib.import_module("examples.transport_campaign")
            mesh = TriangleMesh.unit_square(n)
            result = solve_transport(
                mesh,
                diffusion=0.02,
                velocity=(1.0, 0.0),
                source=1.0,
                degree=3,
                local_refinement=4,
                stabilization="supg",
                diffusive_flux=data.natural_horizontal(mesh),
                dirichlet_enforcement="strong",
                quadrature_order=8,
            )
            norms = {str(q): {"scalar_l2": result.l2_error(data.layer, q)} for q in (20, 24)}
            convention = {
                "problem": (
                    "L11 Section5.1 analytical layer, diffusion .02, velocity (1,0), source1; "
                    "vertical strong Dirichlet and horizontal zero diffusive flux"
                ),
                "spaces": "triangular local P3, r4, SUPG",
                "assembly_order": 8,
            }
        convention["multiplier"] = (
            "RAD half-advection Robin multiplier, "
            "distinct from physical diffusive/conservative flux"
        )
    return result, norms, convention


def run(case: str, variant: str, levels: list[int], output: Path) -> dict:
    """Run a fresh three-level norm-only study with immutable executed source snapshots."""
    if (
        case not in CASES
        or output.exists()
        or len(levels) < 3
        or levels != sorted(set(levels))
        or min(levels) < 1
    ):
        raise ValueError("Select a valid fresh case and at least three increasing positive levels")
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    captures: list[dict[str, Any]] = []
    boundaries: list[Any] = []
    original_solve = HybridSystem.solve
    original_assemble = HybridSystem._assemble_contributions

    def assembled(system: HybridSystem, contributions: Any, boundary_load: Any) -> None:
        original_assemble(system, contributions, boundary_load)
        boundaries.append(None if boundary_load is None else np.asarray(boundary_load).copy())

    def observed(system: HybridSystem, *args: Any, **kwargs: Any) -> Any:
        solution = original_solve(system, *args, **kwargs)
        captures.append(
            hybrid_original(system, solution, kwargs.get("fixed") or {}, boundaries[-1])
        )
        return solution

    record: dict[str, Any] = {
        "schema": "pymhm-initial-scalar-convergence-v1",
        "case": case,
        "variant": variant,
        "acquisition_uuid": str(uuid4()),
        "complete": False,
        "rows": rows,
        "coefficient_vectors_persisted": False,
        "norm_convention": (
            "Separate physical field errors; Euclidean original coefficient-row residuals "
            "are separately scaled algebraic diagnostics"
        ),
        "limitations": (
            "Explicit current meshes; initial rates only, "
            "no matched historical mesh or uniform inf-sup claim"
        ),
    }
    with (
        threadpool_limits(1),
        patch.object(HybridSystem, "solve", observed),
        patch.object(HybridSystem, "_assemble_contributions", assembled),
    ):
        for n in levels:
            started = perf_counter()
            captures.clear()
            boundaries.clear()
            result, norms, conventions = _acquire(case, variant, n)
            originals = captures[-1] if captures else _mh2m_original(result)
            orders = sorted(norms, key=int)
            low, high = (norms[q] for q in orders)
            change = max(
                abs(low[key] - high[key]) / max(high[key], np.finfo(float).tiny) for key in low
            )
            if change > 1e-6:
                raise ArithmeticError(f"Physical error quadrature is unresolved: {change}")
            hashes = source_hashes()
            if rows and hashes != record["source_sha256"]:
                raise RuntimeError("Executed numerical owners changed during the initial study")
            record.update(
                source_sha256=hashes, conventions=conventions, norm_orders=list(map(int, orders))
            )
            row = {
                "level": n,
                "norms": high,
                "quadrature": norms,
                "quadrature_relative_change": change,
                "original_equations": originals,
                "elapsed_seconds": perf_counter() - started,
            }
            rows.append(row)
            (output / "study.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
    for index in range(1, len(rows)):
        a, b = rows[index - 1], rows[index]
        b["observed_rates"] = {
            key: float(np.log(a["norms"][key] / b["norms"][key]) / np.log(b["level"] / a["level"]))
            for key in b["norms"]
            if min(a["norms"][key], b["norms"][key]) > 0
        }
    snapshot = output / "executed-sources"
    for name, expected in record["source_sha256"].items():
        source = ROOT / name
        if digest(source) != expected:
            raise RuntimeError("Executed source changed before completion")
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    record["complete"] = True
    (output / "study.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return record


def main() -> None:
    """Acquire one bounded case or render a completed record without solving again."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--variant", default="triangles")
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.variant, args.levels, args.output)


if __name__ == "__main__":
    main()
