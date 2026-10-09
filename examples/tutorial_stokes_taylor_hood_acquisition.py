"""Qualify smooth MHM with refined local Taylor--Hood Galerkin equations.

The application declares its meshes, P2/P1 local spaces, vector P1 interface,
translation kernel and physical pressure gauge explicitly. Numerical assembly,
condensation, orientation and reconstruction remain in their package owners.
These norm-only records do not claim persisted coefficient-vector replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from examples.core_elasticity_field_archive import ProductionObservation
from examples.formulations.flow import (
    VelocityPressureSpace,
    velocity_pressure_equations,
    velocity_pressure_fields,
)
from examples.minimal_flow_originals import original_diagnostics
from examples.solve_stokes_adaptive import StokesData
from pymhm import (
    Equation,
    ExecutionConfig,
    FaceSpace,
    MeshHierarchy,
    MultiscaleProblem,
    SkeletonSpace,
    TriangleMesh,
    assemble,
    bind_interface,
    bind_problem,
    columns,
)
from pymhm.core.equations import compile_form, compile_local_equations
from pymhm.fem.scalar.operators import boundary_data
from pymhm.fem.scalar.triangle import nodal_space
from pymhm.io.provenance import current_source_manifest, file_digest, workspace_revision
from pymhm.linalg.linear import accurate_residual
from pymhm.postprocessing.solutions import VectorSolution


def source_manifest(root: Path) -> dict[str, str]:
    """Hash the imported application closure and every executed package owner."""
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


def sources_unchanged(root: Path, hashes: dict[str, str]) -> bool:
    """Compare declared source bytes without reinterpreting an archived manifest."""
    return all(file_digest(root / name) == digest for name, digest in hashes.items())


def declared_space(n: int) -> VelocityPressureSpace:
    """Declare SW--NE macrotriangles, r4 continuous P2/P1 and unsplit vector P1."""
    mesh = TriangleMesh.unit_square(n)
    skeleton = SkeletonSpace(mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2)
    return VelocityPressureSpace(mesh, skeleton, 2, 1, 4, 10)


def numerical_rank(matrix: np.ndarray) -> tuple[int, float]:
    """Report finite-dimensional rank with the ordinary SVD roundoff threshold.

    This is an injectivity check, not a quantitative mesh-uniform inf-sup bound.
    """
    singular = linalg.svdvals(matrix)
    threshold = max(matrix.shape) * np.finfo(float).eps * singular[0]
    active = singular[singular > threshold]
    return len(active), float(active[-1] / singular[0])


def local_space_controls(space: VelocityPressureSpace) -> list[dict[str, Any]]:
    """Check the two triangle orientations, translation kernel and trace lifting.

    The restriction of the boundary pairing to zero-mean local velocities must
    have full column rank, the discrete condition (48) of Araya et al. (2017).
    Full-rank divergence rows and exactly two mixed null modes complement this
    check; unscaled ranks alone are not advertised as uniform inf-sup constants.
    """
    rows = []
    data = StokesData()
    for cell in (0, 1):
        equation = velocity_pressure_equations(cell, space=space, source=data.source)
        matrix = equation.a.toarray()
        nv = equation.metadata[1]
        count = len(equation.L)
        moments = np.asarray(equation.moments)[: 2 * nv]
        zero_mean = linalg.null_space(moments.T)
        trace = zero_mean.T @ equation.b[: 2 * nv]
        rank, margin = numerical_rank(matrix)
        divergence_rank, divergence_margin = numerical_rank(matrix[2 * nv :, : 2 * nv])
        trace_rank, trace_margin = numerical_rank(trace)
        kernel = np.asarray(equation.kernel)
        kernel_defect = np.linalg.norm(matrix @ kernel) / np.linalg.norm(matrix)
        accepted = (
            rank == count - 2
            and divergence_rank == count - 2 * nv
            and trace_rank == equation.b.shape[1]
            and kernel_defect <= 1e-10
        )
        rows.append(
            {
                "macrocell": cell,
                "local_unknowns": count,
                "matrix_nullity": count - rank,
                "pressure_rows": count - 2 * nv,
                "divergence_rank": divergence_rank,
                "trace_columns": equation.b.shape[1],
                "zero_mean_velocity_trace_rank": trace_rank,
                "relative_active_singular_margins": {
                    "mixed_operator": margin,
                    "divergence": divergence_margin,
                    "restricted_trace": trace_margin,
                },
                "translation_kernel_relative_defect": float(kernel_defect),
                "accepted": bool(accepted),
            }
        )
    if not all(row["accepted"] for row in rows):
        raise ValueError("The declared local Galerkin spaces fail kernel or lifting checks")
    return rows


def physical_errors(solution: VectorSolution, order: int) -> dict[str, float]:
    """Integrate velocity and physical mean-zero pressure with independent rules."""
    data = StokesData()
    return {
        "velocity_l2": solution.l2_error(data.velocity, order),
        "pressure_l2": solution.pressure_l2_error(data.pressure, order),
    }


def quadrature_control(solution: VectorSolution) -> dict[str, Any]:
    """Require physical field errors to stabilize between Duffy orders 12 and 16."""
    norms = {str(order): physical_errors(solution, order) for order in (12, 16)}
    changes = {
        name: abs(norms["12"][name] - norms["16"][name]) / norms["16"][name] for name in norms["16"]
    }
    if max(changes.values()) > 1e-9:
        raise ValueError("Independent error quadrature has not stabilized")
    return {"orders": [12, 16], "errors": norms, "relative_changes": changes}


def acquire_level(n: int, workers: int) -> dict[str, Any]:
    """Solve one same-space MHM refinement and check every original physical row."""
    started = perf_counter()
    data, space = StokesData(), declared_space(n)
    controls = local_space_controls(space)
    provider = partial(velocity_pressure_equations, space=space, source=data.source)
    boundary, physical = boundary_data(space.skeleton, data.boundary, order=10)
    if physical:
        raise ValueError("This smooth family prescribes velocity on the entire boundary")
    problem = MultiscaleProblem(
        Equation(0, np.r_[-boundary, np.zeros(2 * len(space.mesh.cells))]),
        provider,
        range(len(space.mesh.cells)),
        space.skeleton.size,
        (2,) * len(space.mesh.cells),
    )
    system = assemble(
        problem,
        execution=ExecutionConfig(
            "process" if workers > 1 else "serial", workers=workers, native_threads=1
        ),
    )
    weights = [record[2] for record in system.local_metadata]
    constraints = [system.mean_constraint(weights, 0.0)]
    coefficients = system.solve(constraints=constraints)
    solution = velocity_pressure_fields(system, coefficients, space)
    observed = ProductionObservation(
        system=system,
        applied_boundary=boundary,
        mean_weights=[tuple(weights)],
        mean_values=[0.0],
        constraints=constraints,
    )
    blocks = [
        {"momentum": 2 * row[1], "incompressibility": len(field) - 2 * row[1]}
        for row, field in zip(system.local_metadata, coefficients.fields, strict=True)
    ]
    original = original_diagnostics(solution, observed, blocks)
    quadrature = quadrature_control(solution)
    if not original["accepted"]:
        raise ValueError("The original physical MHM equations exceed their 1e-10 criterion")
    return {
        "macro_resolution": n,
        "H": float(np.sqrt(2) / n),
        "macro_cells": len(space.mesh.cells),
        "fine_cells": sum(len(mesh.cells) for mesh in solution.local_meshes),
        "trace_dofs": space.skeleton.size,
        "local_refinement": 4,
        "trace_degree": 1,
        "trace_subdivisions": 1,
        "formulation": "taylor-hood",
        **quadrature["errors"]["16"],
        "divergence_l2": solution.divergence_l2(),
        "quadrature": quadrature,
        "local_space_controls": controls,
        "original_equations": original,
        "hybrid_residual": coefficients.residual,
        "wall_seconds": perf_counter() - started,
        "accepted": True,
    }


def native_forms(binding: Any) -> tuple[Any, Any, Any]:
    """Derive Stokes load symbolically from the polynomial stream function in UFL."""
    import ufl

    u, p = ufl.TrialFunctions(binding.space)
    v, q = ufl.TestFunctions(binding.space)
    x, y = ufl.SpatialCoordinate(binding.mesh)
    psi = 128 * x**2 * (1 - x) ** 2 * y**2 * (1 - y) ** 2
    exact_u = ufl.as_vector((psi.dx(1), -psi.dx(0)))
    exact_p = 150 * (x - 0.5) * (y - 0.5)
    force = -ufl.div(ufl.grad(exact_u)) + ufl.grad(exact_p)
    dx = ufl.Measure("dx", domain=binding.mesh, metadata={"quadrature_degree": 12})
    return (
        (ufl.inner(ufl.grad(u), ufl.grad(v)) - p * ufl.div(v) - q * ufl.div(u)) * dx,
        (ufl.inner(force, v) * dx),
        q * dx,
    )


def native_element() -> Any:
    """Declare native equispaced P2 vector velocity and P1 scalar pressure."""
    import basix
    import basix.ufl

    return basix.ufl.mixed_element(
        [
            basix.ufl.element(
                "Lagrange",
                "triangle",
                2,
                shape=(2,),
                lagrange_variant=basix.LagrangeVariant.equispaced,
            ),
            basix.ufl.element(
                "Lagrange",
                "triangle",
                1,
                lagrange_variant=basix.LagrangeVariant.equispaced,
            ),
        ]
    )


def native_local_control(n: int = 4) -> dict[str, Any]:
    """Execute independent native volume, load, signed trace and moment assembly.

    Both macrotriangle orientations are checked in the exact nodal map. UFL
    derives the source independently of the polynomial callback data owner.
    """
    import ufl

    from pymhm.backends.spaces import coefficient_map

    space = declared_space(n)
    hierarchy = MeshHierarchy(
        space.mesh, tuple(space.mesh.submesh(c, 4) for c in range(len(space.mesh.cells)))
    )
    bound = bind_problem(
        hierarchy,
        bind_interface(space.skeleton, convention="normal"),
        lambda local: None,
        retained=2,
    )
    rows = []
    for cell in (0, 1):
        portable = compile_local_equations(
            velocity_pressure_equations(cell, space=space, source=StokesData().source)
        ).problem
        with bound.local_context(cell) as local:
            binding = local.native_space(native_element())
            a, load, pressure_mean = native_forms(binding)
            u, _ = ufl.TrialFunctions(binding.space)
            v, _ = ufl.TestFunctions(binding.space)
            dx = ufl.Measure("dx", domain=binding.mesh)
            mapping = np.r_[
                coefficient_map(binding, component=0), coefficient_map(binding, component=1)
            ]
            kernel = np.zeros((binding.size, 2))
            velocity_map = coefficient_map(binding, component=0)
            kernel[velocity_map] = np.tile(np.eye(2), (len(velocity_map) // 2, 1))
            equation = local.equations(
                a=a,
                L=load,
                b=local.trace_pairings(lambda phi, ds, test=v: ufl.inner(phi, test) * ds),
                c=local.trace_pairings(
                    lambda phi, ds, trial=u: -ufl.inner(phi, trial) * ds, axis="rows"
                ),
                kernel=kernel,
                moments=columns(v[0] * dx, v[1] * dx),
            )
            native = compile_local_equations(equation).problem
            pairs = {
                "volume_operator": (
                    native.matrix[mapping][:, mapping].toarray(),
                    portable.matrix.toarray(),
                ),
                "independent_source": (native.load[mapping], portable.load),
                "signed_trace_trial": (native.coupling[mapping], portable.coupling),
                "signed_trace_test": (native.test_coupling[mapping], portable.test_coupling),
                "translation_moments": (native.constraints[mapping], portable.constraints),
                "physical_pressure_mean": (
                    compile_form(pressure_mean)[mapping],
                    velocity_pressure_equations(
                        cell, space=space, source=StokesData().source
                    ).metadata[2],
                ),
            }
            differences = {
                name: float(
                    np.linalg.norm(first - second)
                    / max(np.linalg.norm(second), np.finfo(float).tiny)
                )
                for name, (first, second) in pairs.items()
            }
            rows.append({"macrocell": cell, "relative_differences": differences})
    if max(value for row in rows for value in row["relative_differences"].values()) > 1e-10:
        raise ValueError(f"Native Galerkin block differences exceed 1e-10: {rows}")
    return {"macro_resolution": n, "cells": rows, "criterion": 1e-10, "accepted": True}


def native_reference(n: int) -> dict[str, Any]:
    """Solve independent global native P2/P1 Stokes with strong velocity and mean gauge."""
    from pymhm.backends.spaces import bind_space, coefficient_map

    started = perf_counter()
    mesh = TriangleMesh.unit_square(n)
    binding = bind_space(mesh, native_element())
    try:
        a, load, mean = native_forms(binding)
        matrix, rhs, gauge = compile_form(a), compile_form(load), compile_form(mean)
        u_map, p_map = coefficient_map(binding, component=0), coefficient_map(binding, component=1)
        nodes = nodal_space(mesh, 2)[1]
        boundary = np.flatnonzero(np.any((nodes == 0) | (nodes == 1), axis=1))
        fixed_ids = u_map[(2 * boundary[:, None] + np.arange(2)).ravel()]
        problem = MultiscaleProblem.from_global(
            Equation(matrix, rhs),
            binding.size,
            fixed=dict.fromkeys(map(int, fixed_ids), 0.0),
            constraints=((gauge, 0.0),),
        )
        coefficients = assemble(problem).solve()
        values = coefficients.trace
        skeleton = SkeletonSpace(
            mesh, tuple(FaceSpace.uniform(1) for _ in mesh.faces), components=2
        )
        solution = VectorSolution(
            skeleton, (mesh,), (values[u_map].reshape(-1, 2),), (values[p_map],), coefficients, 2, 1
        )
        quadrature = quadrature_control(solution)
        free = np.ones(binding.size, dtype=bool)
        free[fixed_ids] = False
        defect = accurate_residual(matrix, rhs, values)
        scale = abs(matrix) @ abs(values) + abs(rhs)
        blocks = {}
        for name, ids in (("momentum", u_map[free[u_map]]), ("incompressibility", p_map)):
            blocks[name] = {
                "absolute_row_residual_norm": float(np.linalg.norm(defect[ids])),
                "backward_error": float(np.linalg.norm(defect[ids]) / np.linalg.norm(scale[ids])),
            }
        relative = float(np.linalg.norm(defect[free]) / np.linalg.norm(rhs[free]))
        pressure_mean = float(gauge @ values)
        gauge_defect = abs(pressure_mean) / (abs(gauge) @ abs(values))
        if max(relative, gauge_defect, *(row["backward_error"] for row in blocks.values())) > 1e-10:
            raise ValueError("The classical original equations exceed their 1e-10 criterion")
        return {
            "resolution": n,
            "h": float(np.sqrt(2) / n),
            "cells": len(mesh.cells),
            "unknowns": binding.size,
            "free_unknowns": int(free.sum()),
            **quadrature["errors"]["16"],
            "divergence_l2": solution.divergence_l2(),
            "quadrature": quadrature,
            "pressure_mean": pressure_mean,
            "original_equations": {
                "criterion": 1e-10,
                "accepted": True,
                "full_relative_to_physical_rhs": relative,
                "blocks": blocks,
                "physical_pressure_mean_relative_defect": float(gauge_defect),
            },
            "wall_seconds": perf_counter() - started,
            "accepted": True,
        }
    finally:
        binding.close()


def main() -> None:
    """Acquire one frozen-source campaign without overwriting a numerical record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, default=[16])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--native-control", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("A numerical acquisition record already exists")
    root = Path(__file__).resolve().parents[1]
    hashes = source_manifest(root)
    report = {
        "schema": "pymhm-stokes-taylor-hood-qualification-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "problem": "Araya et al. (2017) polynomial Stokes, nu=1, drag=0",
        "geometry": "unit square with SW-NE diagonal triangles; diameter=sqrt(2)/n",
        "spaces": (
            "continuous global native P2 velocity/P1 pressure"
            if args.reference
            else (
                "continuous local P2 velocity/P1 pressure on r4; "
                "unsplit vector P1 negative pseudotraction"
            )
        ),
        "boundary": "homogeneous exterior velocity; global physical pressure mean zero",
        "norms": "physical broken velocity L2 and physical mean-zero pressure L2",
        "assembly_quadrature": 12 if args.reference else 10,
        "workers": 1 if args.reference else args.workers,
        "native_threads": 1,
        "literature": (
            "Araya et al., CMAME 324 (2017), DOI 10.1016/j.cma.2017.05.027, "
            "section 2.2 and condition (48)"
        ),
        "rate_scope": (
            "refined stable Galerkin family; trace lifting checked numerically; "
            "not the single-element USFEM degree lemma"
        ),
        "source_sha256": hashes,
        "workspace_revision": workspace_revision(root),
        "versions": {
            name: version(name)
            for name in ("numpy", "scipy", "fenics-basix", "fenics-ufl", "fenics-dolfinx")
        },
        "rows": [],
    }
    with threadpool_limits(1):
        if args.native_control:
            report["native_local_control"] = native_local_control()
            print("native local control accepted", flush=True)
        for n in args.levels:
            row = native_reference(n) if args.reference else acquire_level(n, args.workers)
            if not sources_unchanged(root, hashes):
                raise ValueError("Executed sources changed during acquisition")
            report["rows"].append(row)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in row
                        if key
                        in (
                            "macro_resolution",
                            "resolution",
                            "velocity_l2",
                            "pressure_l2",
                            "wall_seconds",
                            "accepted",
                        )
                    }
                ),
                flush=True,
            )
    report["source_unchanged"] = sources_unchanged(root, hashes)
    report["accepted"] = bool(
        report["source_unchanged"] and all(row["accepted"] for row in report["rows"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
