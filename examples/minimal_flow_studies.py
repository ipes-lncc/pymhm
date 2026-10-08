"""Run one bounded analytical flow/GaLS/primal-elasticity case without coefficient archives.

The resulting scalar records support initial three-level convergence studies.
They retain exact executed sources, original equations by physical block and
separate field norms. They do not assert a historical paper reproduction.
"""

from __future__ import annotations

# Preserve direct-file execution alongside the canonical ``python -m examples`` entry point.
if not __package__:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from examples.flow3d_data import Flow3DData
from examples.formulations.application import flow as solve_flow
from examples.formulations.application import flow as solve_flow_3d
from examples.formulations.application import herrmann_elasticity as solve_elasticity_gals_3d
from examples.formulations.application import primal_elasticity as solve_elasticity_3d
from examples.gals3d_data import GaLS3DData
from examples.minimal_flow_originals import (
    capture_sources,
    observe_originals,
    original_diagnostics,
    sources_unchanged,
    write_record,
)
from examples.solve_elasticity3d import ElasticityData3D
from pymhm.fem.scalar.operators import triangle_quadrature
from pymhm.fem.scalar.triangle import tabulate
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.fem.traces.triangle_3d import TriangularSkeleton
from pymhm.meshes.tetrahedron import TetraMesh

ROOT = Path(__file__).resolve().parents[1]

FLOW3D = {
    "stokes-th2": ("stokes", "taylor-hood", 2, 2),
    "brinkman-usfem1": ("brinkman", "usfem", 1, 4),
    "brinkman-usfem2": ("brinkman", "usfem", 2, 2),
    "oseen-p2": ("oseen", "oseen", 2, 2),
}
GALS3D = {"gals-p1": ("gals", 1, 4), "gals-p2": ("gals", 2, 2), "th-p2": ("taylor-hood", 2, 2)}
FLOW2D = {
    "stokes-usfem-l0": ("stokes", 1.0, 0),
    "stokes-usfem-l1": ("stokes", 1.0, 1),
    "stokes-usfem-l2": ("stokes", 1.0, 2),
    "oseen-smooth-l0": ("smooth", 1.0, 0),
    "oseen-smooth-l1": ("smooth", 1.0, 1),
    "oseen-smooth-l2": ("smooth", 1.0, 2),
    "oseen-smooth-nu001": ("smooth", 0.01, 1),
    "oseen-smooth-nu00001": ("smooth", 0.0001, 1),
    "oseen-boundary": ("boundary", 0.01, 1),
    "oseen-internal": ("internal", 0.001, 1),
    "oseen-variable": ("variable", 1.0, 1),
}


def vector_gradient_error(solution: Any, exact_gradient: Any, order: int) -> float:
    """Integrate a 2D vector Jacobian error with component-before-derivative indices."""
    bary, weights = triangle_quadrature(order)
    total = np.longdouble(0)
    for fine, values in zip(solution.local_meshes, solution.values, strict=True):
        dofs, _, _, gradients, _ = tabulate(fine, solution.degree, bary)
        points = np.einsum("qi,tij->tqj", bary, fine.points[fine.cells])
        gradient = np.einsum("tqia,tib->tqba", gradients, values[dofs])
        exact = exact_gradient(points.reshape(-1, 2)).reshape(gradient.shape)
        total += np.sum(
            fine.areas[:, None] * weights * np.sum((gradient - exact) ** 2, axis=(-1, -2)),
            dtype=np.longdouble,
        )
    return float(np.sqrt(total))


def norm_fields(solution: Any, data: Any, family: str, order: int) -> dict[str, float]:
    """Integrate each physical field independently through the unchanged solution API."""
    if family == "elasticity3d":
        return solution.errors(data.displacement, data.stress, order=order)
    displacement = family == "gals3d"
    velocity = data.displacement if displacement else data.velocity
    fields = {
        "displacement_l2" if displacement else "velocity_l2": solution.l2_error(velocity, order),
        "pressure_l2": solution.pressure_l2_error(data.pressure, order),
        "displacement_h1_seminorm" if displacement else "velocity_h1_seminorm": (
            vector_gradient_error(solution, data.gradient, order)
            if family == "flow2d"
            else solution.h1_seminorm_error(data.gradient, order)
        ),
    }
    if displacement:
        fields["stress_l2"] = solution.stress_l2_error(data.stress, order)
        fields["compressibility_l2"] = solution.compressibility_l2(order)
    else:
        fields["divergence_l2"] = (
            solution.divergence_l2() if family == "flow2d" else solution.divergence_l2(order)
        )
    return fields


def acquire(family: str, name: str, n: int, output: Path) -> dict[str, Any]:
    """Run one selected original physical problem with fresh scalar provenance."""
    allowed: Mapping[str, Any] = (
        FLOW3D
        if family == "flow3d"
        else GALS3D
        if family == "gals3d"
        else FLOW2D
        if family == "flow2d"
        else {"anisotropic-p2": 2, "anisotropic-p3": 3}
    )
    levels = (2, 4, 8) if family == "flow2d" else (1, 2, 3)
    if output.exists() or name not in allowed or n not in levels:
        raise ValueError("Fresh output and a declared initial analytical case/level are required")
    output.mkdir(parents=True)
    extra = [
        Path(__file__),
        *(
            ROOT / "examples" / source
            for source in (
                "flow3d_data.py",
                "gals3d_data.py",
                "solve_elasticity3d.py",
                "solve_oseen.py",
                "solve_stokes_adaptive.py",
                "field_sampling.py",
                "plot_mesh.py",
            )
        ),
    ]
    hashes = capture_sources(output, extra)
    started = perf_counter()
    data: Any
    solution: Any
    mesh: Any
    with threadpool_limits(1), observe_originals() as observed:
        if family == "flow3d":
            kind, formulation, degree, refinement = FLOW3D[name]
            data = Flow3DData(kind)
            assembly, orders = 6, (7, 8)
            mesh = TetraMesh.unit_cube(n)
            solution = solve_flow_3d(
                mesh,
                skeleton=TriangularSkeleton(mesh, degree=1),
                formulation=formulation,
                degree=degree,
                local_refinement=refinement,
                quadrature_order=assembly,
                backend="serial",
                workers=1,
                **data.options(),
            )
            vector = solution.velocity
            convention = (
                "Raw grad-grad/Oseen pseudostress multiplier; physical zero-mean pressure; "
                "velocity divergence is a separate L2 diagnostic"
            )
        elif family == "gals3d":
            formulation, degree, refinement = GALS3D[name]
            data = GaLS3DData(1e8)
            assembly, orders = 7, (8, 9)
            solution = solve_elasticity_gals_3d(
                TetraMesh.unit_cube(n),
                formulation=formulation,
                degree=degree,
                local_refinement=refinement,
                lame_lambda=1e8,
                lame_mu=data.shear,
                lame_mu_gradient=data.shear_gradient,
                shear_bounds=data.shear_bounds,
                source=data.source,
                dirichlet=data.displacement,
                quadrature_order=assembly,
                backend="serial",
                workers=1,
            )
            vector = solution.displacement
            convention = (
                "Physical Herrmann pressure with integrated finite compressibility; "
                "lambda=1e8, variable shear; raw symmetric Cauchy stress"
            )
        elif family == "elasticity3d":
            degree = int(allowed[name])
            refinement, formulation = (2 if degree == 2 else 1), "primal-anisotropic"
            data = ElasticityData3D(True)
            assembly = 12 if n == 1 else 10 if n == 2 else 9
            orders = (16, 18) if n == 1 else (14, 16) if n == 2 else (11, 13)
            solution = solve_elasticity_3d(
                TetraMesh.unit_cube(n),
                degree=degree,
                local_refinement=refinement,
                constitutive=data.constitutive,
                source=data.source,
                dirichlet=data.displacement,
                quadrature_order=assembly,
                backend="serial",
                workers=1,
            )
            vector = solution.values
            convention = (
                "Anisotropic Kelvin stiffness (1+x+2y+3z)*SPD; raw symmetric stress; "
                "six retained physical rigid modes"
            )
        else:
            sys.path.insert(0, str(ROOT / "examples"))
            from examples.solve_oseen import OseenData, crisscross
            from examples.solve_stokes_adaptive import StokesData

            kind, viscosity, trace_degree = FLOW2D[name]
            data = StokesData() if kind == "stokes" else OseenData(kind, viscosity)
            degree, refinement = 3, 1
            formulation = "usfem" if kind == "stokes" else "oseen"
            if kind in ("internal", "boundary"):
                assembly, orders = (
                    (80, (96, 112)) if n == 2 else (60, (72, 88)) if n == 4 else (40, (48, 56))
                )
            else:
                assembly, orders = (10 if kind == "stokes" else 16), (16, 18)
            mesh = crisscross(n)
            skeleton = SkeletonSpace(
                mesh, tuple(FaceSpace.uniform(trace_degree) for _ in mesh.faces), 2
            )
            solution = solve_flow(
                mesh,
                skeleton=skeleton,
                formulation=formulation,
                degree=degree,
                local_refinement=refinement,
                quadrature_order=assembly,
                **data.options(),
            )
            vector = solution.values
            convention = (
                "Physical zero-mean pressure; weak full manufactured velocity; "
                "stabilized grad-grad/Oseen pseudo-traction, not physical Darcy flux"
            )
        local_blocks = []
        for cell, values in enumerate(vector):
            blocks = {
                "momentum" if family != "elasticity3d" else "displacement_balance": values.size
            }
            if family != "elasticity3d":
                blocks["continuity" if family != "gals3d" else "compressibility"] = (
                    solution.pressure[cell].size
                )
            local_blocks.append(blocks)
        original = original_diagnostics(solution, observed, local_blocks)
        norms = {str(q): norm_fields(solution, data, family, q) for q in orders}
    lower, upper = norms.values()
    # Field errors drive quadrature acceptance. Near-zero divergence/compressibility
    # remain independently reported diagnostics, whose relative ratios lack scale.
    controlled = [key for key in upper if key not in ("divergence_l2", "compressibility_l2")]
    changes = {
        key: abs(lower[key] - upper[key]) / max(abs(upper[key]), np.finfo(float).tiny)
        for key in controlled
    }
    accepted = bool(original["accepted"] and max(changes.values(), default=0) <= 1e-8)
    if not sources_unchanged(hashes):
        raise ValueError("A numerical source changed during execution")
    origins = {}
    for module_name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if (
                path.is_relative_to(ROOT)
                and path.suffix == ".py"
                and path.relative_to(ROOT).parts[0] in ("src", "examples")
            ):
                origins[module_name] = path.relative_to(ROOT).as_posix()
    if any(origin not in hashes for origin in origins.values()):
        raise ValueError("An executed numerical module was not captured")
    record = {
        "schema": "pymhm-initial-analytical-scalar-record-v1",
        "family": family,
        "case": name,
        "resolution": n,
        "acquisition_uuid": str(uuid4()),
        "source_sha256": hashes,
        "degree": degree,
        "formulation": formulation,
        "local_refinement": refinement,
        "macro_cells": len(solution.local_meshes),
        "fine_cells": sum(len(mesh.cells) for mesh in solution.local_meshes),
        "assembly_order": assembly,
        "norm_orders": list(orders),
        "divergence_integration_order": 4 if family == "flow2d" else list(orders),
        "norms_by_order": norms,
        "terminal_norms": upper,
        "physical_error_quadrature_relative_changes": changes,
        "original_equations": original,
        "accepted": accepted,
        "conventions": convention,
        "elapsed_seconds": perf_counter() - started,
        "workers": 1,
        "native_threads": 1,
        "coefficient_vectors_persisted": False,
        "field_replay_claimed": False,
        "historical_literature_reproduction": False,
        "uniform_inf_sup_proven": False,
        "independent_native_whole_case_verified": False,
        "source_changed": False,
        "runtime_module_origins": origins,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "native_libraries": threadpool_info(),
    }
    write_record(output / "record.json", record)
    print(
        json.dumps(
            {
                "family": family,
                "case": name,
                "resolution": n,
                "accepted": accepted,
                "norms": upper,
                "original": original["full_uncondensed_relative_to_physical_rhs"],
                "elapsed_seconds": record["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    if not accepted:
        raise ArithmeticError(
            "Original equations or physical error quadrature failed; record preserved"
        )
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=("flow3d", "gals3d", "flow2d", "elasticity3d"), required=True
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.family, args.case, args.resolution, args.output)
