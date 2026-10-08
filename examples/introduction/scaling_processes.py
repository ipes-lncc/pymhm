"""Reproducible process scaling measurements and presentation for Darcy.

The notebook retains the public local/global equation workflow. This module
owns timers, norm integration, archives and figures; full campaigns preserve
the original fine grids, trace spaces and inclusive timing conventions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

import examples.introduction.scaling_forms as provider_module
from examples.introduction._scaling_timer import STARTED
from examples.introduction.scaling_forms import (
    DEFAULT_DATA,
    EPSILON,
    PeriodicDarcyData,
    define_ufl_local_equations,
    exact_gradient,
    exact_pressure,
    face_length_snapshot,
    interface_template,
    permeability,
    refined_interface_template,
    source,
    translated_interface,
)
from pymhm import MeshHierarchy, bind_interface, bind_problem
from pymhm.core.contracts import LocalResponse
from pymhm.core.equations import Equation, compile_local_equations
from pymhm.core.multiscale import assemble
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.reference import orthogonal_polynomial_tabulation, simplex_lagrange_basis
from pymhm.fem.scalar.quadrilateral import (
    qk_basis,
    qk_space,
    quadrilateral_operators,
    quadrilateral_quadrature,
    quadrilateral_trace_coupling,
)
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest
from pymhm.linalg.linear import check_linear_solution, solve_linear
from pymhm.meshes.cartesian import CartesianMacroMesh

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "build/introduction/darcy_process_scalability"
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCE_NOTEBOOK = Path(
    os.environ.get(
        "PYMHM_NOTEBOOK_SOURCE", ROOT / "notebooks/introduction/darcy_process_scalability.ipynb"
    )
)
# Import setup is recorded once; no numerical factors are reused.
PARENT_IMPORT_SECONDS = time.perf_counter() - STARTED

SpawnLocalProvider = provider_module.LocalProvider
provider_source_sha256 = hashlib.sha256(Path(provider_module.__file__).read_bytes()).hexdigest()
module_sha256 = provider_source_sha256
module_name = provider_module.__name__
module_path = Path(provider_module.__file__)
EXPORT_IMPORT_SECONDS = 0.0

WORKLOAD = {
    "fine_n": 1000,
    "trace_segments": 8,
    "process_counts": (1, 4, 8, 16),
    "weak_fine_n": 200,
    "weak_trace_segments": 4,
    "crossover_fine_n": 500,
    "crossover_trace_segments": 4,
    "crossover_process_counts": (8, 16),
    "reference_fine_sizes": (200, 500, 1000),
    "selection_reason": (
        "Resolve the fixed medium; compare startup cost, larger local"
        " work and fixed work per process"
    ),
}
REPETITIONS = 3
RANDOM_SEED = 20261004

if WORKLOAD["fine_n"] % 10 or WORKLOAD["weak_fine_n"] % 10:
    raise ValueError("Fine-grid sizes must be multiples of the ten macros per unit coordinate")
PROCESS_COUNTS = tuple(WORKLOAD["process_counts"])
if not PROCESS_COUNTS or any(
    isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in PROCESS_COUNTS
):
    raise ValueError("Every process count must be a positive integer")
if tuple(sorted(set(PROCESS_COUNTS))) != PROCESS_COUNTS or PROCESS_COUNTS[0] != 1:
    raise ValueError("Use increasing distinct positive process counts, including process1")
REFERENCE_FINE_SIZES = tuple(WORKLOAD["reference_fine_sizes"])
if len(REFERENCE_FINE_SIZES) < 3 or any(
    b <= a for a, b in zip(REFERENCE_FINE_SIZES, REFERENCE_FINE_SIZES[1:], strict=False)
):
    raise ValueError("Use at least three increasing conforming refinement meshes")
if any(REFERENCE_FINE_SIZES[-1] % n for n in (*REFERENCE_FINE_SIZES, WORKLOAD["fine_n"])):
    raise ValueError("The common integration partition must resolve every fine-grid breakpoint")
assert WORKLOAD["selection_reason"]


def verify_forms(data: PeriodicDarcyData = DEFAULT_DATA) -> dict[str, Any]:
    """Execute volume, mean and oriented trace equivalence on four macrocells."""
    demo_macro = CartesianMacroMesh(2, 2)

    demo_skeleton = SkeletonSpace(
        demo_macro, tuple(FaceSpace.uniform(1, 4, continuous=True) for _ in demo_macro.faces)
    )

    demo_problem = bind_problem(
        MeshHierarchy(
            demo_macro, tuple(demo_macro.submesh(cell, 20) for cell in range(len(demo_macro.cells)))
        ),
        bind_interface(demo_skeleton, convention="normal"),
        lambda local: define_ufl_local_equations(local, data),
        retained=1,
    )

    native_demonstrations = {}

    for cell in range(len(demo_macro.cells)):
        local = demo_problem.local_context(cell)
        try:
            equations = define_ufl_local_equations(local, data)
            compiled = compile_local_equations(equations)
            native_demonstrations[cell] = (
                compiled.problem,
                equations.metadata["native_to_cartesian"],
            )
        finally:
            local.close()

    print(
        "Executed UFL stiffness, source, mean moments and signed inte"
        "rface forms on four macroelements."
    )

    from pymhm.fem.scalar.quadrilateral import (
        quadrilateral_operators,
        quadrilateral_trace_coupling,
    )

    ufl_block_equivalence = {}

    # Reconcile native node order before comparing the same executed weak forms.
    for cell, (native_problem, native_order) in native_demonstrations.items():
        fine = demo_macro.submesh(cell, 20)
        inverse_order = np.argsort(native_order)
        A, mass, load = quadrilateral_operators(
            fine, 1, permeability=data.permeability, source=data.source, order=4
        )
        B = quadrilateral_trace_coupling(demo_macro, cell, fine, demo_skeleton, 1)
        np.testing.assert_allclose(
            native_problem.matrix[inverse_order][:, inverse_order].toarray(),
            A.toarray(),
            rtol=5e-12,
            atol=5e-13,
        )
        np.testing.assert_allclose(native_problem.load[inverse_order], load, rtol=5e-12, atol=5e-13)
        np.testing.assert_allclose(
            native_problem.coupling[inverse_order], B, rtol=5e-12, atol=5e-13
        )
        np.testing.assert_allclose(
            native_problem.constraints[inverse_order],
            mass @ np.ones((len(load), 1)),
            rtol=5e-12,
            atol=5e-13,
        )
        difference = native_problem.matrix[inverse_order][:, inverse_order] - A
        ufl_block_equivalence[cell] = {
            "stiffness_max_absolute": float(np.max(np.abs(difference.data), initial=0)),
            "source_max_absolute": float(np.max(np.abs(native_problem.load[inverse_order] - load))),
            "trace_max_absolute": float(np.max(np.abs(native_problem.coupling[inverse_order] - B))),
            "mean_max_absolute": float(
                np.max(
                    np.abs(
                        native_problem.constraints[inverse_order] - mass @ np.ones((len(load), 1))
                    )
                )
            ),
        }

    # Verify every normal sign and boundary orientation against direct face integration.
    demo_trace_space = FaceSpace.uniform(1, 4, continuous=True)
    demo_lengths = face_length_snapshot(demo_macro)
    demo_template, demo_ends = interface_template(demo_macro.spacing, demo_trace_space)
    for cell in range(len(demo_macro.cells)):
        fine = demo_macro.submesh(cell, 20)
        direct = quadrilateral_trace_coupling(demo_macro, cell, fine, demo_skeleton, 1)
        transported = translated_interface(
            demo_template, demo_ends, demo_macro, cell, demo_lengths, demo_trace_space
        )
        np.testing.assert_allclose(transported, direct, rtol=5e-12, atol=5e-13)

    return ufl_block_equivalence


def available_cpus() -> tuple[int, dict[str, Any]]:
    """Bound the worker count by CPU affinity and Linux container quotas when present."""
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = len(affinity) if affinity is not None else (os.cpu_count() or 1)
    quota_limits = []
    quota_controls = []
    groups = {}
    if Path("/proc/self/cgroup").exists():
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, group = line.split(":", 2)
            groups[controllers] = group
    candidates = []
    if Path("/proc/self/mountinfo").exists():
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount_fields, fs_fields = left.split(), right.split()
            mount = Path(mount_fields[4])
            if fs_fields[0] == "cgroup2" and "" in groups:
                candidates.append((mount, groups[""], "cpu.max"))
            elif fs_fields[0] == "cgroup" and "cpu" in fs_fields[2].split(","):
                for controllers, group in groups.items():
                    if "cpu" in controllers.split(","):
                        candidates.append((mount, group, "cpu.cfs_quota_us"))
    # Inspect the process group and its ancestors: a parent may impose a quota.
    for mount, group, filename in candidates:
        directory = mount / group.lstrip("/")
        while directory.is_relative_to(mount):
            control = directory / filename
            if control.exists():
                if filename == "cpu.max":
                    value, period = control.read_text().strip().split()
                    limit = None if value == "max" else int(value) / int(period)
                else:
                    value = int(control.read_text().strip())
                    period = int(control.with_name("cpu.cfs_period_us").read_text())
                    limit = value / period if value > 0 else None
                quota_controls.append({"control": str(control), "limit": limit})
                if limit is not None:
                    quota_limits.append(limit)
            if directory == mount:
                break
            directory = directory.parent
    quota = min(quota_limits) if quota_limits else None
    if quota is not None:
        count = min(count, max(1, math.floor(quota)))
    return count, {
        "logical_cpus": os.cpu_count(),
        "affinity": affinity,
        "cpu_quota": quota,
        "quota_controls": quota_controls,
    }


def resource_snapshot() -> dict[str, Any]:
    """Capture actual coordinator limits and cumulative CPU/child usage outside timers."""
    record = {
        "pid": os.getpid(),
        "native_libraries": threadpool_info(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "native_thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    try:
        import resource
    except ImportError:
        return record
    for label, selector in (("self", resource.RUSAGE_SELF), ("children", resource.RUSAGE_CHILDREN)):
        usage = resource.getrusage(selector)
        record[label] = {
            "user_seconds": usage.ru_utime,
            "system_seconds": usage.ru_stime,
            "maximum_resident_set_size": usage.ru_maxrss,
            "resident_set_unit": "bytes" if sys.platform == "darwin" else "KiB",
            "voluntary_context_switches": usage.ru_nvcsw,
            "involuntary_context_switches": usage.ru_nivcsw,
        }
    # Children maximum RSS is the largest individual child, not a simultaneous sum.
    return record


def run_mhm(
    backend: str,
    workers: int,
    fine_n: int,
    trace_segments: int,
    length: int = 1,
) -> tuple[dict[str, Any], Any]:
    """Time fresh declared forms, full worker payloads and complete reconstruction."""
    refinement = fine_n // 10
    if fine_n % 10 or length < 1:
        raise ValueError("Use integer domain lengths and ten macros per unit coordinate")
    resources_before = resource_snapshot()
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        macro = CartesianMacroMesh(10 * length, 10, (0, float(length), 0, 1))
        face_space = FaceSpace.uniform(1, trace_segments, continuous=True)
        skeleton = SkeletonSpace(macro, tuple(face_space for _ in macro.faces))
        lengths = face_length_snapshot(macro)
        template, ends = refined_interface_template(macro.spacing, refinement, face_space)
        provider = SpawnLocalProvider(
            macro, skeleton, template, ends, lengths, face_space, refinement=refinement
        )
        problem = bind_problem(
            MeshHierarchy(macro, provider.local_mesh),
            bind_interface(skeleton, convention="normal"),
            provider,
            global_equation=Equation(0, 0),
            retained=1,
        )
        execution = ExecutionConfig(
            backend=backend,
            workers=workers,
            native_threads=1,
            batch_size=workers,
            pipeline=True,
        )
        prepared = time.perf_counter()
        system = assemble(problem, execution=execution)
        assembled = time.perf_counter()
        solution = system.solve()
        completed = time.perf_counter()
    return {
        "setup": prepared - started,
        "assembly": assembled - prepared,
        "solve_reconstruct": completed - assembled,
        "total": completed - started,
        "cold_total": PARENT_IMPORT_SECONDS + EXPORT_IMPORT_SECONDS + completed - started,
        "resources_before": resources_before,
        "resources_after": resource_snapshot(),
        "fine_grid": [fine_n * length, fine_n],
        "fine_cells": fine_n**2 * length,
        "macro_grid": [10 * length, 10],
        "local_refinement": refinement,
        "trace_segments": trace_segments,
        "global_algebraic_size": system.matrix.shape[0],
    }, (macro, system, solution)


def run_classical(
    fine_n: int,
    solver: str,
    length: int = 1,
    data: PeriodicDarcyData = DEFAULT_DATA,
) -> tuple[dict[str, Any], Any]:
    """Time independent conforming Q1 assembly, exterior elimination and fresh LU/AMG."""
    resources_before = resource_snapshot()
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        mesh = CartesianMacroMesh(fine_n * length, fine_n, (0, float(length), 0, 1))
        _, nodes = qk_space(mesh, 1)
        boundary = (
            np.isclose(nodes[:, 0], 0, rtol=0, atol=1e-14)
            | np.isclose(nodes[:, 0], length, rtol=0, atol=1e-14)
            | np.isclose(nodes[:, 1], 0, rtol=0, atol=1e-14)
            | np.isclose(nodes[:, 1], 1, rtol=0, atol=1e-14)
        )
        free = np.flatnonzero(~boundary)
        prepared = time.perf_counter()
        matrix, _, load = quadrilateral_operators(
            mesh, 1, permeability=data.permeability, source=data.source, order=4
        )
        assembled = time.perf_counter()
        coefficients = np.zeros(len(nodes))
        reduced = matrix[free][:, free]
        coefficients[free] = solve_linear(
            reduced,
            load[free],
            solver=solver,
            rtol=1e-10,
            atol=0,
            maxiter=500 if solver == "pyamg" else None,
            near_nullspace=np.ones((len(free), 1)) if solver == "pyamg" else None,
            refinement_precision="double",
            refinement_steps=2,
            equilibration="none",
        )
        completed = time.perf_counter()
        defect = reduced @ coefficients[free] - load[free]
        relative_residual = float(np.linalg.norm(defect) / np.linalg.norm(load[free]))
    return {
        "setup": prepared - started,
        "assembly": assembled - prepared,
        "solve_reconstruct": completed - assembled,
        "total": completed - started,
        "cold_total": PARENT_IMPORT_SECONDS + EXPORT_IMPORT_SECONDS + completed - started,
        "equation_relative_L2_residual": relative_residual,
        "resources_before": resources_before,
        "resources_after": resource_snapshot(),
        "fine_grid": [fine_n * length, fine_n],
        "fine_cells": fine_n**2 * length,
        "global_algebraic_size": len(free),
    }, (mesh, coefficients)


def evaluate_q1(
    mesh: CartesianMacroMesh,
    coefficients: np.ndarray,
    points: np.ndarray,
    basis_tables: Callable = qk_basis,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the represented pressure and physical broken gradient without smoothing."""
    origin = np.array(mesh.bounds)[[0, 2]]
    cell_coordinate = (points - origin) / mesh.spacing
    cells = np.floor(cell_coordinate).astype(int)
    cells[:, 0] = np.clip(cells[:, 0], 0, mesh.nx - 1)
    cells[:, 1] = np.clip(cells[:, 1], 0, mesh.ny - 1)
    reference = np.clip(cell_coordinate - cells, 0, 1)
    basis, derivative = basis_tables(1, reference)
    first = cells[:, 1] * (mesh.nx + 1) + cells[:, 0]
    local_dofs = first[:, None] + np.array([0, 1, mesh.nx + 1, mesh.nx + 2])
    values = coefficients[local_dofs]
    return np.einsum("qi,qi->q", basis, values), np.einsum(
        "qid,qi->qd", derivative / mesh.spacing, values
    )


def classical_evaluator(result: Any) -> Callable:
    """Return field evaluation in the classical executed nodal basis."""
    mesh, coefficients = result
    return lambda points: evaluate_q1(mesh, coefficients, points)


def mhm_evaluator(result: Any) -> Callable:
    """Return piecewise local evaluation with independent macroelement traces."""
    macro, system, solution = result

    def evaluate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select a one-sided macrocell and evaluate its own Q1 coefficients."""
        origin = np.array(macro.bounds)[[0, 2]]
        cell_coordinate = np.floor((points - origin) / macro.spacing).astype(int)
        cell_coordinate[:, 0] = np.clip(cell_coordinate[:, 0], 0, macro.nx - 1)
        cell_coordinate[:, 1] = np.clip(cell_coordinate[:, 1], 0, macro.ny - 1)
        ids = cell_coordinate[:, 1] * macro.nx + cell_coordinate[:, 0]
        pressure, gradient = np.empty(len(points)), np.empty((len(points), 2))
        for cell in np.unique(ids):
            mask = ids == cell
            fine = system.local_metadata[cell]["mesh"]
            pressure[mask], gradient[mask] = evaluate_q1(fine, solution.fields[cell], points[mask])
        return pressure, gradient

    return evaluate


def field_difference(
    first: Callable,
    second: Callable,
    integration_n: int,
    length: int = 1,
    order: int = 5,
    data: PeriodicDarcyData = DEFAULT_DATA,
) -> dict[str, float]:
    """Integrate physical errors per sqrt(domain area) on a resolving common partition."""
    mesh = CartesianMacroMesh(integration_n * length, integration_n, (0, float(length), 0, 1))
    reference, weights = quadrilateral_quadrature(order)
    measure = float(np.prod(mesh.spacing))
    pressure_square, flux_square = 0.0, 0.0
    for begin in range(0, len(mesh.cells), 256):
        origins = mesh.points[mesh.cells[begin : begin + 256, 0]]
        points = (origins[:, None, :] + reference[None, :, :] * mesh.spacing).reshape(-1, 2)
        pa, ga = first(points)
        pb, gb = second(points)
        dp = (pa - pb).reshape(len(origins), -1)
        dq = (-data.permeability(points)[:, None] * (ga - gb)).reshape(len(origins), -1, 2)
        pressure_square += measure * float(np.einsum("q,tq,tq->", weights, dp, dp))
        flux_square += measure * float(np.einsum("q,tqd,tqd->", weights, dq, dq))
    return {
        "pressure_L2_per_sqrt_area": math.sqrt(pressure_square / length),
        "flux_L2_per_sqrt_area": math.sqrt(flux_square / length),
    }


def exact_evaluator(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the independently differentiated exact pressure and physical gradient."""
    return exact_pressure(points), exact_gradient(points)


def numerical_digest(values: np.ndarray) -> str:
    """Hash the executed dtype, shape and coefficients without changing precision."""
    array = np.ascontiguousarray(values)
    fingerprint = hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode())
    if array.size:
        fingerprint.update(memoryview(array).cast("B"))
    return fingerprint.hexdigest()


def mhm_state_digests(result: Any) -> dict[str, Any]:
    """Fingerprint all operators, response values, bases, maps and solved global coordinates."""
    macro, system, solution = result
    arrays = {}

    def collect(name: str, value: Any) -> None:
        """Preserve dense values or literal CSR data, indices, indptr and shape."""
        if sparse.issparse(value):
            for part in ("data", "indices", "indptr"):
                arrays[name + "." + part] = getattr(value, part)
            arrays[name + ".shape"] = np.asarray(value.shape)
        else:
            arrays[name] = np.asarray(value)

    collect("global.matrix", system.matrix)
    for name in ("rhs", "load_scale", "kernel_offsets"):
        collect("global." + name, getattr(system, name))
    collect("solution.trace", solution.trace)
    collect("solution.gauge_multipliers", solution.gauge_multipliers)
    for name in ("points", "cells", "faces", "cell_faces", "signs", "bounds"):
        collect("macro." + name, getattr(macro, name))
    for cell, (response, record, coarse) in enumerate(
        zip(system.responses, system.cells, solution.coarse, strict=True)
    ):
        prefix = f"macro{cell}."
        for name in (
            "matrix",
            "coupling",
            "load",
            "trace_dofs",
            "kernel",
            "coarse_basis",
            "constraints",
            "test_coupling",
            "left_kernel",
            "test_basis",
            "test_constraints",
            "_retained_action",
            "_test_action",
            "_correct_kernel",
        ):
            collect(prefix + "problem." + name, getattr(response.problem, name))
        for name in ("source", "lifts", "retained_basis"):
            collect(prefix + name, getattr(response, name))
        collect(prefix + "coarse", coarse)
        collect(prefix + "direct.matrix", record.equations.matrix)
        collect(prefix + "direct.load", record.equations.load)
        for name in ("points", "cells", "faces", "cell_faces", "signs", "bounds"):
            collect(prefix + "fine." + name, getattr(record.equations.metadata["mesh"], name))
    return {
        name: {
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "sha256": numerical_digest(values),
        }
        for name, values in arrays.items()
    }


def show_control(
    current_mhm: Any,
    current_classical: Any,
    data: PeriodicDarcyData = DEFAULT_DATA,
) -> dict[str, Any]:
    """Integrate current fields and display current plus available historical figures."""
    from IPython.display import Image, display

    archive_folder = ROOT / "benchmarks/results/execution/introduction-processes-20261004"
    current_errors = {
        "MHM": field_difference(
            mhm_evaluator(current_mhm), exact_evaluator, integration_n=200, data=data
        ),
        "classical": field_difference(
            classical_evaluator(current_classical), exact_evaluator, integration_n=200, data=data
        ),
        "MHM_to_classical": field_difference(
            mhm_evaluator(current_mhm),
            classical_evaluator(current_classical),
            integration_n=200,
            data=data,
        ),
    }
    archive_record = json.loads((archive_folder / "measurements.json").read_text())
    print("Historical campaign provenance: 2026-10-04", archive_record["git_revision"])
    print(
        "Archived strong-scaling samples:", json.dumps(archive_record["strong_summary"], indent=2)
    )
    print("Archived weak-scaling samples:", json.dumps(archive_record["weak_summary"], indent=2))
    print("Current 200 x 200 physical-field control:", json.dumps(current_errors, indent=2))
    print("Current reduced-equation relative residual:", current_mhm[2].residual)
    assert current_mhm[2].residual < 1e-9

    # Show the current fields separately from the historical timing campaign.
    axis_nodes = (np.arange(140) + 0.5) / 140
    xx, yy = np.meshgrid(axis_nodes, axis_nodes)
    sample_points = np.column_stack((xx.ravel(), yy.ravel()))
    field_fig, field_axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, evaluator, title in zip(
        field_axes,
        (mhm_evaluator(current_mhm), classical_evaluator(current_classical)),
        (
            "Current MHM pressure (200 x 200 total fine cells)",
            "Current classical Q1 pressure (200 x 200)",
        ),
        strict=True,
    ):
        values, _ = evaluator(sample_points)
        artist = ax.pcolormesh(xx, yy, values.reshape(xx.shape), shading="nearest")
        for location in np.linspace(0, 1, 11):
            ax.axvline(location, color="black", alpha=0.3, linewidth=0.5)
            ax.axhline(location, color="black", alpha=0.3, linewidth=0.5)
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        field_fig.colorbar(artist, ax=ax, label="pressure")
    plt.show()

    # Published timings retain their original revision. Verify every available
    # artifact and identify the original payloads absent from a source checkout.
    historical_missing = []
    for checksum in (archive_folder / "SHA256SUMS").read_text().splitlines():
        expected, relative = checksum.split(maxsplit=1)
        original = archive_folder / relative
        if original.is_file():
            if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Archive integrity mismatch: {relative}")
        else:
            historical_missing.append(relative)
    print("Historical artifacts absent from this checkout:", historical_missing)
    print("The current numerical control above is independent of these historical timings.")
    print("Available historical campaign figures (original revision retained above):")
    for figure_name in (
        "pressure_fields.png",
        "flux_fields.png",
        "reference_refinement_errors.png",
        "strong_and_weak_scaling.png",
        "strong_stage_costs.png",
        "workload_crossover.png",
        "weak_efficiency_and_physical_errors.png",
    ):
        original_figure = archive_folder / "figures" / figure_name
        if original_figure.is_file():
            display(Image(filename=str(original_figure)))
        else:
            print(
                f"Historical figure unavailable: {figure_name}; "
                "run PYMHM_RUN_CAMPAIGN=1 for new measurements."
            )
    return current_errors


def run_campaign() -> dict[str, Any]:
    """Run all original process strong/weak and crossover studies with fresh workers."""
    step = 2e-6
    rectangle_data_checks = {}

    for length in (1, 4, 8, 16):
        rectangle_probe = np.array(
            [[0.237, 0.419], [length - 0.387, 0.728], [length / 2 + 0.132, 0.147]]
        )
        divergence = np.zeros(len(rectangle_probe))
        for axis in range(2):
            shift = np.eye(2)[axis] * step
            plus = (
                permeability(rectangle_probe + shift)
                * exact_gradient(rectangle_probe + shift)[:, axis]
            )
            minus = (
                permeability(rectangle_probe - shift)
                * exact_gradient(rectangle_probe - shift)[:, axis]
            )
            divergence += (plus - minus) / (2 * step)
        np.testing.assert_allclose(source(rectangle_probe), -divergence, rtol=2e-8, atol=2e-8)
        parameter = np.linspace(0, 1, 51)
        exterior = np.vstack(
            (
                np.column_stack((length * parameter, np.zeros_like(parameter))),
                np.column_stack((length * parameter, np.ones_like(parameter))),
                np.column_stack((np.zeros_like(parameter), parameter)),
                np.column_stack((np.full_like(parameter, length), parameter)),
            )
        )
        np.testing.assert_allclose(exact_pressure(exterior), 0, rtol=0, atol=1e-14)
        rectangle_data_checks[str(length)] = {
            "source_finite_difference_maximum": float(
                np.max(abs(source(rectangle_probe) + divergence))
            ),
            "exterior_pressure_maximum": float(np.max(abs(exact_pressure(exterior)))),
        }

    print({"physical_rectangle_checks": rectangle_data_checks})
    verify_forms()
    CPU_CAPACITY, CPU_METADATA = available_cpus()
    if PROCESS_COUNTS[-1] > CPU_CAPACITY:
        raise ValueError("A process count exceeds the affinity/quota available to this kernel")
    print({"workload": WORKLOAD, "resources": CPU_METADATA, "native_libraries": threadpool_info()})

    control_macro = CartesianMacroMesh(10, 10)
    control_face_space = FaceSpace.uniform(1, WORKLOAD["trace_segments"], continuous=True)
    control_skeleton = SkeletonSpace(
        control_macro, tuple(control_face_space for _ in control_macro.faces)
    )
    control_refinement = WORKLOAD["fine_n"] // 10
    control_lengths = face_length_snapshot(control_macro)
    control_template, control_ends = refined_interface_template(
        control_macro.spacing, control_refinement, control_face_space
    )
    trace_control = {}
    for cell in (0, 1, 10, 11):
        fine = control_macro.submesh(cell, control_refinement)
        direct = quadrilateral_trace_coupling(control_macro, cell, fine, control_skeleton, 1)
        transported = translated_interface(
            control_template, control_ends, control_macro, cell, control_lengths, control_face_space
        )
        np.testing.assert_allclose(transported, direct, rtol=5e-12, atol=5e-13)
        trace_control[str(cell)] = float(np.max(abs(transported - direct)))

    def validate_mhm_state(
        result: Any, reference: Any, expected_digests: dict[str, Any]
    ) -> dict[str, float]:
        """Require exact nonfield contracts and reconstruction roundoff on original rows."""
        if mhm_state_digests(result) != expected_digests:
            raise ValueError(
                "An operator, response value, executed basis, map or global coordinate changed"
            )
        _, system, solution = result
        maximum, defect_square, forcing_square = 0.0, 0.0, 0.0
        for response, actual, expected in zip(
            system.responses, solution.fields, reference[2].fields, strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            maximum = max(maximum, float(np.max(abs(actual - expected), initial=0)))
            local_trace = solution.trace[response.problem.trace_dofs]
            forcing = response.problem.load - response.problem.coupling @ local_trace
            check_linear_solution(response.problem.matrix, forcing, actual, rtol=1e-10, atol=0)
            defect = response.problem.matrix @ actual - forcing
            defect_square += float(np.dot(defect, defect))
            forcing_square += float(np.dot(forcing, forcing))
        return {
            "maximum_pressure_coefficient_difference": maximum,
            "original_local_rows_relative_L2": math.sqrt(defect_square / forcing_square),
            "global_compatibility_residual": float(solution.residual),
        }

    physical_norm_cache = {}

    def physical_norms_once(
        result: Any, kind: str, integration_n: int, length: int = 1
    ) -> dict[str, Any]:
        """Integrate each distinct executed field once, preserving the actual quadrature check."""
        mesh = result[0]
        if kind == "classical":
            fields, evaluator = (result[1],), classical_evaluator(result)
        elif kind == "mhm":
            fields, evaluator = result[2].fields, mhm_evaluator(result)
        else:
            raise ValueError("Use the declared classical or broken MHM field representation")
        key = (
            kind,
            mesh.nx,
            mesh.ny,
            tuple(float(x) for x in mesh.bounds),
            integration_n,
            length,
            EPSILON,
            tuple(numerical_digest(field) for field in fields),
        )
        if key not in physical_norm_cache:
            first = field_difference(evaluator, exact_evaluator, integration_n, length, order=5)
            second = field_difference(evaluator, exact_evaluator, integration_n, length, order=7)
            for quantity in first:
                np.testing.assert_allclose(first[quantity], second[quantity], rtol=1e-3, atol=1e-12)
            physical_norm_cache[key] = {"gauss5": first, "gauss7": second}
        return physical_norm_cache[key]

    reference_fields, reference_accuracy, reference_refinement, reference_timings = {}, {}, {}, {}
    integration_n = REFERENCE_FINE_SIZES[-1]
    for fine_n in REFERENCE_FINE_SIZES:
        reference_timings[str(fine_n)], result = run_classical(fine_n, "scipy")
        reference_fields[fine_n] = result
        reference_accuracy[str(fine_n)] = physical_norms_once(result, "classical", integration_n)[
            "gauss5"
        ]
    for a, b in zip(REFERENCE_FINE_SIZES, REFERENCE_FINE_SIZES[1:], strict=False):
        reference_refinement[f"{a}_to_{b}"] = field_difference(
            classical_evaluator(reference_fields[a]),
            classical_evaluator(reference_fields[b]),
            integration_n,
        )
    for quantity in ("pressure_L2_per_sqrt_area", "flux_L2_per_sqrt_area"):
        errors = [reference_accuracy[str(n)][quantity] for n in REFERENCE_FINE_SIZES]
        if any(b >= a for a, b in zip(errors, errors[1:], strict=False)):
            raise ValueError(
                "The classical refinement has not established decreasing physical field errors"
            )
    print(
        {
            "classical_exact_errors": reference_accuracy,
            "classical_successive_refinement_differences": reference_refinement,
        }
    )

    reference_rates = {}
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for column, quantity in enumerate(("pressure_L2_per_sqrt_area", "flux_L2_per_sqrt_area")):
        errors = np.array([reference_accuracy[str(n)][quantity] for n in REFERENCE_FINE_SIZES])
        sizes = np.array(REFERENCE_FINE_SIZES)
        rates = np.log(errors[:-1] / errors[1:]) / np.log(sizes[1:] / sizes[:-1])
        reference_rates[quantity] = rates.tolist()
        axes[0, column].loglog(1 / sizes, errors, "o-", label="Conforming Q1 reference")
        axes[0, column].set(
            xlabel="Fine-cell size h",
            ylabel=quantity.replace("_", " "),
            title="Measured reference refinement",
        )
        axes[1, column].semilogx(1 / sizes[1:], rates, "s-", label="Observed successive rate")
        axes[1, column].axhline(
            2 if column == 0 else 1,
            color="black",
            linestyle=":",
            label="Smooth Q1 asymptotic guide",
        )
        axes[1, column].set(
            xlabel="Finer cell size h", ylabel="Observed rate", title="Measured rates"
        )
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    figure.savefig(OUTPUT / "reference_refinement_errors.png", dpi=170)
    plt.show()
    print({"observed_reference_rates": reference_rates})

    validation_timings, validation_field = run_mhm(
        "serial", 1, WORKLOAD["fine_n"], WORKLOAD["trace_segments"]
    )
    validation_macro = validation_field[0]
    serial_digests = mhm_state_digests(validation_field)
    process_validation_timings, process_validation_field = run_mhm(
        "process", PROCESS_COUNTS[-1], WORKLOAD["fine_n"], WORKLOAD["trace_segments"]
    )
    process_state_validation = validate_mhm_state(
        process_validation_field, validation_field, serial_digests
    )
    classical_amg_validation_timings, classical_amg_field = run_classical(
        WORKLOAD["fine_n"], "pyamg"
    )
    classical_lu_validation_timings, classical_lu_field = run_classical(WORKLOAD["fine_n"], "scipy")
    evaluators = {
        "MHM serial": mhm_evaluator(validation_field),
        "MHM process": mhm_evaluator(process_validation_field),
        "Classical LU": classical_evaluator(classical_lu_field),
        "Classical AMG-CG": classical_evaluator(classical_amg_field),
    }
    control_fields = {
        "MHM serial": (validation_field, "mhm"),
        "MHM process": (process_validation_field, "mhm"),
        "Classical LU": (classical_lu_field, "classical"),
        "Classical AMG-CG": (classical_amg_field, "classical"),
    }
    control_physical_norms = {
        label: physical_norms_once(result, kind, integration_n)
        for label, (result, kind) in control_fields.items()
    }
    physical_errors = {label: norms["gauss5"] for label, norms in control_physical_norms.items()}
    quadrature_errors = {label: norms["gauss7"] for label, norms in control_physical_norms.items()}
    print("Higher-quadrature physical errors:", quadrature_errors)
    physical_agreement = {
        "serial_to_process": field_difference(
            evaluators["MHM serial"], evaluators["MHM process"], integration_n
        ),
        "classical_LU_to_AMG": field_difference(
            evaluators["Classical LU"], evaluators["Classical AMG-CG"], integration_n
        ),
        "MHM_to_refined_conforming": field_difference(
            evaluators["MHM serial"],
            classical_evaluator(reference_fields[integration_n]),
            integration_n,
        ),
    }
    print(
        {
            "physical_errors": physical_errors,
            "physical_agreement": physical_agreement,
            "fresh_MHM_to_classical_LU_exact_error_ratios": {
                quantity: physical_errors["MHM serial"][quantity]
                / physical_errors["Classical LU"][quantity]
                for quantity in physical_errors["MHM serial"]
            },
            "process_state_validation": process_state_validation,
            "global_MHM_residual": float(validation_field[2].residual),
        }
    )

    def field_panels(evaluators: dict[str, Callable], display_n: int) -> None:
        """Plot pressure, physical flux and errors with distinct colorbars and macrofaces."""
        display = CartesianMacroMesh(display_n, display_n)
        local = np.clip(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 2e-10, 1 - 2e-10
        )
        origins = display.points[display.cells[:, 0]]
        points = (origins[:, None, :] + local[None, :, :] * display.spacing).reshape(-1, 2)
        offsets = 4 * np.arange(len(display.cells))[:, None]
        triangles = np.concatenate((offsets + [0, 1, 3], offsets + [0, 3, 2]))
        exact_p, exact_grad = exact_evaluator(points)
        for quantity in ("pressure", "flux"):
            figure, axes = plt.subplots(
                len(evaluators), 2, figsize=(12, 4 * len(evaluators)), layout="constrained"
            )
            for row, (label, evaluator) in enumerate(evaluators.items()):
                p, gradient = evaluator(points)
                coefficient = permeability(points)[:, None]
                values = (
                    p if quantity == "pressure" else np.linalg.norm(-coefficient * gradient, axis=1)
                )
                errors = (
                    abs(p - exact_p)
                    if quantity == "pressure"
                    else np.linalg.norm(-coefficient * (gradient - exact_grad), axis=1)
                )
                for column, (axis, data, title) in enumerate(
                    zip(axes[row], (values, errors), (label, label + " error"), strict=True)
                ):
                    limits = (
                        {"vmin": 0, "vmax": max(float(np.max(data)), np.finfo(float).eps)}
                        if column or quantity == "flux"
                        else {}
                    )
                    artist = axis.tripcolor(
                        points[:, 0],
                        points[:, 1],
                        triangles,
                        data,
                        shading="gouraud",
                        rasterized=True,
                        **limits,
                    )
                    axis.add_collection(
                        LineCollection(
                            validation_macro.points[validation_macro.faces],
                            colors="black",
                            linewidths=0.6,
                            alpha=0.65,
                        )
                    )
                    axis.set(
                        xlim=(0, 1),
                        ylim=(0, 1),
                        xlabel="x",
                        ylabel="y",
                        aspect="equal",
                        title=title
                        + (" pressure" if quantity == "pressure" else " Darcy flux magnitude"),
                    )
                    figure.colorbar(artist, ax=axis, shrink=0.82, pad=0.025)
            figure.savefig(OUTPUT / (quantity + "_fields.png"), dpi=170)
            plt.show()

    field_panels(
        {
            "Exact": exact_evaluator,
            f"Fine conforming Q1 ({integration_n} per axis)": classical_evaluator(
                reference_fields[integration_n]
            ),
            "Classical LU": evaluators["Classical LU"],
            "Classical AMG-CG": evaluators["Classical AMG-CG"],
            "MHM serial": evaluators["MHM serial"],
            "MHM process": evaluators["MHM process"],
        },
        display_n=200,
    )

    strong_variants = [
        ("serial", 1),
        *(("process", p) for p in PROCESS_COUNTS),
        ("classical_scipy", 1),
        ("classical_pyamg", 1),
    ]

    def execute_strong_variant(backend: str, workers: int) -> tuple[dict[str, Any], Any]:
        """Select scheduling or the declared independent reference, preserving physical data."""
        if backend.startswith("classical_"):
            return run_classical(WORKLOAD["fine_n"], backend.removeprefix("classical_"))
        return run_mhm(backend, workers, WORKLOAD["fine_n"], WORKLOAD["trace_segments"])

    warmup, strong_samples, strong_accuracy = [], [], {}
    for backend, workers in strong_variants:
        timing, result = execute_strong_variant(backend, workers)
        checks = (
            validate_mhm_state(result, validation_field, serial_digests)
            if not backend.startswith("classical_")
            else {}
        )
        warmup.append(
            {
                "backend": backend,
                "workers": workers,
                **timing,
                "scientific_checks_after_timer": checks,
            }
        )
        strong_accuracy[f"{backend}_{workers}"] = physical_norms_once(
            result, "classical" if backend.startswith("classical_") else "mhm", integration_n
        )
        del result
    rng = random.Random(RANDOM_SEED)
    for repetition in range(REPETITIONS):
        order = strong_variants.copy()
        rng.shuffle(order)
        for backend, workers in order:
            timing, result = execute_strong_variant(backend, workers)
            checks = (
                validate_mhm_state(result, validation_field, serial_digests)
                if not backend.startswith("classical_")
                else {}
            )
            row = {"repetition": repetition, "backend": backend, "workers": workers, **timing}
            row["scientific_checks_after_timer"] = checks
            strong_samples.append(row)
            print(
                {
                    key: row[key]
                    for key in ("repetition", "backend", "workers", "total", "cold_total")
                },
                flush=True,
            )
            del result

    crossover_variants = [("process", p) for p in WORKLOAD["crossover_process_counts"]] + [
        ("classical_scipy", 1),
        ("classical_pyamg", 1),
    ]
    crossover_warmup, crossover_samples, crossover_accuracy = [], [], {}
    crossover_control_field = None

    def execute_crossover_variant(backend: str, workers: int) -> tuple[dict[str, Any], Any]:
        """Time the fixed smaller problem with matched material, source and exterior data."""
        if backend.startswith("classical_"):
            return run_classical(WORKLOAD["crossover_fine_n"], backend.removeprefix("classical_"))
        return run_mhm(
            backend, workers, WORKLOAD["crossover_fine_n"], WORKLOAD["crossover_trace_segments"]
        )

    for backend, workers in crossover_variants:
        timing, result = execute_crossover_variant(backend, workers)
        if backend == "process" and crossover_control_field is None:
            crossover_control_field = result
            crossover_digests = mhm_state_digests(result)
        checks = (
            validate_mhm_state(result, crossover_control_field, crossover_digests)
            if backend == "process"
            else {}
        )
        evaluator = mhm_evaluator(result) if backend == "process" else classical_evaluator(result)
        crossover_accuracy[f"{backend}_{workers}"] = {
            "exact": physical_norms_once(
                result, "mhm" if backend == "process" else "classical", integration_n
            ),
            "fine_reference": field_difference(
                evaluator, classical_evaluator(reference_fields[integration_n]), integration_n
            ),
        }
        crossover_warmup.append(
            {
                "backend": backend,
                "workers": workers,
                **timing,
                "scientific_checks_after_timer": checks,
            }
        )
        del result
    for repetition in range(REPETITIONS):
        order = crossover_variants.copy()
        rng.shuffle(order)
        for backend, workers in order:
            timing, result = execute_crossover_variant(backend, workers)
            checks = (
                validate_mhm_state(result, crossover_control_field, crossover_digests)
                if backend == "process"
                else {}
            )
            row = {
                "repetition": repetition,
                "backend": backend,
                "workers": workers,
                **timing,
                "scientific_checks_after_timer": checks,
            }
            crossover_samples.append(row)
            print(
                {
                    key: row[key]
                    for key in ("repetition", "backend", "workers", "total", "cold_total")
                },
                flush=True,
            )
            del result

    weak_warmup, weak_samples, weak_accuracy = [], [], {}
    weak_serial_digests, weak_reference_fields, weak_control_timings = {}, {}, {}
    weak_variants = [
        (backend, workers, length)
        for length in PROCESS_COUNTS
        for backend, workers in (
            ("process", length),
            ("classical_scipy", 1),
            ("classical_pyamg", 1),
        )
    ]

    def execute_weak_variant(backend: str, workers: int, length: int) -> tuple[dict[str, Any], Any]:
        """Solve the explicitly extended physical rectangle with unchanged H,h and period."""
        if backend.startswith("classical_"):
            return run_classical(
                WORKLOAD["weak_fine_n"], backend.removeprefix("classical_"), length
            )
        return run_mhm(
            backend, workers, WORKLOAD["weak_fine_n"], WORKLOAD["weak_trace_segments"], length
        )

    for length in PROCESS_COUNTS:
        timing, result = run_mhm(
            "serial", 1, WORKLOAD["weak_fine_n"], WORKLOAD["weak_trace_segments"], length
        )
        weak_reference_fields[length], weak_serial_digests[length] = (
            result,
            mhm_state_digests(result),
        )
        weak_control_timings[str(length)] = timing
        weak_control_timings[str(length)]["scientific_checks_after_timer"] = validate_mhm_state(
            result, result, weak_serial_digests[length]
        )

    for backend, workers, length in weak_variants:
        timing, result = execute_weak_variant(backend, workers, length)
        weak_warmup.append({"backend": backend, "workers": workers, "length": length, **timing})
        weak_accuracy[f"{backend}_{workers}_L{length}"] = physical_norms_once(
            result,
            "classical" if backend.startswith("classical_") else "mhm",
            WORKLOAD["weak_fine_n"],
            length,
        )
        if backend == "process":
            weak_warmup[-1]["scientific_checks_after_timer"] = validate_mhm_state(
                result, weak_reference_fields[length], weak_serial_digests[length]
            )
        del result
    for repetition in range(REPETITIONS):
        order = weak_variants.copy()
        rng.shuffle(order)
        for backend, workers, length in order:
            timing, result = execute_weak_variant(backend, workers, length)
            checks = (
                validate_mhm_state(
                    result, weak_reference_fields[length], weak_serial_digests[length]
                )
                if not backend.startswith("classical_")
                else {}
            )
            row = {
                "repetition": repetition,
                "backend": backend,
                "workers": workers,
                "length": length,
                **timing,
                "scientific_checks_after_timer": checks,
            }
            weak_samples.append(row)
            print(
                {
                    key: row[key]
                    for key in ("repetition", "backend", "workers", "length", "total", "cold_total")
                },
                flush=True,
            )
            del result

    def summarize(
        samples: list[dict], backend: str, workers: int, length: int | None = None
    ) -> dict[str, float]:
        """Report observed medians and ranges for one matched numerical configuration."""
        selected = [
            row
            for row in samples
            if row["backend"] == backend
            and row["workers"] == workers
            and (length is None or row["length"] == length)
        ]
        values = np.array([row["total"] for row in selected])
        return {
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            **{
                stage: float(np.median([row[stage] for row in selected]))
                for stage in ("setup", "assembly", "solve_reconstruct", "cold_total")
            },
        }

    strong_summary = {
        f"{backend}_{workers}": summarize(strong_samples, backend, workers)
        for backend, workers in strong_variants
    }
    weak_summary = {
        f"{backend}_{workers}_L{length}": summarize(weak_samples, backend, workers, length)
        for backend, workers, length in weak_variants
    }
    crossover_summary = {
        f"{backend}_{workers}": summarize(crossover_samples, backend, workers)
        for backend, workers in crossover_variants
    }
    process_times = np.array([strong_summary[f"process_{p}"]["median"] for p in PROCESS_COUNTS])
    serial_time = strong_summary["serial_1"]["median"]
    one_process_time = strong_summary["process_1"]["median"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    ranges = np.array(
        [
            [strong_summary[f"process_{p}"]["minimum"], strong_summary[f"process_{p}"]["maximum"]]
            for p in PROCESS_COUNTS
        ]
    )
    axes[0].errorbar(
        PROCESS_COUNTS,
        process_times,
        yerr=np.vstack((process_times - ranges[:, 0], ranges[:, 1] - process_times)),
        fmt="o-",
        capsize=4,
        label="MHM processes",
    )
    for label, key, style in (
        ("True serial MHM", "serial_1", "--"),
        ("Classical LU", "classical_scipy_1", ":"),
        ("Classical AMG-CG", "classical_pyamg_1", "-."),
    ):
        line = axes[0].axhline(strong_summary[key]["median"], linestyle=style, label=label)
        axes[0].axhspan(
            strong_summary[key]["minimum"],
            strong_summary[key]["maximum"],
            color=line.get_color(),
            alpha=0.10,
        )
    axes[1].plot(PROCESS_COUNTS, serial_time / process_times, "o-", label="Relative to true serial")
    axes[1].plot(
        PROCESS_COUNTS, one_process_time / process_times, "s-", label="Relative to process1"
    )
    axes[1].plot(PROCESS_COUNTS, PROCESS_COUNTS, "k:", label="Ideal reference line")
    weak_process_times = np.array(
        [weak_summary[f"process_{p}_L{p}"]["median"] for p in PROCESS_COUNTS]
    )
    weak_minima = np.array([weak_summary[f"process_{p}_L{p}"]["minimum"] for p in PROCESS_COUNTS])
    weak_maxima = np.array([weak_summary[f"process_{p}_L{p}"]["maximum"] for p in PROCESS_COUNTS])
    axes[2].errorbar(
        PROCESS_COUNTS,
        weak_process_times,
        yerr=np.vstack((weak_process_times - weak_minima, weak_maxima - weak_process_times)),
        fmt="o-",
        capsize=4,
        label="MHM processes, growing domain",
    )
    for backend, label in (
        ("classical_scipy", "Classical LU"),
        ("classical_pyamg", "Classical AMG-CG"),
    ):
        medians = np.array([weak_summary[f"{backend}_1_L{p}"]["median"] for p in PROCESS_COUNTS])
        minima = np.array([weak_summary[f"{backend}_1_L{p}"]["minimum"] for p in PROCESS_COUNTS])
        maxima = np.array([weak_summary[f"{backend}_1_L{p}"]["maximum"] for p in PROCESS_COUNTS])
        axes[2].errorbar(
            PROCESS_COUNTS,
            medians,
            yerr=np.vstack((medians - minima, maxima - medians)),
            fmt="s--",
            capsize=4,
            label=label,
        )
    for axis, title, ylabel in zip(
        axes,
        ("Strong: complete workflow", "Strong speedup", "Weak: complete growing workflow"),
        ("Time [s]", "Speedup", "Time [s]"),
        strict=True,
    ):
        axis.set(xlabel="Processes", ylabel=ylabel, title=title, xticks=PROCESS_COUNTS)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "strong_and_weak_scaling.png", dpi=170)
    plt.show()
    figure, axis = plt.subplots(figsize=(12, 5), layout="constrained")
    labels = list(strong_summary)
    bottom = np.zeros(len(labels))
    for stage in ("setup", "assembly", "solve_reconstruct"):
        values = np.array([strong_summary[label][stage] for label in labels])
        axis.bar(labels, values, bottom=bottom, label=stage.replace("_", " "))
        bottom += values
    axis.set(ylabel="Time [s]", title="Strong scaling: all timed stages")
    axis.tick_params(axis="x", labelrotation=30)
    axis.legend()
    figure.savefig(OUTPUT / "strong_stage_costs.png", dpi=170)
    plt.show()
    print(
        {
            "strong_efficiency_from_serial": (
                serial_time / process_times / np.array(PROCESS_COUNTS)
            ).tolist(),
            "weak_efficiency_from_process1": (weak_process_times[0] / weak_process_times).tolist(),
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for n, summary, style in (
        (WORKLOAD["crossover_fine_n"], crossover_summary, "o-"),
        (WORKLOAD["fine_n"], strong_summary, "s--"),
    ):
        counts = WORKLOAD["crossover_process_counts"]
        times = np.array([summary[f"process_{p}"]["median"] for p in counts])
        minima = np.array([summary[f"process_{p}"]["minimum"] for p in counts])
        maxima = np.array([summary[f"process_{p}"]["maximum"] for p in counts])
        axes[0].errorbar(
            counts,
            times,
            yerr=np.vstack((times - minima, maxima - times)),
            fmt=style,
            capsize=4,
            label=f"MHM processes, {n}² fine cells",
        )
        for solver, label in (("scipy", "LU"), ("pyamg", "AMG-CG")):
            axes[1].plot(
                counts,
                [summary[f"classical_{solver}_1"]["median"] / value for value in times],
                style,
                label=f"Classical {label}/MHM, {n}² fine cells",
            )
    axes[0].set(ylabel="Complete workflow time [s]", title="Measured workload crossover")
    axes[1].axhline(1, color="black", linestyle=":", label="Equal time")
    axes[1].set(ylabel="Matched classical time / MHM time", title="Gain includes every timed stage")
    for axis in axes:
        axis.set(xlabel="Processes", xticks=WORKLOAD["crossover_process_counts"])
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "workload_crossover.png", dpi=170)
    plt.show()
    print(
        {
            "separately_reported_warmups": {
                "strong": [
                    {key: row[key] for key in ("backend", "workers", "total")} for row in warmup
                ],
                "crossover": [
                    {key: row[key] for key in ("backend", "workers", "total")}
                    for row in crossover_warmup
                ],
                "weak": [
                    {key: row[key] for key in ("backend", "workers", "length", "total")}
                    for row in weak_warmup
                ],
            }
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    axes[0].plot(
        PROCESS_COUNTS,
        weak_process_times[0] / weak_process_times,
        "o-",
        label="Measured process weak efficiency",
    )
    axes[0].axhline(1, color="black", linestyle=":", label="Ideal reference line")
    axes[0].set(ylabel="Weak efficiency", title="Fixed work per process")
    for backend, label in (
        ("process", "MHM processes"),
        ("classical_scipy", "Classical LU"),
        ("classical_pyamg", "Classical AMG-CG"),
    ):
        for axis, quantity in zip(
            axes[1:], ("pressure_L2_per_sqrt_area", "flux_L2_per_sqrt_area"), strict=True
        ):
            values = [
                weak_accuracy[f"{backend}_{p if backend == 'process' else 1}_L{p}"]["gauss5"][
                    quantity
                ]
                for p in PROCESS_COUNTS
            ]
            axis.plot(PROCESS_COUNTS, values, "o-", label=label)
            axis.set(
                ylabel=quantity.replace("_", " "), title="Exact physical error on each rectangle"
            )
    for axis in axes:
        axis.set(xlabel="Processes / physical rectangle length", xticks=PROCESS_COUNTS)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.savefig(OUTPUT / "weak_efficiency_and_physical_errors.png", dpi=170)
    plt.show()

    interval_basis = simplex_lagrange_basis("interval", 1, nodes=np.array([[1.0, 0.0], [0.0, 1.0]]))
    state = {
        "trace": validation_field[2].trace,
        "macro_points": validation_macro.points,
        "macro_cells": validation_macro.cells,
        "macro_faces": validation_macro.faces,
        "macro_cell_faces": validation_macro.cell_faces,
        "macro_signs": validation_macro.signs,
        "macro_grid": np.array([validation_macro.nx, validation_macro.ny]),
        "macro_bounds": np.asarray(validation_macro.bounds),
        "interval_basis_matrix": interval_basis.basis_matrix,
        "interval_native_basis_matrix": interval_basis.element.basis_matrix,
        "interval_nodes": interval_basis.nodes,
        "interval_permutation": interval_basis.permutation,
        "interval_polyset_type": np.asarray(interval_basis.element.polyset_type),
        "interval_backend_version": np.asarray(interval_basis.element.backend_version),
        "trace_parameter_knots": np.linspace(0, 1, WORKLOAD["trace_segments"] + 1),
        "trace_continuous_on_each_face": np.asarray(True),
        "classical_LU_pressure": classical_lu_field[1],
        "classical_AMG_pressure": classical_amg_field[1],
    }
    for cell, (response, field, coarse) in enumerate(
        zip(
            validation_field[1].responses,
            validation_field[2].fields,
            validation_field[2].coarse,
            strict=True,
        )
    ):
        state[f"pressure_{cell}"] = field
        state[f"coarse_{cell}"] = coarse
        state[f"retained_basis_{cell}"] = response.retained_basis
        state[f"source_{cell}"] = response.source
        state[f"lifts_{cell}"] = response.lifts
        state[f"trace_dofs_{cell}"] = response.problem.trace_dofs
        fine = validation_field[1].local_metadata[cell]["mesh"]
        state[f"fine_grid_{cell}"] = np.array([fine.nx, fine.ny])
        state[f"fine_bounds_{cell}"] = np.asarray(fine.bounds)
    for fine_n, (_, coefficients) in reference_fields.items():
        state[f"reference_pressure_{fine_n}"] = coefficients
    np.savez_compressed(OUTPUT / "fields_and_executed_bases.npz", **state)

    def archive_additional_case(result: Any, name: str, trace_segments: int) -> dict[str, Any]:
        """Persist a complete field replay contract for each additional physical mesh."""
        macro, system, solution = result
        arrays = {key: value for key, value in state.items() if key.startswith("interval_")}
        arrays.update(
            {
                "trace": solution.trace,
                "macro_grid": np.array([macro.nx, macro.ny]),
                "macro_bounds": np.asarray(macro.bounds),
                "macro_points": macro.points,
                "macro_cells": macro.cells,
                "macro_faces": macro.faces,
                "macro_cell_faces": macro.cell_faces,
                "macro_signs": macro.signs,
                "trace_parameter_knots": np.linspace(0, 1, trace_segments + 1),
            }
        )
        for cell, (response, field, coarse) in enumerate(
            zip(system.responses, solution.fields, solution.coarse, strict=True)
        ):
            for prefix, value in (
                ("pressure", field),
                ("coarse", coarse),
                ("source", response.source),
                ("lifts", response.lifts),
                ("retained_basis", response.retained_basis),
                ("trace_dofs", response.problem.trace_dofs),
            ):
                arrays[f"{prefix}_{cell}"] = value
            fine = system.local_metadata[cell]["mesh"]
            arrays[f"fine_grid_{cell}"] = np.array([fine.nx, fine.ny])
            arrays[f"fine_bounds_{cell}"] = np.asarray(fine.bounds)
        destination = OUTPUT / (name + ".npz")
        np.savez_compressed(destination, **arrays)
        return {
            "archive": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "executed_basis_digests": {
                key: numerical_digest(value) for key, value in arrays.items() if "basis" in key
            },
        }

    additional_case_archives = {
        f"weak_L{length}": archive_additional_case(
            result, f"weak_L{length}_fields_and_bases", WORKLOAD["weak_trace_segments"]
        )
        for length, result in weak_reference_fields.items()
    }
    additional_case_archives["crossover"] = archive_additional_case(
        crossover_control_field, "crossover_fields_and_bases", WORKLOAD["crossover_trace_segments"]
    )

    def saved_q1_tables(
        degree: int, points: np.ndarray, saved_basis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate archived Q1 coefficients through the shared polynomial owner."""
        if degree != 1:
            raise ValueError("This archived tutorial uses tensor Q1 factors")
        x_table = orthogonal_polynomial_tabulation("interval", 1, points[:, :1], nderiv=1)
        y_table = orthogonal_polynomial_tabulation("interval", 1, points[:, 1:], nderiv=1)
        x, dx = x_table[0] @ saved_basis.T, x_table[1] @ saved_basis.T
        y, dy = y_table[0] @ saved_basis.T, y_table[1] @ saved_basis.T
        values = np.einsum("qi,qj->qij", y, x).reshape(-1, 4)
        gradients = np.stack(
            (np.einsum("qi,qj->qij", y, dx), np.einsum("qi,qj->qij", dy, x)), axis=-1
        )
        return values, gradients.reshape(-1, 4, 2)

    def archived_evaluator(saved: dict[str, np.ndarray], fields: list[np.ndarray]) -> Callable:
        """Restore geometry and evaluate fields through their saved native coefficient factors."""
        macro = CartesianMacroMesh(*saved["macro_grid"], tuple(saved["macro_bounds"]))
        fine_meshes = [
            CartesianMacroMesh(*saved[f"fine_grid_{cell}"], tuple(saved[f"fine_bounds_{cell}"]))
            for cell in range(len(macro.cells))
        ]

        def basis_tables(degree: int, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Use the archived ordered native factor matrix for every field evaluation."""
            return saved_q1_tables(degree, points, saved["interval_basis_matrix"])

        def evaluate(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Select independent local fields using the archived macrogeometry."""
            cell_coordinates = np.floor(
                (points - np.array(macro.bounds)[[0, 2]]) / macro.spacing
            ).astype(int)
            cell_coordinates[:, 0] = np.clip(cell_coordinates[:, 0], 0, macro.nx - 1)
            cell_coordinates[:, 1] = np.clip(cell_coordinates[:, 1], 0, macro.ny - 1)
            indices = cell_coordinates[:, 1] * macro.nx + cell_coordinates[:, 0]
            pressure, gradient = np.empty(len(points)), np.empty((len(points), 2))
            for cell in np.unique(indices):
                mask = indices == cell
                pressure[mask], gradient[mask] = evaluate_q1(
                    fine_meshes[cell], fields[cell], points[mask], basis_tables
                )
            return pressure, gradient

        return evaluate

    replay_checks = {}
    with np.load(OUTPUT / "fields_and_executed_bases.npz", allow_pickle=False) as saved:
        for native_threads in (1, 2):
            maximum = 0.0
            with threadpool_limits(limits=native_threads):
                for cell, original_response in enumerate(validation_field[1].responses):
                    replay_response = LocalResponse(
                        problem=original_response.problem,
                        source=saved[f"source_{cell}"],
                        lifts=saved[f"lifts_{cell}"],
                        coarse_vectors=saved[f"retained_basis_{cell}"],
                    )
                    trace = saved["trace"][saved[f"trace_dofs_{cell}"]]
                    coarse = saved[f"coarse_{cell}"]
                    replayed = replay_response.reconstruct(trace, coarse)
                    np.testing.assert_allclose(
                        replayed, saved[f"pressure_{cell}"], rtol=5e-13, atol=1e-12
                    )
                    # This one-dimensional retained space has an equivalent sign
                    # rotation. Rotate its coordinates and its saved basis together.
                    rotated = replace(
                        replay_response, coarse_vectors=-replay_response.retained_basis
                    )
                    rotated_field = rotated.reconstruct(trace, -coarse)
                    np.testing.assert_allclose(rotated_field, replayed, rtol=5e-13, atol=1e-12)
                    maximum = max(maximum, float(np.max(abs(replayed - saved[f"pressure_{cell}"]))))
            replay_checks[str(native_threads)] = {
                "maximum_pressure_coefficient_difference": maximum,
                "equivalent_retained_sign_rotation": True,
            }
    print({"archived_replay": replay_checks})

    physical_basis_replay = {}
    with np.load(OUTPUT / "fields_and_executed_bases.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    np.testing.assert_array_equal(
        saved["interval_basis_matrix"],
        saved["interval_native_basis_matrix"][saved["interval_permutation"]],
    )
    if str(saved["interval_polyset_type"]) != "standard" or str(
        saved["interval_backend_version"]
    ) != version("fenics-basix"):
        raise ValueError(
            "Replay requires the declared native polynomial convention and backend version"
        )
    reference_points, _ = quadrilateral_quadrature(5)
    saved_tables = saved_q1_tables(1, reference_points, saved["interval_basis_matrix"])
    live_tables = qk_basis(1, reference_points)
    for first, second in zip(saved_tables, live_tables, strict=True):
        np.testing.assert_allclose(first, second, rtol=1e-14, atol=1e-14)
    fields = [saved[f"pressure_{cell}"] for cell in range(len(validation_macro.cells))]
    for native_threads in (1, 2):
        with threadpool_limits(limits=native_threads):
            errors = field_difference(
                archived_evaluator(saved, fields), evaluators["MHM serial"], integration_n
            )
        # Four Q1 cardinal contributions bound coefficient perturbations; raw
        # derivatives add 1/h and the known permeability upper bound exp(1).
        amplitude = max(float(np.max(abs(field))) for field in fields)
        roundoff = 32 * np.finfo(float).eps * max(1, amplitude)
        bounds = {
            "pressure_L2_per_sqrt_area": 4 * roundoff,
            "flux_L2_per_sqrt_area": 4 * math.sqrt(2) * math.e * integration_n * roundoff,
        }
        if any(errors[name] > bounds[name] for name in errors):
            raise ValueError("Physical replay exceeds the declared native-basis roundoff bound")
        physical_basis_replay[str(native_threads)] = {
            "physical_differences": errors,
            "roundoff_bounds": bounds,
        }
    print({"archived_native_basis_physical_replay": physical_basis_replay})
    del saved, fields

    additional_replay_checks = {}
    for name, original in [
        (f"weak_L{length}", result) for length, result in weak_reference_fields.items()
    ] + [("crossover", crossover_control_field)]:
        archive_path = OUTPUT / additional_case_archives[name]["archive"]
        with np.load(archive_path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        checks = {}
        for native_threads in (1, 2):
            maximum = 0.0
            with threadpool_limits(limits=native_threads):
                for cell, original_response in enumerate(original[1].responses):
                    response = LocalResponse(
                        problem=original_response.problem,
                        source=saved[f"source_{cell}"],
                        lifts=saved[f"lifts_{cell}"],
                        coarse_vectors=saved[f"retained_basis_{cell}"],
                    )
                    trace, coarse = (
                        saved["trace"][saved[f"trace_dofs_{cell}"]],
                        saved[f"coarse_{cell}"],
                    )
                    replayed = response.reconstruct(trace, coarse)
                    rotated = replace(
                        response, coarse_vectors=-response.retained_basis
                    ).reconstruct(trace, -coarse)
                    np.testing.assert_allclose(
                        replayed, saved[f"pressure_{cell}"], rtol=0, atol=1e-12
                    )
                    np.testing.assert_allclose(rotated, replayed, rtol=0, atol=1e-12)
                    maximum = max(maximum, float(np.max(abs(replayed - saved[f"pressure_{cell}"]))))
            checks[str(native_threads)] = {
                "maximum_pressure_coefficient_difference": maximum,
                "equivalent_retained_sign_rotation": True,
            }
        fields = [saved[f"pressure_{cell}"] for cell in range(len(original[0].cells))]
        n = WORKLOAD["crossover_fine_n"] if name == "crossover" else WORKLOAD["weak_fine_n"]
        length = 1 if name == "crossover" else int(name.removeprefix("weak_L"))
        physical = field_difference(
            archived_evaluator(saved, fields), mhm_evaluator(original), n, length
        )
        amplitude = max(float(np.max(abs(field))) for field in fields)
        roundoff = 32 * np.finfo(float).eps * max(1, amplitude)
        bounds = {
            "pressure_L2_per_sqrt_area": 4 * roundoff,
            "flux_L2_per_sqrt_area": 4 * math.sqrt(2) * math.e * n * roundoff,
        }
        if any(physical[key] > bounds[key] for key in physical):
            raise ValueError("Additional physical replay exceeds the native-basis roundoff bound")
        additional_replay_checks[name] = {
            "coefficient_replay": checks,
            "physical_replay": physical,
            "physical_roundoff_bounds": bounds,
        }
        del saved, fields
    print({"additional_physical_mesh_replay": additional_replay_checks})

    manifest = current_source_manifest(
        {
            **{
                "importable_provider": module_sha256,
                "source_notebook": hashlib.sha256(SOURCE_NOTEBOOK.read_bytes()).hexdigest(),
                "pixi.lock": hashlib.sha256((ROOT / "pixi.lock").read_bytes()).hexdigest(),
            },
            "examples/introduction/scaling_processes.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "examples/introduction/scaling_forms.py": hashlib.sha256(
                Path(__file__).with_name("scaling_forms.py").read_bytes()
            ).hexdigest(),
            "examples/introduction/_scaling_timer.py": hashlib.sha256(
                Path(__file__).with_name("_scaling_timer.py").read_bytes()
            ).hexdigest(),
        }
    )
    record = {
        "schema": "pymhm.introduction.darcy-process-scalability.v1",
        "workload": WORKLOAD,
        "source_manifest": manifest,
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "fenics-basix",
                "fenics-dolfinx",
                "pyamg",
                "threadpoolctl",
            )
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": CPU_METADATA,
        "affinity_after": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "native_libraries": threadpool_info(),
        "native_threads": 1,
        "one_time_parent_import_seconds": PARENT_IMPORT_SECONDS,
        "one_time_provider_import_seconds": EXPORT_IMPORT_SECONDS,
        "repetitions": REPETITIONS,
        "random_seed": RANDOM_SEED,
        "strong_warmup": warmup,
        "strong_samples": strong_samples,
        "strong_summary": strong_summary,
        "strong_physical_errors_once": strong_accuracy,
        "weak_warmup": weak_warmup,
        "weak_samples": weak_samples,
        "weak_summary": weak_summary,
        "crossover_warmup": crossover_warmup,
        "crossover_samples": crossover_samples,
        "crossover_summary": crossover_summary,
        "crossover_physical_errors": crossover_accuracy,
        "physical_errors": physical_errors,
        "physical_agreement": physical_agreement,
        "error_quadrature_orders": [5, 7],
        "common_error_partition_per_unit_axis": integration_n,
        "reference_exact_errors": reference_accuracy,
        "reference_refinement": reference_refinement,
        "observed_reference_rates": reference_rates,
        "untimed_scientific_controls": {
            "reference_refinement": reference_timings,
            "MHM_serial": validation_timings,
            "MHM_process": process_validation_timings,
            "classical_LU": classical_lu_validation_timings,
            "classical_AMG": classical_amg_validation_timings,
            "weak_serial_each_domain": weak_control_timings,
        },
        "independent_rectangle_data_checks": rectangle_data_checks,
        "weak_physical_errors_per_sqrt_area": weak_accuracy,
        "trace_geometry_controls": trace_control,
        "unique_executed_fields_physically_integrated": len(physical_norm_cache),
        "serial_state_digests": serial_digests,
        "basis_array_digests": {
            name: numerical_digest(array) for name, array in state.items() if "basis" in name
        },
        "field_archive_sha256": hashlib.sha256(
            (OUTPUT / "fields_and_executed_bases.npz").read_bytes()
        ).hexdigest(),
        "archived_replay": replay_checks,
        "archived_native_basis_physical_replay": physical_basis_replay,
        "additional_field_archives": additional_case_archives,
        "additional_mesh_replay": additional_replay_checks,
        "timing_scope": (
            "fresh setup, startup/imports, complete response transfer, or"
            "dered global assembly, join, solve and full reconstruction"
        ),
        "classical_AMG": {
            "rtol": 1e-10,
            "atol": 0,
            "maxiter": 500,
            "near_nullspace": "one constant candidate",
            "refinement_precision": "double",
            "refinement_steps": 2,
            "equilibration": "none",
        },
    }
    (OUTPUT / "measurements.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        {
            "record": str(OUTPUT / "measurements.json"),
            "archive_sha256": record["field_archive_sha256"],
        }
    )

    # All generic assemble calls have completed and joined their workers.

    return record
