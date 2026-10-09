"""Acquire identical-discretization CPU kernel and complete Darcy measurements.

The notebook defines the same local and global equations explicitly. This
module keeps importable manufactured data, spawn callables, timers, physical
controls and report writing outside instructional cells. Numerical operations
are delegated to their PyMHM owners. Every solve rebuilds its local operators,
factors and global system; no material-equivalence cache is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import pymhm
from pymhm import Equation, LocalEquations, MultiscaleProblem, assemble
from pymhm.core.contributions import assemble_hybrid_contributions, local_global_contribution
from pymhm.core.validation import positive_int
from pymhm.execution.cpu import ExecutionConfig
from pymhm.fem.scalar.operators import _scalar_diffusion_blocks, boundary_data
from pymhm.fem.scalar.tetrahedron import tetra_operators
from pymhm.fem.scalar.triangle import scalar_operators, trace_coupling
from pymhm.fem.traces.interval import SkeletonSpace
from pymhm.fem.traces.triangle_3d import (
    TriangularSkeleton,
    tetra_boundary_data,
    tetra_trace_coupling,
)
from pymhm.meshes.tetrahedron import TetraMesh
from pymhm.meshes.triangle import TriangleMesh
from pymhm.postprocessing.nodal import nodal_field
from pymhm.postprocessing.solutions import Darcy3DSolution, DarcySolution


@dataclass(frozen=True)
class KernelCase:
    """Specify one fixed triangle/tetrahedron P2/P0-trace manufactured problem.

    ``macro_resolution`` counts Cartesian divisions before simplicial splitting;
    ``local_refinement`` counts each macro edge's subdivisions. The material is
    smooth, positive and nonaligned with macro translation, with fixed period
    0.137. Its gradient and the manufactured source are evaluated analytically.
    """

    dimension: int
    macro_resolution: int
    local_refinement: int
    degree: int = 2
    quadrature_order: int = 6
    period: float = 0.137
    amplitude: float = 0.25

    def __post_init__(self) -> None:
        """Reject invalid space, grid and coefficient parameters before timing."""
        if self.dimension not in (2, 3):
            raise ValueError("dimension must be two or three")
        for name in ("macro_resolution", "local_refinement", "degree", "quadrature_order"):
            positive_int(getattr(self, name), name)
        if self.dimension == 3 and self.local_refinement & (self.local_refinement - 1):
            raise ValueError("tetrahedral local refinement must be a power of two")
        if not np.isfinite(self.period) or self.period <= 0 or not np.isfinite(self.amplitude):
            raise ValueError("period must be positive and amplitude must be finite")

    @property
    def name(self) -> str:
        """Return a stable filename key containing all mesh and space choices."""
        return (
            f"{self.dimension}d-n{self.macro_resolution}-r{self.local_refinement}"
            f"-p{self.degree}-q{self.quadrature_order}"
        )


def exact_pressure(points: np.ndarray) -> np.ndarray:
    """Return sin(pi*x_i)'s product with homogeneous pressure on the unit box."""
    return np.prod(np.sin(np.pi * points), axis=-1)


def exact_gradient(points: np.ndarray) -> np.ndarray:
    """Differentiate each factor analytically without division near its zeros."""
    dimension = points.shape[-1]
    sine, cosine = np.sin(np.pi * points), np.cos(np.pi * points)
    return np.pi * np.stack(
        [
            cosine[..., i] * np.prod(sine[..., [j for j in range(dimension) if j != i]], axis=-1)
            for i in range(dimension)
        ],
        axis=-1,
    )


def permeability(points: np.ndarray, case: KernelCase) -> np.ndarray:
    """Evaluate exp(amplitude*prod(sin(2*pi*x_i/period))) as scalar K."""
    return np.exp(case.amplitude * np.prod(np.sin(2 * np.pi * points / case.period), axis=-1))


def source(points: np.ndarray, case: KernelCase) -> np.ndarray:
    """Apply -div(K grad(p)) using independent exact product derivatives."""
    dimension, frequency = case.dimension, 2 * np.pi / case.period
    sine, cosine = np.sin(frequency * points), np.cos(frequency * points)
    material = permeability(points, case)
    gradient = (
        frequency
        * case.amplitude
        * material[..., None]
        * np.stack(
            [
                cosine[..., i]
                * np.prod(sine[..., [j for j in range(dimension) if j != i]], axis=-1)
                for i in range(dimension)
            ],
            axis=-1,
        )
    )
    return dimension * np.pi**2 * material * exact_pressure(points) - np.sum(
        gradient * exact_gradient(points), axis=-1
    )


def exact_flux(points: np.ndarray, case: KernelCase) -> np.ndarray:
    """Return the physical Darcy vector field -K grad(p)."""
    return -permeability(points, case)[..., None] * exact_gradient(points)


def local_equations(cell: int, *, case: KernelCase, mesh: Any, skeleton: Any) -> LocalEquations:
    """Declare A*p+B*lambda=f, C=-B.T and a physical-volume pressure moment."""
    fine = mesh.submesh(cell, case.local_refinement)
    material, force = partial(permeability, case=case), partial(source, case=case)
    if case.dimension == 2:
        matrix, mass, load = scalar_operators(
            fine, case.degree, diffusion=material, source=force, order=case.quadrature_order
        )
        coupling = trace_coupling(mesh, cell, fine, skeleton, case.degree)
    else:
        matrix, mass, load = tetra_operators(
            fine, case.degree, diffusion=material, source=force, order=case.quadrature_order
        )
        coupling = tetra_trace_coupling(mesh, cell, fine, skeleton, case.degree)
    constant = np.ones((len(load), 1))
    return LocalEquations(
        matrix,
        load,
        coupling,
        -coupling.T,
        skeleton.cell_dofs(cell),
        kernel=constant,
        moments=mass @ constant,
        metadata=fine,
        field_data=(nodal_field("pressure", fine, case.degree),),
    )


def describe_problem(case: KernelCase) -> tuple[MultiscaleProblem[int], Any]:
    """Create actual macro geometry, physical boundaries and explicit global equation."""
    if case.dimension == 2:
        mesh = TriangleMesh.unit_square(case.macro_resolution)
        skeleton = SkeletonSpace(mesh)
        boundary, fixed = boundary_data(skeleton, 0.0, order=case.quadrature_order)
    else:
        mesh = TetraMesh.unit_cube(case.macro_resolution)
        skeleton = TriangularSkeleton(mesh)
        boundary, fixed = tetra_boundary_data(skeleton, 0.0, {}, case.quadrature_order)
    problem = MultiscaleProblem(
        Equation(0, -np.r_[boundary, np.zeros(len(mesh.cells))]),
        partial(local_equations, case=case, mesh=mesh, skeleton=skeleton),
        range(len(mesh.cells)),
        skeleton.size,
        (1,) * len(mesh.cells),
        fixed=fixed,
    )
    return problem, skeleton


def recover_fields(case: KernelCase, skeleton: Any, system: Any, result: Any) -> Any:
    """Interpret coefficients through existing pressure/physical-flux field owners."""
    material, force = partial(permeability, case=case), partial(source, case=case)
    if case.dimension == 3:
        return Darcy3DSolution(
            skeleton,
            system.local_metadata,
            result.fields,
            result,
            case.degree,
            material,
            force,
            case.quadrature_order,
        )
    # The norm owner evaluates the full raw gradient flux, independently of these
    # optional centroid samples. No H(div) field is claimed for primal MHM.
    return DarcySolution(
        skeleton,
        system.local_metadata,
        result.fields,
        tuple(np.empty((0, 2)) for _ in result.fields),
        result,
        "primal",
        material,
        force,
        case.quadrature_order,
        case.degree,
    )


def physical_controls(case: KernelCase, fields: Any, system: Any) -> dict[str, Any]:
    """Check original local PDE rows, macro balance and independent physical norms.

    Local backward errors divide the L2 defect by the L2 absolute operator
    action, separately from the condensed global compatibility residual.
    Assembly-quadrature macro balance uses the conservative skeleton flux;
    the raw gradient flux is not asserted to conserve every fine cell.
    """
    residuals = []
    for response, field in zip(system.responses, fields.hybrid.fields, strict=True):
        problem = response.problem
        trace = fields.hybrid.trace[problem.trace_dofs]
        defect = problem.matrix @ field + problem.coupling @ trace - problem.load
        scale = abs(problem.matrix) @ abs(field) + abs(problem.coupling) @ abs(trace)
        scale += abs(problem.load)
        residuals.append(
            float(np.linalg.norm(defect) / max(np.linalg.norm(scale), np.finfo(float).tiny))
        )
    controls = {
        "pressure_l2_error": fields.l2_error(exact_pressure, order=case.quadrature_order + 2),
        "physical_flux_l2_error": fields.flux_l2_error(
            partial(exact_flux, case=case), order=case.quadrature_order + 2
        ),
        "local_original_equation_l2_backward_errors": residuals,
        "maximum_local_original_equation_l2_backward_error": max(residuals),
        "macrocell_skeleton_flux_balance_linf": float(max(abs(fields.conservation_residuals()))),
        "global_compatibility_residual": float(fields.hybrid.residual),
    }
    samples = []
    for field in fields.hybrid.field("pressure"):
        points = field.mesh.points[field.mesh.cells].mean(axis=1)
        samples.append(float(np.max(abs(field.evaluate(points) - exact_pressure(points)))))
    controls["pressure_centroid_sample_error_linf"] = max(samples)
    for name in (
        "maximum_local_original_equation_l2_backward_error",
        "macrocell_skeleton_flux_balance_linf",
        "global_compatibility_residual",
    ):
        if controls[name] > 1e-10:
            raise AssertionError(f"Unchanged physical control failed: {name}={controls[name]}")
    return controls


def solve_once(
    case: KernelCase, *, backend: str = "serial", workers: int = 1
) -> tuple[dict[str, Any], Any, Any]:
    """Time the complete fresh CPU workflow, including physical post-processing.

    The total includes mesh/description setup, thread-limit setup, independent
    local assembly/factorization, worker launch/transfer/synchronization/shutdown
    when selected, ordered global assembly, global solve, reconstruction and
    physical error/residual evaluation. Import/process-launch overhead belongs
    to the outer acquisition timer and is reported independently. No GPU is used.
    """
    phases: dict[str, float] = {}
    begin = time.perf_counter()
    with threadpool_limits(limits=1):
        ready = time.perf_counter()
        phases["native_thread_setup"] = ready - begin
        problem, skeleton = describe_problem(case)
        described = time.perf_counter()
        phases["mesh_and_problem_description"] = described - ready
        system = assemble(problem, execution=ExecutionConfig(backend=backend, workers=workers))
        assembled = time.perf_counter()
        phases["assembly_and_local_elimination"] = assembled - described
        result = system.solve()
        solved = time.perf_counter()
        phases["global_solve_and_reconstruction"] = solved - assembled
        fields = recover_fields(case, skeleton, system, result)
        diagnostics = physical_controls(case, fields, system)
        phases["physical_postprocessing"] = time.perf_counter() - solved
    elapsed = time.perf_counter() - begin
    return (
        {
            "complete_seconds": elapsed,
            "phases": phases,
            "physical_controls": diagnostics,
            "macro_cells": len(skeleton.mesh.cells),
            "fine_cells": sum(len(mesh.cells) for mesh in system.local_metadata),
            "trace_dofs": skeleton.size,
            "local_pressure_dofs": [len(field) for field in result.fields],
        },
        system,
        fields,
    )


def source_manifest() -> dict[str, str]:
    """Hash every executed package module and the current acquisition helper."""
    root = Path(pymhm.__file__).resolve().parent
    result = {
        "src/pymhm/" + path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    result["examples/numba_performance.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return result


def save_coefficients(path: Path, fields: Any, system: Any) -> dict[str, Any]:
    """Persist executed nodal coefficients, trace and CSC system for paired replay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    definitions = fields.hybrid.field("pressure")
    np.savez_compressed(
        path,
        trace=fields.hybrid.trace,
        pressure=np.concatenate(fields.hybrid.fields),
        local_offsets=np.r_[0, np.cumsum([len(field) for field in fields.hybrid.fields])],
        global_values=system.matrix.data,
        global_indices=system.matrix.indices,
        global_indptr=system.matrix.indptr,
        global_load=system.rhs,
        local_points=np.stack([field.mesh.points for field in definitions]),
        local_cells=np.stack([field.mesh.cells for field in definitions]),
        local_basis_matrix=np.stack([field.definition.basis_matrix for field in definitions]),
        local_basis_digest=np.asarray([field.basis_digest for field in definitions]),
    )
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare_reports(original: Path, compiled: Path) -> dict[str, Any]:
    """Validate persisted coordinates before calculating any measured speedup.

    The reports must describe identical cases, execution budgets and physical
    conventions. Comparisons include the actual recorded basis, local geometry,
    full original coefficient vectors and assembled reduced operator. Reported
    field errors remain the independent analytical errors from each acquisition.
    """
    old, new = json.loads(original.read_text()), json.loads(compiled.read_text())
    if old["execution"] != new["execution"]:
        raise ValueError("Execution budgets must match for a kernel speedup")
    if len(old["rows"]) != len(new["rows"]):
        raise ValueError("Acquisitions must contain the same cases")
    rows = []
    for prior, current in zip(old["rows"], new["rows"], strict=True):
        if prior["case"] != current["case"]:
            raise ValueError("Discretizations and physical data must match")
        paths = (
            original.parent / prior["coefficients"]["file"],
            compiled.parent / current["coefficients"]["file"],
        )
        for path, row in zip(paths, (prior, current), strict=True):
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["coefficients"]["sha256"]:
                raise ValueError("Coefficient archive checksum differs")
        with (
            np.load(paths[0], allow_pickle=False) as reference,
            np.load(paths[1], allow_pickle=False) as result,
        ):
            for name in (
                "local_offsets",
                "global_indices",
                "global_indptr",
                "local_points",
                "local_cells",
                "local_basis_matrix",
                "local_basis_digest",
            ):
                np.testing.assert_array_equal(result[name], reference[name])
            differences = {}
            for name in ("trace", "pressure", "global_values", "global_load"):
                np.testing.assert_allclose(result[name], reference[name], rtol=1e-10, atol=1e-11)
                difference = result[name] - reference[name]
                differences[name] = {
                    "absolute_linf": float(np.max(abs(difference))),
                    "relative_l2": float(
                        np.linalg.norm(difference)
                        / max(np.linalg.norm(reference[name]), np.finfo(float).tiny)
                    ),
                }
        rows.append(
            {
                "key": prior["key"],
                "case": prior["case"],
                "original_first_seconds": prior["first"]["complete_seconds"],
                "compiled_first_seconds": current["first"]["complete_seconds"],
                "original_warm_median_seconds": prior["warm_median_seconds"],
                "compiled_warm_median_seconds": current["warm_median_seconds"],
                "complete_warm_speedup": prior["warm_median_seconds"]
                / current["warm_median_seconds"],
                "coefficient_differences": differences,
                "micro_speedups": {
                    name: prior["microbenchmarks"][name]["warm_median_seconds"]
                    / current["microbenchmarks"][name]["warm_median_seconds"]
                    for name in (
                        "global_reduction",
                        "shared_global_reduction",
                        "scalar_diffusion_gram",
                    )
                },
            }
        )
    return {
        "schema": "pymhm-numba-paired-comparison-v1",
        "original_report_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "compiled_report_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
        "rows": rows,
    }


def _time_calls(function: Any, repeats: int) -> dict[str, Any]:
    """Keep first-call and actual steady-state samples distinct for one callable."""
    begin = time.perf_counter()
    function()
    first = time.perf_counter() - begin
    durations = []
    for _ in range(repeats):
        begin = time.perf_counter()
        function()
        durations.append(time.perf_counter() - begin)
    return {
        "first_seconds": first,
        "warm_seconds": durations,
        "warm_median_seconds": statistics.median(durations),
    }


def prepared_microbenchmarks(system: Any, *, repeats: int = 7) -> dict[str, Any]:
    """Time real-response global reduction and a nonzero tensor Gram kernel.

    These microbenchmarks exclude tabulation, coefficient evaluation, local
    solves, field norms and process startup. Their speedups cannot be substituted
    for complete MHM speedups. Full quadrature/tensor inputs are deterministic
    binary64 with variable SPD anisotropy and finite affine gradients.
    """
    offsets = np.asarray(system.kernel_offsets)
    contributions = tuple(
        local_global_contribution(response, np.arange(offsets[cell], offsets[cell + 1]))
        for cell, response in enumerate(system.responses)
    )
    reduce = partial(
        assemble_hybrid_contributions,
        contributions,
        trace_size=system.trace_size,
        kernel_offsets=offsets,
        require_local_trace_coverage=False,
    )
    rng = np.random.default_rng(1729)
    dimension = system.local_metadata[0].points.shape[1]
    cells, count, basis = 384, 36 if dimension == 2 else 64, 6 if dimension == 2 else 10
    gradients = rng.standard_normal((cells, count, basis, dimension))
    factor = rng.standard_normal((cells, count, dimension, dimension))
    tensors = factor @ factor.swapaxes(-1, -2) + np.eye(dimension)
    weights = rng.uniform(0.5, 1.5, (cells, count)) / count
    measures = rng.uniform(0.01, 0.1, cells)
    gram = partial(_scalar_diffusion_blocks, weights, gradients, tensors, measures)
    # A separate shared-index workload exercises boxed-entry reduction costs
    # without presenting its timing as an end-to-end multiscale speedup.
    shared_size, local_size, shared_cells = 128, 25, 1024
    synthetic = []
    for cell in range(shared_cells):
        indices = (np.arange(local_size, dtype=np.int64) + cell) % shared_size
        factor = rng.standard_normal((local_size, local_size))
        synthetic.append(
            (indices, factor @ factor.T + np.eye(local_size), rng.standard_normal(local_size))
        )
    shared_reduce = partial(
        assemble_hybrid_contributions,
        tuple(synthetic),
        trace_size=shared_size,
        kernel_offsets=np.full(shared_cells + 1, shared_size, dtype=np.int64),
    )
    with threadpool_limits(limits=1):
        return {
            "global_reduction": _time_calls(reduce, repeats),
            "shared_global_reduction": _time_calls(shared_reduce, repeats),
            "scalar_diffusion_gram": _time_calls(gram, repeats),
            "shared_reduction_input": {
                "cells": shared_cells,
                "local_dofs": local_size,
                "global_trace_dofs": shared_size,
                "retained_dofs": 0,
            },
            "gram_input": {
                "cells": cells,
                "quadrature_points": count,
                "basis": basis,
                "dimension": dimension,
            },
        }


def acquire(
    cases: tuple[KernelCase, ...],
    output: Path,
    *,
    label: str,
    repeats: int = 3,
    backend: str = "serial",
    workers: int = 1,
) -> dict[str, Any]:
    """Write first/warm complete timings, numerical controls and source identities."""
    positive_int(repeats, "repeats")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = source_manifest()
    report: dict[str, Any] = {
        "schema": "pymhm-numba-first-campaign-v1",
        "label": label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": platform.processor(),
        "numpy": np.__version__,
        "numba": version("numba"),
        "llvmlite": version("llvmlite"),
        "scipy": version("scipy"),
        "basix": version("fenics-basix"),
        "pymhm": pymhm.__version__,
        "lockfile_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "pixi.lock").read_bytes()
        ).hexdigest()
        if (Path(__file__).resolve().parents[1] / "pixi.lock").is_file()
        else None,
        "cpu_count": os.cpu_count(),
        "affinity_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "execution": {
            "backend": backend,
            "workers": workers,
            "native_threads": 1,
            "numba_parallel": False,
        },
        "source_sha256": before,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "NUMBA_NUM_THREADS",
                "NUMBA_CACHE_DIR",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "timing_scope": (
            "fresh geometry, description, assembly, local factors, global assembly/solve, "
            "reconstruction and physical postprocessing"
        ),
        "first_call_scope": (
            "first workload call in this acquisition process; "
            "imports and outer launch reported separately"
        ),
        "rows": [],
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        report["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            report["cpu_model"],
        )
    for case in cases:
        first, system, fields = solve_once(case, backend=backend, workers=workers)
        coefficients = save_coefficients(output.parent / f"{label}-{case.name}.npz", fields, system)
        samples = [solve_once(case, backend=backend, workers=workers)[0] for _ in range(repeats)]
        row = {
            "case": asdict(case),
            "key": case.name,
            "first": first,
            "warm": samples,
            "warm_median_seconds": statistics.median(
                value["complete_seconds"] for value in samples
            ),
            "coefficients": coefficients,
            "microbenchmarks": prepared_microbenchmarks(system),
        }
        report["rows"].append(row)
        print(
            json.dumps(
                {
                    "label": label,
                    "case": case.name,
                    "first_seconds": first["complete_seconds"],
                    "warm_median_seconds": row["warm_median_seconds"],
                }
            ),
            flush=True,
        )
    with threadpool_limits(limits=1):
        keys = ("user_api", "internal_api", "version", "num_threads", "architecture")
        report["native_libraries"] = [
            {key: info.get(key) for key in keys} for info in threadpool_info()
        ]
    if source_manifest() != before:
        raise RuntimeError("Executed numerical sources changed during acquisition")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
