"""Acquire the centered-square finite-well Darcy case with the public solver.

The macrogrid is the fixed 200-triangle partition. Every exterior physical
normal flux vanishes and the physical pressure integral is zero. P1 primal
fields and RT0/P0 fields are separate acquisitions; raw primal flux is not an
H(div) field. This selected finite-well case is distinct from point wells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from math import fsum
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

if __package__:
    from .quarter_spot_problem import coefficient, macro_mesh, source
else:
    from quarter_spot_problem import coefficient, macro_mesh, source

from pymhm import DarcySolution, FaceSpace, SkeletonSpace, solve_darcy
from pymhm.darcy import _DarcyLocalFactory
from pymhm.elements import rt0_evaluate
from pymhm.mesh import positive_int
from pymhm.solvers import _accurate_residual

ROOT = Path(__file__).resolve().parents[1]
BARYCENTRIC = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


@dataclass(frozen=True)
class ObstacleConfiguration:
    """Supported fitted local/trace pair on the fixed finite-well geometry.

    Even refinements fit the square and well interfaces. Trace partitions must
    align with fine edges. For P1, at least two fine edges per trace segment
    provide an interior edge test node supported on that segment, establishing
    injectivity of its normal-density moments independently of corner tests.
    RT0 uses the fine integrated normal moments instead of this P1 condition.
    """

    refinement: int
    segments: int
    formulation: Literal["primal", "mixed"] = "primal"
    quadrature_order: int = 6

    def __post_init__(self) -> None:
        """Reject unresolved material geometry and unsupported finite trace pairs."""
        r = positive_int(self.refinement, "refinement")
        s = positive_int(self.segments, "segments")
        positive_int(self.quadrature_order, "quadrature_order")
        if self.formulation not in ("primal", "mixed"):
            raise ValueError("formulation must be primal or mixed")
        if r % 2 or r % s:
            raise ValueError("even refinement and aligned trace segments are required")
        if self.formulation == "primal" and r < 2 * s:
            raise ValueError("this P1 acquisition requires two fine edges per trace segment")


def _fingerprint(path: Path) -> str:
    """Return the unchanged file digest used by acquisition manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    """Digest the actual contiguous matrix bytes together with shape and dtype."""
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _atomic_archive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Flush one complete NPZ before replacing the named acquisition artifact."""
    temporary = path.with_suffix(".npz.pending")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_record(path: Path, record: dict) -> None:
    """Publish JSON only after its referenced archive has been flushed."""
    temporary = path.with_suffix(".json.pending")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _audit(solution: DarcySolution, config: ObstacleConfiguration) -> tuple[dict, dict]:
    """Reassemble through the shared owner to archive its executed basis and equations."""
    macro = solution.skeleton.mesh
    factory = _DarcyLocalFactory(
        macro,
        solution.skeleton,
        coefficient,
        source,
        tuple(np.empty((0, 3)) for _ in macro.cells),
        config.refinement,
        None,
        1,
        config.formulation,
        solution.quadrature_order,
    )
    kernels, constraints, means, relative, absolute, action_relative = [], [], [], [], [], []
    weak = np.zeros(solution.skeleton.size)
    squared_defect, squared_load = 0.0, 0.0
    for cell, field in enumerate(solution.hybrid.fields):
        assembly = factory(cell)
        problem = assembly.problem
        trace = solution.hybrid.trace[problem.trace_dofs]
        operator = sparse.hstack((problem.matrix, sparse.csr_matrix(problem.coupling))).tocsr()
        defect = _accurate_residual(operator, problem.load, np.r_[field, trace])
        rhs = problem.load - problem.coupling @ trace
        scale = float(np.linalg.norm(rhs))
        norm = float(np.linalg.norm(defect))
        relative.append(norm / scale if scale else 0.0 if not norm else None)
        absolute.append(norm)
        action = abs(operator) @ np.abs(np.r_[field, trace]) + np.abs(problem.load)
        action_scale = float(np.linalg.norm(action))
        action_relative.append(norm / action_scale if action_scale else 0.0 if not norm else None)
        squared_defect += norm**2
        squared_load += float(np.linalg.norm(problem.load)) ** 2
        np.add.at(weak, problem.trace_dofs, problem.test_coupling.T @ field)
        means.append(float(assembly.metadata[1] @ field))
        kernels.append(problem.kernel)
        constraints.append(problem.constraints)
        del assembly, problem, operator
    basis = np.stack(kernels)
    mean_rows = np.stack(constraints)
    free = np.ones(solution.skeleton.size, dtype=bool)
    for face in macro.boundary_faces:
        free[solution.skeleton.dofs(int(face))] = False
    checks = {
        "original_local_relative_rhs_residual": relative,
        "original_local_absolute_residual": absolute,
        "original_local_relative_action_residual": action_relative,
        "original_free_equations_relative_load_residual": float(
            np.sqrt((squared_defect + np.linalg.norm(weak[free]) ** 2) / squared_load)
        ),
        "weak_pressure_continuity_linf": float(np.max(np.abs(weak[free]))),
        "macro_conservation_linf": float(np.max(np.abs(solution.conservation_residuals()))),
        "physical_pressure_integral": fsum(means),
        "exterior_trace_linf": float(
            max(
                np.max(np.abs(solution.hybrid.trace[solution.skeleton.dofs(int(face))]))
                for face in macro.boundary_faces
            )
        ),
        "fine_cell_conservation_linf": (
            float(max(np.max(np.abs(value)) for value in solution.fine_conservation_residuals()))
            if config.formulation == "mixed"
            else None
        ),
    }
    return checks, {
        "executed_kernel": basis,
        "executed_constraints": mean_rows,
        "physical_macro_pressure_integrals": np.array(means),
    }


def acquire(
    config: ObstacleConfiguration,
    directory: Path,
    *,
    solver: str = "scipy",
    local_solver: str = "scipy",
    native_threads: int = 1,
    backend: Literal["serial", "thread", "process"] = "serial",
    workers: int = 1,
) -> dict:
    """Execute and persist complete original P1 or RT0 fields with physical checks.

    The archive contains the supplied local kernel/constraint matrices and the
    full final coefficients. It does not claim restart or coarse-only replay.
    No exact solution or reference accuracy is inferred from algebraic checks.
    """
    positive_int(native_threads, "native_threads")
    positive_int(workers, "workers")
    directory.mkdir(parents=True, exist_ok=True)
    macro = macro_mesh()
    skeleton = SkeletonSpace(
        macro, tuple(FaceSpace.uniform(0, config.segments) for _ in macro.faces)
    )
    started = perf_counter()
    with threadpool_limits(native_threads):
        solution = solve_darcy(
            macro,
            skeleton=skeleton,
            permeability=coefficient,
            source=source,
            neumann={int(face): 0.0 for face in macro.boundary_faces},
            mean_pressure=0.0,
            local_refinement=config.refinement,
            quadrature_order=config.quadrature_order,
            formulation=config.formulation,
            solver=solver,
            local_solver=local_solver,
            backend=backend,
            workers=workers,
            parallel_assembly=True,
        )
        checks, basis = _audit(solution, config)
        pools = [
            {key: value for key, value in entry.items() if key != "filepath"}
            for entry in threadpool_info()
        ]
    points = np.vstack([mesh.points for mesh in solution.local_meshes])
    offsets = np.cumsum([0, *[len(mesh.points) for mesh in solution.local_meshes]])
    cells = np.vstack([mesh.cells + offsets[i] for i, mesh in enumerate(solution.local_meshes)])
    pressure = np.concatenate(solution.pressure)
    flux = np.vstack(
        [
            np.repeat(value[:, None, :], 3, axis=1)
            if config.formulation == "primal"
            else rt0_evaluate(mesh, value, BARYCENTRIC)
            for mesh, value in zip(solution.local_meshes, solution.flux, strict=True)
        ]
    )
    areas = np.concatenate([mesh.areas for mesh in solution.local_meshes])
    centers = points[cells].mean(axis=1)
    checks.update(
        obstacle_measured_area=fsum(areas[coefficient(centers) == 1e-4]),
        extraction_integral=fsum(areas * np.minimum(source(centers), 0)),
        injection_integral=fsum(areas * np.maximum(source(centers), 0)),
    )
    archive = directory / f"mhm-{config.formulation}-r{config.refinement}-s{config.segments}.npz"
    arrays = dict(
        points=points,
        cells=cells,
        macro_points=macro.points,
        macro_cells=macro.cells,
        macro_faces=macro.faces,
        macro_cell_faces=macro.cell_faces,
        macro_normals=macro.normals,
        macro_cell=np.repeat(np.arange(len(macro.cells)), config.refinement**2),
        pressure=pressure,
        flux_quadrature=flux,
        quadrature_barycentric=BARYCENTRIC,
        quadrature_weights=np.full(3, 1 / 3),
        permeability=coefficient(centers),
        source_density=source(centers),
        local_fields=np.stack(solution.hybrid.fields),
        trace=solution.hybrid.trace,
        coarse=np.stack(solution.hybrid.coarse),
        macro_signs=macro.signs,
        **basis,
    )
    _atomic_archive(archive, arrays)
    sources = [
        Path(__file__),
        ROOT / "examples/quarter_spot_problem.py",
        *sorted((ROOT / "src/pymhm").glob("*.py")),
    ]
    record = {
        "schema": "pymhm-quarter-obstacle-v1",
        "configuration": asdict(config),
        "executed_quadrature_order": solution.quadrature_order,
        "macro_cells": len(macro.cells),
        "fine_cells": len(cells),
        "pressure_basis": "independent local P1 nodal values"
        if config.formulation == "primal"
        else "independent fine-cell P0 values",
        "flux_convention": "physical Darcy flux -K grad(p); raw, not H(div)"
        if config.formulation == "primal"
        else "physical H(div) RT0 Darcy flux, evaluated at three degree-two quadrature points",
        "boundary": "zero physical outward flux on every exterior macroface",
        "trace_convention": (
            "P0 normal density per ordered uniform segment on each archived macroface; "
            "normal is outward from its first neighboring cell, with archived local signs"
        ),
        "gauge": "physical pressure integral zero",
        "well_convention": (
            "finite square supports [0,0.1]^2 and [0.9,1]^2, integrated strengths -1/+1; "
            "not Dirac wells"
        ),
        "basis_convention": (
            "archived supplied constant-pressure kernel and normalized physical pressure-mass "
            "constraints; final fields archived independently of coarse-only replay"
        ),
        "basis_sha256": _array_digest(basis["executed_kernel"]),
        "constraints_sha256": _array_digest(basis["executed_constraints"]),
        "relative_condensed_residual": float(solution.hybrid.residual),
        "physical_checks": checks,
        "archive": archive.name,
        "archive_sha256": _fingerprint(archive),
        "source_sha256": {str(path.relative_to(ROOT)): _fingerprint(path) for path in sources},
        "lockfile_sha256": _fingerprint(ROOT / "pixi.lock"),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
        ),
        "coefficient_dtype": pressure.dtype.str,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "native_threads_requested": native_threads,
        "native_libraries": pools,
        "backend": backend,
        "workers": workers,
        "solver": solver,
        "local_solver": local_solver,
        "elapsed_seconds": perf_counter() - started,
        "scope": (
            "Selected finite-well square obstacle; physical fields and basis archived. "
            "Classical refinement and independent same-case assembly are separate "
            "acceptance checks."
        ),
    }
    _atomic_record(archive.with_suffix(".json"), record)
    return record


def load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read final fields only with matching case, matrix basis and orientation contracts.

    Nodal primal fields and cellwise mixed pressures retain their different
    coefficient spaces. Reading archived fields does not reconstruct omitted
    local responses or establish reference accuracy.
    """
    record = json.loads(path.with_suffix(".json").read_text())
    config = ObstacleConfiguration(**record["configuration"])
    expected_sources = ("examples/quarter_spot_problem.py", "examples/solve_quarter_obstacle.py")
    if (
        record["schema"] != "pymhm-quarter-obstacle-v1"
        or record["archive"] != path.name
        or record["archive_sha256"] != _fingerprint(path)
        or any(
            record["source_sha256"].get(name) != _fingerprint(ROOT / name)
            for name in expected_sources
        )
    ):
        raise ValueError("quarter-obstacle acquisition or physical case provenance mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    macro = macro_mesh()
    kernel, constraints = arrays["executed_kernel"], arrays["executed_constraints"]
    count = len(macro.cells) * config.refinement**2
    pressure_size = len(arrays["points"]) if config.formulation == "primal" else count
    if (
        not all(np.isrealobj(value) and np.isfinite(value).all() for value in arrays.values())
        or arrays["pressure"].shape != (pressure_size,)
        or arrays["cells"].shape != (count, 3)
        or arrays["flux_quadrature"].shape != (count, 3, 2)
        or kernel.shape != constraints.shape
        or kernel.shape[0] != len(macro.cells)
        or kernel.shape[-1] != 1
        or arrays["local_fields"].shape != kernel.shape[:-1]
        or arrays["coarse"].shape != (len(macro.cells), 1)
        or arrays["trace"].shape != (len(macro.faces) * config.segments,)
        or _array_digest(kernel) != record["basis_sha256"]
        or _array_digest(constraints) != record["constraints_sha256"]
        or not np.array_equal(arrays["macro_points"], macro.points)
        or not np.array_equal(arrays["macro_cells"], macro.cells)
        or not np.array_equal(arrays["macro_faces"], macro.faces)
        or not np.array_equal(arrays["macro_cell_faces"], macro.cell_faces)
        or not np.array_equal(arrays["macro_normals"], macro.normals)
        or not np.array_equal(arrays["macro_signs"], macro.signs)
    ):
        raise ValueError("quarter-obstacle fields, executed basis or trace orientation mismatch")
    return arrays, record


def main() -> None:
    """Acquire requested fixed-macro refinements and distinct primal/mixed formulations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinements", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--segments", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--formulations", choices=["primal", "mixed"], nargs="+", default=["primal"]
    )
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "build/results/quarter-obstacle")
    parser.add_argument("--solver", default="scipy")
    parser.add_argument("--local-solver", default="scipy")
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--backend", choices=["serial", "thread", "process"], default="serial")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    configs = [
        ObstacleConfiguration(r, s, form, args.order)
        for form in args.formulations
        for r in args.refinements
        for s in args.segments
    ]
    for config in configs:
        record = acquire(
            config,
            args.output,
            solver=args.solver,
            local_solver=args.local_solver,
            native_threads=args.native_threads,
            backend=args.backend,
            workers=args.workers,
        )
        checks = record["physical_checks"]
        print(
            json.dumps(
                {
                    "configuration": record["configuration"],
                    "relative_condensed_residual": record["relative_condensed_residual"],
                    "original_free_equations_relative_load_residual": checks[
                        "original_free_equations_relative_load_residual"
                    ],
                    "macro_conservation_linf": checks["macro_conservation_linf"],
                    "physical_pressure_integral": checks["physical_pressure_integral"],
                    "elapsed_seconds": record["elapsed_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
