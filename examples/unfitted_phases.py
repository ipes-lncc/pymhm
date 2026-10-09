"""Bounded-memory acquisition of the printed smooth unfitted trace study.

The sixteen crisscross macrotriangles carry continuous equidistant Pk local
fields and discontinuous Legendre traces. A first pass archives compact MHM
contributions and the executed retained basis. A second pass solves only the
actual combined forcing columns through the shared LocalProblem owner. Shared
streamed refinement checks and, when required, corrects the original equations.
The executed initial fields and ordered physical increments are archived;
replay does not repeat their conditional solves. All original volume and
weak-trace rows are checked before publishing fields.
Local refinement and numerical quadrature are explicit controls; the paper
does not specify its finite local approximation or integration rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast
from uuid import uuid4

import numpy as np
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info, threadpool_limits

from examples.archive_precision import precision_fields, restore_precision
from examples.formulations.local_records import DarcyLocalFactory as _DarcyLocalFactory
from examples.unfitted_convergence import (
    ROOT,
    error_norms,
    smooth_configurations,
    smooth_field,
    smooth_source,
    source_hashes,
)
from examples.unfitted_geometry import macro_mesh
from examples.unfitted_trace_family import nested_trace_injection
from pymhm.core.contracts import HybridSolution, LocalProblem
from pymhm.core.refinement import (
    HybridRefinementCase,
    HybridRefinementLocal,
    HybridStreamRefinement,
    refine_hybrid_stream,
)
from pymhm.core.subspaces import restrict_response
from pymhm.core.system import HybridSystem
from pymhm.core.validation import positive_int
from pymhm.fem.quadrature.orders import nodal_quadrature_order as _assembly_quadrature_order
from pymhm.fem.scalar.triangle import multiindices, nodal_space
from pymhm.fem.traces.interval import FaceSpace, SkeletonSpace
from pymhm.io.provenance import current_source_manifest, optional_file_digest
from pymhm.io.workspace import source_file, source_identity
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError, accurate_residual
from pymhm.meshes.triangle import TriangleMesh


def fingerprint(path: Path) -> str:
    """Hash source or persisted archive bytes without materializing the file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core_source_hashes() -> dict[str, str]:
    """Identify the complete portable-core generation, including trace restriction."""
    return current_source_manifest(
        source_identity(
            ROOT,
            (
                path
                for path in sorted(
                    source_file("src/pymhm/__init__.py", root=ROOT).parent.rglob("*.py")
                )
                if path.name not in ("pixi.lock", "pixi.toml", "pyproject.toml") or path.is_file()
            ),
        ),
        packages=("pymhm", "examples"),
    )


def array_digest(value: np.ndarray) -> str:
    """Identify numerical entries, shape and dtype, excluding long-double padding."""
    array = np.asarray(value)
    digest = hashlib.sha256((array.dtype.str + str(array.shape)).encode("ascii"))
    parts = precision_fields("value", array).values() if array.dtype.itemsize > 8 else (array,)
    for part in parts:
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


def _matrix_digest(matrix: Any) -> str:
    """Identify the canonical sparse operator used in both local passes."""
    matrix = matrix.tocsc(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return hashlib.sha256(
        (
            str(matrix.shape)
            + array_digest(matrix.indptr)
            + array_digest(matrix.indices)
            + array_digest(matrix.data)
        ).encode("ascii")
    ).hexdigest()


def _json(path: Path, record: dict) -> None:
    """Flush an atomic manifest replacement after all referenced bytes exist."""
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _npz(path: Path, **arrays: Any) -> None:
    """Publish complete numerical arrays using an atomic flushed replacement."""
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _peak_rss_kib() -> float | None:
    """Return cumulative peak RSS in KiB where the native resource module exists."""
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 if sys.platform == "darwin" else value


def _restore(arrays: dict[str, np.ndarray], name: str, dtype: np.dtype) -> np.ndarray:
    """Restore portable components in the acquisition's declared precision."""
    return restore_precision(
        arrays[name], arrays[f"{name}_correction"], arrays[f"{name}_tail"]
    ).astype(dtype)


@dataclass
class _NormFields:
    """Only original broken fields and material are needed by the shared norms."""

    local_meshes: tuple[TriangleMesh, ...]
    pressure: tuple[np.ndarray, ...]
    degree: int
    permeability: float = 1.0


class _RefinementStore:
    """Persist executed original-equation arrays in portable acquired precision."""

    def __init__(
        self, acquisition: UnfittedAcquisition, fields: dict, *, check_only: bool = False
    ) -> None:
        """Bind the acquisition and ordered cell fields to a local correction directory."""
        self.acquisition, self.fields = acquisition, fields
        self.directory = acquisition.cell_directory / "original-refinement"
        self.directory.mkdir(exist_ok=True)
        self.records: dict[str, Any] = {}
        self.check_only = check_only
        self.last: tuple[str, np.ndarray] | None = None

    def read_fields(self, cell: int) -> np.ndarray:
        """Read one bounded batch in the declared case and coefficient ordering."""
        return np.column_stack([field[cell] for field in self.fields.values()])

    def write_fields(self, cell: int, fields: np.ndarray) -> None:
        """Retain every executed coefficient digit in transient field storage."""
        for column, field in enumerate(self.fields.values()):
            field[cell] = fields[:, column]

    def write_record(self, name: str, step: int, cell: int, values: np.ndarray) -> None:
        """Atomically archive the executed source, forcing or physical increment."""
        key = f"{name}-{step}-cell-{cell}"
        if self.check_only:
            self.last = (key, values.copy())
            return
        path = self.directory / f"{key}.npz"
        _npz(
            path,
            **precision_fields("values", values),
            acquisition_id=self.acquisition.record["acquisition_id"],
            case_order=np.asarray(list(self.fields)),
        )
        self.records[key] = dict(
            archive=path.name,
            sha256=fingerprint(path),
            values_sha256=array_digest(values),
            case_values_sha256=current_source_manifest(
                {case: array_digest(values[:, column]) for column, case in enumerate(self.fields)},
                packages=("pymhm", "examples"),
            ),
        )

    def read_record(self, name: str, step: int, cell: int) -> np.ndarray:
        """Read actual acquired arrays without repeating their conditional solves."""
        key = f"{name}-{step}-cell-{cell}"
        if self.check_only:
            if self.last is None or self.last[0] != key:
                raise ValueError("unfitted check store has no matching transient record")
            return self.last[1]
        record = self.records[key]
        if Path(record["archive"]).name != record["archive"]:
            raise ValueError("unfitted correction archive must remain within its acquisition")
        arrays = self.acquisition._read(self.directory / record["archive"], record["sha256"])
        if str(arrays["acquisition_id"]) != self.acquisition.record[
            "acquisition_id"
        ] or not np.array_equal(arrays["case_order"], list(self.fields)):
            raise ValueError("unfitted correction acquisition or case ordering mismatch")
        values = _restore(arrays, "values", self.acquisition.dtype)
        if array_digest(values) != record["values_sha256"]:
            raise ValueError("unfitted correction precision or numerical digest mismatch")
        return values


class UnfittedAcquisition:
    """Acquire one immutable smooth trace family with at most one local factor.

    Exterior pressure is weak homogeneous Dirichlet, so there is no artificial
    gauge row. Trace coordinates are physical flux normal to the first adjacent
    macrotriangle. Local coefficients use the declared barycentric cardinal
    ordering. The constant kernel has unit physical mean; its executed corrected
    retained matrix E is archived with the coefficients and used during replay.
    """

    def __init__(
        self,
        directory: Path,
        *,
        refinement: int,
        names: list[str] | tuple[str, ...],
        degree: int = 8,
        assembly_order: int = 13,
        norm_orders: list[int] | tuple[int, ...] = (13, 15),
        native_threads: int = 1,
        local_solver: str = "scipy",
        refinement_precision: Literal["double", "extended"] = "extended",
        original_refinement_steps: int = 2,
    ) -> None:
        """Declare spaces, Gauss counts, backend and coefficient precision.

        Assembly order counts Gauss points per Duffy coordinate, with Darcy's
        degree+2 floor. Trace coupling retains its separate exact polynomial
        rule. Explicit extended precision preserves forcing/correction digits;
        native factors and operators remain double precision. No tolerance is
        changed by the phase or precision choices.
        """
        if refinement_precision not in {"double", "extended"}:
            raise ValueError("refinement_precision must be double or extended")
        if (
            isinstance(original_refinement_steps, (bool, np.bool_))
            or not isinstance(original_refinement_steps, (int, np.integer))
            or original_refinement_steps < 0
        ):
            raise ValueError("original_refinement_steps must be a nonnegative integer")
        if (
            refinement_precision == "extended"
            and np.finfo(np.longdouble).eps >= np.finfo(float).eps
        ):
            raise SolverUnavailableError("extended refinement requires a wider long-double type")
        refinement, degree = positive_int(refinement, "refinement"), positive_int(degree, "degree")
        norms = tuple(sorted(positive_int(order, "norm_orders") for order in norm_orders))
        if not norms or len(norms) != len(set(norms)):
            raise ValueError("norm_orders must be nonempty distinct Gauss counts")
        available = {f"ell{ell}-s{s}": (ell, s) for ell, s in smooth_configurations(32)}
        if not names or len(names) != len(set(names)) or not set(names) <= available.keys():
            raise ValueError("names must be distinct printed smooth trace configurations")
        self.cases = {name: available[name] for name in names}
        self.mesh = macro_mesh(0)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dtype: np.dtype[Any] = np.dtype(
            np.longdouble if refinement_precision == "extended" else float
        )
        self.precision = refinement_precision
        self.native_threads = positive_int(native_threads, "native_threads")
        self.config: dict[str, Any] = dict(
            study="smooth",
            local_degree=degree,
            local_refinement=refinement,
            requested_names=list(names),
            requested_assembly_order=positive_int(assembly_order, "assembly_order"),
            assembly_order=_assembly_quadrature_order(degree, assembly_order),
            norm_orders=list(norms),
            local_solver=local_solver,
            local_refinement_precision=refinement_precision,
            original_refinement_steps=int(original_refinement_steps),
            coefficient_dtype=self.dtype.str,
            coefficient_precision_bits=np.finfo(self.dtype).nmant + 1,
            material_fitted_local_meshes=False,
            source="8*pi^2*sin(2*pi*x)*sin(2*pi*y)",
            coefficient="identity",
            boundary="weak homogeneous Dirichlet pressure",
            physical_flux="-grad(p)",
        )
        self.prefix = f"smooth-p{degree}-r{refinement}-q{self.config['assembly_order']}-phases"
        self.path = self.directory / f"{self.prefix}.json"
        self.cell_directory = self.directory / f"{self.prefix}-cells"
        self.cell_directory.mkdir(exist_ok=True)
        if self.path.exists():
            self.record: dict[str, Any] = json.loads(self.path.read_text())
            if (
                self.record.get("schema") != "pymhm-unfitted-phases-v2"
                or self.record["configuration"] != self.config
            ):
                raise ValueError("unfitted phase manifest configuration mismatch")
            self._current_sources()
        else:
            self.record = dict(
                schema="pymhm-unfitted-phases-v2",
                acquisition_id=str(uuid4()),
                configuration=self.config,
                source_sha256=source_hashes(),
                core_source_sha256=core_source_hashes(),
                phase_source_sha256=fingerprint(Path(__file__)),
                lockfile_sha256=optional_file_digest(ROOT / "pixi.lock"),
                source_changed_during_run=False,
                paper="Chaumont-Frelet, Paredes, Valentin, CAMWA 209 (2026), Section 6.1",
                doi="10.1016/j.camwa.2026.01.016",
                primary_error="absolute broken H1 seminorm of pressure error",
                basis_convention=(
                    "equidistant barycentric cardinal nodes in archived "
                    "multiindex/dof order; archived executed retained E"
                ),
                trace_convention=(
                    "discontinuous Legendre pieces, face parameter from the mesh's "
                    "first endpoint, physical flux normal to first adjacent "
                    "macrotriangle"
                ),
                execution=dict(
                    created_utc=datetime.now(UTC).isoformat(),
                    python=sys.version.split()[0],
                    numpy=np.__version__,
                    scipy=scipy.__version__,
                    platform=platform.platform(),
                    host=platform.node(),
                    host_exclusive=False,
                    memory_convention=(
                        "native ru_maxrss normalized to KiB where available, null "
                        "otherwise; cumulative process maximum includes assembly, solve, "
                        "norms and writes"
                    ),
                    synchronization=(
                        "synchronous CPU operations; archives fsynced before manifest replacement"
                    ),
                ),
                cells={},
                solutions={},
                fields={},
                cases=[],
                complete=False,
            )
            self._checkpoint()

    def _current_sources(self) -> None:
        """Reject replay across numerical sources, phase owner or lock generations."""
        if (
            self.record["source_sha256"] != source_hashes()
            or self.record["core_source_sha256"] != core_source_hashes()
            or self.record["phase_source_sha256"] != fingerprint(Path(__file__))
            or self.record["lockfile_sha256"] != optional_file_digest(ROOT / "pixi.lock")
        ):
            raise ValueError("unfitted acquisition source or lockfile mismatch")

    def _checkpoint(self) -> None:
        """Check identities before committing a completed stage or cell."""
        self._current_sources()
        self.record["execution"]["peak_rss_kib"] = _peak_rss_kib()
        _json(self.path, self.record)

    def _skeleton(self, ell: int, segments: int) -> SkeletonSpace:
        """Use the shared face numbering, parameter direction and physical normals."""
        return SkeletonSpace(
            self.mesh, tuple(FaceSpace.uniform(ell, segments) for _ in self.mesh.faces)
        )

    def _prepared(self) -> SkeletonSpace:
        """Prepare the smallest common printed polynomial trace partition."""
        return self._skeleton(
            max(ell for ell, _ in self.cases.values()), max(s for _, s in self.cases.values())
        )

    def _factory(self, skeleton: SkeletonSpace) -> _DarcyLocalFactory:
        """Reuse the original Darcy owner for both local assembly passes."""
        return _DarcyLocalFactory(
            self.mesh,
            skeleton,
            1.0,
            smooth_source,
            tuple(np.empty((0, 3)) for _ in self.mesh.cells),
            self.config["local_refinement"],
            None,
            self.config["local_degree"],
            "primal",
            self.config["assembly_order"],
        )

    def _injection(
        self, prepared: SkeletonSpace, requested: SkeletonSpace, cell: int
    ) -> np.ndarray:
        """Embed every requested Legendre trace exactly on its prepared macrofaces."""
        return sparse.block_diag(
            [
                sparse.csc_matrix(
                    nested_trace_injection(prepared.faces[face], requested.faces[face])
                )
                for face in self.mesh.cell_faces[cell]
            ]
        ).toarray()

    def _read(self, path: Path, expected: str) -> dict[str, np.ndarray]:
        """Load one complete digest-verified archive and close its descriptor."""
        if fingerprint(path) != expected:
            raise ValueError("unfitted archive digest mismatch")
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def _cell(self, cell: int) -> dict[str, np.ndarray]:
        """Verify the executed nodal, kernel and retained-coordinate data contract."""
        row = self.record["cells"].get(str(cell))
        if row is None:
            raise ValueError("unfitted condensation is incomplete")
        arrays = self._read(self.cell_directory / f"cell-{cell}.npz", row["sha256"])
        retained = _restore(arrays, "retained_basis", self.dtype)
        if (
            array_digest(retained) != row["retained_basis_sha256"]
            or array_digest(arrays["kernel"]) != row["kernel_sha256"]
        ):
            raise ValueError("unfitted executed retained basis or kernel mismatch")
        return arrays

    def condense(self) -> None:
        """Archive compact contributions, releasing each macro's volumetric lifts."""
        self._current_sources()
        with threadpool_limits(self.native_threads):
            self.record["condensation_threadpools"] = [
                {k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()
            ]
            prepared = self._prepared()
            factory = self._factory(prepared)
            for cell in range(len(self.mesh.cells)):
                if str(cell) in self.record["cells"]:
                    self._cell(cell)
                    continue
                started = perf_counter()
                assembly = factory(cell)
                response = assembly.problem.condense(
                    solver=self.config["local_solver"], refinement_precision=self.precision
                )
                problem, fine = response.problem, assembly.metadata[0]
                if not np.array_equal(problem.kernel, np.ones_like(problem.kernel)):
                    raise ValueError("unfitted constant kernel must have unit physical mean")
                dofs, nodes = nodal_space(fine, self.config["local_degree"])
                arrays = dict(
                    kernel=problem.kernel,
                    constraints=problem.constraints,
                    **precision_fields("retained_basis", response.retained_basis),
                    physical_mean=assembly.metadata[1],
                    points=fine.points,
                    cells=fine.cells,
                    nodal_dofs=dofs,
                    nodal_points=nodes,
                    multiindices=multiindices(self.config["local_degree"]),
                    signs=self.mesh.signs[cell],
                    faces=self.mesh.cell_faces[cell],
                )
                for name, (ell, segments) in self.cases.items():
                    skeleton = self._skeleton(ell, segments)
                    injection = self._injection(prepared, skeleton, cell)
                    narrowed = restrict_response(response, injection, skeleton.cell_dofs(cell))
                    indices, block, rhs = narrowed.global_contribution(
                        np.array([skeleton.size + cell])
                    )
                    arrays.update(
                        {
                            f"{key}_{name}": value
                            for key, value in (
                                ("indices", indices),
                                ("block", block),
                                ("rhs", rhs),
                                ("trace_dofs", skeleton.cell_dofs(cell)),
                                ("injection", injection),
                            )
                        }
                    )
                    del narrowed, injection
                path = self.cell_directory / f"cell-{cell}.npz"
                _npz(path, **arrays)
                self.record["cells"][str(cell)] = dict(
                    sha256=fingerprint(path),
                    kernel_sha256=array_digest(problem.kernel),
                    retained_basis_sha256=array_digest(response.retained_basis),
                    matrix_sha256=_matrix_digest(problem.matrix),
                    load_sha256=array_digest(problem.load),
                    coupling_sha256=array_digest(problem.coupling),
                    native_threads=self.native_threads,
                    seconds=perf_counter() - started,
                    pressure_dofs=problem.matrix.shape[0],
                    prepared_trace_dofs=problem.coupling.shape[1],
                )
                self._checkpoint()
                print(f"unfitted condensed macro {cell + 1}/{len(self.mesh.cells)}", flush=True)
                del assembly, response, problem, fine, arrays, dofs, nodes

    def solve(self) -> None:
        """Solve each compact MHM system without volumetric responses in memory."""
        self._current_sources()
        with threadpool_limits(self.native_threads):
            self.record["solution_threadpools"] = [
                {k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()
            ]
            for name, (ell, segments) in self.cases.items():
                if name in self.record["solutions"]:
                    self._solution(name)
                    continue
                started = perf_counter()
                contributions = []
                for cell in range(len(self.mesh.cells)):
                    arrays = self._cell(cell)
                    contributions.append(
                        (arrays[f"indices_{name}"], arrays[f"block_{name}"], arrays[f"rhs_{name}"])
                    )
                system = HybridSystem.from_contributions(
                    contributions,
                    trace_size=self._skeleton(ell, segments).size,
                    coarse_sizes=[1] * len(self.mesh.cells),
                )
                result = system.solve(refinement_precision=self.precision)
                path = self.directory / f"{self.prefix}-{name}-coefficients.npz"
                _npz(
                    path,
                    **precision_fields("trace", result.trace),
                    **precision_fields("coarse", np.array(result.coarse)),
                    acquisition_id=self.record["acquisition_id"],
                    kernel_offsets=system.kernel_offsets,
                )
                self.record["solutions"][name] = dict(
                    archive=path.name,
                    sha256=fingerprint(path),
                    residual=result.residual,
                    seconds=perf_counter() - started,
                    matrix_sha256=_matrix_digest(system.matrix),
                    rhs_sha256=array_digest(system.rhs),
                    native_threads=self.native_threads,
                )
                self._checkpoint()

    def _solution(self, name: str) -> dict[str, np.ndarray]:
        """Verify trace/coarse coefficients belong to this exact acquisition."""
        row = self.record["solutions"].get(name)
        if row is None:
            raise ValueError("unfitted global solution is incomplete")
        if Path(row["archive"]).name != row["archive"]:
            raise ValueError("unfitted coefficient archive must stay within its acquisition")
        arrays = self._read(self.directory / row["archive"], row["sha256"])
        if str(arrays["acquisition_id"]) != self.record["acquisition_id"]:
            raise ValueError("unfitted coefficients belong to a different acquisition")
        return {
            "trace": _restore(arrays, "trace", self.dtype),
            "coarse": _restore(arrays, "coarse", self.dtype),
            "residual": np.asarray(row["residual"]),
        }

    def _restored_local(self, cell: int) -> HybridRefinementLocal:
        """Restore executed operators, cardinal nodes, retained E and oriented maps."""
        arrays = self._cell(cell)
        prepared = self._prepared()
        with threadpool_limits(self.record["cells"][str(cell)]["native_threads"]):
            assembly = self._factory(prepared)(cell)
        original, fine = assembly.problem, assembly.metadata[0]
        row = self.record["cells"][str(cell)]
        dofs, nodes = nodal_space(fine, self.config["local_degree"])
        if (
            _matrix_digest(original.matrix) != row["matrix_sha256"]
            or array_digest(original.load) != row["load_sha256"]
            or array_digest(original.coupling) != row["coupling_sha256"]
            or any(
                not np.array_equal(arrays[key], value)
                for key, value in (
                    ("points", fine.points),
                    ("cells", fine.cells),
                    ("nodal_dofs", dofs),
                    ("nodal_points", nodes),
                    ("kernel", original.kernel),
                    ("constraints", original.constraints),
                    ("signs", self.mesh.signs[cell]),
                    ("faces", self.mesh.cell_faces[cell]),
                )
            )
        ):
            raise ValueError("unfitted replay operator, nodal basis or orientation mismatch")
        problem = LocalProblem(
            original.matrix,
            original.coupling,
            original.load,
            original.trace_dofs,
            kernel=arrays["kernel"],
            constraints=arrays["constraints"],
        )
        maps = []
        for name, configuration in self.cases.items():
            requested = self._skeleton(*configuration)
            indices, injection = arrays[f"trace_dofs_{name}"], arrays[f"injection_{name}"]
            if not np.array_equal(indices, requested.cell_dofs(cell)) or not np.array_equal(
                injection, self._injection(prepared, requested, cell)
            ):
                raise ValueError("unfitted replay trace map mismatch")
            maps.append((indices, injection))
        return HybridRefinementLocal(problem, _restore(arrays, "retained_basis", self.dtype), maps)

    def _compact(self, name: str) -> HybridSystem:
        """Restore the fixed compact operator used by the original global solve."""
        contributions = []
        for cell in range(len(self.mesh.cells)):
            arrays = self._cell(cell)
            contributions.append(
                tuple(arrays[f"{key}_{name}"] for key in ("indices", "block", "rhs"))
            )
        system = HybridSystem.from_contributions(
            contributions,
            trace_size=self._skeleton(*self.cases[name]).size,
            coarse_sizes=[1] * len(self.mesh.cells),
        )
        row = self.record["solutions"][name]
        if (
            _matrix_digest(system.matrix) != row["matrix_sha256"]
            or array_digest(system.rhs) != row["rhs_sha256"]
        ):
            raise ValueError("unfitted executed condensed operator or load mismatch")
        return system

    def _refined_coordinates(self, name: str, result: HybridStreamRefinement) -> dict:
        """Atomically archive actual ordered increments separately from initial coordinates."""
        solution = result.solution
        arrays = dict(
            **precision_fields("trace", solution.trace),
            **precision_fields("coarse", np.array(solution.coarse)),
            residual=solution.residual,
            acquisition_id=self.record["acquisition_id"],
        )
        for step, delta in enumerate(result.increments):
            arrays.update(precision_fields(f"delta_trace_{step}", delta.trace))
            arrays.update(precision_fields(f"delta_coarse_{step}", np.array(delta.coarse)))
        path = self.directory / f"{self.prefix}-{name}-refinement.npz"
        _npz(path, **arrays)
        return dict(
            archive=path.name,
            sha256=fingerprint(path),
            initial_archive=self.record["solutions"][name]["archive"],
            initial_sha256=self.record["solutions"][name]["sha256"],
            original_residual_norms=list(result.residual_norms),
            original_rhs_norm=result.rhs_norm,
            corrections=len(result.increments),
            local_contract_digests=list(result.local_contract_digests),
            trace_sha256=array_digest(solution.trace),
            coarse_sha256=array_digest(np.array(solution.coarse)),
        )

    def _refined_solution(self, name: str, initial: dict) -> dict:
        """Verify final coordinates equal the archived ordered executed increments."""
        row = self.record["refinement"]["cases"][name]
        if (
            row["initial_sha256"] != self.record["solutions"][name]["sha256"]
            or Path(row["archive"]).name != row["archive"]
        ):
            raise ValueError("unfitted refinement coordinate acquisition mismatch")
        arrays = self._read(self.directory / row["archive"], row["sha256"])
        trace, coarse = initial["trace"].copy(), initial["coarse"].copy()
        for step in range(row["corrections"]):
            trace += _restore(arrays, f"delta_trace_{step}", self.dtype)
            coarse += _restore(arrays, f"delta_coarse_{step}", self.dtype)
        if (
            str(arrays["acquisition_id"]) != self.record["acquisition_id"]
            or not np.array_equal(trace, _restore(arrays, "trace", self.dtype))
            or not np.array_equal(coarse, _restore(arrays, "coarse", self.dtype))
            or array_digest(trace) != row["trace_sha256"]
            or array_digest(coarse) != row["coarse_sha256"]
        ):
            raise ValueError("unfitted ordered coordinate increments do not replay")
        return dict(trace=trace, coarse=coarse, residual=arrays["residual"])

    def reconstruct(self) -> None:
        """Reassemble one macro and replay all fields with its executed retained E."""
        self._current_sources()
        if len(self.record["fields"]) == len(self.cases):
            for name in self.cases:
                self._field(name)
            return
        with threadpool_limits(self.native_threads):
            self.record["reconstruction_threadpools"] = [
                {k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()
            ]
            self._reconstruct()

    def _physical_statistics(self, fields: dict, solutions: dict) -> dict:
        """Measure complete original volume/trace rows and declared physical moments."""
        statistics = {
            name: dict(
                volume=np.longdouble(0),
                load=np.longdouble(0),
                weak=np.zeros(self._skeleton(*self.cases[name]).size, dtype=np.longdouble),
                cells=[],
            )
            for name in solutions
        }
        for cell in range(len(self.mesh.cells)):
            local = self._restored_local(cell)
            problem = local.problem
            pressure = np.column_stack([field[cell] for field in fields.values()])
            trace = np.column_stack(
                [
                    injection @ solution["trace"][indices]
                    for (indices, injection), solution in zip(
                        local.trace_maps, solutions.values(), strict=True
                    )
                ]
            )
            load = problem.load.astype(np.longdouble)
            boundary = np.einsum("ij,jk->ik", problem.coupling, trace, dtype=np.longdouble)
            forcing = load[:, None] - boundary
            defects = accurate_residual(problem.matrix.tocsr(), forcing, pressure)
            for column, (name, (indices, injection)) in enumerate(
                zip(solutions, local.trace_maps, strict=True)
            ):
                field, defect = pressure[:, column], defects[:, column]
                coupling = problem.coupling @ injection
                values = statistics[name]
                values["volume"] += defect @ defect
                values["load"] += load @ load
                np.add.at(
                    values["weak"],
                    indices,
                    np.einsum("ij,i->j", coupling, field, dtype=np.longdouble),
                )
                defect_norm, load_norm = np.linalg.norm(defect), np.linalg.norm(load)
                combined_norm = np.linalg.norm(forcing[:, column])
                values["cells"].append(
                    dict(
                        cell=cell,
                        original_volume_defect_norm=float(defect_norm),
                        volume_load_norm=float(load_norm),
                        rhs_after_trace_norm=float(combined_norm),
                        traction_norm=float(np.linalg.norm(boundary[:, column])),
                        original_relative_volume_load=float(defect_norm / load_norm)
                        if load_norm
                        else None,
                        original_relative_rhs_after_trace=float(defect_norm / combined_norm)
                        if combined_norm
                        else None,
                        physical_mean_minus_coarse=float(
                            (problem.constraints.T @ field)[0] - solutions[name]["coarse"][cell, 0]
                        ),
                        macro_outflow_minus_source=float(
                            (problem.kernel[:, 0] @ problem.coupling) @ trace[:, column]
                            - load.sum()
                        ),
                    )
                )
            del (
                local,
                problem,
                pressure,
                trace,
                load,
                boundary,
                forcing,
                defects,
                field,
                defect,
                coupling,
            )
        records = {}
        for name, values in statistics.items():
            weak_norm = np.linalg.norm(values["weak"])
            residual = np.sqrt(values["volume"] + weak_norm**2)
            denominator = np.sqrt(values["load"])
            records[name] = dict(
                original_free_equation_residual_l2=float(residual),
                original_physical_rhs_l2=float(denominator),
                original_free_equation_relative_residual=float(residual / denominator)
                if denominator
                else None,
                weak_trace_defect_l2=float(weak_norm),
                rtol=1e-10,
                accepted=bool(np.isfinite(residual) and residual <= 1e-10 * denominator),
                rows=(
                    "all original volume and weak-trace rows; zero exterior pressure; "
                    "no trace elimination or artificial gauge"
                ),
                cells=values["cells"],
            )
        return records

    def _reconstruct(self) -> None:
        """Use shared original-equation correction and replay actual executed increments."""
        started = perf_counter()
        initial = {name: self._solution(name) for name in self.cases}
        shape = (len(self.mesh.cells), self._cell(0)["kernel"].shape[0])
        fields = {}
        replay = self.record.get("refinement", {}).get("accepted", False)
        try:
            self.record["complete"] = False
            for name in self.cases:
                fields[name] = np.lib.format.open_memmap(
                    self.cell_directory / f"fields-{name}.npy.part",
                    mode="w+",
                    dtype=self.dtype,
                    shape=shape,
                )
            store = _RefinementStore(self, fields)
            if replay:
                store.records = self.record["refinement"]["records"]
            for cell in range(len(self.mesh.cells)):
                if replay:
                    pressure = store.read_record("initial", 0, cell).copy()
                    for step in range(self.record["refinement"]["steps"]):
                        pressure += store.read_record("correction", step, cell)
                else:
                    local = self._restored_local(cell)
                    traces = np.column_stack(
                        [
                            injection @ value["trace"][indices]
                            for (indices, injection), value in zip(
                                local.trace_maps, initial.values(), strict=True
                            )
                        ]
                    )
                    coarse = np.column_stack([value["coarse"][cell] for value in initial.values()])
                    pressure = local.problem.reconstruct(
                        traces,
                        coarse,
                        retained_basis=local.retained_basis,
                        solver=self.config["local_solver"],
                        refinement_precision=self.precision,
                    )
                    del local, traces, coarse
                if pressure.dtype != self.dtype or not np.isfinite(pressure).all():
                    raise ValueError("unfitted reconstructed precision or finite-array mismatch")
                store.write_fields(cell, pressure)
                del pressure
                print(f"unfitted reconstructed macro {cell + 1}/{len(self.mesh.cells)}", flush=True)
            solutions = {
                name: self._refined_solution(name, value) if replay else value
                for name, value in initial.items()
            }
            cases = [
                HybridRefinementCase(
                    self._compact(name),
                    HybridSolution(
                        value["trace"],
                        tuple(value["coarse"]),
                        (),
                        float(value["residual"]),
                        np.empty(0, dtype=self.dtype),
                    ),
                )
                for name, value in solutions.items()
            ]
            try:
                results = refine_hybrid_stream(
                    cases,
                    self._restored_local,
                    _RefinementStore(self, fields, check_only=True) if replay else store,
                    max_steps=0 if replay else self.config["original_refinement_steps"],
                    local_solver=self.config["local_solver"],
                    refinement_precision=self.precision,
                )
            except LinearSolveError as error:
                if not replay:
                    for cell in range(len(self.mesh.cells)):
                        store.write_fields(cell, store.read_record("initial", 0, cell))
                self.record["original_equations"] = self._physical_statistics(fields, solutions)
                self.record["original_refinement_rejection"] = dict(
                    accepted=False, final_original_criterion=str(error), records=store.records
                )
                self._checkpoint()
                raise LinearSolveError("original unfitted saddle residual exceeds 1e-10") from error
            if replay:
                if any(
                    list(result.local_contract_digests)
                    != self.record["refinement"]["cases"][name]["local_contract_digests"]
                    for name, result in zip(initial, results, strict=True)
                ):
                    raise ValueError("unfitted original local execution contract mismatch")
            else:
                self.record["refinement"] = dict(
                    accepted=False,
                    records=store.records,
                    cases={
                        name: self._refined_coordinates(name, result)
                        for name, result in zip(initial, results, strict=True)
                    },
                    steps=max(len(result.increments) for result in results),
                    precision="three portable float64 components; ordered acquired arithmetic",
                    fields="executed initial plus each actual physical correction in step order",
                )
                solutions = {
                    name: dict(
                        trace=result.solution.trace,
                        coarse=np.array(result.solution.coarse),
                        residual=result.solution.residual,
                    )
                    for name, result in zip(initial, results, strict=True)
                }
            self.record["original_equations"] = self._physical_statistics(fields, solutions)
            self._checkpoint()
            if not all(value["accepted"] for value in self.record["original_equations"].values()):
                raise LinearSolveError("original unfitted saddle residual exceeds 1e-10")
            self.record["refinement"]["accepted"] = True
            self._archive_fields(fields, solutions, store, started)
        finally:
            for field in fields.values():
                field.flush()
                cast(Any, field)._mmap.close()
            for name in fields:
                (self.cell_directory / f"fields-{name}.npy.part").unlink(missing_ok=True)

    def _archive_fields(
        self, fields: dict, solutions: dict, store: _RefinementStore, started: float
    ) -> None:
        """Publish actual bases, complete fields and self-contained ordered field history."""
        for column, (name, pressure) in enumerate(fields.items()):
            pressure.flush()
            arrays = dict(
                macro_points=self.mesh.points,
                macro_cells=self.mesh.cells,
                macro_faces=self.mesh.faces,
                macro_face_cells=self.mesh.face_cells,
                macro_normals=self.mesh.normals,
                macro_signs=self.mesh.signs,
                macro_cell_faces=self.mesh.cell_faces,
                degree=np.array(self.config["local_degree"]),
                coefficient_precision_bits=np.array(self.config["coefficient_precision_bits"]),
                acquisition_id=self.record["acquisition_id"],
                refinement_steps=self.record["refinement"]["steps"],
                **precision_fields("initial_trace", self._solution(name)["trace"]),
                **precision_fields("initial_coarse", self._solution(name)["coarse"].ravel()),
                **precision_fields("trace", solutions[name]["trace"]),
                **precision_fields("coarse", solutions[name]["coarse"].ravel()),
            )
            for cell in range(len(self.mesh.cells)):
                stored = self._cell(cell)
                arrays.update(
                    {
                        f"{key}_{cell}": stored[key]
                        for key in (
                            "points",
                            "cells",
                            "nodal_dofs",
                            "nodal_points",
                            "multiindices",
                            "kernel",
                            "constraints",
                        )
                    }
                )
                arrays.update(precision_fields(f"pressure_{cell}", pressure[cell]))
                initial = store.read_record("initial", 0, cell)[:, column]
                arrays.update(precision_fields(f"initial_pressure_{cell}", initial))
                for step in range(self.record["refinement"]["steps"]):
                    delta = store.read_record("correction", step, cell)[:, column]
                    arrays.update(precision_fields(f"executed_delta_{step}_{cell}", delta))
                arrays.update(
                    precision_fields(
                        f"retained_basis_{cell}", _restore(stored, "retained_basis", self.dtype)
                    )
                )
            path = self.directory / f"{self.prefix}-{name}.npz"
            _npz(path, **arrays)
            self.record["fields"][name] = dict(
                archive=path.name, sha256=fingerprint(path), seconds=perf_counter() - started
            )
            ell, segments = self.cases[name]
            self.record["cases"] = [row for row in self.record["cases"] if row["name"] != name] + [
                dict(
                    name=name,
                    trace_degree=ell,
                    segments=segments,
                    macro_diameter=0.5,
                    H=0.5 / segments,
                    macro_cells=len(self.mesh.cells),
                    fine_cells=len(self.mesh.cells) * self.config["local_refinement"] ** 2,
                    trace_dofs=self._skeleton(ell, segments).size,
                    global_dofs=self._skeleton(ell, segments).size + len(self.mesh.cells),
                    archive=path.name,
                    archive_sha256=fingerprint(path),
                    original_free_equation_relative_residual=self.record["original_equations"][
                        name
                    ]["original_free_equation_relative_residual"],
                )
            ]
            self._checkpoint()
            del arrays, stored

    def _field(self, name: str) -> dict[str, np.ndarray]:
        """Load complete field coefficients only from an accepted original saddle."""
        row = self.record["fields"].get(name)
        if row is None:
            raise ValueError("unfitted field reconstruction is incomplete")
        arrays = self._read(self.directory / row["archive"], row["sha256"])
        if (
            str(arrays["acquisition_id"]) != self.record["acquisition_id"]
            or not self.record["original_equations"][name]["accepted"]
            or not self.record["refinement"]["accepted"]
            or arrays["refinement_steps"] != self.record["refinement"]["steps"]
        ):
            raise ValueError("unfitted field acquisition or original-equation mismatch")
        initial = self._solution(name)
        solution = self._refined_solution(name, initial)
        if any(
            not np.array_equal(_restore(arrays, key, self.dtype), value)
            for key, value in (
                ("initial_trace", initial["trace"]),
                ("initial_coarse", initial["coarse"].ravel()),
                ("trace", solution["trace"]),
                ("coarse", solution["coarse"].ravel()),
            )
        ):
            raise ValueError("unfitted field coordinates differ from executed ordered increments")
        if any(
            not np.array_equal(arrays[key], value)
            for key, value in (
                ("macro_points", self.mesh.points),
                ("macro_cells", self.mesh.cells),
                ("macro_faces", self.mesh.faces),
                ("macro_face_cells", self.mesh.face_cells),
                ("macro_normals", self.mesh.normals),
                ("macro_signs", self.mesh.signs),
                ("macro_cell_faces", self.mesh.cell_faces),
            )
        ):
            raise ValueError("unfitted field macro geometry or trace orientation mismatch")
        for cell in range(len(self.mesh.cells)):
            if (
                array_digest(_restore(arrays, f"retained_basis_{cell}", self.dtype))
                != self.record["cells"][str(cell)]["retained_basis_sha256"]
                or array_digest(arrays[f"kernel_{cell}"])
                != self.record["cells"][str(cell)]["kernel_sha256"]
            ):
                raise ValueError("unfitted field executed retained basis or kernel mismatch")
            executed = _restore(arrays, f"initial_pressure_{cell}", self.dtype)
            history = self.record["refinement"]["records"]
            if (
                array_digest(executed)
                != history[f"initial-0-cell-{cell}"]["case_values_sha256"][name]
            ):
                raise ValueError("unfitted field executed initial coefficients mismatch")
            for step in range(self.record["refinement"]["steps"]):
                delta = _restore(arrays, f"executed_delta_{step}_{cell}", self.dtype)
                if (
                    array_digest(delta)
                    != history[f"correction-{step}-cell-{cell}"]["case_values_sha256"][name]
                ):
                    raise ValueError("unfitted field executed correction coefficients mismatch")
                executed += delta
            if not np.array_equal(executed, _restore(arrays, f"pressure_{cell}", self.dtype)):
                raise ValueError("unfitted ordered initial and correction fields do not replay")
        return arrays

    def norms(self) -> None:
        """Evaluate full original fields with independent material/volume quadrature."""
        self._current_sources()
        if len(self.record["fields"]) != len(self.cases):
            raise ValueError("unfitted field reconstruction is incomplete")
        with threadpool_limits(self.native_threads):
            self.record["norm_threadpools"] = [
                {k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()
            ]
            for row in self.record["cases"]:
                arrays = self._field(row["name"])
                solution = _NormFields(
                    tuple(
                        TriangleMesh(arrays[f"points_{cell}"], arrays[f"cells_{cell}"])
                        for cell in range(len(self.mesh.cells))
                    ),
                    tuple(
                        _restore(arrays, f"pressure_{cell}", self.dtype)
                        for cell in range(len(self.mesh.cells))
                    ),
                    self.config["local_degree"],
                )
                started = perf_counter()
                row["norms"] = {
                    f"quadrature_{order}": error_norms(solution, smooth_field, order)
                    for order in self.config["norm_orders"]
                }
                row["norm_seconds"] = perf_counter() - started
                self._checkpoint()
            self.record["complete"] = len(self.record["cases"]) == len(self.cases) and all(
                "norms" in row for row in self.record["cases"]
            )
            self._checkpoint()


def main() -> None:
    """Run reproducible individual phases or their sequential complete acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=int, required=True)
    parser.add_argument("--names", nargs="+", default=["ell2-s32", "ell3-s32"])
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--assembly-order", type=int, default=13)
    parser.add_argument("--norm-orders", nargs="+", type=int, default=[13, 15])
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--local-solver", default="scipy")
    parser.add_argument("--original-refinement-steps", type=int, default=2)
    parser.add_argument(
        "--refinement-precision", choices=["double", "extended"], default="extended"
    )
    parser.add_argument(
        "--stage", choices=["condense", "solve", "reconstruct", "norms", "all"], required=True
    )
    args = parser.parse_args()
    acquisition = UnfittedAcquisition(
        args.output,
        refinement=args.refinement,
        names=args.names,
        degree=args.degree,
        assembly_order=args.assembly_order,
        norm_orders=args.norm_orders,
        native_threads=args.native_threads,
        local_solver=args.local_solver,
        refinement_precision=args.refinement_precision,
        original_refinement_steps=args.original_refinement_steps,
    )
    for stage in ("condense", "solve", "reconstruct", "norms"):
        if args.stage in {stage, "all"}:
            getattr(acquisition, stage)()


if __name__ == "__main__":
    from importlib import import_module

    import_module("examples.unfitted_phases").main()
