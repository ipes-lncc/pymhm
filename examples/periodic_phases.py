"""Cellwise acquisition and immutable coefficient replay for periodic Q1 Darcy.

Condensation retains only small skeletal blocks and declared constant kernels.
After the global solve, a second local pass solves only the actual combined
forcing columns using the archived retained basis, constraints and trace maps.
It checks the original full saddle equations before publishing nodal fields.
Local lifts are never archived or retained for all cells. All checkpoints belong
to one acquisition.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

if __package__:
    from .archive_precision import precision_fields, restore_precision
    from .verify_periodic import (
        ROOT,
        case_conventions,
        fingerprint,
        material,
        source,
        source_hashes,
        validate_case_provenance,
    )
else:
    from examples.archive_precision import precision_fields, restore_precision
    from examples.verify_periodic import (
        ROOT,
        case_conventions,
        fingerprint,
        material,
        source,
        source_hashes,
        validate_case_provenance,
    )

from examples.formulations.local_records import CartesianTask as _QuadTask
from examples.formulations.local_records import assemble_cartesian_task as _assemble_quad
from pymhm import FaceSpace, HybridSystem, SkeletonSpace
from pymhm.core.contracts import HybridSolution, LocalProblem
from pymhm.core.refinement import (
    HybridRefinementCase,
    HybridRefinementLocal,
    refine_hybrid_stream,
)
from pymhm.core.subspaces import restrict_response
from pymhm.core.validation import positive_int
from pymhm.linalg.linear import LinearSolveError, SolverUnavailableError, accurate_residual
from pymhm.meshes.cartesian import CartesianMacroMesh
from pymhm.postprocessing.conforming import ConformingQuadrilateralSolution


def array_digest(value: np.ndarray) -> str:
    """Hash numerical entries and precision, excluding long-double padding bytes."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    parts = precision_fields("value", array).values() if array.dtype.itemsize > 8 else (array,)
    for part in parts:
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


def _sparse_digest(matrix: Any) -> str:
    """Identify an assembled sparse operator without changing its entries."""
    matrix = matrix.tocsc(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return array_digest(
        np.frombuffer(
            (
                str(matrix.shape)
                + array_digest(matrix.indptr)
                + array_digest(matrix.indices)
                + array_digest(matrix.data)
            ).encode("ascii"),
            dtype=np.uint8,
        )
    )


def _json(path: Path, record: dict) -> None:
    """Publish a complete manifest only after flushing its atomic replacement."""
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _npz(path: Path, **arrays: Any) -> None:
    """Save executed arrays atomically, without NumPy changing the suffix."""
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class _RefinementStore:
    """Persist actual cellwise correction arrays as portable precision components."""

    def __init__(
        self, acquisition: PeriodicAcquisition, fields: dict, *, check_only: bool = False
    ) -> None:
        self.acquisition = acquisition
        self.fields = fields
        self.directory = acquisition.cell_directory / "original-refinement"
        self.directory.mkdir(exist_ok=True)
        self.records = {}
        self.check_only = check_only
        self.last = None

    def read_fields(self, cell: int) -> np.ndarray:
        return np.column_stack([field[cell] for field in self.fields.values()])

    def write_fields(self, cell: int, fields: np.ndarray) -> None:
        for column, field in enumerate(self.fields.values()):
            field[cell] = fields[:, column]

    def write_record(self, name: str, step: int, cell: int, values: np.ndarray) -> None:
        key = f"{name}-{step}-cell-{cell}"
        if self.check_only:
            self.last = (key, values.copy())
            return
        path = self.directory / f"{key}.npz"
        _npz(path, **precision_fields("values", values))
        self.records[key] = dict(
            archive=path.name, sha256=fingerprint(path), values_sha256=array_digest(values)
        )

    def read_record(self, name: str, step: int, cell: int) -> np.ndarray:
        key = f"{name}-{step}-cell-{cell}"
        if self.check_only:
            if self.last is None or self.last[0] != key:
                raise ValueError("periodic check store has no matching transient record")
            return self.last[1]
        record = self.records[key]
        path = self.directory / record["archive"]
        if (
            Path(record["archive"]).name != record["archive"]
            or fingerprint(path) != record["sha256"]
        ):
            raise ValueError("periodic correction archive digest mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            values = restore_precision(
                arrays["values"], arrays["values_correction"], arrays["values_tail"]
            ).astype(self.acquisition.coefficient_dtype)
        if array_digest(values) != record["values_sha256"]:
            raise ValueError("periodic correction precision or numerical digest mismatch")
        return values


class PeriodicAcquisition:
    """Run resumable Q1/P0 stages for the unit-square homogeneous-Dirichlet case.

    The kernel is fixed by unit physical mean, Z=1 at every Q1 node. Pressure
    coefficients use tensor equidistant cardinal functions, x index fastest;
    skeleton coefficients are physical normal flux in the first neighbor's
    normal orientation. No periodic boundary condition is applied: 'periodic'
    describes the material. Gauge freedom is removed by exterior pressure zero.
    A configuration/source mismatch or corrupted checkpoint is always rejected.
    """

    def __init__(
        self,
        directory: Path,
        *,
        macro: int,
        refinement: int,
        segments: list[int] | tuple[int, ...],
        native_threads: int = 1,
        order: int = 4,
        refinement_precision: Literal["double", "extended"] = "double",
        original_refinement_steps: int = 2,
    ) -> None:
        """Fix material, mesh, traces, quadrature and coefficient-storage precision.

        Explicit extended precision retains the shared local/global correction
        digits throughout coefficient and field archives; native factors remain
        double precision. It requires a wider NumPy long-double type. Neither
        precision choice changes solver residual tolerances.
        """
        if refinement_precision not in {"double", "extended"}:
            raise ValueError("refinement_precision must be double or extended")
        if (
            refinement_precision == "extended"
            and np.finfo(np.longdouble).eps >= np.finfo(float).eps
        ):
            raise SolverUnavailableError("extended refinement requires a wider long-double type")
        self.refinement_precision = refinement_precision
        if (
            isinstance(original_refinement_steps, (bool, np.bool_))
            or not isinstance(original_refinement_steps, (int, np.integer))
            or original_refinement_steps < 0
        ):
            raise ValueError("original_refinement_steps must be a nonnegative integer")
        self.coefficient_dtype = np.dtype(
            np.longdouble if refinement_precision == "extended" else float
        )
        macro, refinement = positive_int(macro, "macro"), positive_int(refinement, "refinement")
        segments = tuple(sorted(set(positive_int(s, "segments") for s in segments)))
        if not segments or any(max(segments) % s for s in segments):
            raise ValueError("each segment count must divide the largest prepared partition")
        self.mesh = CartesianMacroMesh(macro)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"mhm-{macro}-r{refinement}-phases.json"
        self.cell_directory = self.directory / f"mhm-{macro}-r{refinement}-cells"
        self.cell_directory.mkdir(exist_ok=True)
        self.config = dict(
            macro=macro,
            refinement=refinement,
            segments=list(segments),
            degree=1,
            quadrature_order=max(2, positive_int(order, "order")),
            refinement_precision=refinement_precision,
            original_refinement_steps=int(original_refinement_steps),
            coefficient_dtype=self.coefficient_dtype.str,
            coefficient_precision_bits=np.finfo(self.coefficient_dtype).nmant + 1,
            **case_conventions(),
        )
        self.native_threads = positive_int(native_threads, "native_threads")
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if (
                self.record.get("schema") != "pymhm-periodic-phases-v1"
                or self.record["configuration"] != self.config
            ):
                raise ValueError("periodic phase manifest configuration mismatch")
        else:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout
            )
            self.record = dict(
                schema="pymhm-periodic-phases-v1",
                acquisition_id=str(uuid4()),
                configuration=self.config,
                source_hashes=source_hashes(),
                phase_source_sha256=fingerprint(Path(__file__)),
                precision_source_sha256=fingerprint(
                    Path(__file__).with_name("archive_precision.py")
                ),
                lockfile_sha256=fingerprint(ROOT / "pixi.lock"),
                git_revision=revision,
                git_dirty=dirty,
                python=sys.version.split()[0],
                numpy=np.__version__,
                scipy=scipy.__version__,
                platform=platform.platform(),
                basis_convention="Q1 nodal cardinal x-fastest; retained constant has unit L2 mean",
                trace_convention="physical flux normal to first adjacent macrocell",
                local_solver="scipy",
                global_solver="scipy",
                local_refinement_precision=refinement_precision,
                global_refinement_precision=refinement_precision,
                cells={},
                solutions={},
                fields={},
            )
            _json(self.path, self.record)

    def _current_sources(self) -> None:
        """Require reconstruction to execute the acquisition's local implementation."""
        if (
            self.record["source_hashes"] != source_hashes()
            or self.record["phase_source_sha256"] != fingerprint(Path(__file__))
            or self.record["precision_source_sha256"]
            != fingerprint(Path(__file__).with_name("archive_precision.py"))
            or self.record["lockfile_sha256"] != fingerprint(ROOT / "pixi.lock")
        ):
            raise ValueError("periodic acquisition source or lockfile mismatch")

    def _skeleton(self, segments: int) -> SkeletonSpace:
        """Keep the shared face numbering and first-neighbor physical orientation."""
        return SkeletonSpace(
            self.mesh, tuple(FaceSpace.uniform(0, segments) for _ in self.mesh.faces)
        )

    def _task(self, cell: int, skeleton: SkeletonSpace) -> _QuadTask:
        """Build the same shared quadrilateral assembly task in both local passes."""
        return _QuadTask(
            self.mesh,
            cell,
            (self.config["refinement"],) * 2,
            skeleton,
            1,
            material,
            source,
            self.config["quadrature_order"],
        )

    def _cell(self, cell: int) -> dict[str, np.ndarray]:
        """Load one executed cell basis/block archive and verify its manifest digest."""
        record = self.record["cells"].get(str(cell))
        if record is None:
            raise ValueError("periodic condensation is incomplete")
        path = self.cell_directory / f"cell-{cell}.npz"
        if fingerprint(path) != record["sha256"]:
            raise ValueError("periodic cell archive digest mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        portable_basis = restore_precision(
            arrays["retained_basis_high"],
            arrays["retained_basis_low"],
            arrays["retained_basis_tail"],
        ).astype(self.coefficient_dtype)
        if not np.array_equal(portable_basis, arrays["retained_basis"]):
            raise ValueError("periodic executed retained basis portable components mismatch")
        arrays["retained_basis"] = portable_basis
        if array_digest(arrays["kernel"]) != record["kernel_sha256"]:
            raise ValueError("periodic executed kernel digest mismatch")
        if (
            arrays["retained_basis"].shape != arrays["kernel"].shape
            or not np.isfinite(arrays["retained_basis"]).all()
            or arrays["retained_basis"].dtype.str != record["retained_basis_dtype"]
            or array_digest(arrays["retained_basis"]) != record["retained_basis_sha256"]
        ):
            raise ValueError("periodic executed retained basis digest or dtype mismatch")
        return arrays

    def condense(self) -> None:
        """Acquire each cell once; discard its matrix and lifts after compact blocks."""
        self._current_sources()
        with threadpool_limits(self.native_threads):
            self.record["condensation_threadpools"] = self._threadpools()
            self._condense()
        self._current_sources()

    def _threadpools(self) -> list[dict]:
        """Record actual native numerical pools while their limit is active."""
        return [
            {key: value for key, value in pool.items() if key != "filepath"}
            for pool in threadpool_info()
        ]

    def _condense(self) -> None:
        """Execute one bounded local acquisition pass inside the native limit."""
        self._current_sources()
        maximum = max(self.config["segments"])
        prepared = self._skeleton(maximum)
        for cell in range(len(self.mesh.cells)):
            if str(cell) in self.record["cells"]:
                self._cell(cell)
                continue
            started = perf_counter()
            assembled = _assemble_quad(self._task(cell, prepared))
            response = assembled.problem.condense(refinement_precision=self.refinement_precision)
            problem = response.problem
            # A declared moment fixes the only possible orthogonal rotation (+/-1).
            if not np.array_equal(problem.kernel, np.ones_like(problem.kernel)):
                raise ValueError("periodic kernel must have declared unit physical mean")
            arrays = dict(
                kernel=problem.kernel,
                constraints=problem.constraints,
                retained_basis=response.retained_basis,
                signs=self.mesh.signs[cell],
                faces=self.mesh.cell_faces[cell],
                bounds=np.array(self.mesh.submesh(cell, self.config["refinement"]).bounds),
            )
            parts = precision_fields("basis", response.retained_basis)
            arrays.update(
                retained_basis_high=parts["basis"],
                retained_basis_low=parts["basis_correction"],
                retained_basis_tail=parts["basis_tail"],
            )
            for segments in self.config["segments"]:
                skeleton = self._skeleton(segments)
                face_map = np.eye(segments)[np.arange(maximum) // (maximum // segments)]
                narrowed = restrict_response(
                    response, np.kron(np.eye(4), face_map), skeleton.cell_dofs(cell)
                )
                indices, block, rhs = narrowed.global_contribution(np.array([skeleton.size + cell]))
                arrays.update(
                    {
                        f"{name}_{segments}": value
                        for name, value in (
                            ("indices", indices),
                            ("block", block),
                            ("rhs", rhs),
                            ("trace_dofs", skeleton.cell_dofs(cell)),
                        )
                    }
                )
            path = self.cell_directory / f"cell-{cell}.npz"
            _npz(path, **arrays)
            self.record["cells"][str(cell)] = dict(
                sha256=fingerprint(path),
                kernel_sha256=array_digest(problem.kernel),
                retained_basis_sha256=array_digest(response.retained_basis),
                retained_basis_dtype=response.retained_basis.dtype.str,
                matrix_sha256=_sparse_digest(problem.matrix),
                load_sha256=array_digest(problem.load),
                coupling_sha256=array_digest(problem.coupling),
                native_threads=self.native_threads,
                seconds=perf_counter() - started,
            )
            _json(self.path, self.record)
            print(f"periodic condensed cell {cell + 1}/{len(self.mesh.cells)}", flush=True)
            del response, narrowed, problem, assembled, arrays

    def solve(self) -> None:
        """Solve small skeletal systems without any volumetric responses in memory."""
        self._current_sources()
        with threadpool_limits(self.native_threads):
            self.record["solution_threadpools"] = self._threadpools()
            self._solve()
        self._current_sources()

    def _solve(self) -> None:
        """Execute the compact global phase inside the declared native limit."""
        for segments in self.config["segments"]:
            if str(segments) in self.record["solutions"]:
                self._solution(segments)
                continue
            started = perf_counter()
            contributions = []
            for cell in range(len(self.mesh.cells)):
                arrays = self._cell(cell)
                contributions.append(
                    tuple(arrays[f"{key}_{segments}"] for key in ("indices", "block", "rhs"))
                )
            system = HybridSystem.from_contributions(
                contributions,
                trace_size=self._skeleton(segments).size,
                coarse_sizes=[1] * len(self.mesh.cells),
            )
            result = system.solve(refinement_precision=self.refinement_precision)
            path = (
                self.directory
                / f"mhm-{self.mesh.nx}-r{self.config['refinement']}-s{segments}-coefficients.npz"
            )
            _npz(
                path,
                trace=result.trace,
                coarse=np.array(result.coarse),
                residual=result.residual,
                acquisition_id=self.record["acquisition_id"],
                kernel_offsets=system.kernel_offsets,
            )
            self.record["solutions"][str(segments)] = dict(
                archive=path.name,
                sha256=fingerprint(path),
                residual=result.residual,
                native_threads=self.native_threads,
                seconds=perf_counter() - started,
                matrix_sha256=_sparse_digest(system.matrix),
                rhs_sha256=array_digest(system.rhs),
            )
            _json(self.path, self.record)

    def _solution(self, segments: int) -> dict[str, np.ndarray]:
        """Read solved coefficients from the same acquisition as the cell kernels."""
        record = self.record["solutions"].get(str(segments))
        if record is None:
            raise ValueError("periodic global solution is incomplete")
        if Path(record["archive"]).name != record["archive"]:
            raise ValueError("periodic coefficient archive must remain within its acquisition")
        path = self.directory / record["archive"]
        if fingerprint(path) != record["sha256"]:
            raise ValueError("periodic coefficient archive digest mismatch")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        if str(arrays["acquisition_id"]) != self.record["acquisition_id"]:
            raise ValueError("periodic coefficients belong to a different acquisition")
        if any(
            arrays[name].dtype != self.coefficient_dtype or not np.isfinite(arrays[name]).all()
            for name in ("trace", "coarse")
        ):
            raise ValueError("periodic coefficient precision or finite-array mismatch")
        return arrays

    def reconstruct(self) -> None:
        """Reassemble cellwise and reconstruct using the archived oriented kernel."""
        self._current_sources()
        with threadpool_limits(self.native_threads):
            self.record["reconstruction_threadpools"] = self._threadpools()
            self._reconstruct()
        self._current_sources()

    def _restored_local(self, cell: int) -> HybridRefinementLocal:
        """Restore the original operator, executed basis and common oriented trace maps."""
        maximum = max(self.config["segments"])
        arrays = self._cell(cell)
        original = _assemble_quad(self._task(cell, self._skeleton(maximum))).problem
        record = self.record["cells"][str(cell)]
        if (
            _sparse_digest(original.matrix) != record["matrix_sha256"]
            or array_digest(original.load) != record["load_sha256"]
            or array_digest(original.coupling) != record["coupling_sha256"]
            or not np.array_equal(arrays["signs"], self.mesh.signs[cell])
            or not np.array_equal(arrays["faces"], self.mesh.cell_faces[cell])
        ):
            raise ValueError("periodic replay operator, load or orientation mismatch")
        problem = LocalProblem(
            original.matrix,
            original.coupling,
            original.load,
            original.trace_dofs,
            kernel=arrays["kernel"],
            constraints=arrays["constraints"],
        )
        maps = []
        for segments in self.config["segments"]:
            dofs = arrays[f"trace_dofs_{segments}"]
            if not np.array_equal(dofs, self._skeleton(segments).cell_dofs(cell)):
                raise ValueError("periodic replay trace numbering mismatch")
            face = np.eye(segments)[np.arange(maximum) // (maximum // segments)]
            maps.append((dofs, np.kron(np.eye(4), face)))
        return HybridRefinementLocal(problem, arrays["retained_basis"], maps)

    def _compact(self, segments: int) -> HybridSystem:
        """Restore the same compact operator used by the original coefficient solve."""
        contributions = []
        for cell in range(len(self.mesh.cells)):
            arrays = self._cell(cell)
            contributions.append(
                tuple(arrays[f"{key}_{segments}"] for key in ("indices", "block", "rhs"))
            )
        system = HybridSystem.from_contributions(
            contributions,
            trace_size=self._skeleton(segments).size,
            coarse_sizes=[1] * len(self.mesh.cells),
        )
        record = self.record["solutions"][str(segments)]
        if (
            _sparse_digest(system.matrix) != record["matrix_sha256"]
            or array_digest(system.rhs) != record["rhs_sha256"]
        ):
            raise ValueError("periodic executed condensed operator or load mismatch")
        return system

    def _refined_coordinates(self, segments: int, result: Any, initial: dict) -> dict:
        """Archive actual coordinate increments separately from the original global solve."""
        path = (
            self.directory
            / f"mhm-{self.mesh.nx}-r{self.config['refinement']}-s{segments}-refinement.npz"
        )
        solution = result.solution
        arrays = dict(
            **precision_fields("trace", solution.trace),
            **precision_fields("coarse", np.array(solution.coarse)),
            **precision_fields("gauges", solution.gauge_multipliers),
            residual=solution.residual,
            acquisition_id=self.record["acquisition_id"],
        )
        for step, delta in enumerate(result.increments):
            arrays.update(precision_fields(f"delta_trace_{step}", delta.trace))
            arrays.update(precision_fields(f"delta_coarse_{step}", np.array(delta.coarse)))
            arrays.update(precision_fields(f"delta_gauges_{step}", delta.gauge_multipliers))
        _npz(path, **arrays)
        return dict(
            archive=path.name,
            sha256=fingerprint(path),
            initial_archive=self.record["solutions"][str(segments)]["archive"],
            initial_sha256=self.record["solutions"][str(segments)]["sha256"],
            residual=solution.residual,
            original_residual_norms=list(result.residual_norms),
            original_rhs_norm=result.rhs_norm,
            corrections=len(result.increments),
            local_contract_digests=list(result.local_contract_digests),
            trace_sha256=array_digest(solution.trace),
            coarse_sha256=array_digest(np.array(solution.coarse)),
        )

    def _refined_solution(self, segments: int, initial: dict) -> dict:
        """Restore final coordinates and verify their ordered executed increments."""
        record = self.record["refinement"]["cases"][str(segments)]
        if (
            record["initial_sha256"] != self.record["solutions"][str(segments)]["sha256"]
            or Path(record["archive"]).name != record["archive"]
        ):
            raise ValueError("periodic refinement coordinate acquisition mismatch")
        path = self.directory / record["archive"]
        if fingerprint(path) != record["sha256"]:
            raise ValueError("periodic refined coordinate archive digest mismatch")
        with np.load(path, allow_pickle=False) as data:

            def restored(name: str) -> np.ndarray:
                return restore_precision(
                    data[name], data[f"{name}_correction"], data[f"{name}_tail"]
                ).astype(self.coefficient_dtype)

            trace, coarse = initial["trace"].copy(), initial["coarse"].copy()
            gauges = np.empty(0, dtype=self.coefficient_dtype)
            for step in range(record["corrections"]):
                trace += restored(f"delta_trace_{step}")
                coarse += restored(f"delta_coarse_{step}")
                gauges += restored(f"delta_gauges_{step}")
            if (
                str(data["acquisition_id"]) != self.record["acquisition_id"]
                or not np.array_equal(trace, restored("trace"))
                or not np.array_equal(coarse, restored("coarse"))
                or not np.array_equal(gauges, restored("gauges"))
                or array_digest(trace) != record["trace_sha256"]
                or array_digest(coarse) != record["coarse_sha256"]
            ):
                raise ValueError("periodic ordered coordinate increments do not replay")
            return dict(trace=trace, coarse=coarse, residual=float(data["residual"]))

    def _physical_statistics(self, fields: dict, solutions: dict) -> dict:
        """Measure original physical rows, energy, means and macro conservation."""
        statistics = {
            s: dict(
                volume_defect_squared=np.longdouble(0),
                volume_load_squared=np.longdouble(0),
                weak=np.zeros(self._skeleton(s).size, dtype=np.longdouble),
                energy=np.longdouble(0),
                source_work=np.longdouble(0),
                cells=[],
            )
            for s in solutions
        }
        for cell in range(len(self.mesh.cells)):
            local = self._restored_local(cell)
            p = local.problem
            pressure = np.column_stack([field[cell] for field in fields.values()])
            trace = np.column_stack(
                [
                    injection @ solutions[s]["trace"][indices]
                    for s, (indices, injection) in zip(solutions, local.trace_maps, strict=True)
                ]
            )
            load = p.load.astype(np.longdouble)
            forcing = accurate_residual(
                scipy.sparse.csr_matrix(p.coupling),
                np.broadcast_to(load[:, None], pressure.shape),
                trace,
            )
            defects = accurate_residual(p.matrix.tocsr(), forcing, pressure)
            actions = -accurate_residual(p.matrix.tocsr(), np.zeros_like(pressure), pressure)
            for column, (s, (indices, injection)) in enumerate(
                zip(solutions, local.trace_maps, strict=True)
            ):
                field, defect = pressure[:, column], defects[:, column]
                values = statistics[s]
                coupling = p.coupling @ injection
                values["volume_defect_squared"] += defect @ defect
                values["volume_load_squared"] += load @ load
                np.add.at(
                    values["weak"],
                    indices,
                    coupling.astype(np.longdouble).T @ field.astype(np.longdouble),
                )
                values["energy"] += field @ actions[:, column]
                values["source_work"] += field @ load
                dn, ln, fn = (
                    np.linalg.norm(defect),
                    np.linalg.norm(load),
                    np.linalg.norm(forcing[:, column]),
                )
                values["cells"].append(
                    dict(
                        cell=cell,
                        original_volume_defect_norm=float(dn),
                        volume_load_norm=float(ln),
                        rhs_after_trace_norm=float(fn),
                        original_relative_volume_load=float(dn / ln),
                        original_relative_rhs_after_trace=float(dn / fn),
                        physical_mean_minus_coarse=float(
                            (p.constraints.T @ field)[0] / (1 / self.mesh.nx**2)
                            - solutions[s]["coarse"][cell, 0]
                        ),
                        macro_outflow_minus_source=float(
                            (p.kernel[:, 0] @ p.coupling) @ trace[:, column] - load.sum()
                        ),
                    )
                )
            del local, p, pressure, trace, forcing, defects, actions, field, defect, coupling
        records = {}
        for s, values in statistics.items():
            weak = values["weak"]
            defect_norm = np.sqrt(values["volume_defect_squared"] + weak @ weak)
            load_norm = np.sqrt(values["volume_load_squared"])
            relative = defect_norm / load_norm
            records[str(s)] = dict(
                original_saddle_residual_norm=float(defect_norm),
                volume_load_norm=float(load_norm),
                original_saddle_relative_residual=float(relative),
                weak_pressure_continuity_residual_norm=float(np.linalg.norm(weak)),
                energy_minus_source_work=float(values["energy"] - values["source_work"]),
                rtol=1e-10,
                accepted=bool(np.isfinite(relative) and relative <= 1e-10),
                rows=(
                    "all original volume and weak-trace rows; zero exterior pressure; "
                    "no artificial gauges"
                ),
                accumulation="shared accurate residual; wider NumPy accumulation where available",
                cells=values["cells"],
            )
        return records

    def _reconstruct(self) -> None:
        """Stream shared original-equation correction and replay its executed field history."""
        self._current_sources()
        initial = {s: self._solution(s) for s in self.config["segments"]}
        refinement = self.config["refinement"]
        shape = (len(self.mesh.cells), (refinement + 1) ** 2)
        fields = {}
        replay = self.record.get("refinement", {}).get("accepted", False)
        started = perf_counter()
        try:
            for s in initial:
                fields[s] = np.lib.format.open_memmap(
                    self.cell_directory / f"fields-{s}.npy.part",
                    mode="w+",
                    dtype=self.coefficient_dtype,
                    shape=shape,
                )
            store = _RefinementStore(self, fields)
            if replay:
                store.records = self.record["refinement"]["records"]
            for cell in range(len(self.mesh.cells)):
                local = self._restored_local(cell)
                traces = np.column_stack(
                    [
                        injection @ initial[s]["trace"][indices]
                        for s, (indices, injection) in zip(initial, local.trace_maps, strict=True)
                    ]
                )
                coarse = np.column_stack(
                    [solution["coarse"][cell] for solution in initial.values()]
                )
                pressure = local.problem.reconstruct(
                    traces,
                    coarse,
                    retained_basis=local.retained_basis,
                    refinement_precision=self.refinement_precision,
                )
                if pressure.dtype != self.coefficient_dtype or not np.isfinite(pressure).all():
                    raise ValueError(
                        "periodic reconstructed coefficient precision or finite-array mismatch"
                    )
                if replay:
                    executed = store.read_record("initial", 0, cell)
                    if not np.allclose(pressure, executed, rtol=2e-14, atol=2e-17):
                        raise ValueError("periodic executed initial field replay mismatch")
                    pressure = executed.copy()
                    for step in range(self.record["refinement"]["steps"]):
                        pressure += store.read_record("correction", step, cell)
                store.write_fields(cell, pressure)
                del local, pressure, traces, coarse
            solutions = {
                s: self._refined_solution(s, value) if replay else value
                for s, value in initial.items()
            }
            cases = [
                HybridRefinementCase(
                    self._compact(s),
                    HybridSolution(
                        value["trace"],
                        tuple(value["coarse"]),
                        (),
                        value["residual"],
                        np.empty(0, dtype=self.coefficient_dtype),
                    ),
                )
                for s, value in solutions.items()
            ]
            try:
                results = refine_hybrid_stream(
                    cases,
                    self._restored_local,
                    _RefinementStore(self, fields, check_only=True) if replay else store,
                    max_steps=0 if replay else self.config["original_refinement_steps"],
                    refinement_precision=self.refinement_precision,
                )
            except LinearSolveError as error:
                if not replay:
                    # Only the initial coordinates are available after rejection.
                    # Restore their executed fields before recording that state.
                    for cell in range(len(self.mesh.cells)):
                        store.write_fields(cell, store.read_record("initial", 0, cell))
                self.record["original_equations"] = self._physical_statistics(fields, solutions)
                self.record["original_refinement_rejection"] = dict(
                    accepted=False, final_original_criterion=str(error), records=store.records
                )
                self._current_sources()
                _json(self.path, self.record)
                raise LinearSolveError("original periodic saddle residual exceeds 1e-10") from error
            if not replay:
                refined = {
                    str(s): self._refined_coordinates(s, result, initial[s])
                    for s, result in zip(initial, results, strict=True)
                }
                self.record["refinement"] = dict(
                    accepted=False,
                    records=store.records,
                    cases=refined,
                    steps=max(len(result.increments) for result in results),
                    precision="three portable float64 components; ordered native accumulation",
                    fields="executed initial + each actual physical correction in step order",
                )
                solutions = {
                    s: dict(
                        trace=result.solution.trace,
                        coarse=np.array(result.solution.coarse),
                        residual=result.solution.residual,
                    )
                    for s, result in zip(initial, results, strict=True)
                }
            self.record["original_equations"] = self._physical_statistics(fields, solutions)
            self._current_sources()
            _json(self.path, self.record)
            if not all(value["accepted"] for value in self.record["original_equations"].values()):
                raise LinearSolveError("original periodic saddle residual exceeds 1e-10")
            self.record["refinement"]["accepted"] = True
            for s, pressure in fields.items():
                self._current_sources()
                pressure.flush()
                value = solutions[s]
                path = self.directory / f"mhm-{self.mesh.nx}-r{refinement}-s{s}.npz"
                _npz(
                    path,
                    fields=pressure,
                    residual=value["residual"],
                    trace=value["trace"],
                    coarse=value["coarse"],
                    acquisition_id=self.record["acquisition_id"],
                    macro_points=self.mesh.points,
                    macro_cells=self.mesh.cells,
                    signs=self.mesh.signs,
                    kernel=self._cell(0)["kernel"],
                    **precision_fields("fields_portable", pressure),
                    **precision_fields("trace_portable", value["trace"]),
                    **precision_fields("coarse_portable", value["coarse"]),
                )
                solution_record = dict(
                    self.record["solutions"][str(s)],
                    residual=value["residual"],
                    refinement=self.record["refinement"]["cases"][str(s)],
                )
                metadata = dict(
                    schema=self.record["schema"],
                    acquisition_id=self.record["acquisition_id"],
                    configuration=self.config,
                    segments=s,
                    archive=path.name,
                    archive_sha256=fingerprint(path),
                    kernel_sha256=array_digest(self._cell(0)["kernel"]),
                    basis_convention=self.record["basis_convention"],
                    trace_convention=self.record["trace_convention"],
                    source_hashes=self.record["source_hashes"],
                    phase_source_sha256=self.record["phase_source_sha256"],
                    precision_source_sha256=self.record["precision_source_sha256"],
                    lockfile_sha256=self.record["lockfile_sha256"],
                    git_revision=self.record["git_revision"],
                    git_dirty=self.record["git_dirty"],
                    python=self.record["python"],
                    numpy=self.record["numpy"],
                    scipy=self.record["scipy"],
                    platform=self.record["platform"],
                    cells=self.record["cells"],
                    solution=solution_record,
                    original_equations=self.record["original_equations"][str(s)],
                    refinement_records=self.record["refinement"]["records"],
                    reconstruction_native_threads=self.native_threads,
                    reconstruction_threadpools=self.record["reconstruction_threadpools"],
                    reconstruction_seconds=perf_counter() - started,
                )
                _json(path.with_suffix(".json"), metadata)
                self.record["fields"][str(s)] = metadata
                _json(self.path, self.record)
        finally:
            for pressure in fields.values():
                pressure.flush()
                pressure._mmap.close()
            for s in fields:
                (self.cell_directory / f"fields-{s}.npy.part").unlink(missing_ok=True)


def load_fields(
    path: Path, *, macro: int, refinement: int, segments: int
) -> tuple[ConformingQuadrilateralSolution, ...]:
    """Verify persisted nodal Q1 fields, basis and first-neighbor trace orientation.

    Legacy nodal archives receive shape/finiteness validation; their material
    and basis provenance remain unverified. They cannot certify phased replay.
    """
    mesh = CartesianMacroMesh(macro)
    manifest = path.with_suffix(".json")
    record = json.loads(manifest.read_text()) if manifest.exists() else None
    if record is not None:
        validate_case_provenance(record)
        configuration = record["configuration"]
        precision = configuration.get("refinement_precision", "double")
        dtype = np.dtype(configuration.get("coefficient_dtype", "float64"))
        if (
            precision not in {"double", "extended"}
            or dtype != np.dtype(np.longdouble if precision == "extended" else float)
            or configuration.get("coefficient_precision_bits", np.finfo(dtype).nmant + 1)
            != np.finfo(dtype).nmant + 1
            or (precision == "extended" and np.finfo(dtype).eps >= np.finfo(float).eps)
        ):
            raise ValueError("periodic field coefficient precision contract mismatch")
    if record is not None and (
        record.get("schema") != "pymhm-periodic-phases-v1"
        or record["configuration"]["macro"] != macro
        or record["configuration"]["refinement"] != refinement
        or record["configuration"]["degree"] != 1
        or record["segments"] != segments
        or record["archive"] != path.name
        or record["archive_sha256"] != fingerprint(path)
    ):
        raise ValueError("periodic field manifest or archive digest mismatch")
    with np.load(path, allow_pickle=False) as data:
        fields, residual = data["fields"], float(data["residual"])
        if (
            fields.shape != (len(mesh.cells), (refinement + 1) ** 2)
            or np.iscomplexobj(fields)
            or not np.isfinite(fields).all()
            or not np.isfinite(residual)
            or residual < 0
        ):
            raise ValueError("periodic field arrays violate their nodal acquisition contract")
        if record is not None and (
            fields.dtype != dtype or data["trace"].dtype != dtype or data["coarse"].dtype != dtype
        ):
            raise ValueError("periodic field coefficient precision contract mismatch")
        if record is not None and "precision_source_sha256" in record:
            if record["precision_source_sha256"] != fingerprint(
                Path(__file__).with_name("archive_precision.py")
            ):
                raise ValueError("periodic portable precision implementation mismatch")
            for name in ("fields", "trace", "coarse"):
                portable = restore_precision(
                    data[f"{name}_portable"],
                    data[f"{name}_portable_correction"],
                    data[f"{name}_portable_tail"],
                ).astype(dtype)
                if not np.array_equal(portable, data[name]):
                    raise ValueError("periodic field portable precision components mismatch")
        if record is not None and (
            str(data["acquisition_id"]) != record["acquisition_id"]
            or residual != record["solution"]["residual"]
            or data["trace"].shape != (2 * macro * (macro + 1) * segments,)
            or data["coarse"].shape != (macro * macro, 1)
            or not np.isfinite(data["trace"]).all()
            or not np.isfinite(data["coarse"]).all()
            or array_digest(data["kernel"]) != record["kernel_sha256"]
            or not np.array_equal(data["kernel"], np.ones(((refinement + 1) ** 2, 1)))
            or not np.array_equal(data["macro_points"], mesh.points)
            or not np.array_equal(data["macro_cells"], mesh.cells)
            or not np.array_equal(data["signs"], mesh.signs)
        ):
            raise ValueError("periodic field basis, geometry or orientation mismatch")
    return tuple(
        ConformingQuadrilateralSolution(
            mesh.submesh(cell, refinement), 1, pressure, material, residual
        )
        for cell, pressure in enumerate(fields)
    )
