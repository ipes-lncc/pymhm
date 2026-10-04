"""Optional defect correction of the original uncondensed hybrid equations."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, Protocol

import numpy as np
from scipy import sparse

from pymhm.hybrid import (
    HybridSolution,
    HybridSystem,
    LocalProblem,
    LocalResponse,
    _array,
    _preserved_array,
)
from pymhm.mesh import FloatArray
from pymhm.solvers import (
    LinearSolveError,
    _accurate_residual,
    _refinement_steps,
    _tolerances,
    factorize,
)


@dataclass(frozen=True)
class HybridRefinement:
    """Corrected physical fields and full-equation residual history in input units.

    ``residual_norms`` includes the initial state and every accepted correction.
    It combines the original local rows, unconstrained weak-trace rows and
    supplied physical moments. ``rhs_norm`` uses these same rows and units;
    this algebraic norm is not a finite-element error estimate.
    Persist ``solution.fields`` with their executed evaluation basis: trace and
    coarse coefficients alone cannot replay the added defect-source responses.
    ``solution.residual`` checks free weak-trace and retained-test equations on
    these corrected fields; artificial gauge rows are excluded from that
    diagnostic. Gauge multipliers include all accepted corrections.
    """

    solution: HybridSolution
    residual_norms: tuple[float, ...]
    rhs_norm: float


@dataclass(frozen=True)
class HybridRefinementCase:
    """Original reduced system, executed coordinates and physical data of one case.

    ``moments`` contains local physical weights and original integral targets.
    Compact systems additionally require ``moment_rows``: the executed reduced
    rows expressing these same integrals, without subtracting source moments.
    The number of moments must equal the initial gauge multiplier count.
    """

    system: HybridSystem
    solution: HybridSolution
    boundary_load: Any = None
    fixed: dict[int, float] | None = None
    moments: Sequence[tuple[Sequence[FloatArray], float]] = ()
    moment_rows: Sequence[FloatArray] = ()


@dataclass(frozen=True)
class HybridRefinementLocal:
    """One original local operator and the retained evaluation basis actually used.

    Every streamed case uses the same ``problem`` and ``retained_basis``.
    An optional ``trace_maps`` entry ``(indices, injection)`` for each case maps
    its oriented trace coefficients to the columns of ``problem.coupling``.
    Omission uses ``problem.trace_dofs`` with the identity injection.
    The local test and trial couplings remain those of the original problem.
    """

    problem: LocalProblem
    retained_basis: FloatArray
    trace_maps: Sequence[tuple[Any, FloatArray]] = ()


class HybridRefinementStore(Protocol):
    """Caller-owned bounded storage for executed fields and correction records.

    Arrays have shape ``(local_field_size, number_of_cases)`` and the selected
    correction dtype. ``read_fields``/``write_fields`` access the current physical
    fields. ``write_record``/``read_record`` preserve records named ``initial``
    (step zero), ``load`` (the measured original defect), ``forcing`` (the applied
    defect, zero for already accepted cases), ``source`` (its executed augmented
    inverse response), and ``correction`` (the actual physical delta).
    Replaying fields adds the recorded deltas to ``initial`` in step order and
    the executed dtype. Persist each case's returned coordinate increments and
    the executed local bases as well. Storage must preserve every digit; each
    write is read back and verified before the next operation.
    """

    def read_fields(self, cell: int) -> FloatArray:
        """Read the current physical fields of one cell."""
        ...

    def write_fields(self, cell: int, fields: FloatArray) -> None:
        """Store the current physical fields without narrowing their precision."""
        ...

    def write_record(self, name: str, step: int, cell: int, values: FloatArray) -> None:
        """Persist one executed array in the declared precision."""
        ...

    def read_record(self, name: str, step: int, cell: int) -> FloatArray:
        """Restore an executed array without recomputing its conditional solve."""
        ...


@dataclass(frozen=True)
class HybridStreamRefinement:
    """One streamed case's original residual history and executed coordinate deltas.

    ``solution.fields`` is empty; physical fields remain in the supplied store.
    ``increments`` contains the actual trace, retained and gauge corrections in
    execution order. ``local_contract_digests`` covers original operators, data,
    evaluation bases and oriented trace maps, and is checked on every pass.
    The residual norm has the same physical-row convention as ``HybridRefinement``.
    """

    solution: HybridSolution
    residual_norms: tuple[float, ...]
    rhs_norm: float
    increments: tuple[HybridSolution, ...]
    local_contract_digests: tuple[str, ...]


@dataclass
class _StreamCase:
    original: HybridRefinementCase
    trace: FloatArray
    coarse: tuple[FloatArray, ...]
    gauges: FloatArray
    boundary: FloatArray
    fixed: dict[int, float]
    free: FloatArray
    weights: tuple[tuple[FloatArray, ...], ...]
    targets: FloatArray
    rows: tuple[FloatArray, ...]


def _stream_case(case: HybridRefinementCase, dtype: Any, cells: int) -> _StreamCase:
    """Validate and preserve one case's explicit physical data."""
    system, solution = case.system, case.solution
    sizes = np.diff(system.kernel_offsets)
    if len(sizes) != cells or len(solution.coarse) != cells:
        raise ValueError("one ordered retained partition per streamed cell required")
    trace = _preserved_array(solution.trace, (system.trace_size,), "trace").astype(dtype)
    coarse = tuple(
        _preserved_array(c, (int(size),), "coarse").astype(dtype)
        for c, size in zip(solution.coarse, sizes, strict=True)
    )
    boundary = _preserved_array(
        np.zeros(system.trace_size) if case.boundary_load is None else case.boundary_load,
        trace.shape,
        "boundary_load",
    ).astype(dtype)
    fixed = {} if case.fixed is None else dict(case.fixed)
    if (
        any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or not 0 <= i < system.trace_size
            for i in fixed
        )
        or not np.isfinite(list(fixed.values())).all()
    ):
        raise ValueError("fixed DOFs must be valid trace indices with finite values")
    if any(trace[i] != value for i, value in fixed.items()):
        raise ValueError("solution does not satisfy the supplied fixed trace coefficients")
    weights, targets = [], []
    for local_weights, target in case.moments:
        if len(local_weights) != cells or not np.isfinite(target):
            raise ValueError("one finite weight array per cell and finite moment target required")
        weights.append(tuple(np.array(w, copy=True) for w in local_weights))
        targets.append(dtype(target))
    if len(weights) != len(solution.gauge_multipliers):
        raise ValueError("supply each original physical moment used by the gauged solve")
    if case.moment_rows:
        if len(case.moment_rows) != len(weights):
            raise ValueError("one executed reduced row per physical moment required")
        rows = tuple(
            _preserved_array(row, system.rhs.shape, "moment row").astype(dtype)
            for row in case.moment_rows
        )
    elif system.responses:
        rows = tuple(system.mean_constraint(tuple(w))[0].astype(dtype) for w in weights)
    elif weights:
        raise ValueError("compact systems require executed physical moment rows")
    else:
        rows = ()
    return _StreamCase(
        case,
        trace,
        coarse,
        _preserved_array(solution.gauge_multipliers, (len(weights),), "gauges").astype(dtype),
        boundary,
        fixed,
        np.setdiff1d(np.arange(system.trace_size), list(fixed)),
        tuple(weights),
        np.asarray(targets, dtype=dtype),
        rows,
    )


def _contract_digest(local: HybridRefinementLocal, maps: Sequence[tuple[Any, Any]]) -> str:
    """Hash numerical values, avoiding platform long-double padding bytes."""
    digest = sha256()

    def add(value: Any) -> None:
        array = np.asarray(value)
        digest.update(str((array.shape, array.dtype.kind)).encode())
        if array.dtype.kind == "f":
            digest.update(str(np.finfo(array.dtype).nmant).encode())
            remainder = array.astype(np.longdouble)
            # Three double components cover every supported long-double mantissa.
            for _ in range(3):
                component = remainder.astype("<f8")
                digest.update(component.tobytes())
                remainder = remainder - component.astype(np.longdouble)
        else:
            digest.update(array.astype("<i8").tobytes())

    p = local.problem
    matrix = p.matrix.tocsr()
    for array in (
        matrix.indptr,
        matrix.indices,
        matrix.data,
        p.coupling,
        p.test_coupling,
        p.load,
        p.kernel,
        p.left_kernel,
        p.coarse_basis,
        p.test_basis,
        p.constraints,
        p.test_constraints,
        local.retained_basis,
    ):
        add(array)
    for indices, injection in maps:
        add(indices)
        add(injection)
    return digest.hexdigest()


def _stream_local(
    factory: Callable[[int], HybridRefinementLocal],
    cell: int,
    cases: Sequence[_StreamCase],
    contracts: list[str],
) -> tuple[HybridRefinementLocal, tuple[tuple[Any, FloatArray], ...]]:
    """Restore one fixed local contract and check all case orientation maps."""
    local = factory(cell)
    if not isinstance(local, HybridRefinementLocal):
        raise TypeError("local factory must return HybridRefinementLocal")
    p = local.problem
    _preserved_array(local.retained_basis, p.coarse_basis.shape, "retained basis")
    if any(c.coarse[cell].shape != (p.coarse_basis.shape[1],) for c in cases):
        raise ValueError("local retained width must match each declared cell partition")
    raw_maps = local.trace_maps or tuple((p.trace_dofs, np.eye(p.coupling.shape[1])) for _ in cases)
    if len(raw_maps) != len(cases):
        raise ValueError("one oriented trace map per streamed case required")
    maps = []
    for (indices, injection), case in zip(raw_maps, cases, strict=True):
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            or np.any(indices >= len(case.trace))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("trace maps require distinct valid global indices")
        maps.append(
            (
                indices,
                _preserved_array(injection, (p.coupling.shape[1], len(indices)), "trace injection"),
            )
        )
    contract = _contract_digest(local, maps)
    if not contracts[cell]:
        contracts[cell] = contract
    elif contracts[cell] != contract:
        raise ValueError("original local operator, basis or orientation changed during refinement")
    return local, tuple(maps)


def _stored(
    store: HybridRefinementStore,
    name: str | None,
    step: int,
    cell: int,
    shape: tuple[int, int],
    dtype: Any,
    values: FloatArray | None = None,
) -> FloatArray:
    """Check numerical replay after every caller-owned storage operation."""
    if values is not None:
        if name is None:
            store.write_fields(cell, values)
        else:
            store.write_record(name, step, cell, values)
    restored = store.read_fields(cell) if name is None else store.read_record(name, step, cell)
    _array(restored, shape, "stored fields")
    restored = np.asarray(restored)
    if restored.dtype != np.dtype(dtype) or (
        values is not None and not np.array_equal(restored, values)
    ):
        raise ValueError("streamed storage must preserve the executed correction precision")
    return restored


def _action(matrix: Any, values: FloatArray) -> FloatArray:
    """Apply original rectangular rows with the shared accurate residual owner."""
    shape = (matrix.shape[0], *values.shape[1:])
    return -_accurate_residual(sparse.csr_matrix(matrix), np.zeros(shape), values)


def _global_signature(system: HybridSystem) -> str:
    """Detect mutation of the executed reduced matrix and its declared partition."""
    digest = sha256()
    for array in (
        system.matrix.indptr,
        system.matrix.indices,
        system.matrix.data,
        system.rhs,
        system.kernel_offsets,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _step_limits(max_steps: int, min_steps: int) -> None:
    """Require a consistent explicit interval of nonnegative correction counts."""
    _refinement_steps(max_steps)
    _refinement_steps(min_steps)
    if min_steps > max_steps:
        raise ValueError("min_steps must not exceed max_steps")


def refine_hybrid_stream(
    cases: Sequence[HybridRefinementCase],
    local_factory: Callable[[int], HybridRefinementLocal],
    store: HybridRefinementStore,
    *,
    max_steps: int = 2,
    min_steps: int = 0,
    rtol: float = 1e-10,
    atol: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> tuple[HybridStreamRefinement, ...]:
    """Correct original hybrid equations cellwise with a fixed condensed operator.

    Cases share original local matrices and data, but may have different oriented
    trace spaces, boundary loads, prescribed coefficients and physical moments.
    Each bounded pass calls ``local_factory(cell)`` in cell order; actual local
    bases and all operator/data values must stay identical between passes.
    Local factors solve batched defect sources and are released before the next
    cell. The executed initial field, measured load, inverse response and actual
    physical increment are stored; no conditional solve is repeated for replay.

    Acceptance uses the original volume rows, free weak trace rows and physical
    moments with ``max(atol, rtol*rhs_norm)``. Artificial gauge equations are
    excluded. ``rhs_norm`` includes prescribed physical trace action. The same
    reduced matrix is used for corrections, and prescribed trace increments are
    zero. No tolerance, operator entry or approximation space is modified.
    Compact solutions and per-case coordinate increments are returned; fields
    remain in ``store``. Rejected states raise ``LinearSolveError`` and must not
    be published as accepted acquisitions.
    ``min_steps`` optionally requires further original-equation corrections
    even when the residual is already accepted. It is at most ``max_steps``;
    zero preserves residual-based stopping. It does not certify field accuracy
    or change the original residual criterion.
    """
    _step_limits(max_steps, min_steps)
    _tolerances(rtol, atol)
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        from pymhm.solvers import SolverUnavailableError

        raise SolverUnavailableError("extended refinement requires a wider long-double type")
    if not cases:
        raise ValueError("at least one streamed hybrid case required")
    dtype: Any = np.longdouble if refinement_precision == "extended" else np.float64
    cells = len(cases[0].system.kernel_offsets) - 1
    states = tuple(_stream_case(case, dtype, cells) for case in cases)
    global_contracts = tuple(_global_signature(state.original.system) for state in states)
    contracts = [""] * cells
    histories: list[list[float]] = [[] for _ in states]
    increments: list[list[HybridSolution]] = [[] for _ in states]
    rhs_squared = np.zeros(len(states), dtype=dtype)
    for step in range(max_steps + 1):
        volume_squared = np.zeros(len(states), dtype=dtype)
        weak = [state.boundary.copy() for state in states]
        moment_defects = [state.targets.copy() for state in states]
        retained = [np.zeros_like(state.original.system.rhs, dtype=dtype) for state in states]
        for cell in range(cells):
            local, maps = _stream_local(local_factory, cell, states, contracts)
            p = local.problem
            shape = (len(p.load), len(states))
            fields = _stored(store, None, step, cell, shape, dtype)
            if step == 0:
                _stored(store, "initial", 0, cell, shape, dtype, fields)
            else:
                replay = _stored(store, "initial", 0, cell, shape, dtype).copy()
                for previous in range(step):
                    replay += _stored(store, "correction", previous, cell, shape, dtype)
                if not np.array_equal(replay, fields):
                    raise ValueError(
                        "executed initial fields and ordered corrections do not replay"
                    )
            traces = np.column_stack(
                [
                    injection @ state.trace[indices]
                    for (indices, injection), state in zip(maps, states, strict=True)
                ]
            )
            forcing = _accurate_residual(
                sparse.csr_matrix(p.coupling),
                np.broadcast_to(p.load[:, None], shape).astype(dtype),
                traces,
            )
            defects = _accurate_residual(p.matrix.tocsr(), forcing, fields)
            _stored(store, "load", step, cell, shape, dtype, defects.astype(dtype))
            tests = _action(p.test_coupling.T, fields)
            for j, (state, (indices, injection)) in enumerate(zip(states, maps, strict=True)):
                volume_squared[j] += np.sum(defects[:, j] ** 2, dtype=dtype)
                np.add.at(weak[j], indices, -_action(injection.T, tests[:, j]))
                first, last = state.original.system.kernel_offsets[cell : cell + 2]
                retained[j][first:last] = _action(p.test_basis.T, defects[:, j])
                for m, weights in enumerate(state.weights):
                    w = _preserved_array(weights[cell], (len(p.load),), "moment weights")
                    moment_defects[j][m] -= _action(w[None, :], fields[:, j])[0]
                if step == 0:
                    fixed_trace = np.zeros_like(state.trace)
                    for index, value in state.fixed.items():
                        fixed_trace[index] = value
                    physical_rhs = _accurate_residual(
                        sparse.csr_matrix(p.coupling),
                        p.load.astype(dtype),
                        injection @ fixed_trace[indices],
                    )
                    rhs_squared[j] += np.sum(physical_rhs**2, dtype=dtype)
            del local, p, fields
        residuals, rhs_norms, active = [], [], []
        for j, state in enumerate(states):
            if _global_signature(state.original.system) != global_contracts[j]:
                raise ValueError("executed condensed operator changed during refinement")
            if step == 0:
                rhs_squared[j] += np.sum(state.boundary[state.free] ** 2, dtype=dtype)
                rhs_squared[j] += np.sum(state.targets**2, dtype=dtype)
            norm = float(
                np.sqrt(
                    volume_squared[j]
                    + np.sum(weak[j][state.free] ** 2)
                    + np.sum(moment_defects[j] ** 2)
                )
            )
            rhs_norm = float(np.sqrt(rhs_squared[j]))
            histories[j].append(norm)
            residuals.append(norm)
            rhs_norms.append(rhs_norm)
            active.append(
                step < min_steps or not np.isfinite(norm) or norm > max(atol, rtol * rhs_norm)
            )
        if not any(active):
            results = []
            for j, state in enumerate(states):
                system = state.original.system
                vector = np.r_[state.trace, *state.coarse]
                retained[j][: system.trace_size] = weak[j]
                free = np.setdiff1d(np.arange(len(system.rhs)), list(state.fixed))
                scale = max(
                    float(np.linalg.norm(system.rhs[free])),
                    float(np.linalg.norm((abs(system.matrix) @ abs(vector))[free])),
                    np.finfo(float).tiny,
                )
                solution = HybridSolution(
                    state.trace,
                    state.coarse,
                    (),
                    float(np.linalg.norm(retained[j][free])) / scale,
                    state.gauges,
                )
                results.append(
                    HybridStreamRefinement(
                        solution,
                        tuple(histories[j]),
                        rhs_norms[j],
                        tuple(increments[j]),
                        tuple(contracts),
                    )
                )
            return tuple(results)
        if step == max_steps or not np.isfinite(residuals).all():
            failed = next(j for j, flag in enumerate(active) if flag)
            raise LinearSolveError(
                f"original hybrid case {failed} residual {residuals[failed]:.6e} "
                f"exceeds {max(atol, rtol * rhs_norms[failed]):.6e} "
                f"after {step} corrections"
            )
        loads = [np.zeros_like(state.original.system.rhs, dtype=dtype) for state in states]
        scales = [np.zeros_like(load) for load in loads]
        source_moments = [np.zeros_like(state.targets) for state in states]
        for cell in range(cells):
            local, maps = _stream_local(local_factory, cell, states, contracts)
            p = local.problem
            shape = (len(p.load), len(states))
            defects = _stored(store, "load", step, cell, shape, dtype).copy()
            defects[:, np.logical_not(active)] = 0
            _stored(store, "forcing", step, cell, shape, dtype, defects)
            forcing = np.vstack(
                [defects, np.zeros((p.coarse_basis.shape[1], len(states)), dtype=dtype)]
            )
            with factorize(p.condensation_matrix(), solver=local_solver) as factor:
                source = factor.solve(forcing, refinement_precision=refinement_precision)[
                    : len(p.load)
                ].astype(dtype)
            _stored(store, "source", step, cell, shape, dtype, source)
            reduced = p.condensed_load(source, load=defects)
            for j, (state, (indices, injection)) in enumerate(zip(states, maps, strict=True)):
                first, last = state.original.system.kernel_offsets[cell : cell + 2]
                trace_load = _action(injection.T, reduced[: p.coupling.shape[1], j])
                np.add.at(loads[j], indices, trace_load)
                np.add.at(scales[j], indices, abs(trace_load))
                loads[j][first:last] += reduced[p.coupling.shape[1] :, j]
                scales[j][first:last] += abs(reduced[p.coupling.shape[1] :, j])
                for m, weights in enumerate(state.weights):
                    source_moments[j][m] += _action(
                        np.asarray(weights[cell])[None, :], source[:, j]
                    )[0]
            del local, p, source
        corrections = []
        for j, state in enumerate(states):
            if _global_signature(state.original.system) != global_contracts[j]:
                raise ValueError("executed condensed operator changed during refinement")
            if active[j]:
                size = state.original.system.trace_size
                loads[j][:size] -= weak[j]
                scales[j][:size] += abs(weak[j])
                system = state.original.system.with_rhs(
                    loads[j], load_scale=np.maximum(scales[j], abs(loads[j]))
                )
                constraints = [
                    (row, target - source)
                    for row, target, source in zip(
                        state.rows, moment_defects[j], source_moments[j], strict=True
                    )
                ]
                correction = system.solve(
                    solver=solver,
                    fixed={i: 0.0 for i in state.fixed},
                    constraints=constraints,
                    refinement_precision=refinement_precision,
                )
            else:
                correction = HybridSolution(
                    np.zeros_like(state.trace),
                    tuple(np.zeros_like(c) for c in state.coarse),
                    (),
                    0.0,
                    np.zeros_like(state.gauges),
                )
            corrections.append(correction)
            increments[j].append(correction)
        for cell in range(cells):
            local, maps = _stream_local(local_factory, cell, states, contracts)
            p = local.problem
            shape = (len(p.load), len(states))
            source = _stored(store, "source", step, cell, shape, dtype)
            trace = np.column_stack(
                [
                    injection @ correction.trace[indices]
                    for (indices, injection), correction in zip(maps, corrections, strict=True)
                ]
            )
            coarse = np.column_stack([correction.coarse[cell] for correction in corrections])
            correction_fields = p.with_load(np.zeros_like(p.load)).reconstruct(
                trace,
                coarse,
                retained_basis=local.retained_basis,
                solver=local_solver,
                refinement_precision=refinement_precision,
            )
            delta = (source + correction_fields).astype(dtype)
            _stored(store, "correction", step, cell, shape, dtype, delta)
            fields = _stored(store, None, step, cell, shape, dtype)
            _stored(store, None, step, cell, shape, dtype, fields + delta)
            del local, p, fields
        for state, correction in zip(states, corrections, strict=True):
            state.trace += correction.trace
            state.gauges += correction.gauge_multipliers
            state.coarse = tuple(
                c + dc for c, dc in zip(state.coarse, correction.coarse, strict=True)
            )
    raise AssertionError("unreachable refinement state")


def refine_hybrid(
    system: HybridSystem,
    solution: HybridSolution,
    *,
    boundary_load: Any = None,
    fixed: dict[int, float] | None = None,
    moments: Sequence[tuple[Sequence[FloatArray], float]] = (),
    max_steps: int = 2,
    min_steps: int = 0,
    rtol: float = 1e-10,
    atol: float = 0.0,
    solver: str = "scipy",
    local_solver: str = "scipy",
    refinement_precision: Literal["double", "extended"] = "double",
) -> HybridRefinement:
    """Refine ``A_i u_i+B_i lambda=f_i`` and the original weak trace equations.

    Supply the original ``boundary_load`` (zero by default), prescribed trace
    coefficients ``fixed``, and physical ``moments`` as in the initial solve.
    A moment is ``(local_weights, target)``; do not pass reduced gauge rows.
    Distinct test/trial couplings and multiple retained modes are supported.

    Each correction retains the original lifts and condensed operator. Local
    direct factors solve only the new defect source and are released cellwise;
    no full uncondensed matrix is assembled. The default solution is untouched
    unless this function is explicitly called. ``max_steps=0`` checks it only.
    Acceptance requires the full original residual to be at most
    ``max(atol, rtol*rhs_norm)``. Failure raises ``LinearSolveError`` with the
    achieved residual, including stagnation at a precision floor.
    Extended correction storage is explicit and requires a wider NumPy type.
    ``min_steps`` optionally requires original-equation corrections after the
    residual is already accepted, up to ``max_steps``. Zero preserves the
    default stopping convention. A small residual or a requested correction
    count alone does not certify physical field accuracy.
    """
    _step_limits(max_steps, min_steps)
    _tolerances(rtol, atol)
    if refinement_precision not in {"double", "extended"}:
        raise ValueError("refinement_precision must be double or extended")
    if refinement_precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        from pymhm.solvers import SolverUnavailableError

        raise SolverUnavailableError("extended refinement requires a wider long-double type")
    dtype: Any = np.longdouble if refinement_precision == "extended" else np.float64
    responses = system.responses
    if len(solution.fields) != len(responses) or len(solution.coarse) != len(responses):
        raise ValueError("one field and retained coefficient array per local response required")
    trace = _array(solution.trace, (system.trace_size,), "trace").astype(dtype)
    fields, coarse = [], []
    for response, field, coefficients in zip(
        responses, solution.fields, solution.coarse, strict=True
    ):
        # Validate without discarding the extra digits of an extended input.
        _array(field, response.source.shape, "field")
        _array(coefficients, (response.problem.coarse_basis.shape[1],), "coarse")
        fields.append(np.array(field, dtype=dtype, copy=True))
        coarse.append(np.array(coefficients, dtype=dtype, copy=True))
    trace[:] = solution.trace
    multipliers = np.array(solution.gauge_multipliers, dtype=dtype, copy=True)
    boundary = _preserved_array(
        np.zeros(system.trace_size) if boundary_load is None else boundary_load,
        (system.trace_size,),
        "boundary_load",
    ).astype(dtype)
    prescribed = {} if fixed is None else dict(fixed)
    if (
        any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, (int, np.integer))
            or not 0 <= i < system.trace_size
            for i in prescribed
        )
        or not np.isfinite(list(prescribed.values())).all()
    ):
        raise ValueError("fixed DOFs must be valid trace indices with finite values")
    if any(trace[i] != value for i, value in prescribed.items()):
        raise ValueError("solution does not satisfy the supplied fixed trace coefficients")
    free = np.setdiff1d(np.arange(system.trace_size), list(prescribed))
    weights = []
    targets = []
    for local_weights, target in moments:
        if len(local_weights) != len(responses) or not np.isfinite(target):
            raise ValueError("one finite weight array per cell and finite moment target required")
        weights.append(
            tuple(
                _preserved_array(w, r.source.shape, "moment weights").astype(dtype)
                for w, r in zip(local_weights, responses, strict=True)
            )
        )
        targets.append(dtype(target))
    if len(weights) != len(solution.gauge_multipliers):
        raise ValueError("supply each original physical moment used by the gauged solve")
    fixed_trace = np.zeros(system.trace_size, dtype=dtype)
    for index, value in prescribed.items():
        fixed_trace[index] = value
    rhs = np.concatenate(
        [r.problem.load - r.problem.coupling @ fixed_trace[r.problem.trace_dofs] for r in responses]
        + [boundary[free], np.asarray(targets)]
    )
    rhs_norm = float(np.linalg.norm(rhs))
    threshold = max(atol, rtol * rhs_norm)
    history = []
    step = 0
    while True:
        defects = []
        weak = boundary.astype(dtype)
        for response, field in zip(responses, fields, strict=True):
            problem = response.problem
            forcing = _accurate_residual(
                sparse.csr_matrix(problem.coupling),
                problem.load.astype(dtype),
                trace[problem.trace_dofs],
            )
            defects.append(_accurate_residual(problem.matrix.tocsr(), forcing, field))
            np.add.at(weak, problem.trace_dofs, -problem.test_coupling.astype(dtype).T @ field)
        moment_defects = np.asarray(
            [
                target
                - sum(np.asarray(w, dtype=dtype) @ u for w, u in zip(local, fields, strict=True))
                for local, target in zip(weights, targets, strict=True)
            ],
            dtype=dtype,
        )
        residual = float(np.linalg.norm(np.concatenate([*defects, weak[free], moment_defects])))
        history.append(residual)
        if step >= min_steps and np.isfinite(residual) and residual <= threshold:
            vector = np.r_[trace, *coarse]
            retained = [
                response.problem.test_basis.T @ defect
                for response, defect in zip(responses, defects, strict=True)
            ]
            reduced = np.concatenate([weak[free], *retained])
            active = np.setdiff1d(np.arange(len(system.rhs)), list(prescribed))
            scale = max(
                float(np.linalg.norm(system.rhs[active])),
                float(np.linalg.norm((abs(system.matrix) @ abs(vector))[active])),
                np.finfo(float).tiny,
            )
            corrected = HybridSolution(
                trace,
                tuple(coarse),
                tuple(fields),
                float(np.linalg.norm(reduced)) / scale,
                multipliers,
            )
            return HybridRefinement(corrected, tuple(history), rhs_norm)
        if step == max_steps or not np.isfinite(residual):
            raise LinearSolveError(
                f"original hybrid residual {residual:.6e} exceeds {threshold:.6e} "
                f"after {step} corrections"
            )
        corrected_responses = []
        for response, defect in zip(responses, defects, strict=True):
            problem = response.problem
            matrix, _ = problem.condensation_system()
            forcing = np.r_[
                np.asarray(defect, dtype=dtype),
                np.zeros(problem.coarse_basis.shape[1], dtype=dtype),
            ]
            with factorize(matrix, solver=local_solver) as factor:
                source = factor.solve(forcing, refinement_precision=refinement_precision)[
                    : len(problem.load)
                ]
            corrected_responses.append(
                LocalResponse(
                    problem.with_load(
                        defect, preserve_precision=refinement_precision == "extended"
                    ),
                    source,
                    response.lifts,
                    response.coarse_vectors,
                )
            )
        correction_system = HybridSystem.from_responses(corrected_responses, boundary_load=weak)
        constraints = [
            correction_system.mean_constraint(w, target)
            for w, target in zip(weights, moment_defects, strict=True)
        ]
        correction = correction_system.solve(
            solver=solver,
            fixed={i: 0.0 for i in prescribed},
            constraints=constraints,
            refinement_precision=refinement_precision,
        )
        trace += correction.trace
        multipliers += correction.gauge_multipliers
        fields = [u + du for u, du in zip(fields, correction.fields, strict=True)]
        coarse = [c + dc for c, dc in zip(coarse, correction.coarse, strict=True)]
        step += 1
