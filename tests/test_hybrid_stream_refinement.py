"""Bounded original-equation correction and executed numerical replay contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pymhm.hybrid import HybridSolution, HybridSystem, LocalProblem
from pymhm.hybrid_refinement import (
    HybridRefinementCase,
    HybridRefinementLocal,
    refine_hybrid,
    refine_hybrid_stream,
)
from pymhm.solvers import LinearSolveError, SolverUnavailableError


class MemoryStore:
    """Keep small test fields and actual records using the same public store contract."""

    def __init__(self, fields, dtype=float):
        self.fields = [np.asarray(u, dtype=dtype).copy() for u in fields]
        self.records = {}

    def read_fields(self, cell):
        return self.fields[cell]

    def write_fields(self, cell, fields):
        self.fields[cell] = fields.copy()

    def write_record(self, name, step, cell, values):
        self.records[name, step, cell] = values.copy()

    def read_record(self, name, step, cell):
        return self.records[name, step, cell]


def simple(dtype=float):
    """Provide compact coordinates and deliberately perturbed executed physical fields."""
    p = LocalProblem([[2.0, 0.2], [0.2, 3.0]], np.eye(2), [1.0, 2.0], np.arange(2))
    full = HybridSystem([p], boundary_load=[2.0, 3.0])
    solution = full.solve()
    compact = full.with_rhs(full.rhs, load_scale=full.load_scale)
    case = HybridRefinementCase(compact, replace(solution, fields=()), [2.0, 3.0])
    store = MemoryStore([(solution.fields[0] + [0.01, -0.02])[:, None]], dtype)
    local = HybridRefinementLocal(p, full.responses[0].retained_basis)
    return case, store, lambda _: local, solution


@pytest.mark.parametrize("streamed", [False, True])
def test_minimum_corrections_resolve_a_small_unconstrained_component(streamed):
    """Accepted backward error can still leave a small weak component inaccurate."""
    problem = LocalProblem(np.diag([1.0, 1e-6]), [[1.0], [0.0]], [1.0, 1e-12], [0])
    system = HybridSystem([problem], boundary_load=[1.0])
    exact = system.solve()
    field = exact.fields[0].copy()
    field[1] += 1e-10
    altered = replace(exact, fields=(field,))
    original_matrix = system.matrix.toarray().copy()
    original_load = problem.load.copy()
    if streamed:
        compact = system.with_rhs(system.rhs, load_scale=system.load_scale)
        case = HybridRefinementCase(compact, replace(altered, fields=()), [1.0])
        local = HybridRefinementLocal(problem, system.responses[0].retained_basis)
        unchanged_store, forced_store = MemoryStore([field[:, None]]), MemoryStore([field[:, None]])
        unchanged = refine_hybrid_stream([case], lambda _: local, unchanged_store)[0]
        forced = refine_hybrid_stream(
            [case], lambda _: local, forced_store, min_steps=2, max_steps=2
        )[0]
        unchanged_field, forced_field = (
            unchanged_store.fields[0][:, 0],
            forced_store.fields[0][:, 0],
        )
        replay = forced_store.records["initial", 0, 0].copy()
        for step in range(2):
            replay += forced_store.records["correction", step, 0]
        assert_array_equal(replay, forced_store.fields[0])
    else:
        unchanged = refine_hybrid(system, altered, boundary_load=[1.0])
        forced = refine_hybrid(system, altered, boundary_load=[1.0], min_steps=2, max_steps=2)
        unchanged_field, forced_field = unchanged.solution.fields[0], forced.solution.fields[0]
    assert_array_equal(unchanged_field, field)
    assert len(unchanged.residual_norms) == 1
    assert len(forced.residual_norms) == 3
    assert all(norm <= 1e-10 * forced.rhs_norm for norm in forced.residual_norms)
    assert abs(forced_field[1] - exact.fields[0][1]) <= 10 * np.finfo(float).eps * abs(field[1])
    assert_array_equal(system.matrix.toarray(), original_matrix)
    assert_array_equal(problem.load, original_load)


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("minimum,maximum", [(-1, 2), (True, 2), (0.5, 2), (2, 1)])
def test_minimum_correction_bounds_are_checked_before_assembly(streamed, minimum, maximum):
    """Both storage strategies enforce the same explicit nonnegative step interval."""
    with pytest.raises(ValueError):
        if streamed:
            refine_hybrid_stream([], lambda _: None, None, min_steps=minimum, max_steps=maximum)
        else:
            refine_hybrid(None, None, min_steps=minimum, max_steps=maximum)


@pytest.mark.parametrize("retained", ["none", "kernel", "general"])
@pytest.mark.parametrize("gauged", [False, True])
@pytest.mark.parametrize("prescribed", [False, True])
@pytest.mark.parametrize("boundary_scale", [0.0, 1.0])
def test_streamed_petrov_orientations_match_full_original_saddle(
    retained, gauged, prescribed, boundary_scale
):
    """Preserve independent test modes, rotated traces, weak BCs and physical moments."""
    rng = np.random.default_rng(814)
    left, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    right, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    eigen = np.arange(1.0, 8.0)
    if retained == "kernel":
        eigen[-2:] = 0
    matrix = left @ np.diag(eigen) @ right.T
    coupling, test = rng.normal(size=(7, 3)), rng.normal(size=(7, 3))
    load, boundary = rng.normal(size=7), boundary_scale * rng.normal(size=3)
    options = {}
    if retained != "none":
        options = dict(
            constraints=np.diag(np.arange(1, 8)) @ right[:, -2:],
            test_constraints=np.diag(np.arange(2, 9)) @ left[:, -2:],
        )
        if retained == "kernel":
            options.update(kernel=right[:, -2:], left_kernel=left[:, -2:])
        else:
            options.update(coarse_basis=right[:, -2:], test_basis=left[:, -2:])
    original = LocalProblem(matrix, coupling, load, np.arange(3), test_coupling=test, **options)
    maps, cases, fields, expected = [], [], [], []
    for injection in [np.eye(3), np.linalg.qr(rng.normal(size=(3, 3)))[0]]:
        b, t, g = coupling @ injection, test @ injection, injection.T @ boundary
        p = LocalProblem(matrix, b, load, np.arange(3), test_coupling=t, **options)
        system = HybridSystem([p], boundary_load=g)
        fixed = {0: 0.2} if prescribed else {}
        free = np.setdiff1d(np.arange(3), list(fixed))
        rhs = load - (b[:, 0] * 0.2 if prescribed else 0)
        exact = np.linalg.solve(
            np.block([[matrix, b[:, free]], [t[:, free].T, np.zeros((len(free), len(free)))]]),
            np.r_[rhs, g[free]],
        )
        weights = [rng.normal(size=7)]
        moments = [(weights, weights[0] @ exact[:7])] if gauged else []
        constraints = [system.mean_constraint(w, target) for w, target in moments]
        initial = system.solve(fixed=fixed, constraints=constraints)
        trace = initial.trace.copy()
        trace[free] += 0.003
        coarse = tuple(c + 0.002 for c in initial.coarse)
        field = system.responses[0].reconstruct(trace, coarse[0])
        altered = replace(initial, trace=trace, coarse=coarse, fields=())
        compact = system.with_rhs(system.rhs, load_scale=system.load_scale)
        cases.append(
            HybridRefinementCase(compact, altered, g, fixed, moments, [r for r, _ in constraints])
        )
        maps.append((np.arange(3), injection))
        fields.append(field)
        expected.append(exact)
    basis = original.condense().retained_basis
    store = MemoryStore([np.column_stack(fields)])
    local = HybridRefinementLocal(original, basis, maps)
    result = refine_hybrid_stream(cases, lambda _: local, store, rtol=1e-12)
    for j, (actual, exact) in enumerate(zip(result, expected, strict=True)):
        assert actual.residual_norms[-1] <= 1e-12 * actual.rhs_norm
        assert len(actual.residual_norms) == 2
        assert_allclose(store.fields[0][:, j], exact[:7], rtol=1e-10, atol=1e-10)
        assert_allclose(actual.solution.trace[free], exact[7:], rtol=1e-10, atol=1e-10)
        assert_allclose(actual.solution.gauge_multipliers, 0, atol=2e-12)
        assert actual.solution.fields == ()
        assert actual.solution.residual < 2e-12
        assert len(actual.local_contract_digests[0]) == 64
        replay = store.records["initial", 0, 0][:, j].copy()
        for step, correction in enumerate(actual.increments):
            replay += store.records["correction", step, 0][:, j]
            assert correction.fields == ()
        assert_array_equal(replay, store.fields[0][:, j])


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_two_cell_physical_gauge_and_fixed_flux_agree_with_normal_owner(precision):
    """Stream the same Neumann moments and nonzero fluxes as the in-memory owner."""
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("requires wider accumulation")
    dtype = np.longdouble if precision == "extended" else float
    cells = [
        LocalProblem(
            [[1.0, -1], [-1, 1]],
            np.diag([1.0, -1]),
            load,
            [i, i + 1],
            kernel=np.ones((2, 1)),
        )
        for i, load in enumerate(([1.0, 0.0], [0.0, -1.0]))
    ]
    full = HybridSystem(cells)
    fixed, weights, target = {0: 0.2, 2: 0.2}, [np.ones(2) / 2] * 2, 2.0
    row, reduced_target = full.mean_constraint(weights, target)
    value = full.solve(fixed=fixed, constraints=[(row, reduced_target)])
    trace = value.trace.copy()
    trace[1] += 0.003
    coarse = tuple(c + 0.002 for c in value.coarse)
    fields = tuple(
        r.reconstruct(trace[r.problem.trace_dofs], c)
        for r, c in zip(full.responses, coarse, strict=True)
    )
    initial = replace(value, trace=trace, coarse=coarse, fields=fields)
    expected = refine_hybrid(
        full, initial, fixed=fixed, moments=[(weights, target)], refinement_precision=precision
    )
    compact = full.with_rhs(full.rhs, load_scale=full.load_scale)
    case = HybridRefinementCase(
        compact, initial, fixed=fixed, moments=[(weights, target)], moment_rows=[row]
    )
    store = MemoryStore([u[:, None] for u in fields], dtype)
    result = refine_hybrid_stream(
        [case],
        lambda cell: HybridRefinementLocal(cells[cell], full.responses[cell].retained_basis),
        store,
        refinement_precision=precision,
    )[0]
    assert_allclose(result.solution.trace, expected.solution.trace, atol=2e-14)
    assert_allclose(result.solution.coarse, expected.solution.coarse, atol=2e-14)
    for cell, field in enumerate(expected.solution.fields):
        assert_allclose(store.fields[cell][:, 0], field, atol=2e-14)
        assert store.fields[cell].dtype == np.dtype(dtype)
    assert_allclose(
        sum(w @ store.fields[i][:, 0] for i, w in enumerate(weights)), target, atol=2e-14
    )


def test_accepted_batch_member_is_not_changed_while_another_is_corrected():
    """An already accepted state receives explicit zero increments in a shared batch."""
    case, store, factory, exact = simple()
    accepted = exact.fields[0] + [1e-12, -1e-12]
    store.fields[0] = np.column_stack([accepted, store.fields[0][:, 0]])
    result = refine_hybrid_stream([case, case], factory, store)
    assert_array_equal(store.fields[0][:, 0], accepted)
    assert_array_equal(result[0].increments[0].trace, 0)
    assert np.linalg.norm(store.records["load", 0, 0][:, 0]) > 0
    assert_array_equal(store.records["forcing", 0, 0][:, 0], 0)
    assert_array_equal(store.records["source", 0, 0][:, 0], 0)
    assert_array_equal(store.records["correction", 0, 0][:, 0], 0)


def test_zero_steps_rejects_unaccepted_fields_and_checks_original_replay():
    """Only the same original equations can certify an executed physical field."""
    case, store, factory, _ = simple()
    with pytest.raises(LinearSolveError, match="after 0 corrections"):
        refine_hybrid_stream([case], factory, store, max_steps=0)
    result = refine_hybrid_stream([case], factory, store)[0]
    restored = replace(case, solution=result.solution)
    replay = store.records["initial", 0, 0].copy()
    for step in range(len(result.increments)):
        replay += store.records["correction", step, 0]
    assert_array_equal(replay, store.fields[0])
    checked = refine_hybrid_stream([restored], factory, MemoryStore([replay]), max_steps=0)[0]
    assert checked.residual_norms == (result.residual_norms[-1],)


def test_failing_local_correction_releases_factor_and_preserves_rejection(monkeypatch):
    """An inaccurate local inverse cannot be accepted by the reduced residual alone."""
    from contextlib import contextmanager

    import pymhm.hybrid_refinement as owner

    closed = []

    @contextmanager
    def ineffective(matrix, **options):
        try:
            yield SimpleNamespace(solve=lambda rhs, **kwargs: np.zeros_like(rhs))
        finally:
            closed.append(True)

    case, store, factory, _ = simple()
    monkeypatch.setattr(owner, "factorize", ineffective)
    with pytest.raises(LinearSolveError, match="after 1 corrections"):
        refine_hybrid_stream([case], factory, store, max_steps=1)
    assert closed == [True]


@pytest.mark.parametrize("precision", ["double", "extended"])
def test_changed_original_operator_or_basis_is_rejected_between_passes(precision):
    """A factory cannot silently change the equations during a correction."""
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("requires wider accumulation")
    dtype = np.longdouble if precision == "extended" else float
    case, store, factory, _ = simple(dtype)
    calls = 0

    def changing(cell):
        nonlocal calls
        local = factory(cell)
        calls += 1
        if calls > 1:
            p = local.problem
            return replace(local, problem=p.with_load(p.load + [1e-4, 0]))
        return local

    with pytest.raises(ValueError, match="changed during"):
        refine_hybrid_stream([case], changing, store, refinement_precision=precision)


def test_wide_storage_cannot_silently_cast_executed_fields(monkeypatch):
    """Explicit extended correction requires a store that retains its full dtype."""
    case, store, factory, _ = simple()
    with pytest.raises(ValueError, match="precision"):
        refine_hybrid_stream([case], factory, store, refinement_precision="extended")
    import pymhm.hybrid_refinement as owner

    actual = np.finfo
    monkeypatch.setattr(owner.np, "finfo", lambda dtype: actual(float))
    with pytest.raises(SolverUnavailableError, match="wider"):
        refine_hybrid_stream([case], factory, store, refinement_precision="extended")


@pytest.mark.parametrize(
    "options,match",
    [
        ({"max_steps": -1}, "nonnegative"),
        ({"rtol": -1}, "nonnegative"),
        ({"refinement_precision": "automatic"}, "precision"),
    ],
)
def test_invalid_stream_controls_fail_before_reading_fields(options, match):
    case, store, factory, _ = simple()
    with pytest.raises(ValueError, match=match):
        refine_hybrid_stream([case], factory, store, **options)


def test_empty_stream_and_factory_type_are_explicit_contracts():
    case, store, _, _ = simple()
    with pytest.raises(ValueError, match="at least one"):
        refine_hybrid_stream([], lambda _: None, store)
    with pytest.raises(TypeError, match="HybridRefinementLocal"):
        refine_hybrid_stream([case], lambda _: None, store)


@pytest.mark.parametrize("fixed", [{True: 0}, {2: 0}, {0: np.nan}, {0: 23}])
def test_invalid_fixed_data_is_rejected(fixed):
    case, store, factory, _ = simple()
    with pytest.raises(ValueError, match="fixed"):
        refine_hybrid_stream([replace(case, fixed=fixed)], factory, store)


@pytest.mark.parametrize(
    "alter,match",
    [
        ({"moments": [([], 1.0)]}, "weight array"),
        ({"moments": [([np.ones(2)], np.inf)]}, "finite moment"),
        ({"moments": [([np.ones(2)], 1.0)]}, "each original physical moment"),
        ({"moment_rows": [np.ones(2)]}, "one executed reduced row"),
    ],
)
def test_invalid_physical_moment_contracts(alter, match):
    case, store, factory, _ = simple()
    with pytest.raises(ValueError, match=match):
        refine_hybrid_stream([replace(case, **alter)], factory, store)


@pytest.mark.parametrize("indices", [[0, 0], [-1, 0], [0, 2], [0.0, 1.0], [[0]]])
def test_invalid_trace_orientation_maps(indices):
    case, store, factory, _ = simple()
    local = replace(factory(0), trace_maps=[(indices, np.eye(2))])
    with pytest.raises(ValueError, match="valid global indices"):
        refine_hybrid_stream([case], lambda _: local, store)


def test_compact_physical_moments_require_executed_rows():
    case, store, factory, _ = simple()
    altered = replace(case.solution, gauge_multipliers=np.zeros(1))
    with pytest.raises(ValueError, match="executed physical moment rows"):
        refine_hybrid_stream(
            [replace(case, solution=altered, moments=[([np.ones(2)], 0.0)])], factory, store
        )


def test_full_system_can_supply_physical_moment_rows():
    case, store, factory, exact = simple()
    p = factory(0).problem
    full = HybridSystem([p], boundary_load=case.boundary_load)
    weights = [np.ones(2)]
    target = weights[0] @ exact.fields[0]
    solution = full.solve(constraints=[full.mean_constraint(weights, target)])
    result = refine_hybrid_stream(
        [replace(case, system=full, solution=solution, moments=[(weights, target)])], factory, store
    )[0]
    assert result.residual_norms[-1] <= 1e-10 * result.rhs_norm


@pytest.mark.skipif(
    np.finfo(np.longdouble).eps >= np.finfo(float).eps, reason="requires wider accumulation"
)
def test_original_wide_defect_load_is_preserved_in_both_shared_owners(monkeypatch):
    """Original wide forcing reaches a factor with an explicitly stricter test criterion."""
    import pymhm.hybrid_refinement as owner

    actual_factorize = owner.factorize
    captured = []

    def strict_factor(matrix, **options):
        factor = actual_factorize(matrix, rtol=1e-22, **options)
        original_solve = factor.solve

        def checked_solve(rhs, **kwargs):
            captured.append(rhs.copy())
            return original_solve(rhs, **kwargs)

        factor.solve = checked_solve
        return factor

    monkeypatch.setattr(owner, "factorize", strict_factor)
    expected = np.array([np.longdouble(1) + np.longdouble(2) ** -60])
    problem = LocalProblem([[1.0]], np.empty((1, 0)), [0.0], np.empty(0, dtype=int))
    problem = problem.with_load(expected, preserve_precision=True)
    full = HybridSystem([problem], local_refinement_precision="extended")
    initial = HybridSolution(np.empty(0), (np.empty(0),), (np.zeros(1),), 0.0, np.empty(0))
    normal = refine_hybrid(full, initial, max_steps=1, rtol=1e-22, refinement_precision="extended")
    assert_array_equal(normal.solution.fields[0], expected)
    compact = full.with_rhs(full.rhs)
    store = MemoryStore([np.zeros((1, 1))], np.longdouble)
    local = HybridRefinementLocal(problem, np.empty((1, 0)))
    stream = refine_hybrid_stream(
        [HybridRefinementCase(compact, initial)],
        lambda _: local,
        store,
        max_steps=1,
        rtol=1e-22,
        refinement_precision="extended",
    )[0]
    assert_array_equal(store.fields[0][:, 0], expected)
    assert normal.residual_norms[-1] == stream.residual_norms[-1] == 0
    assert len(captured) == 2
    for forcing in captured:
        assert_array_equal(forcing.ravel(), expected)
        assert forcing.dtype == np.longdouble


def test_online_rhs_reuses_exact_operator_and_preserves_absolute_load_scale():
    """New loads preserve cancellation diagnostics and declare compact reconstruction."""
    case, _, _, _ = simple()
    system = case.system
    online = system.with_rhs(-system.rhs, load_scale=system.load_scale)
    assert online.matrix is system.matrix
    assert_array_equal(online.kernel_offsets, system.kernel_offsets)
    assert_array_equal(online.rhs, -system.rhs)
    assert online.responses == ()
    with pytest.raises(ValueError, match="bound"):
        system.with_rhs(system.rhs, load_scale=-np.ones_like(system.rhs))
    with pytest.raises(ValueError, match="bound"):
        system.with_rhs(system.rhs, load_scale=np.zeros_like(system.rhs))


def test_aliased_initial_records_cannot_hide_a_broken_replay_contract():
    """A store that mutates its archived initial field is rejected before acceptance."""
    case, store, factory, _ = simple()

    class AliasedStore(MemoryStore):
        def write_record(self, name, step, cell, values):
            self.records[name, step, cell] = values

        def write_fields(self, cell, fields):
            self.fields[cell][:] = fields

    store = AliasedStore(store.fields)
    with pytest.raises(ValueError, match="do not replay"):
        refine_hybrid_stream([case], factory, store)


@pytest.mark.parametrize("change_pass", [1, 2])
def test_same_condensed_matrix_cannot_change_during_measurement_or_inverse_pass(change_pass):
    """Factory side effects cannot silently change the operator used for corrections."""
    case, store, factory, _ = simple()
    calls = 0

    def changing(cell):
        nonlocal calls
        calls += 1
        if calls == change_pass:
            case.system.matrix.data[0] += 0.001
        return factory(cell)

    with pytest.raises(ValueError, match="condensed operator changed"):
        refine_hybrid_stream([case], changing, store)


@pytest.mark.parametrize("backend", ["scipy", "pypardiso"])
@pytest.mark.parametrize("precision", ["double", "extended"])
def test_streamed_native_factor_equivalence_and_executed_replay(backend, precision):
    """Native direct factors solve the same original rows and preserve recorded deltas."""
    if backend == "pypardiso":
        pytest.importorskip("pypardiso")
    if precision == "extended" and np.finfo(np.longdouble).eps >= np.finfo(float).eps:
        pytest.skip("requires wider accumulation")
    dtype = np.longdouble if precision == "extended" else float
    case, store, factory, expected = simple(dtype)
    result = refine_hybrid_stream(
        [case], factory, store, solver=backend, local_solver=backend, refinement_precision=precision
    )[0]
    assert_allclose(store.fields[0][:, 0], expected.fields[0], atol=2e-14)
    replay = store.records["initial", 0, 0].copy()
    for step in range(len(result.increments)):
        replay += store.records["correction", step, 0]
    assert_array_equal(replay, store.fields[0])


@pytest.mark.parametrize("which", ["partition", "width", "maps", "injection", "stored"])
def test_streamed_local_layout_contract_is_validated(which):
    """Dimensional agreement includes field, retained and oriented skeleton layouts."""
    case, store, factory, _ = simple()
    local = factory(0)
    if which == "partition":
        case = replace(case, solution=replace(case.solution, coarse=()))
    elif which == "width":
        case = replace(case, solution=replace(case.solution, coarse=(np.ones(1),)))
        case.system.kernel_offsets[-1] += 1
    elif which == "maps":
        local = replace(local, trace_maps=[(np.arange(2), np.eye(2))] * 2)
    elif which == "injection":
        local = replace(local, trace_maps=[(np.arange(2), np.eye(3))])
    else:
        store.fields[0] = np.zeros((3, 1))
    with pytest.raises(
        ValueError, match="partition|retained width|trace map|trace injection|stored fields"
    ):
        refine_hybrid_stream([case], lambda _: local, store)


@pytest.mark.parametrize("corruption", ["dtype", "value"])
def test_record_readback_rejects_narrowing_or_changed_digits(corruption):
    """A caller cannot replace an executed array with numerically different stored data."""
    case, store, factory, _ = simple()

    class IncorrectStore(MemoryStore):
        def write_record(self, name, step, cell, values):
            changed = values.astype(np.float32) if corruption == "dtype" else values + 1e-4
            self.records[name, step, cell] = changed

    with pytest.raises(ValueError, match="executed correction precision"):
        refine_hybrid_stream([case], factory, IncorrectStore(store.fields))


@pytest.mark.parametrize("invalid", [np.zeros((3, 1)), np.zeros((2, 0)), np.zeros(3)])
def test_shared_reduced_load_validates_original_source_layout(invalid):
    case, _, factory, _ = simple()
    with pytest.raises(ValueError, match="source_response"):
        factory(0).problem.condensed_load(invalid)


def test_shared_reduced_load_rejects_ambiguous_retained_convention():
    case, _, factory, _ = simple()
    with pytest.raises(ValueError, match="corrected_retained"):
        factory(0).problem.condensed_load(np.zeros(2), corrected_retained="automatic")


def test_shared_reduced_load_broadcasts_original_vector_to_source_columns():
    case, _, factory, _ = simple()
    p = factory(0).problem
    values = np.column_stack([np.ones(2), np.arange(2.0)])
    assert_array_equal(p.condensed_load(values), np.eye(2) @ values)
    assert_array_equal(
        p.condensed_load(values, load=np.ones(2)),
        p.condensed_load(values, load=np.ones_like(values)),
    )
